// grxBLAS host implementation.
//
// The library ships precompiled device kernels as .vxbin modules and resolves
// them by name, the way a vendor BLAS ships tuned binaries rather than
// compiling at call time. The module is loaded once per handle and per device,
// on first use, so a program that creates a handle and never calls a kernel
// pays nothing.

#include <grx/grxblas.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>   // readlink, for the executable-relative kernel search

#include "hgemm_abi.h"
#include "blas12_abi.h"
#include "sgemm_abi.h"

#ifdef GRXCP_ENABLE_NPU
#include "npu_c930.h"
#endif

namespace {

std::mutex  g_path_mutex;
std::string g_kernel_path;

// The first CONFIGURED source wins outright: if someone set a kernel path and
// the module is not there, that is an error, not an invitation to load a
// different build from somewhere else. Searching on past an explicit setting is
// how a developer ends up benchmarking a kernel they thought they had replaced.
// The unconfigured fallbacks exist so a build tree works with no setup at all.
std::vector<std::string> candidate_paths(const char* module_name) {
  std::vector<std::string> out;
  {
    std::lock_guard<std::mutex> lock(g_path_mutex);
    if (!g_kernel_path.empty()) return {g_kernel_path + "/" + module_name};
  }
  if (const char* env = std::getenv("GRXBLAS_KERNEL_PATH"))
    return {std::string(env) + "/" + module_name};

  char self[4096];
  const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n > 0) {
    self[n] = '\0';
    std::string dir(self);
    const size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos)
      out.push_back(dir.substr(0, slash) + "/" + module_name);
  }
  out.emplace_back(module_name);
  return out;
}

struct Context {
  grxStream_t   stream = nullptr;
  // ONE module for the whole library. Every .vxbin links at the same fixed
  // load address, so a second one cannot be resident at the same time -- see
  // kernels/all.cpp. The combined module is preferred; the scalar-only one is
  // the fallback for a device with no tensor unit.
  grxModule_t   module = nullptr;
  grxFunction_t sgemm_fn = nullptr;
  // The register-blocked sgemm, when the loaded image has one. Optional: an
  // older module still serves sgemm perfectly well, and refusing to load it
  // over a kernel the caller never asked for would be absurd.
  grxFunction_t sgemm_rb_fn = nullptr;
  // The 2D micro-tile, and the entry point that says what tile each blocked
  // kernel produces. Both optional in the same way sgemm_rb is.
  grxFunction_t sgemm_2d_fn = nullptr;
  // The wide micro-tile. Same body, a bigger tile: half a load per multiply-add
  // against the 2D tile's one, and a sixteenth of the threads instead of a
  // quarter.
  grxFunction_t sgemm_wide_fn = nullptr;
  // And the middle rung: 4 x 2, the widest tile that still fits in registers.
  grxFunction_t sgemm_mid_fn = nullptr;
  // Reported by sgemm_shape, never assumed. Zero means the module did not say,
  // and a kernel whose tile is unknown is not launched at all -- the host would
  // be sizing its grid from a guess. See sgemm_abi.h.
  int rb_rows = 0, td_rows = 0, td_cols = 0, wd_rows = 0, wd_cols = 0;
  int md_rows = 0, md_cols = 0;
  std::string   sgemm_path;   // which file actually got loaded
  // Level 1 and 2, resolved from the same module. Absent in an older module,
  // which is why they are looked up lazily and reported rather than assumed.
  grxFunction_t axpy_fn = nullptr;
  grxFunction_t scal_fn = nullptr;
  grxFunction_t gemv_fn = nullptr;
  grxCycleSlot* probe = nullptr;
  int           probe_capacity = 0;

  // Tensor path, resolved from the same module when it carries the entries.
  grxFunction_t hgemm_fn = nullptr;
  uint32_t      tile_m = 0, tile_n = 0, tile_k = 0, tile_smem = 0;
  uint32_t      tcu_types = 0;   // GRXBLAS_TENSOR_* the device build supports
  // The int8 sibling. Its tile is a DIFFERENT shape -- same m and n, twice the
  // depth -- so it gets its own geometry rather than a scaled copy of the fp16
  // one, for the same reason the fp16 geometry is asked for instead of assumed.
  grxFunction_t igemm_fn = nullptr;
  uint32_t      i8_tile_k = 0, i8_smem = 0;
  uint32_t      i8_block_m = 0, i8_block_n = 0;
  // What one warp produces per pass, which is a multiple of the tile: the
  // kernel blocks several tiles together to reuse a staged region.
  uint32_t      block_m = 0, block_n = 0;
  int           slot_a = 0, slot_b = 1;

  std::mutex    mutex;
};

// WHEN TO BLOCK, AND WITH WHICH KERNEL. The rule is
//
//     outputs = m * n * batchCount
//     outputs <  resident        ->  sgemm,     the reference
//     outputs <  16 * resident   ->  sgemm_2d,  a 2 x 2 tile per thread
//     otherwise                  ->  sgemm_4x2, a 4 x 2 tile per thread
//
// where resident = warpSize * maxWarpsPerMultiProcessor * multiProcessorCount,
// 64 on the configuration this was measured on. k does not appear. Neither does
// how m and n split to reach the output count -- which is not an assumption, it
// is the measurement: at 24, 32, 40, 48, 56, 64, 72, 80 and 96 outputs the
// speedup agrees to two decimals however the shape gets there (4x12, 8x6 and
// 12x4 all read 0.88, 0.91, 0.88).
//
// THE FAMILY, AND WHAT DECIDES BETWEEN ITS MEMBERS. Every blocked kernel here
// is the same body with a different tile, so they differ in exactly two things:
// how many memory operations the inner loop pays per multiply-add, and how many
// threads the launch has left. Both are countable, and the second one is only
// countable because the tile is compile-time.
//
//     kernel      tile   inner loop: fp   global loads   stack   mem/FMA
//     sgemm       1x1              1            2           0      2.00
//     sgemm_rb    4x1              4            5           0      1.25
//     sgemm_2d    2x2              4            4           0      1.00
//     sgemm_4x2   4x2              8            6           0      0.75
//     sgemm_4x4   4x4             16            8           7      0.94
//
// Counted from the disassembly of the shipped .vxbin, not from the source.
//
// SO THE LADDER PAYS UNTIL THE REGISTER FILE STOPS IT. 2.00 -> 1.25 -> 1.00 ->
// 0.75 tracks the measured speedup all the way. 4 x 4 breaks it: sixteen
// accumulators plus four A and four B values do not fit, its k loop is the only
// one that touches the stack, and seven spill accesses hand back almost exactly
// the load-count advantage the wider tile was built to have. 0.5 loads per
// multiply-add becomes 0.94 memory operations per multiply-add, and it measures
// like it -- 0.93x to 1.05x against the 2 x 2 tile with no trend, at every
// output count from 576 to 9216.
//
// That was a prediction before it was a result. 4 x 4 was built and measured
// first, failed, and the disassembly said why; 4 x 2 was then built to test
// whether register pressure was the reason, with two distinguishable outcomes.
// It does not spill, its ratio is the 0.75 the arithmetic asks for, and it wins.
//
// WHERE THE SECOND THRESHOLD IS. Bracketed, at k = 16, square shapes, 4 x 2
// against 2 x 2:
//
//     outputs   576    676    784    900   1024   1600   4096   9216
//     ratio    0.93   0.99   1.03   1.02   1.13   1.07   1.16   1.21
//
// A tie from about 676 to 900 and a solid win from 1024, which is 16 * resident
// and the point where a 4 x 2 launch has two full waves of warps. The threshold
// ships there: the ties below it cost nothing either way, and it is a quantity
// the device reports rather than a constant fitted to the tie band.
//
// WHERE THE FIRST ONE IS, and it is not the same kind of boundary. Blocking at
// all crosses over between 48 and 56 outputs -- 0.88 then 1.04 -- long before
// any tile fills the core. It ships at `resident` = 64 rather than at 56 because
// 56 is not a number the device reports and a fitted constant is what went wrong
// last time; the band it declines wins by 3-4% while the band below it LOSES by
// 12%, and being too eager costs about four times what being too shy does.
//
// A HYPOTHESIS ABOUT THE FIRST KNEE, RECORDED AS A HYPOTHESIS. The jump from
// 1.17 to 1.68 between 64 and 72 outputs sits exactly where the reference
// kernel stops fitting in the machine: it needs one thread per output, the core
// holds 64, so at 65 it needs a second wave while the blocked kernels are still
// inside one. That predicts another jump at 136 and none between 112 and 128.
// Measured: 1.87, 1.73, 1.84 across 112-128 and 2.12 at 136 -- the jump is
// there, it is much smaller, and 144 falls back to 2.00. The wave account fits
// the first knee well and the second only partly. The rule does not depend on it.
//
// WHY FIVE KERNELS SHIP AND THE RULE NAMES THREE. sgemm is the ORACLE:
// tests/libs/test_grxblas_rb.cpp runs every tuned kernel against it on the
// device over the same operands and requires agreement BIT FOR BIT, which is a
// far stronger statement than landing inside a tolerance. sgemm_rb is the
// CONTROL for the 2 x 2 tile: it produces the same four outputs per thread, so
// the two launch identical thread counts and the only difference between them
// is the load count -- without it, "the 2 x 2 tile wins because it loads less"
// would be an argument rather than a measurement. sgemm_4x4 is the CONTROL for
// the ladder: it is the evidence that the tile stops paying, and deleting it
// would turn a gated result into a remembered one. All three are re-measured on
// every tier-2 run.
//
// WHAT CAME BEFORE, kept because it is the reason these gates exist. The rule
// was once `k >= 16 || ceil(m/RM) >= warpSize`, fitted to five points with a
// coalescing story attached. The sweep disproved the story -- k never changes
// which kernel wins anywhere in range, and the boundary is not at m = 16 -- and
// the output-count rule that replaced it was then REVERTED for a commit,
// because the block appeared to get slower. It had not. Attention is four
// launches sharing one probe buffer and MCYCLE restarts at zero at every
// launch, so its "cost" was a maximum over four unrelated clocks
// (include/grx/grx_cycles.h). Every number above is a sum of per-launch spans.

// ---------------------------------------------------------------------------
// THE MEASUREMENT HOOKS. Not API: nothing in grxblas.h mentions them, no
// program should depend on them, and they exist so the rule above can be
// measured on the workload it was fitted to instead of on isolated GEMMs.
//
// WHY THE SELECTOR IS A CALL INDEX AND NOT A SHAPE. To find out what one
// stage's kernel choice costs the block, exactly one stage must change while
// the rest stay where they are. Shape cannot express that, because shape does
// not tell the stages apart: at S=8 attention's two GEMMs are BOTH 8x8x8, and
// at S=16 the qkv projection and attention's output GEMM are both 8x16x16. A
// shape filter would move two stages every time it moved one -- which is the
// confound tests/bench/block_sgemm.cpp exists to remove, so it cannot be built
// on top of it.
//
// The counter is process-wide rather than per handle because grxDNN issues its
// GEMMs through a handle of its own that the caller never holds. A counter on
// the caller's handle would number the block's stages with attention missing
// from the sequence.
std::atomic<uint64_t> g_sgemm_calls{0};

// Does an override env var apply to THIS call?
//
// Unset: no. Set to anything not beginning with '#': every call -- the meaning
// these have always had and the one ci/run_real.sh still passes as "=1". Set to
// "#3" or "#3,#7": only those calls, numbered from 0 in issue order.
//
// Parsed on every call rather than cached, so that a bench can setenv() between
// runs of the same process. Host-side work only: the spans these hooks are used
// to compare are device cycles between kernel entry and exit, which nothing
// here is inside of.
bool env_forces(const char* name, uint64_t call) {
  const char* v = std::getenv(name);
  if (!v) return false;
  if (v[0] != '#') return true;
  for (const char* p = v; *p; ) {
    while (*p == '#' || *p == ',' || *p == ' ') ++p;
    if (!*p) break;
    char* end = nullptr;
    const unsigned long long want = std::strtoull(p, &end, 10);
    if (end == p) break;               // malformed: stop, do not guess
    if (want == call) return true;
    p = end;
  }
  return false;
}

enum class SgemmKernel { kNaive, kRegisterBlocked, kTwoD, kMid, kWide };

const char* sgemm_kernel_name(SgemmKernel k) {
  switch (k) {
    case SgemmKernel::kNaive:           return "naive";
    case SgemmKernel::kRegisterBlocked: return "rb";
    case SgemmKernel::kTwoD:            return "2d";
    case SgemmKernel::kMid:             return "4x2";
    case SgemmKernel::kWide:            return "4x4";
  }
  return "?";
}

struct SgemmChoice {
  SgemmKernel kernel = SgemmKernel::kNaive;
  SgemmKernel rule   = SgemmKernel::kNaive;  // what the RULE alone would pick
  const char* why    = "rule";               // what actually picked it
};

// The one place the kernel is chosen. The launch path below and the trace both
// go through it, so a trace line cannot describe a kernel other than the one
// that ran -- which is the whole point of having a trace.
SgemmChoice decide_sgemm_kernel(const Context& ctx, const grxDeviceProp_t& prop,
                                int m, int n, int k, int batch, uint64_t call) {
  (void)k;
  SgemmChoice c;
  const long long resident = (long long)prop.warpSize *
                             prop.maxWarpsPerMultiProcessor *
                             prop.multiProcessorCount;
  const long long outputs = (long long)m * n * batch;
  const bool have_2d  = ctx.sgemm_2d_fn && ctx.td_rows > 0 && ctx.td_cols > 0;
  const bool have_mid = ctx.sgemm_mid_fn && ctx.md_rows > 0 && ctx.md_cols > 0;
  const bool have_rb  = ctx.sgemm_rb_fn && ctx.rb_rows > 0 && m >= ctx.rb_rows;

  // THE RULE, in output count. Two thresholds, both measured:
  //
  //     outputs <  resident        the reference kernel
  //     outputs <  16 * resident   the 2 x 2 micro-tile
  //     otherwise                  the 4 x 2 micro-tile
  //
  // The register-blocked kernel is reached only when the 2 x 2 tile is absent
  // from the module, with its own boundary, because that is the one measured
  // for it.
  if (outputs >= 16 * resident && have_mid)    c.rule = SgemmKernel::kMid;
  else if (outputs >= resident && have_2d)     c.rule = SgemmKernel::kTwoD;
  else if (outputs >= resident && have_mid)    c.rule = SgemmKernel::kMid;
  else if (outputs >= 2 * resident && have_rb) c.rule = SgemmKernel::kRegisterBlocked;

  // force-naive first, because the reference is the ORACLE: a test comparing a
  // tuned kernel against it must be able to reach it from any state.
  if (env_forces("GRXBLAS_SGEMM_NAIVE", call)) { c.why = "forced naive"; return c; }
  // THE WIDE TILE IS NOT IN THE RULE. Reachable only by asking, because
  // nothing has measured it yet -- the same staging the 2D tile went through,
  // and for the same reason: a kernel that ships on an argument about load
  // counts is a kernel that ships on an argument.
  if (ctx.sgemm_mid_fn && ctx.md_rows > 0 && ctx.md_cols > 0 &&
      env_forces("GRXBLAS_SGEMM_4X2", call)) {
    c.kernel = SgemmKernel::kMid;
    c.why = "forced 4x2";
    return c;
  }
  if (ctx.sgemm_wide_fn && ctx.wd_rows > 0 && ctx.wd_cols > 0 &&
      env_forces("GRXBLAS_SGEMM_4X4", call)) {
    c.kernel = SgemmKernel::kWide;
    c.why = "forced 4x4";
    return c;
  }
  if (have_2d && env_forces("GRXBLAS_SGEMM_2D", call)) {
    c.kernel = SgemmKernel::kTwoD;
    c.why = "forced 2d";
    return c;
  }
  if (env_forces("GRXBLAS_SGEMM_RB", call)) {
    if (!ctx.sgemm_rb_fn) { c.why = "module has no sgemm_rb"; return c; }
    if (ctx.rb_rows <= 0) { c.why = "module did not report its tile"; return c; }
    if (m < ctx.rb_rows)  { c.why = "m < RM"; return c; }
    c.kernel = SgemmKernel::kRegisterBlocked;
    c.why = "forced rb";
    return c;
  }

  c.kernel = c.rule;
  if (c.kernel == SgemmKernel::kNaive) {
    // Say WHICH reason, because "the rule said so" and "the module could not
    // tell me its tile" look identical in a trace and mean different things.
    if (outputs < resident)                   c.why = "rule";
    else if (!ctx.sgemm_2d_fn)                c.why = "module has no sgemm_2d";
    else if (ctx.td_rows <= 0)                c.why = "module did not report its tile";
    else                                      c.why = "rule";
  }
  return c;
}

// GRXBLAS_SGEMM_TRACE=<path>: what the library ACTUALLY DID, one line per call.
//
// tests/bench/block_cycles.cpp labels its results with the configuration it
// ASKED FOR and says in a comment that it cannot tell whether the request took
// effect -- the register-blocked kernel is an optional symbol lookup and falls
// back silently. A label that records the request is a label that survives the
// request being ignored. This records the answer instead.
//
// Buffered to memory and written once at exit, so that file I/O can never land
// between two launches and become part of what is being measured.
struct SgemmTraceLine {
  uint64_t call;
  int      m, n, k, batch, transa, transb;
  const char* kernel;   // which one ran: naive, rb, 2d
  const char* rule;     // what the rule alone would have picked
  // Whether a cycle probe was attached to THIS call. A consumer of the trace
  // needs it: a call made with no probe is inside no measured stage, so it has
  // no cost in a stage-sum, and anything that moves when it changes moved
  // through machine state rather than through the stage it is not in.
  bool     probed;
  unsigned grid, block;
  const char* why;
};

class SgemmTrace {
 public:
  ~SgemmTrace() {
    const char* path = std::getenv("GRXBLAS_SGEMM_TRACE");
    if (!path || lines_.empty()) return;
    std::FILE* f = std::fopen(path, "w");
    if (!f) return;
    std::fprintf(f,
                 "call,m,n,k,batch,transa,transb,kernel,rule,probed,grid,block,"
                 "why\n");
    for (const SgemmTraceLine& l : lines_)
      std::fprintf(f, "%llu,%d,%d,%d,%d,%d,%d,%s,%s,%d,%u,%u,%s\n",
                   (unsigned long long)l.call, l.m, l.n, l.k, l.batch,
                   l.transa, l.transb, l.kernel, l.rule, l.probed ? 1 : 0,
                   l.grid, l.block, l.why);
    std::fclose(f);
  }
  void add(const SgemmTraceLine& l) {
    if (!std::getenv("GRXBLAS_SGEMM_TRACE")) return;
    std::lock_guard<std::mutex> g(m_);
    lines_.push_back(l);
  }
 private:
  std::mutex m_;
  std::vector<SgemmTraceLine> lines_;
};

SgemmTrace g_sgemm_trace;

// One warp per block, so one slot per block. Kept as a function because the
// launch geometry below has to agree with it exactly, and two places computing
// the same thing from memory is how they stop agreeing.
int slots_for(int m, int n, int warp_size) {
  if (m <= 0 || n <= 0 || warp_size <= 0) return 0;
  const long long total = (long long)m * n;
  return (int)((total + warp_size - 1) / warp_size);
}

// Ask the module what tile each blocked kernel produces.
//
// The host sizes every blocked launch from these numbers, so getting them from
// the module rather than from a constant beside the launch is the difference
// between a mismatch that cannot happen and one that nothing checks. There WAS
// such a constant here -- `kSgemmRowsPerThread = 4`, with a comment asking the
// reader to keep it equal to RM in kernels/sgemm.cpp -- and a stale .vxbin is
// exactly the case it could not survive.
//
// Anything that goes wrong leaves the geometry at zero, and zero means the
// blocked kernels are not launched at all. Not a failure: sgemm still runs and
// still computes the right answer, which is the right way round. An older
// module that predates sgemm_shape lands here and is served by the reference.
void read_sgemm_shape(Context& ctx) {
  ctx.rb_rows = ctx.td_rows = ctx.td_cols = ctx.wd_rows = ctx.wd_cols = 0;
  ctx.md_rows = ctx.md_cols = 0;
  if (!ctx.module) return;

  grxFunction_t shape_fn = nullptr;
  if (grxModuleGetFunction(&shape_fn, ctx.module, "sgemm_shape") != grxSuccess)
    return;

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess || prop.warpSize <= 0)
    return;

  void* dshape = nullptr;
  const size_t bytes = GRXBLAS_SGEMM_SHAPE_COUNT * sizeof(uint32_t);
  if (grxMalloc(&dshape, bytes) != grxSuccess) return;
  grxMemset(dshape, 0, bytes);

  grxblas_sgemm_shape_args sargs{};
  sargs.abi_version = GRXBLAS_SGEMM_ABI_VERSION;
  sargs.out = (uint64_t)(uintptr_t)dshape;
  grxError_t e = grxLaunchFunction(shape_fn, dim3_t{1, 1, 1},
                                   dim3_t{(unsigned)prop.warpSize, 1, 1},
                                   &sargs, sizeof(sargs), 0, nullptr);
  if (e == grxSuccess) e = grxDeviceSynchronize();

  uint32_t shape[GRXBLAS_SGEMM_SHAPE_COUNT] = {0};
  if (e == grxSuccess)
    e = grxMemcpy(shape, dshape, bytes, grxMemcpyDefault);
  grxFree(dshape);
  if (e != grxSuccess) return;

  // A zero from a kernel that ran is the ABI check having refused: it returns
  // without writing, leaving the buffer as memset left it. Treated the same as
  // no entry point, because it means the same thing -- this module will not say
  // what it does.
  if (shape[GRXBLAS_SGEMM_SHAPE_RB_RM] == 0) return;
  ctx.rb_rows = (int)shape[GRXBLAS_SGEMM_SHAPE_RB_RM];
  // The 2D tile is allowed to be absent while sgemm_rb is present: a module can
  // carry one blocked kernel and not the other, and each is gated on its own
  // geometry rather than on the pair.
  if (shape[GRXBLAS_SGEMM_SHAPE_2D_RM] != 0 &&
      shape[GRXBLAS_SGEMM_SHAPE_2D_RN] != 0) {
    ctx.td_rows = (int)shape[GRXBLAS_SGEMM_SHAPE_2D_RM];
    ctx.td_cols = (int)shape[GRXBLAS_SGEMM_SHAPE_2D_RN];
  }
  if (shape[GRXBLAS_SGEMM_SHAPE_WIDE_RM] != 0 &&
      shape[GRXBLAS_SGEMM_SHAPE_WIDE_RN] != 0) {
    ctx.wd_rows = (int)shape[GRXBLAS_SGEMM_SHAPE_WIDE_RM];
    ctx.wd_cols = (int)shape[GRXBLAS_SGEMM_SHAPE_WIDE_RN];
  }
  if (shape[GRXBLAS_SGEMM_SHAPE_MID_RM] != 0 &&
      shape[GRXBLAS_SGEMM_SHAPE_MID_RN] != 0) {
    ctx.md_rows = (int)shape[GRXBLAS_SGEMM_SHAPE_MID_RM];
    ctx.md_cols = (int)shape[GRXBLAS_SGEMM_SHAPE_MID_RN];
  }
}

// Load the library's module, preferring the one that carries every kernel.
// Called with ctx.mutex held.
grxblasStatus_t ensure_module_locked(Context& ctx) {
  if (ctx.module) return GRXBLAS_STATUS_SUCCESS;

  // Order matters: the combined module carries sgemm too, so trying it first
  // means a device that can run the tensor path gets both entry points, and a
  // device that cannot falls back to the scalar-only image.
  //
  // grxlibs_kernels.vxbin FIRST, and it is the one that matters for a program
  // using more than one GRXCP library. Only ONE module can be resident at a
  // time -- every image links at STARTUP_ADDR, so loading a second returns an
  // address overlap (cuda_mapping.md 7.13). A program calling grxBLAS and
  // grxDNN would otherwise have the second library fail to load its kernels,
  // with an error that says nothing about the real cause. src/libs/
  // kernels_all.cpp builds the image that carries both libraries' entry points.
  //
  // That is necessary and NOT sufficient: both libraries still call
  // grxModuleLoad on it, and the second call overlaps the first. The runtime
  // hands back the resident module instead of loading it twice -- see the
  // reference count in src/runtime/module.cpp. Measured in
  // tests/libs/test_libs_together.cpp, which fails without either half.
  //
  // The library-specific images stay as fallbacks: a grxBLAS-only program does
  // not have to ship grxDNN's kernels, and a device with no tensor unit falls
  // back to the scalar-only one.
  static const char* const kModules[] = {"grxlibs_kernels.vxbin",
                                         "grxblas_kernels.vxbin",
                                         "grxblas_sgemm.vxbin"};
  grxError_t last = grxSuccess;
  for (const char* name : kModules) {
    for (const std::string& path : candidate_paths(name)) {
      grxModule_t mod = nullptr;
      last = grxModuleLoad(&mod, path.c_str());
      if (last != grxSuccess) continue;

      grxFunction_t fn = nullptr;
      if (grxModuleGetFunction(&fn, mod, "sgemm") != grxSuccess) {
        grxModuleUnload(mod);
        continue;
      }
      ctx.module     = mod;
      ctx.sgemm_fn   = fn;
      if (grxModuleGetFunction(&ctx.sgemm_rb_fn, mod, "sgemm_rb") != grxSuccess)
        ctx.sgemm_rb_fn = nullptr;
      if (grxModuleGetFunction(&ctx.sgemm_2d_fn, mod, "sgemm_2d") != grxSuccess)
        ctx.sgemm_2d_fn = nullptr;
      if (grxModuleGetFunction(&ctx.sgemm_wide_fn, mod, "sgemm_4x4") != grxSuccess)
        ctx.sgemm_wide_fn = nullptr;
      if (grxModuleGetFunction(&ctx.sgemm_mid_fn, mod, "sgemm_4x2") != grxSuccess)
        ctx.sgemm_mid_fn = nullptr;
      ctx.sgemm_path = path;
      // What tile each blocked kernel produces, asked of the module. A module
      // that cannot say leaves all three at zero, and read_sgemm_shape drops
      // the blocked kernels rather than sizing a grid from a guess.
      read_sgemm_shape(ctx);
      // Best effort: a module built before these existed still serves sgemm,
      // and the level-1/2 calls report NOT_SUPPORTED instead of the whole
      // library refusing to initialise.
      grxModuleGetFunction(&ctx.axpy_fn, mod, "saxpy");
      grxModuleGetFunction(&ctx.scal_fn, mod, "sscal");
      grxModuleGetFunction(&ctx.gemv_fn, mod, "sgemv");
      return GRXBLAS_STATUS_SUCCESS;
    }
  }

  // Say which file is missing, because "internal error" from a BLAS call is
  // among the least actionable messages a library can produce.
  std::fprintf(stderr,
               "grxblas: cannot load grxlibs_kernels.vxbin, "
               "grxblas_kernels.vxbin or\n"
               "         grxblas_sgemm.vxbin (last error: %s).\n"
               "         Set GRXBLAS_KERNEL_PATH or call grxblasSetKernelPath.\n",
               grxGetErrorString(last));
  return GRXBLAS_STATUS_NOT_INITIALIZED;
}

// Format a tensor-type mask for a human. Says "nothing" rather than printing
// an empty list, because an empty list reads like a formatting bug.
void describe_tensor_types(uint32_t mask, char* out, size_t cap) {
  static const struct { uint32_t bit; const char* name; } kNames[] = {
    {0x01u, "fp16"}, {0x02u, "tf32"}, {0x04u, "fp8"},
    {0x08u, "fp4"},  {0x10u, "int8"}, {0x20u, "int4"},
  };
  out[0] = '\0';
  size_t used = 0;
  for (const auto& n : kNames) {
    if (!(mask & n.bit)) continue;
    const int wrote = std::snprintf(out + used, cap - used, "%s%s",
                                    used ? ", " : "", n.name);
    if (wrote <= 0 || (size_t)wrote >= cap - used) break;
    used += (size_t)wrote;
  }
  if (used == 0) std::snprintf(out, cap, "no input types at all");
}

grxblasStatus_t ensure_sgemm(Context& ctx) {
  std::lock_guard<std::mutex> lock(ctx.mutex);
  return ensure_module_locked(ctx);
}

grxblasStatus_t from_grx(grxError_t e) {
  switch (e) {
    case grxSuccess:                   return GRXBLAS_STATUS_SUCCESS;
    case grxErrorMemoryAllocation:     return GRXBLAS_STATUS_ALLOC_FAILED;
    case grxErrorInvalidValue:         return GRXBLAS_STATUS_INVALID_VALUE;
    case grxErrorInvalidDevicePointer: return GRXBLAS_STATUS_INVALID_VALUE;
    case grxErrorNotSupported:         return GRXBLAS_STATUS_NOT_SUPPORTED;
    default:                           return GRXBLAS_STATUS_EXECUTION_FAILED;
  }
}

// The tensor kernel's tile shape is a property of the DEVICE BUILD, not a
// constant this file may assume: it follows from warp width and registers per
// fragment. So the module is asked, once, and the answer drives the grid and
// the descriptors. See include/grx/device/grx_wmma.h.
grxblasStatus_t ensure_hgemm(Context& ctx) {
  std::lock_guard<std::mutex> lock(ctx.mutex);
  if (ctx.hgemm_fn) return GRXBLAS_STATUS_SUCCESS;

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess)
    return GRXBLAS_STATUS_INTERNAL_ERROR;
  if (!(prop.capabilities & GRX_CAP_TENSOR_CORE) ||
      !(prop.capabilities & GRX_CAP_ASYNC_COPY))
    return GRXBLAS_STATUS_ARCH_MISMATCH;

  const grxblasStatus_t ms = ensure_module_locked(ctx);
  if (ms != GRXBLAS_STATUS_SUCCESS) return ms;

  grxFunction_t fn = nullptr, shape_fn = nullptr;
  if (grxModuleGetFunction(&fn, ctx.module, "hgemm_tcu") != grxSuccess ||
      grxModuleGetFunction(&shape_fn, ctx.module, "hgemm_tcu_shape") !=
          grxSuccess) {
    // The scalar-only module loaded. That is a configuration, not a failure:
    // the device can still do sgemm.
    std::fprintf(stderr,
                 "grxblas: %s has no tensor kernels; build "
                 "src/libs/grxblas/kernels/all.cpp to get them.\n",
                 ctx.sgemm_path.c_str());
    return GRXBLAS_STATUS_ARCH_MISMATCH;
  }

  void* dshape = nullptr;
  if (grxMalloc(&dshape, GRXBLAS_HGEMM_SHAPE_COUNT * sizeof(uint32_t)) !=
      grxSuccess)
    return GRXBLAS_STATUS_ALLOC_FAILED;
  grxMemset(dshape, 0, GRXBLAS_HGEMM_SHAPE_COUNT * sizeof(uint32_t));

  grxblas_hgemm_shape_args sargs{};
  sargs.abi_version = GRXBLAS_HGEMM_ABI_VERSION;
  sargs.out = (uint64_t)(uintptr_t)dshape;
  grxError_t e = grxLaunchFunction(shape_fn, dim3_t{1, 1, 1},
                                   dim3_t{(unsigned)prop.warpSize, 1, 1},
                                   &sargs, sizeof(sargs), 0, nullptr);
  if (e == grxSuccess) e = grxDeviceSynchronize();

  uint32_t shape[GRXBLAS_HGEMM_SHAPE_COUNT] = {0};
  if (e == grxSuccess)
    e = grxMemcpy(shape, dshape, sizeof(shape), grxMemcpyDefault);
  grxFree(dshape);
  if (e != grxSuccess) return from_grx(e);

  if (shape[GRXBLAS_HGEMM_SHAPE_M] == 0 || shape[GRXBLAS_HGEMM_SHAPE_N] == 0 ||
      shape[GRXBLAS_HGEMM_SHAPE_K] == 0)
    return GRXBLAS_STATUS_INTERNAL_ERROR;

  // The kernel reports the warp width it was COMPILED for. If that is not the
  // width this device has, the module was built for a different machine and
  // every fragment layout in it is wrong. See ci/README.md, "configuration
  // provenance".
  if (shape[GRXBLAS_HGEMM_SHAPE_WARP] != (uint32_t)prop.warpSize) {
    std::fprintf(stderr,
                 "grxblas: the tensor kernels were built for a warp width of "
                 "%u, device reports %d.\n",
                 shape[GRXBLAS_HGEMM_SHAPE_WARP], prop.warpSize);
    return GRXBLAS_STATUS_ARCH_MISMATCH;
  }

  ctx.hgemm_fn     = fn;
  ctx.tile_m       = shape[GRXBLAS_HGEMM_SHAPE_M];
  ctx.tile_n       = shape[GRXBLAS_HGEMM_SHAPE_N];
  ctx.tile_k       = shape[GRXBLAS_HGEMM_SHAPE_K];
  ctx.tile_smem    = shape[GRXBLAS_HGEMM_SHAPE_SMEM];
  ctx.tcu_types    = shape[GRXBLAS_HGEMM_SHAPE_TYPES];
  ctx.block_m      = shape[GRXBLAS_HGEMM_SHAPE_BLOCK_M];
  ctx.block_n      = shape[GRXBLAS_HGEMM_SHAPE_BLOCK_N];
  if (ctx.block_m == 0 || ctx.block_n == 0)
    return GRXBLAS_STATUS_INTERNAL_ERROR;

  // The int8 path, if this module and this device have one. Absent is a
  // configuration rather than a failure: the fp16 path still works, and
  // grxblasGemmEx refuses an int8 call with a message that says which types
  // the device does have.
  if ((ctx.tcu_types & GRXBLAS_TCU_INT8) != 0) {
    grxFunction_t ifn = nullptr, ishape = nullptr;
    if (grxModuleGetFunction(&ifn, ctx.module, "igemm_tcu") == grxSuccess &&
        grxModuleGetFunction(&ishape, ctx.module, "igemm_tcu_shape") ==
            grxSuccess) {
      void* dsh = nullptr;
      if (grxMalloc(&dsh, GRXBLAS_HGEMM_SHAPE_COUNT * sizeof(uint32_t)) ==
          grxSuccess) {
        grxMemset(dsh, 0, GRXBLAS_HGEMM_SHAPE_COUNT * sizeof(uint32_t));
        grxblas_hgemm_shape_args ia{};
        ia.abi_version = GRXBLAS_HGEMM_ABI_VERSION;
        ia.out = (uint64_t)(uintptr_t)dsh;
        uint32_t ish[GRXBLAS_HGEMM_SHAPE_COUNT] = {0};
        if (grxLaunchFunction(ishape, dim3_t{1, 1, 1},
                              dim3_t{(unsigned)prop.warpSize, 1, 1}, &ia,
                              sizeof(ia), 0, nullptr) == grxSuccess &&
            grxDeviceSynchronize() == grxSuccess &&
            grxMemcpy(ish, dsh, sizeof(ish), grxMemcpyDefault) == grxSuccess &&
            ish[GRXBLAS_HGEMM_SHAPE_K] != 0) {
          // Same m and n as fp16 or the two paths cannot share a blocking
          // scheme, and this is the place that would notice.
          if (ish[GRXBLAS_HGEMM_SHAPE_M] == ctx.tile_m &&
              ish[GRXBLAS_HGEMM_SHAPE_N] == ctx.tile_n) {
            ctx.igemm_fn   = ifn;
            ctx.i8_tile_k  = ish[GRXBLAS_HGEMM_SHAPE_K];
            ctx.i8_smem    = ish[GRXBLAS_HGEMM_SHAPE_SMEM];
            ctx.i8_block_m = ish[GRXBLAS_HGEMM_SHAPE_BLOCK_M];
            ctx.i8_block_n = ish[GRXBLAS_HGEMM_SHAPE_BLOCK_N];
          } else {
            std::fprintf(stderr,
                         "grxblas: the int8 tile is %ux%u and the fp16 tile is "
                         "%ux%u; they must share m and n. Ignoring int8.\n",
                         ish[GRXBLAS_HGEMM_SHAPE_M], ish[GRXBLAS_HGEMM_SHAPE_N],
                         ctx.tile_m, ctx.tile_n);
          }
        }
        grxFree(dsh);
      }
    }
  }
  return GRXBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// NPU C930 backend (compiled only when GRXCP_ENABLE_NPU is set)
// ---------------------------------------------------------------------------

#ifdef GRXCP_ENABLE_NPU

static npu_c930_device_t g_npu_dev;
static bool              g_npu_detected = false;
static std::once_flag    g_npu_once;

static void ensure_npu_detected() {
  std::call_once(g_npu_once, [] {
    g_npu_detected = npu_c930_detect(&g_npu_dev) != 0;
    if (g_npu_detected)
      std::fprintf(stderr, "grxblas: GRX930 NPU detected at 0x%08x\n",
                   NPU_C930_MMIO_BASE);
  });
}



// NPU GEMM path: INT8 in, INT32 out, via the C930 systolic array.
//
// The NPU's AXI4 DMA reads A/B from DDR and writes C back.
// All pointers are physical DDR addresses (no MMU on bare-metal).
//
// For column-major BLAS convention: A is m x k (lda >= m), B is k x n
// (ldb >= k), C is m x n (ldc >= m).  The NPU expects row-major packed
// INT8.  We pass the A/B/C pointers directly and let the NPU's DMA
// handle the layout — the caller must ensure the buffers are contiguous
// and row-major.
//
// NOTE: this assumes no transpose.  Transpose support would require
// physical copies into row-major buffers, which is a follow-up.
static grxblasStatus_t npu_gemm_path(
    int m, int n, int k,
    const float* alpha, const void* A, int lda,
    const void* B, int ldb,
    const float* beta, void* C, int ldc) {
  ensure_npu_detected();
  if (!g_npu_detected) return GRXBLAS_STATUS_NOT_SUPPORTED;

  // The NPU only does INT8→INT32.  Refuse non-unit alpha/beta for now
  // (the hardware has no alpha/beta scaling — C = A*B, not alpha*A*B + beta*C).
  if (*alpha != 1.0f || *beta != 0.0f) {
    static bool said = false;
    if (!said) {
      said = true;
      std::fprintf(stderr,
                   "grxblas: NPU path requires alpha=1, beta=0."
                   "  alpha=%.1f beta=%.1f not supported.\n",
                   *alpha, *beta);
    }
    return GRXBLAS_STATUS_NOT_SUPPORTED;
  }

  // Validate dimensions against NPU hardware limits.
  if (m > NPU_C930_MAX_M || n > NPU_C930_MAX_N || k > NPU_C930_MAX_K) {
    std::fprintf(stderr,
                 "grxblas: NPU dimensions M=%d N=%d K=%d exceed limits"
                 " (max %d/%d/%d).\n",
                 m, n, k, NPU_C930_MAX_M, NPU_C930_MAX_N, NPU_C930_MAX_K);
    return GRXBLAS_STATUS_NOT_SUPPORTED;
  }

  // The NPU expects contiguous row-major INT8.  The BLAS pointer IS the
  // DDR address on bare-metal (no MMU).  For lda > m, the leading dimension
  // padding means the data is not contiguous — we cannot pass it directly.
  if (lda != m || ldb != k || ldc != m) {
    static bool said = false;
    if (!said) {
      said = true;
      std::fprintf(stderr,
                   "grxblas: NPU path requires contiguous row-major"
                   " (lda= ldb= ldc= m).  lda=%d ldb=%d ldc=%d"
                   " not supported yet.\n",
                   lda, ldb, ldc);
    }
    return GRXBLAS_STATUS_NOT_SUPPORTED;
  }

  // THE ADDRESS REGISTERS ARE 32 BITS. A_BASE/B_BASE/C_BASE are 32-bit MMIO
  // words and a device pointer is 64. Casting one to the other truncates in
  // silence and the DMA then reads whatever lives at the low half: a wrong
  // answer with no error, which is the failure class this project bans. Refused
  // by name instead, so the caller learns the buffer is out of the engine's
  // reach rather than that its GEMM is mysteriously wrong.
  const void* bufs[3] = {A, B, C};
  const char* names[3] = {"A", "B", "C"};
  for (int i = 0; i < 3; ++i) {
    const uint64_t addr = (uint64_t)(uintptr_t)bufs[i];
    if (addr > 0xFFFFFFFFull) {
      std::fprintf(stderr,
                   "grxblas: NPU %s is at 0x%llx, past the 32-bit reach of the"
                   " engine's base registers.\n",
                   names[i], (unsigned long long)addr);
      return GRXBLAS_STATUS_NOT_SUPPORTED;
    }
  }

  // Launch the GEMM through the NPU MMIO interface.
  const uint32_t a_addr = (uint32_t)(uintptr_t)A;
  const uint32_t b_addr = (uint32_t)(uintptr_t)B;
  const uint32_t c_addr = (uint32_t)(uintptr_t)C;

  int rc = npu_c930_gemm(&g_npu_dev, m, n, k, a_addr, b_addr, c_addr);
  if (rc != 0) return GRXBLAS_STATUS_EXECUTION_FAILED;
  return GRXBLAS_STATUS_SUCCESS;
}

#endif  // GRXCP_ENABLE_NPU

// THE DISPATCH RULE, in one place, so that asking and doing cannot disagree.
//
// The current device decides. Nothing here consults a preference, and nothing
// here falls back to another engine: a call the current device's engine cannot
// take is refused, because the alternative is the same source line running on
// different silicon depending on state set somewhere else.
//
// out_device reports the index the decision was made ABOUT. It is the current
// device or this function is wrong, and it is reported rather than assumed so a
// test can say which -- the predecessor of this code passed grxGetDevice(NULL),
// whose return value is an ERROR CODE, and so decided about device 1 forever.
grxblasEngine_t decide_gemm_engine(grxDataType_t Atype, grxDataType_t Btype,
                                   grxDataType_t Ctype, int* out_device) {
  int index = -1;
  if (grxGetDevice(&index) != grxSuccess) return GRXBLAS_ENGINE_NONE;
  if (out_device) *out_device = index;

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, index) != grxSuccess)
    return GRXBLAS_ENGINE_NONE;

  const bool int8 = (Atype == GRX_R_8I && Btype == GRX_R_8I &&
                     Ctype == GRX_R_32I);

  if (prop.deviceType == GRX_DEVICE_TYPE_NPU) {
    // A GEMM-only device has exactly one engine and it does INT8. Anything
    // else has nowhere to go ON THIS DEVICE, and the GPU is not an answer to
    // a question about this one.
#ifdef GRXCP_ENABLE_NPU
    return int8 ? GRXBLAS_ENGINE_NPU_C930 : GRXBLAS_ENGINE_NONE;
#else
    // The device could not have been enumerated without the flag; if it was,
    // saying NONE is the only honest answer.
    (void)int8;
    return GRXBLAS_ENGINE_NONE;
#endif
  }

  // Which pairings the tensor unit accepts is a property of the loaded module,
  // not of the routing, so that stays where it is and this reports the engine.
  return GRXBLAS_ENGINE_GPU_TENSOR;
}

}  // namespace

extern "C" {

const char* grxblasGetEngineString(grxblasEngine_t engine) {
  switch (engine) {
    case GRXBLAS_ENGINE_NONE:       return "none (the call would be refused)";
    case GRXBLAS_ENGINE_GPU_TENSOR: return "GRX-G100 tensor unit";
    case GRXBLAS_ENGINE_NPU_C930:   return "GRX930 c930 NPU";
  }
  return "unknown engine";
}

grxblasStatus_t grxblasGetGemmEngine(grxblasHandle_t handle,
                                     int m, int n, int k,
                                     grxDataType_t Atype, grxDataType_t Btype,
                                     grxDataType_t Ctype,
                                     grxblasEngine_t* engine, int* device) {
  if (!handle || !engine) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m <= 0 || n <= 0 || k <= 0) return GRXBLAS_STATUS_INVALID_VALUE;
  int index = -1;
  *engine = decide_gemm_engine(Atype, Btype, Ctype, &index);
  if (device) *device = index;
  return GRXBLAS_STATUS_SUCCESS;
}

const char* grxblasGetStatusString(grxblasStatus_t s) {
  switch (s) {
    case GRXBLAS_STATUS_SUCCESS:          return "success";
    case GRXBLAS_STATUS_NOT_INITIALIZED:  return "library not initialized (kernels not found?)";
    case GRXBLAS_STATUS_ALLOC_FAILED:     return "allocation failed";
    case GRXBLAS_STATUS_INVALID_VALUE:    return "invalid argument";
    case GRXBLAS_STATUS_ARCH_MISMATCH:    return "device lacks a required capability";
    case GRXBLAS_STATUS_EXECUTION_FAILED: return "kernel execution failed";
    case GRXBLAS_STATUS_NOT_SUPPORTED:    return "not supported";
    case GRXBLAS_STATUS_INTERNAL_ERROR:   return "internal error";
  }
  return "unknown status";
}

grxblasStatus_t grxblasCreate(grxblasHandle_t* handle) {
  if (!handle) return GRXBLAS_STATUS_INVALID_VALUE;
  *handle = reinterpret_cast<grxblasHandle_t>(new Context());
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasDestroy(grxblasHandle_t handle) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  auto* ctx = reinterpret_cast<Context*>(handle);
  if (ctx->module) grxModuleUnload(ctx->module);
  delete ctx;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSetStream(grxblasHandle_t handle, grxStream_t stream) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Context*>(handle)->stream = stream;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGetStream(grxblasHandle_t handle, grxStream_t* stream) {
  if (!handle || !stream) return GRXBLAS_STATUS_INVALID_VALUE;
  *stream = reinterpret_cast<Context*>(handle)->stream;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSetKernelPath(const char* path) {
  std::lock_guard<std::mutex> lock(g_path_mutex);
  g_kernel_path = path ? path : "";
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSetTensorMapSlots(grxblasHandle_t handle,
                                         int slotA, int slotB) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  int slots = 0;
  if (grxTensorMapGetSlotCount(&slots, 0) != grxSuccess)
    return GRXBLAS_STATUS_INTERNAL_ERROR;
  if (slotA < 0 || slotB < 0 || slotA >= slots || slotB >= slots ||
      slotA == slotB)
    return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->slot_a = slotA;
  ctx->slot_b = slotB;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGetTensorTypes(grxblasHandle_t handle,
                                      unsigned* typeMask) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!typeMask) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_hgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  *typeMask = (unsigned)ctx->tcu_types;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGetTensorTile(grxblasHandle_t handle, int* m, int* n,
                                     int* k) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!m || !n || !k) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_hgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  *m = (int)ctx->tile_m;
  *n = (int)ctx->tile_n;
  *k = (int)ctx->tile_k;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGemmEx(grxblasHandle_t handle,
                              grxblasOperation_t transa,
                              grxblasOperation_t transb,
                              int m, int n, int k,
                              const float* alpha,
                              const void* A, grxDataType_t Atype, int lda,
                              const void* B, grxDataType_t Btype, int ldb,
                              const float* beta,
                              void* C, grxDataType_t Ctype, int ldc) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || k < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return GRXBLAS_STATUS_SUCCESS;
  if (k > 0 && (!A || !B)) return GRXBLAS_STATUS_INVALID_VALUE;
  if (!C) return GRXBLAS_STATUS_INVALID_VALUE;

  // Refused rather than emulated. A fallback to the scalar kernel here would
  // report success for a call the tensor path cannot do, and the caller would
  // read the resulting speed as the tensor unit's.
  // Leading-dimension rules follow op(), not the logical shape: a transposed A
  // is STORED k x m, so its leading dimension bounds k rather than m.
  const bool ta = (transa == GRXBLAS_OP_T);
  const bool tb = (transb == GRXBLAS_OP_T);
  const int min_lda = ta ? k : m;
  const int min_ldb = tb ? n : k;
  if (lda < min_lda || ldb < min_ldb || ldc < m)
    return GRXBLAS_STATUS_INVALID_VALUE;

  // ------------------------------------------------------------------
  // NPU C930 dispatch (Phase 7 backend).
  //
  // If the current device is a GRX930 NPU and the type pairing is INT8,
  // route through the NPU's MMIO GEMM path instead of the GPU tensor
  // path.  The NPU has no .vxbin modules, no tensor maps, and no SIMT
  // pipeline — it is a systolic-array accelerator that programs its
  // registers over MMIO and runs the GEMM autonomously via DMA.
  //
  // This check happens BEFORE ensure_hgemm() to avoid loading the
  // expensive GPU tensor kernel module when the NPU path is taken.
  // ------------------------------------------------------------------
  //
  // ROUTED THROUGH THE SAME DECISION grxblasGetGemmEngine reports, so the
  // answer a caller can ask for cannot drift away from what actually happens.
  const grxblasEngine_t engine = decide_gemm_engine(Atype, Btype, Ctype, nullptr);
  if (engine == GRXBLAS_ENGINE_NONE) return GRXBLAS_STATUS_NOT_SUPPORTED;
  if (engine == GRXBLAS_ENGINE_NPU_C930) {
#ifdef GRXCP_ENABLE_NPU
    return npu_gemm_path(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
#else
    return GRXBLAS_STATUS_NOT_SUPPORTED;
#endif
  }

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_hgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;

  // The type check comes AFTER the module is up, so the refusal can say what
  // this device actually has instead of only what it does not. "not supported"
  // with no further information is how a caller ends up assuming the whole
  // tensor path is missing when one type is.
  const bool fp16_pair = (Atype == GRX_R_16F && Btype == GRX_R_16F &&
                          Ctype == GRX_R_32F);
  const bool int8_pair = (Atype == GRX_R_8I && Btype == GRX_R_8I &&
                          Ctype == GRX_R_32I);
  if (!fp16_pair && !(int8_pair && ctx->igemm_fn != nullptr)) {
    static bool said = false;
    if (!said) {
      said = true;
      char have[128];
      describe_tensor_types(ctx->tcu_types, have, sizeof(have));
      std::fprintf(stderr,
                   "grxblas: grxblasGemmEx does fp16 in / fp32 out, and int8 in "
                   "/ int32 out where the\n         device has int8. This one "
                   "accepts %s. Types are a build-time choice on\n"
                   "         GRX-G100 -- query them with "
                   "grxblasGetTensorTypes.\n",
                   have);
    }
    return GRXBLAS_STATUS_NOT_SUPPORTED;
  }

  // One signature cannot carry two scalar types, so alpha and beta are floats
  // for both pairings. For the integer one they have to BE integers: rounding
  // 2.5 to 2 would be a wrong answer the caller never sees happen.
  if (int8_pair) {
    const float a = *alpha, b = *beta;
    if (a != (float)(int32_t)a || b != (float)(int32_t)b)
      return GRXBLAS_STATUS_INVALID_VALUE;
  }

  const uint32_t elem_bytes = int8_pair ? 1u : 2u;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  // The kernel walks BLOCKS, each several WMMA tiles wide and tall, so the
  // descriptor tiles have to cover a block rather than a tile.
  const uint32_t tk = int8_pair ? ctx->i8_tile_k : ctx->tile_k;
  const uint32_t bm = int8_pair ? ctx->i8_block_m : ctx->block_m;
  const uint32_t bn = int8_pair ? ctx->i8_block_n : ctx->block_n;
  const uint32_t m_tiles = ((uint32_t)m + bm - 1) / bm;
  const uint32_t n_tiles = ((uint32_t)n + bn - 1) / bn;
  const uint32_t k_steps = ((uint32_t)k + tk - 1) / tk;

  // The descriptors, and what a transpose does to them.
  //
  // The kernel wants one thing from each operand regardless of how it is
  // stored: sA holding op(A)'s tile ROW major with leading dimension tile::k,
  // and sB holding op(B)'s tile COLUMN major with the same leading dimension.
  // That is what matrix_a and matrix_b col_major fragments read.
  //
  // Dimension 0 of a descriptor is the source's contiguous direction, and the
  // destination layout says whether the engine keeps that order (RowMajor:
  // dest[e1*tile0 + e0]) or transposes it (KMajor: dest[e0*tile1 + e1]). So
  // transposing an operand does three things together, and they are three
  // faces of one change:
  //
  //   the extents swap        (m,k) <-> (k,m)     -- what is stored
  //   the tile extents swap   (bm,tk) <-> (tk,bm)
  //   the destination layout flips                -- to land on the same sA
  //
  // and the kernel swaps its coordinate pair to match, because coordinates are
  // per block and computed on the device.
  //
  //   A, N: stored m x k, dim0 = m. KMajor transposes it into row-major sA.
  //   A, T: stored k x m, dim0 = k. RowMajor already gives row-major sA.
  //   B, N: stored k x n, dim0 = k. RowMajor gives column-major sB.
  //   B, T: stored n x k, dim0 = n. KMajor gives column-major sB.
  grxTensorMapDesc_t da{};
  da.slot = ctx->slot_a;
  da.base = const_cast<void*>(A);
  da.rank = 2;
  da.strideBytes[0] = (unsigned)lda * elem_bytes;
  da.elementBytes = elem_bytes;
  if (!ta) {
    da.size[0] = (unsigned)m;            da.size[1] = (unsigned)(k ? k : 1);
    da.tile[0] = bm;                     da.tile[1] = tk;
    da.layout  = grxTensorMapLayoutKMajor;
  } else {
    da.size[0] = (unsigned)(k ? k : 1);  da.size[1] = (unsigned)m;
    da.tile[0] = tk;                     da.tile[1] = bm;
    da.layout  = grxTensorMapLayoutRowMajor;
  }

  grxTensorMapDesc_t db{};
  db.slot = ctx->slot_b;
  db.base = const_cast<void*>(B);
  db.rank = 2;
  db.strideBytes[0] = (unsigned)ldb * elem_bytes;
  db.elementBytes = elem_bytes;
  if (!tb) {
    db.size[0] = (unsigned)(k ? k : 1);  db.size[1] = (unsigned)n;
    db.tile[0] = tk;                     db.tile[1] = bn;
    db.layout  = grxTensorMapLayoutRowMajor;
  } else {
    db.size[0] = (unsigned)n;            db.size[1] = (unsigned)(k ? k : 1);
    db.tile[0] = bn;                     db.tile[1] = tk;
    db.layout  = grxTensorMapLayoutKMajor;
  }

  if (k > 0) {
    e = grxTensorMapProgramAsync(&da, ctx->stream);
    if (e != grxSuccess) return from_grx(e);
    e = grxTensorMapProgramAsync(&db, ctx->stream);
    if (e != grxSuccess) return from_grx(e);
  }

  grxblas_hgemm_args args{};
  args.abi_version = GRXBLAS_HGEMM_ABI_VERSION;
  args.m = (uint32_t)m; args.n = (uint32_t)n; args.k = (uint32_t)k;
  args.ldc = (uint32_t)ldc;
  args.m_tiles = m_tiles;
  args.tiles   = m_tiles * n_tiles;
  args.k_steps = k_steps;
  args.slot_a = (uint32_t)ctx->slot_a;
  args.slot_b = (uint32_t)ctx->slot_b;
  args.transa = ta ? GRXBLAS_ABI_OP_T : GRXBLAS_ABI_OP_N;
  args.transb = tb ? GRXBLAS_ABI_OP_T : GRXBLAS_ABI_OP_N;
  args.barrier = 0;
  args.alpha = *alpha; args.beta = *beta;
  args.c = (uint64_t)(uintptr_t)C;

  // ONE block, with as many warps as there is work for. Not the natural shape
  // -- that is one CTA per tile -- but a second CTA issuing a tensor
  // instruction deadlocks the current SimX model, so the kernel walks the
  // tiles instead. See the header of kernels/hgemm_tcu.cpp and
  // tests/repro/tcu_multi_cta/. On a one-SM configuration this costs nothing;
  // it is a ceiling everywhere else, and it comes out when the model is fixed.
  const uint32_t tiles = m_tiles * n_tiles;
  uint32_t warps = (uint32_t)(prop.maxThreadsPerBlock / prop.warpSize);
  if (warps == 0) warps = 1;
  if (warps > tiles) warps = tiles;

  if (ctx->probe) {
    if ((uint32_t)ctx->probe_capacity < warps)
      return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }

  e = grxLaunchFunction(int8_pair ? ctx->igemm_fn : ctx->hgemm_fn, dim3_t{1, 1, 1},
                        dim3_t{warps * (unsigned)prop.warpSize, 1, 1}, &args,
                        sizeof(args),
                        (size_t)(int8_pair ? ctx->i8_smem : ctx->tile_smem) * warps,
                        ctx->stream);
  return from_grx(e);
}

grxblasStatus_t grxblasSetCycleProbe(grxblasHandle_t handle,
                                     grxCycleSlot* slots, int capacity) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (slots && capacity <= 0) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->probe          = slots;
  ctx->probe_capacity = slots ? capacity : 0;
  return GRXBLAS_STATUS_SUCCESS;
}

int grxblasCycleSlotsNeeded(grxblasHandle_t handle, int m, int n) {
  (void)handle;
  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess) return 0;
  return slots_for(m, n, prop.warpSize);
}

grxblasStatus_t grxblasGetLoadedKernelPath(grxblasHandle_t handle,
                                           const char** path) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!path) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *path = ctx->sgemm_path.empty() ? nullptr : ctx->sgemm_path.c_str();
  return GRXBLAS_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Level 1 and level 2
// ---------------------------------------------------------------------------

grxblasStatus_t grxblasSaxpy(grxblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (n < 0)   return GRXBLAS_STATUS_INVALID_VALUE;
  if (n == 0)  return GRXBLAS_STATUS_SUCCESS;
  if (!x || !y) return GRXBLAS_STATUS_INVALID_VALUE;
  if (incx == 0 || incy == 0) return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  if (!ctx->axpy_fn) return GRXBLAS_STATUS_NOT_SUPPORTED;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  grxblas_axpy_args args{};
  args.abi_version = GRXBLAS_BLAS12_ABI_VERSION;
  args.n = (uint32_t)n;
  args.incx = incx; args.incy = incy;
  args.alpha = *alpha;
  args.x = (uint64_t)(uintptr_t)x;
  args.y = (uint64_t)(uintptr_t)y;

  const unsigned block = (unsigned)prop.warpSize;
  const unsigned grid  = ((unsigned)n + block - 1) / block;
  if (ctx->probe) {
    if (ctx->probe_capacity < (int)grid) return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }
  e = grxLaunchFunction(ctx->axpy_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), 0, ctx->stream);
  return from_grx(e);
}

grxblasStatus_t grxblasSscal(grxblasHandle_t handle, int n, const float* alpha,
                             float* x, int incx) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (n < 0)   return GRXBLAS_STATUS_INVALID_VALUE;
  if (n == 0)  return GRXBLAS_STATUS_SUCCESS;
  if (!x)      return GRXBLAS_STATUS_INVALID_VALUE;
  // BLAS itself requires a positive increment for scal, so this is the
  // reference behaviour rather than a restriction of ours.
  if (incx <= 0) return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  if (!ctx->scal_fn) return GRXBLAS_STATUS_NOT_SUPPORTED;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  grxblas_scal_args args{};
  args.abi_version = GRXBLAS_BLAS12_ABI_VERSION;
  args.n = (uint32_t)n;
  args.incx = incx;
  args.alpha = *alpha;
  args.x = (uint64_t)(uintptr_t)x;

  const unsigned block = (unsigned)prop.warpSize;
  const unsigned grid  = ((unsigned)n + block - 1) / block;
  if (ctx->probe) {
    if (ctx->probe_capacity < (int)grid) return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }
  e = grxLaunchFunction(ctx->scal_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), 0, ctx->stream);
  return from_grx(e);
}

grxblasStatus_t grxblasSgemv(grxblasHandle_t handle, grxblasOperation_t trans,
                             int m, int n, const float* alpha,
                             const void* A, int lda,
                             const void* x, int incx,
                             const float* beta,
                             void* y, int incy) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return GRXBLAS_STATUS_SUCCESS;
  if (!A || !x || !y)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (incx == 0 || incy == 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (lda < m) return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  if (!ctx->gemv_fn) return GRXBLAS_STATUS_NOT_SUPPORTED;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  // m and n describe A as stored. op(A) is m x n untransposed and n x m
  // transposed, so the output length and the reduction length swap.
  const bool tr = (trans == GRXBLAS_OP_T);
  const uint32_t rows  = tr ? (uint32_t)n : (uint32_t)m;
  const uint32_t depth = tr ? (uint32_t)m : (uint32_t)n;

  grxblas_gemv_args args{};
  args.abi_version = GRXBLAS_BLAS12_ABI_VERSION;
  args.m = (uint32_t)m; args.n = (uint32_t)n;
  args.lda = (uint32_t)lda;
  args.trans = tr ? GRXBLAS_ABI_OP_T : GRXBLAS_ABI_OP_N;
  args.rows = rows; args.depth = depth;
  args.incx = incx; args.incy = incy;
  args.alpha = *alpha; args.beta = *beta;
  args.a = (uint64_t)(uintptr_t)A;
  args.x = (uint64_t)(uintptr_t)x;
  args.y = (uint64_t)(uintptr_t)y;

  // Two launch shapes for two traversals -- see the comment at the top of
  // kernels/blas12.cpp. Untransposed: one thread per output. Transposed: one
  // warp per output, so the lanes read down a column together.
  const unsigned warp = (unsigned)prop.warpSize;
  const unsigned block = warp;
  const unsigned grid  = tr ? rows : ((rows + block - 1) / block);
  if (ctx->probe) {
    if (ctx->probe_capacity < (int)grid) return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }
  e = grxLaunchFunction(ctx->gemv_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), 0, ctx->stream);
  return from_grx(e);
}

// One implementation, with the unbatched call as the batch-of-one case.
//
// Two entry points sharing a body rather than two bodies: the validation rules
// are identical and the launch differs by one grid dimension, and the way a
// batched GEMM usually goes wrong is that its unbatched twin drifted.
static grxblasStatus_t sgemm_batched(grxblasHandle_t handle,
                                     grxblasOperation_t transa,
                                     grxblasOperation_t transb,
                                     int m, int n, int k,
                                     const float* alpha,
                                     const void* A, int lda, long long strideA,
                                     const void* B, int ldb, long long strideB,
                                     const float* beta,
                                     void* C, int ldc, long long strideC,
                                     int batch) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || k < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (batch < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0 || batch == 0) return GRXBLAS_STATUS_SUCCESS;
  if (k > 0 && (!A || !B)) return GRXBLAS_STATUS_INVALID_VALUE;
  if (!C) return GRXBLAS_STATUS_INVALID_VALUE;

  // Leading-dimension rules, checked rather than trusted: a too-small ld reads
  // or writes outside the caller's allocation, and the allocator would not
  // necessarily catch it because the address is still inside some other live
  // buffer.
  const int min_lda = (transa == GRXBLAS_OP_N) ? m : k;
  const int min_ldb = (transb == GRXBLAS_OP_N) ? k : n;
  if (lda < min_lda || ldb < min_ldb || ldc < m)
    return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  grxblas_sgemm_args args{};
  args.abi_version = GRXBLAS_SGEMM_ABI_VERSION;
  args.m = (uint32_t)m; args.n = (uint32_t)n; args.k = (uint32_t)k;
  args.lda = (uint32_t)lda; args.ldb = (uint32_t)ldb; args.ldc = (uint32_t)ldc;
  args.transa = (uint32_t)transa; args.transb = (uint32_t)transb;
  args.alpha = *alpha; args.beta = *beta;
  args.a = (uint64_t)(uintptr_t)A;
  args.b = (uint64_t)(uintptr_t)B;
  args.c = (uint64_t)(uintptr_t)C;
  args.batch = (uint32_t)batch;
  args.stride_a = (int64_t)strideA;
  args.stride_b = (int64_t)strideB;
  args.stride_c = (int64_t)strideC;

  // The rule and both overrides live in decide_sgemm_kernel, above. They are
  // there rather than here so that the trace reads the SAME function the launch
  // does: a trace that recomputed the choice could describe a kernel other than
  // the one that ran, which is the failure it was built to end.
  //
  // The index is taken here, after validation and before the decision, so that
  // two runs of the same program number the same calls the same way -- which is
  // what makes "#3" mean one stage of the block.
  const uint64_t call = g_sgemm_calls.fetch_add(1, std::memory_order_relaxed);
  const SgemmChoice choice =
      decide_sgemm_kernel(*ctx, prop, m, n, k, batch, call);
  // How many THREADS the chosen kernel needs -- one per tile it produces, not
  // one per output. Every one of these divisors comes from what the module
  // reported, so the grid cannot cover a different number of outputs than the
  // kernel slices.
  const unsigned block = (unsigned)prop.warpSize;
  size_t tiles = (size_t)m * (size_t)n;
  switch (choice.kernel) {
    case SgemmKernel::kRegisterBlocked:
      tiles = (size_t)((m + ctx->rb_rows - 1) / ctx->rb_rows) * (size_t)n;
      break;
    case SgemmKernel::kTwoD:
      tiles = (size_t)((m + ctx->td_rows - 1) / ctx->td_rows) *
              (size_t)((n + ctx->td_cols - 1) / ctx->td_cols);
      break;
    case SgemmKernel::kMid:
      tiles = (size_t)((m + ctx->md_rows - 1) / ctx->md_rows) *
              (size_t)((n + ctx->md_cols - 1) / ctx->md_cols);
      break;
    case SgemmKernel::kWide:
      tiles = (size_t)((m + ctx->wd_rows - 1) / ctx->wd_rows) *
              (size_t)((n + ctx->wd_cols - 1) / ctx->wd_cols);
      break;
    case SgemmKernel::kNaive:
      break;
  }
  const unsigned outputs = (unsigned)tiles;
  const unsigned grid = (outputs + block - 1) / block;

  if (ctx->probe) {
    // The probe is one slot per block, and the grid is now two-dimensional.
    // Sized against the LAUNCH rather than against m*n, because the blocked
    // kernel launches fewer blocks and a capacity check that ignored that
    // would refuse probes that are perfectly large enough.
    if (ctx->probe_capacity < (int)grid * batch)
      return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }

  g_sgemm_trace.add(SgemmTraceLine{call, m, n, k, batch, (int)transa,
                                   (int)transb, sgemm_kernel_name(choice.kernel),
                                   sgemm_kernel_name(choice.rule),
                                   ctx->probe != nullptr, grid, block,
                                   choice.why});

  grxFunction_t fn_to_run = ctx->sgemm_fn;
  if (choice.kernel == SgemmKernel::kRegisterBlocked) fn_to_run = ctx->sgemm_rb_fn;
  else if (choice.kernel == SgemmKernel::kTwoD)       fn_to_run = ctx->sgemm_2d_fn;
  else if (choice.kernel == SgemmKernel::kMid)       fn_to_run = ctx->sgemm_mid_fn;
  else if (choice.kernel == SgemmKernel::kWide)      fn_to_run = ctx->sgemm_wide_fn;

  e = grxLaunchFunction(fn_to_run,
                        dim3_t{grid, (unsigned)batch, 1},
                        dim3_t{block, 1, 1}, &args, sizeof(args),
                        /*sharedMem=*/0, ctx->stream);
  return from_grx(e);
}

grxblasStatus_t grxblasSgemm(grxblasHandle_t handle,
                             grxblasOperation_t transa,
                             grxblasOperation_t transb,
                             int m, int n, int k,
                             const float* alpha,
                             const void* A, int lda,
                             const void* B, int ldb,
                             const float* beta,
                             void* C, int ldc) {
  return sgemm_batched(handle, transa, transb, m, n, k, alpha, A, lda, 0,
                       B, ldb, 0, beta, C, ldc, 0, 1);
}

grxblasStatus_t grxblasSgemmStridedBatched(
    grxblasHandle_t handle, grxblasOperation_t transa,
    grxblasOperation_t transb, int m, int n, int k, const float* alpha,
    const void* A, int lda, long long strideA,
    const void* B, int ldb, long long strideB,
    const float* beta, void* C, int ldc, long long strideC, int batchCount) {
  return sgemm_batched(handle, transa, transb, m, n, k, alpha, A, lda, strideA,
                       B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

}  // extern "C"
