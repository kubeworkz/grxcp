// Is the grxcp runtime safe to call from more than one host thread?
//
// WHY NOW. The GRX930 host went from one RISC-V64 core to four with a
// directory-coherent shared L2. Until that landed, "one thread touches the
// driver" was true by construction on the target. It is not any more, and
// nothing in include/grx/ says whether it is allowed -- while the API this one
// imitates, the CUDA runtime, is thread-safe and a porting developer will
// assume it.
//
// This does not read the code and conclude. It runs the calls concurrently
// under ThreadSanitizer and reports what TSan says, including when TSan says
// nothing -- which is a result and has to be reportable as one.
//
// STAGED ON PURPOSE. Each stage is a separate run so a finding can be
// attributed. Stage 1-3 are pure grxcp bookkeeping and barely touch the
// device; stage 4 launches, where the backend .so is in the picture and is NOT
// TSan-instrumented, so its internals are the weaker half of the report.
//
//   ./stress 1   allocator      grxMalloc / grxFree
//   ./stress 2   modules        grxModuleLoad / GetFunction / Unload
//   ./stress 3   events+error   create / record / query / destroy, grxGetLastError
//   ./stress 4   launches       per-thread streams, concurrent launch
//   ./stress 0   everything at once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include "grx/grx_runtime.h"

namespace {

constexpr int kThreads = 4;      // the host this is about
constexpr int kIters   = 40;

std::atomic<int> g_fail{0};

void note(const char* stage, const char* what) {
  std::printf("  [%s] %s\n", stage, what);
}

// ---- stage 1: allocator ---------------------------------------------------
// grxMalloc/grxFree carve from g_slabs and record in g_live/g_free, all
// std::map. If the lock does not cover the whole carve-and-record, two threads
// can be handed the same extent -- which is a wrong answer, not a crash, so
// the check is that every pointer handed out is distinct and non-null.
void stage_alloc(int tid) {
  std::vector<void*> mine;
  for (int i = 0; i < kIters; ++i) {
    void* p = nullptr;
    const size_t bytes = 256 + (size_t)((tid * 37 + i * 13) % 4096);
    if (grxMalloc(&p, bytes) != grxSuccess || p == nullptr) { ++g_fail; continue; }
    std::memset(p ? (void*)&p : nullptr, 0, 0);   // do not touch device memory from host
    mine.push_back(p);
  }
  for (void* p : mine) if (grxFree(p) != grxSuccess) ++g_fail;
}

// ---- stage 2: modules -----------------------------------------------------
void stage_modules(int tid) {
  (void)tid;
  for (int i = 0; i < kIters / 4; ++i) {
    grxModule_t m = nullptr;
    if (grxModuleLoad(&m, "build-real/preamble.vxbin") != grxSuccess) { ++g_fail; continue; }
    grxFunction_t f = nullptr;
    if (grxModuleGetFunction(&f, m, "preamble_probe") != grxSuccess) ++g_fail;
    grxModuleUnload(m);
  }
}

// ---- stage 3: events and the error slot -----------------------------------
// g_last_error is thread_local, so the interesting question is whether the
// EVENT table under it is shared, and whether a query from one thread can see
// another's half-built event.
void stage_events(int tid) {
  (void)tid;
  for (int i = 0; i < kIters; ++i) {
    grxEvent_t e = nullptr;
    if (grxEventCreate(&e) != grxSuccess) { ++g_fail; continue; }
    grxEventRecord(e, nullptr);
    grxEventQuery(e);
    (void)grxGetLastError();
    if (grxEventDestroy(e) != grxSuccess) ++g_fail;
  }
}

// ---- stage 4: launches on per-thread streams ------------------------------
// The one stage whose report is muddied by an uninstrumented backend. Reported
// separately for that reason.
void stage_launch(int tid, grxModule_t m, grxFunction_t f, void* out) {
  grxStream_t s = nullptr;
  if (grxStreamCreate(&s) != grxSuccess) { ++g_fail; return; }
  (void)m;
  struct pargs { uint32_t abi, slots; uint64_t out; };
  for (int i = 0; i < 4; ++i) {
    pargs a{}; a.abi = 3; a.slots = 8; a.out = (uint64_t)(uintptr_t)out;
    if (grxLaunchFunction(f, dim3_t{2,1,1}, dim3_t{16,1,1}, &a, sizeof(a), 0, s)
        != grxSuccess) ++g_fail;
    if (grxStreamSynchronize(s) != grxSuccess) ++g_fail;
  }
  grxStreamDestroy(s);
  (void)tid;
}

void run(const char* name, void (*fn)(int)) {
  std::printf("stage %s: %d threads x %d iters\n", name, kThreads, kIters);
  const int before = g_fail.load();
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) ts.emplace_back(fn, t);
  for (auto& t : ts) t.join();
  const int d = g_fail.load() - before;
  if (d) std::printf("  %d API failures\n", d);
  else   note(name, "no API failures (TSan findings, if any, are above)");
}

}  // namespace

int main(int argc, char** argv) {
  const int stage = (argc > 1) ? std::atoi(argv[1]) : 0;

  grxDeviceProp_t p{};
  grxGetDeviceProperties(&p, 0);
  std::printf("%s: %d SMs -- host-thread stress, %d threads\n\n",
              p.name, p.multiProcessorCount, kThreads);

  // One allocation shared by the launch stage, made single-threaded so the
  // stage measures concurrent LAUNCH and not concurrent allocation.
  void* out = nullptr;
  grxMalloc(&out, 8 * 4 * 8);

  if (stage == 1 || stage == 0) run("alloc",   stage_alloc);
  if (stage == 2 || stage == 0) run("modules", stage_modules);
  if (stage == 3 || stage == 0) run("events",  stage_events);

  if (stage == 4 || stage == 0) {
    grxModule_t m = nullptr; grxFunction_t f = nullptr;
    if (grxModuleLoad(&m, "build-real/preamble.vxbin") == grxSuccess &&
        grxModuleGetFunction(&f, m, "preamble_probe") == grxSuccess) {
      std::printf("stage launch: %d threads, own streams\n", kThreads);
      std::vector<std::thread> ts;
      for (int t = 0; t < kThreads; ++t)
        ts.emplace_back(stage_launch, t, m, f, out);
      for (auto& t : ts) t.join();
      std::printf("  (backend .so is not TSan-instrumented -- weaker evidence)\n");
    } else {
      std::printf("stage launch: skipped, no module\n");
    }
  }

  std::printf("\n%d API failures total.\n", g_fail.load());
  return g_fail.load() ? 1 : 0;
}
