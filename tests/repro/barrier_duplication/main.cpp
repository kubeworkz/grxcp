// Watch for an upstream defect: __syncthreads() gets duplicated across a
// divergent branch and the kernel deadlocks.
//
// MECHANISM, from sw/kernel/include/vx_intrinsics.h and vx_spawn2.h:
//
//   vx_barrier() is `__asm__ volatile(".insn r ..." ::: "memory")`. Volatile
//   inline asm may not be deleted and may not be reordered against other
//   volatile operations. It MAY be duplicated -- nothing in it says otherwise,
//   because LLVM expresses "do not duplicate across control flow that a SIMT
//   machine will re-execute" with the `convergent` and `noduplicate` function
//   attributes, and an asm statement carries neither.
//
//   __syncthreads() in vx_spawn2.h is that asm and nothing more.
//
//   So LLVM is free to tail-duplicate the block containing the barrier into
//   both arms of a preceding divergent branch, and at -O3 it does:
//
//       vx_split_n a0, a7          # diverge
//       beqz  a7, .else
//         ... ; vx_bar a5, a7      # copy 1
//         j .join
//       .else:
//         vx_bar a1, a2            # copy 2
//       .join:
//       vx_join a0
//
//   A warp whose lanes agree runs one arm and arrives once. A DIVERGED warp
//   runs both arms and arrives TWICE. A two-warp CTA then posts three arrivals
//   against a barrier expecting two: the first two release, and the third opens
//   a generation that nobody will ever join. The kernel hangs.
//
// WHY IT HIDES. The extra arrival only happens when a warp actually diverges,
// so a grid that divides evenly into the block size passes and a ragged one
// hangs. It also needs more than one warp per CTA, because a single-warp CTA's
// barrier is satisfied by its own first arrival. Every GRXCP kernel gate before
// phase 4 used one warp per CTA and an exact grid, which is why this survived
// three phases undetected -- the first program grxcc ever compiled had a ragged
// tail and two warps, and hung.
//
// WHAT GRXCP DOES ABOUT IT: include/grx/device/grx_device.h routes
// __syncthreads() through a wrapper marked convergent and noduplicate, and
// grx_cg.h does the same for the cluster and grid forms. That fixes GRXCP's
// kernels. It does not fix the tree: anything else calling vx_barrier,
// vx_barrier_arrive or vx_barrier_wait through the upstream headers has the
// same exposure, which is why this repro exists rather than just a test.
//
// THE FIX upstream is to give vx_barrier and friends a `convergent` wrapper --
// the same three words -- so callers do not each have to know.
//
// This program is BOTH a gate and a watch:
//
//   guarded_good MUST pass. It is the workaround, and a regression in it is a
//   GRXCP bug -- exit non-zero.
//
//   guarded_bad is expected to hang. It runs in a child under a timeout, and
//   when it starts passing the workaround can go. Exit code is unaffected;
//   read the message.

#include <grx/grx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct args_t {
  uint64_t in;
  uint64_t out;
  uint32_t n;
  uint32_t pad;
};

// Run one kernel to completion and check the answer. Returns 0 on a correct
// result, 1 on a wrong one, 2 on an API failure.
int run_one(const char* image, const char* kernel, unsigned warps_per_cta,
            unsigned ctas, unsigned ragged) {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) return 77;
  if (grxSetDevice(0) != grxSuccess) return 2;

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess) return 2;

  const unsigned block = (unsigned)prop.warpSize * warps_per_cta;
  const unsigned n     = block * ctas - ragged;

  grxModule_t   mod = nullptr;
  grxFunction_t fn  = nullptr;
  if (grxModuleLoad(&mod, image) != grxSuccess) return 2;
  if (grxModuleGetFunction(&fn, mod, kernel) != grxSuccess) return 2;

  const size_t bytes = (size_t)n * sizeof(float);
  std::vector<float> hin(n), hout(n, -1.0f);
  for (unsigned i = 0; i < n; ++i) hin[i] = (float)i;

  void *din = nullptr, *dout = nullptr;
  if (grxMalloc(&din, bytes) != grxSuccess) return 2;
  if (grxMalloc(&dout, bytes) != grxSuccess) return 2;
  if (grxMemcpy(din, hin.data(), bytes, grxMemcpyDefault) != grxSuccess) return 2;
  if (grxMemset(dout, 0, bytes) != grxSuccess) return 2;

  args_t a{};
  a.in  = (uint64_t)(uintptr_t)din;
  a.out = (uint64_t)(uintptr_t)dout;
  a.n   = n;

  if (grxLaunchFunction(fn, dim3_t{ctas, 1, 1}, dim3_t{block, 1, 1},
                        &a, sizeof(a), block * sizeof(float),
                        nullptr) != grxSuccess) return 2;
  if (grxDeviceSynchronize() != grxSuccess) return 2;
  if (grxMemcpy(hout.data(), dout, bytes, grxMemcpyDefault) != grxSuccess) return 2;

  unsigned bad = 0;
  for (unsigned i = 0; i < n; ++i) {
    const unsigned base   = (i / block) * block;
    const unsigned mirror = base + block - 1u - (i - base);
    if (mirror >= n) continue;          // never written; the guard is the point
    if (hout[i] != (float)mirror) ++bad;
  }
  grxFree(din);
  grxFree(dout);
  grxModuleUnload(mod);
  return bad ? 1 : 0;
}

// Run `kernel` in a child with a wall-clock limit. -1 means it did not finish.
int run_child(const char* self, const char* image, const char* kernel,
              unsigned seconds) {
  const pid_t pid = ::fork();
  if (pid < 0) return -2;
  if (pid == 0) {
    ::alarm(seconds);
    ::execl(self, self, image, kernel, (char*)nullptr);
    ::_exit(127);
  }
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return -2;
  if (WIFSIGNALED(status)) return -1;                 // SIGALRM: still hung
  return WIFEXITED(status) ? WEXITSTATUS(status) : -2;
}

}  // namespace

int main(int argc, char** argv) {
  // The two-argument form is the child: run one kernel and exit with its code.
  // Two warps per CTA and a ragged tail, which is the shape that diverges.
  if (argc == 3) return run_one(argv[1], argv[2], 2, 6, 5);

  const char* image = (argc > 1) ? argv[1] : "barrier_repro.vxbin";

  std::printf("--- guarded_good: GRXCP's convergent __syncthreads()\n");
  const int good = run_child(argv[0], image, "guarded_good", 60);
  if (good == 77) { std::printf("SKIPPED: no device\n"); return 77; }
  if (good != 0) {
    std::printf("FAILED: the convergent barrier wrapper %s.\n"
                "        This is a GRXCP regression, not an upstream one: see\n"
                "        include/grx/device/grx_device.h.\n",
                good == -1 ? "deadlocked" : "produced a wrong answer");
    return 1;
  }
  std::printf("ok: correct, as the workaround requires\n");

  std::printf("--- guarded_bad: upstream's bare vx_barrier\n");
  const int bad = run_child(argv[0], image, "guarded_bad", 30);
  if (bad == -1) {
    std::printf("STILL BROKEN (expected): the bare barrier deadlocked.\n"
                "        Keep the wrapper. cuda_mapping.md section 7.20.\n");
  } else if (bad == 0) {
    std::printf("FIXED UPSTREAM: the bare barrier now completes correctly.\n"
                "        Re-check the disassembly for a single vx_bar, then the\n"
                "        wrapper in grx_device.h and grx_cg.h can go, along with\n"
                "        cuda_mapping.md section 7.20 and this repro.\n");
  } else {
    std::printf("CHANGED: the bare barrier returned %d rather than hanging.\n"
                "        Investigate before touching the workaround -- a wrong\n"
                "        answer is a different defect from a deadlock.\n", bad);
  }
  return 0;
}
