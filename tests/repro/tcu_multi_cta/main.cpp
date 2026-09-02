// Watch for an upstream defect: the tensor unit deadlocks with two CTAs.
//
// MECHANISM, from sim/simx/tcu/tcu_unit.cpp and sim/simx/core.cpp:
//
//   Core::issue calls tcu_unit_->wgmma_cta_admit() for EVERY TCU micro-op that
//   takes the FU lock, guarded only by #ifdef VX_CFG_EXT_TCU_ENABLE. That sets
//   wgmma_cta_owner_ to the issuing CTA and increments wgmma_admitted_warps_.
//
//   The matching release -- decrement, and clear the owner at zero -- lives
//   inside #ifdef VX_CFG_TCU_WGMMA_ENABLE and additionally only runs for ops
//   where tcu_is_wgmma(tcu_type).
//
//   So plain WMMA acquires the admission slot and nothing ever releases it.
//   wgmma_cta_blocked() then blocks every other CTA at issue, forever. One CTA
//   is fine, since it owns the slot it took.
//
// TURNING WGMMA ON DOES NOT FIX IT, and this was tested rather than reasoned
// about. A sysroot built with -DVX_CFG_TCU_TCU_WGMMA_ENABLE still deadlocks
// here, because the release has TWO conditions and the compile-time flag is
// only one of them: the retiring op must also BE a WGMMA op. A kernel issuing
// plain WMMA never satisfies that, whatever the build says. The distinction
// matters for whoever fixes this -- flipping the flag is not the fix, and a
// bug report that suggests it is sends them to the wrong place.
//
// The fix is to make acquire and release symmetric -- either gate the acquire
// on WGMMA as well, or release on any TCU op that carries fu_unlock.
//
// This program is a WATCH, not a gate. It runs the two-CTA launch in a child
// process under a timeout, so CI notices when the defect is fixed without
// hanging when it is not. Exit code is 0 either way; read the message.

#include <grx/grx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct tcu_repro_args { uint64_t a, b, d; };

int run_launch(const char* image, unsigned blocks) {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) return 77;
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, 0);

  grxModule_t mod = nullptr;
  grxFunction_t fn = nullptr;
  if (grxModuleLoad(&mod, image) != grxSuccess) return 1;
  if (grxModuleGetFunction(&fn, mod, "tcu_repro") != grxSuccess) return 1;

  void *dA = nullptr, *dB = nullptr, *dD = nullptr;
  if (grxMalloc(&dA, 1024) != grxSuccess || grxMalloc(&dB, 1024) != grxSuccess ||
      grxMalloc(&dD, 1024) != grxSuccess)
    return 1;
  grxMemset(dA, 0, 1024);
  grxMemset(dB, 0, 1024);
  grxMemset(dD, 0, 1024);

  tcu_repro_args args{};
  args.a = (uint64_t)(uintptr_t)dA;
  args.b = (uint64_t)(uintptr_t)dB;
  args.d = (uint64_t)(uintptr_t)dD;

  if (grxLaunchFunction(fn, dim3_t{blocks, 1, 1},
                        dim3_t{(unsigned)prop.warpSize, 1, 1}, &args,
                        sizeof(args), 0, nullptr) != grxSuccess)
    return 1;
  return (grxDeviceSynchronize() == grxSuccess) ? 0 : 1;
}

// Run one launch in a child and give it `seconds` to finish. The parent stays
// clean: a deadlocked simulator cannot be interrupted from inside the process
// that called it.
int try_launch(const char* self, const char* image, unsigned blocks,
               unsigned seconds) {
  const pid_t pid = fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    char b[16];
    std::snprintf(b, sizeof(b), "%u", blocks);
    execl(self, self, image, b, "--child", (char*)nullptr);
    _exit(127);
  }
  for (unsigned i = 0; i < seconds * 10; ++i) {
    int status = 0;
    const pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    usleep(100000);
  }
  kill(pid, SIGKILL);
  waitpid(pid, nullptr, 0);
  return -1;   // timed out
}

}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "tcu_repro.vxbin";

  if (argc > 3 && std::strcmp(argv[3], "--child") == 0)
    return run_launch(image, (unsigned)std::atoi(argv[2]));

  const int one = try_launch(argv[0], image, 1, 60);
  if (one == 77) { std::printf("no devices; skipping\n"); return 77; }
  if (one != 0) {
    std::printf("UNEXPECTED: a single-CTA tensor launch did not complete "
                "(status %d)\n", one);
    return 1;
  }
  std::printf("  one CTA:  completes\n");

  const int two = try_launch(argv[0], image, 2, 60);
  if (two == -1) {
    std::printf("  two CTAs: DEADLOCK on this backend -- defect still present\n");
    std::printf("            tcu_unit.cpp takes a CTA admission slot for every "
                "TCU op and\n            releases it only for ops that ARE "
                "WGMMA -- so a WMMA kernel\n            deadlocks whether or "
                "not VX_CFG_TCU_WGMMA_ENABLE is set, which\n            has "
                "been checked both ways. See the header of this file.\n"
                "            grxBLAS works around it with a single persistent "
                "CTA.\n");
    return 0;
  }
  if (two == 0) {
    // "COMPLETES" IS NOT "FIXED", AND THIS USED TO SAY IT WAS.
    //
    // The line here read "THE DEFECT IS FIXED", which is one of two things it
    // could mean and the watch cannot tell them apart from a single run:
    //
    //   * the defect was present on this backend and has been repaired, or
    //   * this backend never had it.
    //
    // The second turned out to be real. The defect is in SimX's C++ --
    // tcu_unit.cpp takes a CTA admission slot for every TCU op and releases it
    // only for ops that ARE WGMMA. The RTL is a different implementation of the
    // same architecture and does not have it: measured, two CTAs complete on
    // `rtlsim` at a configuration whose ISA flags, warp count and TCU settings
    // are identical to the SimX one that deadlocks (cuda_mapping.md 7.12).
    //
    // So the verdict is per backend, and the message says which one answered.
    // A watch that reports a conclusion it has not earned is worse than one
    // that reports less.
    grxDeviceProp_t p{};
    const char* backend = "this backend";
    if (grxGetDeviceProperties(&p, 0) == grxSuccess) {
      switch (p.backend) {
        case GRX_BACKEND_SIMX:    backend = "simx";    break;
        case GRX_BACKEND_RTLSIM:  backend = "rtlsim";  break;
        case GRX_BACKEND_SILICON: backend = "silicon"; break;
        default: break;
      }
    }
    std::printf("  two CTAs: completes on %s\n", backend);
    std::printf("            This backend does not have the defect. It says\n"
                "            nothing about any other: the deadlock is in SimX's\n"
                "            tcu_unit.cpp, and `rtlsim` has been measured\n"
                "            completing the same launch. Run this on the\n"
                "            backend you intend to ship against before removing\n"
                "            the single-CTA workaround in grxblas.cpp -- and\n"
                "            note that on a ONE-SM configuration the workaround\n"
                "            costs nothing, so removing it there will not move\n"
                "            a number.\n");
    return 0;
  }
  std::printf("  two CTAs: failed with status %d (not a deadlock)\n", two);
  return 0;
}
