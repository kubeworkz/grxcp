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
//   So on a build with the tensor unit ON and WGMMA OFF, which is the default
//   configuration, plain WMMA acquires the admission slot and nothing ever
//   releases it. wgmma_cta_blocked() then blocks every other CTA at issue,
//   forever. One CTA is fine, since it owns the slot it took.
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
    std::printf("  two CTAs: DEADLOCK -- upstream defect still present\n");
    std::printf("            tcu_unit.cpp takes a CTA admission slot for every "
                "TCU op but\n            releases it only under "
                "VX_CFG_TCU_WGMMA_ENABLE. See the header of\n            this "
                "file. grxBLAS works around it with a single persistent CTA.\n");
    return 0;
  }
  if (two == 0) {
    std::printf("  two CTAs: completes -- THE DEFECT IS FIXED\n");
    std::printf("            Remove the single-CTA workaround in "
                "src/libs/grxblas/kernels/hgemm_tcu.cpp\n            and let "
                "the host launch one CTA per tile again.\n");
    return 0;
  }
  std::printf("  two CTAs: failed with status %d (not a deadlock)\n", two);
  return 0;
}
