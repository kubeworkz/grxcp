// __launch_bounds__ and per-kernel register metadata: the two pieces of phase 4
// that turn grxFuncGetAttributes from a struct of placeholders into a report.
//
// Both are checked against CONTROLS rather than against themselves:
//
//   __launch_bounds__  a bounded kernel must REFUSE an oversized block, and an
//                      otherwise identical unbounded kernel must ACCEPT the
//                      same launch. Without the second half, a runtime that
//                      refused every large launch would pass.
//
//   numRegs            a deliberately register-hungry kernel must report MORE
//                      than a trivial one. Without that, a driver that
//                      hard-coded any constant would pass -- which is the exact
//                      failure the -1 sentinel existed to prevent, so replacing
//                      it needs more than "the number is not -1".

#include <grx/grx.h>

#include <cstdio>
#include <vector>

// __launch_bounds__ needs a COMPILE-TIME expression, and the device's real
// limits are a run-time property, so the bound here is just a small constant
// that any plausible device can exceed. main() checks that assumption against
// the device it actually got and says so if it cannot -- a gate that silently
// tested nothing is worse than one that reports a skip.
//
// It is spelled as a macro rather than a literal for a second reason: a macro
// is an expression grxcc cannot evaluate, so this also exercises the rule that
// __launch_bounds__ arguments are passed through to the host compiler as source
// text rather than parsed as a number.
#ifndef GRX_TEST_BOUND
#define GRX_TEST_BOUND (2 + 2)
#endif

__global__ void __launch_bounds__(GRX_TEST_BOUND) bounded(float* out,
                                                          unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0f;
}

// Same body, no attribute. This is the control for the launch-bounds check.
__global__ void unbounded(float* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0f;
}

// As few live values as a kernel can have and still do something.
__global__ void frugal(float* out) {
  out[0] = 1.0f;
}

// Deliberately register-hungry: sixteen accumulators alive across the whole
// loop, which the compiler cannot spill to shared memory and has no reason to
// rematerialise. If numRegs is a real measurement, this reads higher than
// `frugal`; if it is a constant, the two agree.
__global__ void greedy(const float* in, float* out, unsigned n) {
  const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  float acc[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j] = (float)j;
  for (unsigned i = t; i < n; i += blockDim.x * gridDim.x) {
    const float v = in[i];
#pragma unroll
    for (int j = 0; j < 16; ++j) acc[j] = acc[j] * v + (float)j;
  }
  float s = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) s += acc[j];
  if (t < n) out[t] = s;
}

int main(int argc, char** argv) {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  grxSetDevice(0);

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, 0);

  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
  };

  // ---- the sentinel still means what it says, and this must go FIRST ------
  //
  // Register counts come from grxcc measuring the ELF it just built. A module
  // loaded from a .vxbin nobody measured has no count, and has to say so.
  //
  // The ordering is forced, not stylistic. Every device image links at the same
  // address, so only ONE module can be resident at a time (cuda_mapping.md
  // section 7.13) -- and touching any of this file's own kernels loads its fat
  // binary and takes that slot. Done after the checks below, this reports
  //
  //     Error: address range overlaps with existing allocation
  //
  // and skips itself while still printing PASSED, which is the shape of a check
  // that quietly stopped checking.
  if (argc > 1) {
    grxModule_t   mod = nullptr;
    grxFunction_t fn  = nullptr;
    if (grxModuleLoad(&mod, argv[1]) == grxSuccess &&
        grxModuleGetFunction(&fn, mod, "vecadd") == grxSuccess) {
      grxFuncAttributes a_loaded{};
      if (grxFuncGetAttributes(&a_loaded, (const void*)fn) == grxSuccess)
        check(a_loaded.numRegs == -1,
              "a module loaded from a .vxbin reports numRegs = -1, unmeasured");
      grxModuleUnload(mod);
    } else {
      std::printf("FAIL  could not load %s to check the unmeasured path\n",
                  argv[1]);
      ++failures;
    }
  }

  // ---- __launch_bounds__ --------------------------------------------------

  grxFuncAttributes a_bounded{}, a_unbounded{};
  if (grxFuncGetAttributes(&a_bounded, (const void*)&bounded) != grxSuccess ||
      grxFuncGetAttributes(&a_unbounded, (const void*)&unbounded) != grxSuccess) {
    std::printf("FAIL  grxFuncGetAttributes failed\n");
    return 1;
  }

  check(a_bounded.maxThreadsPerBlock == GRX_TEST_BOUND,
        "the bounded kernel reports its __launch_bounds__ maximum");
  check(a_unbounded.maxThreadsPerBlock == prop.maxThreadsPerBlock,
        "the unbounded kernel reports the device maximum");

  float* d = nullptr;
  if (grxMalloc((void**)&d, 256 * sizeof(float)) != grxSuccess) return 1;

  const unsigned over = (unsigned)GRX_TEST_BOUND * 2u;
  if ((int)over > prop.maxThreadsPerBlock) {
    std::printf("      (skipping the launch checks: %u threads exceeds this "
                "device's own maximum of %d)\n", over, prop.maxThreadsPerBlock);
  } else {
    bounded<<<1, over>>>(d, 1u);
    const grxError_t refused = grxGetLastError();
    check(refused == grxErrorLaunchOutOfResources,
          "a block larger than __launch_bounds__ is refused");

    // The control. Same shape, no attribute: this one has to go through, or
    // the check above proves only that big blocks fail.
    unbounded<<<1, over>>>(d, 1u);
    const grxError_t allowed = grxGetLastError();
    check(allowed == grxSuccess,
          "the same launch without the attribute is accepted");
    grxDeviceSynchronize();

    // And the bounded kernel still runs at its own limit.
    bounded<<<1, (unsigned)GRX_TEST_BOUND>>>(d, 1u);
    check(grxGetLastError() == grxSuccess,
          "the bounded kernel launches at exactly its maximum");
    grxDeviceSynchronize();
  }

  // ---- numRegs ------------------------------------------------------------

  grxFuncAttributes a_frugal{}, a_greedy{};
  grxFuncGetAttributes(&a_frugal, (const void*)&frugal);
  grxFuncGetAttributes(&a_greedy, (const void*)&greedy);

  std::printf("      registers: frugal=%d greedy=%d\n",
              a_frugal.numRegs, a_greedy.numRegs);

  check(a_frugal.numRegs > 0 && a_greedy.numRegs > 0,
        "both kernels report a register count rather than -1");
  check(a_greedy.numRegs > a_frugal.numRegs,
        "the register-hungry kernel reports more registers than the trivial one");
  check(a_greedy.numRegs <= 64,
        "the count does not exceed the 64 architectural registers a thread has");

  // ptxVersion stays -1 by design, and this gate is the place that would
  // notice if someone made it a number to look more like CUDA.
  check(a_greedy.ptxVersion == -1,
        "ptxVersion is still -1: GRXCP has no PTX analogue");

  grxFree(d);

  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED\n");
  return 0;
}
