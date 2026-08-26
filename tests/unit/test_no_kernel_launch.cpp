// A DEVICE THAT CANNOT RUN KERNELS MUST SAY SO, NOT FALL BACK.
//
// This is groundwork for the GRX930 NPU and it is written before the NPU
// exists, on purpose. The c930 NPU is a systolic array with no SIMT pipeline:
// grxcp_architecture.md section 6 fixes its capability profile as GRX_CAP_GEMM
// without GRX_CAP_KERNEL_LAUNCH, and fixes what grxLaunchKernel must do after
// grxSetDevice(npu) -- return grxErrorNotSupported, and specifically NOT
// silently run the work on the GPU instead.
//
// A silent fallback is the bad outcome and it is worth naming. It does not
// produce a wrong answer; it produces the RIGHT answer on the wrong engine,
// which is invisible until someone measures and cannot explain the number. The
// refusal is the feature.
//
// There is no NPU here, so the condition is reached the way the runtime derives
// it: a device reporting zero warps has no programmable pipeline. That is a
// fact the driver already reports rather than a device-type test, so the mock
// can produce it with GRXMOCK_NUM_WARPS=0 and no new capability ID is invented
// to test a thing that does not exist yet.
//
// The binary runs BOTH ways and ci/build_mock.sh runs it BOTH ways:
//
//   GRXMOCK_NUM_WARPS=0   the capability is absent and the launch must refuse
//   (default)             the capability is present and the launch must NOT
//                         refuse with grxErrorNotSupported
//
// The second run is the control. Without it, this file would pass just as
// happily against a runtime that refused every launch on every device.

#include <grx/grx.h>

#include "grx_test.h"
#include "../mock/vortex_mock.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }

  grxDeviceProp_t prop{};
  GRX_REQUIRE(grxGetDeviceProperties(&prop, 0), "device properties");

  const bool can_launch = (prop.capabilities & GRX_CAP_KERNEL_LAUNCH) != 0;

  // The derivation itself, checked in both directions: the bit and the geometry
  // it comes from must agree. A device advertising kernel launch with no warps
  // to run them on would be the runtime lying about itself.
  check(can_launch == (prop.warpSize > 0 && prop.maxWarpsPerMultiProcessor > 0),
        "GRX_CAP_KERNEL_LAUNCH agrees with the device's warp geometry");

  // Cooperative launch is a narrower claim -- a grid-wide barrier needs the
  // pipeline this bit reports -- so it can never be set without it.
  check(!(prop.capabilities & GRX_CAP_COOPERATIVE_LAUNCH) || can_launch,
        "cooperative launch is never claimed without kernel launch");

  // A module with one entry point, so the launch has something real to aim at
  // and fails on the capability rather than on a missing kernel.
  std::vector<uint8_t> image(1024);
  const char* names[] = {"k"};
  const size_t n = grxmock_build_module(image.data(), image.size(), names, 1);
  image.resize(n);

  // THE REFUSAL MOVED EARLIER, and this is where that is recorded.
  //
  // This block used to assert that "a module loads on a device with no
  // pipeline" and then check the launch. It did load, and that was the bug:
  // grxModuleLoad succeeded, grxModuleGetFunction succeeded, and the program
  // learned that the device cannot run grids at the third call rather than the
  // first. The mistake was grxSetDevice; the report should arrive next to it.
  //
  // Same error from both, on purpose. "This device does not do kernels" is one
  // fact, and a caller that handles grxErrorNotSupported from the launch has
  // already written the handler for the load.
  grxModule_t mod = nullptr;
  const grxError_t load_rc =
      grxModuleLoadData(&mod, image.data(), image.size());

  if (!can_launch) {
    section("a device with no programmable pipeline");
    check(load_rc == grxErrorNotSupported,
          "grxModuleLoad refuses with grxErrorNotSupported");
    check(load_rc != grxErrorInvalidKernelImage,
          "and not 'invalid image', which would blame the binary");
    check(grxPeekAtLastError() == grxErrorNotSupported,
          "the refusal is recorded as the last error");
    // The launch-level check in launch.cpp stays. It is unreachable through
    // this path now -- there is no module to launch -- and it is still the
    // invariant that matters if any future path hands out a function without
    // going through a load.
    std::printf("  note  the launch-level refusal is now unreachable from here:\n"
                "        the load is what fails, which is the point.\n");
    return grxtest::report();
  }

  check(load_rc == grxSuccess, "a module loads on a device that can launch");
  if (load_rc != grxSuccess) return grxtest::report();
  grxFunction_t fn = nullptr;
  GRX_REQUIRE(grxModuleGetFunction(&fn, mod, "k"),
              "its entry point resolves");

  const grxError_t e = grxLaunchFunction(fn, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                                         nullptr, 0, 0, nullptr);

  if (can_launch) {
    section("control: a device that CAN launch");
    // Not "succeeds" -- the mock does not execute anything and other failures
    // are possible and are not this test's business. The claim is narrow: the
    // capability check must not be what stops it.
    check(e != grxErrorNotSupported,
          "the launch is not refused as unsupported");
    std::printf("        (%s)\n", grxGetErrorString(e));
    std::printf("  note  run with GRXMOCK_NUM_WARPS=0 for the refusal itself\n");
  }

  grxModuleUnload(mod);
  return grxtest::report();
}
