// npu_c930_testing.h — the seam that lets a register model stand in for the
// NPU the RUNTIME enumerates.
//
// npu_c930_attach_model() already exists and is enough to drive the backend
// directly: test_npu_c930_model.cc and test_npu_c930_shim.cc both construct an
// npu_c930_device_t, attach a model, and call npu_c930_detect/npu_c930_gemm on
// it. What none of them can reach is the device the runtime enumerates --
// probe_npu_device() constructs its own handle inside a std::call_once and
// hands it to no one, so grxSetDevice, grxMalloc, grxblasGemmEx and everything
// else above them have never once run against an NPU on a machine without a
// c930.
//
// That is not a small hole. tests/libs/test_grxblas_npu.cpp has been compiled
// into every NPU build since it was written and has never executed a single
// case: it looks for a device with deviceType == GRX_DEVICE_TYPE_NPU, finds
// none, and exits 77. A skip that reads like absent hardware, hiding a path
// that was never reachable at all.
//
// So: one function, called BEFORE the first grx call, that installs the model
// probe_npu_device will detect through.
//
// A MODEL IS NOT HARDWARE, and this seam is where that stops being a comment
// and becomes an assertion. A device reached this way reports
// GRX_BACKEND_MODEL and names itself a model in grxDeviceProp_t.name. It is
// not possible to attach a model here and have the device claim to be silicon,
// because the field that says so is derived from this call rather than from a
// constant. Per AGENTS.md no result obtained through it may be reported as the
// NPU working.

#ifndef NPU_C930_TESTING_H
#define NPU_C930_TESTING_H

#include "npu_c930.h"

#ifdef __cplusplus
extern "C" {
#endif

// Install a register model for the NPU device the runtime enumerates.
//
// MUST be called before the first call that initialises the runtime
// (grxGetDeviceCount, grxSetDevice, grxMalloc, ...). Device enumeration runs
// once behind a std::call_once; after it has run this function has nothing to
// attach to and says so by returning 0.
//
// Passing null for both hooks clears a previously installed model, which is
// only useful before initialisation for the same reason.
//
// Returns 1 if the model was installed, 0 if enumeration had already happened.
int grxcp_npu_attach_model_for_testing(npu_c930_read_fn read32,
                                       npu_c930_write_fn write32,
                                       void* ctx);

// Install a DDR model for the same device. Separate from the register model
// because the two paths are separate on hardware too -- the CSRs are AXI4-Lite
// MMIO and the DDR is the engine's own DMA target. A device with registers and
// no DDR is a real state (it is the state every device is in today), and
// grxMemcpy refuses on it rather than accepting a copy that moves nothing.
//
// Same rule as above: before enumeration, or it returns 0.
int grxcp_npu_attach_memory_for_testing(npu_c930_mem_read_fn mem_read,
                                        npu_c930_mem_write_fn mem_write,
                                        void* ctx);

// True once a model has been installed through the seam above. The runtime
// uses it to decide what the device says about itself; a test can use it to
// assert that it is talking to a model and not to hardware.
int grxcp_npu_model_is_attached(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NPU_C930_TESTING_H
