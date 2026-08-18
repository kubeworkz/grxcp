// GRXCP — error mapping and the thread-local sticky error.

#include "internal.h"

#include <grx/grx_runtime.h>

namespace grxcp {

grxError_t map_result(vx_result_t r) {
  switch (r) {
    case VX_SUCCESS:                  return grxSuccess;
    case VX_ERR_INVALID_HANDLE:       return grxErrorInvalidResourceHandle;
    case VX_ERR_INVALID_INFO:         return grxErrorInvalidValue;
    case VX_ERR_INVALID_VALUE:        return grxErrorInvalidValue;
    case VX_ERR_OUT_OF_HOST_MEMORY:   return grxErrorMemoryAllocation;
    case VX_ERR_OUT_OF_DEVICE_MEMORY: return grxErrorMemoryAllocation;
    case VX_ERR_DEVICE_LOST:          return grxErrorDeviceLost;
    case VX_ERR_TIMEOUT:              return grxErrorTimeout;
    case VX_ERR_EVENT_FAILED:         return grxErrorInvalidResourceHandle;
    case VX_ERR_NOT_SUPPORTED:        return grxErrorNotSupported;
    case VX_ERR_INTERNAL:             return grxErrorUnknown;
  }
  return grxErrorUnknown;
}

namespace {
thread_local grxError_t g_last_error = grxSuccess;
}

grxError_t set_error(grxError_t e) {
  if (e != grxSuccess) g_last_error = e;
  return e;
}

grxError_t take_last_error() {
  grxError_t e = g_last_error;
  g_last_error = grxSuccess;
  return e;
}

grxError_t peek_last_error() { return g_last_error; }

}  // namespace grxcp

extern "C" {

grxError_t grxGetLastError(void)    { return grxcp::take_last_error(); }
grxError_t grxPeekAtLastError(void) { return grxcp::peek_last_error(); }

const char* grxGetErrorName(grxError_t e) {
  switch (e) {
    case grxSuccess:                     return "grxSuccess";
    case grxErrorInvalidValue:           return "grxErrorInvalidValue";
    case grxErrorMemoryAllocation:       return "grxErrorMemoryAllocation";
    case grxErrorInitializationError:    return "grxErrorInitializationError";
    case grxErrorInvalidDevice:          return "grxErrorInvalidDevice";
    case grxErrorInvalidDevicePointer:   return "grxErrorInvalidDevicePointer";
    case grxErrorInvalidMemcpyDirection: return "grxErrorInvalidMemcpyDirection";
    case grxErrorInvalidResourceHandle:  return "grxErrorInvalidResourceHandle";
    case grxErrorNotReady:               return "grxErrorNotReady";
    case grxErrorNotSupported:           return "grxErrorNotSupported";
    case grxErrorInvalidKernelImage:     return "grxErrorInvalidKernelImage";
    case grxErrorInvalidDeviceFunction:  return "grxErrorInvalidDeviceFunction";
    case grxErrorLaunchFailure:          return "grxErrorLaunchFailure";
    case grxErrorLaunchOutOfResources:   return "grxErrorLaunchOutOfResources";
    case grxErrorDeviceLost:             return "grxErrorDeviceLost";
    case grxErrorTimeout:                return "grxErrorTimeout";
    case grxErrorFileNotFound:           return "grxErrorFileNotFound";
    case grxErrorUnknown:                return "grxErrorUnknown";
  }
  return "grxErrorUnknown";
}

const char* grxGetErrorString(grxError_t e) {
  switch (e) {
    case grxSuccess:                     return "no error";
    case grxErrorInvalidValue:           return "invalid argument";
    case grxErrorMemoryAllocation:       return "out of memory";
    case grxErrorInitializationError:    return "runtime initialization failed";
    case grxErrorInvalidDevice:          return "invalid device ordinal";
    case grxErrorInvalidDevicePointer:   return "pointer is not a device allocation";
    case grxErrorInvalidMemcpyDirection: return "copy direction contradicts the allocation map";
    case grxErrorInvalidResourceHandle:  return "invalid stream, event, module or kernel handle";
    case grxErrorNotReady:               return "asynchronous operation has not completed";
    case grxErrorNotSupported:           return "not supported on this device or backend";
    case grxErrorInvalidKernelImage:     return "kernel image is malformed or targets another ISA";
    case grxErrorInvalidDeviceFunction:  return "kernel was not registered for this device";
    case grxErrorLaunchFailure:          return "kernel launch failed";
    case grxErrorLaunchOutOfResources:   return "launch exceeds a per-core resource bound";
    case grxErrorDeviceLost:             return "device or backend became unreachable";
    case grxErrorTimeout:                return "wait timed out";
    case grxErrorFileNotFound:           return "no such file";
    case grxErrorUnknown:                return "unknown error";
  }
  return "unknown error";
}

}  // extern "C"
