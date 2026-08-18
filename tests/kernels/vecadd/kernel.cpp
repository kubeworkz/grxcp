// GRXCP device kernel: c[i] = a[i] + b[i].
//
// Written against GRXCP's own device header rather than the driver's, so
// building this exercises grx_device.h through the real VOLT compiler. Once
// grxcc exists the only change here is that the argument struct disappears --
// the kernel takes its parameters directly and the driver unpacks them.
//
// The argument convention today is the driver's: the runtime stages the host
// argument blob into device memory and the kernel receives a POINTER to it in
// its first parameter. __UNIFORM__ tells the compiler the value is the same
// across every lane, so the address computation stays scalar.

#include <grx/device/grx_device.h>

#include "common.h"

__global__ void vecadd(vecadd_args* __UNIFORM__ arg) {
  const float* a = reinterpret_cast<const float*>(arg->a);
  const float* b = reinterpret_cast<const float*>(arg->b);
  float*       c = reinterpret_cast<float*>(arg->c);

  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < arg->n) c[i] = a[i] + b[i];
}
