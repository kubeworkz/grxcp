// Sample a texture once per thread, through grx::tex2D.
//
// Deliberately one sample per thread with coordinates read from memory, so
// every lane in a warp asks for a DIFFERENT coordinate. A test that sampled a
// uniform coordinate would exercise none of the divergence this header had to
// be written around -- __builtin_floorf does not survive divergent codegen on
// this device (cuda_mapping.md 7.24), and a uniform test would never have
// found that.

#include <grx/device/grx_tex.h>

#include "common.h"

__global__ void texture_sample(texture_args* __UNIFORM__ arg) {
  if (arg->abi_version != TEXTURE_ARGS_ABI) return;

  const float* coords = reinterpret_cast<const float*>(arg->coords);
  float*       out    = reinterpret_cast<float*>(arg->out);

  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= arg->count) return;

  out[i] = grx::tex2D<float>((grxTextureObject_t)arg->object,
                             coords[2 * i], coords[2 * i + 1]);
}
