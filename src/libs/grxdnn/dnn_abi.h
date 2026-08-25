// Argument blocks for the grxDNN device kernels, shared by the host library and
// the kernels themselves. One definition, included by both — the same rule
// grxBLAS's sgemm_abi.h states, and for the same reason.

#ifndef GRXDNN_ABI_H
#define GRXDNN_ABI_H

#include <stdint.h>

// Bumped whenever any struct here changes; every kernel checks it and refuses
// to run on a mismatch rather than reading a field that moved.
#define GRXDNN_ABI_VERSION 1u

// Leading dimensions are ROW strides in elements, because grxDNN is row-major
// (grxdnn.h says so in capitals). Signed, so a caller cannot pass a negative
// stride by accident and have it become enormous.
struct grxdnn_softmax_args {
  uint32_t abi_version;
  uint32_t rows;
  uint32_t cols;
  int32_t  ldx;
  int32_t  ldy;
  uint32_t pad;
  uint64_t x;
  uint64_t y;
  uint64_t cycles;   // optional grxCycleSlot[], 0 to disable
};

struct grxdnn_layernorm_args {
  uint32_t abi_version;
  uint32_t rows;
  uint32_t cols;
  int32_t  ldx;
  int32_t  ldy;
  float    eps;
  uint64_t x;
  uint64_t y;
  uint64_t gamma;    // 0 for no scale
  uint64_t beta;     // 0 for no shift
  uint64_t cycles;
};

#endif  // GRXDNN_ABI_H
