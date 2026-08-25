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

// Causal masking of an attention score matrix, viewed as `rows` rows of
// `seq_len` columns with row stride `ld`. `rows` is batch*heads*seq_len and the
// row index encodes both the head and the query position, so nothing here needs
// to know how many heads there are.
// y[i][j] = x[i][j] + bias[j]; bias has `cols` entries.
struct grxdnn_bias_args {
  uint32_t abi_version;
  uint32_t rows;
  uint32_t cols;
  int32_t  ldx;
  int32_t  ldy;
  uint32_t pad;
  uint64_t x;
  uint64_t bias;
  uint64_t y;
  uint64_t cycles;
};

// y = gelu(x). `mode` is 0 for the erf form and 1 for the tanh approximation —
// see grxdnn.h for why the caller has to choose rather than get a default.
struct grxdnn_gelu_args {
  uint32_t abi_version;
  uint32_t rows;
  uint32_t cols;
  int32_t  ldx;
  int32_t  ldy;
  uint32_t mode;
  uint64_t x;
  uint64_t y;
  uint64_t cycles;
};

struct grxdnn_mask_args {
  uint32_t abi_version;
  uint32_t rows;
  uint32_t seq_len;
  int32_t  ld;
  uint64_t scores;
  uint64_t cycles;
};

#endif  // GRXDNN_ABI_H
