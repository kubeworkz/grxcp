// Argument blocks shared by the WMMA gate's kernels and its host program.
//
// The tile shape is NOT declared here. It is a property of the device build,
// and the host learns it by asking the device (the wmma_shape entry point)
// rather than by repeating a constant that would then have to be kept in step
// with the configuration. A host-side copy of the geometry is exactly the kind
// of second definition that looks right until the day it is not.

#ifndef GRXCP_WMMA_COMMON_H
#define GRXCP_WMMA_COMMON_H

#include <stdint.h>

enum {
  WMMA_SHAPE_M = 0,     // tile rows
  WMMA_SHAPE_N,         // tile columns
  WMMA_SHAPE_K,         // tile depth, for the input element type used here
  WMMA_SHAPE_WARP,      // warp width the kernel was COMPILED for
  WMMA_SHAPE_REGS_A,    // registers per lane, matrix_a fragment
  WMMA_SHAPE_REGS_B,    // ... matrix_b
  WMMA_SHAPE_REGS_ACC,  // ... accumulator
  WMMA_SHAPE_COUNT
};

struct wmma_shape_args {
  uint64_t out;   // uint32_t[WMMA_SHAPE_COUNT]
};

// The int8 tile is a DIFFERENT shape from the fp16 one -- same m and n, deeper
// k, because a 32-bit register holds four int8 elements where it holds two
// fp16. So the host asks for both rather than scaling one by a factor it
// believes in.
enum {
  WMMA_I8_M = 0,
  WMMA_I8_N,
  WMMA_I8_K,
  WMMA_I8_REGS_A,
  WMMA_I8_REGS_B,
  WMMA_I8_REGS_ACC,
  WMMA_I8_ENABLED,     // 0 when this build's tensor unit has no int8
  WMMA_I8_COUNT
};

struct wmma_i8_shape_args {
  uint64_t out;   // uint32_t[WMMA_I8_COUNT]
};

struct wmma_i8_gemm_args {
  uint64_t a;          // int8, m x k, row major, leading dimension lda
  uint64_t b;          // int8, k x n, column major, leading dimension ldb
  uint64_t c;          // int32, m x n, row major, leading dimension ldc
  uint64_t d;          // int32, m x n, row major, leading dimension ldc
  uint32_t lda, ldb, ldc;
  uint32_t accumulate;
};

struct wmma_gemm_args {
  uint64_t a;          // fp16, m x k, row major, leading dimension lda
  uint64_t b;          // fp16, k x n, column major, leading dimension ldb
  uint64_t c;          // fp32, m x n, row major, leading dimension ldc
  uint64_t d;          // fp32, m x n, row major, leading dimension ldc
  uint32_t lda, ldb, ldc;
  uint32_t accumulate; // 0: D = A*B      1: D = A*B + C
};

#endif
