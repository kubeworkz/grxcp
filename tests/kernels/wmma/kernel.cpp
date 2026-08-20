// GRXCP device kernels for the WMMA gate.
//
// Two entry points in one module:
//
//   wmma_shape       reports the tile geometry this build actually has, so the
//                    host can size its matrices from the device rather than
//                    from a hardcoded guess.
//   wmma_gemm_tile   one tile of D = A * B (+ C) through grx::wmma.
//
// Both are single-warp: the fragment API is warp-wide, so the launch must use
// a block of exactly one warp with every lane active. A partial warp would put
// lanes into the fragment layout that never loaded their share of the matrix.

#include <grx/device/grx_wmma.h>

#include "common.h"

namespace w = grx::wmma;

using elem = w::half;
using tile = w::tile<elem>;

using frag_a   = w::fragment<w::matrix_a, tile::m, tile::n, tile::k, elem, w::row_major>;
using frag_b   = w::fragment<w::matrix_b, tile::m, tile::n, tile::k, elem, w::col_major>;
using frag_acc = w::fragment<w::accumulator, tile::m, tile::n, tile::k, float>;

__global__ void wmma_shape(wmma_shape_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  if (threadIdx.x != 0) return;
  out[WMMA_SHAPE_M]    = (uint32_t)tile::m;
  out[WMMA_SHAPE_N]    = (uint32_t)tile::n;
  out[WMMA_SHAPE_K]    = (uint32_t)tile::k;
  // The warp width the kernel was COMPILED with, which the host compares
  // against the width the runtime REPORTS. Those two coming from different
  // configurations is the failure this whole build path is arranged to
  // prevent, so the gate checks it rather than assuming it.
  out[WMMA_SHAPE_WARP] = (uint32_t)VX_CFG_NUM_THREADS;
  out[WMMA_SHAPE_REGS_A]   = frag_a::num_elements;
  out[WMMA_SHAPE_REGS_B]   = frag_b::num_elements;
  out[WMMA_SHAPE_REGS_ACC] = frag_acc::num_elements;
}

__global__ void wmma_gemm_tile(wmma_gemm_args* __UNIFORM__ arg) {
  const elem*  A = reinterpret_cast<const elem*>(arg->a);
  const elem*  B = reinterpret_cast<const elem*>(arg->b);
  const float* C = reinterpret_cast<const float*>(arg->c);
  float*       D = reinterpret_cast<float*>(arg->d);

  frag_a   a;
  frag_b   b;
  frag_acc acc, d;

  w::load_matrix_sync(a, A, arg->lda);
  w::load_matrix_sync(b, B, arg->ldb);

  if (arg->accumulate) {
    w::load_matrix_sync(acc, C, arg->ldc, w::mem_row_major);
  } else {
    // Not "C is zero, so skip the load" -- C may hold anything, including the
    // uninitialised bytes an allocator handed back.
    w::fill_fragment(acc, 0.0f);
  }

  w::mma_sync(d, a, b, acc);
  w::store_matrix_sync(D, d, arg->ldc, w::mem_row_major);
}

// ---------------------------------------------------------------------------
// int8 in, int32 out
// ---------------------------------------------------------------------------
//
// The same three calls, a different tile, and an arithmetic that is EXACT --
// which is the point of testing it separately rather than trusting that a
// working fp16 path implies a working int8 one. Nothing about the register
// plumbing changes: wmma_context carries every fragment in float registers
// whatever the format, so an int8 fragment is a bit pattern in a float exactly
// as a packed fp16 pair already is.

#if VX_CFG_TCU_INT8_ENABLED
using i8tile   = w::tile<int8_t>;
using frag_i8a = w::fragment<w::matrix_a, i8tile::m, i8tile::n, i8tile::k, int8_t, w::row_major>;
using frag_i8b = w::fragment<w::matrix_b, i8tile::m, i8tile::n, i8tile::k, int8_t, w::col_major>;
using frag_i32 = w::fragment<w::accumulator, i8tile::m, i8tile::n, i8tile::k, int32_t>;
#endif

__global__ void wmma_i8_shape(wmma_i8_shape_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  if (threadIdx.x != 0) return;
#if VX_CFG_TCU_INT8_ENABLED
  out[WMMA_I8_M]        = (uint32_t)i8tile::m;
  out[WMMA_I8_N]        = (uint32_t)i8tile::n;
  out[WMMA_I8_K]        = (uint32_t)i8tile::k;
  out[WMMA_I8_REGS_A]   = frag_i8a::num_elements;
  out[WMMA_I8_REGS_B]   = frag_i8b::num_elements;
  out[WMMA_I8_REGS_ACC] = frag_i32::num_elements;
  out[WMMA_I8_ENABLED]  = 1;
#else
  // Reported, not omitted: the host has to be able to tell "this build has no
  // int8" from "the kernel never ran".
  for (uint32_t i = 0; i < WMMA_I8_COUNT; ++i) out[i] = 0;
#endif
}

#if VX_CFG_TCU_INT8_ENABLED
__global__ void wmma_i8_gemm_tile(wmma_i8_gemm_args* __UNIFORM__ arg) {
  const int8_t*  A = reinterpret_cast<const int8_t*>(arg->a);
  const int8_t*  B = reinterpret_cast<const int8_t*>(arg->b);
  const int32_t* C = reinterpret_cast<const int32_t*>(arg->c);
  int32_t*       D = reinterpret_cast<int32_t*>(arg->d);

  frag_i8a a;
  frag_i8b b;
  frag_i32 acc, d;

  w::load_matrix_sync(a, A, arg->lda);
  w::load_matrix_sync(b, B, arg->ldb);
  if (arg->accumulate) {
    w::load_matrix_sync(acc, C, arg->ldc, w::mem_row_major);
  } else {
    w::fill_fragment(acc, 0);
  }
  w::mma_sync(d, a, b, acc);
  w::store_matrix_sync(D, d, arg->ldc, w::mem_row_major);
}
#endif
