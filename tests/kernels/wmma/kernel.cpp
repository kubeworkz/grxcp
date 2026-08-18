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
