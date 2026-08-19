// Argument block for the tensor-core GEMM kernel, shared by the host library
// and the device kernel.
//
// A and B are absent on purpose: the kernel never holds a pointer to either.
// It reaches them only through DXA descriptors the host programmed, naming a
// slot and a pair of tile coordinates. That is what makes the staging
// asynchronous, and it is also why the kernel cannot be handed a matrix it was
// not told about.

#ifndef GRXBLAS_HGEMM_ABI_H
#define GRXBLAS_HGEMM_ABI_H

#include <stdint.h>

// Bumped whenever this struct changes; the kernel refuses to run on a
// mismatch. See the same mechanism in sgemm_abi.h and the incident that
// prompted it.
#define GRXBLAS_HGEMM_ABI_VERSION 2u

// Indices into the buffer the shape entry point fills.
enum {
  GRXBLAS_HGEMM_SHAPE_M = 0,   // WMMA tile rows
  GRXBLAS_HGEMM_SHAPE_N,       // WMMA tile columns
  GRXBLAS_HGEMM_SHAPE_K,       // WMMA tile depth, fp16 in
  GRXBLAS_HGEMM_SHAPE_WARP,    // warp width the kernel was compiled for
  GRXBLAS_HGEMM_SHAPE_SMEM,    // bytes of shared memory one warp needs
  GRXBLAS_HGEMM_SHAPE_BLOCK_M, // rows of C one warp produces at a time
  GRXBLAS_HGEMM_SHAPE_BLOCK_N, // columns of C one warp produces at a time
  GRXBLAS_HGEMM_SHAPE_COUNT
};

struct grxblas_hgemm_shape_args {
  uint32_t abi_version;
  uint32_t pad;
  uint64_t out;   // uint32_t[GRXBLAS_HGEMM_SHAPE_COUNT]
};

struct grxblas_hgemm_args {
  uint32_t abi_version;
  uint32_t m, n, k;        // the problem, in elements
  uint32_t ldc;            // C is column major, fp32
  uint32_t m_tiles;        // ceil(m / block M), to decompose a block index
  uint32_t tiles;          // m_tiles * n_tiles: the blocks the warps walk
  uint32_t k_steps;        // ceil(k / tile K)
  uint32_t slot_a, slot_b; // DXA descriptor slots the host programmed
  uint32_t barrier;        // first barrier slot; warp w uses barrier + w
  float    alpha, beta;
  uint64_t c;
  uint64_t cycles;         // optional grxCycleSlot[], 0 to disable
};

#endif  // GRXBLAS_HGEMM_ABI_H
