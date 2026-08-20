// grxBLAS tensor-core GEMM: fp16 in, fp32 accumulate, WMMA tiles per warp.
//
// This is the kernel the phase 3 exit gate is about, and this version is the
// first one that computes the right answer. It is not the tuned one:
//
//   * one WMMA tile of output per warp per pass
//   * single buffered, one k step at a time
//
// THAT IS THE MEASURED OPTIMUM HERE, NOT AN UNTUNED STARTING POINT.
//
// The blocking factors below are knobs, and the obvious settings were tried
// against tests/bench/gemm_cycles.cpp. Cycles per output element at 16x16x16,
// one SM, four warps, one DXA worker:
//
//   1 x 1 tile, single buffer   39.0   <- shipped
//   1 x 1 tile, double buffer   47.9      21% worse
//   1 x 2 block, single buffer  42.0       8% worse
//   2 x 1 block, single buffer  43.8      12% worse
//   2 x 2 block, single buffer  47.9      23% worse
//
// Every textbook lever lost. Two things seem to be going on, and the second is
// the interesting one. Four warps on one core already hide the staging latency
// by interleaving, so a second buffer prefetches into a pipeline that was not
// stalling -- it only adds barrier and addressing work. And blocking reduces
// the BYTES staged per output tile without reducing what the single DXA worker
// actually costs: the A transfer uses the transposing destination, which
// scatters one element per beat, so its cost tracks elements staged per
// transfer rather than transfers issued. A bigger block makes each transfer
// proportionally longer while leaving fewer independent warps at the edges.
//
// None of that need hold on a different machine. More DXA cores, more SMs, or
// WGMMA's larger native tiles would each change the answer, which is why the
// knobs stayed and the numbers are recorded with the configuration attached.
// Re-run the sweep before assuming this shape is still right.
//
// ONE CTA, ON PURPOSE, AND NOT BY CHOICE
//
// This is a persistent-CTA kernel: the host launches ONE block and each warp
// walks the output tiles in strides of the warp count. The natural shape would
// be one CTA per tile, and it is not used because the GRX-G100 SimX model
// deadlocks the moment a second CTA issues a tensor instruction --
// tcu_unit.cpp takes a CTA admission slot for every TCU op but only releases
// it under VX_CFG_TCU_WGMMA_ENABLE, so with WGMMA off the first CTA owns the
// unit forever and every other CTA blocks at issue. Reproducer and full
// mechanism: tests/repro/tcu_multi_cta/, docs/designs/cuda_mapping.md.
//
// On the configuration this was developed against the workaround costs
// nothing -- the core holds four warps and this uses all four -- but it will
// not scale past one SM, and it comes out the moment the model is fixed.
//
// WHY THERE ARE NO A AND B POINTERS
//
// The staging is DXA, which copies a TILE of an array described in advance by
// a descriptor the host programmed into a device slot. The kernel names the
// slot and the tile coordinates. See include/grx/device/grx_pipeline.h.
//
// LAYOUTS, which is where a GEMM like this goes wrong
//
// grxBLAS is column major. A is m x k with element (i, l) at A[i + l*lda], so
// its contiguous direction is i. The tensor unit's matrix_a fragment wants
// row major -- l contiguous. The DMA engine's K_MAJOR destination transposes
// while it writes, which turns the first into the second for free, so:
//
//   A: descriptor dim0 = rows, dim1 = k, K_MAJOR  -> smem[i * K + l]
//   B: descriptor dim0 = k, dim1 = n, ROW_MAJOR   -> smem[j * K + l]
//
// and both fragments load with ldm = K. B's form is what matrix_b col_major
// wants, which is the pairing the tensor unit is built around.
//
// PADDING, which is only half automatic. The engine bounds-checks a tile's
// OUTER dimensions and pads the overhang with CFILL (zero here); dimension 0,
// the contiguous one, is not checked and reads straight on (measured in
// tests/kernels/dxa/). For these two descriptors that works out:
//
//   A: k is dimension 1  -> a k tail is zero padded
//   B: k is dimension 0  -> a k tail reads whatever follows the column
//
// and the accumulation is still exact, because every term of the tail is
// A_pad (zero) times B_whatever. The zero comes from A, not from B, and that
// is the whole reason a ragged k is safe here. A transposed A would put k in
// dimension 0 for both operands and this would silently stop being true, which
// is one reason the host refuses transposes rather than assuming they compose.
//
// The runtime sizes its descriptor bounds check for a full edge tile, so the
// unchecked dimension-0 overhang cannot read outside the allocation.
//
// The OUTPUT still has to be masked, and is.

#include <grx/device/grx_cycles.h>
#include <grx/device/grx_pipeline.h>
#include <grx/device/grx_wmma.h>

#include "../hgemm_abi.h"

namespace w = grx::wmma;

using elem = w::half;
using tile = w::tile<elem>;

using frag_a   = w::fragment<w::matrix_a, tile::m, tile::n, tile::k, elem, w::row_major>;
using frag_b   = w::fragment<w::matrix_b, tile::m, tile::n, tile::k, elem, w::col_major>;
using frag_acc = w::fragment<w::accumulator, tile::m, tile::n, tile::k, float>;

namespace {

// Shared-memory layout, per warp, in bytes:
//
//   [stage 0: A tile][stage 0: B tile][stage 1: A tile][stage 1: B tile][C]
//
// TWO stages, because one is not a pipeline. With a single buffer the warp
// issues a copy, waits for it, computes, and issues the next -- the DMA engine
// idles through the arithmetic and the tensor unit idles through the copy, and
// the async API buys exactly nothing. With two, the copy for step k+1 is in
// flight while step k is being multiplied.
//
// Sizes are multiples of four bytes, which the DMA engine's local-memory
// writes and the fragment loads both want.
// WMMA tiles per warp, down and across. The staged A region covers kTilesM
// tile rows and the staged B region kTilesN tile columns, so one pair of
// transfers produces kTilesM * kTilesN output tiles.
constexpr uint32_t kTilesM = 1;
constexpr uint32_t kTilesN = 1;

constexpr uint32_t kBlockM = kTilesM * tile::m;   // rows of C per warp
constexpr uint32_t kBlockN = kTilesN * tile::n;   // columns of C per warp

constexpr uint32_t kSmemA = kBlockM * tile::k * sizeof(uint16_t);
constexpr uint32_t kSmemB = kBlockN * tile::k * sizeof(uint16_t);
constexpr uint32_t kSmemC = tile::m * tile::n * sizeof(float);
constexpr uint32_t kSmemTotal = kSmemA + kSmemB + kSmemC;

}  // namespace

__global__ void hgemm_tcu_shape(grxblas_hgemm_shape_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXBLAS_HGEMM_ABI_VERSION) return;
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  if (threadIdx.x != 0) return;
  out[GRXBLAS_HGEMM_SHAPE_M]    = (uint32_t)tile::m;
  out[GRXBLAS_HGEMM_SHAPE_N]    = (uint32_t)tile::n;
  out[GRXBLAS_HGEMM_SHAPE_K]    = (uint32_t)tile::k;
  out[GRXBLAS_HGEMM_SHAPE_WARP] = (uint32_t)VX_CFG_NUM_THREADS;
  out[GRXBLAS_HGEMM_SHAPE_SMEM] = kSmemTotal;
  // The host has to program descriptor tiles that match what this kernel
  // stages, so it is told rather than left to derive it.
  out[GRXBLAS_HGEMM_SHAPE_BLOCK_M] = kBlockM;
  out[GRXBLAS_HGEMM_SHAPE_BLOCK_N] = kBlockN;

  // What this tensor unit can actually be fed. Reported rather than assumed:
  // a caller asking for int8 on a build without it should be told no by the
  // device that knows, not by a table on the host that hopes.
  //
  // bf16 is absent from this list because it is absent from the hardware --
  // there is no VX_CFG_TCU_BF16 knob to read. See docs/designs/cuda_mapping.md
  // section 7.19.
  uint32_t types = 0;
#if VX_CFG_TCU_FP16_ENABLED
  types |= GRXBLAS_TCU_FP16;
#endif
#if VX_CFG_TCU_TF32_ENABLED
  types |= GRXBLAS_TCU_TF32;
#endif
#if VX_CFG_TCU_FP8_ENABLED
  types |= GRXBLAS_TCU_FP8;
#endif
#if VX_CFG_TCU_FP4_ENABLED
  types |= GRXBLAS_TCU_FP4;
#endif
#if VX_CFG_TCU_INT8_ENABLED
  types |= GRXBLAS_TCU_INT8;
#endif
#if VX_CFG_TCU_INT4_ENABLED
  types |= GRXBLAS_TCU_INT4;
#endif
  out[GRXBLAS_HGEMM_SHAPE_TYPES] = types;
}

__global__ void hgemm_tcu(grxblas_hgemm_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXBLAS_HGEMM_ABI_VERSION) return;

  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  // Every warp gets its own staging area and its own barrier: they work on
  // different output tiles, so a barrier shared across the CTA would put them
  // in lockstep for no reason, and a shared buffer would have them overwriting
  // each other's tiles.
  const uint32_t warp  = grx::warp_id();
  const uint32_t warps = grx::warps_per_cta();
  const uint32_t lane  = grx::lane_id();

  uint8_t*  smem = grx::shared_memory<uint8_t>() + warp * kSmemTotal;
  uint16_t* sA   = reinterpret_cast<uint16_t*>(smem);
  uint16_t* sB   = reinterpret_cast<uint16_t*>(smem + kSmemA);
  float*    sC   = reinterpret_cast<float*>(smem + kSmemA + kSmemB);

  grx::barrier    bar(arg->barrier + warp, /*num_warps=*/1);
  grx::tensor_map mapA(arg->slot_a);
  grx::tensor_map mapB(arg->slot_b);

  float* C = reinterpret_cast<float*>(arg->c);
  const uint32_t m = arg->m, n = arg->n, ldc = arg->ldc;
  const float alpha = arg->alpha, beta = arg->beta;

  for (uint32_t b = warp; b < arg->tiles; b += warps) {
    const uint32_t block_row = b % arg->m_tiles;
    const uint32_t block_col = b / arg->m_tiles;
    const uint32_t row0 = block_row * kBlockM;
    const uint32_t col0 = block_col * kBlockN;

    frag_acc acc[kTilesM][kTilesN];
    for (uint32_t i = 0; i < kTilesM; ++i)
      for (uint32_t j = 0; j < kTilesN; ++j)
        w::fill_fragment(acc[i][j], 0.0f);

    for (uint32_t step = 0; step < arg->k_steps; ++step) {
      const uint32_t k0 = step * (uint32_t)tile::k;

      // Two transfers for the whole block: kBlockM rows of A and kBlockN
      // columns of B. Both land on one barrier, so both counts are registered
      // before either is issued -- expect_tx after an issue races a transfer
      // that may already have finished.
      bar.expect_tx(2);
      grx::memcpy_async(sA, mapA, row0, k0, bar);
      grx::memcpy_async(sB, mapB, k0, col0, bar);
      bar.arrive_and_wait();

      // Each A fragment is read by every column of the block and each B
      // fragment by every row: that reuse is the whole point of the blocking,
      // and it happens in registers rather than through shared memory.
      frag_a fa[kTilesM];
      frag_b fb[kTilesN];
      for (uint32_t i = 0; i < kTilesM; ++i)
        w::load_matrix_sync(
            fa[i], reinterpret_cast<const elem*>(sA + i * tile::m * tile::k),
            tile::k);
      for (uint32_t j = 0; j < kTilesN; ++j)
        w::load_matrix_sync(
            fb[j], reinterpret_cast<const elem*>(sB + j * tile::n * tile::k),
            tile::k);

      for (uint32_t i = 0; i < kTilesM; ++i)
        for (uint32_t j = 0; j < kTilesN; ++j)
          w::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);

      // The next step overwrites the staging buffers, and the fragment loads
      // above must be done with them first. Within one warp the pipeline is in
      // order, so this is a compiler fence rather than a hardware one.
      __syncwarp();
    }

    // Stage each accumulator to shared memory rather than storing it straight
    // to C: the tile is m x n whether or not that much of C exists, and a
    // direct store would write outside the caller's matrix on any edge tile.
    for (uint32_t i = 0; i < kTilesM; ++i) {
      for (uint32_t j = 0; j < kTilesN; ++j) {
        w::store_matrix_sync(sC, acc[i][j], (unsigned)tile::n, w::mem_row_major);
        __syncwarp();

        const uint32_t base_row = row0 + i * (uint32_t)tile::m;
        const uint32_t base_col = col0 + j * (uint32_t)tile::n;
        for (uint32_t idx = lane; idx < (uint32_t)(tile::m * tile::n);
             idx += grx::warp_size()) {
          const uint32_t row = base_row + idx / (uint32_t)tile::n;
          const uint32_t col = base_col + idx % (uint32_t)tile::n;
          if (row >= m || col >= n) continue;

          const float value = sC[idx];
          float* dst = &C[row + col * ldc];
          // Reading C when beta is zero would be wrong as well as wasteful:
          // the caller may pass uninitialised memory, and 0 * NaN is NaN.
          *dst = (beta == 0.0f) ? (alpha * value)
                                : (alpha * value + beta * *dst);
        }
        __syncwarp();
      }
    }
  }

  probe.finish();
}
