// GRXCP — tensor core API (nvcuda::wmma-shaped).
//
// Backed by vortex::tensor::wmma_context / wgmma_context from the GRX-G100
// kernel sysroot (vx_tensor.h). Those templates already provide fragment
// types, fill, load, store and the MMA step; this header re-presents them in
// the shape CUDA code expects so a ported kernel keeps its structure.
//
// STRUCTURAL NOTE. Kernels using this header CANNOT be compiled through the
// SPIR-V path -- SPIR-V has no route to GRX-G100-specific intrinsics, which
// the GRX-G100 docs call out explicitly. They require the native device
// toolchain (VOLT), which is what grxcc drives and what grxBLAS uses to ship
// prebuilt .vxbin modules before grxcc exists. See the roadmap, phase 3.

#ifndef GRX_WMMA_H
#define GRX_WMMA_H

#include "grx_device.h"
#include <vx_tensor.h>

// A kernel that includes this header intends to use the tensor unit. If the
// device configuration it is being compiled for does not have one, say so at
// compile time. The failure mode this prevents is specific and nasty: the
// sysroot is built with the TCU enabled, the runtime reports tensor cores,
// the kernel is compiled from the baseline VX_config.toml where the TCU is
// off, and the resulting test passes having exercised nothing.
// ci/build_kernel.sh takes the configuration from the installed sysroot for
// exactly this reason; this is the backstop for every other way in.
#if !defined(VX_CFG_EXT_TCU_ENABLED)
#error "grx_wmma.h: no device configuration. Compile with ci/build_kernel.sh, \
which resolves the configuration from the installed sysroot."
#elif !VX_CFG_EXT_TCU_ENABLED
#error "grx_wmma.h: this device configuration has no tensor unit. Rebuild the \
sysroot with ci/build_sysroot.sh --configs \"-DVX_CFG_EXT_TCU_ENABLE\", or do \
not include this header."
#endif

namespace grx {
namespace wmma {

// Fragment roles and layouts, matching the CUDA spelling.
struct matrix_a;
struct matrix_b;
struct accumulator;
struct row_major;
struct col_major;

// fragment<Use, M, N, K, T, Layout>
//
// Implementation: a thin wrapper over vortex::tensor::wmma_context<
//   NT /* threads per warp, from VX_CFG_NUM_THREADS */, M, N, K, T...>::frag,
// so the register layout is the hardware's, not a repacking of it.
template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment;

template <typename Frag, typename T>
__device__ void fill_fragment(Frag& f, const T& value);

template <typename Frag, typename T>
__device__ void load_matrix_sync(Frag& f, const T* ptr, unsigned ldm);

template <typename Frag, typename T>
__device__ void store_matrix_sync(T* ptr, const Frag& f, unsigned ldm,
                                  int layout);

// D = A * B + C
template <typename FragD, typename FragA, typename FragB, typename FragC>
__device__ void mma_sync(FragD& d, const FragA& a, const FragB& b,
                         const FragC& c);

// --- warp-group MMA -------------------------------------------------------
// GRX-G100's TCU implements WGMMA with shared-memory matrix descriptors and
// the lockstep single-CTA gate. wgmma_context composes two wmma_context
// instantiations for the larger per-warp tiles.

template <int M, int N, int K, typename TA, typename TB, typename TAcc>
struct warpgroup;

template <typename WG>
__device__ void wgmma_sync(WG& wg);

// --- 2:4 structured sparsity ---------------------------------------------
// The TCU implements 2:4 sparsity at 2x dense throughput
// (VX_CFG_TCU_SPARSE_ENABLE). The metadata fragment carries the index pairs.

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct sparse_fragment;

template <typename FragD, typename FragA, typename FragMeta,
          typename FragB, typename FragC>
__device__ void mma_sparse_sync(FragD& d, const FragA& a, const FragMeta& meta,
                                const FragB& b, const FragC& c);

// --- block-scaled MX formats ---------------------------------------------
// mxfp8 / mxfp4 are in the TCU format set. There is no stable CUDA analogue
// for these at this level, so GRXCP names them natively rather than pretending
// a CUDA spelling exists.

struct mxfp8_e4m3;
struct mxfp8_e5m2;
struct mxfp4_e2m1;

}  // namespace wmma
}  // namespace grx

#endif  // GRX_WMMA_H
