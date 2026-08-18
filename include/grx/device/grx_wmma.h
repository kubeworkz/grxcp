// GRXCP — tensor core API (nvcuda::wmma-shaped).
//
// Backed by vortex::tensor::wmma_context from the GRX-G100 kernel sysroot
// (vx_tensor.h), which already provides fragment types, fill, load, store and
// the MMA step. This header re-presents them in the shape CUDA code expects so
// a ported kernel keeps its structure.
//
// WHAT IS DIFFERENT FROM CUDA, AND WHY IT HAS TO BE
//
// CUDA fixes the fragment shape at 16x16x16 (and a few others) and the same
// shapes exist on every SM that supports wmma at all. GRX-G100's tile shape is
// DERIVED from the configuration -- warp width, registers per fragment, input
// element size -- so it is not 16x16x16 and it is not even constant across
// builds. Pretending otherwise would mean silently reshaping the caller's data,
// which is the one thing a matrix API must never do.
//
// So the M, N, K a fragment is declared with are CHECKED, not honoured: declare
// the shape this build actually provides or the code does not compile. Write
// portable kernels against grx::wmma::tile<T>::m / ::n / ::k rather than
// hardcoding numbers, and the same source follows the configuration.
//
//   using tile = grx::wmma::tile<grx::wmma::half>;
//   grx::wmma::fragment<grx::wmma::matrix_a, tile::m, tile::n, tile::k,
//                       grx::wmma::half, grx::wmma::row_major> a;
//
// STRUCTURAL NOTE. Kernels using this header CANNOT be compiled through the
// SPIR-V path -- SPIR-V has no route to GRX-G100-specific intrinsics, which
// the GRX-G100 docs call out explicitly. They require the native device
// toolchain (VOLT), which is what grxcc drives and what grxBLAS uses to ship
// prebuilt .vxbin modules before grxcc exists. See the roadmap, phase 3.
//
// STATUS: dense fp16-in/fp32-out and fp32-in/fp32-out, single-warp WMMA. What
// is deliberately NOT here is listed at the bottom of this file rather than
// declared as an empty template that fails at instantiation.

#ifndef GRX_WMMA_H
#define GRX_WMMA_H

#include "grx_device.h"
#include <vx_tensor.h>
#include <type_traits>

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

// Registers per accumulator fragment. This is the one free parameter in the
// tile geometry; everything else follows from it and from the warp width.
// Changing it changes the tile shape, which is why it is a named knob and not
// a literal 8 buried in a template argument list.
#ifndef GRX_WMMA_FRAG_REGS
#define GRX_WMMA_FRAG_REGS 8
#endif

namespace grx {
namespace wmma {

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------
//
// half is storage, not arithmetic: it holds IEEE binary16 bits and nothing in
// this header converts to or from it. Conversion helpers belong with their own
// numerical gate (rounding is where half implementations go wrong), so they are
// not smuggled in here untested. Produce halves on the host, or load them from
// memory the device already holds.

struct half { uint16_t bits; };

// ---------------------------------------------------------------------------
// Fragment roles and layouts
// ---------------------------------------------------------------------------

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};

// Runtime layout selector for the accumulator, matching CUDA's spelling.
enum layout_t { mem_row_major = 0, mem_col_major = 1 };

namespace detail {

namespace vt = vortex::tensor;

// GRXCP element type -> GRX-G100 tensor format descriptor.
template <typename T> struct format;

template <> struct format<half> {
  using type = vt::fp16;
  static_assert(VX_CFG_TCU_FP16_ENABLED,
                "grx::wmma: this build's tensor unit has no fp16 format "
                "(VX_CFG_TCU_FP16_ENABLE).");
};
template <> struct format<float> { using type = vt::fp32; };

static constexpr uint32_t kWarpWidth = VX_CFG_NUM_THREADS;
static constexpr uint32_t kFragRegs  = GRX_WMMA_FRAG_REGS;

template <typename In, typename Out>
using context = vt::wmma_context<kWarpWidth, In, Out, /*is_sparse=*/false,
                                 kFragRegs, /*DK=*/0>;

// The A/B side of the geometry depends only on the input format, and the
// accumulator side only on the output format -- so each fragment can name a
// context without knowing what it will eventually be multiplied against. The
// pairing is checked in mma_sync, where all four fragments are visible.
template <typename T> using in_context  = context<typename format<T>::type, vt::fp32>;
template <typename T> using acc_context = context<vt::fp32, typename format<T>::type>;

template <typename Use, typename T> struct vfragment;
template <typename T> struct vfragment<matrix_a, T> {
  using type = typename in_context<T>::fragment_a;
};
template <typename T> struct vfragment<matrix_b, T> {
  using type = typename in_context<T>::fragment_b;
};
template <typename T> struct vfragment<accumulator, T> {
  using type = typename acc_context<T>::fragment_acc;
};

constexpr vt::mem_layout as_vt_layout(layout_t l) {
  return (l == mem_col_major) ? vt::col_major : vt::row_major;
}

template <typename Layout> struct static_layout;
template <> struct static_layout<row_major> {
  static constexpr vt::mem_layout value = vt::row_major;
};
template <> struct static_layout<col_major> {
  static constexpr vt::mem_layout value = vt::col_major;
};

// Fragments here carry their own register array rather than embedding the
// vortex fragment, because the vortex fragment type is nested inside a context
// that is parameterised by BOTH element types -- an accumulator would have to
// know its future multiplication partner to name its own type. The copies are
// register-to-register moves between always_inline calls and the compiler
// removes them; if that ever stops being true it will show up as spill traffic,
// not as a wrong answer.
template <typename VFrag, uint32_t N>
__forceinline__ void load_regs(VFrag& dst, const float (&src)[N]) {
  static_assert(VFrag::NR == N, "fragment register count mismatch");
  for (uint32_t i = 0; i < N; ++i) dst.data[i] = src[i];
}

template <typename VFrag, uint32_t N>
__forceinline__ void store_regs(float (&dst)[N], const VFrag& src) {
  static_assert(VFrag::NR == N, "fragment register count mismatch");
  for (uint32_t i = 0; i < N; ++i) dst[i] = src.data[i];
}

}  // namespace detail

// ---------------------------------------------------------------------------
// The tile this build actually provides
// ---------------------------------------------------------------------------
//
// m and n do not depend on the element type; k does, because a 32-bit register
// holds two fp16 elements but only one fp32 element.

template <typename T>
struct tile {
  using ctx = detail::in_context<T>;
  static constexpr int m = (int)ctx::tileM;
  static constexpr int n = (int)ctx::tileN;
  static constexpr int k = (int)ctx::tileK;
};

// ---------------------------------------------------------------------------
// fragment
// ---------------------------------------------------------------------------
//
// All of these are WARP-WIDE operations: every lane of the warp must call them,
// with the same arguments, and the fragment lives distributed across the warp's
// registers. Calling one under divergence is undefined -- the same rule CUDA's
// wmma has, for the same reason.
//
// x[] is the raw register storage. For an accumulator that is one float per
// element, exactly as in CUDA. For an fp16 input fragment it is PACKED -- two
// elements per entry -- because that is how the hardware holds them; CUDA
// exposes half x[] there instead. Reading x on an input fragment is therefore
// not portable, and there is no way to make it portable that does not involve
// repacking the registers on every load.

namespace detail {

// An accumulator's K is not checkable here: it depends on the INPUT element
// type, and an accumulator does not know what it will be multiplied against
// (CUDA declares all three fragments with the same M, N, K, and one of the
// three cannot verify the K). m and n do not depend on the element type, so
// those are checked; K is checked in mma_sync where all four fragments meet.
template <typename Use, typename T, int M, int N, int K>
struct check_shape {
  static constexpr bool ok = (M == tile<T>::m && N == tile<T>::n && K == tile<T>::k);
};
template <typename T, int M, int N, int K>
struct check_shape<accumulator, T, M, N, K> {
  static constexpr bool ok = (M == tile<T>::m && N == tile<T>::n);
};

}  // namespace detail

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
  static_assert(detail::check_shape<Use, T, M, N, K>::ok,
                "grx::wmma::fragment: this build provides a different tile "
                "shape. Use grx::wmma::tile<T>::m / ::n / ::k instead of "
                "literal dimensions -- see the header comment.");

  using element_type = T;
  using use_type     = Use;
  using layout_type  = Layout;
  using vfrag_type   = typename detail::vfragment<Use, T>::type;

  static constexpr int m = M, n = N, k = K;
  static constexpr uint32_t num_elements = vfrag_type::NR;

  float x[num_elements];
};

// The accumulator has no compile-time layout: CUDA passes it at the call, and
// so does this. Declaring one with a Layout argument is a mistake worth
// catching, since the argument would silently do nothing.
template <int M, int N, int K, typename T>
struct fragment<accumulator, M, N, K, T, row_major> {
  static_assert(sizeof(T) == 0,
                "grx::wmma: an accumulator fragment takes no layout argument; "
                "the layout is passed to load_matrix_sync / store_matrix_sync.");
};
template <int M, int N, int K, typename T>
struct fragment<accumulator, M, N, K, T, col_major> {
  static_assert(sizeof(T) == 0,
                "grx::wmma: an accumulator fragment takes no layout argument; "
                "the layout is passed to load_matrix_sync / store_matrix_sync.");
};

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

template <typename Use, int M, int N, int K, typename T, typename Layout,
          typename V>
__forceinline__ void fill_fragment(fragment<Use, M, N, K, T, Layout>& f,
                                   V value) {
  using Frag = fragment<Use, M, N, K, T, Layout>;
  typename Frag::vfrag_type v;
  if constexpr (std::is_same<Use, accumulator>::value) {
    detail::acc_context<T>::fill_fragment(v, value);
  } else {
    detail::in_context<T>::fill_fragment(v, value);
  }
  detail::store_regs(f.x, v);
}

// A and B carry their layout in the type, as in CUDA.
template <typename Use, int M, int N, int K, typename T, typename Layout>
__forceinline__ void load_matrix_sync(fragment<Use, M, N, K, T, Layout>& f,
                                      const T* ptr, unsigned ldm) {
  static_assert(!std::is_same<Use, accumulator>::value,
                "grx::wmma: loading an accumulator needs a layout argument.");
  using Frag = fragment<Use, M, N, K, T, Layout>;
  typename Frag::vfrag_type v;
  detail::in_context<T>::template load_matrix_sync<
      detail::static_layout<Layout>::value>(v, ptr, ldm);
  detail::store_regs(f.x, v);
}

template <int M, int N, int K, typename T>
__forceinline__ void load_matrix_sync(fragment<accumulator, M, N, K, T>& f,
                                      const T* ptr, unsigned ldm,
                                      layout_t layout) {
  using Frag = fragment<accumulator, M, N, K, T>;
  typename Frag::vfrag_type v;
  // The backing API takes the layout as a template argument, so both forms are
  // instantiated and one is selected. CUDA's signature is a runtime layout and
  // keeping it means keeping the branch.
  if (layout == mem_col_major) {
    detail::acc_context<T>::template load_matrix_sync<detail::vt::col_major>(
        v, ptr, ldm);
  } else {
    detail::acc_context<T>::template load_matrix_sync<detail::vt::row_major>(
        v, ptr, ldm);
  }
  detail::store_regs(f.x, v);
}

template <int M, int N, int K, typename T>
__forceinline__ void store_matrix_sync(T* ptr,
                                       const fragment<accumulator, M, N, K, T>& f,
                                       unsigned ldm, layout_t layout) {
  using Frag = fragment<accumulator, M, N, K, T>;
  typename Frag::vfrag_type v;
  detail::load_regs(v, f.x);
  if (layout == mem_col_major) {
    detail::acc_context<T>::template store_matrix_sync<detail::vt::col_major>(
        ptr, v, ldm);
  } else {
    detail::acc_context<T>::template store_matrix_sync<detail::vt::row_major>(
        ptr, v, ldm);
  }
}

// D = A * B + C
template <typename FragD, typename FragA, typename FragB, typename FragC>
__forceinline__ void mma_sync(FragD& d, const FragA& a, const FragB& b,
                              const FragC& c) {
  static_assert(std::is_same<typename FragA::use_type, matrix_a>::value, "A must be matrix_a");
  static_assert(std::is_same<typename FragB::use_type, matrix_b>::value, "B must be matrix_b");
  static_assert(std::is_same<typename FragC::use_type, accumulator>::value, "C must be an accumulator");
  static_assert(std::is_same<typename FragD::use_type, accumulator>::value, "D must be an accumulator");
  static_assert(std::is_same<typename FragA::element_type, typename FragB::element_type>::value,
                "A and B must have the same element type");
  static_assert(std::is_same<typename FragC::element_type, typename FragD::element_type>::value,
                "C and D must have the same element type");
  // This is the check the fragment types cannot make on their own: an
  // accumulator does not know which input type it will be paired with, so its
  // K is only meaningful once all four fragments are in the same call.
  static_assert(FragA::m == FragC::m && FragA::n == FragC::n &&
                    FragA::k == FragC::k,
                "grx::wmma::mma_sync: the fragments describe different tiles");

  using in_t  = typename detail::format<typename FragA::element_type>::type;
  using out_t = typename detail::format<typename FragC::element_type>::type;
  using ctx   = detail::context<in_t, out_t>;

  typename ctx::fragment_a   va;
  typename ctx::fragment_b   vb;
  typename ctx::fragment_acc vc, vd;
  detail::load_regs(va, a.x);
  detail::load_regs(vb, b.x);
  detail::load_regs(vc, c.x);
  ctx::mma_sync(vd, va, vb, vc);
  detail::store_regs(d.x, vd);
}

// ---------------------------------------------------------------------------
// Not here yet
// ---------------------------------------------------------------------------
//
// Named rather than declared, because an empty template declaration reads as an
// API and fails at instantiation with a message about nothing in particular.
//
//   WGMMA warp groups        vortex::tensor::wgmma_context exists and takes
//                            shared-memory matrix descriptors; the wrapper does
//                            not. Needs VX_CFG_TCU_WGMMA_ENABLE, which the
//                            configuration this was developed against has off.
//   2:4 structured sparsity  wmma_context supports it (is_sparse plus a
//                            metadata fragment). Needs VX_CFG_TCU_SPARSE_ENABLE
//                            and a metadata-fragment type with its own gate.
//   bf16, fp8, int8, MX      formats exist in tensor_cfg.h. Some have no
//                            corresponding VX_CFG_TCU_*_ENABLE bit exposed in
//                            the generated configuration, so a wrapper could not
//                            tell whether the unit was built with them. Adding
//                            them means fixing that first.
//   half conversion          see the note on grx::wmma::half above.

}  // namespace wmma
}  // namespace grx

#endif  // GRX_WMMA_H
