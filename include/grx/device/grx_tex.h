// GRXCP — sampling a texture from a kernel, in software.
//
// There is no TEX unit reachable from compute (cuda_mapping.md 7.8), so every
// line below runs in YOUR warp: the address arithmetic, the wrap, the four
// loads a bilinear fetch needs, and the interpolation between them. A texture
// read here costs what you would pay to write it by hand, because it is what
// you would write by hand.
//
// grxDeviceProp_t.textureIsEmulated is 1 while that is true. See grx_texture.h.
//
// THE COORDINATE CONVENTION IS CUDA'S, and it is worth stating because it is
// the part ports get wrong. A texel's CENTRE is at integer + 0.5. So an
// unnormalized point fetch at x = 0.0 and at x = 0.9 both return texel 0, and
// a linear fetch at x = 1.0 sits exactly between texels 0 and 1 -- which is
// why the code below subtracts 0.5 before splitting into an index and a
// fraction, and not after.
//
// FILTER WEIGHTS ARE FULL-PRECISION. NVIDIA hardware quantizes the fraction to
// 8 bits, making a bilinear read a 256-tread staircase. This does not, so
// values differ from a CUDA reference by up to half a tread. Reproducing the
// staircase would be emulating a wart to more decimal places.

#ifndef GRX_DEVICE_TEX_H
#define GRX_DEVICE_TEX_H

#include "../grx_texture.h"
#include "grx_device.h"

namespace grx {

struct float4_t { float x, y, z, w; };

namespace tex_detail {

// FLOOR, BY HAND, AND THE REASON IS NOT STYLE.
//
// __builtin_floorf on a DIVERGENT value does not compile for this device:
//
//     error: unimplemented divergent codegen found!
//
// Measured across the family, because knowing which ones is the difference
// between a workaround and a superstition:
//
//     floorf  ceilf  truncf  roundf  rintf   -> all five fail
//     nearbyintf  fabsf  sqrtf              -> all three compile
//
// The five that fail are the ones that lower to a float->int->float sequence
// with an EXPLICIT rounding mode; nearbyintf uses the dynamic mode and lowers
// differently, which is why it survives. Nothing else in the tree had ever
// needed a rounding builtin in divergent code, so this is the first time it
// could have been found. Registered as cuda_mapping.md 7.24.
//
// The replacement is a conversion and a compare, both of which the backend
// handles. Correct for the range a texture coordinate lives in; it is not a
// general floorf, and it is not offered as one -- values beyond 2^31 round
// through int and would be wrong, which coordinates into a 32-bit-indexed
// texture cannot reach.
__device__ inline float floor_f(float v) {
  const int i = (int)v;                 // truncates toward zero
  return (float)(((float)i > v) ? i - 1 : i);
}


__device__ inline int clamp_index(int i, int n, unsigned int mode,
                                  bool* out_of_range) {
  *out_of_range = false;
  switch (mode) {
    case GRX_TEX_ADDRESS_WRAP: {
      int m = i % n;
      return (m < 0) ? m + n : m;
    }
    case GRX_TEX_ADDRESS_MIRROR: {
      const int period = 2 * n;
      int m = i % period;
      if (m < 0) m += period;
      return (m < n) ? m : (period - 1 - m);
    }
    case GRX_TEX_ADDRESS_BORDER:
      if (i < 0 || i >= n) { *out_of_range = true; return 0; }
      return i;
    case GRX_TEX_ADDRESS_CLAMP:
    default:
      return (i < 0) ? 0 : ((i >= n) ? n - 1 : i);
  }
}

__device__ inline float load1(const grx_texture_desc* d, int x, int y) {
  const char* base = (const char*)(unsigned long)d->data;
  return *(const float*)(base + (unsigned)y * d->pitch + (unsigned)x * 4u);
}

__device__ inline float4_t load4(const grx_texture_desc* d, int x, int y) {
  const char* base = (const char*)(unsigned long)d->data;
  const float* p = (const float*)(base + (unsigned)y * d->pitch +
                                  (unsigned)x * 16u);
  float4_t v; v.x = p[0]; v.y = p[1]; v.z = p[2]; v.w = p[3];
  return v;
}

// BRANCH-FREE ON PURPOSE, AND THE REASON IS NOT SETTLED.
//
// This was written with an early return -- `if (ox || oy) return border;` --
// and it produced WRONG VALUES in the border address mode at scale: three of
// four modes agreed with PyTorch to 2e-7 and border was out by 8.9e-1, as
// though the out-of-range tap had returned the in-range texel. The branch-free
// spelling below, which is semantically the same, agrees everywhere.
//
// WHAT IS AND IS NOT KNOWN, because the difference between these two matters
// to whoever reads this next:
//
//   KNOWN. The two spellings disagree in the 81-coordinate, 64-thread gate,
//   reproducibly. The branch-free one is the one that matches a third-party
//   reference.
//
//   NOT KNOWN. Why. A single-lane probe of the ORIGINAL spelling returns the
//   correct 0.0 with every intermediate correct, so it is divergence-related --
//   but three minimal reproducers (a bare divergent early return; one with a
//   bool out-parameter; one with the out-parameter set inside a switch, which
//   is this function's exact shape) all compile and run CORRECTLY at 8 threads.
//   No compiler defect has been isolated and none is being claimed.
//
// So this is a workaround for an unexplained observation, recorded as one. If
// you reduce it, put the reproducer in tests/repro/ and register it; if you
// show the original spelling is fine, delete this comment and simplify.
//
// One gathered texel, with the address modes applied and the border case
// folded in. Templated on the fetch so the two formats share every line of
// the addressing and filtering above them.
template <typename T>
__device__ inline T fetch(const grx_texture_desc* d, int x, int y);

template <> __device__ inline float fetch<float>(const grx_texture_desc* d,
                                                 int x, int y) {
  bool ox = false, oy = false;
  const int xi = clamp_index(x, (int)d->width,  d->address_x, &ox);
  const int yi = clamp_index(y, (int)d->height, d->address_y, &oy);
  const bool oob = ox || oy;
  const float t = load1(d, oob ? 0 : xi, oob ? 0 : yi);
  return oob ? d->border[0] : t;
}

template <> __device__ inline float4_t fetch<float4_t>(const grx_texture_desc* d,
                                                       int x, int y) {
  bool ox = false, oy = false;
  const int xi = clamp_index(x, (int)d->width,  d->address_x, &ox);
  const int yi = clamp_index(y, (int)d->height, d->address_y, &oy);
  const bool oob = ox || oy;
  const float4_t t = load4(d, oob ? 0 : xi, oob ? 0 : yi);
  float4_t r;
  r.x = oob ? d->border[0] : t.x;  r.y = oob ? d->border[1] : t.y;
  r.z = oob ? d->border[2] : t.z;  r.w = oob ? d->border[3] : t.w;
  return r;
}

__device__ inline float lerp1(float a, float b, float t) {
  return a + t * (b - a);
}
__device__ inline float4_t lerp4(float4_t a, float4_t b, float t) {
  float4_t r;
  r.x = lerp1(a.x, b.x, t); r.y = lerp1(a.y, b.y, t);
  r.z = lerp1(a.z, b.z, t); r.w = lerp1(a.w, b.w, t);
  return r;
}
__device__ inline float  mix(float a, float b, float t)  { return lerp1(a, b, t); }
__device__ inline float4_t mix(float4_t a, float4_t b, float t) { return lerp4(a, b, t); }

}  // namespace tex_detail

// Sample a 2D texture. T is float or grx::float4_t.
template <typename T>
__device__ inline T tex2D(grxTextureObject_t object, float x, float y) {
  const grx_texture_desc* d =
      (const grx_texture_desc*)(unsigned long)object;

  // The ABI check, for the same reason every kernel argument struct has one:
  // a host and a device built at different times must produce an obvious
  // wrong answer rather than a subtle one. Zero is the honest return here --
  // there is no error channel inside a sample.
  if (d->abi_version != GRX_TEXTURE_ABI_VERSION) return T{};

  float u = x, v = y;
  if (d->normalized) { u *= (float)d->width; v *= (float)d->height; }

  if (d->filter == GRX_TEX_FILTER_POINT) {
    // floor, not truncate: -0.3 is texel -1 before addressing, and truncation
    // would make it texel 0 and quietly fold the left border into the image.
    const int xi = (int)tex_detail::floor_f(u);
    const int yi = (int)tex_detail::floor_f(v);
    return tex_detail::fetch<T>(d, xi, yi);
  }

  // Linear. The -0.5 puts the sample between texel CENTRES; see the header.
  const float fx = u - 0.5f;
  const float fy = v - 0.5f;
  const float x0 = tex_detail::floor_f(fx);
  const float y0 = tex_detail::floor_f(fy);
  const float ax = fx - x0;
  const float ay = fy - y0;
  const int   ix = (int)x0;
  const int   iy = (int)y0;

  const T c00 = tex_detail::fetch<T>(d, ix,     iy);
  const T c10 = tex_detail::fetch<T>(d, ix + 1, iy);
  const T c01 = tex_detail::fetch<T>(d, ix,     iy + 1);
  const T c11 = tex_detail::fetch<T>(d, ix + 1, iy + 1);

  return tex_detail::mix(tex_detail::mix(c00, c10, ax),
                         tex_detail::mix(c01, c11, ax), ay);
}

// A named sampler, which is what the roadmap called grx::tex<>. It carries the
// handle and nothing else -- the descriptor stays in device memory, read per
// sample, because there is nowhere else for it to live.
template <typename T>
class tex {
 public:
  __device__ explicit tex(grxTextureObject_t object) : object_(object) {}
  __device__ T operator()(float x, float y) const {
    return tex2D<T>(object_, x, y);
  }
  __device__ grxTextureObject_t object() const { return object_; }

 private:
  grxTextureObject_t object_;
};

}  // namespace grx

#endif  // GRX_DEVICE_TEX_H
