// IEEE binary16 conversion for tests, in one place.
//
// Two gates need it -- the WMMA tile gate and the grxBLAS GemmEx gate -- and
// two copies of a rounding routine is two chances to round differently. It
// lives here rather than in the platform headers because nothing on the device
// side has a tested conversion yet (see the note on grx::wmma::half); when one
// lands, this becomes a check against it rather than the only implementation.
//
// Only the normal range is handled with any care. Every caller asserts that
// the values it uses survive a round trip exactly, so a subnormal or an
// overflow surfaces as a failed input check rather than as a wrong matrix.

#ifndef GRXCP_TEST_FP16_H
#define GRXCP_TEST_FP16_H

#include <cstdint>
#include <cstring>

namespace grxtest {

inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign   = (x >> 16) & 0x8000u;
  const uint32_t biased = (x >> 23) & 0xffu;
  const uint32_t mant   = x & 0x7fffffu;

  if (biased == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));

  const int32_t exp = (int32_t)biased - 127 + 15;
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);   // overflow -> inf
  if (exp <= 0)    return (uint16_t)sign;               // underflow -> zero

  uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;   // nearest, ties even
  return (uint16_t)h;
}

inline float half_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      int shift = 0;
      while (!(mant & 0x400u)) { mant <<= 1; ++shift; }
      mant &= 0x3ffu;
      x = sign | ((uint32_t)(127 - 15 - shift) << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    x = sign | 0x7f800000u | (mant << 13);
  } else {
    x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

// True when `f` survives the round trip unchanged, which is what lets a gate
// compare exactly instead of within a tolerance.
inline bool exact_in_half(float f) { return half_to_float(float_to_half(f)) == f; }

}  // namespace grxtest

#endif  // GRXCP_TEST_FP16_H
