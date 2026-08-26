// The texture host API's error surface, on the mock — no kernel needed.
//
// The sampling arithmetic is tier 2's TEXTURE GATE, against PyTorch. What is
// here is everything that can be wrong before a kernel ever runs: formats the
// allocator does not have, bounds on the copy, handles that were never created,
// and the honesty flag that has to keep saying this is software.

#include <grx/grx.h>
#include <grx/grx_texture.h>

#include "grx_test.h"

#include <cstdio>
#include <vector>

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  grxSetDevice(0);

  grxtest::section("the emulation is reported, because it is an emulation");
  grxDeviceProp_t prop{};
  grxtest::check(grxGetDeviceProperties(&prop, 0) == grxSuccess,
                 "device properties");
  // Architecture section 10 rule 5: a software stand-in for hardware is only
  // allowed while a device property says so. This is that check, and it is the
  // reason the three CUDA texture entry points may count as implemented.
  grxtest::check(prop.textureIsEmulated == 1,
                 "textureIsEmulated is set while sampling is done in software");

  grxtest::section("arrays");
  grxArray_t a = nullptr;
  grxtest::check(grxMallocArray(&a, GRX_TEX_FORMAT_FLOAT1, 16, 12) == grxSuccess,
                 "a float array allocates");

  void* data = nullptr;
  size_t pitch = 0, w = 0, h = 0;
  unsigned int fmt = 99;
  grxtest::check(grxArrayGetInfo(a, &data, &pitch, &w, &h, &fmt) == grxSuccess,
                 "and reports where it lives");
  grxtest::check(w == 16 && h == 12 && fmt == GRX_TEX_FORMAT_FLOAT1,
                 "with the shape it was asked for");
  // The pitch is the allocator's business, but it must at least hold a row --
  // a pitch below the row width would corrupt every row after the first.
  grxtest::check(pitch >= 16 * sizeof(float), "and a pitch that holds a row");
  std::printf("        pitch %zu bytes for a %zu-texel row\n", pitch, w);

  grxArray_t bad = nullptr;
  grxtest::check(grxMallocArray(&bad, 77u, 8, 8) == grxErrorInvalidValue,
                 "an unknown channel format is refused, not guessed at");
  grxtest::check(grxMallocArray(&bad, GRX_TEX_FORMAT_FLOAT1, 0, 8) ==
                     grxErrorInvalidValue,
                 "and so is an empty array");

  grxtest::section("copies are bounded in texels");
  std::vector<float> src(16 * 12, 1.0f);
  grxtest::check(grxMemcpy2DToArray(a, 0, 0, src.data(), 16 * sizeof(float), 16,
                                    12, grxMemcpyDefault) == grxSuccess ||
                 grxMemcpy2DToArray(a, 0, 0, src.data(), 16 * sizeof(float), 16,
                                    12, grxMemcpyHostToDevice) == grxSuccess,
                 "a full-array copy is accepted");
  grxtest::check(grxMemcpy2DToArray(a, 1, 0, src.data(), 16 * sizeof(float), 16,
                                    12, grxMemcpyHostToDevice) ==
                     grxErrorInvalidValue,
                 "one texel past the right edge is refused");
  grxtest::check(grxMemcpy2DToArray(a, 0, 1, src.data(), 16 * sizeof(float), 16,
                                    12, grxMemcpyHostToDevice) ==
                     grxErrorInvalidValue,
                 "and one row past the bottom");

  grxtest::section("objects are resources, not values");
  grxTextureDesc_t td{};
  td.addressMode[0] = td.addressMode[1] = GRX_TEX_ADDRESS_CLAMP;
  td.filterMode = GRX_TEX_FILTER_LINEAR;

  grxTextureObject_t o1 = 0, o2 = 0;
  grxtest::check(grxCreateTextureObject(&o1, a, &td) == grxSuccess,
                 "an object is created");
  grxtest::check(grxCreateTextureObject(&o2, a, &td) == grxSuccess,
                 "and a second one over the same array");
  grxtest::check(o1 != o2,
                 "they are distinct: each owns its own device-side descriptor");

  grxTextureDesc_t worse = td;
  worse.addressMode[0] = 42;
  grxTextureObject_t o3 = 0;
  grxtest::check(grxCreateTextureObject(&o3, a, &worse) == grxErrorInvalidValue,
                 "an address mode that does not exist is refused");
  worse = td;
  worse.filterMode = 9;
  grxtest::check(grxCreateTextureObject(&o3, a, &worse) == grxErrorInvalidValue,
                 "and a filter mode that does not exist");

  grxtest::check(grxDestroyTextureObject(o1) == grxSuccess, "an object destroys");
  // The handle is a device ADDRESS, so a stale or invented one looks exactly
  // like a live one. Freeing on the strength of that would be a use-after-free
  // with the caller's name on it.
  grxtest::check(grxDestroyTextureObject(o1) == grxErrorInvalidResourceHandle,
                 "destroying it twice is refused rather than freeing twice");
  grxtest::check(grxDestroyTextureObject((grxTextureObject_t)0x1234) ==
                     grxErrorInvalidResourceHandle,
                 "and a handle nobody created is refused");
  grxtest::check(grxDestroyTextureObject(0) == grxSuccess,
                 "while a null handle is legal, as in CUDA");
  grxtest::check(grxDestroyTextureObject(o2) == grxSuccess,
                 "the second object destroys too");

  grxtest::check(grxFreeArray(a) == grxSuccess, "the array frees");
  grxtest::check(grxFreeArray(nullptr) == grxSuccess,
                 "and freeing null is legal");

  return grxtest::report();
}
