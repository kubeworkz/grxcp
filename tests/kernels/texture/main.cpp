// TEXTURE GATE — grx::tex2D against PyTorch's grid_sample, on a real device.
//
// WHY PYTORCH. A sampler is almost entirely convention: where a texel's centre
// sits, which way floor rounds a negative coordinate, what clamp means at the
// far edge. A reference written from the same reasoning as the implementation
// agrees with it whether or not either is right. torch.nn.functional
// .grid_sample with align_corners=False uses exactly CUDA's convention, and
// three of its padding modes are three of our four address modes.
// texture_ref.py explains the mapping and generates the numbers.
//
// WRAP is the fourth mode and grid_sample has no equivalent, so it is checked
// against a from-specification reference in this file. That is a weaker check
// and it is labelled as one rather than quietly counted with the others.
//
// THIS IS A SOFTWARE SAMPLER. The TEX units are driven by the graphics path
// and are not reachable from compute (cuda_mapping.md 7.8), so what passes
// here is the arithmetic, not the hardware. grxDeviceProp_t.textureIsEmulated
// reads 1 and this gate checks that it does -- a green run that stopped
// reporting the emulation would be the worst outcome available.

#include <grx/grx.h>
#include <grx/grx_texture.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"

#define CHECK(call)                                                      \
  do {                                                                   \
    grxError_t e_ = (call);                                              \
    if (e_ != grxSuccess) {                                              \
      std::fprintf(stderr, "%s -> %s\n", #call, grxGetErrorString(e_));  \
      return 1;                                                          \
    }                                                                    \
  } while (0)

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

struct Ref {
  uint32_t w = 0, h = 0, count = 0, modes = 0;
  std::vector<float> texels, coords, expected;
};

bool load_ref(const char* path, Ref* r) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  uint32_t hdr[4];
  if (std::fread(hdr, sizeof hdr, 1, f) != 1) { std::fclose(f); return false; }
  r->w = hdr[0]; r->h = hdr[1]; r->count = hdr[2]; r->modes = hdr[3];
  r->texels.resize((size_t)r->w * r->h);
  r->coords.resize((size_t)r->count * 2);
  r->expected.resize((size_t)r->modes * r->count);
  bool ok = std::fread(r->texels.data(), 4, r->texels.size(), f) == r->texels.size()
         && std::fread(r->coords.data(), 4, r->coords.size(), f) == r->coords.size()
         && std::fread(r->expected.data(), 4, r->expected.size(), f) == r->expected.size();
  std::fclose(f);
  return ok;
}

// The from-specification reference, used ONLY for WRAP, which grid_sample
// cannot express. Written from CUDA's stated convention rather than from
// grx_tex.h, which is the whole point of having it.
float wrap_index(int i, int n) {
  int m = i % n;
  return (float)((m < 0) ? m + n : m);
}

float reference_wrap(const Ref& r, float x, float y) {
  const float fx = x - 0.5f, fy = y - 0.5f;
  const float x0 = std::floor(fx), y0 = std::floor(fy);
  const float ax = fx - x0, ay = fy - y0;
  auto at = [&](int ix, int iy) {
    const int cx = (int)wrap_index(ix, (int)r.w);
    const int cy = (int)wrap_index(iy, (int)r.h);
    return r.texels[(size_t)cy * r.w + cx];
  };
  const int ix = (int)x0, iy = (int)y0;
  const float top = at(ix, iy)     + ax * (at(ix + 1, iy)     - at(ix, iy));
  const float bot = at(ix, iy + 1) + ax * (at(ix + 1, iy + 1) - at(ix, iy + 1));
  return top + ay * (bot - top);
}

}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "texture.vxbin";
  const char* refpath = (argc > 2) ? argv[2] : "texture_ref.bin";

  Ref ref;
  if (!load_ref(refpath, &ref)) {
    std::printf("no reference at %s; run tests/kernels/texture/texture_ref.py\n",
                refpath);
    return 77;
  }

  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));
  std::printf("%s: sampling in software, textureIsEmulated=%d\n",
              prop.name, prop.textureIsEmulated);

  // The honesty flag is part of the gate, not commentary. If a TEX path ever
  // lands, this fails and whoever landed it has to come and say so here.
  expect(prop.textureIsEmulated == 1,
         "the device reports that texture sampling is emulated");

  grxModule_t mod = nullptr;
  grxError_t e = grxModuleLoad(&mod, image);
  if (e != grxSuccess) {
    std::printf("no kernel image at %s; skipping\n", image);
    return 77;
  }
  grxFunction_t fn = nullptr;
  CHECK(grxModuleGetFunction(&fn, mod, "texture_sample"));

  grxArray_t array = nullptr;
  CHECK(grxMallocArray(&array, GRX_TEX_FORMAT_FLOAT1, ref.w, ref.h));
  CHECK(grxMemcpy2DToArray(array, 0, 0, ref.texels.data(),
                           (size_t)ref.w * sizeof(float), ref.w, ref.h,
                           grxMemcpyHostToDevice));

  void* dCoords = nullptr;
  void* dOut = nullptr;
  CHECK(grxMalloc(&dCoords, ref.coords.size() * sizeof(float)));
  CHECK(grxMalloc(&dOut, (size_t)ref.count * sizeof(float)));
  CHECK(grxMemcpy(dCoords, ref.coords.data(), ref.coords.size() * sizeof(float),
                  grxMemcpyHostToDevice));

  struct Case { unsigned int mode; const char* name; bool from_torch; };
  const Case cases[4] = {
    {GRX_TEX_ADDRESS_CLAMP,  "clamp",  true},
    {GRX_TEX_ADDRESS_BORDER, "border", true},
    {GRX_TEX_ADDRESS_MIRROR, "mirror", true},
    {GRX_TEX_ADDRESS_WRAP,   "wrap",   false},
  };

  std::vector<float> got((size_t)ref.count);
  double worst_overall = 0.0;

  for (int c = 0; c < 4; ++c) {
    grxTextureDesc_t td{};
    td.addressMode[0] = cases[c].mode;
    td.addressMode[1] = cases[c].mode;
    td.filterMode     = GRX_TEX_FILTER_LINEAR;
    td.normalizedCoords = 0;
    for (int i = 0; i < 4; ++i) td.borderColor[i] = 0.0f;

    grxTextureObject_t obj = 0;
    CHECK(grxCreateTextureObject(&obj, array, &td));

    texture_args args{};
    args.abi_version = TEXTURE_ARGS_ABI;
    args.count  = ref.count;
    args.object = obj;
    args.coords = (uint64_t)(uintptr_t)dCoords;
    args.out    = (uint64_t)(uintptr_t)dOut;

    CHECK(grxMemset(dOut, 0, (size_t)ref.count * sizeof(float)));
    const uint32_t block = 64;
    const uint32_t grid  = (ref.count + block - 1) / block;
    CHECK(grxLaunchFunction(fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                            &args, sizeof args, 0, nullptr));
    CHECK(grxDeviceSynchronize());
    CHECK(grxMemcpy(got.data(), dOut, (size_t)ref.count * sizeof(float),
                    grxMemcpyDeviceToHost));

    double worst = 0.0;
    int worst_i = 0;
    for (uint32_t i = 0; i < ref.count; ++i) {
      const float want = cases[c].from_torch
          ? ref.expected[(size_t)c * ref.count + i]
          : reference_wrap(ref, ref.coords[2 * i], ref.coords[2 * i + 1]);
      const double d = std::fabs((double)got[i] - (double)want);
      if (d > worst) { worst = d; worst_i = (int)i; }
    }
    if (worst > worst_overall) worst_overall = worst;

    char what[128];
    std::snprintf(what, sizeof what,
                  "%-6s bilinear, %u coords vs %s (worst %.2e at x=%.2f y=%.2f)",
                  cases[c].name, ref.count,
                  cases[c].from_torch ? "PyTorch" : "the spec", worst,
                  ref.coords[2 * worst_i], ref.coords[2 * worst_i + 1]);
    // 2e-6 absolute. The device computes in fp32 and PyTorch in fp32 with a
    // different association order, so bitwise agreement is not available and
    // claiming it would be a gate nobody could keep.
    expect(worst < 2e-6, what);

    CHECK(grxDestroyTextureObject(obj));
  }

  // POINT filtering, separately, because it shares no arithmetic with the
  // bilinear path and a gate that only ran linear would leave it unchecked.
  {
    grxTextureDesc_t td{};
    td.addressMode[0] = td.addressMode[1] = GRX_TEX_ADDRESS_CLAMP;
    td.filterMode = GRX_TEX_FILTER_POINT;
    td.normalizedCoords = 0;
    grxTextureObject_t obj = 0;
    CHECK(grxCreateTextureObject(&obj, array, &td));

    texture_args args{};
    args.abi_version = TEXTURE_ARGS_ABI;
    args.count = ref.count;
    args.object = obj;
    args.coords = (uint64_t)(uintptr_t)dCoords;
    args.out = (uint64_t)(uintptr_t)dOut;
    CHECK(grxMemset(dOut, 0, (size_t)ref.count * sizeof(float)));
    const uint32_t block = 64;
    CHECK(grxLaunchFunction(fn, dim3_t{(ref.count + block - 1) / block, 1, 1},
                            dim3_t{block, 1, 1}, &args, sizeof args, 0, nullptr));
    CHECK(grxDeviceSynchronize());
    CHECK(grxMemcpy(got.data(), dOut, (size_t)ref.count * sizeof(float),
                    grxMemcpyDeviceToHost));

    int wrong = 0;
    for (uint32_t i = 0; i < ref.count; ++i) {
      const float x = ref.coords[2 * i], y = ref.coords[2 * i + 1];
      int ix = (int)std::floor(x), iy = (int)std::floor(y);
      ix = (ix < 0) ? 0 : ((ix >= (int)ref.w) ? (int)ref.w - 1 : ix);
      iy = (iy < 0) ? 0 : ((iy >= (int)ref.h) ? (int)ref.h - 1 : iy);
      // Point filtering selects a stored texel, so this one IS bitwise.
      if (got[i] != ref.texels[(size_t)iy * ref.w + ix]) ++wrong;
    }
    expect(wrong == 0, "point filtering returns the stored texel, bitwise, at "
                       "every coordinate including the out-of-range ones");
    CHECK(grxDestroyTextureObject(obj));
  }

  // NORMALIZED coordinates must land where the same point in texel units does.
  // Checked as an identity rather than against a second reference: the two
  // spellings of one location cannot disagree.
  {
    grxTextureDesc_t td{};
    td.addressMode[0] = td.addressMode[1] = GRX_TEX_ADDRESS_CLAMP;
    td.filterMode = GRX_TEX_FILTER_LINEAR;
    td.normalizedCoords = 1;
    grxTextureObject_t obj = 0;
    CHECK(grxCreateTextureObject(&obj, array, &td));

    std::vector<float> ncoords(ref.coords.size());
    for (uint32_t i = 0; i < ref.count; ++i) {
      ncoords[2 * i]     = ref.coords[2 * i]     / (float)ref.w;
      ncoords[2 * i + 1] = ref.coords[2 * i + 1] / (float)ref.h;
    }
    void* dn = nullptr;
    CHECK(grxMalloc(&dn, ncoords.size() * sizeof(float)));
    CHECK(grxMemcpy(dn, ncoords.data(), ncoords.size() * sizeof(float),
                    grxMemcpyHostToDevice));

    texture_args args{};
    args.abi_version = TEXTURE_ARGS_ABI;
    args.count = ref.count;
    args.object = obj;
    args.coords = (uint64_t)(uintptr_t)dn;
    args.out = (uint64_t)(uintptr_t)dOut;
    CHECK(grxMemset(dOut, 0, (size_t)ref.count * sizeof(float)));
    const uint32_t block = 64;
    CHECK(grxLaunchFunction(fn, dim3_t{(ref.count + block - 1) / block, 1, 1},
                            dim3_t{block, 1, 1}, &args, sizeof args, 0, nullptr));
    CHECK(grxDeviceSynchronize());
    CHECK(grxMemcpy(got.data(), dOut, (size_t)ref.count * sizeof(float),
                    grxMemcpyDeviceToHost));

    double worst = 0.0;
    for (uint32_t i = 0; i < ref.count; ++i) {
      const double d = std::fabs((double)got[i] -
                                 (double)ref.expected[0 * ref.count + i]);
      if (d > worst) worst = d;
    }
    char what[128];
    std::snprintf(what, sizeof what,
                  "normalized coordinates reach the same texels (worst %.2e)",
                  worst);
    expect(worst < 4e-6, what);
    CHECK(grxDestroyTextureObject(obj));
    grxFree(dn);
  }

  // The object is a resource with a lifetime, and a handle that was never
  // created must not be freed on the strength of looking like an address.
  expect(grxDestroyTextureObject((grxTextureObject_t)0xDEADBEEF) ==
             grxErrorInvalidResourceHandle,
         "destroying a handle that was never created is refused");
  expect(grxDestroyTextureObject(0) == grxSuccess,
         "and destroying a null handle is legal, as it is in CUDA");

  std::printf("        worst absolute deviation across every mode: %.2e\n",
              worst_overall);

  grxFree(dCoords);
  grxFree(dOut);
  grxFreeArray(array);
  grxModuleUnload(mod);

  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
