// WMMA gate: multiply one tile on the tensor unit and check every element.
//
// The comparison is EXACT, not approximate, and that is deliberate. The inputs
// are multiples of 0.5 in [-4, 4]: every one is exactly representable in
// binary16, every product is exact in binary32 (an 11-bit by 11-bit product
// fits in 24 bits of significand), and every partial sum stays well inside the
// exact-integer range. So a correct tensor unit reproduces the reference bit
// for bit, and a tolerance would only serve to hide a wrong answer.
//
// The host asks the DEVICE for the tile shape rather than assuming 16x16x16 or
// any other number: GRX-G100's tile is derived from the configuration. See
// include/grx/device/grx_wmma.h.

#include <grx/grx.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common.h"

#define CHECK(call)                                                      \
  do {                                                                   \
    grxError_t e_ = (call);                                              \
    if (e_ != grxSuccess) {                                              \
      std::fprintf(stderr, "%s -> %s (%s)\n", #call,                     \
                   grxGetErrorString(e_), grxGetErrorName(e_));          \
      return 1;                                                          \
    }                                                                    \
  } while (0)

namespace {

// IEEE binary16 conversion, round to nearest even. Only the normal range is
// handled with any care; the gate asserts that every value it uses survives a
// round trip exactly, so a subnormal or an overflow shows up as a failed input
// check rather than as a silently wrong matrix.
uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t biased = (x >> 23) & 0xffu;
  const uint32_t mant = x & 0x7fffffu;

  if (biased == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));

  const int32_t exp = (int32_t)biased - 127 + 15;
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);   // overflow -> inf
  if (exp <= 0)    return (uint16_t)sign;               // underflow -> zero

  uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return (uint16_t)h;
}

float half_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
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

// Multiples of 0.5 in [-4, 4], deterministic, mixed signs. A matrix of ones
// would give the same answer under most index errors, which is the opposite of
// what a gate is for.
float sample(unsigned& seed) {
  seed = seed * 1664525u + 1013904223u;
  return (float)((int)((seed >> 16) % 17u) - 8) * 0.5f;
}

}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "wmma.vxbin";

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));
  if (!(prop.capabilities & GRX_CAP_TENSOR_CORE)) {
    // A device without a tensor unit is a configuration, not a failure. The
    // kernel could not even have been built for it -- grx_wmma.h refuses.
    std::printf("SKIPPED: %s reports no tensor unit\n", prop.name);
    return 77;
  }

  grxModule_t   mod   = nullptr;
  grxFunction_t shape = nullptr;
  grxFunction_t gemm  = nullptr;
  CHECK(grxModuleLoad(&mod, image));
  CHECK(grxModuleGetFunction(&shape, mod, "wmma_shape"));
  CHECK(grxModuleGetFunction(&gemm, mod, "wmma_gemm_tile"));

  const unsigned warp = (unsigned)prop.warpSize;

  // --- ask the device what tile it has ------------------------------------
  void* d_shape = nullptr;
  CHECK(grxMalloc(&d_shape, WMMA_SHAPE_COUNT * sizeof(uint32_t)));
  CHECK(grxMemset(d_shape, 0, WMMA_SHAPE_COUNT * sizeof(uint32_t)));

  wmma_shape_args sargs{};
  sargs.out = (uint64_t)(uintptr_t)d_shape;
  CHECK(grxLaunchFunction(shape, dim3_t{1, 1, 1}, dim3_t{warp, 1, 1}, &sargs,
                          sizeof(sargs), 0, nullptr));
  CHECK(grxDeviceSynchronize());

  uint32_t sh[WMMA_SHAPE_COUNT] = {0};
  CHECK(grxMemcpy(sh, d_shape, sizeof(sh), grxMemcpyDefault));
  CHECK(grxFree(d_shape));

  const int m = (int)sh[WMMA_SHAPE_M];
  const int n = (int)sh[WMMA_SHAPE_N];
  const int k = (int)sh[WMMA_SHAPE_K];
  std::printf("tile %dx%dx%d (fp16 in, fp32 out), registers per lane a=%u "
              "b=%u acc=%u\n", m, n, k, sh[WMMA_SHAPE_REGS_A],
              sh[WMMA_SHAPE_REGS_B], sh[WMMA_SHAPE_REGS_ACC]);

  if (m <= 0 || n <= 0 || k <= 0) {
    std::fprintf(stderr, "FAILED: device reported a degenerate tile\n");
    return 1;
  }
  // The kernel's compiled-in warp width against the runtime's reported one. If
  // these disagree the module was built for a different machine than the one
  // about to run it, and every number below would be meaningless.
  if (sh[WMMA_SHAPE_WARP] != warp) {
    std::fprintf(stderr,
                 "FAILED: kernel was compiled for a warp width of %u, device "
                 "reports %u. The .vxbin and the runtime come from different\n"
                 "        configurations -- see ci/README.md, "
                 "\"configuration provenance\".\n",
                 sh[WMMA_SHAPE_WARP], warp);
    return 1;
  }

  // --- build the operands -------------------------------------------------
  const int lda = k, ldb = k, ldc = n;
  std::vector<float>    fa((size_t)m * k), fb((size_t)n * k), hc((size_t)m * n);
  std::vector<uint16_t> ha(fa.size()), hb(fb.size());

  unsigned seed = 12345u;
  for (size_t i = 0; i < fa.size(); ++i) fa[i] = sample(seed);
  for (size_t i = 0; i < fb.size(); ++i) fb[i] = sample(seed);
  for (size_t i = 0; i < hc.size(); ++i) hc[i] = sample(seed) * 4.0f;

  for (size_t i = 0; i < fa.size(); ++i) ha[i] = float_to_half(fa[i]);
  for (size_t i = 0; i < fb.size(); ++i) hb[i] = float_to_half(fb[i]);

  // The exactness the comparison relies on, checked rather than assumed.
  for (size_t i = 0; i < fa.size(); ++i) {
    if (half_to_float(ha[i]) != fa[i]) {
      std::fprintf(stderr, "FAILED: A[%zu] = %g is not exact in fp16\n", i, fa[i]);
      return 1;
    }
  }
  for (size_t i = 0; i < fb.size(); ++i) {
    if (half_to_float(hb[i]) != fb[i]) {
      std::fprintf(stderr, "FAILED: B[%zu] = %g is not exact in fp16\n", i, fb[i]);
      return 1;
    }
  }

  void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
  CHECK(grxMalloc(&dA, ha.size() * sizeof(uint16_t)));
  CHECK(grxMalloc(&dB, hb.size() * sizeof(uint16_t)));
  CHECK(grxMalloc(&dC, hc.size() * sizeof(float)));
  CHECK(grxMalloc(&dD, hc.size() * sizeof(float)));
  CHECK(grxMemcpy(dA, ha.data(), ha.size() * sizeof(uint16_t), grxMemcpyDefault));
  CHECK(grxMemcpy(dB, hb.data(), hb.size() * sizeof(uint16_t), grxMemcpyDefault));
  CHECK(grxMemcpy(dC, hc.data(), hc.size() * sizeof(float), grxMemcpyDefault));

  int failures = 0;
  for (int accumulate = 0; accumulate <= 1; ++accumulate) {
    // Poison D so a kernel that writes nothing fails instead of inheriting a
    // previous pass's correct answer.
    std::vector<float> poison(hc.size(), -12345.0f);
    CHECK(grxMemcpy(dD, poison.data(), poison.size() * sizeof(float),
                    grxMemcpyDefault));

    wmma_gemm_args args{};
    args.a = (uint64_t)(uintptr_t)dA;
    args.b = (uint64_t)(uintptr_t)dB;
    args.c = (uint64_t)(uintptr_t)dC;
    args.d = (uint64_t)(uintptr_t)dD;
    args.lda = (uint32_t)lda;
    args.ldb = (uint32_t)ldb;
    args.ldc = (uint32_t)ldc;
    args.accumulate = (uint32_t)accumulate;

    CHECK(grxLaunchFunction(gemm, dim3_t{1, 1, 1}, dim3_t{warp, 1, 1}, &args,
                            sizeof(args), 0, nullptr));
    CHECK(grxDeviceSynchronize());

    std::vector<float> got(hc.size(), 0.0f);
    CHECK(grxMemcpy(got.data(), dD, got.size() * sizeof(float), grxMemcpyDefault));

    int bad = 0;
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float want = accumulate ? hc[(size_t)i * ldc + j] : 0.0f;
        for (int l = 0; l < k; ++l)
          want += fa[(size_t)i * lda + l] * fb[(size_t)j * ldb + l];
        const float have = got[(size_t)i * ldc + j];
        if (have != want) {
          if (bad < 4)
            std::fprintf(stderr, "  (%d,%d) got %g want %g\n", i, j, have, want);
          ++bad;
        }
      }
    }

    const char* label = accumulate ? "D = A*B + C" : "D = A*B";
    if (bad) {
      std::fprintf(stderr, "FAILED: %s, %d/%d elements wrong\n", label, bad, m * n);
      ++failures;
    } else {
      std::printf("PASSED: %s, %d/%d elements exact\n", label, m * n, m * n);
    }
  }

  CHECK(grxFree(dA));
  CHECK(grxFree(dB));
  CHECK(grxFree(dC));
  CHECK(grxFree(dD));
  CHECK(grxModuleUnload(mod));
  return failures ? 1 : 0;
}
