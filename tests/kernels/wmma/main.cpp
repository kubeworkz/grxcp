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

#include "../../common/fp16.h"
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

using grxtest::float_to_half;
using grxtest::half_to_float;

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
  // -------------------------------------------------------------------------
  // int8 in, int32 out
  // -------------------------------------------------------------------------
  //
  // Tested separately rather than assumed from the fp16 result: it is a
  // different tile, a different format id and a different accumulator type,
  // and the only thing the two paths share is the plumbing. The arithmetic is
  // integer, so the comparison is exact by construction -- there is no
  // tolerance here at all, not even the "chosen so it is exact" kind.
  {
    grxFunction_t i8shape = nullptr, i8gemm = nullptr;
    if (grxModuleGetFunction(&i8shape, mod, "wmma_i8_shape") != grxSuccess) {
      std::printf("int8: the module has no int8 entry points\n");
    } else {
      void* d_sh = nullptr;
      CHECK(grxMalloc(&d_sh, WMMA_I8_COUNT * sizeof(uint32_t)));
      CHECK(grxMemset(d_sh, 0, WMMA_I8_COUNT * sizeof(uint32_t)));
      wmma_i8_shape_args a{};
      a.out = (uint64_t)(uintptr_t)d_sh;
      CHECK(grxLaunchFunction(i8shape, dim3_t{1, 1, 1}, dim3_t{warp, 1, 1}, &a,
                              sizeof(a), 0, nullptr));
      CHECK(grxDeviceSynchronize());
      uint32_t ish[WMMA_I8_COUNT] = {0};
      CHECK(grxMemcpy(ish, d_sh, sizeof(ish), grxMemcpyDefault));
      CHECK(grxFree(d_sh));

      if (!ish[WMMA_I8_ENABLED]) {
        std::printf("int8: this build's tensor unit has no int8 format "
                    "(rebuild with -DVX_CFG_TCU_INT8_ENABLE)\n");
      } else if (grxModuleGetFunction(&i8gemm, mod, "wmma_i8_gemm_tile") !=
                 grxSuccess) {
        std::printf("int8: shape reports enabled but the GEMM entry is "
                    "missing\n");
        ++failures;
      } else {
        const int im = (int)ish[WMMA_I8_M], in_ = (int)ish[WMMA_I8_N],
                  ik = (int)ish[WMMA_I8_K];
        std::printf("int8 tile %dx%dx%d (int8 in, int32 out), registers per "
                    "lane a=%u b=%u acc=%u\n", im, in_, ik,
                    ish[WMMA_I8_REGS_A], ish[WMMA_I8_REGS_B],
                    ish[WMMA_I8_REGS_ACC]);

        // The int8 tile must be the fp16 tile with twice the depth: same m and
        // n, k doubled, because a register holds four int8 where it holds two
        // fp16. Checked rather than assumed -- if it ever stops being true,
        // every kernel that sizes a staging buffer from one shape and indexes
        // it with the other is wrong.
        if (im != m || in_ != n || ik != 2 * k) {
          std::printf("  FAIL  int8 tile %dx%dx%d is not the fp16 tile "
                      "%dx%dx%d with twice the depth\n", im, in_, ik, m, n, k);
          ++failures;
        }

        std::vector<int8_t>  A((size_t)im * ik), B((size_t)ik * in_);
        std::vector<int32_t> C((size_t)im * in_), D((size_t)im * in_, -1);
        // Small values with mixed signs. The largest product is 7*7 = 49 and
        // the deepest sum is k of them, so nothing here comes near an int32.
        for (size_t i = 0; i < A.size(); ++i) A[i] = (int8_t)((int)(i * 5 % 15) - 7);
        for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)((int)(i * 3 % 13) - 6);
        for (size_t i = 0; i < C.size(); ++i) C[i] = (int32_t)(i * 11 % 97) - 48;

        std::vector<int32_t> want((size_t)im * in_, 0);
        for (int r = 0; r < im; ++r)
          for (int c = 0; c < in_; ++c) {
            int32_t acc = C[(size_t)r * in_ + c];
            for (int l = 0; l < ik; ++l)
              acc += (int32_t)A[(size_t)r * ik + l] * (int32_t)B[(size_t)c * ik + l];
            want[(size_t)r * in_ + c] = acc;
          }

        void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
        CHECK(grxMalloc(&dA, A.size()));
        CHECK(grxMalloc(&dB, B.size()));
        CHECK(grxMalloc(&dC, C.size() * sizeof(int32_t)));
        CHECK(grxMalloc(&dD, D.size() * sizeof(int32_t)));
        CHECK(grxMemcpy(dA, A.data(), A.size(), grxMemcpyDefault));
        CHECK(grxMemcpy(dB, B.data(), B.size(), grxMemcpyDefault));
        CHECK(grxMemcpy(dC, C.data(), C.size() * sizeof(int32_t), grxMemcpyDefault));
        CHECK(grxMemcpy(dD, D.data(), D.size() * sizeof(int32_t), grxMemcpyDefault));

        wmma_i8_gemm_args g{};
        g.a = (uint64_t)(uintptr_t)dA;
        g.b = (uint64_t)(uintptr_t)dB;
        g.c = (uint64_t)(uintptr_t)dC;
        g.d = (uint64_t)(uintptr_t)dD;
        g.lda = (uint32_t)ik; g.ldb = (uint32_t)ik; g.ldc = (uint32_t)in_;
        g.accumulate = 1;
        CHECK(grxLaunchFunction(i8gemm, dim3_t{1, 1, 1}, dim3_t{warp, 1, 1}, &g,
                                sizeof(g), 0, nullptr));
        CHECK(grxDeviceSynchronize());
        CHECK(grxMemcpy(D.data(), dD, D.size() * sizeof(int32_t), grxMemcpyDefault));

        int bad = 0;
        for (size_t i = 0; i < D.size(); ++i) {
          if (D[i] != want[i]) {
            if (bad < 4)
              std::printf("        D[%zu] got %d want %d\n", i, D[i], want[i]);
            ++bad;
          }
        }
        if (bad) {
          std::printf("  FAIL  int8 tile: %d of %zu elements wrong\n", bad,
                      D.size());
          ++failures;
        } else {
          std::printf("  ok    int8 tile matches an integer reference exactly, "
                      "all %zu elements\n", D.size());
        }
        CHECK(grxFree(dA)); CHECK(grxFree(dB));
        CHECK(grxFree(dC)); CHECK(grxFree(dD));
      }
    }
  }

  CHECK(grxModuleUnload(mod));
  return failures ? 1 : 0;
}
