// grxdnnAttentionForward against PyTorch.
//
// THE REFERENCE IS SOMEBODY ELSE'S ARITHMETIC, and for this op that is not a
// nicety. Attention is where grxDNN's ROW-major convention meets grxBLAS's
// COLUMN-major one, twice, once transposed. It is almost entirely index
// bookkeeping, and a reference written from the same reasoning as the
// implementation agrees with it perfectly whether or not either is right --
// exactly what `unpack()` in test_grxblas.cpp exists to stop, where two copies
// of one misconception passed a transposed GEMM that transposed nothing.
//
// So the expected values come from torch.nn.functional.scaled_dot_product_
// attention, computed in float64, by a library that knows nothing about how
// GRXCP stores a matrix. tests/libs/attention_ref.py generates them and is
// checked in next to them; the vectors are checked in so this gate needs no
// Python and no torch.
//
// That script also simulates the exact grxblasSgemm calls this implementation
// makes -- leading dimensions, transpose flags and all -- on flat memory, and
// refuses to write the vectors unless that simulation reproduces torch to
// 1e-12. So the layout algebra was settled before any of it reached a device.
//
// WHAT THE SHAPES ARE FOR. The 1x1x1x1 case cannot catch a layout bug at all --
// every transpose of a 1x1 matrix is itself -- and it is in the file to check
// the degenerate path, not the algebra. The ragged ones do that work: S=3 is
// shorter than a warp, S=17 and D=8 divide nothing evenly, and S=40 makes a row
// span many warp strides.
//
// WATCHED FAILING, three ways, because a gate nobody has seen fail is a gate
// nobody knows works:
//
//   transa flipped on the QKᵀ GEMM   every case but 1x1 fails -- as "execution
//                                    failed", since grxBLAS's own leading-
//                                    dimension check refuses it before any
//                                    arithmetic happens
//   Q and K swapped                  every case but 1x1 fails NUMERICALLY, by
//                                    0.117 to 0.316. This is the interesting
//                                    one: passing the operands in the order the
//                                    formula reads is dimensionally valid and
//                                    silently computes scoresᵀ
//   causal mask not applied          ONLY the two causal cases fail, by ~1.0,
//                                    and the five unmasked ones still pass --
//                                    so the mask is doing real work and the
//                                    unmasked cases are not accidentally masked
//
// The middle one is why this file exists. It is the mistake a careful person
// makes, it produces plausible numbers, and nothing but an outside reference
// catches it.

#include <grx/grx.h>
#include <grx/grxdnn.h>

#include "../unit/grx_test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

constexpr uint32_t kMagic = 0x4E544147u;   // 'GATN'
constexpr uint32_t kVersion = 1u;

struct Case {
  uint32_t batch, heads, seq, dim, causal;
  std::vector<float> q, k, v, want;
};

bool read_u32(std::FILE* f, uint32_t* out) {
  return std::fread(out, sizeof(*out), 1, f) == 1;
}

bool read_floats(std::FILE* f, size_t n, std::vector<float>* out) {
  out->resize(n);
  return std::fread(out->data(), sizeof(float), n, f) == n;
}

// The vectors live next to the source, and the harness is told where by argv or
// by an environment variable. Neither is guessed at: a gate that silently found
// no cases and reported success would be worse than one that cannot start.
bool load_cases(const char* path, std::vector<Case>* cases) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;

  uint32_t magic = 0, version = 0, count = 0;
  if (!read_u32(f, &magic) || !read_u32(f, &version) || !read_u32(f, &count) ||
      magic != kMagic || version != kVersion || count == 0) {
    std::fclose(f);
    return false;
  }

  for (uint32_t i = 0; i < count; ++i) {
    Case c{};
    if (!read_u32(f, &c.batch) || !read_u32(f, &c.heads) || !read_u32(f, &c.seq) ||
        !read_u32(f, &c.dim) || !read_u32(f, &c.causal)) {
      std::fclose(f);
      return false;
    }
    const size_t n = (size_t)c.batch * c.heads * c.seq * c.dim;
    if (!read_floats(f, n, &c.q) || !read_floats(f, n, &c.k) ||
        !read_floats(f, n, &c.v) || !read_floats(f, n, &c.want)) {
      std::fclose(f);
      return false;
    }
    cases->push_back(std::move(c));
  }
  std::fclose(f);
  return true;
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) { if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr; }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

bool run_case(grxdnnHandle_t h, const Case& c) {
  char label[128];
  std::snprintf(label, sizeof(label), "B%u H%u S%u D%u%s",
                c.batch, c.heads, c.seq, c.dim, c.causal ? " causal" : "");

  const size_t n = c.q.size();
  const size_t bytes = n * sizeof(float);

  size_t ws_bytes = 0;
  grxdnnStatus_t st = grxdnnAttentionWorkspaceSize(
      (int)c.batch, (int)c.heads, (int)c.seq, (int)c.dim, &ws_bytes);
  if (st != GRXDNN_STATUS_SUCCESS) {
    std::printf("  FAIL  %s: workspace size: %s\n", label,
                grxdnnGetStatusString(st));
    ++grxtest::failures();
    return false;
  }

  Buf dq(bytes), dk(bytes), dv(bytes), dout(bytes), dws(ws_bytes);
  if (!dq.p || !dk.p || !dv.p || !dout.p || !dws.p) {
    std::printf("  FAIL  %s (allocation)\n", label);
    ++grxtest::failures();
    return false;
  }
  grxMemcpy(dq.p, c.q.data(), bytes, grxMemcpyDefault);
  grxMemcpy(dk.p, c.k.data(), bytes, grxMemcpyDefault);
  grxMemcpy(dv.p, c.v.data(), bytes, grxMemcpyDefault);
  // Poison the output so a kernel that never writes is caught as a wrong
  // answer rather than passing on whatever happened to be there.
  std::vector<float> poison(n, -1234.5f);
  grxMemcpy(dout.p, poison.data(), bytes, grxMemcpyDefault);

  st = grxdnnAttentionForward(
      h, (int)c.batch, (int)c.heads, (int)c.seq, (int)c.dim,
      (const float*)dq.p, (const float*)dk.p, (const float*)dv.p,
      c.causal ? GRXDNN_ATTN_MASK_CAUSAL : GRXDNN_ATTN_MASK_NONE,
      dws.p, ws_bytes, (float*)dout.p);
  if (st != GRXDNN_STATUS_SUCCESS) {
    std::printf("  FAIL  %s: %s\n", label, grxdnnGetStatusString(st));
    ++grxtest::failures();
    return false;
  }

  std::vector<float> got(n);
  grxMemcpy(got.data(), dout.p, bytes, grxMemcpyDefault);

  // Two GEMMs and a softmax deep, in fp32, against a float64 reference. The
  // bound scales with the reduction length because that is what sets the
  // accumulation error; the constant is not tuned to make this pass -- the
  // observed worst case sits an order of magnitude below it.
  const double tol = 2e-6 * (double)(c.seq + c.dim);
  double worst = 0.0;
  size_t at = 0;
  for (size_t i = 0; i < n; ++i) {
    const double d = std::fabs((double)got[i] - (double)c.want[i]);
    if (d > worst) { worst = d; at = i; }
  }
  if (worst <= tol) {
    std::printf("  ok    %s (worst |diff| %.3g, tol %.3g)\n", label, worst, tol);
    return true;
  }
  std::printf("  FAIL  %s: worst |diff| %.3g at [%zu] (got %.9g, want %.9g)\n",
              label, worst, at, (double)got[at], (double)c.want[at]);
  ++grxtest::failures();
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : std::getenv("GRXDNN_ATTN_REF");
  if (!path) path = "attention_ref.bin";

  std::vector<Case> cases;
  if (!load_cases(path, &cases)) {
    std::printf("cannot read reference vectors from %s\n"
                "  regenerate with: python3 tests/libs/attention_ref.py "
                "--write tests/libs/attention_ref.bin\n", path);
    return 1;
  }

  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxdnnHandle_t h = nullptr;
  if (grxdnnCreate(&h) != GRXDNN_STATUS_SUCCESS) {
    std::printf("grxdnnCreate failed\n");
    return 1;
  }

  // Probe before claiming to test anything: without the device toolchain there
  // is no .vxbin, every case fails identically, and the run reads as "attention
  // is broken" when the truth is "nobody compiled the kernels".
  {
    void* d = nullptr;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxdnnStatus_t probe =
          grxdnnSoftmaxForward(h, 1, 1, (const float*)d, 1, (float*)d, 1);
      grxFree(d);
      if (probe == GRXDNN_STATUS_NOT_INITIALIZED) {
        std::printf("grxdnn device kernels not found; skipping\n");
        grxdnnDestroy(h);
        return 77;
      }
    }
  }

  section("attention against PyTorch's scaled_dot_product_attention");
  for (const Case& c : cases) run_case(h, c);

  section("what it refuses");
  {
    size_t ws = 0;
    check(grxdnnAttentionWorkspaceSize(1, 1, 4, 4, &ws) == GRXDNN_STATUS_SUCCESS &&
          ws == 4u * 4u * sizeof(float),
          "the workspace is seqLen squared per head");
    check(grxdnnAttentionWorkspaceSize(0, 1, 4, 4, &ws) == GRXDNN_STATUS_INVALID_VALUE,
          "a zero batch is refused");
    // 2^31 squared overflows a 64-bit element count once multiplied by four
    // bytes; the size query must say so rather than return a small number.
    check(grxdnnAttentionWorkspaceSize(1024, 64, 1 << 30, 64, &ws) ==
              GRXDNN_STATUS_INVALID_VALUE,
          "a shape whose workspace overflows is refused, not wrapped");

    void* d = nullptr;
    if (grxMalloc(&d, 1024) == grxSuccess) {
      float* f = (float*)d;
      check(grxdnnAttentionForward(h, 1, 1, 2, 2, f, f, f,
                                   GRXDNN_ATTN_MASK_NONE, d, 4096, f) ==
                GRXDNN_STATUS_INVALID_VALUE,
            "out aliasing an input is refused");
      check(grxdnnAttentionForward(h, 1, 1, 2, 2, f, f, f,
                                   GRXDNN_ATTN_MASK_NONE, d, 4, f + 64) ==
                GRXDNN_STATUS_INVALID_VALUE,
            "a workspace too small is refused");
      check(grxdnnAttentionForward(h, 1, 1, 2, 2, f, f, f,
                                   (grxdnnAttnMask_t)99, d, 4096, f + 64) ==
                GRXDNN_STATUS_INVALID_VALUE,
            "an unknown mask mode is refused");
      grxFree(d);
    }
  }

  grxdnnDestroy(h);
  return grxtest::report();
}
