// THE PHASE 6 EXIT GATE: a transformer block, end to end, against PyTorch.
//
// Every op this uses is already gated on its own. What no single-op gate can
// see is whether four correct ops composed are still correct — each one fixes
// its own layout convention and checks it in isolation, and composing them
// means one op's output becomes another's input. A transposed or mis-strided
// hand-off between two correct kernels produces plausible numbers that nothing
// upstream would catch.
//
// STAGE BY STAGE, STOPPING AT THE FIRST DISAGREEMENT. tests/libs/block_ref.py
// dumps every intermediate, not just the block's output, and this compares them
// in order. A block that only checked `y` would report "wrong" over ten stages
// and leave the bisection to a person; this names the op.
//
// NO PERMUTE ANYWHERE, WHICH IS THE POINT.
//
// The reference does what torch does: project to [S, D], reshape to
// [S, H, Dh], permute to [H, S, Dh] for attention, and permute back afterwards.
// GRXCP has no permute op and does not need one. The projection is a
// strided-batched GEMM over the weight matrix's COLUMN BLOCKS -- head h's
// weights are a [D, Dh] slice at offset h*Dh with the same leading dimension,
// so the batch stride is Dh -- and it lands directly in [H][S][Dh], which is
// the layout attention wants. Coming back out, the output projection is one
// accumulating GEMM per head into the same result, because head h's slice of
// Wo is a row block. Two permutes that a naive port would pay for do not exist
// here, and that is a consequence of the layout algebra rather than an
// optimisation.
//
// WATCHED FAILING, and the failures are the whole argument for this file:
//
//   output projection overwrites per head    the H=1 case passes ENTIRELY --
//   instead of accumulating (beta always 0)  correctly, there is nothing to
//                                            accumulate with one head -- and
//                                            both H=2 cases fail at exactly
//                                            `p`, by 0.283 and 0.495, with
//                                            every earlier stage green
//
//   exact GELU where the weights expect      all three cases fail at exactly
//   the tanh form                            `act`, by 2.33e-04
//
// The first is a hand-off bug: every op involved is correct and individually
// gated, and the composition is still wrong. That is what this gate is for, and
// the H=1/H=2 split shows it is measuring the per-head plumbing rather than
// arithmetic in general.
//
// The second one found a defect in THIS FILE. The original tolerance was
// 4e-6*(S+D+Dh) + 2e-6*F -- between 1.5e-04 and 2.6e-04, three orders of
// magnitude above anything observed -- and the wrong activation slipped under
// it on two of the three cases. A bound loose enough to pass a wrong GELU is
// not a bound. It is now set from the measurement; see the note where it is
// computed.
//
// The GEMM arguments below are all `N, N` with the operands SWAPPED, which is
// the row-major-through-a-column-major-library identity applied to each in
// turn: a row-major (r, c) matrix with leading dimension ld is the column-major
// (c, r) matrix over the same bytes, so C = A @ B row-major is Cᵀ = Bᵀ Aᵀ
// column-major, and the transposes vanish into the argument order. Attention is
// the one op that still needs an explicit transpose flag, and it owns it.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grxdnn.h>

#include "../unit/grx_test.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

constexpr uint32_t kMagic = 0x4B4C4247u;   // 'GBLK'
constexpr uint32_t kVersion = 1u;
constexpr float kEps = 1e-5f;

struct Case {
  uint32_t seq = 0, dim = 0, heads = 0, ff = 0, causal = 0;
  std::vector<float> x;
  // Weights, in the order block_ref.py writes them.
  std::vector<float> g1, b1, g2, b2;
  std::vector<float> Wq, Wk, Wv, bq, bk, bv, Wo, bo, W1, bf1, W2, bf2;
  // Every intermediate, so a failure names a stage.
  std::vector<float> h1, q, k, v, a, p, x2, h2, f1, act, f2, y;
};

bool rd(std::FILE* f, std::vector<float>* v, size_t n) {
  v->resize(n);
  return std::fread(v->data(), sizeof(float), n, f) == n;
}

bool load(const char* path, std::vector<Case>* out) {
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  auto u32 = [&](uint32_t* x) { return std::fread(x, sizeof(*x), 1, f) == 1; };
  uint32_t magic = 0, version = 0, count = 0;
  if (!u32(&magic) || !u32(&version) || !u32(&count) || magic != kMagic ||
      version != kVersion || count == 0) {
    std::fclose(f);
    return false;
  }
  bool ok = true;
  for (uint32_t i = 0; i < count && ok; ++i) {
    Case c{};
    ok = u32(&c.seq) && u32(&c.dim) && u32(&c.heads) && u32(&c.ff) &&
         u32(&c.causal);
    if (!ok) break;
    const size_t S = c.seq, D = c.dim, F = c.ff;
    ok = ok && rd(f, &c.x, S * D) &&
         rd(f, &c.g1, D) && rd(f, &c.b1, D) && rd(f, &c.g2, D) &&
         rd(f, &c.b2, D) &&
         rd(f, &c.Wq, D * D) && rd(f, &c.Wk, D * D) && rd(f, &c.Wv, D * D) &&
         rd(f, &c.bq, D) && rd(f, &c.bk, D) && rd(f, &c.bv, D) &&
         rd(f, &c.Wo, D * D) && rd(f, &c.bo, D) &&
         rd(f, &c.W1, D * F) && rd(f, &c.bf1, F) &&
         rd(f, &c.W2, F * D) && rd(f, &c.bf2, D) &&
         rd(f, &c.h1, S * D) &&
         rd(f, &c.q, S * D) && rd(f, &c.k, S * D) && rd(f, &c.v, S * D) &&
         rd(f, &c.a, S * D) &&
         rd(f, &c.p, S * D) && rd(f, &c.x2, S * D) && rd(f, &c.h2, S * D) &&
         rd(f, &c.f1, S * F) && rd(f, &c.act, S * F) && rd(f, &c.f2, S * D) &&
         rd(f, &c.y, S * D);
    if (ok) out->push_back(std::move(c));
  }
  std::fclose(f);
  return ok && !out->empty();
}

// A device buffer that frees itself.
struct Buf {
  void* p = nullptr;
  Buf() = default;
  explicit Buf(size_t bytes) { if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr; }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  float* f() const { return (float*)p; }
};

Buf* upload(const std::vector<float>& v) {
  Buf* b = new Buf(v.size() * sizeof(float));
  if (b->p) grxMemcpy(b->p, v.data(), v.size() * sizeof(float), grxMemcpyDefault);
  return b;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : std::getenv("GRXDNN_BLOCK_REF");
  if (!path) path = "block_ref.bin";

  std::vector<Case> cases;
  if (!load(path, &cases)) {
    std::printf("cannot read reference from %s\n"
                "  regenerate with: python3 tests/libs/block_ref.py "
                "--write tests/libs/block_ref.bin\n", path);
    return 1;
  }

  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxblasHandle_t bh = nullptr;
  grxdnnHandle_t  dh = nullptr;
  if (grxblasCreate(&bh) != GRXBLAS_STATUS_SUCCESS) return 1;
  if (grxdnnCreate(&dh) != GRXDNN_STATUS_SUCCESS) { grxblasDestroy(bh); return 1; }

  {
    void* d = nullptr;
    const float one = 1.0f, zero = 0.0f;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe = grxblasSgemm(
          bh, GRXBLAS_OP_N, GRXBLAS_OP_N, 1, 1, 1, &one, d, 1, d, 1, &zero, d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("no device kernels; skipping\n");
        grxdnnDestroy(dh); grxblasDestroy(bh);
        return 77;
      }
    }
  }

  const float one = 1.0f, zero = 0.0f;

  for (const Case& c : cases) {
    const int S = (int)c.seq, D = (int)c.dim, H = (int)c.heads, F = (int)c.ff;
    const int Dh = D / H;
    char title[128];
    std::snprintf(title, sizeof(title), "S%d D%d H%d F%d%s",
                  S, D, H, F, c.causal ? " causal" : "");
    section(title);

    // SET FROM THE MEASUREMENT, and tightened once already.
    //
    // The first version of this was 4e-6*(S+D+Dh) + 2e-6*F, which came out
    // between 1.5e-04 and 2.6e-04 -- three orders of magnitude above anything
    // this gate actually observes. An ablation found the cost: swapping the
    // exact GELU in where the weights expect the tanh form perturbs `act` by
    // 2.3e-04, and that slipped under the bound on two of the three cases. A
    // tolerance loose enough to pass a wrong activation is not a tolerance.
    //
    // Observed worst across every stage and case is 8.05e-07. This bound scales
    // mildly with the total reduction depth and lands between 1.1e-05 and
    // 1.9e-05 -- roughly 20x headroom over the measurement, and an order of
    // magnitude below the smallest error any ablation produces.
    const double tol = 2e-7 * (double)(S + D + Dh + F);

    bool alive = true;
    auto stage = [&](const char* name, const Buf& buf,
                     const std::vector<float>& want) {
      if (!alive) return;
      std::vector<float> got(want.size());
      grxMemcpy(got.data(), buf.p, want.size() * sizeof(float), grxMemcpyDefault);
      double worst = 0.0;
      size_t at = 0;
      for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs((double)got[i] - (double)want[i]);
        if (d > worst) { worst = d; at = i; }
      }
      if (worst <= tol) {
        std::printf("  ok    %-4s worst |diff| %.3g\n", name, worst);
        return;
      }
      std::printf("  FAIL  %-4s worst |diff| %.3g at [%zu] "
                  "(got %.9g, want %.9g)\n",
                  name, worst, at, (double)got[at], (double)want[at]);
      ++grxtest::failures();
      // Everything after this consumes this stage's output, so continuing would
      // report nine more failures that all say the same thing.
      std::printf("        (stopping this case: later stages read this one)\n");
      alive = false;
    };

    Buf *dx = upload(c.x), *dg1 = upload(c.g1), *db1 = upload(c.b1),
        *dg2 = upload(c.g2), *db2 = upload(c.b2),
        *dWq = upload(c.Wq), *dWk = upload(c.Wk), *dWv = upload(c.Wv),
        *dbq = upload(c.bq), *dbk = upload(c.bk), *dbv = upload(c.bv),
        *dWo = upload(c.Wo), *dbo = upload(c.bo),
        *dW1 = upload(c.W1), *dbf1 = upload(c.bf1),
        *dW2 = upload(c.W2), *dbf2 = upload(c.bf2);

    const size_t sd = (size_t)S * D * sizeof(float);
    const size_t sf = (size_t)S * F * sizeof(float);
    Buf h1(sd), q(sd), k(sd), v(sd), a(sd), p(sd), x2(sd), h2(sd);
    Buf f1(sf), act(sf), f2(sd), y(sd);

    size_t ws_bytes = 0;
    grxdnnAttentionWorkspaceSize(1, H, S, Dh, &ws_bytes);
    Buf ws(ws_bytes);

    // --- 1. h1 = LayerNorm(x) ------------------------------------------------
    grxdnnLayerNormForward(dh, S, D, dx->f(), D, dg1->f(), db1->f(), kEps,
                           h1.f(), D);
    stage("h1", h1, c.h1);

    // --- 2. Q, K, V ----------------------------------------------------------
    //
    // One batched GEMM per projection, batched over HEADS. Head h's weights are
    // the column block starting at h*Dh, so strideA is Dh; h1 is shared by all
    // heads, so strideB is 0 -- which grxblasSgemmStridedBatched documents as
    // broadcasting one operand across the batch. The result lands as
    // [H][S][Dh], which is exactly the layout attention wants.
    struct Proj { Buf* w; Buf* b; Buf* out; const std::vector<float>* want;
                  const char* name; };
    const Proj projs[3] = {{dWq, dbq, &q, &c.q, "q"},
                           {dWk, dbk, &k, &c.k, "k"},
                           {dWv, dbv, &v, &c.v, "v"}};
    for (const Proj& pr : projs) {
      if (!alive) break;
      grxblasSgemmStridedBatched(bh, GRXBLAS_OP_N, GRXBLAS_OP_N,
                                 Dh, S, D, &one,
                                 pr.w->f(), D, (long long)Dh,
                                 h1.f(), D, 0,
                                 &zero, pr.out->f(), Dh, (long long)S * Dh, H);
      // The bias is per head: head h wants bq[h*Dh .. (h+1)*Dh). One call per
      // head rather than one fused epilogue -- a real implementation folds this
      // into the GEMM, and that is a later increment with its own gate.
      for (int hh = 0; hh < H; ++hh)
        grxdnnAddBiasForward(dh, S, Dh, pr.out->f() + (size_t)hh * S * Dh, Dh,
                             pr.b->f() + (size_t)hh * Dh,
                             pr.out->f() + (size_t)hh * S * Dh, Dh);
      stage(pr.name, *pr.out, *pr.want);
    }

    // --- 3. attention --------------------------------------------------------
    if (alive) {
      grxdnnAttentionForward(dh, 1, H, S, Dh, q.f(), k.f(), v.f(),
                             c.causal ? GRXDNN_ATTN_MASK_CAUSAL
                                      : GRXDNN_ATTN_MASK_NONE,
                             ws.p, ws_bytes, a.f());
      stage("a", a, c.a);
    }

    // --- 4. output projection ------------------------------------------------
    //
    // p = sum over heads of a_h @ Wo_h. Head h's slice of Wo is a ROW block, so
    // it is at offset h*Dh*D with the same leading dimension, and the heads
    // accumulate into one result with beta = 1 after the first. That is what
    // replaces the permute back to [S, D].
    if (alive) {
      for (int hh = 0; hh < H; ++hh) {
        const float* beta = (hh == 0) ? &zero : &one;
        grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, D, S, Dh, &one,
                     dWo->f() + (size_t)hh * Dh * D, D,
                     a.f() + (size_t)hh * S * Dh, Dh,
                     beta, p.f(), D);
      }
      grxdnnAddBiasForward(dh, S, D, p.f(), D, dbo->f(), p.f(), D);
      stage("p", p, c.p);
    }

    // --- 5. residual ---------------------------------------------------------
    //
    // x2 = x + p, which is saxpy with alpha = 1. grxDNN needs no elementwise
    // add of its own: grxBLAS already has one and it is already gated.
    if (alive) {
      grxMemcpy(x2.p, p.p, sd, grxMemcpyDefault);
      grxblasSaxpy(bh, S * D, &one, dx->f(), 1, x2.f(), 1);
      stage("x2", x2, c.x2);
    }

    // --- 6..10. the MLP half -------------------------------------------------
    if (alive) {
      grxdnnLayerNormForward(dh, S, D, x2.f(), D, dg2->f(), db2->f(), kEps,
                             h2.f(), D);
      stage("h2", h2, c.h2);
    }
    if (alive) {
      grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, F, S, D, &one,
                   dW1->f(), F, h2.f(), D, &zero, f1.f(), F);
      grxdnnAddBiasForward(dh, S, F, f1.f(), F, dbf1->f(), f1.f(), F);
      stage("f1", f1, c.f1);
    }
    if (alive) {
      // TANH, not exact: block_ref.py uses approximate="tanh", which is what
      // GPT-2's published weights expect. The two forms differ by ~5e-4, well
      // above this gate's tolerance, so picking the wrong one fails here.
      grxdnnGeluForward(dh, S, F, f1.f(), F, GRXDNN_GELU_TANH, act.f(), F);
      stage("act", act, c.act);
    }
    if (alive) {
      grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, D, S, F, &one,
                   dW2->f(), D, act.f(), F, &zero, f2.f(), D);
      grxdnnAddBiasForward(dh, S, D, f2.f(), D, dbf2->f(), f2.f(), D);
      stage("f2", f2, c.f2);
    }
    if (alive) {
      grxMemcpy(y.p, f2.p, sd, grxMemcpyDefault);
      grxblasSaxpy(bh, S * D, &one, x2.f(), 1, y.f(), 1);
      stage("y", y, c.y);
    }

    for (Buf* b : {dx, dg1, db1, dg2, db2, dWq, dWk, dWv, dbq, dbk, dbv,
                   dWo, dbo, dW1, dbf1, dW2, dbf2})
      delete b;
  }

  grxdnnDestroy(dh);
  grxblasDestroy(bh);
  return grxtest::report();
}
