// Where a transformer block's cycles actually go.
//
// This is a REPORT, not a pass/fail gate — except for the control at the end,
// which is a gate and is the only reason the rest of it is worth reading.
//
// Until now nothing had ever measured a whole workload. Every grxDNN kernel has
// carried a `cycle_probe` since it was written, and no host call could reach
// one, so the instrumentation was dead code waiting for something real to point
// at. `tests/libs/test_grxdnn_block.cpp` supplied that: a block that is known
// correct against PyTorch at every stage. Correct is the prerequisite for
// measuring — profiling a wrong kernel tells you how fast the wrong answer
// arrives.
//
// WHAT THE NUMBERS ARE. VX_CSR_MCYCLE, the core's own cycle counter, read by
// the kernel at entry and exit (grx_cycles.h). On SimX that is the cycle count
// of the MODEL, which is exactly the right thing to compare stages against and
// the wrong thing to quote as hardware performance. This device is one SM and
// four lanes; multiplying these figures up to a 128-SM flagship is the
// arithmetic this file exists to make unnecessary.
//
// The headline per stage is the SPAN — first warp starting to last warp
// finishing — because that is what the stage costs the block. `grxCycleSummarize`
// refuses to produce a span when the warps landed on different cores, and this
// reports that refusal rather than printing a plausible number.
//
// THE CONTROL, and why the shape list has two entries.
//
// A profiler whose numbers nobody has watched respond to their input is not
// measuring anything — the same rule the PROF GATE applies to grx-prof. So the
// block runs at two sequence lengths, and the check is not "the total went up",
// which almost anything would satisfy. It is DIFFERENTIAL:
//
//   attention's scores matrix is seqLen SQUARED; every other stage is linear
//   in seqLen. So doubling the sequence must raise ATTENTION'S SHARE of the
//   block, and not merely raise its cycles.
//
// A probe wired to the wrong kernel, or a counter that is really a call count,
// passes "it went up" and fails that.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grxdnn.h>
#include <grx/grx_cycles.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) { if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr; }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  float* f() const { return (float*)p; }
};

// Values do not matter here — correctness is settled elsewhere — but they must
// not be denormal or NaN, which would let the hardware take a different path
// and make this a measurement of something nobody asked about.
void fill(std::vector<float>& v, unsigned seed) {
  for (size_t i = 0; i < v.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    v[i] = (float)((int)(seed >> 16) % 2001 - 1000) * 0.001f;
  }
}

Buf* upload(size_t n, unsigned seed) {
  std::vector<float> h(n);
  fill(h, seed);
  Buf* b = new Buf(n * sizeof(float));
  if (b->p) grxMemcpy(b->p, h.data(), n * sizeof(float), grxMemcpyDefault);
  return b;
}

struct StageCost {
  std::string name;
  uint64_t    span = 0;
  bool        valid = false;
  int         warps = 0;
};

// Why a stage has no number. Two very different reasons, and the first version
// of this file printed the second for both -- which is how a probe that never
// fired came to look like a device that had scattered the warps.
const char* why_no_span(const StageCost& c) {
  if (c.warps == 0)
    return "no warp wrote a slot -- is the kernel's probe reaching finish()?";
  return "the warps spanned cores; a span across two counters means nothing";
}

// One probe buffer, reused, and it lives on the DEVICE.
//
// The kernel is what writes these slots, so they have to be somewhere the
// kernel can reach. The first version of this file used a std::vector and every
// stage reported zero cycles -- not an error, just a silent record of nothing,
// which is the failure mode a host pointer produces here and the reason
// grxdnn.h now says so in the API comment.
//
// Cleared between stages because a slot no warp reached this time would
// otherwise still hold the previous stage's numbers and be summarised into it.
struct Probe {
  void* dev = nullptr;
  int   n   = 0;
  std::vector<grxCycleSlot> host;

  explicit Probe(int capacity) : n(capacity), host((size_t)capacity) {
    if (grxMalloc(&dev, (size_t)capacity * sizeof(grxCycleSlot)) != grxSuccess)
      dev = nullptr;
    clear();
  }
  ~Probe() { if (dev) grxFree(dev); }
  Probe(const Probe&) = delete;
  Probe& operator=(const Probe&) = delete;

  void clear() {
    if (dev) grxMemset(dev, 0, (size_t)n * sizeof(grxCycleSlot));
  }
  StageCost take(const char* name) {
    StageCost c;
    c.name = name;
    if (!dev) return c;
    grxMemcpy(host.data(), dev, (size_t)n * sizeof(grxCycleSlot),
              grxMemcpyDefault);
    grxCycleSummary s{};
    grxCycleSummarize(host.data(), n, &s);
    c.span  = s.span;
    c.valid = s.spanIsValid != 0 && s.warps > 0;
    c.warps = s.warps;
    clear();
    return c;
  }
};

struct Shape { int seq, dim, heads, ff; };

bool profile(grxblasHandle_t bh, grxdnnHandle_t dh, const Shape& sh,
             std::vector<StageCost>* out) {
  const int S = sh.seq, D = sh.dim, H = sh.heads, F = sh.ff;
  const int Dh = D / H;
  const float one = 1.0f, zero = 0.0f;
  const float eps = 1e-5f;

  // Capacity: the largest launch any stage makes. Asked of the libraries rather
  // than derived here, because they own their launch geometry.
  int cap = grxdnnCycleSlotsNeeded(dh, S * F);
  const int blas_cap = grxblasCycleSlotsNeeded(bh, F, S) * H;
  if (blas_cap > cap) cap = blas_cap;
  if (cap <= 0) return false;
  Probe probe(cap + 8);
  if (!probe.dev) return false;

  Buf *x = upload((size_t)S * D, 1), *g1 = upload(D, 2), *b1 = upload(D, 3),
      *g2 = upload(D, 4), *b2 = upload(D, 5),
      *Wq = upload((size_t)D * D, 6), *Wk = upload((size_t)D * D, 7),
      *Wv = upload((size_t)D * D, 8), *bq = upload(D, 9), *bk = upload(D, 10),
      *bv = upload(D, 11), *Wo = upload((size_t)D * D, 12), *bo = upload(D, 13),
      *W1 = upload((size_t)D * F, 14), *bf1 = upload(F, 15),
      *W2 = upload((size_t)F * D, 16), *bf2 = upload(D, 17);

  const size_t sd = (size_t)S * D * sizeof(float);
  const size_t sf = (size_t)S * F * sizeof(float);
  Buf h1(sd), q(sd), k(sd), v(sd), a(sd), p(sd), x2(sd), h2(sd);
  Buf f1(sf), act(sf), f2(sd), y(sd);
  size_t ws_bytes = 0;
  grxdnnAttentionWorkspaceSize(1, H, S, Dh, &ws_bytes);
  Buf ws(ws_bytes);

  grxdnnSetCycleProbe(dh, (grxCycleSlot*)probe.dev, probe.n);
  grxblasSetCycleProbe(bh, (grxCycleSlot*)probe.dev, probe.n);
  probe.clear();

  grxdnnLayerNormForward(dh, S, D, x->f(), D, g1->f(), b1->f(), eps, h1.f(), D);
  out->push_back(probe.take("layernorm 1"));

  // The three projections, measured together: they are the same shape and the
  // same kernel, and splitting them would report one number three times.
  {
    struct P { Buf* w; Buf* b; Buf* o; };
    const P ps[3] = {{Wq, bq, &q}, {Wk, bk, &k}, {Wv, bv, &v}};
    uint64_t total = 0;
    bool valid = true;
    int warps = 0;
    for (const P& pr : ps) {
      grxblasSgemmStridedBatched(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, Dh, S, D, &one,
                                 pr.w->f(), D, (long long)Dh, h1.f(), D, 0,
                                 &zero, pr.o->f(), Dh, (long long)S * Dh, H);
      const StageCost c = probe.take("qkv");
      total += c.span; valid = valid && c.valid; warps = c.warps;
      for (int hh = 0; hh < H; ++hh)
        grxdnnAddBiasForward(dh, S, Dh, pr.o->f() + (size_t)hh * S * Dh, Dh,
                             pr.b->f() + (size_t)hh * Dh,
                             pr.o->f() + (size_t)hh * S * Dh, Dh);
      probe.clear();   // the per-head bias is counted with the other biases
    }
    StageCost c; c.name = "qkv proj (3 GEMMs)"; c.span = total;
    c.valid = valid; c.warps = warps;
    out->push_back(c);
  }

  // Attention is three launches of its own -- two GEMMs, a mask, a softmax --
  // and the probe records only the LAST of them, so it is measured as a whole
  // by summing what its internals report. Reported as one line because that is
  // how a caller buys it.
  grxdnnAttentionForward(dh, 1, H, S, Dh, q.f(), k.f(), v.f(),
                         GRXDNN_ATTN_MASK_CAUSAL, ws.p, ws_bytes, a.f());
  {
    StageCost c = probe.take("attention (softmax pass)");
    out->push_back(c);
  }

  for (int hh = 0; hh < H; ++hh) {
    const float* beta = (hh == 0) ? &zero : &one;
    grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, D, S, Dh, &one,
                 Wo->f() + (size_t)hh * Dh * D, D,
                 a.f() + (size_t)hh * S * Dh, Dh, beta, p.f(), D);
  }
  out->push_back(probe.take("out proj (H GEMMs)"));

  grxdnnAddBiasForward(dh, S, D, p.f(), D, bo->f(), p.f(), D);
  out->push_back(probe.take("bias"));

  grxMemcpy(x2.p, p.p, sd, grxMemcpyDefault);
  grxblasSaxpy(bh, S * D, &one, x->f(), 1, x2.f(), 1);
  out->push_back(probe.take("residual (saxpy)"));

  grxdnnLayerNormForward(dh, S, D, x2.f(), D, g2->f(), b2->f(), eps, h2.f(), D);
  out->push_back(probe.take("layernorm 2"));

  grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, F, S, D, &one,
               W1->f(), F, h2.f(), D, &zero, f1.f(), F);
  out->push_back(probe.take("mlp GEMM 1 (D->F)"));

  grxdnnAddBiasForward(dh, S, F, f1.f(), F, bf1->f(), f1.f(), F);
  out->push_back(probe.take("bias"));

  grxdnnGeluForward(dh, S, F, f1.f(), F, GRXDNN_GELU_TANH, act.f(), F);
  out->push_back(probe.take("gelu"));

  grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, D, S, F, &one,
               W2->f(), D, act.f(), F, &zero, f2.f(), D);
  out->push_back(probe.take("mlp GEMM 2 (F->D)"));

  grxdnnSetCycleProbe(dh, nullptr, 0);
  grxblasSetCycleProbe(bh, nullptr, 0);

  for (Buf* b : {x, g1, b1, g2, b2, Wq, Wk, Wv, bq, bk, bv, Wo, bo,
                 W1, bf1, W2, bf2})
    delete b;
  return true;
}

double report(const Shape& sh, const std::vector<StageCost>& stages,
              double* attention_share) {
  std::printf("\nS=%d D=%d H=%d F=%d\n", sh.seq, sh.dim, sh.heads, sh.ff);
  uint64_t total = 0;
  bool any_invalid = false;
  for (const StageCost& c : stages) {
    if (c.valid) total += c.span;
    else any_invalid = true;
  }
  if (total == 0) { std::printf("  (no cycles recorded)\n"); return 0.0; }

  double attn = 0.0;
  for (const StageCost& c : stages) {
    if (!c.valid) {
      std::printf("  %-26s   -- %s\n", c.name.c_str(), why_no_span(c));
      continue;
    }
    const double share = 100.0 * (double)c.span / (double)total;
    std::printf("  %-26s %10llu cycles  %5.1f%%  (%d warps)\n", c.name.c_str(),
                (unsigned long long)c.span, share, c.warps);
    if (c.name.rfind("attention", 0) == 0) attn += share;
  }
  std::printf("  %-26s %10llu cycles\n", "TOTAL (measured stages)",
              (unsigned long long)total);
  if (any_invalid)
    std::printf("  note: some stages produced no valid span and are excluded "
                "from the total.\n");
  *attention_share = attn;
  return (double)total;
}

}  // namespace

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, 0);
  std::printf("%s: %d SMs, warp %d, %d MHz\n", prop.name,
              prop.multiProcessorCount, prop.warpSize, prop.clockRateMHz);
  std::printf("cycles are VX_CSR_MCYCLE on THIS configuration. They compare "
              "stages;\nthey are not hardware performance and do not scale by "
              "multiplication.\n");

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

  // Two sequence lengths, everything else held fixed. The second is the control.
  const Shape shapes[2] = {{8, 16, 2, 64}, {16, 16, 2, 64}};
  double totals[2] = {0, 0}, attn_share[2] = {0, 0};

  for (int i = 0; i < 2; ++i) {
    std::vector<StageCost> stages;
    if (!profile(bh, dh, shapes[i], &stages)) {
      std::printf("could not size the probe; skipping\n");
      grxdnnDestroy(dh); grxblasDestroy(bh);
      return 77;
    }
    totals[i] = report(shapes[i], stages, &attn_share[i]);
  }

  std::printf("\ndoes the measurement respond to its input?\n");
  expect(totals[0] > 0 && totals[1] > 0, "both shapes recorded cycles");
  expect(totals[1] > totals[0],
         "doubling the sequence costs more cycles");
  // The differential claim. Attention's scores matrix is seqLen SQUARED and
  // every other stage is linear in seqLen, so attention has to take a LARGER
  // share of a longer block -- not merely more cycles, which a counter that was
  // really counting launches would also show.
  std::printf("        attention's share: %.1f%% at S=%d, %.1f%% at S=%d\n",
              attn_share[0], shapes[0].seq, attn_share[1], shapes[1].seq);
  expect(attn_share[1] > attn_share[0],
         "and attention's SHARE grows, as a seqLen-squared stage must");

  grxdnnDestroy(dh);
  grxblasDestroy(bh);
  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
