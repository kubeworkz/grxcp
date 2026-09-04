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
// ONE SPAN PER LAUNCH, AND THIS FILE GOT IT WRONG. MCYCLE restarts at zero at
// every launch: SimX's ProcessorImpl::run() opens with reset(), which assigns a
// fresh PerfStats, and MCYCLE reads PerfStats::cycles. So a span taken across
// two launches is a maximum over two clocks that both started at zero, and it
// looks exactly like a duration. Attention (four launches) and the output
// projection (H launches) were both read that way for three commits. The
// symptom was there to be seen the whole time and nothing was looking at it:
// summarising attention's buffer reported SIXTY-FOUR warps live at once on a
// device that holds sixteen.
//
// What it cost: grxBLAS's sgemm kernel-selection rule was reverted because
// forcing the blocked kernel in attention showed a 27.6% regression. That
// number came from this defect. Every multi-launch stage below now measures
// each launch against its own clock and ADDS the spans, and grxCycleSummary
// carries `maxLive` so the impossible reading is refused instead of printed.
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
#include <cstdlib>
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
  // The most warps live at once, over every slot counted into this stage. It is
  // reported because it is the only thing that can prove the span came from
  // more than one launch -- see grx_cycles.h and `occupancy` below.
  int         maxLive = 0;
  int         overOccupancy = 0;   // maxLive exceeded what the device holds
  // Cross-core evidence, carried so a refusal can say WHY rather than just
  // that. A stage that spans cores is not thereby invalid -- see grx_cycles.h.
  int         cores = 0;
  uint64_t    coreSkew = 0;
  uint64_t    skewSpan = 0;        // the span the skew was judged against
  // Summed launch preamble: reset to first warp, once per launch. EVERY span
  // below excludes it, so a total built from spans alone is not what the block
  // costs -- see the accounting under the table.
  uint64_t    preamble = 0;
  int         launches = 0;
};

// Why a stage has no number. THREE reasons now, and the first version of this
// file printed the same one for the first two -- which is how a probe that never
// fired came to look like a device that had scattered the warps.
const char* why_no_span(const StageCost& c) {
  if (c.warps == 0)
    return "no warp wrote a slot -- is the kernel's probe reaching finish()?";
  if (c.overOccupancy)
    return "more warps live at once than the device holds: these slots come "
           "from more than one launch, and a span across launches is not a "
           "duration on either backend";
  return "no span: the slots did not form one";
}

// DOES MCYCLE RESTART AT EVERY LAUNCH? On one of our two backends it does not,
// and this file used to assert that it did -- in three comments, as a fact.
//
// simx resets the whole device at the top of `ProcessorImpl::run()`, so MCYCLE
// starts from zero at every launch. rtlsim does not: the RTL counter in
// `VX_scheduler.sv` is zeroed only under `reset`, which rtlsim issues once in
// its constructor, and it then free-runs per core, gated on that core's `busy`.
// On rtlsim a launch's firstStart therefore carries every earlier launch that
// ran on that core, and a preamble total summed from it is the cost of the
// whole run rather than of one launch.
//
// SPANS ARE SAFE ON BOTH. A span is a subtraction inside one launch and the
// offset cancels, which is exactly why this hid for as long as it did. It is
// only the ABSOLUTE readings -- the preamble total and its share of the block --
// that it decides, so it decides whether they are printed at all.
//
// Silicon free-runs mcycle, so rtlsim is the faithful one here and simx's
// per-launch reset is the convenience. That makes this a property of the
// backend to be measured, not a bug to be fixed.
//
//   -1 not established   0 cumulative   1 restarts per launch
int  g_mcycle_resets = -1;
#define GRX_CALIB_N 4
uint64_t g_calib[GRX_CALIB_N] = {0, 0, 0, 0};  // the readings the verdict used
// --calibrate-only: establish the counter's behaviour and stop. The full block
// takes hours on rtlsim, which is exactly the backend whose answer differs, so
// the check has to be reachable without paying for the block.
bool g_calibrate_only = false;

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
  // How many warps this device can hold at once. A summary reporting more live
  // than this did not come from one launch, and a span over more than one
  // launch is not a duration -- the launches are separated by a preamble that
  // is not part of either of them, whatever the counter does across the
  // boundary. This is the whole reason the number is here.
  int   occupancy = 0;
  std::vector<grxCycleSlot> host;

  Probe(int capacity, int occ)
      : n(capacity), occupancy(occ), host((size_t)capacity) {
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

  bool fetch() {
    if (!dev) return false;
    return grxMemcpy(host.data(), dev, (size_t)n * sizeof(grxCycleSlot),
                     grxMemcpyDefault) == grxSuccess;
  }

  // One launch's worth of slots, summarised and checked against occupancy.
  void fold(const grxCycleSlot* slots, int count, StageCost* c) const {
    grxCycleSummary s{};
    grxCycleSummarize(slots, count, &s);
    if (s.warps == 0) return;
    c->warps += s.warps;
    if (s.maxLive > c->maxLive) c->maxLive = s.maxLive;
    if (occupancy > 0 && s.maxLive > occupancy) {
      c->overOccupancy = 1;
      c->valid = false;
      return;
    }
    c->preamble += s.firstStart;
    c->launches += 1;
    if (s.cores > c->cores) c->cores = s.cores;
    if (s.coreSkew > c->coreSkew) c->coreSkew = s.coreSkew;
    // On a backend whose per-core counters share an origin, crossCoreSpan is
    // the stage's wall clock and the stagger between cores is part of what the
    // stage costs. Alignment is NOT re-derived here from the skew -- skew
    // cannot see it (grx_cycles.h). It is established once per backend by
    // tests/repro/cross_core_clock/align_probe, which measured 158 cycles of
    // spread against a 37731-cycle span on simx at 4 SMs. Run it before
    // trusting these numbers on a backend it has not been run on.
    if (s.spanCrossesCores) { c->span += s.crossCoreSpan; return; }
    if (s.spanIsValid == 0) { c->valid = false; return; }
    c->span += s.span;
  }

  // A stage that is ONE launch. Summarising the whole buffer is right here and
  // only here.
  StageCost take(const char* name) {
    StageCost c;
    c.name = name;
    if (!fetch()) return c;
    c.valid = true;
    fold(host.data(), n, &c);
    if (c.warps == 0) c.valid = false;
    clear();
    return c;
  }

  // A stage that is SEVERAL launches, each in its own region of the buffer.
  //
  // The spans are ADDED, not spanned: the launches are ordered on one stream,
  // so the stage costs their sum, and each is measured against its own clock.
  // Taking a span across the regions is exactly the mistake this function
  // exists to stop -- it read a maximum over four unrelated counters and called
  // it attention's cost, and a kernel-selection rule was reverted on it.
  StageCost take_regions(const char* name, const grxdnnCycleRegion_t* regions,
                         int count) {
    StageCost c;
    c.name = name;
    if (!fetch()) return c;
    if (count <= 0) { clear(); return c; }   // no regions: nothing to trust
    c.valid = true;
    for (int i = 0; i < count; ++i) {
      const int off = regions[i].offset, len = regions[i].slots;
      if (off < 0 || len <= 0 || off + len > n) { c.valid = false; break; }
      fold(host.data() + off, len, &c);
    }
    if (c.warps == 0) c.valid = false;
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
  // Attention needs room for FOUR launches at once, each in its own region, so
  // that summarising the buffer gives the span of all four rather than of
  // whichever wrote last. It refuses outright if the probe is too small, which
  // is why this is asked rather than guessed.
  const int attn_cap = grxdnnAttentionCycleSlotsNeeded(dh, 1, H, S, Dh);
  if (attn_cap > cap) cap = attn_cap;
  if (cap <= 0) return false;
  grxDeviceProp_t dprop{};
  grxGetDeviceProperties(&dprop, 0);
  const int occupancy =
      dprop.maxWarpsPerMultiProcessor * dprop.multiProcessorCount;
  Probe probe(cap + 8, occupancy);
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

  // Calibrate the counter before trusting an absolute reading from it.
  //
  // FOUR identical launches. Nothing varies between them, so a counter that
  // restarts per launch reports the same firstStart four times; one that does
  // not adds a frame's busy cycles at every step, and the readings climb by a
  // near-constant increment.
  //
  // The test is the SHAPE of the series, not the size of one gap, and the first
  // version of this check got that wrong: it compared two launches and asked
  // whether the second was at least half again the first. On a four-block
  // kernel that is a 124% step and it fires; on this layernorm the frame is
  // only ~2268 cycles against a ~4900 preamble, a 46% step, and the guard
  // reported "restarts per launch" on the backend that does not. It was written
  // against the one measurement that happened to be in front of it.
  //
  // Monotonicity does not care about that ratio. A flat counter wobbles both
  // ways -- simx reads 4727 then 4741 -- so three consecutive strict increases
  // plus growth above a tenth separates them with room to spare either way.
  //
  // This is the check that was missing entirely. The grid sweep in
  // tests/repro/launch_preamble/ ran without it and reported a per-CTA dispatch
  // cost that was really its own position in the sweep.
  if (g_mcycle_resets == -1) {
    bool got_all = true;
    for (int i = 0; i < GRX_CALIB_N && got_all; ++i) {
      probe.clear();
      grxdnnLayerNormForward(dh, S, D, x->f(), D, g1->f(), b1->f(), eps,
                             h1.f(), D);
      grxCycleSummary s{};
      if (!probe.fetch()) { got_all = false; break; }
      grxCycleSummarize(probe.host.data(), probe.n, &s);
      if (s.warps == 0) { got_all = false; break; }
      g_calib[i] = s.firstStart;
    }
    probe.clear();
    if (got_all) {
      bool climbing = true;
      for (int i = 1; i < GRX_CALIB_N; ++i)
        if (g_calib[i] <= g_calib[i - 1]) climbing = false;
      const bool grew =
          g_calib[GRX_CALIB_N - 1] > g_calib[0] + g_calib[0] / 10;
      g_mcycle_resets = (climbing && grew) ? 0 : 1;
    }
    // Printed here, not at the end, so it is watchable on a backend where the
    // rest of this run takes hours (--calibrate-only stops after it).
    std::printf("MCYCLE restarts per launch: %s\n  %d identical launches read",
                g_mcycle_resets == 1 ? "yes" :
                g_mcycle_resets == 0 ? "NO -- absolute readings will report -1"
                                     : "could not establish",
                GRX_CALIB_N);
    for (int i = 0; i < GRX_CALIB_N; ++i)
      std::printf(" %llu", (unsigned long long)g_calib[i]);
    std::printf("\n");
    if (g_calibrate_only) return false;
  }

  grxdnnLayerNormForward(dh, S, D, x->f(), D, g1->f(), b1->f(), eps, h1.f(), D);
  out->push_back(probe.take("layernorm 1"));

  // The three projections, measured together: they are the same shape and the
  // same kernel, and splitting them would report one number three times.
  {
    struct P { Buf* w; Buf* b; Buf* o; };
    const P ps[3] = {{Wq, bq, &q}, {Wk, bk, &k}, {Wv, bv, &v}};
    uint64_t total = 0, bias_total = 0;
    uint64_t pre = 0, bias_pre = 0;
    int launches = 0, bias_launches = 0;
    bool valid = true, bias_valid = true;
    int warps = 0, qkv_live = 0, bias_warps = 0, bias_live = 0;
    for (const P& pr : ps) {
      grxblasSgemmStridedBatched(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, Dh, S, D, &one,
                                 pr.w->f(), D, (long long)Dh, h1.f(), D, 0,
                                 &zero, pr.o->f(), Dh, (long long)S * Dh, H);
      const StageCost c = probe.take("qkv");
      total += c.span; valid = valid && c.valid; warps = c.warps;
      pre += c.preamble; launches += c.launches;
      if (c.maxLive > qkv_live) qkv_live = c.maxLive;

      // THESE SIX LAUNCHES WERE THROWN AWAY, and the comment that threw them
      // away said they were "counted with the other biases". They were not.
      // probe.clear() discarded the slots and no stage absorbed them, so the
      // block total omitted 3 x H real launches -- an estimated 4.2% at S=16
      // and 5.5% at S=8. Every SHARE reported by this bench was therefore a
      // fraction of a denominator that was too small, including the attention
      // share that has been published.
      //
      // The blocking RATIO was unharmed: it is naive over blocked and both
      // totals omitted the same six launches. A ratio survives a wrong
      // denominator when both sides carry it. A share does not.
      //
      // Read after each head and SUMMED, for the reason the out projection is:
      // MCYCLE is not comparable across launches, so one read after the loop
      // would be a maximum over H unrelated clocks with the smaller head
      // vanishing into the larger.
      for (int hh = 0; hh < H; ++hh) {
        grxdnnAddBiasForward(dh, S, Dh, pr.o->f() + (size_t)hh * S * Dh, Dh,
                             pr.b->f() + (size_t)hh * Dh,
                             pr.o->f() + (size_t)hh * S * Dh, Dh);
        const StageCost bc = probe.take("qkv bias");
        bias_total += bc.span;
        bias_pre += bc.preamble; bias_launches += bc.launches;
        bias_valid = bias_valid && bc.valid;
        bias_warps = bc.warps;
        if (bc.maxLive > bias_live) bias_live = bc.maxLive;
      }
    }
    StageCost c; c.name = "qkv proj (3 GEMMs)"; c.span = total;
    c.valid = valid; c.warps = warps; c.maxLive = qkv_live;
    c.preamble = pre; c.launches = launches;
    out->push_back(c);

    // Its own stage rather than folded into the GEMM it follows: it is a
    // different kernel doing a different amount of work, and it is the
    // bias-into-GEMM-epilogue fusion candidate. A fusion cannot be priced
    // against a cost nobody records.
    StageCost b; b.name = "qkv bias (3 x H)"; b.span = bias_total;
    b.preamble = bias_pre; b.launches = bias_launches;
    b.valid = bias_valid; b.warps = bias_warps; b.maxLive = bias_live;
    out->push_back(b);
  }

  // ATTENTION, MEASURED ONE LAUNCH AT A TIME.
  //
  // It is four launches -- the scores GEMM, the mask, the softmax, the output
  // GEMM -- each writing its own region of one probe buffer. This used to
  // summarise the whole buffer and call the result attention's cost. It was
  // not: MCYCLE restarts at zero at EVERY launch (grx_cycles.h), so those four
  // regions carry four unrelated clocks and a span across them is a maximum
  // over strangers. The reading was not merely imprecise -- it reported 64
  // warps live at once on a device that holds 16, and grxBLAS's kernel-
  // selection rule was reverted on a 27.6% "regression" measured with it.
  //
  // grxdnnGetCycleRegions says where each launch wrote. Each is summarised
  // against its own clock and the spans are ADDED, because the launches are
  // ordered on one stream and the stage costs their sum.
  {
    const grxdnnStatus_t st =
        grxdnnAttentionForward(dh, 1, H, S, Dh, q.f(), k.f(), v.f(),
                               GRXDNN_ATTN_MASK_CAUSAL, ws.p, ws_bytes, a.f());
    grxdnnCycleRegion_t regions[8];
    int nregions = 0;
    grxdnnGetCycleRegions(dh, regions, 8, &nregions);
    if (nregions > 8) nregions = 8;
    StageCost c = probe.take_regions("attention (2 GEMMs+mask+softmax)",
                                     regions, nregions);
    if (st != GRXDNN_STATUS_SUCCESS) c.valid = false;
    out->push_back(c);
  }

  // The output projection is H SEPARATE LAUNCHES, and for the same reason it is
  // read after each one and the spans added. Read once at the end it spanned
  // two launches, which on this device meant a maximum over two clocks that
  // both started at zero -- the smaller head simply vanished into the larger.
  {
    StageCost c; c.name = "out proj (H GEMMs)"; c.valid = true;
    for (int hh = 0; hh < H; ++hh) {
      const float* beta = (hh == 0) ? &zero : &one;
      grxblasSgemm(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, D, S, Dh, &one,
                   Wo->f() + (size_t)hh * Dh * D, D,
                   a.f() + (size_t)hh * S * Dh, Dh, beta, p.f(), D);
      const StageCost one_head = probe.take("out proj");
      c.span += one_head.span;
      c.warps = one_head.warps;
      if (one_head.maxLive > c.maxLive) c.maxLive = one_head.maxLive;
      c.overOccupancy |= one_head.overOccupancy;
      c.preamble += one_head.preamble; c.launches += one_head.launches;
      if (!one_head.valid) c.valid = false;
    }
    out->push_back(c);
  }

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
  // When EVERY stage is invalid this used to print "(no cycles recorded)" and
  // return, which skipped the per-stage reasons below -- so the one diagnostic
  // written for this exact case was unreachable in it. Measured at 4 SMs on
  // simx: twelve stages, all invalid, and the output said nothing about why.
  if (total == 0) {
    std::printf("  (no cycles recorded)\n");
    for (const StageCost& c : stages)
      std::printf("  %-26s   -- %s\n", c.name.c_str(), why_no_span(c));
    return 0.0;
  }

  double attn = 0.0;
  for (const StageCost& c : stages) {
    if (!c.valid) {
      std::printf("  %-26s   -- %s\n", c.name.c_str(), why_no_span(c));
      continue;
    }
    const double share = 100.0 * (double)c.span / (double)total;
    if (c.cores > 1)
      std::printf("  %-26s %10llu cycles  %5.1f%%  (%d warps, %d live, "
                  "%d cores, skew %llu)\n",
                  c.name.c_str(), (unsigned long long)c.span, share, c.warps,
                  c.maxLive, c.cores, (unsigned long long)c.coreSkew);
    else
      std::printf("  %-26s %10llu cycles  %5.1f%%  (%d warps, %d live, "
                  "%d launch%s, preamble %llu)\n",
                  c.name.c_str(), (unsigned long long)c.span, share, c.warps,
                  c.maxLive, c.launches, c.launches == 1 ? "" : "es",
                  (unsigned long long)c.preamble);
    if (c.name.rfind("attention", 0) == 0) attn += share;
  }
  std::printf("  %-26s %10llu cycles\n", "TOTAL (measured stages)",
              (unsigned long long)total);
  if (any_invalid)
    std::printf("  note: some stages produced no valid span and are excluded "
                "from the total.\n");

  // WHAT THE TOTAL ABOVE LEAVES OUT, and it is not small.
  //
  // A span runs from the FIRST WARP STARTING to the last warp finishing. The
  // cycles between the launch and that first warp are outside every span in the
  // table -- and on a backend that zeroes MCYCLE at the launch, the first
  // warp's own reading IS that interval, measured on the device.
  //
  // That conditional is the whole point. It is true on simx and false on
  // rtlsim, and this block printed the number unconditionally until a run of
  // six identical launches showed the reading climbing by a constant 8917 each
  // time. Where the counter does not restart, there is no absolute preamble to
  // report from this data and the honest output is -1.
  //
  // Summing spans and calling the result "the block" drops one preamble per
  // launch. This prints it rather than leaving the reader to assume the table
  // is the whole story.
  uint64_t preamble = 0;
  int launches = 0;
  for (const StageCost& c : stages) {
    if (!c.valid) continue;
    preamble += c.preamble;
    launches += c.launches;
  }
  if (launches > 0 && g_mcycle_resets == 1) {
    const uint64_t wall = total + preamble;
    std::printf("  %-26s %10llu cycles  over %d launches (%llu each)\n",
                "launch preamble", (unsigned long long)preamble, launches,
                (unsigned long long)(preamble / (uint64_t)launches));
    std::printf("  %-26s %10llu cycles  -- preamble is %.1f%% of it\n",
                "BLOCK, preamble included", (unsigned long long)wall,
                100.0 * (double)preamble / (double)wall);
  } else if (launches > 0) {
    std::printf("  %-26s %10d          -- %s\n", "launch preamble", -1,
                g_mcycle_resets == 0
                    ? "MCYCLE does not restart per launch on this backend"
                    : "could not establish whether MCYCLE restarts per launch");
    if (g_mcycle_resets == 0)
      std::printf("  %-26s                     %d identical launches read "
                  "%llu .. %llu\n", "", GRX_CALIB_N,
                  (unsigned long long)g_calib[0],
                  (unsigned long long)g_calib[GRX_CALIB_N - 1]);
    std::printf("  %-26s %10d          -- spans above are unaffected: a span "
                "is a subtraction\n", "BLOCK, preamble included", -1);
  }
  *attention_share = attn;
  return (double)total;
}

// The machine-readable half, for ci/perf/baselines/. Written from the SAME
// StageCost records the prose report prints, so the two cannot disagree.
//
// Only raw integers go in: spans, warp counts, and the greatest number of warps
// live at once. Shares, totals-per-element and speedups are derived by
// ci/check_perf.py from these. A baseline holding "4.6%" would drift against
// its own rounding and would have to carry a tolerance to survive it; a
// baseline holding 13834 does not.
//
// maxLive is in the baseline because it is what proves a span came from ONE
// launch. A stage that quietly starts spanning two would move its cycles and
// its maxLive together, and pinning only the cycles is how the last one went
// unnoticed for three commits.
void write_json(const char* path, const grxDeviceProp_t& prop,
                const char* sgemm_config, const Shape* shapes,
                const std::vector<StageCost>* stages, int nshapes) {
  std::FILE* f = std::fopen(path, "w");
  if (!f) {
    std::printf("  could not write %s\n", path);
    ++failures;
    return;
  }
  std::fprintf(f, "{\n  \"bench\": \"block_cycles\",\n");
  std::fprintf(f, "  \"config\": {\"sgemm\": \"%s\"},\n", sgemm_config);
  std::fprintf(f,
               "  \"device\": {\"name\": \"%s\", \"sms\": %d, \"warp\": %d, "
               "\"mhz\": %d},\n",
               prop.name, prop.multiProcessorCount, prop.warpSize,
               prop.clockRateMHz);
  std::fprintf(f, "  \"shapes\": [\n");
  for (int i = 0; i < nshapes; ++i) {
    std::fprintf(f,
                 "    {\"seq\": %d, \"dim\": %d, \"heads\": %d, \"ff\": %d,\n"
                 "     \"stages\": [\n",
                 shapes[i].seq, shapes[i].dim, shapes[i].heads, shapes[i].ff);
    for (size_t j = 0; j < stages[i].size(); ++j) {
      const StageCost& c = stages[i][j];
      std::fprintf(f,
                   "       {\"name\": \"%s\", \"span\": %llu, \"warps\": %d, "
                   "\"maxLive\": %d, \"valid\": %s}%s\n",
                   c.name.c_str(), (unsigned long long)c.span, c.warps,
                   c.maxLive, c.valid ? "true" : "false",
                   j + 1 == stages[i].size() ? "" : ",");
    }
    std::fprintf(f, "     ]}%s\n", i + 1 == nshapes ? "" : ",");
  }
  std::fprintf(f, "  ]\n}\n");
  std::fclose(f);
  std::printf("  wrote %s\n", path);
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = nullptr;
  // --sweep: report the differential control instead of gating it.
  //
  // The control below asks that attention's SHARE of the block grow when the
  // sequence doubles, because its scores matrix is seqLen squared and every
  // other stage is linear. That holds for the block as it SHIPS. It does not
  // have to hold for a block one of whose GEMMs has been forced onto a slower
  // kernel: ci/sweep_block_sgemm.py does exactly that, and forcing the S=16 mlp
  // projection to the reference inflates that stage until attention's share
  // falls below its S=8 value. The bench then reported FAILED for a
  // configuration nobody ships, and the sweep read that as its own flip having
  // failed to run.
  //
  // The flag says which question is being asked. It suppresses nothing else:
  // every stage is still measured, every span still refused if it spans
  // launches, and the JSON is identical. ci/run_real.sh does NOT pass it.
  bool sweep_mode = false;
  // --only-shape <i>: run one of the two sequence lengths instead of both.
  //
  // For ci/sweep_block_sgemm.py, which flips ONE sgemm call and re-runs the
  // block. A call belongs to exactly one shape, so running the other one is
  // work that cannot change the answer -- 73 runs of both shapes took 405
  // seconds where 38 runs of one take about 95. It is also better isolation:
  // each flip is now priced against a baseline of its own configuration, with
  // no chance of the untouched shape contributing anything.
  //
  // Implies --sweep, because a single shape cannot support the differential
  // control (which compares two of them).
  int only_shape = -1;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else if (std::strcmp(argv[i], "--calibrate-only") == 0) {
      g_calibrate_only = true;
    } else if (std::strcmp(argv[i], "--sweep") == 0) {
      sweep_mode = true;
    } else if (std::strcmp(argv[i], "--only-shape") == 0 && i + 1 < argc) {
      only_shape = std::atoi(argv[++i]);
      sweep_mode = true;
    } else {
      std::printf("usage: block_cycles [--out <results.json>] [--sweep] "
                  "[--only-shape <0|1>]\n");
      return 2;
    }
  }
  if (only_shape > 1) {
    std::printf("--only-shape takes 0 or 1; this bench measures two shapes\n");
    return 2;
  }

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
  std::vector<StageCost> stages[2];

  for (int i = 0; i < 2; ++i) {
    if (only_shape >= 0 && i != only_shape) continue;
    if (!profile(bh, dh, shapes[i], &stages[i])) {
      // profile() also returns false on purpose under --calibrate-only, and
      // saying "could not size the probe" there would be a lie about a run
      // that did exactly what was asked.
      std::printf(g_calibrate_only ? "calibration only; stopping here\n"
                                   : "could not size the probe; skipping\n");
      grxdnnDestroy(dh); grxblasDestroy(bh);
      return g_calibrate_only ? 0 : 77;
    }
    totals[i] = report(shapes[i], stages[i], &attn_share[i]);
  }

  std::printf("\ndoes the measurement respond to its input?\n");
  if (only_shape >= 0) {
    std::printf("        --only-shape %d: one sequence length was run, so the "
                "differential\n        control below has nothing to compare "
                "against and is not evaluated.\n", only_shape);
    expect(totals[only_shape] > 0, "the requested shape recorded cycles");
  } else {
    expect(totals[0] > 0 && totals[1] > 0, "both shapes recorded cycles");
    expect(totals[1] > totals[0],
           "doubling the sequence costs more cycles");
  }
  // The differential claim. Attention's scores matrix is seqLen SQUARED and
  // every other stage is linear in seqLen, so attention has to take a LARGER
  // share of a longer block -- not merely more cycles, which a counter that was
  // really counting launches would also show.
  if (only_shape < 0)
    std::printf("        attention's share: %.1f%% at S=%d, %.1f%% at S=%d\n",
                attn_share[0], shapes[0].seq, attn_share[1], shapes[1].seq);
  if (only_shape >= 0) {
    // nothing to gate: one shape
  } else if (sweep_mode)
    std::printf("        --sweep: the share is reported, not gated. One of this "
                "block's GEMMs is on a\n        kernel it does not ship with, so "
                "the stage mix is not the one this control is about.\n");
  else
    expect(attn_share[1] > attn_share[0],
           "and attention's SHARE grows, as a seqLen-squared stage must");

  if (out_path) {
    // The label records what was ASKED FOR, not what grxBLAS did -- the
    // register-blocked kernel is an optional symbol lookup and can fall back
    // silently. Nothing here can tell the difference, so nothing here claims
    // to: ci/check_perf.py runs both configurations and requires their numbers
    // to DIFFER, which is what actually proves the selection took effect.
    const char* naive = std::getenv("GRXBLAS_SGEMM_NAIVE");
    const char* label = (naive && naive[0] == '1') ? "naive" : "register-blocked";
    write_json(out_path, prop, label, shapes, stages, 2);
  }

  grxdnnDestroy(dh);
  grxblasDestroy(bh);
  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED",
              failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
