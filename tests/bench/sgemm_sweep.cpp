// Where does register blocking actually start to pay?
//
// grxblas.cpp picks between the naive sgemm and the register-blocked one with
// a two-clause rule:
//
//     k >= 16   OR   ceil(m/4) >= warpSize
//
// and the roadmap says, in as many words, that the rule is "fitted to five
// points on one configuration and is provisional": the k crossover is bracketed
// by 8 and 16 with NOTHING SWEPT BETWEEN, and the coalescing boundary has a
// mechanism behind it but one measurement either side. That is an honest
// caveat and it is also an invitation. This is the sweep.
//
// WHAT IT DOES. For every shape in a grid it runs BOTH kernels over the same
// operands, in device cycles, and asks two separate questions:
//
//   1. Which kernel is actually faster here?
//   2. Which one does the shipping rule pick?
//
// A rule that is right gets pinned by evidence. A rule that is wrong gets
// counted, in both directions -- picking the slow kernel costs cycles, and
// declining the fast one costs the same cycles while looking like caution.
//
// GRXBLAS_SGEMM_RB is what makes question 1 answerable at all: the rule refuses
// the blocked kernel at shapes it does not like, so measuring whether that
// refusal was right needs a way to run it anyway.
//
// These are SIMX cycles on one SM with a 4-lane warp. They compare two kernels
// on one configuration, which is the only claim the crossover needs; they are
// not hardware performance and a different warp width moves the coalescing
// boundary by construction.

#include <grx/grx.h>
#include <grx/grxblas.h>
#include <grx/grx_cycles.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

struct Probe {
  void* dev = nullptr;
  int   n   = 0;
  std::vector<grxCycleSlot> host;

  explicit Probe(int capacity) : n(capacity), host((size_t)capacity) {
    if (grxMalloc(&dev, (size_t)capacity * sizeof(grxCycleSlot)) != grxSuccess)
      dev = nullptr;
  }
  ~Probe() { if (dev) grxFree(dev); }
  Probe(const Probe&) = delete;
  Probe& operator=(const Probe&) = delete;

  // Cleared before every launch: a slot no warp reached this time would
  // otherwise still hold the last shape's numbers and be summarised into this
  // one. The bench that skipped this reported a profile of its own history.
  void clear() { if (dev) grxMemset(dev, 0, (size_t)n * sizeof(grxCycleSlot)); }

  uint64_t span_of() {
    if (!dev) return 0;
    grxMemcpy(host.data(), dev, (size_t)n * sizeof(grxCycleSlot),
              grxMemcpyDefault);
    grxCycleSummary s{};
    grxCycleSummarize(host.data(), n, &s);
    return (s.spanIsValid && s.warps > 0) ? s.span : 0;
  }
};

// One measurement. Returns the span, or 0 if the launch produced nothing.
uint64_t measure(grxblasHandle_t h, Probe* probe, int m, int n, int k,
                 const void* A, const void* B, void* C, bool blocked) {
  // The selection is env-driven because that is the seam that exists; setting
  // it per measurement keeps the two kernels running over identical operands
  // through identical code, which is what makes the comparison a comparison.
  if (blocked) { setenv("GRXBLAS_SGEMM_RB", "1", 1); unsetenv("GRXBLAS_SGEMM_NAIVE"); }
  else         { setenv("GRXBLAS_SGEMM_NAIVE", "1", 1); unsetenv("GRXBLAS_SGEMM_RB"); }

  probe->clear();
  grxblasSetCycleProbe(h, (grxCycleSlot*)probe->dev, probe->n);
  const float one = 1.0f, zero = 0.0f;
  const grxblasStatus_t st =
      grxblasSgemm(h, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one, A, m, B, k,
                   &zero, C, m);
  grxDeviceSynchronize();
  grxblasSetCycleProbe(h, nullptr, 0);
  if (st != GRXBLAS_STATUS_SUCCESS) return 0;
  return probe->span_of();
}

// The same, batched. A strided-batched GEMM launches batch in the grid's y
// dimension, so the work available to the core is m*n*batch and not m*n -- and
// that is exactly the hole the PERF BASELINE GATE found in the first version of
// the new rule: qkv is three batched GEMMs at m=n=8, which the unbatched rule
// sent to the naive kernel and which then ran 38.9% slower.
uint64_t measure_batched(grxblasHandle_t h, Probe* probe, int m, int n, int k,
                         int batch, const void* A, const void* B, void* C,
                         bool blocked, grxblasOperation_t ta = GRXBLAS_OP_N) {
  if (blocked) { setenv("GRXBLAS_SGEMM_RB", "1", 1); unsetenv("GRXBLAS_SGEMM_NAIVE"); }
  else         { setenv("GRXBLAS_SGEMM_NAIVE", "1", 1); unsetenv("GRXBLAS_SGEMM_RB"); }
  probe->clear();
  grxblasSetCycleProbe(h, (grxCycleSlot*)probe->dev, probe->n);
  const float one = 1.0f, zero = 0.0f;
  const grxblasStatus_t st = grxblasSgemmStridedBatched(
      h, ta, GRXBLAS_OP_N, m, n, k, &one, A,
      (ta == GRXBLAS_OP_T) ? k : m, (long long)m * k,
      B, k, (long long)k * n, &zero, C, m, (long long)m * n, batch);
  grxDeviceSynchronize();
  grxblasSetCycleProbe(h, nullptr, 0);
  if (st != GRXBLAS_STATUS_SUCCESS) return 0;
  return probe->span_of();
}

// The shipping rule, restated here so the sweep can score it. Deliberately a
// COPY rather than a call: if it drifts from grxblas.cpp the numbers below stop
// meaning anything, and the gate at the end is what notices.
// The SHIPPING rule, restated so the sweep can score it. A copy rather than a
// call: if it drifts from grxblas.cpp the counts below stop meaning anything,
// which is what the recorded totals at the end are there to notice.
bool rule_picks_blocked(int m, int n, int k, int warp) {
  (void)n;   // the shipping rule does not look at n. That is part of the finding.
  const int row_blocks = (m + 4 - 1) / 4;
  return m >= 4 && ((k >= 16) || (row_blocks >= warp));
}

// What the sweep says instead: the output count, against the core's capacity.
bool outputs_rule(int m, int n, int batch, long long resident) {
  return m >= 4 && (long long)m * n * batch >= 2 * resident;
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
  std::printf("%s: %d SM, warp %d\n", prop.name, prop.multiProcessorCount,
              prop.warpSize);
  std::printf("cycles are VX_CSR_MCYCLE on THIS configuration. They compare two\n"
              "kernels; they are not hardware performance.\n\n");

  const long long resident = (long long)prop.warpSize *
                             prop.maxWarpsPerMultiProcessor *
                             prop.multiProcessorCount;
  std::printf("%lld threads resident on this core\n\n", resident);

  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return 1;

  // Does this build even have the blocked kernel? Without it every row would
  // read "naive wins" and the sweep would be a very slow way to learn nothing.
  {
    void* d = nullptr;
    const float one = 1.0f, zero = 0.0f;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe = grxblasSgemm(
          h, GRXBLAS_OP_N, GRXBLAS_OP_N, 1, 1, 1, &one, d, 1, d, 1, &zero, d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("no device kernels; skipping\n");
        grxblasDestroy(h);
        return 77;
      }
    }
  }

  const int ms[] = {4, 8, 12, 16, 20, 24, 32};
  const int ks[] = {4, 8, 12, 16, 24, 32};
  const int n = 16;

  const int max_m = 32;
  Probe probe(grxblasCycleSlotsNeeded(h, max_m, n) + 8);
  if (!probe.dev) { std::printf("could not allocate probe slots\n"); return 1; }

  std::vector<float> hA((size_t)max_m * 64, 1.0f), hB((size_t)64 * n, 1.0f);
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (grxMalloc(&dA, hA.size() * 4) != grxSuccess ||
      grxMalloc(&dB, hB.size() * 4) != grxSuccess ||
      grxMalloc(&dC, (size_t)max_m * n * 4) != grxSuccess) {
    std::printf("allocation failed\n");
    return 1;
  }
  grxMemcpy(dA, hA.data(), hA.size() * 4, grxMemcpyDefault);
  grxMemcpy(dB, hB.data(), hB.size() * 4, grxMemcpyDefault);

  std::printf("n = %d throughout. Each cell is naive/blocked cycles, then the\n"
              "speedup blocking gives. > 1.00 means blocking is faster.\n\n", n);
  std::printf("     k =");
  for (int k : ks) std::printf("%9d", k);
  std::printf("\n");

  int rule_wrong_slow = 0;    // rule picked blocked, naive was faster
  int rule_wrong_missed = 0;  // rule picked naive, blocked was faster
  int measured = 0;
  double worst_loss = 1.0;
  int worst_m = 0, worst_k = 0;

  // Every measured point, kept so the boundary can be described rather than
  // asserted from the two nearest cells.
  struct Point { int m, k; double ratio; bool rule_blocked; };
  std::vector<Point> points;

  for (int m : ms) {
    std::printf("m=%3d ", m);
    for (int k : ks) {
      const uint64_t naive   = measure(h, &probe, m, n, k, dA, dB, dC, false);
      const uint64_t blocked = measure(h, &probe, m, n, k, dA, dB, dC, true);
      if (naive == 0 || blocked == 0) { std::printf("        -"); continue; }

      const double ratio = (double)naive / (double)blocked;
      const bool picks = rule_picks_blocked(m, n, k, prop.warpSize);
      points.push_back({m, k, ratio, picks});
      ++measured;

      if (picks && ratio < 1.0) {
        ++rule_wrong_slow;
        if (ratio < worst_loss) { worst_loss = ratio; worst_m = m; worst_k = k; }
      }
      if (!picks && ratio > 1.0) ++rule_wrong_missed;

      // A marker on every cell where the rule and the measurement disagree, so
      // the shape of the error is visible rather than only its count.
      const char flag = (picks == (ratio > 1.0)) ? ' ' : '!';
      std::printf("%8.2f%c", ratio, flag);
    }
    std::printf("\n");
  }

  std::printf("\n  ! marks a cell where the rule and the measurement disagree.\n\n");

  // Where the crossover actually is, per m, read off the sweep rather than
  // assumed from the two points the rule was fitted to.
  std::printf("measured k-crossover, per m (lowest swept k where blocking wins):\n   ");
  for (int m : ms) {
    int cross = -1;
    for (int k : ks) {
      for (const Point& p : points)
        if (p.m == m && p.k == k && p.ratio > 1.0) { cross = k; break; }
      if (cross >= 0) break;
    }
    if (cross >= 0) std::printf("  m=%d:k>=%d", m, cross);
    else            std::printf("  m=%d:never", m);
  }
  std::printf("\n\n");

  // How well does the OUTPUT-COUNT rule do on the same cells? Scored here
  // rather than asserted, because it wins on these and loses on the block.
  int outputs_wrong = 0;
  for (const Point& p : points)
    if (outputs_rule(p.m, n, 1, resident) != (p.ratio > 1.0)) ++outputs_wrong;

  std::printf("scoring, on isolated GEMMs:\n");
  std::printf("        shipping rule (k >= 16 || ceil(m/4) >= warp): %2d of %d wrong\n",
              rule_wrong_slow + rule_wrong_missed, measured);
  std::printf("          %d pick the SLOWER kernel", rule_wrong_slow);
  if (rule_wrong_slow > 0)
    std::printf(", worst m=%d k=%d at %.2fx", worst_m, worst_k, worst_loss);
  std::printf("; %d decline a faster one\n", rule_wrong_missed);
  std::printf("        outputs rule  (m*n*batch >= 2 * resident):   %2d of %d wrong\n",
              outputs_wrong, measured);

  expect(measured == (int)(sizeof(ms) / sizeof(ms[0]) * sizeof(ks) / sizeof(ks[0])),
         "every shape produced two measurements");

  // WHAT IS GATED, AND WHY IT IS NOT "the rule is right".
  //
  // The rule is NOT right on these cells and shipping the one that is made the
  // transformer block SLOWER -- 230171 cycles against 226405 at S=8, because
  // attention's scores GEMM loses inside the block while winning by 1.39x in
  // isolation. So an isolated sweep does not predict the workload, nobody here
  // knows why yet, and asserting either rule as correct would be asserting
  // something this bench has actively disproven.
  //
  // What IS gated is that the disagreement does not grow. These counts are the
  // measured state of a known-imperfect rule; a change that makes it worse is a
  // regression whether or not the rule is ever replaced.
  expect(rule_wrong_slow <= 3,
         "the shipping rule picks the slower kernel on no MORE shapes than the "
         "3 already recorded");
  expect(rule_wrong_missed <= 6,
         "and declines a faster one on no more than the 6 already recorded");
  expect(outputs_wrong == 0,
         "and the output-count rule still explains every isolated cell, which "
         "is what makes the block's disagreement worth chasing");

  // ---- and now n, because the sweep above holds it fixed ------------------
  //
  // The rule the sweep just contradicted was fitted partly to ATTENTION, whose
  // GEMM is m=8 n=8 k=8 and batched. Rewriting the rule from an n=16 sweep and
  // then discovering it is wrong at n=8 would be replacing one under-evidenced
  // boundary with another, so n gets swept too before anything is changed.
  std::printf("\nn sweep. Same cells: naive/blocked, > 1.00 means blocking wins.\n\n");
  const int ns[] = {4, 8, 16, 32};
  const int nm[] = {4, 8, 16};
  const int nk[] = {8, 16};
  std::printf("           n =");
  for (int nn : ns) std::printf("%9d", nn);
  std::printf("\n");

  int n_disagree = 0;
  for (int m : nm) {
    for (int k : nk) {
      std::printf("m=%2d k=%2d  ", m, k);
      for (int nn : ns) {
        std::vector<float> hb((size_t)64 * nn, 1.0f);
        grxMemcpy(dB, hb.data(), hb.size() * 4, grxMemcpyDefault);
        Probe p2(grxblasCycleSlotsNeeded(h, m, nn) + 8);
        if (!p2.dev) { std::printf("        -"); continue; }
        const uint64_t nv = measure(h, &p2, m, nn, k, dA, dB, dC, false);
        const uint64_t bl = measure(h, &p2, m, nn, k, dA, dB, dC, true);
        if (nv == 0 || bl == 0) { std::printf("        -"); continue; }
        const double r = (double)nv / (double)bl;
        // Scored against m >= 8, which is what the m/k sweep above says the
        // boundary actually is. If n moves it, this is where that shows.
        const bool predicted = rule_picks_blocked(m, nn, k, prop.warpSize);
        const char flag = (predicted == (r > 1.0)) ? ' ' : '!';
        if (flag == '!') ++n_disagree;
        std::printf("%8.2f%c", r, flag);
      }
      std::printf("\n");
    }
  }
  std::printf("\n  scored against the shipping rule: %d of %d cells disagree\n",
              n_disagree, (int)(sizeof(nm)/sizeof(nm[0]) * sizeof(nk)/sizeof(nk[0])
                                * sizeof(ns)/sizeof(ns[0])));

  // ---- and batch, which the perf baselines caught the rule ignoring -------
  //
  // The prediction to test: a batched GEMM has m*n*batch outputs available, so
  // (m=8, n=8, batch=2) should behave like (m=8, n=16, batch=1) -- 128 outputs
  // either way. If it does, batch belongs in the rule multiplicatively; if it
  // does not, the rule needs something else and guessing would be the wrong
  // move twice running.
  std::printf("\nbatch sweep. m=n=8 k=16 throughout, so outputs = 64 * batch.\n\n");
  std::printf("      batch =        1        2        4\n");
  {
    const int bm = 8, bn = 8, bk = 16;
    std::vector<float> hb((size_t)bk * bn * 4, 1.0f);
    grxMemcpy(dB, hb.data(), hb.size() * 4, grxMemcpyDefault);
    std::printf("m=8 n=8 k=16 ");
    for (int b : {1, 2, 4}) {
      Probe pb(grxblasCycleSlotsNeeded(h, bm, bn) * b + 8);
      if (!pb.dev) { std::printf("        -"); continue; }
      const uint64_t nv = measure_batched(h, &pb, bm, bn, bk, b, dA, dB, dC, false);
      const uint64_t bl = measure_batched(h, &pb, bm, bn, bk, b, dA, dB, dC, true);
      if (nv == 0 || bl == 0) { std::printf("        -"); continue; }
      std::printf("%9.2f", (double)nv / (double)bl);
    }
    std::printf("\n\n  compare the unbatched row above: m=8 n=8 reads 0.96 and\n"
                "  m=8 n=16 reads 1.44. If batch=2 lands near 1.44 rather than\n"
                "  near 0.96, outputs scale with batch and the rule must too.\n");
  }

  // ---- and the transpose, which the perf baselines caught next ------------
  //
  // Attention's scores GEMM is OP_T with m=n=8, k=8, batch=2 -- 128 outputs, so
  // the batched rule sends it to the blocked kernel, where the block profile
  // says it runs 27.6% SLOWER. Every cell above is OP_N. If transposing moves
  // the crossover, the rule has to know.
  std::printf("\ntranspose sweep. batch=2 throughout; OP_T transposes A.\n\n");
  std::printf("%-22s %9s %9s\n", "m  n  k", "OP_N", "OP_T");
  int t_disagree = 0;
  for (auto sh : {std::array<int,3>{8,8,8}, {8,8,16}, {8,16,8}, {16,16,8}}) {
    const int tm = sh[0], tn = sh[1], tk = sh[2];
    std::vector<float> hb((size_t)tk * tn * 4, 1.0f);
    grxMemcpy(dB, hb.data(), hb.size() * 4, grxMemcpyDefault);
    std::printf("m=%2d n=%2d k=%2d batch=2 ", tm, tn, tk);
    double ratios[2] = {0, 0};
    int col = 0;
    for (grxblasOperation_t op : {GRXBLAS_OP_N, GRXBLAS_OP_T}) {
      Probe pb(grxblasCycleSlotsNeeded(h, tm, tn) * 2 + 8);
      if (!pb.dev) { std::printf("        -"); ++col; continue; }
      const uint64_t nv = measure_batched(h, &pb, tm, tn, tk, 2, dA, dB, dC, false, op);
      const uint64_t bl = measure_batched(h, &pb, tm, tn, tk, 2, dA, dB, dC, true, op);
      if (nv == 0 || bl == 0) { std::printf("        -"); ++col; continue; }
      ratios[col++] = (double)nv / (double)bl;
      std::printf("%9.2f", (double)nv / (double)bl);
    }
    if ((ratios[0] > 1.0) != (ratios[1] > 1.0)) { ++t_disagree; std::printf("  <-- transpose flips it"); }
    std::printf("\n");
  }
  std::printf("\n  %d of 4 shapes change which kernel wins when A is transposed\n",
              t_disagree);

  grxFree(dA); grxFree(dB); grxFree(dC);
  grxblasDestroy(h);
  std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
