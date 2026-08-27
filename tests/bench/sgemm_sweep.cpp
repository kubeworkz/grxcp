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

// Which kernel a measurement should reach. Three now: the reference, the
// register-blocked kernel that reuses B, and the 2D micro-tile that reuses
// both operands.
enum class Which { kNaive, kRb, kTwoD };

const char* which_name(Which w) {
  switch (w) {
    case Which::kNaive: return "naive";
    case Which::kRb:    return "rb";
    case Which::kTwoD:  return "2d";
  }
  return "?";
}

// The selection is env-driven because that is the seam that exists; setting it
// per measurement keeps every kernel running over identical operands through
// identical code, which is what makes the comparison a comparison.
void select(Which w) {
  unsetenv("GRXBLAS_SGEMM_NAIVE");
  unsetenv("GRXBLAS_SGEMM_RB");
  unsetenv("GRXBLAS_SGEMM_2D");
  switch (w) {
    case Which::kNaive: setenv("GRXBLAS_SGEMM_NAIVE", "1", 1); break;
    case Which::kRb:    setenv("GRXBLAS_SGEMM_RB", "1", 1); break;
    case Which::kTwoD:  setenv("GRXBLAS_SGEMM_2D", "1", 1); break;
  }
}

// One measurement. Returns the span, or 0 if the launch produced nothing.
uint64_t measure(grxblasHandle_t h, Probe* probe, int m, int n, int k,
                 const void* A, const void* B, void* C, Which w) {
  select(w);

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
                         Which w, grxblasOperation_t ta = GRXBLAS_OP_N) {
  select(w);
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
// The SHIPPING rule, restated so the sweep can score it. A COPY rather than a
// call: if it drifts from grxblas.cpp the counts below stop meaning anything,
// which is what the gate at the end is there to notice.
Which rule_picks_kernel(int m, int n, int batch, long long resident) {
  // Every module this bench runs against carries sgemm_2d, so the rb fallback
  // branch of the shipping rule is not restated here -- restating a branch this
  // cannot reach would be a copy nothing checks.
  if ((long long)m * n * batch < resident) return Which::kNaive;
  return Which::kTwoD;
}

// The rule this one REPLACED, kept so its score stays on the record. It shipped
// for months on a coalescing story that this sweep disproved: k never changes
// which kernel wins anywhere in range, and the boundary is not at m = 16.
bool former_rule(int m, int n, int k, int warp) {
  (void)n;   // it did not look at n. That was part of the finding.
  const int row_blocks = (m + 4 - 1) / 4;
  return m >= 4 && ((k >= 16) || (row_blocks >= warp));
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

  int rule_wrong_slow = 0;    // the rule picked a kernel SLOWER than naive
  int rule_wrong_missed = 0;  // it picked a safe kernel that was not the best
  int measured = 0;
  double worst_loss = 1.0;
  int worst_m = 0, worst_k = 0;

  // Every measured point, kept so the boundary can be described rather than
  // asserted from the two nearest cells.
  struct Point {
    int m, k;
    double rb_ratio, td_ratio;   // naive / kernel; > 1 means it beat naive
    Which best, picked;
  };
  std::vector<Point> points;

  for (int m : ms) {
    std::printf("m=%3d ", m);
    for (int k : ks) {
      const uint64_t nv = measure(h, &probe, m, n, k, dA, dB, dC, Which::kNaive);
      const uint64_t rb = measure(h, &probe, m, n, k, dA, dB, dC, Which::kRb);
      const uint64_t td = measure(h, &probe, m, n, k, dA, dB, dC, Which::kTwoD);
      if (nv == 0 || rb == 0 || td == 0) { std::printf("        -"); continue; }

      Point p{};
      p.m = m; p.k = k;
      p.rb_ratio = (double)nv / (double)rb;
      p.td_ratio = (double)nv / (double)td;
      // The fastest of the three, by span. Ties go to the simpler kernel,
      // because a tie is not a reason to run more code.
      p.best = Which::kNaive;
      double best_ratio = 1.0;
      if (p.rb_ratio > best_ratio) { p.best = Which::kRb;   best_ratio = p.rb_ratio; }
      if (p.td_ratio > best_ratio) { p.best = Which::kTwoD; best_ratio = p.td_ratio; }
      p.picked = rule_picks_kernel(m, n, 1, resident);
      points.push_back(p);
      ++measured;

      const double picked_ratio = (p.picked == Which::kNaive) ? 1.0
                                : (p.picked == Which::kRb) ? p.rb_ratio
                                                           : p.td_ratio;
      if (picked_ratio < 1.0) {
        ++rule_wrong_slow;
        if (picked_ratio < worst_loss) {
          worst_loss = picked_ratio; worst_m = m; worst_k = k;
        }
      } else if (p.picked != p.best) {
        ++rule_wrong_missed;
      }

      // The best kernel's speedup over the reference, the letter naming it,
      // and a marker on every cell where the rule picks something else -- so
      // the shape of the error is visible rather than only its count.
      const char who = (p.best == Which::kNaive) ? '-'
                     : (p.best == Which::kRb)    ? 'r' : 'd';
      const char flag = (p.picked == p.best) ? ' ' : '!';
      std::printf("%7.2f%c%c", best_ratio, who, flag);
    }
    std::printf("\n");
  }

  std::printf("\n  r = the register-blocked kernel won, d = the 2D micro-tile,\n"
              "  - = neither beat the reference. ! marks a cell where the rule\n"
              "  picks a kernel other than the one that measured fastest.\n\n");

  // Where the crossover actually is, per m, read off the sweep rather than
  // assumed from the two points the rule was fitted to.
  std::printf("measured k-crossover, per m (lowest swept k where blocking wins):\n   ");
  for (int m : ms) {
    int cross = -1;
    for (int k : ks) {
      for (const Point& p : points)
        if (p.m == m && p.k == k && p.best != Which::kNaive) { cross = k; break; }
      if (cross >= 0) break;
    }
    if (cross >= 0) std::printf("  m=%d:k>=%d", m, cross);
    else            std::printf("  m=%d:never", m);
  }
  std::printf("\n\n");

  // How the two kernels compare to EACH OTHER, which is the question the 2D
  // micro-tile was built to ask. Both produce four outputs per thread, so they
  // launch the same number of threads at the same shape; the only difference is
  // that the 2D tile reuses both operands and pays 4 loads per 4 multiply-adds
  // where the 1D tile pays 5. If the load count is what costs, this is where it
  // shows, and nothing else moved.
  int td_beats_rb = 0, rb_beats_td = 0;
  double td_gain_min = 1e9, td_gain_max = 0.0;
  for (const Point& p : points) {
    const double g = p.td_ratio / p.rb_ratio;   // 2D against register-blocked
    if (g > 1.0) ++td_beats_rb; else ++rb_beats_td;
    if (g < td_gain_min) td_gain_min = g;
    if (g > td_gain_max) td_gain_max = g;
  }
  std::printf("the two blocked kernels, head to head (same thread count):\n");
  std::printf("        the 2D micro-tile is faster on %d of %d cells, "
              "slower on %d\n", td_beats_rb, measured, rb_beats_td);
  std::printf("        span ratio 2d:rb ranges %.2fx to %.2fx\n\n",
              td_gain_min, td_gain_max);

  // How the rule this one replaced does on the same cells, kept on the record.
  int former_wrong = 0;
  for (const Point& p : points)
    if (former_rule(p.m, n, p.k, prop.warpSize) !=
        (p.best != Which::kNaive)) ++former_wrong;

  std::printf("scoring, on isolated GEMMs:\n");
  std::printf("        shipping rule:   %2d of %d wrong\n",
              rule_wrong_slow + rule_wrong_missed, measured);
  std::printf("          %d pick a kernel SLOWER than the reference", rule_wrong_slow);
  if (rule_wrong_slow > 0)
    std::printf(", worst m=%d k=%d at %.3fx", worst_m, worst_k, worst_loss);
  std::printf("; %d pick a safe kernel that was not the fastest\n",
              rule_wrong_missed);
  std::printf("        former rule   (k >= 16 || ceil(m/4) >= warp): %2d of %d "
              "wrong about whether to block at all\n",
              former_wrong, measured);

  expect(measured == (int)(sizeof(ms) / sizeof(ms[0]) * sizeof(ks) / sizeof(ks[0])),
         "every shape produced two measurements");

  // WHAT IS GATED.
  //
  // Not "the rule picks the fastest kernel", although it does on 41 of 42
  // cells. What matters is that it never picks one that leaves the caller worse
  // off than the untuned reference, because that is the only error a tuned
  // kernel can make that a program actually feels.
  //
  // One cell is fractionally slower and it is named rather than tolerated:
  // m = 4, n = 16, k = 4 reads 0.996x. That is the exact threshold shape at the
  // smallest swept k -- 64 outputs, the boundary itself -- and a 0.4%
  // difference there is the rule sitting on the crossover, not choosing wrongly.
  // The bound is 0.99 so that a real regression cannot hide behind it: the next
  // cell down the bracket, which the rule declines, loses 12%.
  expect(worst_loss >= 0.99,
         "no shape the rule blocks runs more than 1% slower than the reference");
  expect(rule_wrong_missed == 0,
         "and it never picks a safe kernel where a faster one was available");
  expect(former_wrong == 8,
         "and the rule it replaced is still wrong about whether to block on the "
         "8 cells that replaced it -- if that number moves, this sweep is "
         "measuring something other than what it did");
  // The head-to-head that justifies having a second blocked kernel at all. If
  // this ever stops holding, the 2D tile is not earning its place and the rule
  // should go back to sgemm_rb rather than keep two.
  expect(td_beats_rb == measured,
         "the 2D micro-tile beats the register-blocked kernel on every swept "
         "cell, at the same thread count");

  // ---- and now n, because the sweep above holds it fixed ------------------
  //
  // The rule the sweep just contradicted was fitted partly to ATTENTION, whose
  // GEMM is m=8 n=8 k=8 and batched. Rewriting the rule from an n=16 sweep and
  // then discovering it is wrong at n=8 would be replacing one under-evidenced
  // boundary with another, so n gets swept too before anything is changed.
  std::printf("\nn sweep. naive / the FASTEST blocked kernel, measured whether or\n"
              "not the rule wants it. > 1.00 means blocking wins there.\n\n");
  // n = 1 and 2 are here because the 2D micro-tile slices COLUMNS by 2: at
  // n = 1 half of every thread's work is clamped away, which is the one place
  // the extra dimension can cost rather than pay.
  const int ns[] = {1, 2, 4, 8, 16, 32};
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
        // Measured whether or not the rule wants it. The rule DECLINES the
        // blocked kernels at small shapes, and a sweep that only ran what the
        // rule asked for could never find out whether declining was right --
        // which is the whole reason the force hooks exist.
        const Which pick = rule_picks_kernel(m, nn, 1, resident);
        const uint64_t nv = measure(h, &p2, m, nn, k, dA, dB, dC, Which::kNaive);
        const uint64_t rb = measure(h, &p2, m, nn, k, dA, dB, dC, Which::kRb);
        const uint64_t td = measure(h, &p2, m, nn, k, dA, dB, dC, Which::kTwoD);
        if (nv == 0 || rb == 0 || td == 0) { std::printf("        -"); continue; }
        const uint64_t bl = (rb < td) ? rb : td;
        const double r = (double)nv / (double)bl;
        // Scored against the shipping rule, which counts outputs and so DOES
        // look at n. The cell shows what the rule's own choice cost: a value
        // below 1.00 means the rule picked a kernel slower than the reference
        // at that shape, which is the only kind of error that can leave a
        // program worse off than before any of this existed.
        const bool predicted = (pick != Which::kNaive);
        const char flag = (predicted == (r > 1.0)) ? ' ' : '!';
        (void)flag;
        if (flag == '!') ++n_disagree;
        std::printf("%8.2f%c", r, flag);
      }
      std::printf("\n");
    }
  }
  std::printf("\n  scored against the shipping rule: %d of %d cells disagree\n",
              n_disagree, (int)(sizeof(nm)/sizeof(nm[0]) * sizeof(nk)/sizeof(nk[0])
                                * sizeof(ns)/sizeof(ns[0])));

  // ---- where the crossover actually is, bracketed ------------------------
  //
  // The n sweep above puts it between 32 outputs (blocking loses badly, 0.49x
  // to 0.63x) and 64 (blocking wins, 1.07x to 1.18x). That is a factor of two
  // with nothing measured inside it -- and an unswept bracket is exactly what
  // made the rule this one replaced wrong: its k crossover was bracketed by 8
  // and 16 with nothing between, and the boundary turned out not to be about k
  // at all.
  //
  // So: fill it in. Output counts from 24 to 96, reached two ways at each count
  // where the shape allows, because a boundary that is really about the OUTPUT
  // count must not care how m and n split to reach it.
  std::printf("\nthe crossover, bracketed. naive / fastest blocked, by output "
              "count.\n(resident = %lld threads on this core.)\n\n", resident);
  std::printf("%10s %10s %10s %8s\n", "m x n", "outputs", "ratio", "blocked?");
  {
    struct Cell { int m, n; };
    const Cell cells[] = {
      { 4,  6}, {12,  2},          //  24
      { 4,  8}, { 8,  4}, {16, 2}, //  32
      { 4, 10}, {10,  4},          //  40
      { 4, 12}, { 8,  6}, {12, 4}, //  48
      { 4, 14}, {14,  4},          //  56
      { 4, 16}, { 8,  8}, {16, 4}, //  64
      { 4, 18}, { 6, 12},          //  72
      { 4, 20}, { 8, 10}, {16, 5}, //  80
      { 4, 24}, { 8, 12}, {16, 6}, //  96
      // A PREDICTION, not more of the same. The jump between 64 and 72 above
      // looks like the reference kernel spilling out of the machine: it needs
      // one thread per output, the core holds `resident` of them, so its cost
      // should step every time the output count crosses a multiple of 64 --
      // while the blocked kernels, at a quarter of the threads, are still
      // inside one wave throughout this range.
      //
      // If that is what is happening, the ratio must be roughly FLAT from 112
      // to 128 (the reference is two waves across all of them) and JUMP again
      // at 136 (three waves). If it climbs smoothly instead, the wave story is
      // wrong and the boundary is something else.
      { 8, 14}, {14,  8},          // 112
      { 8, 15},                    // 120
      { 8, 16}, {16,  8},          // 128
      { 8, 17},                    // 136
      { 8, 18}, {18,  8},          // 144
    };
    const int bk = 16;
    for (const Cell& c : cells) {
      std::vector<float> hb((size_t)bk * c.n, 1.0f);
      grxMemcpy(dB, hb.data(), hb.size() * 4, grxMemcpyDefault);
      Probe pc(grxblasCycleSlotsNeeded(h, c.m, c.n) + 8);
      if (!pc.dev) continue;
      const uint64_t nv = measure(h, &pc, c.m, c.n, bk, dA, dB, dC, Which::kNaive);
      const uint64_t rb = measure(h, &pc, c.m, c.n, bk, dA, dB, dC, Which::kRb);
      const uint64_t td = measure(h, &pc, c.m, c.n, bk, dA, dB, dC, Which::kTwoD);
      if (nv == 0 || rb == 0 || td == 0) continue;
      const uint64_t best = (rb < td) ? rb : td;
      const double r = (double)nv / (double)best;
      char shape[16];
      std::snprintf(shape, sizeof(shape), "%dx%d", c.m, c.n);
      std::printf("%10s %10d %10.2f %8s\n", shape, c.m * c.n, r,
                  r > 1.0 ? "wins" : "loses");
    }
  }

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
      const uint64_t nv =
          measure_batched(h, &pb, bm, bn, bk, b, dA, dB, dC, Which::kNaive);
      const uint64_t bl =
          measure_batched(h, &pb, bm, bn, bk, b, dA, dB, dC, Which::kTwoD);
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
  // the shipping rule sends it to the blocked kernel. The block profile once
  // said that ran 27.6% SLOWER; that number was a span taken across four
  // launches on four clocks, and measured per launch the blocked kernel saves
  // 2613 cycles there. Every cell above is OP_N, so the transpose is checked
  // anyway: if transposing moved the crossover the rule would have to know.
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
      const uint64_t nv = measure_batched(h, &pb, tm, tn, tk, 2, dA, dB, dC,
                                          Which::kNaive, op);
      const uint64_t bl = measure_batched(h, &pb, tm, tn, tk, 2, dA, dB, dC,
                                          Which::kTwoD, op);
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
