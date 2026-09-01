// The blocked sgemm kernels against the naive one, on the same device.
//
// THE REFERENCE KERNEL IS THE ORACLE HERE, and that is a stronger check than
// this library has had for a GEMM before. test_grxblas.cpp compares sgemm to a
// host reference at a relative tolerance scaled by k, because the device
// accumulates in a different order than the host and float addition is not
// associative. That tolerance is real headroom, and a tuned kernel that quietly
// changed the arithmetic could hide inside it.
//
// Neither blocked kernel changes the arithmetic. sgemm_rb changes which thread
// computes which output and how often B is loaded; sgemm_2d does the same for
// both operands at once. Every accumulation is still `acc += a * b` over l in
// the same order. So all three kernels must agree BIT FOR BIT, and this
// compares them with ==. No tolerance, nothing to hide in.
//
// All three run on the device over the same operands, selected through the
// environment hooks the host reads, so this is one binary running three kernels
// rather than a kernel compared against a description of itself.
//
// WHAT THE SHAPES ARE FOR. The tails. sgemm_rb slices rows by RM = 4 and
// sgemm_2d slices rows by 2 and COLUMNS by 2, so the interesting cases are the
// ones where m is not a multiple of the row tile and n is not a multiple of the
// column tile -- there the blocked kernels clamp indices they must then
// discard. m = 1, 2, 3 are below RM entirely, where the host falls back to the
// reference for sgemm_rb; m = 5, 7, 13 straddle; n = 1 and n = 3, 5, 7 give
// sgemm_2d a column remainder. All four transpose combinations, because the
// index algebra is written out in all three kernels and duplicated algebra is
// what drifts.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "../unit/grx_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

struct Buf {
  void* p = nullptr;
  Buf() = default;
  explicit Buf(size_t bytes) { alloc(bytes); }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  void alloc(size_t bytes) {
    if (p) { grxFree(p); p = nullptr; }
    if (bytes && grxMalloc(&p, bytes) != grxSuccess) p = nullptr;
  }
  float* f() const { return (float*)p; }
};

void fill(std::vector<float>& v, unsigned seed) {
  for (size_t i = 0; i < v.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    // Quarter-integers in [-2, 2]: exactly representable, so any difference
    // between the two kernels is a difference in what they computed rather
    // than in how a decimal landed.
    v[i] = (float)((int)(seed >> 16) % 17 - 8) * 0.25f;
  }
}

// Which kernel a run should reach.
//
// kRule is not a kernel: it is what a caller gets, and the rb comparison uses
// it deliberately so that what is checked is the SHIPPING path rather than a
// kernel the rule may never select. Below the row tile that means the
// reference, and the cases say so.
enum class Pick { kNaive, kRule, kRb, kTwoD, kMid, kWide };

// Run one sgemm and return C. The kernel is chosen through the same
// environment hooks the library reads, so every path goes through the real
// selection logic rather than a test-only door.
bool run(Pick pick, bool ta, bool tb, int m, int n, int k,
         const std::vector<float>& A, const std::vector<float>& B,
         std::vector<float>* C, int* warps = nullptr) {
  unsetenv("GRXBLAS_SGEMM_NAIVE");
  unsetenv("GRXBLAS_SGEMM_RB");
  unsetenv("GRXBLAS_SGEMM_2D");
  unsetenv("GRXBLAS_SGEMM_4X2");
  unsetenv("GRXBLAS_SGEMM_4X4");
  if (pick == Pick::kNaive) setenv("GRXBLAS_SGEMM_NAIVE", "1", 1);
  if (pick == Pick::kRb)    setenv("GRXBLAS_SGEMM_RB", "1", 1);
  if (pick == Pick::kTwoD)  setenv("GRXBLAS_SGEMM_2D", "1", 1);
  if (pick == Pick::kMid)   setenv("GRXBLAS_SGEMM_4X2", "1", 1);
  if (pick == Pick::kWide)  setenv("GRXBLAS_SGEMM_4X4", "1", 1);

  // A fresh handle each time: the kernel choice is made per call from the
  // environment, but creating the handle here also means neither run can be
  // affected by state the other left behind.
  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return false;

  const int lda = ta ? k : m;
  const int ldb = tb ? n : k;
  const int ldc = m;

  Buf dA(A.size() * sizeof(float)), dB(B.size() * sizeof(float)),
      dC((size_t)ldc * n * sizeof(float));
  if (!dA.p || !dB.p || !dC.p) { grxblasDestroy(h); return false; }
  grxMemcpy(dA.p, A.data(), A.size() * sizeof(float), grxMemcpyDefault);
  grxMemcpy(dB.p, B.data(), B.size() * sizeof(float), grxMemcpyDefault);
  // Poison C: beta is zero, so every element must be written. A kernel that
  // skipped one would otherwise inherit whatever was there.
  std::vector<float> poison((size_t)ldc * n, -7777.0f);
  grxMemcpy(dC.p, poison.data(), poison.size() * sizeof(float), grxMemcpyDefault);

  // Optionally count the warps the launch made. That is the only thing visible
  // from out here that says WHICH kernel ran -- the results are supposed to be
  // identical, so the results cannot say. One slot per block, and the three
  // kernels tile the same output differently, so at the right shape the three
  // counts differ.
  Buf probe;
  int probe_slots = 0;
  if (warps) {
    *warps = -1;
    probe_slots = grxblasCycleSlotsNeeded(h, m, n) + 8;
    if (probe_slots > 0) {
      probe.alloc((size_t)probe_slots * sizeof(grxCycleSlot));
      if (probe.p) {
        grxMemset(probe.p, 0, (size_t)probe_slots * sizeof(grxCycleSlot));
        grxblasSetCycleProbe(h, (grxCycleSlot*)probe.p, probe_slots);
      }
    }
  }

  const float alpha = 1.0f, beta = 0.0f;
  const grxblasStatus_t st = grxblasSgemm(
      h, ta ? GRXBLAS_OP_T : GRXBLAS_OP_N, tb ? GRXBLAS_OP_T : GRXBLAS_OP_N,
      m, n, k, &alpha, dA.p, lda, dB.p, ldb, &beta, dC.p, ldc);
  if (st != GRXBLAS_STATUS_SUCCESS) { grxblasDestroy(h); return false; }

  C->resize(poison.size());
  grxMemcpy(C->data(), dC.p, C->size() * sizeof(float), grxMemcpyDefault);
  if (warps && probe.p) {
    std::vector<grxCycleSlot> host((size_t)probe_slots);
    grxMemcpy(host.data(), probe.p,
              host.size() * sizeof(grxCycleSlot), grxMemcpyDefault);
    grxCycleSummary sum{};
    grxCycleSummarize(host.data(), probe_slots, &sum);
    *warps = sum.warps;
  }
  grxblasDestroy(h);
  return true;
}

}  // namespace

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  {
    grxblasHandle_t h = nullptr;
    if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) return 1;
    void* d = nullptr;
    const float one = 1.0f, zero = 0.0f;
    if (grxMalloc(&d, sizeof(float)) == grxSuccess) {
      const grxblasStatus_t probe = grxblasSgemm(
          h, GRXBLAS_OP_N, GRXBLAS_OP_N, 1, 1, 1, &one, d, 1, d, 1, &zero, d, 1);
      grxFree(d);
      if (probe == GRXBLAS_STATUS_NOT_INITIALIZED) {
        std::printf("grxblas device kernels not found; skipping\n");
        grxblasDestroy(h);
        return 77;
      }
    }
    grxblasDestroy(h);
  }

  struct Case { int m, n, k; const char* what; };
  const Case cases[] = {
    { 1, 3, 4, "m=1, below the row tile -- the host must fall back"},
    { 2, 3, 4, "m=2, below the row tile"},
    { 3, 3, 4, "m=3, below the row tile"},
    { 4, 3, 5, "m=4, exactly one rb tile"},
    { 5, 4, 6, "m=5, one rb tile and a remainder of 1"},
    { 5, 1, 4, "n=1, one column against a 2-wide column tile"},
    { 7, 2, 6, "n=2, exactly one column tile"},
    { 7, 5, 3, "m=7, one rb tile and a remainder of 3"},
    { 8, 6, 8, "m=8, two whole rb tiles"},
    {13, 7, 9, "m=13, three rb tiles and a remainder of 1"},
    {16, 4, 12, "m=16, four whole rb tiles"},
  };

  // The two blocked kernels, each against the reference, over identical
  // operands. Run in one pass rather than two loops so a shape that breaks one
  // of them is reported next to the shape that did not break the other.
  struct Path { Pick pick; const char* name; };
  const Path paths[4] = {{Pick::kRule, "the rule's choice"},
                         {Pick::kTwoD, "sgemm_2d (forced)"},
                         {Pick::kMid,  "sgemm_4x2 (forced)"},
                         {Pick::kWide, "sgemm_4x4 (forced)"}};

  for (const Path& path : paths) {
    char head[128];
    std::snprintf(head, sizeof(head),
                  "%s agrees with the reference kernel, bit for bit", path.name);
    section(head);
    for (const Case& c : cases) {
      for (int t = 0; t < 4; ++t) {
        const bool ta = (t & 1) != 0, tb = (t & 2) != 0;
        const int lda = ta ? c.k : c.m;
        const int ldb = tb ? c.n : c.k;
        std::vector<float> A((size_t)lda * (ta ? c.m : c.k));
        std::vector<float> B((size_t)ldb * (tb ? c.k : c.n));
        fill(A, 5u + (unsigned)t); fill(B, 91u + (unsigned)t);

        std::vector<float> ref, fast;
        char label[160];
        std::snprintf(label, sizeof(label), "%s  [%c%c]", c.what,
                      ta ? 'T' : 'N', tb ? 'T' : 'N');

        if (!run(Pick::kNaive, ta, tb, c.m, c.n, c.k, A, B, &ref) ||
            !run(path.pick, ta, tb, c.m, c.n, c.k, A, B, &fast)) {
          std::printf("  FAIL  %s: a run failed\n", label);
          ++grxtest::failures();
          continue;
        }

        size_t at = ref.size();
        for (size_t i = 0; i < ref.size(); ++i) {
          if (std::memcmp(&ref[i], &fast[i], sizeof(float)) != 0) { at = i; break; }
        }
        if (at == ref.size()) {
          std::printf("  ok    %s\n", label);
        } else {
          std::printf("  FAIL  %s: differ at [%zu]: reference %.9g, blocked %.9g\n",
                      label, at, (double)ref[at], (double)fast[at]);
          ++grxtest::failures();
        }
      }
    }
  }

  section("five kernels actually ran, not one kernel five times");
  {
    // Everything above would pass just as happily if all three runs were the
    // SAME kernel -- if a hook were misspelled, the host ignored it, or the
    // module carried neither blocked entry point. Identical results are the
    // POINT here, so the results cannot be the evidence.
    //
    // The warp count can. One probe slot per BLOCK -- not per output -- and the
    // three kernels tile the same output differently. At warp 4, m=6, n=8:
    //
    //   naive  6*8                 = 48 threads -> 12 blocks
    //   rb     ceil(6/4)*8         = 16         ->  4
    //   2d     ceil(6/2)*ceil(8/2) = 12         ->  3
    //   4x2    ceil(6/4)*ceil(8/2) =  8         ->  2
    //   4x4    ceil(6/4)*ceil(8/4) =  4         ->  1
    //
    // Three different numbers, and getting there took two goes. The first
    // version used m=6 n=4, where rb needs 8 threads and 2d needs 6 -- both two
    // blocks after the divide by the warp width, so the counts collapsed. It
    // also asked for rb through the RULE, which declines it at 24 outputs, so
    // that run was the reference wearing rb's label. Both mistakes showed up
    // here as equal counts, which is what this check is for.
    //
    // All three are FORCED, because what is being established is that three
    // kernels exist and are reachable. Whether the rule would pick them is a
    // different question, asked by the pass above and by the sweeps.
    std::vector<float> A(64), B(64), out;
    fill(A, 3u); fill(B, 17u);
    int w[5] = {-1, -1, -1, -1, -1};
    const Pick picks[5] = {Pick::kNaive, Pick::kRb, Pick::kTwoD, Pick::kMid,
                           Pick::kWide};
    bool ok = true;
    for (int i = 0; i < 5; ++i)
      ok = ok && run(picks[i], false, false, 6, 8, 5, A, B, &out, &w[i]);
    check(ok, "all five probed runs completed");
    std::printf("        warps: naive %d, rb %d, 2d %d, 4x2 %d, 4x4 %d\n",
                w[0], w[1], w[2], w[3], w[4]);
    bool wrote = true, all_distinct = true;
    for (int i = 0; i < 5; ++i) {
      if (w[i] <= 0) wrote = false;
      for (int j = i + 1; j < 5; ++j)
        if (w[i] == w[j]) all_distinct = false;
    }
    check(wrote, "each run wrote probe slots");
    check(all_distinct,
          "all five launches are distinguishable by their warp count, so five "
          "kernels ran");
  }

  section("what the RULE can reach, measured rather than read");
  {
    // The section above establishes that five kernels EXIST. This one asks the
    // different question it explicitly deferred: which of them a caller who
    // asks for nothing can actually get.
    //
    // WHY IT MATTERS OUTSIDE THIS FILE. ci/check_kernel_loops.py ranks kernels
    // by instructions per float op, and its top two entries are sgemm at 24.00
    // and sgemm_rb at 13.25 -- ahead of everything a program reaches. Both
    // numbers are right and neither is a defect: one is the oracle, the other
    // is the 4x1 rung of the tile ladder, and both must stay slow. That census
    // now LABELS them so its ranking cannot be read as a worklist, and a label
    // is a claim. This is where the claim is checked, because reachability
    // lives in decide_sgemm_kernel and a disassembly cannot see it.
    //
    // HOW. The rule is asked at shapes spanning its threshold, and the answer
    // is identified by WARP COUNT -- the same evidence the section above uses,
    // and the only thing visible from out here that distinguishes kernels whose
    // results are deliberately identical.
    //
    // THE YARDSTICK IS ARITHMETIC, NOT A FORCED RUN, and that took a second go.
    // The first version compared the rule's warp count against the count from
    // each kernel forced through its environment hook. That is circular: a hook
    // whose kernel is unavailable falls back to THE RULE, so "forced 2d" can
    // quietly be the rule's own answer wearing 2d's label -- the same mistake
    // the section above records making. Making the rule blind to sgemm_2d as a
    // test of this check produced three columns reading 10, 10 and 10, and the
    // check dutifully reported that the rule had picked 2d when it had picked
    // rb.
    //
    // So the expected counts are COMPUTED from the shape and the tiling, and
    // the forced runs are checked against that arithmetic rather than trusted
    // as it. A force path that silently falls back now fails as itself.
    struct RuleCase {
      int m, n, k;
      const char* what;
    };
    // resident = warpSize * maxWarpsPerMultiProcessor * multiProcessorCount.
    // On the 4-lane, 16-slot, 1-SM configuration that is 64 outputs.
    const RuleCase rcases[] = {
      { 6,  8, 5, "48 outputs, below resident"},
      {10,  7, 5, "70 outputs, above resident"},
      {14, 10, 5, "140 outputs, above TWICE resident -- where rb's branch is"},
    };
    const char* names[5] = {"naive", "rb", "2d", "4x2", "4x4"};
    const Pick  picks[5] = {Pick::kNaive, Pick::kRb, Pick::kTwoD, Pick::kMid,
                            Pick::kWide};
    // Everything the rule is allowed to return. sgemm_rb is NOT here: its
    // branch in decide_sgemm_kernel is an `else if` behind the 2D tile, so a
    // module that has sgemm_2d can never reach it. That is the whole claim.
    const bool reachable[5] = {true, false, true, false, false};

    // The tile of each kernel, which must match kernels/sgemm.cpp. Duplicated
    // knowledge, and deliberately so: the forced runs below are checked against
    // the counts these produce, so a tile that drifts from the kernel fails
    // here rather than being silently absorbed.
    const int tile_m[5] = {1, 4, 2, 4, 4};
    const int tile_n[5] = {1, 1, 2, 2, 4};

    grxDeviceProp_t prop{};
    if (grxGetDeviceProperties(&prop, 0) != grxSuccess) {
      check(false, "device properties");
    } else {
      const int warp = prop.warpSize;
      for (const RuleCase& rc : rcases) {
        std::vector<float> A((size_t)rc.m * rc.k + 64), B((size_t)rc.k * rc.n + 64);
        std::vector<float> out;
        fill(A, 3u); fill(B, 17u);

        // What each tiling WOULD launch, from the shape alone.
        int want[5];
        for (int i = 0; i < 5; ++i) {
          const int threads = ((rc.m + tile_m[i] - 1) / tile_m[i]) *
                              ((rc.n + tile_n[i] - 1) / tile_n[i]);
          want[i] = (threads + warp - 1) / warp;
        }

        int w[5] = {-1, -1, -1, -1, -1}, wrule = -1;
        bool ok = run(Pick::kRule, false, false, rc.m, rc.n, rc.k, A, B, &out,
                      &wrule);
        for (int i = 0; i < 5; ++i)
          ok = ok && run(picks[i], false, false, rc.m, rc.n, rc.k, A, B, &out,
                         &w[i]);
        if (!ok) { check(false, rc.what); continue; }

        std::printf("        %s\n", rc.what);
        std::printf("          rule %3d | expected  naive %3d rb %3d 2d %3d"
                    " 4x2 %3d 4x4 %3d\n", wrule, want[0], want[1], want[2],
                    want[3], want[4]);
        std::printf("                   | forced    naive %3d rb %3d 2d %3d"
                    " 4x2 %3d 4x4 %3d\n", w[0], w[1], w[2], w[3], w[4]);

        // The shapes are chosen so the five tilings are told apart. If one
        // stops being so the identification below is worthless, and this is
        // what says so -- checked on the ARITHMETIC, which is a property of
        // the shape and holds whatever the kernels do.
        bool distinct = true;
        for (int i = 0; i < 5; ++i)
          for (int j = i + 1; j < 5; ++j)
            if (want[i] == want[j]) distinct = false;
        check(distinct, "the five tilings are distinguishable at this shape");

        // Each force path reached its own kernel. This is what catches a hook
        // that fell back to the rule and wore the label anyway.
        for (int i = 0; i < 5; ++i) {
          if (w[i] == want[i]) continue;
          char msg[192];
          std::snprintf(msg, sizeof(msg),
                        "forcing %s launched %d warps, not the %d its tile"
                        " needs -- the hook did not reach it",
                        names[i], w[i], want[i]);
          check(false, msg);
        }

        // And the rule's own answer, identified against the arithmetic.
        int matched = -1, matches = 0;
        for (int i = 0; i < 5; ++i)
          if (want[i] == wrule) { matched = i; ++matches; }
        check(matches == 1, "the rule's launch matches exactly one tiling");
        if (matches == 1) {
          char msg[192];
          std::snprintf(msg, sizeof(msg),
                        "the rule picked %s, which it is allowed to",
                        names[matched]);
          check(reachable[matched], msg);
        }
      }
    }
  }

  section("the comparison can actually fail");
  {
    // And the pipeline is live: perturbing one operand must move the output.
    // Weaker than the warp check above and kept because it fails differently --
    // this catches a device that returned the poison value, which equal warp
    // counts would not.
    std::vector<float> A(64), B(64), out1, out2;
    fill(A, 3u); fill(B, 17u);
    const bool ok1 = run(Pick::kRule, false, false, 8, 8, 8, A, B, &out1);
    A[0] += 1.0f;
    const bool ok2 = run(Pick::kRule, false, false, 8, 8, 8, A, B, &out2);
    check(ok1 && ok2, "both perturbation runs completed");
    bool differs = false;
    for (size_t i = 0; i < out1.size() && !differs; ++i)
      if (std::memcmp(&out1[i], &out2[i], sizeof(float)) != 0) differs = true;
    check(differs, "changing an input changes the output");
  }

  return grxtest::report();
}
