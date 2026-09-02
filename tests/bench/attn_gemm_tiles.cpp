// Which tile should attention's GEMMs use?
//
// attention_cycles.cpp splits attention into four launches and finds the scores
// GEMM is 35.5% of it at S=64 -- the share that grows fastest with sequence
// length. GRXBLAS_SGEMM_TRACE says both of attention's GEMMs run the 2x2 tile,
// because the rule has ONE threshold (outputs >= resident -> 2x2) and the wider
// tiles are reachable only by asking:
//
//     "THE WIDE TILE IS NOT IN THE RULE. Reachable only by asking, because
//      nothing has measured it yet"                          -- grxblas.cpp
//
// This is that measurement. It is the same staging the 2D tile went through.
//
// WHY THIS SHAPE IS THE INTERESTING ONE. At S=64, H=4, Dh=8 the two GEMMs do
// IDENTICAL arithmetic -- 131072 multiply-adds each -- and cost 487869 and
// 279249 cycles. The difference is the shape of the work:
//
//     scores:  m=S,  n=S,  k=Dh    16384 outputs of 8 MACs each
//     out:     m=Dh, n=S,  k=S      2048 outputs of 64 MACs each
//
// The scores GEMM pays per-output setup eight times as often over the same
// FLOPs, and per-output setup is exactly what a wider register tile amortises:
// RM*RN outputs share RM+RN operand loads per k step instead of 2*RM*RN.
// Whether that predicted saving survives contact with the register file is the
// question, and prediction is not measurement.
//
// WHAT IT REPORTS. For both attention shapes across the sequence sweep, every
// available tiling over the SAME operands, in device cycles, each checked
// against the reference kernel's output before its cycles are believed. A
// kernel that is fast and wrong is not a result.
//
// These are SIMX cycles on one SM with a 4-lane warp. They compare kernels on
// one configuration, which is the claim a tiling rule needs; they are not
// hardware performance.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "grx/grx_runtime.h"
#include "grx/grxblas.h"
#include "grx/grx_cycles.h"

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  std::printf("  %-5s %s\n", cond ? "ok" : "FAIL", what);
  if (!cond) ++g_failures;
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) { if (grxMalloc(&p, bytes) != grxSuccess) p = nullptr; }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

void fill(std::vector<float>& v, int seed) {
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = (float)(((int)((i * 37 + seed * 11) % 13)) - 6) * 0.25f;
}

// The five reachable tilings. "" is the rule's own choice, left alone.
struct Tiling { const char* label; const char* env; };
const Tiling kTilings[] = {
  {"naive",     "GRXBLAS_SGEMM_NAIVE"},
  {"rb",        "GRXBLAS_SGEMM_RB"},
  {"2d  (2x2)", "GRXBLAS_SGEMM_2D"},
  {"2d-i",      "GRXBLAS_SGEMM_2D_I"},
  {"4x2",       "GRXBLAS_SGEMM_4X2"},
  {"4x4",       "GRXBLAS_SGEMM_4X4"},
};

void clear_forces() {
  for (const Tiling& t : kTilings) unsetenv(t.env);
}

struct Shape {
  const char* name;
  int m, n, k, batch;
  grxblasOperation_t ta, tb;
  int lda, ldb, ldc;
  long long sa, sb, sc;
};

// One measured run. Returns false when the probe could not produce a span this
// bench is allowed to believe.
bool run_one(grxblasHandle_t blas, const Shape& sh, const char* env,
             void* dA, void* dB, void* dC, grxCycleSlot* slots, int cap,
             int occupancy, uint64_t* out_span, int* out_warps) {
  clear_forces();
  if (env) setenv(env, "1", 1);

  const int need = grxblasCycleSlotsNeeded(blas, sh.m, sh.n) * sh.batch;
  if (need <= 0 || need > cap) return false;
  grxMemset(slots, 0, (size_t)cap * sizeof(grxCycleSlot));
  grxblasSetCycleProbe(blas, slots, cap);

  const float alpha = 1.0f, beta = 0.0f;
  const grxblasStatus_t bs = grxblasSgemmStridedBatched(
      blas, sh.ta, sh.tb, sh.m, sh.n, sh.k, &alpha,
      dA, sh.lda, sh.sa, dB, sh.ldb, sh.sb,
      &beta, dC, sh.ldc, sh.sc, sh.batch);
  grxblasSetCycleProbe(blas, nullptr, 0);
  clear_forces();
  if (bs != GRXBLAS_STATUS_SUCCESS) return false;
  if (grxDeviceSynchronize() != grxSuccess) return false;

  std::vector<grxCycleSlot> host((size_t)cap);
  grxMemcpy(host.data(), slots, host.size() * sizeof(grxCycleSlot),
            grxMemcpyDefault);
  grxCycleSummary sum{};
  grxCycleSummarize(host.data(), need, &sum);
  // Same two refusals attention_cycles makes, and for the same reason: MCYCLE
  // restarts at every launch, so a span across two launches is meaningless and
  // more warps live than the device holds is how that shows up.
  if (!sum.spanIsValid) return false;
  if (sum.maxLive > occupancy) return false;
  *out_span = sum.span;
  *out_warps = sum.warps;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess) {
    std::printf("no device\n");
    return 77;
  }
  std::printf("device: %s, %d SM x %d warps x %d lanes\n", prop.name,
              prop.multiProcessorCount, prop.maxWarpsPerMultiProcessor,
              prop.warpSize);
  const int occupancy = prop.multiProcessorCount * prop.maxWarpsPerMultiProcessor;

  grxblasHandle_t blas = nullptr;
  if (grxblasCreate(&blas) != GRXBLAS_STATUS_SUCCESS) {
    std::printf("grxblasCreate failed\n");
    return 77;
  }

  const int H = 4, Dh = 8;                 // matches attention_cycles.cpp
  const int seqs[] = {8, 16, 32, 64};

  struct Row {
    int S; std::string which; std::string tiling;
    uint64_t span; int warps; bool correct;
  };
  std::vector<Row> rows;

  for (int si = 0; si < (int)(sizeof(seqs) / sizeof(seqs[0])); ++si) {
    const int S = seqs[si];

    // The two shapes exactly as grxdnn issues them (grxdnn.cpp, attention).
    const Shape shapes[2] = {
      {"scores", S, S, Dh, H, GRXBLAS_OP_T, GRXBLAS_OP_N,
       Dh, Dh, S, (long long)S * Dh, (long long)S * Dh, (long long)S * S},
      {"out",    Dh, S, S, H, GRXBLAS_OP_N, GRXBLAS_OP_N,
       Dh, S, Dh, (long long)S * Dh, (long long)S * S, (long long)S * Dh},
    };

    for (const Shape& sh : shapes) {
      const size_t na = (size_t)sh.batch * (size_t)llabs(sh.sa ? sh.sa : 1);
      const size_t nb = (size_t)sh.batch * (size_t)llabs(sh.sb ? sh.sb : 1);
      const size_t nc = (size_t)sh.batch * (size_t)llabs(sh.sc ? sh.sc : 1);

      std::vector<float> ha(na), hb(nb);
      fill(ha, 3); fill(hb, 17);

      Buf A(na * 4), B(nb * 4), C(nc * 4);
      if (!A.p || !B.p || !C.p) { expect(false, "allocate operands"); continue; }
      grxMemcpy(A.p, ha.data(), na * 4, grxMemcpyDefault);
      grxMemcpy(B.p, hb.data(), nb * 4, grxMemcpyDefault);

      const int cap = grxblasCycleSlotsNeeded(blas, sh.m, sh.n) * sh.batch + 8;
      Buf slots((size_t)cap * sizeof(grxCycleSlot));
      if (!slots.p) { expect(false, "allocate probe"); continue; }

      // The reference answer first, from the kernel that IS the oracle.
      std::vector<float> ref(nc, 0.0f);
      {
        uint64_t sp = 0; int w = 0;
        grxMemset(C.p, 0, nc * 4);
        if (!run_one(blas, sh, "GRXBLAS_SGEMM_NAIVE", A.p, B.p, C.p,
                     (grxCycleSlot*)slots.p, cap, occupancy, &sp, &w)) {
          char msg[128];
          std::snprintf(msg, sizeof(msg), "S=%d %s: reference run produced a span",
                        S, sh.name);
          expect(false, msg);
          continue;
        }
        grxMemcpy(ref.data(), C.p, nc * 4, grxMemcpyDefault);
        rows.push_back({S, sh.name, "naive", sp, w, true});
      }

      for (int t = 1; t < (int)(sizeof(kTilings) / sizeof(kTilings[0])); ++t) {
        uint64_t sp = 0; int w = 0;
        grxMemset(C.p, 0, nc * 4);
        if (!run_one(blas, sh, kTilings[t].env, A.p, B.p, C.p,
                     (grxCycleSlot*)slots.p, cap, occupancy, &sp, &w))
          continue;                     // unreachable tiling: not a failure
        std::vector<float> got(nc, 0.0f);
        grxMemcpy(got.data(), C.p, nc * 4, grxMemcpyDefault);
        bool same = true;
        for (size_t i = 0; i < nc && same; ++i)
          if (got[i] != ref[i]) same = false;
        rows.push_back({S, sh.name, kTilings[t].label, sp, w, same});
      }
    }
  }

  // ---- report --------------------------------------------------------------
  std::printf("\n  %-4s %-7s %-10s %10s %7s %9s %s\n",
              "S", "gemm", "tiling", "cycles", "warps", "vs 2x2", "correct");
  for (int si = 0; si < (int)(sizeof(seqs) / sizeof(seqs[0])); ++si) {
    for (const char* which : {"scores", "out"}) {
      uint64_t base = 0;
      for (const Row& r : rows)
        if (r.S == seqs[si] && r.which == which && r.tiling == "2d  (2x2)")
          base = r.span;
      for (const Row& r : rows) {
        if (r.S != seqs[si] || r.which != which) continue;
        char ratio[16] = "     --";
        if (base && r.span)
          std::snprintf(ratio, sizeof(ratio), "%6.2fx", (double)base / (double)r.span);
        std::printf("  %-4d %-7s %-10s %10llu %7d %9s %s\n",
                    r.S, r.which.c_str(), r.tiling.c_str(),
                    (unsigned long long)r.span, r.warps, ratio,
                    r.correct ? "yes" : "*** NO ***");
      }
    }
  }

  // Every tiling that produced a number must have produced the right numbers.
  bool all_correct = true;
  for (const Row& r : rows) if (!r.correct) all_correct = false;
  expect(all_correct, "every tiling that ran agrees with the reference exactly");

  // The 2x2 tile must have been reachable everywhere, or the ratios above are
  // against nothing and the table is decoration.
  bool have_base = true;
  for (int si = 0; si < (int)(sizeof(seqs) / sizeof(seqs[0])); ++si)
    for (const char* which : {"scores", "out"}) {
      bool found = false;
      for (const Row& r : rows)
        if (r.S == seqs[si] && r.which == which && r.tiling == "2d  (2x2)") found = true;
      if (!found) have_base = false;
    }
  expect(have_base, "the 2x2 tile ran at every point, so the ratios have a base");

  if (out_path) {
    FILE* f = std::fopen(out_path, "w");
    if (f) {
      std::fprintf(f, "{\n  \"bench\": \"attn_gemm_tiles\",\n");
      std::fprintf(f, "  \"device\": {\"name\": \"%s\", \"sms\": %d,"
                      " \"warp\": %d, \"warp_slots\": %d},\n",
                   prop.name, prop.multiProcessorCount, prop.warpSize,
                   prop.maxWarpsPerMultiProcessor);
      std::fprintf(f, "  \"rows\": [\n");
      for (size_t i = 0; i < rows.size(); ++i)
        std::fprintf(f, "    {\"seq\": %d, \"gemm\": \"%s\", \"tiling\": \"%s\","
                        " \"span\": %llu, \"warps\": %d, \"correct\": %s}%s\n",
                     rows[i].S, rows[i].which.c_str(), rows[i].tiling.c_str(),
                     (unsigned long long)rows[i].span, rows[i].warps,
                     rows[i].correct ? "true" : "false",
                     i + 1 < rows.size() ? "," : "");
      std::fprintf(f, "  ]\n}\n");
      std::fclose(f);
    }
  }

  grxblasDestroy(blas);
  std::printf("\n%s (%d failures)\n", g_failures ? "FAILED" : "PASSED", g_failures);
  return g_failures ? 1 : 0;
}
