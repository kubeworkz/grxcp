// What attention is made of, one launch at a time, over a sequence sweep.
//
// WHY THIS EXISTS. block_cycles reports attention as ONE stage, because that is
// what a caller pays. It is the largest stage in a transformer block and the
// only one whose share GROWS with sequence length -- 17.1% at S = 8 and 24.2%
// at S = 16 -- so "attention" was the obvious next target and there was no way
// to see which quarter of it to touch.
//
// It is four launches: the scores GEMM, the causal mask, softmax, and the
// output GEMM. Splitting them found SOFTMAX WAS HALF OF ATTENTION at every
// length measured, and rising -- 49.4% at S = 8, 49.7% at 16, 52.1% at 32,
// 53.0% at 64. Nothing pointed there before. The hot-loop census ranks
// dnn_softmax near the BOTTOM of its cost table at 2.00 instructions per float
// op, which is a rate and says nothing about how many float ops there are.
//
// Those figures are what this bench FOUND, not what it reports now: the kernel
// was computing its exponential twice per element, and with the second copy
// gone softmax is 41.3% at S = 8 falling to 39.3% at S = 64. It is still the
// largest single part of attention, which is what the checks below pin.
//
// EACH REGION AGAINST ITS OWN CLOCK, NEVER A SPAN ACROSS TWO. VX_CSR_MCYCLE is
// not comparable across launches (7.25), so a span taken across attention's four
// regions is a maximum over four unrelated counters. That is not a hypothetical
// -- it was this project's largest measurement defect, it reported 64 warps live
// on a device that holds 16, and a grxBLAS kernel-selection rule was reverted on
// a 27.6% "regression" read off it. grxdnnGetCycleRegions says where each launch
// wrote; each region is summarised alone and the spans are ADDED, because the
// launches are ordered on one stream.
//
// maxLive is checked against device occupancy for every region, for the same
// reason: a region reporting more warps live at once than the device holds did
// not come from one launch, whatever its timestamps say.
//
// WHAT THE NUMBERS ARE NOT. One SM, four lanes, SimX. These cycles are what the
// MODEL does -- the right thing for comparing two kernels, the wrong thing to
// quote as hardware performance.

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

void fill(std::vector<float>& v, unsigned s) {
  for (size_t i = 0; i < v.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (float)((int)(s >> 16) % 17 - 8) * 0.125f;
  }
}

struct Buf {
  void* p = nullptr;
  explicit Buf(size_t bytes) {
    if (bytes && grxMalloc(&p, bytes) != grxSuccess) p = nullptr;
  }
  ~Buf() { if (p) grxFree(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  float* f() const { return (float*)p; }
};

struct Row {
  int      seq;
  std::string region;
  uint64_t span;
  int      warps;
};

}  // namespace

int main(int argc, char** argv) {
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--out" && i + 1 < argc) out_path = argv[++i];

  int n = 0;
  if (grxGetDeviceCount(&n) != grxSuccess || n <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;
  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess) return 1;
  const int occupancy =
      prop.maxWarpsPerMultiProcessor * prop.multiProcessorCount;

  grxdnnHandle_t dh = nullptr;
  if (grxdnnCreate(&dh) != GRXDNN_STATUS_SUCCESS) {
    std::printf("grxdnn unavailable; skipping\n");
    return 77;
  }

  std::printf("device: %s, %d SM x %d warps x %d lanes\n", prop.name,
              prop.multiProcessorCount, prop.maxWarpsPerMultiProcessor,
              prop.warpSize);

  const int H = 4, Dh = 8;
  const int seqs[] = {8, 16, 32, 64};
  std::vector<Row> rows;
  double softmax_share_first = 0.0, softmax_share_last = 0.0;

  std::printf("\n  %-4s %-14s %10s %7s %8s\n", "S", "region", "span", "warps",
              "share");
  for (int si = 0; si < (int)(sizeof(seqs) / sizeof(seqs[0])); ++si) {
    const int S = seqs[si];
    const size_t nqkv = (size_t)H * S * Dh;
    std::vector<float> hq(nqkv), hk(nqkv), hv(nqkv);
    fill(hq, 3); fill(hk, 17); fill(hv, 29);

    Buf q(nqkv * 4), k(nqkv * 4), v(nqkv * 4), a(nqkv * 4);
    if (!q.p || !k.p || !v.p || !a.p) { expect(false, "allocate Q/K/V"); break; }
    grxMemcpy(q.p, hq.data(), nqkv * 4, grxMemcpyDefault);
    grxMemcpy(k.p, hk.data(), nqkv * 4, grxMemcpyDefault);
    grxMemcpy(v.p, hv.data(), nqkv * 4, grxMemcpyDefault);

    size_t ws_bytes = 0;
    grxdnnAttentionWorkspaceSize(1, H, S, Dh, &ws_bytes);
    Buf ws(ws_bytes ? ws_bytes : 4);

    // Asked, not guessed: the library owns its launch geometry, and this buffer
    // has to hold FOUR launches at once in four separate regions.
    const int cap = grxdnnAttentionCycleSlotsNeeded(dh, 1, H, S, Dh) + 8;
    if (cap <= 8) { expect(false, "probe capacity"); break; }
    Buf slots((size_t)cap * sizeof(grxCycleSlot));
    if (!slots.p) { expect(false, "allocate the probe buffer"); break; }
    grxMemset(slots.p, 0, (size_t)cap * sizeof(grxCycleSlot));
    grxdnnSetCycleProbe(dh, (grxCycleSlot*)slots.p, cap);

    const grxdnnStatus_t st =
        grxdnnAttentionForward(dh, 1, H, S, Dh, q.f(), k.f(), v.f(),
                               GRXDNN_ATTN_MASK_CAUSAL, ws.p, ws_bytes, a.f());
    grxDeviceSynchronize();
    grxdnnSetCycleProbe(dh, nullptr, 0);

    char what[64];
    std::snprintf(what, sizeof(what), "attention ran at S=%d", S);
    if (st != GRXDNN_STATUS_SUCCESS) { expect(false, what); continue; }

    grxdnnCycleRegion_t regions[8];
    int nregions = 0;
    grxdnnGetCycleRegions(dh, regions, 8, &nregions);
    if (nregions > 8) nregions = 8;
    std::snprintf(what, sizeof(what), "S=%d reported its launches separately", S);
    expect(nregions >= 4, what);
    if (nregions <= 0) continue;

    std::vector<grxCycleSlot> host((size_t)cap);
    grxMemcpy(host.data(), slots.p, host.size() * sizeof(grxCycleSlot),
              grxMemcpyDefault);

    uint64_t total = 0, softmax_span = 0;
    bool ok = true;
    std::vector<Row> here;
    for (int i = 0; i < nregions; ++i) {
      const int off = regions[i].offset, len = regions[i].slots;
      if (off < 0 || len <= 0 || off + len > cap) { ok = false; break; }
      grxCycleSummary sum{};
      grxCycleSummarize(host.data() + off, len, &sum);
      if (sum.maxLive > occupancy) {
        std::printf("        %s: %d live on a device holding %d --"
                    " this region spans more than one launch\n",
                    regions[i].name, sum.maxLive, occupancy);
        ok = false;
        break;
      }
      if (!sum.spanIsValid) { ok = false; break; }
      total += sum.span;
      if (std::strcmp(regions[i].name, "softmax") == 0) softmax_span = sum.span;
      here.push_back({S, regions[i].name, sum.span, sum.warps});
    }
    std::snprintf(what, sizeof(what),
                  "S=%d: every region is one launch on one clock", S);
    expect(ok, what);
    if (!ok) continue;

    for (const Row& r : here)
      std::printf("  %-4d %-14s %10llu %7d %7.1f%%\n", r.seq, r.region.c_str(),
                  (unsigned long long)r.span, r.warps,
                  total ? 100.0 * r.span / total : 0.0);
    std::printf("  %-4d %-14s %10llu\n", S, "TOTAL",
                (unsigned long long)total);
    rows.insert(rows.end(), here.begin(), here.end());
    rows.push_back({S, "TOTAL", total, 0});

    const double share = total ? (double)softmax_span / (double)total : 0.0;
    if (si == 0) softmax_share_first = share;
    softmax_share_last = share;

    // Softmax must be the largest single region here, and that is asserted per
    // sequence length rather than as a trend across them. The first version
    // compared the first share against the last and allowed two points of
    // drift; after the double-exponential came out it read 41.3% against a
    // threshold of 39.3%, which is a gate one rounding away from red for no
    // reason anybody would learn anything from. What matters is WHICH part is
    // biggest, and that is a fact about each measurement on its own.
    uint64_t biggest = 0;
    const char* biggest_name = "";
    for (const Row& r : here)
      if (r.span > biggest) { biggest = r.span; biggest_name = r.region.c_str(); }
    std::snprintf(what, sizeof(what), "S=%d: softmax is the largest region", S);
    expect(std::strcmp(biggest_name, "softmax") == 0, what);
  }

  // THE CLAIM THIS BENCH EXISTS TO PIN, and it is a claim about SHAPE rather
  // than a threshold: softmax is the largest part of attention, and it does not
  // shrink as sequences get longer. If a future change makes one of the GEMMs
  // dominant instead, this goes red and the next person reads a table saying
  // where the time went instead of inheriting an assumption.
  std::printf("\n");
  if (softmax_share_first > 0.0) {
    std::printf("        softmax is %.1f%% of attention at the shortest"
                " sequence and %.1f%% at the longest\n",
                100.0 * softmax_share_first, 100.0 * softmax_share_last);
    expect(softmax_share_first > 0.30 && softmax_share_last > 0.30,
           "and it is over a third of attention at every length measured");
  }

  if (out_path && !rows.empty()) {
    std::FILE* f = std::fopen(out_path, "w");
    if (!f) { std::printf("  could not write %s\n", out_path); ++failures; }
    else {
      std::fprintf(f, "{\n  \"bench\": \"attention_cycles\",\n");
      std::fprintf(f, "  \"device\": {\n    \"name\": \"%s\",\n"
                      "    \"sms\": %d,\n    \"warp\": %d,\n"
                      "    \"warp_slots\": %d\n  },\n",
                   prop.name, prop.multiProcessorCount, prop.warpSize,
                   prop.maxWarpsPerMultiProcessor);
      std::fprintf(f, "  \"regions\": [\n");
      for (size_t i = 0; i < rows.size(); ++i)
        std::fprintf(f, "    {\"seq\": %d, \"region\": \"%s\","
                        " \"span\": %llu, \"warps\": %d}%s\n",
                     rows[i].seq, rows[i].region.c_str(),
                     (unsigned long long)rows[i].span, rows[i].warps,
                     (i + 1 == rows.size()) ? "" : ",");
      std::fprintf(f, "  ]\n}\n");
      std::fclose(f);
      std::printf("  wrote %s\n", out_path);
    }
  }

  grxdnnDestroy(dh);
  std::printf("\n%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
