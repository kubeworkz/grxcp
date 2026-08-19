// Warp-primitive gate. Phase 2's exit gate asks for "a warp-reduction kernel
// using __shfl_down_sync producing correct results through the fallback" --
// and there is no fallback any more, because the ISA turned out to have the
// instructions. So this checks the real thing instead: the reduction, every
// shuffle form against CUDA's segmented semantics at two widths, the vote
// family, and shuffle beside the caller's shared memory.
//
// Everything is integer, so every comparison is exact.

#include <grx/grx.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

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

int failures = 0;
void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "warp.vxbin";

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  grxModule_t mod = nullptr;
  grxFunction_t reduce = nullptr, modes = nullptr, vote = nullptr,
                coexist = nullptr;
  CHECK(grxModuleLoad(&mod, image));
  CHECK(grxModuleGetFunction(&reduce, mod, "warp_reduce"));
  CHECK(grxModuleGetFunction(&modes, mod, "warp_modes"));
  CHECK(grxModuleGetFunction(&vote, mod, "warp_vote"));
  CHECK(grxModuleGetFunction(&coexist, mod, "warp_shared_coexist"));

  const unsigned W = (unsigned)prop.warpSize;
  const unsigned warps_per_block = 2;
  const unsigned block  = W * warps_per_block;
  const unsigned blocks = 2;
  const unsigned threads = block * blocks;
  const unsigned warps   = threads / W;

  std::printf("%s: warp %u, %u blocks x %u warps, shuffle is %s\n", prop.name, W,
              blocks, warps_per_block,
              prop.warpShuffleIsEmulated ? "EMULATED via local memory"
                                         : "native");

  std::vector<uint32_t> in(threads);
  for (unsigned i = 0; i < threads; ++i) in[i] = i + 1;   // 1..threads

  void *dIn = nullptr, *dOut = nullptr, *dAux = nullptr;
  CHECK(grxMalloc(&dIn, in.size() * sizeof(uint32_t)));
  CHECK(grxMalloc(&dOut, (size_t)threads * 4 * sizeof(uint32_t)));
  CHECK(grxMalloc(&dAux, (size_t)threads * sizeof(uint32_t)));
  CHECK(grxMemcpy(dIn, in.data(), in.size() * sizeof(uint32_t), grxMemcpyDefault));

  warp_args args{};
  args.in = (uint64_t)(uintptr_t)dIn;
  args.out = (uint64_t)(uintptr_t)dOut;
  args.aux = (uint64_t)(uintptr_t)dAux;
  args.threads = threads;
  args.width = W;

  auto launch = [&](grxFunction_t fn, size_t shared) -> grxError_t {
    grxError_t e = grxMemset(dOut, 0xff, (size_t)threads * 4 * sizeof(uint32_t));
    if (e != grxSuccess) return e;
    e = grxLaunchFunction(fn, dim3_t{blocks, 1, 1}, dim3_t{block, 1, 1}, &args,
                          sizeof(args), shared, nullptr);
    if (e != grxSuccess) return e;
    return grxDeviceSynchronize();
  };

  auto read_out = [&](unsigned n) {
    std::vector<uint32_t> v(n, 0);
    grxMemcpy(v.data(), dOut, (size_t)n * sizeof(uint32_t), grxMemcpyDefault);
    return v;
  };

  // --- the exit-gate kernel -----------------------------------------------
  std::printf("warp reduction through __shfl_down_sync:\n");
  CHECK(launch(reduce, 0));
  {
    const std::vector<uint32_t> got = read_out(warps);
    int bad = 0;
    for (unsigned w = 0; w < warps; ++w) {
      uint32_t want = 0;
      for (unsigned l = 0; l < W; ++l) want += in[w * W + l];
      if (got[w] != want) {
        if (bad < 4)
          std::printf("        warp %u got %u want %u\n", w, got[w], want);
        ++bad;
      }
    }
    expect(bad == 0, "every warp sums its lanes");
  }

  // --- the whole shuffle family, at two segment widths ---------------------
  for (unsigned width : {W, W / 2}) {
    if (width == 0) continue;
    args.width = width;
    CHECK(launch(modes, 0));
    const std::vector<uint32_t> got = read_out(threads * 4);

    int bad = 0;
    for (unsigned t = 0; t < threads; ++t) {
      const unsigned lane = t % W;
      const unsigned base = lane & ~(width - 1);
      const unsigned limit = base + width;
      const unsigned warp_base = t - lane;

      auto at = [&](unsigned l) { return in[warp_base + l]; };
      const uint32_t want_shfl = at(base + (0u % width));
      const uint32_t want_up   = (lane >= 1 && lane - 1 >= base) ? at(lane - 1) : at(lane);
      const uint32_t want_down = (lane + 1 < limit) ? at(lane + 1) : at(lane);
      const unsigned x = lane ^ 1u;
      const uint32_t want_xor  = (x < limit && x >= base) ? at(x) : at(lane);

      const uint32_t* g = got.data();
      if (g[0 * threads + t] != want_shfl) ++bad;
      if (g[1 * threads + t] != want_up)   ++bad;
      if (g[2 * threads + t] != want_down) ++bad;
      if (g[3 * threads + t] != want_xor)  ++bad;
    }
    char label[96];
    std::snprintf(label, sizeof(label),
                  "shfl / up / down / xor agree with CUDA semantics at width %u",
                  width);
    expect(bad == 0, label);
  }
  args.width = W;

  // --- vote ----------------------------------------------------------------
  std::printf("vote, which is native:\n");
  CHECK(launch(vote, 0));
  {
    const std::vector<uint32_t> got = read_out(warps * 4);
    const uint32_t full = (W >= 32) ? 0xffffffffu : ((1u << W) - 1u);
    uint32_t odd_mask = 0;
    for (unsigned l = 0; l < W; ++l) if (l & 1u) odd_mask |= (1u << l);

    int bad = 0;
    for (unsigned w = 0; w < warps; ++w) {
      const uint32_t* r = &got[w * 4];
      if (r[0] != full)     ++bad;
      if (r[1] != odd_mask) ++bad;
      if (r[2] != 1u)       ++bad;
      if (r[3] != 1u)       ++bad;
      if (bad && w == 0)
        std::printf("        warp 0: active 0x%x (want 0x%x), odd 0x%x (want "
                    "0x%x), any %u, all %u\n", r[0], full, r[1], odd_mask,
                    r[2], r[3]);
    }
    expect(bad == 0, "activemask, ballot, any and all report the full warp");
  }

  // --- shuffle and shared memory in one kernel -----------------------------
  std::printf("shuffle beside the caller's shared memory:\n");
  {
    const size_t shared = (size_t)block * sizeof(uint32_t);
    CHECK(launch(coexist, shared));
    const std::vector<uint32_t> got = read_out(threads);
    std::vector<uint32_t> aux(threads, 0);
    CHECK(grxMemcpy(aux.data(), dAux, aux.size() * sizeof(uint32_t),
                    grxMemcpyDefault));

    int bad_shuffle = 0, bad_shared = 0;
    for (unsigned t = 0; t < threads; ++t) {
      if (got[t] != 0xa000u) ++bad_shuffle;          // lane 0's value, broadcast
      if (aux[t] != 0xc0de0000u + t) ++bad_shared;   // the pattern, untouched
    }
    expect(bad_shuffle == 0, "the shuffle returns lane 0's value to every lane");
    expect(bad_shared == 0,
           "shared memory survives a shuffle -- the scratch is reserved below "
           "it, not on top of it");
  }

  CHECK(grxFree(dIn));
  CHECK(grxFree(dOut));
  CHECK(grxFree(dAux));
  CHECK(grxModuleUnload(mod));

  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED\n");
  return 0;
}
