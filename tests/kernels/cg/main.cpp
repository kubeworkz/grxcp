// Cooperative-groups gate.
//
//   test_cg <image.vxbin>
//
// Every reference here is computed independently on the host from the input
// array, not by mirroring the device's expression. That is the difference
// between checking the kernel and checking that the kernel agrees with itself.

#include <grx/grx.h>

#include <cstdio>
#include <cstdlib>
#include <string>
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
  const char* image = (argc > 1) ? argv[1] : "cg.vxbin";

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  grxModule_t mod = nullptr;
  CHECK(grxModuleLoad(&mod, image));

  const unsigned W          = (unsigned)prop.warpSize;
  const unsigned warps_per_block = 2;
  const unsigned block      = W * warps_per_block;
  const unsigned blocks     = 2;
  const unsigned threads    = block * blocks;

  std::printf("%s: warp %u, %u blocks x %u threads, %d SM(s)\n", prop.name, W,
              blocks, block, prop.multiProcessorCount);

  std::vector<uint32_t> in(threads);
  for (unsigned i = 0; i < threads; ++i) in[i] = i + 1;   // 1..threads

  void *dIn = nullptr, *dOut = nullptr;
  CHECK(grxMalloc(&dIn, in.size() * sizeof(uint32_t)));
  CHECK(grxMalloc(&dOut, (size_t)CG_BANDS * threads * sizeof(uint32_t)));
  CHECK(grxMemcpy(dIn, in.data(), in.size() * sizeof(uint32_t), grxMemcpyDefault));

  cg_args args{};
  args.in      = (uint64_t)(uintptr_t)dIn;
  args.out     = (uint64_t)(uintptr_t)dOut;
  args.threads = threads;
  args.tile    = W;

  std::vector<uint32_t> got((size_t)CG_BANDS * threads);

  auto run = [&](const char* entry, unsigned nblocks, unsigned nthreads,
                 size_t shared, bool cooperative) -> grxError_t {
    grxFunction_t fn = nullptr;
    grxError_t e = grxModuleGetFunction(&fn, mod, entry);
    if (e != grxSuccess) return e;
    e = grxMemset(dOut, 0xff, got.size() * sizeof(uint32_t));
    if (e != grxSuccess) return e;
    e = cooperative
            ? grxLaunchCooperativeFunction(fn, dim3_t{nblocks, 1, 1},
                                           dim3_t{nthreads, 1, 1}, &args,
                                           sizeof(args), shared, nullptr)
            : grxLaunchFunction(fn, dim3_t{nblocks, 1, 1},
                                dim3_t{nthreads, 1, 1}, &args, sizeof(args),
                                shared, nullptr);
    if (e != grxSuccess) return e;
    e = grxDeviceSynchronize();
    if (e != grxSuccess) return e;
    return grxMemcpy(got.data(), dOut, got.size() * sizeof(uint32_t),
                     grxMemcpyDefault);
  };
  auto at = [&](unsigned b, unsigned t) { return got[(size_t)b * threads + t]; };

  // --- thread_block --------------------------------------------------------
  std::printf("thread_block:\n");
  CHECK(run("cg_block", blocks, block, (size_t)block * sizeof(uint32_t), false));
  {
    int bad_rank = 0, bad_size = 0, bad_sync = 0;
    for (unsigned t = 0; t < threads; ++t) {
      const unsigned rank = t % block;
      if (at(CG_BAND_RANK, t) != rank)  ++bad_rank;
      if (at(CG_BAND_SIZE, t) != block) ++bad_size;
      // The neighbour's input, which is only readable after the barrier.
      const unsigned base = t - rank;
      if (at(CG_BAND_REDUCE, t) != in[base + (rank + 1) % block]) ++bad_sync;
    }
    expect(bad_rank == 0, "thread_rank is the linear index within the block");
    expect(bad_size == 0, "num_threads is the block size");
    expect(bad_sync == 0, "sync() orders a shared write against a cross-warp read");
  }

  // --- thread_block_tile ---------------------------------------------------
  for (unsigned tile : {W, W / 2}) {
    if (tile == 0) continue;
    const char* entry = (tile == W) ? "cg_tile_full" : "cg_tile_half";
    grxFunction_t probe = nullptr;
    if (grxModuleGetFunction(&probe, mod, entry) != grxSuccess) continue;

    std::printf("thread_block_tile<%u>:\n", tile);
    CHECK(run(entry, blocks, block, 0, false));

    int bad_rank = 0, bad_size = 0, bad_red = 0, bad_inc = 0, bad_exc = 0,
        bad_shfl = 0, bad_ballot = 0, bad_meta = 0;
    for (unsigned t = 0; t < threads; ++t) {
      const unsigned lane = t % W;
      const unsigned rank = lane % tile;
      const unsigned seg  = t - rank;                  // first thread of the tile

      uint32_t sum = 0, inc = 0;
      for (unsigned i = 0; i < tile; ++i)  sum += in[seg + i];
      for (unsigned i = 0; i <= rank; ++i) inc += in[seg + i];
      const uint32_t exc = inc - in[t];

      uint32_t want_ballot = 0;
      for (unsigned i = 0; i < tile; ++i) if (i & 1u) want_ballot |= (1u << i);

      if (at(CG_BAND_RANK, t)      != rank)          ++bad_rank;
      if (at(CG_BAND_SIZE, t)      != tile)          ++bad_size;
      if (at(CG_BAND_REDUCE, t)    != sum)           ++bad_red;
      if (at(CG_BAND_INCLUSIVE, t) != inc)           ++bad_inc;
      if (at(CG_BAND_EXCLUSIVE, t) != (rank ? exc : 0u)) ++bad_exc;
      if (at(CG_BAND_SHFL, t)      != in[seg])       ++bad_shfl;
      if (at(CG_BAND_BALLOT, t)    != want_ballot)   ++bad_ballot;
      if (at(CG_BAND_META, t)      != lane / tile)   ++bad_meta;
    }
    expect(bad_rank == 0,   "thread_rank is the lane's position in the tile");
    expect(bad_size == 0,   "num_threads is the tile width");
    expect(bad_red == 0,    "reduce sums the tile, and every lane has it");
    expect(bad_inc == 0,    "inclusive_scan");
    expect(bad_exc == 0,    "exclusive_scan, with rank 0 getting the identity");
    expect(bad_shfl == 0,   "shfl broadcasts from the tile's own rank 0");
    expect(bad_ballot == 0, "ballot is over the tile, bit i = rank i");
    expect(bad_meta == 0,   "meta_group_rank identifies the tile in the warp");
  }

  // --- coalesced_group -----------------------------------------------------
  std::printf("coalesced_group, inside a divergent branch:\n");
  CHECK(run("cg_coalesced", blocks, block, 0, false));
  {
    const unsigned active = W / 2;   // the odd lanes
    int bad = 0;
    for (unsigned t = 0; t < threads; ++t) {
      const unsigned lane = t % W;
      if (!(lane & 1u)) {
        if (at(CG_BAND_RANK, t) != 0xffffffffu) ++bad;   // never entered
        continue;
      }
      if (at(CG_BAND_RANK, t) != lane / 2)   ++bad;      // dense rank
      if (at(CG_BAND_SIZE, t) != active)     ++bad;
      // rank 0 of the group is lane 1 of the warp.
      if (at(CG_BAND_SHFL, t) != in[t - lane + 1]) ++bad;
    }
    expect(bad == 0, "only the converged lanes are in the group, ranked densely");
  }

  // --- cluster_group -------------------------------------------------------
  std::printf("cluster_group:\n");
  CHECK(run("cg_cluster", blocks, block, (size_t)block * sizeof(uint32_t), false));
  {
    int bad = 0;
    for (unsigned t = 0; t < threads; ++t) {
      if (at(CG_BAND_RANK, t) != 0) ++bad;              // cluster of one
      if (at(CG_BAND_SIZE, t) != 1) ++bad;
      if (at(CG_BAND_SHFL, t) != in[t]) ++bad;
    }
    expect(bad == 0,
           "the cluster barrier completes and does not disturb shared memory");
  }

  // --- grid_group ----------------------------------------------------------
  std::printf("grid_group, through a cooperative launch:\n");
  {
    // The grid must fit AND cover every core; the runtime refuses anything
    // else, so ask it what fits rather than guessing.
    int per_sm = 0;
    CHECK(grxOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, nullptr,
                                                       (int)block, 0));
    const unsigned coop_blocks =
        (unsigned)(per_sm * prop.multiProcessorCount);
    const unsigned coop_threads = coop_blocks * block;
    std::printf("  grid of %u blocks x %u threads fills the machine\n",
                coop_blocks, block);

    args.threads = coop_threads;
    std::vector<uint32_t> cin(coop_threads);
    for (unsigned i = 0; i < coop_threads; ++i) cin[i] = i + 1;
    void* dIn2 = nullptr; void* dOut2 = nullptr;
    CHECK(grxMalloc(&dIn2, cin.size() * sizeof(uint32_t)));
    CHECK(grxMalloc(&dOut2, (size_t)CG_BANDS * coop_threads * sizeof(uint32_t)));
    CHECK(grxMemcpy(dIn2, cin.data(), cin.size() * sizeof(uint32_t),
                    grxMemcpyDefault));
    args.in  = (uint64_t)(uintptr_t)dIn2;
    args.out = (uint64_t)(uintptr_t)dOut2;

    grxFunction_t fn = nullptr;
    CHECK(grxModuleGetFunction(&fn, mod, "cg_grid"));
    CHECK(grxMemset(dOut2, 0xff,
                    (size_t)CG_BANDS * coop_threads * sizeof(uint32_t)));
    CHECK(grxLaunchCooperativeFunction(fn, dim3_t{coop_blocks, 1, 1},
                                       dim3_t{block, 1, 1}, &args, sizeof(args),
                                       0, nullptr));
    CHECK(grxDeviceSynchronize());

    std::vector<uint32_t> g((size_t)CG_BANDS * coop_threads);
    CHECK(grxMemcpy(g.data(), dOut2, g.size() * sizeof(uint32_t),
                    grxMemcpyDefault));
    auto at2 = [&](unsigned b, unsigned t) { return g[(size_t)b * coop_threads + t]; };

    int bad_peer = 0, bad_rank = 0, bad_size = 0;
    for (unsigned t = 0; t < coop_threads; ++t) {
      const unsigned peer = (t + block) % coop_threads;
      if (at2(CG_BAND_SHFL, t) != cin[peer] + 1u) ++bad_peer;
      if (at2(CG_BAND_RANK, t) != t)              ++bad_rank;
      if (at2(CG_BAND_SIZE, t) != coop_blocks)    ++bad_size;
    }
    expect(bad_peer == 0,
           "after grid.sync() every thread sees another block's writes");
    expect(bad_rank == 0, "grid thread_rank is the linear index in the grid");
    expect(bad_size == 0, "grid num_blocks is the grid size");

    // The control. The same kernel without the barrier must get this wrong --
    // block 0 stalls before publishing, so a block that does not wait reads the
    // sentinel. If this ever comes back correct, the test above it proved
    // nothing and the stall needs to be longer.
    grxFunction_t nosync = nullptr;
    CHECK(grxModuleGetFunction(&nosync, mod, "cg_grid_nosync"));
    CHECK(grxMemset(dOut2, 0xff,
                    (size_t)CG_BANDS * coop_threads * sizeof(uint32_t)));
    CHECK(grxLaunchCooperativeFunction(nosync, dim3_t{coop_blocks, 1, 1},
                                       dim3_t{block, 1, 1}, &args, sizeof(args),
                                       0, nullptr));
    CHECK(grxDeviceSynchronize());
    CHECK(grxMemcpy(g.data(), dOut2, g.size() * sizeof(uint32_t),
                    grxMemcpyDefault));
    int unsynced_wrong = 0;
    for (unsigned t = 0; t < coop_threads; ++t) {
      const unsigned peer = (t + block) % coop_threads;
      if (at2(CG_BAND_SHFL, t) != cin[peer] + 1u) ++unsynced_wrong;
    }
    expect(unsynced_wrong > 0,
           "without grid.sync() a block reads a value that is not published "
           "yet -- so the check above tests the barrier, not luck");

    // A cooperative launch too large to be resident must be refused, not
    // dispatched into a hang.
    const grxError_t too_big = grxLaunchCooperativeFunction(
        fn, dim3_t{coop_blocks * 4 + 1, 1, 1}, dim3_t{block, 1, 1}, &args,
        sizeof(args), 0, nullptr);
    expect(too_big == grxErrorLaunchOutOfResources,
           "a cooperative grid that cannot be resident is refused");
    (void)grxGetLastError();

    CHECK(grxFree(dIn2));
    CHECK(grxFree(dOut2));
  }

  CHECK(grxFree(dIn));
  CHECK(grxFree(dOut));
  CHECK(grxModuleUnload(mod));

  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED\n");
  return 0;
}
