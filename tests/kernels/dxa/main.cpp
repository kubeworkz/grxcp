// DXA gate: program a tensor map, have a kernel stage a tile through the DMA
// engine, and check every element of what landed.
//
// The source array is deliberately PADDED -- the described row is narrower
// than the allocated row -- because a descriptor that ignores the stride and
// assumes tight packing passes a tightly packed test and fails on the first
// real sub-matrix. Values are row*1000 + col, so a misplaced element names
// the place it actually came from.

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

// Described array: 12 x 6 elements, stored with 16 elements per row.
constexpr unsigned kCols       = 12;
constexpr unsigned kRows       = 6;
constexpr unsigned kRowStride  = 16;      // elements actually allocated per row
constexpr unsigned kTile0      = 8;
constexpr unsigned kTile1      = 3;
constexpr unsigned kCoord0     = 4;
constexpr unsigned kCoord1     = 2;
constexpr unsigned kElemBytes  = sizeof(uint32_t);

int failures = 0;

void expect(bool ok, const char* what) {
  std::printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
  if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "dxa.vxbin";

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  int slots = 0;
  CHECK(grxTensorMapGetSlotCount(&slots, 0));
  if (slots == 0) {
    std::printf("SKIPPED: %s reports no DMA engine\n", prop.name);
    return 77;
  }
  std::printf("%s: %d descriptor slots, %d barriers, warp %d\n", prop.name,
              slots, prop.numBarriers, prop.warpSize);

  if ((size_t)kTile0 * kTile1 * kElemBytes > prop.sharedMemPerBlock) {
    std::printf("SKIPPED: the tile does not fit this device's %zu-byte "
                "shared-memory slot\n", prop.sharedMemPerBlock);
    return 77;
  }

  grxModule_t   mod = nullptr;
  grxFunction_t fn  = nullptr;
  CHECK(grxModuleLoad(&mod, image));
  CHECK(grxModuleGetFunction(&fn, mod, "dxa_copy_tile"));

  // --- source array -------------------------------------------------------
  std::vector<uint32_t> src((size_t)kRowStride * kRows, 0xffffffffu);
  for (unsigned r = 0; r < kRows; ++r)
    for (unsigned c = 0; c < kCols; ++c)
      src[(size_t)r * kRowStride + c] = r * 1000u + c;

  const unsigned tile_elems = kTile0 * kTile1;
  void *dSrc = nullptr, *dOut = nullptr;
  // The source is what the engine reads, so it has to be reachable by it.
  CHECK(grxMallocPhysical(&dSrc, src.size() * kElemBytes));
  CHECK(grxMalloc(&dOut, tile_elems * kElemBytes));
  CHECK(grxMemcpy(dSrc, src.data(), src.size() * kElemBytes, grxMemcpyDefault));

  grxTensorMapDesc_t desc{};
  desc.slot            = 0;
  desc.base            = dSrc;
  desc.rank            = 2;
  desc.size[0]         = kCols;
  desc.size[1]         = kRows;
  desc.strideBytes[0]  = kRowStride * kElemBytes;
  desc.tile[0]         = kTile0;
  desc.tile[1]         = kTile1;
  desc.elementBytes    = kElemBytes;
  desc.layout          = grxTensorMapLayoutRowMajor;

  auto run = [&](grxTensorMapLayout_t layout, unsigned warps,
                 std::vector<uint32_t>& got) -> grxError_t {
    desc.layout = layout;
    grxError_t e = grxTensorMapProgram(&desc);
    if (e != grxSuccess) return e;

    std::vector<uint32_t> poison(tile_elems, 0x5a5a5a5au);
    e = grxMemcpy(dOut, poison.data(), poison.size() * kElemBytes,
                  grxMemcpyDefault);
    if (e != grxSuccess) return e;

    dxa_args args{};
    args.out     = (uint64_t)(uintptr_t)dOut;
    args.slot    = (uint32_t)desc.slot;
    args.coord0  = kCoord0;
    args.coord1  = kCoord1;
    args.tile0   = kTile0;
    args.tile1   = kTile1;
    args.barrier = 0;

    const unsigned block = (unsigned)prop.warpSize * warps;
    // The staging buffer is dynamic shared memory: the kernel takes the CTA's
    // slot base from a CSR, and this is where its size is decided.
    const size_t smem_bytes = (size_t)tile_elems * kElemBytes;
    e = grxLaunchFunction(fn, dim3_t{1, 1, 1}, dim3_t{block, 1, 1}, &args,
                          sizeof(args), smem_bytes, nullptr);
    if (e != grxSuccess) return e;
    e = grxDeviceSynchronize();
    if (e != grxSuccess) return e;

    got.assign(tile_elems, 0);
    return grxMemcpy(got.data(), dOut, tile_elems * kElemBytes,
                     grxMemcpyDefault);
  };

  auto source_at = [&](unsigned i0, unsigned i1) {
    return src[(size_t)(kCoord1 + i1) * kRowStride + (kCoord0 + i0)];
  };

  std::printf("staging a %ux%u tile at (%u,%u) of a %ux%u array, row stride %u\n",
              kTile0, kTile1, kCoord0, kCoord1, kCols, kRows, kRowStride);

  // --- row-major destination ----------------------------------------------
  for (unsigned warps : {1u, 2u}) {
    std::vector<uint32_t> got;
    const grxError_t e = run(grxTensorMapLayoutRowMajor, warps, got);
    if (e != grxSuccess) {
      std::printf("  FAIL  row-major, %u warp(s): %s\n", warps,
                  grxGetErrorString(e));
      ++failures;
      continue;
    }
    int bad = 0;
    for (unsigned i1 = 0; i1 < kTile1; ++i1) {
      for (unsigned i0 = 0; i0 < kTile0; ++i0) {
        const uint32_t want = source_at(i0, i1);
        const uint32_t have = got[(size_t)i1 * kTile0 + i0];
        if (have != want) {
          if (bad < 4)
            std::printf("        (%u,%u) got %u want %u\n", i0, i1, have, want);
          ++bad;
        }
      }
    }
    char label[96];
    std::snprintf(label, sizeof(label),
                  "row-major tile lands element for element (%u warp%s)",
                  warps, warps == 1 ? "" : "s");
    expect(bad == 0, label);
  }

  // --- k-major (transposing) destination -----------------------------------
  {
    std::vector<uint32_t> got;
    const grxError_t e = run(grxTensorMapLayoutKMajor, 1, got);
    if (e != grxSuccess) {
      std::printf("  FAIL  k-major: %s\n", grxGetErrorString(e));
      ++failures;
    } else {
      int bad = 0;
      for (unsigned i1 = 0; i1 < kTile1; ++i1) {
        for (unsigned i0 = 0; i0 < kTile0; ++i0) {
          const uint32_t want = source_at(i0, i1);
          const uint32_t have = got[(size_t)i0 * kTile1 + i1];
          if (have != want) {
            if (bad < 4)
              std::printf("        (%u,%u) got %u want %u\n", i0, i1, have, want);
            ++bad;
          }
        }
      }
      expect(bad == 0, "k-major tile arrives transposed, as documented");
    }
  }

  // --- the descriptor validation, which is most of what protects a caller ---
  std::printf("descriptor validation:\n");
  {
    grxTensorMapDesc_t bad = desc;
    bad.slot = slots;
    expect(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
           "a slot past the last one is rejected");

    bad = desc; bad.rank = 3;
    expect(grxTensorMapProgram(&bad) == grxErrorNotSupported,
           "rank 3 is refused rather than half-programmed");

    bad = desc; bad.elementBytes = 3;
    expect(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
           "a non-power-of-two element size is rejected");

    bad = desc; bad.tile[0] = kCols + 1;
    expect(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
           "a tile wider than the array is rejected");

    bad = desc; bad.strideBytes[0] = kElemBytes;   // narrower than one row
    expect(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
           "a stride shorter than a row is rejected");

    bad = desc; bad.size[1] = kRows * 64;          // describes past the end
    expect(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
           "an array larger than its allocation is rejected");

    int one = 0;
    bad = desc; bad.base = &one;                   // a host pointer
    expect(grxTensorMapProgram(&bad) == grxErrorInvalidDevicePointer,
           "a host pointer is rejected");
  }

  CHECK(grxFree(dSrc));
  CHECK(grxFree(dOut));
  CHECK(grxModuleUnload(mod));

  if (failures) {
    std::printf("FAILED (%d)\n", failures);
    return 1;
  }
  std::printf("PASSED\n");
  return 0;
}
