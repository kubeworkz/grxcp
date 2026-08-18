// Phase 1 gate: the allocator, the interval map, and the copy/fill family.
//
// These verify real bytes moving through real offsets against the mock
// driver's memory model. What they cannot verify is concurrency -- the mock
// completes every enqueue before returning (ci/README.md, tier 1).

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdint>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  grxDeviceProp_t prop{};
  GRX_REQUIRE(grxGetDeviceProperties(&prop, 0), "device properties");

  // -------------------------------------------------------------------------
  section("allocation");
  // -------------------------------------------------------------------------
  {
    void* p = nullptr;
    GRX_REQUIRE(grxMalloc(&p, 4096), "grxMalloc 4 KiB");
    check(p != nullptr, "allocation is non-null");

    // Every allocation is aligned to at least the device cache line, so no two
    // allocations share a line. That is the mitigation for the FPGA DMA gap
    // (cuda_mapping.md 7.6) and it must hold on every backend.
    const uint64_t addr = (uint64_t)(uintptr_t)p;
    check(prop.cacheLineSize > 0 && addr % (uint64_t)prop.cacheLineSize == 0,
          "allocation is cache-line aligned");
    check(addr % 256 == 0, "allocation is at least 256-byte aligned");

    GRX_REQUIRE(grxFree(p), "grxFree");
    check(grxFree(nullptr) == grxSuccess, "freeing null is legal");

    void* zero = (void*)0x1234;
    check(grxMalloc(&zero, 0) == grxSuccess && zero == nullptr,
          "zero-size allocation yields null");
  }

  // -------------------------------------------------------------------------
  section("host <-> device round trip");
  // -------------------------------------------------------------------------
  {
    constexpr size_t N = 1024;
    std::vector<uint32_t> src(N), dst(N, 0);
    for (size_t i = 0; i < N; ++i) src[i] = (uint32_t)(i * 2654435761u);

    void* d = nullptr;
    GRX_REQUIRE(grxMalloc(&d, N * sizeof(uint32_t)), "allocate device buffer");
    GRX_REQUIRE(grxMemcpy(d, src.data(), N * sizeof(uint32_t), grxMemcpyDefault),
                "copy host to device");
    GRX_REQUIRE(grxMemcpy(dst.data(), d, N * sizeof(uint32_t), grxMemcpyDefault),
                "copy device to host");
    check(std::memcmp(src.data(), dst.data(), N * sizeof(uint32_t)) == 0,
          "round trip preserves every byte");

    // An interior pointer must resolve to the right buffer AND offset. This is
    // the whole point of the interval map: get the offset wrong and the data
    // lands in the wrong place with no error anywhere.
    const size_t half = N / 2;
    std::vector<uint32_t> tail(half, 0);
    GRX_REQUIRE(grxMemcpy(tail.data(), (uint8_t*)d + half * sizeof(uint32_t),
                          half * sizeof(uint32_t), grxMemcpyDefault),
                "copy from an interior device pointer");
    check(std::memcmp(tail.data(), src.data() + half,
                      half * sizeof(uint32_t)) == 0,
          "interior pointer resolves to the correct offset");

    void* d2 = nullptr;
    GRX_REQUIRE(grxMalloc(&d2, N * sizeof(uint32_t)), "allocate second buffer");
    GRX_REQUIRE(grxMemcpy(d2, d, N * sizeof(uint32_t), grxMemcpyDefault),
                "device to device copy");
    std::fill(dst.begin(), dst.end(), 0u);
    GRX_REQUIRE(grxMemcpy(dst.data(), d2, N * sizeof(uint32_t), grxMemcpyDefault),
                "read back the device-to-device result");
    check(std::memcmp(src.data(), dst.data(), N * sizeof(uint32_t)) == 0,
          "device to device copy preserves every byte");

    GRX_REQUIRE(grxMemset(d, 0xAB, N * sizeof(uint32_t)), "grxMemset");
    GRX_REQUIRE(grxMemcpy(dst.data(), d, N * sizeof(uint32_t), grxMemcpyDefault),
                "read back the memset result");
    bool all_ab = true;
    for (auto v : dst) all_ab = all_ab && (v == 0xABABABABu);
    check(all_ab, "memset writes the pattern across the whole range");

    grxFree(d);
    grxFree(d2);
  }

  // -------------------------------------------------------------------------
  section("direction validation");
  // -------------------------------------------------------------------------
  {
    // CUDA leaves a contradictory explicit kind undefined. GRXCP knows which
    // side is device memory, so it rejects instead of guessing.
    void* d = nullptr;
    GRX_REQUIRE(grxMalloc(&d, 256), "allocate for direction tests");
    std::vector<uint8_t> host(256, 0);

    check(grxMemcpy(d, host.data(), 256, grxMemcpyHostToDevice) == grxSuccess,
          "correct explicit kind is accepted");
    check(grxMemcpy(d, host.data(), 256, grxMemcpyDeviceToHost) ==
              grxErrorInvalidMemcpyDirection,
          "reversed explicit kind is rejected");
    check(grxMemcpy(host.data(), host.data(), 256, grxMemcpyDeviceToDevice) ==
              grxErrorInvalidMemcpyDirection,
          "device-to-device between host pointers is rejected");
    check(grxMemcpy(host.data(), host.data(), 256, grxMemcpyHostToHost) ==
              grxSuccess,
          "host-to-host is accepted");
    grxFree(d);
  }

  // -------------------------------------------------------------------------
  section("pointer attributes");
  // -------------------------------------------------------------------------
  {
    void* d = nullptr;
    GRX_REQUIRE(grxMalloc(&d, 1024), "allocate for attribute tests");
    grxPointerAttributes attr{};
    GRX_REQUIRE(grxPointerGetAttributes(&attr, d), "attributes of a device pointer");
    check(attr.type == grxMemoryTypeDevice, "device pointer reports device memory");
    check(attr.allocationSize >= 1024, "reported size covers the request");

    GRX_REQUIRE(grxPointerGetAttributes(&attr, (uint8_t*)d + 512),
                "attributes of an interior pointer");
    check(attr.type == grxMemoryTypeDevice, "interior pointer is still device memory");

    int stack_variable = 0;
    GRX_REQUIRE(grxPointerGetAttributes(&attr, &stack_variable),
                "attributes of an unregistered pointer");
    check(attr.type == grxMemoryTypeUnregistered,
          "unrelated host pointer reports unregistered");

    check(grxFree((uint8_t*)d + 256) == grxErrorInvalidDevicePointer,
          "freeing an interior pointer is rejected");
    grxFree(d);
  }

  // -------------------------------------------------------------------------
  section("pinned host memory");
  // -------------------------------------------------------------------------
  {
    void* h = nullptr;
    GRX_REQUIRE(grxMallocHost(&h, 4096), "grxMallocHost");
    check(h != nullptr, "pinned allocation is non-null");
    std::memset(h, 0x5A, 4096);   // must be host-dereferenceable

    grxPointerAttributes attr{};
    GRX_REQUIRE(grxPointerGetAttributes(&attr, h), "attributes of pinned memory");
    check(attr.type == grxMemoryTypeHost, "pinned pointer reports host memory");
    check(attr.hostPointer == h, "attributes echo the host pointer");
    GRX_REQUIRE(grxFreeHost(h), "grxFreeHost");
  }

  // -------------------------------------------------------------------------
  section("managed memory gating");
  // -------------------------------------------------------------------------
  {
    void* m = nullptr;
    const grxError_t e = grxMallocManaged(&m, 4096, 0);
    if (prop.managedMemory) {
      check(e == grxSuccess, "managed allocation succeeds where VM works");
      if (e == grxSuccess) {
        grxPointerAttributes attr{};
        grxPointerGetAttributes(&attr, m);
        check(attr.type == grxMemoryTypeManaged, "managed pointer reports managed");
        grxFree(m);
      }
    } else {
      // The FPGA backends have no hardware page-table walker yet, so a managed
      // pointer would be a lie. Refusal is the correct behavior.
      check(e == grxErrorNotSupported,
            "managed allocation is refused where the backend has no MMU");
    }
  }

  // -------------------------------------------------------------------------
  section("fragmentation and reuse");
  // -------------------------------------------------------------------------
  {
    // Allocate many blocks, free every other one, then allocate again into the
    // holes. The invariants: nothing overlaps, and data written before the
    // churn survives it.
    constexpr int kBlocks = 512;
    struct Block { uint8_t* ptr; size_t size; uint8_t tag; };
    std::vector<Block> blocks;

    for (int i = 0; i < kBlocks; ++i) {
      const size_t size = 64 + (size_t)(i % 17) * 96;
      void* p = nullptr;
      if (grxMalloc(&p, size) != grxSuccess) { check(false, "allocation churn"); break; }
      blocks.push_back({(uint8_t*)p, size, (uint8_t)(i & 0xFF)});
    }
    check(blocks.size() == kBlocks, "all churn allocations succeeded");

    bool overlap = false;
    for (size_t i = 0; i < blocks.size() && !overlap; ++i)
      for (size_t j = i + 1; j < blocks.size(); ++j) {
        const uint8_t* a = blocks[i].ptr; const uint8_t* b = blocks[j].ptr;
        if (a < b + blocks[j].size && b < a + blocks[i].size) { overlap = true; break; }
      }
    check(!overlap, "no two live allocations overlap");

    // Keep the even-indexed blocks, tag them, free the odd ones.
    std::vector<uint8_t> pattern(2048);
    for (size_t i = 0; i < blocks.size(); i += 2) {
      std::fill(pattern.begin(), pattern.begin() + blocks[i].size, blocks[i].tag);
      grxMemcpy(blocks[i].ptr, pattern.data(), blocks[i].size, grxMemcpyDefault);
    }
    for (size_t i = 1; i < blocks.size(); i += 2) grxFree(blocks[i].ptr);

    // Refill the holes.
    std::vector<void*> refill;
    for (int i = 0; i < kBlocks / 2; ++i) {
      void* p = nullptr;
      if (grxMalloc(&p, 128) == grxSuccess) refill.push_back(p);
    }
    check(refill.size() == (size_t)kBlocks / 2, "freed space is reusable");

    bool intact = true;
    std::vector<uint8_t> readback(2048);
    for (size_t i = 0; i < blocks.size() && intact; i += 2) {
      grxMemcpy(readback.data(), blocks[i].ptr, blocks[i].size, grxMemcpyDefault);
      for (size_t k = 0; k < blocks[i].size; ++k)
        if (readback[k] != blocks[i].tag) { intact = false; break; }
    }
    check(intact, "surviving allocations are untouched by the churn");

    for (void* p : refill) grxFree(p);
    for (size_t i = 0; i < blocks.size(); i += 2) grxFree(blocks[i].ptr);
  }

  // -------------------------------------------------------------------------
  section("memory info");
  // -------------------------------------------------------------------------
  {
    size_t freeBytes = 0, totalBytes = 0;
    GRX_REQUIRE(grxMemGetInfo(&freeBytes, &totalBytes), "grxMemGetInfo");
    check(totalBytes > 0, "total memory is reported");
    check(freeBytes <= totalBytes, "free memory does not exceed total");
  }

  return grxtest::report();
}
