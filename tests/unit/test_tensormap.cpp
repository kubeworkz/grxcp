// Tier-1 gate for the DXA descriptor encoder.
//
// Programming a tensor map is nothing but a sequence of device-config-register
// writes, so the mock driver records them and this test reads them back. That
// makes the encoding checkable with no device, no simulator and no toolchain --
// which matters more here than usual, because src/runtime/tensormap.cpp is a
// SECOND implementation of an encoding the sysroot also implements (see the
// note at the top of that file). A second implementation nobody checks is how
// two encoders drift apart.
//
// tests/kernels/dxa/ is the other half: it moves real data through a real
// descriptor. This half says the registers hold what they should; that half
// says the engine does what the registers say.

#include <grx/grx.h>

#include "grx_test.h"
#include "../mock/vortex_mock.h"

#include <VX_types.h>

#include <cstdio>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

// Find the value written to one descriptor register, or fall back to a
// sentinel the checks below will not match.
uint32_t written(unsigned slot, uint32_t off, bool* found = nullptr) {
  const uint32_t addr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE + off;
  const grxmock_dcr_record* recs = grxmock_dcr_writes();
  const uint32_t n = grxmock_dcr_count();
  for (uint32_t i = 0; i < n && i < GRXMOCK_MAX_DCR; ++i) {
    if (recs[i].addr == addr) {
      if (found) *found = true;
      return recs[i].value;
    }
  }
  if (found) *found = false;
  return 0xdeadbeefu;
}

bool wrote(unsigned slot, uint32_t off) {
  bool found = false;
  (void)written(slot, off, &found);
  return found;
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

  int slots = 0;
  check(grxTensorMapGetSlotCount(&slots, 0) == grxSuccess,
        "the slot count can be queried");
  if (slots == 0) {
    // A device without a DMA engine is a configuration, not a failure -- but
    // then the encoder cannot be exercised at all, so say so rather than
    // reporting a pass over nothing.
    std::printf("this device reports no DMA engine; nothing to encode\n");
    return grxtest::report();
  }

  // A tensor map may only describe memory the DMA engine can reach.
  void* buf = nullptr;
  if (grxMallocPhysical(&buf, 64 * 1024) != grxSuccess) return 1;

  // 12 x 6 elements of 4 bytes, stored 16 elements to a row, tiled 8 x 3.
  grxTensorMapDesc_t desc{};
  desc.slot           = 2;
  desc.base           = buf;
  desc.rank           = 2;
  desc.size[0]        = 12;
  desc.size[1]        = 6;
  desc.strideBytes[0] = 64;
  desc.tile[0]        = 8;
  desc.tile[1]        = 3;
  desc.elementBytes   = 4;
  desc.layout         = grxTensorMapLayoutRowMajor;

  section("what a rank-2 descriptor puts in the registers");
  grxmock_reset_dcr();
  check(grxTensorMapProgram(&desc) == grxSuccess, "a valid descriptor programs");
  check(grxmock_dcr_count() <= GRXMOCK_MAX_DCR,
        "the write sequence fits the capture buffer");

  const uint64_t base = (uint64_t)(uintptr_t)buf;
  check(written(2, VX_DCR_DXA_DESC_BASE_LO_OFF) == (uint32_t)(base & 0xffffffffu),
        "the low half of the base address");
  check(written(2, VX_DCR_DXA_DESC_BASE_HI_OFF) == (uint32_t)(base >> 32),
        "the high half of the base address");
  check(written(2, VX_DCR_DXA_DESC_SIZE0_OFF) == 12, "size along dim 0");
  check(written(2, VX_DCR_DXA_DESC_SIZE1_OFF) == 6,  "size along dim 1");
  check(written(2, VX_DCR_DXA_DESC_STRIDE0_OFF) == 64, "the row stride, in bytes");
  // TILESIZE01 packs two 16-bit extents; swapping them transposes every tile
  // the engine ever fetches, and nothing downstream would flag it.
  check(written(2, VX_DCR_DXA_DESC_TILESIZE01_OFF) == ((3u << 16) | 8u),
        "the tile extents pack as tile0 in the low half");
  check(written(2, VX_DCR_DXA_DESC_ESTRIDE0_OFF) == 1 &&
        written(2, VX_DCR_DXA_DESC_ESTRIDE1_OFF) == 1,
        "element strides are 1");
  check(written(2, VX_DCR_DXA_DESC_CFILL_OFF) == 0, "the out-of-bounds fill");

  // META: rank in bits 2:0, log2(element bytes) in 4:3, layout in 6:5.
  check(written(2, VX_DCR_DXA_DESC_META_OFF) == ((2u << 0) | (2u << 3) | (0u << 5)),
        "META carries rank, element size and layout");

  check(!wrote(2, VX_DCR_DXA_DESC_SMEM_STRIDE_OFF),
        "no multicast stride is written when none was asked for");

  section("the other slots stay untouched");
  {
    bool other = false;
    const grxmock_dcr_record* recs = grxmock_dcr_writes();
    const uint32_t n = grxmock_dcr_count();
    const uint32_t lo = VX_DCR_DXA_DESC_BASE + 2 * VX_DCR_DXA_DESC_STRIDE;
    const uint32_t hi = lo + VX_DCR_DXA_DESC_STRIDE;
    for (uint32_t i = 0; i < n && i < GRXMOCK_MAX_DCR; ++i)
      if (recs[i].addr < lo || recs[i].addr >= hi) other = true;
    check(!other, "every write lands inside the requested slot's window");
  }

  section("layout and rank change what is written");
  grxmock_reset_dcr();
  desc.layout = grxTensorMapLayoutKMajor;
  check(grxTensorMapProgram(&desc) == grxSuccess, "a k-major descriptor programs");
  check(written(2, VX_DCR_DXA_DESC_META_OFF) == ((2u << 0) | (2u << 3) | (1u << 5)),
        "k-major sets the layout field and nothing else in META");

  grxmock_reset_dcr();
  {
    grxTensorMapDesc_t d1 = desc;
    d1.layout = grxTensorMapLayoutRowMajor;
    d1.rank   = 1;
    d1.tile[0] = 8;
    check(grxTensorMapProgram(&d1) == grxSuccess, "a rank-1 descriptor programs");
    check(written(2, VX_DCR_DXA_DESC_META_OFF) == ((1u << 0) | (2u << 3)),
          "META says rank 1");
    check(!wrote(2, VX_DCR_DXA_DESC_SIZE1_OFF) &&
          !wrote(2, VX_DCR_DXA_DESC_STRIDE0_OFF),
          "a rank-1 descriptor leaves the dim-1 registers alone");
    check(written(2, VX_DCR_DXA_DESC_TILESIZE01_OFF) == 8u,
          "the unused tile extent packs as zero");
  }

  grxmock_reset_dcr();
  {
    grxTensorMapDesc_t d = desc;
    d.layout = grxTensorMapLayoutRowMajor;
    d.elementBytes = 2;
    d.strideBytes[0] = 32;
    check(grxTensorMapProgram(&d) == grxSuccess, "a 2-byte element descriptor programs");
    check(written(2, VX_DCR_DXA_DESC_META_OFF) == ((2u << 0) | (1u << 3)),
          "META encodes log2 of the element size, not the size");
  }

  section("multicast stride");
  grxmock_reset_dcr();
  {
    grxTensorMapDesc_t d = desc;
    d.layout = grxTensorMapLayoutRowMajor;
    d.smemStrideBytes = 512;
    check(grxTensorMapProgram(&d) == grxSuccess, "a multicast descriptor programs");
    check(written(2, VX_DCR_DXA_DESC_SMEM_STRIDE_OFF) == 512,
          "the per-CTA shared-memory stride is written");
  }

  section("what gets refused");
  {
    grxTensorMapDesc_t d = desc;
    d.layout = grxTensorMapLayoutRowMajor;

    grxTensorMapDesc_t bad = d; bad.slot = slots;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue, "a slot past the end");

    bad = d; bad.rank = 0;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue, "rank 0");

    bad = d; bad.rank = 6;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue, "rank 6");

    bad = d; bad.rank = 4;
    check(grxTensorMapProgram(&bad) == grxErrorNotSupported,
          "rank 4 -- refused, not half-programmed");

    bad = d; bad.elementBytes = 0;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue, "a zero element size");

    bad = d; bad.tile[1] = 0;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue, "a zero tile extent");

    bad = d; bad.tile[0] = 100000;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
          "a tile extent past 16 bits");

    bad = d; bad.strideBytes[0] = 8;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
          "a stride shorter than one row");

    bad = d; bad.size[1] = 1u << 20;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidValue,
          "an array bigger than its allocation");

    int stack = 0;
    bad = d; bad.base = &stack;
    check(grxTensorMapProgram(&bad) == grxErrorInvalidDevicePointer,
          "a host pointer");

    // On a device with an MMU, an ordinary allocation's address is not the
    // physical one the engine needs. Where there is no MMU the two are the
    // same and there is nothing to refuse.
    void* plain = nullptr;
    if (grxMalloc(&plain, 64 * 1024) == grxSuccess) {
      bad = d; bad.base = plain;
      const grxError_t got = grxTensorMapProgram(&bad);
      if (prop.unifiedAddressing) {
        check(got == grxErrorInvalidDevicePointer,
              "ordinary memory is refused on a device with virtual memory");
      } else {
        check(got == grxSuccess,
              "ordinary memory is fine where device addresses are physical");
      }
      grxFree(plain);
    }

    check(grxTensorMapProgram(nullptr) == grxErrorInvalidValue, "a null descriptor");
  }

  section("a refused descriptor writes nothing");
  grxmock_reset_dcr();
  {
    grxTensorMapDesc_t bad = desc;
    bad.rank = 4;
    grxTensorMapProgram(&bad);
    check(grxmock_dcr_count() == 0,
          "rejection happens before the first register is touched");
  }

  grxFree(buf);
  return grxtest::report();
}
