// Drive the GRX930 team's register model from a test, through the runtime.
//
// third_party/grx930/npu_dpi_shim.c answers on the c930's CSR map and keeps a
// 64 KB DDR array. Two small mismatches sit between it and our hooks, and both
// are the model's shape rather than a defect:
//
//   1. Its state machine only advances inside npu_dpi_run(), which no host
//      would ever call. So a STATUS read pumps it. That is not a fudge -- it
//      is the model's stand-in for the wall-clock a real poll loop burns, and
//      npu_c930_wait_idle polls STATUS exactly as it would on silicon.
//   2. Its DDR is a byte array behind a word-strobed write and a byte read,
//      while grxMemcpy wants a run of bytes. The conversion is here.
//
// A MODEL IS NOT HARDWARE. Anything green through this says our host drives
// the register map and moves bytes to the right offsets. It says nothing about
// the c930, and per AGENTS.md no result obtained here may be reported as the
// NPU working.

#ifndef GRXCP_TESTS_NPU_SHIM_ADAPTER_H
#define GRXCP_TESTS_NPU_SHIM_ADAPTER_H

#include "npu_c930.h"
#include "npu_c930_testing.h"

extern "C" {
#include "npu_dpi_shim.h"
}

#include <cstdint>
#include <cstring>

namespace grxtest {

// How many model cycles one STATUS read advances. Large enough that any shape
// the SoC accepts finishes inside npu_c930_wait_idle's timeout, small enough
// that the poll loop actually goes round.
inline int& npu_shim_quantum() {
  static int q = 64;
  return q;
}

inline uint32_t npu_shim_read(void*, uint32_t off) {
  if (off == 0x04) npu_dpi_run(npu_shim_quantum());
  return npu_dpi_csr_read(NPU_C930_MMIO_BASE + off);
}

inline void npu_shim_write(void*, uint32_t off, uint32_t v) {
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + off, v);
}

// The shim's mem_write takes a word address plus a byte strobe, so a run of
// bytes is written lane by lane. Slow and obvious, which is what a test wants.
inline int npu_shim_mem_write(void*, uint32_t addr, const void* src,
                              uint32_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(src);
  for (uint32_t i = 0; i < bytes; ++i) {
    const uint32_t a = addr + i;
    if (a >= (uint32_t)NPU_DDR_SIZE) return -1;
    const uint32_t word = a & ~3u, lane = a & 3u;
    npu_dpi_mem_write(word, ((uint32_t)p[i]) << (8 * lane), 1u << lane);
  }
  return 0;
}

inline int npu_shim_mem_read(void*, uint32_t addr, void* dst, uint32_t bytes) {
  uint8_t* p = static_cast<uint8_t*>(dst);
  for (uint32_t i = 0; i < bytes; ++i) {
    const uint32_t a = addr + i;
    if (a >= (uint32_t)NPU_DDR_SIZE) return -1;
    p[i] = (uint8_t)npu_dpi_mem_read(a);
  }
  return 0;
}

// Install both halves. MUST be called before the first grx call -- enumeration
// runs once and the seam refuses afterwards, deliberately. Returns true when
// both were accepted.
inline bool npu_shim_install() {
  npu_dpi_init();
  const int regs = grxcp_npu_attach_model_for_testing(npu_shim_read,
                                                      npu_shim_write, nullptr);
  const int mem  = grxcp_npu_attach_memory_for_testing(npu_shim_mem_read,
                                                       npu_shim_mem_write,
                                                       nullptr);
  return regs == 1 && mem == 1;
}

}  // namespace grxtest

#endif  // GRXCP_TESTS_NPU_SHIM_ADAPTER_H
