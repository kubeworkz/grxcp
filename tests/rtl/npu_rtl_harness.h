// THE SIXTH MODEL, AND THE FIRST THAT EXECUTES THE RTL.
//
// Five models have stood in for the c930 so far: four we wrote (ABSENT, DEAD,
// LIVE, WEDGED) and the GRX930 team's software shim. All five are models of the
// INTERFACE. This one is the design.
//
// Every port of c930_npu_top is top-level, so Verilator hands them to C++ as
// plain members and no DPI is needed at all. This file is the whole bridge: it
// drives the AXI4-Lite slave for CSR access and services the AXI4 master
// against a C++ DDR array, then hands both to the runtime through the seam in
// npu_c930_testing.h. No CPU, no firmware, no .hex, and no dependency on any
// file in the GRX930 tree that does not build.
//
// The memory stays in C++ deliberately. It is a model either way -- their
// c930_ddr_verilator.sv is one too -- so keeping it here means the only RTL in
// the picture is the accelerator, which is the thing under test.
//
// PARAMETERS MATTER AND ARE NOT THE DEFAULTS. c930_npu_top declares
// NUM_ROWS=8, NUM_COLS=8, MAX_M=64, MAX_K=256, MAX_N=8 -- the "core defaults"
// of cuda_mapping.md 7.26. c930_soc_top instantiates 4/4/8/16/12. Build with
// the defaults and you are testing a different machine than the one our
// NPU_C930_MAX_* constants describe, which is that gap in live form. The build
// passes -GNUM_ROWS=4 -GNUM_COLS=4 -GMAX_M=8 -GMAX_K=16 -GMAX_N=12.
//
// STILL NOT SILICON. A device driven this way reports GRX_BACKEND_RTLSIM, not
// GRX_BACKEND_MODEL and not GRX_BACKEND_SILICON -- the seam refuses to let
// anything call itself a chip. Per AGENTS.md no green run here may be reported
// as the NPU working; the phase 7 exit gate needs hardware.

#ifndef GRXCP_TESTS_NPU_RTL_HARNESS_H
#define GRXCP_TESTS_NPU_RTL_HARNESS_H

#include "Vc930_npu_top.h"
#include "verilated.h"

#include "npu_c930.h"
#include "npu_c930_testing.h"
#include <grx/grx_types.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace grxtest {

constexpr uint32_t kDdrBytes = 65536;

struct Rtl {
  Vc930_npu_top top;
  std::vector<uint8_t> ddr = std::vector<uint8_t>(kDdrBytes, 0);
  uint64_t cycles = 0;

  // --- AXI4 read channel state ---
  bool     r_active = false;
  uint32_t r_addr = 0;
  int      r_beats_left = 0;
  int      r_size_bytes = 8;

  // --- AXI4 write channel state ---
  bool     w_active = false;
  uint32_t w_addr = 0;
  bool     b_pending = false;

  void tick() {
    // Everything the master needs answered is computed from the state BEFORE
    // this edge, then applied; that is what makes this a synchronous slave
    // rather than a combinational shortcut the RTL could not have on a bus.
    top.i_clk = 0; top.eval();
    service_master();
    top.i_clk = 1; top.eval();
    ++cycles;
  }

  void service_master() {
    // ---- read address ----
    top.m_axi_arready = !r_active;
    if (top.m_axi_arvalid && !r_active) {
      r_active     = true;
      r_addr       = top.m_axi_araddr;
      r_beats_left = top.m_axi_arlen + 1;
      r_size_bytes = 1 << top.m_axi_arsize;
    }
    // ---- read data ----
    if (r_active) {
      uint64_t d = 0;
      for (int i = 0; i < 8; ++i) {
        const uint32_t a = r_addr + i;
        if (a < kDdrBytes) d |= (uint64_t)ddr[a] << (8 * i);
      }
      top.m_axi_rdata  = d;
      top.m_axi_rresp  = 0;
      top.m_axi_rvalid = 1;
      top.m_axi_rlast  = (r_beats_left == 1);
      if (top.m_axi_rready) {
        r_addr += r_size_bytes;
        if (--r_beats_left == 0) r_active = false;
      }
    } else {
      top.m_axi_rvalid = 0;
      top.m_axi_rlast  = 0;
    }

    // ---- write address ----
    top.m_axi_awready = !w_active;
    if (top.m_axi_awvalid && !w_active) {
      w_active = true;
      w_addr   = top.m_axi_awaddr;
    }
    // ---- write data ----
    top.m_axi_wready = w_active;
    if (w_active && top.m_axi_wvalid) {
      for (int i = 0; i < 8; ++i) {
        if (!((top.m_axi_wstrb >> i) & 1)) continue;
        const uint32_t a = w_addr + i;
        if (a < kDdrBytes) ddr[a] = (uint8_t)(top.m_axi_wdata >> (8 * i));
      }
      w_addr += 8;
      if (top.m_axi_wlast) { w_active = false; b_pending = true; }
    }
    // ---- write response ----
    top.m_axi_bvalid = b_pending;
    top.m_axi_bresp  = 0;
    if (b_pending && top.m_axi_bready) b_pending = false;
  }

  void reset() {
    top.i_rst_n = 0;
    top.s_axi_awvalid = top.s_axi_wvalid = top.s_axi_bready = 0;
    top.s_axi_arvalid = top.s_axi_rready = 0;
    top.m_axi_arready = top.m_axi_rvalid = top.m_axi_rlast = 0;
    top.m_axi_awready = top.m_axi_wready = top.m_axi_bvalid = 0;
    for (int i = 0; i < 16; ++i) tick();
    top.i_rst_n = 1;
    for (int i = 0; i < 4; ++i) tick();
  }

  // AXI4-Lite write: address and data together, then wait for BVALID.
  void csr_write(uint32_t off, uint32_t v) {
    top.s_axi_awaddr  = off;
    top.s_axi_awvalid = 1;
    top.s_axi_wdata   = v;
    top.s_axi_wstrb   = 0xF;
    top.s_axi_wvalid  = 1;
    top.s_axi_bready  = 1;
    bool aw = false, w = false, b = false;
    for (int i = 0; i < 64 && !(aw && w && b); ++i) {
      tick();
      if (top.s_axi_awready) { top.s_axi_awvalid = 0; aw = true; }
      if (top.s_axi_wready)  { top.s_axi_wvalid  = 0; w  = true; }
      if (top.s_axi_bvalid)  { b = true; }
    }
    top.s_axi_awvalid = top.s_axi_wvalid = 0;
    tick();
    top.s_axi_bready = 0;
  }

  uint32_t csr_read(uint32_t off) {
    top.s_axi_araddr  = off;
    top.s_axi_arvalid = 1;
    top.s_axi_rready  = 1;
    uint32_t got = 0;
    for (int i = 0; i < 64; ++i) {
      tick();
      if (top.s_axi_arready) top.s_axi_arvalid = 0;
      if (top.s_axi_rvalid) { got = top.s_axi_rdata; break; }
    }
    top.s_axi_arvalid = 0;
    tick();
    top.s_axi_rready = 0;
    return got;
  }
};

int32_t ref_gemm(const int8_t* A, const int8_t* B, int n, int k, int i, int j) {
  int32_t s = 0;
  for (int p = 0; p < k; ++p) s += (int32_t)A[i * k + p] * (int32_t)B[p * n + j];
  return s;
}

// ---------------------------------------------------------------------------
// The bridge to our hooks
// ---------------------------------------------------------------------------

inline Rtl& rtl() { static Rtl r; return r; }

inline uint32_t rtl_read(void*, uint32_t off)             { return rtl().csr_read(off); }
inline void     rtl_write(void*, uint32_t off, uint32_t v){ rtl().csr_write(off, v); }

inline int rtl_mem_write(void*, uint32_t addr, const void* src, uint32_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(src);
  if (addr >= kDdrBytes || n > kDdrBytes - addr) return -1;
  std::memcpy(rtl().ddr.data() + addr, p, n);
  return 0;
}

inline int rtl_mem_read(void*, uint32_t addr, void* dst, uint32_t n) {
  if (addr >= kDdrBytes || n > kDdrBytes - addr) return -1;
  std::memcpy(dst, rtl().ddr.data() + addr, n);
  return 0;
}

// Install the RTL as the enumerated device's model. MUST be called before the
// first grx call. Returns true when all three halves were accepted.
inline bool rtl_install() {
  rtl().reset();
  const int regs = grxcp_npu_attach_model_for_testing(rtl_read, rtl_write, nullptr);
  const int mem  = grxcp_npu_attach_memory_for_testing(rtl_mem_read, rtl_mem_write,
                                                       nullptr);
  const int kind = grxcp_npu_set_model_backend(GRX_BACKEND_RTLSIM);
  return regs == 1 && mem == 1 && kind == 1;
}

}  // namespace grxtest

#endif  // GRXCP_TESTS_NPU_RTL_HARNESS_H
