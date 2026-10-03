// grxDeviceProp_t.analogGemm, through the front door.
//
// A GRX930 built with a photonic tile does not compute the GEMM it is asked
// for once an impairment is enabled; it computes a noisy approximation of it.
// AGENTS.md section 3 requires every such stand-in to be reported through a
// device property, and this is that property, checked where a user reads it:
// grxGetDeviceProperties on the device the RUNTIME enumerated.
//
// The backend's decision has its own test, against six register files
// (src/backends/npu_c930/test_npu_c930_model.cc). What that cannot reach is the
// part that was wrong here. Everything else in grxDeviceProp_t is established
// once, at the first acquire, and analogGemm was populated the same way -- so
// a tile impaired AFTER the first grxGetDeviceProperties went on being reported
// native for the life of the process. The registers it is read from are
// ordinary read-write state. A property that describes them has to be read
// when it is asked for.
//
// That is why this file is one process walking one register model through
// every state. Enumeration happens once and the seam allows one model, so the
// only way to see a second state is for the property to follow the registers
// -- and with the cached read every section after the first fails.
//
// A MODEL IS NOT HARDWARE. This checks what the runtime reports about a
// register file. It says nothing about a c930.

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdio>
#include <cstring>

#ifdef GRXCP_ENABLE_NPU
#include "npu_c930_testing.h"
#endif

using grxtest::check;
using grxtest::section;

#ifdef GRXCP_ENABLE_NPU
namespace {

// The register file c930_npu_csr.sv decodes since the PTA block was added: 256
// words, with PTA_ID and PTA_CAPS1 read-only. Both are plain members so each
// section below can change what the "build" is.
struct Regs {
  uint32_t r[256] = {0};
  uint32_t id     = 0;      // starts as a register file with no identity word
  uint32_t built  = 0;
};

uint32_t regs_read(void* ctx, uint32_t off) {
  Regs* g = static_cast<Regs*>(ctx);
  if (off == NPU_C930_PTA_ID)    return g->id;
  if (off == NPU_C930_PTA_CAPS1) return g->built;
  return g->r[(off >> 2) & 0xFF];
}

void regs_write(void* ctx, uint32_t off, uint32_t v) {
  Regs* g = static_cast<Regs*>(ctx);
  if (off == NPU_C930_PTA_ID || off == NPU_C930_PTA_CAPS1) return;
  if (off == 0x00 && (v & NPU_C930_CTRL_START)) {
    g->r[1] = NPU_C930_STATUS_DONE;
    return;
  }
  g->r[(off >> 2) & 0xFF] = v;
}

Regs g_regs;

void set(uint32_t off, uint32_t v) { g_regs.r[(off >> 2) & 0xFF] = v; }

const uint32_t kMagic     = 0x50544101u;
const uint32_t kTileBuilt = NPU_C930_PTA_DEFINED & ~NPU_C930_PTA_MZM_NL;

bool model_unreported(const grxAnalogGemm_t& a) {
  return a.activationBits == -1 && a.weightBits == -1 && a.adcBits == -1 &&
         a.adcShift == -1 && a.seed == -1 && a.impairments == -1;
}

grxAnalogGemm_t read(int device) {
  grxDeviceProp_t p{};
  grxGetDeviceProperties(&p, device);
  return p.analogGemm;
}

}  // namespace
#endif

int main() {
#ifndef GRXCP_ENABLE_NPU
  std::printf("built without GRXCP_ENABLE_NPU; there is no NPU backend in this "
              "binary. skipping\n");
  return 77;
#else
  // BEFORE ANY grx CALL: enumeration runs once and the seam refuses afterwards.
  if (grxcp_npu_attach_model_for_testing(regs_read, regs_write, &g_regs) != 1) {
    std::printf("the register model could not be installed; skipping\n");
    return 77;
  }

  int count = 0;
  GRX_REQUIRE(grxGetDeviceCount(&count), "grxGetDeviceCount");
  int npu = -1;
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) == grxSuccess &&
        p.deviceType == GRX_DEVICE_TYPE_NPU) { npu = i; break; }
  }
  section("an NPU reached through a register model");
  check(npu >= 0, "it is enumerated");
  if (npu < 0) return grxtest::report();

  // ---- 1. a register file with no identity word -------------------------
  // This is the device as it was probed, so it is also the one state a cached
  // property could report correctly. Everything after it is a second state.
  section("no PTA identity word");
  {
    const grxAnalogGemm_t a = read(npu);
    check(a.gemmIsAnalogEmulated == -1 && a.tileIsPresent == -1,
          "UNKNOWN in both leading fields -- not 0, which would claim exactness");
    check(model_unreported(a) && a.impairmentsImplemented == -1,
          "and every other field unknown too");
  }

  // ---- 2. the block appears, on a build with no tile --------------------
  section("identified, nothing built");
  g_regs.id    = kMagic;
  g_regs.built = 0;
  {
    const grxAnalogGemm_t a = read(npu);
    check(a.tileIsPresent == 0,
          "tileIsPresent 0 -- the property followed the registers");
    if (a.tileIsPresent != 0)
      std::printf("        Still reporting the state it was probed in. The\n"
                  "        property is being served from a cache, and every\n"
                  "        section below will report the wrong device.\n");
    check(a.gemmIsAnalogEmulated == 0, "GEMMs are exact");
    check(a.impairmentsImplemented == 0 && model_unreported(a),
          "nothing implemented is 0, and no model field is reported");
  }
  set(NPU_C930_PTA_IMPAIR, NPU_C930_PTA_SHOT);
  {
    const grxAnalogGemm_t a = read(npu);
    check(a.gemmIsAnalogEmulated == 0 && a.tileIsPresent == 0 &&
          model_unreported(a),
          "PTA_IMPAIR written on an array does not make its GEMMs analog");
  }
  set(NPU_C930_PTA_IMPAIR, 0);

  // ---- 3. a tile, nothing enabled ---------------------------------------
  section("a tile, PTA_IMPAIR clear");
  g_regs.built = kTileBuilt;
  set(NPU_C930_PTA_BITS, 0x00003688u);   // stale, from whoever was here last
  set(NPU_C930_PTA_SEED, 0x77u);
  set(NPU_C930_PTA_CTRL, 0x1u);          // EN: the calibration engine's
  {
    const grxAnalogGemm_t a = read(npu);
    check(a.tileIsPresent == 1, "tileIsPresent 1");
    check(a.gemmIsAnalogEmulated == 0,
          "GEMMs are exact, with PTA_CTRL.EN set: EN is not what makes them analog");
    check(a.impairmentsImplemented == (int64_t)kTileBuilt,
          "what the tile can model is reported: 0x5f, all but MZM_NL");
    check(model_unreported(a),
          "the stale BITS and SEED are not reported for a model that is not running");
  }

  // ---- 4. the tile, impaired, with EN clear -----------------------------
  section("a tile, PTA_IMPAIR set, PTA_CTRL.EN clear");
  set(NPU_C930_PTA_CTRL, 0x0u);
  set(NPU_C930_PTA_IMPAIR, kTileBuilt);
  set(NPU_C930_PTA_BITS, 8u | (8u << 4) | (6u << 8) | (3u << 12));
  set(NPU_C930_PTA_SEED, 0x2au);
  {
    const grxAnalogGemm_t a = read(npu);
    check(a.gemmIsAnalogEmulated == 1,
          "EMULATED -- a GEMM here is not the GEMM that was asked for");
    if (a.gemmIsAnalogEmulated != 1)
      std::printf("        A noisy GEMM is being reported as native. That is\n"
                  "        the failure this property exists to prevent.\n");
    check(a.tileIsPresent == 1, "tile present");
    check(a.activationBits == 8 && a.weightBits == 8 && a.adcBits == 6 &&
          a.adcShift == 3, "a8, w8, ADC 6 bits << 3");
    check(a.seed == 0x2a, "seed 0x2a -- what makes the answer reproducible");
    check(a.impairments == (int64_t)kTileBuilt,
          "impairments QUANT|THERMAL|SHOT|DRIFT|XTALK|PROG_ERR");
    check((a.impairments & ~a.impairmentsImplemented) == 0,
          "none of them outside what the tile implements");
  }

  // ---- 5. an enable the tile does not have ------------------------------
  section("PTA_IMPAIR asks for MZM_NL");
  set(NPU_C930_PTA_IMPAIR, NPU_C930_PTA_QUANT | NPU_C930_PTA_MZM_NL);
  {
    const grxAnalogGemm_t a = read(npu);
    check((a.impairments & ~a.impairmentsImplemented) == GRX_ANALOG_MZM_NL,
          "impairments & ~impairmentsImplemented names the bit that will refuse");
  }

  // ---- 6. and back ------------------------------------------------------
  section("PTA_IMPAIR cleared again");
  set(NPU_C930_PTA_IMPAIR, 0);
  {
    const grxAnalogGemm_t a = read(npu);
    check(a.gemmIsAnalogEmulated == 0 && a.tileIsPresent == 1 &&
          model_unreported(a),
          "native again, and the model's fields withdrawn with it");
  }

  // ---- 7. the property is the device's, not the process's ---------------
  // Device 0 is a GPU on every configuration this runs in. Whatever the NPU's
  // registers say, the GPU's GEMMs are its own.
  section("the GPU beside it");
  set(NPU_C930_PTA_IMPAIR, kTileBuilt);
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) != grxSuccess) continue;
    if (p.deviceType != GRX_DEVICE_TYPE_GPU) continue;
    check(p.analogGemm.gemmIsAnalogEmulated == 0 &&
          p.analogGemm.tileIsPresent == 0,
          "reports native and no tile, while the NPU reports emulated");
    break;
  }
  check(read(npu).gemmIsAnalogEmulated == 1,
        "and the NPU still reports emulated");

  return grxtest::report();
#endif
}
