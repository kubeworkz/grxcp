// The lines grx-smi prints for grxDeviceProp_t.analogGemm.
//
// In a header, and not in grx-smi's main.cpp, for one reason: three of the four
// forms below can only be produced by a c930 with a photonic tile, and there is
// none in CI. Left inside the tool, the only line anything ever executed was
// "native (no PTA tile in this build)" -- the EMULATED and UNKNOWN forms, which
// are the two a user most needs to be right, would have shipped unread.
// tests/unit/test_analog_gemm_text.cpp holds each form to its exact text.
//
// The layout is docs/designs/pta_cpu_integration.md section 7.1: the stand-ins
// section's 23-column label field, so the value starts at column 28 and a
// continuation line is 27 spaces.

#ifndef GRXCP_TOOLS_ANALOG_GEMM_TEXT_H
#define GRXCP_TOOLS_ANALOG_GEMM_TEXT_H

#include <grx/grx_types.h>

#include <cstdio>
#include <string>
#include <vector>

namespace grxtools {

// The impairment mask as names, in the register's bit order.
inline std::string impair_names(int64_t mask) {
  static const char* const kNames[7] = {"QUANT", "THERMAL", "SHOT", "DRIFT",
                                        "XTALK", "MZM_NL", "PROG_ERR"};
  std::string s;
  for (int b = 0; b < 7; ++b) {
    if (!(mask & (int64_t(1) << b))) continue;
    if (!s.empty()) s += "|";
    s += kNames[b];
  }
  return s.empty() ? "none" : s;
}

// One string per printed line, without the newline.
//
// UNKNOWN is as loud as EMULATED. A device that cannot say whether its GEMMs
// are exact is not a device whose GEMMs are exact, and a quiet line here would
// be read as one.
inline std::vector<std::string> analog_gemm_lines(const grxAnalogGemm_t& a) {
  const std::string label = "    analog GEMM            ";
  const std::string cont  = "                           ";
  std::vector<std::string> out;

  if (a.gemmIsAnalogEmulated < 0 || a.tileIsPresent < 0) {
    out.push_back(label +
        "UNKNOWN: this device's register file has no PTA identity word");
    out.push_back(cont +
        "so whether a GEMM here is exact cannot be read from it");
    return out;
  }
  if (!a.tileIsPresent) {
    out.push_back(label + "native (no PTA tile in this build)");
    return out;
  }
  if (!a.gemmIsAnalogEmulated) {
    out.push_back(label + "native (PTA tile present, PTA_IMPAIR clear)");
    return out;
  }

  // Zero bits is a real setting -- unquantised -- and "a0" would read as a
  // zero-bit activation, so it is spelled out.
  char buf[160];
  std::string act = "a=full", wgt = "w=full", adc = "ADC unquantised";
  if (a.activationBits > 0) {
    std::snprintf(buf, sizeof(buf), "a%d", a.activationBits);
    act = buf;
  }
  if (a.weightBits > 0) {
    std::snprintf(buf, sizeof(buf), "w%d", a.weightBits);
    wgt = buf;
  }
  if (a.adcBits > 0) {
    std::snprintf(buf, sizeof(buf), "ADC %d bits << %d", a.adcBits, a.adcShift);
    adc = buf;
  }
  std::snprintf(buf, sizeof(buf), "seed 0x%08llx", (unsigned long long)a.seed);
  out.push_back(label + "EMULATED on the PTA tile: " + act + "/" + wgt + ", " +
                adc + ", " + buf);
  out.push_back(cont + "impairments " + impair_names(a.impairments));

  // A bit that is off and a bit that cannot be on look the same in the mask
  // above, and a START asking for the second kind is refused outright. If one
  // of those is actually ENABLED the device will run nothing, which outranks
  // a note about what it could not do if asked.
  const int64_t unbuilt = GRX_ANALOG_DEFINED & ~a.impairmentsImplemented;
  const int64_t refused = a.impairments & ~a.impairmentsImplemented;
  if (refused)
    out.push_back(cont + impair_names(refused) +
                  " is ENABLED and not implemented: every START is refused");
  else if (unbuilt)
    out.push_back(cont + impair_names(unbuilt) +
                  " is defined but not implemented; a START with it is refused");
  out.push_back(cont +
      "this device does NOT compute the same function as a digital c930");
  return out;
}

}  // namespace grxtools

#endif  // GRXCP_TOOLS_ANALOG_GEMM_TEXT_H
