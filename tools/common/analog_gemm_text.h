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

// A Q8.8 register as the number it stands for, to three places. Every sigma in
// the tile's map is in this format; printing the raw integer would make a
// reader do the division, and get it wrong for the one that is Q0.8.
inline std::string q8(int v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.3f", v / 256.0);
  return buf;
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
  char buf[200];
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
  // "of 16": a bit count means nothing without the width it is a count of. Six
  // bits of a sixteen-bit operand is not a six-bit quantiser for int8 data, it
  // is a quantiser that rounds all of it to zero.
  std::string of;
  if (a.operandBits > 0 && (a.activationBits > 0 || a.weightBits > 0)) {
    std::snprintf(buf, sizeof(buf), " of %d", a.operandBits);
    of = buf;
  }
  std::snprintf(buf, sizeof(buf), "seed 0x%08llx", (unsigned long long)a.seed);
  out.push_back(label + "EMULATED on the PTA tile: " + act + "/" + wgt + of +
                ", " + adc + ", " + buf);
  out.push_back(cont + "impairments " + impair_names(a.impairments));

  // The quantiser keeps the TOP of the operand word. An int8 operand lives in
  // the low eight bits, so a setting that keeps eight or more bits fewer than
  // the word has rounds every one of them to zero -- and int8 is the only
  // operand grxBLAS sends this device. A C of all zeros that is exactly what
  // was configured is worth one line before somebody debugs it as a fault.
  if ((a.impairments & GRX_ANALOG_QUANT) && a.operandBits > 0) {
    const bool act_dead = a.activationBits > 0 && a.operandBits - a.activationBits >= 8;
    const bool wgt_dead = a.weightBits > 0 && a.operandBits - a.weightBits >= 8;
    if (act_dead || wgt_dead) {
      std::snprintf(buf, sizeof(buf),
                    "%s of %d keeps the top %d bits: every int8 %s rounds to zero",
                    act_dead ? act.c_str() : wgt.c_str(), a.operandBits,
                    act_dead ? a.activationBits : a.weightBits,
                    act_dead && wgt_dead ? "operand"
                        : (act_dead ? "activation" : "weight"));
      out.push_back(cont + buf);
    }
  }

  // How much. All four are printed whether or not their impairment is enabled:
  // the line above says which of them the tile applies, and a sigma left over
  // from an earlier run is worth seeing before it is switched back on.
  out.push_back(cont + "noise: thermal " + q8(a.thermalSigmaQ8) + " LSB, shot k " +
                q8(a.shotCoefficientQ8) + ", programming " +
                q8(a.programmingSigmaQ8) + " LSB, crosstalk " + q8(a.crosstalkQ8));
  if (a.impairments & GRX_ANALOG_DRIFT) {
    std::snprintf(buf, sizeof(buf),
                  "drift: %s LSB a step, a step every 2^%d shots, clamp %s LSB",
                  q8(a.driftSigmaQ8).c_str(), a.driftLog2Shots,
                  q8(a.driftClampQ8).c_str());
    out.push_back(cont + buf);
    out.push_back(cont +
        "drift is device state: the answer depends on every shot since MODEL_RST");
  }
  if (a.tileRows > 0) {
    std::snprintf(buf, sizeof(buf), "tile %d x %d, %d-bit operands, %d-bit sums",
                  a.tileRows, a.tileCols, a.operandBits, a.accumulatorBits);
    out.push_back(cont + buf);
  }

  // The two states in which the reference model is the wrong model for this
  // device, even with every number above.
  if (a.loopModes > 0) {
    std::snprintf(buf, sizeof(buf),
                  "PTA_CTRL loop modes 0x%x are set: not the order the reference "
                  "model walks", a.loopModes);
    out.push_back(cont + buf);
  }
  if (a.calibrationValid > 0)
    out.push_back(cont +
        "a calibration's trims are in force, which the reference model does not hold");

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
