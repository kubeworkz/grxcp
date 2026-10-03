// What grx-smi prints for grxDeviceProp_t.analogGemm, held to its exact text.
//
// grx-smi is where a user finds out that the numbers in front of them came
// from a noisy approximation of the GEMM they asked for. Three of the four
// forms it can print need a c930 with a photonic tile to produce, and CI has
// none, so without this file the only line that ever ran was the one saying
// nothing is wrong.
//
// The lines are docs/designs/pta_cpu_integration.md section 7.1's, as built.
// This test needs no device: it formats structs.

#include <grx/grx.h>

#include "grx_test.h"
#include "../../tools/common/analog_gemm_text.h"

#include <cstdio>
#include <string>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

const std::string kLabel = "    analog GEMM            ";
const std::string kCont  = "                           ";

grxAnalogGemm_t make(int emulated, int present) {
  grxAnalogGemm_t a;
  a.gemmIsAnalogEmulated   = emulated;
  a.tileIsPresent          = present;
  a.activationBits         = -1;
  a.weightBits             = -1;
  a.adcBits                = -1;
  a.adcShift               = -1;
  a.seed                   = -1;
  a.impairments            = -1;
  a.impairmentsImplemented = -1;
  a.thermalSigmaQ8         = -1;
  a.shotCoefficientQ8      = -1;
  a.programmingSigmaQ8     = -1;
  a.driftSigmaQ8           = -1;
  a.driftLog2Shots         = -1;
  a.driftClampQ8           = -1;
  a.crosstalkQ8            = -1;
  a.loopModes              = -1;
  a.calibrationValid       = -1;
  a.tileRows               = -1;
  a.tileCols               = -1;
  a.operandBits            = -1;
  a.accumulatorBits        = -1;
  return a;
}

const int64_t kTile = GRX_ANALOG_DEFINED & ~GRX_ANALOG_MZM_NL;   // 0x5f

// An impaired tile at the board plan's version 0 allowances, on the 8-bit
// bench tile they were measured on: thermal 1 LSB, 3 photons an ADC LSB
// (k = 1/sqrt(3)), programming 4 LSB, 10% crosstalk, and TFLT's drift fit.
grxAnalogGemm_t v0_on_the_bench_tile() {
  grxAnalogGemm_t a = make(1, 1);
  a.activationBits = 8; a.weightBits = 8; a.adcBits = 6; a.adcShift = 3;
  a.seed = 0x2a;
  a.impairments = kTile;
  a.impairmentsImplemented = kTile;
  a.thermalSigmaQ8 = 256; a.shotCoefficientQ8 = 148;
  a.programmingSigmaQ8 = 1024; a.crosstalkQ8 = 26;
  a.driftSigmaQ8 = 55; a.driftLog2Shots = 31; a.driftClampQ8 = 8643;
  a.loopModes = 0; a.calibrationValid = 0;
  a.tileRows = 8; a.tileCols = 8; a.operandBits = 8; a.accumulatorBits = 48;
  return a;
}

void show(const std::vector<std::string>& lines) {
  for (const std::string& l : lines) std::printf("  |%s\n", l.c_str());
}

bool has(const std::vector<std::string>& lines, const std::string& want) {
  for (const std::string& l : lines)
    if (l == want) return true;
  return false;
}

}  // namespace

int main() {
  section("a device that cannot say");
  {
    const std::vector<std::string> l = grxtools::analog_gemm_lines(make(-1, -1));
    show(l);
    check(l.size() == 2, "two lines");
    check(l.size() == 2 && l[0] == kLabel +
          "UNKNOWN: this device's register file has no PTA identity word",
          "it says UNKNOWN, not native");
    check(l.size() == 2 && l[1] == kCont +
          "so whether a GEMM here is exact cannot be read from it",
          "and what that means, on a continuation line 27 spaces in");
    // Either leading field unknown is unknown: there is no half-knowledge here
    // that would justify printing "native".
    check(grxtools::analog_gemm_lines(make(0, -1)).size() == 2 &&
          grxtools::analog_gemm_lines(make(-1, 0)).size() == 2,
          "one unknown field is enough to say so");
  }

  section("no tile");
  {
    grxAnalogGemm_t a = make(0, 0);
    a.impairmentsImplemented = 0;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 1 && l[0] == kLabel + "native (no PTA tile in this build)",
          "one line, and the value starts at column 28");
    check(kLabel.size() == 27, "the label field is 27 columns: 4 of indent and 23");
  }

  section("a tile, nothing enabled");
  {
    grxAnalogGemm_t a = make(0, 1);
    a.impairmentsImplemented = kTile;
    a.tileRows = 8; a.tileCols = 8; a.operandBits = 8; a.accumulatorBits = 48;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 1 && l[0] == kLabel +
          "native (PTA tile present, PTA_IMPAIR clear)",
          "native, naming the register that decides it");
  }

  section("a tile, impaired: version 0's allowances on the 8-bit bench tile");
  {
    const std::vector<std::string> l =
        grxtools::analog_gemm_lines(v0_on_the_bench_tile());
    show(l);
    check(l.size() == 8, "eight lines");
    check(l.size() == 8 && l[0] == kLabel +
          "EMULATED on the PTA tile: a8/w8 of 8, ADC 6 bits << 3, seed 0x0000002a",
          "the quantisers, and the width their bit counts are counts of");
    check(l.size() == 8 && l[1] == kCont +
          "impairments QUANT|THERMAL|SHOT|DRIFT|XTALK|PROG_ERR",
          "the enables in force, by name, in the register's order");
    check(l.size() == 8 && l[2] == kCont +
          "noise: thermal 1.000 LSB, shot k 0.578, programming 4.000 LSB, "
          "crosstalk 0.102",
          "how much: the Q8.8 registers as the numbers they stand for");
    check(l.size() == 8 && l[3] == kCont +
          "drift: 0.215 LSB a step, a step every 2^31 shots, clamp 33.762 LSB",
          "the drift walk");
    check(l.size() == 8 && l[4] == kCont +
          "drift is device state: the answer depends on every shot since MODEL_RST",
          "and that no struct can carry where it has got to");
    check(l.size() == 8 && l[5] == kCont +
          "tile 8 x 8, 8-bit operands, 48-bit sums",
          "the tile, which decides the order the noise is drawn in");
    check(l.size() == 8 && l[6] == kCont +
          "MZM_NL is defined but not implemented; a START with it is refused",
          "what is defined and absent, derived from the mask");
    check(l.size() == 8 && l[7] == kCont +
          "this device does NOT compute the same function as a digital c930",
          "and the sentence the dispatch rule depends on");
  }

  section("the same int8 precision on a 16-bit tile");
  {
    // Five and six bits of an int8 operand are thirteen and fourteen of the
    // sixteen-bit word it sits in. Printed without the width, "a13/w14" reads
    // as a far finer quantiser than version 0's.
    grxAnalogGemm_t a = v0_on_the_bench_tile();
    a.activationBits = 13; a.weightBits = 14; a.adcShift = 10;
    a.tileRows = 4; a.tileCols = 4; a.operandBits = 16;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(!l.empty() && l[0] == kLabel +
          "EMULATED on the PTA tile: a13/w14 of 16, ADC 6 bits << 10, seed 0x0000002a",
          "a13/w14 of 16");
    check(has(l, kCont + "tile 4 x 4, 16-bit operands, 48-bit sums"),
          "and the tile's own line agrees");
  }

  section("a setting that rounds every int8 operand to zero");
  {
    // Six activation bits is a six-bit DAC on the 8-bit bench tile. On a
    // 16-bit tile it keeps the top six of sixteen, and an int8 operand is in
    // the bottom eight.
    grxAnalogGemm_t a = v0_on_the_bench_tile();
    a.activationBits = 6; a.weightBits = 14;
    a.tileRows = 4; a.tileCols = 4; a.operandBits = 16;
    std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(has(l, kCont + "a6 of 16 keeps the top 6 bits: every int8 activation "
                         "rounds to zero"),
          "a6 of 16 is said to zero the activations");
    a.activationBits = 13; a.weightBits = 8;
    l = grxtools::analog_gemm_lines(a);
    check(has(l, kCont + "w8 of 16 keeps the top 8 bits: every int8 weight "
                         "rounds to zero"),
          "w8 of 16 is said to zero the weights: eight bits is not enough here");
    a.weightBits = 9;
    l = grxtools::analog_gemm_lines(a);
    bool any = false;
    for (const std::string& s : l) any = any || s.find("rounds to zero") != std::string::npos;
    check(!any, "w9 of 16 keeps one bit of an int8, and is not flagged");
    check(!has(grxtools::analog_gemm_lines(v0_on_the_bench_tile()),
               kCont + "a8 of 8 keeps the top 8 bits: every int8 activation "
                       "rounds to zero") &&
          grxtools::analog_gemm_lines(v0_on_the_bench_tile()).size() == 8,
          "and on the 8-bit tile nothing is: there the count is the precision");
  }

  section("unquantised, no drift, and a seed that fills the register");
  {
    grxAnalogGemm_t a = v0_on_the_bench_tile();
    a.activationBits = 0; a.weightBits = 0; a.adcBits = 0; a.adcShift = 0;
    a.seed = 0xFFFFFFFFLL;
    a.impairments = GRX_ANALOG_THERMAL;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(!l.empty() && l[0] == kLabel +
          "EMULATED on the PTA tile: a=full/w=full, ADC unquantised, seed 0xffffffff",
          "zero bits is printed as unquantised, never as 'a0', and with no 'of'");
    check(l.size() == 6 && l[1] == kCont + "impairments THERMAL",
          "a single impairment, and six lines: the two drift lines are gone");
    check(has(l, kCont + "noise: thermal 1.000 LSB, shot k 0.578, programming "
                         "4.000 LSB, crosstalk 0.102"),
          "the sigmas are still all shown: the mask says which of them apply");
  }

  section("an enable the tile does not have");
  {
    grxAnalogGemm_t a = v0_on_the_bench_tile();
    a.impairments = GRX_ANALOG_QUANT | GRX_ANALOG_MZM_NL;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(has(l, kCont +
              "MZM_NL is ENABLED and not implemented: every START is refused"),
          "it says the device will run nothing, not merely what it lacks");
    check(!has(l, kCont +
               "MZM_NL is defined but not implemented; a START with it is refused"),
          "and does not also print the milder line");
  }

  section("a tile that implements everything");
  {
    grxAnalogGemm_t a = v0_on_the_bench_tile();
    a.impairments = GRX_ANALOG_DEFINED;
    a.impairmentsImplemented = GRX_ANALOG_DEFINED;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 7,
          "seven lines: there is no not-implemented line to print");
    check(l.size() == 7 && l[1] == kCont +
          "impairments QUANT|THERMAL|SHOT|DRIFT|XTALK|MZM_NL|PROG_ERR",
          "all seven named");
  }

  section("the two states the reference model does not cover");
  {
    grxAnalogGemm_t a = v0_on_the_bench_tile();
    a.loopModes = 0x4;           // MORDER
    a.calibrationValid = 1;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(has(l, kCont + "PTA_CTRL loop modes 0x4 are set: not the order the "
                         "reference model walks"),
          "a loop order the model does not walk is said");
    check(has(l, kCont + "a calibration's trims are in force, which the "
                         "reference model does not hold"),
          "and so are trims it does not hold");
    check(l.size() == 10, "ten lines, the longest form there is");
  }

  return grxtest::report();
}
