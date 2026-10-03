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
  return a;
}

const int64_t kTile = GRX_ANALOG_DEFINED & ~GRX_ANALOG_MZM_NL;   // 0x5f

void show(const std::vector<std::string>& lines) {
  for (const std::string& l : lines) std::printf("  |%s\n", l.c_str());
}

}  // namespace

int main() {
  section("a device that cannot say");
  {
    const std::vector<std::string> l = grxtools::analog_gemm_lines(make(-1, -1));
    show(l);
    check(l.size() == 2, "two lines");
    check(l.size() == 2 && l[0] ==
          "    analog GEMM            UNKNOWN: this device's register file "
          "has no PTA identity word",
          "it says UNKNOWN, not native");
    check(l.size() == 2 && l[1] ==
          "                           so whether a GEMM here is exact cannot "
          "be read from it",
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
    check(l.size() == 1 && l[0] ==
          "    analog GEMM            native (no PTA tile in this build)",
          "one line, and the value starts at column 28");
  }

  section("a tile, nothing enabled");
  {
    grxAnalogGemm_t a = make(0, 1);
    a.impairmentsImplemented = kTile;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 1 && l[0] ==
          "    analog GEMM            native (PTA tile present, PTA_IMPAIR clear)",
          "native, naming the register that decides it");
  }

  section("a tile, impaired -- the specification's own example");
  {
    grxAnalogGemm_t a = make(1, 1);
    a.activationBits = 8; a.weightBits = 8; a.adcBits = 6; a.adcShift = 3;
    a.seed = 0x2a;
    a.impairments = kTile;
    a.impairmentsImplemented = kTile;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 4, "four lines");
    check(l.size() == 4 && l[0] ==
          "    analog GEMM            EMULATED on the PTA tile: a8/w8, "
          "ADC 6 bits << 3, seed 0x0000002a",
          "the model, enough of it to reproduce the answer");
    check(l.size() == 4 && l[1] ==
          "                           impairments "
          "QUANT|THERMAL|SHOT|DRIFT|XTALK|PROG_ERR",
          "the enables in force, by name, in the register's order");
    check(l.size() == 4 && l[2] ==
          "                           MZM_NL is defined but not implemented; "
          "a START with it is refused",
          "what is defined and absent, derived from the mask and not written "
          "for MZM_NL");
    check(l.size() == 4 && l[3] ==
          "                           this device does NOT compute the same "
          "function as a digital c930",
          "and the sentence the dispatch rule depends on");
  }

  section("unquantised, and a seed that fills the register");
  {
    grxAnalogGemm_t a = make(1, 1);
    a.activationBits = 0; a.weightBits = 0; a.adcBits = 0; a.adcShift = 0;
    a.seed = 0xFFFFFFFFLL;
    a.impairments = GRX_ANALOG_THERMAL;
    a.impairmentsImplemented = kTile;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 4 && l[0] ==
          "    analog GEMM            EMULATED on the PTA tile: a=full/w=full, "
          "ADC unquantised, seed 0xffffffff",
          "zero bits is printed as unquantised, never as 'a0'");
    check(l.size() == 4 && l[1] == "                           impairments THERMAL",
          "a single impairment");
  }

  section("an enable the tile does not have");
  {
    grxAnalogGemm_t a = make(1, 1);
    a.activationBits = 8; a.weightBits = 8; a.adcBits = 6; a.adcShift = 3;
    a.seed = 1;
    a.impairments = GRX_ANALOG_QUANT | GRX_ANALOG_MZM_NL;
    a.impairmentsImplemented = kTile;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 4 && l[2] ==
          "                           MZM_NL is ENABLED and not implemented: "
          "every START is refused",
          "it says the device will run nothing, not merely what it lacks");
  }

  section("a tile that implements everything");
  {
    grxAnalogGemm_t a = make(1, 1);
    a.activationBits = 6; a.weightBits = 6; a.adcBits = 7; a.adcShift = 9;
    a.seed = 7;
    a.impairments = GRX_ANALOG_DEFINED;
    a.impairmentsImplemented = GRX_ANALOG_DEFINED;
    const std::vector<std::string> l = grxtools::analog_gemm_lines(a);
    show(l);
    check(l.size() == 3,
          "three lines: there is no not-implemented line to print");
    check(l.size() == 3 && l[1] ==
          "                           impairments "
          "QUANT|THERMAL|SHOT|DRIFT|XTALK|MZM_NL|PROG_ERR",
          "all seven named");
  }

  return grxtest::report();
}
