"""
The proposal's performance model with an operand supply in it
(pta_tpaqcn_review.md section 7, and section 10 item 5).

evaluate_pta() in GRX_AI_Photonics_PTA_TPA_QCN.md (tpaqcn/pta_arch.py) divides
MACs by time of flight: log2(N) MZI stages of 200 um at n_eff 2.5, plus a 1 ps
activation.  Nothing puts data into the light or takes it out again, and four
parameters its own TensorCoreConfig defines -- clock_freq_hz and the DAC, ADC
and modulator energies per bit -- are never read.  This script:

  1. ports evaluate_pta() as written and reproduces its numbers for the
     proposal's own per-chip configuration: make_pta() in its multichip
     example, four layers of N = 16 over 32 WDM channels, four activation
     units per layer, 100 aJ and 1 ps per activation;
  2. adds the operand supply -- a shot rate set by the symbol rate the
     converters run at, the conversions at the two ends of the optical stack
     priced with the proposal's own per-bit energies, and the host feed each
     shot's inputs need;
  3. prices the activations with the review's measured-anchored energies at a
     stated pulse length (pta_tpaqcn_measured.py), beside the analog
     optoelectronic neuron and the digital O-E-O path of review section 4.5.

The proposal's headline (its section 3.1): 10-100 POPS peak throughput,
0.1-1 fJ per MAC, 10-100 aJ per activation, against 4 POPS for an H100.

Assumed, beyond the proposal's own numbers:
    * 6-bit values in and out, the model's only precision parameter;
    * one activation per output per channel -- N x channels a layer, where
      the model counts n_activation_units x channels = 4 x 32;
    * a fully pipelined optical stack: a new shot every symbol, whatever the
      stack's time of flight, so time of flight never binds;
    * conversions only at the stack's two ends -- the all-optical premise.
      An O-E-O stack converts at every layer instead, which only adds.
    * host feed markers: PCIe 6.0 x16 at its raw 64 GT/s per lane, and the
      proposal's own 1-10 Tb/s/mm^2 bandwidth-density claim.

Standard library only.  Run:  python3 docs/designs/pta_operand_supply.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_tpaqcn_measured as meas
import pta_material_scorecard as score

# ---- the proposal's own numbers (GRX_AI_Photonics_PTA_TPA_QCN.md) -----------
N = 16                        # TensorCoreConfig(N=16, ...) in make_pta()
CHANNELS = 32                 # wavelength_channels
MODES = 1
LAYERS = 4
ACT_UNITS = 4                 # PTALayer(n_activation_units=4)
BITS = 6                      # weight_precision_bits
CLOCK_HZ = 10e9               # clock_freq_hz, never read by evaluate_pta
MAC_J = 0.5e-15               # mac_energy_mzi
DAC_J_BIT, ADC_J_BIT, MOD_J_BIT = 50e-15, 200e-15, 30e-15   # never read
ACT_J, ACT_S = 100e-18, 1e-12                             # ActivationUnitConfig
ACT_BW_HZ = 100e9
LINK_BAUD = 50e9              # LinkConfig(symbol_rate_Gbaud=50.0)
GPU_MACS_S, GPU_J_MAC = 4e15, 5e-12                       # baseline_gpu_metrics
HEADLINE_POPS = (10, 100)
DENSITY_TBPS_MM2 = (1, 10)    # "Bandwidth density 1-10 Tb/s/mm^2"

# ---- markers and baselines ---------------------------------------------------
PCIE6_X16_BPS = 64e9 * 16     # raw signalling rate, one direction
PULSES = (("10 ps", 10e-12), ("1 ps", 1e-12), ("100 fs", 100e-15))
POPS = 1e15


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def fmt_rate(bps):
    for unit, scale in (("Tb/s", 1e12), ("Gb/s", 1e9), ("Mb/s", 1e6)):
        if bps >= scale:
            return f"{bps / scale:.3g} {unit}"
    return f"{bps:.3g} b/s"


# ---- 1. evaluate_pta(), ported as written -----------------------------------
def tensor_core_latency_s(n):
    """MZI mesh: int(log2 N) stages of 200 um at n_eff 2.5, c = 3e8."""
    return 2.5 * int(math.log2(n)) * 200e-6 / 3e8


def evaluate_pta():
    n_macs = N * N * CHANNELS * MODES
    lat = tensor_core_latency_s(N) + ACT_S
    energy = MAC_J * n_macs + ACT_J * ACT_UNITS * CHANNELS
    total_macs, total_lat, total_e = LAYERS * n_macs, LAYERS * lat, LAYERS * energy
    return {
        "macs_per_layer": n_macs,
        "latency_s": total_lat,
        "energy_per_mac_J": total_e / total_macs,
        "throughput_macs_s": total_macs / total_lat,
    }


# ---- 2. the operand supply ---------------------------------------------------
MACS_PER_SHOT = LAYERS * N * N * CHANNELS * MODES
VALUES_IN = VALUES_OUT = N * CHANNELS * MODES
BITS_IN_PER_SHOT = VALUES_IN * BITS
IO_J_PER_SHOT = (VALUES_IN * BITS * (DAC_J_BIT + MOD_J_BIT)
                 + VALUES_OUT * BITS * ADC_J_BIT)
ACTS_PER_SHOT = LAYERS * N * CHANNELS * MODES


def energy_per_mac(act_j):
    """MAC energy, the conversions at the stack's ends, and one activation per output."""
    return MAC_J + (IO_J_PER_SHOT + ACTS_PER_SHOT * act_j) / MACS_PER_SHOT


def main():
    section("1. evaluate_pta() as written, the proposal's per-chip configuration")
    r = evaluate_pta()
    print(f"  {LAYERS} layers x (N={N}, {CHANNELS} channels): {r['macs_per_layer']:,} MACs a layer,"
          f" {LAYERS * r['macs_per_layer']:,} a shot")
    print(f"  latency {r['latency_s'] * 1e12:.2f} ps (time of flight + {ACT_S * 1e12:.0f} ps per"
          f" activation), energy {r['energy_per_mac_J'] * 1e15:.3f} fJ per MAC")
    print(f"  throughput {r['throughput_macs_s']:.3e} MACs/s = {r['throughput_macs_s'] / POPS:.2f} POPS"
          f"  ({r['throughput_macs_s'] / GPU_MACS_S:.2f}x the model's own H100 at"
          f" {GPU_MACS_S / POPS:.0f} POPS)")
    print(f"  so the table's {HEADLINE_POPS[0]}-{HEADLINE_POPS[1]} POPS is not one chip: it needs"
          f" {HEADLINE_POPS[0] * POPS / r['throughput_macs_s']:.0f}-"
          f"{HEADLINE_POPS[1] * POPS / r['throughput_macs_s']:.0f} of them")
    print(f"  never read: clock_freq_hz {CLOCK_HZ / 1e9:.0f} GHz; DAC {DAC_J_BIT * 1e15:.0f},"
          f" ADC {ADC_J_BIT * 1e15:.0f}, modulator {MOD_J_BIT * 1e15:.0f} fJ per bit")
    print(f"  activations counted: {ACT_UNITS * CHANNELS} a layer; outputs to activate:"
          f" {N * CHANNELS} a layer")

    section("2. Throughput with an operand supply: a shot per symbol")
    print(f"  one shot moves {BITS_IN_PER_SHOT:,} bits in ({VALUES_IN} values x {BITS} bits) and"
          f" as many out, for {MACS_PER_SHOT:,} MACs")
    print(f"  {'symbol rate':<34}{'POPS':>8}{'vs H100':>9}{'host feed needed':>18}"
          f"{'I/O area at 1-10 Tb/s/mm^2':>29}")
    rates = (("the proposal's control clock", CLOCK_HZ),
             ("the proposal's optical link", LINK_BAUD),
             ("the activation's bandwidth", ACT_BW_HZ))
    for label, rate in rates:
        rate = min(rate, ACT_BW_HZ)
        macs_s = MACS_PER_SHOT * rate
        feed = BITS_IN_PER_SHOT * rate
        area = f"{feed / 1e12 / DENSITY_TBPS_MM2[1]:.3g}-{feed / 1e12 / DENSITY_TBPS_MM2[0]:.3g} mm^2"
        print(f"  {label + ' ' + str(int(rate / 1e9)) + ' GBd':<34}{macs_s / POPS:8.2f}"
              f"{macs_s / GPU_MACS_S:8.2f}x{fmt_rate(feed):>18}{area:>29}")
    feed_bound = MACS_PER_SHOT * PCIE6_X16_BPS / BITS_IN_PER_SHOT
    print(f"  fed over one PCIe 6.0 x16 link ({fmt_rate(PCIE6_X16_BPS)} raw): "
          f"{PCIE6_X16_BPS / BITS_IN_PER_SHOT / 1e6:.0f} M shots/s = {feed_bound / POPS:.3f} POPS,"
          f" {feed_bound / GPU_MACS_S:.4f}x the H100")
    print(f"  a chip needs {HEADLINE_POPS[0] * POPS / MACS_PER_SHOT:.3g}-"
          f"{HEADLINE_POPS[1] * POPS / MACS_PER_SHOT:.3g} shots/s for the headline --"
          f" {HEADLINE_POPS[0] * POPS / MACS_PER_SHOT / ACT_BW_HZ:.0f}-"
          f"{HEADLINE_POPS[1] * POPS / MACS_PER_SHOT / ACT_BW_HZ:.0f}x its own 100 GHz activation")

    section("3. Energy per MAC with the conversions and a real activation")
    io_per_mac = IO_J_PER_SHOT / MACS_PER_SHOT
    print(f"  conversions at the stack's ends: {IO_J_PER_SHOT * 1e12:.0f} pJ a shot ="
          f" {io_per_mac * 1e15:.1f} fJ per MAC, with the proposal's own per-bit energies")
    kap4 = score.kappa_tpaqcn() * score.CHI31_C4 / meas.CHI31
    p_c4 = meas.knee_power(6.0, kap4, 5.0, 5.0)
    p_tfln = meas.knee_power(10.0, math.sqrt(score.TFLN_ETA * 1e4), 0.0, 0.0)
    neuron = score.neuron_j(10, 1.0)
    rows = [("proposal: 100 aJ, 4 a layer counted", None)]
    rows += [(f"proposal: 100 aJ, one per output", ACT_J)]
    rows += [(f"compound 4, 6 mm, at {name}", p_c4 * tau) for name, tau in PULSES]
    rows += [(f"TFLN-class, 1 cm lossless, at {name}", p_tfln * tau) for name, tau in PULSES]
    rows += [(f"analog O-E-O neuron, {meas.fmt_energy(neuron)}", neuron),
             (f"digital O-E-O path, {meas.fmt_energy(meas.OEO_J)}", meas.OEO_J)]
    print(f"  {'activation':<40}{'per activation':>15}{'fJ per MAC':>12}{'x claim top':>13}"
          f"{'H100 / this':>13}")
    for label, act_j in rows:
        if act_j is None:
            e = evaluate_pta()["energy_per_mac_J"]
            per_act = meas.fmt_energy(ACT_J)
        else:
            e = energy_per_mac(act_j)
            per_act = meas.fmt_energy(act_j)
        print(f"  {label:<40}{per_act:>15}{e * 1e15:12.3g}{e / 1e-15:12.3g}x{GPU_J_MAC / e:12.0f}x")
    print("  (x claim top: over the proposal's 1 fJ, the top of its 0.1-1 fJ claim.  H100 / this:")
    print(f"   its {GPU_J_MAC * 1e12:.0f} pJ over this row.)  With conversions at every layer, as O-E-O"
          f" stacks need, add {io_per_mac * (LAYERS - 1) * 1e15:.0f} fJ per MAC.")


if __name__ == "__main__":
    main()
