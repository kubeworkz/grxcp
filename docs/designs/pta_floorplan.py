"""
Open question 1 of board_program_plan.md, the other half: how big is the chiplet?

pta_shot_rate.py bounded how fast the tile fires.  This bounds how large it is.
Nothing here is a layout.  It is a floorplan in the sense P1 needs first: which
things have to be on the photonic die and under the interface chip, how much
room each takes as a function of the figures nobody has fixed, and which of
them ends up setting the size.

What is on the die is read off the program's own error model, not chosen here:

  inputs    a Mach-Zehnder modulator a row.  PTA_IMPAIR has a bit for its
            nonlinearity (MZM_NL), and the CPU document's section 1 has
            "activation DAC -> modulator array"
  weights   k x n cells, "a DAC-held voltage each" (CPU document 4.4, B4)
  outputs   a detector a column (CPU document 1)

Six things take room, and each is priced from a figure with a source:

  1. the inputs       as long as a modulator is, which the platform sets
  2. the weights      either as long as a modulator or as small as a resonant
                      cell: the two readings differ by two orders
  3. the connections  every weight is held on one die and applied on the other
  4. the facet        a fiber takes a pitch of die edge
  5. the link         UCIe modules have a width and a depth
  6. the heat         the laser's light lands somewhere

Held by the program already, and reused:
    1.96 V cm       TFLT's half-wave voltage-length product (CPU document 4.4;
                    C. Wang et al., Nature 629, 784, 2024).  The same paper's
                    abstract gives 5.6 dB/m of propagation loss
    20 mm           the length of the 45 GHz TFLN device pta_tw_sweep.py is
                    anchored on (C. Wang et al., Nature 562, 101, 2018)
    pta_chiplet_link.py   the candidate geometries, the module, the layer
    pta_shot_rate.py      the laser's power under section 4.3's version 1

From board_program_plan.md section 8, question 1, where each is cited:
    100 um          flip-chip pad pitch on modulators and detectors (Urino 2014)
    25 um           a weight cell -- AN ASSUMPTION in a pending patent
                    application, for another topology.  Used as the floor
    127, 250 um     fiber pitch; 24 channels an array, four of them given to
                    alignment on an edge-coupled array (Europractice v1.7)
    37 at 40 um     couplers under one multicore fiber (Kopp 2015)
    8, 16           lines a fiber in the announced products
    56              wavelength channels a bus, by the same application
    9.5 x 13 mm     the reference package, its C4 bumps at 250 um, its
                    through-mold vias at 300 um and its redistribution at 15 um
                    line and space on two front layers (Li 2025)
    10 x 10 mm      the largest die a standard packaging service takes
    4,000 mm2       the M1000's interposer

New here, from the UCIe Consortium's Hot Chips 2023 tutorial ("Electrical,
Form-Factor, and Compliance", its Electrical Summary):
    571.5 x 1540 um a standard-package x16 module's PHY at 24 and 32 Gb/s
    100-130, 25-55  bump pitch in um, standard and advanced package, with 110
                    and 45 the pitches its own density figures are estimated at
    224 and 145     GByte/s per mm of die edge and per mm2, at 32 Gb/s.  Both
                    directions and bytes -- which the checks confirm, because
                    it is the only reading the dimensions agree with

Assumed, and marked again where each is used:
    - the voltage a weight's or an input's DAC swings.  1, 2 and 5 V are swept;
      nothing has designed a DAC
    - a modulator's lateral pitch.  Nothing sizes one, so it is swept at the
      same pitches as the bond, and a real pair of arms with its electrodes is
      unlikely to be as narrow as the finest of them
    - that a resonant cell is no smaller than the pad that reaches it, so the
      bond pitch and the cell pitch are one number
    - one set of electrodes.  B4's second weight bank is taken as a second set
      of held voltages on the interface chip, not a second set of cells
    - that all the light the detectors do not receive is absorbed on the die,
      which is an upper bound on the heat and not an estimate of it

Not priced: the interface chip's own circuits.  Its converters, its weight
store and its buffers need device figures this program does not hold, so the
chip appears here only as the area it has to span to reach its pads.
(pta_power.py has since priced the converters, the drive and the link from
published parts.  The weight store and the buffers are still not.)

Standard library only.  Run:  python3 docs/designs/pta_floorplan.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_chiplet_link as link
import pta_shot_rate as shot

# ---- the platform ----------------------------------------------------------
TFLT_VPI_L_V_CM = 1.96      # held: CPU document 4.4
TFLT_LOSS_DB_M = 5.6        # the same paper's abstract; propagation only
TFLN_ANCHOR_MM = 20.0       # the device behind pta_tw_sweep.py's 45 GHz
SWINGS_V = (1.0, 2.0, 5.0)  # ASSUMED: no DAC has been designed

# ---- pitches, in um --------------------------------------------------------
P_FLIP = 100                # Urino et al.: designed and built
P_UCIE_S, UCIE_S_RANGE = 110, (100, 130)
P_UCIE_A, UCIE_A_RANGE = 45, (25, 55)
P_CELL = 25                 # the application's assumed cell, and UCIe-A's floor
BOND_PITCHES = (
    (P_FLIP, "flip-chip, as built in 2014"),
    (UCIE_A_RANGE[1], "the advanced package's coarsest"),
    (P_UCIE_A, "the advanced package's estimating pitch"),
    (P_CELL, "its finest, and the assumed cell"),
)

# ---- the facet --------------------------------------------------------------
FIBER_PITCHES = (250, 127)
ARRAY_CHANNELS, EDGE_SHUNTS, ARRAYS_A_DIE = 24, 4, 2
LINES_A_FIBER = (8, 16)
MULTICORE = 37
LINES_A_BUS = 56

# ---- the reference outlines -------------------------------------------------
PKG_MM = (9.5, 13.0)
C4_PITCH, TMV_PITCH = 250, 300
RDL_LINE_SPACE, RDL_FRONT_LAYERS = 15, 2
STD_DIE_MM = (10.0, 10.0)
M1000_MM2 = 4000.0

# ---- one UCIe-S module at 32 GT/s ------------------------------------------
UCIE_W_UM, UCIE_D_UM = 571.5, 1540.0
UCIE_EDGE_GBS_MM, UCIE_AREA_GBS_MM2 = 224, 145
B8_FLIP_MODULES = shot.B8_FLIP_MODULES

# ---- densities offered for comparison, neither of them a limit --------------
RING_LINK_W_MM2 = 1.4       # Lightmatter reports a ring link stable above this
COMPUTE_W_MM2 = 2.0         # "200+ W/cm2", the die the keep-out article fears

X2_TILE = shot.X2_TILE
MM = 1e-3                   # um to mm


# ---- 1. the inputs ---------------------------------------------------------
def mzm_mm(volts):
    """How long a Pockels Mach-Zehnder is, to turn fully on at this swing."""
    return TFLT_VPI_L_V_CM / volts * 10.0


def strip_mm2(count, length_mm, pitch_um):
    """Long devices side by side: a count of them, each a pitch wide."""
    return count * length_mm * pitch_um * MM


def propagation_db(length_mm):
    return TFLT_LOSS_DB_M * length_mm * 1e-3


# ---- 2. the weights --------------------------------------------------------
def field_mm2(count, pitch_um):
    """A square array of things on a pitch: cells, pads or bumps."""
    return count * (pitch_um * MM) ** 2


def cells_that_fit(area_mm2, cell_mm2):
    return int(area_mm2 / cell_mm2 + 1e-9)


# ---- 3. the connections ----------------------------------------------------
def connections(k, n):
    """Lines between the two dies: a weight each, an input each, an output each."""
    return k * n + k + n


def side_by_side_mm(lines):
    """Die edge the lines take if the two dies sit beside each other."""
    return lines * 2 * RDL_LINE_SPACE * MM / RDL_FRONT_LAYERS


def lines_across_mm(edge_mm):
    return int(edge_mm * RDL_FRONT_LAYERS / (2 * RDL_LINE_SPACE * MM) + 1e-9)


# ---- 4. the facet ----------------------------------------------------------
def fibers(k, lines_a_fiber):
    return -(-k // lines_a_fiber)


def facet_mm(count, pitch_um):
    return count * pitch_um * MM


# ---- 5. the link -----------------------------------------------------------
def link_modules(k, n, mb):
    """UCIe-S modules for the link model's layer at 1 GS/s: a module is both ways."""
    into, out = link.rates(k=k, n=n, fs=shot.X2_FS, **dict(link.BIG, mb=mb))
    return max(link.modules(into), link.modules(out))


def module_mm2():
    return UCIE_W_UM * UCIE_D_UM * MM * MM


def module_sites():
    """Bump sites under one module's PHY at the pitch its densities assume."""
    return UCIE_W_UM * UCIE_D_UM / P_UCIE_S ** 2


def escape_mm2(mods, pitch_um):
    """The same sites spread to a coarser pitch on the way to the substrate."""
    return field_mm2(mods * module_sites(), pitch_um)


# ---- 7. the die ------------------------------------------------------------
def die_mm2(k, n, pitch_um, volts):
    """(weights, inputs): resonant weights on the bond pitch, an MZM a row.

    volts=None makes the inputs resonant too: one more cell a row.
    """
    weights = field_mm2(k * n, pitch_um)
    if volts is None:
        return weights, field_mm2(k, pitch_um)
    return weights, strip_mm2(k, mzm_mm(volts), pitch_um)


def pitch_ceiling_um(k, n, volts, area_mm2):
    """The coarsest bond pitch at which die_mm2() still fits an area."""
    a = k * n + (k if volts is None else 0)
    b = 0.0 if volts is None else k * mzm_mm(volts)
    return (-b + math.sqrt(b * b + 4 * a * area_mm2)) / (2 * a) / MM


def area(mm):
    return mm[0] * mm[1]


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    k, n = X2_TILE
    v1 = shot.REQ["v1"]
    std, pkg = area(STD_DIE_MM), area(PKG_MM)

    print("Open question 1, the other half: what sizes the PTA chiplet.")
    print("Areas are lower bounds on a die, not a layout of one.")

    # ------------------------------------------------------------------ 0
    section("0. What is on the die, by geometry")
    print(f"  {'tile':<10}{'input MZMs':>11}{'weights':>9}{'detectors':>11}{'lines between dies':>20}")
    for gk, gn in link.GEOMETRIES:
        print(f"  {f'{gk}x{gn}':<10}{gk:>11}{gk * gn:>9,}{gn:>11}{connections(gk, gn):>20,}")
    print("  A line between dies is one held voltage reaching one electrode, or one")
    print("  detector reaching its amplifier.  TFLT has no transistor to hold or to")
    print("  switch with, so nothing on the photonic die can share them.")

    # ------------------------------------------------------------------ 1
    section("1. The inputs: a Pockels modulator is as long as its voltage is low")
    print(f"  TFLT is {TFLT_VPI_L_V_CM} V cm.  Length = that / swing, and the swing is ASSUMED:")
    print(f"  {'swing':>7}{'one MZM':>10}{f'{k} of them':>13}{'at 25 um':>11}{'at 100 um':>11}"
          f"{'its light lost':>16}")
    for v in SWINGS_V:
        L = mzm_mm(v)
        print(f"  {v:>5.0f} V{L:>7.2f} mm{k * L / 1e3:>11.2f} m"
              f"{strip_mm2(k, L, P_CELL):>7.0f} mm2{strip_mm2(k, L, P_FLIP):>7.0f} mm2"
              f"{propagation_db(L):>13.2f} dB")
    print("  The last two area columns are the lateral pitch swept, not known.")
    print(f"  The die is at least one modulator long.  At 1 V that is {mzm_mm(1.0):.1f} mm:")
    print(f"  longer than the reference package's {PKG_MM[1]:.0f} mm side, and the same length")
    print(f"  as the {TFLN_ANCHOR_MM:.0f} mm TFLN device the shot-rate model's modulator bound is")
    print("  anchored on.  Length costs area and not light: 5.6 dB/m is a tenth of")
    print("  a decibel over the longest of these.")

    # ------------------------------------------------------------------ 2
    section(f"2. The weights: two readings of a cell, two orders apart ({k}x{n})")
    print("  NOT RESONANT.  A weight that attenuates by interference is a modulator,")
    print(f"  and as long as one.  {k * n:,} of them, {P_CELL} um apart:")
    for v in SWINGS_V:
        a = strip_mm2(k * n, mzm_mm(v), P_CELL)
        print(f"    {v:.0f} V  {a:>7,.0f} mm2   {a / M1000_MM2:>5.2f} of the M1000's interposer,"
              f" {a / std:>4.0f} standard dies")
    print("  RESONANT.  A ring needs its resonance moved by a linewidth, not its")
    print("  phase by pi, so its size is a pitch and not a length:")
    for p, note in BOND_PITCHES:
        ex = (k * p * MM, n * p * MM)
        print(f"    {p:>3} um  {field_mm2(k * n, p):>7.1f} mm2   {ex[0]:.1f} x {ex[1]:.1f} mm   {note}")
    print("  Those are the bond's pitches standing in.  No source sizes a TFLT ring,")
    print("  and three figures are missing behind this reading: the ring's size, its")
    print("  linewidth, and the swing that moves it by one.  The last is governed by")
    print("  the same 1.96 V cm, through a group index this program does not hold.")
    print("  (pta_ring.py, 2026-10-05, has since read five published rings into those three")
    print("  places.  The smallest is 30 um in radius, a 60 um cell: no row above is as fine.)")
    print("  What a non-resonant tile CAN be, at 5 V and 25 um:")
    cell = mzm_mm(5.0) * P_CELL * MM
    for name, a in (("the largest standard die", std), ("the reference package", pkg)):
        print(f"    {name:<26}{cells_that_fit(a, cell):>6,} cells")
    fits = [f"{gk}x{gn}" for gk, gn in link.GEOMETRIES if gk * gn <= cells_that_fit(std, cell)]
    print(f"  so of the candidates only {' and '.join(fits)} fit a standard die that way.")

    # ------------------------------------------------------------------ 3
    section(f"3. The connections: {connections(k, n):,} lines between two dies ({k}x{n})")
    print("  STACKED, a pad a line.  The interface chip has to span this to reach them:")
    print(f"  {'bond pitch':>11}{'weights alone':>15}{'all lines':>11}")
    for p, note in BOND_PITCHES:
        print(f"  {p:>8} um{field_mm2(k * n, p):>11.1f} mm2{field_mm2(connections(k, n), p):>7.1f} mm2"
              f"   {note}")
    edge = side_by_side_mm(connections(k, n))
    print(f"  SIDE BY SIDE, a trace a line.  At the reference package's {RDL_LINE_SPACE} um line")
    print(f"  and space on {RDL_FRONT_LAYERS} layers the lines need {edge:.0f} mm of shared edge.  Its")
    print(f"  {PKG_MM[1]:.0f} mm side carries {lines_across_mm(PKG_MM[1]):,}, a tile of that many cells.")

    # ------------------------------------------------------------------ 4
    section(f"4. The facet: fibers by how the light arrives ({k} inputs)")
    print(f"  {'arrangement':<34}{'fibers':>7}{'at 250 um':>11}{'at 127 um':>11}")
    rows = [("one laser, one fiber (B5)", 1),
            ("a fiber an input row", k)]
    rows += [(f"{l} lines a fiber", fibers(k, l)) for l in LINES_A_FIBER]
    rows += [(f"multicore, {MULTICORE} a fiber", fibers(k, MULTICORE))]
    for name, f in rows:
        cells_ = "".join(f"{facet_mm(f, p):>8.1f} mm" for p in FIBER_PITCHES)
        print(f"  {name:<34}{f:>7}{cells_}")
    usable = ARRAYS_A_DIE * (ARRAY_CHANNELS - EDGE_SHUNTS)
    print(f"  A standard service attaches {usable} usable fibers to a die.  The multicore")
    print("  row is grating couplers in two dimensions and takes no edge at all.")
    print(f"  And if the weights are rings, {k} lines are {fibers(k, LINES_A_BUS)} buses at"
          f" {LINES_A_BUS} lines a bus,")
    print(f"  or {fibers(k, 16)} at the 16 a source carries.")

    # ------------------------------------------------------------------ 5
    section("5. The link: what UCIe-S asks of the interface chip")
    m2, sites = module_mm2(), module_sites()
    print(f"  one x16 module at 32 GT/s is {UCIE_W_UM} x {UCIE_D_UM:.0f} um = {m2:.2f} mm2,"
          f" about {sites:.0f} bump")
    print(f"  sites at {P_UCIE_S} um.  A module is both directions.")
    print(f"  {'tile':<10}" + "".join(f"{f'batch {mb}':>10}" for mb in (16, 64, 256))
          + f"{'at batch 64:':>15}{'PHY':>6}{'die edge':>10}{'via field':>11}{'C4 field':>10}")
    for gk, gn in link.GEOMETRIES:
        mods = [link_modules(gk, gn, mb) for mb in (16, 64, 256)]
        m = mods[1]
        print(f"  {f'{gk}x{gn}':<10}" + "".join(f"{x:>10}" for x in mods) + f"{'':>15}"
              f"{m * m2:>4.1f}mm2{m * UCIE_W_UM * MM:>7.1f} mm"
              f"{escape_mm2(m, TMV_PITCH):>7.0f} mm2{escape_mm2(m, C4_PITCH):>6.0f} mm2")
    c4_sites = int(pkg / (C4_PITCH * MM) ** 2)
    print(f"  The via and C4 fields are the same sites at {TMV_PITCH} and {C4_PITCH} um: in the reference")
    print("  package the chip sits on top, so its link comes down through the mold")
    print(f"  to the substrate.  That package has {c4_sites:,} C4 sites in all, and B8's")
    b8 = B8_FLIP_MODULES * sites
    print(f"  {B8_FLIP_MODULES} modules are {b8:,.0f}.  It is a statement about a package of that size,")
    print("  which was built for eight lanes, and not a limit on one built for this.")

    # ------------------------------------------------------------------ 6
    section(f"6. Which term sizes the die ({k}x{n}, resonant weights, an MZM a row)")
    print("  The weights sit on the bond pitch and the modulators' lateral pitch is")
    print("  taken as the same number, ASSUMED.")
    print(f"  {'bond pitch':>11}{'weights':>9}{'inputs, 5 V':>13}{'inputs, 2 V':>13}"
          f"{'die, 5 V':>10}{'die, 2 V':>10}   against a {std:.0f} mm2 standard die")
    for p, _ in BOND_PITCHES:
        w, i5 = die_mm2(k, n, p, 5.0)
        _, i2 = die_mm2(k, n, p, 2.0)
        verdict = "fits at 5 V" if w + i5 <= std else f"{(w + i5) / std:.2f}x at 5 V"
        verdict += ", fits at 2 V" if w + i2 <= std else f", {(w + i2) / std:.2f}x at 2 V"
        print(f"  {p:>8} um{w:>9.1f}{i5:>13.1f}{i2:>13.1f}{w + i5:>10.1f}{w + i2:>10.1f}   {verdict}")
    print(f"  1 V is not in the table because it does not get as far as an area: its")
    print(f"  modulator is {mzm_mm(1.0):.1f} mm, and a {STD_DIE_MM[0]:.0f} mm die does not hold one at any pitch.")
    print()
    print("  The coarsest bond pitch at which each candidate fits a standard die,")
    print("  by what an input is (um):")
    print(f"  {'tile':<10}{'MZM at 2 V':>12}{'MZM at 5 V':>12}{'resonant':>10}")
    for gk, gn in link.GEOMETRIES:
        cells_ = "".join(f"{pitch_ceiling_um(gk, gn, v, std):>12.0f}" for v in (2.0, 5.0))
        print(f"  {f'{gk}x{gn}':<10}{cells_}{pitch_ceiling_um(gk, gn, None, std):>10.0f}")
    print(f"  UCIe's advanced package runs {UCIE_A_RANGE[0]} to {UCIE_A_RANGE[1]} um and its standard"
          f" {UCIE_S_RANGE[0]} to {UCIE_S_RANGE[1]}.")

    # ------------------------------------------------------------------ 7
    section(f"7. The heat the light brings ({k}x{n}, version 1, 1 GS/s)")
    lo, hi = (shot.laser_power(v1, shot.X2_FS, n, d, 0.5) for d in shot.LOSS_DB)
    print(f"  The laser is {lo:.2f} to {hi:.1f} W (pta_shot_rate.py).  If every watt the detectors")
    print("  do not receive were absorbed on the die -- an upper bound:")
    print(f"  {'bond pitch':>11}{'die, 5 V':>10}{'at 10 dB':>14}{'at 20 dB':>14}")
    for p, _ in BOND_PITCHES:
        a = sum(die_mm2(k, n, p, 5.0))
        print(f"  {p:>8} um{a:>6.0f} mm2{lo / a:>8.3f} W/mm2{hi / a:>8.3f} W/mm2")
    print(f"  For comparison, and neither is a limit: Lightmatter reports a ring link")
    print(f"  stable above {RING_LINK_W_MM2} W/mm2, and the die the keep-out article fears is"
          f" {COMPUTE_W_MM2:.0f}.")
    print("  The converters' power sits on the same area, and pta_power.py has it.")

    findings()
    checks()


def findings():
    k, n = X2_TILE
    v1 = shot.REQ["v1"]
    std, pkg = area(STD_DIE_MM), area(PKG_MM)
    nonres5 = strip_mm2(k * n, mzm_mm(5.0), P_CELL)
    w25, i25 = die_mm2(k, n, P_CELL, 5.0)
    w100, i100 = die_mm2(k, n, P_FLIP, 5.0)
    hi = shot.laser_power(v1, shot.X2_FS, n, shot.LOSS_DB[1], 0.5)

    print()
    print("What this says, six readings.")
    print()
    print("1. ON THIS PLATFORM THE WEIGHTS ARE RESONANT, OR THE TILE IS NOT A CHIPLET.")
    print(f"   A TFLT weight that is not resonant is {mzm_mm(5.0):.1f} mm long at 5 V and {mzm_mm(1.0):.1f} at"
          f" 1 V, and")
    print(f"   {k * n:,} of them at the finest pitch here are {nonres5:,.0f} mm2 at 5 V: {nonres5 / std:.0f} standard"
          f" dies,")
    print(f"   {nonres5 / M1000_MM2:.0%} of the largest photonic part in the board plan's references."
          f"  Short of")
    print("   a smaller tile, another material or a cell no figure here describes,")
    print("   the weights sit on resonances.  That much of board plan question 8 is")
    print("   no longer open, and what comes with it stops being conditional: a")
    print("   level held on the side of a resonance moves with temperature and with")
    print("   the laser's wavelength, by figures nobody holds for TFLT.  Whether the")
    print("   inputs are also told apart by wavelength -- the ring bank, with a line")
    print("   an input and a limit on lines a bus -- the area does not decide.")
    print()
    print("2. THE INPUTS ARE THE LARGEST OPTICS ON THE DIE, NOT THE WEIGHTS.  The error")
    print(f"   model has a modulator a row.  {k} of them are {k * mzm_mm(5.0) / 1e3:.1f} m of modulator at"
          f" 5 V and {k * mzm_mm(1.0) / 1e3:.1f} at")
    print(f"   1 V: {i25:.0f} mm2 at a 25 um lateral pitch against the weights' {w25:.1f}.  Only at")
    print(f"   flip-chip pitch do the weights catch up, {w100:.0f} against {i100:.0f}.  And the die is")
    print("   one modulator long whatever else is on it.  The board plan's \"the bond")
    print("   pitch sizes the die, and the optics do not\" was written about the weights")
    print("   and had not counted the inputs.  If an input is a ring and not an MZM")
    print("   this reading goes away, and MZM_NL is then the wrong impairment.")
    print()
    c5, c2 = pitch_ceiling_um(k, n, 5.0, std), pitch_ceiling_um(k, n, 2.0, std)
    cr = pitch_ceiling_um(k, n, None, std)
    print("3. THE BOND PITCH IS THE NUMBER, AND THE SWING IS THE OTHER ONE.  The tile")
    print(f"   fits the largest die a standard service packages at a bond pitch of {c5:.0f} um")
    print(f"   or finer with 5 V inputs, {c2:.0f} um with 2 V ones, and {cr:.0f} um if the inputs are")
    print(f"   resonant too.  The first two are inside UCIe's advanced-package range of")
    print(f"   {UCIE_A_RANGE[0]} to {UCIE_A_RANGE[1]} um, and none reaches its standard package's {UCIE_S_RANGE[0]}.  At 1 V"
          f" the")
    print(f"   modulator is {mzm_mm(1.0):.1f} mm and no pitch helps.  So the volts a DAC swings is a")
    print("   floorplan input as much as the pitch is, and nothing has specified it.")
    print()
    edge = side_by_side_mm(connections(k, n))
    print("4. B4's \"STACKED OR SIDE BY SIDE\" IS NOT OPEN FOR A TILE THIS SIZE.  Side by")
    print(f"   side, {connections(k, n):,} lines need {edge:.0f} mm of shared edge at the reference")
    print(f"   package's own redistribution; its long side carries {lines_across_mm(PKG_MM[1]):,}.  That is fan-out")
    print("   wiring, the finest B2's organic packaging has; a silicon bridge is finer")
    print("   and is what B2 put off.  Within B2, stacked is the only arrangement that")
    print("   reaches the weights, and it puts the converters directly over the tile.")
    print()
    m64 = link_modules(k, n, 64)
    print("5. THE LINK IS SMALL UNTIL IT HAS TO COME DOWN THROUGH THE PACKAGE.  X2's")
    print(f"   {m64} modules are {m64 * module_mm2():.1f} mm2 of PHY and {m64 * UCIE_W_UM * MM:.1f} mm of die edge."
          f"  But a stacked chip's")
    print(f"   link reaches the substrate through the mold: {escape_mm2(m64, TMV_PITCH):.0f} mm2 of vias at"
          f" {TMV_PITCH} um,")
    print(f"   three times the dense tile's weights.  At B8's {B8_FLIP_MODULES} modules the bumps"
          f" outnumber")
    print("   every C4 site under a package the reference one's size.  The module")
    print("   count is a package-area question before it is a signalling one.")
    print()
    print("6. THE LIGHT IS NOT WHAT HEATS THE TILE.  Even all of a 20 dB laser absorbed")
    print(f"   on the densest die is {hi / (w25 + i25):.2f} W/mm2, a fifteenth of what a ring link is")
    print("   reported stable under.  The facet does not bind either: one fiber, or")
    print(f"   {fibers(k, 16)} at sixteen lines each.  What this left unpriced was the same thing as")
    print("   before, with an address: the converters, on top of the tile.  pta_power.py")
    print("   has since priced them.")
    print()


def checks():
    """Every claim above, as an assert."""
    k, n = X2_TILE
    v1 = shot.REQ["v1"]
    std, pkg = area(STD_DIE_MM), area(PKG_MM)
    assert (k, n) == (256, 64) and std == 100.0 and pkg == 123.5

    # 1. The modulator's length is the held figure over the swing, and at 1 V it
    #    is the anchor device's own length to within 2%.
    assert abs(mzm_mm(5.0) - 3.92) < 1e-9 and abs(mzm_mm(1.0) - 19.6) < 1e-9
    assert abs(mzm_mm(1.0) / TFLN_ANCHOR_MM - 1.0) < 0.03
    assert mzm_mm(1.0) > PKG_MM[1] > STD_DIE_MM[0] > mzm_mm(2.0)

    # 2. Length costs no light worth counting: the longest modulator here loses
    #    a tenth of a decibel in propagation.
    assert propagation_db(mzm_mm(1.0)) < 0.12

    # 3. Non-resonant weights at 256x64 are 1,606 mm2 at 5 V and the finest
    #    pitch: sixteen standard dies and two fifths of the M1000.  Of the link
    #    model's candidates only the two smallest fit a standard die that way.
    nonres = strip_mm2(k * n, mzm_mm(5.0), P_CELL)
    assert abs(nonres - 1605.6) < 0.1, nonres
    assert round(nonres / std) == 16 and abs(nonres / M1000_MM2 - 0.40) < 0.005
    cell = mzm_mm(5.0) * P_CELL * MM
    assert cells_that_fit(std, cell) == 1020 and cells_that_fit(pkg, cell) == 1260
    fit = [g for g in link.GEOMETRIES if g[0] * g[1] <= cells_that_fit(std, cell)]
    assert fit == [(8, 8), (64, 8)], fit

    # 4. The board plan's two areas, reproduced: 164 mm2 at 100 um, more than
    #    the reference package, and 10.2 at 25 -- a factor of sixteen.
    assert abs(field_mm2(k * n, P_FLIP) - 163.84) < 1e-9
    assert abs(field_mm2(k * n, P_CELL) - 10.24) < 1e-9
    assert field_mm2(k * n, P_FLIP) > pkg
    assert abs(field_mm2(k * n, P_FLIP) / field_mm2(k * n, P_CELL) - 16.0) < 1e-9

    # 5. The resonant and non-resonant readings are two orders apart at the
    #    same pitch, by exactly the modulator's length over the pitch.
    ratio = nonres / field_mm2(k * n, P_CELL)
    assert abs(ratio - mzm_mm(5.0) / (P_CELL * MM)) < 1e-6 and ratio > 100

    # 6. At a 25 um lateral pitch the inputs outweigh the weights at every swing;
    #    at flip-chip pitch the weights lead at 5 V and not at 1 V.
    for v in SWINGS_V:
        w, i = die_mm2(k, n, P_CELL, v)
        assert i > w, (v, w, i)
    w, i = die_mm2(k, n, P_FLIP, 5.0)
    assert w > i and die_mm2(k, n, P_FLIP, 1.0)[1] > w

    # 7. pitch_ceiling_um() inverts die_mm2(): at the ceiling the die is the
    #    area asked for, for every candidate and every kind of input.
    for gk, gn in link.GEOMETRIES:
        for v in (1.0, 2.0, 5.0, None):
            p = pitch_ceiling_um(gk, gn, v, std)
            assert abs(sum(die_mm2(gk, gn, p, v)) - std) < 1e-6 * std, (gk, gn, v)

    # 8. Reading 3's three pitches, and where they sit in UCIe's ranges.
    c5, c2 = pitch_ceiling_um(k, n, 5.0, std), pitch_ceiling_um(k, n, 2.0, std)
    cr = pitch_ceiling_um(k, n, None, std)
    assert round(c5) == 53 and round(c2) == 33 and round(cr) == 78, (c5, c2, cr)
    assert UCIE_A_RANGE[0] < c2 < c5 < UCIE_A_RANGE[1]
    assert UCIE_A_RANGE[1] < cr < UCIE_S_RANGE[0]
    # At 1 V the modulator is longer than the die's side, so its area is moot;
    # at 2 V it fits the side with 0.2 mm to spare.
    assert mzm_mm(1.0) > max(STD_DIE_MM) and 0 < STD_DIE_MM[0] - mzm_mm(2.0) < 0.25

    # 9. Side by side: 16,704 lines want a quarter of a metre of shared edge,
    #    and the reference package's long side carries 866.
    lines = connections(k, n)
    assert lines == 16_704
    assert abs(side_by_side_mm(lines) - 250.56) < 1e-9
    assert lines_across_mm(PKG_MM[1]) == 866
    assert abs(side_by_side_mm(lines_across_mm(PKG_MM[1])) - PKG_MM[1]) < 2 * RDL_LINE_SPACE * MM

    # 10. The facet: a fiber a row is the board plan's 64 mm and 32.5 mm, 16
    #     lines a fiber is 16 fibers and 4 mm, and a multicore of 37 is seven.
    assert facet_mm(k, 250) == 64.0 and abs(facet_mm(k, 127) - 32.512) < 1e-9
    assert fibers(k, 16) == 16 and facet_mm(16, 250) == 4.0
    assert fibers(k, MULTICORE) == 7 and fibers(k, LINES_A_BUS) == 5
    assert ARRAYS_A_DIE * (ARRAY_CHANNELS - EDGE_SHUNTS) == 40

    # 11. The UCIe figures agree with each other only as bytes and both ways:
    #     a module is 2 x 16 lanes x 32 Gb/s = 128 GByte/s, and that over its
    #     width and its area is the tutorial's 224 and 145.
    both_ways_gbyte = 2 * link.LANES * link.GT_S / 8
    assert both_ways_gbyte == 128
    assert round(both_ways_gbyte / (UCIE_W_UM * MM)) == UCIE_EDGE_GBS_MM
    assert round(both_ways_gbyte / module_mm2()) == UCIE_AREA_GBS_MM2
    assert UCIE_S_RANGE[0] <= P_UCIE_S <= UCIE_S_RANGE[1]

    # 12. X2's five modules, as this model counts them, and what they take.
    assert link_modules(k, n, 64) == shot.X2_MODULES == 5
    assert abs(5 * module_mm2() - 4.40) < 0.005
    assert abs(5 * UCIE_W_UM * MM - 2.86) < 0.005
    assert 72 < module_sites() < 73

    # 13. Through the mold the link is three times the dense tile's weights, and
    #     B8's thirty modules have more bumps than the package has C4 sites.
    via = escape_mm2(5, TMV_PITCH)
    assert abs(via - 32.7) < 0.1, via
    assert 3.0 < via / field_mm2(k * n, P_CELL) < 3.3
    c4_sites = int(pkg / (C4_PITCH * MM) ** 2)
    assert c4_sites == 1976
    assert B8_FLIP_MODULES * module_sites() > c4_sites > 5 * module_sites()

    # 14. The light: at most 0.09 W/mm2 on the densest die, about a fifteenth of
    #     the ring link's density, and less on every larger one.  It was 0.19
    #     and a seventh while pta_shot_rate.py read v1's receiver noise as a
    #     quarter of an 8-bit LSB; it is half of one.
    hi = shot.laser_power(v1, shot.X2_FS, n, shot.LOSS_DB[1], 0.5)
    dense = sum(die_mm2(k, n, P_CELL, 5.0))
    assert abs(hi / dense - 0.093) < 0.001, hi / dense
    assert 15.0 < RING_LINK_W_MM2 / (hi / dense) < 16.0
    for p, _ in BOND_PITCHES:
        assert hi / sum(die_mm2(k, n, p, 5.0)) <= hi / dense + 1e-12

    print("All checks pass.")


if __name__ == "__main__":
    main()
