"""
S3 of board_program_plan.md: the dispatch cost model.

Which device should a GEMM go to -- the PTA chiplet, the GRX-G100 or the c930's
NPU?  The plan's row asks for "the section 2.1 model with X2's link terms and
C1's accuracy", with its predictions checked on rev 0.  This is the model and
the predictions, stated before rev 0 exists so that rev 0 has something to check.

A GEMM here is one layer on one batch: M rows of K inputs against a K x N weight
set, M * K * N multiply-accumulates (MACs).

THE FOUR TERMS ARE NOT THE SAME KIND OF NUMBER, and the tables say which is which.

  the NPU    MEASURED in RTL.  Section 2.1 of pta_cpu_integration.md with the
             digital array's own constants reproduces the core's cycles at
             M=64 N=8 K=256 (pta_tw_sweep.py asserts it); F0 and F2 measured the
             DMA around it and the host's share of a GEMM on the SoC
             (pta_feed_model.py).
  the GPU    MEASURED in SimX, by grxgpu, at two shapes, in fp16:
             grxgpu/docs/proposals/grxgpu_tensor_engine.md sections 8.1 and 8.4,
             the G100 configuration of 8 clusters of 16 cores.  The clock is
             CONFIGURED: VX_CFG_PLATFORM_CLOCK_RATE = 400 in grxgpu's
             VX_config.toml, which is what the device reports and not a
             measurement of silicon.  A launch's fixed cost was MEASURED in simx
             by grxcp, on other kernels at another shape
             (developer_interface.md section 3), so here it is an INDICATION.
  the PTA    PREDICTED.  There is no chiplet.  Its time is X2's link
             (pta_chiplet_link.py), the weight path of pta_shot_rate.py
             section 5, and the board plan's section 4.3: a 256 x 64 tile at
             1 GS/s with two weight banks.  Each of those is a candidate or an
             assumption of the document it comes from.
  accuracy   MEASURED on one network: grx930's harness on D3, 784-100-10 on
             MNIST (the board plan's section 4.3), ON THE TILE THIS MODEL
             PRICES, 256 x 64.  It first quoted the 8 x 8 tile's figures, which
             were the only ones there were; grx930's design note section 5 now
             has both.  No other network has a number, and this model does not
             supply one.

WHAT IT DOES NOT PRICE, and says so where it matters.

  energy                    pta_power.py has the chiplet's watts and nothing in
                            the three repositories has the GPU's, so a comparison
                            in joules would have one side.
  the chiplet's command     Proposed to grxgpu and not built
                            (grxgpu/docs/proposals/pta_chiplet_host_path.md), so
                            it has no cost to quote.  Sections 2 and 4 turn that
                            around: what the cost may be, and what it has to be.
  the host's path to the    The GPU's fixed cost here is on the device.  What the
  GPU, over link 1          driver and the link add to a launch has not been
                            measured.  It cancels between the GPU and the chiplet,
                            which share it, and it does not cancel against the NPU
                            (section 5).

Standard library only.  Run:  python3 docs/designs/pta_dispatch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pta_chiplet_link as link
import pta_feed_model as feed
import pta_shot_rate as shot
import pta_tw_sweep as sweep

# ---- the GPU ---------------------------------------------------------------
# grxgpu_tensor_engine.md 8.1 and 8.4: SimX, G100 configuration, fp16 in and
# fp32 out, current HEAD.  (M, K, N) and the run's cycles.
GPU_MEASURED = (((512, 512, 512), 13_760_999), ((1024, 512, 1024), 55_253_809))
GPU_MHZ = 400                     # CONFIGURED: VX_CFG_PLATFORM_CLOCK_RATE
# developer_interface.md 3: simx, grxcp's own kernels at S = 16.  The fixed cost
# inside a stage's span, and the preamble before its first warp.  An indication:
# another kernel, another shape, and the preamble "is a figure for this shape
# rather than a property".
GPU_LAUNCH_SPAN = 2_776
GPU_LAUNCH_PREAMBLE = 9_418

# ---- the NPU ---------------------------------------------------------------
# The digital array, 8 x 8, at 100 MHz.  What the SoC builds takes at most this
# shape in one command (pta_tw_sweep.py: MAX_M=8, MAX_K=16, MAX_N=12).
NPU_ARRAY = (sweep.NUM_ROWS, sweep.NUM_COLS)
NPU_MAX = dict(m=sweep.SOC_M, k=sweep.SOC_K, n=sweep.SOC_N)
NPU_READ_A = feed.LEVELS["NPU bench"]["read_a"]       # the rows read before launch
# F0's whole GEMM at M=64 N=8 K=256, with F2's rebuilt write burst in place of
# the one F0 measured: the reading the model below is held to.
NPU_F0_GEMM = (feed.MEASURED[("NPU bench", sweep.DIGITAL_RUN)][2]
               - feed.LEVELS["NPU bench"]["write_c"] + feed.BENCH_F2["write_c"])
# F2: the wall clock less the DMA's count, a GEMM, on the SoC.  The submit's
# MMIO writes and the drain's poll.
NPU_HOST = min(d["wall"] - d["gemm"] for d in feed.MEASURED_F2.values())

# ---- the PTA chiplet -------------------------------------------------------
TILE = (256, 64)                  # the working geometry: the board plan's B10, 2026-10-05
FS = shot.X2_FS                   # 1 GS/s.  Not settled: B10 fixed the tile and not the rate.
BANKS = 2                         # 4.3: "held to two weight banks"
BATCH = link.BIG["mb"]            # 64, the batch X2 and 4.3 are quoted at
# 4.3: "a write path one output's column wide at batch 64".
PER_BEAT = shot.per_beat_two_banks(TILE[0], TILE[1], BATCH)
LINK1_GBS = 64.0                  # board_icd.md, link 1: "about 64 GB/s a direction before overhead"

# ---- accuracy, the board plan's 4.3 ------------------------------------------
D3 = ((784, 100), (100, 10))
D3_EXACT = 97.45                  # 8-bit operands, 6-bit weights, nothing else impaired
# v1's loss in points on a 256 x 64 tile, after an hour of TFLT's drift and
# after six minutes of it (grx930's `pta_mnist.sh geometry`, five networks,
# standard errors 0.06 and 0.04), and the same on the c930 core's 8 x 8 tile,
# which is what this quoted until the harness could be asked for another.
D3_V1_LOSS = {"hourly calibration": 0.24, "every six minutes": 0.21}
D3_V1_LOSS_8X8 = {"hourly calibration": 0.81, "every six minutes": 0.37}

WORK = (
    # name, M, K, N
    ("D3 layer 1", BATCH, 784, 100),
    ("D3 layer 2", BATCH, 100, 10),
    ("section 6.2's shape", sweep.M, sweep.K, sweep.N),
    ("the SoC NPU's largest", sweep.SOC_M, sweep.SOC_K, sweep.SOC_N),
    ("4096 x 4096", link.BIG["mb"], link.BIG["kin"], link.BIG["nout"]),
)


def macs(m, k, n):
    return m * k * n


# ---- the GPU ---------------------------------------------------------------
def gpu_cycles_per_mac():
    """Both measured runs as one rate.  checks() holds them to each other."""
    return (sum(c for _, c in GPU_MEASURED)
            / sum(macs(*shape) for shape, _ in GPU_MEASURED))


def gpu_launch_s(preamble=True):
    """A launch's fixed cost on the device.  The preamble is the part that
    "scales with occupancy", so a launch of one small GEMM may not pay it all."""
    return (GPU_LAUNCH_SPAN + (GPU_LAUNCH_PREAMBLE if preamble else 0)) / (GPU_MHZ * 1e6)


def gpu_s(m, k, n, launch=True):
    """One launch.  The measured runs were proportional to MACs, so this is."""
    return (gpu_cycles_per_mac() * macs(m, k, n) / (GPU_MHZ * 1e6)
            + (gpu_launch_s() if launch else 0.0))


def slack_s(m, k, n, **kw):
    """How much more than a launch the chiplet's command may cost and still win.

    The GPU's arithmetic less the chiplet's whole time.  Both are reached through
    the GPU's command processor, so what they share cancels and what is left is
    the difference between a launch and a command to the chiplet.
    """
    return gpu_s(m, k, n, launch=False) - pta(m, k, n, **kw)["seconds"]


# ---- the NPU ---------------------------------------------------------------
def npu_cycles(m, k, n):
    """The engine's cycles for one GEMM: the core and the DMA around it.

    The core is section 2.1's interchanged order with the restore unfolded, which
    is what the DMA-fed array runs (F0's 71,734 is 71,168 plus its A-row wait).
    The DMA reads B before the launch and writes C after it; A streams behind the
    compute.  The A-row wait is left out -- 566 cycles of F0's 73,601 -- because
    the race that sets it is modelled at one shape only.
    """
    rows, cols = NPU_ARRAY
    kt, nt = -(-k // rows), -(-n // cols)
    core = sweep.interchanged(sweep.SCAN, sweep.DIGITAL_RUN, sweep.TD,
                              m=m, nt=nt, kt=kt, restore=True)
    read_b = feed.feed_part(feed.beats_of(k * n, 1), "read")
    write_c = feed.feed_part(feed.c_beats_of(m * n), "write")
    return NPU_READ_A + read_b + core + feed.HANDOFF + write_c + feed.DONE


def npu_s(m, k, n, host=True):
    return (npu_cycles(m, k, n) + (NPU_HOST if host else 0)) * sweep.CYCLE_NS * 1e-9


def npu_takes(m, k, n):
    """Whether the NPU the SoC builds can be asked for this in one command."""
    return m <= NPU_MAX["m"] and k <= NPU_MAX["k"] and n <= NPU_MAX["n"]


# ---- the PTA chiplet -------------------------------------------------------
def pta(m, kin, nout, tile=TILE, fs=FS, mods=1, per_beat=PER_BEAT, banks=BANKS,
        resident=False, padded=True):
    """One GEMM on the chiplet, from the command arriving to the results back.

    A weight set is one (K tile, N tile) of the walk: k * n cells, shot m times.
    Writing a set takes ceil(k * n / per_beat) beats, a beat being one shot
    period (pta_shot_rate.py section 5).  With one bank the tile writes, then
    shoots.  With two the next set loads behind the current set's shots, so only
    the first set's write is seen, unless a set takes longer to write than to
    shoot.  Resident means the sets are already in banks on the chiplet: nothing
    is written and no weight crosses the link, which needs a bank a set.

    The link is X2's: B4's split, so the activations cross once and the results
    once.  It caps the tile and does not add to it, as in X2.  X2 counts a weight
    set whole, padding and all; padded=False sends only the layer's own weights.
    """
    k, n = tile
    t = link.layer_traffic(kin, nout, m, k, n)
    sets = t["shots"] // m
    wb = -(-(k * n) // per_beat)
    if resident:
        assert sets <= banks, (sets, banks)
        beats = sets * m
    elif banks >= 2:
        beats = wb + (sets - 1) * max(m, wb) + m
    else:
        beats = sets * (wb + m)
    weights = t["weights"] if padded else kin * nout * link.W_BYTES
    into = t["acts"] if resident else weights + t["acts"]
    tile_s = beats / fs
    link_s = max(into, t["out"]) / (mods * link.MODULE_GBS * 1e9)
    return dict(sets=sets, shots=t["shots"], beats=beats, tile_s=tile_s,
                link_s=link_s, into=into, out=t["out"],
                binds="link" if link_s > tile_s else "tile",
                fill=kin * nout / (sets * k * n),
                seconds=max(tile_s, link_s) + link.ROUND_TRIP_NS * 1e-9)


def cross_s(m, k, n):
    """Operands from CPU memory to the GPU's and results back, over link 1."""
    return (m * k + k * n + m * n * link.ACC_BYTES) / (LINK1_GBS * 1e9)


def npu_gpu_break_even(preamble=True):
    """MACs above which a launch on the GPU beats a command to the NPU.

    Fixed costs against rates.  The NPU's rate is F0's whole GEMM over its MACs.
    Negative means the GPU's fixed cost is the smaller and it is ahead at every
    size.
    """
    npu_rate = NPU_F0_GEMM * sweep.CYCLE_NS * 1e-9 / macs(sweep.M, sweep.K, sweep.N)
    gpu_rate = gpu_cycles_per_mac() / (GPU_MHZ * 1e6)
    fixed = gpu_launch_s(preamble) - NPU_HOST * sweep.CYCLE_NS * 1e-9
    return fixed / (npu_rate - gpu_rate)


def npu_keeps_s():
    """What a launch has to cost, in all, for the NPU to keep its largest command."""
    shape = (NPU_MAX["m"], NPU_MAX["k"], NPU_MAX["n"])
    return npu_s(*shape) - gpu_s(*shape, launch=False)


def batch_for_half(kin, nout, command_s, **kw):
    """The smallest batch at which the chiplet's own time reaches `command_s`."""
    lo, hi = 1, 1
    while pta(hi, kin, nout, **kw)["seconds"] < command_s:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if pta(mid, kin, nout, **kw)["seconds"] < command_s:
            lo = mid + 1
        else:
            hi = mid
    return lo


def d3_submitted_s(round_trips, launch_s, round_trip_s):
    """D3 at a batch of 64 as a host sees it.

    The chiplet's two GEMMs, the host's round trips, and whatever turns the
    first layer's sums into the second's operands: a launch on the GPU costing
    `launch_s` on the device, or None with that step on the chiplet.  The step's
    own arithmetic is not priced.
    """
    chiplet = sum(pta(BATCH, a, b)["seconds"] for a, b in D3)
    return chiplet + round_trips * round_trip_s + (launch_s or 0.0)


def fmt_s(s):
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if s >= scale * 0.9995:
            return f"{s / scale:.3g} {unit}"
    return f"{s * 1e12:.3g} ps"


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    k, n = TILE
    c_gpu = gpu_cycles_per_mac()
    host_s = NPU_HOST * sweep.CYCLE_NS * 1e-9

    section("1. The three devices: a rate, a fixed cost, and where each number is from")
    f0 = macs(sweep.M, sweep.K, sweep.N)
    big = pta(link.BIG["mb"], link.BIG["kin"], link.BIG["nout"])
    big_macs = macs(link.BIG["mb"], link.BIG["kin"], link.BIG["nout"])
    print(f"  {'device':<14}{'a MAC':>10}{'fixed, a GEMM':>16}   what the numbers are")
    print(f"  {'NPU':<14}{fmt_s(NPU_F0_GEMM * sweep.CYCLE_NS * 1e-9 / f0):>10}{fmt_s(host_s):>16}"
          f"   MEASURED, RTL: F0's GEMM at M=64 N=8 K=256; F2's host share on the SoC")
    print(f"  {'GPU':<14}{fmt_s(c_gpu / (GPU_MHZ * 1e6)):>10}{fmt_s(gpu_launch_s()):>16}"
          f"   MEASURED, SimX, fp16: {c_gpu:.4f} cycles a MAC; clock CONFIGURED at"
          f" {GPU_MHZ} MHz; launch an INDICATION")
    print(f"  {'PTA chiplet':<14}{fmt_s(big['seconds'] / big_macs):>10}{'not known':>16}"
          f"   PREDICTED: {k} x {n}, {FS / 1e9:.0f} GS/s, {BANKS} banks, one module,"
          f" on the 4096-square layer")
    for shape, cycles in GPU_MEASURED:
        print(f"    GPU, measured: M={shape[0]} K={shape[1]} N={shape[2]}  {cycles:,} cycles,"
              f" {cycles / macs(*shape):.4f} a MAC")
    print("    The chiplet's GEMM is int8 and the GPU's int8 path has no measurement in")
    print("    that document, so its fp16 rate stands in for it.")
    print(f"    The GPU's launch is {GPU_LAUNCH_SPAN:,} cycles inside a stage's span and"
          f" {GPU_LAUNCH_PREAMBLE:,} before its first warp,")
    print("    measured by grxcp in simx on other kernels at S = 16.  The G100 runs above")
    print("    include their own launch and do not separate it.")
    print(f"    The NPU the SoC builds takes at most M={NPU_MAX['m']} K={NPU_MAX['k']}"
          f" N={NPU_MAX['n']} in a command: {macs(*NPU_MAX.values()):,} MACs.")
    print("    The chiplet's fixed cost is its command path, which is proposed and not built.")

    section("2. Each GEMM on each device, with no fixed cost on any of them")
    print(f"  chiplet as 4.3 has it: {k} x {n}, {FS / 1e9:.0f} GS/s, {BANKS} banks,"
          f" {PER_BEAT} cells written a beat, one module")
    print(f"  {'GEMM':<24}{'MACs':>15}{'NPU':>10}{'GPU':>10}{'PTA':>10}{'bound by':>10}"
          f"{'sets':>6}{'fill':>6}{'GPU/PTA':>9}{'slack':>10}")
    for name, m, kk, nn in WORK:
        p = pta(m, kk, nn)
        npu = fmt_s(npu_s(m, kk, nn, host=False)) + ("" if npu_takes(m, kk, nn) else "*")
        g = gpu_s(m, kk, nn, launch=False)
        print(f"  {name:<24}{macs(m, kk, nn):>15,}{npu:>10}{fmt_s(g):>10}"
              f"{fmt_s(p['seconds']):>10}{p['binds']:>10}{p['sets']:>6}{p['fill']:>6.0%}"
              f"{g / p['seconds']:>9,.1f}{fmt_s(slack_s(m, kk, nn)):>10}")
    print("  * more than the SoC's NPU takes in one command: the array's rate, were it asked.")
    print("  Fill is the share of the programmed cells the layer uses: a 100 x 10 layer")
    print("  lights 6% of one set.  Slack is the GPU's arithmetic less the chiplet's whole")
    print("  time: how much MORE than a launch a command to the chiplet may cost before")
    print("  the GPU is the quicker of the two.  Both are reached through the GPU's command")
    print("  processor, so what the two paths share cancels.")
    print(f"  To each column add its fixed cost: {fmt_s(host_s)} for the NPU, {fmt_s(gpu_launch_s())}"
          f" for the GPU, and for the")
    print("  chiplet a figure nobody has (section 4).")

    section("3. What moves the chiplet's column")
    print(f"  {'GEMM':<24}{'as 4.3':>10}{'0.1 GS/s':>10}{'scan, 1':>10}{'5 modules':>11}"
          f"{'unpadded':>10}{'resident':>10}{'GPU':>10}")
    for name, m, kk, nn in WORK:
        base = pta(m, kk, nn)
        res = fmt_s(pta(m, kk, nn, resident=True)["seconds"]) if base["sets"] <= BANKS else "no"
        print(f"  {name:<24}{fmt_s(base['seconds']):>10}"
              f"{fmt_s(pta(m, kk, nn, fs=FS / 10)['seconds']):>10}"
              f"{fmt_s(pta(m, kk, nn, banks=1, per_beat=1)['seconds']):>10}"
              f"{fmt_s(pta(m, kk, nn, mods=5)['seconds']):>11}"
              f"{fmt_s(pta(m, kk, nn, padded=False)['seconds']):>10}{res:>10}"
              f"{fmt_s(gpu_s(m, kk, nn, launch=False)):>10}")
    print("  'scan, 1' is one bank written a cell a beat, as the c930's scan writes its")
    print("  tile.  'unpadded' sends a layer's own weights and not whole sets: X2 counts")
    print(f"  the padding.  'resident' needs a bank a weight set, and 4.3 holds the chip to {BANKS}:")
    print("  D3's first layer is 8 sets and the 4096-square one is 1,024.")

    section("4. The command path: what it has to cost for the tile's speed to be seen")
    print("  The chiplet's time above is the tile and the link.  A command to it also")
    print("  costs whatever the path from the host costs, and that path does not exist.")
    print("  The program has measured two per-command costs, and neither is this one:")
    print(f"    {fmt_s(host_s):>8}  the c930's MMIO path to its own NPU ({NPU_HOST:,} cycles at 100 MHz)")
    print(f"    {fmt_s(gpu_launch_s()):>8}  a launch on the GPU, in simx"
          f" ({GPU_LAUNCH_SPAN + GPU_LAUNCH_PREAMBLE:,} cycles at {GPU_MHZ} MHz)")
    print(f"  {'GEMM':<24}{'chiplet':>10}{'its share at ' + fmt_s(host_s):>24}"
          f"{'at ' + fmt_s(gpu_launch_s()):>14}{'batch for half':>20}")
    for name, m, kk, nn in WORK:
        p = pta(m, kk, nn)["seconds"]
        halves = " / ".join(f"{batch_for_half(kk, nn, c):,}" for c in (host_s, gpu_launch_s()))
        print(f"  {name:<24}{fmt_s(p):>10}{p / (p + host_s):>24.0%}"
              f"{p / (p + gpu_launch_s()):>14.0%}{halves:>20}")
    print("  'Batch for half' is the batch at which the chiplet's own time equals the")
    print("  command's, at each of the two costs.  Below it the command is most of the GEMM.")

    section("5. The NPU against the GPU: not settled by these numbers")
    m, kk, nn = sweep.SOC_M, sweep.SOC_K, sweep.SOC_N
    print("  The two fixed costs are not the same kind of thing.")
    print(f"    {fmt_s(host_s):>8}  the NPU's, and it is the HOST's: MMIO writes and a poll, on the SoC")
    print(f"    {fmt_s(gpu_launch_s(False)):>8}  the GPU's, ON THE DEVICE: what a launch pays inside a span")
    print(f"    {fmt_s(gpu_launch_s()):>8}  the same with the preamble, which scales with occupancy and was")
    print("              measured at S = 16: a launch of one small GEMM may pay less of it")
    print("  Nothing has measured the host's path to the GPU, which is on top of either.")
    print(f"  With the preamble a launch beats the NPU above {npu_gpu_break_even():,.0f} MACs, and the"
          f" NPU the SoC builds")
    print(f"  takes {macs(*NPU_MAX.values()):,}: it keeps everything it can be asked for.  Without it the"
          f" GPU's fixed cost is")
    print("  under the NPU's host path and the GPU is ahead at every size.")
    print(f"  At the NPU's largest shape: NPU {fmt_s(npu_s(m, kk, nn))}; GPU"
          f" {fmt_s(gpu_s(m, kk, nn, launch=False))} of arithmetic and its launch.")
    print(f"  So the NPU keeps that command if a launch costs more than {fmt_s(npu_keeps_s())} in all,"
          f" the host's path")
    print(f"  included: {npu_keeps_s() * GPU_MHZ * 1e6:,.0f} cycles at {GPU_MHZ} MHz.  That is rev 0's to measure.")

    section("6. D3 end to end, a batch of 64, fixed costs included")
    for label, fn, note in (("NPU*", npu_s, ""), ("GPU", gpu_s, "; its launches an indication")):
        total = sum(fn(BATCH, a, b) for a, b in D3)
        print(f"  {label:<26}{fmt_s(total):>10}   {D3_EXACT:.2f}%   integer arithmetic, exact{note}")
    tile = sum(pta(BATCH, a, b)["seconds"] for a, b in D3)
    for label, loss in D3_V1_LOSS.items():
        print(f"  {'PTA, ' + label:<26}{fmt_s(tile):>10}   {D3_EXACT - loss:.2f}%   "
              f"{loss:.2f} points lost; and two commands")
    print(f"  Those two are the {TILE[0]} x {TILE[1]} tile's.  On the c930 core's 8 x 8 they are "
          + " and ".join(f"{D3_EXACT - x:.2f}%" for x in D3_V1_LOSS_8X8.values())
          + ": drift costs a small tile more.")
    print(f"  With a command at the two measured costs the chiplet's D3 is"
          f" {fmt_s(tile + 2 * host_s)} to {fmt_s(tile + 2 * gpu_launch_s())}.")
    print(f"  An activation stage on the GPU puts a launch between the two GEMMs:"
          f" {fmt_s(gpu_launch_s())},")
    print(f"  {gpu_launch_s() / tile:.0f} times the chiplet's time for both.  B4 put the stage on"
          f" the chiplet for the link's sake;")
    print("  this is the same answer from the time side.  And with one command queue on")
    print("  the GPU the launch and the GEMMs are serial, so nothing overlaps to hide it.")
    print("  * the array's rate: the SoC's NPU cannot be asked for either layer in one command.")

    section("7. One round trip for the network: what a command list is worth")
    lo, hi = gpu_launch_s(False), gpu_launch_s()
    print("  grxgpu's runtime already submits a list of commands under one doorbell and")
    print("  one completion poll (vx_enqueue_commands, and vx_enqueue_draw as one command")
    print("  the device expands).  The host-path proposal's GEMM was a call of its own.  Its")
    print("  amendment makes it a member of the list.")
    print(f"  D3 at a batch of {BATCH}.  The host's round trip is the c930's measured"
          f" {fmt_s(host_s)}, a STAND-IN:")
    print("  nobody has measured the G100's.  The step between the layers is a launch,")
    print(f"  {fmt_s(lo)} to {fmt_s(hi)} on the device (section 5); its arithmetic is not priced.")
    print(f"  {'how it is submitted':<40}{'round trips':>12}{'total':>20}{'the chiplet is':>20}")
    rows = (("a call a GEMM, and the launch", 3, True),
            ("one list", 1, True),
            ("one list, the step on the chiplet", 1, False))
    for label, trips, launch in rows:
        ts = [d3_submitted_s(trips, x, host_s) for x in ((lo, hi) if launch else (None,))]
        total = " to ".join(fmt_s(t) for t in ts)
        share = " to ".join(f"{tile / t:.0%}" for t in ts)
        print(f"  {label:<40}{trips:>12}{total:>20}{share:>20}")
    print(f"  The list removes two round trips, {fmt_s(2 * host_s)}, whatever the launch costs.")
    print("  The last row needs the chiplet's activation stage.  The register map's section")
    print("  8 specifies it and the twin runs it (the plan's X6).  grxgpu has been asked to")
    print("  carry it and has not answered, so no host can submit that row yet.")

    findings()
    checks()


def findings():
    host_s = NPU_HOST * sweep.CYCLE_NS * 1e-9
    d3 = sum(pta(BATCH, a, b)["seconds"] for a, b in D3)
    ratio = {name: gpu_s(m, kk, nn, launch=False) / pta(m, kk, nn)["seconds"]
             for name, m, kk, nn in WORK}
    layers = [r for name, r in ratio.items() if name != "the SoC NPU's largest"]
    l2 = (BATCH,) + D3[1]
    print()
    print("What this says, seven readings.")
    print()
    print("  1. ABOVE TENS OF THOUSANDS OF MACS, TIME DOES NOT CHOOSE.  On every layer here")
    print(f"     the chiplet as 4.3 has it is {min(layers):.0f} to {max(layers):.0f} times the GPU's"
          f" arithmetic, and its command")
    print(f"     may cost {fmt_s(slack_s(*l2))} to {fmt_s(slack_s(*WORK[-1][1:]))} more than a launch"
          f" before that turns.  What chooses is")
    print("     whether the GEMM may go there at all: int8 operands, and a network whose")
    print("     loss at this budget has been measured and accepted.  That is one network.")
    print()
    print("  2. AT 1,536 MACS THEY ARE LEVEL.  The chiplet spends its time moving one")
    print(f"     padded weight set, and the GPU's arithmetic takes as long:"
          f" {ratio[WORK[3][0]]:.2f} to one.")
    print()
    print("  3. THE CHIPLET'S GEMM IS ITS COMMAND.  D3's two layers at batch 64 are")
    print(f"     {fmt_s(d3)} on the chiplet.  Its two commands, at either cost the program has"
          f" measured,")
    print(f"     are {2 * host_s / d3:.1f} or {2 * gpu_launch_s() / d3:.0f} times that.  So what"
          f" rev 0 can measure of the chiplet's speed")
    print("     is the command path's, and the proposal to grxgpu is where that is decided.")
    print("     One list for the network takes two of D3's three round trips away (section")
    print("     7); keeping the step between its layers on the chiplet takes the launch too.")
    print()
    print("  4. THE LINK BINDS BEFORE THE TILE DOES, at one module, on every shape, and")
    print("     what crosses is weights.  D3's first layer programs eight whole sets to")
    print("     use 60% of them.  The batch is the lever, as X2 said: the weights cross")
    print("     once a batch whatever its size.")
    print()
    print("  5. THE WRITE PATH IS WORTH AS MUCH AS THE SHOT RATE.  A tenth of the shot rate")
    print(f"     leaves D3's second layer {gpu_s(*l2, launch=False) / pta(*l2, fs=FS / 10)['seconds']:.0f}"
          f" times the GPU.  The c930's one-cell scan on one bank")
    print(f"     leaves it {gpu_s(*l2, launch=False) / pta(*l2, banks=1, per_beat=1)['seconds']:.1f}"
          f" times: level.  4.3's two banks and its 256 cells a beat are what")
    print("     the chiplet's column rests on.")
    print()
    print("  6. THE NPU'S SHARE IS NOT SETTLED.  Its fixed cost is the host's and measured;")
    print("     the GPU's is the device's, in simx, at another shape, and the host's path")
    print("     to the GPU is unmeasured.  The NPU keeps its largest command if a launch")
    print(f"     costs more than {fmt_s(npu_keeps_s())} in all.  The two figures in hand are"
          f" {fmt_s(gpu_launch_s(False))} and {fmt_s(gpu_launch_s())},")
    print("     one on each side.")
    print()
    print("  7. PLACEMENT IS A RULE AND NOT A SEARCH, on these numbers:")
    print("       the chiplet   an int8 GEMM in a network measured at this budget, batched")
    print("       the NPU       a GEMM it can take, if rev 0 puts a launch above reading 6's figure")
    print("       the GPU       everything else")
    print("     It is applied above grxBLAS, by whoever sets the device: the library runs a")
    print("     GEMM where it is told and does not fall back.")
    print()
    print("  WHAT REV 0 CAN AND CANNOT CHECK.  It can check the GPU's rate and launch, the")
    print("  NPU's, the break-even between them, and the cost of a command to the PTA.  It")
    print("  cannot check the chiplet's column: on rev 0 the PTA is the error-model tile,")
    print("  and its time is whatever PTA_TW and PTA_TS are set to.")


def checks():
    """Every claim above, as an assert."""
    k, n = TILE
    assert (k, n, FS, BANKS, BATCH, PER_BEAT) == (256, 64, 1e9, 2, 64, 256)
    host_s = NPU_HOST * sweep.CYCLE_NS * 1e-9
    small = WORK[3]
    assert small[0] == "the SoC NPU's largest" and macs(*small[1:]) == 1_536
    layers = [w for w in WORK if w is not small]

    # 1. The GPU's two runs are one rate, to half a percent, so a proportional
    #    model is what they support.
    rates = [c / macs(*shape) for shape, c in GPU_MEASURED]
    assert max(rates) / min(rates) < 1.005, rates
    c_gpu = gpu_cycles_per_mac()
    assert abs(c_gpu - 0.1028) < 0.0001, c_gpu
    assert abs(gpu_launch_s() * 1e6 - 30.5) < 0.05

    # 2. The NPU's model is held to what was measured: the core to the cycle at
    #    F0's shape, and the whole GEMM short by exactly the A-row wait it leaves
    #    out.
    assert sweep.interchanged(sweep.SCAN, sweep.DIGITAL_RUN, sweep.TD,
                              restore=True) == sweep.MEASURED_INTERCHANGED
    f0 = (sweep.M, sweep.K, sweep.N)
    arow = feed.MEASURED[("NPU bench", sweep.DIGITAL_RUN)][1]
    assert NPU_F0_GEMM - npu_cycles(*f0) == arow == 566, (NPU_F0_GEMM, npu_cycles(*f0))
    assert all(1_085 <= d["wall"] - d["gemm"] <= 1_105 for d in feed.MEASURED_F2.values())
    assert abs(host_s * 1e6 - 10.9) < 0.05

    # 3. The chiplet's model is X2's where X2 has a number: the 4096-square layer
    #    is bound by the link, at X2's capped shot rate, and takes X2's shots.
    big = pta(link.BIG["mb"], link.BIG["kin"], link.BIG["nout"])
    assert big["binds"] == "link" and big["shots"] == 65_536 and big["sets"] == 1_024
    cap = link.max_shot_rate(k=k, n=n, mods=1, **link.BIG)
    assert abs(big["shots"] / big["link_s"] - cap) < 1.0, (big["shots"] / big["link_s"], cap)
    #    The weight path is pta_shot_rate.py's: one bank shoots the share of its
    #    time that its section 5 tabulates.
    for per_beat in (1, 64, 256, 16_384):
        one = pta(BATCH, k, n, banks=1, per_beat=per_beat)
        assert abs(BATCH / one["beats"] - shot.duty_one_bank(k, n, BATCH, per_beat)) < 1e-12
    #    Two banks at 4.3's write path never wait: only the first set's write shows.
    two = pta(BATCH, 4 * k, 4 * n)
    assert two["sets"] == 16 and two["beats"] == two["shots"] + BATCH

    # 4. Reading 1.  On the four layers the chiplet is 33 to 933 times the GPU's
    #    arithmetic, and its command may cost 16 us to 276 ms more than a launch.
    ratio = {w[0]: gpu_s(*w[1:], launch=False) / pta(*w[1:])["seconds"] for w in WORK}
    lay = [ratio[w[0]] for w in layers]
    assert 33 < min(lay) < 34 and 932 < max(lay) < 934, ratio
    slack = [slack_s(*w[1:]) for w in layers]
    assert abs(min(slack) * 1e6 - 16.0) < 0.1 and abs(max(slack) * 1e3 - 276) < 1, slack
    assert min(slack) == slack_s(BATCH, *D3[1])

    # 5. Reading 2.  At the SoC NPU's largest shape the two are level, to 3%, and
    #    the chiplet's time there is the link's: one padded set of 16,384 bytes.
    assert abs(ratio[small[0]] - 1.0) < 0.03, ratio[small[0]]
    p = pta(*small[1:])
    assert p["binds"] == "link" and p["sets"] == 1 and p["into"] == k * n + small[1] * small[2]
    #    Unpadded it is the tile's and 2.3 times the GPU, which is the size of
    #    X2's convention at this shape.
    u = pta(*small[1:], padded=False)
    assert u["binds"] == "tile"
    assert abs(gpu_s(*small[1:], launch=False) / u["seconds"] - 2.3) < 0.05

    # 6. Reading 3.  D3 at batch 64 is 3.74 us on the chiplet, and its two
    #    commands are 5.8 or 16 times that.
    d3 = sum(pta(BATCH, a, b)["seconds"] for a, b in D3)
    assert abs(d3 * 1e6 - 3.74) < 0.005, d3
    assert abs(2 * host_s / d3 - 5.8) < 0.05 and abs(2 * gpu_launch_s() / d3 - 16.3) < 0.05
    #    The tile is under a quarter of its own GEMM on both D3 layers at either
    #    cost, and the batch that makes it half is in the hundreds at the least.
    for a, b in D3:
        t = pta(BATCH, a, b)["seconds"]
        assert t / (t + host_s) < 0.25
        assert batch_for_half(a, b, host_s) > 9 * BATCH

    # 7. Reading 4.  The link binds on every shape, and D3's first layer fills
    #    60% of the eight sets it programs.
    assert all(pta(*w[1:])["binds"] == "link" for w in WORK)
    l1 = pta(BATCH, *D3[0])
    assert l1["sets"] == 8 and abs(l1["fill"] - 0.598) < 0.001
    assert abs(pta(BATCH, *D3[1])["fill"] - 0.061) < 0.001
    #    Doubling the batch adds under 40% to the first layer: the weights cross
    #    once.
    assert pta(2 * BATCH, *D3[0])["seconds"] < 1.4 * l1["seconds"]
    #    Five modules move every layer, which a tile-bound one would not.
    assert all(pta(*w[1:], mods=5)["seconds"] < 0.5 * pta(*w[1:])["seconds"] for w in layers)

    # 8. Reading 5.  D3's second layer: a tenth of the shot rate keeps it 12 times
    #    the GPU, and the c930's scan on one bank brings it level.
    l2 = (BATCH,) + D3[1]
    g = gpu_s(*l2, launch=False)
    assert abs(g / pta(*l2, fs=FS / 10)["seconds"] - 12.0) < 0.1
    assert abs(g / pta(*l2, banks=1, per_beat=1)["seconds"] - 1.0) < 0.01
    #    With two banks a second bank does not help a layer of one set: the first
    #    write is seen either way.
    assert pta(*l2)["beats"] == pta(*l2, banks=1)["beats"]

    # 9. Reading 6.  With the preamble the break-even sits above the SoC NPU's
    #    largest command; without it the GPU's fixed cost is under the NPU's host
    #    path, so there is none.  The figure that separates them is 24.4 us, and
    #    the two launch costs in hand are one on each side of it.
    be = npu_gpu_break_even()
    assert 3_600 < be < 3_800, be
    assert macs(*NPU_MAX.values()) < be
    assert npu_gpu_break_even(preamble=False) < 0
    assert gpu_launch_s(False) < host_s < gpu_launch_s()
    keeps = npu_keeps_s()
    assert abs(keeps * 1e6 - 24.4) < 0.05, keeps
    assert gpu_launch_s(False) < keeps < gpu_launch_s()
    assert abs(gpu_launch_s(False) * 1e6 - 6.9) < 0.05
    #    Past a hundred thousand MACs the GPU wins whichever launch cost is right.
    assert gpu_s(*f0) < npu_s(*f0)
    assert npu_takes(*small[1:]) and not any(npu_takes(*w[1:]) for w in layers)

    # 10. The accuracy term is one network's, at two calibration intervals.
    assert sorted(D3_V1_LOSS.values()) == [0.21, 0.24]
    #     They are this tile's, and the tile is the one the model prices.  The
    #     8 x 8 tile's are dearer at both intervals.
    assert TILE == (256, 64)
    assert all(D3_V1_LOSS[k] < D3_V1_LOSS_8X8[k] for k in D3_V1_LOSS)
    assert sorted(D3_V1_LOSS_8X8.values()) == [0.37, 0.81]

    # 11. Section 6.  A launch between D3's two GEMMs is eight times what the
    #     chiplet spends on both.
    assert abs(gpu_launch_s() / d3 - 8.1) < 0.1

    # 13. Section 7.  D3 as three submissions, as one list, and as one list with
    #     the step between its layers on the chiplet.  The list takes 21.8 us off
    #     at either launch cost, and the chiplet's share of its own network goes
    #     from under a tenth to a quarter.
    lo, hi = gpu_launch_s(False), gpu_launch_s()
    calls = [d3_submitted_s(3, x, host_s) for x in (lo, hi)]
    one = [d3_submitted_s(1, x, host_s) for x in (lo, hi)]
    held = d3_submitted_s(1, None, host_s)
    assert [fmt_s(t) for t in calls] == ["43.4 us", "67 us"], calls
    assert [fmt_s(t) for t in one] == ["21.6 us", "45.1 us"], one
    assert fmt_s(held) == "14.7 us", held
    assert all(abs((a - b) - 2 * host_s) < 1e-12 for a, b in zip(calls, one))
    assert fmt_s(2 * host_s) == "21.8 us"
    assert [f"{d3 / t:.0%}" for t in calls] == ["9%", "6%"]
    assert [f"{d3 / t:.0%}" for t in one] == ["17%", "8%"]
    assert f"{d3 / held:.0%}" == "26%"
    #     The intermediate that would stop crossing is not what the last row
    #     saves: D3's second layer loses its 6,400 activation bytes from the
    #     link and a tenth of a microsecond.
    l2 = pta(BATCH, *D3[1])
    assert l2["into"] == k * n + BATCH * D3[1][0] and BATCH * D3[1][0] == 6_400
    assert abs(BATCH * D3[1][0] / (link.MODULE_GBS * 1e9) * 1e6 - 0.11) < 0.005

    # 12. board_program_plan.md 3.4 quotes these, figure for figure.
    f0_macs = macs(*f0)
    assert fmt_s(NPU_F0_GEMM * sweep.CYCLE_NS * 1e-9 / f0_macs) == "5.54 ns"
    assert fmt_s(c_gpu / (GPU_MHZ * 1e6)) == "257 ps"
    assert fmt_s(big["seconds"] / macs(link.BIG["mb"], link.BIG["kin"], link.BIG["nout"])) == "0.276 ps"
    quoted = {
        "D3 layer 1": ("1.29 ms", "3.25 us", 397, "1.29 ms"),
        "D3 layer 2": ("16.5 us", "496 ns", 33, "16 us"),
        "section 6.2's shape": ("33.7 us", "669 ns", 50, "33 us"),
        "the SoC NPU's largest": ("395 ns", "387 ns", 1, "8.24 ns"),
        "4096 x 4096": ("276 ms", "296 us", 933, "276 ms"),
    }
    for name, m, kk, nn in WORK:
        got = (fmt_s(gpu_s(m, kk, nn, launch=False)), fmt_s(pta(m, kk, nn)["seconds"]),
               round(ratio[name]), fmt_s(slack_s(m, kk, nn)))
        assert got == quoted[name], (name, got)
    assert [batch_for_half(a, b, host_s) for a, b in D3] == [628, 6_063]
    assert round(keeps * GPU_MHZ * 1e6) == 9_758
    assert fmt_s(sum(npu_s(BATCH, a, b) for a, b in D3)) == "29.3 ms"
    assert fmt_s(sum(gpu_s(BATCH, a, b) for a, b in D3)) == "1.37 ms"
    assert (fmt_s(d3 + 2 * host_s), fmt_s(d3 + 2 * gpu_launch_s())) == ("25.6 us", "64.7 us")
    assert [round(D3_EXACT - x, 2) for x in D3_V1_LOSS.values()] == [97.21, 97.24]
    assert [round(D3_EXACT - x, 2) for x in D3_V1_LOSS_8X8.values()] == [96.64, 97.08]

    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
