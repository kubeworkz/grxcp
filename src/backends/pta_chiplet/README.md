# The PTA chiplet: its digital twin, and its driver

**Board plan steps X5 and S4.** X5 is a C model that presents the chiplet's
register map
([`pta_chiplet_regmap.md`](../../../docs/designs/pta_chiplet_regmap.md), X4)
with the photonic tile's error model behind it, so that a driver, grxcp and the
dispatch model can be brought up before there is a chiplet. S4 is that driver,
and the runtime's use of it.

**It is a model, and there is no chiplet.** `PTA_CAPS2` says so in bit 31.
Nothing that runs through the twin is a statement about a photonic device. What
it is evidence of: that the map can be implemented as written, that this
implementation is the error model bit for bit, and what that model gives for a
workload.

## Files

| File | What it is |
|---|---|
| `pta_chiplet_twin.h` | The map's offsets and bits, the build, and the ways in: the window, the command path, the clock |
| `pta_chiplet_twin.c` | The twin: registers, the chiplet's own command queue, a clock, and the calibration contract, in front of `pta_gemm()` and `pta_cal_bank()` |
| `test_pta_chiplet_twin.cc` | The gate |
| `pta_mnist_via_twin.c` | grx930's accuracy harness with the twin where the model was — the software half of the board plan's P2 gate |
| `pta_mnist_budget_via_twin.sh` | Runs that harness over grx930's whole accuracy budget and holds each line to the one recorded |
| `pta_mnist_act_via_twin.c` | The activation stage held to that harness: grx930's own batch routine beside a twin that keeps every intermediate |
| `pta_chiplet.h`, `pta_chiplet.cpp` | The host's driver: detection, the property's reader, the completion test, one GEMM start to end. In `libgrxrt` under `-DGRXCP_ENABLE_PTA=ON` |
| `pta_chiplet_testing.h` | The seam that attaches a model as the chiplet the runtime enumerates |
| `test_pta_chiplet_driver.cc` | The driver's gate, against the twin and against windows and links that misbehave |
| `CMakeLists.txt` | `pta_chiplet_twin`, a static library, and the two gates as tests |

The tile's arithmetic is not in this directory. It is
`third_party/grx930/pta_tile_model.c`, grx930's, vendored byte for byte and held
to grx930's own vectors by `ci/build_mock.sh`. The twin calls it and adds
nothing to it.

**The twin is not in `libgrxrt`.** A runtime that linked it would be carrying
its own device. The driver is, under the flag, and a test attaches the twin to
it through the seam.

## The driver, and the device

The chiplet is a device of its own in grxcp's table (the board plan's B9):
`GRX_DEVICE_TYPE_PTA`, GEMM-only, with its GPU as its parent. It has no memory,
so a GEMM on it takes pointers allocated on `grxDeviceProp_t.parentDevice`.

**It has no hardware path.** The chiplet's registers are a page of the GPU's
BAR and the GPU's driver has no call that reaches it, and the link that carries
its work has no command format. So the driver's three ways in are hooks, and
only a model fills them. A runtime built with the flag and nothing attached
enumerates no PTA device. One with a window and no link enumerates it, reports
it, and refuses a GEMM as not supported with C untouched.

What would replace the hooks has been proposed to grxgpu
(`grxgpu/docs/proposals/pta_chiplet_host_path.md`): the registers through its
driver's register commands, and a GEMM as one command of its command processor.
Until that is answered, nothing here changes.

The gates that need the runtime are
`tests/unit/test_pta_chiplet_device.cpp`, what the device reports and refuses,
and `tests/libs/test_grxblas_pta_chiplet.cpp`, `grxblasGemmEx` on it held bit
for bit to the model built from the device property alone.
`tests/common/pta_twin_adapter.h` is how a test asks for the twin.

## Using it

```c
#include "pta_chiplet_twin.h"

pta_twin_build build = { 256, 64, 8, 48, 4 };   /* rows, columns, DIN_W, ACC_W, queue depth */
pta_twin *t = pta_twin_new(&build);

/* The control path: the window, as CXL.io will reach it. */
pta_twin_write32(t, PTA_TWIN_SEED, 20261004);    /* once a run; restarts PTA_GEMM_CT */
pta_twin_write32(t, PTA_TWIN_IMPAIR, 0x03);      /* QUANT | THERMAL */
pta_twin_write32(t, PTA_TWIN_BITS, 6 | 6 << 4 | 7 << 8 | 16 << 12);
pta_twin_write32(t, PTA_TWIN_SIGMA_TH, 0x0080);  /* half an LSB of that 7-bit ADC */

/* The data path: a whole GEMM, standing in for link 2. */
int status;
pta_twin_cmd cmd = { 0, M, N, K, A, B, C, &status, NULL };
if (pta_twin_submit(t, &cmd) != 0) { /* not accepted: the queue is full */ }

/* The map's completion test: one read, neither BUSY nor CAL_BUSY. */
while (pta_twin_read32(t, PTA_TWIN_STATUS) & (PTA_TWIN_STATUS_BUSY | PTA_TWIN_STATUS_CAL_BUSY))
    pta_twin_run(t, 1000);
/* status is PTA_TWIN_DONE and C holds the result, or PTA_TWIN_REFUSED and C is untouched. */
```

The geometry is the caller's to name, because it is the map's first open
question. The gate runs 4 × 4 (the c930 register model's tile), 8 × 8 (the
tile the accuracy budget was measured on) and 256 × 64 (one of the link
model's candidates for the chiplet).

## The gate

`ci/build_mock.sh` builds and runs it in tier 1, as the **PTA CHIPLET TWIN
GATE**, and `ctest` runs it as `pta_chiplet_twin`. It has three parts.

- **The map.** Identity and capability, the per-GEMM seed and `PTA_GEMM_CT`,
  64-bit counters with a latched upper half, the completion contract behind a
  link, interrupts, the calibration engine's words, refusals, reset.
- **The model.** Every GEMM through the twin equals `pta_gemm()` called
  directly, on a device the twin never sees, and every calibration equals
  `pta_cal_bank()`. The clear case is held to an integer product computed
  without the model.
- **The driver.** `npu_c930_read_analog()`, the c930 backend's reader, pointed
  at the twin through a change of base and nothing else. Then two GEMMs
  reproduced from what that driver read, and `PTA_GEMM_CT`, alone.

- **The activation stage.** Its function against the harness's three lines, a
  network held on the chiplet against the same network brought out at every
  layer, what it refuses, and how long held operands last. Below.

Then it is built five more times, each against a twin that is wrong in one way,
and has to fail each time:

| Switch | The twin it builds | What catches it |
|---|---|---|
| `PTA_TWIN_ABLATE_SEED` | Every GEMM runs on `PTA_SEED` itself | The bitwise checks, wherever noise is drawn |
| `PTA_TWIN_ABLATE_CAL_GUARD` | A command arriving during a calibration is taken at once | Occupancy 1, 2, 3 during `CAL_BUSY`, and "nothing is delivered" |
| `PTA_TWIN_ABLATE_RST_GUARD` | `MODEL_RST` is honoured whatever is running | "leaves the device as it was" |
| `PTA_TWIN_ABLATE_ACT_ROUND` | The activation stage's shift truncates | The stage's function, and every GEMM through it |
| `PTA_TWIN_ABLATE_HELD_GUARD` | Held operands outlive the command after them | "a layer that is refused leaves nothing held" |

## The activation stage

`docs/designs/pta_chiplet_regmap.md` §8 is the specification. A build has the
stage when `pta_twin_build.act_hold` is not zero; that is how many operands it
can hold for the next command, a power of two.

It is the step grx930's harness takes on the host between a network's layers:
a bias, a ReLU, a rounding shift back to an operand, and the operand's clamp.
`pta_twin_activate()` is the function. It is the one piece of arithmetic in the
twin that is not grx930's, and so it is the one piece held to grx930 from
outside.

A command asks for the stage, and nothing is left switched on afterwards.

```c
pta_twin_cmd c;
memset(&c, 0, sizeof c);                 /* zero asks nothing of the stage */
/* layer 1: its operands stay on the chiplet */
c.M = 64; c.K = 784; c.N = 100; c.A = pixels; c.B = w1;
c.flags = PTA_TWIN_CMD_ACT | PTA_TWIN_CMD_HOLD;
c.act_shift = 12; c.act_bits = 8; c.bias = b1;
pta_twin_submit(twin, &c);
/* layer 2: its activations are the ones held */
c.K = 100; c.N = 10; c.A = NULL; c.B = w2; c.C = sums; c.bias = NULL;
c.flags = PTA_TWIN_CMD_FROM_HELD;
pta_twin_submit(twin, &c);
```

Held operands are the next command's or nobody's. Any command's start takes or
discards them, a calibration between two layers does not, and a reset drops
them.

**Held to the harness itself, 2026-10-04**, against grx930 at `838c7cd`.
`pta_mnist_act_via_twin.c` compiles grx930's `pta_mnist.c` into itself with its
`main` renamed, runs the harness's own `tile_batch()` on a random network, and
runs the same network on a twin with every intermediate held:

```
cc -std=c99 -O2 -I$GRX930/c930/sim -I$GRXCP/third_party/grx930 \
   -I$GRXCP/src/backends/pta_chiplet pta_mnist_act_via_twin.c \
   pta_chiplet_twin.c $GRXCP/third_party/grx930/pta_tile_model.c -lm \
   -o pta_mnist_act_twin && ./pta_mnist_act_twin
```

| Network | Tile | The harness | The twin | Operands handed on | Last layer's sums |
|---|---|---|---|---|---|
| D3's shape, exact | 256 × 64 | 54 GEMMs | 2 commands | 6,400, none differ | 640, none differ |
| Three hidden layers, exact | 256 × 64 | 80 GEMMs | 4 commands | 19,200, none differ | 640, none differ |
| D3's shape, quantisers on | 8 × 8 | 54 GEMMs | 2 commands | 6,400, none differ | 640, none differ |
| Three hidden layers, quantisers on | 8 × 8 | 80 GEMMs | 4 commands | 14,400, none differ | 480, none differ |

Built against a twin whose shift truncates, it fails all four: 1,646 of the
first case's 6,400 operands differ.

It is not in CI, for the reason the budget run is not: grx930's harness is not
in this tree. And it compares nothing with noise enabled. The harness and the
twin cut a layer into different GEMMs, each GEMM draws from its own seed, and so
with noise they are two different runs. What holds a held network to anything
under noise is the gate above, where both runs are cut the same way.

**The network is random.** No MNIST file is read and no accuracy is measured.
D3 held on the chiplet, on the trained networks, has not been run.

## The D3 network through the twin

The board plan's P2 gate asks that "the D3 network runs through the emulated
PTA bit-identical to `pta_mnist`'s C reference". The enumeration half of that
gate needs a board. This half does not.

grx930's harness touches its device through three of the model's functions:
`pta_model_reset()` once, on the run's seed; `pta_drift_age()`, if the run asks
for hours of drift; and `pta_gemm()` for every GEMM of the network.
`pta_mnist_via_twin.c` defines the same three, with the same signatures, which
reach the model only through the twin. The harness is compiled unedited with
the names redirected:

```sh
cc -std=c99 -O2 -Ithird_party/grx930 -Dpta_gemm=pta_gemm_via_twin \
   -Dpta_model_reset=pta_model_reset_via_twin \
   -Dpta_drift_age=pta_drift_age_via_twin \
   -c $GRX930/c930/sim/pta_mnist.c -o pta_mnist.o
cc -std=c99 -O2 -Ithird_party/grx930 -Isrc/backends/pta_chiplet pta_mnist.o \
   src/backends/pta_chiplet/pta_mnist_via_twin.c \
   src/backends/pta_chiplet/pta_chiplet_twin.c \
   third_party/grx930/pta_tile_model.c -lm -o pta_mnist_twin
```

The reset goes through the map: `PTA_SEED`, then `PTA_CTRL.MODEL_RST`. Every
GEMM's configuration is written to the registers before it, the GEMM is a
command, and its end is read from `PTA_STATUS`. The ageing is `pta_twin_age()`,
which is the twin's and not the map's.

`pta_mnist_budget_via_twin.sh` then gives that program every setting of the
harness's budget on every network, and compares what it prints with what the
harness printed when it ran the budget itself.

**Measured 2026-10-04**, against grx930 at `838c7cd`: every evaluation of the
accuracy budget of 2026-10-03, run again through the twin. 44 settings on five
networks, each network on its own seed, is 220 passes over the 10,000 test
images and 1,865,160 GEMMs, on the 8 × 8 tile the budget was measured on. All
220 printed the line the harness had written the day before, byte for
byte — accuracy, saturations and every probe figure. 10 of them age the tile
first, by six minutes of drift or by an hour.

It is not in CI: grx930's harness and the MNIST files are not in this tree, and
are not vendored for this.

It refuses what no register can do. The harness calibrates through
`pta_trim_write()`, a host's trim write, and the map has no home for one. A run
with `--calibrate` stops with a message rather than compare two different
devices.

## What the map does not give

Building the twin found these. Each is recorded against the map, in its
section 6, and none is papered over in the code.

- **No way to issue work.** The map is the control window. Work reaches the
  chiplet over link 2, and nothing says what a command on that link looks
  like. `pta_twin_submit()` is a function call standing in for a protocol
  nobody has written.
- **The queue's depth** is a build parameter here, and no register reports it.
- **`BUSY` has to cover the chiplet's queue**, or one read of `PTA_STATUS`
  cannot be the device's half of "has the work finished". Here it does.
- **`PTA_GAIN[j]` and `PTA_OFFS[j]` are eight words each** and a chiplet's tile
  has 64 columns or more. Columns past the eighth have no register.
- **`PTA_IRQ_STATUS.SAT_THRESHOLD` has no threshold.** The twin never raises it.
- **A GEMM cannot be reproduced from `PTA_SEED` alone.** Its seed is derived
  from `PTA_SEED` and the GEMM's index, so a report that carries one and not
  the other describes a run and not a result.

## What the twin does not model

Each reads zero in the register that would say otherwise.

- The calibration scheduler (`PTA_CTRL[6:4]`). A calibration runs when
  `PTA_CTRL.CAL_NOW` asks for one. The comparison of schedulers that the
  calibration document assigns to this twin is still owed.
- The loop-order and residency modes (`PTA_CTRL[9:7]`). The model walks one
  order.
- The activation stage, unless the build asks for one (`PTA_CAPS2[17]`), and
  its time when it does: the stage adds no cycle here.
- A shot rate (`PTA_CAPS2[15:0]`). A shot lasts `PTA_TS` of the twin's own
  cycles.
- A calibration that interrupts a GEMM. Here it waits for the running command.
- Drift that comes with time. The model's drift is clocked by shots, so an idle
  twin does not drift. `pta_twin_age()` is the model's own fast-forward, for
  sweeps over hours; no register does it.
- Any timing beyond two formulas: a GEMM holds the tile for
  `programmings × PTA_TW + shots × PTA_TS` cycles and a calibration for
  `passes × repeats × (PTA_TW + rows × PTA_TS)`.
