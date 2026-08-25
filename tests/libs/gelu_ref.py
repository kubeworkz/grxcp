#!/usr/bin/env python3
"""Reference vectors for grxdnnGeluForward, from PyTorch.

GELU is where grxDNN grew its own transcendentals. The device build is
-nostdlib, so there is no erf and no tanh; both are built on the dev_exp that
softmax already uses, and the whole question is how much error that stack
accumulates.

This library has been wrong about exactly that before. dev_rsqrt shipped under a
comment claiming "about 2e-6 relative" for a bit-hack that is actually 1.7e-3,
and the layer-norm gate caught it only because every one of its cases failed
while every softmax case passed. So the numbers in elementwise.cpp are MEASURED
by the gate that reads this file's output, and this script's job is to provide
something to measure against that GRXCP had no hand in.

    python3 tests/libs/gelu_ref.py            # print the two forms' spread
    python3 tests/libs/gelu_ref.py --write PATH

Format, all little-endian:
    u32 magic 'GELU'  u32 version  u32 rows  u32 cols
    f32 x[rows*cols]  f32 exact[rows*cols]  f32 tanh[rows*cols]
    f32 bias[cols]    f32 biased[rows*cols]      (x + bias broadcast down rows)
"""

import argparse
import struct
import sys

import numpy as np
import torch

MAGIC = 0x554C4547  # 'GELU' little-endian
VERSION = 1

ROWS, COLS = 32, 64          # 2048 samples, and a shape with ragged warps


def build_x():
    """The sweep.

    Linear over [-10, 10] rather than random: GELU's interesting structure is
    all in |x| < 4 and its saturation behaviour is at the ends, and a uniform
    sweep hits both without leaving gaps to luck. The specific values that get
    substituted in are the ones where an implementation tends to break --
    exactly zero, the saturation thresholds dev_erf and dev_tanh use, and
    magnitudes far outside them.
    """
    x = np.linspace(-10.0, 10.0, ROWS * COLS, dtype=np.float64)
    edge = [0.0, -0.0, 1e-8, -1e-8, 4.0, -4.0, 9.0, -9.0, 30.0, -30.0,
            88.0, -88.0, 1e3, -1e3]
    x[:len(edge)] = edge
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="PATH")
    args = ap.parse_args()

    x = build_x()
    tx = torch.from_numpy(x.astype(np.float64))

    exact = torch.nn.functional.gelu(tx, approximate="none").numpy()
    tanh_ = torch.nn.functional.gelu(tx, approximate="tanh").numpy()

    # THE TWO FORMS ARE DIFFERENT FUNCTIONS, and the gate needs to know by how
    # much. If they agreed to within its tolerance, a gate that passed both
    # would be checking one implementation twice and the mode argument could be
    # ignored entirely without anything noticing.
    spread = float(np.max(np.abs(exact - tanh_)))
    at = int(np.argmax(np.abs(exact - tanh_)))
    print(f"  exact vs tanh: worst |diff| {spread:.3g} at x = {x[at]:.4g}")
    if spread < 1e-4:
        print("  FAILED: the two forms are too close to tell apart")
        return 1

    rng = np.random.default_rng(20260825)
    bias = (rng.random(COLS, dtype=np.float64) - 0.5) * 4.0
    biased = x.astype(np.float64).reshape(ROWS, COLS) + bias[None, :]

    print(f"  sweep: {ROWS}x{COLS} over [{x.min():.4g}, {x.max():.4g}]")

    if args.write:
        with open(args.write, "wb") as f:
            f.write(struct.pack("<IIII", MAGIC, VERSION, ROWS, COLS))
            for arr in (x, exact, tanh_):
                f.write(np.asarray(arr, dtype="<f4").tobytes())
            f.write(np.asarray(bias, dtype="<f4").tobytes())
            f.write(np.asarray(biased, dtype="<f4").tobytes())
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
