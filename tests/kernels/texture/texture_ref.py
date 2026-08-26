#!/usr/bin/env python3
"""Reference samples for the TEXTURE GATE, from PyTorch rather than from us.

A texture sampler is almost entirely convention: where a texel's centre sits,
which way floor rounds a negative coordinate, what "clamp" means at the far
edge. A reference written from the same reasoning as the implementation agrees
with it whether or not either is right -- the same trap the attention gate was
written to avoid, and the reason that one uses PyTorch too.

torch.nn.functional.grid_sample with align_corners=False uses exactly CUDA's
convention: a texel centre at integer + 0.5, coordinates normalised so that -1
and +1 are the OUTER edges of the first and last texels. So the mapping from an
unnormalized texture coordinate u in [0, width] is

    g = 2 * u / width - 1

and grid_sample's padding modes line up with three of the four address modes:

    padding_mode="border"      -> GRX_TEX_ADDRESS_CLAMP
    padding_mode="zeros"       -> GRX_TEX_ADDRESS_BORDER with border 0
    padding_mode="reflection"  -> GRX_TEX_ADDRESS_MIRROR

WRAP has no grid_sample equivalent and is NOT covered here. It is checked
against a from-specification host reference in main.cpp instead, and that
weaker check is stated rather than hidden.

Writes tests/kernels/texture/texture_ref.bin:
    uint32 width, height, count, mode_count
    float  texels[height * width]
    float  coords[2 * count]                 (unnormalized, texel units)
    float  expected[mode_count * count]      (one block per address mode)
"""

import pathlib
import struct

import torch
import torch.nn.functional as F

W, H = 16, 12

# A field with CURVATURE. A linear ramp is the classic bad texture fixture:
# bilinear interpolation of a linear function is exact no matter what weights
# you use, so wrong filter weights pass. This one does not have that mercy.
torch.manual_seed(7)
ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                        torch.arange(W, dtype=torch.float64), indexing="ij")
texels = (torch.sin(xs * 0.7) * torch.cos(ys * 0.5)
          + 0.25 * torch.sin((xs + ys) * 1.3)).float()

# Coordinates: texel centres, midpoints, and well outside on every side, so the
# address modes are what decides the answer rather than an accident of range.
coords = []
for cy in (-2.5, -0.5, 0.0, 0.5, 3.25, 6.0, 11.5, 12.5, 14.0):
    for cx in (-3.0, -0.5, 0.0, 0.5, 1.0, 7.75, 15.5, 16.5, 19.0):
        coords.append((cx, cy))

MODES = [("border", "CLAMP"), ("zeros", "BORDER"), ("reflection", "MIRROR")]

img = texels.view(1, 1, H, W)
grid = torch.tensor(
    [[[[2.0 * cx / W - 1.0, 2.0 * cy / H - 1.0] for (cx, cy) in coords]]],
    dtype=torch.float32)

blocks = []
for padding, name in MODES:
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode=padding,
                        align_corners=False)
    blocks.append(out.reshape(-1))

dest = pathlib.Path(__file__).with_name("texture_ref.bin")
with dest.open("wb") as f:
    f.write(struct.pack("<IIII", W, H, len(coords), len(MODES)))
    f.write(texels.reshape(-1).numpy().astype("<f4").tobytes())
    for cx, cy in coords:
        f.write(struct.pack("<ff", cx, cy))
    for b in blocks:
        f.write(b.numpy().astype("<f4").tobytes())

print(f"wrote {dest.name}: {W}x{H} texels, {len(coords)} coords, "
      f"{len(MODES)} address modes ({', '.join(n for _, n in MODES)})")
print(f"torch {torch.__version__}, grid_sample align_corners=False")
