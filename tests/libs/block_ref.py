#!/usr/bin/env python3
"""A transformer block from PyTorch: the phase 6 exit gate's reference.

The gate is "a transformer inference workload runs end-to-end through GRXCP
libraries with numerical agreement against a PyTorch CPU reference". Every op it
names is already gated on its own. This is the part those gates cannot cover:
whether four correct ops, composed, are still correct.

They are not automatically. Each individual gate fixes its own layout
convention and checks it in isolation; composing them means one op's output
becomes another's input, and a transposed or mis-strided hand-off produces
plausible numbers that no single-op gate can see.

THE BLOCK, pre-norm, GPT-2 shaped:

    h1  = LayerNorm(x, g1, b1)
    Q,K,V = h1 @ Wq,Wk,Wv  + bq,bk,bv        per head
    a   = Attention(Q, K, V, causal)
    p   = a @ Wo + bo
    x2  = x + p                              residual
    h2  = LayerNorm(x2, g2, b2)
    f1  = h2 @ W1 + bf1
    act = GELU(f1)
    f2  = act @ W2 + bf2
    y   = x2 + f2                            residual

INTERMEDIATES ARE DUMPED, NOT JUST y. A block that disagrees only at the end
tells you nothing about where — and with ten stages, bisecting by hand is the
expensive way to find out. The gate compares stage by stage and stops at the
first disagreement, so a failure names the op.

    python3 tests/libs/block_ref.py --write PATH

Format, little-endian, all tensors ROW-major and contiguous:
    u32 magic 'GBLK'  u32 version  u32 num_cases
    per case: u32 seq, model_dim, heads, ff_dim, causal
              f32 x[S*D]
              f32 g1[D] b1[D] g2[D] b2[D]
              f32 Wq[D*D] Wk[D*D] Wv[D*D]      (row-major [D][D])
              f32 bq[D] bk[D] bv[D]
              f32 Wo[D*D] bo[D]
              f32 W1[D*F] bf1[F] W2[F*D] bf2[D]
              f32 h1[S*D] q[H*S*Dh] k[H*S*Dh] v[H*S*Dh] a[H*S*Dh]
              f32 p[S*D] x2[S*D] h2[S*D] f1[S*F] act[S*F] f2[S*D] y[S*D]
"""

import argparse
import struct
import sys

import numpy as np
import torch

MAGIC = 0x4B4C4247  # 'GBLK'
VERSION = 1

# seq, model_dim, heads, ff_dim, causal
#
# Small, because every one of these runs on a functional simulator. What they
# have to cover is the composition, not scale: one head to isolate the plumbing
# from the per-head slicing, then two heads so the slicing is actually
# exercised, and a causal case because that is the shape inference has.
CASES = [
    (6, 8, 1, 32, False),
    (8, 16, 2, 64, False),
    (8, 16, 2, 64, True),
]

EPS = 1e-5


def block(x, w, heads, causal):
    """The reference. Plain torch ops in float64, one line per stage."""
    S, D = x.shape
    Dh = D // heads

    h1 = torch.nn.functional.layer_norm(x, (D,), w["g1"], w["b1"], EPS)

    # [S, D] -> per head [H, S, Dh]. In torch this is a reshape and a transpose;
    # GRXCP gets the same layout straight out of a batched GEMM over the weight
    # matrix's column blocks, with no permute at all.
    def project(wgt, bias):
        y = h1 @ wgt + bias                      # [S, D]
        return y.reshape(S, heads, Dh).permute(1, 0, 2).contiguous()

    q = project(w["Wq"], w["bq"])
    k = project(w["Wk"], w["bk"])
    v = project(w["Wv"], w["bv"])

    a = torch.nn.functional.scaled_dot_product_attention(
        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=causal
    ).squeeze(0)                                  # [H, S, Dh]

    # Back to [S, D] for the output projection. GRXCP does this without a
    # permute either, by accumulating one GEMM per head into the same result.
    a_merged = a.permute(1, 0, 2).reshape(S, D)
    p = a_merged @ w["Wo"] + w["bo"]
    x2 = x + p

    h2 = torch.nn.functional.layer_norm(x2, (D,), w["g2"], w["b2"], EPS)
    f1 = h2 @ w["W1"] + w["bf1"]
    act = torch.nn.functional.gelu(f1, approximate="tanh")
    f2 = act @ w["W2"] + w["bf2"]
    y = x2 + f2

    return dict(h1=h1, q=q, k=k, v=v, a=a, p=p, x2=x2, h2=h2,
                f1=f1, act=act, f2=f2, y=y)


def make_weights(D, F, gen):
    def t(*shape, scale=1.0):
        return (torch.rand(shape, generator=gen, dtype=torch.float64) - 0.5) * 2.0 * scale

    return {
        # Layer-norm gains near 1 and shifts near 0, as a trained model has them:
        # a gain centred on zero would make the norm's output mostly noise and
        # hide a sign error in the following GEMM.
        "g1": 1.0 + t(D, scale=0.1), "b1": t(D, scale=0.1),
        "g2": 1.0 + t(D, scale=0.1), "b2": t(D, scale=0.1),
        # 1/sqrt(fan_in), the usual init, so activations stay O(1) through the
        # block instead of growing and putting the comparison at the mercy of
        # fp32 range rather than of correctness.
        "Wq": t(D, D, scale=D ** -0.5), "bq": t(D, scale=0.1),
        "Wk": t(D, D, scale=D ** -0.5), "bk": t(D, scale=0.1),
        "Wv": t(D, D, scale=D ** -0.5), "bv": t(D, scale=0.1),
        "Wo": t(D, D, scale=D ** -0.5), "bo": t(D, scale=0.1),
        "W1": t(D, F, scale=D ** -0.5), "bf1": t(F, scale=0.1),
        "W2": t(F, D, scale=F ** -0.5), "bf2": t(D, scale=0.1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="PATH")
    args = ap.parse_args()

    blobs = []
    for idx, (S, D, H, F, causal) in enumerate(CASES):
        gen = torch.Generator().manual_seed(4000 + idx)
        x = (torch.rand((S, D), generator=gen, dtype=torch.float64) - 0.5) * 2.0
        w = make_weights(D, F, gen)
        out = block(x, w, H, causal)

        tag = f"S{S} D{D} H{H} F{F}{' causal' if causal else ''}"
        print(f"  {tag:<28} |y| max {float(out['y'].abs().max()):.4g}")
        blobs.append((S, D, H, F, causal, x, w, out))

    if args.write:
        order_w = ["g1", "b1", "g2", "b2", "Wq", "Wk", "Wv", "bq", "bk", "bv",
                   "Wo", "bo", "W1", "bf1", "W2", "bf2"]
        order_o = ["h1", "q", "k", "v", "a", "p", "x2", "h2",
                   "f1", "act", "f2", "y"]
        with open(args.write, "wb") as f:
            f.write(struct.pack("<III", MAGIC, VERSION, len(blobs)))
            for S, D, H, F, causal, x, w, out in blobs:
                f.write(struct.pack("<IIIII", S, D, H, F, 1 if causal else 0))
                f.write(x.numpy().astype("<f4").tobytes())
                for name in order_w:
                    f.write(w[name].numpy().astype("<f4").tobytes())
                for name in order_o:
                    f.write(out[name].numpy().astype("<f4").tobytes())
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
