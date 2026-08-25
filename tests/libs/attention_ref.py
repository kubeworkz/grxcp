#!/usr/bin/env python3
"""Reference vectors for grxdnnAttentionForward, from PyTorch.

WHY PYTORCH AND NOT A REFERENCE WE WROTE
----------------------------------------
Attention is where grxDNN's ROW-major convention meets grxBLAS's COLUMN-major
one, in a single op, twice. The whole thing is transpose bookkeeping, and a
reference derived from the same reasoning as the implementation agrees with it
perfectly whether or not either is right -- the failure `unpack()` in
tests/libs/test_grxblas.cpp exists to prevent, where two copies of one
misconception passed a transposed GEMM that transposed nothing.

So the reference is somebody else's arithmetic: torch's own
scaled_dot_product_attention, which knows nothing about how GRXCP stores a
matrix. It runs here, not in CI -- the vectors are checked in so the gate needs
no torch, and this script is checked in so anyone can regenerate them and diff.

This file ALSO simulates the exact grxblasSgemm calls the implementation makes,
on flat memory, with the leading dimensions and transpose flags spelled out. If
that simulation reproduces torch, the index algebra is right before a line of
device code exists. If it does not, the mistake is here, on a laptop, and not in
a kernel at three in the morning.

    python3 tests/libs/attention_ref.py            # check the algebra
    python3 tests/libs/attention_ref.py --write    # and write the .bin

Format of attention_ref.bin, all little-endian:
    u32 magic 'GATN'  u32 version  u32 num_cases
    per case: u32 B, H, S, D, causal   then f32 Q, K, V, expected  (row-major,
                                       each [B][H][S][D], contiguous)
"""

import argparse
import struct
import sys

import numpy as np
import torch

MAGIC = 0x4E544147  # 'GATN' little-endian
VERSION = 1

# B, H, S, D, causal
CASES = [
    (1, 1,  1,  1, False),   # the degenerate one: softmax over a single element
    (1, 1,  3,  2, False),   # sequence shorter than a warp (4 lanes)
    (1, 2,  8,  4, False),   # exactly two warps of keys
    (2, 2, 17,  8, False),   # ragged: neither S nor D is a multiple of the warp
    (1, 1, 40, 16, False),   # long enough that a row spans many warp strides
    (1, 2,  8,  4, True),    # causal, the shape an LLM decode step actually has
    (2, 2, 17,  8, True),    # causal and ragged together
]


def make_inputs(b, h, s, d, seed):
    g = torch.Generator().manual_seed(seed)
    # Bounded and modest: the point is the layout algebra, not the numerics, and
    # a huge logit would put the comparison at the mercy of exp() differences
    # rather than of a transpose being right.
    def t():
        return (torch.rand((b, h, s, d), generator=g, dtype=torch.float64) - 0.5) * 2.0
    return t(), t(), t()


def torch_reference(q, k, v, causal):
    # float64 through the reference so the tolerance the gate uses is about the
    # DEVICE's fp32 arithmetic and not about torch's.
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal)
    return out


# ---------------------------------------------------------------------------
# The implementation's arithmetic, simulated on flat memory.
# ---------------------------------------------------------------------------

def gemm_colmajor(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """grxblasSgemm, exactly: every operand read COLUMN-major out of flat memory.

    C := alpha * op(A) * op(B) + beta * C, where op(A) is m x k and op(B) is
    k x n. Element (i, j) of a column-major matrix with leading dimension ld
    lives at flat offset i + j*ld -- that one line is the whole convention, and
    it is written out rather than expressed with a reshape so there is nothing
    hiding in a stride order.
    """
    def read(M, ld, rows, cols):
        out = np.empty((rows, cols), dtype=np.float64)
        for j in range(cols):
            for i in range(rows):
                out[i, j] = M[i + j * ld]
        return out

    opA = read(A, lda, k, m).T if transa else read(A, lda, m, k)
    opB = read(B, ldb, n, k).T if transb else read(B, ldb, k, n)
    prod = alpha * (opA @ opB)

    for j in range(n):
        for i in range(m):
            off = i + j * ldc
            C[off] = prod[i, j] + (beta * C[off] if beta != 0.0 else 0.0)


def grxcp_attention(q, k, v, causal):
    """What grxdnnAttentionForward does, in the order it does it.

    Tensors arrive ROW-major as [B][H][S][D], which is grxDNN's convention and
    what a transformer's activations already look like. grxBLAS is COLUMN-major.
    The bridge is the one identity that makes this cheap: a row-major (r, c)
    matrix with leading dimension ld IS the column-major (c, r) matrix with the
    same ld and the same bytes. No data moves; the transpose lives in the
    arguments.
    """
    B, H, S, D = q.shape
    qm = q.reshape(-1).copy()
    km = k.reshape(-1).copy()
    vm = v.reshape(-1).copy()
    scores = np.zeros(B * H * S * S, dtype=np.float64)
    out = np.zeros(B * H * S * D, dtype=np.float64)

    scale = 1.0 / np.sqrt(D)

    for head in range(B * H):
        qh, kh, vh = head * S * D, head * S * D, head * S * D
        sh, oh = head * S * S, head * S * D

        # GEMM 1: scores(row-major S x S, ld=S) = scale * Q K^T
        #
        # Read column-major, Q's memory is Q^T (D x S) and K's memory is K^T.
        # Writing the result column-major with ld=S produces scores^T, and
        # scores^T = (Q K^T)^T = K Q^T -- so A is K with transa=T and B is Q.
        # The scale is alpha; it does not need a kernel of its own.
        gemm_colmajor(True, False, S, S, D, scale,
                      km[kh:], D, qm[qh:], D, 0.0, scores[sh:], S)

        # Causal mask, on the row-major view: element (i, j) is at i*S + j and
        # a key at j > i is in the future.
        if causal:
            for i in range(S):
                for j in range(i + 1, S):
                    scores[sh + i * S + j] = -np.inf

        # Softmax along each row of the row-major view -- which is exactly what
        # grxdnnSoftmaxForward already does, over B*H*S rows of S columns.
        for i in range(S):
            row = scores[sh + i * S: sh + i * S + S]
            mx = row.max()
            e = np.exp(row - mx)
            scores[sh + i * S: sh + i * S + S] = e / e.sum()

        # GEMM 2: out(row-major S x D, ld=D) = P V
        #
        # out^T = (P V)^T = V^T P^T, and V^T / P^T are just V's and P's memory
        # read column-major. So both operands are untransposed and the operands
        # swap places.
        gemm_colmajor(False, False, D, S, S, 1.0,
                      vm[vh:], D, scores[sh:], S, 0.0, out[oh:], D)

    return out.reshape(B, H, S, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="PATH", nargs="?", const="attention_ref.bin")
    args = ap.parse_args()

    blobs = []
    worst_overall = 0.0
    ok = True

    for idx, (b, h, s, d, causal) in enumerate(CASES):
        q, k, v = make_inputs(b, h, s, d, seed=1000 + idx)
        want = torch_reference(q, k, v, causal).numpy()
        got = grxcp_attention(q.numpy(), k.numpy(), v.numpy(), causal)

        worst = float(np.max(np.abs(got - want)))
        worst_overall = max(worst_overall, worst)
        tag = f"B{b} H{h} S{s} D{d}{' causal' if causal else ''}"
        # float64 on both sides: anything above rounding here is a real
        # disagreement about layout, not an accumulation difference.
        good = worst < 1e-12
        ok = ok and good
        print(f"  {'ok  ' if good else 'FAIL'}  {tag:<28} worst |diff| {worst:.3g}")

        blobs.append((b, h, s, d, causal, q.numpy(), k.numpy(), v.numpy(), want))

    print(f"\n{'PASSED' if ok else 'FAILED'}: the column-major composition "
          f"reproduces torch (worst {worst_overall:.3g})")
    if not ok:
        return 1

    if args.write:
        with open(args.write, "wb") as f:
            f.write(struct.pack("<III", MAGIC, VERSION, len(blobs)))
            for b, h, s, d, causal, q, k, v, want in blobs:
                f.write(struct.pack("<IIIII", b, h, s, d, 1 if causal else 0))
                for arr in (q, k, v, want):
                    f.write(arr.astype("<f4").tobytes())
        print(f"wrote {args.write}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
