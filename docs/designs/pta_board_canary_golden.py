"""
The first board's canary: its golden tensors, checked.

The bundle in canary/ is what grx930's harness wrote for the canary
(`SHAPE=chain sim/pta_mnist.sh MNIST_GZ WORK golden`, c930/doc/
pta_error_model_design_note.md section 5, "The first board's canary, its golden
tensors"): the network B20 fixed and the owner's seed rule chose, its 1,000
acceptance vectors and 1,000 calibration vectors, every stage's declared-format
output, FC3's declared reference in the converter's codes as the owner ruled on
2026-10-14, and a double-precision pass for the task's quality. manifest.json
is GRXCP's candidate manifest for it.

This file is written apart from the harness. It holds every artifact to the
manifest's length and digest; the pixels, inputs, labels and ids of both sets
to the corpus worked out apart from MNIST's files; and then works every integer
tensor out again from the inputs and the integer weights, with arithmetic of
its own, and the double-precision logits from the network's file. Each has to
be the bundle's, byte for byte. Its own arithmetic is held first to cases
worked by hand at every rounding and saturation boundary.

WHAT THIS IS NOT. A frozen canary: the freeze is the owner's, on 2026-11-25. A
check of how the harness derived the integer weights, biases and shifts from
the trained network: those are the bundle's, and what they are is what is
checked. Device evidence of any kind.

Standard library only.  Run:  python3 docs/designs/pta_board_canary_golden.py
(--recompute-only skips the manifest's digests; --dir DIR checks a bundle
elsewhere.)
"""
import hashlib
import json
import os
import struct
import sys
from operator import mul

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else os.path.join(HERE, "canary")
N, C, IN, HID, OUT = 1000, 1000, 64, 64, 16
SH = (9, 9)                 # FC1's and FC2's rescale
S, H_A, H_W = 9, 2, 2       # FC3's converter step 2^S of operand sums; 8 - 6 bits off activations and weights
STEP = 1 << (S - H_A - H_W)     # the converter's step in the codes' own sums: 32
LIM = 128                   # an 8-bit converter's codes: -128..127
# the corpus, worked out apart from the harness from MNIST's own files (B20): the acceptance set's as
# B20 records them, and the calibration set's and the pixels by the same script
CORPUS = {"inputs_int8.bin": "d968e50d76c4c3bfe3b0d0020db73fc0a1bade8f794cba389216f0777f0f23c4",
          "labels_u8.bin": "d14ba0a1a3818edefa35518ed001ece84b7ac0eabe9394551c73928493175d3f",
          "ids.txt": "81e4e1694930066d6e678be2ac3b828a0677756e8dc0632048e605541c754711",
          "pixels_u8.bin": "49008ed471ac4de4f40e3779615eab182248422ead8bf269ad840424b48da444",
          "cal_inputs_int8.bin": "c23d0a98df1376ce6dd6a43fa94926d1a0f37978945274f2674c9a5586b34f1c",
          "cal_labels_u8.bin": "e9b9c3471c38b1ac45a646a16235eb657be54efdb54e30974f4f839d24a7c7a2"}


def raw(name):
    return open(os.path.join(DIR, name), "rb").read()


def i8(b):
    return list(struct.unpack(f"<{len(b)}b", b))


def u8(b):
    return list(b)


def i32(b):
    return list(struct.unpack(f"<{len(b) // 4}i", b))


def f64(b):
    return list(struct.unpack(f"<{len(b) // 8}d", b))


def rows(xs, width):
    return [xs[i:i + width] for i in range(0, len(xs), width)]


# ---- the declared format's arithmetic, written here ----------------------------------------------
def rescale(v, s):
    """A hidden layer's output: the ReLU, then round(v / 2^s), half up, held at 127."""
    v = max(v, 0)
    v = (v + (1 << (s - 1))) >> s if s else v
    return min(v, 127)


def act_code(x):
    """FC2's output as FC3's activation code: q(x, 6) at 8 bits, in its own step."""
    return min((x + 2) >> 2, 31)


def converter(s):
    """The converter's code of an exact sum in the codes' own units: floor(s / 32 + 1/2), and whether it
    was held.  Python's >> floors, as the tile's arithmetic shift does."""
    t = ((s * (1 << (H_A + H_W)) * 256 >> S) + 128) >> 8
    return max(-LIM, min(LIM - 1, t)), not -LIM <= t <= LIM - 1


def layer(x, w, b, s):
    return [rescale(sum(map(mul, x, wr)) + bn, s) for wr, bn in zip(w, b)]


def classify(codes, b):
    best = None
    for o, c in enumerate(codes):
        v = c * (1 << S) + b[o]
        if best is None or v > best[0]:
            best = (v, o)
    return best[1]


def net_floats(path):
    """The harness's network file: a 48-byte header, the depth and the host's layers for PTAMLP4, then
    each layer's weights and biases as 32-bit floats."""
    b = open(path, "rb").read()
    assert b[:8] == b"PTAMLP4\0", b[:8]
    din, wbits, seed = struct.unpack("<3i", b[8:20])
    hidden, host = struct.unpack("<2i", b[48:56])
    at, w, bias = 56, [], []
    for l, (i, o) in enumerate(((IN, HID), (HID, HID), (HID, OUT))):
        w.append(rows(list(struct.unpack(f"<{i * o}f", b[at:at + 4 * i * o])), i))
        at += 4 * i * o
        bias.append(list(struct.unpack(f"<{o}f", b[at:at + 4 * o])))
        at += 4 * o
    assert at == len(b) and (din, wbits, seed, hidden, host) == (8, 6, 1, 2, 2)
    return w, bias


def high_precision(px, w, b):
    """The network as trained, in double precision on the pixels over 255, as the harness's golden pass
    adds: the bias first, then each weight times its input in order."""
    x = [p / 255.0 for p in px]
    for l in range(3):
        out = []
        for wr, bn in zip(w[l], b[l]):
            z = bn
            for wk, xk in zip(wr, x):
                z += wk * xk
            out.append(z if l == 2 or z > 0.0 else 0.0)
        x = out
    return x


def has_float(x):
    if isinstance(x, float):
        return True
    if isinstance(x, dict):
        return any(map(has_float, x.values()))
    if isinstance(x, list):
        return any(map(has_float, x))
    return False


def hand_cases():
    """The arithmetic, at its boundaries, against figures worked by hand."""
    assert [rescale(v, 9) for v in (-5, 0, 255, 256, 767, 768, 65023, 65024, 1 << 30)] == [0, 0, 0, 1, 1, 2, 127, 127, 127]
    assert rescale(65279, 9) == 127 and rescale(64767, 9) == 126           # 126.5 rounds up, 126.498 down
    assert [act_code(x) for x in (0, 1, 2, 5, 6, 121, 122, 125, 126, 127)] == [0, 0, 1, 1, 2, 30, 31, 31, 31, 31]
    # the converter: 32 sum units a code, half up, held at -128..127
    assert [converter(s)[0] for s in (0, 15, 16, 47, 48, -16, -17, -48, -49)] == [0, 0, 1, 1, 2, 0, -1, -1, -2]
    assert converter(127 * 32 + 15) == (127, False) and converter(127 * 32 + 16) == (127, True)
    assert converter(-128 * 32 - 16) == (-128, False) and converter(-128 * 32 - 17) == (-128, True)
    assert STEP == 32
    # the class: the largest code at its step with its bias, the lowest index on a tie
    assert classify([0, 0, 1], [0, 0, 0]) == 2 and classify([1, 1, 0], [0, 0, 0]) == 0
    assert classify([1, 0], [0, 513]) == 1 and classify([1, 0], [0, 512]) == 0
    # a job of sixteen: 62 of them and one of eight
    assert divmod(N, 16) == (62, 8)


def main():
    recompute_only = "--recompute-only" in sys.argv[1:]
    hand_cases()
    manifest = json.loads(raw("manifest.json"))
    arts = {a["path"]: a for a in manifest["artifacts"]}
    files = sorted(f for f in os.listdir(DIR) if f not in ("manifest.json", "SHA256SUMS"))
    assert sorted(arts) == files, (sorted(arts), files)
    if not recompute_only:
        for f, a in arts.items():
            b = raw(f)
            assert len(b) == a["byte_length"] and hashlib.sha256(b).hexdigest() == a["sha256"], f
        sums = dict(reversed(line.split("  ")) for line in raw("SHA256SUMS").decode().splitlines())
        assert sums == {**{f: a["sha256"] for f, a in arts.items()}, "manifest.json": hashlib.sha256(raw("manifest.json")).hexdigest()}
    for f, d in CORPUS.items():
        assert hashlib.sha256(raw(f)).hexdigest() == d, f
    assert raw("ids.txt").decode() == "".join(f"mnist-t10k-{n:05d}\n" for n in range(N))
    assert raw("cal_ids.txt").decode() == "".join(f"mnist-train-{54000 + n:05d}\n" for n in range(C))

    px, x0 = rows(u8(raw("pixels_u8.bin")), IN), rows(i8(raw("inputs_int8.bin")), IN)
    labels, cal_labels = u8(raw("labels_u8.bin")), u8(raw("cal_labels_u8.bin"))
    w1, b1 = rows(i8(raw("fc1_weights_int8.bin")), IN), i32(raw("fc1_bias_int32.bin"))
    w2, b2 = rows(i8(raw("fc2_weights_int8.bin")), HID), i32(raw("fc2_bias_int32.bin"))
    w3, b3 = rows(i8(raw("fc3_weights_int8.bin")), HID), i32(raw("fc3_bias_int32.bin"))
    assert len(px) == len(x0) == len(labels) == N and len(cal_labels) == C
    assert len(w1) == len(w2) == HID and len(w3) == OUT and len(b1) == len(b2) == HID and len(b3) == OUT
    assert all(-32 <= v <= 31 for r in w3 for v in r) and all(0 <= v <= 9 for v in labels + cal_labels)
    # the inputs are the pixels as FC1 takes them
    assert all(x == (127 * p + 127) // 255 for pr, xr in zip(px, x0) for p, x in zip(pr, xr))
    # an int32 sum holds every product and bias, with room
    assert max(map(abs, b1 + b2)) + IN * 127 * 128 < 1 << 31

    got = {"fc1_relu_int8.bin": [], "fc2_relu_int8.bin": [], "fc3_act_codes_u8.bin": [], "fc3_sums_int32.bin": [],
           "fc3_codes_int8.bin": [], "class_ids_u8.bin": [], "cal_fc3_codes_int8.bin": []}
    held = 0
    lo, hi = LIM, -LIM
    for which, xs in (("acc", x0), ("cal", rows(i8(raw("cal_inputs_int8.bin")), IN))):
        for x in xs:
            a1 = layer(x, w1, b1, SH[0])
            a2 = layer(a1, w2, b2, SH[1])
            ca = [act_code(v) for v in a2]
            s = [sum(map(mul, ca, wr)) for wr in w3]
            cv = [converter(v) for v in s]
            codes = [c for c, _ in cv]
            held += sum(h for _, h in cv)
            if which == "cal":
                got["cal_fc3_codes_int8.bin"] += codes
                continue
            lo, hi = min(lo, *codes), max(hi, *codes)
            got["fc1_relu_int8.bin"] += a1
            got["fc2_relu_int8.bin"] += a2
            got["fc3_act_codes_u8.bin"] += ca
            got["fc3_sums_int32.bin"] += s
            got["fc3_codes_int8.bin"] += codes
            got["class_ids_u8.bin"].append(classify(codes, b3))
    assert held == 0, held
    for f, xs in got.items():
        b = raw(f)
        if f.endswith("int32.bin"):
            mine = struct.pack(f"<{len(xs)}i", *xs)
        elif "_u8" in f:
            mine = bytes(xs)
        else:
            mine = struct.pack(f"<{len(xs)}b", *xs)
        assert mine == b, f

    # the double-precision pass, from the network's own file
    w, b = net_floats(os.path.join(DIR, "canary_seed1.net"))
    hp = [high_precision(p, w, b) for p in px]
    theirs = rows(f64(raw("highprec_logits_f64.bin")), OUT)
    worst = max(abs(a - c) for hr, tr in zip(hp, theirs) for a, c in zip(hr, tr))
    assert worst == 0.0, worst
    top = [max(range(OUT), key=lambda o: (r[o], -o)) for r in hp]
    cls = got["class_ids_u8.bin"]
    declared_right = 100.0 * sum(c == y for c, y in zip(cls, labels)) / N
    hp_right = 100.0 * sum(t == y for t, y in zip(top, labels)) / N
    agree = 100.0 * sum(t == c for t, c in zip(top, cls)) / N
    line = manifest["generation"]["harness_line"]
    assert f"declared_right={declared_right:.2f} highprec_right={hp_right:.2f} highprec_agree={agree:.2f}" in line
    assert " clipped=0 tile_differ=0 of 32000 act_differ=0 of 256000 " in line
    assert manifest["freeze_authority"] == "Gridworkz" and manifest["calendar_freeze_date_utc"] == "2026-11-25"
    assert manifest["gate_f"]["reference"].startswith("fc3_declared_format_reference")
    # with no number but integers and no character past ASCII, keys sorted and no whitespace is RFC 8785's form
    assert all(c < 128 for c in raw("manifest.json")) and not has_float(manifest)
    canon = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    print("The first board's canary: its golden tensors, checked.")
    print(f"  {len(arts)} artifacts, {sum(a['byte_length'] for a in arts.values()):,} bytes"
          + ("" if recompute_only else ", each its manifest's length and SHA-256"))
    print("  the pixels, inputs, labels and ids of both sets: the corpus worked out apart from MNIST's files, byte for byte;")
    print("  the inputs the pixels as FC1 takes them")
    print(f"  worked out again here, byte for byte: {', '.join(sorted(got))}")
    print(f"  FC3's codes on the acceptance vectors from {lo} to {hi}, none held at the converter's range")
    print(f"  the double-precision logits: the bundle's to the last bit")
    print(f"  the declared format right on {declared_right:.2f}% of the acceptance vectors, the double-precision"
          f" network on {hp_right:.2f}%, their classes the same on {agree:.2f}%")
    print(f"  the manifest's canonical SHA-256 (RFC 8785's form): {hashlib.sha256(canon).hexdigest()}")
    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
