"""
The ADC survey, as far as the tile needs it: what a converter near 1 GS/s costs.

pta_power.py priced the tile's ADCs from three converters taken one at a time
from their papers, and said that B. Murmann's ADC Performance Survey was the
place to widen that.  This is the survey, read from the plots of his ISSCC 2022
short course ("Introduction to ADCs/DACs: Metrics, Topologies, Trade Space, and
Applications", February 2022), which draw every converter published at ISSCC
and the VLSI Circuits Symposium from 1997 to 2021.

HOW IT WAS READ.  The slides' plots are vector drawings, so every marker's
position is in the file to five figures and nothing was read off by eye.  Three
of them show the same converters on different axes:

    slide 45   conversion rate against SNDR
    slide 59   SNDR against energy a conversion, P / fs
    slide 64   conversion rate against the Schreier figure of merit,
               FoM_S = SNDR + 10 log10((fs / 2) / P)

A converter is a point in each, and the three have to agree: slide 59's point
gives a FoM_S, and slide 64 has to show it at slide 45's rate.  546 converters
are matched that way, of 596, 613 and 569 markers; the rest fall outside one
plot's axes.  Each axis is fitted to its own grid lines, to within 0.0014 of a
decade or a decibel.

HOW IT WAS CHECKED.  Two of the matched points are converters pta_power.py
already held from their own papers, and the plots return what the papers say:

    Kull et al., JSSC 2013        1.2 GS/s, 39.3 dB, 3.1 mW   read: 1.20, 39.3, 3.06
    Verbruggen et al., JSSC 2010  2.2 GS/s, 31.1 dB, 2.6 mW   read: 2.20, 31.1, 2.60

The plots also return the slides' own annotations: a best FoM_S of 186.8 dB
beside the 185 dB line drawn on slide 64, and a largest fs x 2^ENOB of 4.65
beside the "5 TS/s-levels" frontier of slide 45.

WHAT IS KEPT HERE.  The matched converters between 30 and 50 dB of SNDR at
40 MS/s and above: the low-resolution, fast corner a photonic tile's column
lives in.  Rate in GS/s, SNDR in dB, power in mW.  The survey itself is
B. Murmann, "ADC Performance Survey 1997-2021", github.com/bmurmann/ADC-survey;
these rows are its data as his slides plot it, not a new measurement of
anything.

WHAT IT IS NOT.  The plots carry no names, so a row is a converter and not a
citation.  They carry no architecture, node or year either, and 1997's parts
sit beside 2021's: the median of this table is a history, and only its best
end says what can be built now.  The power is the converter's as its paper
reported it, which usually leaves out the reference and the clock.

With the slides' PDF as an argument this re-reads the plots and holds the
table to them.  Without, it reports from the table.

Standard library only.  Run:  python3 docs/designs/pta_adc_survey.py [SLIDES.pdf]
"""
import math
import re
import sys
import zlib

# fs in GS/s, SNDR in dB, P in mW
TABLE = (
    (0.04, 39.0, 8.599),
    (0.04, 44.5, 0.113),
    (0.04, 44.5, 30.01),
    (0.04, 50.0, 53.03),
    (0.05, 32.0, 0.2401),
    (0.05, 47.7, 1.44),
    (0.075, 43.3, 70.04),
    (0.08, 40.0, 0.024),
    (0.1, 41.5, 30.01),
    (0.125, 47.5, 21.0),
    (0.15, 40.0, 0.1335),
    (0.15, 45.4, 70.99),
    (0.2, 37.7, 1.0),
    (0.2, 39.3, 1.381),
    (0.2, 40.3, 8.501),
    (0.2, 47.3, 30.02),
    (0.3, 38.4, 34.02),
    (0.32, 50.0, 40.01),
    (0.4, 40.4, 3.999),
    (0.45, 36.0, 49.99),
    (0.6, 31.0, 10.01),
    (0.6, 33.0, 5.303),
    (0.6, 37.0, 38.02),
    (0.6, 40.0, 207.0),
    (0.6, 43.0, 30.01),
    (0.6, 46.9, 200.1),
    (0.75, 43.3, 4.499),
    (0.8, 33.7, 12.01),
    (0.8, 44.2, 300.2),
    (0.82, 37.4, 4.261),
    (0.9, 33.4, 0.6999),
    (0.95, 45.0, 2.301),
    (1.0, 33.8, 55.03),
    (1.0, 34.6, 1.261),
    (1.0, 38.9, 26.5),
    (1.0, 40.0, 16.0),
    (1.0, 42.7, 3.801),
    (1.0, 45.5, 2.55),
    (1.1, 40.9, 46.02),
    (1.1, 45.0, 4.002),
    (1.2, 39.3, 3.06),
    (1.25, 32.0, 32.02),
    (1.3, 33.1, 22.0),
    (1.35, 48.1, 168.5),
    (1.6, 45.8, 1269.0),
    (1.62, 48.0, 93.01),
    (1.75, 30.0, 7.602),
    (2.0, 40.7, 21.0),
    (2.2, 31.1, 2.601),
    (2.2, 34.0, 50.01),
    (2.2, 37.4, 27.41),
    (2.4, 40.1, 4.999),
    (2.6, 48.5, 480.0),
    (2.7, 33.6, 50.02),
    (2.8, 48.2, 44.62),
    (3.0, 33.1, 11.01),
    (3.3, 34.2, 5.503),
    (3.5, 31.2, 98.02),
    (3.6, 50.0, 795.3),
    (5.0, 30.8, 5.501),
    (5.0, 30.9, 8.502),
    (5.0, 32.0, 320.1),
    (5.0, 45.2, 22.71),
    (5.0, 46.1, 150.0),
    (5.0, 48.5, 29.01),
    (5.4, 50.0, 500.1),
    (8.0, 42.4, 26.01),
    (8.0, 49.0, 300.1),
    (8.8, 37.0, 35.01),
    (10.0, 30.3, 79.03),
    (10.0, 32.5, 240.0),
    (10.0, 33.8, 31.99),
    (10.0, 40.1, 50.81),
    (10.3, 31.6, 390.3),
    (16.0, 37.0, 320.1),
    (18.0, 48.0, 1300.0),
    (20.0, 30.7, 69.54),
    (20.0, 32.5, 175.0),
    (20.0, 38.8, 129.3),
    (28.0, 31.5, 280.0),
    (30.0, 32.0, 53.49),
    (32.0, 37.8, 199.1),
    (64.0, 37.6, 950.6),
    (72.0, 34.2, 235.1),
    (90.0, 33.0, 667.1),
    (100.0, 35.1, 311.4),
)

MATCHED, MARKERS = 546, (596, 613, 569)
PUBLISHED = (                    # name, GS/s, dB, mW: from the papers themselves
    ("Kull et al., JSSC 2013", 1.2, 39.3, 3.1),
    ("Verbruggen et al., JSSC 2010", 2.2, 31.1, 2.6),
)
WINDOW = 2.5                     # "near" a rate: within this factor of it


def enob(sndr_db):
    return (sndr_db - 1.76) / 6.02


def walden_j(row):
    """Joules a conversion step, P / (2^ENOB fs)."""
    fs, sndr, p = row
    return p * 1e-3 / (2 ** enob(sndr) * fs * 1e9)


def near(fs_hz, table=TABLE):
    """The table's converters within WINDOW of a rate, most efficient first."""
    g = fs_hz / 1e9
    return sorted((r for r in table if g / WINDOW <= r[0] <= g * WINDOW), key=walden_j)


def frontier(fs_hz, table=TABLE):
    """(best, quartile) figure of merit near a rate, in joules a step.

    The quartile is the worst of the best quarter: what a part in the leading
    quarter of everything published near this rate achieves.
    """
    rows = near(fs_hz, table)
    return walden_j(rows[0]), walden_j(rows[len(rows) // 4])


# ---- re-reading the slides ----------------------------------------------------
NUM = r'(-?\d+(?:\.\d+)?)'


def _inflate(raw):
    try:
        return zlib.decompress(raw)
    except zlib.error:
        return zlib.decompressobj().decompress(raw)


def _objects(data):
    """Every object in the file: number -> (dictionary bytes, decoded stream or None)."""
    objs = {}
    for m in re.finditer(rb'(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj', data, re.S):
        body = m.group(3)
        s = re.search(rb'stream\r?\n', body)
        if not s:
            objs[int(m.group(1))] = (body, None)
            continue
        head, raw = body[:s.start()], body[s.end():]
        end = raw.rfind(b'endstream')
        raw = (raw[:end] if end >= 0 else raw).rstrip(b'\r\n')
        objs[int(m.group(1))] = (head, _inflate(raw) if b'/FlateDecode' in head else raw)
    for head, stream in list(objs.values()):
        if stream is None or b'/ObjStm' not in head:
            continue
        n = int(re.search(rb'/N\s+(\d+)', head).group(1))
        first = int(re.search(rb'/First\s+(\d+)', head).group(1))
        nums = stream[:first].split()
        pairs = [(int(nums[2 * i]), int(nums[2 * i + 1])) for i in range(n)]
        for i, (num, off) in enumerate(pairs):
            end = pairs[i + 1][1] if i + 1 < n else len(stream) - first
            objs.setdefault(num, (stream[first + off:first + end], None))
    return objs


def _value(objs, head, key):
    """/key's value in a dictionary, following one reference."""
    m = re.search(rb'/' + key + rb'\b\s*', head)
    if not m:
        return b''
    rest = head[m.end():]
    r = re.match(rb'(\d+)\s+\d+\s+R', rest)
    return objs[int(r.group(1))][0] if r else rest


def _pages(objs):
    def walk(num, out, seen):
        if num in seen or num not in objs:
            return
        seen.add(num)
        head = objs[num][0]
        if re.search(rb'/Type\s*/Pages\b', head):
            kids = _value(objs, head, b'Kids')
            for k in re.findall(rb'(\d+)\s+\d+\s+R', kids[:kids.index(b']') + 1]):
                walk(int(k), out, seen)
        elif re.search(rb'/Type\s*/Page\b', head):
            out.append(num)

    out, seen = [], set()
    for head, _ in objs.values():
        if re.search(rb'/Type\s*/Catalog\b', head):
            root = re.search(rb'/Pages\s+(\d+)\s+\d+\s+R', head)
            walk(int(root.group(1)), out, seen)
    return out


def _plot(objs, pages, number):
    """A slide's plot: its grid lines and its markers' centres, in drawing units."""
    head = objs[pages[number - 1]][0]
    c = re.search(rb'/Contents\s+(\d+)\s+\d+\s+R', head)
    name = re.search(rb'/(Meta\d+)\s+Do', objs[int(c.group(1))][1]).group(1)
    xo = _value(objs, _value(objs, head, b'Resources'), b'XObject')
    form = re.search(rb'/' + name + rb'\s+(\d+)\s+\d+\s+R', xo)
    s = objs[int(form.group(1))][1].decode('latin-1')
    area = re.search(NUM + ' ' + NUM + ' ' + NUM + ' ' + NUM + r' re\s*\nf\*', s)
    w, h = float(area.group(3)), float(area.group(4))
    vx, hy = set(), set()
    for m in re.finditer(NUM + ' ' + NUM + r' m\s*\n' + NUM + ' ' + NUM + r' l\s*\nS', s):
        ax, ay, bx, by = map(float, m.groups())
        if ax == bx and abs(abs(by - ay) - h) < 1.0:
            vx.add(ax)
        elif ay == by and abs(abs(bx - ax) - w) < 1.0:
            hy.add(ay)
    pts = []
    for m in re.finditer(NUM + ' ' + NUM + ' ' + NUM + ' ' + NUM + r' re\s*\nf\*', s[s.index('0 0 1 rg'):]):
        x, y, mw, mh = map(float, m.groups())
        if 3.0 < mw <= 13.0 and 3.0 < mh <= 13.0:
            pts.append((x + mw / 2, y + mh / 2))
    return sorted(vx), sorted(hy), pts


def _axis(positions, values):
    """Drawing units to data, least squares through the grid lines; and the worst miss."""
    n = len(positions)
    assert n == len(values), (n, len(values))
    mp, mv = sum(positions) / n, sum(values) / n
    k = sum((p - mp) * (v - mv) for p, v in zip(positions, values)) / sum((p - mp) ** 2 for p in positions)
    return (lambda p: mv + k * (p - mp)), max(abs(mv + k * (p - mp) - v) for p, v in zip(positions, values))


def read_slides(path):
    """Every converter the three plots agree on, as (GS/s, dB, mW); and the counts."""
    objs = _objects(open(path, 'rb').read())
    pages = _pages(objs)
    (avx, ahy, a), (bvx, bhy, b), (cvx, chy, c) = (_plot(objs, pages, n) for n in (45, 59, 64))
    ax, e1 = _axis(avx, list(range(4, 12)))                 # log10 of 10^4 .. 10^11 samples a second
    ay, e2 = _axis(ahy, [20, 40, 60, 80, 100])              # dB
    bx, e3 = _axis(bvx, list(range(10, 111, 10)))           # dB
    by, e4 = _axis(bhy, [-14, -12, -10, -8, -6])            # log10 of joules
    cx, e5 = _axis(cvx, list(range(3, 12)))
    cy, e6 = _axis(chy, list(range(130, 191, 10)))          # dB
    A = [(ax(x), ay(y)) for x, y in a]
    B = [(bx(x), by(y)) for x, y in b]
    C = [(cx(x), cy(y)) for x, y in c]
    rows, used_b, used_c = [], set(), set()
    for lf, sndr in A:
        best = None
        for i, (s2, le) in enumerate(B):
            if i in used_b or abs(s2 - sndr) > 0.06:
                continue
            foms = s2 - 10 * (math.log10(2.0) + le)
            for j, (lf2, f2) in enumerate(C):
                if j in used_c or abs(lf2 - lf) > 0.004 or abs(f2 - foms) > 0.12:
                    continue
                err = abs(s2 - sndr) / 0.06 + abs(lf2 - lf) / 0.004 + abs(f2 - foms) / 0.12
                if best is None or err < best[0]:
                    best = (err, i, j, le, f2)
        if best:
            used_b.add(best[1])
            used_c.add(best[2])
            rows.append((10 ** (lf - 9), sndr, 10 ** (best[3] + lf + 3), best[4]))
    return rows, (len(A), len(B), len(C)), max(e1, e2, e3, e4, e5, e6)


def kept(rows):
    """The part of the survey this file keeps, rounded as TABLE is."""
    out = [(float('%.3g' % fs), round(sndr, 1), float('%.4g' % p))
           for fs, sndr, p, _ in rows if 30.0 <= round(sndr, 1) <= 50.0 and fs >= 0.04]
    return tuple(sorted(out))


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    print("The ADC survey, near where a photonic tile's column converts.")
    print(f"{len(TABLE)} converters between 30 and 50 dB at 40 MS/s and above, of {MATCHED} read from the")
    print("plots of B. Murmann's ISSCC 2022 short course.  Energy a conversion step is")
    print("P / (2^ENOB fs).")

    section("1. Near each shot rate, within a factor of 2.5")
    print(f"  {'rate':>9}{'parts':>7}{'best':>10}{'quartile':>10}{'median':>10}   the best part")
    for fs in (0.1e9, 1e9, 10e9):
        rows = near(fs)
        f = [walden_j(r) * 1e15 for r in rows]
        b = rows[0]
        print(f"  {fs / 1e9:>4.1f} GS/s{len(rows):>7}{f[0]:>7.1f} fJ{f[len(f) // 4]:>7.1f} fJ{f[len(f) // 2]:>7.0f} fJ"
              f"   {b[0]:.2f} GS/s, {b[1]:.1f} dB, {b[2]:.2f} mW")
    print("  The median is a history of the field since 1997.  The best and the")
    print("  quartile are what the leading parts do.")

    section("2. What that is in milliwatts, at an effective B bits")
    print("  Below 50 dB the survey's own trend is twice the energy a bit (its slide 59,")
    print("  \"technology limited\"), which is what a fixed figure of merit says:")
    print(f"  {'rate':>9}" + "".join(f"{f'{b} bits':>20}" for b in (6, 7, 8)))
    for fs in (0.1e9, 1e9, 10e9):
        lo, hi = frontier(fs)
        print(f"  {fs / 1e9:>4.1f} GS/s" + "".join(
            f"{lo * 2 ** b * fs * 1e3:>9.2f} to {hi * 2 ** b * fs * 1e3:<7.2f}" for b in (6, 7, 8)) + " mW")

    section("3. The parts near 1 GS/s, most efficient first")
    print(f"  {'GS/s':>6}{'SNDR':>8}{'bits':>7}{'mW':>9}{'fJ a step':>11}")
    for r in near(1e9)[:12]:
        print(f"  {r[0]:>6.2f}{r[1]:>6.1f} dB{enob(r[1]):>7.2f}{r[2]:>9.2f}{walden_j(r) * 1e15:>11.1f}")
    print(f"  ... and {len(near(1e9)) - 12} more, to {walden_j(near(1e9)[-1]) * 1e15:,.0f} fJ.")

    checks()
    if len(sys.argv) > 1:
        rows, counts, fit = read_slides(sys.argv[1])
        got = kept(rows)
        print(f"\nRe-read from {sys.argv[1]}: {counts[0]}, {counts[1]} and {counts[2]} markers,"
              f" {len(rows)} matched, grid fit {fit:.4f}.")
        assert (len(rows), counts) == (MATCHED, MARKERS), (len(rows), counts)
        assert fit < 0.002, fit
        assert got == tuple(sorted(TABLE)), sorted(set(got) ^ set(TABLE))[:6]
        assert abs(max(r[3] for r in rows) - 186.8) < 0.05
        assert abs(max(r[0] * 1e9 * 2 ** enob(r[1]) for r in rows) / 1e12 - 4.65) < 0.01
        print("The table is what the slides plot.")


def checks():
    """Every claim above, as an assert."""
    assert len(TABLE) == len(set(TABLE)) == 86
    assert all(30.0 <= r[1] <= 50.0 and r[0] >= 0.04 for r in TABLE)

    # 1. The plots return the two converters whose papers were read directly:
    #    rate and SNDR as published, power within 2%.
    for name, fs, sndr, p in PUBLISHED:
        hit = [r for r in TABLE if abs(r[0] - fs) < 0.005 and abs(r[1] - sndr) < 0.05]
        assert len(hit) == 1 and abs(hit[0][2] / p - 1.0) < 0.02, (name, hit)

    # 2. Near 1 GS/s: 34 parts, the best at 16.6 fJ a step and the quartile at 40.
    rows = near(1e9)
    best, quart = frontier(1e9)
    assert len(rows) == 34, len(rows)
    assert abs(best * 1e15 - 16.6) < 0.1 and abs(quart * 1e15 - 40.3) < 0.1, (best, quart)
    assert rows[0] == (1.0, 45.5, 2.55)

    # 3. The best part near 1 GS/s is better than seven bits and under 3 mW, so
    #    version 1's converter exists as a published part and is not a projection.
    assert enob(rows[0][1]) > 7.0 and rows[0][2] < 3.0

    # 4. The frontier is dearer the faster the converter: 3.7, 16.6 and 26.7 fJ
    #    at 0.1, 1 and 10 GS/s.
    f = [frontier(fs)[0] * 1e15 for fs in (0.1e9, 1e9, 10e9)]
    assert f[0] < f[1] < f[2] and all(abs(a - b) < 0.1 for a, b in zip(f, (3.7, 16.6, 26.7))), f

    # 5. frontier() is the head and the quarter mark of near(), and near() is
    #    sorted by the figure of merit.
    for fs in (0.1e9, 1e9, 10e9):
        rows = near(fs)
        j = [walden_j(r) for r in rows]
        assert j == sorted(j) and frontier(fs) == (j[0], j[len(j) // 4])

    print("\nAll checks pass.")


if __name__ == "__main__":
    main()
