"""
The first board's canary: Gate F's band.

grx930's harness ran the canary, and the four other networks of its recipe,
through its model of the PTA, with FC3 held to its converter's codes, the
declared reference the owner ruled on 2026-10-14 (B21), and one scale fixed
on the calibration set (`SHAPE=chain sim/pta_mnist.sh MNIST_GZ WORK band`;
c930/doc/pta_error_model_design_note.md section 5, "The first board's
canary, its band"). The rows: the format alone; version 2's budget with no
source; the same lit by B16's source, the board's own case; lit at 0.4 to
1.4 times every row of the budget; and lit on ten sources, each run a
hundred times. pta_board_band_runs.csv.gz is every run's figures as the
harness printed them, and TABLES is what the mode printed.

This file works every table out again from the runs. Then it applies the
rule written before the sweep (band_rule below, 2026-10-15T01:13:20Z): the
band's edges, the class threshold, the tail rule, and what moves from
repeat to repeat. And it says how often a tile at each factor of its budget
passes them, and how the predictions written with the rule fared. The owner
ruled the rule's figures (B22), and canary/manifest.json carries them from
revision 0.2-candidate on: this file holds it to them and to these runs.

WHAT THIS IS NOT. A ruling: the band is the owner's to set (B19). Device
evidence of any kind. A band for any network but the canary. A model of a
source other than B16's.

Standard library only.  Run:  python3 docs/designs/pta_board_band.py
"""
import gzip
import hashlib
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "pta_board_band_runs.csv.gz")
RUNS_SHA256 = "0df5f693167be82e27fd47a4d82104ca4e779c415657aa723b82890e487dc03f"     # of the CSV the file holds, unpacked
TAIL = 32                   # the harness counts outputs off by 1 to 32 codes
SEL = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32)   # the tail table's columns
ROWS = ["format only", "as budgeted", "as budgeted, lit", "lit, 0.4 of the budget", "lit, 0.6 of the budget",
        "lit, 0.8 of the budget", "lit, 1.2 of the budget", "lit, 1.4 of the budget", "as budgeted, lit, ten sources"]
FACTOR = [0.0, 1.0, 1.0, 0.4, 0.6, 0.8, 1.2, 1.4, 1.0]
CANARY, DRAWS, OTHER, OUTPUTS = 1, 1000, 100, 16000
FMT, DARK, LIT, LOWER_ROW, TEN = 0, 1, 2, 4, 8      # indices into ROWS
FIRST_SWEEP = "stopped before its check: 20 runs of the 1.4 row refused, which the mode did not yet record"
TABLES = r"""== band: 8805 runs, each of them the network, the row and the draw its place says: yes
== band: runs refused for a line of the source all but dark when the lines were read, which
== the reading cannot correct, as a board's calibration would refuse that source: each row's
== runs on every network, and of them refused.  The tables after are over the runs that ran
                                 runs  refused
format only                         5        0
as budgeted                      1400        0
as budgeted, lit                 1400        0
lit, 0.4 of the budget           1000        0
lit, 0.6 of the budget           1000        0
lit, 0.8 of the budget           1000        0
lit, 1.2 of the budget           1000        0
lit, 1.4 of the budget           1000       20
as budgeted, lit, ten sources    1000        0
== band: the format alone, each network: how often its reference, FC3's codes, is right, and the
== tile's accuracy, percent; its rms error over the reference's rms, its largest error in codes,
== and the scale
                      ref      acc      rms      max    scale
seed 1, canary      96.60    96.60    0.000     0.00  1.00000
seed 2              96.30    96.30    0.000     0.00  1.00000
seed 3              96.30    96.30    0.000     0.00  1.00000
seed 4              96.20    96.20    0.000     0.00  1.00000
seed 5              96.70    96.70    0.000     0.00  1.00000
== band: the canary, seed 1, over its 1000 draws: the last layer's rms error over its
== reference's rms with the scale, percent: mean, standard deviation, smallest, 1st percentile,
== median, 99th percentile and largest
                                  mean     sd   least      1%  median     99%    most
as budgeted                      3.322  0.020   3.260   3.276   3.321   3.367   3.385
as budgeted, lit                 5.180  0.533   4.040   4.164   5.129   6.598   7.212
lit, 0.4 of the budget           2.219  0.156   1.909   1.953   2.195   2.652   2.881
lit, 0.6 of the budget           3.118  0.271   2.567   2.643   3.084   3.886   4.257
lit, 0.8 of the budget           4.101  0.399   3.261   3.368   4.058   5.204   5.686
lit, 1.2 of the budget           6.283  0.680   4.789   4.992   6.223   8.090   8.788
lit, 1.4 of the budget           7.424  0.829   5.534   5.780   7.370   9.599  10.408
as budgeted, lit, ten sources    4.879  0.474   4.170   4.228   4.784   5.763   5.823
== band: the same without the scale
                                  mean     sd   least      1%  median     99%    most
as budgeted                      4.711  0.035   4.595   4.636   4.712   4.790   4.814
as budgeted, lit                 5.538  0.472   4.375   4.611   5.498   6.831   7.419
lit, 0.4 of the budget           2.385  0.141   2.049   2.102   2.369   2.777   2.971
lit, 0.6 of the budget           3.359  0.243   2.775   2.889   3.331   4.030   4.381
lit, 0.8 of the budget           4.406  0.356   3.534   3.710   4.372   5.392   5.857
lit, 1.2 of the budget           6.676  0.596   5.239   5.509   6.635   8.293   9.002
lit, 1.4 of the budget           7.837  0.719   6.127   6.407   7.779   9.838  10.571
as budgeted, lit, ten sources    5.306  0.423   4.242   4.361   5.361   6.037   6.125
== band: the canary: how often the class with the scale is the reference's, percent, and the
== largest error in codes with the scale, over its draws
                                          mean     sd   least      1%  median     99%    most
as budgeted: class                      99.678  0.151  99.100  99.300  99.700 100.000 100.000
as budgeted: largest                     5.772  0.497   4.820   4.960   5.640   7.020   7.790
as budgeted, lit: class                 99.361  0.238  98.500  98.800  99.400  99.800 100.000
as budgeted, lit: largest               11.153  2.247   6.530   7.330  10.820  18.250  21.910
lit, 0.4 of the budget: class           99.750  0.133  99.300  99.400  99.800 100.000 100.000
lit, 0.4 of the budget: largest          4.510  0.852   2.750   3.190   4.350   7.000   9.270
lit, 0.6 of the budget: class           99.654  0.161  99.000  99.200  99.700 100.000 100.000
lit, 0.6 of the budget: largest          6.626  1.316   4.110   4.410   6.380  10.230  12.990
lit, 0.8 of the budget: class           99.536  0.193  98.800  99.000  99.500  99.900 100.000
lit, 0.8 of the budget: largest          8.852  1.794   5.270   5.900   8.550  14.220  17.290
lit, 1.2 of the budget: class           99.141  0.283  98.100  98.400  99.200  99.700 100.000
lit, 1.2 of the budget: largest         13.484  2.732   7.780   8.870  13.030  21.900  26.340
lit, 1.4 of the budget: class           98.885  0.339  97.600  98.000  98.900  99.600  99.800
lit, 1.4 of the budget: largest         15.793  3.155   8.590  10.400  15.310  25.800  31.130
as budgeted, lit, ten sources: class    99.392  0.232  98.500  98.800  99.400  99.800 100.000
as budgeted, lit, ten sources: largest   9.916  1.409   6.880   7.390   9.860  13.490  15.210
== band: the canary: outputs of the 16,000 a run that are t or more codes off with the scale,
== to the nearest code: the mean over its draws, and the most
                                        t=1          t=2          t=3          t=4          t=6          t=8         t=12         t=16         t=20         t=24         t=28         t=32
as budgeted                    10411.0 10625  3275.2 3453   772.7  860   123.2  159     1.1    6     0.0    1     0.0    0     0.0    0     0.0    0     0.0    0     0.0    0     0.0    0
as budgeted, lit               12198.9 13137  6119.4 8219  2688.1 4646  1095.5 2329   184.1  774    38.3  385     2.1  118     0.1   20     0.0    2     0.0    0     0.0    0     0.0    0
lit, 0.4 of the budget          8192.9 9262  1060.3 1953   106.8  544    14.8  238     0.4   31     0.0    2     0.0    0     0.0    0     0.0    0     0.0    0     0.0    0     0.0    0
lit, 0.6 of the budget         10025.7 11105  2723.0 4121   583.6 1380   131.1  631    10.1  222     0.9   67     0.0    2     0.0    0     0.0    0     0.0    0     0.0    0     0.0    0
lit, 0.8 of the budget         11287.2 12301  4484.5 6320  1490.8 2827   467.5 1234    57.3  455     8.9  223     0.2   28     0.0    2     0.0    0     0.0    0     0.0    0     0.0    0
lit, 1.2 of the budget         12843.7 13718  7457.9 9541  3910.7 6244  1918.4 3807   428.3 1299   105.9  606     8.8  218     0.8   69     0.1   18     0.0    2     0.0    0     0.0    0
lit, 1.4 of the budget         13326.3 14190  8564.5 10606  5079.5 7625  2845.2 5182   807.3 2045   230.7  921    24.5  326     3.0  145     0.4   44     0.0   15     0.0    2     0.0    0
as budgeted, lit, ten sources  12055.0 12852  5835.1 7172  2436.1 3516   920.7 1609   116.4  308    14.2   56     0.1    2     0.0    0     0.0    0     0.0    0     0.0    0     0.0    0
== band: the five networks on the first 100 draws of the budget's two rows: the rms error with
== the scale, percent, and the class with it, over each network's draws
                                               mean     sd   least    most   class   least
as budgeted                seed 1, canary     3.318  0.021   3.276   3.370   99.68   99.30
as budgeted                seed 2             3.109  0.020   3.065   3.159   99.59   99.10
as budgeted                seed 3             3.366  0.022   3.318   3.422   99.59   99.30
as budgeted                seed 4             3.422  0.023   3.356   3.483   99.49   99.10
as budgeted                seed 5             3.216  0.018   3.175   3.256   99.62   99.20
as budgeted, lit           seed 1, canary     5.236  0.544   4.246   6.653   99.37   98.90
as budgeted, lit           seed 2             4.994  0.657   3.942   7.473   99.34   98.80
as budgeted, lit           seed 3             4.988  0.423   4.051   6.288   99.35   98.70
as budgeted, lit           seed 4             5.007  0.481   4.093   6.395   99.35   98.90
as budgeted, lit           seed 5             5.126  0.519   4.015   6.438   99.48   99.00
== band: the canary lit at its budget on ten sources, the lit row's first ten draws' line
== levels, a draw's the one its last digit says: the rms error with the scale, percent, over
== each source's repeats; and the standard deviation from repeat to repeat, pooled within a
== source, against that of the ten sources' means
                            runs    mean     sd   least    most
source 0                     100   4.693  0.045   4.595   4.811
source 1                     100   5.339  0.052   5.235   5.464
source 2                     100   4.890  0.044   4.777   4.983
source 3                     100   4.275  0.035   4.170   4.366
source 4                     100   4.320  0.038   4.186   4.415
source 5                     100   5.061  0.040   4.979   5.158
source 6                     100   4.532  0.039   4.437   4.640
source 7                     100   4.520  0.035   4.453   4.622
source 8                     100   5.460  0.047   5.349   5.559
source 9                     100   5.703  0.052   5.583   5.823
repeat to repeat            1000   4.879  0.043
source to source              10   4.879  0.497
"""


# ---- the runs -----------------------------------------------------------------------------------
def load():
    raw = gzip.open(RUNS, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == RUNS_SHA256, "the runs are not the file this model holds"
    lines = raw.decode("ascii").split("\n")
    assert lines[-1] == ""
    head = lines[0].split(",")
    assert head[-1] == "stail"
    runs = []
    for text in lines[1:-1]:
        f = text.split(",")
        assert len(f) == len(head), text
        r = {k: (int(x) if x != "" else None) for k, x in zip(head[:-1], f[:-1])}
        r["row"] -= 1                                 # the harness counts rows from one
        r["refused"] = r["srms"] is None
        st = [int(x) for x in f[-1].split(";")] if f[-1] else []
        assert len(st) <= TAIL and all(a >= b for a, b in zip(st, st[1:]))
        r["st"] = st + [0] * (TAIL - len(st))         # outputs t + 1 codes off or more, with the scale
        runs.append(r)
    return runs


RUN = load()


def of(row, seed=CANARY, most=None, ran=True):
    """A row's runs on a network, in the order of their files' names, those that ran unless asked."""
    got = [r for r in RUN if r["row"] == row and r["seed"] == seed and (most is None or r["draw"] < most)]
    got.sort(key=lambda r: "d%d.txt" % r["draw"])
    return [r for r in got if not r["refused"]] if ran else got


# ---- the mode's tables, worked out again ---------------------------------------------------------
def ofmt(x):
    """A number as awk prints it and reads it back: an integer as such, else to six figures."""
    return float("%d" % x) if x == int(x) else float("%.6g" % x)


def dist(xs):
    """band_dist: mean, standard deviation, smallest, 1st percentile, median, 99th, largest."""
    x = sorted(xs)
    n = len(x)
    t = t2 = 0.0
    for v in x:
        t += v
        t2 += v * v
    m = t / n
    sd = math.sqrt((t2 - n * m * m) / (n - 1)) if n > 1 else 0.0

    def at(p):
        k = int(p * n)
        if k < p * n:
            k += 1
        return x[max(k, 1) - 1]

    return " %7.3f %6.3f %7.3f %7.3f %7.3f %7.3f %7.3f" % (m, sd, x[0], at(0.01), at(0.5), at(0.99), x[-1])


def srms(r):
    return r["srms"] / 1e5


def tables():
    out = []
    w = out.append
    n_run = len(RUN)
    w("== band: %d runs, each of them the network, the row and the draw its place says: yes" % n_run)
    w("== band: runs refused for a line of the source all but dark when the lines were read, which")
    w("== the reading cannot correct, as a board's calibration would refuse that source: each row's")
    w("== runs on every network, and of them refused.  The tables after are over the runs that ran")
    w("%-30s %6s %8s" % ("", "runs", "refused"))
    for i, row in enumerate(ROWS):
        n = 5 if i == FMT else DRAWS + 4 * OTHER if i in (DARK, LIT) else DRAWS
        assert n == sum(1 for r in RUN if r["row"] == i)
        w("%-30s %6d %8d" % (row, n, sum(r["refused"] for r in RUN if r["row"] == i)))
    w("== band: the format alone, each network: how often its reference, FC3's codes, is right, and the")
    w("== tile's accuracy, percent; its rms error over the reference's rms, its largest error in codes,")
    w("== and the scale")
    w("%-16s %8s %8s %8s %8s %8s" % ("", "ref", "acc", "rms", "max", "scale"))
    for sd in range(1, 6):
        r = of(FMT, sd)[0]
        w("%-16s %8.2f %8.2f %8.3f %8.2f %8.5f" % ("seed %d%s" % (sd, ", canary" if sd == CANARY else ""), r["ref"] / 100,
                                                   r["acc"] / 100, 100 * (r["rms"] / 1e5), r["maxcode"] / 100, r["scale"] / 1e5))
    w("== band: the canary, seed %d, over its %d draws: the last layer's rms error over its" % (CANARY, DRAWS))
    w("== reference's rms with the scale, percent: mean, standard deviation, smallest, 1st percentile,")
    w("== median, 99th percentile and largest")
    hdr = "%-30s %7s %6s %7s %7s %7s %7s %7s" % ("", "mean", "sd", "least", "1%", "median", "99%", "most")
    w(hdr)
    for i in range(1, len(ROWS)):
        w("%-30s" % ROWS[i] + dist([ofmt(srms(r) * 100) for r in of(i)]))
    w("== band: the same without the scale")
    w(hdr)
    for i in range(1, len(ROWS)):
        w("%-30s" % ROWS[i] + dist([ofmt(r["rms"] / 1e5 * 100) for r in of(i)]))
    w("== band: the canary: how often the class with the scale is the reference's, percent, and the")
    w("== largest error in codes with the scale, over its draws")
    w("%-38s %7s %6s %7s %7s %7s %7s %7s" % ("", "mean", "sd", "least", "1%", "median", "99%", "most"))
    for i in range(1, len(ROWS)):
        w("%-38s" % (ROWS[i] + ": class") + dist([ofmt(r["sagree"] / 100 * 1) for r in of(i)]))
        w("%-38s" % (ROWS[i] + ": largest") + dist([ofmt(r["smaxcode"] / 100 * 1) for r in of(i)]))
    w("== band: the canary: outputs of the 16,000 a run that are t or more codes off with the scale,")
    w("== to the nearest code: the mean over its draws, and the most")
    w("%-30s" % "" + "".join(" %12s" % ("t=%d" % t) for t in SEL))
    for i in range(1, len(ROWS)):
        rs = of(i)
        w("%-30s" % ROWS[i] + "".join(" %7.1f %4d" % (sum(r["st"][t - 1] for r in rs) / len(rs), max(r["st"][t - 1] for r in rs))
                                      for t in SEL))
    w("== band: the five networks on the first %d draws of the budget's two rows: the rms error with" % OTHER)
    w("== the scale, percent, and the class with it, over each network's draws")
    w("%-26s %-16s %7s %6s %7s %7s %7s %7s" % ("", "", "mean", "sd", "least", "most", "class", "least"))
    for i in (DARK, LIT):
        for sd in range(1, 6):
            rs = of(i, sd, OTHER)
            t = t2 = c = 0.0
            lo = hi = cl = None
            for r in rs:
                x = 100 * srms(r)
                t += x
                t2 += x * x
                lo = x if lo is None or x < lo else lo
                hi = x if hi is None or x > hi else hi
                a = r["sagree"] / 100
                c += a
                cl = a if cl is None or a < cl else cl
            n = len(rs)
            m = t / n
            w("%-26s %-16s" % (ROWS[i], "seed %d%s" % (sd, ", canary" if sd == CANARY else "")) +
              " %7.3f %6.3f %7.3f %7.3f %7.2f %7.2f" % (m, math.sqrt((t2 - n * m * m) / (n - 1)), lo, hi, c / n, cl))
    w("== band: the canary lit at its budget on ten sources, the lit row's first ten draws' line")
    w("== levels, a draw's the one its last digit says: the rms error with the scale, percent, over")
    w("== each source's repeats; and the standard deviation from repeat to repeat, pooled within a")
    w("== source, against that of the ten sources' means")
    w("%-26s %5s %7s %6s %7s %7s" % ("", "runs", "mean", "sd", "least", "most"))
    rep = sources()
    for g in range(10):
        c, m, sd, lo, hi = rep["source"][g]
        w("%-26s %5d %7.3f %6.3f %7.3f %7.3f" % ("source %d" % g, c, m, sd, lo, hi))
    w("%-26s %5d %7.3f %6.3f" % ("repeat to repeat", rep["N"], rep["mean"], rep["within"]))
    w("%-26s %5d %7.3f %6.3f" % ("source to source", rep["K"], rep["mean"], rep["between"]))
    return "\n".join(out) + "\n"


def sources():
    """The ten sources: each source's repeats, the spread within them pooled, and between their means."""
    c, s, s2, lo, hi = {}, {}, {}, {}, {}
    N = 0
    for r in of(TEN):
        g, x = r["draw"] % 10, 100 * srms(r)
        c[g] = c.get(g, 0) + 1
        s[g] = s.get(g, 0.0) + x
        s2[g] = s2.get(g, 0.0) + x * x
        lo[g] = x if c[g] == 1 or x < lo[g] else lo[g]
        hi[g] = x if c[g] == 1 or x > hi[g] else hi[g]
        N += 1
    rep, wsum, K, mm, mm2 = {"source": {}}, 0.0, 0, 0.0, 0.0
    for g in range(10):
        m = s[g] / c[g]
        wsum += s2[g] - c[g] * m * m
        K += 1
        mm += m
        mm2 += m * m
        rep["source"][g] = (c[g], m, math.sqrt((s2[g] - c[g] * m * m) / (c[g] - 1)) if c[g] > 1 else 0.0, lo[g], hi[g])
    bm = mm / K
    rep.update(N=N, K=K, mean=bm, within=math.sqrt(wsum / (N - K)) if N > K else 0.0,
               between=math.sqrt((mm2 - K * bm * bm) / (K - 1)) if K > 1 else 0.0)
    return rep


# ---- the band, by the rule written before the sweep ------------------------------------------------
def band_rule():
    """The rule of 2026-10-15T01:13:20Z, in the runs' own units: srms in 1e-5 of the reference's rms,
    so a tenth of a point is 100; the class in hundredths of a point, so a tenth is 10."""
    lit = of(LIT)
    upper = -(-max(r["srms"] for r in lit) // 100) * 100          # rounded up to a tenth of a point
    lower = (min(r["srms"] for r in of(LOWER_ROW)) // 100) * 100  # rounded down
    cls = (min(r["sagree"] for r in lit) // 10) * 10              # rounded down
    most = {t: max(r["st"][t - 1] for r in lit) for t in range(1, TAIL + 1)}
    t = min(t for t in most if most[t] <= 16)
    rep = sources()
    return {"lower": lower, "upper": upper, "class": cls, "tail_t": t, "tail_n": most[t], "most": most,
            "repeat": rep["within"], "source": rep["between"]}


B = band_rule()


def passes(r, b=B):
    """A run passes Gate F as the band proposes it: inside the band, the class, and the tail."""
    return (not r["refused"] and b["lower"] <= r["srms"] <= b["upper"] and r["sagree"] >= b["class"] and
            r["st"][b["tail_t"] - 1] <= b["tail_n"])


def share(rows, test):
    """The share of a row's draws, refused ones counted, that pass a test, percent."""
    return 100.0 * sum(1 for r in rows if test(r)) / len(rows)


def operating():
    """Each canary row: its draws, refused, and the share passing each part and all of them."""
    out = []
    for i in range(1, len(ROWS)):
        rs = of(i, ran=False)
        out.append((ROWS[i], len(rs), sum(r["refused"] for r in rs),
                    share(rs, lambda r: not r["refused"] and B["lower"] <= r["srms"] <= B["upper"]),
                    share(rs, lambda r: not r["refused"] and r["sagree"] >= B["class"]),
                    share(rs, lambda r: not r["refused"] and r["st"][B["tail_t"] - 1] <= B["tail_n"]),
                    share(rs, passes)))
    return out


def others():
    """The four other networks, lit at the budget on their draws, against the canary's band."""
    return {sd: share(of(LIT, sd, ran=False), passes) for sd in range(2, 6)}


# ---- what was predicted with the rule -----------------------------------------------------------------
def mean(xs):
    return sum(xs) / len(xs)


def predictions():
    pct = lambda rs: [r["srms"] / 1000 for r in rs]
    fmt = [of(FMT, sd)[0] for sd in range(1, 6)]
    lit, dark = pct(of(LIT)), pct(of(DARK))
    sdl = math.sqrt(sum((x - mean(lit)) ** 2 for x in lit) / (len(lit) - 1))
    rising = [mean(pct(of(i))) for i in (3, 4, 5, LIT, 6, 7)]
    over = pct(of(6))
    eight = [r["st"][7] for r in of(LIT)]
    cls = [r["sagree"] / 100 for r in of(LIT)]
    return [
        ("format alone exact on every network; the canary's reference 96.60%",
         all(r["rms"] == 0 and r["sagree"] == 10000 and r["st"][0] == 0 and r["scale"] == 100000 for r in fmt) and fmt[0]["ref"] == 9660),
        ("lit at the budget: mean 4.5 to 4.9, sd 0.05 to 0.25, largest at most 5.5",
         4.5 <= mean(lit) <= 4.9 and 0.05 <= sdl <= 0.25 and max(lit) <= 5.5),
        ("as budgeted, no source: mean 3.1 to 3.5", 3.1 <= mean(dark) <= 3.5),
        ("the lit rows rise with the factor; the 0.6 row's mean 2.4 to 3.3",
         all(a < b for a, b in zip(rising, rising[1:])) and 2.4 <= rising[1] <= 3.3),
        ("the 1.2 row's mean 5.3 to 6.0, under a tenth of it at or under the budget row's largest",
         5.3 <= mean(over) <= 6.0 and sum(x <= max(lit) for x in over) < len(over) / 10),
        ("class lit at the budget: mean 99.1 to 99.6, least at least 98.5", 99.1 <= mean(cls) <= 99.6 and min(cls) >= 98.5),
        ("eight or more codes off, lit at the budget: mean 2 to 20, most at most 60", 2 <= mean(eight) <= 20 and max(eight) <= 60),
        ("seeds 2 to 5 lit at the budget: each within 0.6 of the canary",
         all(abs(mean(pct(of(LIT, sd, OTHER))) - mean(lit)) <= 0.6 for sd in range(2, 6))),
        ("the mode's check says yes on its first sweep", False),       # FIRST_SWEEP
    ]


# ---- what the owner ruled, and what this file holds the runs to ---------------------------------------
# Ruled by Gridworkz on 2026-10-15, as GRXCP recommended (B22): the rule's figures, as they stand.
RULED = {"lower": 2500, "upper": 7300, "class": 9850, "tail_t": 17, "tail_n": 16}


def manifest_holds():
    """The canary's manifest (canary/manifest.json) says what was ruled, on these runs."""
    import json
    m = json.load(open(os.path.join(HERE, "canary", "manifest.json"), encoding="utf-8"))
    g = m["gate_f"]
    assert float(m["manifest_revision"].split("-")[0]) >= 0.2      # the band is in every revision from 0.2 on
    assert g["physical_rms_band_percent"] == ["%.1f" % (RULED["lower"] / 1000), "%.1f" % (RULED["upper"] / 1000)]
    assert g["class_agreement_threshold_percent"] == "%.1f" % (RULED["class"] / 100)
    t = g["physical_tail_threshold_and_allowed_count"]
    assert (t["codes_off_at_least"], t["outputs_allowed"], t["of_outputs"]) == (RULED["tail_t"], RULED["tail_n"], OUTPUTS)
    assert g["band_evidence"]["runs_csv_sha256"] == RUNS_SHA256, "the manifest names other runs"


def findings():
    assert {k: B[k] for k in RULED} == RULED, "the rule does not give what the owner ruled"
    manifest_holds()
    assert (round(B["repeat"], 4), round(B["source"], 4)) == (0.0430, 0.4971)
    assert B["most"][16] == 20 and B["most"][17] <= 16 and B["most"][20] == 2
    shares = {row: round(al, 1) for row, _, _, _, _, _, al in operating()}
    assert shares == {"as budgeted": 100.0, "as budgeted, lit": 100.0, "lit, 0.4 of the budget": 5.2,
                      "lit, 0.6 of the budget": 100.0, "lit, 0.8 of the budget": 100.0, "lit, 1.2 of the budget": 91.6,
                      "lit, 1.4 of the budget": 44.3, "as budgeted, lit, ten sources": 100.0}, shares
    refused = {row: ref for row, _, ref, _, _, _, _ in operating()}
    assert sum(refused.values()) == 20 == refused["lit, 1.4 of the budget"]
    assert {sd: round(p) for sd, p in others().items()} == {2: 99, 3: 100, 4: 100, 5: 100}
    assert [right for _, right in predictions()] == [True, False, True, True, False, True, False, True, False]


def main():
    assert tables() == TABLES, "the tables are not the harness's"
    findings()
    print("The first board's canary: Gate F's band, as the rule gives it and the owner ruled it (B22).")
    print("  %d runs; every table the mode printed, worked out again from them" % len(RUN))
    print("  the band, by the rule written before the sweep, on the rms error over the reference's rms")
    print("  with the scale: %.1f%% to %.1f%%" % (B["lower"] / 1000, B["upper"] / 1000))
    print("  the class with the scale the reference's on at least %.1f%% of the 1,000 vectors" % (B["class"] / 100))
    print("  the tail: at most %d of the 16,000 outputs %d or more codes off" % (B["tail_n"], B["tail_t"]))
    print("  from repeat to repeat, one source: %.3f points; from source to source: %.3f" % (B["repeat"], B["source"]))
    print()
    print("  %-30s %6s %8s %7s %7s %7s %7s" % ("the canary's draws", "draws", "refused", "band", "class", "tail", "all"))
    for row, n, ref, bnd, cl, tl, al in operating():
        print("  %-30s %6d %8d %7.1f %7.1f %7.1f %7.1f" % (row, n, ref, bnd, cl, tl, al))
    print("  the four others lit at the budget, passing all three: " +
          ", ".join("seed %d %.0f%%" % (sd, p) for sd, p in others().items()))
    print()
    print("  predictions written with the rule:")
    for text, right in predictions():
        print("    %s: %s" % ("right" if right else "WRONG", text))
    print()
    print("All checks pass.")


if __name__ == "__main__":
    main()
