/*
 * pta_mnist_act_via_twin.c - the twin's activation stage, held to grx930's
 * accuracy harness and not to a copy of it.
 *
 * The stage (docs/designs/pta_chiplet_regmap.md section 8) is defined as the
 * step grx930's c930/sim/pta_mnist.c takes on the host between a network's
 * layers.  The twin's gate holds pta_twin_activate() to those three lines as
 * the gate's author copied them out.  This file holds it to the lines
 * themselves: it compiles the harness's source into itself, runs the harness's
 * own tile_batch() on a network, and runs the same network on a twin with
 * every intermediate held on the chiplet.
 *
 *   cc -std=c99 -O2 -I$GRX930/c930/sim -I$GRXCP/third_party/grx930 \
 *      -I$GRXCP/src/backends/pta_chiplet pta_mnist_act_via_twin.c \
 *      $GRXCP/src/backends/pta_chiplet/pta_chiplet_twin.c \
 *      $GRXCP/third_party/grx930/pta_tile_model.c -lm -o pta_mnist_act_twin
 *   ./pta_mnist_act_twin
 *
 * The harness is not edited.  It is included with its main() renamed, which is
 * what lets this file call tile_batch() and read what it computed.  grx930's
 * source is not vendored here, so this is run by hand against a checkout and
 * is not in CI; README.md says when it was last run and on what.
 *
 * WHAT IS COMPARED.  The harness cuts a layer into GEMMs its core accepts, at
 * most 64 x 256 x 8, adds them up on the host, and takes the step between
 * layers on the host.  The twin is given each layer as ONE command, walks it in
 * its own tiles, and takes the step in its activation stage with the result
 * held for the next command.  Two things have to come out the same: the
 * operands every hidden layer hands on, and the last layer's sums.
 *
 * WHAT THAT CAN AND CANNOT COVER.  The two cut the work differently, and each
 * GEMM draws its noise from its own seed, so with any noise enabled they are
 * two different runs and nothing here compares them.  The cases are the exact
 * product, on a tile of the chiplet's size, and the quantisers alone, on the
 * harness's own 8 x 8 tile, where a shot sees the same operands either way.
 * With noise, what is held to what is the twin's gate: a network held on the
 * chiplet against the same network brought out at every layer, on one cut.
 *
 * The network is random.  It is the harness's arithmetic that is under test,
 * not MNIST, and a random network needs no data.
 */
#define main pta_mnist_main
#include "pta_mnist.c"
#undef main

#include "pta_chiplet_twin.h"

static uint32_t lcg_state;

static uint32_t lcg(void)
{
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 8;
}

/* A value in [-amax, amax]. */
static int32_t draw(int amax)
{
    return (int32_t)(lcg() % (uint32_t)(2 * amax + 1)) - amax;
}

static void drain(pta_twin *t)
{
    int i;
    for (i = 0; i < 1000000 && (pta_twin_read32(t, PTA_TWIN_STATUS) &
                                (PTA_TWIN_STATUS_BUSY | PTA_TWIN_STATUS_CAL_BUSY)); ++i)
        pta_twin_run(t, 1u << 20);
}

/* The registers, from the harness's configuration for one layer. */
static void program(pta_twin *t, const pta_cfg *c)
{
    pta_twin_write32(t, PTA_TWIN_IMPAIR, c->impair);
    pta_twin_write32(t, PTA_TWIN_BITS, c->act_bits | (c->w_bits << 4) | (c->adc_bits << 8) |
                                           (c->adc_shift << 12));
}

/*
 * One network, both ways.  `quant` turns the three quantisers on; the twin's
 * tile is rows x cols.  Returns the number of things that differed.
 */
static int run_case(const char *what, int hidden, int M, int quant, int rows, int cols, uint32_t seed)
{
    const int D = 8, amax = 127;
    host_net   hn;
    pta_cfg    cfg[MAX_L];
    pta_tile   tile;
    pta_device dev;
    pta_twin_build build;
    pta_twin  *held, *out;
    gemm_buf  *g  = (gemm_buf *)malloc(sizeof *g);
    batch_ws  *bw = (batch_ws *)malloc(sizeof *bw);
    int32_t   *a0 = (int32_t *)malloc((size_t)M * N_IN * sizeof *a0);
    int32_t   *B  = (int32_t *)malloc((size_t)N_IN * N_HID * sizeof *B);
    int64_t   *C  = (int64_t *)malloc((size_t)M * N_HID * sizeof *C);
    int64_t   *A2 = (int64_t *)malloc((size_t)M * N_HID * sizeof *A2);
    uint32_t   gemm = 0;
    long       clipped = 0, zeros = 0, operands = 0;
    int        l, i, bad = 0, bad_ops = 0, bad_sums = 0, status, refused = 0;

    if (!g || !bw || !a0 || !B || !C || !A2 || host_alloc(&hn, D, hidden) != 0) {
        printf("FAIL  %s: out of memory\n", what);
        return 1;
    }
    lcg_state = seed;
    for (l = 0; l <= hidden; ++l) {
        const int in = layer_in(l), outn = layer_out(hidden, l);
        for (i = 0; i < in * outn; ++i)
            hn.w[l][i] = draw(amax);
        /* A hidden layer's sum is about 2^18 wide on this network, so 2^11 a
         * step leaves operands across the range with some at each end. */
        for (i = 0; i < outn; ++i)
            hn.b[l][i] = (int64_t)draw(60) * 1024 + draw(511);
        if (l < hidden)
            hn.sh[l] = l == 0 ? 12 : 11;
        memset(&cfg[l], 0, sizeof cfg[l]);
        if (quant) {
            cfg[l].impair    = PTA_QUANT;
            cfg[l].act_bits  = 6;
            cfg[l].w_bits    = 6;
            cfg[l].adc_bits  = 7;
            cfg[l].adc_shift = 12;       /* 8 inputs of 127 x 127 in seven bits */
        }
    }
    for (i = 0; i < M * N_IN; ++i)
        a0[i] = (int32_t)(lcg() % 128u);  /* pixels: operands that are not negative */

    /* ---- the harness, as it runs a batch ---- */
    tile.rows  = ROWS;
    tile.cols  = COLS;
    tile.din_w = D;
    tile.acc_w = ACC_W;
    if (pta_device_init(&dev, &tile) != 0)
        return 1;
    pta_model_reset(&dev, 0);
    if (tile_batch(&hn, cfg, &tile, &dev, seed, &gemm, g, M, a0, bw) < 0) {
        printf("FAIL  %s: the harness's own run did not complete\n", what);
        return 1;
    }

    /* ---- the twin, twice: every intermediate held, and every one brought out ---- */
    build.rows = rows;
    build.cols = cols;
    build.din_w = D;
    build.acc_w = ACC_W;
    build.queue_depth = 4;
    build.act_hold = 1 << 14;
    held = pta_twin_new(&build);
    out  = pta_twin_new(&build);
    if (!held || !out)
        return 1;
    for (l = 0; l <= hidden; ++l) {
        const int in = layer_in(l), outn = layer_out(hidden, l);
        pta_twin_cmd c;
        int k, n;

        for (k = 0; k < in; ++k)
            for (n = 0; n < outn; ++n)
                B[k * outn + n] = hn.w[l][n * in + k];
        memset(&c, 0, sizeof c);
        c.bank = l & 1;
        c.M = M;
        c.N = outn;
        c.K = in;
        c.B = B;
        c.status = &status;
        if (l < hidden) {
            c.act_shift = hn.sh[l];
            c.act_bits  = D;
            c.bias      = hn.b[l];
        }

        /* held: this layer's operands stay on the chiplet for the next */
        program(held, &cfg[l]);
        c.A = l == 0 ? a0 : NULL;
        c.C = C;
        c.flags = (l > 0 ? PTA_TWIN_CMD_FROM_HELD : 0u) |
                  (l < hidden ? (PTA_TWIN_CMD_ACT | PTA_TWIN_CMD_HOLD) : 0u);
        status = -1;
        if (pta_twin_submit(held, &c) != 0)
            ++refused;
        drain(held);
        refused += status != PTA_TWIN_DONE;
        if (l == hidden)
            for (i = 0; i < M * N_OUT; ++i)
                bad_sums += C[i] != bw->y[l][i];

        /* brought out: the same layer, its operands returned and compared */
        if (l < hidden) {
            program(out, &cfg[l]);
            c.A = l == 0 ? a0 : bw->a[l];
            c.C = A2;
            c.flags = PTA_TWIN_CMD_ACT;
            status = -1;
            if (pta_twin_submit(out, &c) != 0)
                ++refused;
            drain(out);
            refused += status != PTA_TWIN_DONE;
            for (i = 0; i < M * N_HID; ++i) {
                bad_ops += A2[i] != (int64_t)bw->a[l + 1][i];
                zeros   += bw->a[l + 1][i] == 0;
                clipped += bw->a[l + 1][i] == amax;
                ++operands;
            }
        }
    }
    bad = bad_ops + bad_sums + refused;
    printf("%s  %s\n", bad ? "FAIL" : "ok  ", what);
    printf("        %d hidden layer%s, a batch of %d, the twin's tile %d x %d; the harness's is %d x %d\n",
           hidden, hidden == 1 ? "" : "s", M, rows, cols, ROWS, COLS);
    printf("        the harness ran %u GEMMs and the twin %d commands a pass\n", (unsigned)gemm, hidden + 1);
    printf("        %ld operands handed on by the hidden layers: %d differ; %ld are zero and %ld at the clamp\n",
           operands, bad_ops, zeros, clipped);
    printf("        %d sums out of the last layer: %d differ.  Commands not done: %d\n", M * N_OUT, bad_sums,
           refused);
    if (zeros == 0 || clipped == 0 || zeros + clipped == operands) {
        printf("FAIL  %s: the stage was given nothing to do at one end of its range\n", what);
        ++bad;
    }
    pta_twin_free(held);
    pta_twin_free(out);
    pta_device_free(&dev);
    host_free(&hn);
    free(g);
    free(bw);
    free(a0);
    free(B);
    free(C);
    free(A2);
    return bad;
}

int main(void)
{
    int bad = 0;

    printf("The twin's activation stage against grx930's pta_mnist.c, compiled into this program.\n"
           "A random network: the harness's arithmetic is what is under test.  A model, not a chiplet.\n\n");
    bad += run_case("D3's shape, the exact product, on a tile of the chiplet's size", 1, 64, 0, 256, 64, 0xD3);
    bad += run_case("three hidden layers, the exact product, the same tile", 3, 64, 0, 256, 64, 0x3D3);
    bad += run_case("D3's shape, the three quantisers on, on the harness's own tile", 1, 64, 1, ROWS, COLS, 0x9D3);
    bad += run_case("three hidden layers, the quantisers on, the harness's tile", 3, 48, 1, ROWS, COLS, 0x7D3);
    printf("\n%s\n", bad ? "FAILED" : "PASSED: held on the twin, each network is the harness's, element for element");
    return bad != 0;
}
