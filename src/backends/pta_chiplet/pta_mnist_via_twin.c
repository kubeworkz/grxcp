/*
 * pta_mnist_via_twin.c - grx930's accuracy harness, with the twin where the
 * model was.
 *
 * The board plan's P2 gate has two halves: Linux on a CXL host enumerates the
 * device, and "the D3 network runs through the emulated PTA bit-identical to
 * pta_mnist's C reference".  The first needs a board.  The second needs only
 * the twin, and this file is how it is run.
 *
 * grx930's c930/sim/pta_mnist.c touches its device through three of the
 * model's functions on the way to an accuracy: pta_model_reset() once, on the
 * run's seed; pta_drift_age(), if the run asks for hours of drift; and
 * pta_gemm() for every GEMM of the network.  Compiled with those three names
 * redirected, the same source calls the functions below, which have the same
 * signatures and reach the model only through the twin:
 *
 *   cc -std=c99 -O2 -Dpta_gemm=pta_gemm_via_twin \
 *      -Dpta_model_reset=pta_model_reset_via_twin \
 *      -Dpta_drift_age=pta_drift_age_via_twin -c $GRX930/c930/sim/pta_mnist.c
 *   cc -std=c99 -O2 pta_mnist.o pta_mnist_via_twin.c pta_chiplet_twin.c \
 *      $GRXCP/third_party/grx930/pta_tile_model.c -lm -o pta_mnist_twin
 *
 * The harness is not edited, and whatever it prints is then a statement about
 * the twin.  The same `eval` line, given to that program and to the harness
 * built as grx930 builds it, has to print the same line.  README.md has what
 * was run.
 *
 * What goes through the map and what does not.  The reset is PTA_SEED and
 * PTA_CTRL.MODEL_RST.  Every GEMM's configuration is written to the registers
 * before it, the GEMM is a command, and its end is read from PTA_STATUS.  The
 * ageing is pta_twin_age(), which is the twin's and not the map's: the model
 * itself says the RTL has no such port.
 *
 * WHAT IT CANNOT RUN, and refuses.  The harness calibrates through
 * pta_trim_write(), a host's trim write, and the map has no register for one.
 * A run with --calibrate leaves the harness's own device holding trims the
 * twin's cannot be given, and this file stops rather than compare two
 * different devices.
 *
 * Nothing about the run is taken on trust.  Every GEMM's seed, as the harness
 * derived it, is checked against the one the twin is about to use, and a
 * mismatch stops the run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pta_chiplet_twin.h"
#include "pta_tile_model.h"

void pta_model_reset_via_twin(pta_device *dev, uint32_t seed);
void pta_drift_age_via_twin(pta_device *dev, const pta_cfg *cfg, uint64_t steps);
long pta_gemm_via_twin(const pta_cfg *cfg, const pta_tile *tile, pta_device *dev, int bank,
                       int M, int N, int K, const int32_t *A, const int32_t *B, int64_t *C);

#define AGES_MAX 8

static pta_twin *twin;
static pta_tile  built;
static uint32_t  base_seed;
static int       have_seed, starting;
static struct { pta_cfg cfg; uint64_t steps; } ages[AGES_MAX];
static int       n_ages;
static unsigned long gemms;
static unsigned long long aged;

static void die(const char *why)
{
    fprintf(stderr, "pta_mnist_via_twin: %s\n", why);
    exit(3);
}

static void report(void)
{
    /* On stderr, so that stdout stays the harness's own line and nothing else. */
    if (twin)
        fprintf(stderr,
                "pta_mnist_via_twin: %lu GEMMs through the twin's map, %llu drift steps aged; "
                "PTA_GEMM_CT %u, PTA_SHOT_CT %u, PTA_WLOAD_CT %u\n",
                gemms, aged, pta_twin_read32(twin, PTA_TWIN_GEMM_CT),
                pta_twin_read32(twin, PTA_TWIN_SHOT_CT), pta_twin_read32(twin, PTA_TWIN_WLOAD_CT));
}

/* The registers a start samples, from a configuration of the harness's. */
static void program(const pta_cfg *cfg)
{
    uint32_t trim_log2 = 0;

    /* The DAC's step and clamp matter to a trim write and to nothing else, and
     * nothing here writes one.  They are carried across all the same, where
     * the map can say them: a step that is a power of two, a 16-bit clamp. */
    if ((cfg->trim_step & (cfg->trim_step - 1)) != 0 || cfg->trim_step > 0x8000u ||
        cfg->trim_max > 0xFFFFu)
        die("the harness's trim step or clamp is one PTA_TRIM cannot hold");
    while ((1u << trim_log2) < cfg->trim_step)
        ++trim_log2;
    pta_twin_write32(twin, PTA_TWIN_TRIM, trim_log2 | (cfg->trim_max << 16));
    pta_twin_write32(twin, PTA_TWIN_IMPAIR, cfg->impair);
    pta_twin_write32(twin, PTA_TWIN_BITS, cfg->act_bits | (cfg->w_bits << 4) |
                                          (cfg->adc_bits << 8) | (cfg->adc_shift << 12));
    pta_twin_write32(twin, PTA_TWIN_SIGMA_TH, cfg->sigma_th);
    pta_twin_write32(twin, PTA_TWIN_SIGMA_SH, cfg->k_shot);
    pta_twin_write32(twin, PTA_TWIN_SIGMA_PR, cfg->sigma_pr);
    pta_twin_write32(twin, PTA_TWIN_DRIFT, cfg->drift_sigma | (cfg->drift_log2 << 16));
    pta_twin_write32(twin, PTA_TWIN_DRIFT_MAX, cfg->drift_max);
    pta_twin_write32(twin, PTA_TWIN_XTALK, cfg->xtalk);
}

/* The harness's own device is never aged or reset here, so at its first GEMM it
 * must still be as pta_device_init() left it.  If it is not, the harness has
 * written it directly -- a calibration's trims -- and the twin's is another
 * device. */
static void require_untouched(const pta_device *dev)
{
    int b, i;
    for (b = 0; b < 2; ++b)
        for (i = 0; i < dev->rows * dev->cols; ++i)
            if (dev->drift[b][i] != 0 || dev->trim[b][i] != 0)
                die("the harness calibrated its device, and no register writes a trim");
    for (i = 0; i < dev->cols; ++i)
        if (dev->gain[i] != 256 || dev->offs[i] != 0)
            die("the harness set a column's affine before the first GEMM");
}

void pta_model_reset_via_twin(pta_device *dev, uint32_t seed)
{
    (void)dev;
    base_seed = seed;
    have_seed = 1;
    starting  = 1;
    n_ages    = 0;
}

void pta_drift_age_via_twin(pta_device *dev, const pta_cfg *cfg, uint64_t steps)
{
    (void)dev;
    if (!starting)
        die("the harness aged its device in the middle of a run");
    if (n_ages == AGES_MAX)
        die("more ageing steps before a run than this file holds");
    ages[n_ages].cfg   = *cfg;
    ages[n_ages].steps = steps;
    ++n_ages;
}

/* A run's start, now that its first GEMM has said what the tile is. */
static void begin(const pta_tile *tile, const pta_device *dev)
{
    pta_twin_build b;
    int i;

    if (!have_seed)
        die("a GEMM before any model reset: the run has no seed");
    require_untouched(dev);
    if (!twin || memcmp(&built, tile, sizeof built) != 0) {
        const int first = twin == NULL;
        pta_twin_free(twin);
        b.rows = tile->rows;
        b.cols = tile->cols;
        b.din_w = tile->din_w;
        b.acc_w = tile->acc_w;
        b.queue_depth = 4;
        twin = pta_twin_new(&b);
        if (!twin)
            die("the twin cannot be built at the harness's tile");
        built = *tile;
        if (first)
            atexit(report);
    } else {
        pta_twin_reset(twin);
    }
    /* pta_model_reset(&dev, seed), through the map. */
    pta_twin_write32(twin, PTA_TWIN_SEED, base_seed);
    pta_twin_write32(twin, PTA_TWIN_CTRL, PTA_TWIN_CTRL_MODEL_RST);
    /* pta_drift_age(&dev, &cfg, steps), which no register does. */
    for (i = 0; i < n_ages; ++i) {
        program(&ages[i].cfg);
        if (pta_twin_age(twin, ages[i].steps) != 0)
            die("the twin would not be aged");
        aged += ages[i].steps;
    }
    n_ages   = 0;
    starting = 0;
}

long pta_gemm_via_twin(const pta_cfg *cfg, const pta_tile *tile, pta_device *dev, int bank,
                       int M, int N, int K, const int32_t *A, const int32_t *B, int64_t *C)
{
    pta_twin_cmd cmd;
    int   status = -1;
    long  sats = -1;

    memset(&cmd, 0, sizeof cmd);       /* nothing asked of the activation stage */
    if (!twin || starting)
        begin(tile, dev);
    if (cfg->seed != pta_twin_gemm_seed(base_seed, pta_twin_read32(twin, PTA_TWIN_GEMM_CT)))
        die("this GEMM's seed is not the one the twin is about to use: a calibration's "
            "probe, or a GEMM outside the run");

    /* The configuration, as a driver would write it: before the GEMM it is for. */
    program(cfg);

    memset(&cmd, 0, sizeof cmd);
    cmd.bank = bank;
    cmd.M = M;
    cmd.N = N;
    cmd.K = K;
    cmd.A = A;
    cmd.B = B;
    cmd.C = C;
    cmd.status = &status;
    cmd.sats = &sats;
    if (pta_twin_submit(twin, &cmd) != 0)
        return -1;
    /* The map's completion test: one read of PTA_STATUS, neither BUSY nor CAL_BUSY. */
    while (pta_twin_read32(twin, PTA_TWIN_STATUS) &
           (PTA_TWIN_STATUS_BUSY | PTA_TWIN_STATUS_CAL_BUSY))
        pta_twin_run(twin, 1u << 20);
    if (status != PTA_TWIN_DONE)
        return -1;
    ++gemms;
    return sats;
}
