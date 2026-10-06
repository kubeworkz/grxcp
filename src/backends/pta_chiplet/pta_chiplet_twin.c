/*
 * pta_chiplet_twin.c - the PTA chiplet's digital twin (board plan, step X5).
 *
 * See pta_chiplet_twin.h for what this is and is not, and
 * docs/designs/pta_chiplet_regmap.md for the map it presents.
 *
 * The tile's arithmetic is not here.  A GEMM is pta_gemm() and a calibration is
 * pta_cal_bank(), both grx930's, vendored under third_party/grx930 and held to
 * grx930's own vectors by ci/build_mock.sh.  This file is what stands between
 * those two calls and a driver: registers, a queue, a clock, and the rules the
 * map's section 5 gives for what happens when they meet.
 *
 * Three switches build a twin that is deliberately wrong, each in the one way
 * a gate exists to catch.  ci/build_mock.sh builds all three and requires the
 * gate to fail against each:
 *
 *   PTA_TWIN_ABLATE_SEED       every GEMM runs on PTA_SEED itself, not on the
 *                              seed derived from it and the GEMM's index
 *   PTA_TWIN_ABLATE_CAL_GUARD  a command that arrives during a calibration is
 *                              taken at once, into a tile that is not free
 *   PTA_TWIN_ABLATE_RST_GUARD  a MODEL_RST is honoured whatever is running
 *   PTA_TWIN_ABLATE_ACT_ROUND  the activation stage's shift truncates
 *   PTA_TWIN_ABLATE_HELD_GUARD held operands outlive the command after them,
 *                              so a later one can run on a network's leftovers
 */
#include "pta_chiplet_twin.h"

#include <stdlib.h>
#include <string.h>

#include "pta_tile_model.h"

/* The largest a command may be.  A bound on the twin's own arithmetic and on
 * the model's, which counts elements in an int; no tile is this big. */
#define DIM_MAX  65536
#define ELEM_MAX ((int64_t)1 << 28)

enum { T_IDLE, T_GEMM, T_CAL };

typedef struct {
    int      bank, M, N, K;
    int32_t *A, *B;        /* the twin's copies; A is NULL when the activations are the held ones */
    int64_t *C;            /* the caller's */
    int     *status;
    long    *sats;
    unsigned flags;
    int      act_shift, act_bits;
    int64_t *bias;         /* the twin's copy, or NULL */
    long    *clips;
} cmd_t;

struct pta_twin {
    pta_twin_build build;
    pta_tile       tile;
    pta_device     dev;

    /* ---- the registers ---- */
    uint32_t en;
    uint32_t impair, bits, seed, sig_th, sig_sh, sig_pr, drift, xtalk;
    uint32_t tw, ts, cal_per, cal_thr, dmax, cal_cfg, trim, cal_seed;
    uint32_t irq_status, irq_mask;
    uint32_t gemm_ct, cal_ct;
    uint64_t cal_cyc, shot_ct, wload_ct, sat_ct;
    uint32_t cal_cyc_hi, shot_hi, wload_hi, sat_hi;      /* latched by a read of the low half */
    uint32_t err_max, err_found;
    int      cal_valid, drift_alarm;
    int32_t  gain[PTA_TWIN_AFFINE_WORDS];
    int32_t  offs[PTA_TWIN_AFFINE_WORDS];

    /* ---- what holds the tile ---- */
    int      state;
    uint64_t left;           /* cycles until it lets go */
    uint64_t now;
    int      cal_pending;    /* a CAL_NOW not yet served */

    cmd_t    cur;            /* the running command, */
    int64_t *cur_c;          /* its result, held until it ends, */
    long     cur_sats;       /* its saturations */
    long     cur_clips;      /* and what the activation stage clamped */

    /* ---- the activation stage's held operands ---- */
    int32_t *held;           /* build.act_hold of them, or NULL with no stage */
    int      held_m, held_n; /* what is held is held_m x held_n */
    int      held_valid;
    uint32_t act_clip_ct;

    struct {                 /* the running calibration's outcome, published at its end */
        int      refused, clamped;
        uint32_t err_max, err_found;
        uint64_t cycles, shots;
        long     sats;
    } cal;

    cmd_t   *q;              /* the chiplet's own queue, a ring */
    int      q_head, q_len;
};

/* ---- the two seed functions -------------------------------------------------- */
static uint64_t splitmix64(uint64_t *s)
{
    uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* grx930's pta_mnist.c, gemm_seed(), to the letter. */
uint32_t pta_twin_gemm_seed(uint32_t seed, uint32_t index)
{
    uint64_t s = ((uint64_t)seed << 32) | index;
    return (uint32_t)(splitmix64(&s) >> 32);
}

/* The map's section 4, and c930_pta_cal.sv's o_seed. */
uint32_t pta_twin_cal_seed(uint32_t cal_seed, uint32_t index)
{
    return cal_seed ^ (index * 0x9E3779B1u);
}

/* The map's section 4: an interval in cycles, as PTA_CAL_PER holds it. */
uint32_t pta_twin_cal_per(uint64_t cycles)
{
    const uint64_t half = (uint64_t)1 << (PTA_TWIN_CAL_PER_LOG2 - 1);
    uint64_t units;
    if (cycles == 0)
        return 0;
    if (cycles > UINT64_MAX - half)
        return UINT32_MAX;
    units = (cycles + half) >> PTA_TWIN_CAL_PER_LOG2;
    if (units == 0)
        return 1;
    return units > UINT32_MAX ? UINT32_MAX : (uint32_t)units;
}

/* ---- small things --------------------------------------------------------------- */
static void raise_irq(pta_twin *t, uint32_t bits)
{
    t->irq_status |= bits;
}

static int affine_words(const pta_twin *t)
{
    return t->build.cols < PTA_TWIN_AFFINE_WORDS ? t->build.cols : PTA_TWIN_AFFINE_WORDS;
}

/* What a start samples, from the registers as they stand. */
static void cfg_from_regs(const pta_twin *t, pta_cfg *cfg)
{
    memset(cfg, 0, sizeof *cfg);
    cfg->impair      = t->impair;
    cfg->act_bits    = t->bits & 0xFu;
    cfg->w_bits      = (t->bits >> 4) & 0xFu;
    cfg->adc_bits    = (t->bits >> 8) & 0xFu;
    cfg->adc_shift   = (t->bits >> 12) & 0x3Fu;
    cfg->seed        = t->seed;
    cfg->sigma_th    = t->sig_th;
    cfg->k_shot      = t->sig_sh;
    cfg->sigma_pr    = t->sig_pr;
    cfg->drift_sigma = t->drift & 0xFFFFu;
    cfg->drift_log2  = (t->drift >> 16) & 0x1Fu;
    cfg->drift_max   = t->dmax;
    cfg->xtalk       = t->xtalk;
    cfg->trim_step   = 1u << (t->trim & 0xFu);
    cfg->trim_max    = t->trim >> 16;
}

static void cmd_release(cmd_t *c)
{
    free(c->A);
    free(c->B);
    free(c->bias);
    memset(c, 0, sizeof *c);
}

/* A command's end, however it ends. */
static void cmd_end(cmd_t *c, int status)
{
    if (c->status)
        *c->status = status;
    cmd_release(c);
}

static uint64_t at_least_one(uint64_t cycles)
{
    return cycles ? cycles : 1;
}

/* ---- the activation stage ------------------------------------------------------------ */
int64_t pta_twin_activate(int64_t sum, int64_t bias, int shift, int bits, int *clipped)
{
    int64_t v, amax;

    if (clipped)
        *clipped = 0;
    if (shift < 0 || shift > 62 || bits < 2 || bits > 32)
        return 0;
    amax = ((int64_t)1 << (bits - 1)) - 1;
    if (bias > 0 && sum > INT64_MAX - bias)
        v = INT64_MAX;
    else if (bias < 0 && sum < INT64_MIN - bias)
        v = INT64_MIN;
    else
        v = sum + bias;
    if (v < 0)
        v = 0;
    if (shift > 0) {
#ifdef PTA_TWIN_ABLATE_ACT_ROUND
        v >>= shift;
#else
        /* (v + 2^(shift-1)) >> shift, written so that it cannot overflow */
        v = (v >> shift) + ((v >> (shift - 1)) & 1);
#endif
    }
    if (v > amax) {
        v = amax;
        if (clipped)
            *clipped = 1;
    }
    return v;
}

/* What the stage refuses, judged when the command's turn comes. */
static int act_refused(const pta_twin *t, const cmd_t *c)
{
    const unsigned f = c->flags;

    if (!f)
        return 0;
    if ((f & ~(PTA_TWIN_CMD_ACT | PTA_TWIN_CMD_HOLD | PTA_TWIN_CMD_FROM_HELD)) != 0)
        return 1;
    if (t->build.act_hold == 0)
        return 1;
    if ((f & PTA_TWIN_CMD_HOLD) && !(f & PTA_TWIN_CMD_ACT))
        return 1;
    if ((f & PTA_TWIN_CMD_ACT) && (c->act_shift < 0 || c->act_shift > 62 ||
                                   c->act_bits < 2 || c->act_bits > t->build.din_w))
        return 1;
    if ((f & PTA_TWIN_CMD_HOLD) && (int64_t)c->M * c->N > (int64_t)t->build.act_hold)
        return 1;
    if ((f & PTA_TWIN_CMD_FROM_HELD) &&
        (!t->held_valid || t->held_m != c->M || t->held_n != c->K))
        return 1;
    return 0;
}

/* ---- a GEMM ------------------------------------------------------------------------ */
/*
 * The model's arithmetic for one command, into a buffer of the twin's own, and
 * the activation stage's after it if the command asks.  Returns the buffer, or
 * NULL if the tile or the stage refuses the command, in which case nothing has
 * been counted and the model has not been called.
 */
static int64_t *gemm_arith(pta_twin *t, const cmd_t *c, long *sats, long *clips)
{
    pta_cfg  cfg;
    int64_t *out;
    long     r;
    const int32_t *a = (c->flags & PTA_TWIN_CMD_FROM_HELD) ? t->held : c->A;

    *clips = 0;
    if (act_refused(t, c))
        return NULL;

    cfg_from_regs(t, &cfg);
    /* The c930 core's own refusal (c930_npu_core.sv, pta_bad): a bit the tile
     * does not build, or an ADC shift the model's arithmetic cannot hold. */
    if (cfg.impair != 0 && ((cfg.impair & ~PTA_TWIN_BUILT) != 0 || cfg.adc_shift > 40))
        return NULL;
    if (c->bank < 0 || c->bank >= (int)PTA_TWIN_BANKS)
        return NULL;

#ifdef PTA_TWIN_ABLATE_SEED
    cfg.seed = t->seed;
#else
    cfg.seed = pta_twin_gemm_seed(t->seed, t->gemm_ct);
#endif
    out = (int64_t *)malloc((size_t)c->M * (size_t)c->N * sizeof *out);
    if (!out)
        return NULL;
    r = pta_gemm(&cfg, &t->tile, &t->dev, c->bank, c->M, c->N, c->K, a, c->B, out);
    if (r < 0) {
        free(out);
        return NULL;
    }
    *sats = r;
    if (c->flags & PTA_TWIN_CMD_ACT) {
        int m, n, clipped;
        for (m = 0; m < c->M; ++m)
            for (n = 0; n < c->N; ++n) {
                int64_t *v = &out[(size_t)m * (size_t)c->N + (size_t)n];
                *v = pta_twin_activate(*v, c->bias ? c->bias[n] : 0, c->act_shift, c->act_bits,
                                       &clipped);
                *clips += clipped;
            }
    }
    return out;
}

/* Take the tile for a command, or refuse it.  The tile is free on entry. */
static void gemm_begin(pta_twin *t, cmd_t *c)
{
    const uint64_t tiles = (uint64_t)((c->N + t->build.cols - 1) / t->build.cols) *
                           (uint64_t)((c->K + t->build.rows - 1) / t->build.rows);
    const uint64_t shots = tiles * (uint64_t)c->M;
    long     sats = 0, clips = 0;
    int64_t *out  = gemm_arith(t, c, &sats, &clips);

    /* Held operands are the next command's or nobody's.  This command has had
     * its turn: it took them, or it did not want them, or it was refused. */
#ifndef PTA_TWIN_ABLATE_HELD_GUARD
    t->held_valid = 0;
#endif
    if (!out) {
        raise_irq(t, PTA_TWIN_IRQ_ERR);
        cmd_end(c, PTA_TWIN_REFUSED);
        return;
    }
    t->gemm_ct  += 1;
    t->sat_ct    = (uint64_t)sats;       /* the count restarts at a GEMM start */
    t->act_clip_ct = (uint32_t)clips;    /* and so does this one */
    t->cur_clips = clips;
    t->shot_ct  += shots;
    t->wload_ct += tiles;
    t->cur       = *c;
    t->cur_c     = out;
    t->cur_sats  = sats;
    t->state     = T_GEMM;
    t->left      = at_least_one(tiles * t->tw + shots * t->ts);
}

static void gemm_end(pta_twin *t)
{
    const size_t n = (size_t)t->cur.M * (size_t)t->cur.N;

    if (t->cur.flags & PTA_TWIN_CMD_HOLD) {
        size_t i;
        for (i = 0; i < n; ++i)
            t->held[i] = (int32_t)t->cur_c[i];      /* 0 .. 2^(bits-1) - 1: it fits */
        t->held_m = t->cur.M;
        t->held_n = t->cur.N;
        t->held_valid = 1;
    } else {
        memcpy(t->cur.C, t->cur_c, n * sizeof *t->cur_c);
    }
    if (t->cur.sats)
        *t->cur.sats = t->cur_sats;
    if (t->cur.clips)
        *t->cur.clips = t->cur_clips;
    free(t->cur_c);
    t->cur_c = NULL;
    cmd_end(&t->cur, PTA_TWIN_DONE);
    t->state = T_IDLE;
}

/* ---- a calibration ------------------------------------------------------------------- */
static void cal_begin(pta_twin *t)
{
    pta_cfg     cfg;
    pta_cal_cfg cal;
    pta_streams st;
    long        sats = 0;
    int         clamped = 0;
    int64_t     found = 0, left;
    const int   bank = (int)((t->cal_cfg >> 10) & 1u);

    cfg_from_regs(t, &cfg);
    cal.amp_log2  = t->cal_cfg & 0xFu;
    cal.reps_log2 = (t->cal_cfg >> 4) & 0xFu;
    cal.passes    = (t->cal_cfg >> 8) & 0x3u;
    if (cal.passes == 0)
        cal.passes = 1;                  /* as the engine: zero means one */

    /* A calibration is not a GEMM start: it has streams of its own, loaded
     * from PTA_CAL_SEED and the count of calibrations before it. */
    pta_start(&st, pta_twin_cal_seed(t->cal_seed, t->cal_ct));
    left = pta_cal_bank(&t->dev, &cfg, &t->tile, bank, &st, &cal, &sats, &clamped, &found);

    memset(&t->cal, 0, sizeof t->cal);
    t->cal_valid = 0;                    /* nothing is in force until this one ends */
    if (left < 0) {
        /* The engine's refusal: a probe amplitude it could not read back. */
        t->cal.refused = 1;
        t->cal.cycles  = 1;
    } else {
        /* With the ADC unquantised one pass is exact and the model takes one. */
        const int      quantised = (cfg.impair & PTA_QUANT) && cfg.adc_bits != 0;
        const uint64_t passes    = quantised ? cal.passes : 1;
        const uint64_t reps      = passes << cal.reps_log2;
        t->cal.err_max   = (uint32_t)left;
        t->cal.err_found = (uint32_t)found;
        t->cal.clamped   = clamped;
        t->cal.sats      = sats;
        t->cal.shots     = reps * (uint64_t)t->build.rows;
        t->cal.cycles    = at_least_one(reps * ((uint64_t)t->tw +
                                                (uint64_t)t->build.rows * t->ts));
    }
    t->state = T_CAL;
    t->left  = t->cal.cycles;
}

static void cal_end(pta_twin *t)
{
    t->cal_cyc += t->cal.cycles;
    if (t->cal.refused) {
        raise_irq(t, PTA_TWIN_IRQ_ERR);
        t->drift_alarm = 0;
    } else {
        t->err_max     = t->cal.err_max;
        t->err_found   = t->cal.err_found;
        t->drift_alarm = t->cal.clamped;
        t->cal_valid   = 1;
        t->cal_ct     += 1;
        t->shot_ct    += t->cal.shots;                  /* the probe's shots are shots */
        t->sat_ct     += (uint64_t)t->cal.sats;
        if (t->cal.clamped)
            raise_irq(t, PTA_TWIN_IRQ_DRIFT_ALARM);
    }
    raise_irq(t, PTA_TWIN_IRQ_CAL_DONE);
    t->state = T_IDLE;
}

/* ---- who gets the tile next ------------------------------------------------------------ */
/* A calibration first, then the queue's head.  A refused command never holds
 * the tile, so the loop goes on to the one behind it. */
static void start_next(pta_twin *t)
{
    while (t->state == T_IDLE) {
        if (t->cal_pending) {
            t->cal_pending = 0;
            cal_begin(t);
        } else if (t->q_len > 0) {
            cmd_t c = t->q[t->q_head];
            memset(&t->q[t->q_head], 0, sizeof t->q[t->q_head]);
            t->q_head = (t->q_head + 1) % t->build.queue_depth;
            t->q_len -= 1;
            gemm_begin(t, &c);
        } else {
            break;
        }
    }
}

/* ---- the build and reset ---------------------------------------------------------------- */
static void regs_reset(pta_twin *t)
{
    int j;
    t->en = 0;
    t->impair = t->bits = t->seed = 0;
    t->sig_th = t->sig_sh = t->sig_pr = t->drift = t->xtalk = 0;
    t->tw = t->ts = t->cal_per = t->cal_thr = t->dmax = 0;
    t->cal_cfg = t->trim = t->cal_seed = 0;
    t->irq_status = t->irq_mask = 0;
    t->gemm_ct = t->cal_ct = 0;
    t->act_clip_ct = 0;
    t->cal_cyc = t->shot_ct = t->wload_ct = t->sat_ct = 0;
    t->cal_cyc_hi = t->shot_hi = t->wload_hi = t->sat_hi = 0;
    t->err_max = t->err_found = 0;
    t->cal_valid = t->drift_alarm = 0;
    for (j = 0; j < PTA_TWIN_AFFINE_WORDS; ++j) {
        t->gain[j] = 256;                /* unity */
        t->offs[j] = 0;
    }
}

pta_twin *pta_twin_new(const pta_twin_build *b)
{
    pta_twin *t;

    if (!b || b->rows < 1 || b->rows > 1023 || b->cols < 1 || b->cols > 1023 ||
        b->din_w < 2 || b->din_w > 32 || b->acc_w < 2 || b->acc_w > 63 ||
        b->queue_depth < 1 || b->queue_depth > 64)
        return NULL;
    /* No stage, or room for a power of two of operands: PTA_CAPS2 reports a log2. */
    if (b->act_hold < 0 || (int64_t)b->act_hold > ELEM_MAX ||
        (b->act_hold & (b->act_hold - 1)) != 0)
        return NULL;
    t = (pta_twin *)calloc(1, sizeof *t);
    if (!t)
        return NULL;
    t->build      = *b;
    t->tile.rows  = b->rows;
    t->tile.cols  = b->cols;
    t->tile.din_w = b->din_w;
    t->tile.acc_w = b->acc_w;
    t->q = (cmd_t *)calloc((size_t)b->queue_depth, sizeof *t->q);
    if (b->act_hold > 0)
        t->held = (int32_t *)calloc((size_t)b->act_hold, sizeof *t->held);
    if (!t->q || (b->act_hold > 0 && !t->held) || pta_device_init(&t->dev, &t->tile) != 0) {
        free(t->held);
        free(t->q);
        free(t);
        return NULL;
    }
    regs_reset(t);
    pta_model_reset(&t->dev, 0);
    return t;
}

/* Everything that was accepted and has not ended, discarded. */
static void drop_all(pta_twin *t)
{
    if (t->state == T_GEMM) {
        free(t->cur_c);
        t->cur_c = NULL;
        cmd_end(&t->cur, PTA_TWIN_LOST);
    }
    while (t->q_len > 0) {
        cmd_end(&t->q[t->q_head], PTA_TWIN_LOST);
        t->q_head = (t->q_head + 1) % t->build.queue_depth;
        t->q_len -= 1;
    }
    t->q_head = 0;
    t->state = T_IDLE;
    t->left = 0;
    t->cal_pending = 0;
    t->held_valid = 0;
}

void pta_twin_reset(pta_twin *t)
{
    if (!t)
        return;
    drop_all(t);
    regs_reset(t);
    pta_model_reset(&t->dev, 0);
    pta_cal_reset(&t->dev);
    t->now = 0;
}

void pta_twin_free(pta_twin *t)
{
    if (!t)
        return;
    drop_all(t);
    pta_device_free(&t->dev);
    free(t->held);
    free(t->q);
    free(t);
}

/* ---- the data path --------------------------------------------------------------------- */
static int32_t *copy_i32(const int32_t *src, size_t n)
{
    int32_t *dst = (int32_t *)malloc(n * sizeof *dst);
    if (dst)
        memcpy(dst, src, n * sizeof *dst);
    return dst;
}

int pta_twin_submit(pta_twin *t, const pta_twin_cmd *cmd)
{
    cmd_t c;

    /* A is not read when the activations are the held ones, and C is not
     * written when the results are to be held. */
    if (!t || !cmd || !cmd->B ||
        (!cmd->A && !(cmd->flags & PTA_TWIN_CMD_FROM_HELD)) ||
        (!cmd->C && !(cmd->flags & PTA_TWIN_CMD_HOLD)))
        return -1;
    if (cmd->M < 1 || cmd->N < 1 || cmd->K < 1 ||
        cmd->M > DIM_MAX || cmd->N > DIM_MAX || cmd->K > DIM_MAX ||
        (int64_t)cmd->M * cmd->N > ELEM_MAX || (int64_t)cmd->M * cmd->K > ELEM_MAX ||
        (int64_t)cmd->K * cmd->N > ELEM_MAX)
        return -1;
    /* Nowhere to put it: the tile is taken, or about to be, and the queue is
     * full.  The caller keeps the command. */
    if ((t->state != T_IDLE || t->cal_pending) && t->q_len >= t->build.queue_depth) {
#ifdef PTA_TWIN_ABLATE_CAL_GUARD
        if (t->state != T_CAL)
#endif
        return -1;
    }

    memset(&c, 0, sizeof c);
    c.bank = cmd->bank;
    c.M = cmd->M;
    c.N = cmd->N;
    c.K = cmd->K;
    c.C = cmd->C;
    c.status = cmd->status;
    c.sats = cmd->sats;
    c.flags = cmd->flags;
    c.act_shift = cmd->act_shift;
    c.act_bits = cmd->act_bits;
    c.clips = cmd->clips;
    if (!(cmd->flags & PTA_TWIN_CMD_FROM_HELD))
        c.A = copy_i32(cmd->A, (size_t)cmd->M * (size_t)cmd->K);
    c.B = copy_i32(cmd->B, (size_t)cmd->K * (size_t)cmd->N);
    if ((cmd->flags & PTA_TWIN_CMD_ACT) && cmd->bias) {
        c.bias = (int64_t *)malloc((size_t)cmd->N * sizeof *c.bias);
        if (c.bias)
            memcpy(c.bias, cmd->bias, (size_t)cmd->N * sizeof *c.bias);
    }
    if ((!c.A && !(cmd->flags & PTA_TWIN_CMD_FROM_HELD)) || !c.B ||
        ((cmd->flags & PTA_TWIN_CMD_ACT) && cmd->bias && !c.bias)) {
        free(c.A);
        free(c.B);
        free(c.bias);
        return -1;
    }
    if (c.status)
        *c.status = PTA_TWIN_PENDING;

#ifdef PTA_TWIN_ABLATE_CAL_GUARD
    /* The guard removed: a command that arrives during a calibration goes
     * straight into the tile, and comes straight out of it. */
    if (t->state == T_CAL) {
        long     sats = 0, clips = 0;
        int64_t *out  = gemm_arith(t, &c, &sats, &clips);
        if (out) {
            if (c.C)
                memcpy(c.C, out, (size_t)c.M * (size_t)c.N * sizeof *out);
            free(out);
            t->gemm_ct += 1;
            if (c.sats)
                *c.sats = sats;
        }
        cmd_end(&c, out ? PTA_TWIN_DONE : PTA_TWIN_REFUSED);
        return 0;
    }
#endif

    if (t->state == T_IDLE && !t->cal_pending && t->q_len == 0) {
        gemm_begin(t, &c);
        return 0;
    }
    t->q[(t->q_head + t->q_len) % t->build.queue_depth] = c;
    t->q_len += 1;
    return 0;
}

int pta_twin_pending(const pta_twin *t)
{
    return t ? t->q_len + (t->state == T_GEMM ? 1 : 0) : 0;
}

/* ---- time ------------------------------------------------------------------------------ */
void pta_twin_run(pta_twin *t, uint64_t cycles)
{
    if (!t)
        return;
    while (cycles > 0) {
        uint64_t step;
        if (t->state == T_IDLE) {
            t->now += cycles;
            return;
        }
        step = t->left < cycles ? t->left : cycles;
        t->now  += step;
        t->left -= step;
        cycles  -= step;
        if (t->left == 0) {
            if (t->state == T_GEMM)
                gemm_end(t);
            else
                cal_end(t);
            start_next(t);
        }
    }
}

uint64_t pta_twin_now(const pta_twin *t)
{
    return t ? t->now : 0;
}

int pta_twin_age(pta_twin *t, uint64_t steps)
{
    pta_cfg cfg;

    if (!t || t->state != T_IDLE || t->q_len > 0)
        return -1;
    cfg_from_regs(t, &cfg);
    if (cfg.impair & PTA_DRIFT)
        pta_drift_age(&t->dev, &cfg, steps);
    return 0;
}

int pta_twin_irq(const pta_twin *t)
{
    return t && (t->irq_status & t->irq_mask) != 0;
}

/* ---- the window ------------------------------------------------------------------------ */
static uint32_t caps0(const pta_twin *t)
{
    return (uint32_t)t->build.rows | ((uint32_t)t->build.cols << 10) |
           ((uint32_t)t->build.din_w << 20) | ((uint32_t)t->build.acc_w << 26);
}

/* The widest activation and weight settings that quantise: DIN_W - 1, capped by
 * the four bits PTA_BITS has.  As the c930's core computes PTA_QMAX. */
static uint32_t caps1(const pta_twin *t)
{
    const uint32_t qmax = t->build.din_w - 1 > 15 ? 15u : (uint32_t)(t->build.din_w - 1);
    return PTA_TWIN_BUILT | (PTA_TWIN_BANKS << 8) | (qmax << 16) | (qmax << 20) | (15u << 24);
}

/* The activation stage, if this build has one, and the log2 of what it holds. */
static uint32_t caps2_act(const pta_twin *t)
{
    uint32_t log2 = 0;

    if (t->build.act_hold == 0)
        return 0;
    while (((uint32_t)1 << log2) < (uint32_t)t->build.act_hold)
        ++log2;
    return PTA_TWIN_CAPS2_ACT | (log2 << PTA_TWIN_CAPS2_HOLD_SHIFT);
}

static uint32_t status(const pta_twin *t)
{
    const int busy = t->state == T_GEMM || t->q_len > 0;
    return (t->state == T_CAL ? PTA_TWIN_STATUS_CAL_BUSY : 0u) |
           (t->cal_valid ? PTA_TWIN_STATUS_CAL_VALID : 0u) |
           (t->sat_ct ? PTA_TWIN_STATUS_SAT : 0u) |
           (t->drift_alarm ? PTA_TWIN_STATUS_DRIFT_ALARM : 0u) |
           (busy ? PTA_TWIN_STATUS_BUSY : 0u) |
           ((t->err_max & 0xFFFFu) << 8);
}

uint32_t pta_twin_read32(pta_twin *t, uint32_t off)
{
    if (!t || off >= PTA_TWIN_WINDOW || (off & 3u))
        return 0;

    if (off >= PTA_TWIN_GAIN0 && off < PTA_TWIN_GAIN0 + 4u * PTA_TWIN_AFFINE_WORDS) {
        const int j = (int)((off - PTA_TWIN_GAIN0) >> 2);
        return j < affine_words(t) ? ((uint32_t)t->gain[j] & 0x3FFFFu) : 0u;
    }
    if (off >= PTA_TWIN_OFFS0 && off < PTA_TWIN_OFFS0 + 4u * PTA_TWIN_AFFINE_WORDS) {
        const int j = (int)((off - PTA_TWIN_OFFS0) >> 2);
        return j < affine_words(t) ? (uint32_t)t->offs[j] : 0u;
    }

    switch (off) {
    case PTA_TWIN_ID:         return PTA_TWIN_ID_VALUE;
    case PTA_TWIN_CAPS0:      return caps0(t);
    case PTA_TWIN_CAPS1:      return caps1(t);
    case PTA_TWIN_CAPS2:      return PTA_TWIN_CAPS2_TWIN | (PTA_TWIN_KIND << 18) | PTA_TWIN_CAPS2_CAL |
                                     caps2_act(t);
    case PTA_TWIN_ACT_CLIP_CT: return t->act_clip_ct;
    case PTA_TWIN_IRQ_STATUS: return t->irq_status;
    case PTA_TWIN_IRQ_MASK:   return t->irq_mask;
    case PTA_TWIN_GEMM_CT:    return t->gemm_ct;
    case PTA_TWIN_CTRL:       return t->en;     /* the pulses and the unbuilt modes read zero */
    case PTA_TWIN_STATUS:     return status(t);
    case PTA_TWIN_IMPAIR:     return t->impair;
    case PTA_TWIN_BITS:       return t->bits;
    case PTA_TWIN_SEED:       return t->seed;
    case PTA_TWIN_SIGMA_TH:   return t->sig_th;
    case PTA_TWIN_SIGMA_SH:   return t->sig_sh;
    case PTA_TWIN_SIGMA_PR:   return t->sig_pr;
    case PTA_TWIN_DRIFT:      return t->drift;
    case PTA_TWIN_XTALK:      return t->xtalk;
    case PTA_TWIN_TW:         return t->tw;
    case PTA_TWIN_TS:         return t->ts;
    case PTA_TWIN_CAL_PER:    return t->cal_per;
    case PTA_TWIN_CAL_THR:    return t->cal_thr;
    case PTA_TWIN_CAL_CT:     return t->cal_ct;
    /* A counter's low half, and its high half held for the read that follows. */
    case PTA_TWIN_CAL_CYC:    t->cal_cyc_hi = (uint32_t)(t->cal_cyc >> 32);  return (uint32_t)t->cal_cyc;
    case PTA_TWIN_SHOT_CT:    t->shot_hi    = (uint32_t)(t->shot_ct >> 32);  return (uint32_t)t->shot_ct;
    case PTA_TWIN_WLOAD_CT:   t->wload_hi   = (uint32_t)(t->wload_ct >> 32); return (uint32_t)t->wload_ct;
    case PTA_TWIN_SAT_CT:     t->sat_hi     = (uint32_t)(t->sat_ct >> 32);   return (uint32_t)t->sat_ct;
    case PTA_TWIN_CAL_CYC_HI:  return t->cal_cyc_hi;
    case PTA_TWIN_SHOT_CT_HI:  return t->shot_hi;
    case PTA_TWIN_WLOAD_CT_HI: return t->wload_hi;
    case PTA_TWIN_SAT_CT_HI:   return t->sat_hi;
    case PTA_TWIN_ERR_MAX:    return t->err_max & 0xFFFFFFu;
    case PTA_TWIN_ERR_FOUND:  return t->err_found & 0xFFFFFFu;
    case PTA_TWIN_DRIFT_MAX:  return t->dmax;
    case PTA_TWIN_CAL_CFG:    return t->cal_cfg;
    case PTA_TWIN_TRIM:       return t->trim;
    case PTA_TWIN_CAL_SEED:   return t->cal_seed;
    default:                  return 0;
    }
}

/* A column's pair, to the tile: a write to either word sends both. */
static void send_affine(pta_twin *t, int j)
{
    pta_column_cal(&t->dev, j, t->gain[j], t->offs[j]);
}

static void write_ctrl(pta_twin *t, uint32_t v)
{
    t->en = v & PTA_TWIN_CTRL_EN;

    if (v & PTA_TWIN_CTRL_MODEL_RST) {
        const int idle = t->state == T_IDLE && t->q_len == 0;
#ifdef PTA_TWIN_ABLATE_RST_GUARD
        const int honoured = 1;
#else
        const int honoured = idle;
#endif
        if (honoured) {
            int j;
            pta_model_reset(&t->dev, t->seed);
            pta_cal_reset(&t->dev);
            for (j = 0; j < PTA_TWIN_AFFINE_WORDS; ++j) {
                t->gain[j] = 256;
                t->offs[j] = 0;
            }
            t->cal_valid   = 0;
            t->drift_alarm = 0;
        }
        if (!idle)
            raise_irq(t, PTA_TWIN_IRQ_ERR);      /* refused, and the driver is told */
    }

    /* CAL_NOW is the engine's, and the engine listens only when enabled. */
    if ((v & PTA_TWIN_CTRL_CAL_NOW) && t->en) {
        t->cal_pending = 1;
        start_next(t);
    }
}

void pta_twin_write32(pta_twin *t, uint32_t off, uint32_t v)
{
    if (!t || off >= PTA_TWIN_WINDOW || (off & 3u))
        return;

    if (off >= PTA_TWIN_GAIN0 && off < PTA_TWIN_GAIN0 + 4u * PTA_TWIN_AFFINE_WORDS) {
        const int j = (int)((off - PTA_TWIN_GAIN0) >> 2);
        if (j < affine_words(t)) {
            const uint32_t g = v & 0x3FFFFu;                 /* eighteen bits, signed */
            t->gain[j] = (g & 0x20000u) ? (int32_t)g - 0x40000 : (int32_t)g;
            send_affine(t, j);
        }
        return;
    }
    if (off >= PTA_TWIN_OFFS0 && off < PTA_TWIN_OFFS0 + 4u * PTA_TWIN_AFFINE_WORDS) {
        const int j = (int)((off - PTA_TWIN_OFFS0) >> 2);
        if (j < affine_words(t)) {
            t->offs[j] = (int32_t)v;
            send_affine(t, j);
        }
        return;
    }

    switch (off) {
    case PTA_TWIN_IRQ_STATUS: t->irq_status &= ~(v & 0xFu);  break;   /* write 1 to clear */
    case PTA_TWIN_IRQ_MASK:   t->irq_mask = v & 0xFu;        break;
    case PTA_TWIN_CTRL:       write_ctrl(t, v);              break;
    case PTA_TWIN_IMPAIR:     t->impair   = v & 0x7Fu;       break;
    case PTA_TWIN_BITS:       t->bits     = v & 0x3FFFFu;    break;
    /* One seed for a run, and the count of GEMMs under it starts again. */
    case PTA_TWIN_SEED:       t->seed = v; t->gemm_ct = 0;   break;
    case PTA_TWIN_SIGMA_TH:   t->sig_th   = v & 0xFFFFu;     break;
    case PTA_TWIN_SIGMA_SH:   t->sig_sh   = v & 0xFFFFu;     break;
    case PTA_TWIN_SIGMA_PR:   t->sig_pr   = v & 0xFFFFu;     break;
    case PTA_TWIN_DRIFT:      t->drift    = v & 0x1FFFFFu;   break;
    case PTA_TWIN_XTALK:      t->xtalk    = v & 0xFFu;       break;
    case PTA_TWIN_TW:         t->tw       = v;               break;
    case PTA_TWIN_TS:         t->ts       = v;               break;
    case PTA_TWIN_CAL_PER:    t->cal_per  = v;               break;
    case PTA_TWIN_CAL_THR:    t->cal_thr  = v & 0xFFFFFFu;   break;
    case PTA_TWIN_DRIFT_MAX:  t->dmax     = v & 0xFFFFu;     break;
    case PTA_TWIN_CAL_CFG:    t->cal_cfg  = v & 0x7FFu;      break;
    case PTA_TWIN_TRIM:       t->trim     = v & 0xFFFF000Fu; break;
    case PTA_TWIN_CAL_SEED:   t->cal_seed = v;               break;
    default:                  break;                         /* read-only, or nothing there */
    }
}
