/*
 * pta_chiplet_twin.h - the PTA chiplet's digital twin (board plan, step X5).
 *
 * docs/designs/pta_chiplet_regmap.md is the map (X4).  This is a C model that
 * presents that map, with the photonic tile's error model behind it:
 * third_party/grx930/pta_tile_model.c, the reference grx930 holds its RTL to
 * bit for bit.  No arithmetic of the tile's is written here.  The twin is a
 * register file, a command queue, a clock and the calibration contract, in
 * front of pta_gemm() and pta_cal_bank().
 *
 * IT IS A MODEL.  PTA_CAPS2 says so in bit 31, and its tile kind reads 3, "the
 * model on its own".  Nothing that runs through it is a statement about a
 * photonic device or about a chiplet: there is no chiplet.  What it can say is
 * whether a driver uses the map correctly, and what the error model's
 * arithmetic gives for a GEMM -- which is what bring-up before silicon needs
 * (board plan P2) and all it is evidence of.
 *
 * WHAT IT PRESENTS, offsets from the window's base (the map's sections 2-4):
 *
 *   0x000-0x00C  PTA_ID and PTA_CAPS0..2, read-only: what this build is
 *   0x010-0x014  PTA_IRQ_STATUS (write 1 to clear) and PTA_IRQ_MASK
 *   0x018        PTA_GEMM_CT, GEMMs started since the last PTA_SEED write
 *   0x01C        PTA_ACT_CLIP_CT, outputs the activation stage clamped in the
 *                last GEMM
 *   0x040-0x0D0  the block the c930 has at 0x140: CTRL, STATUS, IMPAIR, BITS,
 *                SEED, the sigmas, DRIFT, XTALK, TW, TS, CAL_PER, CAL_THR,
 *                CAL_CT, CAL_CYC, SHOT_CT, WLOAD_CT, SAT_CT, ERR_MAX, GAIN[j],
 *                OFFS[j], DRIFT_MAX
 *   0x0D4-0x0DC  PTA_CAL_CFG, PTA_TRIM, PTA_CAL_SEED
 *   0x0E0-0x0EC  the counters' upper halves, latched when the lower is read
 *   0x0F0        PTA_ERR_FOUND
 *
 * Every other word of the 4 KiB window reads zero and ignores writes.
 *
 * WHAT THE MAP DOES NOT GIVE, and what the twin does about each.  These are
 * findings against the map, recorded in its section 6, and none of them is
 * papered over here:
 *
 *   - No way to issue work.  The map is the control window, in the GPU's BAR.
 *     Work reaches the chiplet over link 2, from the GPU's copy engine, and
 *     nothing yet says what a command on that link looks like.  So the twin's
 *     data path is a function call, pta_twin_submit(), and a command is a whole
 *     integer GEMM in pta_gemm()'s own terms.  That is a stand-in for a
 *     protocol nobody has written, not a proposal for one.
 *   - Its own command queue's depth (the map's open question 3).  A build
 *     parameter here.  A submit the queue cannot take is not accepted, and
 *     says so; nothing reports the depth in a register.
 *   - BUSY and the chiplet's queue.  The map makes one read of PTA_STATUS the
 *     device's half of "has the work finished", and gives the chiplet a queue
 *     of its own.  For that one read to be sound, BUSY has to cover a command
 *     that is queued as well as one that is running, so here it does.
 *   - PTA_GAIN[j] and PTA_OFFS[j] are eight words each, and a chiplet's tile
 *     has 64 columns or more.  The twin has the eight.  Columns past the
 *     eighth have no register and stay at unity and zero.
 *   - PTA_IRQ_STATUS.SAT_THRESHOLD has no threshold in the map.  The twin
 *     never raises it.
 *
 * WHAT THE TWIN DOES NOT MODEL, each visible in a register and not assumed:
 *
 *   - PTA_CTRL[6:4], the calibration scheduler.  Only "off" is built: a
 *     calibration runs when PTA_CTRL.CAL_NOW asks for one.  The field reads
 *     zero whatever is written.  PTA_CAL_PER and PTA_CAL_THR store and read
 *     back, and schedule nothing.  PTA_CAL_PER is in the map's unit, 2^16
 *     cycles, and pta_twin_cal_per() converts to it; nothing here counts in it.
 *   - PTA_CTRL[9:7], the loop-order and residency modes.  The model walks one
 *     order, the shipped one.  The bits read zero.
 *   - PTA_CTRL[11:10], which on the c930 are its DMA's options.  Read zero.
 *   - The activation stage, unless the build asks for one (below).  Without
 *     it PTA_CAPS2[17] is clear and a command that asks for the stage is
 *     refused.
 *   - A shot rate.  PTA_CAPS2[15:0] reads zero: a shot here lasts PTA_TS of the
 *     twin's own cycles and has no rate in hertz.
 *   - A calibration that interrupts a GEMM.  pta_gemm() is one call, so here a
 *     calibration waits for the running command and runs between commands.
 *   - Drift that comes with time.  The model's drift is clocked by shots, so an
 *     idle twin does not drift.  pta_twin_age() is the fast-forward.
 *
 * TIME.  The twin has a clock that only pta_twin_run() advances.  A GEMM
 * occupies the tile for
 *
 *     programmings * PTA_TW  +  shots * PTA_TS      cycles, and at least one,
 *
 * a programming being one (N tile, K tile) of the walk and a shot one output
 * row of one.  A calibration occupies it for
 *
 *     passes * repeats * (PTA_TW + rows * PTA_TS)   cycles, and at least one.
 *
 * Those two formulas are the twin's, and they are all its timing is.  Its
 * arithmetic happens when a command is dispatched, with the configuration the
 * registers hold at that moment (the map's section 5: configuration takes
 * effect at the next GEMM start).  Its result is delivered when the time is up
 * and not before.
 *
 * THE ACTIVATION STAGE (the map's section 8).  What turns one layer's sums into
 * the next layer's operands, so that a network need not leave the chiplet
 * between its layers.  It is the step grx930's accuracy harness takes on the
 * host between layers (c930/sim/pta_mnist.c, tile_batch), and nothing more:
 *
 *     a = min( round( max(sum + bias, 0) / 2^shift ), 2^(bits-1) - 1 )
 *
 * A command asks for it and it applies to that command alone: nothing is left
 * switched on for whoever submits next.  Its output goes back to the caller,
 * or with PTA_TWIN_CMD_HOLD stays here to be the activations of the NEXT
 * command, which says PTA_TWIN_CMD_FROM_HELD.  Held operands are the next
 * command's or nobody's: any command's start takes or discards them, and a
 * calibration between the two does not.
 *
 * Three things to know about it.  It is not grx930's S_ACT, which models an
 * optical nonlinearity and has no bias.  pta_twin_activate() is the twin's own
 * arithmetic, the one piece here that is not grx930's; the gate holds it to the
 * harness's three lines.  And it adds no time: the stage is taken to sit in the
 * shot's own pipeline, one unit a column, which is a requirement on a chiplet
 * and not a finding about one.
 *
 * C99, the standard library, and pta_tile_model.h.
 */
#ifndef PTA_CHIPLET_TWIN_H
#define PTA_CHIPLET_TWIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- The map (docs/designs/pta_chiplet_regmap.md) -------------------------- */
#define PTA_TWIN_WINDOW        0x1000u   /* one page a PTA instance */

#define PTA_TWIN_ID            0x000u    /* R    [31:8] "PTA", [7:0] the map's version */
#define PTA_TWIN_CAPS0         0x004u    /* R    [9:0] rows [19:10] cols [25:20] DIN_W [31:26] ACC_W */
#define PTA_TWIN_CAPS1         0x008u    /* R    [6:0] built [15:8] banks [19:16] [23:20] [27:24] widest bits */
#define PTA_TWIN_CAPS2         0x00Cu    /* R    [15:0] MHz [16] cal [17] act [19:18] kind [31] twin */
#define PTA_TWIN_IRQ_STATUS    0x010u    /* RW1C */
#define PTA_TWIN_IRQ_MASK      0x014u    /* RW */
#define PTA_TWIN_GEMM_CT       0x018u    /* R    GEMMs started since the last PTA_SEED write */
#define PTA_TWIN_ACT_CLIP_CT   0x01Cu    /* R    outputs the activation stage clamped, since the last GEMM start */
#define PTA_TWIN_CTRL          0x040u    /* RW   [0] EN [1] CAL_NOW [3] MODEL_RST */
#define PTA_TWIN_STATUS        0x044u    /* R    [0] CAL_BUSY [1] CAL_VALID [2] SAT [3] DRIFT_ALARM [4] BUSY [23:8] residual */
#define PTA_TWIN_IMPAIR        0x048u    /* RW   [6:0] */
#define PTA_TWIN_BITS          0x04Cu    /* RW   [3:0] B_a [7:4] B_w [11:8] B_adc [17:12] S */
#define PTA_TWIN_SEED          0x050u    /* RW   a write restarts PTA_GEMM_CT */
#define PTA_TWIN_SIGMA_TH      0x054u    /* RW   [15:0] Q8.8, in ADC LSB */
#define PTA_TWIN_SIGMA_SH      0x058u    /* RW   [15:0] Q8.8 */
#define PTA_TWIN_SIGMA_PR      0x05Cu    /* RW   [15:0] Q8.8, in weight LSB */
#define PTA_TWIN_DRIFT         0x060u    /* RW   [15:0] step sigma, [20:16] log2 shots a step */
#define PTA_TWIN_XTALK         0x064u    /* RW   [7:0] Q0.8 */
#define PTA_TWIN_TW            0x068u    /* RW   cycles a programming */
#define PTA_TWIN_TS            0x06Cu    /* RW   cycles a shot */
#define PTA_TWIN_CAL_PER       0x070u    /* RW   in units of 2^16 cycles; stored, and no scheduler reads it */
#define PTA_TWIN_CAL_THR       0x074u    /* RW   [23:0], stored likewise */
#define PTA_TWIN_CAL_CT        0x078u    /* R    calibrations that ran to the end */
#define PTA_TWIN_CAL_CYC       0x07Cu    /* R    cycles spent calibrating, low half */
#define PTA_TWIN_SHOT_CT       0x080u    /* R    shots issued, probes included, low half */
#define PTA_TWIN_WLOAD_CT      0x084u    /* R    a GEMM's weight programmings, low half */
#define PTA_TWIN_SAT_CT        0x088u    /* R    ADC saturations since the last GEMM start, low half */
#define PTA_TWIN_ERR_MAX       0x08Cu    /* R    [23:0] what the last calibration left */
#define PTA_TWIN_GAIN0         0x090u    /* RW   [17:0] signed Q8.8, eight words */
#define PTA_TWIN_OFFS0         0x0B0u    /* RW   signed, eight words */
#define PTA_TWIN_DRIFT_MAX     0x0D0u    /* RW   [15:0] Q8.8, in weight LSB */
#define PTA_TWIN_CAL_CFG       0x0D4u    /* RW   [3:0] amplitude [7:4] repeats [9:8] passes [10] bank */
#define PTA_TWIN_TRIM          0x0D8u    /* RW   [3:0] log2 of the DAC's step, [31:16] its clamp */
#define PTA_TWIN_CAL_SEED      0x0DCu    /* RW */
#define PTA_TWIN_SHOT_CT_HI    0x0E0u    /* R    latched when PTA_SHOT_CT is read */
#define PTA_TWIN_WLOAD_CT_HI   0x0E4u    /* R    latched when PTA_WLOAD_CT is read */
#define PTA_TWIN_SAT_CT_HI     0x0E8u    /* R    latched when PTA_SAT_CT is read */
#define PTA_TWIN_CAL_CYC_HI    0x0ECu    /* R    latched when PTA_CAL_CYC is read */
#define PTA_TWIN_ERR_FOUND     0x0F0u    /* R    [23:0] what the last calibration found */

#define PTA_TWIN_AFFINE_WORDS  8         /* the map's PTA_GAIN[j] and PTA_OFFS[j] */

#define PTA_TWIN_ID_VALUE      0x50544101u   /* "PTA", map version 1 */
#define PTA_TWIN_BUILT         0x5Fu         /* every impairment but MZM_NL, as the c930's tile */
#define PTA_TWIN_BANKS         2u            /* the model's weight banks */
#define PTA_TWIN_KIND          3u            /* PTA_CAPS2[19:18]: the model on its own */
#define PTA_TWIN_CAPS2_CAL     0x00010000u   /* the calibration engine is present */
#define PTA_TWIN_CAL_PER_LOG2  16u           /* PTA_CAL_PER counts 2^16 cycles: the map's section 4 */
#define PTA_TWIN_CAPS2_ACT     0x00020000u   /* the activation stage is built */
#define PTA_TWIN_CAPS2_HOLD_SHIFT 20         /* [24:20] log2 of the operands the stage can hold */
#define PTA_TWIN_CAPS2_HOLD_MASK  0x01F00000u
#define PTA_TWIN_CAPS2_TWIN    0x80000000u   /* this is a model and not silicon */

#define PTA_TWIN_CTRL_EN         0x01u
#define PTA_TWIN_CTRL_CAL_NOW    0x02u   /* a pulse; honoured only with EN set */
#define PTA_TWIN_CTRL_MODEL_RST  0x08u   /* a pulse; refused while BUSY or CAL_BUSY */

#define PTA_TWIN_STATUS_CAL_BUSY    0x01u
#define PTA_TWIN_STATUS_CAL_VALID   0x02u
#define PTA_TWIN_STATUS_SAT         0x04u   /* PTA_SAT_CT is not zero */
#define PTA_TWIN_STATUS_DRIFT_ALARM 0x08u
#define PTA_TWIN_STATUS_BUSY        0x10u   /* a command is running, or queued */

#define PTA_TWIN_IRQ_CAL_DONE      0x1u
#define PTA_TWIN_IRQ_ERR           0x2u
#define PTA_TWIN_IRQ_DRIFT_ALARM   0x4u
#define PTA_TWIN_IRQ_SAT_THRESHOLD 0x8u    /* never raised: the map gives it no threshold */

/* ---- The build ------------------------------------------------------------- */
/*
 * What a chiplet is, as opposed to what state it is in.  The geometry is the
 * map's first open question, so nothing here picks one: a caller names it and
 * PTA_CAPS0 reports it.  queue_depth is the map's third.
 */
typedef struct {
    int rows;          /* inputs a shot sums, k: 1 .. 1023 */
    int cols;          /* outputs a shot yields, n: 1 .. 1023 */
    int din_w;         /* operand width: 2 .. 32 */
    int acc_w;         /* accumulator width: 2 .. 63 */
    int queue_depth;   /* commands that can wait behind the running one: 1 .. 64 */
    int act_hold;      /* operands the activation stage can hold for the next command:
                          0 builds no stage, else a power of two up to 2^28 */
} pta_twin_build;

typedef struct pta_twin pta_twin;

/* A twin as it leaves reset, or NULL if the build is out of range or memory
 * ran out. */
pta_twin *pta_twin_new(const pta_twin_build *build);
void      pta_twin_free(pta_twin *t);

/* The chiplet's reset: every register to its reset value, the tile to no
 * drift and no correction, the clock to zero, and whatever was running or
 * queued discarded -- its status says PTA_TWIN_LOST.  The build is unchanged. */
void      pta_twin_reset(pta_twin *t);

/* ---- The control path: the window, as CXL.io reaches it --------------------- */
uint32_t  pta_twin_read32(pta_twin *t, uint32_t offset);
void      pta_twin_write32(pta_twin *t, uint32_t offset, uint32_t value);

/* The interrupt line: PTA_IRQ_STATUS & PTA_IRQ_MASK is not zero. */
int       pta_twin_irq(const pta_twin *t);

/* ---- The data path: what link 2 will carry ---------------------------------- */
#define PTA_TWIN_PENDING   0   /* accepted; queued or running */
#define PTA_TWIN_DONE      1   /* C holds the result */
#define PTA_TWIN_REFUSED   2   /* the tile cannot do this; C is untouched and IRQ ERR is raised */
#define PTA_TWIN_LOST      3   /* a reset discarded it; C is untouched */

/* What a command may ask of the activation stage.  Zero asks nothing. */
#define PTA_TWIN_CMD_ACT        0x1u   /* the results go through the stage: C receives operands, not sums */
#define PTA_TWIN_CMD_HOLD       0x2u   /* and stay on the chiplet for the next command; C is not written */
#define PTA_TWIN_CMD_FROM_HELD  0x4u   /* this command's activations are the ones held; A is not read */

/*
 * One command: C = A * B on weight bank `bank`, A being M x K and B K x N, row
 * major, in the model's own integers.  A and B are copied when the command is
 * accepted, and so is bias.  C, status, sats and clips are written when it ends,
 * so they have to outlive it; status, sats and clips may be NULL.
 *
 * Zero every field this file may add later: a command is memset to zero and
 * then filled, and a zero in a field means the command does not use it.
 *
 * With PTA_TWIN_CMD_ACT each of the M x N sums goes through
 * pta_twin_activate(sum, bias[n], act_shift, act_bits) on its way out.
 * With PTA_TWIN_CMD_HOLD the operands that come out are kept for the next
 * command and C may be NULL.  With PTA_TWIN_CMD_FROM_HELD the activations are
 * the M x K operands the command before it held, and A may be NULL.  One
 * command may take the held operands and hold its own: a middle layer.
 */
typedef struct {
    int            bank;
    int            M, N, K;
    const int32_t *A;
    const int32_t *B;
    int64_t       *C;
    int           *status;   /* PTA_TWIN_PENDING from acceptance, then one of the other three */
    long          *sats;     /* the GEMM's ADC saturations, with PTA_TWIN_DONE */
    unsigned       flags;    /* PTA_TWIN_CMD_*, or zero */
    int            act_shift;   /* 0 .. 62 */
    int            act_bits;    /* the operand's width, clamp included: 2 .. DIN_W */
    const int64_t *bias;     /* N of them, in the sums' own units, or NULL for none */
    long          *clips;    /* outputs the stage clamped, with PTA_TWIN_DONE */
} pta_twin_cmd;

/*
 * Hand the twin a command.  Returns 0 if it was accepted, and -1 if it was
 * not: the queue is full, a pointer or a dimension is unusable, or memory ran
 * out.  A command that is not accepted has not happened -- its status is not
 * written and nothing is counted -- and the caller still holds it.
 *
 * A command the TILE cannot honour is a different thing: it is accepted, and
 * when its turn comes it ends PTA_TWIN_REFUSED with PTA_IRQ_STATUS.ERR raised.
 * That is the map's "refused, not dropped".  The tile refuses what the c930's
 * does: an impairment it does not build, or an ADC shift past 40 with any
 * impairment enabled; and a bank it does not have.
 *
 * The activation stage refuses, the same way: any flag on a build without the
 * stage, or a flag it does not know; PTA_TWIN_CMD_HOLD without
 * PTA_TWIN_CMD_ACT; a shift or a width out of range; more operands to hold than
 * it has room for; and PTA_TWIN_CMD_FROM_HELD when nothing is held, or when
 * what is held is not M x K.  A command that is refused has still taken its
 * turn, so whatever was held before it is gone.
 */
int       pta_twin_submit(pta_twin *t, const pta_twin_cmd *cmd);

/* Commands accepted and not yet ended: the one running and those behind it. */
int       pta_twin_pending(const pta_twin *t);

/* ---- Time --------------------------------------------------------------------- */
/* Advance the clock, completing and starting whatever falls due. */
void      pta_twin_run(pta_twin *t, uint64_t cycles);
uint64_t  pta_twin_now(const pta_twin *t);

/*
 * Age the tile by `steps` drift steps: the state steps * 2^log2 shots would
 * leave, under PTA_DRIFT and PTA_DRIFT_MAX as they stand, without issuing the
 * shots.  It is the model's own pta_drift_age(), for sweeps over hours of
 * drift, and like the clock it is the twin's and not the map's: no register
 * does this, and silicon will not need one, because its drift comes with time.
 * No shot is counted and the clock does not move.  With DRIFT clear in
 * PTA_IMPAIR it does nothing, as a shot would step nothing.  Returns 0, or -1
 * if anything holds the tile, in which case nothing was aged.
 */
int       pta_twin_age(pta_twin *t, uint64_t steps);

/* ---- What a driver needs and the map only describes ------------------------- */
/*
 * The seed of GEMM `index` under PTA_SEED `seed`: the map's section 4, which
 * points at grx930's pta_mnist.c for the function.  It is that function, and
 * with the twin implementing it, it is the one to hold silicon to.
 */
uint32_t  pta_twin_gemm_seed(uint32_t seed, uint32_t index);

/* The seed of calibration `index`, PTA_CAL_CT before it runs. */
uint32_t  pta_twin_cal_seed(uint32_t cal_seed, uint32_t index);

/*
 * PTA_CAL_PER for an interval of that many cycles: the map's section 4.  The
 * register counts units of 2^16 cycles, because 32 bits of single cycles is
 * 4.3 seconds of a 1 GHz shot clock and a calibration interval is minutes.
 * Rounds to the nearest unit, a half up.  Zero cycles is zero, which is the
 * period switched off, and no other interval comes back as zero.  One past
 * the register's range comes back as its largest value.
 */
uint32_t  pta_twin_cal_per(uint64_t cycles);

/*
 * The activation stage's function, for one sum: the map's section 8.
 *
 *     v = max(sum + bias, 0)
 *     v = round(v / 2^shift)              a half rounds up
 *     a = min(v, 2^(bits-1) - 1)
 *
 * which is grx930's pta_mnist.c between its layers.  shift is 0 .. 62 and bits
 * 2 .. 32; outside that it returns 0.  *clipped is set to whether the last line
 * changed the value, and may be NULL.  sum + bias saturates where it would not
 * fit 64 bits, which the harness's own addition would leave undefined.
 */
int64_t   pta_twin_activate(int64_t sum, int64_t bias, int shift, int bits, int *clipped);

#ifdef __cplusplus
}
#endif

#endif /* PTA_CHIPLET_TWIN_H */
