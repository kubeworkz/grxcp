/*
 * pta_chiplet_module.h - the first board's PTA module: the transport's queue,
 * in front of the chiplet's twin.
 *
 * docs/designs/pta_board_transport.md is the design (the board plan's B24 and
 * B25).  On the first board the PTA is a module of its own, a CXL Type-1
 * device on the GRX930 host, and no GPU stands between them.  So the module
 * carries its own queue: a ring of GEMM descriptors in host memory, a doorbell
 * on the second page of its BAR, and the results and a record written back
 * into host memory over CXL.cache.  This file models that module.  Its tile is
 * the twin, pta_chiplet_twin.h, unchanged and on the BAR's first page, and
 * nothing of the tile's arithmetic is here.
 *
 * IT IS A MODEL.  The twin it fronts says so in PTA_CAPS2[31], and that word
 * reads the same through this file.  There is no module.  What it can say is
 * whether a driver uses the queue as the design has it, and what a GEMM gives
 * when it does.
 *
 * WHAT IT PRESENTS, offsets in the BAR:
 *
 *   0x0000-0x0FFF  page 0: the twin's map, every word as the twin has it
 *   0x1000-0x102C  page 1: the queue, PTQ_*, as the design's section 4 lays out
 *
 * Every other word reads zero and ignores writes.  The MSI-X table of the
 * design's pages 2 and 3 is not modelled: an interrupt is a call, below.
 *
 * HOST MEMORY is two calls the caller supplies, one for each thing CXL.cache
 * does here: read a whole 64-byte line as it now is (RdCurr), and write a
 * whole line (ItoMWr).  A write call that has returned is ordered: what the
 * design calls the GO.  The module keeps no line of the host's.  Addresses
 * are the host's physical ones, and the calls are where a test decides what
 * one means.
 *
 * WHAT IT DOES, which is the design's sections 4 to 6 and nothing more:
 *
 *   - One command at a time, in index order, from PTQ_HEAD to PTQ_TAIL.  It
 *     reads a slot, decides, and either ends it at once -- not issued, not run
 *     -- or hands the GEMM to the twin, which may yet refuse it.
 *   - Only version 1 descriptors, 64 bytes.  PTQ_CAPS says so: a version 2
 *     descriptor, the activation stage's, ends NOT ISSUED here.
 *   - Results go out a whole line at a time, the last line's pad as zeros,
 *     and each result is the low 32 bits of the twin's sum: format 0.  Then,
 *     and only once every line of them is written, the record: line 0 of the
 *     slot rewritten whole, the host's fields as they were read and the
 *     module's own -- status, gemm_index, saturations.  Then PTQ_HEAD moves,
 *     and then vector 0 is raised if PTQ_IRQ_AT names the slot.
 *   - A command that did not start reports gemm_index 0xFFFFFFFF.
 *   - A teardown -- PTQ_CTRL.ENABLE written 0 -- drops the command it was
 *     running.  The twin has no way to cancel one, so the GEMM runs on in the
 *     tile and has used its index, and nothing of it is ever written.  The
 *     next slot waits until the tile has finished with it.  PTQ_STATUS.ENABLED
 *     reads 0 at once, because nothing here is in flight: every write has
 *     returned.
 *   - A doorbell that goes back, or past the ring, halts the queue (reason 1),
 *     and so does a host-memory call that fails (reason 2).  A halted queue
 *     takes nothing more until it is enabled again.
 *
 * TIME.  The twin's clock moves only in pta_module_run(), which runs the twin
 * and services the ring before and after.  A doorbell is serviced when it is
 * written, so a command that ends at once ends then.
 *
 * Five switches build a module that is deliberately wrong, each in one way the
 * queue's gate exists to catch.  ci/build_mock.sh builds all five and requires
 * the gate to fail against each:
 *
 *   PTA_MODULE_ABLATE_RECORD_FIRST    the record goes out before the results
 *   PTA_MODULE_ABLATE_REFUSED_INDEX   a refusal reports an index
 *   PTA_MODULE_ABLATE_AFTER_SUCCESS   bit 16 of target is not honoured
 *   PTA_MODULE_ABLATE_ORPHAN          a teardown writes the dropped command's
 *                                     results and record when the tile ends it
 *   PTA_MODULE_ABLATE_DOORBELL        a doorbell is taken whatever it says
 *
 * C99, the standard library, and pta_chiplet_twin.h.
 */
#ifndef PTA_CHIPLET_MODULE_H
#define PTA_CHIPLET_MODULE_H

#include <stdint.h>

#include "pta_chiplet_twin.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Page 1, the queue (the design's section 4) ---------------------------- */
#define PTA_MODULE_PTQ          0x1000u
#define PTA_MODULE_PTQ_ID       0x1000u   /* R    [31:8] "PTQ", [7:0] the page's version */
#define PTA_MODULE_PTQ_CAPS     0x1004u   /* R    [4:0] log2 most slots [8] v1 [9] v2 [15:12] vectors */
#define PTA_MODULE_PTQ_CTRL     0x1008u   /* RW   [0] ENABLE [1] IRQ_EN */
#define PTA_MODULE_PTQ_STATUS   0x100Cu   /* R    [0] ENABLED [1] HALTED [15:8] why */
#define PTA_MODULE_PTQ_BASE_LO  0x1010u   /* RW   the ring, 4 KiB aligned; only while ENABLE is 0 */
#define PTA_MODULE_PTQ_BASE_HI  0x1014u   /* RW */
#define PTA_MODULE_PTQ_SIZE     0x1018u   /* RW   log2 of the ring's slots; only while ENABLE is 0 */
#define PTA_MODULE_PTQ_TAIL     0x1020u   /* RW   the doorbell, a free-running index */
#define PTA_MODULE_PTQ_HEAD     0x1024u   /* R    the next slot the module will end */
#define PTA_MODULE_PTQ_IRQ_AT   0x1028u   /* RW   vector 0 when PTQ_HEAD reaches this */
#define PTA_MODULE_PTQ_FAULT_AT 0x102Cu   /* R    the index the queue halted at */

#define PTA_MODULE_PTQ_ID_VALUE   0x50545101u   /* "PTQ", version 1 */
#define PTA_MODULE_RING_LOG2_MAX  10u
#define PTA_MODULE_VECTORS        2u
#define PTA_MODULE_CAPS_VALUE     (PTA_MODULE_RING_LOG2_MAX | (1u << 8) | (PTA_MODULE_VECTORS << 12))

#define PTA_MODULE_CTRL_ENABLE    0x1u
#define PTA_MODULE_CTRL_IRQ_EN    0x2u
#define PTA_MODULE_STATUS_ENABLED 0x1u
#define PTA_MODULE_STATUS_HALTED  0x2u
#define PTA_MODULE_HALT_DOORBELL  1u      /* PTQ_STATUS[15:8] */
#define PTA_MODULE_HALT_MEMORY    2u

/* ---- The ring and its descriptor: the host path's version 1, field for field */
#define PTA_MODULE_LINE           64u
#define PTA_MODULE_SLOT           128u
#define PTA_MODULE_D_SIZE_VERSION 0x00u
#define PTA_MODULE_D_TARGET       0x04u   /* [7:0] instance [15:8] bank [31:16] flags */
#define PTA_MODULE_D_FORMAT       0x08u   /* 0: int8 operands, int32 results */
#define PTA_MODULE_D_M            0x0Cu
#define PTA_MODULE_D_K            0x10u
#define PTA_MODULE_D_N            0x14u
#define PTA_MODULE_D_ACT          0x18u   /* M x K, row-major */
#define PTA_MODULE_D_WGT          0x20u   /* K x N, row-major */
#define PTA_MODULE_D_OUT          0x28u   /* M x N, row-major, whole lines */
#define PTA_MODULE_D_STATUS       0x30u   /* the module's, from here on */
#define PTA_MODULE_D_INDEX        0x34u
#define PTA_MODULE_D_SATS         0x38u
#define PTA_MODULE_V1             ((1u << 16) | 64u)
#define PTA_MODULE_AFTER_SUCCESS  0x00010000u   /* target bit 16 */

#define PTA_MODULE_PENDING        0
#define PTA_MODULE_DONE           1
#define PTA_MODULE_REFUSED        2
#define PTA_MODULE_LOST           3       /* written by the host, in a teardown */
#define PTA_MODULE_NOT_ISSUED     4
#define PTA_MODULE_NOT_RUN        5
#define PTA_MODULE_NO_INDEX       0xFFFFFFFFu

/* The largest a command may be, in elements of any one operand or of its
 * results: the model's own bound on what it copies, not a module's.  Past it a
 * command ends NOT ISSUED. */
#define PTA_MODULE_ELEM_MAX       (1u << 24)

/* ---- Host memory, as CXL.cache reaches it ----------------------------------- */
/* Each returns 0, or not 0 if the access failed; the queue then halts. */
typedef int  (*pta_module_read_line_fn)(void *ctx, uint64_t addr, uint8_t line[64]);
typedef int  (*pta_module_write_line_fn)(void *ctx, uint64_t addr, const uint8_t line[64]);
/* An MSI-X: vector 0 for the queue, 1 for the map's PTA_IRQ_STATUS. */
typedef void (*pta_module_msi_fn)(void *ctx, unsigned vector);

typedef struct {
    pta_module_read_line_fn  read_line;
    pta_module_write_line_fn write_line;
    pta_module_msi_fn        msi;      /* may be NULL */
    void                    *ctx;
} pta_module_host;

typedef struct pta_module pta_module;

/* A module in front of `tile`, which it does not own, or NULL if either call
 * of host memory is missing or memory ran out.  The queue leaves reset
 * disabled, with every register zero. */
pta_module *pta_module_new(pta_twin *tile, const pta_module_host *host);
void        pta_module_free(pta_module *m);
pta_twin   *pta_module_tile(pta_module *m);

/* The BAR, as CXL.io reaches it. */
uint32_t    pta_module_read32(pta_module *m, uint32_t offset);
void        pta_module_write32(pta_module *m, uint32_t offset, uint32_t value);

/* Advance the tile's clock, servicing the ring before and after. */
void        pta_module_run(pta_module *m, uint64_t cycles);

/* A function-level reset: the twin's reset, and the queue's registers to zero.
 * A command in the tile is discarded and nothing of it is written. */
void        pta_module_reset(pta_module *m);

#ifdef __cplusplus
}
#endif

#endif /* PTA_CHIPLET_MODULE_H */
