/*
 * pta_chiplet_module.c - the first board's PTA module: the transport's queue,
 * in front of the chiplet's twin.
 *
 * See pta_chiplet_module.h for what this is and is not, and
 * docs/designs/pta_board_transport.md for the design it models.  The tile is
 * pta_chiplet_twin.c, called through its own interface and nothing else.
 */
#include "pta_chiplet_module.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    int      live;         /* a command is in the tile */
    int      orphan;       /* dropped by a teardown: it runs on, and nothing is written */
    uint32_t idx;          /* its slot's index */
    uint8_t  desc[PTA_MODULE_LINE];   /* line 0 as it was read */
    uint32_t M, N;
    uint64_t out;
    uint32_t index;        /* PTA_GEMM_CT as it was handed to the tile */
    int64_t *C;
    int      status;
    long     sats;
} inflight_t;

struct pta_module {
    pta_twin       *tile;
    pta_module_host host;

    uint32_t enable, irq_en, enabled;
    uint32_t base_lo, base_hi, size_log2;
    uint32_t tail, head, irq_at, fault_at;
    uint32_t halted;       /* 0, or why: PTA_MODULE_HALT_* */
    int      prev;         /* how the last command ended since ENABLE; 0 for none */
    int      irq_level;    /* the tile's line, for vector 1's edge */
    inflight_t cur;
};

/* ---- little-endian bytes, whatever the host is ---- */
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static uint64_t get64(const uint8_t *p) { return (uint64_t)get32(p) | (uint64_t)get32(p + 4) << 32; }
static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static void put64(uint8_t *p, uint64_t v) { put32(p, (uint32_t)v); put32(p + 4, (uint32_t)(v >> 32)); }

static uint64_t lines_of(uint64_t bytes) { return (bytes + PTA_MODULE_LINE - 1) / PTA_MODULE_LINE; }

static uint64_t slot_addr(const pta_module *m, uint32_t idx) {
    const uint64_t base = (uint64_t)m->base_hi << 32 | (m->base_lo & ~0xFFFu);
    return base + (uint64_t)(idx & ((1u << m->size_log2) - 1u)) * PTA_MODULE_SLOT;
}

static void halt(pta_module *m, uint32_t why) {
    m->halted = why;
    m->fault_at = m->head;
}

/* The record: line 0 of the slot, the host's fields as they were read and the
 * module's own.  Then PTQ_HEAD, then the interrupt, in that order. */
static int record(pta_module *m, uint32_t idx, const uint8_t desc[PTA_MODULE_LINE], int status,
                  uint32_t index, uint64_t sats) {
    uint8_t line[PTA_MODULE_LINE];
    memcpy(line, desc, sizeof line);
    put32(line + PTA_MODULE_D_STATUS, (uint32_t)status);
    put32(line + PTA_MODULE_D_INDEX, index);
    put64(line + PTA_MODULE_D_SATS, sats);
    if (m->host.write_line(m->host.ctx, slot_addr(m, idx), line) != 0) {
        halt(m, PTA_MODULE_HALT_MEMORY);
        return -1;
    }
    m->head = idx + 1u;
    m->prev = status;
    if (m->irq_en && m->head == m->irq_at && m->host.msi) m->host.msi(m->host.ctx, 0);
    return 0;
}

/* Format 0's results: the low 32 bits of each sum, a whole line at a time, and
 * the last line's pad as zeros. */
static int write_results(pta_module *m, const inflight_t *f) {
    const uint64_t count = (uint64_t)f->M * f->N;
    const uint64_t n = lines_of(count * 4u);
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t line[PTA_MODULE_LINE];
        memset(line, 0, sizeof line);
        for (uint64_t e = i * 16u; e < count && e < (i + 1u) * 16u; ++e)
            put32(line + (e - i * 16u) * 4u, (uint32_t)(uint64_t)f->C[e]);
        if (m->host.write_line(m->host.ctx, f->out + i * PTA_MODULE_LINE, line) != 0) return -1;
    }
    return 0;
}

static void drop(inflight_t *f) {
    free(f->C);
    memset(f, 0, sizeof *f);
}

/* The tile has ended the command in flight: write what the design says. */
static void finish(pta_module *m) {
    inflight_t *f = &m->cur;
    if (!f->live || f->status == PTA_TWIN_PENDING) return;
#ifndef PTA_MODULE_ABLATE_ORPHAN
    if (f->orphan) { drop(f); return; }
#endif
    if (f->status == PTA_TWIN_DONE) {
#ifdef PTA_MODULE_ABLATE_RECORD_FIRST
        uint8_t desc[PTA_MODULE_LINE];
        memcpy(desc, f->desc, sizeof desc);
        if (record(m, f->idx, desc, PTA_MODULE_DONE, f->index, (uint64_t)f->sats) == 0 &&
            write_results(m, f) != 0)
            halt(m, PTA_MODULE_HALT_MEMORY);
#else
        if (write_results(m, f) != 0) halt(m, PTA_MODULE_HALT_MEMORY);
        else (void)record(m, f->idx, f->desc, PTA_MODULE_DONE, f->index, (uint64_t)f->sats);
#endif
    } else if (f->status == PTA_TWIN_REFUSED) {
#ifdef PTA_MODULE_ABLATE_REFUSED_INDEX
        (void)record(m, f->idx, f->desc, PTA_MODULE_REFUSED, f->index, 0);
#else
        (void)record(m, f->idx, f->desc, PTA_MODULE_REFUSED, PTA_MODULE_NO_INDEX, 0);
#endif
    } else {
        (void)record(m, f->idx, f->desc, PTA_MODULE_LOST, PTA_MODULE_NO_INDEX, 0);
    }
    drop(f);
}

/* Read `count` int8 operands from `addr`, a line at a time, into the twin's
 * integers. */
static int read_operands(pta_module *m, uint64_t addr, uint64_t count, int32_t *out) {
    const uint64_t n = lines_of(count);
    for (uint64_t i = 0; i < n; ++i) {
        uint8_t line[PTA_MODULE_LINE];
        if (m->host.read_line(m->host.ctx, addr + i * PTA_MODULE_LINE, line) != 0) return -1;
        for (uint64_t e = i * PTA_MODULE_LINE; e < count && e < (i + 1u) * PTA_MODULE_LINE; ++e)
            out[e] = (int32_t)(int8_t)line[e - i * PTA_MODULE_LINE];
    }
    return 0;
}

/* Take the slot at PTQ_HEAD: read it, decide, and end it or hand it to the tile. */
static void take(pta_module *m) {
    const uint32_t idx = m->head;
    uint8_t d[PTA_MODULE_LINE];
    if (m->host.read_line(m->host.ctx, slot_addr(m, idx), d) != 0) {
        halt(m, PTA_MODULE_HALT_MEMORY);
        return;
    }
    const uint32_t sv = get32(d + PTA_MODULE_D_SIZE_VERSION), target = get32(d + PTA_MODULE_D_TARGET);
    const uint32_t fmt = get32(d + PTA_MODULE_D_FORMAT);
    const uint32_t M = get32(d + PTA_MODULE_D_M), K = get32(d + PTA_MODULE_D_K), N = get32(d + PTA_MODULE_D_N);
    const uint64_t act = get64(d + PTA_MODULE_D_ACT), wgt = get64(d + PTA_MODULE_D_WGT);
    const uint64_t out = get64(d + PTA_MODULE_D_OUT);
    const uint32_t flags = target & 0xFFFF0000u;

    /* Not issued: what the module cannot use, decided before the tile sees it. */
    if (sv != PTA_MODULE_V1 || fmt != 0 || (target & 0xFFu) != 0 || (flags & ~PTA_MODULE_AFTER_SUCCESS) ||
        M == 0 || K == 0 || N == 0 || (act | wgt | out) % PTA_MODULE_LINE != 0 ||
        (uint64_t)M * K > PTA_MODULE_ELEM_MAX || (uint64_t)K * N > PTA_MODULE_ELEM_MAX ||
        (uint64_t)M * N > PTA_MODULE_ELEM_MAX) {
        (void)record(m, idx, d, PTA_MODULE_NOT_ISSUED, PTA_MODULE_NO_INDEX, 0);
        return;
    }
#ifndef PTA_MODULE_ABLATE_AFTER_SUCCESS
    /* Not run: it asked to follow a success, and the command before it did not
     * end done.  The first command since ENABLE has none before it. */
    if ((flags & PTA_MODULE_AFTER_SUCCESS) && m->prev != 0 && m->prev != PTA_MODULE_DONE) {
        (void)record(m, idx, d, PTA_MODULE_NOT_RUN, PTA_MODULE_NO_INDEX, 0);
        return;
    }
#endif
    int32_t *A = (int32_t *)malloc((size_t)M * K * sizeof *A);
    int32_t *B = (int32_t *)malloc((size_t)K * N * sizeof *B);
    int64_t *C = (int64_t *)malloc((size_t)M * N * sizeof *C);
    if (!A || !B || !C) {
        free(A); free(B); free(C);
        (void)record(m, idx, d, PTA_MODULE_NOT_ISSUED, PTA_MODULE_NO_INDEX, 0);
        return;
    }
    if (read_operands(m, wgt, (uint64_t)K * N, B) != 0 || read_operands(m, act, (uint64_t)M * K, A) != 0) {
        free(A); free(B); free(C);
        halt(m, PTA_MODULE_HALT_MEMORY);
        return;
    }
    inflight_t *f = &m->cur;
    memset(f, 0, sizeof *f);
    f->idx = idx;
    memcpy(f->desc, d, sizeof f->desc);
    f->M = M; f->N = N; f->out = out; f->C = C;
    f->index = pta_twin_read32(m->tile, PTA_TWIN_GEMM_CT);
    f->status = PTA_TWIN_PENDING;

    pta_twin_cmd c;
    memset(&c, 0, sizeof c);
    c.bank = (int)((target >> 8) & 0xFFu);
    c.M = (int)M; c.N = (int)N; c.K = (int)K;
    c.A = A; c.B = B; c.C = C;
    c.status = &f->status;
    c.sats = &f->sats;
    const int accepted = pta_twin_submit(m->tile, &c);
    free(A);                         /* the twin copied both on accepting */
    free(B);
    if (accepted != 0) {
        drop(f);
        (void)record(m, idx, d, PTA_MODULE_NOT_ISSUED, PTA_MODULE_NO_INDEX, 0);
        return;
    }
    f->live = 1;
}

/* Vector 1 on a rising edge of the map's interrupt line. */
static void irq_edge(pta_module *m) {
    const int level = pta_twin_irq(m->tile);
    if (level && !m->irq_level && m->host.msi) m->host.msi(m->host.ctx, 1);
    m->irq_level = level;
}

/* Everything that can happen now: end what the tile has ended, then take
 * slots until one is in the tile or none is left.  A command the tile refuses
 * as it is handed over ends here too, so the interrupt line is looked at last. */
static void service(pta_module *m) {
    for (;;) {
        if (m->cur.live) {
            finish(m);
            if (m->cur.live) break;       /* still in the tile */
            continue;
        }
        if (!m->enable || m->halted || m->head == m->tail) break;
        take(m);
    }
    irq_edge(m);
}

pta_module *pta_module_new(pta_twin *tile, const pta_module_host *host) {
    if (!tile || !host || !host->read_line || !host->write_line) return NULL;
    pta_module *m = (pta_module *)calloc(1, sizeof *m);
    if (!m) return NULL;
    m->tile = tile;
    m->host = *host;
    return m;
}

void pta_module_free(pta_module *m) {
    if (!m) return;
    if (m->cur.live) pta_twin_reset(m->tile);   /* the tile must not write into freed memory */
    drop(&m->cur);
    free(m);
}

pta_twin *pta_module_tile(pta_module *m) { return m ? m->tile : NULL; }

uint32_t pta_module_read32(pta_module *m, uint32_t off) {
    if (!m) return 0;
    if (off < PTA_MODULE_PTQ) return pta_twin_read32(m->tile, off);
    switch (off) {
        case PTA_MODULE_PTQ_ID:       return PTA_MODULE_PTQ_ID_VALUE;
        case PTA_MODULE_PTQ_CAPS:     return PTA_MODULE_CAPS_VALUE;
        case PTA_MODULE_PTQ_CTRL:     return m->enable | (m->irq_en << 1);
        case PTA_MODULE_PTQ_STATUS:
            return m->enabled | (m->halted ? PTA_MODULE_STATUS_HALTED : 0u) | (m->halted << 8);
        case PTA_MODULE_PTQ_BASE_LO:  return m->base_lo & ~0xFFFu;
        case PTA_MODULE_PTQ_BASE_HI:  return m->base_hi;
        case PTA_MODULE_PTQ_SIZE:     return m->size_log2;
        case PTA_MODULE_PTQ_TAIL:     return m->tail;
        case PTA_MODULE_PTQ_HEAD:     return m->head;
        case PTA_MODULE_PTQ_IRQ_AT:   return m->irq_at;
        case PTA_MODULE_PTQ_FAULT_AT: return m->fault_at;
    }
    return 0;
}

void pta_module_write32(pta_module *m, uint32_t off, uint32_t v) {
    if (!m) return;
    if (off < PTA_MODULE_PTQ) {
        pta_twin_write32(m->tile, off, v);
        irq_edge(m);
        return;
    }
    switch (off) {
        case PTA_MODULE_PTQ_CTRL: {
            const uint32_t en = v & PTA_MODULE_CTRL_ENABLE;
            m->irq_en = (v & PTA_MODULE_CTRL_IRQ_EN) ? 1u : 0u;
            if (en && !m->enable) {
                m->enable = m->enabled = 1;
                m->head = m->tail;              /* the ring starts empty */
                m->prev = 0;
                m->halted = 0;
                service(m);
            } else if (!en && m->enable) {
                m->enable = 0;
                if (m->cur.live) m->cur.orphan = 1;
                m->enabled = 0;                 /* every write has returned: nothing is in flight */
            }
            return;
        }
        case PTA_MODULE_PTQ_BASE_LO: if (!m->enable) m->base_lo = v & ~0xFFFu; return;
        case PTA_MODULE_PTQ_BASE_HI: if (!m->enable) m->base_hi = v; return;
        case PTA_MODULE_PTQ_SIZE:
            if (!m->enable && v <= PTA_MODULE_RING_LOG2_MAX) m->size_log2 = v;
            return;
        case PTA_MODULE_PTQ_TAIL:
            if (!m->enable) { m->tail = v; return; }
#ifndef PTA_MODULE_ABLATE_DOORBELL
            if (v - m->head > (1u << m->size_log2) || v - m->tail > 0x80000000u) {
                halt(m, PTA_MODULE_HALT_DOORBELL);
                return;
            }
#endif
            m->tail = v;
            service(m);
            return;
        case PTA_MODULE_PTQ_IRQ_AT: m->irq_at = v; return;
    }
}

void pta_module_run(pta_module *m, uint64_t cycles) {
    if (!m) return;
    service(m);
    pta_twin_run(m->tile, cycles);
    service(m);
}

void pta_module_reset(pta_module *m) {
    if (!m) return;
    pta_twin_reset(m->tile);       /* writes LOST into the command's status, which is still ours */
    drop(&m->cur);
    pta_twin *tile = m->tile;
    const pta_module_host host = m->host;
    memset(m, 0, sizeof *m);
    m->tile = tile;
    m->host = host;
}
