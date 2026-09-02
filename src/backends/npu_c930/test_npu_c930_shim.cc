// The NPU backend against the GRX930 team's own register model.
//
// test_npu_c930_model.cc drives this backend through four models we wrote
// ourselves -- ABSENT, DEAD, LIVE, WEDGED. Each was built to produce one
// decision, which makes it excellent evidence about the decision and no
// evidence at all about the register map: we invented both sides of that
// conversation.
//
// third_party/grx930/npu_dpi_shim.c is the fifth model and the first one we did
// not write. It comes from the team that owns the RTL, carries their CSR
// addresses, and computes a real INT8 GEMM into a 64 KB DDR array. Driving
// npu_c930_gemm() through it tests the whole sequence -- idle, program, start,
// poll, DONE, read C -- against someone else's idea of what those registers do.
//
// A MODEL IS NOT HARDWARE. Their shim's own header says it is not cycle
// accurate, and its GEMM is a C triple loop. Passing here says our host drives
// the register map correctly. It says nothing about the c930, it does not meet
// the phase 7 exit gate, and per AGENTS.md no green run here may be reported as
// the NPU working.
//
// FIVE DEFECTS WERE MEASURED ON IMPORT AND ALL FIVE ARE NOW FIXED UPSTREAM --
// the history is in third_party/grx930/README.md. Two of them shaped this file
// and are worth keeping in view:
//
//   - STATUS.BUSY was never asserted, so npu_c930_wait_idle returned on its
//     first poll whether or not the model had finished. Fixed, and the RTL
//     answer came with it: c930_npu_csr.sv drives bit 0 from the core's i_busy,
//     so the map was right and the model was wrong. case_busy_covers_the_gemm
//     now pins the property our poll loop rests on.
//   - The GEMM path indexed ddr[] unchecked (watched under ASAN). Fixed
//     upstream, twice: first with an extent check, then -- after this side
//     showed a base near 2^32 wrapping straight past it -- with a wrap-safe
//     form. The adapter below still does its own check in 64-bit. That
//     duplication is deliberate and stays: the guard that matters is the one in
//     the process being protected.
//
// A sixth followed from reading the fix rather than the original: STATUS.DONE
// was inferred from the counters instead of latched. Also fixed. See
// case_done_is_latched, which now gates the latch and the two properties on
// our side that made the inference harmless while it lasted.
//
// The layout the shim reads is its own (A row-major m x k, B row-major k x n).
// Whether that is the layout a column-major BLAS caller means is a separate
// question and is not what this file tests: this file tests the register
// sequence.

#include "npu_c930.h"

extern "C" {
#include "npu_dpi_shim.h"
}

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

int g_failures = 0;

void check(bool cond, const char* what) {
  std::printf("  %s  %s\n", cond ? "ok  " : "FAIL", what);
  if (!cond) ++g_failures;
}

constexpr uint32_t kDdrBytes = 65536;   // MEM_BYTES in c930_soc_top

// ---------------------------------------------------------------------------
// The adapter
// ---------------------------------------------------------------------------

// Does [base, base+bytes) fit inside the model's DDR window? In 64-bit, so a
// base near 2^32 cannot wrap into looking small. This is the guard that stops a
// bad C_BASE from writing past the shim's array, so it is checked on its own
// below before any case that would depend on it.
bool in_window(uint64_t base, uint64_t bytes) {
  return bytes == 0 ? base <= kDdrBytes
                    : (base < kDdrBytes && bytes <= (uint64_t)kDdrBytes - base);
}

struct Shim {
  int      quantum = 4096;  // model cycles advanced per STATUS read
  int      polls   = 0;     // STATUS reads, i.e. how far model time advanced
  int      starts  = 0;     // STARTs actually forwarded to the model
  int      refused = 0;     // STARTs the address decoder rejected
  uint32_t err     = 0;     // STATUS.ERROR raised by the decoder
  uint32_t m = 0, n = 0, k = 0;
  uint32_t a = 0, b = 0, c = 0;
};

// Reading STATUS is what advances model time. The shim's state machine only
// moves inside npu_dpi_run(), which no host would ever call -- so the adapter
// supplies it, on the poll. That is not a fudge: it is the model's stand-in for
// the wall-clock a real poll loop burns.
uint32_t shim_read(void* ctx, uint32_t off) {
  Shim* s = static_cast<Shim*>(ctx);
  if (off == 0x04) {
    ++s->polls;
    npu_dpi_run(s->quantum);
    return npu_dpi_csr_read(NPU_C930_MMIO_BASE + off) | s->err;
  }
  return npu_dpi_csr_read(NPU_C930_MMIO_BASE + off);
}

void shim_write(void* ctx, uint32_t off, uint32_t v) {
  Shim* s = static_cast<Shim*>(ctx);
  switch (off) {
    case 0x08: s->m = v; break;
    case 0x0C: s->n = v; break;
    case 0x10: s->k = v; break;
    case 0x14: s->a = v; break;
    case 0x18: s->b = v; break;
    case 0x1C: s->c = v; break;
    default: break;
  }

  if (off == 0x00 && (v & NPU_C930_CTRL_START)) {
    s->err = 0;
    const uint64_t a_bytes = (uint64_t)s->m * s->k;        // INT8
    const uint64_t b_bytes = (uint64_t)s->k * s->n;        // INT8
    const uint64_t c_bytes = (uint64_t)s->m * s->n * 4u;   // INT32
    if (!in_window(s->a, a_bytes) || !in_window(s->b, b_bytes) ||
        !in_window(s->c, c_bytes)) {
      // An address decoder answers a transaction outside its window with an
      // error, not by writing somewhere else. The START is not forwarded.
      s->err = NPU_C930_STATUS_ERROR;
      ++s->refused;
      return;
    }
    ++s->starts;
  }
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + off, v);
}

// ---------------------------------------------------------------------------
// DDR helpers (byte lanes through the strobe, the way the AXI write channel is)
// ---------------------------------------------------------------------------

void ddr_put8(uint32_t addr, int8_t v) {
  const uint32_t word = addr & ~3u, lane = addr & 3u;
  npu_dpi_mem_write(word, ((uint32_t)(uint8_t)v) << (8 * lane), 1u << lane);
}

int32_t ddr_get32(uint32_t addr) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= ((uint32_t)npu_dpi_mem_read(addr + i)) << (8 * i);
  return (int32_t)v;
}

int32_t reference(const int8_t* A, const int8_t* B, int n, int k, int i, int j) {
  int32_t s = 0;
  for (int p = 0; p < k; ++p) s += (int32_t)A[i * k + p] * (int32_t)B[p * n + j];
  return s;
}

// Fill DDR with a deterministic A and B and hand back the host copies.
void load_operands(int8_t* A, int8_t* B, int m, int n, int k,
                   uint32_t a_addr, uint32_t b_addr) {
  for (int i = 0; i < m * k; ++i) A[i] = (int8_t)(((i * 7) % 11) - 5);
  for (int i = 0; i < k * n; ++i) B[i] = (int8_t)(((i * 5) % 9) - 4);
  for (int i = 0; i < m * k; ++i) ddr_put8(a_addr + i, A[i]);
  for (int i = 0; i < k * n; ++i) ddr_put8(b_addr + i, B[i]);
}

// ---------------------------------------------------------------------------

void case_guard() {
  std::printf("the adapter's address decoder, before anything depends on it:\n");
  check(in_window(0x0400, 64), "a 64-byte result at 0x0400 is inside 64 KB");
  check(in_window(0xffc0, 64), "and one that ends exactly at the top is too");
  check(!in_window(0xfff0, 64), "one that ends past the top is not");
  check(!in_window(0x69ed12b0ull & 0xffffffffull, 64),
        "and a truncated host pointer is nowhere near it");
  check(!in_window(0xffffffffull, 4), "a base at 2^32-1 cannot wrap into range");
}

void case_detect() {
  std::printf("their model, seen through npu_c930_detect():\n");
  npu_dpi_init();
  Shim s;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);

  check(npu_c930_detect(&dev) == 1, "it is detected as present");
  check(dev.present == 1, "and dev.present is set");
  check(npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x08) == 0u,
        "the write-readback probe put DIM_M back the way it found it");
  check(s.starts == 0 && s.refused == 0, "detection started nothing");
}

void case_gemm() {
  std::printf("a GEMM, end to end, and C checked against a host reference:\n");
  const int M = 4, N = 4, K = 8;
  const uint32_t A_ADDR = 0x0100, B_ADDR = 0x0200, C_ADDR = 0x0400;

  npu_dpi_init();
  int8_t A[4 * 8], B[8 * 4];
  load_operands(A, B, M, N, K, A_ADDR, B_ADDR);

  Shim s;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);
  if (npu_c930_detect(&dev) != 1) { check(false, "detected"); return; }

  const int rc = npu_c930_gemm(&dev, M, N, K, A_ADDR, B_ADDR, C_ADDR);
  check(rc == 0, "npu_c930_gemm reports success");
  check(s.starts == 1, "CTRL.START was pulsed exactly once");
  check(s.refused == 0, "and the decoder let it through");

  int bad = 0;
  int32_t first_got = 0, first_want = 0;
  int fi = -1, fj = -1;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      const int32_t got  = ddr_get32(C_ADDR + (i * N + j) * 4);
      const int32_t want = reference(A, B, N, K, i, j);
      if (got != want) {
        if (!bad) { fi = i; fj = j; first_got = got; first_want = want; }
        ++bad;
      }
    }
  if (bad)
    std::printf("        first mismatch C[%d][%d]: model %d, host %d\n",
                fi, fj, first_got, first_want);
  check(bad == 0, "all M*N results match INT8 x INT8 -> INT32 done on the host");
}

// This case used to be "a launch that has not finished when the host looks",
// and it existed because the imported shim never asserted STATUS.BUSY, so
// npu_c930_wait_idle returned on its first poll and DONE was the only evidence
// of completion there was. The GRX930 team fixed that -- and answered the
// question behind it with RTL: c930_npu_csr.sv line 180 reads
// `{29'd0, i_error, done_latch, i_busy}`, so bit 0 IS driven on silicon, and
// their Icarus run shows the core holding i_busy for 2171 cycles of a
// 4x4x8 GEMM. The shim was wrong, not the register map.
//
// So the property worth pinning is the one our poll loop actually rests on:
// BUSY must stay up for the whole GEMM and drop in the same breath as DONE.
// A BUSY that dropped early -- at the end of the 2-cycle pipeline fill, say --
// would send wait_idle home before the result existed, and npu_c930_gemm would
// report "no DONE" on a launch that was merely still running.
void case_busy_covers_the_gemm() {
  std::printf("BUSY is driven, and drops exactly when DONE arrives:\n");
  const int M = 8, N = 12, K = 16;
  const uint32_t A_ADDR = 0x0100, B_ADDR = 0x0400, C_ADDR = 0x0800;

  npu_dpi_init();
  int8_t A[8 * 16], B[16 * 12];
  load_operands(A, B, M, N, K, A_ADDR, B_ADDR);

  // One model cycle per STATUS read, so the poll loop has to go round many
  // times. That is the point: a quantum this small was fatal before the fix.
  Shim s;
  s.quantum = 1;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);
  if (npu_c930_detect(&dev) != 1) { check(false, "detected"); return; }

  const int polls_before = s.polls;
  const int rc = npu_c930_gemm(&dev, M, N, K, A_ADDR, B_ADDR, C_ADDR);
  // AFTER the launch: expected_cycles() reads the live CSRs, so asking before
  // npu_c930_gemm has programmed DIM_M/N/K compares the shape against nothing.
  const int expected = npu_dpi_expected_cycles();
  const int polls = s.polls - polls_before;
  std::printf("        shape needs %d model cycles; the poll went round %d times\n",
              expected, polls);
  check(rc == 0, "the GEMM completes");
  check(polls > expected, "and wait_idle really did have to wait for it");
  check(s.starts == 1, "one START");

  int bad = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
      if (ddr_get32(C_ADDR + (i * N + j) * 4) != reference(A, B, N, K, i, j)) ++bad;
  check(bad == 0, "and C is right at 8x12x16, the largest shape the SoC takes");

  // The transition itself, cycle by cycle, below npu_c930_gemm.
  npu_dpi_init();
  load_operands(A, B, M, N, K, A_ADDR, B_ADDR);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x08, M);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x0C, N);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x10, K);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x14, A_ADDR);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x18, B_ADDR);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x1C, C_ADDR);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x00, NPU_C930_CTRL_START);
  int last_busy = 0, first_done = -1;
  for (int c = 1; c <= 4096; ++c) {
    npu_dpi_run(1);
    const uint32_t st = npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x04);
    if (st & NPU_C930_STATUS_BUSY) last_busy = c;
    if ((st & NPU_C930_STATUS_DONE) && first_done < 0) first_done = c;
    if (first_done >= 0) break;
  }
  std::printf("        BUSY high through cycle %d, DONE first seen at %d\n",
              last_busy, first_done);
  check(first_done == last_busy + 1,
        "DONE appears on the cycle after BUSY drops -- no window in between");
}

// STATUS.DONE IS A LATCH, AND THAT TOOK TWO ROUNDS TO BECOME TRUE.
//
// On import it was inferred rather than latched:
//
//   build_status() = ... | ((CYCLE_LO || OP_COUNT) && !busy ? 2 : 0) | ...
//
// which reads "some counter is non-zero and I am not busy" -- true of any idle
// device that ran an earlier GEMM. Measured then, in one process after one good
// GEMM: a START the model refused outright (DIM_M = 0) read STATUS = 0x02, a
// completion flag for a launch never accepted; and an out-of-window START read
// 0x06, ERROR and DONE together, which says it both failed and completed.
//
// Their own test for the bounds check saw the correct 0x04 because it ran on a
// freshly initialised shim, where CYCLE_LO is still zero. That is the part
// worth remembering: a register whose value depends on prior state cannot be
// characterised from a clean slate alone, and every case below therefore runs a
// good GEMM FIRST and only then asks the awkward question.
//
// Fixed upstream with a real done_latch, set when the cycle countdown reaches
// zero and cleared on CTRL.START -- which is what c930_npu_csr.sv does. Both
// readings are now 0x00 and 0x04.
//
// What this case gates is the pair of properties on OUR side that made those
// readings unable to reach a caller in the first place, and that still stand
// between us and any future model with the same idea: dimensions are validated
// before a register is touched, and ERROR is checked before DONE. Both are
// watched failing -- reorder the two status checks and the last one goes red.
void case_done_is_latched() {
  std::printf("DONE is a latch, and the two checks that guard us either way:\n");
  const int M = 4, N = 4, K = 8;
  const uint32_t A_ADDR = 0x0100, B_ADDR = 0x0200, C_ADDR = 0x0400;

  npu_dpi_init();
  int8_t A[4 * 8], B[8 * 4];
  load_operands(A, B, M, N, K, A_ADDR, B_ADDR);

  Shim s;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);
  if (npu_c930_detect(&dev) != 1) { check(false, "detected"); return; }

  // one good GEMM, so the model's cycle counter is non-zero from here on
  check(npu_c930_gemm(&dev, M, N, K, A_ADDR, B_ADDR, C_ADDR) == 0,
        "a good GEMM first, which leaves CYCLE_LO non-zero");

  // Show the inference directly, below our backend.
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x08, 0);            // DIM_M = 0
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x00, NPU_C930_CTRL_START);
  const uint32_t refused = npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x04);
  std::printf("        a START the model refuses (M=0) reads STATUS=0x%02x\n",
              refused);

  // OUR PROTECTION #1: dimensions are validated before any register is touched,
  // so npu_c930_gemm can never issue the START that produces that reading.
  const int starts_before = s.starts;
  check(npu_c930_gemm(&dev, 0, N, K, A_ADDR, B_ADDR, C_ADDR) != 0,
        "npu_c930_gemm refuses M=0");
  check(s.starts == starts_before,
        "and refuses it BEFORE writing CTRL -- the device is never asked");

  // An out-of-window START, issued below the adapter so the adapter's own guard
  // does not intercept it. ERROR without DONE: it failed, and it did not
  // complete. Before the latch landed this read 0x06 -- both at once.
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x08, M);
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x1C, 0xfff0);   // C_BASE off the end
  npu_dpi_csr_write(NPU_C930_MMIO_BASE + 0x00, NPU_C930_CTRL_START);
  const uint32_t after_err = npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x04);
  std::printf("        an out-of-window START, second time in the process, "
              "reads STATUS=0x%02x\n", after_err);
  check((after_err & NPU_C930_STATUS_ERROR) &&
        !(after_err & NPU_C930_STATUS_DONE),
        "ERROR without DONE -- the two are mutually exclusive");

  // OUR PROTECTION #2: npu_c930_gemm checks ERROR before DONE, so a device
  // reporting both is a failure and not a success. Driven through a hook built
  // for exactly this, rather than through the shim -- the point is our ordering,
  // and a test of our ordering should not depend on how someone else's model
  // happens to reach that state.
  struct Both {
    static uint32_t rd(void*, uint32_t off) {
      if (off == 0x04) return NPU_C930_STATUS_ERROR | NPU_C930_STATUS_DONE;
      return 0x5u;   // any R/W readback, so detect() sees a live register file
    }
    static void wr(void*, uint32_t, uint32_t) {}
  };
  npu_c930_device_t contradictory;
  npu_c930_attach_model(&contradictory, Both::rd, Both::wr, nullptr);
  check(npu_c930_gemm(&contradictory, M, N, K, A_ADDR, B_ADDR, C_ADDR) != 0,
        "npu_c930_gemm reads ERROR|DONE as FAILED, not as DONE");
}

void case_out_of_window() {
  std::printf("a launch whose result would leave the 64 KB window:\n");
  const int M = 4, N = 4, K = 8;
  const uint32_t A_ADDR = 0x0100, B_ADDR = 0x0200;

  npu_dpi_init();
  int8_t A[4 * 8], B[8 * 4];
  load_operands(A, B, M, N, K, A_ADDR, B_ADDR);

  Shim s;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);
  if (npu_c930_detect(&dev) != 1) { check(false, "detected"); return; }

  // 4x4 INT32 is 64 bytes; 0xfff0 leaves 16. Unguarded, the shim writes 48
  // bytes past its array -- watched under ASAN, see the README.
  const int rc = npu_c930_gemm(&dev, M, N, K, A_ADDR, B_ADDR, 0xfff0);
  check(rc != 0, "the GEMM is refused");
  check(s.refused == 1, "by the address decoder");
  check(s.starts == 0, "and START never reached the model");

  // TRUNCATING A HOST POINTER DOES NOT PRODUCE AN OUT-OF-RANGE ADDRESS. IT
  // PRODUCES AN ARBITRARY ONE, AND THE DECODER CANNOT SAVE YOU FROM THE HALF
  // THAT LANDS INSIDE.
  //
  // This check used to truncate a live malloc() and assert the result was
  // refused. It passed -- and it passed by luck. Measured, same machine, same
  // program:
  //
  //   glibc malloc:  0x55ef9bfc62a0  ->  low 32 = 0x9bfc62a0   outside
  //   ASAN  malloc:  0x506000000020  ->  low 32 = 0x00000020   INSIDE
  //
  // Under ASAN the low word is 32, a perfectly ordinary DDR offset, so the
  // decoder waves it through and the DMA writes to an address that has nothing
  // to do with the buffer. The flaky version of this check was found by
  // building it under ASAN during a sabotage run, not by reading it.
  //
  // So both cases are pinned to the measured constants rather than to whatever
  // the allocator does today. What actually stands between a real allocation
  // and this is grxblasGemmEx's own refusal of any pointer above 0xFFFFFFFF --
  // which, on a 64-bit host, is every allocation there is.
  const uint32_t kGlibcLow32 = 0x9bfc62a0u;   // outside the window
  const uint32_t kAsanLow32  = 0x00000020u;   // inside it, by accident
  {
    Shim s2;
    npu_c930_device_t dev2;
    npu_c930_attach_model(&dev2, shim_read, shim_write, &s2);
    npu_c930_detect(&dev2);
    const int rc2 = npu_c930_gemm(&dev2, M, N, K, A_ADDR, B_ADDR, kGlibcLow32);
    check(rc2 != 0 && s2.starts == 0,
          "a truncated glibc pointer (0x9bfc62a0) is refused by the decoder");
  }
  {
    Shim s3;
    npu_c930_device_t dev3;
    npu_c930_attach_model(&dev3, shim_read, shim_write, &s3);
    npu_c930_detect(&dev3);
    const int rc3 = npu_c930_gemm(&dev3, M, N, K, A_ADDR, B_ADDR, kAsanLow32);
    check(rc3 == 0 && s3.starts == 1,
          "a truncated ASAN pointer (0x20) is NOT -- it looks like a real offset");
    std::printf("        so the 32-bit base registers cannot be fed a host\n"
                "        pointer at all: the cast does not fail loudly, it\n"
                "        fails to whatever the low word happens to be.\n");
  }
}

// THIS WAS AN OBSERVATION AND IS NOW A GATE, which is the whole point of having
// written it down.
//
// On import, three of the four counters their header names as accurate read the
// wrong register: OP_COUNT was written at 0x28 and read at 0x2c, so 0x2c
// returned STALL_COUNT, 0x30 returned DMA_CT, and 0x34 read zero forever. That
// was not gated -- a red gate on someone else's defect turns our suite amber for
// a bug we cannot fix here -- so it was printed every run and reported instead,
// with the promise that the gate would arrive with the fix.
//
// It arrived. The cause was not the re-index we guessed at: `ADDR_CYCLE_HI`
// (0x28) exists as a localparam in c930_npu_csr.sv and appears in no case
// statement, so the RTL has a dead register there and the header had simply
// omitted it. CYCLE_COUNT is 32 bits, not 64. Our guess was wrong and theirs was
// checkable, which is why the question was asked rather than assumed.
void case_counters() {
  std::printf("the counters, at the addresses the header documents:\n");
  const int M = 4, N = 4, K = 8;
  npu_dpi_init();
  int8_t A[4 * 8], B[8 * 4];
  load_operands(A, B, M, N, K, 0x0100, 0x0200);
  Shim s;
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);
  npu_c930_detect(&dev);
  npu_c930_gemm(&dev, M, N, K, 0x0100, 0x0200, 0x0400);

  static const struct { uint32_t off; const char* name; } kRegs[] = {
    {0x24, "CYCLE_LO   "},
    {0x28, "(dead)     "},
    {0x2c, "OP_COUNT   "},
    {0x30, "STALL_COUNT"},
    {0x34, "DMA_CT     "},
  };
  for (const auto& r : kRegs)
    std::printf("        0x%02x  %s = %u\n", r.off, r.name,
                npu_dpi_csr_read(NPU_C930_MMIO_BASE + r.off));

  check(npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x2c) ==
            (uint32_t)(M * N * K * 2),
        "OP_COUNT reads M*N*K*2 at 0x2c");
  check(npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x28) == 0,
        "0x28 reads 0 -- ADDR_CYCLE_HI is dead in the RTL");
  check(npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x24) != 0,
        "CYCLE_LO counted something");
  check(npu_dpi_csr_read(NPU_C930_MMIO_BASE + 0x30) == 0,
        "STALL_COUNT is 0 (this model does not stall)");
}

}  // namespace

int main() {
  std::printf("=== GRX930 NPU backend, against the GRX930 team's register model ===\n");
  std::printf("NOTE: their shim is a MODEL. It computes the GEMM in C and its\n"
              "      own header says it is not cycle accurate. Passing here says\n"
              "      our host drives the register map. It says nothing about the\n"
              "      c930, and does not meet the phase 7 exit gate.\n\n");
  case_guard();
  case_detect();
  case_gemm();
  case_busy_covers_the_gemm();
  case_done_is_latched();
  case_out_of_window();
  case_counters();
  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
