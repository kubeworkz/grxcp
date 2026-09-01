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
// TWO DEFECTS IN THE SHIM SHAPE THIS FILE, both measured on import and both
// recorded in third_party/grx930/README.md:
//
//   - STATUS.BUSY is never asserted, so npu_c930_wait_idle() returns on its
//     first poll whether or not the model has finished. DONE is the only thing
//     that separates running from finished here -- which our backend checks,
//     and which is the only reason this is safe to wire up at all.
//   - The GEMM path indexes ddr[] unchecked, while mem_read/mem_write both
//     range-check. A C_BASE near the top of the window writes past the array
//     (watched under ASAN). So the adapter below refuses a launch whose A/B/C
//     extents leave the window, the way an address decoder would, rather than
//     forwarding a START that corrupts the process.
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

void case_not_finished() {
  std::printf("a launch that has not finished when the host looks:\n");
  std::printf("        (their STATUS.BUSY is never asserted, so wait_idle\n"
              "         returns at once -- DONE is the only evidence there is)\n");
  const int M = 8, N = 12, K = 16;
  const uint32_t A_ADDR = 0x0100, B_ADDR = 0x0400, C_ADDR = 0x0800;

  npu_dpi_init();
  int8_t A[8 * 16], B[16 * 12];
  load_operands(A, B, M, N, K, A_ADDR, B_ADDR);

  Shim s;
  s.quantum = 1;   // far less model time than this shape needs
  npu_c930_device_t dev;
  npu_c930_attach_model(&dev, shim_read, shim_write, &s);
  if (npu_c930_detect(&dev) != 1) { check(false, "detected"); return; }

  const int polls_before = s.polls;
  const int rc = npu_c930_gemm(&dev, M, N, K, A_ADDR, B_ADDR, C_ADDR);
  // Read AFTER the launch: expected_cycles() reads the live CSRs, so asking
  // before npu_c930_gemm has programmed DIM_M/N/K returns 0 and the printed
  // comparison is between the shape and nothing.
  const int expected = npu_dpi_expected_cycles();
  const int advanced = (s.polls - polls_before) * s.quantum;
  std::printf("        shape needs %d model cycles; %d poll(s) advanced %d\n",
              expected, s.polls - polls_before, advanced);
  check(advanced < expected, "the poll really did stop short of the shape");
  check(rc != 0, "the GEMM is reported FAILED, not as success");
  check(s.starts == 1, "even though START was accepted");

  // And the direction of the mistake matters. This model writes C at START,
  // before it will admit to being done -- so the refusal above threw away an
  // answer that happened to be correct. That is the safe direction: on hardware
  // the same state is a DMA still in flight, and the buffer holds whatever it
  // held before.
  const int32_t c00 = ddr_get32(C_ADDR);
  check(c00 == reference(A, B, N, K, 0, 0),
        "C was already correct in DDR -- the refusal was conservative, not wrong");
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

// Not a gate. Their header names four registers as accurate and three of them
// read the wrong value; gating on someone else's defect turns our suite amber
// for a bug we cannot fix here. Recorded, reported, and printed every run so it
// cannot quietly stop being true -- the gate arrives with the fix.
void observe_counters() {
  std::printf("counter layout, as OBSERVED (not gated -- see the README):\n");
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
    {0x24, "CYCLE_COUNT (header)"},
    {0x28, "(undocumented)     "},
    {0x2c, "OP_COUNT    (header)"},
    {0x30, "STALL_COUNT (header)"},
    {0x34, "DMA_CT      (header)"},
  };
  for (const auto& r : kRegs)
    std::printf("        0x%02x  %s = %u\n", r.off, r.name,
                npu_dpi_csr_read(NPU_C930_MMIO_BASE + r.off));
  std::printf("        M*N*K*2 = %d, which is at 0x28, not at OP_COUNT's 0x2c\n",
              M * N * K * 2);
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
  case_not_finished();
  case_out_of_window();
  observe_counters();
  std::printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "PASSED",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures ? 1 : 0;
}
