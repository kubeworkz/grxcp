// grxblasGemmEx, all the way down to the c930 RTL.
//
// The whole stack in one line of test: grxMalloc carves the DDR window,
// grxMemcpy moves A and B into it, grxblasGemmEx routes to the NPU engine and
// programs the CSRs, the Verilated RTL runs the GEMM through its own AXI
// master, and grxMemcpy reads C back. Nothing here is a stand-in except the
// memory, and the memory is a model on every path including the real one.
//
// This file is built only when the c930 RTL is on hand
// (-DGRXCP_C930_RTL_DIR=/path/to/c930). Without it there is nothing to
// verilate and the target does not exist, which is why there is no skip inside
// main: a test that cannot be built should not be a test that reports 77.
//
// A MODEL IS NOT HARDWARE, AND NEITHER IS A SIMULATION. Passing here says the
// host and the RTL agree. It says nothing about silicon: no timing closure, no
// physical DDR, no clock domain crossing, no c930 in a socket. Per AGENTS.md no
// run through this may be reported as the NPU working, and the phase 7 exit
// gate is unchanged.
//
// What it CAN say, which nothing before it could: the counters our other
// models report are not the RTL's. See the table this prints.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"
#include "npu_rtl_harness.h"

#include <cstdio>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

// Column-major reference, matching grxBLAS's convention (grxblas.h).
int32_t ref_cm(const int8_t* A, const int8_t* B, int m, int k, int i, int j) {
  int32_t s = 0;
  for (int l = 0; l < k; ++l)
    s += (int32_t)A[i + (size_t)l * m] * (int32_t)B[l + (size_t)j * k];
  return s;
}

bool run_case(grxblasHandle_t h, int m, int n, int k, const char* label) {
  const int lda = m, ldb = k, ldc = m;
  std::vector<int8_t> A((size_t)m * k), B((size_t)k * n);
  for (size_t i = 0; i < A.size(); ++i) A[i] = (int8_t)(((i * 7) % 11) - 5);
  for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)(((i * 5) % 9) - 4);

  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (grxMalloc(&dA, A.size()) != grxSuccess ||
      grxMalloc(&dB, B.size()) != grxSuccess ||
      grxMalloc(&dC, (size_t)m * n * sizeof(int32_t)) != grxSuccess) {
    std::printf("  FAIL  %s (allocation)\n", label);
    return false;
  }
  grxMemcpy(dA, A.data(), A.size(), grxMemcpyDefault);
  grxMemcpy(dB, B.data(), B.size(), grxMemcpyDefault);

  const float alpha = 1.0f, beta = 0.0f;
  const grxblasStatus_t s =
      grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &alpha,
                    dA, GRX_R_8I, lda, dB, GRX_R_8I, ldb, &beta,
                    dC, GRX_R_32I, ldc);
  if (s != GRXBLAS_STATUS_SUCCESS) {
    std::printf("  FAIL  %s (%s)\n", label, grxblasGetStatusString(s));
    grxFree(dA); grxFree(dB); grxFree(dC);
    return false;
  }

  std::vector<int32_t> got((size_t)m * n, 0);
  grxMemcpy(got.data(), dC, got.size() * sizeof(int32_t), grxMemcpyDefault);
  grxFree(dA); grxFree(dB); grxFree(dC);

  int bad = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      const int32_t want = ref_cm(A.data(), B.data(), m, k, i, j);
      const int32_t g = got[(size_t)i + (size_t)j * ldc];
      if (g != want) {
        if (bad < 3)
          std::printf("        C[%d][%d]: RTL %d, host %d\n", i, j, g, want);
        ++bad;
      }
    }
  if (bad) std::printf("  FAIL  %s (%d/%d wrong)\n", label, bad, m * n);
  else     std::printf("  ok    %s\n", label);
  return bad == 0;
}

}  // namespace

int main() {
  // BEFORE THE FIRST grx CALL.
  const bool installed = grxtest::rtl_install();

  section("the RTL is installed as the enumerated device");
  check(installed, "register, memory and backend-kind hooks all accepted");

  int count = 0;
  check(grxGetDeviceCount(&count) == grxSuccess, "grxGetDeviceCount");

  int npu = -1;
  grxDeviceProp_t prop{};
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) != grxSuccess) continue;
    if (p.deviceType == GRX_DEVICE_TYPE_NPU) { npu = i; prop = p; break; }
  }
  check(npu >= 0, "an NPU device appears in the device table");
  if (npu < 0) return grxtest::report();

  section("and it says what it is");
  std::printf("  note  device %d: \"%s\", backend %d\n", npu, prop.name,
              (int)prop.backend);
  check(prop.backend == GRX_BACKEND_RTLSIM,
        "backend is GRX_BACKEND_RTLSIM, not MODEL and not SILICON");
  check(std::strstr(prop.name, "NOT hardware") != nullptr,
        "and the name still says NOT hardware, because it is not");

  std::printf("\n  *** THIS IS THE c930 RTL UNDER VERILATOR, NOT SILICON. ***\n"
              "  It executes the design, which no model before it did. It does\n"
              "  not close timing, drive a physical DDR, or sit in a socket.\n"
              "  The phase 7 exit gate is unchanged.\n\n");

  check(grxSetDevice(npu) == grxSuccess, "grxSetDevice");

  grxblasHandle_t h = nullptr;
  check(grxblasCreate(&h) == GRXBLAS_STATUS_SUCCESS, "grxblasCreate");

  int failures = 0;

  // The caller's dimensions. npu_gemm_path swaps the operands to turn this
  // column-major call into the row-major one the engine computes, so the
  // caller's n is bounded by MAX_M (8) and the caller's m by MAX_N (12).
  section("INT8 GEMM through the whole stack, against the RTL");
  failures += !run_case(h, 2, 2, 2, "2x2x2");
  failures += !run_case(h, 4, 4, 4, "4x4x4");
  failures += !run_case(h, 8, 8, 8, "8x8x8");
  failures += !run_case(h, 1, 1, 1, "1x1x1");
  failures += !run_case(h, 1, 4, 4, "m=1");
  failures += !run_case(h, 4, 1, 4, "n=1");
  failures += !run_case(h, 4, 4, 1, "k=1");
  failures += !run_case(h, 4, 4, 16, "k=MAX_K");

  section("the crossed bounds, on the RTL rather than on our arithmetic");
  failures += !run_case(h, 12, 8, 16, "m=MAX_N, n=MAX_M, k=MAX_K (largest legal)");
  {
    // One past the crossed bound: the caller's n = 9 makes the engine's M = 9,
    // which exceeds MAX_M = 8. Our library refuses before the launch; this
    // checks that the refusal and the RTL agree about where the wall is.
    const float alpha = 1.0f, beta = 0.0f;
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    grxMalloc(&dA, 4 * 4); grxMalloc(&dB, 4 * 9);
    grxMalloc(&dC, (size_t)4 * 9 * sizeof(int32_t));
    const grxblasStatus_t s =
        grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, 4, 9, 4, &alpha,
                      dA, GRX_R_8I, 4, dB, GRX_R_8I, 4, &beta,
                      dC, GRX_R_32I, 4);
    check(s == GRXBLAS_STATUS_NOT_SUPPORTED,
          "n=9 is refused: the engine's M would be 9 > MAX_M=8");
    grxFree(dA); grxFree(dB); grxFree(dC);
  }

  section("what the RTL's counters actually say");
  std::printf("  note  after the last GEMM, read straight from the RTL:\n");
  std::printf("        CYCLE_LO 0x24 = %-8u  OP_COUNT 0x2c = %u\n",
              grxtest::rtl().csr_read(0x24), grxtest::rtl().csr_read(0x2c));
  std::printf("        STALL    0x30 = %-8u  DMA_CT   0x34 = %u\n",
              grxtest::rtl().csr_read(0x30), grxtest::rtl().csr_read(0x34));
  check(grxtest::rtl().csr_read(0x28) == 0,
        "0x28 reads 0 -- ADDR_CYCLE_HI is dead in the RTL, confirmed here "
        "rather than taken from the header");
  check(grxtest::rtl().csr_read(0x24) != 0, "CYCLE_LO counted something");
  // Not a gate. The software shim reports 22 cycles for the 4x4x8 shape where
  // the RTL reports 224, OP_COUNT 256 where the RTL says 1280, STALL 0 where
  // the RTL says 64, and DMA_CT 22 where the RTL says 320 -- and the RTL's
  // DMA_CT exceeds its own CYCLE_LO, so the two count different windows. The
  // shim's counters are a model of a model; nothing should quote them as the
  // c930's. Recorded in cuda_mapping.md 7.34.

  check(grxblasDestroy(h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  if (failures) std::printf("\n%d case(s) FAILED\n", failures);
  return grxtest::report();
}
