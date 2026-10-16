// The first board's PTA as a device the RUNTIME enumerates: a module on the
// host, with no GPU behind it.
//
// docs/designs/pta_board_transport.md, and the board plan's B24 and B25. On the
// first board the PTA is a CXL Type-1 device of its own on the GRX930 host, so
// its operands are host memory and nothing is between them and it. This is the
// device that board has: parentDevice -1, a queue in host memory, and
// grxMallocHost on it handing out memory its queue reaches. A GEMM on three
// such buffers runs in place; on any other host memory it is copied through
// the driver's own. This file is the reporting, the refusals, the memory and
// both ways a GEMM can go. The GEMM held bit for bit to the model, through the
// same device, is tests/libs/test_grxblas_pta_chiplet.cpp run with "module".
//
// Run with a tile, and with "nomemory" as a second word: then the module is
// attached with no host memory to reach, and it must not be enumerated at all,
// because without a queue a PTA on the host has no way to be given work.
//
// A MODEL IS NOT HARDWARE. The twin is behind the module, the device says it
// is a model, and a pass is a statement about the runtime's path to it.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef GRXCP_ENABLE_PTA
#include "pta_twin_adapter.h"
#endif

using grxtest::check;
using grxtest::section;

#ifdef GRXCP_ENABLE_PTA
namespace {

const int32_t kSentinel = (int32_t)0x5A5A5A5A;

// C = A . B, column-major, as grxblasGemmEx is asked for it below.
std::vector<int32_t> exact(int m, int n, int k, const int8_t* A, const int8_t* B) {
  std::vector<int32_t> c((size_t)m * n, 0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      int32_t s = 0;
      for (int l = 0; l < k; ++l) s += (int32_t)A[i + (size_t)m * l] * (int32_t)B[l + (size_t)k * j];
      c[i + (size_t)m * j] = s;
    }
  return c;
}

void fill(int8_t* p, size_t n, uint32_t seed) {
  uint32_t s = seed * 2654435761u + 7u;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    p[i] = (int8_t)((int)((s >> 8) % 255u) - 127);
  }
}

bool same(const int32_t* c, const std::vector<int32_t>& want) {
  for (size_t i = 0; i < want.size(); ++i) if (c[i] != want[i]) return false;
  return true;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#ifndef GRXCP_ENABLE_PTA
  (void)argc; (void)argv;
  std::printf("built without GRXCP_ENABLE_PTA; there is no PTA backend in this build. skipping\n");
  return 77;
#else
  pta_twin_build build = grxtest::pta_twin_working_tile;
  bool nomemory = false, ok = argc <= 3;
  if (ok && argc >= 2) {
    if (std::strcmp(argv[1], "256x64") == 0) build = grxtest::pta_twin_first_tile;
    else ok = std::strcmp(argv[1], "128x64") == 0;
  }
  if (ok && argc == 3) {
    nomemory = std::strcmp(argv[2], "nomemory") == 0;
    ok = nomemory;
  }
  if (!ok) {
    std::printf("usage: %s [128x64 | 256x64] [nomemory]\n", argv[0]);
    return 2;
  }

  // BEFORE THE FIRST grx CALL: enumeration runs once.
  const bool installed = grxtest::pta_twin_install_module(build, /*with_memory=*/!nomemory);
  section("the seam");
  check(installed, nomemory ? "the module is installed on the host, with no host memory given it"
                            : "the module is installed on the host, with host memory and no link");
  if (!installed) return grxtest::report();
  pta_twin* twin = grxtest::pta_twin_installed();

  int count = 0;
  GRX_REQUIRE(grxGetDeviceCount(&count), "grxGetDeviceCount");
  int pta = -1, ptas = 0, gpus = 0;
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) != grxSuccess) continue;
    if (p.deviceType == GRX_DEVICE_TYPE_PTA) { pta = i; ++ptas; }
    if (p.deviceType == GRX_DEVICE_TYPE_GPU) ++gpus;
  }

  if (nomemory) {
    section("a module on the host with nothing for its queue to reach");
    check(ptas == 0 && gpus >= 1,
          "is not enumerated: with no queue a PTA on the host has no way to be given work");
    return grxtest::report();
  }

  // ---- enumeration ----------------------------------------------------------
  section("enumeration");
  check(ptas == 1 && gpus >= 1 && pta == count - 1, "one PTA, beside the GPUs and after them");
  if (pta < 0) return grxtest::report();
  grxDeviceProp_t prop{};
  GRX_REQUIRE(grxGetDeviceProperties(&prop, pta), "grxGetDeviceProperties");
  std::printf("  note  %s\n", prop.name);
  check(prop.parentDevice == -1, "parentDevice is -1: no GPU behind it, and its operands are host memory");
  check(prop.backend == GRX_BACKEND_MODEL && std::strstr(prop.name, "module on the host") != nullptr &&
            std::strstr(prop.name, "NOT hardware") != nullptr,
        "a model, and its name says a module on the host and not hardware");
  check(prop.capabilities == GRX_CAP_GEMM && prop.totalGlobalMem == 0 && prop.multiProcessorCount == 0,
        "GEMM and nothing else: no device memory, no pipeline");
  {
    const grxAnalogGemm_t& a = prop.analogGemm;
    check(a.tileIsPresent == 1 && a.tileRows == build.rows && a.tileCols == build.cols && a.gemmIndex == 0,
          "its tile is the build's, and GEMM 0 is next");
    std::printf("  note  tile %dx%d, as the device reports it\n", a.tileRows, a.tileCols);
  }

  // ---- what it refuses, and what it does not -----------------------------------
  section("no device memory, and host memory its queue reaches");
  GRX_REQUIRE(grxSetDevice(pta), "grxSetDevice");
  {
    void* p = nullptr;
    check(grxMalloc(&p, 256) == grxErrorNotSupported && p == nullptr, "grxMalloc: it has no memory of its own");
    check(grxMallocManaged(&p, 256, 0) == grxErrorNotSupported, "grxMallocManaged");
    grxStream_t s = nullptr;
    check(grxStreamCreate(&s) == grxErrorNotSupported, "grxStreamCreate: no streams");
    void* h = nullptr;
    check(grxMallocHost(&h, 100) == grxSuccess && h != nullptr && ((uintptr_t)h % 4096) == 0,
          "grxMallocHost: host memory its queue reaches, on a page");
    std::memset(h, 0x3C, 100);
    check(static_cast<uint8_t*>(h)[99] == 0x3C, "and the host reads and writes it as its own");
    check(grxFreeHost(h) == grxSuccess, "grxFreeHost gives it back");
    check(grxFreeHost(h) == grxErrorInvalidValue, "and a second free of it is refused");
  }

  // ---- a GEMM, in place and staged ---------------------------------------------
  section("a GEMM: in place on its own memory, copied from any other");
  grxblasHandle_t bh = nullptr;
  check(grxblasCreate(&bh) == GRXBLAS_STATUS_SUCCESS, "grxblasCreate");
  {
    grxblasEngine_t engine = GRXBLAS_ENGINE_NONE;
    int device = -1;
    grxblasGetGemmEngine(bh, 8, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I, &engine, &device);
    check(engine == GRXBLAS_ENGINE_PTA_CHIPLET && device == pta, "an int8 GEMM here goes to the PTA");
  }
  const int m = 6, n = 5, k = 9;
  const float one = 1.0f, zero = 0.0f;
  void *hA = nullptr, *hB = nullptr, *hC = nullptr;
  GRX_REQUIRE(grxMallocHost(&hA, (size_t)m * k), "grxMallocHost A");
  GRX_REQUIRE(grxMallocHost(&hB, (size_t)k * n), "grxMallocHost B");
  GRX_REQUIRE(grxMallocHost(&hC, (size_t)m * n * 4 + 64), "grxMallocHost C");
  int8_t* A = static_cast<int8_t*>(hA);
  int8_t* B = static_cast<int8_t*>(hB);
  int32_t* C = static_cast<int32_t*>(hC);
  fill(A, (size_t)m * k, 1);
  fill(B, (size_t)k * n, 2);
  const std::vector<int32_t> want = exact(m, n, k, A, B);

  for (int i = 0; i < m * n; ++i) C[i] = kSentinel;
  grxtest::pta_host_lines().clear();
  grxblasStatus_t s = grxblasGemmEx(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one, A, GRX_R_8I, m,
                                    B, GRX_R_8I, k, &zero, C, GRX_R_32I, m);
  check(s == GRXBLAS_STATUS_SUCCESS && same(C, want), "on grxMallocHost's memory: the exact product");
  check(grxtest::pta_host_lines_in(A, (size_t)m * k, false) > 0 &&
            grxtest::pta_host_lines_in(B, (size_t)k * n, false) > 0 &&
            grxtest::pta_host_lines_in(C, (size_t)m * n * 4, true) > 0,
        "in place: the module read A and B and wrote C where the caller put them");
  GRX_REQUIRE(grxGetDeviceProperties(&prop, pta), "grxGetDeviceProperties");
  check(prop.analogGemm.gemmIndex == 1, "and it took an index");

  // C a word into its buffer: off a line, so the module cannot write it whole.
  int32_t* C4 = C + 1;
  for (int i = 0; i < m * n; ++i) C4[i] = kSentinel;
  grxtest::pta_host_lines().clear();
  s = grxblasGemmEx(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one, A, GRX_R_8I, m,
                    B, GRX_R_8I, k, &zero, C4, GRX_R_32I, m);
  check(s == GRXBLAS_STATUS_SUCCESS && same(C4, want) &&
            grxtest::pta_host_lines_in(C, (size_t)m * n * 4 + 64, true) == 0,
        "C off a line: copied instead, the same product, and the module never wrote the caller's C");

  std::vector<int8_t> pA(A, A + (size_t)m * k), pB(B, B + (size_t)k * n);
  std::vector<int32_t> pC((size_t)m * n, kSentinel);
  grxtest::pta_host_lines().clear();
  s = grxblasGemmEx(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one, pA.data(), GRX_R_8I, m,
                    pB.data(), GRX_R_8I, k, &zero, pC.data(), GRX_R_32I, m);
  check(s == GRXBLAS_STATUS_SUCCESS && same(pC.data(), want) &&
            grxtest::pta_host_lines_in(pA.data(), pA.size(), false) == 0,
        "plain host memory: copied through the driver's own, the same product");

  void* gpu = nullptr;
  GRX_REQUIRE(grxSetDevice(0), "grxSetDevice(0)");
  GRX_REQUIRE(grxMalloc(&gpu, (size_t)m * k), "grxMalloc on the GPU");
  GRX_REQUIRE(grxSetDevice(pta), "grxSetDevice(pta)");
  std::vector<int32_t> gC((size_t)m * n, kSentinel);
  s = grxblasGemmEx(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one, gpu, GRX_R_8I, m,
                    pB.data(), GRX_R_8I, k, &zero, gC.data(), GRX_R_32I, m);
  bool untouched = true;
  for (int32_t v : gC) untouched = untouched && v == kSentinel;
  check(s == GRXBLAS_STATUS_INVALID_VALUE && untouched,
        "a GPU's pointer: refused by name, and C untouched; no GPU is behind this device");

  pta_twin_write32(twin, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_MZM_NL);
  for (int i = 0; i < m * n; ++i) C[i] = kSentinel;
  s = grxblasGemmEx(bh, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one, A, GRX_R_8I, m,
                    B, GRX_R_8I, k, &zero, C, GRX_R_32I, m);
  untouched = true;
  for (int i = 0; i < m * n; ++i) untouched = untouched && C[i] == kSentinel;
  check(s == GRXBLAS_STATUS_EXECUTION_FAILED && untouched,
        "a refusal in place: the GEMM fails, and the caller's C is as it was");
  pta_twin_write32(twin, PTA_TWIN_IMPAIR, 0);

  grxSetDevice(0);
  grxFree(gpu);
  grxSetDevice(pta);
  grxFreeHost(hA); grxFreeHost(hB); grxFreeHost(hC);
  check(grxblasDestroy(bh) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  grxtest::pta_transport_note(grxtest::PtaTransport::Module);
  return grxtest::report();
#endif
}
