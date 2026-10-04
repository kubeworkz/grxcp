// The PTA chiplet as a device the RUNTIME enumerates, through the digital twin.
//
// The board plan's S4: the chiplet reported through the device property, from
// the twin now and from silicon later. This file is the reporting half and the
// refusals. The GEMM, held bit for bit to the model, is
// tests/libs/test_grxblas_pta_chiplet.cpp.
//
// THE CHIPLET IS INSTALLED HERE WITH A WINDOW AND NO LINK. That is a real
// state and the one every chiplet is in off a model: its registers can be read
// and nothing can be handed to it. So this file also holds the refusal that
// state has to produce -- a GEMM that returns "not supported" and leaves C
// alone, rather than a success over a product nobody computed.
//
// WHAT A PTA DEVICE IS, as the device table has it (board plan, B9): a device
// of its own, GEMM-only, with no memory. Its operands are its parent GPU's.
// Every call below that is not a property read or a GEMM is refused, and each
// refusal is checked for being "not supported" and not something that blames
// the caller's argument.
//
// A MODEL IS NOT HARDWARE. The device says so in two fields, and both are
// checked.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"

#include <cstdio>
#include <cstring>
#include <vector>

#ifdef GRXCP_ENABLE_PTA
#include "pta_twin_adapter.h"
#endif

using grxtest::check;
using grxtest::section;

int main() {
#ifndef GRXCP_ENABLE_PTA
  std::printf("built without GRXCP_ENABLE_PTA; there is no PTA chiplet backend in "
              "this build. skipping\n");
  return 77;
#else
  // One of the link model's candidate geometries for the chiplet.
  const pta_twin_build build = {256, 64, 8, 48, 4};

  // BEFORE THE FIRST grx CALL: enumeration runs once.
  const bool installed = grxtest::pta_twin_install(build, /*parent=*/0, /*with_link=*/false);
  section("the seam");
  check(installed, "the twin's window is installed, and no link");
  check(grxcp_pta_model_is_attached() == 1, "and the seam says a model is attached");
  if (!installed) return grxtest::report();
  pta_twin* twin = grxtest::pta_twin_installed();

  int count = 0;
  GRX_REQUIRE(grxGetDeviceCount(&count), "grxGetDeviceCount");
  check(grxcp_pta_attach_model_for_testing(nullptr, nullptr, nullptr) == 0 &&
        grxcp_pta_attach_link_for_testing(nullptr, nullptr) == 0 &&
        grxcp_pta_set_parent_for_testing(0) == 0,
        "after enumeration the seam refuses: there is nothing left to attach to");

  // ---- enumeration ----------------------------------------------------------
  section("enumeration");
  int pta = -1, gpus = 0, ptas = 0;
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) != grxSuccess) continue;
    if (p.deviceType == GRX_DEVICE_TYPE_PTA) { pta = i; ++ptas; }
    if (p.deviceType == GRX_DEVICE_TYPE_GPU) ++gpus;
  }
  check(ptas == 1 && gpus >= 1, "one PTA chiplet, beside the GPUs");
  if (pta < 0) return grxtest::report();
  check(pta == count - 1, "appended after them, so no existing device changes index");

  grxDeviceProp_t prop{};
  GRX_REQUIRE(grxGetDeviceProperties(&prop, pta), "grxGetDeviceProperties");
  std::printf("  note  %s\n", prop.name);
  check(prop.backend == GRX_BACKEND_MODEL, "its backend is a model");
  check(std::strstr(prop.name, "NOT hardware") != nullptr, "and its name says it is not hardware");
  check(prop.capabilities == GRX_CAP_GEMM, "its profile is GEMM and nothing else");
  check(prop.totalGlobalMem == 0 && prop.maxThreadsPerBlock == 0 && prop.multiProcessorCount == 0,
        "no memory and no pipeline");

  // ---- whose memory ---------------------------------------------------------
  section("whose memory its operands are in");
  check(prop.parentDevice == 0, "parentDevice names device 0");
  {
    grxDeviceProp_t parent{};
    grxGetDeviceProperties(&parent, prop.parentDevice);
    check(parent.deviceType == GRX_DEVICE_TYPE_GPU && parent.parentDevice == -1,
          "which is a GPU, with memory of its own and no parent");
    check(parent.analogGemm.tileIsPresent == 0 && parent.analogGemm.gemmIndex == -1,
          "and whose own GEMMs are still reported native: the tile is the chiplet's, not the GPU's");
  }

  // ---- the property ---------------------------------------------------------
  section("the property, from the chiplet's registers");
  {
    const grxAnalogGemm_t& a = prop.analogGemm;
    check(a.tileIsPresent == 1 && a.gemmIsAnalogEmulated == 0,
          "a tile is present, and with PTA_IMPAIR clear its GEMMs are exact");
    check(a.tileRows == 256 && a.tileCols == 64 && a.operandBits == 8 && a.accumulatorBits == 48,
          "the tile is the build's: 256 x 64, 8-bit operands, 48-bit sums");
    check(a.impairmentsImplemented == 0x5F, "it implements all but MZM_NL");
    check(a.gemmIndex == 0, "GEMM 0 is next: the index is reported with nothing impaired");
    check(a.seed == -1 && a.thermalSigmaQ8 == -1 && a.activationBits == -1,
          "and no configuration is reported for a model that is not running");
  }
  // Written behind the runtime's back, the way another holder of the chiplet
  // would. The property is read from the device on every call.
  pta_twin_write32(twin, PTA_TWIN_SEED, 0x2A);
  pta_twin_write32(twin, PTA_TWIN_IMPAIR, PTA_QUANT | PTA_THERMAL);
  pta_twin_write32(twin, PTA_TWIN_BITS, 6u | (6u << 4) | (7u << 8) | (16u << 12));
  pta_twin_write32(twin, PTA_TWIN_SIGMA_TH, 0x0040);
  grxGetDeviceProperties(&prop, pta);
  {
    const grxAnalogGemm_t& a = prop.analogGemm;
    check(a.gemmIsAnalogEmulated == 1 && a.impairments == (PTA_QUANT | PTA_THERMAL) &&
          a.activationBits == 6 && a.weightBits == 6 && a.adcBits == 7 && a.adcShift == 16 &&
          a.seed == 0x2A && a.thermalSigmaQ8 == 0x40,
          "impairments switched on in the registers are in the next read: it is live");
    check(a.gemmIndex == 0, "and the seed's write restarted the count");
  }

  // ---- what it refuses --------------------------------------------------------
  section("everything that is not a GEMM is refused, as not supported");
  GRX_REQUIRE(grxSetDevice(pta), "grxSetDevice");
  {
    void* p = nullptr;
    check(grxMalloc(&p, 256) == grxErrorNotSupported && p == nullptr,
          "grxMalloc: the chiplet has no memory");
    check(grxMallocManaged(&p, 256, 0) == grxErrorNotSupported, "grxMallocManaged");
    check(grxMallocHost(&p, 256) == grxErrorNotSupported, "grxMallocHost");
    size_t free_bytes = 1, total_bytes = 1;
    check(grxMemGetInfo(&free_bytes, &total_bytes) == grxSuccess && free_bytes == 0 && total_bytes == 0,
          "grxMemGetInfo answers: none free, of none");
    grxStream_t s = nullptr;
    check(grxStreamCreate(&s) == grxErrorNotSupported, "grxStreamCreate: no streams");
    grxEvent_t e = nullptr;
    check(grxEventCreate(&e) == grxErrorNotSupported, "grxEventCreate: no events");
    grxModule_t m = nullptr;
    const unsigned char image[16] = {0};
    check(grxModuleLoadData(&m, image, sizeof image) == grxErrorNotSupported,
          "grxModuleLoadData: no pipeline to load a kernel for");
    check(grxDeviceSynchronize() == grxSuccess, "grxDeviceSynchronize has nothing to wait for");
  }

  // ---- a parent's pointer is not this device's --------------------------------
  section("a parent's pointer, outside a GEMM");
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  const int m = 4, n = 3, k = 5;
  GRX_REQUIRE(grxSetDevice(prop.parentDevice), "grxSetDevice(parent)");
  GRX_REQUIRE(grxMalloc(&dA, (size_t)m * k), "grxMalloc A on the parent");
  GRX_REQUIRE(grxMalloc(&dB, (size_t)k * n), "grxMalloc B on the parent");
  GRX_REQUIRE(grxMalloc(&dC, (size_t)m * n * sizeof(int32_t)), "grxMalloc C on the parent");
  std::vector<int8_t> hA((size_t)m * k, 3), hB((size_t)k * n, -2);
  std::vector<int32_t> sentinel((size_t)m * n, (int32_t)0x5A5A5A5A), back((size_t)m * n, 0);
  grxMemcpy(dA, hA.data(), hA.size(), grxMemcpyDefault);
  grxMemcpy(dB, hB.data(), hB.size(), grxMemcpyDefault);
  grxMemcpy(dC, sentinel.data(), sentinel.size() * sizeof(int32_t), grxMemcpyDefault);
  GRX_REQUIRE(grxSetDevice(pta), "grxSetDevice(pta)");
  check(grxMemcpy(back.data(), dC, back.size() * sizeof(int32_t), grxMemcpyDefault) ==
            grxErrorInvalidDevicePointer,
        "grxMemcpy on the chiplet refuses it: the exception is the GEMM's alone");

  // ---- a chiplet with no link ---------------------------------------------------
  section("a window and no link");
  grxblasHandle_t h = nullptr;
  check(grxblasCreate(&h) == GRXBLAS_STATUS_SUCCESS, "grxblasCreate");
  {
    grxblasEngine_t engine = GRXBLAS_ENGINE_NONE;
    int device = -1;
    grxblasGetGemmEngine(h, m, n, k, GRX_R_8I, GRX_R_8I, GRX_R_32I, &engine, &device);
    check(engine == GRXBLAS_ENGINE_PTA_CHIPLET && device == pta,
          "an int8 GEMM here is routed to the PTA chiplet, on this device");
    grxblasGetGemmEngine(h, m, n, k, GRX_R_16F, GRX_R_16F, GRX_R_32F, &engine, &device);
    check(engine == GRXBLAS_ENGINE_NONE,
          "and an fp16 one has nowhere to go: the GPU is not an answer about this device");
  }
  const float one = 1.0f, zero = 0.0f;
  const grxblasStatus_t s = grxblasGemmEx(h, GRXBLAS_OP_N, GRXBLAS_OP_N, m, n, k, &one,
                                          dA, GRX_R_8I, m, dB, GRX_R_8I, k, &zero,
                                          dC, GRX_R_32I, m);
  check(s == GRXBLAS_STATUS_NOT_SUPPORTED,
        "the GEMM is refused as not supported: there is no way to hand the chiplet work");
  GRX_REQUIRE(grxSetDevice(prop.parentDevice), "grxSetDevice(parent)");
  grxMemcpy(back.data(), dC, back.size() * sizeof(int32_t), grxMemcpyDefault);
  check(back == sentinel, "and C is exactly as the caller left it");
  grxGetDeviceProperties(&prop, pta);
  check(prop.analogGemm.gemmIndex == 0, "no GEMM started, so none was counted");

  grxFree(dA); grxFree(dB); grxFree(dC);
  check(grxblasDestroy(h) == GRXBLAS_STATUS_SUCCESS, "grxblasDestroy");
  return grxtest::report();
#endif
}
