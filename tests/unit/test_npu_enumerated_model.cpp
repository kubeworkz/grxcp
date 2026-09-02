// The NPU device the RUNTIME enumerates, driven through a register model.
//
// Every other NPU test in this tree talks to the backend directly: construct an
// npu_c930_device_t, attach a model, call npu_c930_detect and npu_c930_gemm on
// it. That covers the backend's decisions well and covers the runtime not at
// all -- probe_npu_device built its own handle inside a std::call_once and gave
// it to nobody, so grxSetDevice, grxGetDeviceProperties, grxMalloc and
// grxblasGemmEx had never once run against an NPU on a machine without a c930.
//
// tests/libs/test_grxblas_npu.cpp is the evidence. It has been compiled into
// every NPU build since it was written and has never executed a single case: no
// device reports GRX_DEVICE_TYPE_NPU, so it exits 77. A skip that reads like
// absent hardware, standing in front of a path that was not reachable at all.
//
// This file is the first thing to go through the front door.
//
// A MODEL IS NOT HARDWARE. The point of the seam is not only that a model can
// be attached, but that a device reached through one CANNOT claim to be
// silicon: the backend field is derived from the same variable probe_npu_device
// detects through. That is what the middle of this file checks, and it is the
// half that was wrong -- populate_npu_properties used to assign
// GRX_BACKEND_SILICON unconditionally, with the comment "NPU is always real
// hardware", which was a claim made by a function with no way to know.

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdio>
#include <cstring>

#ifdef GRXCP_ENABLE_NPU
#include "npu_c930_testing.h"
#endif

using grxtest::check;
using grxtest::section;

#ifdef GRXCP_ENABLE_NPU
namespace {

// A register file, and nothing more. Enough for the write-readback probe in
// npu_c930_detect to find it, which is all this file needs -- the GEMM path is
// covered against the GRX930 team's own model in test_npu_c930_shim.cc.
struct Regs {
  uint32_t r[16] = {0};
  int      reads = 0;
  int      writes = 0;
};

uint32_t regs_read(void* ctx, uint32_t off) {
  Regs* g = static_cast<Regs*>(ctx);
  ++g->reads;
  return g->r[(off >> 2) & 0xF];
}

void regs_write(void* ctx, uint32_t off, uint32_t v) {
  Regs* g = static_cast<Regs*>(ctx);
  ++g->writes;
  if (off == 0x00 && (v & NPU_C930_CTRL_START)) {
    g->r[1] = NPU_C930_STATUS_DONE;
    return;
  }
  g->r[(off >> 2) & 0xF] = v;
}

Regs g_regs;

}  // namespace
#endif

int main() {
#ifndef GRXCP_ENABLE_NPU
  std::printf("built without GRXCP_ENABLE_NPU; there is no NPU backend in this "
              "binary. skipping\n");
  return 77;
#else
  // BEFORE ANY grx CALL. Enumeration runs once behind a std::call_once, and a
  // model installed after it has nothing to attach to.
  const int installed =
      grxcp_npu_attach_model_for_testing(regs_read, regs_write, &g_regs);

  section("the seam installs before enumeration");
  check(installed == 1, "the model was accepted");
  check(grxcp_npu_model_is_attached() == 1, "and the runtime knows it is there");

  int count = 0;
  check(grxGetDeviceCount(&count) == grxSuccess, "grxGetDeviceCount");
  std::printf("  note  %d device(s) enumerated\n", count);

  int npu = -1;
  grxDeviceProp_t prop{};
  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) != grxSuccess) continue;
    if (p.deviceType == GRX_DEVICE_TYPE_NPU) { npu = i; prop = p; break; }
  }

  section("the model is enumerated as a device");
  check(npu >= 0, "an NPU device appears in the device table");
  if (npu < 0) {
    std::printf("        Without this the rest of the file is meaningless, and\n"
                "        every NPU gate above the backend is unreachable.\n");
    return grxtest::report();
  }
  check(g_regs.reads > 0 && g_regs.writes > 0,
        "detection actually went through the model's hooks");

  section("and it does not claim to be hardware");
  std::printf("  note  device %d: \"%s\", backend %d\n", npu, prop.name,
              (int)prop.backend);
  check(prop.backend == GRX_BACKEND_MODEL,
        "backend is GRX_BACKEND_MODEL");
  check(prop.backend != GRX_BACKEND_SILICON,
        "and specifically NOT silicon");
  check(std::strstr(prop.name, "NOT hardware") != nullptr,
        "the device name says so too, where grx-smi will print it");
  check(prop.managedMemory == 0,
        "no managed memory: a model has no MMU either");

  section("the seam refuses after enumeration");
  // Attaching now would leave populate_npu_properties reporting a model for a
  // device that was found some other way -- the exact fabrication the derived
  // backend field exists to prevent. Refusing is the only honest answer.
  check(grxcp_npu_attach_model_for_testing(regs_read, regs_write, &g_regs) == 0,
        "a second attach, after enumeration, is refused");

  section("selecting it works");
  check(grxSetDevice(npu) == grxSuccess, "grxSetDevice on the NPU device");
  int current = -1;
  check(grxGetDevice(&current) == grxSuccess && current == npu,
        "and it is the current device");

  // NOT A GATE YET. grxMalloc has no NPU branch: allocate_device calls
  // vx_buffer_create(d->handle, ...) and an NPU device's handle is null, so the
  // request goes to the Vortex driver and comes back VX_ERR_INVALID_VALUE --
  // reported to the caller as "invalid value", blaming the size argument for a
  // device that has no allocator at all (cuda_mapping.md 7.28).
  //
  // Printed rather than asserted, because a red gate on a hole we are about to
  // fill is noise. The gate arrives with the allocator.
  section("what an allocation on this device does today");
  void* p = nullptr;
  const grxError_t e = grxMalloc(&p, 256);
  std::printf("  note  grxMalloc(256) -> %s (%s)\n",
              grxGetErrorName(e), grxGetErrorString(e));
  std::printf("        totalGlobalMem is %zu bytes and nothing allocates from "
              "it.\n", prop.totalGlobalMem);
  if (e == grxSuccess) grxFree(p);

  return grxtest::report();
#endif
}
