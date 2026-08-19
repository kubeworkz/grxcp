// Phase 0 gate: assert that grxGetDeviceProperties reports a self-consistent,
// sourced device description.
//
// Several of these are CONTRACT tests rather than behavior tests -- they fail
// when someone clears an honesty flag without removing the emulation it
// describes (AGENTS.md section 3). If one of those fires because the underlying
// gap was genuinely closed, the fix is to delete the assertion in the same
// change that deletes the emulation, never to relax it on its own.

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdio>
#include <cstdlib>

using grxtest::check;

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; cannot run device property tests\n");
    return 77;  // skip
  }

  for (int i = 0; i < count; ++i) {
    grxDeviceProp_t p{};
    if (grxGetDeviceProperties(&p, i) != grxSuccess) {
      std::printf("device %d: query failed\n", i);
      return 1;
    }
    std::printf("device %d (%s):\n", i, p.name);

    // --- sourced values must be present ---
    check(p.warpSize > 0,                    "warpSize > 0");
    check(p.maxWarpsPerMultiProcessor > 0,   "warps per SM > 0");
    check(p.multiProcessorCount > 0,         "SM count > 0");
    check(p.clusterCount > 0,                "cluster count > 0");
    check(p.totalGlobalMem > 0,              "global memory > 0");
    check(p.cacheLineSize > 0,               "cache line size > 0");
    check(p.name[0] != '\0',                 "device name is set");

    // --- derived values must follow their documented formula ---
    // A CTA occupies one dispatcher slot and cannot exceed a core's warp
    // capacity (cta_clustering_and_dispatch.md 3.1).
    check(p.maxThreadsPerBlock == p.warpSize * p.maxWarpsPerMultiProcessor,
          "maxThreadsPerBlock == warpSize * warpsPerSM");
    check(p.sharedMemPerBlock <= p.sharedMemPerMultiprocessor,
          "shared per block <= shared per SM");
    check(p.maxThreadsDim[0] == p.maxThreadsPerBlock &&
          p.maxThreadsDim[1] == p.maxThreadsPerBlock &&
          p.maxThreadsDim[2] == p.maxThreadsPerBlock,
          "per-dimension thread limits match the block limit");

    // --- unknowns must be sentinels, never plausible numbers ---
    check(p.numBarriers == -1,
          "barrier count reports unknown (no host capability ID exists)");

    // --- capability profile must agree with the rest of the record ---
    check(!p.managedMemory || p.unifiedAddressing,
          "managed memory implies unified addressing");
    check(((p.capabilities & GRX_CAP_UNIFIED_ADDRESSING) != 0) ==
              (p.unifiedAddressing != 0),
          "UVA capability bit agrees with unifiedAddressing");
    check((p.capabilities & GRX_CAP_KERNEL_LAUNCH) != 0,
          "GPU device advertises kernel launch");
    check(!(p.capabilities & GRX_CAP_TENSOR_CORE) ||
           (p.capabilities & GRX_CAP_GEMM),
          "tensor core implies GEMM capability");

    // --- managed memory must be gated by backend, not just by VM support ---
    // VM silently no-ops on the FPGA paths (command_processor.md 10, item 2).
    const bool fpga = (p.backend == GRX_BACKEND_XRT || p.backend == GRX_BACKEND_OPAE);
    check(!fpga || !p.managedMemory,
          "managed memory is off on FPGA backends");

    // --- honesty flags: see the note at the top of this file ---
    // Was `== 1` while the shuffle was staged through local memory. The ISA
    // has SHFL.* and VOTE.*, grx_warp.h issues them, and tests/kernels/warp/
    // checks the semantics on a real device -- so the flag says native, and
    // this says it must keep saying so. Flipping it back without restoring an
    // emulation is the defect this catches.
    check(p.warpShuffleIsEmulated == 0,
          "warp shuffle reports native, because the ISA has SHFL.*");
    check(p.eventTimingIsDeviceSide == 0,
          "event timing still reports host clock (CP profiling writeback pending)");
    check(p.constantMemoryIsGlobal == 1,
          "__constant__ still reports read-only global lowering");
  }

  // --- error surface ---
  std::printf("error surface:\n");
  check(grxGetErrorName(grxSuccess) != nullptr &&
        grxGetErrorString(grxErrorNotSupported) != nullptr,
        "error name and string lookups return text");
  check(grxGetDeviceProperties(nullptr, 0) == grxErrorInvalidValue,
        "null property pointer is rejected");
  check(grxGetLastError() == grxErrorInvalidValue,
        "grxGetLastError reports the sticky error");
  check(grxGetLastError() == grxSuccess,
        "grxGetLastError clears the sticky error");
  {
    grxDeviceProp_t p{};
    check(grxGetDeviceProperties(&p, 9999) == grxErrorInvalidDevice,
          "out-of-range device ordinal is rejected");
    check(grxPeekAtLastError() == grxErrorInvalidDevice,
          "grxPeekAtLastError does not clear");
    check(grxGetLastError() == grxErrorInvalidDevice,
          "sticky error survives until taken");
  }

  return grxtest::report();
}
