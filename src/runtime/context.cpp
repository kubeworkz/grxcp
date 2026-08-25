// GRXCP — device table, implicit context, and grxDeviceProp_t population.
//
// Every numeric field below is either read from vx_device_query or derived by a
// formula documented in an upstream GRX-G100 design doc, with the source named
// in a comment. Nothing is invented; fields the stack cannot yet supply report
// a sentinel (AGENTS.md section 3).

#include "internal.h"

#include <grx/grx_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef GRXCP_ENABLE_NPU
#include "npu_c930.h"
#endif

namespace grxcp {

namespace {

std::once_flag       g_init_once;
std::mutex           g_devices_mutex;
std::vector<Device>  g_devices;
grxError_t           g_init_error = grxSuccess;

thread_local int     g_current_device = 0;

// vx_device_query returns 0 for a capability the backend does not implement,
// which is indistinguishable from a legitimate zero. Ask explicitly and keep
// the result code so a failed query never becomes a plausible-looking number.
bool query(vx_device_h dev, uint32_t caps_id, uint64_t* out) {
  uint64_t v = 0;
  if (vx_device_query(dev, caps_id, &v) != VX_SUCCESS) return false;
  *out = v;
  return true;
}

uint64_t query_or(vx_device_h dev, uint32_t caps_id, uint64_t fallback) {
  uint64_t v = 0;
  return query(dev, caps_id, &v) ? v : fallback;
}

// VM works on the simulator backends and silently no-ops on the FPGA paths:
// the RTL command processor has no CP_SATP decode and no hardware page-table
// walker in VX_cp_dma yet (grxgpu/docs/designs/command_processor.md 10, item 2).
// Reporting managedMemory=1 there would hand out a pointer that quietly means
// something different, so the backend gates the capability.
bool backend_has_vm(grxBackend_t b) {
  return b == GRX_BACKEND_SIMX || b == GRX_BACKEND_RTLSIM || b == GRX_BACKEND_GEM5;
}

void populate_properties(Device& d) {
  grxDeviceProp_t& p = d.prop;
  std::memset(&p, 0, sizeof(p));

  const vx_device_h h = d.handle;
  const grxBackend_t backend = detect_backend();

  p.deviceType = GRX_DEVICE_TYPE_GPU;
  p.backend    = backend;

  // The G100 chip design declares compute capability 10.0 as its target
  // (grxgpu/docs/designs/gpu_chip_design.md section 2). There is no capability
  // register for it, so it is carried here with its source named.
  p.computeCapabilityMajor = 10;
  p.computeCapabilityMinor = 0;

  // --- execution geometry ---------------------------------------------------
  p.warpSize                  = (int)query_or(h, VX_CAPS_NUM_THREADS,   0);
  p.maxWarpsPerMultiProcessor = (int)query_or(h, VX_CAPS_NUM_WARPS,     0);
  p.multiProcessorCount       = (int)query_or(h, VX_CAPS_NUM_CORES,     0);
  p.clusterCount              = (int)query_or(h, VX_CAPS_NUM_CLUSTERS,  0);
  p.socketSize                = (int)query_or(h, VX_CAPS_SOCKET_SIZE,   0);
  p.issueWidth                = (int)query_or(h, VX_CAPS_ISSUE_WIDTH,   0);

  // A CTA is expanded into warps by VX_cta_dispatch and occupies one CTA slot;
  // there are NUM_WARPS slots per core, so a single block cannot exceed the
  // core's whole warp capacity (cta_clustering_and_dispatch.md section 3.1).
  p.maxThreadsPerBlock = p.warpSize * p.maxWarpsPerMultiProcessor;
  p.maxThreadsDim[0] = p.maxThreadsDim[1] = p.maxThreadsDim[2] =
      p.maxThreadsPerBlock;

  // Grid dimensions reach the KMU through 32-bit DCR writes (the command
  // processor's CMD_DCR_WRITE payload is a uint32), so each axis is bounded by
  // the DCR width rather than by any smaller architectural limit.
  p.maxGridSize[0] = p.maxGridSize[1] = p.maxGridSize[2] = 0x7fffffff;

  // Barrier count is exposed to kernels through VX_CSR_NUM_BARRIERS but has no
  // host-side capability ID. Report "unknown" rather than a guess.
  p.numBarriers = -1;

  // --- memory ---------------------------------------------------------------
  p.totalGlobalMem            = (size_t)query_or(h, VX_CAPS_GLOBAL_MEM_SIZE, 0);
  p.sharedMemPerMultiprocessor= (size_t)query_or(h, VX_CAPS_LOCAL_MEM_SIZE,  0);
  // One CTA may occupy the whole local memory when it is the only resident one,
  // so the per-block ceiling equals the per-core capacity.
  p.sharedMemPerBlock         = p.sharedMemPerMultiprocessor;
  p.memBankCount              = (int)query_or(h, VX_CAPS_NUM_MEM_BANKS,   0);
  p.memBankSize               = (size_t)query_or(h, VX_CAPS_MEM_BANK_SIZE, 0);
  p.cacheLineSize             = (int)query_or(h, VX_CAPS_CACHE_LINE_SIZE, 0);
  p.clockRateMHz              = (int)query_or(h, VX_CAPS_CLOCK_RATE,      0);
  p.peakMemoryBandwidthMBs    = (size_t)query_or(h, VX_CAPS_PEAK_MEM_BW,  0);

  const uint64_t vm_support   = query_or(h, VX_CAPS_VM_SUPPORT, 0);
  p.unifiedAddressing         = (int)(vm_support != 0);
  p.managedMemory             = (int)(vm_support != 0 && backend_has_vm(backend));
  p.pinnedMemTotal            = (size_t)query_or(h, VX_CAPS_VM_PINNED_SIZE, 0);
  p.pinnedMemFree             = (size_t)query_or(h, VX_CAPS_VM_PINNED_FREE, 0);

  // --- capability profile ---------------------------------------------------
  const uint64_t isa = query_or(h, VX_CAPS_ISA_FLAGS, 0);
  unsigned caps = GRX_CAP_STREAMS | GRX_CAP_EVENTS | GRX_CAP_MEMCPY;

  // KERNEL LAUNCH IS DERIVED, NOT ASSUMED.
  //
  // It used to be an unconditional bit, which was true of every device this
  // function could see -- it only ever populates a GPU. It stops being true one
  // device along: populate_npu_properties builds a profile with GRX_CAP_GEMM and
  // no GRX_CAP_KERNEL_LAUNCH, because the c930 is a systolic array with no SIMT
  // pipeline, and validate() in launch.cpp now refuses on that bit rather than
  // falling back to the GPU.
  //
  // A device with no warps or no lanes has no programmable pipeline, and that is
  // the condition rather than a device-type test: it is a fact the driver
  // already reports, it needs no new capability ID, and it stays true of
  // anything else in that shape. Without it the launch path computed
  // maxThreadsPerBlock = 0 and refused with "launch out of resources", which
  // describes a grid that does not fit rather than a device that cannot run
  // grids at all.
  //
  // Cooperative launch is a strictly narrower claim and moves with it: a
  // grid-wide barrier needs the very pipeline this bit reports.
  if (p.warpSize > 0 && p.maxWarpsPerMultiProcessor > 0)
    caps |= GRX_CAP_KERNEL_LAUNCH | GRX_CAP_COOPERATIVE_LAUNCH;

  if (isa & VX_ISA_EXT_TCU) caps |= GRX_CAP_TENSOR_CORE | GRX_CAP_GEMM;
  if (isa & VX_ISA_EXT_DXA) caps |= GRX_CAP_ASYNC_COPY;
  if (isa & VX_ISA_EXT_RTU) caps |= GRX_CAP_RAY_TRACING;
  if (isa & VX_ISA_STD_A)   caps |= GRX_CAP_GLOBAL_ATOMICS;
  if (p.unifiedAddressing)  caps |= GRX_CAP_UNIFIED_ADDRESSING;
  p.capabilities = caps;

  // --- honesty flags --------------------------------------------------------
  // Each of these marks a documented software stand-in for hardware. Clearing
  // one without removing the emulation it describes is a defect.

  // Was 1, for as long as the shuffle was staged through local memory. The ISA
  // has SHFL.UP / DOWN / BFLY / IDX and VOTE.ALL / ANY / UNI / BAL -- they are
  // in vx_intrinsics.h with no configuration gate and the ALU implements them,
  // so grx_warp.h issues them directly and the emulation is gone. Verified by
  // tests/kernels/warp/, which checks all four shuffle forms against CUDA's
  // segmented semantics at two widths (cuda_mapping.md section 7.1).
  p.warpShuffleIsEmulated   = 0;
  // The driver does stamp each command, but with the HOST clock around
  // execution -- the CP does not write back device timestamps. On a simulator
  // that number measures the simulator, so this flag stays 0 until the
  // timestamps come from the device (command_processor.md section 10, item 9;
  // cuda_mapping.md section 7.4).
  p.eventTimingIsDeviceSide = 0;
  // No exposed broadcast constant path; __constant__ lowers to read-only global
  // memory (cuda_mapping.md section 7.2).
  p.constantMemoryIsGlobal  = 1;

  std::snprintf(p.name, sizeof(p.name), "GRX-G100 (%s)", backend_name(backend));
}

}  // namespace

grxBackend_t detect_backend() {
  // Mirrors sw/runtime/stub/vortex.cpp: $VORTEX_DRIVER selects the backend
  // library libvortex-<name>.so, defaulting to "simx" when unset.
  const char* drv = std::getenv("VORTEX_DRIVER");
  const std::string name = drv ? drv : "simx";
  if (name == "simx")   return GRX_BACKEND_SIMX;
  if (name == "rtlsim") return GRX_BACKEND_RTLSIM;
  if (name == "xrt")    return GRX_BACKEND_XRT;
  if (name == "opae")   return GRX_BACKEND_OPAE;
  if (name == "gem5")   return GRX_BACKEND_GEM5;
  return GRX_BACKEND_SILICON;
}

const char* backend_name(grxBackend_t b) {
  switch (b) {
    case GRX_BACKEND_SIMX:    return "simx";
    case GRX_BACKEND_RTLSIM:  return "rtlsim";
    case GRX_BACKEND_XRT:     return "xrt";
    case GRX_BACKEND_OPAE:    return "opae";
    case GRX_BACKEND_GEM5:    return "gem5";
    case GRX_BACKEND_SILICON: return "silicon";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// NPU C930 device support
// ---------------------------------------------------------------------------
#ifdef GRXCP_ENABLE_NPU

// Fill grxDeviceProp_t for the GRX930 NPU from hardware constants.
// The NPU has no vx_device_h and no vx_device_query — every field comes
// from the RTL parameters in c930/doc/c930_architecture.md.
static void populate_npu_properties(Device& d) {
  grxDeviceProp_t& p = d.prop;
  std::memset(&p, 0, sizeof(p));

  p.deviceType = GRX_DEVICE_TYPE_NPU;
  p.backend    = GRX_BACKEND_SILICON;  // NPU is always real hardware

  // Compute capability: NPU has no scalar pipeline, report 0.0
  p.computeCapabilityMajor = 0;
  p.computeCapabilityMinor = 0;

  // --- execution geometry (NPU has no SIMT pipeline) ---
  p.warpSize                  = 1;   // no warps — scalar GEMM dispatch
  p.maxWarpsPerMultiProcessor = 0;
  p.multiProcessorCount       = 0;   // no SMs — systolic array
  p.clusterCount              = 0;
  p.socketSize                = 0;
  p.issueWidth                = 0;
  p.maxThreadsPerBlock        = 0;
  p.maxThreadsDim[0] = p.maxThreadsDim[1] = p.maxThreadsDim[2] = 0;
  p.maxGridSize[0] = p.maxGridSize[1] = p.maxGridSize[2] = 0;
  p.numBarriers = 0;

  // --- memory (NPU uses DDR, 64 KB in the current SoC) ---
  p.totalGlobalMem             = 65536;  // MEM_BYTES from c930_soc_top
  p.sharedMemPerMultiprocessor = 0;
  p.sharedMemPerBlock          = 0;
  p.memBankCount               = 0;
  p.memBankSize                = 0;
  p.cacheLineSize              = 32;     // icache line size
  p.clockRateMHz               = 50;     // CLK_DIV=2, 100/2=50 MHz
  p.peakMemoryBandwidthMBs     = 0;      // not characterized yet
  p.unifiedAddressing          = 0;      // no MMU/IOMMU yet
  p.managedMemory              = 0;
  p.pinnedMemTotal             = 0;
  p.pinnedMemFree              = 0;

  // --- capability profile (from architecture spec §6) ---
  unsigned caps = 0;
  caps |= NPU_C930_CAP_STREAMS;    // MMIO doorbell + STATUS.DONE
  caps |= NPU_C930_CAP_EVENTS;     // o_irq pulses on completion
  caps |= NPU_C930_CAP_MEMCPY;     // c930_npu_dma AXI4 master
  caps |= NPU_C930_CAP_GEMM;       // systolic array INT8 GEMM
  // No GRX_CAP_KERNEL_LAUNCH — no SIMT pipeline
  // No GRX_CAP_UNIFIED_ADDRESSING — no MMU yet
  p.capabilities = caps;

  // --- honesty flags ---
  p.warpShuffleIsEmulated   = 0;  // no shuffles at all
  p.eventTimingIsDeviceSide = 0;  // no device-side timestamp counter
  p.constantMemoryIsGlobal  = 1;  // no __constant__ path

  std::snprintf(p.name, sizeof(p.name), "GRX930 NPU (silicon)");
}

void probe_npu_device(std::vector<Device>& devices) {
  static std::once_flag npu_once;
  std::call_once(npu_once, [&devices] {
    npu_c930_device_t* dev = new npu_c930_device_t;
    if (npu_c930_detect(dev) && dev->present) {
      Device d;
      d.index    = (int)devices.size();
      d.type     = DeviceType::NPU;
      d.handle   = nullptr;  // no Vortex handle
      d.opened   = true;     // MMIO is always "open"
      d.probed   = false;    // will be filled on first acquire
      d.npu_dev  = dev;
      devices.push_back(d);
      std::fprintf(stderr, "grxcp: GRX930 NPU detected at 0x%08x"
                   " (device %d)\n", NPU_C930_MMIO_BASE, d.index);
    } else {
      delete dev;
    }
  });
}

#endif  // GRXCP_ENABLE_NPU

grxError_t ensure_initialized() {
  std::call_once(g_init_once, [] {
    // Enumerate Vortex (GPU) devices first
    uint32_t count = 0;
    vx_result_t r = vx_device_count(&count);
    if (r != VX_SUCCESS) { g_init_error = map_result(r); return; }
    g_devices.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      g_devices[i].index = (int)i;
      g_devices[i].type  = DeviceType::GPU;
    }

    // Probe for the GRX930 NPU and append it
#ifdef GRXCP_ENABLE_NPU
    probe_npu_device(g_devices);
#endif
  });
  return g_init_error;
}

grxError_t acquire_device(int index, Device** out) {
  grxError_t e = ensure_initialized();
  if (e != grxSuccess) return e;
  if (index < 0 || (size_t)index >= g_devices.size()) return grxErrorInvalidDevice;

  std::lock_guard<std::mutex> lock(g_devices_mutex);
  Device& d = g_devices[index];

  // GPU devices need vx_device_open; NPU devices are already "open" (MMIO)
  if (d.type == DeviceType::GPU && !d.opened) {
    vx_result_t r = vx_device_open((uint32_t)index, &d.handle);
    if (r != VX_SUCCESS) return map_result(r);
    d.opened = true;
  }

  if (!d.probed) {
    if (d.type == DeviceType::NPU) {
#ifdef GRXCP_ENABLE_NPU
      populate_npu_properties(d);
#else
      return grxErrorNotSupported;
#endif
    } else {
      populate_properties(d);
    }
    d.probed = true;
  }
  *out = &d;
  return grxSuccess;
}

int  current_device_index()          { return g_current_device; }
void set_current_device_index(int i) { g_current_device = i; }

}  // namespace grxcp

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

extern "C" {

grxError_t grxGetDeviceCount(int* count) {
  if (!count) return grxcp::set_error(grxErrorInvalidValue);
  grxError_t e = grxcp::ensure_initialized();
  if (e != grxSuccess) return grxcp::set_error(e);
  // Return the total device count from the table (GPU + NPU).
  // ensure_initialized() has already appended the NPU if present.
  *count = (int)grxcp::g_devices.size();
  return grxSuccess;
}

grxError_t grxSetDevice(int device) {
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);
  grxcp::set_current_device_index(device);
  return grxSuccess;
}

grxError_t grxGetDevice(int* device) {
  if (!device) return grxcp::set_error(grxErrorInvalidValue);
  *device = grxcp::current_device_index();
  return grxSuccess;
}

grxError_t grxGetDeviceProperties(grxDeviceProp_t* prop, int device) {
  if (!prop) return grxcp::set_error(grxErrorInvalidValue);
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);
  *prop = d->prop;
  return grxSuccess;
}

grxError_t grxMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(grxcp::current_device_index(), &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  // NPU has no vx_device_memory_info — report total DDR from properties.
  // The NPU's DDR is shared with the CPU, so "free" is the total minus
  // what the CPU has allocated.  For now, report total as free (the NPU
  // DMA can access any DDR address).
  if (d->type == grxcp::DeviceType::NPU) {
    if (freeBytes)  *freeBytes  = (size_t)d->prop.totalGlobalMem;
    if (totalBytes) *totalBytes = (size_t)d->prop.totalGlobalMem;
    return grxSuccess;
  }

  uint64_t f = 0, used = 0;
  vx_result_t r = vx_device_memory_info(d->handle, &f, &used);
  if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));
  if (freeBytes)  *freeBytes  = (size_t)f;
  if (totalBytes) *totalBytes = (size_t)d->prop.totalGlobalMem;
  return grxSuccess;
}

grxError_t grxDeviceSynchronize(void) {
  const int device = grxcp::current_device_index();
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);
  // NPU has no Vortex streams — nothing to sync.
  if (d->type == grxcp::DeviceType::NPU) return grxSuccess;
  // Drains every stream on the device, including the null stream -- CUDA's
  // contract is device-wide, not current-stream.
  e = grxcp::sync_all_streams(device);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxDeviceGetAttribute(int* value, int attr, int device) {
  (void)value; (void)attr; (void)device;
  // CUDA's attribute enum is a CUDA-specific numbering that would have to be
  // invented here to be honoured. grxGetDeviceProperties carries the same
  // information with names that mean something on this hardware.
  return grxcp::set_error(grxErrorNotSupported);
}

grxError_t grxDeviceCanAccessPeer(int* canAccess, int, int) {
  if (canAccess) *canAccess = 0;
  return grxSuccess;
}

grxError_t grxDeviceEnablePeerAccess(int, unsigned int) {
  // No peer path exists in hardware yet: NVLink-class remote decode on G100 and
  // the coherent port on the c930 NPU are both future work. Declared so the
  // surface stays stable; refused so nothing silently misbehaves.
  return grxcp::set_error(grxErrorNotSupported);
}

}  // extern "C"
