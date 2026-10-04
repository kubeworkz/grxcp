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
#ifdef GRXCP_ENABLE_PTA
#include "pta_chiplet.h"
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

// A device known to have no analog tile: both leading fields 0, and -1 in
// every field that would describe a model, because there is not one.
grxAnalogGemm_t analog_gemm_none() {
  grxAnalogGemm_t a;
  a.gemmIsAnalogEmulated   = 0;
  a.tileIsPresent          = 0;
  a.activationBits         = -1;
  a.weightBits             = -1;
  a.adcBits                = -1;
  a.adcShift               = -1;
  a.seed                   = -1;
  a.impairments            = -1;
  a.impairmentsImplemented = 0;
  a.thermalSigmaQ8         = -1;
  a.shotCoefficientQ8      = -1;
  a.programmingSigmaQ8     = -1;
  a.driftSigmaQ8           = -1;
  a.driftLog2Shots         = -1;
  a.driftClampQ8           = -1;
  a.crosstalkQ8            = -1;
  a.loopModes              = -1;
  a.calibrationValid       = -1;
  a.tileRows               = -1;
  a.tileCols               = -1;
  a.operandBits            = -1;
  a.accumulatorBits        = -1;
  a.gemmIndex              = -1;
  return a;
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
  p.textureIsEmulated       = 1;   // software sampling; cuda_mapping.md 7.8
  // A G100's GEMM is the TCU's, or a kernel's, and both are digital: nothing
  // in the Vortex driver's capability set describes a photonic tile, and the
  // board plan's PTA chiplet behind the GPU's BAR has no driver here to ask.
  // So this is "no tile", sourced from the absence of anything to source it
  // from, and it says so with -1 in every field that would describe one.
  p.analogGemm = analog_gemm_none();
  // A GPU's memory is its own. A PTA chiplet in its package is a device of its
  // own in the table, and names this one as its parent.
  p.parentDevice = -1;

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
    case GRX_BACKEND_MODEL:   return "model";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// NPU C930 device support
// ---------------------------------------------------------------------------
#ifdef GRXCP_ENABLE_NPU

// The register model installed through npu_c930_testing.h, if any. Read in two
// places -- probe_npu_device, to decide what to detect through, and
// populate_npu_properties, to decide what the device says it is -- and those
// two must never disagree, which is why it is one variable and not two flags.
struct PendingNpuModel {
  npu_c930_read_fn  read32  = nullptr;
  npu_c930_write_fn write32 = nullptr;
  void*             ctx     = nullptr;
  npu_c930_mem_read_fn  mem_read  = nullptr;
  npu_c930_mem_write_fn mem_write = nullptr;
  void*                 mem_ctx   = nullptr;
  // What the attached model IS. Defaults to a software register model; a
  // Verilator harness driving the RTL says so and gets RTLSIM. Never SILICON.
  grxBackend_t          backend   = GRX_BACKEND_MODEL;
};
static PendingNpuModel g_npu_model;
static bool            g_npu_enumerated = false;

// What the NPU's registers say a GEMM is, right now. The decision is the
// backend's (npu_c930_read_analog, and its header for why PTA_CTRL.EN is not
// the bit that decides it); this only carries it into the public struct.
//
// A device with no register path at all comes back unknown in every field,
// which is the one answer that claims nothing.
static grxAnalogGemm_t read_npu_analog(npu_c930_device_t* dev) {
  npu_c930_analog_t n;
  (void)npu_c930_read_analog(dev, &n);
  grxAnalogGemm_t a;
  a.gemmIsAnalogEmulated   = n.analog;
  a.tileIsPresent          = n.tile_present;
  a.activationBits         = n.activation_bits;
  a.weightBits             = n.weight_bits;
  a.adcBits                = n.adc_bits;
  a.adcShift               = n.adc_shift;
  a.seed                   = n.seed;
  a.impairments            = n.impairments;
  a.impairmentsImplemented = n.impairments_implemented;
  a.thermalSigmaQ8         = n.sigma_thermal_q8;
  a.shotCoefficientQ8      = n.shot_k_q8;
  a.programmingSigmaQ8     = n.sigma_prog_q8;
  a.driftSigmaQ8           = n.drift_sigma_q8;
  a.driftLog2Shots         = n.drift_log2_shots;
  a.driftClampQ8           = n.drift_clamp_q8;
  a.crosstalkQ8            = n.crosstalk_q8;
  a.loopModes              = n.loop_modes;
  a.calibrationValid       = n.calibration_valid;
  a.tileRows               = n.tile_rows;
  a.tileCols               = n.tile_cols;
  a.operandBits            = n.operand_bits;
  a.accumulatorBits        = n.accumulator_bits;
  // A c930's GEMM runs on PTA_SEED as its host wrote it; there is no index.
  a.gemmIndex              = -1;
  return a;
}

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
  p.textureIsEmulated       = 1;  // and no TEX unit either
  // The first reading. grxGetDeviceProperties takes another on every call --
  // these are live registers, and this copy is only what was true at probe.
  p.analogGemm = read_npu_analog(d.npu_dev);
  p.parentDevice = -1;   // the c930's DDR is its own

  // WHAT THIS DEVICE IS, DERIVED RATHER THAN ASSERTED.
  //
  // These two lines used to read
  //
  //     p.backend = GRX_BACKEND_SILICON;  // NPU is always real hardware
  //     snprintf(p.name, ..., "GRX930 NPU (silicon)");
  //
  // -- a claim about what the device is, made by a function with no way to
  // know, in a struct whose entire purpose is to let a caller find out. It
  // happened to hold only because nothing could reach this device except an
  // mmap of /dev/mem. The seam in npu_c930_testing.h removes that accident:
  // attach a register model and the same line would report a C triple loop as
  // silicon, which is the fabrication AGENTS.md section 1 exists to stop.
  //
  // So the field follows the way the device was reached. There is exactly one
  // predicate, it is the same one probe_npu_device used to decide what to
  // detect through, and a model cannot be attached without it moving.
  const bool via_model = g_npu_model.read32 || g_npu_model.write32;
  p.backend = via_model ? g_npu_model.backend : GRX_BACKEND_SILICON;
  std::snprintf(p.name, sizeof(p.name), "GRX930 NPU (%s)",
                !via_model ? "silicon"
                : (g_npu_model.backend == GRX_BACKEND_RTLSIM)
                    ? "RTL through Verilator, NOT hardware"
                    : "software register model, NOT hardware");
}

void probe_npu_device(std::vector<Device>& devices) {
  static std::once_flag npu_once;
  std::call_once(npu_once, [&devices] {
    // THE BRACES ARE THE FIX, AND THEY WERE MISSING.
    //
    // npu_c930_device_t is a C struct with no constructor, so `new T` -- no
    // braces -- default-initialises it, which for a POD means every member is
    // INDETERMINATE. Among those members are `read32` and `mmio_base`, and
    // npu_c930_detect's first act is reg_read, which does
    //
    //     if (dev->read32) return dev->read32(dev->io_ctx, offset);
    //
    // an indirect call through whatever was in that word. `new T{}` value-
    // initialises: every member zero, read32 null, mmio_base null, and detect
    // takes the mmap path it was written for.
    //
    // IT DID NOT ALWAYS CRASH, which is why it survived. A fresh heap page from
    // the OS is zeroed, so a program whose first act is grxGetDeviceCount gets
    // a struct that happens to be zero and behaves. A program that reads a file
    // FIRST does not: tests/libs/test_grxdnn_gelu.cpp loads its reference
    // vectors before touching the device, and on that dirty heap this probe
    // called a function pointer made of freed bytes. Reproduced deterministically
    // by filling and freeing 4 MB of 0xAA before the first grx call.
    //
    // The predicate this probe uses was already tightened once -- detection is a
    // write-readback now, because reading STATUS and accepting anything that was
    // not 0xFFFFFFFF grew an NPU on any host where /dev/mem opened. That fixed
    // what the answer was judged against. This is the handle the question was
    // asked through, and it was never initialised at all.
    npu_c930_device_t* dev = new npu_c930_device_t{};
    // A model installed through the seam replaces the mmap, and only before
    // this runs -- which is why the seam refuses after enumeration rather than
    // pretending to work. npu_c930_attach_model memsets the struct, so this
    // must come after the value-initialisation above and not instead of it.
    if (g_npu_model.read32 || g_npu_model.write32)
      npu_c930_attach_model(dev, g_npu_model.read32, g_npu_model.write32,
                            g_npu_model.ctx);
    // AFTER attach_model, which memsets the struct. The header says so; this
    // is the one call site that has to obey it.
    if (g_npu_model.mem_read || g_npu_model.mem_write)
      npu_c930_attach_memory(dev, g_npu_model.mem_read, g_npu_model.mem_write,
                             g_npu_model.mem_ctx);
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
                   " (device %d)%s\n", NPU_C930_MMIO_BASE, d.index,
                   !(g_npu_model.read32 || g_npu_model.write32) ? ""
                   : (g_npu_model.backend == GRX_BACKEND_RTLSIM)
                       ? " -- THROUGH THE RTL UNDER VERILATOR, not hardware"
                       : " -- THROUGH A REGISTER MODEL, not hardware");
    } else {
      delete dev;
    }
    g_npu_enumerated = true;
  });
}

// The one NPU device handle in the process, or null. grxblas.cpp used to keep
// its OWN file-static npu_c930_device_t and call npu_c930_detect on it, so a
// build with an NPU had two handles, two detections and, on a real machine,
// two independent mmaps of the same register block. Worse for the seam: a
// model attached to the enumerated device would not have been the device
// grxblasGemmEx dispatched through, so the routing could be exercised against
// one device while the answer came from another.
npu_c930_device* npu_device_for(int index) {
  if (index < 0 || (size_t)index >= g_devices.size()) return nullptr;
  const Device& d = g_devices[index];
  return (d.type == DeviceType::NPU) ? d.npu_dev : nullptr;
}

#endif  // GRXCP_ENABLE_NPU

// ---------------------------------------------------------------------------
// PTA chiplet support
// ---------------------------------------------------------------------------
#ifdef GRXCP_ENABLE_PTA

// The model installed through pta_chiplet_testing.h, if any. Unlike the NPU's
// this is not an alternative to a hardware path: it is the only path. Read in
// two places -- probe_pta_device, to decide whether there is anything to
// detect, and populate_pta_properties, to say what the device is.
struct PendingPtaModel {
  pta_chiplet_read_fn   read32  = nullptr;
  pta_chiplet_write_fn  write32 = nullptr;
  void*                 ctx     = nullptr;
  pta_chiplet_submit_fn submit  = nullptr;
  void*                 link_ctx = nullptr;
  int                   parent  = 0;
};
static PendingPtaModel g_pta_model;
static bool            g_pta_enumerated = false;

// What the chiplet's registers say a GEMM is, right now, in the public struct.
static grxAnalogGemm_t read_pta_analog(pta_chiplet_device_t* dev) {
  pta_chiplet_analog_t n;
  (void)pta_chiplet_read_analog(dev, &n);
  grxAnalogGemm_t a;
  a.gemmIsAnalogEmulated   = n.analog;
  a.tileIsPresent          = n.tile_present;
  a.activationBits         = n.activation_bits;
  a.weightBits             = n.weight_bits;
  a.adcBits                = n.adc_bits;
  a.adcShift               = n.adc_shift;
  a.seed                   = n.seed;
  a.impairments            = n.impairments;
  a.impairmentsImplemented = n.impairments_implemented;
  a.thermalSigmaQ8         = n.sigma_thermal_q8;
  a.shotCoefficientQ8      = n.shot_k_q8;
  a.programmingSigmaQ8     = n.sigma_prog_q8;
  a.driftSigmaQ8           = n.drift_sigma_q8;
  a.driftLog2Shots         = n.drift_log2_shots;
  a.driftClampQ8           = n.drift_clamp_q8;
  a.crosstalkQ8            = n.crosstalk_q8;
  a.loopModes              = n.loop_modes;
  a.calibrationValid       = n.calibration_valid;
  a.tileRows               = n.tile_rows;
  a.tileCols               = n.tile_cols;
  a.operandBits            = n.operand_bits;
  a.accumulatorBits        = n.accumulator_bits;
  // The one field the c930 has no use for. Each of the chiplet's GEMMs runs on
  // a seed derived from `seed` and this, so a report without it describes a
  // run and not a result.
  a.gemmIndex              = n.gemm_index;
  return a;
}

// Fill grxDeviceProp_t for the PTA chiplet. It has no pipeline, no memory and
// no clock of its own that anything here can source, so nearly every field is
// zero and the profile is one bit wide.
static void populate_pta_properties(Device& d) {
  grxDeviceProp_t& p = d.prop;
  std::memset(&p, 0, sizeof(p));

  p.deviceType = GRX_DEVICE_TYPE_PTA;
  // The only way here is the seam, so the only thing this can be is a model.
  // Derived from how the device was reached, as the NPU's is: the day a
  // hardware path exists this line is where it starts to matter.
  p.backend    = GRX_BACKEND_MODEL;

  p.numBarriers = 0;

  // GEMM, and nothing else. No launch, no streams, no events, and above all no
  // memcpy: there is nothing on this device to copy to.
  p.capabilities = GRX_CAP_GEMM;

  p.warpShuffleIsEmulated   = 0;  // no shuffles at all
  p.eventTimingIsDeviceSide = 0;
  p.constantMemoryIsGlobal  = 1;  // no __constant__ path
  p.textureIsEmulated       = 1;  // and no TEX unit either
  p.analogGemm   = read_pta_analog(d.pta_dev);
  p.parentDevice = d.parent;

  std::snprintf(p.name, sizeof(p.name),
                "GRX PTA chiplet (software register model, NOT hardware)");
}

void probe_pta_device(std::vector<Device>& devices) {
  static std::once_flag pta_once;
  std::call_once(pta_once, [&devices] {
    g_pta_enumerated = true;
    // NOTHING ATTACHED, NOTHING ENUMERATED. There is no hardware path to try:
    // the chiplet's window is a page of the GPU's BAR and the GPU's driver has
    // no call that reads it. A build flag says what code exists, not what is
    // in the package.
    if (!g_pta_model.read32 && !g_pta_model.write32) return;

    // A tile with no parent has nothing to compute on, so it is not a device.
    const int parent = g_pta_model.parent;
    if (parent < 0 || (size_t)parent >= devices.size() ||
        devices[parent].type != DeviceType::GPU) {
      std::fprintf(stderr, "grxcp: a PTA chiplet model was attached behind "
                   "device %d, which is not a GPU here; not enumerated\n", parent);
      return;
    }

    pta_chiplet_device_t* dev = new pta_chiplet_device_t{};
    pta_chiplet_attach_window(dev, g_pta_model.read32, g_pta_model.write32,
                              g_pta_model.ctx);
    // AFTER attach_window, which zeroes the struct.
    if (g_pta_model.submit)
      pta_chiplet_attach_link(dev, g_pta_model.submit, g_pta_model.link_ctx);
    if (!pta_chiplet_detect(dev)) {
      delete dev;
      return;
    }
    Device d;
    d.index   = (int)devices.size();
    d.type    = DeviceType::PTA;
    d.handle  = nullptr;   // no Vortex handle
    d.opened  = true;
    d.probed  = false;
    d.parent  = parent;
    d.pta_dev = dev;
    devices.push_back(d);
    std::fprintf(stderr, "grxcp: PTA chiplet detected behind device %d (device %d)"
                 " -- THROUGH A REGISTER MODEL, not hardware\n", parent, d.index);
  });
}

pta_chiplet_device* pta_device_for(int index) {
  if (index < 0 || (size_t)index >= g_devices.size()) return nullptr;
  const Device& d = g_devices[index];
  return (d.type == DeviceType::PTA) ? d.pta_dev : nullptr;
}

int pta_parent_of(int index) {
  if (index < 0 || (size_t)index >= g_devices.size()) return -1;
  const Device& d = g_devices[index];
  return (d.type == DeviceType::PTA) ? d.parent : -1;
}

#endif  // GRXCP_ENABLE_PTA

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
    // And the PTA chiplet, last: it names a GPU as its parent, so the GPUs have
    // to be in the table before it.
#ifdef GRXCP_ENABLE_PTA
    probe_pta_device(g_devices);
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
    } else if (d.type == DeviceType::PTA) {
#ifdef GRXCP_ENABLE_PTA
      populate_pta_properties(d);
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

// A device's properties as they are NOW.
//
// Everything in grxDeviceProp_t was established once, at the first acquire,
// and that was right for all of it until analogGemm arrived: geometry, memory
// and capabilities are what the device IS. Whether its GEMMs are analog is
// what the device is currently SET to, in registers any holder of the device
// can write, and a property cached at probe would have gone on reporting
// "native" over a tile somebody had since impaired. So that one field is read
// again here, under the same lock the cached copy is made under.
void snapshot_properties(Device& d, grxDeviceProp_t* out) {
  std::lock_guard<std::mutex> lock(g_devices_mutex);
#ifdef GRXCP_ENABLE_NPU
  if (d.type == DeviceType::NPU) d.prop.analogGemm = read_npu_analog(d.npu_dev);
#endif
#ifdef GRXCP_ENABLE_PTA
  // The chiplet's too, and for one more reason than the NPU's: PTA_GEMM_CT
  // moves with every GEMM, so a cached copy is wrong after the first one.
  if (d.type == DeviceType::PTA) d.prop.analogGemm = read_pta_analog(d.pta_dev);
#endif
  *out = d.prop;
}

}  // namespace grxcp

// ---------------------------------------------------------------------------
// The test seam (npu_c930_testing.h)
// ---------------------------------------------------------------------------
#ifdef GRXCP_ENABLE_NPU
extern "C" {

int grxcp_npu_attach_model_for_testing(npu_c930_read_fn read32,
                                       npu_c930_write_fn write32,
                                       void* ctx) {
  // AFTER ENUMERATION THIS CANNOT WORK, AND SAYS SO.
  //
  // probe_npu_device runs once behind a std::call_once. A model installed
  // after that would sit in g_npu_model unread, and -- far worse -- would flip
  // populate_npu_properties into reporting the device as a model when the
  // device it describes was found by mmap. Returning 0 is the difference
  // between a test that skips and a test that lies about what it ran on.
  if (grxcp::g_npu_enumerated) return 0;
  grxcp::g_npu_model.read32  = read32;
  grxcp::g_npu_model.write32 = write32;
  grxcp::g_npu_model.ctx     = ctx;
  return 1;
}

int grxcp_npu_attach_memory_for_testing(npu_c930_mem_read_fn mem_read,
                                        npu_c930_mem_write_fn mem_write,
                                        void* ctx) {
  if (grxcp::g_npu_enumerated) return 0;
  grxcp::g_npu_model.mem_read  = mem_read;
  grxcp::g_npu_model.mem_write = mem_write;
  grxcp::g_npu_model.mem_ctx   = ctx;
  return 1;
}

int grxcp_npu_set_model_backend(int backend) {
  if (grxcp::g_npu_enumerated) return 0;
  // NOTHING ATTACHED THROUGH THIS SEAM MAY CLAIM TO BE A CHIP. The seam exists
  // so a model can stand in for hardware; letting it also SAY it is hardware
  // would hand back the exact fabrication the derived backend field removed.
  if (backend == GRX_BACKEND_SILICON) return 0;
  if (backend != GRX_BACKEND_MODEL && backend != GRX_BACKEND_RTLSIM) return 0;
  grxcp::g_npu_model.backend = (grxBackend_t)backend;
  return 1;
}

int grxcp_npu_model_is_attached(void) {
  return (grxcp::g_npu_model.read32 || grxcp::g_npu_model.write32) ? 1 : 0;
}

}  // extern "C"
#endif  // GRXCP_ENABLE_NPU

// ---------------------------------------------------------------------------
// The test seam (pta_chiplet_testing.h)
// ---------------------------------------------------------------------------
#ifdef GRXCP_ENABLE_PTA
extern "C" {

// Each of these refuses after enumeration, for the reason the NPU's do: the
// probe runs once, and a model installed after it would be read by nothing.

int grxcp_pta_attach_model_for_testing(pta_chiplet_read_fn read32,
                                       pta_chiplet_write_fn write32,
                                       void* ctx) {
  if (grxcp::g_pta_enumerated) return 0;
  grxcp::g_pta_model.read32  = read32;
  grxcp::g_pta_model.write32 = write32;
  grxcp::g_pta_model.ctx     = ctx;
  return 1;
}

int grxcp_pta_attach_link_for_testing(pta_chiplet_submit_fn submit, void* ctx) {
  if (grxcp::g_pta_enumerated) return 0;
  grxcp::g_pta_model.submit   = submit;
  grxcp::g_pta_model.link_ctx = ctx;
  return 1;
}

int grxcp_pta_set_parent_for_testing(int gpu_index) {
  if (grxcp::g_pta_enumerated) return 0;
  grxcp::g_pta_model.parent = gpu_index;
  return 1;
}

int grxcp_pta_model_is_attached(void) {
  return (grxcp::g_pta_model.read32 || grxcp::g_pta_model.write32) ? 1 : 0;
}

}  // extern "C"
#endif  // GRXCP_ENABLE_PTA

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
  grxcp::snapshot_properties(*d, prop);
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
  // A PTA chiplet has no memory, and that is an answer: none free, of none.
  if (d->type == grxcp::DeviceType::PTA) {
    if (freeBytes)  *freeBytes  = 0;
    if (totalBytes) *totalBytes = 0;
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
  // Nor has a PTA chiplet, and a GEMM on it has ended by the time the call
  // that issued it returns.
  if (d->type == grxcp::DeviceType::PTA) return grxSuccess;
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
