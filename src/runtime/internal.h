// GRXCP runtime internals. Not installed; not part of the public ABI.

#ifndef GRXCP_INTERNAL_H
#define GRXCP_INTERNAL_H

#include <grx/grx_abi.h>
#include <grx/grx_types.h>
#include <vortex2.h>

#include <cstdint>
#include <mutex>
#include <vector>

#ifdef GRXCP_ENABLE_NPU
struct npu_c930_device;  // forward declaration (npu_c930.h)
#endif

namespace grxcp {

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

// Device type tag: GPU (Vortex) or NPU (GRX930 systolic array).
// The NPU is a memory-mapped accelerator with no Vortex driver dependency.
enum class DeviceType { GPU, NPU };

// One entry per device. GPU devices come from vx_device_count and are opened
// lazily via vx_device_open. NPU devices are detected by MMIO probe and
// have no vx_device_h (handle stays nullptr).
struct Device {
  int             index    = -1;
  DeviceType      type     = DeviceType::GPU;
  vx_device_h     handle   = nullptr;  // nullptr for NPU devices
  bool            opened   = false;
  bool            probed   = false;   // properties populated
  grxDeviceProp_t prop     {};
#ifdef GRXCP_ENABLE_NPU
  struct npu_c930_device* npu_dev = nullptr;  // owned, only for NPU devices
#endif
};

// The single mapping point from a driver result to a GRXCP error. Nothing else
// in the runtime is allowed to construct a grxError_t from a vx_result_t --
// keeping it here is what makes the mapping auditable (AGENTS.md section 4).
grxError_t map_result(vx_result_t r);

// Record and return an error. Every public entry point funnels its failure
// through this so grxGetLastError has something to report.
grxError_t set_error(grxError_t e);

grxError_t ensure_initialized();
grxError_t acquire_device(int index, Device** out);

int  current_device_index();
void set_current_device_index(int index);

// Backend the driver dlopened for this process, from $VORTEX_DRIVER. The
// driver stub defaults to "simx" when the variable is unset and loads
// libvortex-<name>.so; there is no runtime query for this, so reading the same
// variable the stub reads is the only honest way to report it.
grxBackend_t detect_backend();
const char*  backend_name(grxBackend_t b);

// ---------------------------------------------------------------------------
// NPU device support (GRXCP_ENABLE_NPU)
// ---------------------------------------------------------------------------
#ifdef GRXCP_ENABLE_NPU
// Probe for the GRX930 NPU and append it to the device table.
// Called once during ensure_initialized(), after Vortex device enumeration.
void probe_npu_device(std::vector<Device>& devices);
#endif

// ---------------------------------------------------------------------------
// Allocations
// ---------------------------------------------------------------------------

// What the interval map returns for a device address. This is the mechanism
// that lets the public API traffic in plain void* device pointers while the
// driver traffics in refcounted buffer handles.
struct Mapping {
  vx_buffer_h buffer = nullptr;
  uint64_t    base   = 0;   // device address of the allocation
  uint64_t    offset = 0;   // byte offset of the queried address within buffer
  uint64_t    size   = 0;   // bytes remaining from the queried address
  int         device = 0;
  bool        managed = false;
  bool        physical = false;  // allocated where the DMA engine can reach it
};

// Resolve any address inside a live device allocation. Returns false for a
// host pointer, a stale pointer, or an address in no allocation at all.
bool lookup_device_pointer(const void* ptr, Mapping* out);

// Resolve a pinned host allocation made by grxMallocHost.
bool lookup_host_pointer(const void* ptr, Mapping* out);

// Release every allocation and slab owned by a device. Called by grxDeviceReset.
void release_all_allocations(int device);

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

// The device queue behind a stream. Passing nullptr yields the device's null
// stream, created on first use.
grxError_t resolve_stream(grxStream_t stream, int device, vx_queue_h* out_queue,
                          grxStream_t* out_stream);

// Legacy default-stream ordering. Before enqueuing on `stream`, collect the
// events this operation must wait on: work on the null stream waits for every
// blocking stream, and work on a blocking stream waits for the null stream.
void collect_wait_events(grxStream_t stream, int device,
                         std::vector<vx_event_h>* out);

// Record the completion event of the operation just enqueued on `stream`, so
// later operations can order against it. Takes ownership of `event`.
void set_stream_last_event(grxStream_t stream, int device, vx_event_h event);

// Wait for everything currently outstanding on a stream.
grxError_t sync_stream(grxStream_t stream, int device);

// Wait for every stream on a device, including the null stream.
grxError_t sync_all_streams(int device);

// Monotonic host nanoseconds, used for the event-timing fallback.
uint64_t host_now_ns();

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Everything the launch path needs about a resolved kernel. `params` is null
// and `has_layout` false when the toolchain did not supply a parameter layout,
// which is the difference between being able to pack a CUDA-style void**
// argument array and not.
struct KernelBinding {
  vx_kernel_h             kernel      = nullptr;
  const grx_kernel_param* params      = nullptr;
  uint32_t                num_params  = 0;
  uint32_t                args_size   = 0;
  uint32_t                static_smem = 0;
  int32_t                 num_regs    = -1;
  uint32_t                max_threads_per_block = 0;
  bool                    has_layout  = false;
  int                     device      = 0;
  // Provenance for grx-sanitize. These point into the owning ModuleState and
  // are valid for as long as the module is loaded; a kernel reached through
  // the host-stub registry leaves them empty, because a fat binary registered
  // by a static initializer has no file the sanitizer could read symbols from.
  const char*             name        = "";
  const char*             module_path = "";
  const char*             module_elf  = "";
  bool                    sanitized   = false;
};

// Load a device image and register it, remembering the path it came from.
grxError_t load_module_tracked(grxModule_t* module, const void* image,
                               size_t size, const char* path);

// Resolve a registered host stub for a device, loading its module on first use.
bool lookup_registration(const void* stub, int device, KernelBinding* out);

// Resolve a grxFunction_t obtained from the module path.
bool lookup_function(grxFunction_t func, KernelBinding* out);

// Resident CTAs per SM, from the CTA dispatcher's three bounds. Shared by the
// occupancy API and by launch validation.
int resident_blocks_per_sm(const grxDeviceProp_t& prop, int block_size,
                           size_t smem_per_block);

// ---------------------------------------------------------------------------
// grx-sanitize (src/runtime/sanitize.cpp)
// ---------------------------------------------------------------------------
//
// All of these are no-ops when GRX_SANITIZE is unset, which is what keeps the
// call sites in the allocator and the launch path unconditional and short.

bool     sanitize_enabled();
uint64_t sanitize_redzone_bytes();   // trailing pad per allocation, 0 when off

void sanitize_note_region(uint64_t base, uint64_t size);
void sanitize_forget_region(uint64_t base);
void sanitize_note_alloc(uint64_t base, uint64_t requested);
void sanitize_note_free(uint64_t base);
void sanitize_forget_all();

// Patch the control-block address into a .vxbin image. True when the image was
// built with --sanitize and now points at the block.
bool sanitize_patch_image(std::vector<uint8_t>& image, const char* elf_path,
                          int device);

grxError_t sanitize_arm(Device& d, uint32_t shared_bytes, uint32_t grid_threads,
                        const char* kernel, const char* module_path,
                        const char* elf_path, bool instrumented);
void sanitize_drain(int device);
int  sanitize_findings();
void sanitize_report_summary();

// ---------------------------------------------------------------------------
// grx-prof (src/runtime/profile.cpp)
// ---------------------------------------------------------------------------
//
// One measured operation, bracketed. `profile_begin` returns false when
// profiling is off, which is what lets a call site read as
//
//   ProfileSample s;
//   const bool p = profile_begin(device, &s);
//   ... do the work ...
//   if (p) profile_end_kernel(&s, ...);
//
// with no cost and no branching on an environment variable at the call site.
struct ProfileSample {
  int   device = 0;
  void* opaque = nullptr;
};

bool profile_enabled();
bool profile_begin(int device, ProfileSample* out);
void profile_end_kernel(ProfileSample* sample, const char* kernel,
                        const char* module_path, uint32_t grid[3],
                        uint32_t block[3], size_t shared, const void* stream);
void profile_end_transfer(ProfileSample* sample, const char* op, uint64_t bytes,
                          const char* kind, const void* stream);
void profile_abandon(ProfileSample* sample);

}  // namespace grxcp

#endif  // GRXCP_INTERNAL_H
