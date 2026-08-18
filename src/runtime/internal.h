// GRXCP runtime internals. Not installed; not part of the public ABI.

#ifndef GRXCP_INTERNAL_H
#define GRXCP_INTERNAL_H

#include <grx/grx_types.h>
#include <vortex2.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace grxcp {

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------

// One entry per device reported by vx_device_count. Devices are opened lazily:
// enumerating must not cost a device open, because grx-smi and any program that
// merely counts devices should not spin up every backend.
struct Device {
  int             index    = -1;
  vx_device_h     handle   = nullptr;
  bool            opened   = false;
  bool            probed   = false;   // properties populated
  grxDeviceProp_t prop     {};
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

}  // namespace grxcp

#endif  // GRXCP_INTERNAL_H
