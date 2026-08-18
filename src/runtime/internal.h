// GRXCP runtime internals. Not installed; not part of the public ABI.

#ifndef GRXCP_INTERNAL_H
#define GRXCP_INTERNAL_H

#include <grx/grx_types.h>
#include <vortex2.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace grxcp {

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

// Enumerate devices once per process. Safe to call repeatedly.
grxError_t ensure_initialized();

// Fetch a device entry by index, opening and probing it if needed.
grxError_t acquire_device(int index, Device** out);

// The current thread's device index (default 0).
int  current_device_index();
void set_current_device_index(int index);

// Backend the driver dlopened for this process, from $VORTEX_DRIVER. The
// driver stub defaults to "simx" when the variable is unset and loads
// libvortex-<name>.so; there is no runtime query for this, so reading the same
// variable the stub reads is the only honest way to report it.
grxBackend_t detect_backend();
const char*  backend_name(grxBackend_t b);

}  // namespace grxcp

#endif  // GRXCP_INTERNAL_H
