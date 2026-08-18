// Inspection hooks for the mock driver.
//
// The mock cannot execute a kernel -- there is no RISC-V core behind it. What
// it CAN do is record the launch descriptor the runtime handed the driver,
// which is exactly the part of the launch path that is worth testing without a
// simulator: dimension mapping, shared-memory sizing, cluster dimensions, and
// argument packing. Getting any of those wrong produces a wrong answer on real
// hardware with no error anywhere, so they get asserted here.

#ifndef GRXCP_VORTEX_MOCK_H
#define GRXCP_VORTEX_MOCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GRXMOCK_MAX_ARGS 256

typedef struct {
  int      valid;
  uint64_t entry_pc;
  uint32_t ndim;
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t cluster_dim[3];
  uint32_t lmem_size;
  uint32_t args_size;
  uint8_t  args[GRXMOCK_MAX_ARGS];
} grxmock_launch_record;

// The most recent launch the driver received, and how many it has seen.
const grxmock_launch_record* grxmock_last_launch(void);
uint32_t                     grxmock_launch_count(void);
void                         grxmock_reset_launches(void);

// Build a mock module image: a magic tag followed by NUL-terminated kernel
// names. The mock does not parse real .vxbin encoding -- it models the
// contract (named entries resolve to kernels), not the file format.
// Returns the number of bytes written, or 0 if the buffer is too small.
size_t grxmock_build_module(void* buffer, size_t capacity,
                            const char* const* kernel_names, uint32_t count);

#ifdef __cplusplus
}
#endif

#endif  // GRXCP_VORTEX_MOCK_H
