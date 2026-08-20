// GRXCP — the memcheck ABI shared by the host runtime and the device-side
// sanitizer runtime.
//
// The host owns every field here; the device only reads them, plus one atomic
// counter and the report array it writes into. The struct lives as a global
// (`__grx_san_state`) in a sanitized kernel image, and the host finds its
// device address by reading the module's sibling ELF symbol table -- the
// .vxbin footer carries kernel entries only.
//
// Layout rules, because this crosses an ABI boundary between an x86-64 host
// and an rv64 device: explicit fixed-width types, explicit padding, 8-byte
// alignment for every 64-bit field, no bitfields, no pointers (device
// addresses are uint64_t on both sides). The version field is checked by the
// device before it reads anything else.

#ifndef GRX_SANITIZE_ABI_H
#define GRX_SANITIZE_ABI_H

#include <stdint.h>

#define GRX_SAN_ABI_VERSION 1u

// What a finding is.
#define GRX_SAN_KIND_OOB_GLOBAL   1u  // in a region GRXCP owns, in no allocation
#define GRX_SAN_KIND_OOB_STRADDLE 2u  // starts inside an allocation, runs past its end
#define GRX_SAN_KIND_USE_AFTER_FREE 3u
#define GRX_SAN_KIND_OOB_SHARED   4u  // outside this CTA's shared-memory slot

#define GRX_SAN_FLAG_WRITE  0x1u

// Extent states.
#define GRX_SAN_EXTENT_LIVE  1u
#define GRX_SAN_EXTENT_FREED 2u

// One live-or-quarantined allocation, exactly as the caller asked for it: the
// rounding slop the allocator adds is deliberately NOT part of the extent, so
// a one-element overflow of a 100-byte buffer is a finding even though the
// allocator handed out 256 bytes.
typedef struct {
  uint64_t base;
  uint64_t size;
  uint32_t state;    // GRX_SAN_EXTENT_*
  uint32_t id;       // allocation ordinal, for "allocation #7" in the report
} grxSanExtent;

// A region the GRXCP allocator owns (a slab, or a direct buffer). Only
// addresses inside one of these are checked against the extent table; anything
// else is memory this runtime did not hand out and cannot judge.
typedef struct {
  uint64_t base;
  uint64_t size;
} grxSanRegion;

// One report slot. `kind` is zero for an empty slot, which is what makes the
// slot table readable without a count.
typedef struct {
  uint64_t addr;         // faulting address
  uint64_t pc;           // device PC just after the access, for symbolization
  uint64_t extent_base;  // owning or nearest allocation, 0 if none
  uint64_t extent_size;
  uint32_t size;         // access width in bytes
  uint32_t flags;        // GRX_SAN_FLAG_*
  uint32_t kind;         // GRX_SAN_KIND_*, 0 = empty slot
  uint32_t extent_id;
  uint32_t block;        // linearized blockIdx
  uint32_t thread;       // linearized threadIdx
  uint32_t warp;         // hardware warp id within the CTA
  uint32_t lane;         // hardware lane id within the warp
} grxSanReport;

// The device-visible control block.
//
// There is no finding counter, and that is deliberate. A counter needs an
// atomic increment, and GRX-G100 builds with VX_CFG_EXT_A_ENABLED off -- which
// this device configuration is -- abort in the LSU on any AMO instruction. So
// the report array is indexed by grid-linear thread instead: every thread owns
// one slot, writes its FIRST finding there and nothing after, and the host
// counts the slots whose kind is non-zero. No contention, no atomics, and the
// same run produces the same report twice.
//
// The cost is a slot per thread. `max_reports` is how many slots exist and
// `grid_threads` is how many the launch wanted; when the second exceeds the
// first, threads above the limit record nothing and the host says so.
typedef struct {
  uint32_t abi_version;
  uint32_t enabled;
  uint64_t extents;      // device address of grxSanExtent[num_extents]
  uint64_t regions;      // device address of grxSanRegion[num_regions]
  uint64_t reports;      // device address of grxSanReport[max_reports]
  uint32_t num_extents;
  uint32_t num_regions;
  uint32_t max_reports;  // report slots available
  uint32_t grid_threads; // threads this launch dispatches
  uint32_t shared_bytes; // this launch's sharedMem request, per CTA
  uint32_t reserved;
} grxSanState;

#endif  // GRX_SANITIZE_ABI_H
