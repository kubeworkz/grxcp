// GRXCP — device side of cycle measurement. See include/grx/grx_cycles.h for
// what the numbers mean and why events cannot provide them.

#ifndef GRX_DEVICE_CYCLES_H
#define GRX_DEVICE_CYCLES_H

#include "grx_device.h"
#include <grx/grx_cycles.h>

namespace grx {

// Serializing read of the core's cycle counter. vx_rdcycle_sync flushes the
// warp pipeline first: without that the read can retire before the work it is
// supposed to be timing, which produces a small, stable, entirely wrong number.
// grx::clock64() is the unserialized read, for code that wants a timestamp
// rather than a measurement.
__forceinline__ uint64_t cycle_counter() { return vx_rdcycle_sync(); }

// Measures one warp's span through a kernel and writes it to slot
// [linear block index * warps per block + warp index].
//
//   __global__ void k(args* a) {
//     grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(a->cycles));
//     ... the work ...
//     probe.finish();
//   }
//
// A null pointer disables it, so the measured and unmeasured builds are the
// same kernel -- which matters, because a kernel compiled differently for
// measurement is not the kernel whose speed was measured.
//
// The cost when enabled is two serializing reads and one 24-byte store per
// warp. The pipeline flush is not free and it IS inside the measured region;
// for a kernel short enough that this matters, measure a loop of them.
//
// IT FINISHES ITSELF. The destructor calls finish(), so a kernel that
// constructs one and returns is measured whether or not anyone remembered the
// call. That is not tidiness: forgetting it produces a probe that records
// NOTHING and reports no error, and the host then summarises an array of zeros
// into "no warps wrote a slot" -- a silent absence that reads like a device
// problem. Every grxDNN kernel shipped in exactly that state, and it took
// pointing the profiler at a real workload to notice, because nothing else had
// ever read a grxDNN slot.
//
// finish() is idempotent, so calling it explicitly to end the measured region
// early still works and the destructor then does nothing. grxBLAS's kernels do
// that and their numbers are unchanged.
class cycle_probe {
 public:
  __forceinline__ explicit cycle_probe(grxCycleSlot* slots)
      : slots_(slots), start_(slots ? cycle_counter() : 0) {}

  __forceinline__ ~cycle_probe() { finish(); }

  __forceinline__ void finish() {
    if (!slots_) return;
    const uint64_t end = cycle_counter();
    grxCycleSlot* const slots = slots_;
    slots_ = nullptr;              // idempotent: the destructor will not redo it
    // One writer per warp. Every lane holds the same values, so letting them
    // all store would be correct and wasteful.
    if (lane_id() != 0) return;
    grxCycleSlot* s = &slots[block_index() * warps_per_cta() + warp_id()];
    s->start = start_;
    s->end   = end;
    s->core  = (uint32_t)vx_core_id();
    s->warp  = warp_id();
  }

 private:
  // Linear block index, so a 2D or 3D grid indexes slots without the host
  // having to know which shape the kernel was launched with.
  static __forceinline__ uint32_t block_index() {
    return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  }

  grxCycleSlot* slots_;
  uint64_t      start_;
};

}  // namespace grx

#endif  // GRX_DEVICE_CYCLES_H
