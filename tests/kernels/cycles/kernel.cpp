// A kernel whose cost is known to be proportional to a number the host picks.
//
// This is the calibration weight for the cycle probe. Measuring something is
// only worth doing if the measurement has been shown to respond to the thing
// it claims to measure: double the work, the count should double. Without this
// the probe could be returning a constant, or a host clock, or nanoseconds,
// and every number derived from it would look reasonable.
//
// The loop is a dependent multiply-add chain: the compiler cannot vectorise it
// away, cannot hoist it, and each iteration costs the same as the last.

#include <grx/device/grx_cycles.h>

#include "common.h"

__global__ void spin(cycles_args* __UNIFORM__ arg) {
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->slots));

  uint64_t acc = (uint64_t)arg->seed + threadIdx.x;
  for (uint32_t i = 0; i < arg->iters; ++i)
    acc = acc * 6364136223846793005ull + 1442695040888963407ull;

  // The sink is what stops the chain from being dead code. Only one thread
  // writes it; the value is not checked, only its existence matters.
  if (arg->sink && threadIdx.x == 0)
    *reinterpret_cast<uint64_t*>(arg->sink) = acc;

  probe.finish();
}
