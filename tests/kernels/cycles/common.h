// Argument block for the cycle-measurement validation gate.

#ifndef GRXCP_CYCLES_COMMON_H
#define GRXCP_CYCLES_COMMON_H

#include <stdint.h>

struct cycles_args {
  uint64_t slots;   // grxCycleSlot[blocks * warps_per_block], or 0 to disable
  uint64_t sink;    // uint64_t, so the work cannot be optimised away
  uint32_t iters;   // dependent-chain iterations
  uint32_t seed;
};

#endif
