// Argument block for the warp-primitive gate.

#ifndef GRXCP_WARP_COMMON_H
#define GRXCP_WARP_COMMON_H

#include <stdint.h>

struct warp_args {
  uint64_t in;      // uint32_t[threads]
  uint64_t out;     // uint32_t[threads * 4] for the shuffle modes, or per-warp
  uint64_t aux;     // second output, meaning depends on the entry point
  uint32_t threads; // total threads in the launch
  uint32_t width;   // segment width for the shuffle modes
};

#endif
