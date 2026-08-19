// Argument block shared by the DXA gate's kernel and its host program.

#ifndef GRXCP_DXA_COMMON_H
#define GRXCP_DXA_COMMON_H

#include <stdint.h>

struct dxa_args {
  uint64_t out;      // uint32_t[tile0 * tile1], global
  uint32_t slot;     // descriptor slot the host programmed
  uint32_t coord0;   // tile origin along dim 0, in elements
  uint32_t coord1;   // tile origin along dim 1, in elements
  uint32_t tile0;
  uint32_t tile1;
  uint32_t barrier;  // which CTA barrier slot to use
};

#endif
