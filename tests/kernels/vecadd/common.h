// Argument block shared by the kernel and the host.
//
// Until grxcc exists, a kernel takes one pointer to a struct the host packs,
// so both sides must agree on the layout exactly -- which is why this is one
// header included by both rather than two declarations that look alike.

#ifndef GRXCP_VECADD_COMMON_H
#define GRXCP_VECADD_COMMON_H

#include <stdint.h>

struct vecadd_args {
  uint32_t n;
  uint64_t a;
  uint64_t b;
  uint64_t c;
};

#endif
