// Shared between the sanitizer gate's host and device sides.

#ifndef GRX_TEST_SANITIZE_COMMON_H
#define GRX_TEST_SANITIZE_COMMON_H

#include <stdint.h>

struct san_args {
  uint64_t out;      // device pointer to `n` uint32_t
  uint32_t n;        // elements the host actually allocated
  uint32_t pad;
};

#endif  // GRX_TEST_SANITIZE_COMMON_H
