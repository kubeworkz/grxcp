// Shared between the texture sampling kernel and its host driver.
#ifndef GRXCP_TEST_TEXTURE_COMMON_H
#define GRXCP_TEST_TEXTURE_COMMON_H

#include <stdint.h>

// One sample request per thread. Coordinates come from the host so the same
// list drives the device and the reference.
struct texture_args {
  uint32_t abi_version;
  uint32_t count;
  uint64_t object;   // grxTextureObject_t
  uint64_t coords;   // const float[2 * count], x then y
  uint64_t out;      // float[count]
};

#define TEXTURE_ARGS_ABI 1u

#endif
