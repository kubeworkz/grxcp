// Shared between the cooperative-groups gate's host and device sides.

#ifndef GRX_TEST_CG_COMMON_H
#define GRX_TEST_CG_COMMON_H

#include <stdint.h>

// Output slots, so a wrong collective is identifiable rather than just a wrong
// total. Each kernel writes one uint32 per thread into its own band.
#define CG_BAND_RANK      0
#define CG_BAND_SIZE      1
#define CG_BAND_REDUCE    2
#define CG_BAND_INCLUSIVE 3
#define CG_BAND_EXCLUSIVE 4
#define CG_BAND_SHFL      5
#define CG_BAND_BALLOT    6
#define CG_BAND_META      7
#define CG_BANDS          8

struct cg_args {
  uint64_t in;        // uint32_t[threads]
  uint64_t out;       // uint32_t[CG_BANDS * threads]
  uint32_t threads;   // total threads in the grid
  uint32_t tile;      // tile width for the tile kernel
};

#endif  // GRX_TEST_CG_COMMON_H
