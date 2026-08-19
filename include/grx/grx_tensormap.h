// GRXCP — tensor maps: descriptors for DXA asynchronous tile copies.
//
// A tensor map tells the DMA engine how a multi-dimensional array is laid out
// in global memory and what a tile of it looks like. A kernel then asks for
// "the tile at these coordinates" and the engine delivers it into shared
// memory while the kernel does something else. This is the machinery behind
// cp.async / TMA-style pipelining, and it is what a blocked GEMM stands on.
//
// HOW THIS DIFFERS FROM CUDA, WHICH MATTERS FOR PORTING
//
// CUDA's tensor map (`CUtensorMap`, built by `cuTensorMapEncodeTiled`) is an
// opaque object the program owns and passes to a kernel as an argument. Each
// kernel launch can carry its own, and there is no shared resource to manage.
//
// GRX-G100's descriptors live in the DEVICE, in a small fixed set of slots --
// grxTensorMapGetSlotCount reports how many. Programming one is a device-state
// change, not the construction of a value, so:
//
//   * Slots are a shared resource. Two kernels that need different maps in the
//     same slot must not be in flight together, and nothing in the hardware
//     will tell you that you got it wrong.
//   * Programming is STREAM ORDERED. grxTensorMapProgramAsync goes into the
//     stream's command sequence ahead of the launches that follow it on that
//     stream -- which is what makes the slot's contents predictable at all.
//     Two streams sharing a slot are racing, exactly as two streams writing
//     the same buffer would be.
//
// GRXCP does not paper over this by copying descriptors around behind the
// caller's back. Slot management is the program's, and it is visible.

#ifndef GRX_TENSORMAP_H
#define GRX_TENSORMAP_H

#include "grx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// How the tile is written into shared memory.
typedef enum {
  // smem[i1 * tile0 + i0]: the tile lands in the same order it is read.
  grxTensorMapLayoutRowMajor = 0,
  // smem[i0 * tile1 + i1]: transposing write, rank <= 2 only. The engine
  // scatters one element per beat here rather than streaming, so it is
  // substantially slower per tile -- worth it when it saves a transpose in
  // the consumer, not worth it otherwise.
  grxTensorMapLayoutKMajor   = 1
} grxTensorMapLayout_t;

// The array, and the tile shape carved out of it.
//
// Sizes and tile extents are in ELEMENTS; strides are in BYTES, because that
// is what the engine consumes and converting silently is how a stride ends up
// scaled twice. stride[i] is the byte distance between consecutive entries
// along dimension i+1 -- so a row-major m x n array of 4-byte elements has
// size = {n, m} and stride = {n * 4}.
typedef struct {
  int      slot;             // which descriptor slot to program
  void*    base;             // device pointer to the array
  unsigned rank;             // 1 or 2 today; see the note below
  unsigned size[5];          // extent per dimension, in elements
  unsigned strideBytes[4];   // byte stride for dimensions 1..rank-1
  unsigned tile[5];          // tile extent per dimension, in elements
  unsigned elementBytes;     // 1, 2, 4 or 8
  grxTensorMapLayout_t layout;
  // Multicast: byte distance between the shared-memory bases of consecutive
  // CTAs in a cluster, so one issue can fill every peer. 0 disables it.
  unsigned smemStrideBytes;
} grxTensorMapDesc_t;

// Descriptor slots available on this device, or 0 when the device has no DMA
// engine. This is a hardware limit, not a GRXCP policy.
grxError_t grxTensorMapGetSlotCount(int* count, int device);

// Allocate memory the DMA engine can reach.
//
// The engine's bus master bypasses the per-core MMU, so the base address in a
// descriptor is a PHYSICAL address. On a device with virtual memory an ordinary
// grxMalloc pointer is not one, and an engine pointed at it would stream from
// whatever that bit pattern happens to name -- silently, and into the caller's
// shared memory. So a tensor map may only describe memory allocated here.
//
// There is no CUDA analogue: TMA goes through the same address translation as
// everything else. On a device without virtual memory this is grxMalloc with a
// different name, and the distinction still matters -- it says which
// allocations may be described, so the same source keeps working when the
// device gains an MMU.
//
// Freed with grxFree.
grxError_t grxMallocPhysical(void** ptr, size_t size);

// Program a slot. The async form is ordered within `stream`; the synchronous
// form runs on the null stream and returns once the write has landed.
//
// RANK. Only 1 and 2 are accepted today, and a higher rank returns
// grxErrorNotSupported rather than a partly-programmed descriptor. The engine
// handles up to 5, and the extra ranks are mechanical to add -- they are
// absent because nothing here exercises them yet, and an untested descriptor
// encoder is worse than an honest refusal.
grxError_t grxTensorMapProgramAsync(const grxTensorMapDesc_t* desc,
                                    grxStream_t stream);
grxError_t grxTensorMapProgram     (const grxTensorMapDesc_t* desc);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRX_TENSORMAP_H
