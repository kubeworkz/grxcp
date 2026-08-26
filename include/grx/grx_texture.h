// GRXCP — texture objects, and the honest name for what they are here.
//
// READ THIS BEFORE USING IT. GRX-G100 HAS TEX UNITS AND THIS DOES NOT USE THEM.
//
// The TEX units and TCACHE exist in hardware and are driven by the graphics
// path; nothing exposes them to compute kernels (cuda_mapping.md 7.8, still
// open). So every sample taken through this header is ADDRESSED AND FILTERED
// IN SOFTWARE, by the calling kernel, out of ordinary global memory. There is
// no texture cache, no dedicated interpolator, and no free clamping.
//
// Architecture section 10 rule 5 bans emulating a hardware feature behind an
// API that implies hardware, with one sanctioned exception: an emulation that
// is REPORTED THROUGH A DEVICE PROPERTY, the way the warp-shuffle fallback is
// reported through grxDeviceProp_t.warpShuffleIsEmulated. This is built to that
// exception and no further:
//
//     grxDeviceProp_t.textureIsEmulated == 1
//
// while this remains software. grx-conform prints it. A program that cares
// about texture throughput can ask, and a program that just wants the arithmetic
// to be right does not have to.
//
// WHAT YOU GET, THEN. Ported CUDA texture code compiles and computes the right
// values. What you do not get is the performance shape texture code is usually
// written for: a bilinear fetch here is four global loads and the arithmetic
// between them, issued by your own warp.
//
// TWO MORE DIFFERENCES, both deliberate and both why the compat entries are
// PARTIAL rather than MAPPED:
//
//   * FILTER WEIGHTS ARE FULL-PRECISION FLOAT. NVIDIA hardware quantizes the
//     fractional coordinate to 8 bits before interpolating, so a bilinear read
//     there is a step function with 256 treads. Reproducing that would mean
//     emulating a wart to more decimal places; the values here differ from a
//     CUDA reference by up to half a tread, and that is the reason.
//   * ONE CHANNEL FORMAT FAMILY: float and float4. Integer formats with
//     normalized reads (uchar4 and friends) are the other half of CUDA's
//     texture surface and are not here. A port that needs them fails to compile
//     rather than silently reading garbage.
//
// THE HANDLE IS A DEVICE ADDRESS, and that is the emulation showing through.
// In CUDA a cudaTextureObject_t is an opaque token the TEX unit understands.
// Here the KERNEL is what has to understand it, so the handle is the device
// address of a grx_texture_desc that the kernel loads and reads. It is passed
// to a kernel as an ordinary argument, exactly like a pointer, because that is
// what it is.

#ifndef GRX_TEXTURE_H
#define GRX_TEXTURE_H

#include <stddef.h>
#include <stdint.h>

#include "grx_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// The shared descriptor
//
// Written by the host, read by the DEVICE, so it obeys the same rules as every
// other cross-boundary struct here: fixed widths, explicit padding, addresses
// as uint64_t. ci/check_abi.py compares its layout across all four targets.
// ---------------------------------------------------------------------------

#define GRX_TEXTURE_ABI_VERSION 1u

// Channel formats. float and float4 only -- see the header comment.
#define GRX_TEX_FORMAT_FLOAT1  0u
#define GRX_TEX_FORMAT_FLOAT4  1u

// Out-of-range coordinates.
#define GRX_TEX_ADDRESS_CLAMP  0u   // hold the edge value
#define GRX_TEX_ADDRESS_WRAP   1u   // repeat
#define GRX_TEX_ADDRESS_MIRROR 2u   // repeat, reflected
#define GRX_TEX_ADDRESS_BORDER 3u   // return borderColor

#define GRX_TEX_FILTER_POINT   0u
#define GRX_TEX_FILTER_LINEAR  1u

typedef struct {
  uint32_t abi_version;      // GRX_TEXTURE_ABI_VERSION -- first field, never moves
  uint32_t format;           // GRX_TEX_FORMAT_*
  uint32_t width;            // texels
  uint32_t height;           // texels; 1 for a 1D texture
  uint32_t pitch;            // BYTES between rows, >= width * bytes-per-texel
  uint32_t address_x;        // GRX_TEX_ADDRESS_*
  uint32_t address_y;
  uint32_t filter;           // GRX_TEX_FILTER_*
  uint32_t normalized;       // 1: coordinates are 0..1 rather than 0..width
  uint32_t pad;
  uint64_t data;             // device address of the texel storage
  float    border[4];        // used only by GRX_TEX_ADDRESS_BORDER
} grx_texture_desc;

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

typedef struct grxArrayImpl* grxArray_t;

// The device address of a grx_texture_desc. Not opaque, and deliberately so:
// see the header comment.
typedef uint64_t grxTextureObject_t;

typedef struct {
  unsigned int addressMode[2];   // GRX_TEX_ADDRESS_*, x then y
  unsigned int filterMode;       // GRX_TEX_FILTER_*
  int          normalizedCoords; // 0 or 1
  float        borderColor[4];
} grxTextureDesc_t;

// A 2D array of texels, pitched for the device's alignment. height may be 1.
//
// Not a plain grxMalloc: the pitch is the allocator's business, not the
// caller's, and a texture read has to know it. grxArrayGetInfo hands it back
// for the memcpy the caller will want to do.
grxError_t grxMallocArray(grxArray_t* array, unsigned int format,
                          size_t width, size_t height);
grxError_t grxFreeArray(grxArray_t array);

// Where the texels live and how they are laid out, for a caller that would
// rather write them with the ordinary memcpy family than through
// grxMemcpy2DToArray.
grxError_t grxArrayGetInfo(grxArray_t array, void** devicePtr, size_t* pitch,
                           size_t* width, size_t* height, unsigned int* format);

// Row-wise copy into the array, honouring its pitch. `spitch` is the SOURCE
// pitch in bytes; `width` is in TEXELS, matching cudaMemcpy2DToArray's shape
// only in that both take a destination offset -- CUDA's width is in bytes,
// which is a trap this does not reproduce. The offsets are in texels too.
grxError_t grxMemcpy2DToArray(grxArray_t array, size_t xOffsetTexels,
                              size_t yOffset, const void* src, size_t spitch,
                              size_t widthTexels, size_t height,
                              grxMemcpyKind kind);

// Build a sampler over an array. This ALLOCATES DEVICE MEMORY for the
// descriptor the kernel will read, so it is a resource with a lifetime, not a
// value -- destroy it.
grxError_t grxCreateTextureObject(grxTextureObject_t* object, grxArray_t array,
                                  const grxTextureDesc_t* desc);
grxError_t grxDestroyTextureObject(grxTextureObject_t object);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRX_TEXTURE_H
