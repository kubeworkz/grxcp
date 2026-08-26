// GRXCP — texture arrays and sampler objects.
//
// The whole of what makes this different from a pitched allocation is in
// grx_texture.h's header comment: there is no TEX unit reachable from compute,
// so the KERNEL samples, and the object handle is therefore the device address
// of a descriptor the kernel loads. This file allocates the texels, allocates
// the descriptor, fills it, and hands back its address.
//
// Everything here is ordinary device memory. That is the point, and it is why
// grxDeviceProp_t.textureIsEmulated reads 1.

#include "internal.h"

#include <grx/grx_runtime.h>
#include <grx/grx_texture.h>

#include <cstring>
#include <map>
#include <mutex>

namespace grxcp {
namespace {

struct ArrayImpl {
  void*        data   = nullptr;
  size_t       pitch  = 0;     // bytes per row, as allocated
  size_t       width  = 0;     // texels
  size_t       height = 0;
  unsigned int format = GRX_TEX_FORMAT_FLOAT1;
  int          device = 0;
};

std::mutex                          g_tex_mutex;
std::map<grxTextureObject_t, void*> g_objects;   // handle -> the same address

size_t texel_bytes(unsigned int format) {
  switch (format) {
    case GRX_TEX_FORMAT_FLOAT1: return 4;
    case GRX_TEX_FORMAT_FLOAT4: return 16;
    default: return 0;
  }
}

// Rows are padded to the device's cache line so a row start is never a
// straddling access. Not a texture-unit swizzle -- there is no texture unit --
// just the same alignment the allocator gives everything else.
size_t row_pitch(const Device& d, size_t width, unsigned int format) {
  const size_t bytes = width * texel_bytes(format);
  const size_t align = (d.prop.cacheLineSize > 0)
                           ? (size_t)d.prop.cacheLineSize : (size_t)256;
  return (bytes + align - 1) / align * align;
}

}  // namespace
}  // namespace grxcp

extern "C" {

grxError_t grxMallocArray(grxArray_t* array, unsigned int format,
                          size_t width, size_t height) {
  if (!array || width == 0 || height == 0)
    return grxcp::set_error(grxErrorInvalidValue);
  if (grxcp::texel_bytes(format) == 0)
    return grxcp::set_error(grxErrorInvalidValue);

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(grxcp::current_device_index(), &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  const size_t pitch = grxcp::row_pitch(*d, width, format);
  void* data = nullptr;
  e = grxMalloc(&data, pitch * height);
  if (e != grxSuccess) return grxcp::set_error(e);

  auto* a = new grxcp::ArrayImpl();
  a->data   = data;
  a->pitch  = pitch;
  a->width  = width;
  a->height = height;
  a->format = format;
  a->device = grxcp::current_device_index();
  *array = reinterpret_cast<grxArray_t>(a);
  return grxSuccess;
}

grxError_t grxFreeArray(grxArray_t array) {
  if (!array) return grxSuccess;          // CUDA: freeing null is legal
  auto* a = reinterpret_cast<grxcp::ArrayImpl*>(array);
  if (a->data) grxFree(a->data);
  delete a;
  return grxSuccess;
}

grxError_t grxArrayGetInfo(grxArray_t array, void** devicePtr, size_t* pitch,
                           size_t* width, size_t* height, unsigned int* format) {
  if (!array) return grxcp::set_error(grxErrorInvalidValue);
  auto* a = reinterpret_cast<grxcp::ArrayImpl*>(array);
  if (devicePtr) *devicePtr = a->data;
  if (pitch)     *pitch     = a->pitch;
  if (width)     *width     = a->width;
  if (height)    *height    = a->height;
  if (format)    *format    = a->format;
  return grxSuccess;
}

grxError_t grxMemcpy2DToArray(grxArray_t array, size_t xOffsetTexels,
                              size_t yOffset, const void* src, size_t spitch,
                              size_t widthTexels, size_t height,
                              grxMemcpyKind kind) {
  if (!array || !src) return grxcp::set_error(grxErrorInvalidValue);
  auto* a = reinterpret_cast<grxcp::ArrayImpl*>(array);
  const size_t tb = grxcp::texel_bytes(a->format);

  // Bounds first, and in TEXELS, because that is the unit the caller is
  // thinking in. CUDA's cudaMemcpy2DToArray takes a width in BYTES next to
  // offsets in bytes-for-x and rows-for-y, which is a trap this does not
  // reproduce -- see the declaration.
  if (xOffsetTexels + widthTexels > a->width ||
      yOffset + height > a->height)
    return grxcp::set_error(grxErrorInvalidValue);

  const auto* s = static_cast<const unsigned char*>(src);
  auto* dst = static_cast<unsigned char*>(a->data);
  for (size_t row = 0; row < height; ++row) {
    grxError_t e = grxMemcpy(dst + (yOffset + row) * a->pitch +
                                 xOffsetTexels * tb,
                             s + row * spitch, widthTexels * tb, kind);
    if (e != grxSuccess) return grxcp::set_error(e);
  }
  return grxSuccess;
}

grxError_t grxCreateTextureObject(grxTextureObject_t* object, grxArray_t array,
                                  const grxTextureDesc_t* desc) {
  if (!object || !array || !desc) return grxcp::set_error(grxErrorInvalidValue);
  auto* a = reinterpret_cast<grxcp::ArrayImpl*>(array);

  for (int i = 0; i < 2; ++i)
    if (desc->addressMode[i] > GRX_TEX_ADDRESS_BORDER)
      return grxcp::set_error(grxErrorInvalidValue);
  if (desc->filterMode > GRX_TEX_FILTER_LINEAR)
    return grxcp::set_error(grxErrorInvalidValue);

  grx_texture_desc d{};
  d.abi_version = GRX_TEXTURE_ABI_VERSION;
  d.format      = a->format;
  d.width       = (uint32_t)a->width;
  d.height      = (uint32_t)a->height;
  d.pitch       = (uint32_t)a->pitch;
  d.address_x   = desc->addressMode[0];
  d.address_y   = desc->addressMode[1];
  d.filter      = desc->filterMode;
  d.normalized  = desc->normalizedCoords ? 1u : 0u;
  d.data        = (uint64_t)(uintptr_t)a->data;
  for (int i = 0; i < 4; ++i) d.border[i] = desc->borderColor[i];

  // The descriptor lives on the DEVICE because the device is what reads it.
  void* dev = nullptr;
  grxError_t e = grxMalloc(&dev, sizeof d);
  if (e != grxSuccess) return grxcp::set_error(e);
  e = grxMemcpy(dev, &d, sizeof d, grxMemcpyHostToDevice);
  if (e != grxSuccess) { grxFree(dev); return grxcp::set_error(e); }

  {
    std::lock_guard<std::mutex> lock(grxcp::g_tex_mutex);
    grxcp::g_objects[(grxTextureObject_t)(uintptr_t)dev] = dev;
  }
  *object = (grxTextureObject_t)(uintptr_t)dev;
  return grxSuccess;
}

grxError_t grxDestroyTextureObject(grxTextureObject_t object) {
  if (object == 0) return grxSuccess;
  void* dev = nullptr;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_tex_mutex);
    auto it = grxcp::g_objects.find(object);
    // Tracked rather than trusted. The handle is a device address, so a stale
    // or invented one is indistinguishable from a live one by inspection --
    // freeing whatever it points at would be a use-after-free with the caller's
    // name on it.
    if (it == grxcp::g_objects.end())
      return grxcp::set_error(grxErrorInvalidResourceHandle);
    dev = it->second;
    grxcp::g_objects.erase(it);
  }
  return grxFree(dev);
}

}  // extern "C"
