// GRXCP — DXA descriptor slots (tensor maps).
//
// The descriptor lives in device configuration registers, so programming one
// is a sequence of DCR writes. They go through vx_enqueue_dcr_write rather
// than an immediate write, which is what makes the slot's contents ordered
// with respect to the launches that follow on the same stream. A kernel that
// issues against a slot the host is still programming would read whatever was
// in flight, and nothing would report it.
//
// ON DUPLICATING THE ENCODER. The sysroot ships vortex::dxa::program_Nd in
// dxa.h, which encodes the same registers. This file does not call it: that
// version writes immediately through the v1 driver API, and mixing v1 and v2
// in the runtime is a boundary GRXCP does not cross (AGENTS.md section 2). So
// the field layout is written out a second time here, which is a real risk --
// the mitigation is that tests/kernels/dxa/ moves actual data through an
// actual descriptor and checks it element by element, so a divergence between
// the two encoders fails a test rather than passing quietly. dxa.h is the
// reference; diff against it when either changes.
//
// dxa_meta.h, which defines the META field widths, is NOT installed into the
// sysroot -- the widths below are transcribed from it. Installing that header
// would remove the transcription; it is a small upstream ask.

#include "internal.h"

#include <grx/grx_tensormap.h>

#include <VX_types.h>

#include <vector>

namespace {

// META field layout, from sw/common/dxa_meta.h.
constexpr uint32_t kMetaDimLsb    = 0;   // 3 bits: rank
constexpr uint32_t kMetaElemszLsb = 3;   // 2 bits: log2(element bytes)
constexpr uint32_t kMetaLayoutLsb = 5;   // 2 bits: destination layout

constexpr uint32_t pack_2x16(uint32_t lo, uint32_t hi) {
  return ((hi & 0xffffu) << 16) | (lo & 0xffffu);
}

uint32_t elem_size_enc(unsigned elem_bytes) {
  uint32_t enc = 0;
  for (unsigned v = elem_bytes; v > 1; v >>= 1) ++enc;
  return enc;
}

bool has_async_copy(const grxDeviceProp_t& prop) {
  return (prop.capabilities & GRX_CAP_ASYNC_COPY) != 0;
}

struct DcrWrite { uint32_t off, val; };

}  // namespace

extern "C" {

grxError_t grxTensorMapGetSlotCount(int* count, int device) {
  if (!count) return grxcp::set_error(grxErrorInvalidValue);
  *count = 0;

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  if (has_async_copy(d->prop)) *count = VX_DCR_DXA_DESC_COUNT;
  return grxSuccess;
}

grxError_t grxTensorMapProgramAsync(const grxTensorMapDesc_t* desc,
                                    grxStream_t stream) {
  if (!desc || !desc->base) return grxcp::set_error(grxErrorInvalidValue);

  grxcp::Mapping m{};
  if (!grxcp::lookup_device_pointer(desc->base, &m))
    return grxcp::set_error(grxErrorInvalidDevicePointer);

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(m.device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  if (!has_async_copy(d->prop)) return grxcp::set_error(grxErrorNotSupported);

  // The DXA worker's bus master bypasses the per-core MMU, so a descriptor's
  // base is a PHYSICAL address. On a backend with virtual memory an ordinary
  // allocation's device pointer is not one, and the engine would stream from
  // whatever that bit pattern names -- into the caller's shared memory, with
  // nothing reporting it. grxMallocPhysical exists for exactly this, and it is
  // required here whenever the device has an MMU to bypass.
  if (d->prop.unifiedAddressing && !m.physical)
    return grxcp::set_error(grxErrorInvalidDevicePointer);

  if (desc->slot < 0 || desc->slot >= VX_DCR_DXA_DESC_COUNT)
    return grxcp::set_error(grxErrorInvalidValue);
  if (desc->rank == 0 || desc->rank > 5)
    return grxcp::set_error(grxErrorInvalidValue);
  if (desc->rank > 2) return grxcp::set_error(grxErrorNotSupported);

  const unsigned eb = desc->elementBytes;
  if (eb != 1 && eb != 2 && eb != 4 && eb != 8)
    return grxcp::set_error(grxErrorInvalidValue);

  for (unsigned i = 0; i < desc->rank; ++i) {
    if (desc->size[i] == 0 || desc->tile[i] == 0)
      return grxcp::set_error(grxErrorInvalidValue);
    // A tile LARGER than the array is legal and useful: the engine pads the
    // overhang with the descriptor's fill value rather than reading past the
    // end. A blocked GEMM depends on it -- the edge tiles of a matrix whose
    // dimensions are not multiples of the tile arrive zero-padded, and a zero
    // contributes nothing to an accumulation. Only the 16-bit field is a
    // limit here.
    if (desc->tile[i] > 0xffffu) return grxcp::set_error(grxErrorInvalidValue);
  }

  // Bounds. A descriptor that describes more array than was allocated is a
  // DMA engine reading into someone else's buffer, and the engine will not
  // complain -- it has no idea where the allocation ends.
  //
  // The reach is computed from the TILE, not the array, because the engine
  // bounds-checks only the outer dimensions: a tile that overhangs along
  // dimension 0 reads straight past size0 into whatever follows (measured --
  // see tests/kernels/dxa/). So the last row must have room for a full tile,
  // or an edge tile reads off the end of the allocation.
  const uint64_t row_elems =
      (desc->tile[0] > desc->size[0]) ? desc->tile[0] : desc->size[0];
  uint64_t span = 0;
  if (desc->rank == 1) {
    span = row_elems * eb;
  } else {
    if (desc->strideBytes[0] < (uint64_t)desc->size[0] * eb)
      return grxcp::set_error(grxErrorInvalidValue);
    span = (uint64_t)(desc->size[1] - 1) * desc->strideBytes[0] + row_elems * eb;
  }
  if (span > m.size) return grxcp::set_error(grxErrorInvalidValue);

  vx_queue_h q = nullptr;
  e = grxcp::resolve_stream(stream, m.device, &q, nullptr);
  if (e != grxSuccess) return grxcp::set_error(e);

  const uint64_t base = (uint64_t)(uintptr_t)desc->base;
  const uint32_t meta = ((desc->rank & 0x7u) << kMetaDimLsb) |
                        ((elem_size_enc(eb) & 0x3u) << kMetaElemszLsb) |
                        (((uint32_t)desc->layout & 0x3u) << kMetaLayoutLsb);

  std::vector<DcrWrite> writes;
  writes.push_back({VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base & 0xffffffffu)});
  writes.push_back({VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base >> 32)});
  writes.push_back({VX_DCR_DXA_DESC_SIZE0_OFF,   desc->size[0]});
  if (desc->rank >= 2) {
    writes.push_back({VX_DCR_DXA_DESC_SIZE1_OFF,   desc->size[1]});
    writes.push_back({VX_DCR_DXA_DESC_STRIDE0_OFF, desc->strideBytes[0]});
  }
  writes.push_back({VX_DCR_DXA_DESC_META_OFF,     meta});
  writes.push_back({VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1});
  if (desc->rank >= 2) writes.push_back({VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1});
  writes.push_back({VX_DCR_DXA_DESC_TILESIZE01_OFF,
                    pack_2x16(desc->tile[0], desc->rank >= 2 ? desc->tile[1] : 0)});
  writes.push_back({VX_DCR_DXA_DESC_CFILL_OFF, 0});
  if (desc->smemStrideBytes)
    writes.push_back({VX_DCR_DXA_DESC_SMEM_STRIDE_OFF, desc->smemStrideBytes});

  std::vector<vx_event_h> waits;
  grxcp::collect_wait_events(stream, m.device, &waits);

  // Queues execute in order, so only the first write needs the stream's wait
  // events and only the last needs to produce one.
  const uint32_t dcr_base =
      VX_DCR_DXA_DESC_BASE + (uint32_t)desc->slot * VX_DCR_DXA_DESC_STRIDE;
  vx_event_h completion = nullptr;
  for (size_t i = 0; i < writes.size(); ++i) {
    const bool first = (i == 0);
    const bool last  = (i + 1 == writes.size());
    vx_result_t r = vx_enqueue_dcr_write(
        q, dcr_base + writes[i].off, writes[i].val,
        first ? (uint32_t)waits.size() : 0,
        (first && !waits.empty()) ? waits.data() : nullptr,
        last ? &completion : nullptr);
    if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));
  }

  grxcp::set_stream_last_event(stream, m.device, completion);
  return grxSuccess;
}

grxError_t grxTensorMapProgram(const grxTensorMapDesc_t* desc) {
  grxError_t e = grxTensorMapProgramAsync(desc, nullptr);
  if (e != grxSuccess) return e;
  return grxcp::sync_stream(nullptr, grxcp::current_device_index());
}

}  // extern "C"
