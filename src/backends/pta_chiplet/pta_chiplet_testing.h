// pta_chiplet_testing.h -- the seam that lets a model stand in for the PTA
// chiplet the RUNTIME enumerates.
//
// The same shape as npu_c930_testing.h, with one difference that matters: the
// NPU has a hardware path and the seam replaces it; the chiplet has none, so
// the seam is the only way a chiplet is ever enumerated. A runtime built with
// GRXCP_ENABLE_PTA and nothing attached here has no PTA device, on any
// machine, because there is no chiplet and no call in the GPU's driver that
// would find one (pta_chiplet.h).
//
// A MODEL IS NOT HARDWARE. A device reached this way reports GRX_BACKEND_MODEL
// and names itself a model in grxDeviceProp_t.name. Those are derived from
// this call having been made, not from a constant, so nothing attached here can
// claim to be a chip. Per AGENTS.md no result obtained through it may be
// reported as the chiplet working.

#ifndef PTA_CHIPLET_TESTING_H
#define PTA_CHIPLET_TESTING_H

#include "pta_chiplet.h"

#ifdef __cplusplus
extern "C" {
#endif

// Install the chiplet's window: a model of its register page.
//
// MUST be called before the first call that initialises the runtime. Device
// enumeration runs once; after it has run this has nothing to attach to and
// says so by returning 0. Passing null for both hooks clears a previously
// installed model. Returns 1 if the model was installed.
int grxcp_pta_attach_model_for_testing(pta_chiplet_read_fn read32,
                                       pta_chiplet_write_fn write32,
                                       void* ctx);

// Install the link: the function call that stands in for the command path.
// Separate from the window because the two are separate on the board -- the
// window is CXL.io through the GPU's BAR and the link is die to die. A chiplet
// with a window and no link is a real state: it is enumerated and reported,
// and a GEMM on it is refused. Same rule: before enumeration, or it returns 0.
int grxcp_pta_attach_link_for_testing(pta_chiplet_submit_fn submit, void* ctx);

// Which GPU the chiplet is in the package of: the device whose memory its
// operands live in. Device 0 unless this says otherwise. It has to be a GPU
// that enumeration finds, or the chiplet is not enumerated at all -- a tile
// with no parent has nothing to compute on. Before enumeration, or 0.
int grxcp_pta_set_parent_for_testing(int gpu_index);

// True once a model has been installed through the seam above.
int grxcp_pta_model_is_attached(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PTA_CHIPLET_TESTING_H
