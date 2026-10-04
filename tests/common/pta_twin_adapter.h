// The PTA chiplet's digital twin, as the device the runtime enumerates.
//
// src/backends/pta_chiplet/pta_chiplet_twin.c presents the chiplet's register
// map with grx930's error model behind it. This installs one through the seam
// in pta_chiplet_testing.h, so that grxGetDeviceProperties, grxSetDevice and
// grxblasGemmEx run against a PTA device on a machine that has none -- which is
// every machine, because there is no chiplet.
//
// Two things sit between the twin and the driver's hooks, and both are the
// twin's shape rather than a workaround:
//
//   1. Its clock only advances inside pta_twin_run(), which no host calls. So a
//      read of PTA_STATUS pumps it: that read is where a host's completion poll
//      spends its time, and the driver polls exactly as it would across a link.
//   2. Its command path is a function call, which is what the driver's link
//      hook is too. The two status codes are the same numbers by construction,
//      and the static asserts below are what hold them there.
//
// A MODEL IS NOT HARDWARE. Anything green through this says the runtime and the
// driver use the map correctly and that the answer is the error model's. It
// says nothing about a chiplet, and no result obtained here may be reported as
// one working.

#ifndef GRXCP_TESTS_PTA_TWIN_ADAPTER_H
#define GRXCP_TESTS_PTA_TWIN_ADAPTER_H

#include "pta_chiplet.h"
#include "pta_chiplet_testing.h"

extern "C" {
#include "pta_chiplet_twin.h"
#include "pta_tile_model.h"
}

#include <cstdint>
#include <cstring>

static_assert(PTA_CHIPLET_CMD_PENDING == PTA_TWIN_PENDING &&
              PTA_CHIPLET_CMD_DONE == PTA_TWIN_DONE &&
              PTA_CHIPLET_CMD_REFUSED == PTA_TWIN_REFUSED &&
              PTA_CHIPLET_CMD_LOST == PTA_TWIN_LOST,
              "the driver's link results are the twin's");

namespace grxtest {

// The twin behind the device, for a test that programs it the way another
// holder of the chiplet would: behind the runtime's back.
inline pta_twin*& pta_twin_installed() {
  static pta_twin* t = nullptr;
  return t;
}

// How many of the twin's cycles one PTA_STATUS read advances.
inline uint64_t& pta_twin_quantum() {
  static uint64_t q = 1u << 16;
  return q;
}

inline uint32_t pta_twin_read_hook(void* ctx, uint32_t off) {
  pta_twin* t = static_cast<pta_twin*>(ctx);
  if (off == PTA_CHIPLET_STATUS) pta_twin_run(t, pta_twin_quantum());
  return pta_twin_read32(t, off);
}

inline void pta_twin_write_hook(void* ctx, uint32_t off, uint32_t v) {
  pta_twin_write32(static_cast<pta_twin*>(ctx), off, v);
}

inline int pta_twin_submit_hook(void* ctx, int bank, int M, int N, int K,
                                const int32_t* A, const int32_t* B, int64_t* C,
                                int* status) {
  pta_twin_cmd c;
  std::memset(&c, 0, sizeof(c));
  c.bank = bank;
  c.M = M;
  c.N = N;
  c.K = K;
  c.A = A;
  c.B = B;
  c.C = C;
  c.status = status;
  return pta_twin_submit(static_cast<pta_twin*>(ctx), &c);
}

// Build a twin and install it as the chiplet in device `parent`'s package.
// BEFORE the first grx call. `with_link` false installs the window alone: a
// chiplet that can be read and cannot be given work. Returns false if the twin
// could not be built or the seam refused.
inline bool pta_twin_install(const pta_twin_build& build, int parent = 0,
                             bool with_link = true) {
  pta_twin* t = pta_twin_new(&build);
  if (!t) return false;
  pta_twin_installed() = t;
  if (grxcp_pta_attach_model_for_testing(pta_twin_read_hook, pta_twin_write_hook, t) != 1)
    return false;
  if (with_link && grxcp_pta_attach_link_for_testing(pta_twin_submit_hook, t) != 1)
    return false;
  return grxcp_pta_set_parent_for_testing(parent) == 1;
}

}  // namespace grxtest

#endif  // GRXCP_TESTS_PTA_TWIN_ADAPTER_H
