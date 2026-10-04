// The GRX930 team's register model, as a build WITH the photonic tile.
//
// npu_shim_adapter.h installs their shim as it comes: a c930 with no tile,
// which computes an exact product and refuses any START that asks for an
// impairment. That is the right default and it cannot report an emulated GEMM,
// because it does not compute one.
//
// Their shim can also be the other build. Compiled with NPU_DPI_WITH_PTA and
// linked with third_party/grx930/pta_tile_model.c -- which tests/CMakeLists.txt
// does for any test that includes this header -- npu_dpi_set_tile() puts the
// reference error model behind the same register map. Then a START with
// PTA_IMPAIR set runs, and C is the impaired result.
//
// THE MODEL IS THE ONE THE RTL IS HELD TO. pta_tile_model.c is what grx930's
// P gates compare the tile's RTL against bit for bit, and what
// third_party/grx930/pta_vectors.txt pins. So a result through this adapter is
// a statement about the error model's arithmetic. It is still not a statement
// about a c930, and nothing through it may be reported as hardware: the device
// says GRX_BACKEND_MODEL, and the tile says it is an emulation of kind 3, the
// model on its own, which no RTL build reports.

#ifndef GRXCP_TESTS_NPU_TILE_ADAPTER_H
#define GRXCP_TESTS_NPU_TILE_ADAPTER_H

#include "npu_shim_adapter.h"

extern "C" {
#include "pta_tile_model.h"
}

namespace grxtest {

// Install the shim as a tile build. BEFORE the first grx call, for the same
// reason npu_shim_install has to be. Returns false if the shim was compiled
// without the model, in which case nothing was installed.
inline bool npu_tile_install() {
  if (npu_dpi_set_tile(NPU_DPI_TILE_MODEL) != 0) return false;
  return npu_shim_install();
}

}  // namespace grxtest

#endif  // GRXCP_TESTS_NPU_TILE_ADAPTER_H
