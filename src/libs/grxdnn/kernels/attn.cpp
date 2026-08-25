// The one piece of attention that is not already a library call.
//
// scores = Q Kᵀ is a batched GEMM, softmax is grxdnnSoftmaxForward, and
// out = P V is another batched GEMM. What none of those can do is stop a query
// from seeing a key that comes after it, so that is what is here.
//
// WHY A SEPARATE KERNEL RATHER THAN A FLAG ON SOFTMAX.
//
// Folding the mask into dnn_softmax would save a launch, and softmax is already
// reading the row it would have to modify. It is not done because dnn_softmax is
// gated — five shapes, two numerical controls — and the gate is a statement
// about the kernel that ships. Growing a mode argument makes it a statement
// about one of two kernels that share a name. This costs one launch over a
// matrix that the following softmax reads anyway, so the traffic is already
// paid for; when the flash-style fused form arrives it replaces all three
// launches at once and this file goes with them.
//
// -FLT_MAX rather than -inf: dev_exp clamps at -88 and returns exactly zero
// below it, so a masked entry contributes nothing to the sum. An actual
// infinity would give inf - inf = NaN in the max-subtraction the moment a row
// were entirely masked. No row here ever is — element (i, i) is always visible —
// but the kernel should not depend on the caller never asking.

#include <grx/device/grx_cg.h>
#include <grx/device/grx_cycles.h>

#include "../dnn_abi.h"
#include "dnn_device.h"

namespace { using grxdnn_dev::kNegInf; }

// One warp per row of the [batch*heads*seqLen, seqLen] score matrix, matching
// dnn_softmax so the two agree on which lane touches which element.
//
// The row index carries both meanings at once: r / seq_len is the head, and
// r % seq_len is the query position within it. That is what makes the mask a
// pure function of the flat row index and needs no per-head dispatch.
__global__ void dnn_causal_mask(grxdnn_mask_args* __UNIFORM__ arg) {
  if (arg->abi_version != GRXDNN_ABI_VERSION) return;
  grx::cycle_probe probe(reinterpret_cast<grxCycleSlot*>(arg->cycles));

  float* s = reinterpret_cast<float*>(arg->scores);
  const uint32_t rows = arg->rows;
  const uint32_t seq_len = arg->seq_len;
  const int32_t  ld = arg->ld;

  const uint32_t w = grx::warp_size();
  const uint32_t lane = grx::lane_id();
  const uint32_t warps_per_block = (blockDim.x + w - 1u) / w;
  const uint32_t row0 = blockIdx.x * warps_per_block + (threadIdx.x / w);
  const uint32_t stride = gridDim.x * warps_per_block;

  for (uint32_t r = row0; r < rows; r += stride) {
    const uint32_t query = r % seq_len;      // position of this query in its head
    float* row = s + (size_t)r * (size_t)ld;
    // Keys strictly after the query are in the future. Lanes start past the
    // diagonal rather than testing every element, so a row near the start of a
    // long sequence does almost no work.
    for (uint32_t j = query + 1u + lane; j < seq_len; j += w)
      row[j] = kNegInf;
  }
}
