// Two kernels that are the same algorithm, differing only in which barrier
// they call. Built as one module so a single run compares them.
//
//   guarded_bad  -- upstream's __syncthreads(), i.e. a bare vx_barrier
//   guarded_good -- GRXCP's, through the convergent wrapper
//
// The divergence in front of the barrier is the point. Both kernels write
// out[i] = in[mirror] through shared memory, and every thread reaches the
// barrier in the source; whether every thread reaches it in the OBJECT CODE is
// what is being measured.

#include <grx/device/grx_device.h>

#include <vx_spawn2.h>

// grx_device.h replaced __syncthreads() with the convergent form. Recover the
// upstream spelling under a different name so both can appear in one module.
#define __grx_upstream_syncthreads() \
  vx_barrier(get_local_group_id(), get_num_sub_groups())

struct args_t {
  uint64_t in;
  uint64_t out;
  uint32_t n;
  uint32_t pad;
};

namespace {

// The parameters are taken BY VALUE, which is not incidental. Pass `n` through
// the argument struct and the compiler reloads it for the second comparison,
// the two branches stop being recognizably the same test, and the tail
// duplication that causes the bug does not happen -- the barrier stays at the
// join and this repro reports a fix that has not occurred. Taking them by value
// reproduces the shape grxcc generates: an always_inline body called from a
// wrapper that unpacks the struct once, which is also the shape every ordinary
// CUDA kernel has.
template <bool Convergent>
__forceinline__ void body(const float* in, float* out, unsigned n) {
  float* s = grx::shared_memory<float>();

  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned t = threadIdx.x;

  if (i < n) s[t] = in[i];
  if (Convergent) __syncthreads(); else __grx_upstream_syncthreads();
  if (i < n) out[i] = s[blockDim.x - 1u - t];
}

}  // namespace

__global__ void guarded_bad(args_t* __UNIFORM__ a) {
  body<false>((const float*)(uintptr_t)a->in, (float*)(uintptr_t)a->out, a->n);
}
__global__ void guarded_good(args_t* __UNIFORM__ a) {
  body<true>((const float*)(uintptr_t)a->in, (float*)(uintptr_t)a->out, a->n);
}
