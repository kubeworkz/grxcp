// Kernels with known-good and known-bad memory behaviour, for the grx-sanitize
// gate.
//
// The bugs here are deliberate and each one is a single line, so the gate can
// assert on the line number the sanitizer reports. Keep the line numbers stable
// when editing this file, or update ci/run_real.sh with it -- the gate greps
// for the file and checks the reported line is one of the planted ones.
//
// Every wrong access is a plain indexed store or load through a pointer whose
// bound the compiler cannot see, which is exactly the class of bug that
// survives review and then corrupts something three kernels later.

#include <grx/device/grx_device.h>

#include "common.h"

// Correct: the negative control. Every thread writes inside the allocation and
// nowhere else, so a sanitizer that reports anything here is reporting noise.
__global__ void san_clean(san_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < arg->n) out[tid] = tid;
}

// Wrong: one element past the end, from one thread. The classic off-by-one --
// the bound is right in the guard and wrong in the index.
__global__ void san_oob_write(san_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < arg->n) out[tid] = tid;
  if (tid == 0) out[arg->n] = 0xbadbad;      // PLANTED: one past the end
}

// Wrong: reads before the start. Lands in the previous allocation's redzone,
// which is the reason every allocation gets one.
__global__ void san_oob_read(san_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    const uint32_t v = out[-4];              // PLANTED: before the start
    out[0] = v;
  }
}

// Wrong only because of what the host did: the buffer was freed before the
// launch. The access itself is in bounds.
__global__ void san_use_after_free(san_args* __UNIFORM__ arg) {
  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out);
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < arg->n) out[tid] = 0x5eeded;     // PLANTED: buffer already freed
}

// Wrong: past the end of the CTA's shared-memory slot. The launch asks for
// blockDim.x words; this writes word blockDim.x. The dispatcher rounds the
// per-CTA stride up, so the store lands in slack that belongs to no one and
// corrupts nothing today -- and would corrupt the next CTA's slot the moment
// the block size changed.
__global__ void san_oob_shared(san_args* __UNIFORM__ arg) {
  uint32_t* out  = reinterpret_cast<uint32_t*>(arg->out);
  uint32_t* smem = grx::shared_memory<uint32_t>();

  smem[threadIdx.x] = threadIdx.x;
  if (threadIdx.x == 0) smem[blockDim.x] = 0xfeed;   // PLANTED: past sharedMem
  __syncthreads();

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < arg->n) out[tid] = smem[threadIdx.x];
}
