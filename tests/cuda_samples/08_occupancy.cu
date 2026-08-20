// The occupancy API: cudaFuncGetAttributes, the occupancy calculator, and
// cudaOccupancyMaxPotentialBlockSize picking a launch shape.
//
// This is what a CUDA program does when it does not want to hard-code a block
// size. It is also the sample most likely to expose a platform reporting
// numbers it does not really have, which is why it checks them for internal
// consistency rather than printing them.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__global__ void square(float* data, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] * data[i];
}

int main() {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void*)square));

  int minGridSize = 0, blockSize = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                                (const void*)square, 0, 0));

  int activeBlocks = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &activeBlocks, (const void*)square, blockSize, 0));

  std::printf("  maxThreadsPerBlock=%d numRegs=%d sharedSizeBytes=%zu\n",
              attr.maxThreadsPerBlock, attr.numRegs, attr.sharedSizeBytes);
  std::printf("  suggested blockSize=%d, minGridSize=%d, blocks/SM=%d\n",
              blockSize, minGridSize, activeBlocks);

  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL %s\n", what); ++failures; }
  };

  check(blockSize > 0 && blockSize <= prop.maxThreadsPerBlock,
        "the suggested block size is within the device's own limit");
  check(blockSize % prop.warpSize == 0,
        "the suggested block size is a whole number of warps");
  check(activeBlocks > 0, "at least one block fits per SM");
  check((long)activeBlocks * blockSize <=
            (long)prop.maxWarpsPerMultiProcessor * prop.warpSize,
        "blocks/SM times block size fits in the SM's resident threads");
  check(minGridSize > 0, "the suggested grid covers the machine");

  // And the shape it suggested has to actually launch and compute.
  const int n = blockSize * 4;
  std::vector<float> h(n);
  for (int i = 0; i < n; ++i) h[i] = (float)(i % 7);

  float* d = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  square<<<(n + blockSize - 1) / blockSize, blockSize>>>(d, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> out(n, -1.0f);
  CUDA_CHECK(cudaMemcpy(out.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i)
    if (fabsf(out[i] - h[i] * h[i]) > 1e-4f) { ++failures; break; }

  CUDA_CHECK(cudaFree(d));

  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED  occupancy API, and the shape it suggested runs\n");
  return 0;
}
