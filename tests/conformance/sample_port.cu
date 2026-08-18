// A deliberately awkward CUDA program: mostly portable, with three calls that
// GRXCP cannot provide. grxify must rewrite the first group and name the second.
#include <cuda_runtime.h>
#include <cstdio>

__global__ void saxpy(float a, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  float *dx = nullptr, *dy = nullptr;
  cudaMalloc((void**)&dx, 1024);
  cudaMalloc((void**)&dy, 1024);
  cudaMemset(dy, 0, 1024);

  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, s);

  cudaMemcpyAsync(dx, dy, 1024, cudaMemcpyDeviceToDevice, s);
  cudaEventRecord(stop, s);
  cudaStreamSynchronize(s);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, start, stop);
  std::printf("%f ms on %s\n", ms, prop.name);

  // These three have no GRXCP equivalent and must be reported.
  cudaGraph_t g;
  cudaGraphCreate(&g, 0);
  cudaMemcpyToSymbol(nullptr, dx, 4);
  cudaStreamAddCallback(s, nullptr, nullptr, 0);

  // Declared but refuses at runtime -- also worth reporting.
  cudaHostRegister(dx, 1024, 0);

  cudaFree(dx);
  cudaFree(dy);
  cudaStreamDestroy(s);
  return 0;
}
