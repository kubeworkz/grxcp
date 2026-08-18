// A CUDA host program using only the portable subset. grxify translates it and
// the result must COMPILE, LINK and RUN against libgrxrt -- which is the real
// test of the compatibility header, not just of the rename table.
//
// Kernels are deliberately absent: launching one needs a device toolchain, and
// this test is about the host API surface.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call)                                                     \
  do {                                                                  \
    cudaError_t e_ = (call);                                            \
    if (e_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "%s -> %s\n", #call, cudaGetErrorString(e_));\
      return 1;                                                         \
    }                                                                   \
  } while (0)

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(cudaSetDevice(0));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));

  constexpr size_t N = 2048;
  std::vector<unsigned char> host(N), back(N, 0);
  for (size_t i = 0; i < N; ++i) host[i] = (unsigned char)(i * 7 + 3);

  void *da = nullptr, *db = nullptr;
  CHECK(cudaMalloc(&da, N));
  CHECK(cudaMalloc(&db, N));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  CHECK(cudaEventRecord(start, stream));
  CHECK(cudaMemcpyAsync(da, host.data(), N, cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(db, da, N, cudaMemcpyDeviceToDevice, stream));
  CHECK(cudaMemcpyAsync(back.data(), db, N, cudaMemcpyDeviceToHost, stream));
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaStreamSynchronize(stream));

  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  if (std::memcmp(host.data(), back.data(), N) != 0) {
    std::fprintf(stderr, "data mismatch after round trip\n");
    return 1;
  }

  cudaPointerAttributes attr;
  CHECK(cudaPointerGetAttributes(&attr, da));

  size_t freeBytes = 0, totalBytes = 0;
  CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaDeviceSynchronize());

  std::printf("PASSED on %s (%d SMs, warp %d, %.3f ms, %zu MiB free)\n",
              prop.name, prop.multiProcessorCount, prop.warpSize, ms,
              (size_t)(freeBytes >> 20));
  return 0;
}
