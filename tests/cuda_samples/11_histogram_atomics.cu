// histogram with atomicAdd -- and the one sample in this directory that is NOT
// expected to build on this configuration.
//
// It is here deliberately. GRX-G100's A extension is a BUILD OPTION, and this
// sysroot has VX_CFG_EXT_A_ENABLE off (cuda_mapping.md section 7.16). The
// device toolchain still compiles -march=rv64imafd, so clang will happily lower
// an atomic to an AMO that the hardware then aborts on -- with no message, no
// line, and nothing in the stack having said the word "atomic".
//
// So the platform's job here is to REFUSE AT COMPILE TIME with a message that
// names the reason, and the gate's job is to check that it does. On a sysroot
// built with the extension enabled this sample compiles and runs, and the gate
// checks that instead. Either way nobody meets the abort.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define NBINS 16

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__global__ void histogram(const unsigned int* in, unsigned int* bins, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&bins[in[i] % NBINS], 1u);
}

int main() {
  const int n = 1024;
  const int threads = 32;

  std::vector<unsigned int> h_in(n);
  for (int i = 0; i < n; ++i) h_in[i] = (unsigned int)(i * 7 + 3);

  unsigned int *d_in = nullptr, *d_bins = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(unsigned int)));
  CUDA_CHECK(cudaMalloc((void**)&d_bins, NBINS * sizeof(unsigned int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(unsigned int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_bins, 0, NBINS * sizeof(unsigned int)));

  histogram<<<(n + threads - 1) / threads, threads>>>(d_in, d_bins, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned int> h_bins(NBINS, 0);
  CUDA_CHECK(cudaMemcpy(h_bins.data(), d_bins, NBINS * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned int> want(NBINS, 0);
  for (int i = 0; i < n; ++i) ++want[h_in[i] % NBINS];

  int bad = 0;
  for (int b = 0; b < NBINS; ++b)
    if (h_bins[b] != want[b]) {
      if (bad < 4)
        std::printf("  bin %d got %u want %u\n", b, h_bins[b], want[b]);
      ++bad;
    }

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_bins));

  if (bad) { std::printf("FAILED: %d of %d bins wrong\n", bad, NBINS); return 1; }
  std::printf("PASSED  histogram over %d elements into %d bins\n", n, NBINS);
  return 0;
}
