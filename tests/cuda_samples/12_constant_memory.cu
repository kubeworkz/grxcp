// __constant__ memory and cudaMemcpyToSymbol -- the way a CUDA convolution
// gets its filter taps, and the construct 09_stencil_1d.cu would normally use
// for its coefficients.
//
// This is also the sample that documents an asymmetry, because the platform has
// one and hiding it would be worse than the feature not existing:
//
//   __constant__  works. The device cannot write it, so the runtime's own copy
//                 of the image is authoritative and a read-back is exact.
//   __device__    refuses on read-back. A kernel CAN write one, the driver
//                 gives the host no way to see what it wrote, and returning the
//                 stale host copy would be a wrong answer.
//
// docs/designs/cuda_mapping.md section 7.23.

#include <grx/grx_cuda_compat.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define TAPS 7

__constant__ float c_filter[TAPS];
__constant__ int   c_bias;

// A writable device global, here to be REFUSED rather than to be used.
__device__ int d_counter = 1234;

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #call, \
                   cudaGetErrorString(err));                               \
      return EXIT_FAILURE;                                                 \
    }                                                                      \
  } while (0)

__global__ void convolve(const float* in, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float acc = 0.0f;
  for (int t = 0; t < TAPS; ++t) {
    const int j = i + t - TAPS / 2;
    if (j >= 0 && j < n) acc += c_filter[t] * in[j];
  }
  out[i] = acc + (float)c_bias + (float)d_counter;
}

int main() {
  const int n = 256;
  const int threads = 32;

  std::vector<float> h_filter(TAPS), h_in(n), h_out(n, -1.0f);
  for (int t = 0; t < TAPS; ++t) h_filter[t] = (float)(t + 1);
  for (int i = 0; i < n; ++i) h_in[i] = (float)(i % 5);
  const int h_bias = 100;

  // The whole point: the taps live on the host until this call.
  CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_filter.data(),
                                TAPS * sizeof(float), 0,
                                cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(c_bias, &h_bias, sizeof(int), 0,
                                cudaMemcpyHostToDevice));

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_in, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  convolve<<<(n + threads - 1) / threads, threads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
  };

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    float want = 0.0f;
    for (int t = 0; t < TAPS; ++t) {
      const int j = i + t - TAPS / 2;
      if (j >= 0 && j < n) want += h_filter[t] * h_in[j];
    }
    want += (float)h_bias + 1234.0f;
    if (fabsf(h_out[i] - want) > 1e-3f) {
      if (bad < 3)
        std::printf("  [%d] got %g want %g\n", i, (double)h_out[i],
                    (double)want);
      ++bad;
    }
  }
  check(bad == 0, "the kernel sees the taps the host wrote after the launch "
                  "was compiled");

  // A second write, AFTER the module has been loaded. This is the expensive
  // path -- it invalidates the module and the next launch reloads it -- and it
  // has to produce a different answer, or the first check only proved that the
  // initializers happened to be right.
  std::vector<float> h_filter2(TAPS, 0.0f);
  h_filter2[TAPS / 2] = 2.0f;             // an impulse: out = 2 * in
  CUDA_CHECK(cudaMemcpyToSymbol(c_filter, h_filter2.data(),
                                TAPS * sizeof(float), 0,
                                cudaMemcpyHostToDevice));
  convolve<<<(n + threads - 1) / threads, threads>>>(d_in, d_out, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bad = 0;
  for (int i = 0; i < n; ++i) {
    const float want = 2.0f * h_in[i] + (float)h_bias + 1234.0f;
    if (fabsf(h_out[i] - want) > 1e-3f) {
      if (bad < 3)
        std::printf("  [%d] got %g want %g\n", i, (double)h_out[i],
                    (double)want);
      ++bad;
    }
  }
  check(bad == 0, "a second write after the module was loaded reaches the "
                  "device too");

  // Read-back of a __constant__ symbol is exact.
  std::vector<float> readback(TAPS, -1.0f);
  CUDA_CHECK(cudaMemcpyFromSymbol(readback.data(), c_filter,
                                  TAPS * sizeof(float), 0,
                                  cudaMemcpyDeviceToHost));
  bool same = true;
  for (int t = 0; t < TAPS; ++t) if (readback[t] != h_filter2[t]) same = false;
  check(same, "cudaMemcpyFromSymbol returns what was written");

  // And a partial window, with an offset.
  float one = -1.0f;
  CUDA_CHECK(cudaMemcpyFromSymbol(&one, c_filter, sizeof(float),
                                  (TAPS / 2) * sizeof(float),
                                  cudaMemcpyDeviceToHost));
  check(one == 2.0f, "an offset read returns that element and not the first");

  // The refusals, which are the honest half of this sample.
  check(cudaMemcpyFromSymbol(&one, d_counter, sizeof(int), 0,
                             cudaMemcpyDeviceToHost) == cudaErrorNotSupported,
        "reading back a __device__ symbol is refused, not approximated");

  int not_a_symbol = 0;
  check(cudaMemcpyToSymbol(&not_a_symbol, &h_bias, sizeof(int), 0,
                           cudaMemcpyHostToDevice) != cudaSuccess,
        "a host variable that is not a device symbol is rejected");

  check(cudaMemcpyToSymbol(c_bias, &h_bias, sizeof(int) * 4, 0,
                           cudaMemcpyHostToDevice) != cudaSuccess,
        "a write past the end of a symbol is rejected");

  size_t sz = 0;
  CUDA_CHECK(cudaGetSymbolSize(&sz, c_filter));
  check(sz == TAPS * sizeof(float), "cudaGetSymbolSize reports the real size");

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));

  if (failures) { std::printf("FAILED (%d)\n", failures); return 1; }
  std::printf("PASSED  __constant__ memory through cudaMemcpyToSymbol\n");
  return 0;
}
