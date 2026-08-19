// grxBLAS host implementation.
//
// The library ships precompiled device kernels as .vxbin modules and resolves
// them by name, the way a vendor BLAS ships tuned binaries rather than
// compiling at call time. The module is loaded once per handle and per device,
// on first use, so a program that creates a handle and never calls a kernel
// pays nothing.

#include <grx/grxblas.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <unistd.h>   // readlink, for the executable-relative kernel search

#include "sgemm_abi.h"

namespace {

std::mutex  g_path_mutex;
std::string g_kernel_path;

// The first CONFIGURED source wins outright: if someone set a kernel path and
// the module is not there, that is an error, not an invitation to load a
// different build from somewhere else. Searching on past an explicit setting is
// how a developer ends up benchmarking a kernel they thought they had replaced.
// The unconfigured fallbacks exist so a build tree works with no setup at all.
std::vector<std::string> candidate_paths(const char* module_name) {
  std::vector<std::string> out;
  {
    std::lock_guard<std::mutex> lock(g_path_mutex);
    if (!g_kernel_path.empty()) return {g_kernel_path + "/" + module_name};
  }
  if (const char* env = std::getenv("GRXBLAS_KERNEL_PATH"))
    return {std::string(env) + "/" + module_name};

  char self[4096];
  const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  if (n > 0) {
    self[n] = '\0';
    std::string dir(self);
    const size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos)
      out.push_back(dir.substr(0, slash) + "/" + module_name);
  }
  out.emplace_back(module_name);
  return out;
}

struct Context {
  grxStream_t   stream = nullptr;
  grxModule_t   sgemm_module = nullptr;
  grxFunction_t sgemm_fn = nullptr;
  std::string   sgemm_path;   // which file actually got loaded
  grxCycleSlot* probe = nullptr;
  int           probe_capacity = 0;
  std::mutex    mutex;
};

// One warp per block, so one slot per block. Kept as a function because the
// launch geometry below has to agree with it exactly, and two places computing
// the same thing from memory is how they stop agreeing.
int slots_for(int m, int n, int warp_size) {
  if (m <= 0 || n <= 0 || warp_size <= 0) return 0;
  const long long total = (long long)m * n;
  return (int)((total + warp_size - 1) / warp_size);
}

grxblasStatus_t ensure_sgemm(Context& ctx) {
  std::lock_guard<std::mutex> lock(ctx.mutex);
  if (ctx.sgemm_fn) return GRXBLAS_STATUS_SUCCESS;

  grxError_t last = grxSuccess;
  for (const std::string& path : candidate_paths("grxblas_sgemm.vxbin")) {
    grxModule_t mod = nullptr;
    last = grxModuleLoad(&mod, path.c_str());
    if (last != grxSuccess) continue;

    grxFunction_t fn = nullptr;
    last = grxModuleGetFunction(&fn, mod, "sgemm");
    if (last != grxSuccess) { grxModuleUnload(mod); continue; }

    ctx.sgemm_module = mod;
    ctx.sgemm_fn     = fn;
    ctx.sgemm_path   = path;
    return GRXBLAS_STATUS_SUCCESS;
  }

  // Say which file is missing, because "internal error" from a BLAS call is
  // among the least actionable messages a library can produce.
  std::fprintf(stderr,
               "grxblas: cannot load grxblas_sgemm.vxbin (last error: %s).\n"
               "         Set GRXBLAS_KERNEL_PATH or call grxblasSetKernelPath.\n",
               grxGetErrorString(last));
  return GRXBLAS_STATUS_NOT_INITIALIZED;
}

grxblasStatus_t from_grx(grxError_t e) {
  switch (e) {
    case grxSuccess:                   return GRXBLAS_STATUS_SUCCESS;
    case grxErrorMemoryAllocation:     return GRXBLAS_STATUS_ALLOC_FAILED;
    case grxErrorInvalidValue:         return GRXBLAS_STATUS_INVALID_VALUE;
    case grxErrorInvalidDevicePointer: return GRXBLAS_STATUS_INVALID_VALUE;
    case grxErrorNotSupported:         return GRXBLAS_STATUS_NOT_SUPPORTED;
    default:                           return GRXBLAS_STATUS_EXECUTION_FAILED;
  }
}

}  // namespace

extern "C" {

const char* grxblasGetStatusString(grxblasStatus_t s) {
  switch (s) {
    case GRXBLAS_STATUS_SUCCESS:          return "success";
    case GRXBLAS_STATUS_NOT_INITIALIZED:  return "library not initialized (kernels not found?)";
    case GRXBLAS_STATUS_ALLOC_FAILED:     return "allocation failed";
    case GRXBLAS_STATUS_INVALID_VALUE:    return "invalid argument";
    case GRXBLAS_STATUS_ARCH_MISMATCH:    return "device lacks a required capability";
    case GRXBLAS_STATUS_EXECUTION_FAILED: return "kernel execution failed";
    case GRXBLAS_STATUS_NOT_SUPPORTED:    return "not supported";
    case GRXBLAS_STATUS_INTERNAL_ERROR:   return "internal error";
  }
  return "unknown status";
}

grxblasStatus_t grxblasCreate(grxblasHandle_t* handle) {
  if (!handle) return GRXBLAS_STATUS_INVALID_VALUE;
  *handle = reinterpret_cast<grxblasHandle_t>(new Context());
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasDestroy(grxblasHandle_t handle) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  auto* ctx = reinterpret_cast<Context*>(handle);
  if (ctx->sgemm_module) grxModuleUnload(ctx->sgemm_module);
  delete ctx;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSetStream(grxblasHandle_t handle, grxStream_t stream) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Context*>(handle)->stream = stream;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGetStream(grxblasHandle_t handle, grxStream_t* stream) {
  if (!handle || !stream) return GRXBLAS_STATUS_INVALID_VALUE;
  *stream = reinterpret_cast<Context*>(handle)->stream;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSetKernelPath(const char* path) {
  std::lock_guard<std::mutex> lock(g_path_mutex);
  g_kernel_path = path ? path : "";
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSetCycleProbe(grxblasHandle_t handle,
                                     grxCycleSlot* slots, int capacity) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (slots && capacity <= 0) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->probe          = slots;
  ctx->probe_capacity = slots ? capacity : 0;
  return GRXBLAS_STATUS_SUCCESS;
}

int grxblasCycleSlotsNeeded(grxblasHandle_t handle, int m, int n) {
  (void)handle;
  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess) return 0;
  return slots_for(m, n, prop.warpSize);
}

grxblasStatus_t grxblasGetLoadedKernelPath(grxblasHandle_t handle,
                                           const char** path) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!path) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *path = ctx->sgemm_path.empty() ? nullptr : ctx->sgemm_path.c_str();
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasSgemm(grxblasHandle_t handle,
                             grxblasOperation_t transa,
                             grxblasOperation_t transb,
                             int m, int n, int k,
                             const float* alpha,
                             const void* A, int lda,
                             const void* B, int ldb,
                             const float* beta,
                             void* C, int ldc) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || k < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return GRXBLAS_STATUS_SUCCESS;
  if (k > 0 && (!A || !B)) return GRXBLAS_STATUS_INVALID_VALUE;
  if (!C) return GRXBLAS_STATUS_INVALID_VALUE;

  // Leading-dimension rules, checked rather than trusted: a too-small ld reads
  // or writes outside the caller's allocation, and the allocator would not
  // necessarily catch it because the address is still inside some other live
  // buffer.
  const int min_lda = (transa == GRXBLAS_OP_N) ? m : k;
  const int min_ldb = (transb == GRXBLAS_OP_N) ? k : n;
  if (lda < min_lda || ldb < min_ldb || ldc < m)
    return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  grxblas_sgemm_args args{};
  args.abi_version = GRXBLAS_SGEMM_ABI_VERSION;
  args.m = (uint32_t)m; args.n = (uint32_t)n; args.k = (uint32_t)k;
  args.lda = (uint32_t)lda; args.ldb = (uint32_t)ldb; args.ldc = (uint32_t)ldc;
  args.transa = (uint32_t)transa; args.transb = (uint32_t)transb;
  args.alpha = *alpha; args.beta = *beta;
  args.a = (uint64_t)(uintptr_t)A;
  args.b = (uint64_t)(uintptr_t)B;
  args.c = (uint64_t)(uintptr_t)C;

  // One thread per output element. The block is a whole warp so no lane is
  // wasted, and the grid covers m*n with the tail masked by the kernel.
  const unsigned block = (unsigned)prop.warpSize;
  const unsigned total = (unsigned)((size_t)m * (size_t)n);
  const unsigned grid  = (total + block - 1) / block;

  if (ctx->probe) {
    if (ctx->probe_capacity < slots_for(m, n, prop.warpSize))
      return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }

  e = grxLaunchFunction(ctx->sgemm_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), /*sharedMem=*/0, ctx->stream);
  return from_grx(e);
}

}  // extern "C"
