// grxDNN — the host side. Shapes in, one launch out.
//
// The library ships precompiled device kernels and resolves them by name from
// a .vxbin, exactly as grxBLAS does, and prefers the SAME image: only one
// module can be resident at a time, so a program using both libraries must
// find both libraries' entry points in one file. src/libs/kernels_all.cpp is
// that file, and the reasoning is written down there.

#include <grx/grxdnn.h>

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "dnn_abi.h"

namespace {

std::mutex  g_path_mutex;
std::string g_kernel_path;

// THE SAME SEARCH ORDER AS grxBLAS, and that matters more than it looks.
//
// Both libraries load the same file, and the runtime hands the second caller
// the module the first one already has -- but only when the two arrive at the
// same bytes. Two different search orders find the same file in most layouts
// and a different one in some, and the case where they differ is a program that
// loaded two images and got an address overlap on the second.
//
// The executable's own directory is in the list because that is where the build
// puts the .vxbin next to the binary that needs it, and a program run from
// anywhere other than its build directory finds nothing without it. grxDNN did
// not have this until tests/libs/test_libs_together.cpp compared the two
// libraries' loaded paths and they disagreed.
std::vector<std::string> candidate_paths(const char* module_name) {
  std::vector<std::string> out;
  {
    std::lock_guard<std::mutex> lock(g_path_mutex);
    if (!g_kernel_path.empty()) return {g_kernel_path + "/" + module_name};
  }
  if (const char* env = std::getenv("GRXDNN_KERNEL_PATH"))
    return {std::string(env) + "/" + module_name};
  // grxBLAS's variable too: the shared image is built once and both libraries
  // look for it, so a program that already sets one should not have to set two.
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
  std::mutex    mutex;
  grxStream_t   stream    = nullptr;
  grxModule_t   module    = nullptr;
  grxFunction_t softmax   = nullptr;
  grxFunction_t layernorm = nullptr;
  std::string   path;
};

grxdnnStatus_t ensure_module_locked(Context& ctx) {
  if (ctx.module) return GRXDNN_STATUS_SUCCESS;

  static const char* const kModules[] = {"grxlibs_kernels.vxbin",
                                         "grxdnn_kernels.vxbin"};
  grxError_t last = grxSuccess;
  for (const char* name : kModules) {
    for (const std::string& path : candidate_paths(name)) {
      grxModule_t mod = nullptr;
      last = grxModuleLoad(&mod, path.c_str());
      if (last != grxSuccess) continue;

      grxFunction_t sm = nullptr, ln = nullptr;
      if (grxModuleGetFunction(&sm, mod, "dnn_softmax") != grxSuccess ||
          grxModuleGetFunction(&ln, mod, "dnn_layernorm") != grxSuccess) {
        grxModuleUnload(mod);
        continue;
      }
      ctx.module    = mod;
      ctx.softmax   = sm;
      ctx.layernorm = ln;
      ctx.path      = path;
      return GRXDNN_STATUS_SUCCESS;
    }
  }

  std::fprintf(stderr,
               "grxdnn: cannot load grxlibs_kernels.vxbin or "
               "grxdnn_kernels.vxbin (last error: %s).\n"
               "        Set GRXDNN_KERNEL_PATH or call grxdnnSetKernelPath.\n",
               grxGetErrorString(last));
  return GRXDNN_STATUS_NOT_INITIALIZED;
}

// A launch shape for a one-warp-per-row kernel.
//
// The kernels stride over rows, so the grid does not have to cover them --
// but a grid far larger than the work is warp slots left idle at the tail, and
// a grid far smaller serialises rows that could have run together. This asks
// the device how wide it is and covers the rows once.
struct Shape { unsigned grid, block; };

grxdnnStatus_t launch_shape(int rows, Shape* out) {
  int device = 0;
  if (grxGetDevice(&device) != grxSuccess) return GRXDNN_STATUS_NOT_INITIALIZED;
  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, device) != grxSuccess)
    return GRXDNN_STATUS_NOT_INITIALIZED;
  if (prop.warpSize <= 0) return GRXDNN_STATUS_ARCH_MISMATCH;

  const unsigned warp = (unsigned)prop.warpSize;
  // Four warps per block unless the device's own limit is smaller. Four is not
  // tuned -- it is a starting point that keeps the block small enough to be
  // schedulable on the narrow default configuration and large enough not to pay
  // a block's dispatch cost per warp.
  unsigned warps_per_block = 4u;
  while (warps_per_block > 1u &&
         warp * warps_per_block > (unsigned)prop.maxThreadsPerBlock)
    warps_per_block /= 2u;

  out->block = warp * warps_per_block;
  const unsigned needed = ((unsigned)rows + warps_per_block - 1u) / warps_per_block;
  out->grid = needed ? needed : 1u;
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t map_launch(grxError_t e) {
  return (e == grxSuccess) ? GRXDNN_STATUS_SUCCESS
                           : GRXDNN_STATUS_EXECUTION_FAILED;
}

}  // namespace

extern "C" {

const char* grxdnnGetStatusString(grxdnnStatus_t s) {
  switch (s) {
    case GRXDNN_STATUS_SUCCESS:          return "success";
    case GRXDNN_STATUS_NOT_INITIALIZED:  return "not initialized";
    case GRXDNN_STATUS_ALLOC_FAILED:     return "allocation failed";
    case GRXDNN_STATUS_INVALID_VALUE:    return "invalid value";
    case GRXDNN_STATUS_ARCH_MISMATCH:    return "device cannot run this";
    case GRXDNN_STATUS_EXECUTION_FAILED: return "execution failed";
    case GRXDNN_STATUS_NOT_SUPPORTED:    return "not supported";
    case GRXDNN_STATUS_KERNEL_NOT_FOUND: return "kernel not found in module";
  }
  return "unknown grxdnn status";
}

grxdnnStatus_t grxdnnCreate(grxdnnHandle_t* handle) {
  if (!handle) return GRXDNN_STATUS_INVALID_VALUE;
  Context* ctx = new (std::nothrow) Context();
  if (!ctx) return GRXDNN_STATUS_ALLOC_FAILED;
  *handle = reinterpret_cast<grxdnnHandle_t>(ctx);
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnDestroy(grxdnnHandle_t handle) {
  if (!handle) return GRXDNN_STATUS_INVALID_VALUE;
  Context* ctx = reinterpret_cast<Context*>(handle);
  if (ctx->module) grxModuleUnload(ctx->module);
  delete ctx;
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnSetStream(grxdnnHandle_t handle, grxStream_t stream) {
  if (!handle) return GRXDNN_STATUS_INVALID_VALUE;
  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->stream = stream;
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnGetStream(grxdnnHandle_t handle, grxStream_t* stream) {
  if (!handle || !stream) return GRXDNN_STATUS_INVALID_VALUE;
  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  *stream = ctx->stream;
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnSetKernelPath(const char* dir) {
  std::lock_guard<std::mutex> lock(g_path_mutex);
  g_kernel_path = dir ? dir : "";
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnGetLoadedKernelPath(grxdnnHandle_t handle,
                                         const char** path) {
  if (!handle || !path) return GRXDNN_STATUS_INVALID_VALUE;
  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  // NULL rather than "" when nothing is loaded: loading is lazy, and an empty
  // string would read as "loaded from the current directory".
  *path = ctx->path.empty() ? nullptr : ctx->path.c_str();
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnSoftmaxForward(grxdnnHandle_t handle,
                                    int rows, int cols,
                                    const float* x, int ldx,
                                    float* y, int ldy) {
  if (!handle || !x || !y) return GRXDNN_STATUS_INVALID_VALUE;
  // cols == 0 would make every row's sum zero and every output a division by
  // it. Refused here rather than producing a buffer of NaN that gets blamed on
  // whatever consumes it.
  if (rows <= 0 || cols <= 0) return GRXDNN_STATUS_INVALID_VALUE;
  if (ldx < cols || ldy < cols) return GRXDNN_STATUS_INVALID_VALUE;

  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  grxdnnStatus_t st = ensure_module_locked(*ctx);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  Shape shape{};
  st = launch_shape(rows, &shape);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  grxdnn_softmax_args args{};
  args.abi_version = GRXDNN_ABI_VERSION;
  args.rows = (uint32_t)rows;
  args.cols = (uint32_t)cols;
  args.ldx  = ldx;
  args.ldy  = ldy;
  args.x    = (uint64_t)(uintptr_t)x;
  args.y    = (uint64_t)(uintptr_t)y;

  return map_launch(grxLaunchFunction(ctx->softmax,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &args, sizeof(args), 0, ctx->stream));
}

grxdnnStatus_t grxdnnLayerNormForward(grxdnnHandle_t handle,
                                      int rows, int cols,
                                      const float* x, int ldx,
                                      const float* gamma, const float* beta,
                                      float eps,
                                      float* y, int ldy) {
  if (!handle || !x || !y) return GRXDNN_STATUS_INVALID_VALUE;
  if (rows <= 0 || cols <= 0) return GRXDNN_STATUS_INVALID_VALUE;
  if (ldx < cols || ldy < cols) return GRXDNN_STATUS_INVALID_VALUE;
  // A negative epsilon can make var + eps negative and rsqrt produce a NaN.
  // Zero is allowed -- it is what a caller means by "no floor" -- so the test
  // is on the sign, not on being positive.
  if (!(eps >= 0.0f)) return GRXDNN_STATUS_INVALID_VALUE;

  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  grxdnnStatus_t st = ensure_module_locked(*ctx);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  Shape shape{};
  st = launch_shape(rows, &shape);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  grxdnn_layernorm_args args{};
  args.abi_version = GRXDNN_ABI_VERSION;
  args.rows  = (uint32_t)rows;
  args.cols  = (uint32_t)cols;
  args.ldx   = ldx;
  args.ldy   = ldy;
  args.eps   = eps;
  args.x     = (uint64_t)(uintptr_t)x;
  args.y     = (uint64_t)(uintptr_t)y;
  args.gamma = (uint64_t)(uintptr_t)gamma;
  args.beta  = (uint64_t)(uintptr_t)beta;

  return map_launch(grxLaunchFunction(ctx->layernorm,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &args, sizeof(args), 0, ctx->stream));
}

}  // extern "C"
