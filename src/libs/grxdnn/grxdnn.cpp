// grxDNN — the host side. Shapes in, one launch out.
//
// The library ships precompiled device kernels and resolves them by name from
// a .vxbin, exactly as grxBLAS does, and prefers the SAME image: only one
// module can be resident at a time, so a program using both libraries must
// find both libraries' entry points in one file. src/libs/kernels_all.cpp is
// that file, and the reasoning is written down there.

#include <grx/grxdnn.h>

#include <unistd.h>

#include <cmath>
#include <cstdint>
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
  grxFunction_t causal_mask = nullptr;
  grxFunction_t add_bias    = nullptr;
  grxFunction_t gelu        = nullptr;
  std::string   path;
  // Attention's two GEMMs are grxBLAS calls. The handle is created on first use
  // rather than in grxdnnCreate: a program that only ever calls softmax should
  // not pay for a grxBLAS context, and grxBLAS loads its module lazily too, so
  // creating one early would not even warm anything.
  grxblasHandle_t blas = nullptr;

  // Device cycle instrumentation. Null is off and off is the default: the same
  // kernel runs either way, so a measured run and a shipped run are the same
  // code. See grxdnnSetCycleProbe.
  grxCycleSlot* probe = nullptr;
  int           probe_capacity = 0;
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
      // Looked up but NOT required: an older image predating attention still
      // serves softmax and layer norm perfectly well, and refusing to load it
      // would break those callers over a kernel they never asked for.
      // grxdnnAttentionForward reports the absence by name if it is asked to
      // run without it.
      if (grxModuleGetFunction(&ctx.causal_mask, mod, "dnn_causal_mask") !=
          grxSuccess)
        ctx.causal_mask = nullptr;
      if (grxModuleGetFunction(&ctx.add_bias, mod, "dnn_add_bias") != grxSuccess)
        ctx.add_bias = nullptr;
      if (grxModuleGetFunction(&ctx.gelu, mod, "dnn_gelu") != grxSuccess)
        ctx.gelu = nullptr;
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

// How many warps a launch over `rows` rows will actually use. This is the one
// place that answers it, because the probe's capacity check and the launch
// geometry have to agree exactly and two places computing the same thing from
// memory is how they stop agreeing.
unsigned warps_in(const Shape& shape, unsigned warp_size) {
  return warp_size ? (shape.grid * (shape.block / warp_size)) : 0u;
}

// Point a kernel's argument block at the probe, if one is attached, after
// checking the capacity against the launch that is about to happen. Returns
// KERNEL_NOT_FOUND-free status; a probe too small is INVALID_VALUE rather than
// a partial record, matching grxblasSetCycleProbe.
grxdnnStatus_t attach_probe(const Context& ctx, const Shape& shape,
                            uint64_t* cycles_field) {
  if (!ctx.probe) return GRXDNN_STATUS_SUCCESS;
  int device = 0;
  grxDeviceProp_t prop{};
  if (grxGetDevice(&device) != grxSuccess ||
      grxGetDeviceProperties(&prop, device) != grxSuccess || prop.warpSize <= 0)
    return GRXDNN_STATUS_NOT_INITIALIZED;
  const unsigned warps = warps_in(shape, (unsigned)prop.warpSize);
  if ((unsigned)ctx.probe_capacity < warps) return GRXDNN_STATUS_INVALID_VALUE;
  *cycles_field = (uint64_t)(uintptr_t)ctx.probe;
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
  // grxBLAS first: it holds a reference to the same module this is about to
  // release, and destroying it after the unload would leave it briefly owning
  // functions from a module already let go.
  if (ctx->blas) grxblasDestroy(ctx->blas);
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

grxdnnStatus_t grxdnnSetCycleProbe(grxdnnHandle_t handle,
                                   grxCycleSlot* slots, int capacity) {
  if (!handle) return GRXDNN_STATUS_NOT_INITIALIZED;
  if (slots && capacity <= 0) return GRXDNN_STATUS_INVALID_VALUE;
  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->probe          = slots;
  ctx->probe_capacity = slots ? capacity : 0;
  return GRXDNN_STATUS_SUCCESS;
}

int grxdnnCycleSlotsNeeded(grxdnnHandle_t handle, int rows) {
  if (!handle || rows <= 0) return 0;
  // Asked of the same launch_shape the call itself will use, so the answer is
  // the launch's warp count rather than an estimate that has to be kept in
  // step with it.
  Shape shape{};
  if (launch_shape(rows, &shape) != GRXDNN_STATUS_SUCCESS) return 0;
  int device = 0;
  grxDeviceProp_t prop{};
  if (grxGetDevice(&device) != grxSuccess ||
      grxGetDeviceProperties(&prop, device) != grxSuccess || prop.warpSize <= 0)
    return 0;
  return (int)warps_in(shape, (unsigned)prop.warpSize);
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

  st = attach_probe(*ctx, shape, &args.cycles);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

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

  st = attach_probe(*ctx, shape, &args.cycles);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  return map_launch(grxLaunchFunction(ctx->layernorm,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &args, sizeof(args), 0, ctx->stream));
}

// ---------------------------------------------------------------------------
// Elementwise
// ---------------------------------------------------------------------------

grxdnnStatus_t grxdnnAddBiasForward(grxdnnHandle_t handle,
                                    int rows, int cols,
                                    const float* x, int ldx,
                                    const float* bias,
                                    float* y, int ldy) {
  if (!handle || !x || !bias || !y) return GRXDNN_STATUS_INVALID_VALUE;
  if (rows <= 0 || cols <= 0) return GRXDNN_STATUS_INVALID_VALUE;
  if (ldx < cols || ldy < cols) return GRXDNN_STATUS_INVALID_VALUE;

  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  grxdnnStatus_t st = ensure_module_locked(*ctx);
  if (st != GRXDNN_STATUS_SUCCESS) return st;
  if (!ctx->add_bias) return GRXDNN_STATUS_KERNEL_NOT_FOUND;

  Shape shape{};
  st = launch_shape(rows, &shape);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  grxdnn_bias_args args{};
  args.abi_version = GRXDNN_ABI_VERSION;
  args.rows = (uint32_t)rows;
  args.cols = (uint32_t)cols;
  args.ldx  = ldx;
  args.ldy  = ldy;
  args.x    = (uint64_t)(uintptr_t)x;
  args.bias = (uint64_t)(uintptr_t)bias;
  args.y    = (uint64_t)(uintptr_t)y;

  st = attach_probe(*ctx, shape, &args.cycles);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  return map_launch(grxLaunchFunction(ctx->add_bias,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &args, sizeof(args), 0, ctx->stream));
}

grxdnnStatus_t grxdnnGeluForward(grxdnnHandle_t handle,
                                 int rows, int cols,
                                 const float* x, int ldx,
                                 grxdnnGeluMode_t mode,
                                 float* y, int ldy) {
  if (!handle || !x || !y) return GRXDNN_STATUS_INVALID_VALUE;
  if (rows <= 0 || cols <= 0) return GRXDNN_STATUS_INVALID_VALUE;
  if (ldx < cols || ldy < cols) return GRXDNN_STATUS_INVALID_VALUE;
  // An unrecognised mode is refused rather than defaulted. The two forms differ
  // by ~1e-3 and a caller who has not chosen has a question to answer.
  if (mode != GRXDNN_GELU_EXACT && mode != GRXDNN_GELU_TANH)
    return GRXDNN_STATUS_INVALID_VALUE;

  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  grxdnnStatus_t st = ensure_module_locked(*ctx);
  if (st != GRXDNN_STATUS_SUCCESS) return st;
  if (!ctx->gelu) return GRXDNN_STATUS_KERNEL_NOT_FOUND;

  Shape shape{};
  st = launch_shape(rows, &shape);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  grxdnn_gelu_args args{};
  args.abi_version = GRXDNN_ABI_VERSION;
  args.rows = (uint32_t)rows;
  args.cols = (uint32_t)cols;
  args.ldx  = ldx;
  args.ldy  = ldy;
  args.mode = (mode == GRXDNN_GELU_TANH) ? 1u : 0u;
  args.x    = (uint64_t)(uintptr_t)x;
  args.y    = (uint64_t)(uintptr_t)y;

  st = attach_probe(*ctx, shape, &args.cycles);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  return map_launch(grxLaunchFunction(ctx->gelu,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &args, sizeof(args), 0, ctx->stream));
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

grxdnnStatus_t grxdnnAttentionWorkspaceSize(int batch, int heads, int seqLen,
                                            int headDim, size_t* bytes) {
  if (!bytes) return GRXDNN_STATUS_INVALID_VALUE;
  if (batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0)
    return GRXDNN_STATUS_INVALID_VALUE;

  // seqLen SQUARED per head, and the multiplication is checked rather than
  // trusted: at 64 heads and a 16k sequence this passes 2^38 elements, and a
  // size_t that has silently wrapped becomes an allocation far too small for
  // the writes that follow.
  const uint64_t heads_total = (uint64_t)batch * (uint64_t)heads;
  const uint64_t per_head    = (uint64_t)seqLen * (uint64_t)seqLen;
  if (per_head != 0 && heads_total > UINT64_MAX / per_head)
    return GRXDNN_STATUS_INVALID_VALUE;
  const uint64_t elems = heads_total * per_head;
  if (elems > UINT64_MAX / sizeof(float)) return GRXDNN_STATUS_INVALID_VALUE;
  const uint64_t need = elems * sizeof(float);
  if (need > (uint64_t)SIZE_MAX) return GRXDNN_STATUS_INVALID_VALUE;

  *bytes = (size_t)need;
  return GRXDNN_STATUS_SUCCESS;
}

grxdnnStatus_t grxdnnAttentionForward(grxdnnHandle_t handle,
                                      int batch, int heads,
                                      int seqLen, int headDim,
                                      const float* Q, const float* K,
                                      const float* V,
                                      grxdnnAttnMask_t mask,
                                      void* workspace, size_t workspaceBytes,
                                      float* out) {
  if (!handle || !Q || !K || !V || !out) return GRXDNN_STATUS_INVALID_VALUE;
  if (batch <= 0 || heads <= 0 || seqLen <= 0 || headDim <= 0)
    return GRXDNN_STATUS_INVALID_VALUE;
  if (mask != GRXDNN_ATTN_MASK_NONE && mask != GRXDNN_ATTN_MASK_CAUSAL)
    return GRXDNN_STATUS_INVALID_VALUE;

  // Aliasing is refused rather than tolerated. The second GEMM writes `out`
  // while reading V, so out == V produces a wrong answer and not a crash --
  // the worst kind, since it looks like an attention bug.
  if (out == Q || out == K || out == V) return GRXDNN_STATUS_INVALID_VALUE;

  size_t need = 0;
  grxdnnStatus_t st =
      grxdnnAttentionWorkspaceSize(batch, heads, seqLen, headDim, &need);
  if (st != GRXDNN_STATUS_SUCCESS) return st;
  if (!workspace || workspaceBytes < need) return GRXDNN_STATUS_INVALID_VALUE;

  Context* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  st = ensure_module_locked(*ctx);
  if (st != GRXDNN_STATUS_SUCCESS) return st;

  if (mask == GRXDNN_ATTN_MASK_CAUSAL && !ctx->causal_mask) {
    std::fprintf(stderr,
                 "grxdnn: this kernel image has no dnn_causal_mask, so a "
                 "causal mask cannot be applied.\n"
                 "        Rebuild it from src/libs/kernels_all.cpp (loaded: "
                 "%s).\n",
                 ctx->path.c_str());
    return GRXDNN_STATUS_KERNEL_NOT_FOUND;
  }

  if (!ctx->blas) {
    if (grxblasCreate(&ctx->blas) != GRXBLAS_STATUS_SUCCESS)
      return GRXDNN_STATUS_NOT_INITIALIZED;
  }
  // Every launch below goes on the caller's stream, including grxBLAS's, or the
  // GEMMs and the softmax between them would be ordered against each other only
  // by luck.
  if (grxblasSetStream(ctx->blas, ctx->stream) != GRXBLAS_STATUS_SUCCESS)
    return GRXDNN_STATUS_NOT_INITIALIZED;

  float* scores = reinterpret_cast<float*>(workspace);
  const int nheads = batch * heads;              // heads are contiguous
  const long long qkv_stride = (long long)seqLen * headDim;
  const long long s_stride   = (long long)seqLen * seqLen;

  // ---- GEMM 1: scores = Q Kᵀ / sqrt(headDim) --------------------------------
  //
  // grxBLAS is COLUMN-major and these tensors are ROW-major, and no data moves
  // to bridge that: a row-major (r, c) matrix with leading dimension ld IS the
  // column-major (c, r) matrix over the same bytes. So Q's memory read
  // column-major is Qᵀ, K's is Kᵀ, and writing the result column-major with
  // ld = seqLen produces scoresᵀ.
  //
  //   scoresᵀ = (Q Kᵀ)ᵀ = K Qᵀ
  //
  // which is A = K with transa = T, B = Q untransposed. The operands swap and
  // the transpose lands on K -- the opposite of how the formula reads, which is
  // exactly why this is checked against PyTorch in tests/libs/test_grxdnn_attn
  // rather than against a reference written from the same reasoning.
  //
  // The scale is alpha. It needs no kernel and no second pass over the matrix.
  const float scale = 1.0f / std::sqrt((float)headDim);
  const float zero = 0.0f, one = 1.0f;
  grxblasStatus_t bs = grxblasSgemmStridedBatched(
      ctx->blas, GRXBLAS_OP_T, GRXBLAS_OP_N,
      seqLen, seqLen, headDim, &scale,
      K, headDim, qkv_stride,
      Q, headDim, qkv_stride,
      &zero, scores, seqLen, s_stride, nheads);
  if (bs != GRXBLAS_STATUS_SUCCESS) return GRXDNN_STATUS_EXECUTION_FAILED;

  // ---- mask ----------------------------------------------------------------
  //
  // The score matrix is now [batch*heads*seqLen] rows of seqLen columns in the
  // row-major view, and the row index carries the query position: r % seqLen.
  const uint32_t rows = (uint32_t)nheads * (uint32_t)seqLen;
  if (mask == GRXDNN_ATTN_MASK_CAUSAL) {
    Shape shape{};
    st = launch_shape((int)rows, &shape);
    if (st != GRXDNN_STATUS_SUCCESS) return st;

    grxdnn_mask_args margs{};
    margs.abi_version = GRXDNN_ABI_VERSION;
    margs.rows    = rows;
    margs.seq_len = (uint32_t)seqLen;
    margs.ld      = seqLen;
    margs.scores  = (uint64_t)(uintptr_t)scores;

    st = attach_probe(*ctx, shape, &margs.cycles);
    if (st != GRXDNN_STATUS_SUCCESS) return st;

    st = map_launch(grxLaunchFunction(ctx->causal_mask,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &margs, sizeof(margs), 0, ctx->stream));
    if (st != GRXDNN_STATUS_SUCCESS) return st;
  }

  // ---- softmax along the key axis ------------------------------------------
  //
  // The same kernel the library already ships, over the same row-major view.
  // Nothing about it knows this is attention, which is the point: it is gated
  // on its own and this call inherits that.
  {
    Shape shape{};
    st = launch_shape((int)rows, &shape);
    if (st != GRXDNN_STATUS_SUCCESS) return st;

    grxdnn_softmax_args sargs{};
    sargs.abi_version = GRXDNN_ABI_VERSION;
    sargs.rows = rows;
    sargs.cols = (uint32_t)seqLen;
    sargs.ldx  = seqLen;
    sargs.ldy  = seqLen;
    sargs.x    = (uint64_t)(uintptr_t)scores;
    sargs.y    = (uint64_t)(uintptr_t)scores;   // in place

    st = attach_probe(*ctx, shape, &sargs.cycles);
    if (st != GRXDNN_STATUS_SUCCESS) return st;

    st = map_launch(grxLaunchFunction(ctx->softmax,
                                      dim3_t{shape.grid, 1, 1},
                                      dim3_t{shape.block, 1, 1},
                                      &sargs, sizeof(sargs), 0, ctx->stream));
    if (st != GRXDNN_STATUS_SUCCESS) return st;
  }

  // ---- GEMM 2: out = P V ---------------------------------------------------
  //
  //   outᵀ = (P V)ᵀ = Vᵀ Pᵀ
  //
  // and Vᵀ and Pᵀ are just V's and P's memory read column-major. So neither
  // operand is transposed and the two simply swap places -- V first. m is
  // headDim here, not seqLen, because the column-major result is outᵀ.
  bs = grxblasSgemmStridedBatched(
      ctx->blas, GRXBLAS_OP_N, GRXBLAS_OP_N,
      headDim, seqLen, seqLen, &one,
      V, headDim, qkv_stride,
      scores, seqLen, s_stride,
      &zero, out, headDim, qkv_stride, nheads);
  return (bs == GRXBLAS_STATUS_SUCCESS) ? GRXDNN_STATUS_SUCCESS
                                        : GRXDNN_STATUS_EXECUTION_FAILED;
}

}  // extern "C"
