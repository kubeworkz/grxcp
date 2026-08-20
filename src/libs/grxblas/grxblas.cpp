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

#include "hgemm_abi.h"
#include "blas12_abi.h"
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
  // ONE module for the whole library. Every .vxbin links at the same fixed
  // load address, so a second one cannot be resident at the same time -- see
  // kernels/all.cpp. The combined module is preferred; the scalar-only one is
  // the fallback for a device with no tensor unit.
  grxModule_t   module = nullptr;
  grxFunction_t sgemm_fn = nullptr;
  std::string   sgemm_path;   // which file actually got loaded
  // Level 1 and 2, resolved from the same module. Absent in an older module,
  // which is why they are looked up lazily and reported rather than assumed.
  grxFunction_t axpy_fn = nullptr;
  grxFunction_t scal_fn = nullptr;
  grxFunction_t gemv_fn = nullptr;
  grxCycleSlot* probe = nullptr;
  int           probe_capacity = 0;

  // Tensor path, resolved from the same module when it carries the entries.
  grxFunction_t hgemm_fn = nullptr;
  uint32_t      tile_m = 0, tile_n = 0, tile_k = 0, tile_smem = 0;
  uint32_t      tcu_types = 0;   // GRXBLAS_TENSOR_* the device build supports
  // What one warp produces per pass, which is a multiple of the tile: the
  // kernel blocks several tiles together to reuse a staged region.
  uint32_t      block_m = 0, block_n = 0;
  int           slot_a = 0, slot_b = 1;

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

// Load the library's module, preferring the one that carries every kernel.
// Called with ctx.mutex held.
grxblasStatus_t ensure_module_locked(Context& ctx) {
  if (ctx.module) return GRXBLAS_STATUS_SUCCESS;

  // Order matters: the combined module carries sgemm too, so trying it first
  // means a device that can run the tensor path gets both entry points, and a
  // device that cannot falls back to the scalar-only image.
  static const char* const kModules[] = {"grxblas_kernels.vxbin",
                                         "grxblas_sgemm.vxbin"};
  grxError_t last = grxSuccess;
  for (const char* name : kModules) {
    for (const std::string& path : candidate_paths(name)) {
      grxModule_t mod = nullptr;
      last = grxModuleLoad(&mod, path.c_str());
      if (last != grxSuccess) continue;

      grxFunction_t fn = nullptr;
      if (grxModuleGetFunction(&fn, mod, "sgemm") != grxSuccess) {
        grxModuleUnload(mod);
        continue;
      }
      ctx.module     = mod;
      ctx.sgemm_fn   = fn;
      ctx.sgemm_path = path;
      // Best effort: a module built before these existed still serves sgemm,
      // and the level-1/2 calls report NOT_SUPPORTED instead of the whole
      // library refusing to initialise.
      grxModuleGetFunction(&ctx.axpy_fn, mod, "saxpy");
      grxModuleGetFunction(&ctx.scal_fn, mod, "sscal");
      grxModuleGetFunction(&ctx.gemv_fn, mod, "sgemv");
      return GRXBLAS_STATUS_SUCCESS;
    }
  }

  // Say which file is missing, because "internal error" from a BLAS call is
  // among the least actionable messages a library can produce.
  std::fprintf(stderr,
               "grxblas: cannot load grxblas_kernels.vxbin or "
               "grxblas_sgemm.vxbin (last error: %s).\n"
               "         Set GRXBLAS_KERNEL_PATH or call grxblasSetKernelPath.\n",
               grxGetErrorString(last));
  return GRXBLAS_STATUS_NOT_INITIALIZED;
}

// Format a tensor-type mask for a human. Says "nothing" rather than printing
// an empty list, because an empty list reads like a formatting bug.
void describe_tensor_types(uint32_t mask, char* out, size_t cap) {
  static const struct { uint32_t bit; const char* name; } kNames[] = {
    {0x01u, "fp16"}, {0x02u, "tf32"}, {0x04u, "fp8"},
    {0x08u, "fp4"},  {0x10u, "int8"}, {0x20u, "int4"},
  };
  out[0] = '\0';
  size_t used = 0;
  for (const auto& n : kNames) {
    if (!(mask & n.bit)) continue;
    const int wrote = std::snprintf(out + used, cap - used, "%s%s",
                                    used ? ", " : "", n.name);
    if (wrote <= 0 || (size_t)wrote >= cap - used) break;
    used += (size_t)wrote;
  }
  if (used == 0) std::snprintf(out, cap, "no input types at all");
}

grxblasStatus_t ensure_sgemm(Context& ctx) {
  std::lock_guard<std::mutex> lock(ctx.mutex);
  return ensure_module_locked(ctx);
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

// The tensor kernel's tile shape is a property of the DEVICE BUILD, not a
// constant this file may assume: it follows from warp width and registers per
// fragment. So the module is asked, once, and the answer drives the grid and
// the descriptors. See include/grx/device/grx_wmma.h.
grxblasStatus_t ensure_hgemm(Context& ctx) {
  std::lock_guard<std::mutex> lock(ctx.mutex);
  if (ctx.hgemm_fn) return GRXBLAS_STATUS_SUCCESS;

  grxDeviceProp_t prop{};
  if (grxGetDeviceProperties(&prop, 0) != grxSuccess)
    return GRXBLAS_STATUS_INTERNAL_ERROR;
  if (!(prop.capabilities & GRX_CAP_TENSOR_CORE) ||
      !(prop.capabilities & GRX_CAP_ASYNC_COPY))
    return GRXBLAS_STATUS_ARCH_MISMATCH;

  const grxblasStatus_t ms = ensure_module_locked(ctx);
  if (ms != GRXBLAS_STATUS_SUCCESS) return ms;

  grxFunction_t fn = nullptr, shape_fn = nullptr;
  if (grxModuleGetFunction(&fn, ctx.module, "hgemm_tcu") != grxSuccess ||
      grxModuleGetFunction(&shape_fn, ctx.module, "hgemm_tcu_shape") !=
          grxSuccess) {
    // The scalar-only module loaded. That is a configuration, not a failure:
    // the device can still do sgemm.
    std::fprintf(stderr,
                 "grxblas: %s has no tensor kernels; build "
                 "src/libs/grxblas/kernels/all.cpp to get them.\n",
                 ctx.sgemm_path.c_str());
    return GRXBLAS_STATUS_ARCH_MISMATCH;
  }

  void* dshape = nullptr;
  if (grxMalloc(&dshape, GRXBLAS_HGEMM_SHAPE_COUNT * sizeof(uint32_t)) !=
      grxSuccess)
    return GRXBLAS_STATUS_ALLOC_FAILED;
  grxMemset(dshape, 0, GRXBLAS_HGEMM_SHAPE_COUNT * sizeof(uint32_t));

  grxblas_hgemm_shape_args sargs{};
  sargs.abi_version = GRXBLAS_HGEMM_ABI_VERSION;
  sargs.out = (uint64_t)(uintptr_t)dshape;
  grxError_t e = grxLaunchFunction(shape_fn, dim3_t{1, 1, 1},
                                   dim3_t{(unsigned)prop.warpSize, 1, 1},
                                   &sargs, sizeof(sargs), 0, nullptr);
  if (e == grxSuccess) e = grxDeviceSynchronize();

  uint32_t shape[GRXBLAS_HGEMM_SHAPE_COUNT] = {0};
  if (e == grxSuccess)
    e = grxMemcpy(shape, dshape, sizeof(shape), grxMemcpyDefault);
  grxFree(dshape);
  if (e != grxSuccess) return from_grx(e);

  if (shape[GRXBLAS_HGEMM_SHAPE_M] == 0 || shape[GRXBLAS_HGEMM_SHAPE_N] == 0 ||
      shape[GRXBLAS_HGEMM_SHAPE_K] == 0)
    return GRXBLAS_STATUS_INTERNAL_ERROR;

  // The kernel reports the warp width it was COMPILED for. If that is not the
  // width this device has, the module was built for a different machine and
  // every fragment layout in it is wrong. See ci/README.md, "configuration
  // provenance".
  if (shape[GRXBLAS_HGEMM_SHAPE_WARP] != (uint32_t)prop.warpSize) {
    std::fprintf(stderr,
                 "grxblas: the tensor kernels were built for a warp width of "
                 "%u, device reports %d.\n",
                 shape[GRXBLAS_HGEMM_SHAPE_WARP], prop.warpSize);
    return GRXBLAS_STATUS_ARCH_MISMATCH;
  }

  ctx.hgemm_fn     = fn;
  ctx.tile_m       = shape[GRXBLAS_HGEMM_SHAPE_M];
  ctx.tile_n       = shape[GRXBLAS_HGEMM_SHAPE_N];
  ctx.tile_k       = shape[GRXBLAS_HGEMM_SHAPE_K];
  ctx.tile_smem    = shape[GRXBLAS_HGEMM_SHAPE_SMEM];
  ctx.tcu_types    = shape[GRXBLAS_HGEMM_SHAPE_TYPES];
  ctx.block_m      = shape[GRXBLAS_HGEMM_SHAPE_BLOCK_M];
  ctx.block_n      = shape[GRXBLAS_HGEMM_SHAPE_BLOCK_N];
  if (ctx.block_m == 0 || ctx.block_n == 0)
    return GRXBLAS_STATUS_INTERNAL_ERROR;
  return GRXBLAS_STATUS_SUCCESS;
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
  if (ctx->module) grxModuleUnload(ctx->module);
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

grxblasStatus_t grxblasSetTensorMapSlots(grxblasHandle_t handle,
                                         int slotA, int slotB) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  int slots = 0;
  if (grxTensorMapGetSlotCount(&slots, 0) != grxSuccess)
    return GRXBLAS_STATUS_INTERNAL_ERROR;
  if (slotA < 0 || slotB < 0 || slotA >= slots || slotB >= slots ||
      slotA == slotB)
    return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  std::lock_guard<std::mutex> lock(ctx->mutex);
  ctx->slot_a = slotA;
  ctx->slot_b = slotB;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGetTensorTypes(grxblasHandle_t handle,
                                      unsigned* typeMask) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!typeMask) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_hgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  *typeMask = (unsigned)ctx->tcu_types;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGetTensorTile(grxblasHandle_t handle, int* m, int* n,
                                     int* k) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!m || !n || !k) return GRXBLAS_STATUS_INVALID_VALUE;
  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_hgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  *m = (int)ctx->tile_m;
  *n = (int)ctx->tile_n;
  *k = (int)ctx->tile_k;
  return GRXBLAS_STATUS_SUCCESS;
}

grxblasStatus_t grxblasGemmEx(grxblasHandle_t handle,
                              grxblasOperation_t transa,
                              grxblasOperation_t transb,
                              int m, int n, int k,
                              const float* alpha,
                              const void* A, grxDataType_t Atype, int lda,
                              const void* B, grxDataType_t Btype, int ldb,
                              const float* beta,
                              void* C, grxDataType_t Ctype, int ldc) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || k < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return GRXBLAS_STATUS_SUCCESS;
  if (k > 0 && (!A || !B)) return GRXBLAS_STATUS_INVALID_VALUE;
  if (!C) return GRXBLAS_STATUS_INVALID_VALUE;

  // Refused rather than emulated. A fallback to the scalar kernel here would
  // report success for a call the tensor path cannot do, and the caller would
  // read the resulting speed as the tensor unit's.
  // Leading-dimension rules follow op(), not the logical shape: a transposed A
  // is STORED k x m, so its leading dimension bounds k rather than m.
  const bool ta = (transa == GRXBLAS_OP_T);
  const bool tb = (transb == GRXBLAS_OP_T);
  const int min_lda = ta ? k : m;
  const int min_ldb = tb ? n : k;
  if (lda < min_lda || ldb < min_ldb || ldc < m)
    return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_hgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;

  // The type check comes AFTER the module is up, so the refusal can say what
  // this device actually has instead of only what it does not. "not supported"
  // with no further information is how a caller ends up assuming the whole
  // tensor path is missing when one type is.
  if (Atype != GRX_R_16F || Btype != GRX_R_16F || Ctype != GRX_R_32F) {
    static bool said = false;
    if (!said) {
      said = true;
      char have[128];
      describe_tensor_types(ctx->tcu_types, have, sizeof(have));
      std::fprintf(stderr,
                   "grxblas: grxblasGemmEx wants fp16 in and fp32 out; this "
                   "device's tensor unit accepts %s.\n"
                   "         Types are a build-time choice on GRX-G100 -- "
                   "query them with grxblasGetTensorTypes.\n",
                   have);
    }
    return GRXBLAS_STATUS_NOT_SUPPORTED;
  }

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  // The kernel walks BLOCKS, each several WMMA tiles wide and tall, so the
  // descriptor tiles have to cover a block rather than a tile.
  const uint32_t tk = ctx->tile_k;
  const uint32_t bm = ctx->block_m, bn = ctx->block_n;
  const uint32_t m_tiles = ((uint32_t)m + bm - 1) / bm;
  const uint32_t n_tiles = ((uint32_t)n + bn - 1) / bn;
  const uint32_t k_steps = ((uint32_t)k + tk - 1) / tk;

  // The descriptors, and what a transpose does to them.
  //
  // The kernel wants one thing from each operand regardless of how it is
  // stored: sA holding op(A)'s tile ROW major with leading dimension tile::k,
  // and sB holding op(B)'s tile COLUMN major with the same leading dimension.
  // That is what matrix_a and matrix_b col_major fragments read.
  //
  // Dimension 0 of a descriptor is the source's contiguous direction, and the
  // destination layout says whether the engine keeps that order (RowMajor:
  // dest[e1*tile0 + e0]) or transposes it (KMajor: dest[e0*tile1 + e1]). So
  // transposing an operand does three things together, and they are three
  // faces of one change:
  //
  //   the extents swap        (m,k) <-> (k,m)     -- what is stored
  //   the tile extents swap   (bm,tk) <-> (tk,bm)
  //   the destination layout flips                -- to land on the same sA
  //
  // and the kernel swaps its coordinate pair to match, because coordinates are
  // per block and computed on the device.
  //
  //   A, N: stored m x k, dim0 = m. KMajor transposes it into row-major sA.
  //   A, T: stored k x m, dim0 = k. RowMajor already gives row-major sA.
  //   B, N: stored k x n, dim0 = k. RowMajor gives column-major sB.
  //   B, T: stored n x k, dim0 = n. KMajor gives column-major sB.
  grxTensorMapDesc_t da{};
  da.slot = ctx->slot_a;
  da.base = const_cast<void*>(A);
  da.rank = 2;
  da.strideBytes[0] = (unsigned)lda * 2u;
  da.elementBytes = 2;
  if (!ta) {
    da.size[0] = (unsigned)m;            da.size[1] = (unsigned)(k ? k : 1);
    da.tile[0] = bm;                     da.tile[1] = tk;
    da.layout  = grxTensorMapLayoutKMajor;
  } else {
    da.size[0] = (unsigned)(k ? k : 1);  da.size[1] = (unsigned)m;
    da.tile[0] = tk;                     da.tile[1] = bm;
    da.layout  = grxTensorMapLayoutRowMajor;
  }

  grxTensorMapDesc_t db{};
  db.slot = ctx->slot_b;
  db.base = const_cast<void*>(B);
  db.rank = 2;
  db.strideBytes[0] = (unsigned)ldb * 2u;
  db.elementBytes = 2;
  if (!tb) {
    db.size[0] = (unsigned)(k ? k : 1);  db.size[1] = (unsigned)n;
    db.tile[0] = tk;                     db.tile[1] = bn;
    db.layout  = grxTensorMapLayoutRowMajor;
  } else {
    db.size[0] = (unsigned)n;            db.size[1] = (unsigned)(k ? k : 1);
    db.tile[0] = bn;                     db.tile[1] = tk;
    db.layout  = grxTensorMapLayoutKMajor;
  }

  if (k > 0) {
    e = grxTensorMapProgramAsync(&da, ctx->stream);
    if (e != grxSuccess) return from_grx(e);
    e = grxTensorMapProgramAsync(&db, ctx->stream);
    if (e != grxSuccess) return from_grx(e);
  }

  grxblas_hgemm_args args{};
  args.abi_version = GRXBLAS_HGEMM_ABI_VERSION;
  args.m = (uint32_t)m; args.n = (uint32_t)n; args.k = (uint32_t)k;
  args.ldc = (uint32_t)ldc;
  args.m_tiles = m_tiles;
  args.tiles   = m_tiles * n_tiles;
  args.k_steps = k_steps;
  args.slot_a = (uint32_t)ctx->slot_a;
  args.slot_b = (uint32_t)ctx->slot_b;
  args.transa = ta ? GRXBLAS_ABI_OP_T : GRXBLAS_ABI_OP_N;
  args.transb = tb ? GRXBLAS_ABI_OP_T : GRXBLAS_ABI_OP_N;
  args.barrier = 0;
  args.alpha = *alpha; args.beta = *beta;
  args.c = (uint64_t)(uintptr_t)C;

  // ONE block, with as many warps as there is work for. Not the natural shape
  // -- that is one CTA per tile -- but a second CTA issuing a tensor
  // instruction deadlocks the current SimX model, so the kernel walks the
  // tiles instead. See the header of kernels/hgemm_tcu.cpp and
  // tests/repro/tcu_multi_cta/. On a one-SM configuration this costs nothing;
  // it is a ceiling everywhere else, and it comes out when the model is fixed.
  const uint32_t tiles = m_tiles * n_tiles;
  uint32_t warps = (uint32_t)(prop.maxThreadsPerBlock / prop.warpSize);
  if (warps == 0) warps = 1;
  if (warps > tiles) warps = tiles;

  if (ctx->probe) {
    if ((uint32_t)ctx->probe_capacity < warps)
      return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }

  e = grxLaunchFunction(ctx->hgemm_fn, dim3_t{1, 1, 1},
                        dim3_t{warps * (unsigned)prop.warpSize, 1, 1}, &args,
                        sizeof(args), (size_t)ctx->tile_smem * warps,
                        ctx->stream);
  return from_grx(e);
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

// ---------------------------------------------------------------------------
// Level 1 and level 2
// ---------------------------------------------------------------------------

grxblasStatus_t grxblasSaxpy(grxblasHandle_t handle, int n, const float* alpha,
                             const float* x, int incx, float* y, int incy) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (n < 0)   return GRXBLAS_STATUS_INVALID_VALUE;
  if (n == 0)  return GRXBLAS_STATUS_SUCCESS;
  if (!x || !y) return GRXBLAS_STATUS_INVALID_VALUE;
  if (incx == 0 || incy == 0) return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  if (!ctx->axpy_fn) return GRXBLAS_STATUS_NOT_SUPPORTED;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  grxblas_axpy_args args{};
  args.abi_version = GRXBLAS_BLAS12_ABI_VERSION;
  args.n = (uint32_t)n;
  args.incx = incx; args.incy = incy;
  args.alpha = *alpha;
  args.x = (uint64_t)(uintptr_t)x;
  args.y = (uint64_t)(uintptr_t)y;

  const unsigned block = (unsigned)prop.warpSize;
  const unsigned grid  = ((unsigned)n + block - 1) / block;
  if (ctx->probe) {
    if (ctx->probe_capacity < (int)grid) return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }
  e = grxLaunchFunction(ctx->axpy_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), 0, ctx->stream);
  return from_grx(e);
}

grxblasStatus_t grxblasSscal(grxblasHandle_t handle, int n, const float* alpha,
                             float* x, int incx) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (n < 0)   return GRXBLAS_STATUS_INVALID_VALUE;
  if (n == 0)  return GRXBLAS_STATUS_SUCCESS;
  if (!x)      return GRXBLAS_STATUS_INVALID_VALUE;
  // BLAS itself requires a positive increment for scal, so this is the
  // reference behaviour rather than a restriction of ours.
  if (incx <= 0) return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  if (!ctx->scal_fn) return GRXBLAS_STATUS_NOT_SUPPORTED;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  grxblas_scal_args args{};
  args.abi_version = GRXBLAS_BLAS12_ABI_VERSION;
  args.n = (uint32_t)n;
  args.incx = incx;
  args.alpha = *alpha;
  args.x = (uint64_t)(uintptr_t)x;

  const unsigned block = (unsigned)prop.warpSize;
  const unsigned grid  = ((unsigned)n + block - 1) / block;
  if (ctx->probe) {
    if (ctx->probe_capacity < (int)grid) return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }
  e = grxLaunchFunction(ctx->scal_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), 0, ctx->stream);
  return from_grx(e);
}

grxblasStatus_t grxblasSgemv(grxblasHandle_t handle, grxblasOperation_t trans,
                             int m, int n, const float* alpha,
                             const void* A, int lda,
                             const void* x, int incx,
                             const float* beta,
                             void* y, int incy) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return GRXBLAS_STATUS_SUCCESS;
  if (!A || !x || !y)  return GRXBLAS_STATUS_INVALID_VALUE;
  if (incx == 0 || incy == 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (lda < m) return GRXBLAS_STATUS_INVALID_VALUE;

  auto* ctx = reinterpret_cast<Context*>(handle);
  const grxblasStatus_t s = ensure_sgemm(*ctx);
  if (s != GRXBLAS_STATUS_SUCCESS) return s;
  if (!ctx->gemv_fn) return GRXBLAS_STATUS_NOT_SUPPORTED;

  grxDeviceProp_t prop{};
  grxError_t e = grxGetDeviceProperties(&prop, 0);
  if (e != grxSuccess) return from_grx(e);

  // m and n describe A as stored. op(A) is m x n untransposed and n x m
  // transposed, so the output length and the reduction length swap.
  const bool tr = (trans == GRXBLAS_OP_T);
  const uint32_t rows  = tr ? (uint32_t)n : (uint32_t)m;
  const uint32_t depth = tr ? (uint32_t)m : (uint32_t)n;

  grxblas_gemv_args args{};
  args.abi_version = GRXBLAS_BLAS12_ABI_VERSION;
  args.m = (uint32_t)m; args.n = (uint32_t)n;
  args.lda = (uint32_t)lda;
  args.trans = tr ? GRXBLAS_ABI_OP_T : GRXBLAS_ABI_OP_N;
  args.rows = rows; args.depth = depth;
  args.incx = incx; args.incy = incy;
  args.alpha = *alpha; args.beta = *beta;
  args.a = (uint64_t)(uintptr_t)A;
  args.x = (uint64_t)(uintptr_t)x;
  args.y = (uint64_t)(uintptr_t)y;

  // Two launch shapes for two traversals -- see the comment at the top of
  // kernels/blas12.cpp. Untransposed: one thread per output. Transposed: one
  // warp per output, so the lanes read down a column together.
  const unsigned warp = (unsigned)prop.warpSize;
  const unsigned block = warp;
  const unsigned grid  = tr ? rows : ((rows + block - 1) / block);
  if (ctx->probe) {
    if (ctx->probe_capacity < (int)grid) return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }
  e = grxLaunchFunction(ctx->gemv_fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                        &args, sizeof(args), 0, ctx->stream);
  return from_grx(e);
}

// One implementation, with the unbatched call as the batch-of-one case.
//
// Two entry points sharing a body rather than two bodies: the validation rules
// are identical and the launch differs by one grid dimension, and the way a
// batched GEMM usually goes wrong is that its unbatched twin drifted.
static grxblasStatus_t sgemm_batched(grxblasHandle_t handle,
                                     grxblasOperation_t transa,
                                     grxblasOperation_t transb,
                                     int m, int n, int k,
                                     const float* alpha,
                                     const void* A, int lda, long long strideA,
                                     const void* B, int ldb, long long strideB,
                                     const float* beta,
                                     void* C, int ldc, long long strideC,
                                     int batch) {
  if (!handle) return GRXBLAS_STATUS_NOT_INITIALIZED;
  if (!alpha || !beta) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || k < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (batch < 0) return GRXBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0 || batch == 0) return GRXBLAS_STATUS_SUCCESS;
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
  args.batch = (uint32_t)batch;
  args.stride_a = (int64_t)strideA;
  args.stride_b = (int64_t)strideB;
  args.stride_c = (int64_t)strideC;

  // One thread per output element. The block is a whole warp so no lane is
  // wasted, and the grid covers m*n with the tail masked by the kernel. The
  // batch is the grid's second dimension, which is the whole reason batching
  // costs one launch here instead of `batch` of them.
  const unsigned block = (unsigned)prop.warpSize;
  const unsigned total = (unsigned)((size_t)m * (size_t)n);
  const unsigned grid  = (total + block - 1) / block;

  if (ctx->probe) {
    // The probe is one slot per block, and the grid is now two-dimensional.
    if (ctx->probe_capacity < slots_for(m, n, prop.warpSize) * batch)
      return GRXBLAS_STATUS_INVALID_VALUE;
    args.cycles = (uint64_t)(uintptr_t)ctx->probe;
  }

  e = grxLaunchFunction(ctx->sgemm_fn, dim3_t{grid, (unsigned)batch, 1},
                        dim3_t{block, 1, 1}, &args, sizeof(args),
                        /*sharedMem=*/0, ctx->stream);
  return from_grx(e);
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
  return sgemm_batched(handle, transa, transb, m, n, k, alpha, A, lda, 0,
                       B, ldb, 0, beta, C, ldc, 0, 1);
}

grxblasStatus_t grxblasSgemmStridedBatched(
    grxblasHandle_t handle, grxblasOperation_t transa,
    grxblasOperation_t transb, int m, int n, int k, const float* alpha,
    const void* A, int lda, long long strideA,
    const void* B, int ldb, long long strideB,
    const float* beta, void* C, int ldc, long long strideC, int batchCount) {
  return sgemm_batched(handle, transa, transb, m, n, k, alpha, A, lda, strideA,
                       B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

}  // extern "C"
