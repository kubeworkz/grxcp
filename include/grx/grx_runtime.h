// GRXCP — host runtime API (layer L1).
//
// CUDA Runtime semantics, GRX names. Every entry point here has a CUDA
// analogue with matching semantics; grx_cuda_compat.h maps one onto the other.
//
// Implementation contract: this API is expressible entirely on the shipped
// vortex2.h driver surface. Where a CUDA feature needs GRX-G100 driver or
// hardware work, the entry point returns grxErrorNotSupported and the gap is
// recorded in docs/designs/cuda_mapping.md section 7 -- it is never faked.

#ifndef GRX_RUNTIME_H
#define GRX_RUNTIME_H

#include "grx_types.h"
#include "grx_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

grxError_t grxGetDeviceCount      (int* count);
grxError_t grxSetDevice           (int device);
grxError_t grxGetDevice           (int* device);
grxError_t grxGetDeviceProperties (grxDeviceProp_t* prop, int device);
grxError_t grxDeviceSynchronize   (void);
grxError_t grxDeviceReset         (void);
grxError_t grxMemGetInfo          (size_t* freeBytes, size_t* totalBytes);
grxError_t grxDeviceGetAttribute  (int* value, int attr, int device);

// Peer access is declared in v1 and returns grxErrorNotSupported until the
// hardware has a peer path (NVLink-class decode on G100, or the coherent
// port on the c930 NPU). Declaring it now keeps the surface stable.
grxError_t grxDeviceCanAccessPeer     (int* canAccess, int device, int peer);
grxError_t grxDeviceEnablePeerAccess  (int peerDevice, unsigned int flags);

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

grxError_t grxGetLastError    (void);   // returns and clears
grxError_t grxPeekAtLastError (void);   // returns, does not clear

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

grxError_t grxMalloc        (void** ptr, size_t size);
grxError_t grxFree          (void* ptr);
grxError_t grxMallocHost    (void** ptr, size_t size);
grxError_t grxFreeHost      (void* ptr);
grxError_t grxMallocManaged (void** ptr, size_t size, unsigned int flags);
grxError_t grxHostRegister  (void* ptr, size_t size, unsigned int flags);
grxError_t grxHostUnregister(void* ptr);

grxError_t grxMemcpy        (void* dst, const void* src, size_t count,
                             grxMemcpyKind kind);
grxError_t grxMemcpyAsync   (void* dst, const void* src, size_t count,
                             grxMemcpyKind kind, grxStream_t stream);
grxError_t grxMemcpy2D      (void* dst, size_t dpitch,
                             const void* src, size_t spitch,
                             size_t width, size_t height, grxMemcpyKind kind);
grxError_t grxMemcpy2DAsync (void* dst, size_t dpitch,
                             const void* src, size_t spitch,
                             size_t width, size_t height, grxMemcpyKind kind,
                             grxStream_t stream);

grxError_t grxMemset        (void* dst, int value, size_t count);
grxError_t grxMemsetAsync   (void* dst, int value, size_t count,
                             grxStream_t stream);

grxError_t grxPointerGetAttributes(grxPointerAttributes* attr, const void* ptr);

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------
//
// grxStream_t wraps a vx_queue_h one-to-one. Stream SEMANTICS are correct from
// the first release; stream CONCURRENCY is not yet real -- the command
// processor defaults to a single queue and the driver serializes launches.
// A multi-stream program is correct and portable today and gets faster when
// the QMD-style atomic launch lands upstream. See cuda_mapping.md section 7.3.

grxError_t grxStreamCreate            (grxStream_t* stream);
grxError_t grxStreamCreateWithFlags   (grxStream_t* stream, unsigned int flags);
grxError_t grxStreamCreateWithPriority(grxStream_t* stream, unsigned int flags,
                                       int priority);
grxError_t grxStreamDestroy           (grxStream_t stream);
grxError_t grxStreamSynchronize       (grxStream_t stream);
grxError_t grxStreamQuery             (grxStream_t stream);
grxError_t grxStreamWaitEvent         (grxStream_t stream, grxEvent_t event,
                                       unsigned int flags);

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
//
// Backed by vx_event timeline counters, which are strictly more expressive
// than CUDA's binary events: one underlying event can carry many recordings.
// grxEventElapsedTime uses device-side profiling timestamps where the command
// processor supplies them and a host monotonic clock otherwise; which one was
// used is reported by grxDeviceProp_t.eventTimingIsDeviceSide.

grxError_t grxEventCreate          (grxEvent_t* event);
grxError_t grxEventCreateWithFlags (grxEvent_t* event, unsigned int flags);
grxError_t grxEventDestroy         (grxEvent_t event);
grxError_t grxEventRecord          (grxEvent_t event, grxStream_t stream);
grxError_t grxEventSynchronize     (grxEvent_t event);
grxError_t grxEventQuery           (grxEvent_t event);
grxError_t grxEventElapsedTime     (float* ms, grxEvent_t start, grxEvent_t end);

// ---------------------------------------------------------------------------
// Kernel launch
// ---------------------------------------------------------------------------
//
// `func` is the address of the host-side stub that grxcc emits for a
// __global__ function; the runtime resolves it to a vx_kernel_h through the
// registration table built at static-init time (architecture doc section 4.3).

grxError_t grxLaunchKernel  (const void* func, dim3_t gridDim, dim3_t blockDim,
                             void** args, size_t sharedMem, grxStream_t stream);
grxError_t grxLaunchKernelEx(const grxLaunchConfig_t* config, const void* func,
                             void** args);
grxError_t grxLaunchCooperativeKernel(const void* func, dim3_t gridDim,
                                      dim3_t blockDim, void** args,
                                      size_t sharedMem, grxStream_t stream);

grxError_t grxFuncGetAttributes(grxFuncAttributes* attr, const void* func);

// Driver-style launch: the caller supplies the argument blob already packed to
// the kernel's device layout, so no parameter descriptor is required. This is
// the entry point for language runtimes and translators that know their own
// ABI, and the one to use before grxcc exists to emit descriptors.
grxError_t grxLaunchFunction(grxFunction_t func, dim3_t gridDim,
                             dim3_t blockDim, const void* argsBlob,
                             size_t argsSize, size_t sharedMem,
                             grxStream_t stream);

// ---------------------------------------------------------------------------
// Modules (driver-style path, for JIT and language runtimes)
// ---------------------------------------------------------------------------

grxError_t grxModuleLoad       (grxModule_t* module, const char* path);
grxError_t grxModuleLoadData   (grxModule_t* module, const void* image,
                                size_t size);
grxError_t grxModuleUnload     (grxModule_t module);
grxError_t grxModuleGetFunction(grxFunction_t* func, grxModule_t module,
                                const char* name);

// ---------------------------------------------------------------------------
// Occupancy
// ---------------------------------------------------------------------------
//
// Not a passthrough. Resident CTAs per SM is the minimum of three bounds, the
// third of which is the CTA dispatcher's fixed-stride LMEM allocator formula
// taken verbatim from grxgpu/docs/designs/cta_clustering_and_dispatch.md 3.1:
//
//   warps : floor(NUM_WARPS / ceil(blockSize / NUM_THREADS))
//   slots : NUM_CTA_SLOTS  (== NUM_WARPS)
//   smem  : floor(LOCAL_MEM_SIZE / align_up(smemPerBlock, MEM_BLOCK_SIZE))
//
// There is no register bound: the dispatcher does not gate CTA admission on
// register count. Do not add one to match CUDA's model.

grxError_t grxOccupancyMaxActiveBlocksPerMultiprocessor(
    int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
grxError_t grxOccupancyMaxPotentialBlockSize(
    int* minGridSize, int* blockSize, const void* func,
    size_t dynamicSMemSize, int blockSizeLimit);

// ---------------------------------------------------------------------------
// Registration (emitted by grxcc; not called by application code)
// ---------------------------------------------------------------------------

void** __grxRegisterFatBinary  (void* fatCubin);
void   __grxUnregisterFatBinary(void** handle);
void   __grxRegisterFunction   (void** handle, const char* hostStub,
                                const char* deviceName,
                                int minBlocks, int maxThreads);
// Additive registration carrying the parameter layout, static shared-memory
// size, and register count. grxcc emits this alongside __grxRegisterFunction;
// without it grxLaunchKernel cannot pack a void** argument array and says so
// rather than guessing at parameter widths.
void   __grxRegisterKernelDesc (void** handle, const char* hostStub,
                                const grx_kernel_desc* desc);
void   __grxRegisterVar        (void** handle, const char* hostVar,
                                const char* deviceName, size_t size,
                                int isConstant);
grxError_t __grxPushCallConfiguration(dim3_t gridDim, dim3_t blockDim,
                                      size_t sharedMem, grxStream_t stream);
grxError_t __grxPopCallConfiguration (dim3_t* gridDim, dim3_t* blockDim,
                                      size_t* sharedMem, grxStream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRX_RUNTIME_H
