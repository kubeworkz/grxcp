// The CUDA Runtime API surface GRXCP tracks, and what each entry maps to.
//
// This file is the single source of truth for two tools that would otherwise
// drift apart: grx-conform (which publishes the coverage number) and grxify
// (which translates source and has to explain what it cannot translate). A new
// entry point becomes visible to both by being added here once.
//
// THE DENOMINATOR IS A CHOICE, SO IT IS STATED. The full CUDA Runtime API is
// several hundred entry points, most of them variants, deprecated forms, or
// interop with graphics APIs GRXCP has no analogue for. Counting against all of
// them would produce a flattering-looking number that measures nothing. The
// surface tracked here is the one that actually appears in portable CUDA code:
// device management, memory, streams, events, launch, occupancy, and error
// handling, plus the major features a port is likely to reach for and find
// missing (graphs, dynamic parallelism, IPC, textures). Anything a port would
// hit is either in this table or is a bug in this table.
//
// Status values:
//   MAPPED       implemented with matching semantics
//   PARTIAL      implemented, with a documented behavioural caveat
//   UNSUPPORTED  declared and refuses with grxErrorNotSupported
//   ABSENT       deliberately not provided; a port using it fails to compile
//
// Columns: GRX_CUDA_API(cuda_name, grx_name, status, category, note)
// `grx_name` is empty for ABSENT entries.
//
// Types and enum constants live in a second list, GRX_CUDA_TYPE, at the bottom.
// They are tracked separately because they are not entry points: counting a
// typedef toward an API coverage percentage would inflate it. grxify needs them
// so it does not flag `cudaStream_t` as an unknown CUDA symbol when the compat
// header already covers it. A consumer that only cares about functions can
// ignore GRX_CUDA_TYPE entirely -- it defaults to a no-op.

// --- device management -----------------------------------------------------
GRX_CUDA_API(cudaGetDeviceCount,       grxGetDeviceCount,       MAPPED,      "device", "")
GRX_CUDA_API(cudaSetDevice,            grxSetDevice,            MAPPED,      "device", "")
GRX_CUDA_API(cudaGetDevice,            grxGetDevice,            MAPPED,      "device", "")
GRX_CUDA_API(cudaGetDeviceProperties,  grxGetDeviceProperties,  PARTIAL,     "device", "fields the stack cannot source report -1 rather than a guess; numRegs and barrier count are unknown today")
GRX_CUDA_API(cudaDeviceSynchronize,    grxDeviceSynchronize,    MAPPED,      "device", "")
GRX_CUDA_API(cudaDeviceReset,          grxDeviceReset,          MAPPED,      "device", "")
GRX_CUDA_API(cudaMemGetInfo,           grxMemGetInfo,           MAPPED,      "device", "")
GRX_CUDA_API(cudaDeviceGetAttribute,   grxDeviceGetAttribute,   UNSUPPORTED, "device", "the attribute enum is CUDA-specific; use grxGetDeviceProperties")
GRX_CUDA_API(cudaDeviceCanAccessPeer,  grxDeviceCanAccessPeer,  PARTIAL,     "device", "always reports no: there is no peer path in hardware yet")
GRX_CUDA_API(cudaDeviceEnablePeerAccess, grxDeviceEnablePeerAccess, UNSUPPORTED, "device", "needs NVLink-class remote decode on G100 or the coherent port on the c930 NPU")
GRX_CUDA_API(cudaSetDeviceFlags,       ,                        ABSENT,      "device", "the flags select spin/yield/blocking sync behaviour that has no analogue here")
GRX_CUDA_API(cudaDeviceSetLimit,       ,                        ABSENT,      "device", "no tunable device limits are exposed")

// --- error handling --------------------------------------------------------
GRX_CUDA_API(cudaGetLastError,         grxGetLastError,         MAPPED,      "error", "")
GRX_CUDA_API(cudaPeekAtLastError,      grxPeekAtLastError,      MAPPED,      "error", "")
GRX_CUDA_API(cudaGetErrorName,         grxGetErrorName,         MAPPED,      "error", "")
GRX_CUDA_API(cudaGetErrorString,       grxGetErrorString,       MAPPED,      "error", "")

// --- memory ----------------------------------------------------------------
GRX_CUDA_API(cudaMalloc,               grxMalloc,               MAPPED,      "memory", "")
GRX_CUDA_API(cudaFree,                 grxFree,                 MAPPED,      "memory", "an interior pointer is rejected rather than silently rounded down")
GRX_CUDA_API(cudaMallocHost,           grxMallocHost,           MAPPED,      "memory", "")
GRX_CUDA_API(cudaHostAlloc,            grxMallocHost,           PARTIAL,     "memory", "the write-combining and portable flags are ignored; there is no hardware distinction to honour")
GRX_CUDA_API(cudaFreeHost,             grxFreeHost,             MAPPED,      "memory", "")
GRX_CUDA_API(cudaMallocManaged,        grxMallocManaged,        PARTIAL,     "memory", "refused on backends whose command processor has no page-table walker, rather than returning a pointer that is not really managed")
GRX_CUDA_API(cudaHostRegister,         grxHostRegister,         UNSUPPORTED, "memory", "pinning arbitrary existing host pages needs a driver path into the command processor host aperture that does not exist")
GRX_CUDA_API(cudaHostUnregister,       grxHostUnregister,       UNSUPPORTED, "memory", "see cudaHostRegister")
GRX_CUDA_API(cudaMemcpy,               grxMemcpy,               PARTIAL,     "memory", "a copy kind that contradicts the allocation map is rejected where CUDA leaves it undefined")
GRX_CUDA_API(cudaMemcpyAsync,          grxMemcpyAsync,          MAPPED,      "memory", "")
GRX_CUDA_API(cudaMemcpy2D,             grxMemcpy2D,             PARTIAL,     "memory", "decomposed into row transfers; a strided descriptor path is a follow-up")
GRX_CUDA_API(cudaMemcpy2DAsync,        grxMemcpy2DAsync,        PARTIAL,     "memory", "see cudaMemcpy2D")
GRX_CUDA_API(cudaMemcpy3D,             ,                        ABSENT,      "memory", "the descriptor struct pulls in pitched pointers and arrays; use grxMemcpy2D per slice")
GRX_CUDA_API(cudaMemset,               grxMemset,               MAPPED,      "memory", "")
GRX_CUDA_API(cudaMemsetAsync,          grxMemsetAsync,          MAPPED,      "memory", "")
GRX_CUDA_API(cudaMemset2D,             ,                        ABSENT,      "memory", "no strided fill path yet")
GRX_CUDA_API(cudaPointerGetAttributes, grxPointerGetAttributes, MAPPED,      "memory", "")
GRX_CUDA_API(cudaMallocPitch,          ,                        ABSENT,      "memory", "pitched allocation belongs with the strided copy path")
GRX_CUDA_API(cudaMemPrefetchAsync,     ,                        ABSENT,      "memory", "managed-memory migration hints need a page-migration engine")
GRX_CUDA_API(cudaMemAdvise,            ,                        ABSENT,      "memory", "see cudaMemPrefetchAsync")
GRX_CUDA_API(cudaMemcpyToSymbol,       grxMemcpyToSymbol,       PARTIAL,     "memory", "grxcc registers device symbols from the ELF; a write after the module is loaded reloads it (cuda_mapping.md 7.23)")
GRX_CUDA_API(cudaMemcpyFromSymbol,     grxMemcpyFromSymbol,     PARTIAL,     "memory", "__constant__ only: the driver gives the host no view of a loaded module, so a __device__ symbol a kernel wrote cannot be read back and is refused")
GRX_CUDA_API(cudaGetSymbolAddress,     grxGetSymbolAddress,     PARTIAL,     "memory", "returns the link address, which is where the loader puts it; grxMemcpy will not accept it, because the allocation map does not own it")
GRX_CUDA_API(cudaGetSymbolSize,        grxGetSymbolSize,        MAPPED,      "memory", "")

// --- streams ---------------------------------------------------------------
GRX_CUDA_API(cudaStreamCreate,           grxStreamCreate,           PARTIAL, "stream", "ordering is correct; streams do not yet run concurrently because the command processor defaults to one queue")
GRX_CUDA_API(cudaStreamCreateWithFlags,  grxStreamCreateWithFlags,  PARTIAL, "stream", "see cudaStreamCreate")
GRX_CUDA_API(cudaStreamCreateWithPriority, grxStreamCreateWithPriority, PARTIAL, "stream", "priority reaches the queue but has no effect until multi-queue arbitration is wired")
GRX_CUDA_API(cudaStreamDestroy,          grxStreamDestroy,          MAPPED,  "stream", "")
GRX_CUDA_API(cudaStreamSynchronize,      grxStreamSynchronize,      MAPPED,  "stream", "")
GRX_CUDA_API(cudaStreamQuery,            grxStreamQuery,            MAPPED,  "stream", "")
GRX_CUDA_API(cudaStreamWaitEvent,        grxStreamWaitEvent,        MAPPED,  "stream", "")
GRX_CUDA_API(cudaStreamAddCallback,      ,                          ABSENT,  "stream", "host callbacks need a driver-side completion thread the runtime does not run")
GRX_CUDA_API(cudaStreamBeginCapture,     ,                          ABSENT,  "stream", "graph capture belongs with CUDA graphs")
GRX_CUDA_API(cudaStreamEndCapture,       ,                          ABSENT,  "stream", "see cudaStreamBeginCapture")

// --- events ----------------------------------------------------------------
GRX_CUDA_API(cudaEventCreate,          grxEventCreate,          MAPPED,      "event", "")
GRX_CUDA_API(cudaEventCreateWithFlags, grxEventCreateWithFlags, MAPPED,      "event", "")
GRX_CUDA_API(cudaEventDestroy,         grxEventDestroy,         MAPPED,      "event", "")
GRX_CUDA_API(cudaEventRecord,          grxEventRecord,          MAPPED,      "event", "")
GRX_CUDA_API(cudaEventSynchronize,     grxEventSynchronize,     MAPPED,      "event", "")
GRX_CUDA_API(cudaEventQuery,           grxEventQuery,           MAPPED,      "event", "")
GRX_CUDA_API(cudaEventElapsedTime,     grxEventElapsedTime,     PARTIAL,     "event", "brackets execution with the driver's host-clock timestamps, not device ones -- on a simulator that measures the simulator; reported by eventTimingIsDeviceSide")

// --- launch and occupancy --------------------------------------------------
GRX_CUDA_API(cudaLaunchKernel,            grxLaunchKernel,            PARTIAL, "launch", "requires a registered parameter layout; a stub without one is refused rather than packed by guesswork")
GRX_CUDA_API(cudaLaunchCooperativeKernel, grxLaunchCooperativeKernel, MAPPED,  "launch", "")
GRX_CUDA_API(cudaLaunchKernelExC,         grxLaunchKernelEx,          PARTIAL, "launch", "cluster dimension, cooperative and priority attributes are honoured; the shared-memory carve-out attribute is refused because the hardware register does not exist")
GRX_CUDA_API(cudaFuncGetAttributes,       grxFuncGetAttributes,       PARTIAL, "launch", "numRegs reports -1 until the toolchain emits register metadata")
GRX_CUDA_API(cudaFuncSetAttribute,        ,                           ABSENT,  "launch", "the attributes worth setting need the unified L1/shared carve-out register")
GRX_CUDA_API(cudaFuncSetCacheConfig,      ,                           ABSENT,  "launch", "see cudaFuncSetAttribute")
GRX_CUDA_API(cudaOccupancyMaxActiveBlocksPerMultiprocessor, grxOccupancyMaxActiveBlocksPerMultiprocessor, PARTIAL, "launch", "computed from the CTA dispatcher's warp, slot and shared-memory bounds; there is no register bound because the dispatcher does not gate admission on registers")
GRX_CUDA_API(cudaOccupancyMaxPotentialBlockSize, grxOccupancyMaxPotentialBlockSize, PARTIAL, "launch", "see cudaOccupancyMaxActiveBlocksPerMultiprocessor")
GRX_CUDA_API(cudaLaunchHostFunc,          ,                           ABSENT,  "launch", "see cudaStreamAddCallback")

// --- modules (driver API, commonly reached for by language runtimes) -------
GRX_CUDA_API(cuModuleLoad,             grxModuleLoad,           MAPPED,      "module", "")
GRX_CUDA_API(cuModuleLoadData,         grxModuleLoadData,       MAPPED,      "module", "")
GRX_CUDA_API(cuModuleUnload,           grxModuleUnload,         MAPPED,      "module", "")
GRX_CUDA_API(cuModuleGetFunction,      grxModuleGetFunction,    MAPPED,      "module", "")
GRX_CUDA_API(cuLaunchKernel,           grxLaunchFunction,       PARTIAL,     "module", "takes one packed argument blob rather than CUDA's void** plus extra array")
GRX_CUDA_API(cuModuleGetGlobal,        ,                        ABSENT,      "module", "the symbol path is keyed on a host stand-in grxcc registers, not on a module handle and a name")

// --- features with no GRXCP implementation ---------------------------------
GRX_CUDA_API(cudaGraphCreate,          ,                        ABSENT,      "graph",   "CUDA graphs are deferred; the driver's batched command submission is the natural substrate when they land")
GRX_CUDA_API(cudaGraphLaunch,          ,                        ABSENT,      "graph",   "see cudaGraphCreate")
GRX_CUDA_API(cudaGraphInstantiate,     ,                        ABSENT,      "graph",   "see cudaGraphCreate")
GRX_CUDA_API(cudaIpcGetMemHandle,      ,                        ABSENT,      "ipc",     "no cross-process device memory sharing")
GRX_CUDA_API(cudaIpcOpenMemHandle,     ,                        ABSENT,      "ipc",     "see cudaIpcGetMemHandle")
GRX_CUDA_API(cudaCreateTextureObject,  ,                        ABSENT,      "texture", "the texture units exist in hardware but are driven by the graphics path and not exposed to compute yet")
GRX_CUDA_API(cudaDestroyTextureObject, ,                        ABSENT,      "texture", "see cudaCreateTextureObject")
GRX_CUDA_API(cudaMallocArray,          ,                        ABSENT,      "texture", "see cudaCreateTextureObject")
GRX_CUDA_API(cudaGraphicsGLRegisterBuffer, ,                    ABSENT,      "interop", "no graphics interop surface")
GRX_CUDA_API(cudaProfilerStart,        ,                        ABSENT,      "profiler", "use grx-prof; there is no in-process profiler control API")
GRX_CUDA_API(cudaProfilerStop,         ,                        ABSENT,      "profiler", "see cudaProfilerStart")

// ---------------------------------------------------------------------------
// Types and enum constants
//
// These are handled by grx_cuda_compat.h as macro renames, so translated source
// may keep using the CUDA spelling. They are listed here so grxify recognises
// them as covered rather than reporting them as unknown CUDA symbols.
//
// Columns: GRX_CUDA_TYPE(cuda_name, grx_name, status, note)
// ---------------------------------------------------------------------------

#ifndef GRX_CUDA_TYPE
#define GRX_CUDA_TYPE(cuda, grx, status, note)
#define GRX_CUDA_TYPE_LOCAL_NOOP
#endif

GRX_CUDA_TYPE(cudaError_t,               grxError_t,               MAPPED, "")
GRX_CUDA_TYPE(cudaSuccess,               grxSuccess,               MAPPED, "")
GRX_CUDA_TYPE(cudaErrorInvalidValue,     grxErrorInvalidValue,     MAPPED, "")
GRX_CUDA_TYPE(cudaErrorMemoryAllocation, grxErrorMemoryAllocation, MAPPED, "")
GRX_CUDA_TYPE(cudaErrorInvalidDevice,    grxErrorInvalidDevice,    MAPPED, "")
GRX_CUDA_TYPE(cudaErrorNotReady,         grxErrorNotReady,         MAPPED, "")
GRX_CUDA_TYPE(cudaErrorNotSupported,     grxErrorNotSupported,     MAPPED, "")
GRX_CUDA_TYPE(cudaErrorLaunchFailure,    grxErrorLaunchFailure,    MAPPED, "")
GRX_CUDA_TYPE(cudaErrorLaunchOutOfResources, grxErrorLaunchOutOfResources, MAPPED, "")
GRX_CUDA_TYPE(cudaErrorFileNotFound,     grxErrorFileNotFound,     MAPPED, "")

GRX_CUDA_TYPE(cudaDeviceProp,            grxDeviceProp_t,          PARTIAL, "field set differs: GRX-specific fields are added and CUDA fields the stack cannot source are absent")
GRX_CUDA_TYPE(cudaStream_t,              grxStream_t,              MAPPED, "")
GRX_CUDA_TYPE(cudaEvent_t,               grxEvent_t,               MAPPED, "")
GRX_CUDA_TYPE(cudaFuncAttributes,        grxFuncAttributes,        PARTIAL, "numRegs reports -1 until the toolchain supplies it")
GRX_CUDA_TYPE(cudaPointerAttributes,     grxPointerAttributes,     MAPPED, "")
GRX_CUDA_TYPE(cudaMemcpyKind,            grxMemcpyKind,            MAPPED, "")
GRX_CUDA_TYPE(cudaMemcpyHostToHost,      grxMemcpyHostToHost,      MAPPED, "")
GRX_CUDA_TYPE(cudaMemcpyHostToDevice,    grxMemcpyHostToDevice,    MAPPED, "")
GRX_CUDA_TYPE(cudaMemcpyDeviceToHost,    grxMemcpyDeviceToHost,    MAPPED, "")
GRX_CUDA_TYPE(cudaMemcpyDeviceToDevice,  grxMemcpyDeviceToDevice,  MAPPED, "")
GRX_CUDA_TYPE(cudaMemcpyDefault,         grxMemcpyDefault,         MAPPED, "")
GRX_CUDA_TYPE(cudaStreamDefault,         grxStreamDefault,         MAPPED, "")
GRX_CUDA_TYPE(cudaStreamNonBlocking,     grxStreamNonBlocking,     MAPPED, "")
GRX_CUDA_TYPE(cudaEventDefault,          grxEventDefault,          MAPPED, "")
GRX_CUDA_TYPE(cudaEventBlockingSync,     grxEventBlockingSync,     MAPPED, "")
GRX_CUDA_TYPE(cudaEventDisableTiming,    grxEventDisableTiming,    MAPPED, "")

GRX_CUDA_TYPE(cudaGraph_t,               ,                         ABSENT, "CUDA graphs are deferred")
GRX_CUDA_TYPE(cudaGraphExec_t,           ,                         ABSENT, "CUDA graphs are deferred")
GRX_CUDA_TYPE(cudaTextureObject_t,       ,                         ABSENT, "textures are not exposed to compute yet")
GRX_CUDA_TYPE(cudaArray_t,               ,                         ABSENT, "textures are not exposed to compute yet")
GRX_CUDA_TYPE(cudaIpcMemHandle_t,        ,                         ABSENT, "no cross-process sharing")

#ifdef GRX_CUDA_TYPE_LOCAL_NOOP
#undef GRX_CUDA_TYPE
#undef GRX_CUDA_TYPE_LOCAL_NOOP
#endif
