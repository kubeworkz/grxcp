// GRXCP — CUDA source compatibility shim.
//
// Include this INSTEAD OF <cuda_runtime.h> to compile CUDA-shaped source
// against GRXCP. It is a pure rename layer: no behavior is defined here, so a
// program that works through this header works identically when written
// against the native grx* names.
//
// What this header deliberately does NOT do:
//   * it does not paper over the gaps in docs/designs/cuda_mapping.md 7.
//     cudaGraph*, dynamic parallelism, IPC, and texture objects are simply
//     absent -- a port that needs them fails at compile time with a clear
//     "undeclared identifier" rather than at runtime with a mystery.
//   * it does not emulate CUDA semantics GRXCP intentionally tightened.
//     cudaMemcpy with a kind that contradicts the allocation map returns an
//     error here where CUDA is undefined. That is a feature.

#ifndef GRX_CUDA_COMPAT_H
#define GRX_CUDA_COMPAT_H

#include "grx.h"

// --- errors ---------------------------------------------------------------
#define cudaError_t                     grxError_t
#define cudaSuccess                     grxSuccess
#define cudaErrorInvalidValue           grxErrorInvalidValue
#define cudaErrorMemoryAllocation       grxErrorMemoryAllocation
#define cudaErrorInvalidDevice          grxErrorInvalidDevice
#define cudaErrorNotReady               grxErrorNotReady
#define cudaErrorNotSupported           grxErrorNotSupported
#define cudaErrorLaunchFailure          grxErrorLaunchFailure
#define cudaErrorLaunchOutOfResources   grxErrorLaunchOutOfResources
#define cudaGetErrorName                grxGetErrorName
#define cudaGetErrorString              grxGetErrorString
#define cudaGetLastError                grxGetLastError
#define cudaPeekAtLastError             grxPeekAtLastError

// --- device ---------------------------------------------------------------
#define cudaDeviceProp                  grxDeviceProp_t
#define cudaGetDeviceCount              grxGetDeviceCount
#define cudaSetDevice                   grxSetDevice
#define cudaGetDevice                   grxGetDevice
#define cudaGetDeviceProperties         grxGetDeviceProperties
#define cudaDeviceSynchronize           grxDeviceSynchronize
#define cudaDeviceReset                 grxDeviceReset
#define cudaMemGetInfo                  grxMemGetInfo
#define cudaDeviceGetAttribute          grxDeviceGetAttribute
#define cudaDeviceCanAccessPeer         grxDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess      grxDeviceEnablePeerAccess

// --- memory ---------------------------------------------------------------
#define cudaMalloc                      grxMalloc
#define cudaFree                        grxFree
#define cudaMallocHost                  grxMallocHost
#define cudaHostAlloc(p, s, f)          grxMallocHost((p), (s))
#define cudaFreeHost                    grxFreeHost
#define cudaMallocManaged               grxMallocManaged
#define cudaHostRegister                grxHostRegister
#define cudaHostUnregister              grxHostUnregister
#define cudaMemcpy                      grxMemcpy
#define cudaMemcpyAsync                 grxMemcpyAsync
#define cudaMemcpy2D                    grxMemcpy2D
#define cudaMemcpy2DAsync               grxMemcpy2DAsync
#define cudaMemset                      grxMemset
#define cudaMemsetAsync                 grxMemsetAsync
#define cudaPointerGetAttributes        grxPointerGetAttributes
#define cudaPointerAttributes           grxPointerAttributes
#define cudaMemcpyKind                  grxMemcpyKind
#define cudaMemcpyHostToHost            grxMemcpyHostToHost
#define cudaMemcpyHostToDevice          grxMemcpyHostToDevice
#define cudaMemcpyDeviceToHost          grxMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice        grxMemcpyDeviceToDevice
#define cudaMemcpyDefault               grxMemcpyDefault

// --- streams --------------------------------------------------------------
#define cudaStream_t                    grxStream_t
#define cudaStreamCreate                grxStreamCreate
#define cudaStreamCreateWithFlags       grxStreamCreateWithFlags
#define cudaStreamCreateWithPriority    grxStreamCreateWithPriority
#define cudaStreamDestroy               grxStreamDestroy
#define cudaStreamSynchronize           grxStreamSynchronize
#define cudaStreamQuery                 grxStreamQuery
#define cudaStreamWaitEvent             grxStreamWaitEvent
#define cudaStreamDefault               grxStreamDefault
#define cudaStreamNonBlocking           grxStreamNonBlocking

// --- events ---------------------------------------------------------------
#define cudaEvent_t                     grxEvent_t
#define cudaEventCreate                 grxEventCreate
#define cudaEventCreateWithFlags        grxEventCreateWithFlags
#define cudaEventDestroy                grxEventDestroy
#define cudaEventRecord                 grxEventRecord
#define cudaEventSynchronize            grxEventSynchronize
#define cudaEventQuery                  grxEventQuery
#define cudaEventElapsedTime            grxEventElapsedTime
#define cudaEventDefault                grxEventDefault
#define cudaEventBlockingSync           grxEventBlockingSync
#define cudaEventDisableTiming          grxEventDisableTiming

// --- launch and occupancy -------------------------------------------------
#define cudaLaunchKernel                grxLaunchKernel
#define cudaLaunchCooperativeKernel     grxLaunchCooperativeKernel
#define cudaLaunchKernelExC             grxLaunchKernelEx
#define cudaFuncAttributes              grxFuncAttributes
#define cudaFuncGetAttributes           grxFuncGetAttributes
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
        grxOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize \
        grxOccupancyMaxPotentialBlockSize

#endif  // GRX_CUDA_COMPAT_H
