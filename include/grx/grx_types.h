// GRXCP — core types shared by the host runtime API.
//
// Design note: this header is deliberately self-contained (standard C headers
// only). Nothing here depends on a GRX-G100 build-time configuration header --
// device capabilities are discovered at runtime through grxGetDeviceProperties,
// exactly as vortex2.h discovers them through vx_device_query.

#ifndef GRX_TYPES_H
#define GRX_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

typedef enum {
  grxSuccess                      = 0,
  grxErrorInvalidValue            = 1,
  grxErrorMemoryAllocation        = 2,
  grxErrorInitializationError     = 3,
  grxErrorInvalidDevice           = 4,
  grxErrorInvalidDevicePointer    = 5,
  grxErrorInvalidMemcpyDirection  = 6,
  grxErrorInvalidResourceHandle   = 7,
  grxErrorNotReady                = 8,
  grxErrorNotSupported            = 9,
  grxErrorInvalidKernelImage      = 10,
  grxErrorInvalidDeviceFunction   = 11,
  grxErrorLaunchFailure           = 12,
  grxErrorLaunchOutOfResources    = 13,
  grxErrorDeviceLost              = 14,
  grxErrorTimeout                 = 15,
  grxErrorFileNotFound            = 16,
  grxErrorUnknown                 = 999
} grxError_t;

const char* grxGetErrorName   (grxError_t e);
const char* grxGetErrorString (grxError_t e);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

typedef struct { unsigned int x, y, z; } dim3_t;

#ifdef __cplusplus
struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};
#else
typedef dim3_t dim3;
#endif

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

typedef struct grxStream*   grxStream_t;
typedef struct grxEvent*    grxEvent_t;
typedef struct grxModule*   grxModule_t;
typedef struct grxFunction* grxFunction_t;

#define grxStreamDefault      0x00u
#define grxStreamNonBlocking  0x01u
#define grxEventDefault       0x00u
#define grxEventBlockingSync  0x01u
#define grxEventDisableTiming 0x02u

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

typedef enum {
  grxMemcpyHostToHost     = 0,
  grxMemcpyHostToDevice   = 1,
  grxMemcpyDeviceToHost   = 2,
  grxMemcpyDeviceToDevice = 3,
  // Preferred form: the runtime resolves direction from its allocation map.
  // Unlike CUDA, an explicit kind that contradicts the map is an error rather
  // than undefined behavior.
  grxMemcpyDefault        = 4
} grxMemcpyKind;

typedef enum {
  grxMemoryTypeUnregistered = 0,
  grxMemoryTypeHost         = 1,
  grxMemoryTypeDevice       = 2,
  grxMemoryTypeManaged      = 3
} grxMemoryType;

typedef struct {
  grxMemoryType type;
  int           device;
  void*         devicePointer;
  void*         hostPointer;
  size_t        allocationSize;
} grxPointerAttributes;

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------
//
// Named after CUDA's cudaDataType (CUDA_R_16F and friends) because library
// entry points that take one -- grxblasGemmEx -- are shaped after the CUDA
// calls that take the other. Only the types something actually implements are
// listed: an enumerator for a format no kernel can consume is a promise the
// compiler will let a caller make and the runtime will then break.
typedef enum {
  GRX_R_16F = 2,   // IEEE binary16
  GRX_R_32F = 0    // IEEE binary32
} grxDataType_t;

// ---------------------------------------------------------------------------
// Device model
// ---------------------------------------------------------------------------

typedef enum { GRX_DEVICE_TYPE_GPU = 0, GRX_DEVICE_TYPE_NPU = 1 } grxDeviceType_t;

// Which execution backend a device is running on. Programs that reason about
// wall-clock time must check this: a simx device is several orders of
// magnitude slower than silicon, and a benchmark that does not say which
// backend it ran on is not reporting a result.
typedef enum {
  GRX_BACKEND_SIMX    = 0,
  GRX_BACKEND_RTLSIM  = 1,
  GRX_BACKEND_XRT     = 2,
  GRX_BACKEND_OPAE    = 3,
  GRX_BACKEND_GEM5    = 4,
  GRX_BACKEND_SILICON = 5
} grxBackend_t;

// Capability profile: which GRXCP subsystems a device actually implements.
// Present from v1 so the GRX930 NPU can join as a second device without an
// API break -- it will advertise GEMM and memcpy but not kernel launch.
#define GRX_CAP_KERNEL_LAUNCH       (1u << 0)
#define GRX_CAP_STREAMS             (1u << 1)
#define GRX_CAP_EVENTS              (1u << 2)
#define GRX_CAP_MEMCPY              (1u << 3)
#define GRX_CAP_GEMM                (1u << 4)
#define GRX_CAP_UNIFIED_ADDRESSING  (1u << 5)
#define GRX_CAP_TENSOR_CORE         (1u << 6)
#define GRX_CAP_ASYNC_COPY          (1u << 7)
#define GRX_CAP_RAY_TRACING         (1u << 8)
#define GRX_CAP_COOPERATIVE_LAUNCH  (1u << 9)

typedef struct {
  char            name[128];
  grxDeviceType_t deviceType;
  grxBackend_t    backend;
  unsigned int    capabilities;      // GRX_CAP_* bitmask
  int             computeCapabilityMajor;   // G100 design target: 10
  int             computeCapabilityMinor;   //                      0

  // --- execution geometry (all sourced from vx_device_query) ---
  int    warpSize;                       // VX_CAPS_NUM_THREADS
  int    maxWarpsPerMultiProcessor;      // VX_CAPS_NUM_WARPS
  int    multiProcessorCount;            // VX_CAPS_NUM_CORES (cores x clusters)
  int    clusterCount;                   // VX_CAPS_NUM_CLUSTERS
  int    socketSize;                     // VX_CAPS_SOCKET_SIZE
  int    issueWidth;                     // VX_CAPS_ISSUE_WIDTH
  int    maxThreadsPerBlock;
  int    maxThreadsDim[3];
  int    maxGridSize[3];
  int    numBarriers;

  // --- memory ---
  size_t totalGlobalMem;                 // VX_CAPS_GLOBAL_MEM_SIZE
  size_t sharedMemPerMultiprocessor;     // VX_CAPS_LOCAL_MEM_SIZE
  size_t sharedMemPerBlock;              // usable per-CTA share of the above
  int    memBankCount;                   // VX_CAPS_NUM_MEM_BANKS
  size_t memBankSize;                    // VX_CAPS_MEM_BANK_SIZE
  int    cacheLineSize;                  // VX_CAPS_CACHE_LINE_SIZE
  int    clockRateMHz;                   // VX_CAPS_CLOCK_RATE
  size_t peakMemoryBandwidthMBs;         // VX_CAPS_PEAK_MEM_BW
  int    unifiedAddressing;              // VX_CAPS_VM_SUPPORT
  int    managedMemory;                  // VX_CAPS_VM_SUPPORT and backend has VM
  size_t pinnedMemTotal;                 // VX_CAPS_VM_PINNED_SIZE
  size_t pinnedMemFree;                  // VX_CAPS_VM_PINNED_FREE

  // --- honesty flags: see AGENTS.md section 3 ---
  // Set when a documented software emulation stands in for hardware. Never
  // remove one of these without removing the emulation it describes.
  int    warpShuffleIsEmulated;      // 0: the ISA has SHFL.* and VOTE.*
  int    eventTimingIsDeviceSide;    // 0 while CP profiling writeback is a stub
  int    constantMemoryIsGlobal;     // 1 while __constant__ lowers to RO global
} grxDeviceProp_t;

// Attributes of a compiled kernel. Fields the toolchain cannot yet supply
// report -1 rather than a plausible-looking invention.
typedef struct {
  size_t sharedSizeBytes;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int    maxThreadsPerBlock;
  int    numRegs;              // -1 until grxcc emits register metadata (phase 4)
  int    ptxVersion;           // -1: GRXCP has no PTX analogue by design
  int    binaryVersion;
  uint64_t deviceEntryPC;      // vx_kernel_address
} grxFuncAttributes;

// ---------------------------------------------------------------------------
// Extended launch attributes
// ---------------------------------------------------------------------------

typedef enum {
  GRX_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION   = 0,
  GRX_LAUNCH_ATTRIBUTE_COOPERATIVE         = 1,
  GRX_LAUNCH_ATTRIBUTE_PRIORITY            = 2,
  GRX_LAUNCH_ATTRIBUTE_SHARED_MEM_CARVEOUT = 3
} grxLaunchAttributeID;

typedef struct {
  grxLaunchAttributeID id;
  union {
    dim3_t clusterDim;
    int    cooperative;
    int    priority;
    int    sharedMemCarveoutPercent;
  } val;
} grxLaunchAttribute;

typedef struct {
  dim3_t              gridDim;
  dim3_t              blockDim;
  size_t              dynamicSharedMem;
  grxStream_t         stream;
  unsigned int        numAttributes;
  grxLaunchAttribute* attributes;
} grxLaunchConfig_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRX_TYPES_H
