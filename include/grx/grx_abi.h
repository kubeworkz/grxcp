// GRXCP — on-disk and cross-tool ABI structures.
//
// This is the one header shared by the host runtime and the device toolchain,
// so it contains POD layouts only: no functions, no C++ types, nothing that
// depends on either side's headers (AGENTS.md section 2).
//
// It defines the .grxfat fat binary that grxcc embeds in a host ELF, and the
// kernel parameter descriptors the runtime needs to pack a CUDA-style
// void** argument array into the flat blob the driver wants.

#ifndef GRX_ABI_H
#define GRX_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Fat binary container
//
// Layout:  [grx_fatbin_header][grx_fatbin_entry × n][payload bytes …]
//
// Payload offsets are measured from the start of the header. Multiple entries
// let one host binary carry, say, an rv64 image with tensor-core instructions
// and an rv64 image without, and let the runtime pick by matching required_isa
// against the device's VX_CAPS_ISA_FLAGS. That is CUDA's fatbin idea with
// GRX's actual capability bits in place of SM version numbers.
// ---------------------------------------------------------------------------

#define GRX_FATBIN_MAGIC   0x46585247u   // 'G','R','X','F' little-endian
#define GRX_FATBIN_VERSION 1u

// What a payload contains.
#define GRX_IMAGE_VXBIN    0u   // linked device binary, ready for the loader
#define GRX_IMAGE_SPIRV    1u   // SPIR-V, JIT-compiled by the device stack
#define GRX_IMAGE_LLVM_IR  2u   // LLVM bitcode, for a future runtime compiler

typedef struct {
  uint32_t magic;         // GRX_FATBIN_MAGIC
  uint16_t version;       // GRX_FATBIN_VERSION
  uint16_t num_entries;
  uint64_t total_size;    // header + entries + payloads
} grx_fatbin_header;

typedef struct {
  uint32_t kind;          // GRX_IMAGE_*
  uint32_t xlen;          // 32 or 64
  // Extension bits this image REQUIRES, in the VX_ISA_EXT_* encoding shifted
  // down by 32 (that is, bit 0 here is VX_ISA_EXT_ICACHE). An image is
  // loadable when the device advertises every bit set here. Zero means the
  // image needs no optional extension.
  uint32_t required_isa;
  uint32_t reserved;
  uint64_t offset;        // from the start of the header
  uint64_t size;
} grx_fatbin_entry;

// ---------------------------------------------------------------------------
// Kernel parameter descriptors
//
// The driver takes kernel arguments as one flat host blob that it stages into
// a device scratch slot. CUDA's launch API takes void** -- an array of
// POINTERS to the values -- which carries no size information. Bridging the
// two requires knowing each parameter's offset and width, which is what these
// descriptors carry. grxcc emits one array per kernel; a program that calls
// grxLaunchKernel on a stub with no registered layout gets a clear error
// rather than a silently corrupt argument blob.
// ---------------------------------------------------------------------------

typedef struct {
  uint16_t offset;      // byte offset within the packed blob
  uint16_t size;        // byte width of this parameter on the DEVICE
  uint8_t  is_pointer;  // 1 if the value is a device address
  uint8_t  reserved[3];
} grx_kernel_param;

// ---------------------------------------------------------------------------
// Device variable descriptors
//
// CUDA's cudaMemcpyToSymbol takes the HOST address of a `__device__` or
// `__constant__` variable and finds the device one. grxcc supplies the link:
// it reads the address and size out of the device ELF it just built and
// registers them against the host stand-in's address.
//
// `device_vma` is the symbol's link address, which for GRX-G100 is also where
// it lands -- every image links at STARTUP_ADDR and is loaded there. The
// runtime still computes the payload offset as (vma - min_vma) rather than
// trusting that, so the day a loader relocates, one subtraction changes.
// ---------------------------------------------------------------------------

typedef struct {
  const char* device_name;   // as it appears in the device ELF's symbol table
  uint64_t    device_vma;    // link address of the symbol
  uint32_t    size;          // bytes
  uint32_t    is_constant;   // 1 for __constant__, 0 for __device__
} grx_var_desc;

typedef struct {
  const char*             device_name;   // entry name in the .vxbin symbol table
  const grx_kernel_param* params;
  uint32_t                num_params;
  uint32_t                args_size;     // total packed blob size
  uint32_t                static_smem;   // per-kernel local-memory need, bytes
  int32_t                 num_regs;      // -1 when the toolchain cannot say
  uint32_t                max_threads_per_block;  // 0 = no __launch_bounds__
  uint32_t                reserved;
} grx_kernel_desc;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRX_ABI_H
