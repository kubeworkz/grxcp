// GRXCP — kernel launch and occupancy.
//
// The launch descriptor is the whole job: take a grid, a block, a shared-memory
// size, and an argument array, and produce one vx_launch_info_t the command
// processor can act on. The driver stages the argument blob into a device
// scratch slot and programs the kernel-management unit's argument registers
// itself, which is why nothing here allocates an argument buffer.
//
// Argument packing is where a runtime quietly corrupts data if it is careless.
// CUDA's void** carries pointers to values and no sizes, so packing requires
// the kernel's parameter layout. When a layout was registered, arguments are
// packed to the device's widths -- including narrowing a 64-bit host pointer to
// a 32-bit device pointer, with a check that no significant bits are lost.
// When no layout was registered, the launch is REFUSED. Guessing that every
// argument is pointer-sized is the kind of assumption that produces a wrong
// answer instead of an error message.

#include "internal.h"

#include <grx/grx_runtime.h>

#include <cstring>
#include <vector>

namespace grxcp {

namespace {

// Per-thread <<<>>> configuration stack. grxcc lowers a launch into a push,
// a call to the host stub, and a pop inside that stub.
struct CallConfig {
  dim3_t      grid{1, 1, 1};
  dim3_t      block{1, 1, 1};
  size_t      shared = 0;
  grxStream_t stream = nullptr;
};
thread_local std::vector<CallConfig> g_call_configs;

uint32_t volume(const dim3_t& d) {
  return (d.x ? d.x : 1) * (d.y ? d.y : 1) * (d.z ? d.z : 1);
}

// Fewer dimensions means a shorter grid walk in the kernel-management unit, so
// report the smallest rank that describes this launch.
uint32_t rank_of(const dim3_t& grid, const dim3_t& block) {
  if (grid.z > 1 || block.z > 1) return 3;
  if (grid.y > 1 || block.y > 1) return 2;
  return 1;
}

grxError_t pack_arguments(const KernelBinding& k, void** args,
                          std::vector<uint8_t>* out) {
  if (!k.has_layout) {
    // No parameter layout: the widths are unknowable. See the file header.
    return grxErrorInvalidDeviceFunction;
  }
  if (k.num_params > 0 && !args) return grxErrorInvalidValue;

  out->assign(k.args_size, 0);
  for (uint32_t i = 0; i < k.num_params; ++i) {
    const grx_kernel_param& p = k.params[i];
    if ((size_t)p.offset + p.size > out->size()) return grxErrorInvalidValue;
    if (!args[i]) return grxErrorInvalidValue;

    if (p.is_pointer && p.size < sizeof(void*)) {
      // 32-bit device, 64-bit host. The high bytes must be zero or the pointer
      // does not survive the narrowing -- the failure the SPIR-V path accepts
      // as an unfixed risk, refused here instead.
      uint64_t value = 0;
      std::memcpy(&value, args[i], sizeof(void*));
      if ((value >> (p.size * 8)) != 0) return grxErrorInvalidValue;
      std::memcpy(out->data() + p.offset, &value, p.size);
    } else {
      std::memcpy(out->data() + p.offset, args[i], p.size);
    }
  }
  return grxSuccess;
}

grxError_t validate(const grxDeviceProp_t& prop, const KernelBinding& k,
                    const dim3_t& grid, const dim3_t& block, size_t shared) {
  const uint32_t threads = volume(block);
  if (threads == 0) return grxErrorInvalidValue;

  if ((int)threads > prop.maxThreadsPerBlock) return grxErrorLaunchOutOfResources;
  if (k.max_threads_per_block > 0 && threads > k.max_threads_per_block)
    return grxErrorLaunchOutOfResources;

  if (grid.x > (unsigned)prop.maxGridSize[0] ||
      grid.y > (unsigned)prop.maxGridSize[1] ||
      grid.z > (unsigned)prop.maxGridSize[2])
    return grxErrorInvalidValue;

  const size_t total_smem = shared + k.static_smem;
  if (total_smem > prop.sharedMemPerBlock) return grxErrorLaunchOutOfResources;

  // Deliberately NOT rejected: a block size that is not a multiple of the warp
  // size. The chip design recommends multiples of the warp width for
  // scheduling efficiency, but the dispatcher handles a partial warp through
  // the thread mask, and CUDA code in the wild launches 100-thread blocks. A
  // correctness error for a performance recommendation would break ports for
  // no safety gain.

  // A cooperative launch must fit the whole grid on the device at once, or the
  // grid-wide barrier deadlocks. That is a property of the launch, not of the
  // kernel, so it is checked by the caller that sets the cooperative flag.
  return grxSuccess;
}

grxError_t launch_common(const KernelBinding& k, const dim3_t& grid,
                         const dim3_t& block, const dim3_t& cluster,
                         size_t shared, const void* args_blob,
                         size_t args_size, int device, grxStream_t stream) {
  Device* d = nullptr;
  grxError_t e = acquire_device(device, &d);
  if (e != grxSuccess) return e;

  e = validate(d->prop, k, grid, block, shared);
  if (e != grxSuccess) return e;

  vx_queue_h q = nullptr;
  e = resolve_stream(stream, device, &q, nullptr);
  if (e != grxSuccess) return e;

  // grx-sanitize arms the device-side checker with the allocation map as it
  // stands right now. Draining first is not optional: the map is a single
  // device-resident table, so a kernel still in flight would be checked
  // against a table that is being rewritten underneath it. Sanitized runs
  // therefore serialize at every launch, which is a cost the mode accepts.
  if (sanitize_enabled()) {
    e = sync_all_streams(device);
    if (e != grxSuccess) return e;
    const uint32_t grid_threads =
        (grid.x ? grid.x : 1) * (grid.y ? grid.y : 1) * (grid.z ? grid.z : 1) *
        (block.x ? block.x : 1) * (block.y ? block.y : 1) *
        (block.z ? block.z : 1);
    e = sanitize_arm(*d, (uint32_t)(shared + k.static_smem), grid_threads,
                     k.name, k.module_path, k.module_elf, k.sanitized);
    if (e != grxSuccess) return e;
  }

  vx_launch_info_t info{};
  info.struct_size = sizeof(info);
  info.next        = nullptr;
  info.kernel      = k.kernel;
  info.args_host   = args_blob;
  info.args_size   = args_size;
  info.ndim        = rank_of(grid, block);
  info.grid_dim[0]  = grid.x  ? grid.x  : 1;
  info.grid_dim[1]  = grid.y  ? grid.y  : 1;
  info.grid_dim[2]  = grid.z  ? grid.z  : 1;
  info.block_dim[0] = block.x ? block.x : 1;
  info.block_dim[1] = block.y ? block.y : 1;
  info.block_dim[2] = block.z ? block.z : 1;
  // The dispatcher's per-CTA local-memory stride: the launch's request plus
  // whatever per-kernel need the toolchain recorded. Nothing is reserved on
  // top -- the warp shuffle is a hardware instruction and stages through no
  // memory (include/grx/device/grx_warp.h).
  info.lmem_size    = (uint32_t)(shared + k.static_smem);
  info.cluster_dim[0] = cluster.x ? cluster.x : 1;
  info.cluster_dim[1] = cluster.y ? cluster.y : 1;
  info.cluster_dim[2] = cluster.z ? cluster.z : 1;

  std::vector<vx_event_h> waits;
  collect_wait_events(stream, device, &waits);

  // grx-prof brackets the launch with a device sync on each side so the
  // performance-counter delta belongs to this kernel and nothing else.
  ProfileSample sample;
  const bool profiling = profile_begin(device, &sample);

  vx_event_h completion = nullptr;
  vx_result_t r = vx_enqueue_launch(q, &info, (uint32_t)waits.size(),
                                    waits.empty() ? nullptr : waits.data(),
                                    &completion);
  if (r != VX_SUCCESS) { profile_abandon(&sample); return map_result(r); }

  set_stream_last_event(stream, device, completion);

  if (profiling) {
    uint32_t g[3] = {info.grid_dim[0], info.grid_dim[1], info.grid_dim[2]};
    uint32_t b[3] = {info.block_dim[0], info.block_dim[1], info.block_dim[2]};
    profile_end_kernel(&sample, k.name, k.module_path, g, b, shared, stream);
  }
  return grxSuccess;
}

}  // namespace

// Is a cooperative launch legal on this device?
//
// Two conditions, and the second is easy to miss. The grid-wide barrier only
// terminates if every block is RESIDENT at once -- that is the familiar one.
// It also only terminates if every core has at least one block, because the
// hardware counts arrivals per core: a core with no active warps never
// forwards an arrival, and the cluster waits for it forever. A grid smaller
// than the machine is the shape that hangs, which is exactly the shape a
// first test tends to have.
grxError_t check_cooperative(const grxDeviceProp_t& prop, const dim3_t& grid,
                             const dim3_t& block, size_t smem_per_block) {
  const int per_sm = resident_blocks_per_sm(prop, (int)volume(block),
                                            smem_per_block);
  const uint64_t blocks = (uint64_t)(grid.x ? grid.x : 1) *
                          (grid.y ? grid.y : 1) * (grid.z ? grid.z : 1);
  if (per_sm <= 0) return grxErrorLaunchOutOfResources;
  if (blocks > (uint64_t)per_sm * (uint64_t)prop.multiProcessorCount)
    return grxErrorLaunchOutOfResources;
  if (blocks < (uint64_t)prop.multiProcessorCount)
    return grxErrorLaunchOutOfResources;
  return grxSuccess;
}

int resident_blocks_per_sm(const grxDeviceProp_t& prop, int block_size,
                           size_t smem_per_block) {
  if (block_size <= 0 || prop.warpSize <= 0) return 0;

  // Bound 1: warps. A CTA's warps must all fit in the core's warp slots.
  const int warps_per_cta =
      (block_size + prop.warpSize - 1) / prop.warpSize;
  if (warps_per_cta <= 0) return 0;
  const int warp_bound = prop.maxWarpsPerMultiProcessor / warps_per_cta;

  // Bound 2: CTA slots. VX_cta_dispatch has NUM_CTA_SLOTS == NUM_WARPS.
  const int slot_bound = prop.maxWarpsPerMultiProcessor;

  // Bound 3: shared memory, via the fixed-stride local-memory allocator:
  //   usable_slots = floor(LOCAL_MEM_SIZE / align_up(smem_per_CTA, BLOCK))
  // A zero-shared-memory kernel has zero stride, which the allocator treats as
  // "all slots usable" rather than as a division by zero
  // (cta_clustering_and_dispatch.md 3.1).
  int smem_bound = slot_bound;
  if (smem_per_block > 0) {
    // VX_CFG_MEM_BLOCK_SIZE is the stride granule; VX_config.toml derives the
    // L1 line size from the same expression, so the cache line the driver
    // reports is that granule.
    const size_t granule = (prop.cacheLineSize > 0)
                               ? (size_t)prop.cacheLineSize : 64u;
    const size_t stride = ((smem_per_block + granule - 1) / granule) * granule;
    smem_bound = (stride == 0) ? slot_bound
                               : (int)(prop.sharedMemPerMultiprocessor / stride);
  }

  // There is deliberately no register bound. Unlike CUDA, the CTA dispatcher
  // does not gate admission on register count, so adding one to look familiar
  // would report an occupancy the hardware does not enforce.
  int n = warp_bound;
  if (slot_bound < n) n = slot_bound;
  if (smem_bound < n) n = smem_bound;
  return (n < 0) ? 0 : n;
}

}  // namespace grxcp

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

extern "C" {

grxError_t __grxPushCallConfiguration(dim3_t gridDim, dim3_t blockDim,
                                      size_t sharedMem, grxStream_t stream) {
  grxcp::g_call_configs.push_back({gridDim, blockDim, sharedMem, stream});
  return grxSuccess;
}

grxError_t __grxPopCallConfiguration(dim3_t* gridDim, dim3_t* blockDim,
                                     size_t* sharedMem, grxStream_t* stream) {
  if (grxcp::g_call_configs.empty()) return grxErrorInvalidValue;
  const grxcp::CallConfig c = grxcp::g_call_configs.back();
  grxcp::g_call_configs.pop_back();
  if (gridDim)   *gridDim   = c.grid;
  if (blockDim)  *blockDim  = c.block;
  if (sharedMem) *sharedMem = c.shared;
  if (stream)    *stream    = c.stream;
  return grxSuccess;
}

grxError_t grxLaunchKernel(const void* func, dim3_t gridDim, dim3_t blockDim,
                           void** args, size_t sharedMem, grxStream_t stream) {
  if (!func) return grxcp::set_error(grxErrorInvalidDeviceFunction);
  const int device = grxcp::current_device_index();

  grxcp::KernelBinding k{};
  if (!grxcp::lookup_registration(func, device, &k))
    return grxcp::set_error(grxErrorInvalidDeviceFunction);

  std::vector<uint8_t> blob;
  grxError_t e = grxcp::pack_arguments(k, args, &blob);
  if (e != grxSuccess) return grxcp::set_error(e);

  e = grxcp::launch_common(k, gridDim, blockDim, dim3_t{1, 1, 1}, sharedMem,
                           blob.empty() ? nullptr : blob.data(), blob.size(),
                           device, stream);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxLaunchKernelEx(const grxLaunchConfig_t* config, const void* func,
                             void** args) {
  if (!config || !func) return grxcp::set_error(grxErrorInvalidValue);
  const int device = grxcp::current_device_index();

  grxcp::KernelBinding k{};
  if (!grxcp::lookup_registration(func, device, &k))
    return grxcp::set_error(grxErrorInvalidDeviceFunction);

  dim3_t cluster{1, 1, 1};
  bool cooperative = false;
  for (unsigned i = 0; i < config->numAttributes; ++i) {
    const grxLaunchAttribute& a = config->attributes[i];
    switch (a.id) {
      case GRX_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION: cluster = a.val.clusterDim; break;
      case GRX_LAUNCH_ATTRIBUTE_COOPERATIVE:       cooperative = a.val.cooperative != 0; break;
      case GRX_LAUNCH_ATTRIBUTE_PRIORITY:          break;  // set at stream creation
      case GRX_LAUNCH_ATTRIBUTE_SHARED_MEM_CARVEOUT:
        // The unified L1/shared carve-out register does not exist in the
        // hardware yet, so honouring this would be theatre.
        return grxcp::set_error(grxErrorNotSupported);
    }
  }

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  if (cooperative && !(d->prop.capabilities & GRX_CAP_COOPERATIVE_LAUNCH))
    return grxcp::set_error(grxErrorNotSupported);

  if (cooperative) {
    grxError_t ce = grxcp::check_cooperative(d->prop, config->gridDim,
                                             config->blockDim,
                                             config->dynamicSharedMem +
                                                 k.static_smem);
    if (ce != grxSuccess) return grxcp::set_error(ce);
  }

  std::vector<uint8_t> blob;
  e = grxcp::pack_arguments(k, args, &blob);
  if (e != grxSuccess) return grxcp::set_error(e);

  e = grxcp::launch_common(k, config->gridDim, config->blockDim, cluster,
                           config->dynamicSharedMem,
                           blob.empty() ? nullptr : blob.data(), blob.size(),
                           device, config->stream);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxLaunchCooperativeKernel(const void* func, dim3_t gridDim,
                                      dim3_t blockDim, void** args,
                                      size_t sharedMem, grxStream_t stream) {
  grxLaunchAttribute attr{};
  attr.id = GRX_LAUNCH_ATTRIBUTE_COOPERATIVE;
  attr.val.cooperative = 1;

  grxLaunchConfig_t config{};
  config.gridDim          = gridDim;
  config.blockDim         = blockDim;
  config.dynamicSharedMem = sharedMem;
  config.stream           = stream;
  config.numAttributes    = 1;
  config.attributes       = &attr;
  return grxLaunchKernelEx(&config, func, args);
}

grxError_t grxLaunchFunction(grxFunction_t func, dim3_t gridDim,
                             dim3_t blockDim, const void* argsBlob,
                             size_t argsSize, size_t sharedMem,
                             grxStream_t stream) {
  grxcp::KernelBinding k{};
  if (!grxcp::lookup_function(func, &k))
    return grxcp::set_error(grxErrorInvalidResourceHandle);

  grxError_t e = grxcp::launch_common(k, gridDim, blockDim, dim3_t{1, 1, 1},
                                      sharedMem, argsBlob, argsSize,
                                      k.device, stream);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxLaunchCooperativeFunction(grxFunction_t func, dim3_t gridDim,
                                        dim3_t blockDim, const void* argsBlob,
                                        size_t argsSize, size_t sharedMem,
                                        grxStream_t stream) {
  grxcp::KernelBinding k{};
  if (!grxcp::lookup_function(func, &k))
    return grxcp::set_error(grxErrorInvalidResourceHandle);

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(k.device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  if (!(d->prop.capabilities & GRX_CAP_COOPERATIVE_LAUNCH))
    return grxcp::set_error(grxErrorNotSupported);

  e = grxcp::check_cooperative(d->prop, gridDim, blockDim,
                               sharedMem + k.static_smem);
  if (e != grxSuccess) return grxcp::set_error(e);

  e = grxcp::launch_common(k, gridDim, blockDim, dim3_t{1, 1, 1},
                           sharedMem, argsBlob, argsSize, k.device, stream);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxFuncGetAttributes(grxFuncAttributes* attr, const void* func) {
  if (!attr || !func) return grxcp::set_error(grxErrorInvalidValue);
  const int device = grxcp::current_device_index();

  grxcp::KernelBinding k{};
  if (!grxcp::lookup_registration(func, device, &k) &&
      !grxcp::lookup_function((grxFunction_t)func, &k))
    return grxcp::set_error(grxErrorInvalidDeviceFunction);

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  *attr = grxFuncAttributes{};
  attr->sharedSizeBytes    = k.static_smem;
  attr->constSizeBytes     = 0;
  attr->localSizeBytes     = 0;
  attr->maxThreadsPerBlock = (k.max_threads_per_block > 0)
                                 ? (int)k.max_threads_per_block
                                 : d->prop.maxThreadsPerBlock;
  // Reported as unknown until grxcc emits register metadata into the .vxbin
  // footer. A fabricated number here would be worse than -1: code tunes on it.
  attr->numRegs       = k.num_regs;
  attr->ptxVersion    = -1;   // GRXCP has no PTX analogue by design
  attr->binaryVersion = 1;

  uint64_t pc = 0;
  if (k.kernel && vx_kernel_address(k.kernel, &pc) == VX_SUCCESS)
    attr->deviceEntryPC = pc;
  return grxSuccess;
}

grxError_t grxOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                        const void* func,
                                                        int blockSize,
                                                        size_t dynamicSMemSize) {
  if (!numBlocks || blockSize <= 0)
    return grxcp::set_error(grxErrorInvalidValue);

  const int device = grxcp::current_device_index();
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  size_t smem = dynamicSMemSize;
  if (func) {
    grxcp::KernelBinding k{};
    if (grxcp::lookup_registration(func, device, &k) ||
        grxcp::lookup_function((grxFunction_t)func, &k))
      smem += k.static_smem;
  }

  *numBlocks = grxcp::resident_blocks_per_sm(d->prop, blockSize, smem);
  return grxSuccess;
}

grxError_t grxOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                             const void* func,
                                             size_t dynamicSMemSize,
                                             int blockSizeLimit) {
  if (!minGridSize || !blockSize) return grxcp::set_error(grxErrorInvalidValue);

  const int device = grxcp::current_device_index();
  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  size_t static_smem = 0;
  if (func) {
    grxcp::KernelBinding k{};
    if (grxcp::lookup_registration(func, device, &k) ||
        grxcp::lookup_function((grxFunction_t)func, &k))
      static_smem = k.static_smem;
  }

  const int limit = (blockSizeLimit > 0 &&
                     blockSizeLimit < d->prop.maxThreadsPerBlock)
                        ? blockSizeLimit : d->prop.maxThreadsPerBlock;

  // Search whole warps only: a partial warp costs a full warp slot, so a block
  // size between multiples can never beat the multiple below it.
  int best_block = d->prop.warpSize;
  int best_threads = 0;
  for (int bs = d->prop.warpSize; bs <= limit; bs += d->prop.warpSize) {
    const int blocks = grxcp::resident_blocks_per_sm(
        d->prop, bs, dynamicSMemSize + static_smem);
    const int threads = blocks * bs;
    if (threads > best_threads) { best_threads = threads; best_block = bs; }
  }

  *blockSize   = best_block;
  *minGridSize = grxcp::resident_blocks_per_sm(
                     d->prop, best_block, dynamicSMemSize + static_smem) *
                 d->prop.multiProcessorCount;
  return grxSuccess;
}

}  // extern "C"
