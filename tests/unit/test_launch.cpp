// Phase 1 gate: fat binaries, modules, the kernel registry, launch, occupancy.
//
// The mock cannot run a kernel, so these assert the thing that IS checkable
// without a simulator and that silently corrupts results when wrong: the
// launch descriptor. Grid and block dimensions, the shared-memory size, the
// cluster dimensions, and the packed argument blob all have to arrive at the
// driver exactly as the caller meant them.

#include <grx/grx.h>

#include "grx_test.h"
#include "../mock/vortex_mock.h"

#include <cstdint>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

// Extension bit positions within grx_fatbin_entry::required_isa, which is the
// VX_ISA_EXT_* encoding shifted down by 32.
constexpr uint32_t kExtTCU = 1u << 9;
constexpr uint32_t kExtRTU = 1u << 11;

// Build a .grxfat holding one mock module image per entry.
std::vector<uint8_t> build_fatbin(
    const std::vector<std::pair<uint32_t, std::vector<const char*>>>& entries) {
  std::vector<std::vector<uint8_t>> payloads;
  for (const auto& e : entries) {
    std::vector<uint8_t> buf(1024);
    const size_t n = grxmock_build_module(buf.data(), buf.size(),
                                          e.second.data(),
                                          (uint32_t)e.second.size());
    buf.resize(n);
    payloads.push_back(std::move(buf));
  }

  const size_t header_bytes = sizeof(grx_fatbin_header);
  const size_t table_bytes  = sizeof(grx_fatbin_entry) * entries.size();
  size_t total = header_bytes + table_bytes;
  for (const auto& p : payloads) total += p.size();

  std::vector<uint8_t> out(total, 0);
  grx_fatbin_header header{};
  header.magic       = GRX_FATBIN_MAGIC;
  header.version     = GRX_FATBIN_VERSION;
  header.num_entries = (uint16_t)entries.size();
  header.total_size  = total;
  std::memcpy(out.data(), &header, sizeof(header));

  size_t payload_at = header_bytes + table_bytes;
  for (size_t i = 0; i < entries.size(); ++i) {
    grx_fatbin_entry e{};
    e.kind         = GRX_IMAGE_VXBIN;
    e.xlen         = 64;
    e.required_isa = entries[i].first;
    e.offset       = payload_at;
    e.size         = payloads[i].size();
    std::memcpy(out.data() + header_bytes + i * sizeof(grx_fatbin_entry), &e,
                sizeof(e));
    std::memcpy(out.data() + payload_at, payloads[i].data(), payloads[i].size());
    payload_at += payloads[i].size();
  }
  return out;
}

// Stand-ins for the host stubs grxcc emits. Only their addresses matter.
char g_stub_vecadd;
char g_stub_noparams;

}  // namespace

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  grxDeviceProp_t prop{};
  GRX_REQUIRE(grxGetDeviceProperties(&prop, 0), "device properties");

  // -------------------------------------------------------------------------
  section("fat binary image selection");
  // -------------------------------------------------------------------------
  // Two images: a portable one and one that needs tensor cores. The device has
  // tensor cores, so the more capable image must win -- that is the whole point
  // of shipping both.
  auto fat = build_fatbin({{0u,        {"portable_kernel"}},
                           {kExtTCU,   {"tensor_kernel"}}});

  grxModule_t mod = nullptr;
  GRX_REQUIRE(grxModuleLoadData(&mod, fat.data(), fat.size()),
              "load a multi-image fat binary");

  grxFunction_t fn = nullptr;
  check(grxModuleGetFunction(&fn, mod, "tensor_kernel") == grxSuccess,
        "the tensor-core image was selected on a tensor-core device");
  check(grxModuleGetFunction(&fn, mod, "portable_kernel") ==
            grxErrorInvalidDeviceFunction,
        "the portable image was not the one loaded");
  check(grxModuleGetFunction(&fn, mod, "no_such_kernel") ==
            grxErrorInvalidDeviceFunction,
        "an unknown entry name is rejected");

  {
    // An image demanding an extension this device lacks must fail at load,
    // loudly, rather than trap somewhere inside the kernel later.
    auto rt_only = build_fatbin({{kExtRTU, {"raytrace_kernel"}}});
    grxModule_t bad = nullptr;
    check(grxModuleLoadData(&bad, rt_only.data(), rt_only.size()) ==
              grxErrorInvalidKernelImage,
          "an image needing an absent extension is refused");
  }

  // -------------------------------------------------------------------------
  section("module-path launch");
  // -------------------------------------------------------------------------
  {
    GRX_REQUIRE(grxModuleGetFunction(&fn, mod, "tensor_kernel"),
                "resolve the kernel");
    grxmock_reset_launches();

    // Driver-style launch: the caller packed the blob itself, so no parameter
    // layout is needed.
    struct { uint64_t a; uint64_t b; uint32_t n; uint32_t pad; } blob{
        0xAAAA'0000'0000'1000ull, 0xBBBB'0000'0000'2000ull, 4096u, 0u};

    GRX_REQUIRE(grxLaunchFunction(fn, dim3_t{16, 1, 1}, dim3_t{4, 1, 1},
                                  &blob, sizeof(blob), 0, nullptr),
                "grxLaunchFunction");

    const grxmock_launch_record* r = grxmock_last_launch();
    check(r->valid && grxmock_launch_count() == 1, "the driver saw one launch");
    check(r->grid_dim[0] == 16 && r->grid_dim[1] == 1 && r->grid_dim[2] == 1,
          "grid dimensions reach the driver");
    check(r->block_dim[0] == 4 && r->block_dim[1] == 1 && r->block_dim[2] == 1,
          "block dimensions reach the driver");
    check(r->ndim == 1, "a 1D launch is described as rank 1");
    check(r->args_size == sizeof(blob) &&
              std::memcmp(r->args, &blob, sizeof(blob)) == 0,
          "the argument blob arrives byte for byte");
    check(r->cluster_dim[0] == 1 && r->cluster_dim[1] == 1 &&
              r->cluster_dim[2] == 1,
          "cluster dimensions default to 1");

    GRX_REQUIRE(grxLaunchFunction(fn, dim3_t{2, 3, 4}, dim3_t{2, 2, 1},
                                  &blob, sizeof(blob), 128, nullptr),
                "3D launch with dynamic shared memory");
    r = grxmock_last_launch();
    check(r->ndim == 3, "a launch with depth is described as rank 3");
    check(r->grid_dim[1] == 3 && r->grid_dim[2] == 4, "y and z reach the driver");
    check(r->lmem_size == 128, "dynamic shared memory reaches the driver");
  }

  // -------------------------------------------------------------------------
  section("registration-path launch and argument packing");
  // -------------------------------------------------------------------------
  {
    void** handle = __grxRegisterFatBinary(fat.data());
    check(handle != nullptr, "fat binary registration returns a handle");

    // What grxcc will emit for  __global__ void k(float* a, float* b, int n)
    static const grx_kernel_param params[] = {
        {0,  8, 1, {0, 0, 0}},   // float* a
        {8,  8, 1, {0, 0, 0}},   // float* b
        {16, 4, 0, {0, 0, 0}},   // int n
    };
    grx_kernel_desc desc{};
    desc.device_name = "tensor_kernel";
    desc.params      = params;
    desc.num_params  = 3;
    desc.args_size   = 24;
    desc.static_smem = 64;
    desc.num_regs    = -1;
    __grxRegisterFunction(handle, &g_stub_vecadd, "tensor_kernel", 0, 0);
    __grxRegisterKernelDesc(handle, &g_stub_vecadd, &desc);

    void*    dev_a = (void*)0x1000'0000'0000'0100ull;
    void*    dev_b = (void*)0x1000'0000'0000'0200ull;
    int      n     = 777;
    void*    args[] = {&dev_a, &dev_b, &n};

    grxmock_reset_launches();
    GRX_REQUIRE(grxLaunchKernel(&g_stub_vecadd, dim3_t{8, 1, 1},
                                dim3_t{4, 1, 1}, args, 32, nullptr),
                "grxLaunchKernel through the registry");

    const grxmock_launch_record* r = grxmock_last_launch();
    check(r->valid, "the registered stub resolved to a device kernel");
    check(r->args_size == 24, "packed blob is the declared size");

    uint64_t a_packed = 0, b_packed = 0;
    uint32_t n_packed = 0;
    std::memcpy(&a_packed, r->args + 0,  8);
    std::memcpy(&b_packed, r->args + 8,  8);
    std::memcpy(&n_packed, r->args + 16, 4);
    check(a_packed == (uint64_t)(uintptr_t)dev_a, "first pointer lands at offset 0");
    check(b_packed == (uint64_t)(uintptr_t)dev_b, "second pointer lands at offset 8");
    check(n_packed == 777u, "the scalar lands at offset 16");

    // A kernel's own recorded local-memory need is added to whatever the
    // launch asks for dynamically, because the dispatcher's per-CTA stride has
    // to cover both. (Nothing emits that number yet -- there is no static
    // __shared__ on this platform, see include/grx/device/grx_device.h -- but
    // the launch path has to honour it when a toolchain does.)
    check(r->lmem_size == 32 + 64,
          "shared memory is dynamic plus the kernel's recorded requirement");

    // A stub with no registered layout cannot have its void** packed. Refusing
    // beats assuming every argument is pointer-sized.
    __grxRegisterFunction(handle, &g_stub_noparams, "tensor_kernel", 0, 0);
    check(grxLaunchKernel(&g_stub_noparams, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                          args, 0, nullptr) == grxErrorInvalidDeviceFunction,
          "launching a stub with no parameter layout is refused");

    check(grxLaunchKernel((const void*)&n, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                          args, 0, nullptr) == grxErrorInvalidDeviceFunction,
          "an unregistered stub address is rejected");

    // ---------------------------------------------------------------------
    section("launch attributes and validation");
    // ---------------------------------------------------------------------
    grxLaunchAttribute cluster_attr{};
    cluster_attr.id = GRX_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
    cluster_attr.val.clusterDim = dim3_t{2, 1, 1};

    grxLaunchConfig_t cfg{};
    cfg.gridDim          = dim3_t{8, 1, 1};
    cfg.blockDim         = dim3_t{4, 1, 1};
    cfg.dynamicSharedMem = 0;
    cfg.stream           = nullptr;
    cfg.numAttributes    = 1;
    cfg.attributes       = &cluster_attr;

    grxmock_reset_launches();
    GRX_REQUIRE(grxLaunchKernelEx(&cfg, &g_stub_vecadd, args),
                "grxLaunchKernelEx with a cluster dimension");
    check(grxmock_last_launch()->cluster_dim[0] == 2,
          "the cluster dimension reaches the driver");

    grxLaunchAttribute carveout{};
    carveout.id = GRX_LAUNCH_ATTRIBUTE_SHARED_MEM_CARVEOUT;
    carveout.val.sharedMemCarveoutPercent = 50;
    cfg.attributes = &carveout;
    check(grxLaunchKernelEx(&cfg, &g_stub_vecadd, args) == grxErrorNotSupported,
          "the shared-memory carve-out attribute is refused, not ignored");

    const int too_many = prop.maxThreadsPerBlock + prop.warpSize;
    check(grxLaunchKernel(&g_stub_vecadd, dim3_t{1, 1, 1},
                          dim3_t{(unsigned)too_many, 1, 1}, args, 0, nullptr) ==
              grxErrorLaunchOutOfResources,
          "a block larger than the core's warp capacity is rejected");
    check(grxLaunchKernel(&g_stub_vecadd, dim3_t{1, 1, 1}, dim3_t{4, 1, 1},
                          args, prop.sharedMemPerBlock + 1, nullptr) ==
              grxErrorLaunchOutOfResources,
          "a shared-memory request beyond the per-block limit is rejected");

    // A grid too large to be co-resident would deadlock at the grid barrier.
    check(grxLaunchCooperativeKernel(&g_stub_vecadd, dim3_t{1u << 20, 1, 1},
                                     dim3_t{4, 1, 1}, args, 0, nullptr) ==
              grxErrorLaunchOutOfResources,
          "a cooperative launch that cannot be co-resident is rejected");

    // ---------------------------------------------------------------------
    section("kernel attributes");
    // ---------------------------------------------------------------------
    grxFuncAttributes fa{};
    GRX_REQUIRE(grxFuncGetAttributes(&fa, &g_stub_vecadd), "grxFuncGetAttributes");
    check(fa.sharedSizeBytes == 64, "static shared memory is reported");
    check(fa.numRegs == -1,
          "register count reports unknown until the toolchain supplies it");
    check(fa.ptxVersion == -1, "no PTX version is claimed");
    check(fa.deviceEntryPC != 0, "the device entry PC is reported");

    __grxUnregisterFatBinary(handle);
    check(grxLaunchKernel(&g_stub_vecadd, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                          args, 0, nullptr) == grxErrorInvalidDeviceFunction,
          "launching after unregistration is rejected");
  }

  // -------------------------------------------------------------------------
  section("occupancy");
  // -------------------------------------------------------------------------
  {
    // Hand-computed against the documented three bounds:
    //   warps : NUM_WARPS / ceil(block / warpSize)
    //   slots : NUM_WARPS
    //   smem  : LOCAL_MEM_SIZE / align_up(smem, cache line)
    const int W = prop.maxWarpsPerMultiProcessor;
    const int T = prop.warpSize;

    int blocks = 0;
    GRX_REQUIRE(grxOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, nullptr,
                                                             T, 0),
                "occupancy for a one-warp block");
    check(blocks == W, "a one-warp block fills every CTA slot");

    GRX_REQUIRE(grxOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, nullptr,
                                                             T * 2, 0),
                "occupancy for a two-warp block");
    check(blocks == W / 2, "a two-warp block halves residency");

    // Shared memory bound: asking for half the per-core local memory admits
    // exactly two CTAs regardless of how few warps they use.
    const size_t half = prop.sharedMemPerMultiprocessor / 2;
    GRX_REQUIRE(grxOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, nullptr,
                                                             T, half),
                "occupancy under a shared-memory bound");
    check(blocks == 2, "shared memory bounds residency to two CTAs");

    // A tiny shared-memory request rounds up to one stride granule and must not
    // reduce residency at all.
    GRX_REQUIRE(grxOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, nullptr,
                                                             T, 1),
                "occupancy with a one-byte shared request");
    check(blocks == W, "a sub-granule shared request does not cost residency");

    int min_grid = 0, block_size = 0;
    GRX_REQUIRE(grxOccupancyMaxPotentialBlockSize(&min_grid, &block_size,
                                                  nullptr, 0, 0),
                "grxOccupancyMaxPotentialBlockSize");
    check(block_size % T == 0 && block_size > 0,
          "the suggested block size is a whole number of warps");
    check(block_size <= prop.maxThreadsPerBlock,
          "the suggested block size is launchable");
    check(min_grid > 0, "a minimum grid size is suggested");
  }

  grxModuleUnload(mod);
  return grxtest::report();
}
