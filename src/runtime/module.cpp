// GRXCP — fat binaries, modules, kernels, and the host-stub registry.
//
// Two paths converge here.
//
//   Module path   grxModuleLoad* / grxModuleGetFunction: explicit, driver
//                 shaped, for language runtimes and translators.
//   Runtime path  a host stub address, registered at static-init time by
//                 constructors grxcc emits, resolved to a device kernel on
//                 first launch. This is what <<<>>> compiles down to.
//
// Both end at the same place: a vx_kernel_h resolved from a .vxbin entry name
// through the VXSYMTAB footer.
//
// Image selection is the part worth reading. A .grxfat can carry several
// device images, each declaring the extension bits it requires. The runtime
// picks the most capable image the device can actually run, matched against
// VX_CAPS_ISA_FLAGS -- so a binary built with a tensor-core kernel and a
// portable fallback runs on both, and a binary built only for tensor cores
// fails loudly on a device without them instead of trapping mid-kernel.

#include "internal.h"

#include <grx/grx_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace grxcp {

namespace {

struct ModuleState {
  vx_module_h module = nullptr;
  int         device = 0;
  // Provenance, kept for grx-sanitize: the path this image came from, the
  // sibling ELF that carries its symbols and line tables, and whether the
  // sanitizer's control-block anchor was found and patched. A module loaded
  // from memory has no path and therefore no ELF, which is a reportable fact
  // rather than a failure.
  std::string path;
  std::string elf;
  bool        sanitized = false;
};

struct FunctionState {
  vx_kernel_h  kernel = nullptr;
  ModuleState* owner  = nullptr;
  std::string  name;
  uint32_t     static_smem = 0;
  int32_t      num_regs    = -1;
  uint32_t     max_threads_per_block = 0;
};

// A fat binary registered by a host program's static initializers. Modules are
// loaded per device on first use: a four-device process loads each image once
// per device, not once per launch.
struct FatBinary {
  const void*                image = nullptr;
  std::map<int, vx_module_h> modules;
};

struct Registration {
  std::string                   device_name;
  FatBinary*                    fatbin = nullptr;
  std::vector<grx_kernel_param> params;
  uint32_t                      args_size   = 0;
  uint32_t                      static_smem = 0;
  int32_t                       num_regs    = -1;
  uint32_t                      max_threads_per_block = 0;
  bool                          has_layout  = false;
  std::map<int, vx_kernel_h>    resolved;   // device -> kernel
};

std::mutex g_module_mutex;
std::map<grxModule_t, ModuleState*>     g_modules;
std::map<grxFunction_t, FunctionState*> g_functions;
std::map<const void*, Registration>     g_registry;   // host stub -> kernel
std::vector<FatBinary*>                 g_fatbins;

// Select the device image to load. Preference order: images the device can run,
// most-demanding first, so a tensor-core build wins over a portable one on
// hardware that has tensor cores.
grxError_t select_image(const void* blob, size_t blob_size, const Device& d,
                        const void** out_image, size_t* out_size) {
  if (blob_size < sizeof(grx_fatbin_header)) {
    // Not a container; treat the bytes as a bare device binary.
    *out_image = blob;
    *out_size  = blob_size;
    return grxSuccess;
  }

  grx_fatbin_header header{};
  std::memcpy(&header, blob, sizeof(header));
  if (header.magic != GRX_FATBIN_MAGIC) {
    *out_image = blob;
    *out_size  = blob_size;
    return grxSuccess;
  }
  if (header.version != GRX_FATBIN_VERSION) return grxErrorInvalidKernelImage;
  if (header.num_entries == 0)              return grxErrorInvalidKernelImage;
  if (header.total_size > blob_size)        return grxErrorInvalidKernelImage;

  uint64_t isa = 0;
  if (vx_device_query(d.handle, VX_CAPS_ISA_FLAGS, &isa) != VX_SUCCESS)
    return grxErrorInvalidDevice;
  const uint32_t device_ext = (uint32_t)(isa >> 32);

  const auto* entries = reinterpret_cast<const grx_fatbin_entry*>(
      static_cast<const uint8_t*>(blob) + sizeof(grx_fatbin_header));

  const grx_fatbin_entry* best = nullptr;
  int best_score = -1;
  for (uint16_t i = 0; i < header.num_entries; ++i) {
    const grx_fatbin_entry& e = entries[i];
    if (e.offset + e.size > header.total_size) return grxErrorInvalidKernelImage;
    // The device must provide every extension the image requires.
    if ((e.required_isa & device_ext) != e.required_isa) continue;
    // The runtime only loads ready-to-run device binaries. SPIR-V and bitcode
    // entries need a JIT that does not exist here yet; skipping them beats
    // handing the loader something it cannot parse.
    if (e.kind != GRX_IMAGE_VXBIN) continue;

    const int score = __builtin_popcount(e.required_isa);
    if (score > best_score) { best = &e; best_score = score; }
  }
  if (!best) return grxErrorInvalidKernelImage;

  *out_image = static_cast<const uint8_t*>(blob) + best->offset;
  *out_size  = (size_t)best->size;
  return grxSuccess;
}

grxError_t load_module(Device& d, const void* blob, size_t size,
                       vx_module_h* out) {
  const void* image = nullptr;
  size_t image_size = 0;
  grxError_t e = select_image(blob, size, d, &image, &image_size);
  if (e != grxSuccess) return e;
  return map_result(vx_module_load_bytes(d.handle, image, image_size, out));
}

// Resolve a registered host stub to a device kernel, loading the module for
// this device the first time it is needed.
grxError_t resolve_registration(Registration& reg, int device,
                                vx_kernel_h* out) {
  auto cached = reg.resolved.find(device);
  if (cached != reg.resolved.end()) { *out = cached->second; return grxSuccess; }

  Device* d = nullptr;
  grxError_t e = acquire_device(device, &d);
  if (e != grxSuccess) return e;
  if (!reg.fatbin || !reg.fatbin->image) return grxErrorInvalidDeviceFunction;

  vx_module_h mod = nullptr;
  auto it = reg.fatbin->modules.find(device);
  if (it != reg.fatbin->modules.end()) {
    mod = it->second;
  } else {
    // The registered image is a bare pointer with no length: the fat binary's
    // own header carries its size, which is why the container is required for
    // the registration path even when it holds a single entry.
    grx_fatbin_header header{};
    std::memcpy(&header, reg.fatbin->image, sizeof(header));
    const size_t size = (header.magic == GRX_FATBIN_MAGIC)
                            ? (size_t)header.total_size : 0;
    if (size == 0) return grxErrorInvalidKernelImage;

    e = load_module(*d, reg.fatbin->image, size, &mod);
    if (e != grxSuccess) return e;
    reg.fatbin->modules[device] = mod;
  }

  vx_kernel_h kernel = nullptr;
  vx_result_t r = vx_module_get_kernel(mod, reg.device_name.c_str(), &kernel);
  if (r != VX_SUCCESS) return map_result(r);

  reg.resolved[device] = kernel;
  *out = kernel;
  return grxSuccess;
}

}  // namespace

bool lookup_registration(const void* stub, int device, KernelBinding* out) {
  std::lock_guard<std::mutex> lock(g_module_mutex);
  auto it = g_registry.find(stub);
  if (it == g_registry.end()) return false;

  vx_kernel_h kernel = nullptr;
  if (resolve_registration(it->second, device, &kernel) != grxSuccess)
    return false;

  if (out) {
    out->kernel      = kernel;
    out->params      = it->second.params.data();
    out->num_params  = (uint32_t)it->second.params.size();
    out->args_size   = it->second.args_size;
    out->static_smem = it->second.static_smem;
    out->num_regs    = it->second.num_regs;
    out->max_threads_per_block = it->second.max_threads_per_block;
    out->has_layout  = it->second.has_layout;
  }
  return true;
}

// Load an image and register it, remembering where it came from.
//
// `path` is null for grxModuleLoadData, which is the whole reason this is one
// function rather than two: the sanitizer needs the sibling ELF, the ELF is
// found from the path, and a module loaded from memory simply does not have
// one. Keeping both paths here makes that difference explicit instead of
// leaving grxModuleLoadData to silently lose provenance.
grxError_t load_module_tracked(grxModule_t* module, const void* image,
                               size_t size, const char* path) {
  if (!module || !image || size == 0) return set_error(grxErrorInvalidValue);

  Device* d = nullptr;
  grxError_t e = acquire_device(current_device_index(), &d);
  if (e != grxSuccess) return set_error(e);

  std::string elf_path;
  bool sanitized = false;

  // Patching rewrites 8 bytes of the payload, so it needs a mutable copy. Only
  // taken when the sanitizer is on -- the ordinary path still loads the
  // caller's bytes in place.
  std::vector<uint8_t> patched;
  if (path && sanitize_enabled()) {
    elf_path = path;
    const size_t dot = elf_path.rfind(".vxbin");
    if (dot != std::string::npos && dot + 6 == elf_path.size())
      elf_path.replace(dot, 6, ".elf");
    else
      elf_path += ".elf";

    patched.assign((const uint8_t*)image, (const uint8_t*)image + size);
    sanitized = sanitize_patch_image(patched, elf_path.c_str(), d->index);
    if (sanitized) image = patched.data();
  }

  vx_module_h mod = nullptr;
  e = load_module(*d, image, size, &mod);
  if (e != grxSuccess) return set_error(e);

  auto* s     = new ModuleState();
  s->module   = mod;
  s->device   = d->index;
  s->path     = path ? path : "";
  s->elf      = sanitized ? elf_path : "";
  s->sanitized = sanitized;

  auto handle = reinterpret_cast<grxModule_t>(s);
  {
    std::lock_guard<std::mutex> lock(g_module_mutex);
    g_modules[handle] = s;
  }
  *module = handle;
  return grxSuccess;
}

bool lookup_function(grxFunction_t func, KernelBinding* out) {
  std::lock_guard<std::mutex> lock(g_module_mutex);
  auto it = g_functions.find(func);
  if (it == g_functions.end()) return false;
  if (out) {
    out->kernel      = it->second->kernel;
    out->params      = nullptr;
    out->num_params  = 0;
    out->args_size   = 0;
    out->static_smem = it->second->static_smem;
    out->num_regs    = it->second->num_regs;
    out->max_threads_per_block = it->second->max_threads_per_block;
    out->has_layout  = false;
    out->device      = it->second->owner ? it->second->owner->device : 0;
    out->name        = it->second->name.c_str();
    if (const ModuleState* m = it->second->owner) {
      out->module_path = m->path.c_str();
      out->module_elf  = m->elf.c_str();
      out->sanitized   = m->sanitized;
    }
  }
  return true;
}

}  // namespace grxcp

// ---------------------------------------------------------------------------
// Public entry points — module path
// ---------------------------------------------------------------------------

extern "C" {

grxError_t grxModuleLoadData(grxModule_t* module, const void* image,
                             size_t size) {
  return grxcp::load_module_tracked(module, image, size, nullptr);
}

grxError_t grxModuleLoad(grxModule_t* module, const char* path) {
  if (!module || !path) return grxcp::set_error(grxErrorInvalidValue);

  // Read the file here rather than calling vx_module_load_file, so that the
  // same fat-binary selection applies whether an image arrives from disk or
  // from memory. A bare .vxbin still works: select_image passes it through.
  // A missing file is not a malformed image. Reporting it as one sends the
  // caller off to debug their toolchain when the answer is a wrong path.
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return grxcp::set_error(grxErrorFileNotFound);
  std::fseek(f, 0, SEEK_END);
  const long len = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (len <= 0) { std::fclose(f); return grxcp::set_error(grxErrorInvalidKernelImage); }

  std::vector<uint8_t> bytes((size_t)len);
  const size_t got = std::fread(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  if (got != bytes.size()) return grxcp::set_error(grxErrorInvalidKernelImage);

  return grxcp::load_module_tracked(module, bytes.data(), bytes.size(), path);
}

grxError_t grxModuleUnload(grxModule_t module) {
  grxcp::ModuleState* s = nullptr;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);
    auto it = grxcp::g_modules.find(module);
    if (it == grxcp::g_modules.end())
      return grxcp::set_error(grxErrorInvalidResourceHandle);
    s = it->second;
    grxcp::g_modules.erase(it);

    // Functions resolved from this module become invalid with it.
    for (auto fit = grxcp::g_functions.begin(); fit != grxcp::g_functions.end();) {
      if (fit->second->owner == s) {
        vx_kernel_release(fit->second->kernel);
        delete fit->second;
        fit = grxcp::g_functions.erase(fit);
      } else {
        ++fit;
      }
    }
  }
  vx_module_release(s->module);
  delete s;
  return grxSuccess;
}

grxError_t grxModuleGetFunction(grxFunction_t* func, grxModule_t module,
                                const char* name) {
  if (!func || !name) return grxcp::set_error(grxErrorInvalidValue);

  grxcp::ModuleState* s = nullptr;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);
    auto it = grxcp::g_modules.find(module);
    if (it == grxcp::g_modules.end())
      return grxcp::set_error(grxErrorInvalidResourceHandle);
    s = it->second;
  }

  vx_kernel_h kernel = nullptr;
  vx_result_t r = vx_module_get_kernel(s->module, name, &kernel);
  if (r != VX_SUCCESS) return grxcp::set_error(grxErrorInvalidDeviceFunction);

  auto* fs  = new grxcp::FunctionState();
  fs->kernel = kernel;
  fs->owner  = s;
  fs->name   = name;

  // Per-kernel metadata comes from the .vxbin footer when the toolchain
  // records it. Until grxcc does, the device's natural block dims are the only
  // available answer and the register count stays unknown.
  uint32_t x = 0, y = 0, z = 0;
  if (vx_kernel_get_max_block_size(kernel, &x, &y, &z) == VX_SUCCESS)
    fs->max_threads_per_block = x * (y ? y : 1) * (z ? z : 1);

  auto handle = reinterpret_cast<grxFunction_t>(fs);
  {
    std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);
    grxcp::g_functions[handle] = fs;
  }
  *func = handle;
  return grxSuccess;
}

// ---------------------------------------------------------------------------
// Public entry points — registration (emitted by grxcc, not called by users)
// ---------------------------------------------------------------------------

void** __grxRegisterFatBinary(void* fatCubin) {
  auto* fb  = new grxcp::FatBinary();
  fb->image = fatCubin;
  std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);
  grxcp::g_fatbins.push_back(fb);
  return reinterpret_cast<void**>(fb);
}

void __grxUnregisterFatBinary(void** handle) {
  if (!handle) return;
  auto* fb = reinterpret_cast<grxcp::FatBinary*>(handle);
  std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);

  for (auto it = grxcp::g_registry.begin(); it != grxcp::g_registry.end();) {
    if (it->second.fatbin == fb) {
      for (auto& kv : it->second.resolved) vx_kernel_release(kv.second);
      it = grxcp::g_registry.erase(it);
    } else {
      ++it;
    }
  }
  for (auto& kv : fb->modules) vx_module_release(kv.second);

  auto pos = std::find(grxcp::g_fatbins.begin(), grxcp::g_fatbins.end(), fb);
  if (pos != grxcp::g_fatbins.end()) grxcp::g_fatbins.erase(pos);
  delete fb;
}

void __grxRegisterFunction(void** handle, const char* hostStub,
                           const char* deviceName, int minBlocks,
                           int maxThreads) {
  (void)minBlocks;
  if (!handle || !hostStub || !deviceName) return;
  std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);
  auto& reg = grxcp::g_registry[(const void*)hostStub];
  reg.device_name = deviceName;
  reg.fatbin      = reinterpret_cast<grxcp::FatBinary*>(handle);
  if (maxThreads > 0) reg.max_threads_per_block = (uint32_t)maxThreads;
}

void __grxRegisterKernelDesc(void** handle, const char* hostStub,
                             const grx_kernel_desc* desc) {
  if (!handle || !hostStub || !desc) return;
  std::lock_guard<std::mutex> lock(grxcp::g_module_mutex);
  auto& reg = grxcp::g_registry[(const void*)hostStub];
  reg.fatbin = reinterpret_cast<grxcp::FatBinary*>(handle);
  if (desc->device_name) reg.device_name = desc->device_name;
  reg.params.assign(desc->params, desc->params + desc->num_params);
  reg.args_size   = desc->args_size;
  reg.static_smem = desc->static_smem;
  reg.num_regs    = desc->num_regs;
  if (desc->max_threads_per_block > 0)
    reg.max_threads_per_block = desc->max_threads_per_block;
  reg.has_layout  = true;
}

void __grxRegisterVar(void** handle, const char* hostVar,
                      const char* deviceName, size_t size, int isConstant) {
  (void)handle; (void)hostVar; (void)deviceName; (void)size; (void)isConstant;
  // __device__ and __constant__ variable registration needs a device symbol
  // table the .vxbin footer does not carry yet. Recording the request without
  // being able to honour grxMemcpyToSymbol would be worse than not accepting
  // it: this is a no-op until the symbol path exists.
}

}  // extern "C"
