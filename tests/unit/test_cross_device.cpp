// Two devices, and what the runtime must not silently do with them.
//
// WHY THIS COULD NOT EXIST BEFORE. The mock's vx_device_open handed out
// `&g_devices[0]` for every index, so GRXMOCK_DEVICE_COUNT=2 gave the runtime
// two device SLOTS over one mock device, one bump allocator and one memory
// pool. Every cross-device operation in CI succeeded for the uninteresting
// reason that there was only ever one device. The mock now models N distinct
// devices, each with its own address space starting at the same base -- which
// is what per-device DDR means, and what makes an address ambiguous.
//
// WHAT THAT IMMEDIATELY FOUND, before any of the checks below were written:
//
//   grxMalloc on device 1 returned DEVICE 0's MEMORY. take_best_fit searched
//   the whole free list with no device filter, so device 1's allocation was
//   carved out of device 0's slab and recorded as device 1's. Device 1's
//   grxMemGetInfo still read zero bytes in use while holding a live 4 KiB
//   allocation. Silent, on every multi-device machine, always.
//
//   Fixing that produced the second bug on the next run, as predicted: with
//   device 1 allocating from its own slab, both devices returned the SAME
//   address, and g_live -- keyed by address alone -- kept only one of the two
//   records. The map is keyed by (device, address) now.
//
// THE HONEST LIMIT, checked below as its own case. When an address is live on
// both devices, nothing in a bare void* says which was meant, and "this
// pointer, on this device" is the only defensible reading. What is refused is
// a pointer live on ANOTHER device and on no local allocation: there is no
// reading of that which is correct.

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess) {
    std::printf("no device count; skipping\n");
    return 77;
  }
  if (count < 2) {
    std::printf("needs two devices; run with GRXMOCK_DEVICE_COUNT=2\n");
    return 77;
  }

  // ---- a device-0 pointer, used from device 1 before device 1 owns anything
  //
  // FIRST, because it is the case the (device, address) key is really for.
  // Keys sort by device then address, so a lookup on a device with NO live
  // allocations walks back past every key it has and lands on the LAST entry
  // of a lower-numbered device -- which, at a plausible address, covers it.
  // Every other ordering in this file hides that: once device 1 has any
  // allocation of its own, its own entry is the one the walk stops at.
  size_t free0 = 0, free1 = 0, total = 0;
  grxtest::check(grxSetDevice(0) == grxSuccess, "select device 0");
  grxtest::check(grxMemGetInfo(&free0, &total) == grxSuccess, "device 0 memory info");
  grxtest::check(grxSetDevice(1) == grxSuccess, "select device 1");
  grxtest::check(grxMemGetInfo(&free1, &total) == grxSuccess, "device 1 memory info");

  grxSetDevice(0);
  void* first0 = nullptr;
  grxtest::check(grxMalloc(&first0, 1 << 20) == grxSuccess,
                 "a 1 MiB allocation on device 0, and nothing yet on device 1");
  void* deep0 = (char*)first0 + (1 << 19);

  unsigned char probe[64];
  std::memset(probe, 0x3C, sizeof probe);

  grxSetDevice(1);
  grxtest::check(grxMemset(deep0, 0, 64) == grxErrorInvalidDevicePointer,
                 "grxMemset on device 0's pointer is refused from device 1");
  grxtest::check(grxMemcpy(deep0, probe, 64, grxMemcpyHostToDevice) ==
                     grxErrorInvalidDevicePointer,
                 "grxMemcpy to device 0's pointer is refused from device 1");
  grxtest::check(grxMemcpy(probe, deep0, 64, grxMemcpyDeviceToHost) ==
                     grxErrorInvalidDevicePointer,
                 "and reading from it is refused too, not just writing");
  grxtest::check(grxFree(first0) == grxErrorInvalidDevicePointer,
                 "grxFree of device 0's allocation is refused from device 1");

  // POSITIVE CONTROL, here rather than only at the end: without it a runtime
  // that refused every pointer on every device would pass all four.
  grxSetDevice(0);
  grxtest::check(grxMemset(deep0, 0, 64) == grxSuccess,
                 "and all of it is accepted on device 0, which owns it");

  // ---- allocation lands on the device that was asked for -----------------
  //
  // Device 0 allocated FIRST, above, which is what makes this bite: an
  // unfiltered free-list search finds device 0's slab with room in it and
  // carves device 1's allocation out of that. Ask device 1 first and the list
  // is empty, there is nothing to steal, and the bug hides.
  size_t before1 = 0, before0 = 0;
  grxSetDevice(1);
  grxMemGetInfo(&before1, &total);
  grxSetDevice(0);
  grxMemGetInfo(&before0, &total);

  grxSetDevice(1);
  void* p1 = nullptr;
  grxtest::check(grxMalloc(&p1, 4096) == grxSuccess, "grxMalloc on device 1");

  size_t after1 = 0, after0 = 0;
  grxtest::check(grxMemGetInfo(&after1, &total) == grxSuccess, "device 1 memory info again");
  grxSetDevice(0);
  grxtest::check(grxMemGetInfo(&after0, &total) == grxSuccess, "device 0 memory info again");

  // The load-bearing assertion. Device 1's allocation must come out of device
  // 1. Before take_best_fit took a device this read "device 1 used 0 bytes"
  // while device 1 held a live allocation.
  grxtest::check(after1 < before1, "allocating on device 1 consumed device 1's memory");
  grxtest::check(after0 == before0, "and did not consume device 0's");

  // ---- the same address, live on both devices ----------------------------
  grxSetDevice(0);
  void* p0 = nullptr;
  grxtest::check(grxMalloc(&p0, 4096) == grxSuccess, "grxMalloc on device 0");
  std::printf("        device 0: %p   device 1: %p\n", p0, p1);

  // Not asserted as equal: whether the two spaces collide at this size is the
  // allocator's business and could reasonably change. What IS asserted is that
  // each pointer works on its own device, whether or not they alias.
  std::vector<unsigned char> host(256, 0xA5);
  std::vector<unsigned char> back(256, 0x00);

  grxSetDevice(0);
  grxtest::check(grxMemcpy(p0, host.data(), 256, grxMemcpyHostToDevice) == grxSuccess,
          "write to device 0's pointer, on device 0");
  grxtest::check(grxMemcpy(back.data(), p0, 256, grxMemcpyDeviceToHost) == grxSuccess,
          "read it back");
  grxtest::check(std::memcmp(host.data(), back.data(), 256) == 0,
          "and the bytes are the ones that were written");

  grxSetDevice(1);
  std::vector<unsigned char> other(256, 0x5A);
  grxtest::check(grxMemcpy(p1, other.data(), 256, grxMemcpyHostToDevice) == grxSuccess,
          "write to device 1's pointer, on device 1");

  // If the two addresses alias, this is the case that matters: device 1's
  // write must not have landed in device 0's memory.
  grxSetDevice(0);
  std::memset(back.data(), 0, back.size());
  grxtest::check(grxMemcpy(back.data(), p0, 256, grxMemcpyDeviceToHost) == grxSuccess,
          "re-read device 0's pointer");
  grxtest::check(std::memcmp(host.data(), back.data(), 256) == 0,
          "device 0's bytes survived a write to device 1 at the same address");

  // ---- a pointer that is live on ANOTHER device and nowhere here ---------
  //
  // Built by allocating far enough into device 1 that device 0 has no live
  // allocation covering the address, so the refusal is unambiguous.
  grxSetDevice(1);
  void* far1 = nullptr;
  grxtest::check(grxMalloc(&far1, 1 << 20) == grxSuccess, "a 1 MiB allocation on device 1");
  void* deep = (char*)far1 + (1 << 19);

  grxSetDevice(0);
  const grxError_t memset_rc = grxMemset(deep, 0, 64);
  const grxError_t memcpy_rc = grxMemcpy(deep, host.data(), 64, grxMemcpyHostToDevice);
  const grxError_t free_rc   = grxFree(far1);

  // Is that address ALSO live on device 0? Asked by trying a read of it as a
  // device pointer here; if the spaces happen to overlap at this offset there
  // is nothing to refuse, and asserting a coincidence would make this test
  // pass or fail on allocator sizing rather than on the rule it is about.
  const bool also_local =
      (grxMemcpy(back.data(), deep, 64, grxMemcpyDeviceToHost) == grxSuccess);
  if (also_local) {
    std::printf("        note: that address is also live on device 0 here, so\n"
                "              there is nothing to refuse. Skipping the three\n"
                "              refusal cases rather than asserting a coincidence.\n");
  } else {
    grxtest::check(memset_rc == grxErrorInvalidDevicePointer,
            "grxMemset on another device's pointer is refused");
    grxtest::check(memcpy_rc == grxErrorInvalidDevicePointer,
            "grxMemcpy to another device's pointer is refused");
    grxtest::check(free_rc == grxErrorInvalidDevicePointer,
            "grxFree of another device's pointer is refused");
  }

  // POSITIVE CONTROL. Without it, a runtime that refused every pointer
  // everywhere would pass all three assertions above.
  grxSetDevice(1);
  grxtest::check(grxMemset(deep, 0, 64) == grxSuccess,
          "and the same pointer is accepted on the device that owns it");
  grxtest::check(grxFree(far1) == grxSuccess, "as is freeing it there");

  grxSetDevice(0);
  grxFree(p0);
  grxFree(first0);
  grxSetDevice(1);
  grxFree(p1);

  return grxtest::report();
}
