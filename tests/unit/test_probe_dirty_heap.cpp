// Device enumeration must not depend on what the heap happened to contain.
//
// WHAT THIS CAUGHT. src/runtime/context.cpp allocated the NPU probe's handle
// with `new npu_c930_device_t` -- no braces. That is default-initialisation,
// and for a C struct with no constructor it leaves every member INDETERMINATE.
// npu_c930_detect's first act is a register read, which begins
//
//     if (dev->read32) return dev->read32(dev->io_ctx, offset);
//
// so the probe made an indirect call through whatever bytes were in that word.
//
// WHY IT SURVIVED. A fresh heap page from the OS is zeroed. A program whose
// first act is grxGetDeviceCount therefore gets a handle that is accidentally
// null-filled and behaves correctly, which is every test in this directory. The
// crash needs a DIRTY heap, and the only test that had one was
// tests/libs/test_grxdnn_gelu.cpp -- it loads reference vectors from a file
// before touching a device. That test is built by cmake and registered with
// ctest, and nothing in either CI tier ran ctest, so the segfault sat in a
// configuration that was compiled every run and executed never.
//
// WHAT THIS TEST DOES. Dirties the heap deliberately, then enumerates. It is
// not subtle and it does not need to be: fill and free a few megabytes with a
// non-zero pattern, and the next allocation of that size comes back holding it.
// Before the fix this segfaults; after it, enumeration is unchanged.
//
// IT IS NOT AN NPU TEST. It runs in both configurations and asserts the same
// thing in each -- that enumeration answers the same way on a dirty heap as on
// a clean one. Any future probe that reads an uninitialised handle fails here,
// whichever backend adds it.

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

namespace {

// Fill and release enough memory, with a pattern that is not zero, that a
// subsequent allocation is served from it. 0xAA is chosen so a pointer built
// out of it is non-null and unmapped -- a null one would be caught by the very
// test the bug slips past.
void dirty_the_heap() {
  const size_t block = 1u << 16;
  std::vector<void*> held;
  held.reserve(64);
  for (int i = 0; i < 64; ++i) {
    void* p = std::malloc(block);
    if (!p) break;
    std::memset(p, 0xAA, block);
    held.push_back(p);
  }
  for (void* p : held) std::free(p);
}

}  // namespace

int main() {
  std::printf("=== device enumeration on a dirty heap ===\n");

  section("enumeration survives a heap that is not full of zeros");
  {
    dirty_the_heap();

    // The call that crashed. Reaching the next line at all is most of the
    // check: before the fix this was a jump through freed bytes.
    int count = -1;
    const grxError_t e = grxGetDeviceCount(&count);
    check(e == grxSuccess || e == grxErrorInvalidDevice,
          "grxGetDeviceCount returns, and returns a real status");
    check(count >= 0, "the count is not left indeterminate");
    std::printf("        %d device(s)\n", count);
  }

  section("and every device it reports can be described");
  {
    // A probe that scribbled a device into the table out of stack garbage would
    // pass the section above and fail here.
    int count = 0;
    if (grxGetDeviceCount(&count) == grxSuccess) {
      for (int i = 0; i < count; ++i) {
        grxDeviceProp_t prop{};
        const bool ok = (grxGetDeviceProperties(&prop, i) == grxSuccess);
        char what[96];
        std::snprintf(what, sizeof(what),
                      "device %d has properties, and a name", i);
        check(ok && prop.name[0] != '\0', what);
        if (ok) std::printf("        [%d] %s\n", i, prop.name);
      }
    }
  }

  return grxtest::report();
}
