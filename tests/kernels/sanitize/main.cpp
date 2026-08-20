// grx-sanitize gate driver.
//
//   test_sanitize <image.vxbin> <scenario>
//
// One scenario per run, because each is a separate claim: the sanitizer finds
// this bug, and finds nothing in the version without it. The program itself
// always exits 0 when the device work completes -- whether the run was clean
// is grx-sanitize's verdict to give, not this program's.
//
// Scenarios:
//   clean            correct kernel, correct host  -> no findings
//   oob-write        one element past the end      -> oob-global, write
//   oob-read         four elements before the base -> oob-global, read
//   use-after-free   launch against a freed buffer -> use-after-free
//   oob-shared       one word past sharedMem       -> oob-shared, write

#include <grx/grx.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"

#define CHECK(call)                                                      \
  do {                                                                   \
    grxError_t e_ = (call);                                              \
    if (e_ != grxSuccess) {                                              \
      std::fprintf(stderr, "%s -> %s (%s)\n", #call,                     \
                   grxGetErrorString(e_), grxGetErrorName(e_));          \
      return 1;                                                          \
    }                                                                    \
  } while (0)

int main(int argc, char** argv) {
  const char* image    = (argc > 1) ? argv[1] : "sanitize.vxbin";
  const std::string sc = (argc > 2) ? argv[2] : "clean";

  int count = 0;
  CHECK(grxGetDeviceCount(&count));
  if (count == 0) { std::fprintf(stderr, "no devices\n"); return 77; }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  grxModule_t mod = nullptr;
  CHECK(grxModuleLoad(&mod, image));

  const char* entry =
      (sc == "clean")          ? "san_clean" :
      (sc == "oob-write")      ? "san_oob_write" :
      (sc == "oob-read")       ? "san_oob_read" :
      (sc == "use-after-free") ? "san_use_after_free" :
      (sc == "oob-shared")     ? "san_oob_shared" : nullptr;
  if (!entry) { std::fprintf(stderr, "unknown scenario: %s\n", sc.c_str()); return 2; }

  grxFunction_t fn = nullptr;
  CHECK(grxModuleGetFunction(&fn, mod, entry));

  const unsigned block = (unsigned)prop.warpSize * 2;
  const unsigned n     = block * 2;

  // A first allocation, so "before the start" of the second lands in this
  // one's trailing redzone rather than off the front of the slab.
  void* guard = nullptr;
  CHECK(grxMalloc(&guard, 64 * sizeof(uint32_t)));

  void* out = nullptr;
  CHECK(grxMalloc(&out, (size_t)n * sizeof(uint32_t)));
  CHECK(grxMemset(out, 0, (size_t)n * sizeof(uint32_t)));

  san_args args{};
  args.out = (uint64_t)(uintptr_t)out;
  args.n   = n;

  const size_t shared = (sc == "oob-shared") ? (size_t)block * sizeof(uint32_t) : 0;

  if (sc == "use-after-free") {
    // The pointer in args is now stale. Nothing about the launch says so.
    CHECK(grxFree(out));
  }

  std::printf("%s: scenario %s, %u threads, %zu bytes shared\n", prop.name,
              sc.c_str(), n, shared);

  CHECK(grxLaunchFunction(fn, dim3_t{2, 1, 1}, dim3_t{block, 1, 1}, &args,
                          sizeof(args), shared, nullptr));
  CHECK(grxDeviceSynchronize());

  if (sc != "use-after-free") {
    std::vector<uint32_t> got(n, 0xffffffffu);
    CHECK(grxMemcpy(got.data(), out, got.size() * sizeof(uint32_t),
                    grxMemcpyDefault));
    std::printf("  first four words: %u %u %u %u\n", got[0], got[1], got[2],
                got[3]);
    CHECK(grxFree(out));
  }
  CHECK(grxFree(guard));
  CHECK(grxModuleUnload(mod));

  std::printf("device work completed\n");
  return 0;
}
