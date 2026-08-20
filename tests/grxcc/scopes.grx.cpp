// grxcc's parser, exercised on the constructs a real C++ file contains.
//
// vecadd.grx.cpp is the exit gate: it proves a program compiles and computes.
// This one proves the LEXER AND SCOPE TRACKING are right, which is a separate
// claim and fails in a different way -- not a wrong answer but a mangled
// generated source, usually with an error naming something the author never
// wrote.
//
// Everything here computes a value the host checks, so a construct that is
// mis-parsed into a kernel that never runs is a FAIL rather than a silent pass.

#include <grx/grx.h>

#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Decoys. None of these is a kernel or a launch, and every one of them would
// be if the lexer did not know what is not code.
// ---------------------------------------------------------------------------

// __global__ void ghost(float*) and a launch ghost<<<1, 1>>>(nullptr)
/* __global__ void block_comment_ghost(float*); k<<<1,1>>>(0) */
static const char* kDecoy1 = "__global__ void in_a_string(float*); k<<<1,1>>>(0)";
static const char* kDecoy2 = R"raw(__global__ void in_raw(float*); k<<<1,1>>>(0))raw";
static const char  kDecoyChar = '<';

// A namespace ALIAS and a using-directive: `namespace` appears, no scope opens.
namespace alias_target { const int kUnused = 1; }
namespace at = alias_target;

// ---------------------------------------------------------------------------
// Kernels at three different scopes
// ---------------------------------------------------------------------------

// File scope.
__global__ void at_file_scope(float* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0f;
}

// Anonymous namespace: reachable unqualified from file scope, so the generated
// registration tables need no qualification at all.
namespace {
__global__ void in_anon(float* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += 2.0f;
}
}  // namespace

// Named, and nested. The stub and the args struct are emitted here, inside
// `outer::inner`; the registration tables are emitted at file scope and have to
// name them through the full path.
namespace outer {
namespace inner {
__global__ void nested(float* out, unsigned n) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] *= 10.0f;
}
}  // namespace inner
}  // namespace outer

// ---------------------------------------------------------------------------

#define CHECK(call)                                                        \
  do {                                                                     \
    grxError_t e_ = (call);                                                \
    if (e_ != grxSuccess) {                                                \
      std::printf("%s -> %s\n", #call, grxGetErrorString(e_));             \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  CHECK(grxSetDevice(0));

  grxDeviceProp_t prop{};
  CHECK(grxGetDeviceProperties(&prop, 0));

  const unsigned block = (unsigned)prop.warpSize * 2u;
  const unsigned n     = block * 3u + 1u;          // ragged on purpose
  const unsigned grid  = (n + block - 1u) / block;

  float* d = nullptr;
  CHECK(grxMalloc((void**)&d, n * sizeof(float)));

  // Three launches, three spellings of the callee: unqualified at file scope,
  // unqualified out of an anonymous namespace, and fully qualified.
  at_file_scope<<<grid, block>>>(d, n);
  CHECK(grxDeviceSynchronize());
  in_anon<<<grid, block>>>(d, n);
  CHECK(grxDeviceSynchronize());
  outer::inner::nested<<<grid, block>>>(d, n);
  CHECK(grxDeviceSynchronize());

  std::vector<float> got(n, -1.0f);
  CHECK(grxMemcpy(got.data(), d, n * sizeof(float), grxMemcpyDefault));

  int bad = 0;
  for (unsigned i = 0; i < n; ++i)
    if (got[i] != 30.0f) {                       // (1 + 2) * 10
      if (bad < 4) std::printf("  [%u] got %g want 30\n", i, (double)got[i]);
      ++bad;
    }
  std::printf("%s  three kernels at three scopes, %u elements\n",
              bad ? "FAIL" : "ok  ", n);

  CHECK(grxFree(d));

  // The decoys have to have survived as data. Touching them here also keeps the
  // compiler from deciding they were unused and removing the evidence.
  const bool decoys_intact = kDecoy1[0] == '_' && kDecoy2[0] == '_' &&
                             kDecoyChar == '<' && at::kUnused == 1;
  std::printf("%s  decoy strings and comments stayed data\n",
              decoys_intact ? "ok  " : "FAIL");

  if (bad || !decoys_intact) { std::printf("FAILED\n"); return 1; }
  std::printf("PASSED\n");
  return 0;
}
