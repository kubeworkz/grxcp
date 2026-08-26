// Where a GEMM lands, and that the answer is about the device you are on.
//
// THE RULE UNDER TEST: the current device decides which engine a grxBLAS call
// runs on. Nothing redirects it, nothing falls back, and grxblasGetGemmEngine
// reports the decision from the same function that makes it.
//
// WHAT THIS FOUND. The NPU routing read
//
//     grxGetDeviceProperties(&prop, grxGetDevice(nullptr))
//
// and grxGetDevice writes through its argument and RETURNS AN ERROR CODE. With
// nullptr it returns grxErrorInvalidValue, the value 1, so the properties query
// was `(&prop, 1)`: the routing consulted device index 1 on every call and the
// current device on none of them.
//
// Two ways that goes wrong. On a machine whose index 1 is not an NPU,
// grxSetDevice(npu) routed to the GPU. On a machine whose index 1 IS the NPU --
// which is every machine with one GPU and one NPU, because the NPU is appended
// after the GPU -- an INT8 GemmEx issued while the current device was the GPU
// routed to the NPU. Work on silicon the caller did not select, with no error.
//
// It survived because nothing asked. The decision was a private static with no
// way to observe it short of owning a c930, so the assertion this file makes --
// that the reported DEVICE INDEX is the current one -- needs no NPU at all.

#include <grx/grx.h>
#include <grx/grxblas.h>

#include "grx_test.h"

#include <cstdio>

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }

  grxblasHandle_t h = nullptr;
  if (grxblasCreate(&h) != GRXBLAS_STATUS_SUCCESS) {
    std::printf("grxblasCreate failed; skipping\n");
    return 77;
  }

  grxtest::section("the reported engine is about the CURRENT device");

  for (int d = 0; d < count; ++d) {
    if (grxSetDevice(d) != grxSuccess) continue;

    grxblasEngine_t engine = GRXBLAS_ENGINE_NONE;
    int reported = -1;
    const grxblasStatus_t s =
        grxblasGetGemmEngine(h, 16, 16, 16, GRX_R_16F, GRX_R_16F, GRX_R_32F,
                             &engine, &reported);
    char what[96];
    std::snprintf(what, sizeof what,
                  "device %d: the decision was made about device %d", d,
                  reported);
    grxtest::check(s == GRXBLAS_STATUS_SUCCESS && reported == d, what);
    std::printf("        -> %s\n", grxblasGetEngineString(engine));

    // The same question in INT8, which is the pairing the NPU routing keys on.
    // Its answer must also be about device d and not about a fixed index.
    int reported8 = -1;
    grxblasEngine_t engine8 = GRXBLAS_ENGINE_NONE;
    grxblasGetGemmEngine(h, 8, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I, &engine8,
                         &reported8);
    std::snprintf(what, sizeof what,
                  "device %d: and so was the INT8 decision (device %d)", d,
                  reported8);
    grxtest::check(reported8 == d, what);
  }

  // The whole point of the bug was a decision that did not move when the
  // current device did. With one device that is unobservable, so say so rather
  // than claim a pass.
  if (count < 2) {
    std::printf("  note  one device here: the index cannot be seen to FOLLOW\n"
                "        grxSetDevice. Run with GRXMOCK_DEVICE_COUNT=2 for that.\n");
  } else {
    grxtest::section("and it follows grxSetDevice");
    int a = -1, b = -1;
    grxblasEngine_t e{};
    grxSetDevice(0);
    grxblasGetGemmEngine(h, 8, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I, &e, &a);
    grxSetDevice(1);
    grxblasGetGemmEngine(h, 8, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I, &e, &b);
    grxtest::check(a == 0 && b == 1,
                   "selecting device 1 moves the decision to device 1");
    if (!(a == 0 && b == 1))
      std::printf("        got %d then %d\n", a, b);
  }

  grxtest::section("the query refuses what it cannot answer");
  grxblasEngine_t e{};
  grxtest::check(grxblasGetGemmEngine(nullptr, 8, 8, 8, GRX_R_8I, GRX_R_8I,
                                      GRX_R_32I, &e, nullptr) ==
                     GRXBLAS_STATUS_INVALID_VALUE,
                 "a null handle is refused");
  grxtest::check(grxblasGetGemmEngine(h, 8, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I,
                                      nullptr, nullptr) ==
                     GRXBLAS_STATUS_INVALID_VALUE,
                 "a null engine pointer is refused");
  grxtest::check(grxblasGetGemmEngine(h, 0, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I,
                                      &e, nullptr) ==
                     GRXBLAS_STATUS_INVALID_VALUE,
                 "an empty shape is refused");

  // POSITIVE CONTROL: a runtime that refused everything would pass the three
  // above.
  grxSetDevice(0);
  grxtest::check(grxblasGetGemmEngine(h, 8, 8, 8, GRX_R_8I, GRX_R_8I, GRX_R_32I,
                                      &e, nullptr) == GRXBLAS_STATUS_SUCCESS,
                 "and a well-formed question is answered");

  grxblasDestroy(h);
  return grxtest::report();
}
