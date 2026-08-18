// Minimal assertion harness for the GRXCP unit tests.
//
// No framework dependency on purpose: these tests must build and run anywhere
// the runtime does, including on a riscv64 host, without pulling in a test
// library that may not be packaged there.

#ifndef GRXCP_TEST_H
#define GRXCP_TEST_H

#include <cstdio>

namespace grxtest {

inline int& failures() { static int n = 0; return n; }

inline void check(bool cond, const char* what) {
  if (!cond) { std::printf("  FAIL  %s\n", what); ++failures(); }
  else       { std::printf("  ok    %s\n", what); }
}

inline void section(const char* name) { std::printf("%s:\n", name); }

inline int report() {
  const int n = failures();
  std::printf("\n%s (%d failure%s)\n", n ? "FAILED" : "PASSED", n,
              n == 1 ? "" : "s");
  return n ? 1 : 0;
}

}  // namespace grxtest

#define GRX_REQUIRE(call, what)                                          \
  do {                                                                   \
    grxError_t e_ = (call);                                              \
    grxtest::check(e_ == grxSuccess, what);                              \
    if (e_ != grxSuccess) {                                              \
      std::printf("        -> %s\n", grxGetErrorString(e_));             \
      return grxtest::report();                                          \
    }                                                                    \
  } while (0)

#endif  // GRXCP_TEST_H
