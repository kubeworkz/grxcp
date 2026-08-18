// Phase 1 gate: streams and events.
//
// SCOPE WARNING. The mock driver completes every enqueue before returning, so
// these tests verify the API contract -- handle lifetimes, ordering
// bookkeeping, error surface, data correctness through a stream -- and NOT
// concurrency. No test here can fail because of a race, which means none of
// them can pass because there isn't one. Overlap and asynchrony are tier-2
// properties and need a real backend (ci/README.md).

#include <grx/grx.h>

#include "grx_test.h"

#include <cstdint>
#include <cstring>
#include <vector>

using grxtest::check;
using grxtest::section;

int main() {
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) {
    std::printf("no devices; skipping\n");
    return 77;
  }
  grxDeviceProp_t prop{};
  GRX_REQUIRE(grxGetDeviceProperties(&prop, 0), "device properties");

  // -------------------------------------------------------------------------
  section("stream lifecycle");
  // -------------------------------------------------------------------------
  {
    grxStream_t s = nullptr;
    GRX_REQUIRE(grxStreamCreate(&s), "grxStreamCreate");
    check(s != nullptr, "stream handle is non-null");
    check(grxStreamQuery(s) == grxSuccess, "a fresh stream is idle");
    GRX_REQUIRE(grxStreamSynchronize(s), "grxStreamSynchronize");
    GRX_REQUIRE(grxStreamDestroy(s), "grxStreamDestroy");

    check(grxStreamDestroy(s) == grxErrorInvalidResourceHandle,
          "destroying a stream twice is rejected");
    check(grxStreamDestroy((grxStream_t)0x1234) == grxErrorInvalidResourceHandle,
          "a bogus stream handle is rejected");

    grxStream_t nb = nullptr;
    GRX_REQUIRE(grxStreamCreateWithFlags(&nb, grxStreamNonBlocking),
                "create a non-blocking stream");
    GRX_REQUIRE(grxStreamDestroy(nb), "destroy the non-blocking stream");

    grxStream_t hi = nullptr;
    GRX_REQUIRE(grxStreamCreateWithPriority(&hi, grxStreamDefault, -1),
                "create a high-priority stream");
    GRX_REQUIRE(grxStreamDestroy(hi), "destroy the high-priority stream");
  }

  // -------------------------------------------------------------------------
  section("data through a stream");
  // -------------------------------------------------------------------------
  {
    constexpr size_t N = 4096;
    std::vector<uint8_t> src(N), dst(N, 0);
    for (size_t i = 0; i < N; ++i) src[i] = (uint8_t)(i * 31 + 7);

    grxStream_t s = nullptr;
    GRX_REQUIRE(grxStreamCreate(&s), "create a work stream");

    void* d = nullptr;
    GRX_REQUIRE(grxMalloc(&d, N), "allocate device buffer");
    GRX_REQUIRE(grxMemcpyAsync(d, src.data(), N, grxMemcpyDefault, s),
                "async copy host to device");
    GRX_REQUIRE(grxMemsetAsync((uint8_t*)d + N / 2, 0, N / 2, s),
                "async memset the upper half");
    GRX_REQUIRE(grxMemcpyAsync(dst.data(), d, N, grxMemcpyDefault, s),
                "async copy device to host");
    GRX_REQUIRE(grxStreamSynchronize(s), "synchronize the stream");

    check(std::memcmp(dst.data(), src.data(), N / 2) == 0,
          "lower half survives the stream sequence");
    bool upper_zero = true;
    for (size_t i = N / 2; i < N; ++i) upper_zero = upper_zero && (dst[i] == 0);
    check(upper_zero, "upper half reflects the in-stream memset");

    GRX_REQUIRE(grxFree(d), "free device buffer");
    GRX_REQUIRE(grxStreamDestroy(s), "destroy the work stream");
  }

  // -------------------------------------------------------------------------
  section("event lifecycle");
  // -------------------------------------------------------------------------
  {
    grxEvent_t e = nullptr;
    GRX_REQUIRE(grxEventCreate(&e), "grxEventCreate");
    check(e != nullptr, "event handle is non-null");

    // An event that was never recorded is complete by definition, matching
    // CUDA: querying or synchronizing it must not block or fail.
    check(grxEventQuery(e) == grxSuccess, "an unrecorded event queries complete");
    GRX_REQUIRE(grxEventSynchronize(e), "synchronizing an unrecorded event");

    GRX_REQUIRE(grxEventRecord(e, nullptr), "record on the null stream");
    GRX_REQUIRE(grxEventSynchronize(e), "synchronize after record");
    check(grxEventQuery(e) == grxSuccess, "a completed event queries complete");

    // Re-recording must reuse the same underlying timeline rather than leak a
    // new driver event each time round a loop.
    for (int i = 0; i < 64; ++i)
      if (grxEventRecord(e, nullptr) != grxSuccess) {
        check(false, "repeated record");
        break;
      }
    GRX_REQUIRE(grxEventSynchronize(e), "synchronize after repeated records");

    GRX_REQUIRE(grxEventDestroy(e), "grxEventDestroy");
    check(grxEventDestroy(e) == grxErrorInvalidResourceHandle,
          "destroying an event twice is rejected");
  }

  // -------------------------------------------------------------------------
  section("event timing");
  // -------------------------------------------------------------------------
  {
    grxEvent_t start = nullptr, stop = nullptr;
    GRX_REQUIRE(grxEventCreate(&start), "create start event");
    GRX_REQUIRE(grxEventCreate(&stop),  "create stop event");

    void* d = nullptr;
    GRX_REQUIRE(grxMalloc(&d, 1 << 16), "allocate for a timed copy");
    std::vector<uint8_t> host(1 << 16, 0xC3);

    GRX_REQUIRE(grxEventRecord(start, nullptr), "record start");
    GRX_REQUIRE(grxMemcpy(d, host.data(), host.size(), grxMemcpyDefault),
                "timed copy");
    GRX_REQUIRE(grxEventRecord(stop, nullptr), "record stop");
    GRX_REQUIRE(grxEventSynchronize(stop), "synchronize stop");

    float ms = -1.0f;
    GRX_REQUIRE(grxEventElapsedTime(&ms, start, stop), "grxEventElapsedTime");
    check(ms >= 0.0f, "elapsed time is non-negative");

    // Which clock produced that number is not a detail the caller should have
    // to guess at. While the command processor's profiling writeback is a
    // skeleton the runtime falls back to a host clock and says so.
    check(prop.eventTimingIsDeviceSide == 0,
          "device property reports the host-clock fallback is in effect");

    grxEvent_t untimed = nullptr;
    GRX_REQUIRE(grxEventCreateWithFlags(&untimed, grxEventDisableTiming),
                "create an event with timing disabled");
    GRX_REQUIRE(grxEventRecord(untimed, nullptr), "record the untimed event");
    check(grxEventElapsedTime(&ms, start, untimed) ==
              grxErrorInvalidResourceHandle,
          "elapsed time on a timing-disabled event is rejected");

    grxEvent_t fresh = nullptr;
    GRX_REQUIRE(grxEventCreate(&fresh), "create an unrecorded event");
    check(grxEventElapsedTime(&ms, start, fresh) == grxErrorNotReady,
          "elapsed time against an unrecorded event reports not-ready");

    grxEventDestroy(fresh);
    grxEventDestroy(untimed);
    grxEventDestroy(start);
    grxEventDestroy(stop);
    grxFree(d);
  }

  // -------------------------------------------------------------------------
  section("cross-stream ordering");
  // -------------------------------------------------------------------------
  {
    grxStream_t a = nullptr, b = nullptr;
    GRX_REQUIRE(grxStreamCreate(&a), "create stream a");
    GRX_REQUIRE(grxStreamCreate(&b), "create stream b");

    grxEvent_t gate = nullptr;
    GRX_REQUIRE(grxEventCreate(&gate), "create the gating event");

    constexpr size_t N = 1024;
    std::vector<uint8_t> src(N, 0x77), dst(N, 0);
    void *d1 = nullptr, *d2 = nullptr;
    GRX_REQUIRE(grxMalloc(&d1, N), "allocate first buffer");
    GRX_REQUIRE(grxMalloc(&d2, N), "allocate second buffer");

    GRX_REQUIRE(grxMemcpyAsync(d1, src.data(), N, grxMemcpyDefault, a),
                "produce on stream a");
    GRX_REQUIRE(grxEventRecord(gate, a), "record the gate on stream a");
    GRX_REQUIRE(grxStreamWaitEvent(b, gate, 0),
                "stream b waits on the gate");
    GRX_REQUIRE(grxMemcpyAsync(d2, d1, N, grxMemcpyDefault, b),
                "consume on stream b");
    GRX_REQUIRE(grxMemcpyAsync(dst.data(), d2, N, grxMemcpyDefault, b),
                "read back on stream b");
    GRX_REQUIRE(grxStreamSynchronize(b), "synchronize stream b");

    check(std::memcmp(dst.data(), src.data(), N) == 0,
          "data crosses the stream boundary intact");

    // Waiting on an event that was never recorded is a no-op in CUDA, not an
    // error, and code in the wild relies on that.
    grxEvent_t never = nullptr;
    GRX_REQUIRE(grxEventCreate(&never), "create an unrecorded event");
    check(grxStreamWaitEvent(a, never, 0) == grxSuccess,
          "waiting on an unrecorded event is a no-op");

    grxEventDestroy(never);
    grxEventDestroy(gate);
    grxFree(d1);
    grxFree(d2);
    grxStreamDestroy(a);
    grxStreamDestroy(b);
  }

  // -------------------------------------------------------------------------
  section("device synchronize");
  // -------------------------------------------------------------------------
  {
    grxStream_t s = nullptr;
    GRX_REQUIRE(grxStreamCreate(&s), "create a stream");
    void* d = nullptr;
    GRX_REQUIRE(grxMalloc(&d, 4096), "allocate");
    std::vector<uint8_t> host(4096, 0x11);
    GRX_REQUIRE(grxMemcpyAsync(d, host.data(), 4096, grxMemcpyDefault, s),
                "enqueue work");
    GRX_REQUIRE(grxDeviceSynchronize(), "grxDeviceSynchronize drains every stream");
    check(grxStreamQuery(s) == grxSuccess,
          "the stream is idle after a device synchronize");
    grxFree(d);
    grxStreamDestroy(s);
  }

  return grxtest::report();
}
