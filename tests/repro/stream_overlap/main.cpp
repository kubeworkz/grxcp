// Do two GRXCP streams run at the same time?
//
// WHY THIS EXISTS. src/runtime/stream.cpp says "the driver serializes
// launches", and docs/designs/grxcp_roadmap.md gates the whole of phase 5 on
// that being true -- multi-queue scheduling, copy/compute overlap, retiring the
// launch serialization, all of it waits on the upstream QMD-style atomic
// launch. That is a large decision resting on a sentence nobody had re-checked
// since it was written, so this checks it.
//
// THE TEST. Two kernels rendezvous through a device global:
//
//   stream A   waiter  -- spins reading g_flag, up to a bounded limit
//   stream B   setter  -- writes g_flag
//
// The waiter is enqueued FIRST, and reports the iteration it saw the flag on.
// The budget is bounded, so both arrangements terminate -- a barrier would hang
// on a serialized device and could not be run in CI.
//
// THE ITERATION NUMBER IS THE WHOLE ANSWER, and the first version of this test
// got that wrong by only asking "did you see it".
//
//   saw at iteration > 0   OVERLAP. The waiter was already spinning when the
//                          setter ran. Nothing but concurrency produces this.
//   never saw it           SERIALIZED, for this run. The waiter ran to
//                          completion and the setter had still not run.
//   saw at iteration 0     INCONCLUSIVE. The flag was already set on the very
//                          first read, so the setter finished before the waiter
//                          began -- which says the launches were REORDERED, not
//                          that they overlapped.
//
// That third case is not rare -- measured over eight runs: five serialized,
// three of them, zero overlaps. Two streams are two driver worker threads
// racing to submit, and CUDA promises no ordering between independent streams
// either. Every "saw it" was at iteration 0 EXACTLY, never at 500 or 1200,
// which is what makes the reordering reading the right one rather than a
// convenient one. A test that counted iteration 0 as overlap would report
// overlap a third of the time on a device that has none.
//
// (The first explanation offered for those iteration-0 runs was a stale flag
// surviving the module reload. It was wrong -- a reload resets both .data and
// .bss, checked separately. Recorded because the wrong answer was the more
// obvious one.)
//
// So this runs the trial several times and concludes from the set: any run
// seeing the flag mid-spin proves overlap; otherwise at least one run has to
// have gone the distance before "serialized" is claimed.
//
// This is a WATCH, not a gate. Exit code is 0 whichever way it lands; read the
// message. The day a mid-spin sighting appears, phase 5 is unblocked.

#include <grx/grx.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

struct spin_args { uint64_t out; uint32_t limit; uint32_t expect; uint32_t self_at; };
struct set_args  { uint64_t unused; uint32_t value; uint32_t pad; };

#define CHECK(call)                                                        \
  do {                                                                     \
    grxError_t e_ = (call);                                                \
    if (e_ != grxSuccess) {                                                \
      std::printf("  %s -> %s\n", #call, grxGetErrorString(e_));           \
      return -2;                                                           \
    }                                                                      \
  } while (0)

// Run the rendezvous once. `setter_first` picks the positive-control
// arrangement. Returns the iteration the waiter saw the flag on, -1 if it never
// did, or -2 on an API failure.
//
// Each run uses its own `tag`, so a flag left set by an earlier run cannot be
// mistaken for this one's setter. (A module reload does reset both .data and
// .bss -- checked separately -- but the tag costs nothing and removes the
// question.)
int rendezvous(grxModule_t mod, uint32_t limit, bool setter_first,
               uint32_t tag, uint32_t self_at = 0,
               bool launch_setter = true) {
  grxFunction_t fn_wait = nullptr, fn_set = nullptr;
  if (grxModuleGetFunction(&fn_wait, mod, "waiter") != grxSuccess) return -2;
  if (grxModuleGetFunction(&fn_set, mod, "setter") != grxSuccess) return -2;

  int* d_out = nullptr;
  CHECK(grxMalloc((void**)&d_out, sizeof(int)));
  const int sentinel = -7;
  CHECK(grxMemcpy(d_out, &sentinel, sizeof(int), grxMemcpyHostToDevice));

  grxStream_t sa = nullptr, sb = nullptr;
  CHECK(grxStreamCreate(&sa));
  CHECK(grxStreamCreate(&sb));

  spin_args wa{};
  wa.out    = (uint64_t)(uintptr_t)d_out;
  wa.limit  = limit;
  wa.expect  = tag;
  wa.self_at = self_at;
  set_args sb_args{};
  sb_args.value = tag;

  if (!launch_setter) {
    // The self-set control runs the waiter ALONE. Launching a setter as well
    // would let the same reordering race the real test suffers from set the
    // flag before the waiter's first read, which is what made the first
    // version of this control report iteration 0 and fail.
    CHECK(grxLaunchFunction(fn_wait, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                            &wa, sizeof(wa), 0, sa));
  } else if (setter_first) {
    CHECK(grxLaunchFunction(fn_set, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                            &sb_args, sizeof(sb_args), 0, sb));
    CHECK(grxStreamSynchronize(sb));
    CHECK(grxLaunchFunction(fn_wait, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                            &wa, sizeof(wa), 0, sa));
  } else {
    CHECK(grxLaunchFunction(fn_wait, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                            &wa, sizeof(wa), 0, sa));
    CHECK(grxLaunchFunction(fn_set, dim3_t{1, 1, 1}, dim3_t{1, 1, 1},
                            &sb_args, sizeof(sb_args), 0, sb));
  }

  CHECK(grxDeviceSynchronize());

  int saw = -7;
  CHECK(grxMemcpy(&saw, d_out, sizeof(int), grxMemcpyDeviceToHost));

  grxStreamDestroy(sa);
  grxStreamDestroy(sb);
  grxFree(d_out);
  return saw;
}

}  // namespace

int main(int argc, char** argv) {
  const char* image = (argc > 1) ? argv[1] : "stream_overlap.vxbin";
  const uint32_t limit = (argc > 2) ? (uint32_t)std::atoi(argv[2]) : 3000u;
  const int trials = (argc > 3) ? std::atoi(argv[3]) : 6;

  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count == 0) {
    std::printf("SKIPPED: no device\n");
    return 77;
  }
  if (grxSetDevice(0) != grxSuccess) return 1;

  grxDeviceProp_t prop{};
  grxGetDeviceProperties(&prop, 0);

  grxModule_t mod = nullptr;
  if (grxModuleLoad(&mod, image) != grxSuccess) {
    std::printf("SKIPPED: cannot load %s\n", image);
    return 77;
  }

  // The positive control. If a device global written by one kernel is not
  // visible to the next, nothing below would mean anything.
  const int control = rendezvous(mod, limit, /*setter_first=*/true, 11u);
  if (control < 0) {
    std::printf("FAILED: the control (setter first, then waiter) reported %d.\n"
                "        The rendezvous itself is broken.\n", control);
    grxModuleUnload(mod);
    return 1;
  }
  std::printf("  control: setter first -> saw the flag at iteration %d\n",
              control);

  // A control for the DETECTOR. The waiter sets the flag itself halfway
  // through its own spin, so a mid-spin sighting definitely exists. If this
  // came back 0 or -1, the measurement could not report one at all and
  // "never saw it" below would be worth nothing.
  grxModuleUnload(mod);
  if (grxModuleLoad(&mod, image) != grxSuccess) return 1;
  const uint32_t half = limit / 2;
  const int self = rendezvous(mod, limit, /*setter_first=*/false, 33u, half,
                              /*launch_setter=*/false);
  if (self <= 0 || (uint32_t)self < half) {
    std::printf("FAILED: the self-set control reported %d, expected about "
                "%u.\n"
                "        This measurement cannot see a mid-spin sighting, so "
                "it cannot tell\n"
                "        overlap from its absence.\n", self, half);
    grxModuleUnload(mod);
    return 1;
  }
  std::printf("  control: waiter sets the flag at %u -> saw it at %d "
              "(a mid-spin sighting IS visible)\n", half, self);

  int overlaps = 0, serialized = 0, inconclusive = 0, first_overlap = -1;
  for (int t = 0; t < trials; ++t) {
    // A fresh module each trial, and only one may be resident at a time
    // (cuda_mapping.md 7.13), so unload before loading.
    grxModuleUnload(mod);
    if (grxModuleLoad(&mod, image) != grxSuccess) return 1;

    const int saw = rendezvous(mod, limit, /*setter_first=*/false,
                               (uint32_t)(100 + t));
    if (saw == -2) { std::printf("  trial %d: API failure\n", t); return 1; }
    if (saw > 0)       { ++overlaps; if (first_overlap < 0) first_overlap = saw; }
    else if (saw == 0) { ++inconclusive; }
    else               { ++serialized; }
  }
  grxModuleUnload(mod);

  std::printf("  %d trials, %u iteration budget: %d overlapped, %d ran the "
              "distance, %d reordered\n",
              trials, limit, overlaps, serialized, inconclusive);

  if (overlaps > 0) {
    std::printf("\nSTREAMS OVERLAP on %s.\n", prop.name);
    std::printf("  A waiter saw the flag at iteration %d -- mid-spin, so the "
                "setter ran while\n"
                "  it was still going. src/runtime/stream.cpp's note that "
                "\"the driver\n"
                "  serializes launches\" is now WRONG, and phase 5's first "
                "item is unblocked:\n"
                "  multi-queue stream scheduling has something to schedule.\n"
                "  Re-read grxcp_roadmap.md phase 5 and cuda_mapping.md 7.3.\n",
                first_overlap);
  } else if (serialized > 0) {
    std::printf("\nSTREAMS ARE SERIALIZED on %s (expected).\n", prop.name);
    std::printf("  %d of %d trials ran the waiter's whole budget without the "
                "setter ever\n"
                "  running -- which it could only do if the second stream "
                "waited for the\n"
                "  first to retire. No trial saw the flag mid-spin. This is "
                "what\n"
                "  cuda_mapping.md 7.3 describes and what phase 5 waits on.\n",
                serialized, trials);
    if (inconclusive > 0)
      std::printf("  (%d trials were reordered -- the setter finished before "
                  "the waiter began.\n"
                  "   Two streams are two driver worker threads racing to "
                  "submit, and CUDA\n"
                  "   promises no ordering between independent streams either. "
                  "Those trials\n"
                  "   say nothing either way, which is why this runs several.)\n",
                  inconclusive);
  } else {
    std::printf("\nINCONCLUSIVE on %s: all %d trials were reordered.\n",
                prop.name, trials);
    std::printf("  Every waiter found the flag already set on its first read, "
                "so none of them\n"
                "  ever spun against a running setter. Raise the trial count "
                "or the budget.\n");
  }
  return 0;
}
