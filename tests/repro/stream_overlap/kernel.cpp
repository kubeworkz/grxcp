// Two kernels that rendezvous through a device global, for finding out whether
// two streams run at the same time.
//
// The test is a BOUNDED spin, not a barrier: `waiter` gives up after a fixed
// number of iterations and reports how it ended. So a device that serializes
// its streams answers the question in finite time instead of hanging, which is
// what makes this runnable in CI rather than a thing somebody tries by hand.

#include <grx/device/grx_device.h>

// Deliberately NOT atomic. The A extension is off in this configuration and an
// AMO aborts the simulator (cuda_mapping.md 7.16). A single writer and a single
// reader over one aligned word need no atomicity, only that the compiler is not
// allowed to cache it -- hence volatile.
__attribute__((used, retain)) volatile int g_flag = 0;

struct spin_args { uint64_t out; uint32_t limit; uint32_t expect; uint32_t self_at; };
struct set_args  { uint64_t unused; uint32_t value; uint32_t pad; };

// Spin until the flag holds `expect`, or until `limit` iterations run out.
// out[0] is the iteration it saw it on, or -1 if it never did.
//
// It waits for a SPECIFIC value rather than for non-zero, so that a flag left
// set by an earlier trial cannot be mistaken for this trial's setter.
//
// That is belt and braces rather than a fix for a known problem, and the
// distinction is worth recording because the first guess was wrong. When an
// early version reported "saw it at iteration 0" after a reload, the
// explanation reached for was a stale flag surviving in .bss. Checked
// separately: a module reload DOES reset both .data and .bss, so that was not
// it. The real cause is that two streams are two driver worker threads racing
// to submit, and about a third of the time the setter wins and finishes before
// the waiter's first read. The tag costs nothing and removes one variable; the
// iteration number is what actually distinguishes the cases.
__global__ void waiter(spin_args* __UNIFORM__ a) {
  int* out = (int*)(uintptr_t)a->out;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const int want = (int)a->expect;
  int saw = -1;
  for (uint32_t i = 0; i < a->limit; ++i) {
    // self_at is a control for the DETECTOR, not for the device: it makes the
    // waiter set the flag itself partway through its own spin, so a mid-spin
    // sighting is guaranteed to exist. If the host then reports iteration 0 or
    // -1, the measurement cannot see a mid-spin sighting at all, and its
    // "never saw it" on the real test would mean nothing. Zero disables it,
    // which is every real trial.
    if (a->self_at != 0 && i == a->self_at) g_flag = want;
    if (g_flag == want) { saw = (int)i; break; }
  }
  out[0] = saw;
}

// Set the flag the waiter is watching.
__global__ void setter(set_args* __UNIFORM__ a) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  g_flag = (int)a->value;
}

// A control: the same spin with nothing to wait for. If `waiter` reports -1
// this must too, and in about the same wall-clock -- otherwise "never saw it"
// might mean "never ran".
__global__ void spin_only(spin_args* __UNIFORM__ a) {
  int* out = (int*)(uintptr_t)a->out;
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  volatile int sink = 0;
  for (uint32_t i = 0; i < a->limit; ++i) sink += (int)i;
  out[0] = (sink == 0) ? 0 : -1;
}
