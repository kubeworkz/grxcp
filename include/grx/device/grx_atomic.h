// GRXCP — CUDA's device-side atomic family, over the RISC-V A extension.
//
// WHY THIS HEADER IS MOSTLY A REFUSAL
//
// The device toolchain compiles `-march=rv64imafd`. The **a** is the atomic
// extension, so clang lowers a `std::atomic`, a `__atomic_fetch_add` or an
// `__sync_*` builtin to an AMO without a word of complaint. A GRX-G100 built
// with VX_CFG_EXT_A_ENABLE off -- which this sysroot is -- decodes that
// instruction, routes it to the LSU, and calls std::abort() in the simulator.
// No message, no line, no trace: the process is gone, and nothing in the stack
// has said the word "atomic".
//
// That is not hypothetical. It is how grx-sanitize's first draft died, counting
// findings with __atomic_fetch_add on a device-memory word. See
// docs/designs/cuda_mapping.md section 7.16.
//
// So on a build without the extension every entry point here is `unavailable`,
// and a program that calls one gets a compile error that NAMES THE REASON:
//
//   error: 'atomicAdd' is unavailable: this GRX-G100 build has
//   VX_CFG_EXT_A_ENABLE off, so an atomic would lower to an AMO the hardware
//   aborts on...
//
// cuda_mapping.md 7.16 said this header should `#error` when the extension is
// off. Per-function `unavailable` is better and this is the deviation: an
// #error breaks every kernel in every file that includes the header, including
// the ones that never wanted an atomic -- and grxcc includes it in every device
// pass, the way a CUDA frontend supplies these names implicitly. Only a program
// that actually calls one should hear about it.
//
// WHAT IS AND IS NOT PROVIDED
//
// The integer forms map to real AMO instructions. The FLOAT forms do not: there
// is no floating-point AMO in RISC-V, and CUDA's float atomicAdd would have to
// be a compare-and-swap loop. That loop is written here because a CAS loop over
// a hardware CAS is a faithful implementation, unlike the barrier-based fake
// 7.16 rejected. 64-bit forms need the A extension at XLEN 64, which is the
// only configuration GRXCP supports.

#ifndef GRX_ATOMIC_H
#define GRX_ATOMIC_H

#include <grx/device/grx_device.h>

#ifndef VX_CFG_EXT_A_ENABLED
// Not built through ci/build_kernel.sh, so the configuration is unknown.
// Assume absent: refusing something that would have worked costs a compile
// error, and permitting something that will not costs a silent abort.
#define VX_CFG_EXT_A_ENABLED 0
#endif

#if VX_CFG_EXT_A_ENABLED

#define __GRX_ATOMIC_ATTR __attribute__((always_inline)) inline

// The compiler's own atomic builtins, which lower to AMOs on this target.
// Relaxed ordering matches CUDA: an atomic is atomic, not a fence. A program
// that wants ordering uses __threadfence().
#define __GRX_AMO(name, op, T)                                              \
  __GRX_ATOMIC_ATTR T name(T* address, T val) {                             \
    return __atomic_##op(address, val, __ATOMIC_RELAXED);                   \
  }

__GRX_AMO(atomicAdd, fetch_add, int)
__GRX_AMO(atomicAdd, fetch_add, unsigned int)
__GRX_AMO(atomicAdd, fetch_add, unsigned long long)
__GRX_AMO(atomicSub, fetch_sub, int)
__GRX_AMO(atomicSub, fetch_sub, unsigned int)
__GRX_AMO(atomicAnd, fetch_and, int)
__GRX_AMO(atomicAnd, fetch_and, unsigned int)
__GRX_AMO(atomicOr,  fetch_or,  int)
__GRX_AMO(atomicOr,  fetch_or,  unsigned int)
__GRX_AMO(atomicXor, fetch_xor, int)
__GRX_AMO(atomicXor, fetch_xor, unsigned int)

#undef __GRX_AMO

__GRX_ATOMIC_ATTR int atomicExch(int* address, int val) {
  return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__GRX_ATOMIC_ATTR unsigned int atomicExch(unsigned int* address,
                                          unsigned int val) {
  return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}

__GRX_ATOMIC_ATTR int atomicMin(int* address, int val) {
  return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__GRX_ATOMIC_ATTR unsigned int atomicMin(unsigned int* address,
                                         unsigned int val) {
  return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__GRX_ATOMIC_ATTR int atomicMax(int* address, int val) {
  return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__GRX_ATOMIC_ATTR unsigned int atomicMax(unsigned int* address,
                                         unsigned int val) {
  return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}

__GRX_ATOMIC_ATTR int atomicCAS(int* address, int compare, int val) {
  __atomic_compare_exchange_n(address, &compare, val, false, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
  return compare;   // CUDA returns the value that was there
}
__GRX_ATOMIC_ATTR unsigned int atomicCAS(unsigned int* address,
                                         unsigned int compare,
                                         unsigned int val) {
  __atomic_compare_exchange_n(address, &compare, val, false, __ATOMIC_RELAXED,
                              __ATOMIC_RELAXED);
  return compare;
}

// float atomicAdd, as a CAS loop -- RISC-V has no floating-point AMO. This is
// what CUDA did on sm_1x and what every portable implementation still does.
__GRX_ATOMIC_ATTR float atomicAdd(float* address, float val) {
  unsigned int* as_uint = (unsigned int*)address;
  unsigned int old = __atomic_load_n(as_uint, __ATOMIC_RELAXED), assumed;
  do {
    assumed = old;
    float f;
    __builtin_memcpy(&f, &assumed, sizeof(f));
    f += val;
    unsigned int next;
    __builtin_memcpy(&next, &f, sizeof(next));
    old = atomicCAS(as_uint, assumed, next);
  } while (assumed != old);
  float result;
  __builtin_memcpy(&result, &old, sizeof(result));
  return result;
}

#undef __GRX_ATOMIC_ATTR

#else   // !VX_CFG_EXT_A_ENABLED

#define __GRX_NO_ATOMIC                                                     \
  __attribute__((unavailable(                                               \
      "this GRX-G100 build has VX_CFG_EXT_A_ENABLE off, so an atomic would " \
      "lower to an AMO the hardware aborts on -- with no message and no "   \
      "line. Ask grxDeviceProp_t::capabilities for GRX_CAP_GLOBAL_ATOMICS " \
      "before using one, restructure so each thread owns its output (the "  \
      "way grx-sanitize indexes its report table by grid-linear thread), "  \
      "or build a sysroot with the extension enabled. "                     \
      "docs/designs/cuda_mapping.md section 7.16.")))

// Declared, not defined. The declarations are what turn "undeclared
// identifier 'atomicAdd'" -- which sends the reader looking for a missing
// include -- into a message about this device.
__GRX_NO_ATOMIC int          atomicAdd(int*, int);
__GRX_NO_ATOMIC unsigned int atomicAdd(unsigned int*, unsigned int);
__GRX_NO_ATOMIC unsigned long long atomicAdd(unsigned long long*,
                                             unsigned long long);
__GRX_NO_ATOMIC float        atomicAdd(float*, float);
__GRX_NO_ATOMIC int          atomicSub(int*, int);
__GRX_NO_ATOMIC unsigned int atomicSub(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicExch(int*, int);
__GRX_NO_ATOMIC unsigned int atomicExch(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicMin(int*, int);
__GRX_NO_ATOMIC unsigned int atomicMin(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicMax(int*, int);
__GRX_NO_ATOMIC unsigned int atomicMax(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicAnd(int*, int);
__GRX_NO_ATOMIC unsigned int atomicAnd(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicOr(int*, int);
__GRX_NO_ATOMIC unsigned int atomicOr(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicXor(int*, int);
__GRX_NO_ATOMIC unsigned int atomicXor(unsigned int*, unsigned int);
__GRX_NO_ATOMIC int          atomicCAS(int*, int, int);
__GRX_NO_ATOMIC unsigned int atomicCAS(unsigned int*, unsigned int,
                                       unsigned int);

#undef __GRX_NO_ATOMIC

#endif  // VX_CFG_EXT_A_ENABLED
#endif  // GRX_ATOMIC_H
