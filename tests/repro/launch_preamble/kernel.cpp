// What is inside the launch preamble?
//
// A block's cycles are 51% per-launch fixed cost, and 39.7 of those points are
// the PREAMBLE: the interval between the launch (which zeroes MCYCLE) and the
// first warp reaching its cycle probe. Measured at 9418 cycles per launch,
// near-constant across twelve different kernels and independent of grid size.
// Nothing has ever looked inside it.
//
// Three candidate stories, and they want different fixes:
//
//   1. Device bring-up -- reset, scheduler start, CTA dispatch, instruction
//      fetch. Not ours, not shortenable from software; fusion is the only
//      lever, and section 3 of developer_interface.md is right as written.
//   2. The FIRST MEMORY ACCESS. Every kernel opens by loading its ABI version
//      from the argument blob -- a cold read through a cold dcache. If that is
//      most of it, prefetch or a warmer path is worth more than fusion.
//   3. THE INSTRUMENT. `cycle_probe` reads with vx_rdcycle_sync, which emits a
//      CUSTOM0 fence before the CSR read. If that fence is expensive at entry,
//      part of the "preamble" is the measurement, and the 51% needs revising.
//
// This kernel takes readings that separate them. It does the least possible
// before its first timestamp: an UNSERIALIZED read, so story 3 cannot hide
// inside the first number.
//
//   t_raw    plain csrr, first instruction of the kernel
//   t_sync   serialized read straight after   -> t_sync - t_raw is the fence
//   t_arg    after loading one word from the argument blob -> first-access cost
//   t_end    after a second serialized read   -> steady-state fence cost
//
// Every value is a cycle count on THIS core since the launch zeroed MCYCLE, so
// t_raw is the preamble as the device sees it, with nothing of ours in front.

#include <grx/device/grx_device.h>
#include <grx/device/grx_cycles.h>

struct preamble_args {
  uint32_t abi_version;
  uint32_t slots;
  uint64_t out;        // uint64_t[8]: t_raw, t_sync, t_arg, t_end, core, warp, abi, 0
};

__global__ void preamble_probe(preamble_args* __UNIFORM__ arg) {
  // FIRST. No fence, no memory access, nothing before it.
  const uint64_t t_raw = grx::clock64();

  // The serialized read the real probe uses. The difference is what the fence
  // costs at kernel entry, which is the part that could be instrument rather
  // than machine.
  const uint64_t t_sync = grx::cycle_counter();

  // The first memory access any kernel makes: one word out of the argument
  // blob, through a cold dcache.
  const uint32_t abi = arg->abi_version;
  const uint64_t t_arg = grx::clock64();

  // A second serialized read, to price the fence once the pipeline is warm.
  const uint64_t t_end = grx::cycle_counter();

  // EVERY block records its own entry time, so the host can see whether work
  // overlaps dispatch or waits for it. One slot per block: {t_raw, core}.
  if (grx::lane_id() != 0) return;
  if (grx::warp_id() != 0) return;
  if (blockIdx.x >= arg->slots) return;

  uint64_t* o = reinterpret_cast<uint64_t*>(arg->out);
  o[blockIdx.x * 4 + 0] = t_raw;
  o[blockIdx.x * 4 + 1] = t_sync;
  o[blockIdx.x * 4 + 2] = t_arg + (uint64_t)(abi & 0u);
  o[blockIdx.x * 4 + 3] = (uint64_t)vx_core_id() + 1;
}
