// CUDA spells it <cooperative_groups.h>. GRXCP spells it
// <grx/device/grx_cg.h>. This header exists so a CUDA source file does not
// have to know the difference.
//
// It is a forwarding header, not a shim: grx_cg.h already closes with
// `namespace cooperative_groups = grx::cg;`, so `cg::this_thread_block()`
// resolves to the real implementation and its documented differences from CUDA
// (tile width, grid-barrier scope, map_shared_rank) apply unchanged. Nothing is
// papered over here.
//
// DEVICE PASS ONLY. grxcc compiles the file twice, and a .cu file's
// `#include <cooperative_groups.h>` is seen by both halves -- but grx_cg.h
// bottoms out in vx_spawn2.h and the CTA CSRs, which do not exist on the host.
// CUDA solves this by guarding the device-only parts with __CUDA_ARCH__; this
// is the same fence with grxcc's spelling.
//
// The host half still gets the NAMESPACE, because a file writes
// `namespace cg = cooperative_groups;` at file scope and that line is in both
// passes. An empty namespace is enough to make the alias legal, and any actual
// use of it is inside a kernel body, which the host pass replaces with a launch
// stub.
#ifndef GRX_COOPERATIVE_GROUPS_FORWARD_H
#define GRX_COOPERATIVE_GROUPS_FORWARD_H

#if defined(__GRX_DEVICE_PASS__) || !defined(__GRX_HOST_PASS__)
#include <grx/device/grx_cg.h>
#else
namespace cooperative_groups {}
#endif

#endif
