// Every GRXCP library's device kernels, in ONE module.
//
// Not a convenience: a requirement, and one that grew. grxBLAS's
// kernels/all.cpp already existed because grxblasSgemm and grxblasGemmEx are
// two entry points a single program calls, and each .vxbin links at the same
// fixed load address (STARTUP_ADDR) -- so a program can have exactly one module
// resident (cuda_mapping.md 7.13).
//
// grxDNN makes that bite across LIBRARIES rather than within one. The phase 6
// exit gate is a transformer: attention and GEMM from grxBLAS, layer norm and
// softmax from grxDNN, in one process. Two library-specific modules cannot both
// be resident, so the second library to initialise would fail to load its
// kernels and report something about a missing file.
//
// So the libraries share an image. Each still prefers this one and falls back
// to its own, which keeps a grxBLAS-only program from shipping grxDNN's kernels
// and keeps a device without a tensor unit on the scalar path.
//
// THIS FILE IS HALF THE FIX, AND THE SMALLER HALF.
//
// The first version of this comment stopped at the paragraph above and said the
// shared image was the answer. It is not, and the gate that proved it is
// tests/libs/test_libs_together.cpp: with this exact file built and both
// libraries pointed at it, grxDNN still failed to load, because grxBLAS had
// already called grxModuleLoad on it and grxDNN's own call asked the driver to
// reserve a range grxBLAS was holding. The same bytes collide with themselves.
//
//   Error: address range overlaps with existing allocation -
//     requested=[0x180000000-0x180002000], existing=[0x180000000, 0x180002000]
//
// The other half is in src/runtime/module.cpp: an image already resident on the
// device, byte for byte, is handed back to the second caller with a reference
// count rather than loaded again. Both halves are needed and neither is
// sufficient -- two DIFFERENT images still collide no matter how the runtime
// counts, which is what this file prevents.
//
// When device images become relocatable, this file can go away. The reference
// counting in the runtime should not: it is what cuModuleLoad does with a file
// already loaded in the context, and it stays correct when the addresses stop
// colliding.

#include "grxblas/kernels/all.cpp"   // NOLINT(bugprone-suspicious-include)
#include "grxdnn/kernels/all.cpp"    // NOLINT(bugprone-suspicious-include)
