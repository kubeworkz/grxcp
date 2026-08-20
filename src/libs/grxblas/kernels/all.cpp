// Every grxBLAS device kernel, in one module.
//
// Not a convenience: a requirement. Each .vxbin is linked at the same fixed
// load address (STARTUP_ADDR), so loading a second one fails with an address
// overlap -- a program can have exactly one module resident unless someone
// hand-assigns link addresses. A library that ships its kernels as separate
// modules therefore cannot offer two of them at once, which is how this file
// came to exist: grxblasSgemm and grxblasGemmEx in the same program.
//
// Registered as a platform gap in docs/designs/cuda_mapping.md. When device
// images become relocatable this file can go away; until then, one module.
//
// This module requires a tensor unit and a DMA engine, because hgemm_tcu does.
// On a device with neither, build kernels/sgemm.cpp on its own -- the library
// falls back to it and reports the tensor path as unavailable.

#include "sgemm.cpp"       // NOLINT(bugprone-suspicious-include)
#include "blas12.cpp"      // NOLINT(bugprone-suspicious-include)
#include "hgemm_tcu.cpp"   // NOLINT(bugprone-suspicious-include)
