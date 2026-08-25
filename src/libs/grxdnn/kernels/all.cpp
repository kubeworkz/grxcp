// Every grxDNN device kernel, in one translation unit.
//
// Same reason grxBLAS has one: each .vxbin links at STARTUP_ADDR, so the device
// holds one module at a time (cuda_mapping.md 7.13), and grxdnnAttentionForward
// needs the softmax kernel and the mask kernel in the same image.
//
// src/libs/kernels_all.cpp includes THIS file rather than the individual
// kernels, so a new grxDNN kernel is added here once and reaches both the
// grxDNN-only image and the combined one.

#include "norm.cpp"          // NOLINT(bugprone-suspicious-include)
#include "attn.cpp"          // NOLINT(bugprone-suspicious-include)
#include "elementwise.cpp"   // NOLINT(bugprone-suspicious-include)
