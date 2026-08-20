// GRXCP — the tiny support surface grxcc's generated host code calls into.
//
// Everything here exists because a SOURCE REWRITER cannot see types. grxcc
// turns `kernel<<<g, b>>>(args)` into a call, and at that point `g` might be an
// `int`, a `dim3`, a `dim3_t`, or an expression whose type only the compiler
// knows. Resolving that in the rewriter would mean writing a C++ type checker;
// resolving it here costs three overloads and hands the job to the compiler
// that already has the answer.
//
// This header is included by generated code, not written by hand. It is
// installed because the generated code is compiled in the user's build, not in
// grxcc's.

#ifndef GRX_LAUNCH_SHIM_H
#define GRX_LAUNCH_SHIM_H

#include <grx/grx_types.h>

#ifdef __cplusplus

namespace grx {
namespace shim {

// A launch dimension, however the caller spelled it. CUDA's <<<n, m>>> takes
// ints and its <<<dim3(a,b), dim3(c,d)>>> takes dim3s, and both have to reach
// the same POD the runtime entry point wants.
inline dim3_t as_dim(unsigned int n)       { return dim3_t{n, 1u, 1u}; }
inline dim3_t as_dim(int n)                { return dim3_t{(unsigned)n, 1u, 1u}; }
inline dim3_t as_dim(unsigned long n)      { return dim3_t{(unsigned)n, 1u, 1u}; }
inline dim3_t as_dim(long n)               { return dim3_t{(unsigned)n, 1u, 1u}; }
inline dim3_t as_dim(const dim3_t& d)      { return d; }
inline dim3_t as_dim(const dim3& d)        { return dim3_t{d.x, d.y, d.z}; }

// The stream argument of <<<g, b, s, st>>>. Absent means the null stream, and
// a caller writing 0 means the same thing -- which is a literal that would
// otherwise be ambiguous between a pointer and an integer.
inline grxStream_t as_stream(grxStream_t s) { return s; }
inline grxStream_t as_stream(int)           { return (grxStream_t)0; }
inline grxStream_t as_stream(long)          { return (grxStream_t)0; }

}  // namespace shim
}  // namespace grx

#endif  // __cplusplus
#endif  // GRX_LAUNCH_SHIM_H
