# `grxcc` — the single-source driver

`grxcc` compiles a file that contains both halves of a GPU program:

```cpp
__global__ void axpy(const float* x, float* y, unsigned n, float a) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  ...
  axpy<<<grid, block>>>(dx, dy, n, 3.0f);
  grxDeviceSynchronize();
}
```

No module load, no `.vxbin` named anywhere, no `grxLaunchKernel`. That is the
whole claim, and `tests/grxcc/vecadd.grx.cpp` is the gate that holds it.

---

## 1. Why an orchestrator and not a frontend

The architecture document's open question 1 asked whether `grxcc` should be a
real Clang `ToolChain` — a frontend that parses the file once, with
`__global__` a genuine attribute and `<<<>>>` genuine syntax — or an
orchestrator that rewrites text and shells out to two ordinary compilers.

v1 is the orchestrator, for one reason that outweighs the rest: **it can be
wrong in visible ways.** A frontend that mis-splits a file produces a confusing
diagnostic from inside a compiler nobody in this project maintains. An
orchestrator writes both generated sources to disk (`--keep`, `--emit-only`),
and every failure is a compiler error with a line number in a file you can open
and read. When `grxcc` gets a rewrite wrong, the evidence is a text file.

The cost is real and is not hidden: `grxcc` does not have a type system. It
finds `__global__` by text and matches brackets. Section 4 lists exactly what
that costs.

Promoting it to a Clang `ToolChain` is Phase 6 scope.

---

## 2. The two passes

`grxcc` reads the source once, rewrites every `<<<>>>` launch, and then builds
two files from the result.

### The device pass

```
#define __GRX_DEVICE_PASS__ 1
<everything up to the last file-scope boundary before the first __global__>
#include <grx/device/grx_device.h>
<from there to the last kernel's closing brace, kernels rewritten>
<closing braces for anything that span left open>
                                    <-- and nothing after the last kernel
```

Two placement decisions here are load-bearing.

**The pass stops at the last kernel.** A frontend compiles the whole file for
the device and simply never diagnoses host-only function bodies it does not
need. An orchestrator has no such option: it hands clang the text, and `main`'s
`std::printf` is an error for a bare-metal `riscv64` target whether or not the
device would ever run it. So the device pass keeps the file up to the end of the
last `__global__` and drops the rest. This makes *device code first, host code
after* a rule of `grxcc` rather than a convention — which it already is in most
CUDA files — and it fails visibly, as an undeclared name at the device compile,
if a kernel needed something declared below it.

**The device header goes at the last file-scope boundary before the first
kernel** — later than the top of the file, earlier than the kernel itself. Two
constraints have to hold at once. It must come *after* the user's includes,
because `grx_device.h` defines `printf` and `assert` as macros — which is what
makes a kernel able to call them — and those macros poison `<cstdio>` if the
standard header is parsed afterwards:

```
cstdio:127:11: error: no member named 'printf' in the global namespace
```

And it must be at *file scope*, which "immediately before the first kernel" is
not the moment a kernel lives inside `namespace ns {` — putting an `#include`
there renames everything in the header. The last top-level boundary before the
kernel is the only point that satisfies both.

The same nesting is why the device pass appends closing braces: cutting at the
last kernel's `}` can leave a namespace or an `extern "C" {` open, and the
resulting error names end-of-file rather than anything the author wrote.

### The host pass

```
#include <grx/grx.h>
#include <grx/grx_abi.h>
#include <grx/grx_launch_shim.h>
namespace grx { namespace shim { inline grxError_t push(...); }}
#define __GRX_HOST_PASS__ 1
<the whole file, with each kernel BODY replaced by a launch stub>
<the .vxbin as a fat-binary byte array>
<a registration constructor>
```

Each kernel becomes a host function with the same signature that packs its
arguments into a struct and calls `__grxLaunchKernelDesc`. The registration
constructor keys each stub's **address** to its kernel descriptor, which is what
makes `axpy<<<...>>>(...)` resolvable at run time without the program ever
naming a string.

---

## 3. `grx_launch_shim.h`, or: a rewriter cannot see types

`kernel<<<g, b>>>(args)` becomes

```cpp
(::grx::shim::push(::grx::shim::as_dim(g), ::grx::shim::as_dim(b),
                   (size_t)0, (grxStream_t)0), kernel(args))
```

`g` might be an `int`, an `unsigned`, a `dim3`, a `dim3_t`, or an expression
whose type only the compiler knows. Resolving that in the rewriter would mean
writing a C++ type checker. Resolving it in `as_dim` costs six overloads and
hands the job to the compiler that already has the answer. `as_stream` exists
for the same reason, plus one of its own: a caller writing `0` for the stream
means the null stream, and a bare `0` is a literal that would otherwise be
ambiguous between a pointer and an integer.

The comma expression is deliberate. It keeps the launch a single expression, so
it survives everywhere a launch can legally appear — inside a loop body without
braces, inside a ternary, as the operand of a cast.

---

## 4. What the text-level parser costs

These are the honest limits of a rewriter without a type system. Each one is
either diagnosed or is a documented restriction; none of them is silent.

**A `<<<>>>` must name a `__global__` defined in this file.** `grxcc` builds the
host stub from the kernel's body, so a declaration alone is not enough — unlike
`nvcc`, which can launch a kernel defined in another translation unit because
its fat binary is linked rather than generated per file. `grxcc` diagnoses this
by name and lists the kernels the file does define, because the alternative is a
wall of template errors out of the device compile.

**Template kernels are not supported.** `template <typename T> __global__ void
k(...)` has no single argument struct to emit and no single stub address to
register. The template head sits *before* `__global__`, so detecting it means
walking back to the end of the previous declaration and looking for the keyword
— which also catches `template <int N> static __global__ ...`, where the two are
not adjacent.

**Dynamic parallelism is not supported.** A `<<<>>>` inside a kernel body is
rewritten like any other, and then fails the device compile — where the error
names `::grx::shim::push`, which does not exist on the device.

**`__global__` must be at file or namespace scope.** Namespaces are tracked:
`ns::kernel<<<...>>>` resolves, the generated stub and args struct are emitted
in place inside the namespace, and the registration tables — which sit at file
scope — qualify every name they mention. Anonymous namespaces contribute no
qualification, which is correct, since their members are reachable unqualified
from file scope. A `__global__` inside a class or a function body is diagnosed:
a host stub has to be a free function whose *address* keys the registry.

**Two kernels may not share an unqualified name**, even in different namespaces.
The device entry point is `__vx_kentry_<name>`, derived from the `__global__`
alone, so `a::run` and `b::run` are one symbol on the device however distinct
they are on the host. C++ would accept the file and the device would run
whichever one the linker kept, which is why this is an error and not a warning.

**The lexer understands comments, string and character literals, and raw string
literals**, so `<<<` inside a string is not a launch and `__global__` inside a
comment is not a kernel. It does not understand `#if`, so a kernel inside a
disabled preprocessor branch is still seen. That is conservative in the safe
direction: the kernel is compiled and registered, and nothing launches it.

---

## 5. Attributes and metadata

### `__launch_bounds__`

```cpp
__global__ void __launch_bounds__(256, 2) k(float* p) { ... }
```

The attribute brings its own parentheses, and taking the first `(` after
`__global__` as the parameter list reads `__launch_bounds__` as the kernel's
name and `256, 2` as its parameters. That is not hypothetical — it is what
`grxcc` did until the attribute loop existed, and the only reason it was not
silent is that the launch-site check then failed to find a kernel called `k`.
So the parser walks the parenthesised groups, consuming any whose preceding
identifier is an attribute (`__launch_bounds__`, `__attribute__`, `alignas`)
and keeping the rest as the declaration head.

The maximum is carried as the **source expression the author wrote**, not a
parsed integer. `__launch_bounds__(kBlock * 2)` is legal CUDA; `grxcc` has no
constant evaluator and the host compiler does, so the expression is emitted into
the descriptor as `(uint32_t)(kBlock * 2)`. Same principle as `as_dim`.

Neither pass ever sees the attribute: both emitters generate a fresh signature,
so it disappears without anyone having to strip it.

The second argument — `minBlocksPerMultiprocessor` — has nothing to do here, and
`grxcc` says so rather than accepting it silently. `cuda_mapping.md` section
7.21 has the reasoning.

### `numRegs`

After the device compile, `grxcc` disassembles the ELF `build_kernel.sh` left
beside the `.vxbin`, walks each kernel's directly-reachable call graph, and
counts distinct architectural registers — integer and floating-point together,
`x0` excluded. An indirect call or an unresolvable target reports **-1** rather
than a lower bound.

Two details worth knowing before touching this:

- **The entry point has two names.** `build_kernel.sh` emits both
  `__vx_kentry_<name>` and `<name>` at the same address, and `llvm-objdump`
  labels the block with whichever alias it picks — it prints `<axpy>` for a
  symbol `llvm-nm` lists under both. Looking for one spelling found nothing and
  reported -1 for every kernel, which is indistinguishable from the honest
  "cannot measure this" answer. That is why the gate compares two kernels
  against each other rather than just checking for a number.
- **Failure is not fatal.** No `llvm-objdump`, or a disassembly that will not
  parse, leaves every kernel at -1 and the program still builds. Refusing to
  compile because an optional statistic could not be gathered would trade a
  working program for a number nobody asked for.

---

## 6. Sharing one device-build recipe

The device pass is not compiled by `grxcc` directly. It shells out to
`ci/build_kernel.sh` — the same script the grxBLAS kernels and every kernel gate
use — found via `--build-kernel`, `$GRXCC_BUILD_KERNEL`, or a search relative to
the binary.

This is a deliberate constraint rather than a convenience. A kernel compiled by
`grxcc` and one compiled by hand must come from one set of flags and one
recorded device configuration, or "it works when I build it myself" becomes a
real bug report that costs a day. The cost is that `grxcc` has to be told where
the script is.

---

## 7. Two failures that shaped the design

Both were found by the first program `grxcc` ever compiled, and both are the
kind that a smaller sample would not have caught.

**Static initialization order.** `__grxRegisterFatBinary` runs from a static
initializer in the *user's* translation unit. The runtime's registry was a
file-scope `std::map` in `src/runtime/module.cpp`, and the order those two
initialize in is the linker's choice. With the generated object first on the
link line, the registrar ran before the map was constructed and the program
died in `_Rb_tree_decrement` before printing anything. The registry, the module
and function tables and the fat-binary list are now function-local statics: the
first registration constructs the map it is about to insert into, and because
construction completes *inside* the registrar's constructor, destruction happens
after it — which fixes the mirror-image problem at exit too.

**`__syncthreads()` duplicated across divergence.** The sample's third kernel
reverses a block through shared memory, over a grid whose last block is partial.
It hung. The cause was not in `grxcc` at all: upstream's `vx_barrier` is a
`volatile` inline asm with no `convergent` or `noduplicate`, so LLVM
tail-duplicated it into both arms of the preceding divergent branch and a
diverged warp arrived at the barrier twice. Every kernel gate before phase 4
launched one warp per CTA over an evenly-dividing grid, which is precisely the
shape that cannot expose it. `cuda_mapping.md` section 7.20 has the disassembly,
the attribute ablation, and the workaround.

The general lesson is in the sample's own header comment: it is deliberately not
minimal. A single kernel with one pointer argument would not have exercised
argument packing, a second kernel, shared memory, a stream, a launch in a loop,
or a ragged grid — and four of those six were needed to find these two bugs.

---

## 8. Using it

```
grxcc [options] <source> [objects...] -o <output>

  --grxgpu <path>    grxgpu checkout, for the device compile
  --tooldir <path>   device toolchain (default $TOOLDIR or $HOME/tools)
  --build-kernel <p> ci/build_kernel.sh to shell out to
  -I <dir>           include directory, passed to both passes
  -L <dir>, -l<lib>  passed to the host link
  <file>.o|.a|.so    passed to the host link
  -c                 compile the host pass only, do not link
  --keep             leave the generated sources in place
  --emit-only        write the generated sources and stop
  -v                 print every command
```

A file with no `__global__` at all is compiled as an ordinary host translation
unit and needs no `--grxgpu` — most translation units of a real program have no
kernels, and demanding a device toolchain to compile them would be wrong.

`--emit-only` is the first thing to reach for when a generated program
misbehaves: it writes `<output>.grxcc.dev.cpp` and `<output>.grxcc.host.cpp` and
stops, and the answer is usually visible in one of them.
