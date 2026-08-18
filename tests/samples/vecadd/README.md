# vecadd

The same program in the two forms GRXCP supports, so the difference between
"what the platform will feel like" and "what works today" is visible in one
place.

## Target form — `vecadd.grx.cpp`

```cpp
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
...
vecadd<<<grid, block>>>(da, db, dc, N);
```

Needs `grxcc`, which lands in Phase 4. The driver splits the file, compiles the
device side through VOLT into a `.vxbin`, embeds it as a `.grxfat` section,
emits the registration constructors and the parameter descriptor, and rewrites
`<<<>>>` into `__grxPushCallConfiguration` plus a call to the host stub.

## Today's form — the module path

Until then the same work is expressed explicitly, which is also what a language
runtime or a translator would do:

```cpp
grxModule_t   mod = nullptr;
grxFunction_t fn  = nullptr;
grxModuleLoad(&mod, "vecadd.grxfat");
grxModuleGetFunction(&fn, mod, "vecadd");

struct { uint64_t a, b, c; uint32_t n, pad; } args{
    (uint64_t)da, (uint64_t)db, (uint64_t)dc, (uint32_t)N, 0};

grxLaunchFunction(fn, dim3_t{grid, 1, 1}, dim3_t{block, 1, 1},
                  &args, sizeof(args), /*sharedMem=*/0, /*stream=*/nullptr);
grxDeviceSynchronize();
```

`grxLaunchFunction` takes the argument blob already packed to the kernel's
device layout, so it needs no parameter descriptor. That is deliberate: the
caller knows its own ABI, and inventing one on its behalf is how a runtime
silently corrupts arguments.

The `grxLaunchKernel(func, ..., void** args, ...)` path — the one `<<<>>>`
compiles to — requires a registered parameter layout, because a `void**`
carries no widths. A stub without one is refused rather than guessed at.

## Running it

Both forms need a device that can execute a kernel, which means a built and
installed GRX-G100 sysroot and `$VORTEX_DRIVER` pointing at `simx` or
`rtlsim`. The mock driver in `tests/mock/` records launch descriptors but
computes nothing, so it can prove the descriptor is correct and cannot prove
the arithmetic is.
