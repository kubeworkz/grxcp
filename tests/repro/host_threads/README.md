# Calling grxcp from four host threads: our bookkeeping holds, the runtime under it does not

The GRX930 host went from one RISC-V64 core to four with a directory-coherent
shared L2. Until that landed, "one thread touches the driver" was true by
construction on the target. It is not any more — and **nothing in
`include/grx/` or the design docs says whether more than one thread is
allowed**, while the API this one imitates, the CUDA runtime, is thread-safe
and a porting developer will assume it.

So: four threads, real calls, under ThreadSanitizer. Not a reading of the code.

## Calibrate the instrument first

A clean TSan run means nothing until TSan has been watched firing. Same flags,
same link, a deliberate race on a plain global:

```
races found: 1
SUMMARY: ThreadSanitizer: data race /tmp/tsan_calib.cc:9 in operator()
```

It fires. Now a zero is a result.

## And the sequential control

`sequential_control.cpp` does the same operations on one thread, so a
concurrent failure cannot be blamed on the operation itself:

| | result |
|---|---|
| load the same module twice, both still open | both succeed |
| load / unload / load | both succeed |
| 40 × event create-record-query-destroy | 0 failures |

Everything below is therefore **caused by concurrency**, not by the call.

## Measured, simx, 4 threads

| stage | grxcp's own state | underneath |
|---|---|---|
| **1 allocator** — `grxMalloc`/`grxFree`, 160 ops | **0 races, 0 failures** | — |
| **2 modules** — load / getFunction / unload | 0 races | **19 of 40 loads fail**: `address range overlaps with existing allocation` |
| **3 events** — create / record / query / destroy | 0 races; takes its lock correctly | **5 data races**, all in `libvortex` |
| **4 launches** — per-thread streams | 0 races | **7 data races**, in `libvortex` *and* the simulator's DRAM model |

**No grxcp code owns a racing access.** Every `#0`/`#1` frame in every report
belongs to `libvortex`; grxcp appears at frame #2 or deeper as the *caller*, and
once as the owner of a mutex TSan noted it taking (`resolve_stream`'s
`g_streams_mutex`). That was checked rather than assumed — grxcp's objects are
linked into this binary, so its frames show up as `stress+0x…` and could easily
have been mistaken for the backend's.

### The event race, which is the sharp one

```
Write of size 8 by thread T5:
  #0 pthread_cond_destroy
  #1 vx_event_release
  #2 grxEventDestroy            event.cpp:136

Previous read of size 8 by thread T4:
  #0 pthread_cond_broadcast
  #1 vx::Queue::enqueue_signal(…)::{lambda}

Location is heap block of size 152 allocated by thread T5:
  #1 vx::Event::create(vx::Device*, vx::Event**)
  #2 grxEventCreate             event.cpp:85
```

**A condition variable is destroyed while another thread is inside a broadcast
on it.** POSIX makes that undefined, and it is reachable from an ordinary
sequence: create an event on one thread, let another thread's queue signal it,
destroy it. The other racing sites in the same run are `vx::Event::signal`,
`vx::Event::complete`, `vx::Event::wait_value` and `vx::Queue::worker_loop` —
the event object's whole lifetime, not one call.

### The module failure is a TOCTOU, not a race on memory

TSan reports nothing for stage 2 because nothing races on a *word* — the driver
loses on its own bookkeeping. `vx_module_load_bytes` reserves the image's
address range, and two concurrent loads of the same `.vxbin` both try to reserve
`[0x180000000, 0x180002000]`. Sequentially both succeed, so the reservation is
reference-counted or deduplicated; concurrently the check and the reserve are
not atomic and **19 of 40 loads fail**. `module.cpp`'s device-variable comment
already documented that message from the single-threaded side; this is the same
mechanism arriving from a new direction.

### Launches: two different racing owners, and only one of them ships

Stage 4's seven races split:

* `vx::Queue::worker_loop`, `vx::Queue::finish`, `vx::Event::wait_value` — the
  **runtime library**, which runs on the real host. These matter.
* `vortex::DramSim::Impl`'s request deque — the **simulator's** DRAM model.
  That one cannot exist on hardware and should not be reported as if it could.

## What this means for grxcp

**Our own shared state came through clean**, which is worth stating as plainly
as the failures: the allocator's `g_mem_mutex` covers the whole carve-and-record,
the stream table locks, and `launch.cpp` — which has no mutex at all — was not
implicated, because its per-call state is `thread_local`. That is evidence, not
proof: TSan only sees interleavings that actually happened.

**The contract is undocumented, and today it can only be "one thread".** Not
because of our code but because of what is under it. That is a gap in its own
right: a CUDA-shaped API that is silent about thread safety will be assumed
thread-safe, and the failure mode is a destroyed condvar rather than an error
code.

## Reproducing

```
# build the runtime with TSan
g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude $(pkg-config --cflags vortex-runtime) \
    -c src/runtime/*.cpp                       # into build-tsan/
g++ -fsanitize=thread build-tsan/*.o $(pkg-config --libs vortex-runtime) -o build-tsan/stress

VORTEX_DRIVER=simx TSAN_OPTIONS="halt_on_error=0" ./build-tsan/stress 1   # allocator
                                                  ./build-tsan/stress 2   # modules
                                                  ./build-tsan/stress 3   # events
                                                  ./build-tsan/stress 4   # launches

./build-real/sequential_control                   # the one-thread control
```

Stage 4's report is the weakest of the four: the backend `.so` is not
TSan-instrumented, so TSan sees its accesses but not everything it does. That
weakens a *negative*. The positives it reported there are still positives.
