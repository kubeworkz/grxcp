// GRXCP — streams over vx_queue_h, including legacy default-stream ordering.
//
// A grxStream_t wraps one device queue. The interesting part is not the
// wrapping, it is the ordering contract: CUDA's legacy null stream implicitly
// synchronizes with every blocking stream, and that behavior is load-bearing
// for ported code. It is implemented here by threading each stream's last
// completion event into the wait list of the next operation on the streams it
// must order against.
//
// Note on concurrency, so nobody reads more into this file than it delivers:
// the ordering is correct, but the hardware does not yet run streams
// concurrently. The command processor defaults to a single queue and the
// driver serializes launches; real overlap needs the upstream QMD-style atomic
// launch (cuda_mapping.md section 7.3). Programs written against these
// semantics get faster when that lands, without source changes.

#include "internal.h"

#include <grx/grx_runtime.h>

#include <algorithm>
#include <chrono>
#include <map>

namespace grxcp {

namespace {

struct StreamState {
  vx_queue_h  queue      = nullptr;
  int         device     = 0;
  unsigned    flags      = 0;
  bool        is_null    = false;
  vx_event_h  last_event = nullptr;   // completion of the most recent enqueue
};

std::mutex g_streams_mutex;
std::map<grxStream_t, StreamState*> g_streams;      // user handle -> state
std::map<int, StreamState*>         g_null_streams; // device -> null stream

grxError_t make_queue(int device, vx_queue_priority_e priority,
                      vx_queue_h* out) {
  Device* d = nullptr;
  grxError_t e = acquire_device(device, &d);
  if (e != grxSuccess) return e;

  vx_queue_info_t info{};
  info.struct_size = sizeof(info);
  info.next        = nullptr;
  info.priority    = priority;
  // Profiling is requested unconditionally so grxEventElapsedTime can use
  // device timestamps the moment the command processor starts producing them.
  info.flags       = VX_QUEUE_PROFILING_ENABLE;
  return map_result(vx_queue_create(d->handle, &info, out));
}

// The null stream is created on first use rather than at device open, so a
// program that never touches the default stream never pays for a queue.
StreamState* null_stream_locked(int device) {
  auto it = g_null_streams.find(device);
  if (it != g_null_streams.end()) return it->second;

  vx_queue_h q = nullptr;
  if (make_queue(device, VX_QUEUE_PRIORITY_NORMAL, &q) != grxSuccess)
    return nullptr;

  auto* s = new StreamState();
  s->queue   = q;
  s->device  = device;
  s->is_null = true;
  g_null_streams[device] = s;
  return s;
}

StreamState* state_of(grxStream_t stream, int device) {
  if (stream == nullptr) return null_stream_locked(device);
  auto it = g_streams.find(stream);
  return (it == g_streams.end()) ? nullptr : it->second;
}

void replace_last_event(StreamState* s, vx_event_h ev) {
  if (s->last_event) vx_event_release(s->last_event);
  s->last_event = ev;
}

}  // namespace

uint64_t host_now_ns() {
  using clock = std::chrono::steady_clock;
  return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch())
      .count();
}

grxError_t resolve_stream(grxStream_t stream, int device, vx_queue_h* out_queue,
                          grxStream_t* out_stream) {
  std::lock_guard<std::mutex> lock(g_streams_mutex);
  StreamState* s = state_of(stream, device);
  if (!s) return grxErrorInvalidResourceHandle;
  if (out_queue)  *out_queue  = s->queue;
  if (out_stream) *out_stream = stream;
  return grxSuccess;
}

void collect_wait_events(grxStream_t stream, int device,
                         std::vector<vx_event_h>* out) {
  std::lock_guard<std::mutex> lock(g_streams_mutex);
  StreamState* self = state_of(stream, device);
  if (!self) return;

  if (self->is_null) {
    // Null-stream work waits for every blocking stream on this device.
    for (auto& kv : g_streams) {
      StreamState* s = kv.second;
      if (s->device != device) continue;
      if (s->flags & grxStreamNonBlocking) continue;
      if (s->last_event) out->push_back(s->last_event);
    }
  } else if (!(self->flags & grxStreamNonBlocking)) {
    // Blocking-stream work waits for the null stream.
    auto it = g_null_streams.find(device);
    if (it != g_null_streams.end() && it->second->last_event)
      out->push_back(it->second->last_event);
  }

  // A stream is always ordered against itself.
  if (self->last_event &&
      std::find(out->begin(), out->end(), self->last_event) == out->end())
    out->push_back(self->last_event);
}

void set_stream_last_event(grxStream_t stream, int device, vx_event_h event) {
  std::lock_guard<std::mutex> lock(g_streams_mutex);
  StreamState* s = state_of(stream, device);
  if (!s) { if (event) vx_event_release(event); return; }
  replace_last_event(s, event);
}

grxError_t sync_stream(grxStream_t stream, int device) {
  vx_queue_h q = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_streams_mutex);
    StreamState* s = state_of(stream, device);
    if (!s) return grxErrorInvalidResourceHandle;
    q = s->queue;
  }
  return map_result(vx_queue_finish(q, VX_TIMEOUT_INFINITE));
}

grxError_t sync_all_streams(int device) {
  std::vector<vx_queue_h> queues;
  {
    std::lock_guard<std::mutex> lock(g_streams_mutex);
    auto it = g_null_streams.find(device);
    if (it != g_null_streams.end()) queues.push_back(it->second->queue);
    for (auto& kv : g_streams)
      if (kv.second->device == device) queues.push_back(kv.second->queue);
  }
  for (vx_queue_h q : queues) {
    vx_result_t r = vx_queue_finish(q, VX_TIMEOUT_INFINITE);
    if (r != VX_SUCCESS) return map_result(r);
  }
  return grxSuccess;
}

// --- helpers used only by this file's public entry points ------------------

namespace {

// Stream handles are opaque to callers. Using the state pointer itself as the
// handle keeps lookup O(log n) and makes a stale handle detectable rather than
// a wild pointer dereference.
grxStream_t register_stream(vx_queue_h q, int device, unsigned flags) {
  auto* s = new StreamState();
  s->queue  = q;
  s->device = device;
  s->flags  = flags;
  auto handle = reinterpret_cast<grxStream_t>(s);
  std::lock_guard<std::mutex> lock(g_streams_mutex);
  g_streams[handle] = s;
  return handle;
}

grxError_t destroy_stream(grxStream_t stream) {
  StreamState* s = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_streams_mutex);
    auto it = g_streams.find(stream);
    if (it == g_streams.end()) return grxErrorInvalidResourceHandle;
    s = it->second;
    g_streams.erase(it);
  }
  // CUDA's destroy is asynchronous: outstanding work completes first.
  vx_queue_finish(s->queue, VX_TIMEOUT_INFINITE);
  if (s->last_event) vx_event_release(s->last_event);
  vx_queue_release(s->queue);
  delete s;
  return grxSuccess;
}

// Non-blocking completion check. With nothing ever enqueued the stream is
// trivially idle; otherwise the last operation's completion event answers it.
grxError_t query_stream(grxStream_t stream, int device) {
  vx_event_h last = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_streams_mutex);
    StreamState* s = state_of(stream, device);
    if (!s) return grxErrorInvalidResourceHandle;
    last = s->last_event;
    if (!last) return grxSuccess;
    vx_event_retain(last);
  }
  uint64_t value = 0;
  vx_result_t r = vx_event_get_value(last, &value);
  vx_event_release(last);
  if (r != VX_SUCCESS) return map_result(r);
  return (value >= 1) ? grxSuccess : grxErrorNotReady;
}

}  // namespace

}  // namespace grxcp

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

extern "C" {

grxError_t grxStreamCreateWithPriority(grxStream_t* stream, unsigned int flags,
                                       int priority) {
  if (!stream) return grxcp::set_error(grxErrorInvalidValue);
  const int device = grxcp::current_device_index();

  // CUDA's convention: numerically lower means higher priority.
  vx_queue_priority_e prio = (priority < 0) ? VX_QUEUE_PRIORITY_HIGH
                           : (priority > 0) ? VX_QUEUE_PRIORITY_LOW
                                            : VX_QUEUE_PRIORITY_NORMAL;

  vx_queue_h q = nullptr;
  grxError_t e = grxcp::make_queue(device, prio, &q);
  if (e != grxSuccess) return grxcp::set_error(e);

  *stream = grxcp::register_stream(q, device, flags);
  return grxSuccess;
}

grxError_t grxStreamCreateWithFlags(grxStream_t* stream, unsigned int flags) {
  return grxStreamCreateWithPriority(stream, flags, 0);
}

grxError_t grxStreamCreate(grxStream_t* stream) {
  return grxStreamCreateWithPriority(stream, grxStreamDefault, 0);
}

grxError_t grxStreamDestroy(grxStream_t stream) {
  if (!stream) return grxcp::set_error(grxErrorInvalidResourceHandle);
  grxError_t e = grxcp::destroy_stream(stream);
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxStreamSynchronize(grxStream_t stream) {
  grxError_t e = grxcp::sync_stream(stream, grxcp::current_device_index());
  return (e == grxSuccess) ? e : grxcp::set_error(e);
}

grxError_t grxStreamQuery(grxStream_t stream) {
  grxError_t e = grxcp::query_stream(stream, grxcp::current_device_index());
  return (e == grxSuccess || e == grxErrorNotReady) ? e : grxcp::set_error(e);
}

}  // extern "C"
