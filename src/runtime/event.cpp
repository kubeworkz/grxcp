// GRXCP — events over vx_event timeline counters.
//
// The driver's events are strictly more expressive than CUDA's: each carries a
// monotonically increasing uint64 counter rather than a set-once flag. That is
// exploited here -- one vx_event_h backs every recording of a grxEvent_t, with
// each record advancing the counter to a fresh target -- so re-recording an
// event in a loop allocates nothing.
//
// Elapsed time is the one place this file has to be careful. The command
// processor's profiling writeback is still a skeleton upstream, so
// vx_event_get_profiling refuses. Rather than silently substituting a host
// clock and letting someone publish it as a device measurement, the fallback
// is used AND reported: grxDeviceProp_t.eventTimingIsDeviceSide is 0 while it
// is in effect (cuda_mapping.md section 7.4).

#include "internal.h"

#include <grx/grx_runtime.h>

#include <map>

namespace grxcp {
namespace {

struct EventState {
  vx_event_h event      = nullptr;
  uint64_t   target     = 0;        // counter value the latest record waits for
  int        device     = 0;
  unsigned   flags      = 0;
  bool       recorded   = false;
  uint64_t   host_ns    = 0;        // host timestamp captured at record
  vx_event_h completion = nullptr;  // the record op's own completion event
};

std::mutex g_events_mutex;
std::map<grxEvent_t, EventState*> g_events;

EventState* state_of(grxEvent_t handle) {
  auto it = g_events.find(handle);
  return (it == g_events.end()) ? nullptr : it->second;
}

}  // namespace
}  // namespace grxcp

extern "C" {

grxError_t grxEventCreateWithFlags(grxEvent_t* event, unsigned int flags) {
  if (!event) return grxcp::set_error(grxErrorInvalidValue);
  const int device = grxcp::current_device_index();

  grxcp::Device* d = nullptr;
  grxError_t e = grxcp::acquire_device(device, &d);
  if (e != grxSuccess) return grxcp::set_error(e);

  vx_event_h ev = nullptr;
  vx_result_t r = vx_event_create(d->handle, &ev);
  if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));

  auto* s   = new grxcp::EventState();
  s->event  = ev;
  s->device = device;
  s->flags  = flags;

  auto handle = reinterpret_cast<grxEvent_t>(s);
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    grxcp::g_events[handle] = s;
  }
  *event = handle;
  return grxSuccess;
}

grxError_t grxEventCreate(grxEvent_t* event) {
  return grxEventCreateWithFlags(event, grxEventDefault);
}

grxError_t grxEventDestroy(grxEvent_t event) {
  grxcp::EventState* s = nullptr;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    auto it = grxcp::g_events.find(event);
    if (it == grxcp::g_events.end())
      return grxcp::set_error(grxErrorInvalidResourceHandle);
    s = it->second;
    grxcp::g_events.erase(it);
  }
  if (s->completion) vx_event_release(s->completion);
  vx_event_release(s->event);
  delete s;
  return grxSuccess;
}

grxError_t grxEventRecord(grxEvent_t event, grxStream_t stream) {
  grxcp::EventState* s = nullptr;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    s = grxcp::state_of(event);
    if (!s) return grxcp::set_error(grxErrorInvalidResourceHandle);
  }

  vx_queue_h q = nullptr;
  grxError_t e = grxcp::resolve_stream(stream, s->device, &q, nullptr);
  if (e != grxSuccess) return grxcp::set_error(e);

  std::vector<vx_event_h> waits;
  grxcp::collect_wait_events(stream, s->device, &waits);

  const uint64_t target = s->target + 1;
  vx_event_h completion = nullptr;
  vx_result_t r = vx_enqueue_signal(q, s->event, target,
                                    (uint32_t)waits.size(),
                                    waits.empty() ? nullptr : waits.data(),
                                    &completion);
  if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));

  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    s->target   = target;
    s->recorded = true;
    s->host_ns  = grxcp::host_now_ns();
    if (s->completion) vx_event_release(s->completion);
    s->completion = completion;
    if (completion) vx_event_retain(completion);
  }
  grxcp::set_stream_last_event(stream, s->device, completion);
  return grxSuccess;
}

grxError_t grxEventSynchronize(grxEvent_t event) {
  grxcp::EventState* s = nullptr;
  uint64_t target = 0;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    s = grxcp::state_of(event);
    if (!s) return grxcp::set_error(grxErrorInvalidResourceHandle);
    if (!s->recorded) return grxSuccess;   // never recorded: nothing to wait for
    target = s->target;
  }
  return grxcp::set_error(
      grxcp::map_result(vx_event_wait_value(s->event, target,
                                            VX_TIMEOUT_INFINITE)));
}

grxError_t grxEventQuery(grxEvent_t event) {
  grxcp::EventState* s = nullptr;
  uint64_t target = 0;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    s = grxcp::state_of(event);
    if (!s) return grxcp::set_error(grxErrorInvalidResourceHandle);
    if (!s->recorded) return grxSuccess;
    target = s->target;
  }
  uint64_t value = 0;
  vx_result_t r = vx_event_get_value(s->event, &value);
  if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));
  return (value >= target) ? grxSuccess : grxErrorNotReady;
}

grxError_t grxEventElapsedTime(float* ms, grxEvent_t start, grxEvent_t end) {
  if (!ms) return grxcp::set_error(grxErrorInvalidValue);

  grxcp::EventState *a = nullptr, *b = nullptr;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    a = grxcp::state_of(start);
    b = grxcp::state_of(end);
    if (!a || !b) return grxcp::set_error(grxErrorInvalidResourceHandle);
    if (!a->recorded || !b->recorded) return grxcp::set_error(grxErrorNotReady);
  }
  if ((a->flags & grxEventDisableTiming) || (b->flags & grxEventDisableTiming))
    return grxcp::set_error(grxErrorInvalidResourceHandle);

  // Preferred path: device-side timestamps from the command processor.
  vx_profile_info_t pa{}, pb{};
  if (a->completion && b->completion &&
      vx_event_get_profiling(a->completion, &pa) == VX_SUCCESS &&
      vx_event_get_profiling(b->completion, &pb) == VX_SUCCESS) {
    const double delta_ns = (double)pb.end_ns - (double)pa.end_ns;
    *ms = (float)(delta_ns / 1.0e6);
    return grxSuccess;
  }

  // Fallback: host timestamps captured at record. This measures submission
  // time, not device execution time, and grxDeviceProp_t.eventTimingIsDeviceSide
  // reports 0 so a caller can tell which clock produced the number.
  const double delta_ns = (double)b->host_ns - (double)a->host_ns;
  *ms = (float)(delta_ns / 1.0e6);
  return grxSuccess;
}

grxError_t grxStreamWaitEvent(grxStream_t stream, grxEvent_t event,
                              unsigned int flags) {
  (void)flags;
  grxcp::EventState* s = nullptr;
  uint64_t target = 0;
  {
    std::lock_guard<std::mutex> lock(grxcp::g_events_mutex);
    s = grxcp::state_of(event);
    if (!s) return grxcp::set_error(grxErrorInvalidResourceHandle);
    // Waiting on an unrecorded event is a no-op in CUDA, not an error.
    if (!s->recorded) return grxSuccess;
    target = s->target;
  }

  const int device = grxcp::current_device_index();
  vx_queue_h q = nullptr;
  grxError_t e = grxcp::resolve_stream(stream, device, &q, nullptr);
  if (e != grxSuccess) return grxcp::set_error(e);

  std::vector<vx_event_h> waits;
  grxcp::collect_wait_events(stream, device, &waits);

  vx_event_h completion = nullptr;
  vx_result_t r = vx_enqueue_wait_value(q, s->event, target,
                                        (uint32_t)waits.size(),
                                        waits.empty() ? nullptr : waits.data(),
                                        &completion);
  if (r != VX_SUCCESS) return grxcp::set_error(grxcp::map_result(r));
  grxcp::set_stream_last_event(stream, device, completion);
  return grxSuccess;
}

}  // extern "C"
