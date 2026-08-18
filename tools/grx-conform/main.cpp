// grx-conform — publish what GRXCP does and does not implement.
//
// The AGENTS.md honesty rules say to publish the conformance number and track
// it as a trend rather than hiding it. This tool is that number. It reports
// three things:
//
//   1. Coverage across the tracked CUDA Runtime surface, by category.
//   2. Behavioural verification that every entry point marked UNSUPPORTED
//      actually refuses at runtime, rather than being marked that way in a
//      table and doing something else in the code.
//   3. What is still blocked on hardware, named explicitly.
//
// It deliberately does NOT claim a kernel-execution pass rate. Nothing here
// runs a kernel; that needs a real backend. Reporting an API coverage number as
// if it were a conformance pass rate would be exactly the kind of flattering
// measurement the rules forbid.
//
//   grx-conform            human-readable report
//   grx-conform --markdown emit docs/conformance.md content on stdout

#include <grx/grx.h>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

enum Status { MAPPED, PARTIAL, UNSUPPORTED, ABSENT };

struct Entry {
  const char* cuda_name;
  const char* grx_name;
  Status      status;
  const char* category;
  const char* note;
};

const Entry kEntries[] = {
#define GRX_CUDA_API(cuda, grx, status, category, note) \
  {#cuda, #grx, status, category, note},
#include "../common/cuda_api_table.inc"
#undef GRX_CUDA_API
};

constexpr size_t kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);

// Implemented, from a caller's point of view: the call exists and does the
// work. PARTIAL counts, because a documented behavioural caveat is still a
// working entry point; the caveat is printed alongside so it is not lost.
bool implemented(Status s) { return s == MAPPED || s == PARTIAL; }

struct CategoryStats {
  int mapped = 0, partial = 0, unsupported = 0, absent = 0;
  int total() const { return mapped + partial + unsupported + absent; }
  int implemented_count() const { return mapped + partial; }
};

// ---------------------------------------------------------------------------
// Behavioural verification
//
// A table entry claiming an entry point refuses is worth nothing unless the
// entry point actually refuses. These calls are hand-written because the table
// carries no signatures -- if an UNSUPPORTED entry is added without a check
// here, the count mismatch below reports it.
// ---------------------------------------------------------------------------

struct Check {
  const char* name;
  bool        passed;
  const char* detail;
};

std::vector<Check> run_refusal_checks() {
  std::vector<Check> checks;

  auto refuses = [](grxError_t e) { return e == grxErrorNotSupported; };

  {
    int value = 0;
    const grxError_t e = grxDeviceGetAttribute(&value, 0, 0);
    checks.push_back({"grxDeviceGetAttribute", refuses(e),
                      grxGetErrorName(e)});
  }
  {
    const grxError_t e = grxDeviceEnablePeerAccess(0, 0);
    checks.push_back({"grxDeviceEnablePeerAccess", refuses(e),
                      grxGetErrorName(e)});
  }
  {
    static char buffer[64];
    const grxError_t e = grxHostRegister(buffer, sizeof(buffer), 0);
    checks.push_back({"grxHostRegister", refuses(e), grxGetErrorName(e)});
  }
  {
    static char buffer[64];
    const grxError_t e = grxHostUnregister(buffer);
    checks.push_back({"grxHostUnregister", refuses(e), grxGetErrorName(e)});
  }

  // Clear the sticky errors these deliberate failures left behind, so a
  // program running grx-conform inline does not inherit them.
  grxGetLastError();
  return checks;
}

// Device-side facts worth publishing next to the coverage number, because they
// change what a ported program will actually experience.
struct DeviceFacts {
  bool        have_device = false;
  std::string name;
  bool        warp_shuffle_emulated = false;
  bool        event_timing_host     = false;
  bool        constant_is_global    = false;
  bool        managed_memory        = false;
};

DeviceFacts probe_device() {
  DeviceFacts f;
  int count = 0;
  if (grxGetDeviceCount(&count) != grxSuccess || count <= 0) return f;
  grxDeviceProp_t p{};
  if (grxGetDeviceProperties(&p, 0) != grxSuccess) return f;
  f.have_device           = true;
  f.name                  = p.name;
  f.warp_shuffle_emulated = p.warpShuffleIsEmulated != 0;
  f.event_timing_host     = p.eventTimingIsDeviceSide == 0;
  f.constant_is_global    = p.constantMemoryIsGlobal != 0;
  f.managed_memory        = p.managedMemory != 0;
  return f;
}

std::map<std::string, CategoryStats> tally() {
  std::map<std::string, CategoryStats> by_category;
  for (size_t i = 0; i < kEntryCount; ++i) {
    CategoryStats& s = by_category[kEntries[i].category];
    switch (kEntries[i].status) {
      case MAPPED:      ++s.mapped; break;
      case PARTIAL:     ++s.partial; break;
      case UNSUPPORTED: ++s.unsupported; break;
      case ABSENT:      ++s.absent; break;
    }
  }
  return by_category;
}

int count_status(Status want) {
  int n = 0;
  for (size_t i = 0; i < kEntryCount; ++i)
    if (kEntries[i].status == want) ++n;
  return n;
}

void print_human(const std::vector<Check>& checks, const DeviceFacts& facts) {
  const auto by_category = tally();
  int implemented_total = 0;
  for (size_t i = 0; i < kEntryCount; ++i)
    if (implemented(kEntries[i].status)) ++implemented_total;

  std::printf("GRXCP conformance against the tracked CUDA Runtime surface\n\n");
  std::printf("  %d of %zu entry points implemented (%.0f%%)\n",
              implemented_total, kEntryCount,
              100.0 * implemented_total / (double)kEntryCount);
  std::printf("    mapped  %d    partial %d    refused %d    absent %d\n\n",
              count_status(MAPPED), count_status(PARTIAL),
              count_status(UNSUPPORTED), count_status(ABSENT));

  std::printf("  %-10s %6s %8s %8s %8s %8s\n", "category", "total", "mapped",
              "partial", "refused", "absent");
  for (const auto& kv : by_category) {
    const CategoryStats& s = kv.second;
    std::printf("  %-10s %6d %8d %8d %8d %8d\n", kv.first.c_str(), s.total(),
                s.mapped, s.partial, s.unsupported, s.absent);
  }

  std::printf("\nRefusal checks (an entry marked refused must actually refuse)\n");
  int failed = 0;
  for (const Check& c : checks) {
    std::printf("  %-28s %s", c.name, c.passed ? "ok" : "FAIL");
    if (!c.passed) { std::printf("  (returned %s)", c.detail); ++failed; }
    std::printf("\n");
  }
  const int expected = count_status(UNSUPPORTED);
  if ((int)checks.size() != expected) {
    std::printf("  %-28s FAIL  (%zu checks for %d refused entries)\n",
                "check coverage", checks.size(), expected);
    ++failed;
  }

  std::printf("\nBehaviour a port will notice\n");
  if (!facts.have_device) {
    std::printf("  no device reachable; device-side facts not probed\n");
  } else {
    std::printf("  device                 %s\n", facts.name.c_str());
    std::printf("  warp shuffle           %s\n",
                facts.warp_shuffle_emulated
                    ? "EMULATED through local memory (no WSHFL instruction)"
                    : "native");
    std::printf("  event elapsed time     %s\n",
                facts.event_timing_host ? "HOST CLOCK (no device timestamps yet)"
                                        : "device-side");
    std::printf("  __constant__           %s\n",
                facts.constant_is_global ? "read-only global (no broadcast path)"
                                         : "constant cache");
    std::printf("  managed memory         %s\n",
                facts.managed_memory ? "available" : "refused on this backend");
  }

  std::printf("\nNot measured here\n");
  std::printf("  Kernel execution. Nothing in THIS REPORT runs a kernel, so\n");
  std::printf("  none of the numbers above say whether a kernel computes the\n");
  std::printf("  right answer. API coverage is not a conformance pass rate and\n");
  std::printf("  is not reported as one. Kernel execution is covered separately\n");
  std::printf("  by the phase 1 gate in ci/run_real.sh.\n");

  if (failed) std::printf("\n%d refusal check(s) FAILED\n", failed);
}

void print_markdown(const std::vector<Check>& checks, const DeviceFacts& facts) {
  const auto by_category = tally();
  int implemented_total = 0;
  for (size_t i = 0; i < kEntryCount; ++i)
    if (implemented(kEntries[i].status)) ++implemented_total;

  std::printf("# GRXCP Conformance Report\n\n");
  std::printf("Generated by `grx-conform --markdown`. Do not hand-edit.\n\n");
  std::printf("## Headline\n\n");
  std::printf("**%d of %zu tracked CUDA Runtime entry points implemented "
              "(%.0f%%).**\n\n",
              implemented_total, kEntryCount,
              100.0 * implemented_total / (double)kEntryCount);
  std::printf("The denominator is a curated surface, not the whole CUDA API — "
              "see the note at the top of `tools/common/cuda_api_table.inc` "
              "for what is counted and why. This is an **API coverage** "
              "number. It is not a conformance pass rate: nothing in this "
              "report executes a kernel.\n\n");

  std::printf("| Category | Total | Mapped | Partial | Refused | Absent |\n");
  std::printf("|---|---|---|---|---|---|\n");
  for (const auto& kv : by_category) {
    const CategoryStats& s = kv.second;
    std::printf("| %s | %d | %d | %d | %d | %d |\n", kv.first.c_str(),
                s.total(), s.mapped, s.partial, s.unsupported, s.absent);
  }

  std::printf("\n## Partial: works, with a caveat\n\n");
  std::printf("| CUDA | GRXCP | Caveat |\n|---|---|---|\n");
  for (size_t i = 0; i < kEntryCount; ++i) {
    if (kEntries[i].status != PARTIAL) continue;
    std::printf("| `%s` | `%s` | %s |\n", kEntries[i].cuda_name,
                kEntries[i].grx_name, kEntries[i].note);
  }

  std::printf("\n## Refused: declared, returns `grxErrorNotSupported`\n\n");
  std::printf("| CUDA | GRXCP | Why |\n|---|---|---|\n");
  for (size_t i = 0; i < kEntryCount; ++i) {
    if (kEntries[i].status != UNSUPPORTED) continue;
    std::printf("| `%s` | `%s` | %s |\n", kEntries[i].cuda_name,
                kEntries[i].grx_name, kEntries[i].note);
  }

  std::printf("\n## Absent: a port using these fails to compile\n\n");
  std::printf("Failing at compile time is the point. A mystery at runtime "
              "would be worse.\n\n");
  std::printf("| CUDA | Why |\n|---|---|\n");
  for (size_t i = 0; i < kEntryCount; ++i) {
    if (kEntries[i].status != ABSENT) continue;
    std::printf("| `%s` | %s |\n", kEntries[i].cuda_name, kEntries[i].note);
  }

  std::printf("\n## Refusal verification\n\n");
  int failed = 0;
  for (const Check& c : checks) if (!c.passed) ++failed;
  std::printf("%zu of %d refused entry points verified to actually refuse at "
              "runtime%s.\n\n",
              checks.size() - (size_t)failed, count_status(UNSUPPORTED),
              failed ? " — **with failures, see below**" : "");
  if (failed) {
    for (const Check& c : checks)
      if (!c.passed)
        std::printf("- `%s` returned `%s` instead of `grxErrorNotSupported`\n",
                    c.name, c.detail);
    std::printf("\n");
  }

  std::printf("## Behaviour a port will notice\n\n");
  if (!facts.have_device) {
    std::printf("No device was reachable when this report was generated, so "
                "the device-side facts below were not probed.\n\n");
  } else {
    std::printf("| Property | State |\n|---|---|\n");
    std::printf("| Device | %s |\n", facts.name.c_str());
    std::printf("| Warp shuffle | %s |\n",
                facts.warp_shuffle_emulated
                    ? "emulated through local memory, roughly an order of "
                      "magnitude slower than a register shuffle"
                    : "native");
    std::printf("| Event elapsed time | %s |\n",
                facts.event_timing_host
                    ? "host clock; measures submission, not device execution"
                    : "device-side timestamps");
    std::printf("| `__constant__` | %s |\n",
                facts.constant_is_global
                    ? "lowered to read-only global memory; no broadcast "
                      "bandwidth advantage"
                    : "constant cache");
    std::printf("| Managed memory | %s |\n",
                facts.managed_memory ? "available"
                                     : "refused on this backend");
    std::printf("\n");
  }

  std::printf("## What this report does not measure\n\n");
  std::printf("Kernel execution. No kernel runs in *this report*, so nothing "
              "above says whether a kernel produces the right answer. That is "
              "covered separately: `ci/run_real.sh` compiles a kernel with the "
              "device toolchain, runs it on a real backend and checks the "
              "arithmetic.\n\nStream concurrency is still genuinely unproven. "
              "The command processor runs a single queue, so no test anywhere "
              "can currently fail because of a race between streams.\n");
}

}  // namespace

int main(int argc, char** argv) {
  bool markdown = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--markdown")) markdown = true;
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: grx-conform [--markdown]\n");
      return 0;
    } else {
      std::fprintf(stderr, "grx-conform: unknown option '%s'\n", argv[i]);
      return 2;
    }
  }

  const DeviceFacts facts  = probe_device();
  const auto        checks = run_refusal_checks();

  if (markdown) print_markdown(checks, facts);
  else          print_human(checks, facts);

  int failed = 0;
  for (const Check& c : checks) if (!c.passed) ++failed;
  if ((int)checks.size() != count_status(UNSUPPORTED)) ++failed;
  return failed ? 1 : 0;
}
