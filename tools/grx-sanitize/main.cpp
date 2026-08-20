// grx-sanitize — run a GRXCP program with device memory checking on, and turn
// what the runtime reports into file:line.
//
//   grx-sanitize [options] -- <program> [args...]
//
// The split of work is the same one AddressSanitizer uses on the host. The
// runtime does the detection, because only the runtime knows what was
// allocated and how big it really was; this tool does the symbolization,
// because only the build tree has the ELF and llvm-symbolizer.
//
// It sets GRX_SANITIZE=1, reads the child's stderr, and passes everything that
// is not a GRXSAN line straight through so the program's own output is not
// swallowed. Findings are printed at the end, in order, with the source line
// of the offending access.
//
// Exit code: the child's, if it failed. Otherwise 1 when there were findings,
// 0 when there were none. That makes it usable as a gate.
//
// What it does NOT do is decide whether the run was meaningful. A module built
// without ci/build_kernel.sh --sanitize carries no instrumentation, and the
// runtime says so on a GRXSAN|status line; this tool prints that as a warning
// and exits non-zero, because "no findings" from an uninstrumented binary is
// not a clean bill of health.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

struct Finding {
  std::map<std::string, std::string> f;
};

void usage() {
  std::fprintf(stderr,
      "usage: grx-sanitize [--symbolizer PATH] [--elf PATH] [--quiet]"
      " -- <program> [args...]\n"
      "\n"
      "  --symbolizer PATH  llvm-symbolizer to use (default: $GRX_SYMBOLIZER,\n"
      "                     then $TOOLDIR/llvm-vortex/bin, then PATH)\n"
      "  --elf PATH         ELF with line tables for the device image; by\n"
      "                     default the one the runtime names in its report\n"
      "  --quiet            findings only, no summary banner\n");
}

std::string field(const Finding& fi, const char* k, const char* dflt = "") {
  auto it = fi.f.find(k);
  return (it == fi.f.end()) ? std::string(dflt) : it->second;
}

bool file_exists(const std::string& p) {
  return !p.empty() && ::access(p.c_str(), X_OK) == 0;
}

std::string find_symbolizer(const std::string& explicit_path) {
  if (!explicit_path.empty()) return explicit_path;
  if (const char* e = std::getenv("GRX_SYMBOLIZER")) {
    if (file_exists(e)) return e;
  }
  std::vector<std::string> candidates;
  if (const char* t = std::getenv("TOOLDIR"))
    candidates.push_back(std::string(t) + "/llvm-vortex/bin/llvm-symbolizer");
  if (const char* h = std::getenv("HOME"))
    candidates.push_back(std::string(h) + "/tools/llvm-vortex/bin/llvm-symbolizer");
  for (const std::string& c : candidates)
    if (file_exists(c)) return c;
  // Last resort: whatever is on PATH. An llvm-symbolizer built for another
  // target still reads DWARF fine -- the line tables are target independent.
  return "llvm-symbolizer";
}

// One "function\nfile:line:col" pair, or an empty string when the address
// cannot be resolved. Deliberately reports failure rather than guessing.
std::string symbolize(const std::string& symbolizer, const std::string& elf,
                      unsigned long long pc) {
  if (elf.empty()) return "";
  char cmd[4096];
  std::snprintf(cmd, sizeof(cmd),
                "'%s' --obj='%s' --functions=short --demangle --output-style=LLVM"
                " 0x%llx 2>/dev/null",
                symbolizer.c_str(), elf.c_str(), pc);
  std::FILE* p = ::popen(cmd, "r");
  if (!p) return "";

  char line[1024];
  std::string func, loc;
  if (std::fgets(line, sizeof(line), p)) func = line;
  if (std::fgets(line, sizeof(line), p)) loc  = line;
  ::pclose(p);

  auto chomp = [](std::string& s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  };
  chomp(func);
  chomp(loc);
  if (loc.empty() || loc.rfind("??", 0) == 0) return "";
  if (func.empty() || func == "??") return loc;
  return loc + "  in " + func;
}

Finding parse(const std::string& line) {
  Finding fi;
  size_t pos = 0;
  while (pos < line.size()) {
    const size_t bar = line.find('|', pos);
    const std::string tok = line.substr(pos, (bar == std::string::npos)
                                                 ? std::string::npos
                                                 : bar - pos);
    const size_t eq = tok.find('=');
    if (eq != std::string::npos) fi.f[tok.substr(0, eq)] = tok.substr(eq + 1);
    if (bar == std::string::npos) break;
    pos = bar + 1;
  }
  return fi;
}

}  // namespace

int main(int argc, char** argv) {
  std::string symbolizer_opt, elf_opt;
  bool quiet = false;

  int i = 1;
  for (; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--") { ++i; break; }
    else if (a == "--symbolizer" && i + 1 < argc) symbolizer_opt = argv[++i];
    else if (a == "--elf" && i + 1 < argc)        elf_opt = argv[++i];
    else if (a == "--quiet")                      quiet = true;
    else if (a == "-h" || a == "--help")          { usage(); return 0; }
    else { std::fprintf(stderr, "grx-sanitize: unknown option %s\n", a.c_str());
           usage(); return 2; }
  }
  if (i >= argc) { usage(); return 2; }

  std::vector<char*> child_argv;
  for (int k = i; k < argc; ++k) child_argv.push_back(argv[k]);
  child_argv.push_back(nullptr);

  int pipefd[2];
  if (::pipe(pipefd) != 0) {
    std::fprintf(stderr, "grx-sanitize: pipe: %s\n", std::strerror(errno));
    return 2;
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    std::fprintf(stderr, "grx-sanitize: fork: %s\n", std::strerror(errno));
    return 2;
  }
  if (pid == 0) {
    ::close(pipefd[0]);
    ::dup2(pipefd[1], STDERR_FILENO);
    ::close(pipefd[1]);
    ::setenv("GRX_SANITIZE", "1", 1);
    ::execvp(child_argv[0], child_argv.data());
    std::fprintf(stderr, "grx-sanitize: cannot run %s: %s\n", child_argv[0],
                 std::strerror(errno));
    ::_exit(127);
  }

  ::close(pipefd[1]);
  std::FILE* err = ::fdopen(pipefd[0], "r");

  std::vector<Finding> findings;
  std::vector<std::string> statuses;
  int declared = -1, dropped = 0;

  char buf[8192];
  while (err && std::fgets(buf, sizeof(buf), err)) {
    std::string line = buf;
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
      line.pop_back();

    if (line.rfind("GRXSAN|", 0) != 0) {
      std::fprintf(stderr, "%s\n", line.c_str());
      continue;
    }
    const Finding fi = parse(line);
    if (line.rfind("GRXSAN|summary", 0) == 0) {
      declared = std::atoi(field(fi, "findings", "-1").c_str());
      dropped  = std::atoi(field(fi, "dropped", "0").c_str());
    } else if (line.rfind("GRXSAN|status", 0) == 0) {
      statuses.push_back(line.substr(std::strlen("GRXSAN|")));
    } else {
      findings.push_back(fi);
    }
  }
  if (err) std::fclose(err);

  int status = 0;
  ::waitpid(pid, &status, 0);
  const int child_rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128;

  const std::string symbolizer = find_symbolizer(symbolizer_opt);

  for (size_t k = 0; k < findings.size(); ++k) {
    const Finding& fi = findings[k];
    const std::string elf = elf_opt.empty() ? field(fi, "elf") : elf_opt;
    const unsigned long long pc =
        std::strtoull(field(fi, "pc", "0").c_str(), nullptr, 0);

    // The recorded PC is a RETURN address -- the instruction after the call
    // that reported. Symbolizing pc-1 lands inside the call, which is the
    // statement that made the access.
    const std::string where = (pc > 0) ? symbolize(symbolizer, elf, pc - 1) : "";

    std::fprintf(stderr, "\n=== grx-sanitize: %s %s of %s bytes ===\n",
                 field(fi, "kind", "?").c_str(), field(fi, "access", "?").c_str(),
                 field(fi, "size", "?").c_str());
    if (!where.empty())
      std::fprintf(stderr, "  at %s\n", where.c_str());
    else
      std::fprintf(stderr, "  at pc %s (no source line: %s)\n",
                   field(fi, "pc", "?").c_str(),
                   elf.empty() ? "the runtime reported no ELF for this module"
                               : "llvm-symbolizer could not resolve it");

    // Everything below is derived from the raw fields the runtime reported.
    // "past the end" is measured from the LAST byte the access touches, so a
    // 4-byte store landing exactly on the boundary reads as 4 bytes past
    // rather than 0 -- which is the number that tells you how much to shrink.
    const std::string kind  = field(fi, "kind");
    const std::string alloc = field(fi, "alloc");
    const unsigned long long addr =
        std::strtoull(field(fi, "addr", "0").c_str(), nullptr, 0);
    const unsigned long long base =
        std::strtoull(alloc.empty() ? "0" : alloc.c_str(), nullptr, 0);
    const unsigned long long asize =
        std::strtoull(field(fi, "allocsize", "0").c_str(), nullptr, 0);
    const unsigned long long width =
        std::strtoull(field(fi, "size", "0").c_str(), nullptr, 0);
    const std::string id = field(fi, "allocid", "?");

    std::fprintf(stderr, "  address 0x%llx\n", addr);
    if (base != 0) {
      if (kind == "oob-shared") {
        std::fprintf(stderr,
                     "  this CTA's shared-memory slot is %llu bytes at 0x%llx;"
                     " the access ends %llu bytes past it\n",
                     asize, base, (addr + width) - (base + asize));
      } else if (kind == "use-after-free") {
        std::fprintf(stderr,
                     "  allocation #%s (%llu bytes at 0x%llx) was freed before"
                     " this access\n", id.c_str(), asize, base);
      } else if (kind == "oob-straddle") {
        std::fprintf(stderr,
                     "  the access starts inside allocation #%s (%llu bytes at"
                     " 0x%llx) and ends %llu bytes past its end\n",
                     id.c_str(), asize, base, (addr + width) - (base + asize));
      } else if (addr < base) {
        std::fprintf(stderr,
                     "  no live allocation covers it; the nearest above is"
                     " #%s (%llu bytes at 0x%llx), %llu bytes higher\n",
                     id.c_str(), asize, base, base - addr);
      } else {
        std::fprintf(stderr,
                     "  no live allocation covers it; the nearest below is"
                     " #%s (%llu bytes at 0x%llx), which ends %llu bytes"
                     " before it\n",
                     id.c_str(), asize, base, addr - (base + asize));
      }
    } else if (kind != "oob-shared") {
      std::fprintf(stderr,
                   "  no live allocation covers it, and none lies below it\n");
    }
    std::fprintf(stderr, "  kernel %s, block %s, thread %s (warp %s lane %s)\n",
                 field(fi, "kernel", "?").c_str(), field(fi, "block", "?").c_str(),
                 field(fi, "thread", "?").c_str(), field(fi, "warp", "?").c_str(),
                 field(fi, "lane", "?").c_str());
  }

  for (const std::string& s : statuses)
    std::fprintf(stderr, "\ngrx-sanitize: NOTE %s\n", s.c_str());

  const bool uninstrumented = [&] {
    for (const std::string& s : statuses)
      if (s.find("instrumented=0") != std::string::npos) return true;
    return false;
  }();

  if (!quiet) {
    if (findings.empty() && !uninstrumented)
      std::fprintf(stderr,
                   "\ngrx-sanitize: no findings%s\n",
                   (declared < 0) ? " (the runtime reported no summary --"
                                    " was GRX_SANITIZE honoured?)" : "");
    else
      std::fprintf(stderr, "\ngrx-sanitize: %zu finding%s%s\n", findings.size(),
                   findings.size() == 1 ? "" : "s",
                   dropped ? " (plus more than the report buffer held)" : "");
    if (uninstrumented)
      std::fprintf(stderr,
                   "grx-sanitize: at least one kernel carried no"
                   " instrumentation -- rebuild it with"
                   " ci/build_kernel.sh --sanitize.\n"
                   "              A clean result from an uninstrumented binary"
                   " means nothing was checked.\n");
  }

  if (child_rc != 0) return child_rc;
  if (!findings.empty() || uninstrumented) return 1;
  return 0;
}
