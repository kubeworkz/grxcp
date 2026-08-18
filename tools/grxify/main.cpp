// grxify — translate CUDA source to GRXCP.
//
// A mechanical rename driven by the same table grx-conform publishes from, so
// the translator and the coverage report can never disagree about what is
// supported.
//
// The interesting output is not the rewritten file, it is the diagnostics. A
// port fails on the calls that have no GRXCP equivalent, and the useful thing
// a tool can do is name them up front, with the reason, before someone spends a
// day discovering them one compile error at a time. `--check` does exactly that
// and nothing else, so it can gate a port in CI.
//
//   grxify in.cu                 rewrite to stdout
//   grxify in.cu -o out.grx.cpp  rewrite to a file
//   grxify --check in.cu         report only; exit 1 if anything is unmappable

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

// Functions and types both matter to a translator: a type the compat header
// already renames must not be reported as an unknown CUDA symbol.
const Entry kEntries[] = {
#define GRX_CUDA_API(cuda, grx, status, category, note) \
  {#cuda, #grx, status, category, note},
#define GRX_CUDA_TYPE(cuda, grx, status, note) \
  {#cuda, #grx, status, "type", note},
#include "../common/cuda_api_table.inc"
#undef GRX_CUDA_TYPE
#undef GRX_CUDA_API
};

constexpr size_t kEntryCount = sizeof(kEntries) / sizeof(kEntries[0]);

const std::map<std::string, const Entry*>& table() {
  static const std::map<std::string, const Entry*> m = [] {
    std::map<std::string, const Entry*> t;
    for (size_t i = 0; i < kEntryCount; ++i) t[kEntries[i].cuda_name] = &kEntries[i];
    return t;
  }();
  return m;
}

bool is_ident_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}

struct Finding {
  int         line;
  std::string identifier;
  std::string reason;
};

struct Result {
  std::string          output;
  std::vector<Finding> unmappable;
  int                  rewritten = 0;
};

Result translate(const std::string& src) {
  Result r;
  r.output.reserve(src.size() + src.size() / 8);

  int line = 1;
  size_t i = 0;
  while (i < src.size()) {
    // Include-directive rewriting has to come first: the CUDA headers are what
    // pull in every name being renamed below.
    if (src.compare(i, 8, "#include") == 0 &&
        (i == 0 || src[i - 1] == '\n')) {
      const size_t eol = src.find('\n', i);
      const std::string directive = src.substr(i, (eol == std::string::npos)
                                                      ? std::string::npos
                                                      : eol - i);
      if (directive.find("cuda_runtime.h")   != std::string::npos ||
          directive.find("cuda_runtime_api.h") != std::string::npos ||
          directive.find("<cuda.h>")         != std::string::npos) {
        r.output += "#include <grx/grx_cuda_compat.h>";
        ++r.rewritten;
        i = (eol == std::string::npos) ? src.size() : eol;
        continue;
      }
    }

    if (!is_ident_char(src[i]) || (i > 0 && is_ident_char(src[i - 1]))) {
      if (src[i] == '\n') ++line;
      r.output += src[i++];
      continue;
    }

    size_t j = i;
    while (j < src.size() && is_ident_char(src[j])) ++j;
    const std::string ident = src.substr(i, j - i);

    auto it = table().find(ident);
    if (it != table().end()) {
      const Entry* e = it->second;
      if (e->status == ABSENT) {
        // Left untouched on purpose: the compile error that follows names the
        // exact identifier, which is more useful than a silently mangled call.
        r.unmappable.push_back({line, ident, e->note});
        r.output += ident;
      } else if (std::strcmp(e->category, "type") == 0) {
        // The compat header renames types with a #define, so the CUDA spelling
        // keeps working. Rewriting it would be churn for no benefit.
        r.output += ident;
      } else {
        r.output += e->grx_name;
        ++r.rewritten;
        if (e->status == UNSUPPORTED) {
          r.unmappable.push_back(
              {line, ident,
               std::string("refused at runtime: ") + e->note});
        }
      }
    } else if ((ident.compare(0, 4, "cuda") == 0 && ident.size() > 4) ||
               (ident.compare(0, 2, "cu") == 0 && ident.size() > 2 &&
                ident[2] >= 'A' && ident[2] <= 'Z')) {
      // Looks like a CUDA entry point but is not tracked. Reporting it beats
      // guessing, and it is also how the table finds out it has a hole.
      r.unmappable.push_back(
          {line, ident, "not in the tracked API surface; add it to "
                        "tools/common/cuda_api_table.inc if it should be"});
      r.output += ident;
    } else {
      r.output += ident;
    }
    i = j;
  }
  return r;
}

std::string read_file(const char* path, bool* ok) {
  *ok = false;
  std::FILE* f = std::fopen(path, "rb");
  if (!f) return {};
  std::string s;
  char buf[8192];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  *ok = true;
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  const char* input  = nullptr;
  const char* output = nullptr;
  bool check_only = false;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--check")) check_only = true;
    else if (!std::strcmp(argv[i], "-o") && i + 1 < argc) output = argv[++i];
    else if (!std::strcmp(argv[i], "--help")) {
      std::printf("usage: grxify [--check] [-o out] in.cu\n");
      return 0;
    } else if (argv[i][0] == '-') {
      std::fprintf(stderr, "grxify: unknown option '%s'\n", argv[i]);
      return 2;
    } else {
      input = argv[i];
    }
  }
  if (!input) {
    std::fprintf(stderr, "grxify: no input file\n");
    return 2;
  }

  bool ok = false;
  const std::string src = read_file(input, &ok);
  if (!ok) {
    std::fprintf(stderr, "grxify: cannot read %s\n", input);
    return 2;
  }

  const Result r = translate(src);

  for (const Finding& f : r.unmappable)
    std::fprintf(stderr, "%s:%d: %s: %s\n", input, f.line,
                 f.identifier.c_str(), f.reason.c_str());

  if (!check_only) {
    if (output) {
      std::FILE* out = std::fopen(output, "wb");
      if (!out) {
        std::fprintf(stderr, "grxify: cannot write %s\n", output);
        return 2;
      }
      std::fwrite(r.output.data(), 1, r.output.size(), out);
      std::fclose(out);
    } else {
      std::fwrite(r.output.data(), 1, r.output.size(), stdout);
    }
  }

  std::fprintf(stderr, "grxify: %d identifier(s) rewritten, %zu need attention\n",
               r.rewritten, r.unmappable.size());
  return r.unmappable.empty() ? 0 : 1;
}
