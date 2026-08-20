// grxcc — the GRXCP single-source driver.
//
//   grxcc [options] source.grx.cpp -o program
//
// One file with __global__ kernels and <<<>>> launches in it, compiled twice
// and linked once. This is the piece that makes GRXCP a programming platform
// rather than a library binding.
//
// AN ORCHESTRATOR, NOT A FRONTEND
//
// The architecture doc's open question 1 asks whether grxcc should be a real
// Clang driver -- a ToolChain and an Action graph, upstream-shaped -- or a
// program that rewrites source and shells out to clang twice. The answer it
// settles on is the second now and the first in phase 6, once the flag surface
// has stopped moving, and this is that second thing.
//
// What that buys: the risk here is integration, not compiler research. VOLT
// already lowers SIMT kernels, vxbin.py already builds multi-entry binaries,
// and the runtime already has every registration entry point this emits. What
// it costs is written down under "WHAT THE REWRITER CANNOT SEE" below, because
// a source rewriter that pretends to understand C++ is worse than one that
// says where it stops.
//
// THE PIPELINE
//
//   1. Rewrite the source into two.
//        device pass  each __global__ keeps its body, wrapped in an entry
//                     point that unpacks the argument blob (see below)
//        host pass    each __global__ becomes a stub that packs its arguments
//                     and calls grxLaunchKernel; each <<<>>> becomes a
//                     configuration push plus a call to that stub
//   2. Compile the device pass with ci/build_kernel.sh -> .vxbin
//   3. Wrap the .vxbin in a .grxfat container and emit it as a byte array
//      appended to the host pass, next to the registration constructor
//   4. Compile the host pass and link it against libgrxrt and the driver
//
// WHY THE KERNEL GETS A WRAPPER
//
// A __global__ function with several parameters compiles fine for the device:
// the backend gives it the ordinary RISC-V convention, a0/a1/fa0 and so on.
// The launch path cannot fill those. The command processor stages ONE blob and
// hands the kernel ONE pointer to it, so a multi-parameter kernel would read
// whatever happened to be in a1.
//
// Verified rather than assumed -- disassembling `__global__ void k(int, float*,
// double)` shows it reading a0, a1 and fa0 directly. So grxcc emits the entry
// point itself: a one-pointer kernel that unpacks a generated struct and calls
// the user's body. The struct is emitted into BOTH passes, so the offsets the
// host packs to and the offsets the device reads from come from one
// declaration compiled twice rather than from two calculations that agree
// until they do not.
//
// WHAT THE REWRITER CANNOT SEE
//
// It lexes: it tracks strings, character literals, raw strings, and both
// comment forms, so none of those can be mistaken for code. It does not parse
// types or resolve names. Concretely:
//
//   - A kernel's parameters are split on top-level commas and the last
//     identifier is taken as the name. `int a`, `const float* p` and
//     `T (*fn)(int)` all work; a parameter with no name does not, and is
//     reported rather than guessed at.
//   - `is_pointer` is decided by looking for a `*` outside brackets. A typedef
//     that hides a pointer -- `typedef float* fptr;` -- is NOT detected, and
//     that matters only on a device whose pointers are narrower than the
//     host's. rv64 is the supported target and the widths match there; the
//     runtime's packing path refuses a pointer that does not fit rather than
//     truncating it, which is the backstop for the day rv32 matters.
//   - Templated kernels are not supported: `template <...> __global__ ...` is
//     reported. CUDA's own instantiation rules for those are subtle and a
//     rewriter would get them wrong quietly.
//   - A kernel must be at FILE OR NAMESPACE scope. Namespaces are tracked, so
//     `ns::kernel<<<...>>>` resolves and the registration tables qualify their
//     references; a __global__ inside a class or a function body is reported,
//     because a host stub has to be a free function whose address keys the
//     registry.
//   - Two kernels may not share an unqualified name even across namespaces:
//     the device entry point is `__vx_kentry_<name>`, so `a::run` and `b::run`
//     are one symbol on the device.
//   - Overloaded kernel names are not supported, for the same reason -- and
//     because the stub takes its own address as its registry key, where `&foo`
//     would be ambiguous.
//   - `#if` is not evaluated, so a kernel inside a disabled preprocessor branch
//     is still compiled and registered. That is conservative in the safe
//     direction: nothing launches it.
//
// Every one of those is a diagnostic, not a silent miscompile.

#include <grx/grx_abi.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// ---------------------------------------------------------------------------
// A lexer that knows what is not code
// ---------------------------------------------------------------------------
//
// Small on purpose. It exists so that `//` inside a string, a `<<<` inside a
// comment, and a brace inside a character literal cannot move the rewriter.

bool ident_char(char c) {
  return std::isalnum((unsigned char)c) || c == '_';
}

// Advance past whatever non-code construct starts at i, or return i unchanged.
size_t skip_noncode(const std::string& s, size_t i) {
  if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '/') {
    const size_t nl = s.find('\n', i);
    return (nl == std::string::npos) ? s.size() : nl;
  }
  if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '*') {
    const size_t e = s.find("*/", i + 2);
    return (e == std::string::npos) ? s.size() : e + 2;
  }
  // Raw string: R"delim( ... )delim". Checked before the ordinary string case
  // because its body may contain unescaped quotes and backslashes.
  if (s[i] == 'R' && i + 1 < s.size() && s[i + 1] == '"' &&
      (i == 0 || !ident_char(s[i - 1]))) {
    const size_t open = s.find('(', i + 2);
    if (open != std::string::npos) {
      const std::string delim = s.substr(i + 2, open - (i + 2));
      const std::string close = ")" + delim + "\"";
      const size_t e = s.find(close, open);
      return (e == std::string::npos) ? s.size() : e + close.size();
    }
  }
  if (s[i] == '"' || s[i] == '\'') {
    const char q = s[i];
    size_t j = i + 1;
    while (j < s.size()) {
      if (s[j] == '\\') { j += 2; continue; }
      if (s[j] == q) return j + 1;
      ++j;
    }
    return s.size();
  }
  return i;
}

// Match a bracket run starting at `open` (which must be one of ({[), returning
// the index just past its partner. Nested brackets and non-code are handled.
size_t match_bracket(const std::string& s, size_t open) {
  const char o = s[open];
  const char c = (o == '(') ? ')' : (o == '{') ? '}' : ']';
  int depth = 0;
  size_t i = open;
  while (i < s.size()) {
    const size_t skipped = skip_noncode(s, i);
    if (skipped != i) { i = skipped; continue; }
    if (s[i] == o) ++depth;
    else if (s[i] == c) { if (--depth == 0) return i + 1; }
    ++i;
  }
  return std::string::npos;
}

// Find the next occurrence of `c` at or after `from` that is real code.
size_t find_code_char(const std::string& s, size_t from, char c) {
  size_t i = from;
  while (i < s.size()) {
    const size_t skipped = skip_noncode(s, i);
    if (skipped != i) { i = skipped; continue; }
    if (s[i] == c) return i;
    ++i;
  }
  return std::string::npos;
}

// Brace depth at `pos`. Zero means file scope; one means inside a namespace,
// an `extern "C" {` block, or a class.
int brace_depth(const std::string& s, size_t pos) {
  int depth = 0;
  size_t i = 0;
  while (i < pos && i < s.size()) {
    const size_t skipped = skip_noncode(s, i);
    if (skipped != i) { i = skipped; continue; }
    if (s[i] == '{') ++depth;
    else if (s[i] == '}') --depth;
    ++i;
  }
  return depth;
}

// The end of the LAST TOP-LEVEL PREPROCESSOR DIRECTIVE before `pos`, or 0.
//
// This is where grxcc inserts the device headers, and the position is pinned
// between two constraints that nearly meet.
//
//   Not earlier than the user's includes. The device header defines `printf`
//   and `assert` as macros -- which is what makes a kernel able to call them --
//   and those macros poison <cstdio> if the standard header is parsed
//   afterwards.
//
//   Not later than the user's first DEVICE DECLARATION. A CUDA file routinely
//   puts a `__device__` helper above its kernels:
//
//       __inline__ __device__ float warpReduceSum(float v) { ... }
//
//   and `__device__` is one of the names the header defines. Inserting "just
//   before the first __global__" put the header after that helper, and the
//   device compile said `unknown type name '__device__'`.
//
// Every C++ file in practice has its includes and macros first and its
// declarations after, so the end of the last directive lies between them. It is
// also necessarily at file scope, which "just before the first kernel" is not
// when the kernel lives inside a namespace.
//
// A file with no directives at all gets 0, the top of the file, which is right
// for the same reason: there are no includes to poison.
size_t top_level_boundary_before(const std::string& s, size_t pos) {
  int depth = 0;
  size_t best = 0, i = 0;
  bool line_start = true;
  while (i < pos && i < s.size()) {
    const size_t skipped = skip_noncode(s, i);
    if (skipped != i) { i = skipped; continue; }
    const char c = s[i];
    if (c == '#' && line_start && depth == 0) {
      size_t j = i;
      while (j < s.size()) {
        if (s[j] == '\\' && j + 1 < s.size() && s[j + 1] == '\n') { j += 2; continue; }
        if (s[j] == '\n') break;
        ++j;
      }
      i = j;
      if (i <= pos) best = i;
      continue;
    }
    if (c == '{') ++depth;
    else if (c == '}') --depth;
    line_start = (c == '\n') || (line_start && (c == ' ' || c == '\t'));
    ++i;
  }
  return best;
}

// The enclosing named-namespace path at `pos`, with a trailing "::".
//
// Anonymous namespaces contribute nothing: their members are reachable by
// unqualified name from file scope, which is exactly what the generated
// registration tables need.
//
// `namespace a = b;` and `using namespace a;` are not scopes and are skipped --
// both are recognised by there being no `{` after the name.
std::string scope_at(const std::string& s, size_t pos) {
  struct Frame { std::string name; int depth; };
  std::vector<Frame> stack;
  int depth = 0;
  std::string pending;
  bool have_pending = false;
  size_t i = 0;
  while (i < pos && i < s.size()) {
    const size_t skipped = skip_noncode(s, i);
    if (skipped != i) { i = skipped; continue; }
    if (s[i] == '{') {
      ++depth;
      if (have_pending) { stack.push_back({pending, depth}); have_pending = false; }
      ++i;
      continue;
    }
    if (s[i] == '}') {
      if (!stack.empty() && stack.back().depth == depth) stack.pop_back();
      --depth;
      ++i;
      continue;
    }
    if (s.compare(i, 9, "namespace") == 0 &&
        (i == 0 || !ident_char(s[i - 1])) &&
        (i + 9 >= s.size() || !ident_char(s[i + 9]))) {
      size_t j = i + 9;
      while (j < s.size() && std::isspace((unsigned char)s[j])) ++j;
      std::string name;
      while (j < s.size()) {
        if (ident_char(s[j])) { name += s[j++]; continue; }
        if (s[j] == ':' && j + 1 < s.size() && s[j + 1] == ':') { name += "::"; j += 2; continue; }
        break;
      }
      while (j < s.size() && std::isspace((unsigned char)s[j])) ++j;
      if (j < s.size() && s[j] == '{') { pending = name; have_pending = true; i = j; continue; }
      i += 9;
      continue;
    }
    ++i;
  }
  std::string out;
  for (const Frame& f : stack)
    if (!f.name.empty()) out += f.name + "::";
  return out;
}

// How many braces are open at `pos` that are NOT namespaces. Non-zero means a
// class, a struct or a function body, none of which can hold a __global__ that
// grxcc knows how to emit a host stub for.
int non_namespace_depth(const std::string& s, size_t pos) {
  struct Frame { bool is_ns; int depth; };
  std::vector<Frame> stack;
  int depth = 0, ns_open = 0;
  bool have_pending = false;
  size_t i = 0;
  while (i < pos && i < s.size()) {
    const size_t skipped = skip_noncode(s, i);
    if (skipped != i) { i = skipped; continue; }
    if (s[i] == '{') {
      ++depth;
      if (have_pending) { stack.push_back({true, depth}); ++ns_open; have_pending = false; }
      ++i;
      continue;
    }
    if (s[i] == '}') {
      if (!stack.empty() && stack.back().depth == depth) { stack.pop_back(); --ns_open; }
      --depth;
      ++i;
      continue;
    }
    if (s.compare(i, 9, "namespace") == 0 &&
        (i == 0 || !ident_char(s[i - 1])) &&
        (i + 9 >= s.size() || !ident_char(s[i + 9]))) {
      size_t j = i + 9;
      while (j < s.size() && std::isspace((unsigned char)s[j])) ++j;
      while (j < s.size() &&
             (ident_char(s[j]) || (s[j] == ':' && j + 1 < s.size() && s[j + 1] == ':')))
        j += (s[j] == ':') ? 2 : 1;
      while (j < s.size() && std::isspace((unsigned char)s[j])) ++j;
      if (j < s.size() && s[j] == '{') { have_pending = true; i = j; continue; }
      i += 9;
      continue;
    }
    ++i;
  }
  return depth - ns_open;
}

// Split a parameter list on commas that are not inside brackets or angle
// brackets. Angle depth is tracked so `std::pair<int, int> p` stays one
// parameter -- imperfectly, since `a < b, c > d` is ambiguous in C++ itself,
// but a kernel parameter list is a declaration context where it is not.
std::vector<std::string> split_params(const std::string& list) {
  std::vector<std::string> out;
  int round = 0, square = 0, angle = 0;
  size_t start = 0;
  for (size_t i = 0; i < list.size();) {
    const size_t skipped = skip_noncode(list, i);
    if (skipped != i) { i = skipped; continue; }
    const char c = list[i];
    if (c == '(') ++round; else if (c == ')') --round;
    else if (c == '[') ++square; else if (c == ']') --square;
    else if (c == '<') ++angle; else if (c == '>') --angle;
    else if (c == ',' && round == 0 && square == 0 && angle == 0) {
      out.push_back(list.substr(start, i - start));
      start = i + 1;
    }
    ++i;
  }
  const std::string tail = list.substr(start);
  bool only_space = true;
  for (char ch : tail) if (!std::isspace((unsigned char)ch)) only_space = false;
  if (!only_space || !out.empty()) out.push_back(tail);
  return out;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) ++a;
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

// ---------------------------------------------------------------------------
// What a kernel looks like once it has been read
// ---------------------------------------------------------------------------

struct Param {
  std::string decl;      // the parameter as written, e.g. "const float* p"
  std::string name;      // "p"
  bool        is_pointer = false;
};

// A file-scope __device__ or __constant__ variable.
//
// CUDA declares one of these on BOTH sides: the device gets the storage, and
// the host gets an object of the same name whose only job is to have an address
// -- `cudaMemcpyToSymbol(coeffs, ...)` passes that address, and the runtime
// looks the device symbol up by it. So unlike a __device__ FUNCTION, which is
// dropped from the host pass, a __device__ VARIABLE is kept in both.
struct DeviceVar {
  std::string name;
  bool        is_constant = false;
  int         line = 0;
  uint64_t    vma = 0;      // filled in from the device ELF after it is built
  uint32_t    size = 0;
};

struct SharedDecl {
  std::string decl;        // "float tile[4][5]", as written, minus __shared__
  std::string name;        // "tile"
  bool        is_extern = false;
  size_t      start = 0;   // offset of "extern"/"__shared__" in the body
  size_t      end   = 0;   // offset just past the ';'
};

struct Kernel {
  std::string        name;
  // The enclosing named-namespace path, with a trailing "::" -- "ns::" for a
  // kernel in `namespace ns`, empty at file scope. The generated stub and args
  // struct are emitted IN PLACE, so they land inside the namespace; the
  // registration tables are emitted at file scope and have to name them
  // through this.
  std::string        scope;
  // __launch_bounds__(maxThreadsPerBlock[, minBlocksPerMultiprocessor]).
  // The first argument is kept as SOURCE TEXT rather than a parsed number:
  // `__launch_bounds__(kBlock * 2)` is legal CUDA and grxcc has no constant
  // evaluator, so the expression is emitted into the host pass and the host
  // compiler -- which does have one -- works it out. Same principle as
  // grx_launch_shim.h's as_dim overloads.
  std::string        launch_bounds;
  bool               launch_bounds_had_min = false;
  // Architectural registers the kernel touches, filled in after the device
  // compile by reading the ELF. -1 means "not measured", which is what the
  // descriptor carries when the call graph could not be resolved.
  int32_t            num_regs = -1;
  // Static and dynamic __shared__ declarations found in the body.
  std::vector<SharedDecl> shared;
  bool has_static_shared() const {
    for (const SharedDecl& d : shared) if (!d.is_extern) return true;
    return false;
  }
  std::vector<Param> params;
  size_t             decl_start = 0;   // index of "__global__"
  size_t             body_start = 0;   // index of '{'
  size_t             body_end   = 0;   // index just past '}'
  int                line       = 0;

  std::string qualified() const { return scope + name; }
};

struct Diagnostic {
  int         line;
  std::string text;
};

struct Parsed {
  std::vector<Kernel>     kernels;
  std::vector<Diagnostic> errors;
};

int line_of(const std::string& s, size_t pos) {
  int line = 1;
  for (size_t i = 0; i < pos && i < s.size(); ++i)
    if (s[i] == '\n') ++line;
  return line;
}

// The last identifier in a parameter declaration is its name. Everything before
// it is the type, whatever shape that type has.
bool split_param(const std::string& decl_in, Param* out) {
  const std::string decl = trim(decl_in);
  if (decl.empty() || decl == "void") return false;

  // A pointer is a '*' at top level. Brackets are skipped so that a function
  // pointer's inner '*' does not make the parameter itself look like one.
  int depth = 0;
  for (size_t i = 0; i < decl.size(); ++i) {
    const char c = decl[i];
    if (c == '(' || c == '[') ++depth;
    else if (c == ')' || c == ']') --depth;
    else if ((c == '*' || c == '&') && depth == 0) out->is_pointer = (c == '*');
  }

  size_t end = decl.size();
  while (end > 0 && !ident_char(decl[end - 1])) --end;
  size_t start = end;
  while (start > 0 && ident_char(decl[start - 1])) --start;
  if (start == end) return false;
  out->decl = decl;
  out->name = decl.substr(start, end - start);
  return true;
}

// File-scope __device__ and __constant__ VARIABLES.
//
// Distinguished from a __device__ function by what follows the declaration: a
// `{` opens a function body, anything reaching a `;` first is a variable. That
// is the same test find_device_only makes, and the two have to agree -- a
// variable dropped from the host pass takes its address with it, and
// cudaMemcpyToSymbol has nothing to name.
std::vector<DeviceVar> find_device_vars(const std::string& src) {
  std::vector<DeviceVar> out;
  size_t i = 0;
  while (i < src.size()) {
    const size_t skipped = skip_noncode(src, i);
    if (skipped != i) { i = skipped; continue; }

    bool is_const = false;
    size_t kw_len = 0;
    if (src.compare(i, 12, "__constant__") == 0) { is_const = true; kw_len = 12; }
    else if (src.compare(i, 10, "__device__") == 0) { kw_len = 10; }
    if (kw_len == 0 || (i > 0 && ident_char(src[i - 1])) ||
        (i + kw_len < src.size() && ident_char(src[i + kw_len]))) {
      ++i;
      continue;
    }
    // Only at file or namespace scope; a __constant__ inside a function body is
    // not a thing CUDA allows either.
    if (non_namespace_depth(src, i) != 0) { i += kw_len; continue; }

    size_t j = i + kw_len, end = std::string::npos;
    bool is_function = false;
    while (j < src.size()) {
      const size_t sk = skip_noncode(src, j);
      if (sk != j) { j = sk; continue; }
      if (src[j] == '(') {          // a parameter list: this is a function
        is_function = true;
        break;
      }
      if (src[j] == '{' || src[j] == ';') {
        // A brace here is an initializer (`= {1, 2, 3}`), not a body, because
        // a function's `(` would have been seen first.
        end = j;
        break;
      }
      ++j;
    }
    if (is_function || end == std::string::npos) { i += kw_len; continue; }

    const size_t semi = find_code_char(src, i + kw_len, ';');
    if (semi == std::string::npos) { i += kw_len; continue; }

    DeviceVar v;
    v.is_constant = is_const;
    v.line = line_of(src, i);
    // The name is the identifier before the first '[', '=' or ';'.
    {
      const std::string decl = src.substr(i + kw_len, semi - (i + kw_len));
      size_t e = decl.size();
      for (char c : {'[', '='}) {
        const size_t at = decl.find(c);
        if (at != std::string::npos && at < e) e = at;
      }
      while (e > 0 && !ident_char(decl[e - 1])) --e;
      size_t b = e;
      while (b > 0 && ident_char(decl[b - 1])) --b;
      v.name = decl.substr(b, e - b);
    }
    if (!v.name.empty()) out.push_back(v);
    i = semi + 1;
  }
  return out;
}

// The first place the file says something only the device understands:
// __global__, __device__ or __constant__, whichever comes first.
//
// This is what the device headers have to be inserted before. "Before the first
// __global__" is not enough -- a file routinely writes
//
//     __constant__ float c_filter[7];
//     #define CUDA_CHECK(call) ...
//     __global__ void convolve(...) { ... }
//
// and inserting after the LAST directive before the kernel puts the headers
// after the __constant__, which then does not name a type.
size_t first_device_token(const std::string& src) {
  static const char* kToks[] = {"__global__", "__device__", "__constant__"};
  size_t i = 0;
  while (i < src.size()) {
    const size_t skipped = skip_noncode(src, i);
    if (skipped != i) { i = skipped; continue; }
    for (const char* t : kToks) {
      const size_t n = std::strlen(t);
      if (src.compare(i, n, t) == 0 && (i == 0 || !ident_char(src[i - 1])) &&
          (i + n >= src.size() || !ident_char(src[i + n])))
        return i;
    }
    ++i;
  }
  return src.size();
}

// Spans of `__device__`-only code, to be dropped from the HOST pass.
//
// A CUDA file routinely puts a device helper above its kernels:
//
//     __inline__ __device__ float warpReduceSum(float v) {
//       for (int o = warpSize / 2; o > 0; o /= 2)
//         v += __shfl_down_sync(0xffffffff, v, o);
//       return v;
//     }
//
// The host pass is the whole file with kernel bodies swapped for launch stubs,
// so without this it keeps that function -- and the host compiler, which has
// never heard of `__device__`, `warpSize` or `__shfl_down_sync`, reports three
// errors about a function the host was never going to call. nvcc drops these
// from its host pass; so does grxcc.
//
// `__host__ __device__` is NOT dropped: it is the CUDA spelling for "compile
// this for both", and both is what it gets.
struct Span { size_t start, end; };

std::vector<Span> find_device_only(const std::string& src) {
  std::vector<Span> out;
  size_t i = 0;
  while (i < src.size()) {
    const size_t skipped = skip_noncode(src, i);
    if (skipped != i) { i = skipped; continue; }
    if (src.compare(i, 10, "__device__") != 0 ||
        (i > 0 && ident_char(src[i - 1])) ||
        (i + 10 < src.size() && ident_char(src[i + 10]))) {
      ++i;
      continue;
    }

    // Back up to the start of the declaration: the previous top-level
    // boundary. Any preceding qualifiers (`static`, `__inline__`, a comment)
    // come with it.
    size_t start = i;
    while (start > 0 && src[start - 1] != ';' && src[start - 1] != '{' &&
           src[start - 1] != '}' && src[start - 1] != '\n')
      --start;
    // A `\n` boundary can leave a partial line; walk back over blank ones so a
    // multi-line declaration head is not cut in half.
    {
      size_t p2 = start;
      while (p2 > 0) {
        size_t line_begin = src.rfind('\n', p2 - 1);
        line_begin = (line_begin == std::string::npos) ? 0 : line_begin + 1;
        const std::string prev = trim(src.substr(line_begin, p2 - line_begin));
        if (prev.empty() || prev.back() == ';' || prev.back() == '}' ||
            prev.back() == '{' || prev[0] == '#' ||
            (prev.size() >= 2 && prev.compare(0, 2, "//") == 0))
          break;
        p2 = line_begin;
      }
      start = p2;
    }

    const std::string head = src.substr(start, i - start);
    if (head.find("__host__") != std::string::npos) { i += 10; continue; }
    if (head.find("__global__") != std::string::npos) { i += 10; continue; }

    // Find the end: a `{` opens a definition, a `;` ends a declaration.
    size_t j = i + 10;
    size_t end = std::string::npos;
    bool is_function = false;
    while (j < src.size()) {
      const size_t sk = skip_noncode(src, j);
      if (sk != j) { j = sk; continue; }
      // A '(' before any '{' or ';' means a parameter list, so this is a
      // function. Without that test a `__device__ float g[4] = {1,2,3,4};`
      // reads as a definition and gets dropped from the host pass, taking the
      // address cudaMemcpyToSymbol needs with it.
      if (src[j] == '(') { is_function = true; ++j; continue; }
      if (src[j] == '{') { end = match_bracket(src, j); break; }
      if (src[j] == ';') { end = j + 1; break; }
      if (src.compare(j, 10, "__host__") == 0) { end = 0; break; }
      ++j;
    }
    if (!is_function) { i += 10; continue; }          // a variable: keep it
    if (end == 0) { i += 10; continue; }              // __device__ __host__
    if (end == std::string::npos) { i += 10; continue; }
    out.push_back({start, end});
    i = end;
  }
  return out;
}

// Find the static and dynamic __shared__ declarations in a kernel body.
//
// Only at the body's own statement level: a __shared__ inside an `if` or a loop
// is not something CUDA allows to mean anything useful, and treating it as a
// hoisted declaration would silently change its scope. Nested ones are left
// alone, so the unavailable-attribute error still fires on them.
std::vector<SharedDecl> find_shared(const std::string& body,
                                    std::vector<Diagnostic>* errors, int line0) {
  std::vector<SharedDecl> out;
  int depth = 0;
  size_t i = 0;
  while (i < body.size()) {
    const size_t skipped = skip_noncode(body, i);
    if (skipped != i) { i = skipped; continue; }
    if (body[i] == '{') { ++depth; ++i; continue; }
    if (body[i] == '}') { --depth; ++i; continue; }

    const bool is_kw = body.compare(i, 10, "__shared__") == 0;
    if (!is_kw || (i > 0 && ident_char(body[i - 1])) ||
        (i + 10 < body.size() && ident_char(body[i + 10]))) {
      ++i;
      continue;
    }

    SharedDecl d;
    d.start = i;
    // `extern __shared__` -- take the `extern` with it.
    {
      size_t b = i;
      while (b > 0 && (body[b - 1] == ' ' || body[b - 1] == '\t')) --b;
      if (b >= 6 && body.compare(b - 6, 6, "extern") == 0 &&
          (b == 6 || !ident_char(body[b - 7]))) {
        d.is_extern = true;
        d.start = b - 6;
      }
    }

    const size_t semi = find_code_char(body, i + 10, ';');
    if (semi == std::string::npos) {
      errors->push_back({line0, "unterminated __shared__ declaration"});
      break;
    }
    d.end = semi + 1;
    d.decl = trim(body.substr(i + 10, semi - (i + 10)));

    if (depth != 1) {
      errors->push_back({line0,
          "__shared__ '" + d.decl + "' is not at the kernel body's top level; "
          "grxcc hoists static __shared__ into one per-kernel block and cannot "
          "do that for a declaration inside a nested scope"});
      i = d.end;
      continue;
    }

    // The name is the identifier before the first '[' or '=' or the end.
    {
      size_t e = d.decl.size();
      const size_t br = d.decl.find('[');
      if (br != std::string::npos) e = br;
      const size_t eq = d.decl.find('=');
      if (eq != std::string::npos && eq < e) e = eq;
      while (e > 0 && !ident_char(d.decl[e - 1])) --e;
      size_t b = e;
      while (b > 0 && ident_char(d.decl[b - 1])) --b;
      d.name = d.decl.substr(b, e - b);
    }
    if (d.name.empty()) {
      errors->push_back({line0, "cannot find the name in '__shared__ " +
                                    d.decl + "'"});
      i = d.end;
      continue;
    }
    if (d.is_extern) {
      // `extern __shared__ float sdata[];` -- an unsized array by definition.
      // Take the element type as everything before the name.
      const size_t at = d.decl.find(d.name);
      d.decl = trim(d.decl.substr(0, at));
    }
    out.push_back(d);
    i = d.end;
  }
  return out;
}

Parsed find_kernels(const std::string& src) {
  Parsed p;
  size_t i = 0;
  while (i < src.size()) {
    const size_t skipped = skip_noncode(src, i);
    if (skipped != i) { i = skipped; continue; }

    if (src.compare(i, 10, "__global__") != 0 ||
        (i > 0 && ident_char(src[i - 1])) ||
        (i + 10 < src.size() && ident_char(src[i + 10]))) {
      ++i;
      continue;
    }

    Kernel k;
    k.decl_start = i;
    k.line = line_of(src, i);

    // A template head sits BEFORE __global__, so looking only at what follows
    // misses it entirely. Walk back to the end of the previous declaration --
    // the nearest ; { } or the start of file -- and look for the keyword in
    // between. That also catches `template <int N> static __global__ ...`,
    // where the template head and __global__ are not adjacent.
    //
    // A templated kernel has no single argument struct to emit and no single
    // stub address to register, so this is a real limit and not a parser
    // shortcut. Saying so by name beats letting the device pass fail on
    // generated code the author never wrote.
    {
      size_t b = i;
      while (b > 0 && src[b - 1] != ';' && src[b - 1] != '{' &&
             src[b - 1] != '}') --b;
      const std::string prefix = src.substr(b, i - b);
      size_t t = prefix.find("template");
      while (t != std::string::npos) {
        const bool lhs = (t == 0) || !ident_char(prefix[t - 1]);
        const bool rhs = (t + 8 >= prefix.size()) || !ident_char(prefix[t + 8]);
        if (lhs && rhs) break;
        t = prefix.find("template", t + 1);
      }
      if (t != std::string::npos) {
        p.errors.push_back({k.line,
            "templated __global__ kernels are not supported by grxcc; "
            "instantiate them behind a non-template wrapper"});
        i += 10;
        continue;
      }
    }

    // Between __global__ and the name there is a return type, which CUDA
    // requires to be void -- and possibly an ATTRIBUTE, which brings its own
    // parentheses:
    //
    //   __global__ void __launch_bounds__(256, 2) k(float* p) { ... }
    //
    // Taking the first '(' as the parameter list reads `__launch_bounds__` as
    // the kernel's name and `256, 2` as its parameters. That is not a
    // hypothetical: it is what grxcc did before this loop existed, and the only
    // reason it was not silent is that the launch-site check then failed to
    // find a kernel called `bounded`.
    //
    // So walk the parenthesised groups, consuming any whose preceding
    // identifier is an attribute, and keep everything else as the declaration
    // head.
    size_t j = i + 10;
    size_t paren = std::string::npos;
    std::string head_raw;
    bool attr_error = false;
    for (;;) {
      paren = find_code_char(src, j, '(');
      if (paren == std::string::npos) break;
      size_t e = paren;
      while (e > j && std::isspace((unsigned char)src[e - 1])) --e;
      size_t s0 = e;
      while (s0 > j && ident_char(src[s0 - 1])) --s0;
      const std::string tok = src.substr(s0, e - s0);
      const bool is_attr = (tok == "__launch_bounds__" ||
                            tok == "__attribute__" || tok == "alignas");
      if (!is_attr) { head_raw += src.substr(j, paren - j); break; }

      const size_t attr_end = match_bracket(src, paren);
      if (attr_end == std::string::npos) {
        p.errors.push_back({k.line, "unterminated " + tok + " on a __global__"});
        attr_error = true;
        break;
      }
      if (tok == "__launch_bounds__") {
        const std::vector<std::string> a =
            split_params(src.substr(paren + 1, attr_end - paren - 2));
        if (a.empty() || trim(a[0]).empty()) {
          p.errors.push_back({k.line,
              "__launch_bounds__ needs a maximum thread count"});
          attr_error = true;
          break;
        }
        k.launch_bounds = trim(a[0]);
        k.launch_bounds_had_min = a.size() > 1 && !trim(a[1]).empty();
      }
      head_raw += src.substr(j, s0 - j);
      j = attr_end;
    }
    if (attr_error) { i += 10; continue; }
    if (paren == std::string::npos) {
      p.errors.push_back({k.line, "__global__ with no parameter list"});
      break;
    }
    const std::string head = trim(head_raw);
    if (head.find('<') != std::string::npos) {
      p.errors.push_back({k.line,
          "templated __global__ kernels are not supported by grxcc; "
          "instantiate them behind a non-template wrapper"});
      i = paren;
      continue;
    }
    size_t ne = head.size();
    while (ne > 0 && !ident_char(head[ne - 1])) --ne;
    size_t ns = ne;
    while (ns > 0 && ident_char(head[ns - 1])) --ns;
    k.name = head.substr(ns, ne - ns);
    const std::string ret = trim(head.substr(0, ns));
    if (k.name.empty()) {
      p.errors.push_back({k.line, "cannot find the kernel's name"});
      i = paren;
      continue;
    }
    // Namespace scope is supported; anything else nested is not. A __global__
    // inside a class or a function body has no place grxcc can put the host
    // stub -- the stub has to be an ordinary free function whose ADDRESS keys
    // the registry.
    k.scope = scope_at(src, k.decl_start);
    if (non_namespace_depth(src, k.decl_start) != 0) {
      p.errors.push_back({k.line, "kernel '" + k.name +
          "' is not at namespace scope; grxcc can only emit a host stub for a "
          "__global__ declared at file or namespace scope"});
      i = paren;
      continue;
    }

    if (ret != "void") {
      p.errors.push_back({k.line, "kernel '" + k.name +
          "' returns '" + ret + "'; a __global__ function must return void"});
    }

    const size_t after_params = match_bracket(src, paren);
    if (after_params == std::string::npos) {
      p.errors.push_back({k.line, "unterminated parameter list"});
      break;
    }
    const std::string list = src.substr(paren + 1, after_params - paren - 2);
    for (const std::string& d : split_params(list)) {
      const std::string t = trim(d);
      if (t.empty() || t == "void") continue;
      Param prm;
      if (!split_param(t, &prm)) {
        p.errors.push_back({k.line, "kernel '" + k.name + "': parameter '" + t +
            "' has no name. grxcc needs one to pack the argument blob."});
        continue;
      }
      k.params.push_back(prm);
    }

    // Skip whatever sits between the parameter list and the body -- a
    // __launch_bounds__ attribute lives here in CUDA.
    size_t b = after_params;
    while (b < src.size()) {
      const size_t sk = skip_noncode(src, b);
      if (sk != b) { b = sk; continue; }
      if (src[b] == '{') break;
      if (src[b] == ';') break;
      ++b;
    }
    if (b >= src.size() || src[b] == ';') {
      // A declaration without a body. Nothing to emit, and nothing wrong.
      i = (b < src.size()) ? b + 1 : src.size();
      continue;
    }
    const size_t body_end = match_bracket(src, b);
    if (body_end == std::string::npos) {
      p.errors.push_back({k.line, "unterminated body for kernel '" + k.name + "'"});
      break;
    }
    k.body_start = b;
    k.body_end   = body_end;
    k.shared = find_shared(src.substr(b, body_end - b), &p.errors, k.line);
    p.kernels.push_back(k);
    i = body_end;
  }

  // Two kernels may not share an UNQUALIFIED name, even in different
  // namespaces. The device entry point is `__vx_kentry_<name>`, derived from
  // the __global__ alone, so `a::run` and `b::run` are one symbol on the
  // device however distinct they are on the host. C++ would accept the file and
  // the device would run whichever one the linker kept.
  for (size_t a = 0; a < p.kernels.size(); ++a)
    for (size_t b = a + 1; b < p.kernels.size(); ++b)
      if (p.kernels[a].name == p.kernels[b].name)
        p.errors.push_back({p.kernels[b].line,
            "a second __global__ named '" + p.kernels[b].name + "' (the first "
            "is at line " + std::to_string(p.kernels[a].line) + "). The device "
            "entry point comes from the unqualified name, so '" +
            p.kernels[a].qualified() + "' and '" + p.kernels[b].qualified() +
            "' would be one symbol."});
  return p;
}

// ---------------------------------------------------------------------------
// <<< >>>
// ---------------------------------------------------------------------------
//
// `name<<<g, b[, s[, st]]>>>(args)` becomes
//
//   (grx::shim::push(g, b, s, st), name(args))
//
// A comma expression rather than a statement, because a launch is legal
// anywhere a call is -- including inside an if, which CUDA code does.

struct LaunchSite {
  std::string callee;
  int         line = 0;
};

struct LaunchRewrite {
  std::string output;
  int         count = 0;
  std::vector<LaunchSite> sites;
  std::vector<Diagnostic> errors;
};

LaunchRewrite rewrite_launches(const std::string& src) {
  LaunchRewrite r;
  r.output.reserve(src.size() + src.size() / 8);

  size_t i = 0;
  while (i < src.size()) {
    const size_t skipped = skip_noncode(src, i);
    if (skipped != i) { r.output.append(src, i, skipped - i); i = skipped; continue; }

    if (src.compare(i, 3, "<<<") != 0) { r.output += src[i++]; continue; }

    // The kernel name is the identifier immediately before, which is already
    // in the output. Take it back out: it becomes the callee of the rewritten
    // call rather than a prefix of it.
    // Qualified names count: `ns::kernel<<<...>>>` has to take `ns::kernel` as
    // the callee, not `kernel`, or the `ns::` is left stranded in front of the
    // rewritten expression.
    size_t ne = r.output.size();
    while (ne > 0 && std::isspace((unsigned char)r.output[ne - 1])) --ne;
    size_t ns = ne;
    for (;;) {
      size_t start = ns;
      while (start > 0 && ident_char(r.output[start - 1])) --start;
      if (start == ns) break;
      ns = start;
      if (ns >= 2 && r.output[ns - 1] == ':' && r.output[ns - 2] == ':') {
        ns -= 2;
        continue;
      }
      break;
    }
    const std::string callee = r.output.substr(ns, ne - ns);
    if (callee.empty()) {
      r.errors.push_back({line_of(src, i),
          "<<< with no kernel name before it"});
      r.output += src[i++];
      continue;
    }

    const size_t close = src.find(">>>", i + 3);
    if (close == std::string::npos) {
      r.errors.push_back({line_of(src, i), "unterminated <<<"});
      break;
    }
    const std::string cfg = src.substr(i + 3, close - (i + 3));
    const std::vector<std::string> parts = split_params(cfg);
    if (parts.size() < 2 || parts.size() > 4) {
      r.errors.push_back({line_of(src, i),
          "a launch configuration takes 2 to 4 arguments, found " +
          std::to_string(parts.size())});
      r.output += src[i++];
      continue;
    }

    size_t call = close + 3;
    while (call < src.size() && std::isspace((unsigned char)src[call])) ++call;
    if (call >= src.size() || src[call] != '(') {
      r.errors.push_back({line_of(src, i),
          "expected an argument list after >>>"});
      r.output += src[i++];
      continue;
    }
    const size_t call_end = match_bracket(src, call);
    if (call_end == std::string::npos) {
      r.errors.push_back({line_of(src, i), "unterminated argument list"});
      break;
    }
    const std::string args = src.substr(call, call_end - call);

    r.output.resize(ns);
    r.output += "(::grx::shim::push(::grx::shim::as_dim(" + trim(parts[0]) +
                "), ::grx::shim::as_dim(" + trim(parts[1]) + "), " +
                (parts.size() > 2 ? "(size_t)(" + trim(parts[2]) + ")" : "(size_t)0") +
                ", " +
                (parts.size() > 3 ? "::grx::shim::as_stream(" + trim(parts[3]) + ")"
                                  : "(grxStream_t)0") +
                "), " + callee + args + ")";
    r.sites.push_back({callee, line_of(src, i)});
    ++r.count;
    i = call_end;
  }
  return r;
}

// ---------------------------------------------------------------------------
// Emitters
// ---------------------------------------------------------------------------

std::string args_struct_name(const Kernel& k) { return "__grx_args_" + k.name; }

std::string smem_struct_name(const Kernel& k) { return "__grx_smem_" + k.name; }

// ---------------------------------------------------------------------------
// Static __shared__
// ---------------------------------------------------------------------------
//
// `__shared__ float tile[TILE][TILE + 1];` is in five of the eleven programs in
// tests/cuda_samples, and until now GRXCP answered it with a compile error: the
// definition that used to exist put the array in a section the device link
// script does not mention, so it landed in GLOBAL memory, compiled, ran, and
// was not shared memory. Refusing was the honest fix. Implementing it is the
// better one, and the missing piece turned out to be small.
//
// grxcc collects a kernel's static __shared__ declarations into one struct,
// places that struct over the CTA's local-memory slot, and replaces each
// declaration with a reference to the matching member:
//
//     struct __grx_smem_k { float tile[4][5]; float sums[32]; };
//     ...
//     auto& tile = ((__grx_smem_k*)::grx::shared_memory<void>())->tile;
//
// The compiler computes every size and offset from the user's own declaration
// text -- grxcc never parses a type. That is the same trick the argument struct
// uses, and it is why `float tile[TILE][TILE + 1]` works without grxcc knowing
// what TILE is.
//
// sizeof(__grx_smem_k) then goes into the kernel descriptor's `static_smem`,
// which src/runtime/launch.cpp ALREADY adds to the CTA's lmem_size:
//
//     info.lmem_size = (uint32_t)(shared + k.static_smem);
//
// That line has been there since phase 1 and nothing had ever set the field.
// The struct is emitted into both passes, so the host's sizeof and the device's
// offsets come from one declaration compiled twice.
//
// DYNAMIC shared memory -- `extern __shared__ float sdata[];` -- is the
// launch's sharedMem argument, and it lands AFTER the static block:
//
//     float* sdata = (float*)((char*)::grx::shared_memory<void>()
//                             + sizeof(__grx_smem_k));
//
// A kernel with both gets both, in that order, which is the layout the runtime
// is already reserving.

// The body, with each __shared__ declaration replaced by a reference into the
// CTA's local-memory slot.
std::string rewrite_shared(const std::string& body,
                           const std::vector<SharedDecl>& decls,
                           const std::string& smem_struct, bool has_static) {
  std::string out;
  size_t cursor = 0;
  for (const SharedDecl& d : decls) {
    out.append(body, cursor, d.start - cursor);
    if (d.is_extern) {
      // Dynamic shared memory starts where the static block ends. When there
      // is no static block the offset is zero and no struct is emitted --
      // sizeof of an empty struct is 1, not 0, and a one-byte shift of every
      // dynamic allocation is the kind of bug that shows up as an off-by-one
      // in someone else's kernel a month later.
      out += d.decl + "* " + d.name + " = (" + d.decl + "*)((char*)" +
             "::grx::shared_memory<void>()" +
             (has_static ? (" + sizeof(" + smem_struct + ")") : "") + ");";
    } else {
      // `auto&` so grxcc never has to name the type it did not parse.
      out += "auto& " + d.name + " = ((" + smem_struct + "*)" +
             "::grx::shared_memory<void>())->" + d.name + ";";
    }
    cursor = d.end;
  }
  out.append(body, cursor, std::string::npos);
  return out;
}

// The per-kernel static shared block, emitted into both passes so the host's
// sizeof and the device's member offsets come from one declaration.
//
// alignas(16) is not decoration: dynamic shared memory begins at
// sizeof(this struct), and without a known alignment a struct ending on an odd
// byte would hand `extern __shared__ double[]` a misaligned pointer.
std::string emit_smem_struct(const Kernel& k) {
  std::string s = "struct alignas(16) " + smem_struct_name(k) + " {\n";
  bool any = false;
  for (const SharedDecl& d : k.shared)
    if (!d.is_extern) { s += "  " + d.decl + ";\n"; any = true; }
  if (!any) return "";
  s += "};\n";
  return s;
}

std::string emit_args_struct(const Kernel& k) {
  std::string s = "struct " + args_struct_name(k) + " {\n";
  for (const Param& p : k.params) s += "  " + p.decl + ";\n";
  if (k.params.empty()) s += "  char __grx_empty;\n";
  s += "};\n";
  return s;
}

// The device pass: the user's body becomes an inlined function, and the entry
// point is a one-pointer kernel that unpacks the blob into it. See the header
// comment for why the wrapper is not optional.
std::string emit_device_kernel(const Kernel& k, const std::string& body) {
  const std::string st = args_struct_name(k);
  std::string s = emit_args_struct(k);
  s += emit_smem_struct(k);
  const std::string rewritten =
      k.shared.empty() ? body
                       : rewrite_shared(body, k.shared, smem_struct_name(k),
                                        k.has_static_shared());
  s += "static __attribute__((always_inline)) inline void __grx_body_" + k.name + "(";
  for (size_t i = 0; i < k.params.size(); ++i) {
    if (i) s += ", ";
    s += k.params[i].decl;
  }
  s += ") " + rewritten + "\n";
  s += "__global__ void " + k.name + "(" + st + "* __UNIFORM__ __grx_a) {\n";
  s += "  __grx_body_" + k.name + "(";
  for (size_t i = 0; i < k.params.size(); ++i) {
    if (i) s += ", ";
    s += "__grx_a->" + k.params[i].name;
  }
  s += ");\n}\n";
  return s;
}

// The host pass: the same signature, a body that packs and launches.
std::string emit_host_stub(const Kernel& k) {
  std::string s = emit_args_struct(k);
  // The same static-shared struct the device pass declares. The host does not
  // use its members -- it needs its SIZE, for the descriptor -- and getting
  // that from one declaration compiled twice is the same discipline the
  // argument struct follows.
  s += emit_smem_struct(k);
  s += "void " + k.name + "(";
  for (size_t i = 0; i < k.params.size(); ++i) {
    if (i) s += ", ";
    s += k.params[i].decl;
  }
  s += ") {\n";
  s += "  void* __grx_argv[] = {";
  for (size_t i = 0; i < k.params.size(); ++i) {
    if (i) s += ", ";
    s += "(void*)&" + k.params[i].name;
  }
  if (k.params.empty()) s += "nullptr";
  s += "};\n";
  s += "  dim3_t __grx_g, __grx_b; size_t __grx_s; grxStream_t __grx_st;\n";
  s += "  if (__grxPopCallConfiguration(&__grx_g, &__grx_b, &__grx_s, "
       "&__grx_st) != grxSuccess) return;\n";
  s += "  grxLaunchKernel((const void*)&" + k.name + ", __grx_g, __grx_b, "
       "__grx_argv, __grx_s, __grx_st);\n";
  s += "}\n";
  return s;
}

// The registration constructor, appended to the host pass rather than put in a
// TU of its own: the stub and its parameter types are already in scope here,
// and a separate TU would have to redeclare both.
std::string emit_registration(const std::vector<Kernel>& kernels,
                              const std::vector<DeviceVar>& vars,
                              const std::string& fatbin_symbol) {
  std::string s;
  s += "\n// ---- generated by grxcc: fat binary registration ----\n";
  s += "namespace {\n";
  for (const Kernel& k : kernels) {
    // These tables sit at file scope, so every name they mention needs the
    // kernel's namespace in front of it. The kernel's own NAME in the
    // descriptor stays unqualified: it is the device entry-point symbol, and
    // build_kernel.sh derives that from the __global__ alone.
    const std::string st = k.scope + args_struct_name(k);
    s += "const grx_kernel_param __grx_p_" + k.name + "[] = {\n";
    for (const Param& p : k.params) {
      s += "  { (uint16_t)__builtin_offsetof(" + st + ", " + p.name + "), "
           "(uint16_t)sizeof(((" + st + "*)0)->" + p.name + "), " +
           (p.is_pointer ? "1" : "0") + ", {0,0,0} },\n";
    }
    if (k.params.empty()) s += "  { 0, 0, 0, {0,0,0} },\n";
    s += "};\n";
    // __launch_bounds__ goes in as the SOURCE EXPRESSION the author wrote, cast
    // here rather than evaluated by grxcc. The runtime refuses a launch whose
    // block exceeds it (src/runtime/launch.cpp), which is the whole behavioural
    // content of the attribute on this hardware.
    const std::string bounds =
        k.launch_bounds.empty() ? "0u"
                                : "(uint32_t)(" + k.launch_bounds + ")";
    const std::string static_smem =
        k.has_static_shared()
            ? "(uint32_t)sizeof(" + k.scope + smem_struct_name(k) + ")"
            : "0u";
    s += "const grx_kernel_desc __grx_d_" + k.name + " = { \"" + k.name +
         "\", __grx_p_" + k.name + ", " + std::to_string(k.params.size()) +
         "u, (uint32_t)sizeof(" + st + "), " + static_smem + ", " +
         std::to_string(k.num_regs) + ", " + bounds + ", 0u };\n";
  }
  for (const DeviceVar& v : vars) {
    if (v.vma == 0) continue;   // not found in the ELF; see the caller
    s += "const grx_var_desc __grx_v_" + v.name + " = { \"" + v.name +
         "\", " + std::to_string(v.vma) + "ull, " + std::to_string(v.size) +
         "u, " + (v.is_constant ? "1u" : "0u") + " };\n";
  }
  s += "struct __grx_registrar {\n";
  s += "  void** handle;\n";
  s += "  __grx_registrar() {\n";
  s += "    handle = __grxRegisterFatBinary((void*)" + fatbin_symbol + ");\n";
  for (const Kernel& k : kernels) {
    s += "    __grxRegisterKernelDesc(handle, (const char*)(const void*)&" +
         k.qualified() + ", &__grx_d_" + k.name + ");\n";
  }
  for (const DeviceVar& v : vars) {
    if (v.vma == 0) continue;
    s += "    __grxRegisterVar(handle, (const void*)&" + v.name +
         ", &__grx_v_" + v.name + ");\n";
  }
  s += "  }\n";
  s += "  ~__grx_registrar() { __grxUnregisterFatBinary(handle); }\n";
  s += "};\n";
  s += "__grx_registrar __grx_registrar_instance;\n";
  s += "}  // namespace\n";
  return s;
}

// ---------------------------------------------------------------------------
// Files and processes
// ---------------------------------------------------------------------------

bool read_file(const std::string& path, std::string* out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) { std::fclose(f); return false; }
  out->resize((size_t)n);
  const size_t got = std::fread(&(*out)[0], 1, (size_t)n, f);
  std::fclose(f);
  return got == (size_t)n;
}

bool write_file(const std::string& path, const std::string& data) {
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const size_t n = std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  return n == data.size();
}

int run(const std::vector<std::string>& argv, bool verbose) {
  if (verbose) {
    std::fprintf(stderr, "grxcc:");
    for (const std::string& a : argv) std::fprintf(stderr, " %s", a.c_str());
    std::fprintf(stderr, "\n");
  }
  std::vector<char*> raw;
  for (const std::string& a : argv) raw.push_back(const_cast<char*>(a.c_str()));
  raw.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) return -1;
  if (pid == 0) {
    ::execvp(raw[0], raw.data());
    std::fprintf(stderr, "grxcc: cannot run %s: %s\n", raw[0], std::strerror(errno));
    ::_exit(127);
  }
  int status = 0;
  ::waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128;
}

// Run a command and capture its standard output. Errors go to grxcc's own
// stderr rather than being swallowed, so a missing llvm-objdump says so.
bool run_capture(const std::string& cmd, std::string* out, bool verbose) {
  if (verbose) std::fprintf(stderr, "grxcc: %s\n", cmd.c_str());
  std::FILE* f = ::popen(cmd.c_str(), "r");
  if (!f) return false;
  out->clear();
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
  return ::pclose(f) == 0;
}

// ---------------------------------------------------------------------------
// Register measurement
// ---------------------------------------------------------------------------
//
// grxFuncGetAttributes.numRegs has reported -1 since phase 1, because nothing
// in the toolchain emits a per-kernel register count: there is no ptxas here to
// print one, and the .vxbin footer carries entry points and nothing else.
//
// What it CAN be measured from is the object code, so that is what grxcc does:
// disassemble the device ELF, walk each kernel's reachable call graph, and
// count the distinct architectural registers that appear as operands.
//
// The definition is narrow on purpose, and stated in cuda_mapping.md so nobody
// has to infer it:
//
//   - Integer and floating-point registers are counted TOGETHER, because a
//     GRX-G100 thread has one of each file and CUDA's single number has no
//     place to put two.
//   - x0 (`zero`) is excluded. It is a wire, not storage.
//   - The count covers the kernel entry point and every function reachable from
//     it by a DIRECT call.
//   - An INDIRECT call (jalr), or a direct call to a symbol not in this image,
//     makes the count unknowable, and the kernel reports -1 rather than a
//     number that is really a lower bound. A lower bound presented as a
//     measurement is the failure mode this whole exercise exists to avoid.
//
// It is worth saying what the number does NOT do here: it does not bound
// occupancy. src/runtime/launch.cpp's resident_blocks_per_sm has no register
// term, because the CTA dispatcher does not gate admission on register count
// the way CUDA's does. The number is a fact about the code, not an input to a
// scheduling decision.

struct DisasmFunc {
  std::vector<bool>        regs;          // 64 slots: x0-x31 then f0-f31
  std::vector<std::string> calls;         // direct call targets, by symbol
  bool                     indirect = false;
};

// Map a RISC-V ABI register name to a slot, or -1 if it is not a register.
// Names as llvm-objdump prints them.
int reg_slot(const std::string& t) {
  static const char* kInt[32] = {
      "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2",
      "s0",   "s1", "a0", "a1", "a2", "a3", "a4", "a5",
      "a6",   "a7", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8",   "s9", "s10", "s11", "t3", "t4", "t5", "t6"};
  static const char* kFlt[32] = {
      "ft0", "ft1", "ft2",  "ft3",  "ft4", "ft5", "ft6", "ft7",
      "fs0", "fs1", "fa0",  "fa1",  "fa2", "fa3", "fa4", "fa5",
      "fa6", "fa7", "fs2",  "fs3",  "fs4", "fs5", "fs6", "fs7",
      "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"};
  for (int i = 0; i < 32; ++i) if (t == kInt[i]) return i;
  for (int i = 0; i < 32; ++i) if (t == kFlt[i]) return 32 + i;
  if (t == "fp") return 8;         // s0's other name
  return -1;
}

// Parse `llvm-objdump -d` output into a per-function register and call-target
// map. The format relied on is minimal: a line `<addr> <name>:` opens a
// function, and instruction lines are `<addr>: <bytes>\t<mnemonic>\t<operands>`.
void parse_disasm(const std::string& text,
                  std::vector<std::pair<std::string, DisasmFunc>>* out) {
  DisasmFunc* cur = nullptr;
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t nl = text.find('\n', pos);
    if (nl == std::string::npos) nl = text.size();
    const std::string line = text.substr(pos, nl - pos);
    pos = nl + 1;
    if (line.empty()) continue;

    // A function header: "0000000180000100 <__vx_kentry_nested>:"
    const size_t lt = line.find('<');
    const size_t gt = line.find(">:");
    if (lt != std::string::npos && gt != std::string::npos && gt > lt &&
        line.find(':') == gt + 1) {
      out->push_back({line.substr(lt + 1, gt - lt - 1), DisasmFunc{}});
      cur = &out->back().second;
      cur->regs.assign(64, false);
      continue;
    }
    if (!cur) continue;

    const size_t colon = line.find(':');
    if (colon == std::string::npos) continue;
    std::string body = line.substr(colon + 1);

    // Drop the encoded bytes: the first whitespace-delimited field.
    size_t b = body.find_first_not_of(" \t");
    if (b == std::string::npos) continue;
    const size_t be = body.find_first_of(" \t", b);
    if (be == std::string::npos) continue;
    body = body.substr(be);

    // A call target, if this is one: llvm-objdump appends "<symbol>".
    const size_t tl = body.find('<');
    const size_t tg = body.find('>', tl == std::string::npos ? 0 : tl);

    // Tokenise on anything that cannot appear in a register name.
    std::string tok;
    bool saw_jal = false, saw_jalr = false;
    for (size_t i = 0; i <= body.size(); ++i) {
      const char c = (i < body.size()) ? body[i] : ' ';
      if (ident_char(c) || c == '.') { tok += c; continue; }
      if (!tok.empty()) {
        if (tok == "jal") saw_jal = true;
        else if (tok == "jalr") saw_jalr = true;
        const int slot = reg_slot(tok);
        if (slot > 0) cur->regs[(size_t)slot] = true;   // slot 0 is x0
        tok.clear();
      }
    }
    if (saw_jalr) cur->indirect = true;
    if (saw_jal) {
      if (tl != std::string::npos && tg != std::string::npos && tg > tl)
        cur->calls.push_back(body.substr(tl + 1, tg - tl - 1));
      else
        cur->indirect = true;   // a jal grxcc cannot resolve is not a fact
    }
  }
}

// Registers reachable from one of `entries`, or -1 if the call graph does not
// close.
//
// Several names, because the entry point has two. build_kernel.sh emits both
// `__vx_kentry_<name>` and `<name>` at the same address, and llvm-objdump
// labels the block with whichever alias it happens to pick -- it prints
// `<axpy>` for a symbol that llvm-nm lists under both. Looking for one spelling
// and giving up found nothing, and reported -1 for every kernel, which looked
// exactly like the honest "cannot measure this" answer. Trying both is the
// difference between a measurement and a sentinel that agrees with it by
// accident.
int32_t count_registers(
    const std::vector<std::pair<std::string, DisasmFunc>>& funcs,
    const std::vector<std::string>& entries) {
  std::string entry;
  for (const std::string& cand : entries) {
    for (const auto& e : funcs) if (e.first == cand) { entry = cand; break; }
    if (!entry.empty()) break;
  }
  if (entry.empty()) return -1;

  std::vector<bool> seen_regs(64, false);
  std::vector<std::string> work{entry}, visited;
  bool found_entry = false;
  while (!work.empty()) {
    const std::string name = work.back();
    work.pop_back();
    bool already = false;
    for (const std::string& v : visited) if (v == name) { already = true; break; }
    if (already) continue;
    visited.push_back(name);

    const DisasmFunc* f = nullptr;
    for (const auto& e : funcs) if (e.first == name) { f = &e.second; break; }
    if (!f) return -1;              // a call into something we cannot see
    if (name == entry) found_entry = true;
    if (f->indirect) return -1;
    for (size_t i = 1; i < 64; ++i) if (f->regs[i]) seen_regs[i] = true;
    for (const std::string& c : f->calls) work.push_back(c);
  }
  if (!found_entry) return -1;
  int32_t n = 0;
  for (size_t i = 1; i < 64; ++i) if (seen_regs[i]) ++n;
  return n;
}

// The .vxbin, wrapped in a .grxfat container and spelled as a C array. An
// object file made with objcopy would be smaller to compile, but it would also
// make grxcc depend on a particular objcopy's section naming; a byte array
// works with any host compiler, which is what a driver should prefer.
std::string emit_fatbin_array(const std::string& vxbin, const std::string& symbol,
                              uint32_t required_isa) {
  grx_fatbin_header h{};
  h.magic       = GRX_FATBIN_MAGIC;
  h.version     = GRX_FATBIN_VERSION;
  h.num_entries = 1;

  grx_fatbin_entry e{};
  e.kind         = GRX_IMAGE_VXBIN;
  e.xlen         = 64;
  e.required_isa = required_isa;
  e.offset       = sizeof(h) + sizeof(e);
  e.size         = vxbin.size();
  h.total_size   = e.offset + vxbin.size();

  std::string blob;
  blob.append((const char*)&h, sizeof(h));
  blob.append((const char*)&e, sizeof(e));
  blob += vxbin;

  std::string s = "\n// ---- generated by grxcc: device image ----\n";
  s += "alignas(8) static const unsigned char " + symbol + "[] = {\n";
  char buf[8];
  for (size_t i = 0; i < blob.size(); ++i) {
    std::snprintf(buf, sizeof(buf), "0x%02x,", (unsigned char)blob[i]);
    s += buf;
    if ((i % 16) == 15) s += "\n";
  }
  s += "\n};\n";
  return s;
}

// A positional argument that is not the source file. A compiler driver gets
// handed object files and archives alongside the source it compiles -- that is
// how every build system in existence invokes one -- and rejecting them as "a
// second source file" would force the caller to run the link by hand, which
// defeats the point of having a driver.
bool is_link_input(const std::string& a) {
  static const char* kExt[] = {".o", ".a", ".so", ".obj", ".lib"};
  for (const char* e : kExt) {
    const size_t n = std::strlen(e);
    if (a.size() > n && a.compare(a.size() - n, n, e) == 0) return true;
  }
  return a.find(".so.") != std::string::npos;   // libfoo.so.1
}

void usage() {
  std::fprintf(stderr,
      "usage: grxcc [options] <source> [objects...] -o <output>\n"
      "\n"
      "  --grxgpu <path>    grxgpu checkout, for the device compile (required)\n"
      "  --tooldir <path>   device toolchain (default $TOOLDIR or $HOME/tools)\n"
      "  --build-kernel <p> ci/build_kernel.sh to shell out to (or\n"
      "                     $GRXCC_BUILD_KERNEL; grxcc deliberately shares the\n"
      "                     repo's one device-build recipe rather than keeping\n"
      "                     a second copy of the flags)\n"
      "  -I <dir>           include directory, passed to both passes\n"
      "  -L <dir>, -l<lib>  passed to the host link\n"
      "  <file>.o|.a|.so    passed to the host link\n"
      "  -c                 compile the host pass only, do not link\n"
      "  --keep             leave the generated sources in place\n"
      "  --emit-only        write the generated sources and stop\n"
      "  -v                 print every command\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string source, output, grxgpu, tooldir, build_kernel;
  std::vector<std::string> includes, link_flags, extra_host;
  bool verbose = false, keep = false, emit_only = false, compile_only = false;

  if (const char* t = std::getenv("TOOLDIR")) tooldir = t;
  if (tooldir.empty()) {
    if (const char* h = std::getenv("HOME")) tooldir = std::string(h) + "/tools";
  }
  if (const char* g = std::getenv("GRXGPU")) grxgpu = g;
  if (const char* b = std::getenv("GRXCC_BUILD_KERNEL")) build_kernel = b;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-o" && i + 1 < argc)            output = argv[++i];
    else if (a == "--grxgpu" && i + 1 < argc)  grxgpu = argv[++i];
    else if (a == "--tooldir" && i + 1 < argc) tooldir = argv[++i];
    else if (a == "--build-kernel" && i + 1 < argc) build_kernel = argv[++i];
    else if (a == "-I" && i + 1 < argc)        includes.push_back(argv[++i]);
    else if (a.compare(0, 2, "-I") == 0)       includes.push_back(a.substr(2));
    else if (a == "-L" && i + 1 < argc)        link_flags.push_back("-L" + std::string(argv[++i]));
    else if (a.compare(0, 2, "-L") == 0 || a.compare(0, 2, "-l") == 0)
      link_flags.push_back(a);
    else if (a == "-c")                        compile_only = true;
    else if (a == "--keep")                    keep = true;
    else if (a == "--emit-only")               emit_only = true;
    else if (a == "-v")                        verbose = true;
    else if (a == "-h" || a == "--help")       { usage(); return 0; }
    else if (a.compare(0, 1, "-") == 0)        extra_host.push_back(a);
    else if (is_link_input(a))                 link_flags.push_back(a);
    else if (source.empty())                   source = a;
    else { std::fprintf(stderr, "grxcc: more than one source file\n"); return 2; }
  }

  if (source.empty()) { usage(); return 2; }
  if (output.empty()) output = compile_only ? "a.o" : "a.out";

  std::string src;
  if (!read_file(source, &src)) {
    std::fprintf(stderr, "grxcc: cannot read %s\n", source.c_str());
    return 2;
  }

  // ---- 1. rewrite -----------------------------------------------------------
  const Parsed parsed = find_kernels(src);
  for (const Diagnostic& d : parsed.errors)
    std::fprintf(stderr, "%s:%d: error: %s\n", source.c_str(), d.line,
                 d.text.c_str());
  if (!parsed.errors.empty()) return 1;
  // No warning for a file with no kernels. Most translation units of a real
  // program have none, and the summary line below already says "0 kernels".

  // Both passes start from the same launch-rewritten text, so a <<<>>> that
  // appears inside a kernel body -- which is dynamic parallelism, and not
  // supported -- fails the device compile rather than being silently dropped
  // from one pass.
  const LaunchRewrite lr = rewrite_launches(src);
  for (const Diagnostic& d : lr.errors)
    std::fprintf(stderr, "%s:%d: error: %s\n", source.c_str(), d.line,
                 d.text.c_str());
  if (!lr.errors.empty()) return 1;

  // Every <<<>>> must name a __global__ DEFINED IN THIS FILE.
  //
  // grxcc builds the host stub from the kernel's body, so a declaration alone
  // is not enough -- unlike nvcc, which can call a kernel defined in another
  // translation unit because its fat binary is linked, not generated per file.
  // Catching it here is the difference between one line naming the symbol and a
  // wall of template errors out of the device compile, which is what the
  // generated text produces if this check is skipped.
  {
    bool unknown = false;
    for (const LaunchSite& s : lr.sites) {
      // `::kernel`, `ns::kernel` and -- from inside `namespace ns` -- a bare
      // `kernel` all name the same thing. grxcc is not a name resolver, so it
      // accepts any of the three: a launch that matches no kernel at all is the
      // error worth catching, and one that matches the wrong overload of a name
      // is a problem grxcc does not have, having refused templates already.
      const std::string bare =
          s.callee.compare(0, 2, "::") == 0 ? s.callee.substr(2) : s.callee;
      bool found = false;
      for (const Kernel& k : parsed.kernels)
        if (bare == k.name || bare == k.qualified()) { found = true; break; }
      if (found) continue;
      unknown = true;
      std::fprintf(stderr,
                   "%s:%d: error: launched `%s` with <<<>>>, but no "
                   "__global__ function of that name is defined in this file\n",
                   source.c_str(), s.line, s.callee.c_str());
    }
    if (unknown) {
      if (parsed.kernels.empty()) {
        std::fprintf(stderr,
                     "grxcc: this file defines no __global__ functions at all. "
                     "A kernel grxcc launches has to be compiled by grxcc.\n");
      } else {
        std::fprintf(stderr, "grxcc: this file defines:");
        for (const Kernel& k : parsed.kernels)
          std::fprintf(stderr, " %s", k.name.c_str());
        std::fprintf(stderr, "\n");
      }
      return 1;
    }
  }

  // __launch_bounds__'s second argument asks the compiler to spill enough that
  // N blocks fit per SM. There is nothing here to ask: GRX-G100's CTA
  // dispatcher does not gate admission on register count, so the request is
  // neither honoured nor needed -- see resident_blocks_per_sm in
  // src/runtime/launch.cpp, which has no register term. Saying so beats
  // accepting it silently, which would let an author believe a spill happened.
  for (const Kernel& k : parsed.kernels)
    if (k.launch_bounds_had_min)
      std::fprintf(stderr,
                   "%s:%d: note: __launch_bounds__ on `%s` asks for a minimum "
                   "blocks-per-SM; GRX-G100 does not bound occupancy by "
                   "register count, so there is nothing to trade against it. "
                   "The maximum thread count IS enforced.\n",
                   source.c_str(), k.line, k.name.c_str());

  // The kernels were located in the ORIGINAL text; relocate them in the
  // rewritten one by finding each body again. Simpler and more robust than
  // threading offsets through the rewrite: kernel bodies are unique text.
  Parsed relocated = find_kernels(lr.output);
  std::vector<DeviceVar> device_vars = find_device_vars(lr.output);
  if (relocated.kernels.size() != parsed.kernels.size()) {
    std::fprintf(stderr,
                 "grxcc: internal error: %zu kernels before the launch rewrite "
                 "and %zu after\n", parsed.kernels.size(),
                 relocated.kernels.size());
    return 1;
  }

  // The two passes are assembled from explicit pieces rather than by slicing a
  // shared buffer, because they do not cover the same span of the file.
  //
  //   host   = [0 .. end of file), kernel bodies replaced by launch stubs
  //   device = [0 .. insertion point) + device header
  //            + [insertion point .. last kernel's closing brace), bodies kept
  //            + whatever braces that span left open
  //
  // THE DEVICE PASS STOPS AT THE LAST KERNEL.
  //
  // A frontend compiles the whole file for the device and simply does not
  // diagnose host-only function bodies it never needs. An orchestrator has no
  // such option: it hands clang the text, and `main`'s std::printf is an error
  // for a bare-metal target whether or not the device will ever run it.
  //
  // So the device pass keeps the file up to the end of the last __global__ and
  // drops the rest. That makes "device code first, host code after" a rule of
  // grxcc rather than a convention -- which it already is in most CUDA files --
  // and it fails visibly (an undeclared name at the device compile) rather than
  // quietly if a kernel needed something declared below it.
  //
  // THE DEVICE HEADER GOES AT THE LAST FILE-SCOPE BOUNDARY BEFORE THE FIRST
  // KERNEL, which is later than the top of the file and earlier than the kernel
  // itself. Two constraints have to hold at once:
  //
  //   After the user's includes. The device header defines `printf` and
  //   `assert` as macros -- which is what makes a kernel able to call them --
  //   and those macros poison <cstdio> if the standard header is parsed
  //   afterwards.
  //
  //   At file scope. "Immediately before the first kernel" fails the moment a
  //   kernel lives inside `namespace ns {`, where it would put an #include
  //   inside the namespace and rename everything in it.
  //
  // The last top-level boundary before the kernel is the only point that
  // satisfies both.
  //
  // The same nesting is why the device pass appends closing braces: cutting at
  // the last kernel's `}` can leave a namespace or `extern "C" {` open.
  std::string device_head, device_body, host_src;
  {
    const size_t device_start =
        relocated.kernels.empty()
            ? lr.output.size()
            : top_level_boundary_before(lr.output,
                                        first_device_token(lr.output));
    device_head.assign(lr.output, 0, device_start);

    // `__device__`-only definitions are dropped from the host pass; the device
    // pass keeps them, which is the point of them.
    const std::vector<Span> device_only = find_device_only(lr.output);
    auto append_host = [&](size_t from, size_t to) {
      size_t at = from;
      for (const Span& sp : device_only) {
        if (sp.end <= at || sp.start >= to) continue;
        if (sp.start > at) host_src.append(lr.output, at, sp.start - at);
        at = sp.end;
      }
      if (to > at) host_src.append(lr.output, at, to - at);
    };

    size_t cursor = 0;        // host: the whole file
    size_t dcursor = device_start;   // device: from the insertion point on
    for (const Kernel& k : relocated.kernels) {
      append_host(cursor, k.decl_start);
      device_body.append(lr.output, dcursor, k.decl_start - dcursor);
      const std::string body =
          lr.output.substr(k.body_start, k.body_end - k.body_start);
      device_body += emit_device_kernel(k, body);
      host_src    += emit_host_stub(k);
      cursor  = k.body_end;
      dcursor = k.body_end;
    }
    append_host(cursor, lr.output.size());

    // Close whatever the cut left open. A kernel inside `namespace ns {` ends
    // with the namespace still unclosed, and the device compile would report an
    // error at end-of-file that names nothing the author wrote.
    if (!relocated.kernels.empty()) {
      const int open = brace_depth(lr.output, relocated.kernels.back().body_end);
      if (open > 0) {
        device_body += "\n// grxcc: closing " + std::to_string(open) +
                       " scope(s) the device pass cut through\n";
        for (int d = 0; d < open; ++d) device_body += "}\n";
      }
    }
  }

  // The host pass needs the runtime, the ABI structures and the launch shim.
  // Prepending rather than requiring the user to include them: generated code
  // should not impose includes the author did not write.
  host_src = "#include <grx/grx.h>\n#include <grx/grx_abi.h>\n"
             "#include <grx/grx_launch_shim.h>\n"
             "namespace grx { namespace shim {\n"
             "inline grxError_t push(dim3_t g, dim3_t b, size_t s, grxStream_t st) {\n"
             "  return __grxPushCallConfiguration(g, b, s, st);\n"
             "}\n}}\n"
             "#define __GRX_HOST_PASS__ 1\n"
             // A __device__ or __constant__ VARIABLE is declared on both
             // sides: the device gets the storage, the host gets an object of
             // the same name whose address is the key cudaMemcpyToSymbol
             // passes. So the host pass keeps the declaration and needs the
             // qualifiers to mean nothing. __device__ FUNCTIONS are a
             // different case and are dropped entirely.
             "#define __device__\n"
             "#define __constant__\n"
             "#define __forceinline__ inline\n" + host_src;

  // WHAT THE DEVICE PASS INCLUDES, AND WHY IT IS MORE THAN ONE HEADER.
  //
  // A CUDA file writes `__shfl_down_sync`, `cooperative_groups::reduce` and
  // `atomicAdd` without including anything for them: the frontend supplies
  // those names. grxcc has no frontend, so it supplies them here. A user who
  // had to add `#include <grx/device/grx_warp.h>` to get a shuffle would be
  // editing the file, which is exactly what tests/cuda_samples exists to
  // prevent.
  //
  // grx_atomic.h is included even on a build with no A extension: every entry
  // point in it is `unavailable` there, so including it costs nothing and turns
  // `undeclared identifier 'atomicAdd'` -- which sends the reader looking for a
  // missing include -- into a message about this device's configuration.
  //
  // grx_wmma.h, grx_pipeline.h and grx_cycles.h are NOT here. Those are GRX
  // APIs with no CUDA spelling a source file would already be using, so
  // including them would only slow every device compile down.
  const std::string device_src =
      "#define __GRX_DEVICE_PASS__ 1\n" + device_head +
      "\n#include <grx/device/grx_device.h>\n"
      "#include <grx/device/grx_warp.h>\n"
      "#include <grx/device/grx_cg.h>\n"
      "#include <grx/device/grx_atomic.h>\n" +
      device_body;

  const std::string stem = output + ".grxcc";
  const std::string dev_path  = stem + ".dev.cpp";
  const std::string host_path = stem + ".host.cpp";
  const std::string vxbin_path = stem + ".vxbin";

  if (!write_file(dev_path, device_src) || !write_file(host_path, host_src)) {
    std::fprintf(stderr, "grxcc: cannot write the generated sources\n");
    return 2;
  }
  std::fprintf(stderr, "grxcc: %zu kernel%s, %d launch%s rewritten\n",
               relocated.kernels.size(),
               relocated.kernels.size() == 1 ? "" : "s", lr.count,
               lr.count == 1 ? "" : "es");
  if (emit_only) {
    std::fprintf(stderr, "grxcc: wrote %s and %s\n", dev_path.c_str(),
                 host_path.c_str());
    return 0;
  }

  // ---- 2. device compile ----------------------------------------------------
  //
  // A file with no kernels has no device half. That is an ordinary situation in
  // a real build -- most translation units of a CUDA program contain no
  // __global__ at all -- so grxcc compiles the host pass and stops, rather than
  // demanding a --grxgpu it has no use for. There is nothing to register: the
  // kernels this file's code calls are registered by the file that defines
  // them.
  const bool has_device_half = !relocated.kernels.empty();
  if (has_device_half && grxgpu.empty()) {
    std::fprintf(stderr,
                 "grxcc: --grxgpu <path> is required for the device compile "
                 "(or set $GRXGPU).\n");
    return 2;
  }
  if (has_device_half) {
    std::string self_dir = ".";
    char self[4096];
    const ssize_t n = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n > 0) {
      self[n] = '\0';
      const std::string p(self);
      const size_t slash = p.find_last_of('/');
      if (slash != std::string::npos) self_dir = p.substr(0, slash);
    }
    // build_kernel.sh is the same script the library kernels are built with.
    // Sharing it is the point: a kernel compiled by grxcc and one compiled by
    // hand must come from the same flags and the same recorded configuration,
    // or "it works when I build it myself" becomes a real bug report.
    // Candidates, most specific first. grxcc shells out to the SAME script the
    // library kernels are built with rather than carrying its own copy of the
    // flags: a kernel compiled by grxcc and one compiled by hand must come
    // from one recipe, or "it works when I build it myself" becomes a real bug
    // report. The cost is that grxcc has to find the script, hence the flag.
    std::string script = build_kernel;
    if (script.empty()) {
      const std::string candidates[] = {
        self_dir + "/../ci/build_kernel.sh",
        self_dir + "/ci/build_kernel.sh",
        "ci/build_kernel.sh",
      };
      struct stat sb;
      for (const std::string& c : candidates)
        if (::stat(c.c_str(), &sb) == 0) { script = c; break; }
    }
    if (script.empty()) {
      std::fprintf(stderr,
                   "grxcc: cannot find ci/build_kernel.sh. Pass "
                   "--build-kernel <path> or set $GRXCC_BUILD_KERNEL.\n");
      return 2;
    }
    std::vector<std::string> cmd = {script,
                                    "--grxgpu", grxgpu,
                                    "--tooldir", tooldir};
    for (const std::string& inc : includes) { cmd.push_back("-I"); cmd.push_back(inc); }
    cmd.push_back(dev_path);
    cmd.push_back("-o");
    cmd.push_back(vxbin_path);

    const int rc = run(cmd, verbose);
    if (rc != 0) {
      std::fprintf(stderr, "grxcc: the device compile failed (%d)\n", rc);
      return rc;
    }

    // Measure each kernel's register use from the ELF build_kernel.sh left
    // beside the .vxbin. A failure here is not fatal: every kernel keeps
    // num_regs == -1, which is exactly what the field means, and the program
    // still builds. A driver that refused to compile because it could not
    // gather an OPTIONAL statistic would be trading a working program for a
    // number nobody asked for.
    {
      const std::string elf = vxbin_path.substr(0, vxbin_path.size() - 6) + ".elf";
      const std::string objdump = tooldir + "/llvm-vortex/bin/llvm-objdump";
      std::string text;
      struct stat sb;
      if (::stat(objdump.c_str(), &sb) != 0) {
        if (verbose)
          std::fprintf(stderr, "grxcc: no %s; kernels report numRegs = -1\n",
                       objdump.c_str());
      } else if (!run_capture(objdump + " -d " + elf + " 2>/dev/null", &text,
                              verbose)) {
        std::fprintf(stderr,
                     "grxcc: could not disassemble %s; kernels report "
                     "numRegs = -1\n", elf.c_str());
      } else {
        std::vector<std::pair<std::string, DisasmFunc>> funcs;
        parse_disasm(text, &funcs);
        for (Kernel& k : relocated.kernels)
          k.num_regs =
              count_registers(funcs, {"__vx_kentry_" + k.name, k.name});
      }
    }

    // Device variables: address and size, from the ELF's own symbol table.
    //
    // A __constant__ or __device__ declaration gives grxcc a NAME; only the
    // linker knows where it ended up and how big it is. llvm-nm prints both,
    // and the format relied on is minimal -- `<addr> <size> <type> <name>`.
    //
    // A variable that is declared but never referenced by a kernel is dropped
    // by --gc-sections and simply is not in the table. It gets no registration,
    // and grxMemcpyToSymbol on it reports grxErrorInvalidSymbol -- which is the
    // truth: there is no device symbol to copy to.
    if (!device_vars.empty()) {
      const std::string elf = vxbin_path.substr(0, vxbin_path.size() - 6) + ".elf";
      const std::string nm = tooldir + "/llvm-vortex/bin/llvm-nm";
      std::string text;
      struct stat sb;
      if (::stat(nm.c_str(), &sb) != 0) {
        std::fprintf(stderr,
                     "grxcc: no %s, so __device__/__constant__ symbols cannot "
                     "be registered; grxMemcpyToSymbol will not find them\n",
                     nm.c_str());
      } else if (run_capture(nm + " --print-size --defined-only " + elf +
                             " 2>/dev/null", &text, verbose)) {
        for (DeviceVar& v : device_vars) {
          size_t pos = 0;
          while (pos <= text.size()) {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos) nl = text.size();
            const std::string line = text.substr(pos, nl - pos);
            pos = nl + 1;
            if (line.size() < 20) continue;
            const size_t sp = line.rfind(' ');
            if (sp == std::string::npos || line.substr(sp + 1) != v.name) continue;
            v.vma  = std::strtoull(line.c_str(), nullptr, 16);
            v.size = (uint32_t)std::strtoull(line.c_str() + 17, nullptr, 16);
            break;
          }
          if (v.vma == 0)
            std::fprintf(stderr,
                         "%s:%d: warning: `%s` is not in the device image -- "
                         "declared but never used by a kernel, so the linker "
                         "dropped it. grxMemcpyToSymbol will not find it.\n",
                         source.c_str(), v.line, v.name.c_str());
        }
      }
    }
  }

  // ---- 3. fat binary + registration ----------------------------------------
  if (has_device_half) {
    std::string vxbin;
    if (!read_file(vxbin_path, &vxbin)) {
      std::fprintf(stderr, "grxcc: the device compile produced no %s\n",
                   vxbin_path.c_str());
      return 1;
    }
    host_src += emit_fatbin_array(vxbin, "__grx_fatbin_data", 0);
    host_src += emit_registration(relocated.kernels, device_vars,
                                  "__grx_fatbin_data");
    if (!write_file(host_path, host_src)) {
      std::fprintf(stderr, "grxcc: cannot rewrite %s\n", host_path.c_str());
      return 2;
    }
  }

  // ---- 4. host compile and link --------------------------------------------
  {
    const char* cxx = std::getenv("CXX");
    std::vector<std::string> cmd = {cxx ? cxx : "g++", "-std=c++17"};
    for (const std::string& inc : includes) cmd.push_back("-I" + inc);
    for (const std::string& f : extra_host)  cmd.push_back(f);
    if (compile_only) cmd.push_back("-c");
    cmd.push_back(host_path);
    cmd.push_back("-o");
    cmd.push_back(output);
    for (const std::string& f : link_flags) cmd.push_back(f);
    const int rc = run(cmd, verbose);
    if (rc != 0) {
      std::fprintf(stderr, "grxcc: the host compile failed (%d)\n", rc);
      return rc;
    }
  }

  if (!keep) {
    ::unlink(dev_path.c_str());
    ::unlink(host_path.c_str());
  }
  return 0;
}
