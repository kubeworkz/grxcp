#!/usr/bin/env python3
"""Check that every reference the documentation makes to this tree resolves.

This exists because a sweep done by hand found six dangling references that
several careful readings had not: `tests/bench/sgemm_cycles.cpp` renamed out
from under two documents, three GRXGPU paths written as if they were ours, and
a phase 6 exit clause quoting a Phase 4 target that was never set. None of them
were wrong when written. They rotted, quietly, because nothing checked them.

Reading does not catch this class of drift. Only a machine that opens every
path does. So:

  1. every `path/like/this` in backticks that names something in this tree
     resolves to a file or directory that exists;
  2. every markdown link to a local page resolves, relative to the page that
     makes the link;
  3. every reference of the form `some_doc.md 7.13` names a section that
     some_doc.md actually has;
  4. every bare `§7.13` or `(section 7.13)` names a section number that exists
     in SOME document here.

WHAT EACH CHECK DOES NOT PROVE, stated because a gate that overstates itself is
worse than no gate:

  * Check 1 adjudicates a path only when its first component is a directory of
    this repository. `grxgpu/...` and `c930/...` name other repositories and
    cannot be resolved from here; `hw/dpi` and `kernels/sgemm.cpp` are
    fragments relative to a tree this checkout does not contain. Those are
    counted and printed, not silently dropped -- if `tests/` were ever deleted,
    all 70-odd `tests/...` references would move from checked to unchecked and
    the count would say so.
  * Check 3 is skipped when the named .md is not in this repository, for the
    same reason.
  * Check 4 proves the number exists somewhere, NOT that the reference points
    at the right document. A bare `section 7.10` three lines under a sentence
    naming cuda_mapping.md is only readable as such by a human. Checking it
    against every document is the strongest thing that has no false positives,
    and it still catches the failure that matters: a section number that exists
    nowhere at all.
  * Fenced blocks are not read at all. They hold intended-layout diagrams and
    shell transcripts -- grxcp_architecture.md 9 lists a dozen files nobody has
    written yet -- and opening those would report a plan as a defect. This
    excludes nothing today: measured against the current tree, zero
    reference-shaped strings live inside a fence. It is a rule for the next
    person who puts one there.
  * Nothing here checks link ANCHORS, because no document currently uses one.
  * A bare `filename.md` in backticks, with no directory, is not checked --
    too many of those are other repositories' files.

--self-test plants one broken reference of each of the four kinds in a fixture
tree and fails unless all four are caught, because a gate nobody has watched
fail is not a gate. It runs first, every time.

  ./ci/check_docs.py [--list-unchecked] [--self-test]
"""

import pathlib
import re
import sys
import tempfile

# Build outputs hold copies of the sources and would be checked twice, with the
# second copy resolving against the wrong root.
SKIP_DIRS = {".git", "_sync", "_to_delete", "__pycache__"}

BACKTICK_PATH = re.compile(r"`([A-Za-z0-9_][A-Za-z0-9_./+*-]*/[A-Za-z0-9_./+*-]*)`")
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")

# A named reference takes two forms and they need different strictness. With an
# explicit marker -- `foo.md` §15, foo.md section 7.4 -- any number is a section
# number. Without one, only a DOTTED number is: "foo.md 7.13" is a reference,
# "foo.md and 3 others" is a sentence. Getting this wrong is not academic; the
# first version of this file demanded only dotted numbers, so `gpu_chip_design.md`
# §15 fell through to check 4 and was reported as a section existing in no
# document, when in truth it names a section of a document this checkout does
# not contain and cannot be adjudicated at all.
#
# The leading path is captured, not skipped over, because it decides WHOSE
# document is meant. `grxgpu/AGENTS.md` §4 must not be adjudicated against our
# AGENTS.md -- and would have been, silently and successfully, since ours does
# happen to have a section 4.
NAMED_SECTION = re.compile(
    r"`?((?:[A-Za-z0-9_./-]*/)?[A-Za-z0-9_]+\.md)`?[ ,]*"
    r"(?:(?:§ ?|section |sec\. )(\d+(?:\.\d+)*)|(\d+\.\d+))"
)
BARE_SECTION = re.compile(r"(?:§ ?|section |sec\. )(\d+(?:\.\d+)*)")
HEADING = re.compile(r"^#{2,6}\s+(\d+(?:\.\d+)*)\.?(?:\s|$)")


def resolves(base: pathlib.Path, target: str) -> bool:
    """Does target exist under base? Globs count if they match anything."""
    if any(c in target for c in "*?["):
        try:
            return any(base.glob(target))
        except (ValueError, IndexError):
            return False
    return (base / target).exists()


def scan(root: pathlib.Path):
    """Returns (docs, checked, unchecked, failures) for one tree."""

    def skipped(path):
        rel = path.relative_to(root)
        return any(p in SKIP_DIRS or p.startswith("build-") for p in rel.parts[:-1])

    docs = sorted(p for p in root.rglob("*.md") if not skipped(p))

    # The set of first components this checker is entitled to adjudicate: the
    # directories that actually exist at the top of this repository. Deriving
    # it rather than hardcoding it means a new top-level directory is covered
    # the day it appears, and -- more to the point -- a path naming a directory
    # that no longer exists stops being adjudicated LOUDLY, in the unchecked
    # count, instead of quietly passing.
    repo_dirs = {
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and not p.name.startswith((".", "build-"))
        and p.name not in SKIP_DIRS
    }

    failures = []   # (relative document, line number, kind, what is missing)
    unchecked = {}  # first component -> [(document, line, text)]
    checked = {"paths": 0, "links": 0, "named": 0, "bare": 0}

    def note_unchecked(key, doc, lineno, text):
        unchecked.setdefault(key, []).append((doc, lineno, text))

    # Section numbers, per document and pooled. Pooled is what check 4 consults.
    headings = {}
    for doc in docs:
        nums = set()
        for line in doc.read_text(encoding="utf-8", errors="replace").splitlines():
            m = HEADING.match(line)
            if m:
                nums.add(m.group(1))
        headings[doc] = nums
    all_headings = set().union(*headings.values()) if headings else set()

    by_basename = {}
    for doc in docs:
        by_basename.setdefault(doc.name, doc)

    for doc in docs:
        rel = doc.relative_to(root).as_posix()
        here = doc.parent
        text = doc.read_text(encoding="utf-8", errors="replace")
        fenced = False
        for lineno, line in enumerate(text.splitlines(), 1):
            # Fenced blocks are skipped. They hold intended-layout diagrams and
            # shell transcripts, where a path is a plan or an example rather
            # than a claim about this tree -- grxcp_architecture.md 9 is a tree
            # listing with a dozen files nobody has written yet. Prose is where
            # a backticked path asserts something, so prose is what is opened.
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue

            # 1. backticked paths
            for target in BACKTICK_PATH.findall(line):
                target = target.rstrip("/")
                if resolves(root, target) or resolves(here, target):
                    checked["paths"] += 1
                    continue
                head = target.split("/", 1)[0]
                if head in repo_dirs:
                    checked["paths"] += 1
                    failures.append((rel, lineno, "path", target))
                else:
                    note_unchecked(head, rel, lineno, target)

            # 2. markdown links to local pages
            for target in MD_LINK.findall(line):
                if target.startswith(("http:", "https:", "mailto:", "#", "ftp:")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                if resolves(here, target) or resolves(root, target):
                    checked["links"] += 1
                    continue
                head = target.split("/", 1)[0]
                if target.endswith(".md") or head in repo_dirs:
                    checked["links"] += 1
                    failures.append((rel, lineno, "link", target))
                else:
                    note_unchecked(head, rel, lineno, target)

            # 3. "some_doc.md 7.13" -- checked against that document
            for name, marked, dotted in NAMED_SECTION.findall(line):
                num = marked or dotted
                if "/" in name:
                    # Qualified: it resolves as written, or it is someone else's.
                    target_doc = next(
                        (b / name for b in (here, root) if (b / name).exists()), None
                    )
                else:
                    target_doc = by_basename.get(name)
                if target_doc is None or target_doc not in headings:
                    note_unchecked(name, rel, lineno, f"{name} {num}")
                    continue
                checked["named"] += 1
                if num not in headings[target_doc]:
                    failures.append((rel, lineno, "section", f"{name} {num}"))

            # 4. bare "§7.13" / "(section 7.13)" -- must exist somewhere here
            for num in BARE_SECTION.findall(NAMED_SECTION.sub("", line)):
                checked["bare"] += 1
                if num not in all_headings:
                    failures.append((rel, lineno, "section", f"§{num} (no document)"))

    return docs, checked, unchecked, failures


FIXTURE = {
    "src/real.cpp": "int main() { return 0; }\n",
    "b.md": "# B\n\n## 2. Two\n\n### 2.1 Two point one\n",
    "a.md": """# A

Good path `src/real.cpp`, bad path `src/gone.cpp`.
Good link [b](b.md), bad link [gone](missing.md).
Good section b.md 2.1, bad section b.md 9.9.
Good bare §2.1, bad bare §8.8.
Not ours: `upstream/thing.cpp` and `upstream/notes.md` §4.

```
A planned tree, not a claim: `src/planned.cpp` [p](planned.md) §7.7
```
""",
}

EXPECTED = {
    ("path", "src/gone.cpp"),
    ("link", "missing.md"),
    ("section", "b.md 9.9"),
    ("section", "§8.8 (no document)"),
}


def self_test() -> bool:
    """Plant one broken reference of each kind and require all four to be caught."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for name, body in FIXTURE.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")

        _, checked, unchecked, failures = scan(root)
        got = {(kind, target) for _, _, kind, target in failures}

        ok = True
        for kind, target in sorted(EXPECTED):
            if (kind, target) not in got:
                print(f"  FAIL  self-test: a broken {kind} was NOT caught: {target}")
                ok = False
        for kind, target in sorted(got - EXPECTED):
            print(f"  FAIL  self-test: a sound reference was reported: {kind} {target}")
            ok = False
        # The good half of each pair has to have been checked and passed, or the
        # four catches above could come from a checker that fails everything.
        for key, want in (("paths", 1), ("links", 1), ("named", 2), ("bare", 2)):
            if checked[key] < want:
                print(f"  FAIL  self-test: only {checked[key]} {key} checked, want {want}")
                ok = False
        if "upstream" not in unchecked or "upstream/notes.md" not in unchecked:
            print("  FAIL  self-test: another repository's paths were adjudicated")
            ok = False
        if ok:
            print("  ok    self-test: all four kinds of broken reference are caught")
        return ok


def main() -> int:
    if not self_test():
        return 1

    root = pathlib.Path(__file__).resolve().parent.parent
    docs, checked, unchecked, failures = scan(root)

    print(f"  {len(docs)} markdown files, {sum(checked.values())} references checked")
    print(
        f"        {checked['paths']} backticked paths, {checked['links']} links, "
        f"{checked['named']} named sections, {checked['bare']} bare sections"
    )

    if unchecked:
        total = sum(len(v) for v in unchecked.values())
        tops = sorted(unchecked.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        # Truncated for the build log, never silently: the total is the count
        # that matters, and --list-unchecked prints every one of them.
        head, rest = tops[:8], tops[8:]
        summary = ", ".join(f"{k} ({len(v)})" for k, v in head)
        if rest:
            summary += f", +{sum(len(v) for _, v in rest)} more in {len(rest)} kinds"
        print(f"  {total} not adjudicated -- not of this tree: {summary}")
        if "--list-unchecked" in sys.argv:
            for key, items in tops:
                for doc, lineno, text in items:
                    print(f"        {doc}:{lineno}: {text}")

    if failures:
        print(f"  FAIL  {len(failures)} references do not resolve:")
        for rel, lineno, kind, target in failures:
            print(f"        {rel}:{lineno}: {kind}: {target}")
        return 1

    print("  ok    every documented path, link and section reference resolves")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    sys.exit(main())
