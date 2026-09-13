#!/usr/bin/env bash
# Land the PTA design documents and the TPA-QCN review on main.
#
#   cd /c/Users/kubew/grxcp && bash _pta_staging/land_docs.sh
#
# Two commits: the earlier design work plus the analyses it cites, then the
# TPA-QCN review and the physics code that produced it.  Every file it touches
# is already on disk; this only stages and commits.  Stops at the first error.
set -euo pipefail

cd "$(dirname "$0")/.."
S=_pta_staging

[ -d .git ] || { echo "not a git repo: $PWD" >&2; exit 1; }
for f in msg_docs1.txt msg_docs2.txt; do
  [ -f "$S/$f" ] || { echo "missing $S/$f" >&2; exit 1; }
done

# Stale locks a sandboxed shell could create but not remove.
rm -f .git/HEAD.lock .git/index.lock .git/objects/maintenance.lock 2>/dev/null || true
find .git/objects -name 'tmp_obj_*' -delete 2>/dev/null || true

COMMIT1=(
  docs/GRX_PTA_Integration.md
  docs/GRX_AI_Photonics_PTA_TPA_QCN.md
  docs/designs/pta_cpu_integration.md
  docs/designs/pta_gpu_integration.md
)
COMMIT2=(
  docs/designs/pta_tpaqcn_review.md
  docs/designs/pta_tpaqcn_audit.py
  docs/designs/pta_tpaqcn_waveguide_pol.py
  docs/designs/pta_tpaqcn_phase_match.py
  docs/designs/pta_tpaqcn_measured.py
  docs/index.md
)

for f in "${COMMIT1[@]}" "${COMMIT2[@]}"; do
  [ -f "$f" ] || { echo "expected file missing: $f" >&2; exit 1; }
done

git checkout main
echo

echo "=== commit 1/2: PTA designs and their source analyses ==="
git add -- "${COMMIT1[@]}"
git commit -F "$S/msg_docs1.txt"
echo

echo "=== commit 2/2: TPA-QCN review and the corrected physics ==="
git add -- "${COMMIT2[@]}"
git commit -F "$S/msg_docs2.txt"
echo

git log --oneline -3
echo
echo "Deliberately NOT committed:"
echo "  docs/Pta_CPU_Integration.md   duplicate of docs/designs/pta_cpu_integration.md"
echo "  docs/Pta_GPU_Integration.md   duplicate of docs/designs/pta_gpu_integration.md"
echo "    (older downloads that landed in docs/ with capitalised names -- delete"
echo "     them, or keep them; either way they are not the live copies)"
echo "  docs/GRX930_SoC_Architectural_Spec.md, docs/GRXIConnect.md,"
echo "  docs/GRX_GCPU.md, docs/notes_from_grx930_team.md, 'Claude outputs/'"
echo "    (your material, left for you to decide on)"
echo
echo "Ready to push.  Remove $S when you are happy."
