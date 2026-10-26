#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checker="$repo_root/scripts/check-workflow-policy.sh"
workflow="$repo_root/.github/workflows/ci.yml"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

fail() { printf 'workflow policy test failed: %s\n' "$*" >&2; exit 1; }
[[ -x "$checker" ]] || fail 'workflow checker is missing or not executable'

expect_failure() {
  expected=$1
  shift
  if "$@" >"$scratch/output" 2>&1; then fail "accepted $expected"; fi
  grep -Fq "$expected" "$scratch/output" || {
    sed -n '1,60p' "$scratch/output" >&2
    fail "missing diagnostic: $expected"
  }
}
copy_workflow() { cp "$workflow" "$scratch/workflow.yml"; }

"$checker" "$workflow"
copy_workflow
sed -i 's/^  contents: read$/  contents: write-all/' "$scratch/workflow.yml"
expect_failure 'workflow must grant only contents: read' "$checker" "$scratch/workflow.yml"
copy_workflow
printf '%s\n' '    permissions:' '      contents: write' >> "$scratch/workflow.yml"
expect_failure 'workflow must grant only contents: read' "$checker" "$scratch/workflow.yml"
copy_workflow
sed -i 's#actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1#actions/checkout@v7#' "$scratch/workflow.yml"
expect_failure 'action reference must use an immutable commit SHA' "$checker" "$scratch/workflow.yml"
copy_workflow
sed -i 's/runs-on: ubuntu-24.04/runs-on: self-hosted/' "$scratch/workflow.yml"
expect_failure 'self-hosted runners are not permitted' "$checker" "$scratch/workflow.yml"
copy_workflow
printf '\n  pull_request_target:\n' >> "$scratch/workflow.yml"
expect_failure 'pull_request_target is not permitted' "$checker" "$scratch/workflow.yml"
copy_workflow
sed -i '/persist-credentials: false/d' "$scratch/workflow.yml"
expect_failure 'checkout must disable persisted credentials' "$checker" "$scratch/workflow.yml"
copy_workflow
sed -i '/run: |/a\          echo "${{ secrets.TEST_TOKEN }}"' "$scratch/workflow.yml"
expect_failure 'workflow must not interpolate secrets into shell commands' "$checker" "$scratch/workflow.yml"
copy_workflow
printf '%s\n' '      - uses: unreviewed/action@0123456789abcdef0123456789abcdef01234567' >> "$scratch/workflow.yml"
expect_failure 'external action owner is not in the reviewed inventory' "$checker" "$scratch/workflow.yml"
copy_workflow
printf '%s\n' \
  '      - uses: actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f' \
  '        with:' \
  '          name: report' \
  '          path: report.txt' >> "$scratch/workflow.yml"
expect_failure 'every artifact upload must specify retention-days' "$checker" "$scratch/workflow.yml"
printf '%s\n' 'Workflow policy tests passed.'
