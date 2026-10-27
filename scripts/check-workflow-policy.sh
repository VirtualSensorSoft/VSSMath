#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workflow=${1:-$repo_root/.github/workflows/ci.yml}
if [[ ! -f "$workflow" ]]; then
  printf 'workflow file is missing: %s\n' "$workflow" >&2
  exit 1
fi

failed=0
fail() { printf 'workflow policy violation: %s\n' "$1" >&2; failed=1; }

permission_count=$(grep -Ec '^permissions:[[:space:]]*$' "$workflow" || true)
if [[ "$permission_count" -ne 1 ]] || grep -Eq '^[[:space:]]+permissions:' "$workflow" || ! awk '
  /^permissions:[[:space:]]*$/ { found = 1; in_permissions = 1; next }
  in_permissions && /^[^[:space:]#]/ { in_permissions = 0 }
  in_permissions && $0 !~ /^[[:space:]]*$/ && $0 !~ /^[[:space:]]*#/ && $0 !~ /^  contents:[[:space:]]+read[[:space:]]*$/ { invalid = 1 }
  END { exit !(found && !invalid) }
' "$workflow"; then
  fail 'workflow must grant only contents: read'
fi

grep -Eq '(^|[[:space:]])pull_request_target([[:space:]:]|$)' "$workflow" && \
  fail 'pull_request_target is not permitted'
grep -Eq 'runs-on:.*self-hosted|\[[^]]*self-hosted' "$workflow" && \
  fail 'self-hosted runners are not permitted'
if awk '
  /^    if:/ { in_job_if = 1 }
  in_job_if && /^    [A-Za-z0-9_-]+:/ && $0 !~ /^    if:/ { in_job_if = 0 }
  in_job_if && /matrix[.]|matrix\[/ { invalid = 1 }
  END { exit !invalid }
' "$workflow"; then
  fail 'job-level if must not reference matrix context'
fi
grep -Fq '${{ secrets.' "$workflow" && \
  fail 'workflow must not interpolate secrets into shell commands'
grep -Eq 'clang-format[^#]*[[:space:]]-i([[:space:]]|$)' "$workflow" && \
  fail 'workflow formatter must remain check-only'
grep -Eq 'clang-tidy[^#]*--fix(-errors)?([[:space:]]|$)' "$workflow" && \
  fail 'workflow analyzer must not apply fixes'

while IFS= read -r action_ref; do
  action_name=${action_ref%@*}
  case "$action_name" in
    actions/checkout|actions/cache|actions/cache/restore|actions/cache/save|\
    actions/upload-artifact|aquasecurity/trivy-action) ;;
    *) fail 'external action owner is not in the reviewed inventory'; continue ;;
  esac
  action_sha=${action_ref##*@}
  if ! [[ "$action_sha" =~ ^[0-9a-f]{40}$ ]]; then
    fail 'action reference must use an immutable commit SHA'
  fi
done < <(sed -nE 's/^[[:space:]]*-?[[:space:]]*uses:[[:space:]]*([^[:space:]#]+).*/\1/p' "$workflow")

if ! awk '
  function finish_checkout() {
    if (checkout && !persist_disabled) { missing = 1 }
    checkout = 0
    persist_disabled = 0
  }
  /^[[:space:]]*-[[:space:]]+(name:|uses:)/ {
    if ($0 !~ /uses:[[:space:]]*actions\/checkout@/) finish_checkout()
  }
  /uses:[[:space:]]*actions\/checkout@/ { finish_checkout(); checkout = 1 }
  checkout && /^[[:space:]]+persist-credentials:[[:space:]]*false([[:space:]#]|$)/ { persist_disabled = 1 }
  END { finish_checkout(); exit missing }
' "$workflow"; then
  fail 'checkout must disable persisted credentials'
fi

if ! awk '
  function finish_upload() {
    if (upload && !retention) { missing = 1 }
    upload = 0
    retention = 0
  }
  /^[[:space:]]*-[[:space:]]+(name:|uses:)/ &&
      $0 !~ /uses:[[:space:]]*actions\/upload-artifact@/ { finish_upload() }
  /uses:[[:space:]]*actions\/upload-artifact@/ { finish_upload(); upload = 1 }
  upload && /^[[:space:]]+retention-days:[[:space:]]*[^[:space:]#]+/ { retention = 1 }
  END { finish_upload(); exit missing }
' "$workflow"; then
  fail 'every artifact upload must specify retention-days'
fi

workflow_contains() {
  grep -Fq -- "$1" "$workflow"
}
require_workflow_text() {
  if ! workflow_contains "$1"; then fail "$2"; fi
}

require_workflow_text "cron: '30 23 * * 5'" 'full suite must run on the approved Friday schedule'
require_workflow_text 'timezone: Europe/Berlin' 'weekly schedule must use Europe/Berlin time'
require_workflow_text 'type: boolean' 'manual full_suite input must be boolean'
require_workflow_text 'default: true' 'manual full_suite must default to full acceptance'
require_workflow_text "github.repository == 'VirtualSensorSoft/VSSMath'" 'scheduled full runs must be restricted to the canonical repository'
require_workflow_text "ctest --test-dir build --output-on-failure --output-junit build/acceptance/ctest-fast.xml -LE '^ci-extended$'" 'fast tests must exclude the ci-extended selection'
require_workflow_text "-LE '^ci-once-per-full$'" 'normal full lanes must exclude once-per-full tests'
require_workflow_text "-L '^ci-once-per-full$'" 'one shared full-suite lane must select once-per-full tests'

for required in \
  'CCACHE_MAXSIZE: 512M' \
  'CCACHE_COMPILERCHECK: content' \
  'CCACHE_BASEDIR: ${{ github.workspace }}' \
  'path: ${{ runner.temp }}/vssmath-ccache' \
  'CMAKE_CXX_COMPILER_LAUNCHER=ccache' \
  'steps.compiler.outputs.version' \
  '${{ github.sha }}'; do
  require_workflow_text "$required" "compiler cache contract is missing: $required"
done
home_cache_pattern='/ho''me/'
if grep -Eq 'CCACHE_SLOPPINESS' "$workflow" || grep -Eq "CCACHE_DIR:.*${home_cache_pattern}" "$workflow"; then
  fail 'compiler cache must not use sloppiness or a home-directory cache'
fi
if [[ $(grep -Ec 'uses: actions/cache@' "$workflow" || true) -ne 3 ]]; then
  fail 'only the fast compiler, full compiler, and coverage profiles may use the cache action'
fi
cache_block=$(awk '
  /uses: actions\/cache@/ { in_cache = 1; found = 1 }
  in_cache { print }
  in_cache && /^[[:space:]]*restore-keys:/ { in_cache = 0 }
  END { if (!found) exit 1 }
' "$workflow")
if grep -Eq "path:.*(build|\\.cache|${home_cache_pattern})" <<<"$cache_block"; then
  fail 'only the runner.temp ccache directory may be persisted'
fi
analysis_block=$(awk '
  /^  static-analysis:/ { in_job = 1 }
  in_job && /^  [A-Za-z0-9_-]+:/ && $0 !~ /^  static-analysis:/ { exit }
  in_job { print }
' "$workflow")
if grep -Eq 'actions/cache@|CCACHE_DIR|CMAKE_CXX_COMPILER_LAUNCHER=ccache' <<<"$analysis_block"; then
  fail 'static analysis must remain uncached'
fi
if grep -Eq 'test[_ -]result cache|cache.*ctest|ctest.*cache' "$workflow"; then
  fail 'test results must never be cached'
fi

if [[ "$failed" -ne 0 ]]; then exit 1; fi
printf '%s\n' 'Workflow policy checks passed.'
