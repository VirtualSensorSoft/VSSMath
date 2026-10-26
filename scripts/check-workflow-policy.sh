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

if [[ "$failed" -ne 0 ]]; then exit 1; fi
printf '%s\n' 'Workflow policy checks passed.'
