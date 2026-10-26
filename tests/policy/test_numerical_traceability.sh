#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
checker="$root/scripts/check-numerical-traceability.sh"
requirements="$root/docs/development/numerical-requirements.tsv"

[[ -x "$checker" ]] || { echo 'traceability checker is missing or not executable' >&2; exit 1; }
"$checker" --root "$root" --requirements docs/development/numerical-requirements.tsv

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp "$requirements" "$tmp/requirements.tsv"

if "$checker" --root "$root" --requirements "$tmp/requirements.tsv" >/dev/null 2>&1; then
  echo 'checker accepted a traceability file outside the repository' >&2
  exit 1
fi

awk 'NR != 2' "$requirements" > "$tmp/requirements.tsv"
if "$checker" --root "$root" --requirements "$tmp/requirements.tsv" >/dev/null 2>&1; then
  echo 'checker accepted an incomplete requirement inventory' >&2
  exit 1
fi

cp "$requirements" "$tmp/requirements.tsv"
sed -i '2s/NUM-/NUM-DUPLICATE-/' "$tmp/requirements.tsv"
if "$checker" --root "$root" --requirements "$tmp/requirements.tsv" >/dev/null 2>&1; then
  echo 'checker accepted a malformed requirement identity' >&2
  exit 1
fi

echo 'Numerical traceability policy checks passed.'
