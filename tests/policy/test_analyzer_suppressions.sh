#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
checker="$root/scripts/check-analyzer-suppressions.sh"
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/include" "$scratch/src" "$scratch/tests" "$scratch/benchmarks" \
  "$scratch/docs/development/deviations"
cp "$root/docs/development/deviations/analyzer-suppressions.md" \
  "$scratch/docs/development/deviations/analyzer-suppressions.md"
[[ -x "$checker" ]] || { echo 'analyzer suppression checker is missing' >&2; exit 1; }

"$checker" "$root"
printf '%s\n' 'int value; // NOLINT' > "$scratch/include/example.hpp"
if "$checker" "$scratch" >"$scratch/output" 2>&1; then
  echo 'unregistered suppression was accepted' >&2; exit 1
fi
grep -Fq 'unregistered suppression' "$scratch/output"

printf '%s\n' '| include/example.hpp:1 | readability-identifier-naming | test | controlled fixture | unit test | Reviewed |' \
  >> "$scratch/docs/development/deviations/analyzer-suppressions.md"
if "$checker" "$scratch" >"$scratch/output" 2>&1; then
  echo 'bare suppression was accepted' >&2; exit 1
fi
grep -Fq 'bare NOLINT suppression' "$scratch/output"

sed -i 's/NOLINT/NOLINT(readability-identifier-naming)/' "$scratch/include/example.hpp"
"$checker" "$scratch"
sed -i 's/NOLINT(readability-identifier-naming)/NOLINTNEXTLINE(readability-identifier-naming)/' \
  "$scratch/include/example.hpp"
"$checker" "$scratch"

echo 'Analyzer suppression policy tests passed.'
