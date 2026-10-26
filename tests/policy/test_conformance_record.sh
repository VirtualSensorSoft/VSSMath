#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
checker="$root/scripts/check-conformance-record.sh"
manifest="$root/config/industrial-cpp-v1.0.0-rules.txt"
scratch=$(mktemp -d)
fixture_root="$scratch/repository"
mkdir -p "$fixture_root/config" "$fixture_root/scripts" "$fixture_root/docs/development"
cp "$manifest" "$fixture_root/config/industrial-cpp-v1.0.0-rules.txt"
cp "$checker" "$fixture_root/scripts/check-conformance-record.sh"
git -C "$fixture_root" init -q
git -C "$fixture_root" config user.name 'Policy fixture'
git -C "$fixture_root" config user.email 'policy-fixture@example.invalid'
git -C "$fixture_root" add config scripts
git -C "$fixture_root" commit -qm 'create conformance checker fixture'
trap 'rm -rf "$scratch"' EXIT
record="$fixture_root/docs/development/conformance-record.tsv"
fail() { echo "conformance record test failed: $*" >&2; exit 1; }
[[ -x "$checker" ]] || fail 'conformance checker is missing or not executable'

write_record() {
  status=$1; commit=$2; result=$3
  {
    printf 'record_status\t%s\nsource_commit\t%s\n' "$status" "$commit"
    printf 'rule_id\tassessment\tevidence\tdeviation\n'
    while IFS= read -r id; do
      evidence='not yet assessed'; [[ "$result" == Pending ]] || evidence='reviewed test evidence'
      printf '%s\t%s\t%s\t%s\n' "$id" "$result" "$evidence" 'None'
    done < "$manifest"
  } > "$record"
}
run_checker() { "$checker" --root "$fixture_root" --record "$record"; }

write_record Under-review 'Not assessed' Pending
run_checker >/dev/null || fail 'provisional record was rejected'

write_record Final 0000000000000000000000000000000000000000 Passed
if run_checker >"$scratch/output" 2>&1; then fail 'nonexistent source commit was accepted'; fi
grep -Fq 'assessed source commit is not present in repository' "$scratch/output" || fail 'missing source-identity diagnostic'

source_commit=$(git -C "$fixture_root" rev-parse HEAD)
write_record Final "$source_commit" Passed
run_checker >/dev/null || fail 'complete final record was rejected'

write_record Final "$source_commit" Pending
if run_checker >"$scratch/output" 2>&1; then fail 'final record with pending results was accepted'; fi
grep -Fq 'final record has unresolved assessment' "$scratch/output" || fail 'missing pending-result diagnostic'

write_record Final "$source_commit" Passed
missing_record="$fixture_root/docs/development/conformance-missing.tsv"
awk 'NR != 4' "$record" > "$missing_record"
if "$checker" --root "$fixture_root" --record "$missing_record" >"$scratch/output" 2>&1; then fail 'incomplete final record was accepted'; fi
grep -Fq 'rule coverage does not match the manifest' "$scratch/output" || fail 'missing coverage diagnostic'

echo 'Conformance record policy tests passed.'
