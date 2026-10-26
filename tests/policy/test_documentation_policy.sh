#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "$0")/../.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
case_root="$scratch/candidate"
mkdir -p "$case_root"
cp -a "$source_root/." "$case_root/"
rm -rf "$case_root/.git" "$case_root/build" "$case_root/build-"*
git -C "$case_root" init -q
checker="$case_root/scripts/check-documentation-policy.sh"
metadata="$case_root/docs/development/documentation-policy.env"

fail() { echo "documentation policy test failed: $*" >&2; exit 1; }

prepare_case() {
  rm -rf "$case_root"
  mkdir -p "$case_root"
  cp -a "$source_root/." "$case_root/"
  rm -rf "$case_root/.git" "$case_root/build" "$case_root/build-"*
  git -C "$case_root" init -q
  checker="$case_root/scripts/check-documentation-policy.sh"
  metadata="$case_root/docs/development/documentation-policy.env"
}
write_metadata() {
  cat > "$metadata" <<EOF_META
VSS_DOCUMENTATION_POLICY_VERSION=1.0.0
VSS_DOCUMENTATION_POLICY_SOURCE_REPOSITORY=VirtualSensorSoft/VSSMath
VSS_DOCUMENTATION_POLICY_SOURCE_COMMIT=0000000000000000000000000000000000000000
VSS_DOCUMENTATION_POLICY_CHECKER_SHA256=$(sha256sum "$checker" | awk '{print $1}')
VSS_DOCUMENTATION_POLICY_STATUS=LocallyMaintained
VSS_DOCUMENTATION_DEVELOPMENT_ROOT=docs/development
VSS_DOCUMENTATION_ARCHIVE_ROOT=docs/development/archive
EOF_META
}
run() {
  "$checker" --root "$case_root" --adoption docs/development/documentation-policy.env \
    --inventory docs/development/migration-inventory.tsv
}
expect_failure() {
  label=$1; shift
  if run >"$scratch/output" 2>&1; then fail "accepted $label"; fi
  grep -Fq "$label" "$scratch/output" || { cat "$scratch/output" >&2; fail "missing diagnostic for $label"; }
}

[[ -x "$checker" ]] || fail 'checker is missing or not executable'
write_metadata
run >"$scratch/output" 2>&1 || { cat "$scratch/output" >&2; fail 'valid local profile was rejected'; }
grep -Fq 'Documentation policy 1.0.0 checks passed' "$scratch/output" || fail 'success evidence is missing'

prepare_case
write_metadata
sed -i 's#VirtualSensorSoft/VSSMath#ExampleOrg/OtherLibrary#' "$metadata"
expect_failure 'documentation policy is not maintained by VirtualSensorSoft/VSSMath'

prepare_case
write_metadata
sed -i 's/^VSS_DOCUMENTATION_POLICY_STATUS=.*/VSS_DOCUMENTATION_POLICY_STATUS=Adopted/' "$metadata"
expect_failure 'public documentation policy status must be LocallyMaintained'

prepare_case
write_metadata
sed -i 's/^VSS_DOCUMENTATION_POLICY_CHECKER_SHA256=.*/VSS_DOCUMENTATION_POLICY_CHECKER_SHA256=0000000000000000000000000000000000000000000000000000000000000000/' "$metadata"
expect_failure 'vendored checker digest does not match adoption metadata'

prepare_case
write_metadata
rm -rf "$case_root/docs/development/archive"
ln -s /etc "$case_root/docs/development/archive"
expect_failure 'archive root escapes the repository through a symlink'

prepare_case
write_metadata
ln -s /etc/passwd "$case_root/docs/external-link"
expect_failure 'symlink escapes the repository'

prepare_case
write_metadata
mkdir -p "$case_root/build/policy-fixture"
printf '%s\n' '<html></html>' > "$case_root/build/policy-fixture/index.html"
git -C "$case_root" add --all --force
expect_failure 'generated or build output is present'

echo 'Documentation policy tests passed.'
