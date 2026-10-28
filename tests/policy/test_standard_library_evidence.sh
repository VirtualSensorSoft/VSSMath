#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
recorder="$repo_root/scripts/record-standard-library.sh"
compiler=${1:?usage: test_standard_library_evidence.sh CXX_COMPILER [compiler options...]}
shift
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

fail() { printf 'standard-library evidence test failed: %s\n' "$*" >&2; exit 1; }
field() {
  local name=$1 file=$2
  sed -n "s/^${name}=//p" "$file"
}
require_field() {
  local name=$1 file=$2 value
  value=$(field "$name" "$file")
  [[ -n "$value" ]] || fail "missing or empty ${name}"
  [[ "$value" != not-applicable ]] || fail "${name} is unexpectedly not applicable"
}
expect_failure_without_output() {
  local label=$1 command=$2 output=$3
  shift 3
  printf 'stale success record\n' > "$output"
  if "$recorder" "$command" "$output" "$@" >"$scratch/stdout" 2>"$scratch/stderr"; then
    fail "accepted ${label}"
  fi
  [[ ! -e "$output" ]] || fail "${label} left a stale success record"
}

"$recorder" "$compiler" "$scratch/actual.txt" "$@"
require_field source_commit "$scratch/actual.txt"
require_field compiler_command "$scratch/actual.txt"
[[ $(field language_standard "$scratch/actual.txt") == 202002L ]] ||
  fail 'capture did not identify C++20'
library=$(field standard_library "$scratch/actual.txt")
[[ "$library" == libstdc++ || "$library" == libc++ ]] ||
  fail "unexpected standard library: ${library:-empty}"
require_field standard_library_version "$scratch/actual.txt"
[[ -n $(field standard_library_release "$scratch/actual.txt") ]] ||
  fail 'standard_library_release is empty'
[[ -n $(field cxx_abi "$scratch/actual.txt") ]] || fail 'cxx_abi is empty'

cat > "$scratch/failing-compiler" <<'EOF'
#!/usr/bin/env bash
exit 42
EOF
cat > "$scratch/unknown-library-compiler" <<'EOF'
#!/usr/bin/env bash
printf '#define __cplusplus 202002L\n'
EOF
cat > "$scratch/empty-macro-compiler" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$scratch/failing-compiler" "$scratch/unknown-library-compiler" "$scratch/empty-macro-compiler"
expect_failure_without_output 'missing compiler' "$scratch/no-such-compiler" "$scratch/missing.txt"
expect_failure_without_output 'failing compiler' "$scratch/failing-compiler" "$scratch/failing.txt"
expect_failure_without_output 'unknown standard library' "$scratch/unknown-library-compiler" "$scratch/unknown.txt"
expect_failure_without_output 'empty compiler macros' "$scratch/empty-macro-compiler" "$scratch/empty.txt"

printf 'Standard-library evidence tests passed for %s.\n' "$compiler"
