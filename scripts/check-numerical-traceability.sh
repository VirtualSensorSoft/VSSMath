#!/usr/bin/env bash
set -euo pipefail

root_arg=
requirements_arg=
while (($#)); do
  case "$1" in
    --root) (($# >= 2)) || { echo '--root requires a value' >&2; exit 2; }; root_arg=$2; shift 2 ;;
    --requirements) (($# >= 2)) || { echo '--requirements requires a value' >&2; exit 2; }; requirements_arg=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$root_arg" && -n "$requirements_arg" ]] || {
  echo 'usage: check-numerical-traceability.sh --root DIR --requirements RELATIVE_PATH' >&2
  exit 2
}
root=$(realpath -e -- "$root_arg")
case "$requirements_arg" in
  ''|/*|.|..|./*|../*|*/./*|*/../*|*//* ) echo 'requirements path must be normalized and relative' >&2; exit 1 ;;
esac
requirements="$root/$requirements_arg"
[[ -f "$requirements" ]] || { echo 'requirements file is missing' >&2; exit 1; }
resolved=$(realpath -e -- "$requirements")
case "$resolved" in "$root"/*) ;; *) echo 'requirements path escapes the repository' >&2; exit 1 ;; esac

header=$'requirement_id\tcontract\tstable_fixtures\tctest_evidence'
IFS= read -r actual_header < "$requirements" || true
[[ "$actual_header" == "$header" ]] || { echo 'numerical requirements header is invalid' >&2; exit 1; }
cmake_list="$root/tests/CMakeLists.txt"
[[ -f "$cmake_list" ]] || { echo 'test CMake inventory is missing' >&2; exit 1; }
declare -A seen=()
rows=0
while IFS=$'\t' read -r id contract fixtures ctests extra || [[ -n "${id:-}" ]]; do
  rows=$((rows + 1))
  [[ -z "${extra:-}" && -n "${id:-}" && -n "${contract:-}" && -n "${fixtures:-}" && -n "${ctests:-}" ]] || {
    echo "invalid or empty numerical requirement row $((rows + 1))" >&2; exit 1;
  }
  [[ "$id" =~ ^NUM-[A-Z]+-[0-9]{3}$ ]] || { echo "invalid public requirement ID: $id" >&2; exit 1; }
  [[ -z "${seen[$id]+x}" ]] || { echo "duplicate numerical requirement ID: $id" >&2; exit 1; }
  seen[$id]=1
  IFS=',' read -r -a fixture_list <<< "$fixtures"
  ((${#fixture_list[@]} > 0)) || { echo "requirement has no fixture: $id" >&2; exit 1; }
  for fixture in "${fixture_list[@]}"; do
    [[ "$fixture" =~ ^(TRI|LU|QR|CHO|EIG|COV|DIF|STA|PRO)-[0-9]{2}$ || "$fixture" =~ ^SUP-(TRI|LU|QR|EIG|DIF|STA|PRO)-[0-9]{2}$ ]] || {
      echo "invalid public fixture ID '$fixture' for $id" >&2; exit 1;
    }
    if ! grep -R -F -q --include='*.cpp' --include='*.hpp' -- "$fixture" "$root/tests"; then
      echo "fixture is not referenced by a test source: $fixture" >&2; exit 1;
    fi
  done
  IFS=',' read -r -a ctest_list <<< "$ctests"
  ((${#ctest_list[@]} > 0)) || { echo "requirement has no CTest evidence: $id" >&2; exit 1; }
  for ctest_name in "${ctest_list[@]}"; do
    [[ "$ctest_name" =~ ^math\.[a-z0-9_.]+$ ]] || { echo "invalid CTest evidence '$ctest_name' for $id" >&2; exit 1; }
    if ! grep -Fq -- "$ctest_name" "$cmake_list"; then
      echo "CTest evidence is not registered: $ctest_name" >&2; exit 1;
    fi
  done
done < <(tail -n +2 "$requirements")
[[ "$rows" -eq 95 ]] || { echo "expected 95 numerical requirements, found $rows" >&2; exit 1; }
printf 'Numerical traceability 1.0.0 checks passed: %s requirements.\n' "$rows"
