#!/usr/bin/env bash
set -euo pipefail

root_arg=
record_arg=
while (($#)); do
  case "$1" in
    --root) (($# >= 2)) || { echo '--root requires a value' >&2; exit 2; }; root_arg=$2; shift 2 ;;
    --record) (($# >= 2)) || { echo '--record requires a value' >&2; exit 2; }; record_arg=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$root_arg" && -n "$record_arg" ]] || { echo 'usage: check-conformance-record.sh --root DIR --record TSV' >&2; exit 2; }
root=$(realpath -e -- "$root_arg")
record=$(realpath -e -- "$record_arg")
manifest="$root/config/industrial-cpp-v1.0.0-rules.txt"
[[ -f "$manifest" && -f "$record" ]] || { echo 'conformance record or rule manifest is missing' >&2; exit 1; }
case "$record" in "$root"/*) ;; *) echo 'conformance record is outside the repository' >&2; exit 1 ;; esac
mapfile -t manifest_ids < "$manifest"
((${#manifest_ids[@]} > 0)) || { echo 'rule manifest is empty' >&2; exit 1; }
declare -A expected=() seen=()
for id in "${manifest_ids[@]}"; do
  [[ "$id" =~ ^ENG-(CPP-[0-9]{3}|CON-[0-9]{3}|DATA-[0-9]{3})$ ]] || { echo "invalid public rule ID: $id" >&2; exit 1; }
  [[ -z "${expected[$id]+x}" ]] || { echo "duplicate public rule ID in manifest: $id" >&2; exit 1; }
  expected[$id]=1
done
IFS=$'\t' read -r status_key record_status extra < "$record" || true
[[ "$status_key" == record_status && -z "${extra:-}" ]] || { echo 'record status header is invalid' >&2; exit 1; }
IFS=$'\t' read -r commit_key source_commit extra < <(sed -n '2p' "$record")
[[ "$commit_key" == source_commit && -z "${extra:-}" ]] || { echo 'record source identity is invalid' >&2; exit 1; }
case "$record_status" in Under-review|Final) ;; *) echo 'record status must be Under-review or Final' >&2; exit 1 ;; esac
if [[ "$record_status" == Final ]]; then
  [[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'final record requires an assessed source commit' >&2; exit 1; }
  if ! git -C "$root" cat-file -e "${source_commit}^{commit}" 2>/dev/null; then
    echo 'assessed source commit is not present in repository' >&2
    exit 1
  fi
else
  [[ "$source_commit" == 'Not assessed' ]] || { echo 'under-review record cannot claim an assessed source commit' >&2; exit 1; }
fi
header=$'rule_id\tassessment\tevidence\tdeviation'
IFS= read -r actual_header < <(sed -n '3p' "$record") || true
[[ "$actual_header" == "$header" ]] || { echo 'rule evidence table header is invalid' >&2; exit 1; }
line=3
while IFS=$'\t' read -r id assessment evidence deviation extra || [[ -n "${id:-}" ]]; do
  line=$((line + 1))
  [[ -z "${extra:-}" && -n "${id:-}" && -n "${assessment:-}" && -n "${evidence:-}" && -n "${deviation:-}" ]] || {
    echo "invalid conformance evidence row $line" >&2; exit 1;
  }
  [[ -n "${expected[$id]+x}" ]] || { echo "unknown rule in conformance record: $id" >&2; exit 1; }
  [[ -z "${seen[$id]+x}" ]] || { echo "duplicate rule in conformance record: $id" >&2; exit 1; }
  seen[$id]=1
  case "$assessment" in Pending|Passed|Failed|'Approved deviation') ;; *) echo "invalid assessment for $id" >&2; exit 1 ;; esac
  if [[ "$assessment" == 'Approved deviation' ]]; then
    [[ "$deviation" != None ]] || { echo "approved deviation lacks a record for $id" >&2; exit 1; }
  else
    [[ "$deviation" == None ]] || { echo "unexpected deviation reference for $id" >&2; exit 1; }
  fi
  if [[ "$record_status" == Final ]]; then
    [[ "$assessment" == Passed || "$assessment" == 'Approved deviation' ]] || {
      echo "final record has unresolved assessment: $id" >&2; exit 1;
    }
    [[ "$evidence" != 'not yet assessed' ]] || { echo "final record lacks evidence: $id" >&2; exit 1; }
  fi
done < <(tail -n +4 "$record")
if ((${#seen[@]} != ${#expected[@]})); then
  echo 'rule coverage does not match the manifest' >&2
  exit 1
fi
for id in "${manifest_ids[@]}"; do
  [[ -n "${seen[$id]+x}" ]] || { echo "rule is missing from conformance record: $id" >&2; exit 1; }
done
printf 'Conformance record policy checks passed: %s rules, status %s.\n' "${#expected[@]}" "$record_status"
