#!/usr/bin/env bash
set -eu

repo_root=${1:-.}
register=${2:-"$repo_root/docs/development/deviations/analyzer-suppressions.md"}
failures=0

report_failure() {
    printf 'analyzer suppression check failed: %s\n' "$1" >&2
    failures=1
}

if [ ! -f "$register" ]; then
    report_failure "suppression register is missing: $register"
    exit 1
fi

occurrences=$(mktemp)
trap 'rm -f "$occurrences"' EXIT
for directory in include src tests benchmarks; do
    if [ -d "$repo_root/$directory" ]; then
        grep -RInE --include='*.h' --include='*.hpp' --include='*.cpp' \
            'NOLINT(NEXTLINE)?([[:space:]]|\(|$)' "$repo_root/$directory" >>"$occurrences" || true
    fi
done

if [ -s "$occurrences" ]; then
    while IFS=: read -r path line _; do
        relative=${path#"$repo_root/"}
        if ! grep -Fq -- "| ${relative}:${line} |" "$register"; then
            report_failure "unregistered suppression at ${relative}:${line}"
        fi
    done <"$occurrences"
fi

if grep -RInE --include='*.h' --include='*.hpp' --include='*.cpp' \
    'NOLINT($|[[:space:]]|\(\))' "$repo_root/include" "$repo_root/src" \
    "$repo_root/tests" "$repo_root/benchmarks" 2>/dev/null; then
    report_failure 'bare NOLINT suppression found'
fi

if grep -nE '^\| [^|]+ \| (\*|[^[:space:]|]+\*)' "$register"; then
    report_failure 'wildcard suppression check found in register'
fi

if [ "$failures" -ne 0 ]; then
    exit 1
fi
printf '%s\n' 'Analyzer suppression checks passed'
