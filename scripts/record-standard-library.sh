#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s CXX_COMPILER OUTPUT [compiler-options...]\n' "${0##*/}" >&2
  exit 2
}

[[ $# -ge 2 ]] || usage
compiler=$1
output=$2
shift 2
[[ -n "$compiler" && -n "$output" ]] || usage

# Remove any previous success record first: a failed fresh capture must never
# leave an older artifact that could be mistaken for current evidence.
if [[ -d "$output" && ! -L "$output" ]]; then
  printf 'standard-library evidence: output path is a directory: %s\n' "$output" >&2
  exit 1
fi
rm -f -- "$output"

output_dir=$(dirname -- "$output")
if [[ ! -d "$output_dir" ]]; then
  printf 'standard-library evidence: output directory does not exist: %s\n' "$output_dir" >&2
  exit 1
fi
compiler_path=$(command -v -- "$compiler" 2>/dev/null || true)
if [[ -z "$compiler_path" || ! -x "$compiler_path" ]]; then
  printf 'standard-library evidence: compiler is not executable or not found: %s\n' "$compiler" >&2
  exit 1
fi

macro_file=$(mktemp)
compiler_stderr=$(mktemp)
temporary_output=$(mktemp "${output}.tmp.XXXXXX")
success=0
cleanup() {
  rm -f -- "$macro_file" "$compiler_stderr" "$temporary_output"
  if [[ "$success" -ne 1 ]]; then
    rm -f -- "$output"
  fi
}
trap cleanup EXIT

source_commit=$(git -C "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" \
  rev-parse --verify HEAD 2>/dev/null || true)
if [[ ! "$source_commit" =~ ^[0-9a-f]{40}$ ]]; then
  printf 'standard-library evidence: repository HEAD is unavailable or not a full commit ID\n' >&2
  exit 1
fi

if ! "$compiler_path" "$@" -std=c++20 -dM -E -x c++ -include version /dev/null \
    >"$macro_file" 2>"$compiler_stderr"; then
  printf 'standard-library evidence: C++20 macro capture failed for %s\n' "$compiler_path" >&2
  cat "$compiler_stderr" >&2
  exit 1
fi

macro_value() {
  awk -v name="$1" '$1 == "#define" && $2 == name { print $3 }' "$macro_file"
}

language_standard=$(macro_value __cplusplus)
if [[ "$language_standard" != 202002L ]]; then
  printf 'standard-library evidence: expected __cplusplus=202002L, got %s\n' \
    "${language_standard:-empty}" >&2
  exit 1
fi

glibcxx_version=$(macro_value __GLIBCXX__)
glibcxx_release=$(macro_value _GLIBCXX_RELEASE)
libcpp_version=$(macro_value _LIBCPP_VERSION)
if [[ -n "$glibcxx_version" && -n "$libcpp_version" ]]; then
  printf 'standard-library evidence: compiler exposed multiple standard libraries\n' >&2
  exit 1
elif [[ -n "$glibcxx_version" ]]; then
  [[ "$glibcxx_version" =~ ^[0-9]+$ && "$glibcxx_release" =~ ^[0-9]+$ ]] || {
    printf 'standard-library evidence: libstdc++ version or release macro is missing/invalid\n' >&2
    exit 1
  }
  cxx_abi=$(macro_value _GLIBCXX_USE_CXX11_ABI)
  [[ "$cxx_abi" =~ ^[01]$ ]] || {
    printf 'standard-library evidence: libstdc++ C++ ABI macro is missing/invalid\n' >&2
    exit 1
  }
  standard_library=libstdc++
  standard_library_version=$glibcxx_version
  standard_library_release=$glibcxx_release
  cxx_abi="libstdc++-cxx11-${cxx_abi}"
elif [[ -n "$libcpp_version" ]]; then
  [[ "$libcpp_version" =~ ^[0-9]+$ ]] || {
    printf 'standard-library evidence: libc++ version macro is invalid\n' >&2
    exit 1
  }
  libcxx_abi=$(macro_value __LIBCPP_ABI_VERSION)
  [[ "$libcxx_abi" =~ ^[0-9]+$ ]] || {
    printf 'standard-library evidence: libc++ ABI version macro is missing/invalid\n' >&2
    exit 1
  }
  standard_library=libc++
  standard_library_version=$libcpp_version
  standard_library_release=not-applicable
  cxx_abi="libc++-abi-${libcxx_abi}"
else
  printf 'standard-library evidence: compiler macros do not identify libstdc++ or libc++\n' >&2
  exit 1
fi

{
  printf 'source_commit=%s\n' "$source_commit"
  printf 'compiler_command='
  printf '%q' "$compiler_path"
  for option in "$@"; do printf ' %q' "$option"; done
  printf ' -std=c++20 -dM -E -x c++ -include version /dev/null\n'
  printf 'language_standard=%s\n' "$language_standard"
  printf 'standard_library=%s\n' "$standard_library"
  printf 'standard_library_version=%s\n' "$standard_library_version"
  printf 'standard_library_release=%s\n' "$standard_library_release"
  printf 'cxx_abi=%s\n' "$cxx_abi"
} >"$temporary_output"

mv -f -- "$temporary_output" "$output"
success=1
