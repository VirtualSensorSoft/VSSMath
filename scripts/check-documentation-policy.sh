#!/usr/bin/env bash
set -eu

root_arg=
adoption_arg=
inventory_arg=
root=
dev_root=
archive_root=
failures=0

declare -A adoption_values
declare -A adoption_seen
declare -a adoption_order
declare -A record_path
declare -A record_status
declare -A record_superseded_by
declare -A record_supersedes
declare -A inventory_seen
declare -A path_map_seen

report_failure() {
    printf 'documentation policy check failed: %s\n' "$1" >&2
    failures=1
}

parse_arguments() {
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --root)
                if [ "$#" -lt 2 ]; then
                    report_failure '--root requires a value'
                    return
                fi
                root_arg=$2
                shift 2
                ;;
            --adoption)
                if [ "$#" -lt 2 ]; then
                    report_failure '--adoption requires a value'
                    return
                fi
                adoption_arg=$2
                shift 2
                ;;
            --inventory)
                if [ "$#" -lt 2 ]; then
                    report_failure '--inventory requires a value'
                    return
                fi
                inventory_arg=$2
                shift 2
                ;;
            *)
                report_failure "unknown argument: $1"
                shift
                ;;
        esac
    done

    [ -n "$root_arg" ] || report_failure '--root is required'
    [ -n "$adoption_arg" ] || report_failure '--adoption is required'
    [ -n "$inventory_arg" ] || report_failure '--inventory is required'
}

validate_relative_path() {
    label=$1
    value=$2
    case "$value" in
        ''|/*|.|..|./*|../*|*/./*|*/../*|*//*|*$'\r'*|*$'\n'*)
            report_failure "$label is not a normalized repository-relative path: $value"
            return
            ;;
    esac

    if [ -e "$root/$value" ] || [ -L "$root/$value" ]; then
        resolved=
        resolved=$(readlink -f -- "$root/$value" 2>/dev/null || true)
        case "$resolved" in
            "$root"|"$root"/*) ;;
            *) report_failure "$label escapes the repository through a symlink: $value" ;;
        esac
    fi
}

load_adoption() {
    adoption_file="$root/$adoption_arg"
    if [ ! -f "$adoption_file" ]; then
        report_failure "adoption file is missing: $adoption_arg"
        return
    fi

    expected_keys=(
        VSS_DOCUMENTATION_POLICY_VERSION
        VSS_DOCUMENTATION_POLICY_SOURCE_REPOSITORY
        VSS_DOCUMENTATION_POLICY_SOURCE_COMMIT
        VSS_DOCUMENTATION_POLICY_CHECKER_SHA256
        VSS_DOCUMENTATION_POLICY_STATUS
        VSS_DOCUMENTATION_DEVELOPMENT_ROOT
        VSS_DOCUMENTATION_ARCHIVE_ROOT
    )
    line_number=0
    while IFS= read -r line || [ -n "$line" ]; do
        line_number=$((line_number + 1))
        if [ -z "$line" ]; then
            report_failure "blank adoption line at $line_number"
            continue
        fi
        case "$line" in
            *=*) ;;
            *) report_failure "adoption line has no equals sign at $line_number"; continue ;;
        esac
        key=$(printf '%s\n' "$line" | sed 's/=.*//')
        value=$(printf '%s\n' "$line" | sed 's/^[^=]*=//')
        if ! [[ "$key" =~ ^[A-Z][A-Z0-9_]*$ ]]; then
            report_failure "invalid adoption key at $line_number"
            continue
        fi
        case "$key" in
            VSS_DOCUMENTATION_POLICY_VERSION|VSS_DOCUMENTATION_POLICY_SOURCE_REPOSITORY|\
            VSS_DOCUMENTATION_POLICY_SOURCE_COMMIT|VSS_DOCUMENTATION_POLICY_CHECKER_SHA256|\
            VSS_DOCUMENTATION_POLICY_STATUS|VSS_DOCUMENTATION_DEVELOPMENT_ROOT|\
            VSS_DOCUMENTATION_ARCHIVE_ROOT) ;;
            *) report_failure "unknown adoption key: $key"; continue ;;
        esac
        if [[ -v "adoption_seen[$key]" ]]; then
            report_failure "duplicate adoption key: $key"
            continue
        fi
        adoption_seen[$key]=1
        adoption_values[$key]=$value
        adoption_order+=( "$key" )
    done < "$adoption_file"

    expected_index=0
    for key in "${expected_keys[@]}"; do
        if ! [[ -v "adoption_values[$key]" ]]; then
            report_failure "required adoption key is missing: $key"
            continue
        fi
        if [ "${adoption_order[$expected_index]-}" != "$key" ]; then
            report_failure "adoption keys are not in the required order at $key"
        fi
        expected_index=$((expected_index + 1))
    done
    if [ "${#adoption_order[@]}" -ne "${#expected_keys[@]}" ]; then
        report_failure 'adoption file contains an incorrect number of keys'
    fi

    if ! [[ "${adoption_values[VSS_DOCUMENTATION_POLICY_VERSION]-}" == 1.0.0 ]]; then
        report_failure 'unsupported documentation policy version'
    fi
    if [ "${adoption_values[VSS_DOCUMENTATION_POLICY_SOURCE_REPOSITORY]-}" != VirtualSensorSoft/VSSMath ]; then
        report_failure 'documentation policy is not maintained by VirtualSensorSoft/VSSMath'
    fi
    if ! [[ "${adoption_values[VSS_DOCUMENTATION_POLICY_SOURCE_COMMIT]-}" =~ ^[0-9a-f]{40}$ ]]; then
        report_failure 'adoption source commit is malformed'
    fi
    if ! [[ "${adoption_values[VSS_DOCUMENTATION_POLICY_CHECKER_SHA256]-}" =~ ^[0-9a-f]{64}$ ]]; then
        report_failure 'adoption checker digest is malformed'
    fi
    if [ "${adoption_values[VSS_DOCUMENTATION_POLICY_STATUS]-}" != LocallyMaintained ]; then
        report_failure 'public documentation policy status must be LocallyMaintained'
    fi

    dev_root=${adoption_values[VSS_DOCUMENTATION_DEVELOPMENT_ROOT]-}
    archive_root=${adoption_values[VSS_DOCUMENTATION_ARCHIVE_ROOT]-}
    validate_relative_path 'development root' "$dev_root"
    validate_relative_path 'archive root' "$archive_root"

    checker="$root/scripts/check-documentation-policy.sh"
    if [ ! -x "$checker" ]; then
        report_failure 'vendored checker is missing or not executable'
    elif [[ "${adoption_values[VSS_DOCUMENTATION_POLICY_CHECKER_SHA256]-}" =~ ^[0-9a-f]{64}$ ]]; then
        actual_digest=$(sha256sum "$checker" | awk '{print $1}')
        if [ "$actual_digest" != "${adoption_values[VSS_DOCUMENTATION_POLICY_CHECKER_SHA256]}" ]; then
            report_failure 'vendored checker digest does not match adoption metadata'
        fi
    fi
}

validate_required_files() {
    required_files=(
        README.md
        docs/README.md
        docs/architecture.md
        docs/status.md
        "$dev_root/README.md"
        "$adoption_arg"
        "$inventory_arg"
        "$archive_root/README.md"
        "$archive_root/path-map.tsv"
        scripts/check-documentation-policy.sh
    )
    for required in "${required_files[@]}"; do
        [ -f "$root/$required" ] || report_failure "required file is missing: $required"
    done
}

validate_inventory() {
    inventory_file="$root/$inventory_arg"
    if [ ! -f "$inventory_file" ]; then return 0; fi
    header=
    IFS= read -r header < "$inventory_file" || true
    expected_header=$'source_path\tclass\tdisposition\tdestination_path\ttracking_url\tcompatibility_stub\tnotes'
    [ "$header" = "$expected_header" ] || report_failure 'migration inventory header is invalid'

    line_number=1
    while IFS= read -r line || [ -n "$line" ]; do
        line_number=$((line_number + 1))
        [ "$line_number" -eq 2 ] && continue
        [ -n "$line" ] || { report_failure "empty migration inventory row at $line_number"; continue; }
        field_count=$(printf '%s\n' "$line" | awk -F '\t' '{print NF}')
        if [ "$field_count" -ne 7 ]; then
            report_failure "migration inventory row $line_number has $field_count fields"
            continue
        fi
        IFS=$'\t' read -r source class disposition destination tracking stub notes <<<"$line"
        if [[ -v "inventory_seen[$source]" ]]; then
            report_failure "duplicate migration inventory source: $source"
        fi
        inventory_seen[$source]=1
        validate_relative_path 'inventory source path' "$source"
        [ -n "$destination" ] && validate_relative_path 'inventory destination path' "$destination"
        case "$class" in
            living-current|active-change|implemented-history|formal-evidence|\
            superseded|routine-review-duplicate|obsolete-no-unique-value) ;;
            *) report_failure "unrecognized migration class: $class" ;;
        esac
        case "$disposition" in
            retain|rewrite-in-place|move|consolidate|replace|remove) ;;
            *) report_failure "unrecognized migration disposition: $disposition" ;;
        esac
        case "$stub" in
            yes|no) ;;
            *) report_failure "unrecognized compatibility-stub value: $stub" ;;
        esac
        case "$tracking" in
            https://github.com/VirtualSensorSoft/*|None) ;;
            *) report_failure "migration tracking URL is not canonical: $source" ;;
        esac
        case "$disposition" in
            retain|rewrite-in-place)
                [ -e "$root/$source" ] || report_failure "retained inventory source is missing: $source"
                ;;
            remove)
                [ ! -e "$root/$source" ] || report_failure "removed inventory source still exists: $source"
                ;;
        esac
        if [ "$stub" = yes ]; then
            if [ -z "$destination" ] || [ ! -e "$root/$destination" ]; then
                report_failure "compatibility stub destination is missing: $source"
            fi
            if [ ! -f "$root/$source" ]; then
                report_failure "compatibility stub source is missing: $source"
            else
                grep -Fq '# Historical Record Link' "$root/$source" || \
                    report_failure "compatibility stub is not minimal: $source"
                grep -Fq 'Move commit:' "$root/$source" || \
                    report_failure "compatibility stub has no move commit: $source"
            fi
            if [ -n "$destination" ] && [ -f "$root/$source" ] && [ -f "$root/$destination" ] && \
                cmp -s "$root/$source" "$root/$destination"; then
                report_failure "old path retains a full duplicate: $source"
            fi
        fi
    done < "$inventory_file"
}

validate_path_map() {
    path_map_file="$root/$archive_root/path-map.tsv"
    if [ ! -f "$path_map_file" ]; then return 0; fi
    header=
    IFS= read -r header < "$path_map_file" || true
    expected_header=$'old_path\tnew_path\tmove_commit\ttracking_url\tcompatibility_stub'
    [ "$header" = "$expected_header" ] || report_failure 'archive path-map header is invalid'
    line_number=1
    while IFS= read -r line || [ -n "$line" ]; do
        line_number=$((line_number + 1))
        [ "$line_number" -eq 2 ] && continue
        [ -n "$line" ] || { report_failure "empty path-map row at $line_number"; continue; }
        field_count=$(printf '%s\n' "$line" | awk -F '\t' '{print NF}')
        if [ "$field_count" -ne 5 ]; then
            report_failure "path-map row $line_number has $field_count fields"
            continue
        fi
        IFS=$'\t' read -r old_path new_path move_commit tracking stub <<<"$line"
        if [[ -v "path_map_seen[$old_path]" ]]; then
            report_failure "duplicate path-map source: $old_path"
        fi
        path_map_seen[$old_path]=1
        validate_relative_path 'path-map old path' "$old_path"
        validate_relative_path 'path-map new path' "$new_path"
        if ! [[ "$move_commit" =~ ^[0-9a-f]{40}$ ]]; then
            report_failure "path-map move commit is malformed: $old_path"
        fi
        case "$tracking" in
            https://github.com/VirtualSensorSoft/*|None) ;;
            *) report_failure "path-map tracking URL is not canonical: $old_path" ;;
        esac
        case "$stub" in yes|no) ;; *) report_failure "path-map stub value is invalid: $old_path" ;; esac
    done < "$path_map_file"
}

metadata_value() {
    key=$1
    file=$2
    sed -n "s/^- $key: //p" "$file" | head -n 1
}

validate_record() {
    record_kind=$1
    record_directory=$2
    specification="$record_directory/specification.md"
    plan="$record_directory/implementation-plan.md"
    if [ ! -f "$specification" ]; then
        report_failure "$record_kind record is missing specification.md: $record_directory"
        return
    fi
    record_id=$(metadata_value 'Record ID' "$specification")
    status=$(metadata_value 'Status' "$specification")
    owner=$(metadata_value 'Owner' "$specification")
    scope=$(metadata_value 'Scope' "$specification")
    issue=$(metadata_value 'Tracking issue' "$specification")
    approval=$(metadata_value 'Approval pull request' "$specification")
    implementation=$(metadata_value 'Implementation pull request(s)' "$specification")
    superseded_by=$(metadata_value 'Superseded by' "$specification")
    supersedes=$(metadata_value 'Supersedes' "$specification")
    [ -n "$record_id" ] || report_failure "record ID is missing: $record_directory"
    [ -n "$status" ] || report_failure "record status is missing: $record_directory"
    [ -n "$owner" ] || report_failure "record owner is missing: $record_directory"
    [ -n "$scope" ] || report_failure "record scope is missing: $record_directory"
    [ -n "$issue" ] || report_failure "record tracking issue is missing: $record_directory"
    [ -n "$approval" ] || report_failure "record approval pull request is missing: $record_directory"
    [ -n "$implementation" ] || report_failure "record implementation pull request is missing: $record_directory"
    if [ -n "$record_id" ]; then
        if [[ -v "record_path[$record_id]" ]]; then
            report_failure "duplicate record ID $record_id: $record_directory and ${record_path[$record_id]}"
        else
            record_path[$record_id]=$record_directory
            record_status[$record_id]=$status
            record_superseded_by[$record_id]=$superseded_by
            record_supersedes[$record_id]=$supersedes
        fi
    fi
    if [ "$record_kind" = active ]; then
        case "$status" in
            Provisional|Accepted|Implementing|Approved|Active) ;;
            *) report_failure "terminal or invalid record is in active navigation: $record_directory" ;;
        esac
    else
        case "$status" in
            Implemented|Completed|Superseded|Rejected|Withdrawn|Deferred|Abandoned) ;;
            *) report_failure "non-terminal record is in archive navigation: $record_directory" ;;
        esac
    fi
    if [ "$record_kind" = archive ] || [ "$status" != Provisional ]; then
        if [ ! -f "$plan" ]; then
            report_failure "$record_kind record is missing implementation-plan.md: $record_directory"
        fi
    fi
    case "$status" in
        Accepted|Implementing|Approved|Active)
            while IFS= read -r -d '' file; do
                if grep -Eq '(^|[^A-Za-z])(TBD|TODO|FIXME)([^A-Za-z]|$)' "$file"; then
                    report_failure "accepted record contains unresolved marker: $file"
                fi
            done < <(find -P "$record_directory" -type f -name '*.md' -print0)
            ;;
    esac
    if [ -f "$record_directory/evidence.md" ]; then
        grep -Fq '## Assessed Identity' "$record_directory/evidence.md" || \
            report_failure "formal evidence lacks Assessed Identity: $record_directory"
        grep -Fq '## Limitations' "$record_directory/evidence.md" || \
            report_failure "formal evidence lacks Limitations: $record_directory"
    fi
}

validate_active_records() {
    changes="$root/$dev_root/changes"
    if [ ! -d "$changes" ]; then return 0; fi
    while IFS= read -r -d '' directory; do
        validate_record active "$directory"
    done < <(find -P "$changes" -mindepth 1 -maxdepth 1 -type d -print0)
}

validate_archive_records() {
    changes="$root/$archive_root/changes"
    if [ ! -d "$changes" ]; then return 0; fi
    while IFS= read -r -d '' directory; do
        validate_record archive "$directory"
    done < <(find -P "$changes" -mindepth 1 -maxdepth 1 -type d -print0)
}

validate_record_relationships() {
    for record_id in "${!record_status[@]}"; do
        if [ "${record_status[$record_id]}" = Superseded ]; then
            successor=${record_superseded_by[$record_id]-}
            if [ -z "$successor" ] || [ "$successor" = None ] || ! [[ -v "record_path[$successor]" ]]; then
                report_failure "superseded record has no two-way successor: $record_id"
            elif [ "${record_supersedes[$successor]}" != "$record_id" ]; then
                report_failure "superseded relationship is not two-way: $record_id"
            fi
        fi
    done
}

validate_relative_links() {
    mapfile -d '' markdown_files < <(find -P "$root" -type f -name '*.md' -not -path "$root/.git/*" -print0)
    for file in "${markdown_files[@]}"; do
        while IFS= read -r link; do
            case "$link" in
                ''|http://*|https://*|mailto:*|$#*) continue ;;
            esac
            target=${link%%#*}
            [ -n "$target" ] || continue
            case "$target" in *'<'*|*'>'*) continue ;; esac
            candidate="$(dirname "$file")/$target"
            resolved=$(realpath -m -- "$candidate")
            case "$resolved" in
                "$root"|"$root"/*) ;;
                *) report_failure "relative link escapes repository: $file -> $link"; continue ;;
            esac
            [ -e "$candidate" ] || report_failure "broken relative link: $file -> $link"
        done < <(grep -Eo '\]\([^)]*\)' "$file" | sed -E 's/^.*\]\(([^)]*)\)$/\1/' || true)
done
}

validate_repository_symlinks() {
    while IFS= read -r -d '' path; do
        relative=${path#"$root/"}
        resolved=$(realpath -e -- "$path" 2>/dev/null || true)
        if [ -z "$resolved" ]; then
            report_failure "broken repository symlink: $relative"
            continue
        fi
        case "$resolved" in
            "$root"|"$root"/*) ;;
            *) report_failure "symlink escapes the repository: $relative" ;;
        esac
    done < <(find -P "$root" -path "$root/.git" -prune -o -type l -print0)
}

validate_proposed_output() {
    while IFS= read -r -d '' relative; do
        file="$root/$relative"
        [ -f "$file" ] && [ ! -L "$file" ] || continue
        case "$relative" in
            *.html|*.htm|*/build/*|build/*|*/dist/*|dist/*|*/CMakeFiles/*)
                report_failure "generated or build output is present: $relative"
                ;;
        esac
    done < <(git -C "$root" ls-files -z --cached --others --exclude-standard)
}

validate_confidentiality_markers() {
    marker_one=$(printf '%s%s' mar lin)
    marker_two=$(printf '%s%s' ilc soft)
    marker_three=$(printf '%s%s' lc io)
    marker_four=$(printf '%s%s' mar lin reco)
    while IFS= read -r -d '' relative; do
        file="$root/$relative"
        [ -f "$file" ] && [ ! -L "$file" ] || continue
        [ "$file" = "$root/scripts/check-documentation-policy.sh" ] && continue
        for marker in "$marker_one" "$marker_two" "$marker_three" "$marker_four"; do
            if grep -Fqi -- "$marker" "$file"; then report_failure "prohibited external-name marker found: $file"; fi
        done
        home_marker=$(printf '%s%s' '/ho' 'me/')
        users_marker=$(printf '%s%s' '/U' 'sers/')
        for marker in 'ghp_' 'github_pat_' 'BEGIN PRIVATE KEY' 'token=' 'token:' "$home_marker" "$users_marker"; do
            if grep -Fqi -- "$marker" "$file"; then report_failure "confidentiality marker found: $file"; fi
        done
    done < <(git -C "$root" ls-files -z --cached --others --exclude-standard)
}

parse_arguments "$@"
if [ "$failures" -ne 0 ]; then
    exit 2
fi
if ! root=$(cd "$root_arg" && pwd -P); then
    report_failure "repository root is not a directory: $root_arg"
    exit 2
fi
repository_top=$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || true)
if [ "$repository_top" != "$root" ]; then
    report_failure 'repository root must be the top-level directory of a Git working tree'
    exit 2
fi
validate_relative_path 'adoption argument' "$adoption_arg"
validate_relative_path 'inventory argument' "$inventory_arg"
if [ "$failures" -ne 0 ]; then
    exit 2
fi
load_adoption
if [ "$failures" -ne 0 ]; then
    exit 2
fi
validate_required_files
validate_inventory
validate_path_map
validate_active_records
validate_archive_records
validate_record_relationships
validate_relative_links
validate_repository_symlinks
validate_proposed_output
validate_confidentiality_markers

if [ "$failures" -ne 0 ]; then
    exit 2
fi
printf '%s\n' 'Documentation policy 1.0.0 checks passed'
