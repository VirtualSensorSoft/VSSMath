cmake_minimum_required(VERSION 3.28)
include("${CMAKE_CURRENT_LIST_DIR}/qualification_contract.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_session_schema.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_batch_plan.cmake")

macro(vss_stage_fail token reason)
  message(FATAL_ERROR "VSSMath ${token}: ${reason}")
endmacro()

foreach(required VSS_PERF_SESSION_DIR VSS_PERF_BASELINE_DIR VSS_PERF_SOURCE_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    vss_stage_fail(invalid_record "${required} is required")
  endif()
endforeach()
foreach(path_var VSS_PERF_SESSION_DIR VSS_PERF_BASELINE_DIR VSS_PERF_SOURCE_DIR)
  if(NOT IS_ABSOLUTE "${${path_var}}")
    vss_stage_fail(invalid_record "${path_var} must be absolute")
  endif()
endforeach()
if(NOT IS_DIRECTORY "${VSS_PERF_SESSION_DIR}" OR
   NOT IS_DIRECTORY "${VSS_PERF_SOURCE_DIR}" OR
   IS_SYMLINK "${VSS_PERF_SESSION_DIR}" OR IS_SYMLINK "${VSS_PERF_SOURCE_DIR}")
  vss_stage_fail(invalid_record "session or source directory is missing or symlinked")
endif()
file(REAL_PATH "${VSS_PERF_SESSION_DIR}" session_real)
file(REAL_PATH "${VSS_PERF_SOURCE_DIR}" source_real)

set(manifest "${session_real}/qualification.env")
vss_contract_read_kv("${manifest}" VSS_MANIFEST "${VSS_ALGORITHM_MANIFEST_KEYS}"
  manifest_ok manifest_reason)
if(NOT manifest_ok)
  vss_stage_fail(invalid_record "${manifest_reason}")
endif()
if(NOT VSS_MANIFEST_SCHEMA_VERSION STREQUAL "2" OR
   NOT VSS_MANIFEST_RESULT STREQUAL "qualified_pass" OR
   NOT VSS_MANIFEST_HOST_PROFILE STREQUAL "performance-primary-local-linux-v1" OR
   NOT VSS_MANIFEST_CONFIGURATION MATCHES "^(Debug|Release)$" OR
   NOT VSS_MANIFEST_RECORD_PROFILE MATCHES "^performance-primary-gcc-(debug|release)$")
  vss_stage_fail(invalid_record "schema, qualification token, or identity is not approved")
endif()
if((VSS_MANIFEST_CONFIGURATION STREQUAL "Debug" AND
    NOT VSS_MANIFEST_RECORD_PROFILE STREQUAL "performance-primary-gcc-debug") OR
   (VSS_MANIFEST_CONFIGURATION STREQUAL "Release" AND
    NOT VSS_MANIFEST_RECORD_PROFILE STREQUAL "performance-primary-gcc-release"))
  vss_stage_fail(invalid_record "configuration-to-profile mapping is invalid")
endif()
vss_contract_require_commit("${VSS_MANIFEST_MATH_COMMIT}" MATH_COMMIT ok reason)
if(NOT ok)
  vss_stage_fail(invalid_record "${reason}")
endif()
foreach(hash_key IN ITEMS BENCHMARK_SHA256 HOST_FINGERPRINT_SHA256)
  vss_contract_require_hex64("${VSS_MANIFEST_${hash_key}}" "${hash_key}" ok reason)
  if(NOT ok)
    vss_stage_fail(invalid_record "${reason}")
  endif()
endforeach()
if(VSS_MANIFEST_HOST_FINGERPRINT_SHA256 MATCHES
   "^0000000000000000000000000000000000000000000000000000000000000000$")
  vss_stage_fail(invalid_record "host fingerprint is a zero placeholder")
endif()
foreach(integer_key IN ITEMS AFFINITY_CPU MAX_PACKAGE_TEMP_MILLIC
    STABLE_WINDOW_SECONDS SAMPLE_INTERVAL_SECONDS MAX_WAIT_SECONDS)
  vss_contract_require_integer("${VSS_MANIFEST_${integer_key}}" "${integer_key}" ok reason)
  if(NOT ok)
    vss_stage_fail(invalid_record "${reason}")
  endif()
endforeach()
if(NOT VSS_MANIFEST_AFFINITY_CPU STREQUAL "4" OR
   NOT VSS_MANIFEST_MAX_LOAD_1M STREQUAL "0.50" OR
   NOT VSS_MANIFEST_MAX_PACKAGE_TEMP_MILLIC STREQUAL "55000" OR
   NOT VSS_MANIFEST_STABLE_WINDOW_SECONDS STREQUAL "60" OR
   NOT VSS_MANIFEST_SAMPLE_INTERVAL_SECONDS STREQUAL "10" OR
   NOT VSS_MANIFEST_MAX_WAIT_SECONDS STREQUAL "900")
  vss_stage_fail(invalid_record "manifest host qualification limits differ from the reviewed profile")
endif()

set(expected_paths
  CANDIDATE_PATH "candidate/${VSS_MANIFEST_RECORD_PROFILE}.json"
  VERIFICATION_1_PATH verification-1.json
  VERIFICATION_2_PATH verification-2.json
  PREFLIGHT_CANDIDATE_PATH preflight-candidate.env
  PREFLIGHT_VERIFICATION_1_PATH preflight-verification-1.env
  PREFLIGHT_VERIFICATION_2_PATH preflight-verification-2.env
  BATCH_PLAN_PATH batch-plan.env
  CANDIDATE_QUALITY_PATH candidate-quality.result
  VERIFICATION_1_QUALITY_PATH verification-1-quality.result
  VERIFICATION_2_QUALITY_PATH verification-2-quality.result
  REPEATABILITY_PATH repeatability.result)
while(expected_paths)
  list(POP_FRONT expected_paths key expected)
  if(NOT VSS_MANIFEST_${key} STREQUAL expected)
    vss_stage_fail(invalid_record "${key} is not the canonical session path")
  endif()
  vss_contract_require_relative_path("${VSS_MANIFEST_${key}}" "${key}" ok reason)
  if(NOT ok)
    vss_stage_fail(invalid_record "${reason}")
  endif()
endwhile()

set(candidate "${session_real}/${VSS_MANIFEST_CANDIDATE_PATH}")
set(verification_1 "${session_real}/${VSS_MANIFEST_VERIFICATION_1_PATH}")
set(verification_2 "${session_real}/${VSS_MANIFEST_VERIFICATION_2_PATH}")
set(preflight_candidate "${session_real}/${VSS_MANIFEST_PREFLIGHT_CANDIDATE_PATH}")
set(preflight_1 "${session_real}/${VSS_MANIFEST_PREFLIGHT_VERIFICATION_1_PATH}")
set(preflight_2 "${session_real}/${VSS_MANIFEST_PREFLIGHT_VERIFICATION_2_PATH}")
set(batch_plan "${session_real}/${VSS_MANIFEST_BATCH_PLAN_PATH}")
set(candidate_quality "${session_real}/${VSS_MANIFEST_CANDIDATE_QUALITY_PATH}")
set(verification_1_quality "${session_real}/${VSS_MANIFEST_VERIFICATION_1_QUALITY_PATH}")
set(verification_2_quality "${session_real}/${VSS_MANIFEST_VERIFICATION_2_QUALITY_PATH}")
set(repeatability_result "${session_real}/${VSS_MANIFEST_REPEATABILITY_PATH}")
if(IS_SYMLINK "${session_real}/candidate")
  vss_stage_fail(invalid_record "candidate evidence directory is symlinked")
endif()
set(evidence_paths "${candidate}" "${verification_1}" "${verification_2}"
  "${preflight_candidate}" "${preflight_1}" "${preflight_2}" "${batch_plan}"
  "${candidate_quality}" "${verification_1_quality}"
  "${verification_2_quality}" "${repeatability_result}")
set(evidence_keys CANDIDATE_SHA256 VERIFICATION_1_SHA256 VERIFICATION_2_SHA256
  PREFLIGHT_CANDIDATE_SHA256 PREFLIGHT_VERIFICATION_1_SHA256
  PREFLIGHT_VERIFICATION_2_SHA256 BATCH_PLAN_SHA256 CANDIDATE_QUALITY_SHA256
  VERIFICATION_1_QUALITY_SHA256 VERIFICATION_2_QUALITY_SHA256 REPEATABILITY_SHA256)
list(LENGTH evidence_paths evidence_count)
math(EXPR evidence_last "${evidence_count} - 1")
foreach(index RANGE 0 "${evidence_last}")
  list(GET evidence_paths "${index}" evidence_path)
  list(GET evidence_keys "${index}" hash_key)
  vss_contract_read_sha256("${evidence_path}" "${VSS_MANIFEST_${hash_key}}"
    "${hash_key}" evidence_ok evidence_reason)
  if(NOT evidence_ok)
    vss_stage_fail(invalid_record "${evidence_reason}")
  endif()
  file(REAL_PATH "${evidence_path}" evidence_real)
  string(FIND "${evidence_real}" "${session_real}/" session_prefix)
  if(NOT session_prefix EQUAL 0)
    vss_stage_fail(invalid_record "evidence escapes the canonical session")
  endif()
endforeach()

execute_process(COMMAND git -C "${source_real}" rev-parse HEAD
  OUTPUT_VARIABLE actual_commit OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE git_status)
if(NOT git_status EQUAL 0 OR NOT actual_commit STREQUAL VSS_MANIFEST_MATH_COMMIT)
  vss_stage_fail(invalid_record "source HEAD does not match the manifest commit")
endif()
execute_process(COMMAND git -C "${source_real}" diff --quiet
  RESULT_VARIABLE unstaged_status)
execute_process(COMMAND git -C "${source_real}" diff --cached --quiet
  RESULT_VARIABLE staged_status)
if(NOT unstaged_status EQUAL 0 OR NOT staged_status EQUAL 0)
  vss_stage_fail(invalid_record "tracked source worktree or index is not clean")
endif()

function(vss_stage_record path require_path out_json out_timestamp)
  vss_algorithm_record_load("${path}" "${require_path}" json token reason)
  if(NOT token STREQUAL "record_valid")
    vss_stage_fail("${token}" "${path}: ${reason}")
  endif()
  string(JSON profile GET "${json}" profile_id)
  string(JSON environment GET "${json}" environment)
  foreach(pair IN ITEMS
      "math_commit|${VSS_MANIFEST_MATH_COMMIT}"
      "benchmark_sha256|${VSS_MANIFEST_BENCHMARK_SHA256}"
      "configuration|${VSS_MANIFEST_CONFIGURATION}"
      "frequency_governor|powersave"
      "turbo_state|enabled"
      "runner_label_or_host|${VSS_MANIFEST_HOST_PROFILE}")
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 key)
    list(GET fields 1 expected)
    string(JSON value GET "${environment}" "${key}")
    if(NOT value STREQUAL expected)
      vss_stage_fail(incompatible "${path}: record identity differs at ${key}")
    endif()
  endforeach()
  if(NOT profile STREQUAL VSS_MANIFEST_RECORD_PROFILE)
    vss_stage_fail(incompatible "${path}: profile differs from manifest")
  endif()
  string(JSON timestamp GET "${environment}" utc_timestamp)
  vss_contract_utc_seconds("${timestamp}" timestamp_ok timestamp_seconds timestamp_reason)
  if(NOT timestamp_ok)
    vss_stage_fail(invalid_record "${path}: ${timestamp_reason}")
  endif()
  set("${out_json}" "${json}" PARENT_SCOPE)
  set("${out_timestamp}" "${timestamp_seconds}" PARENT_SCOPE)
endfunction()

vss_stage_record("${candidate}" TRUE candidate_json candidate_seconds)
vss_stage_record("${verification_1}" FALSE verification_1_json verification_1_seconds)
vss_stage_record("${verification_2}" FALSE verification_2_json verification_2_seconds)

function(vss_stage_preflight_field path key out_value)
  file(STRINGS "${path}" matching_line REGEX "^${key}=")
  if(NOT matching_line)
    vss_stage_fail(invalid_record "${path}: missing ${key}")
  endif()
  string(REGEX REPLACE "^[^=]*=" "" value "${matching_line}")
  set("${out_value}" "${value}" PARENT_SCOPE)
endfunction()

function(vss_stage_bind_record_hardware record_json preflight_path label)
  string(JSON environment GET "${record_json}" environment)
  foreach(pair IN ITEMS
      "os_name|OS_NAME"
      "architecture|ARCHITECTURE"
      "cpu_model|CPU_MODEL"
      "logical_cores|LOGICAL_CPU_COUNT")
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 record_key)
    list(GET fields 1 preflight_key)
    vss_stage_preflight_field("${preflight_path}" "${preflight_key}" preflight_value)
    string(JSON record_value GET "${environment}" "${record_key}")
    if(NOT "${record_value}" STREQUAL "${preflight_value}")
      vss_stage_fail(incompatible
        "${label}: record ${record_key} '${record_value}' differs from paired preflight '${preflight_value}'")
    endif()
  endforeach()
endfunction()

vss_stage_bind_record_hardware("${candidate_json}" "${preflight_candidate}" "candidate")
vss_stage_bind_record_hardware("${verification_1_json}" "${preflight_1}" "verification 1")
vss_stage_bind_record_hardware("${verification_2_json}" "${preflight_2}" "verification 2")

foreach(prefix IN ITEMS "${preflight_1}" "${preflight_2}")
  foreach(key IN ITEMS HOST_FINGERPRINT_SHA256 OS_NAME ARCHITECTURE CPU_MODEL
      LOGICAL_CPU_COUNT AFFINITY_CPU AFFINITY_SIBLINGS BATTERY_STATUS
      INTEL_PSTATE_MODE SCALING_GOVERNOR INTEL_PSTATE_NO_TURBO)
    vss_stage_preflight_field("${preflight_candidate}" "${key}" candidate_value)
    vss_stage_preflight_field("${prefix}" "${key}" verification_value)
    if(NOT "${verification_value}" STREQUAL "${candidate_value}")
      vss_stage_fail(preflight_failed "preflight environment differs at ${key}")
    endif()
  endforeach()
endforeach()

foreach(role IN ITEMS CANDIDATE VERIFICATION_1 VERIFICATION_2)
  if(role STREQUAL "CANDIDATE")
    set(record_json "${candidate_json}")
    set(manifest_token "${VSS_MANIFEST_CANDIDATE_QUALITY_RESULT}")
    set(result_path "${candidate_quality}")
    set(label candidate)
  elseif(role STREQUAL "VERIFICATION_1")
    set(record_json "${verification_1_json}")
    set(manifest_token "${VSS_MANIFEST_VERIFICATION_1_QUALITY_RESULT}")
    set(result_path "${verification_1_quality}")
    set(label "verification 1")
  else()
    set(record_json "${verification_2_json}")
    set(manifest_token "${VSS_MANIFEST_VERIFICATION_2_QUALITY_RESULT}")
    set(result_path "${verification_2_quality}")
    set(label "verification 2")
  endif()
  if(NOT manifest_token STREQUAL stable_pass)
    vss_stage_fail(invalid_record "${label} manifest quality token is not stable_pass")
  endif()
  file(READ "${result_path}" result_bytes)
  if(NOT result_bytes STREQUAL "stable_pass\n")
    vss_stage_fail(invalid_record "${label} quality result file is not the exact stable_pass token")
  endif()
  vss_algorithm_record_quality("${record_json}" quality_token quality_reason)
  if(NOT quality_token STREQUAL stable_pass)
    vss_stage_fail("${quality_token}" "${label} quality validation: ${quality_reason}")
  endif()
endforeach()
vss_algorithm_batch_plan_validate("${batch_plan}" "${candidate}" plan_token plan_reason)
if(NOT plan_token STREQUAL batch_plan_valid)
  vss_stage_fail("${plan_token}" "batch plan validation: ${plan_reason}")
endif()
file(READ "${repeatability_result}" repeatability_bytes)
if(NOT VSS_MANIFEST_REPEATABILITY_RESULT STREQUAL repeatable_pass OR
   NOT repeatability_bytes STREQUAL "repeatable_pass\n")
  vss_stage_fail(invalid_record "repeatability result is not the exact success token")
endif()
vss_algorithm_repeatability("${candidate}" "${verification_1}" "${verification_2}"
  repeatability_token repeatability_reason)
if(NOT repeatability_token STREQUAL repeatable_pass)
  vss_stage_fail("${repeatability_token}" "repeatability validation: ${repeatability_reason}")
endif()

function(vss_stage_load_hundredths value out_ok out_value)
  if(NOT "${value}" MATCHES "^([0-9]+)(\\.([0-9][0-9]?))?$")
    set("${out_ok}" FALSE PARENT_SCOPE)
    set("${out_value}" 0 PARENT_SCOPE)
    return()
  endif()
  set(whole "${CMAKE_MATCH_1}")
  set(fraction "${CMAKE_MATCH_3}")
  if(fraction STREQUAL "")
    set(fraction 00)
  elseif(fraction MATCHES "^[0-9]$")
    string(APPEND fraction 0)
  endif()
  if(whole GREATER 4)
    set("${out_ok}" FALSE PARENT_SCOPE)
    set("${out_value}" 0 PARENT_SCOPE)
    return()
  endif()
  math(EXPR result "${whole} * 100 + ${fraction}")
  set("${out_ok}" TRUE PARENT_SCOPE)
  set("${out_value}" "${result}" PARENT_SCOPE)
endfunction()

macro(vss_stage_preflight path prefix)
  vss_contract_read_kv("${path}" "${prefix}" "${VSS_ALGORITHM_PREFLIGHT_KEYS}"
    preflight_ok preflight_reason)
  if(NOT preflight_ok)
    vss_stage_fail(preflight_failed "${preflight_reason}")
  endif()
  if(NOT "${${prefix}_SCHEMA_VERSION}" STREQUAL "1" OR
     NOT "${${prefix}_RESULT}" STREQUAL "preflight_pass" OR
     NOT "${${prefix}_HOST_PROFILE}" STREQUAL VSS_MANIFEST_HOST_PROFILE OR
     NOT "${${prefix}_RECORD_PROFILE}" STREQUAL VSS_MANIFEST_RECORD_PROFILE OR
     NOT "${${prefix}_CONFIGURATION}" STREQUAL VSS_MANIFEST_CONFIGURATION OR
     NOT "${${prefix}_MATH_COMMIT}" STREQUAL VSS_MANIFEST_MATH_COMMIT OR
     NOT "${${prefix}_BENCHMARK_SHA256}" STREQUAL VSS_MANIFEST_BENCHMARK_SHA256 OR
     NOT "${${prefix}_HOST_FINGERPRINT_SHA256}" STREQUAL VSS_MANIFEST_HOST_FINGERPRINT_SHA256)
    vss_stage_fail(preflight_failed "${prefix} identity/result mismatch")
  endif()
  foreach(pair IN ITEMS
      "OS_NAME|Linux" "ARCHITECTURE|x86_64" "CPU_MODEL|Synthetic test processor"
      "LOGICAL_CPU_COUNT|8" "AFFINITY_CPU|${VSS_MANIFEST_AFFINITY_CPU}"
      "AFFINITY_SIBLINGS|0,4" "AC_ONLINE|1" "INTEL_PSTATE_MODE|active"
      "SCALING_GOVERNOR|powersave" "INTEL_PSTATE_NO_TURBO|0"
      "MAX_LOAD_1M|${VSS_MANIFEST_MAX_LOAD_1M}"
      "MAX_PACKAGE_TEMP_MILLIC|${VSS_MANIFEST_MAX_PACKAGE_TEMP_MILLIC}"
      "STABLE_WINDOW_SECONDS|${VSS_MANIFEST_STABLE_WINDOW_SECONDS}"
      "SAMPLE_INTERVAL_SECONDS|${VSS_MANIFEST_SAMPLE_INTERVAL_SECONDS}"
      "MAX_WAIT_SECONDS|${VSS_MANIFEST_MAX_WAIT_SECONDS}")
    string(REPLACE "|" ";" fields "${pair}")
    list(GET fields 0 key)
    list(GET fields 1 expected)
    if(NOT "${${prefix}_${key}}" STREQUAL expected)
      vss_stage_fail(preflight_failed "${prefix} policy differs at ${key}")
    endif()
  endforeach()
  if("${${prefix}_BATTERY_STATUS}" STREQUAL "Discharging")
    vss_stage_fail(preflight_failed "${prefix} was captured on battery power")
  endif()
  vss_stage_load_hundredths("${${prefix}_LOAD_1M}" load_ok load_hundredths)
  if(NOT load_ok OR load_hundredths GREATER 50)
    vss_stage_fail(preflight_failed "${prefix} load or temperature exceeded reviewed limits")
  endif()
  set(package_temp "${${prefix}_PACKAGE_TEMP_MILLIC}")
  if(NOT package_temp MATCHES "^[0-9]+$")
    vss_stage_fail(preflight_failed "${prefix} package temperature is malformed")
  endif()
  if(package_temp GREATER 55000)
    vss_stage_fail(preflight_failed "${prefix} load or temperature exceeded reviewed limits")
  endif()
  foreach(timestamp_key IN ITEMS START_UTC END_UTC)
    vss_contract_utc_seconds("${${prefix}_${timestamp_key}}" time_ok "${prefix}_${timestamp_key}_SECONDS" time_reason)
    if(NOT time_ok)
      vss_stage_fail(preflight_failed "${prefix}: ${time_reason}")
    endif()
  endforeach()
  math(EXPR observed_window "${${prefix}_END_UTC_SECONDS} - ${${prefix}_START_UTC_SECONDS}")
  if(observed_window LESS VSS_MANIFEST_STABLE_WINDOW_SECONDS)
    vss_stage_fail(preflight_failed "${prefix} does not span the full stable window")
  endif()
endmacro()

vss_stage_preflight("${preflight_candidate}" CANDIDATE_PREFLIGHT)
vss_stage_preflight("${preflight_1}" VERIFICATION_1_PREFLIGHT)
vss_stage_preflight("${preflight_2}" VERIFICATION_2_PREFLIGHT)
foreach(prefix IN ITEMS VERIFICATION_1_PREFLIGHT VERIFICATION_2_PREFLIGHT)
  foreach(key IN ITEMS HOST_FINGERPRINT_SHA256 OS_NAME ARCHITECTURE CPU_MODEL
      LOGICAL_CPU_COUNT AFFINITY_CPU AFFINITY_SIBLINGS BATTERY_STATUS
      INTEL_PSTATE_MODE SCALING_GOVERNOR INTEL_PSTATE_NO_TURBO)
    if(NOT "${${prefix}_${key}}" STREQUAL "${CANDIDATE_PREFLIGHT_${key}}")
      vss_stage_fail(preflight_failed "preflight environment differs at ${key}")
    endif()
  endforeach()
endforeach()

vss_contract_utc_seconds("${VSS_MANIFEST_START_UTC}" time_ok manifest_start_seconds time_reason)
if(NOT time_ok)
  vss_stage_fail(invalid_record "manifest: ${time_reason}")
endif()
vss_contract_utc_seconds("${VSS_MANIFEST_END_UTC}" time_ok manifest_end_seconds time_reason)
if(NOT time_ok)
  vss_stage_fail(invalid_record "manifest: ${time_reason}")
endif()
foreach(prefix IN ITEMS CANDIDATE_PREFLIGHT VERIFICATION_1_PREFLIGHT VERIFICATION_2_PREFLIGHT)
  if(${prefix}_START_UTC_SECONDS LESS manifest_start_seconds OR
     ${prefix}_END_UTC_SECONDS GREATER manifest_end_seconds)
    vss_stage_fail(invalid_record "${prefix} falls outside the manifest interval")
  endif()
endforeach()
if(CANDIDATE_PREFLIGHT_END_UTC_SECONDS GREATER candidate_seconds OR
   candidate_seconds GREATER VERIFICATION_1_PREFLIGHT_START_UTC_SECONDS OR
   VERIFICATION_1_PREFLIGHT_END_UTC_SECONDS GREATER verification_1_seconds OR
   verification_1_seconds GREATER VERIFICATION_2_PREFLIGHT_START_UTC_SECONDS OR
   VERIFICATION_2_PREFLIGHT_END_UTC_SECONDS GREATER verification_2_seconds OR
   verification_2_seconds GREATER manifest_end_seconds)
  vss_stage_fail(invalid_record "record/preflight UTC role ordering is invalid")
endif()

execute_process(COMMAND git -C "${source_real}" diff --cached --quiet --ignore-submodules --
  RESULT_VARIABLE index_status)
if(NOT index_status EQUAL 0)
  vss_stage_fail(invalid_record "source index is not clean")
endif()
set(expected_baseline_dir "${source_real}/tests/performance/baselines/v2")
cmake_path(NORMAL_PATH expected_baseline_dir)
set(requested_baseline_dir "${VSS_PERF_BASELINE_DIR}")
cmake_path(NORMAL_PATH requested_baseline_dir)
if(NOT requested_baseline_dir STREQUAL expected_baseline_dir)
  vss_stage_fail(invalid_record "destination must be the canonical schema-v2 baseline directory")
endif()
foreach(relative IN ITEMS tests tests/performance tests/performance/baselines
    tests/performance/baselines/v2)
  set(ancestor "${source_real}/${relative}")
  if(EXISTS "${ancestor}" AND IS_SYMLINK "${ancestor}")
    vss_stage_fail(invalid_record "canonical destination has a symlink ancestor")
  endif()
endforeach()
if(IS_SYMLINK "${expected_baseline_dir}")
  vss_stage_fail(invalid_record "canonical destination directory is symlinked")
endif()
set(destination "${expected_baseline_dir}/${VSS_MANIFEST_RECORD_PROFILE}.json")
set(staging_record "${expected_baseline_dir}/.${VSS_MANIFEST_RECORD_PROFILE}.json.tmp")
if(EXISTS "${destination}" OR IS_SYMLINK "${destination}" OR
   EXISTS "${staging_record}" OR IS_SYMLINK "${staging_record}")
  vss_stage_fail(invalid_record "refusing to overwrite a baseline or reuse a staging temporary")
endif()
file(MAKE_DIRECTORY "${expected_baseline_dir}")
file(REAL_PATH "${expected_baseline_dir}" baseline_real)
if(NOT baseline_real STREQUAL expected_baseline_dir)
  vss_stage_fail(invalid_record "canonical baseline directory escaped through a symlink")
endif()
file(COPY_FILE "${candidate}" "${staging_record}" RESULT copy_status)
if(NOT copy_status STREQUAL "0")
  file(REMOVE "${staging_record}")
  vss_stage_fail(qualification_incomplete "candidate copy failed: ${copy_status}")
endif()
file(SHA256 "${staging_record}" staged_sha)
if(NOT staged_sha STREQUAL VSS_MANIFEST_CANDIDATE_SHA256)
  file(REMOVE "${staging_record}")
  vss_stage_fail(invalid_record "staged candidate SHA-256 differs from manifest")
endif()
file(RENAME "${staging_record}" "${destination}" NO_REPLACE RESULT rename_status)
if(NOT rename_status STREQUAL "0")
  file(REMOVE "${staging_record}")
  vss_stage_fail(qualification_incomplete "atomic baseline publication failed: ${rename_status}")
endif()
message(STATUS "VSSMath staged qualified schema-v2 baseline ${destination}")
