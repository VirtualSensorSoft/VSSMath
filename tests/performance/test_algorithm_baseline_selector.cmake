cmake_minimum_required(VERSION 3.28)

foreach(required IN ITEMS VSS_PERF_CMAKE_COMMAND VSS_PERF_SOURCE_DIR
    VSS_PERF_SCRATCH)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
include("${VSS_PERF_SOURCE_DIR}/tests/performance/support/algorithm_fixtures.cmake")
set(selector "${VSS_PERF_SOURCE_DIR}/tests/performance/select_algorithm_baseline.cmake")
set(checker "${VSS_PERF_SOURCE_DIR}/tests/performance/check_algorithm_regression.cmake")
file(MAKE_DIRECTORY "${VSS_PERF_SCRATCH}")

function(write_record path profile configuration median allocations)
  vss_algorithm_fixture_record(json "${profile}" "${configuration}"
    "${median}" 1 "${allocations}")
  file(WRITE "${path}" "${json}")
endfunction()

set(current "${VSS_PERF_SCRATCH}/current.json")
write_record("${current}" performance-test-profile Debug 125 2)

# Selection depends only on v2 identity compatibility, never on timing.
set(single_dir "${VSS_PERF_SCRATCH}/single")
file(MAKE_DIRECTORY "${single_dir}")
set(single_baseline "${single_dir}/performance-test-profile.json")
write_record("${single_baseline}" performance-test-profile Debug 100 2)
set(selected "${VSS_PERF_SCRATCH}/selected.txt")
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BASELINE_DIR=${single_dir}"
    "-DVSS_PERF_CURRENT=${current}"
    "-DVSS_PERF_SELECTED_FILE=${selected}"
    -P "${selector}"
  RESULT_VARIABLE selected_status
  OUTPUT_VARIABLE selected_stdout
  ERROR_VARIABLE selected_stderr)
if(NOT selected_status STREQUAL "0" OR NOT EXISTS "${selected}")
  message(FATAL_ERROR
    "selector rejected one compatible baseline: ${selected_stdout}${selected_stderr}")
endif()
file(READ "${selected}" selected_path)
string(STRIP "${selected_path}" selected_path)
if(NOT selected_path STREQUAL single_baseline)
  message(FATAL_ERROR "selector chose the wrong compatible baseline")
endif()
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BASELINE=${selected_path}"
    "-DVSS_PERF_CURRENT=${current}"
    "-DVSS_PERF_RESULT=${VSS_PERF_SCRATCH}/selected-comparison.result"
    -DVSS_PERF_MAX_PERCENT=120
    -P "${checker}"
  RESULT_VARIABLE comparison_status)
file(READ "${VSS_PERF_SCRATCH}/selected-comparison.result" comparison_token)
string(STRIP "${comparison_token}" comparison_token)
if(comparison_status STREQUAL "0" OR
   NOT comparison_token STREQUAL "timing_regression")
  message(FATAL_ERROR
    "selection improperly used timing as a selector: ${comparison_token}")
endif()

# Incompatible profiles are excluded; no compatible result removes stale output.
set(incompatible_dir "${VSS_PERF_SCRATCH}/incompatible")
file(MAKE_DIRECTORY "${incompatible_dir}")
write_record("${incompatible_dir}/performance-test-profile.json"
  performance-test-profile Release 100 2)
write_record("${incompatible_dir}/other-profile.json" other-profile Debug 100 2)
file(WRITE "${selected}" "stale selection\n")
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BASELINE_DIR=${incompatible_dir}"
    "-DVSS_PERF_CURRENT=${current}"
    "-DVSS_PERF_SELECTED_FILE=${selected}"
    -P "${selector}"
  RESULT_VARIABLE incompatible_status)
if(incompatible_status STREQUAL "0" OR EXISTS "${selected}")
  message(FATAL_ERROR "zero compatible matches retained or wrote a selection")
endif()

# A malformed v2 candidate is an error, not an incompatible record to skip.
set(malformed_dir "${VSS_PERF_SCRATCH}/malformed")
file(MAKE_DIRECTORY "${malformed_dir}")
file(WRITE "${malformed_dir}/performance-test-profile.json"
  "{\"schema_version\":2,\"profile_id\":\"performance-test-profile\"}\n")
file(WRITE "${selected}" "stale selection\n")
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BASELINE_DIR=${malformed_dir}"
    "-DVSS_PERF_CURRENT=${current}"
    "-DVSS_PERF_SELECTED_FILE=${selected}"
    -P "${selector}"
  RESULT_VARIABLE malformed_status)
if(malformed_status STREQUAL "0" OR EXISTS "${selected}")
  message(FATAL_ERROR "malformed v2 candidate was skipped or selected")
endif()

# Schema-v1 candidates are excluded rather than interpreted with v2 rules.
set(legacy_dir "${VSS_PERF_SCRATCH}/legacy")
file(MAKE_DIRECTORY "${legacy_dir}")
file(WRITE "${legacy_dir}/historical.json" "{\"schema_version\":1}\n")
file(WRITE "${selected}" "stale selection\n")
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BASELINE_DIR=${legacy_dir}"
    "-DVSS_PERF_CURRENT=${current}"
    "-DVSS_PERF_SELECTED_FILE=${selected}"
    -P "${selector}"
  RESULT_VARIABLE legacy_status)
if(legacy_status STREQUAL "0" OR EXISTS "${selected}")
  message(FATAL_ERROR "historical v1 record was treated as a compatible baseline")
endif()

message(STATUS "VSS-MATH-PERF baseline selector tests passed")
