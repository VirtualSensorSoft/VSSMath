cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_ci_diagnostic.cmake")

foreach(required IN ITEMS VSS_PERF_BENCHMARK VSS_PERF_OUTPUT_DIR
    VSS_PERF_RECORD_PROFILE_ID VSS_PERF_FREQUENCY_GOVERNOR
    VSS_PERF_TURBO_STATE VSS_PERF_RUNNER_LABEL)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VSS-MATH-PERF CI diagnostic requires ${required}")
  endif()
endforeach()

if(NOT VSS_PERF_RECORD_PROFILE_ID MATCHES "^[a-z0-9][a-z0-9._-]*$")
  message(FATAL_ERROR "VSS-MATH-PERF CI diagnostic profile ID is malformed")
endif()
foreach(metadata_pair
    "frequency governor;${VSS_PERF_FREQUENCY_GOVERNOR}"
    "turbo state;${VSS_PERF_TURBO_STATE}"
    "runner label;${VSS_PERF_RUNNER_LABEL}")
  list(GET metadata_pair 0 metadata_label)
  list(GET metadata_pair 1 metadata_value)
  if(NOT metadata_value MATCHES "^[A-Za-z0-9._-]+$")
    message(FATAL_ERROR "VSS-MATH-PERF CI diagnostic ${metadata_label} is malformed")
  endif()
endforeach()

vss_algorithm_regular_file("${VSS_PERF_BENCHMARK}" benchmark_is_regular
  benchmark_reason)
if(NOT benchmark_is_regular)
  message(FATAL_ERROR "VSS-MATH-PERF CI diagnostic benchmark is invalid: ${benchmark_reason}")
endif()
file(SHA256 "${VSS_PERF_BENCHMARK}" benchmark_sha256)

get_filename_component(output_dir "${VSS_PERF_OUTPUT_DIR}" ABSOLUTE)
if(output_dir MATCHES "/tests/performance/baselines(/|$)" OR
   EXISTS "${output_dir}" OR IS_SYMLINK "${output_dir}")
  message(FATAL_ERROR
    "VSS-MATH-PERF CI diagnostic output must be a new, non-canonical directory")
endif()
file(MAKE_DIRECTORY "${output_dir}")

function(write_diagnostic_result result token detail)
  string(REPLACE "\r" " " safe_detail "${detail}")
  string(REPLACE "\n" " " safe_detail "${safe_detail}")
  file(WRITE "${output_dir}/diagnostic.result"
    "RESULT=${result}\nREASON_CLASS=${token}\nREASON=${safe_detail}\n")
endfunction()

set(record_paths)
set(capture_failure FALSE)
set(capture_failure_token "")
set(capture_failure_reason "")
foreach(capture_index RANGE 1 2)
  set(capture_dir "${output_dir}/capture-${capture_index}")
  file(MAKE_DIRECTORY "${capture_dir}")
  set(record_path "${capture_dir}/${VSS_PERF_RECORD_PROFILE_ID}.json")
  set(capture_result_path "${capture_dir}/capture.result")
  set(wrapper_stdout_path "${capture_dir}/wrapper.stdout.log")
  set(wrapper_stderr_path "${capture_dir}/wrapper.stderr.log")

  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_BENCHMARK=${VSS_PERF_BENCHMARK}"
      "-DVSS_PERF_OUTPUT=${record_path}"
      "-DVSS_PERF_CAPTURE_RESULT=${capture_result_path}"
      "-DVSS_PERF_FREQUENCY_GOVERNOR=${VSS_PERF_FREQUENCY_GOVERNOR}"
      "-DVSS_PERF_TURBO_STATE=${VSS_PERF_TURBO_STATE}"
      "-DVSS_PERF_RUNNER_LABEL=${VSS_PERF_RUNNER_LABEL}"
      -P "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_benchmark.cmake"
    RESULT_VARIABLE wrapper_status
    OUTPUT_VARIABLE wrapper_stdout
    ERROR_VARIABLE wrapper_stderr
    TIMEOUT 2280)
  file(WRITE "${wrapper_stdout_path}" "${wrapper_stdout}")
  file(WRITE "${wrapper_stderr_path}" "${wrapper_stderr}")

  if(EXISTS "${capture_result_path}")
    file(READ "${capture_result_path}" capture_token)
    string(STRIP "${capture_token}" capture_token)
  else()
    set(capture_token invalid_record)
  endif()
  vss_algorithm_ci_diagnostic_classify_capture("${capture_token}"
    capture_ok capture_reason)
  if(NOT wrapper_status STREQUAL "0" OR NOT capture_ok)
    set(capture_failure TRUE)
    set(capture_failure_token "${capture_token}")
    if(capture_reason STREQUAL "")
      set(capture_failure_reason "capture ${capture_index}: ${wrapper_stderr}")
    else()
      set(capture_failure_reason "capture ${capture_index}: ${capture_reason}; ${wrapper_stderr}")
    endif()
    break()
  endif()

  vss_algorithm_record_load("${record_path}" TRUE record_json record_token
    record_reason)
  if(NOT record_token STREQUAL "record_valid")
    set(capture_failure TRUE)
    set(capture_failure_token "${record_token}")
    set(capture_failure_reason "capture ${capture_index}: ${record_reason}")
    break()
  endif()
  string(JSON record_profile GET "${record_json}" profile_id)
  string(JSON environment GET "${record_json}" environment)
  string(JSON record_digest GET "${environment}" benchmark_sha256)
  string(JSON record_governor GET "${environment}" frequency_governor)
  string(JSON record_turbo GET "${environment}" turbo_state)
  string(JSON record_runner GET "${environment}" runner_label_or_host)
  if(NOT record_profile STREQUAL VSS_PERF_RECORD_PROFILE_ID OR
     NOT record_digest STREQUAL benchmark_sha256 OR
     NOT record_governor STREQUAL VSS_PERF_FREQUENCY_GOVERNOR OR
     NOT record_turbo STREQUAL VSS_PERF_TURBO_STATE OR
     NOT record_runner STREQUAL VSS_PERF_RUNNER_LABEL)
    set(capture_failure TRUE)
    set(capture_failure_token invalid_record)
    set(capture_failure_reason
      "capture ${capture_index}: record identity or requested metadata differs")
    break()
  endif()
  list(APPEND record_paths "${record_path}")
endforeach()

if(capture_failure)
  write_diagnostic_result(diagnostic_failure "${capture_failure_token}"
    "${capture_failure_reason}")
  message(FATAL_ERROR
    "VSS-MATH-PERF CI diagnostic capture failed (${capture_failure_token}): ${capture_failure_reason}")
endif()

list(GET record_paths 0 first_record)
list(GET record_paths 1 second_record)
vss_algorithm_regression("${first_record}" "${second_record}"
  comparison_token comparison_reason)
vss_algorithm_ci_diagnostic_classify_comparison("${comparison_token}"
  diagnostic_result comparison_ok classification_reason)
if(comparison_reason STREQUAL "")
  set(diagnostic_reason "${classification_reason}")
else()
  set(diagnostic_reason "${classification_reason}; ${comparison_reason}")
endif()
write_diagnostic_result("${diagnostic_result}" "${comparison_token}"
  "${diagnostic_reason}")
if(NOT comparison_ok)
  message(FATAL_ERROR
    "VSS-MATH-PERF CI diagnostic failed (${comparison_token}): ${comparison_reason}")
endif()
message(STATUS "VSS-MATH-PERF CI diagnostic result: ${diagnostic_result} (${comparison_token})")
