cmake_minimum_required(VERSION 3.28)

foreach(required IN ITEMS VSS_PERF_BENCHMARK VSS_PERF_BASELINE
    VSS_PERF_OUTPUT_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VSSMath ${required} is required")
  endif()
endforeach()
if(IS_SYMLINK "${VSS_PERF_OUTPUT_DIR}" OR
   (EXISTS "${VSS_PERF_OUTPUT_DIR}" AND
    NOT IS_DIRECTORY "${VSS_PERF_OUTPUT_DIR}"))
  message(FATAL_ERROR "VSSMath output path must be a real directory")
endif()
file(MAKE_DIRECTORY "${VSS_PERF_OUTPUT_DIR}")
if(DEFINED VSS_PERF_MAX_ATTEMPTS AND
   NOT "${VSS_PERF_MAX_ATTEMPTS}" STREQUAL "1")
  file(WRITE "${VSS_PERF_OUTPUT_DIR}/summary.txt" "invalid_record\n")
  message(FATAL_ERROR "VSSMath only one capture attempt is supported")
endif()

set(summary "${VSS_PERF_OUTPUT_DIR}/summary.txt")
set(current "${VSS_PERF_OUTPUT_DIR}/current.json")
set(capture_result "${VSS_PERF_OUTPUT_DIR}/capture.result")
set(comparison_result "${VSS_PERF_OUTPUT_DIR}/comparison.result")
set(checker "${CMAKE_CURRENT_LIST_DIR}/check_algorithm_regression.cmake")
set(benchmark_launcher "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_benchmark.cmake")
set(digest_validator "${CMAKE_CURRENT_LIST_DIR}/test_executable_digest.cmake")
set(benchmark_args)
foreach(metadata_variable IN ITEMS
    VSS_PERF_FREQUENCY_GOVERNOR VSS_PERF_TURBO_STATE VSS_PERF_RUNNER_LABEL)
  if(DEFINED ${metadata_variable} AND
     NOT "${${metadata_variable}}" STREQUAL "")
    list(APPEND benchmark_args "-D${metadata_variable}=${${metadata_variable}}")
  endif()
endforeach()
if(DEFINED VSS_PERF_BATCH_PLAN AND
   NOT "${VSS_PERF_BATCH_PLAN}" STREQUAL "")
  list(APPEND benchmark_args "-DVSS_PERF_BATCH_PLAN=${VSS_PERF_BATCH_PLAN}")
endif()

# Capture once. The launcher writes a specific result token even when the
# executable fails, and confines its stdout/stderr files to this output dir.
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${VSS_PERF_BENCHMARK}"
    "-DVSS_PERF_OUTPUT=${current}"
    "-DVSS_PERF_CAPTURE_RESULT=${capture_result}"
    ${benchmark_args}
    -P "${benchmark_launcher}"
  RESULT_VARIABLE capture_status
  OUTPUT_VARIABLE capture_stdout
  ERROR_VARIABLE capture_stderr)
if(NOT EXISTS "${capture_result}")
  file(WRITE "${summary}" "invalid_record\n")
  message(FATAL_ERROR
    "VSSMath capture launcher produced no result: ${capture_stdout}${capture_stderr}")
endif()
file(READ "${capture_result}" capture_token)
string(STRIP "${capture_token}" capture_token)
if(NOT capture_token MATCHES
    "^(capture_pass|invalid_record|allocation_inconsistent|qualification_incomplete)$")
  file(WRITE "${summary}" "invalid_record\n")
  message(FATAL_ERROR "VSSMath capture launcher returned unknown token")
endif()
if(NOT capture_status STREQUAL "0" OR NOT capture_token STREQUAL "capture_pass")
  file(WRITE "${summary}" "${capture_token}\n")
  message(FATAL_ERROR
    "VSSMath ${capture_token} during the single capture: ${capture_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${VSS_PERF_BENCHMARK}"
    "-DVSS_PERF_RECORD=${current}"
    -P "${digest_validator}"
  RESULT_VARIABLE digest_status
  OUTPUT_VARIABLE digest_output
  ERROR_VARIABLE digest_error)
if(NOT digest_status STREQUAL "0")
  file(WRITE "${summary}" "invalid_record\n")
  message(FATAL_ERROR
    "VSSMath executable digest validation failed: ${digest_output}${digest_error}")
endif()

# One exact schema-v2 comparison; there is deliberately no timing retry loop.
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BASELINE=${VSS_PERF_BASELINE}"
    "-DVSS_PERF_CURRENT=${current}"
    "-DVSS_PERF_RESULT=${comparison_result}"
    -DVSS_PERF_MAX_PERCENT=120
    -P "${checker}"
  RESULT_VARIABLE comparison_status
  OUTPUT_VARIABLE comparison_output
  ERROR_VARIABLE comparison_error)
if(NOT EXISTS "${comparison_result}")
  file(WRITE "${summary}" "invalid_record\n")
  message(FATAL_ERROR
    "VSSMath checker produced no result: ${comparison_output}${comparison_error}")
endif()
file(READ "${comparison_result}" token)
string(STRIP "${token}" token)
file(WRITE "${summary}" "${token}\n")
if(token STREQUAL "compatible_pass")
  if(NOT comparison_status STREQUAL "0")
    message(FATAL_ERROR
      "VSSMath checker status disagrees with ${token}: ${comparison_error}")
  endif()
  return()
endif()
if(NOT token MATCHES
    "^(incompatible|invalid_record|allocation_regression|timing_regression|unstable_measurement)$")
  file(WRITE "${summary}" "invalid_record\n")
  message(FATAL_ERROR "VSSMath checker returned unknown token ${token}")
endif()
message(FATAL_ERROR "VSSMath ${token}: ${comparison_error}")
