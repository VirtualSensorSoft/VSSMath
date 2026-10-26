cmake_minimum_required(VERSION 3.28)

foreach(required IN ITEMS VSS_PERF_CMAKE_COMMAND VSS_PERF_STUB
    VSS_PERF_SOURCE_DIR VSS_PERF_SCRATCH)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
file(MAKE_DIRECTORY "${VSS_PERF_SCRATCH}")
set(wrapper "${VSS_PERF_SOURCE_DIR}/tests/performance/run_algorithm_benchmark.cmake")
set(regression_runner
  "${VSS_PERF_SOURCE_DIR}/tests/performance/run_algorithm_regression.cmake")
set(plan "${VSS_PERF_SCRATCH}/plan with spaces.env")
file(WRITE "${plan}" "fixture plan\n")

function(run_capture name failure expected_status expected_token)
  set(output_dir "${VSS_PERF_SCRATCH}/${name}")
  file(MAKE_DIRECTORY "${output_dir}")
  set(output "${output_dir}/capture.json")
  set(result "${output_dir}/capture.result")
  if(NOT "${failure}" STREQUAL "")
    set(ENV{VSS_TEST_BENCHMARK_FAILURE} "${failure}")
  else()
    unset(ENV{VSS_TEST_BENCHMARK_FAILURE})
  endif()
  execute_process(
    COMMAND "${VSS_PERF_CMAKE_COMMAND}"
      "-DVSS_PERF_BENCHMARK=${VSS_PERF_STUB}"
      "-DVSS_PERF_OUTPUT=${output}"
      "-DVSS_PERF_BATCH_PLAN=${plan}"
      "-DVSS_PERF_CAPTURE_RESULT=${result}"
      -DVSS_PERF_FREQUENCY_GOVERNOR=performance
      -DVSS_PERF_TURBO_STATE=enabled
      -DVSS_PERF_RUNNER_LABEL=wrapper-test
      -P "${wrapper}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  unset(ENV{VSS_TEST_BENCHMARK_FAILURE})
  if("${expected_status}" STREQUAL "PASS" AND NOT status STREQUAL "0")
    message(FATAL_ERROR "${name}: capture failed: ${stdout}${stderr}")
  elseif("${expected_status}" STREQUAL "FAIL" AND status STREQUAL "0")
    message(FATAL_ERROR "${name}: expected capture failure")
  endif()
  if(NOT EXISTS "${result}")
    message(FATAL_ERROR "${name}: wrapper did not write capture result")
  endif()
  file(READ "${result}" token)
  string(STRIP "${token}" token)
  if(NOT token STREQUAL expected_token)
    message(FATAL_ERROR "${name}: expected ${expected_token}, got ${token}")
  endif()
  if("${expected_token}" STREQUAL "capture_pass")
    if(NOT EXISTS "${output}" OR
       NOT EXISTS "${output}.stdout.log" OR
       NOT EXISTS "${output}.stderr.log")
      message(FATAL_ERROR "${name}: capture record or private logs are missing")
    endif()
    file(READ "${output}.stdout.log" saved_stdout)
    if(NOT saved_stdout MATCHES "stub stdout diagnostic")
      message(FATAL_ERROR "${name}: stdout was not captured in output directory")
    endif()
  elseif(EXISTS "${output}")
    message(FATAL_ERROR "${name}: failed capture left a final JSON record")
  endif()
endfunction()

run_capture(success "" PASS capture_pass)
run_capture(allocation-failure 2 FAIL allocation_inconsistent)

# A raw capture path must not reach a canonical baseline through a directory
# alias; pre-existing reviewed bytes must survive validation failure.
set(alias_case "${VSS_PERF_SCRATCH}/canonical-alias-case")
file(REMOVE_RECURSE "${alias_case}")
set(canonical_fixture_dir
  "${alias_case}/fixture-repository/tests/performance/baselines/v2")
file(MAKE_DIRECTORY "${canonical_fixture_dir}")
set(alias_path "${alias_case}/alias")
file(CREATE_LINK "${canonical_fixture_dir}" "${alias_path}" SYMBOLIC)
set(aliased_record "${canonical_fixture_dir}/performance-wrapper-test-profile.json")
set(reviewed_sentinel "reviewed baseline sentinel\n")
file(WRITE "${aliased_record}" "${reviewed_sentinel}")
set(alias_output "${alias_path}/performance-wrapper-test-profile.json")
set(alias_result "${alias_path}/capture.result")
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${VSS_PERF_STUB}"
    "-DVSS_PERF_OUTPUT=${alias_output}"
    "-DVSS_PERF_CAPTURE_RESULT=${alias_result}"
    "-DVSS_PERF_BATCH_PLAN=${plan}"
    -DVSS_PERF_FREQUENCY_GOVERNOR=performance
    -DVSS_PERF_TURBO_STATE=enabled
    -DVSS_PERF_RUNNER_LABEL=wrapper-test
    -P "${wrapper}"
  RESULT_VARIABLE alias_status OUTPUT_VARIABLE alias_stdout
  ERROR_VARIABLE alias_stderr)
file(READ "${aliased_record}" sentinel_after)
if(alias_status STREQUAL "0" OR NOT sentinel_after STREQUAL reviewed_sentinel)
  message(FATAL_ERROR
    "directory alias was accepted or changed reviewed bytes: ${alias_stdout}${alias_stderr}")
endif()
foreach(alias_artifact IN ITEMS "${alias_output}.stdout.log"
    "${alias_output}.stderr.log" "${alias_result}")
  if(EXISTS "${alias_artifact}")
    message(FATAL_ERROR "alias validation created or retained ${alias_artifact}")
  endif()
endforeach()

# The complete regression runner must reject an incompatible comparison while
# preserving its specific diagnostic token.
include("${VSS_PERF_SOURCE_DIR}/tests/performance/support/algorithm_fixtures.cmake")
set(regression_profile performance-wrapper-test-profile)
set(regression_baseline "${VSS_PERF_SCRATCH}/${regression_profile}.json")
set(regression_current "${VSS_PERF_SCRATCH}/incompatible-current.json")
file(SHA256 "${VSS_PERF_STUB}" stub_sha256)
vss_algorithm_fixture_record(baseline_json "${regression_profile}" Debug 100 64 2)
vss_algorithm_fixture_record(current_json "${regression_profile}" Debug 100 64 2)
foreach(record_variable IN ITEMS baseline_json current_json)
  string(JSON "${record_variable}" SET "${${record_variable}}" environment
    benchmark_sha256 "\"${stub_sha256}\"")
  string(JSON "${record_variable}" SET "${${record_variable}}" environment
    frequency_governor "\"performance\"")
  string(JSON "${record_variable}" SET "${${record_variable}}" environment
    turbo_state "\"enabled\"")
  string(JSON "${record_variable}" SET "${${record_variable}}" environment
    runner_label_or_host "\"wrapper-test\"")
endforeach()
string(JSON current_json SET "${current_json}" environment configuration
  "\"Release\"")
file(WRITE "${regression_baseline}" "${baseline_json}\n")
file(WRITE "${regression_current}" "${current_json}\n")
set(ENV{VSS_TEST_BENCHMARK_RECORD} "${regression_current}")
set(regression_output "${VSS_PERF_SCRATCH}/incompatible-regression")
execute_process(
  COMMAND "${VSS_PERF_CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${VSS_PERF_STUB}"
    "-DVSS_PERF_BASELINE=${regression_baseline}"
    "-DVSS_PERF_OUTPUT_DIR=${regression_output}"
    "-DVSS_PERF_BATCH_PLAN=${plan}"
    -DVSS_PERF_FREQUENCY_GOVERNOR=performance
    -DVSS_PERF_TURBO_STATE=enabled
    -DVSS_PERF_RUNNER_LABEL=wrapper-test
    -P "${regression_runner}"
  RESULT_VARIABLE regression_status
  OUTPUT_VARIABLE regression_stdout
  ERROR_VARIABLE regression_stderr)
unset(ENV{VSS_TEST_BENCHMARK_RECORD})
if(regression_status STREQUAL "0" OR
   NOT EXISTS "${regression_output}/summary.txt")
  message(FATAL_ERROR
    "incompatible regression was not rejected: ${regression_stdout}${regression_stderr}")
endif()
file(READ "${regression_output}/summary.txt" regression_summary)
if(NOT regression_summary STREQUAL "incompatible\n" OR
   NOT "${regression_stdout}${regression_stderr}" MATCHES "VSSMath incompatible:" OR
   "${regression_stdout}${regression_stderr}" MATCHES "checker status disagrees")
  message(FATAL_ERROR
    "incompatible regression diagnostics were not preserved: ${regression_summary}\n${regression_stdout}${regression_stderr}")
endif()

message(STATUS "VSS-MATH-PERF benchmark wrapper tests passed")
