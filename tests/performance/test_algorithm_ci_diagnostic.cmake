cmake_minimum_required(VERSION 3.28)

foreach(required IN ITEMS VSS_PERF_CMAKE_COMMAND VSS_PERF_STUB
    VSS_PERF_SOURCE_DIR VSS_PERF_SCRATCH)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${VSS_PERF_SCRATCH}")
file(MAKE_DIRECTORY "${VSS_PERF_SCRATCH}/fixtures")
include("${VSS_PERF_SOURCE_DIR}/tests/performance/support/algorithm_fixtures.cmake")
include("${VSS_PERF_SOURCE_DIR}/tests/performance/algorithm_ci_diagnostic.cmake")

set(profile "performance-ci-diagnostic-test")
set(frequency_governor performance)
set(turbo_state enabled)
set(runner_label performance-ci-diagnostic-test)
file(SHA256 "${VSS_PERF_STUB}" stub_sha256)

function(write_record path configuration median allocations)
  vss_algorithm_fixture_record(json "${profile}" "${configuration}"
    "${median}" 64 "${allocations}")
  string(JSON json SET "${json}" environment benchmark_sha256 "\"${stub_sha256}\"")
  string(JSON json SET "${json}" environment frequency_governor "\"${frequency_governor}\"")
  string(JSON json SET "${json}" environment turbo_state "\"${turbo_state}\"")
  string(JSON json SET "${json}" environment runner_label_or_host "\"${runner_label}\"")
  file(WRITE "${path}" "${json}\n")
endfunction()

function(write_unstable_record path)
  write_record("${path}" Debug 100 2)
  file(READ "${path}" stable_text)
  string(JSON unstable_case GET "${stable_text}" cases 0)
  set(samples "[")
  foreach(sample_index RANGE 0 29)
    if(sample_index GREATER 0)
      string(APPEND samples ",")
    endif()
    if(sample_index LESS 15)
      string(APPEND samples "90")
    else()
      string(APPEND samples "110")
    endif()
  endforeach()
  string(APPEND samples "]")
  string(JSON unstable_case SET "${unstable_case}" batch_durations_ns "${samples}")
  string(JSON unstable_case SET "${unstable_case}" median_batch_ns 100)
  string(JSON unstable_case SET "${unstable_case}" mad_batch_ns 10)
  string(JSON unstable_text SET "${stable_text}" cases 0 "${unstable_case}")
  file(WRITE "${path}" "${unstable_text}\n")
endfunction()

function(run_diagnostic name mode second_kind expected_result expected_reason
    expected_status)
  set(case_dir "${VSS_PERF_SCRATCH}/${name}")
  set(fixture_dir "${case_dir}/fixtures")
  set(output_dir "${case_dir}/output")
  file(MAKE_DIRECTORY "${fixture_dir}")
  write_record("${fixture_dir}/first.json" Debug 100 2)
  if(second_kind STREQUAL "stable")
    write_record("${fixture_dir}/second.json" Debug 100 2)
  elseif(second_kind STREQUAL "timing")
    write_record("${fixture_dir}/second.json" Debug 121 2)
  elseif(second_kind STREQUAL "allocation")
    write_record("${fixture_dir}/second.json" Debug 100 3)
  elseif(second_kind STREQUAL "incompatible")
    write_record("${fixture_dir}/second.json" Release 100 2)
  elseif(second_kind STREQUAL "unstable")
    write_unstable_record("${fixture_dir}/second.json")
  else()
    message(FATAL_ERROR "unknown test fixture kind: ${second_kind}")
  endif()

  set(invocation_file "${case_dir}/invocations.txt")
  set(ENV{VSS_TEST_FIXTURE_DIR} "${fixture_dir}")
  set(ENV{VSS_TEST_INVOCATION_FILE} "${invocation_file}")
  set(ENV{VSS_TEST_DIAGNOSTIC_MODE} "${mode}")
  execute_process(
    COMMAND "${VSS_PERF_CMAKE_COMMAND}"
      "-DVSS_PERF_BENCHMARK=${VSS_PERF_STUB}"
      "-DVSS_PERF_OUTPUT_DIR=${output_dir}"
      "-DVSS_PERF_RECORD_PROFILE_ID=${profile}"
      "-DVSS_PERF_FREQUENCY_GOVERNOR=${frequency_governor}"
      "-DVSS_PERF_TURBO_STATE=${turbo_state}"
      "-DVSS_PERF_RUNNER_LABEL=${runner_label}"
      -P "${VSS_PERF_SOURCE_DIR}/tests/performance/run_algorithm_ci_diagnostic.cmake"
    RESULT_VARIABLE status OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
  unset(ENV{VSS_TEST_FIXTURE_DIR})
  unset(ENV{VSS_TEST_INVOCATION_FILE})
  unset(ENV{VSS_TEST_DIAGNOSTIC_MODE})

  set(result_file "${output_dir}/diagnostic.result")
  if(NOT EXISTS "${result_file}")
    message(FATAL_ERROR "${name}: runner did not preserve its result: ${stdout}${stderr}")
  endif()
  file(READ "${result_file}" result_text)
  if(NOT result_text MATCHES "RESULT=${expected_result}([\n]|$)")
    message(FATAL_ERROR "${name}: expected result ${expected_result}: ${result_text}")
  endif()
  if(NOT result_text MATCHES "REASON_CLASS=${expected_reason}([\n]|$)")
    message(FATAL_ERROR "${name}: expected reason ${expected_reason}: ${result_text}")
  endif()
  if(expected_status STREQUAL "PASS" AND NOT status EQUAL 0)
    message(FATAL_ERROR "${name}: expected successful diagnostic: ${stdout}${stderr}")
  elseif(expected_status STREQUAL "FAIL" AND status EQUAL 0)
    message(FATAL_ERROR "${name}: expected failed diagnostic")
  endif()
  file(READ "${invocation_file}" invocation_count)
  string(STRIP "${invocation_count}" invocation_count)
  if(NOT invocation_count STREQUAL "2")
    message(FATAL_ERROR "${name}: expected exactly two captures, got ${invocation_count}")
  endif()
  foreach(artifact IN ITEMS
      "capture-1/${profile}.json.stdout.log"
      "capture-1/${profile}.json.stderr.log"
      "capture-1/wrapper.stdout.log"
      "capture-1/wrapper.stderr.log"
      "capture-2/wrapper.stdout.log"
      "capture-2/wrapper.stderr.log")
    if(NOT EXISTS "${output_dir}/${artifact}")
      message(FATAL_ERROR "${name}: diagnostic artifact is missing ${artifact}")
    endif()
  endforeach()
endfunction()

run_diagnostic(stable stable stable diagnostic_comparison_pass compatible_pass PASS)
run_diagnostic(unstable unstable unstable diagnostic_inconclusive unstable_measurement PASS)
run_diagnostic(timing timing timing diagnostic_inconclusive timing_regression PASS)
run_diagnostic(incompatible incompatible incompatible diagnostic_inconclusive incompatible PASS)
run_diagnostic(allocation allocation allocation diagnostic_failure allocation_regression FAIL)
run_diagnostic(missing missing-second stable diagnostic_failure invalid_record FAIL)

vss_algorithm_ci_diagnostic_classify_capture(unknown_capture_token
  unknown_capture_ok unknown_capture_reason)
if(unknown_capture_ok OR NOT unknown_capture_reason MATCHES "unknown")
  message(FATAL_ERROR "unknown capture tokens must fail closed")
endif()
vss_algorithm_ci_diagnostic_classify_comparison(unknown_comparison_token
  unknown_comparison_result unknown_comparison_ok unknown_comparison_reason)
if(NOT unknown_comparison_result STREQUAL "diagnostic_failure" OR
   unknown_comparison_ok OR NOT unknown_comparison_reason MATCHES "unknown")
  message(FATAL_ERROR "unknown comparison tokens must fail closed")
endif()

message(STATUS "VSS-MATH-PERF CI diagnostic tests passed")
