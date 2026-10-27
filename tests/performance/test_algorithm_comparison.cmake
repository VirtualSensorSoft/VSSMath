cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/support/algorithm_fixtures.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")

set(work "${CMAKE_CURRENT_BINARY_DIR}/algorithm-comparison-contract")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")

function(assert_equal label actual expected)
  if(NOT "${actual}" STREQUAL "${expected}")
    message(FATAL_ERROR
      "VSSMath0 Task 3 ${label}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

function(assert_contains label value needle)
  string(FIND "${value}" "${needle}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "VSSMath0 Task 3 ${label}: '${value}' does not contain '${needle}'")
  endif()
endfunction()

function(write_record path profile median iterations allocations)
  vss_algorithm_fixture_record(json "${profile}" Release "${median}"
    "${iterations}" "${allocations}")
  file(WRITE "${path}" "${json}\n")
endfunction()

function(set_case path case_index median iterations allocations dispersion)
  file(READ "${path}" json)
  string(JSON item GET "${json}" cases "${case_index}")
  string(JSON item SET "${item}" median_batch_ns "${median}")
  string(JSON item SET "${item}" mad_batch_ns "${dispersion}")
  string(JSON item SET "${item}" iterations_per_sample "${iterations}")
  string(JSON item SET "${item}" allocations_per_operation "${allocations}")
  vss_algorithm_decimal_divide_small("${median}" "${iterations}" display
    remainder)
  string(JSON item SET "${item}" median_ns_per_operation "${display}")

  set(samples "[")
  foreach(sample_index RANGE 0 29)
    if(sample_index GREATER 0)
      string(APPEND samples ",")
    endif()
    if(dispersion GREATER 0)
      if(sample_index LESS 15)
        string(APPEND samples "90")
      else()
        string(APPEND samples "110")
      endif()
    else()
      string(APPEND samples "${median}")
    endif()
  endforeach()
  string(APPEND samples "]")
  string(JSON item SET "${item}" batch_durations_ns "${samples}")
  string(JSON json SET "${json}" cases "${case_index}" "${item}")
  file(WRITE "${path}" "${json}\n")
endfunction()

function(assert_regression label baseline current expected_token reason_part)
  vss_algorithm_regression("${baseline}" "${current}" token reason)
  assert_equal("${label} token" "${token}" "${expected_token}")
  if(NOT "${reason_part}" STREQUAL "")
    assert_contains("${label} reason" "${reason}" "${reason_part}")
  endif()
endfunction()

function(assert_repeatability label candidate verification_1 verification_2
    expected_token reason_part)
  vss_algorithm_repeatability("${candidate}" "${verification_1}"
    "${verification_2}" token reason)
  if(NOT "${token}" STREQUAL "${expected_token}")
    message(FATAL_ERROR
      "VSSMath0 Task 3 ${label}: expected '${expected_token}', got '${token}': ${reason}")
  endif()
  if(NOT "${reason_part}" STREQUAL "")
    assert_contains("${label} reason" "${reason}" "${reason_part}")
  endif()
endfunction()

set(baseline "${work}/task3-profile.json")
set(current "${work}/current.json")
write_record("${baseline}" task3-profile 100 4 2)
write_record("${current}" task3-profile 240 8 1)
assert_regression("normalized 120-percent equality" "${baseline}" "${current}"
  compatible_pass "")

write_record("${current}" another-profile 241 8 1)
assert_regression("identity mismatch precedes timing" "${baseline}" "${current}"
  incompatible "")

set(misnamed_baseline "${work}/misnamed-baseline.json")
write_record("${misnamed_baseline}" task3-profile 100 4 2)
write_record("${current}" task3-profile 240 8 1)
assert_regression("baseline profile filename contract" "${misnamed_baseline}"
  "${current}" invalid_record "filename does not match profile_id")

write_record("${baseline}" task3-profile 100 4 2)
write_record("${current}" task3-profile 241 8 1)
assert_regression("just above 120 percent" "${baseline}" "${current}"
  timing_regression
  "case 0 tri.solve.n16.rhs4 current_median=241 current_iterations=8 baseline_median=100 baseline_iterations=4")

write_record("${current}" task3-profile 250 8 1)
assert_regression("125-percent slowdown" "${baseline}" "${current}"
  timing_regression "case 0")

write_record("${baseline}" task3-profile 10 8 2)
write_record("${current}" task3-profile 13 8 1)
assert_regression("truncated display is not the decision value" "${baseline}"
  "${current}" timing_regression "case 0")

write_record("${baseline}" task3-profile 100 4 2)
write_record("${current}" task3-profile 100 8 1)
assert_regression("fewer allocations pass" "${baseline}" "${current}"
  compatible_pass "")
write_record("${current}" task3-profile 100 8 3)
assert_regression("allocation regression precedes timing" "${baseline}"
  "${current}" allocation_regression
  "case 0 tri.solve.n16.rhs4 allocations_per_operation baseline=2 current=3")

set(huge_baseline "${work}/huge-profile.json")
set(huge_current "${work}/huge-current.json")
write_record("${huge_baseline}" huge-profile 9000000000000000000 8 0)
write_record("${huge_current}" huge-profile 10800000000000000000 8 0)
assert_regression("arbitrary precision equality" "${huge_baseline}"
  "${huge_current}" compatible_pass "")

set(legacy "${work}/legacy.json")
file(READ "${huge_current}" legacy_json)
string(JSON legacy_json SET "${legacy_json}" schema_version 1)
file(WRITE "${legacy}" "${legacy_json}\n")
assert_regression("known schema-v1 pair" "${huge_baseline}" "${legacy}"
  incompatible "")

set(legacy_symlink "${work}/legacy-symlink.json")
file(CREATE_LINK "${legacy}" "${legacy_symlink}" SYMBOLIC RESULT symlink_result)
if(symlink_result STREQUAL "0")
  assert_regression("schema-v1 symbolic link is rejected" "${huge_baseline}"
    "${legacy_symlink}" invalid_record "symbolic links are not accepted")
else()
  message(STATUS "Skipping symbolic-link fixture: ${symlink_result}")
endif()

set(malformed "${work}/malformed.json")
file(READ "${huge_current}" malformed_json)
string(JSON malformed_json SET "${malformed_json}" schema_version "\"2\"")
file(WRITE "${malformed}" "${malformed_json}\n")
assert_regression("malformed schema-v2" "${huge_baseline}" "${malformed}"
  invalid_record "")

set(candidate "${work}/candidate.json")
set(verification_1 "${work}/verification-1.json")
set(verification_2 "${work}/verification-2.json")
write_record("${candidate}" task3-profile 100 8 2)
write_record("${verification_1}" task3-profile 90 8 2)
write_record("${verification_2}" task3-profile 108 8 2)
assert_repeatability("exact 120-percent range equality" "${candidate}"
  "${verification_1}" "${verification_2}" repeatable_pass "")

write_record("${verification_1}" task3-profile 80 8 2)
write_record("${verification_2}" task3-profile 96 8 2)
assert_repeatability("wide three-record range" "${candidate}"
  "${verification_1}" "${verification_2}" nonrepeatable_timing "case 0")

write_record("${verification_1}" task3-profile 100 8 2)
write_record("${verification_2}" task3-profile 121 8 2)
set_case("${verification_2}" 1 999 8 2 0)
assert_repeatability("first canonical timing failure" "${candidate}"
  "${verification_1}" "${verification_2}" nonrepeatable_timing "case 0")

write_record("${verification_1}" task3-profile 90 4 2)
write_record("${verification_2}" task3-profile 108 8 2)
assert_repeatability("unequal iteration counts" "${candidate}"
  "${verification_1}" "${verification_2}" incompatible "iteration counts differ")

write_record("${verification_1}" task3-profile 90 8 1)
write_record("${verification_2}" task3-profile 108 8 2)
assert_repeatability("lower allocation is inconsistent" "${candidate}"
  "${verification_1}" "${verification_2}" allocation_inconsistent
  "case 0 tri.solve.n16.rhs4 allocations_per_operation candidate=2 verification_1=1")

write_record("${verification_1}" task3-profile 90 8 3)
assert_repeatability("higher allocation is inconsistent" "${candidate}"
  "${verification_1}" "${verification_2}" allocation_inconsistent "case 0")

# A later allocation mismatch must win over an earlier timing failure.
write_record("${candidate}" task3-profile 100 8 2)
write_record("${verification_1}" task3-profile 80 8 2)
write_record("${verification_2}" task3-profile 108 8 2)
set_case("${verification_1}" 1 100 8 3 0)
assert_repeatability("all allocations precede timing" "${candidate}"
  "${verification_1}" "${verification_2}" allocation_inconsistent
  "case 1 tri.solve.n64.rhs4 allocations_per_operation candidate=2 verification_1=3")

write_record("${verification_1}" task3-profile 90 8 2)
set_case("${candidate}" 0 100 8 2 10)
assert_repeatability("unstable candidate" "${candidate}" "${verification_1}"
  "${verification_2}" unstable_measurement "case 0")
write_record("${candidate}" task3-profile 100 8 2)
set_case("${verification_1}" 0 100 8 2 10)
assert_repeatability("unstable verification" "${candidate}"
  "${verification_1}" "${verification_2}" unstable_measurement "case 0")
write_record("${verification_1}" task3-profile 90 8 2)
set_case("${verification_2}" 0 100 8 2 10)
assert_repeatability("unstable second verification" "${candidate}"
  "${verification_1}" "${verification_2}" unstable_measurement "case 0")

set(quality_result "${work}/quality.result")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_RECORD=${verification_2}"
    "-DVSS_PERF_RESULT=${quality_result}"
    -P "${CMAKE_CURRENT_LIST_DIR}/check_algorithm_quality.cmake"
  RESULT_VARIABLE quality_exit
  OUTPUT_VARIABLE quality_output
  ERROR_VARIABLE quality_error)
if(quality_exit EQUAL 0)
  message(FATAL_ERROR "unstable quality wrapper unexpectedly succeeded")
endif()
file(READ "${quality_result}" quality_token)
assert_equal("quality wrapper result" "${quality_token}" "unstable_measurement\n")
assert_contains("quality wrapper case details"
  "${quality_output}${quality_error}" "case 0")
assert_contains("quality wrapper value details"
  "${quality_output}${quality_error}" "MAD=10")

write_record("${verification_1}" task3-profile 90 8 2)
write_record("${verification_2}" task3-profile 108 8 2)
set(repeatability_result "${work}/repeatability.result")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_CANDIDATE=${candidate}"
    "-DVSS_PERF_VERIFICATION_1=${verification_1}"
    "-DVSS_PERF_VERIFICATION_2=${verification_2}"
    "-DVSS_PERF_RESULT=${repeatability_result}"
    -P "${CMAKE_CURRENT_LIST_DIR}/check_algorithm_repeatability.cmake"
  RESULT_VARIABLE repeatability_exit
  OUTPUT_VARIABLE repeatability_output
  ERROR_VARIABLE repeatability_error)
assert_equal("repeatability wrapper exit" "${repeatability_exit}" "0")
file(READ "${repeatability_result}" repeatability_token)
assert_equal("repeatability wrapper result" "${repeatability_token}"
  "repeatable_pass\n")

write_record("${verification_2}" task3-profile 121 8 2)
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_CANDIDATE=${candidate}"
    "-DVSS_PERF_VERIFICATION_1=${verification_1}"
    "-DVSS_PERF_VERIFICATION_2=${verification_2}"
    "-DVSS_PERF_RESULT=${repeatability_result}"
    -P "${CMAKE_CURRENT_LIST_DIR}/check_algorithm_repeatability.cmake"
  RESULT_VARIABLE repeatability_failure_exit
  OUTPUT_VARIABLE repeatability_failure_output
  ERROR_VARIABLE repeatability_failure_error)
if(repeatability_failure_exit EQUAL 0)
  message(FATAL_ERROR "unstable repeatability wrapper unexpectedly succeeded")
endif()
file(READ "${repeatability_result}" repeatability_failure_token)
assert_equal("repeatability failure wrapper result"
  "${repeatability_failure_token}" "nonrepeatable_timing\n")
assert_contains("repeatability wrapper case detail"
  "${repeatability_failure_output}${repeatability_failure_error}" "case 0")
assert_contains("repeatability wrapper value detail"
  "${repeatability_failure_output}${repeatability_failure_error}"
  "verification_2_median=121")

message(STATUS "VSSMath0 Task 3 algorithm-comparison tests passed")
