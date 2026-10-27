cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_batch_plan.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_session_schema.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/support/algorithm_fixtures.cmake")

if(NOT DEFINED VSS_MATH_BATCH_PLAN_READER OR
   NOT EXISTS "${VSS_MATH_BATCH_PLAN_READER}")
  message(FATAL_ERROR "C++ batch-plan reader executable was not provided")
endif()

set(work "${CMAKE_CURRENT_BINARY_DIR}/algorithm-batch-plan-contract")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}/candidate")

function(assert_equal label expected actual)
  if(NOT "${expected}" STREQUAL "${actual}")
    message(FATAL_ERROR
      "VSSMath0 Task 4 ${label}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

function(assert_plan label path expected_token expected_candidate)
  vss_algorithm_batch_plan_validate("${path}" "${expected_candidate}"
    token reason)
  if(NOT token STREQUAL expected_token)
    message(FATAL_ERROR
      "VSSMath0 Task 4 ${label}: expected ${expected_token}, got ${token}: ${reason}")
  endif()
  set(reader_command "${VSS_MATH_BATCH_PLAN_READER}" --validate "${path}")
  if(ARGC GREATER 4)
    list(APPEND reader_command "${ARGV4}")
  endif()
  execute_process(COMMAND ${reader_command} RESULT_VARIABLE reader_status
    ERROR_VARIABLE reader_error)
  if(expected_token STREQUAL "batch_plan_valid" OR
     "${label}" STREQUAL "changed_count")
    if(NOT reader_status EQUAL 0)
      message(FATAL_ERROR
        "VSSMath0 Task 4 ${label}: C++ reader rejected valid input: ${reader_error}")
    endif()
  elseif(reader_status EQUAL 0)
    message(FATAL_ERROR
      "VSSMath0 Task 4 ${label}: C++ reader accepted malformed input")
  endif()
endfunction()

function(write_variant name content)
  set(path "${work}/${name}.env")
  file(WRITE "${path}" "${content}")
  set("${name}_PATH" "${path}" PARENT_SCOPE)
endfunction()

function(expect_invalid_plan name content expected_token)
  write_variant("${name}" "${content}")
  if(ARGC GREATER 3)
    assert_plan("${name}" "${${name}_PATH}" "${expected_token}"
      "${candidate}" "${ARGV3}")
  else()
    assert_plan("${name}" "${${name}_PATH}" "${expected_token}"
      "${candidate}")
  endif()
endfunction()

vss_algorithm_fixture_record(candidate_json performance-test-profile Debug 100 64 0)
set(candidate "${work}/candidate/performance-test-profile.json")
file(WRITE "${candidate}" "${candidate_json}\n")

set(canonical_fixture
  "${CMAKE_CURRENT_LIST_DIR}/fixtures/valid-batch-plan.env")
file(READ "${canonical_fixture}" canonical)

vss_algorithm_batch_plan_write("${candidate}" "${work}/written.env"
  token reason)
assert_equal("writer success" batch_plan_valid "${token}")
file(READ "${work}/written.env" written)
assert_equal("canonical bytes" "${canonical}" "${written}")
assert_plan("canonical fixture" "${canonical_fixture}" batch_plan_valid
  "${candidate}")

string(JSON semicolon_candidate_json SET "${candidate_json}" environment
  configuration "\"Debug;Release\"")
string(JSON equals_candidate_json SET "${candidate_json}" environment
  configuration "\"Debug=Release\"")
foreach(index IN ITEMS 1 2)
  if(index EQUAL 1)
    set(unsafe_candidate_json "${semicolon_candidate_json}")
  else()
    set(unsafe_candidate_json "${equals_candidate_json}")
  endif()
  set(unsafe_candidate "${work}/candidate/unsafe-${index}.json")
  file(WRITE "${unsafe_candidate}" "${unsafe_candidate_json}\n")
  set(unsafe_plan "${work}/unsafe-${index}.env")
  vss_algorithm_batch_plan_write("${unsafe_candidate}" "${unsafe_plan}"
    unsafe_token unsafe_reason)
  assert_equal("writer rejects unsafe configuration ${index}" invalid_record
    "${unsafe_token}")
  if(EXISTS "${unsafe_plan}" OR EXISTS "${unsafe_plan}.tmp")
    message(FATAL_ERROR
      "VSSMath0 Task 4 unsafe configuration ${index} produced a plan file")
  endif()
endforeach()

string(JSON unstable_case GET "${candidate_json}" cases 0)
set(unstable_samples "[")
foreach(sample_index RANGE 0 29)
  if(sample_index GREATER 0)
    string(APPEND unstable_samples ",")
  endif()
  if(sample_index LESS 15)
    string(APPEND unstable_samples "90")
  else()
    string(APPEND unstable_samples "110")
  endif()
endforeach()
string(APPEND unstable_samples "]")
string(JSON unstable_case SET "${unstable_case}" batch_durations_ns
  "${unstable_samples}")
string(JSON unstable_case SET "${unstable_case}" median_batch_ns 100)
string(JSON unstable_case SET "${unstable_case}" mad_batch_ns 10)
string(JSON unstable_case SET "${unstable_case}" median_ns_per_operation 1)
string(JSON unstable_json SET "${candidate_json}" cases 0 "${unstable_case}")
file(MAKE_DIRECTORY "${work}/unstable")
set(unstable_candidate "${work}/unstable/performance-test-profile.json")
file(WRITE "${unstable_candidate}" "${unstable_json}\n")
vss_algorithm_batch_plan_write("${unstable_candidate}"
  "${work}/must-not-publish.env" token reason)
assert_equal("writer enforces candidate quality" unstable_measurement "${token}")
if(EXISTS "${work}/must-not-publish.env" OR
   EXISTS "${work}/must-not-publish.env.tmp")
  message(FATAL_ERROR
    "VSSMath0 Task 4 unstable candidate produced a plan file")
endif()

# The manifest and preflight key lists are organization-shared repository
# contracts. Keep an independent expected order here so drift fails locally.
set(expected_manifest_keys
  SCHEMA_VERSION RESULT HOST_PROFILE RECORD_PROFILE CONFIGURATION MATH_COMMIT
  BENCHMARK_SHA256 HOST_FINGERPRINT_SHA256 AFFINITY_CPU MAX_LOAD_1M
  MAX_PACKAGE_TEMP_MILLIC STABLE_WINDOW_SECONDS SAMPLE_INTERVAL_SECONDS
  MAX_WAIT_SECONDS START_UTC END_UTC CANDIDATE_PATH CANDIDATE_SHA256
  VERIFICATION_1_PATH VERIFICATION_1_SHA256 VERIFICATION_2_PATH
  VERIFICATION_2_SHA256 PREFLIGHT_CANDIDATE_PATH PREFLIGHT_CANDIDATE_SHA256
  PREFLIGHT_VERIFICATION_1_PATH PREFLIGHT_VERIFICATION_1_SHA256
  PREFLIGHT_VERIFICATION_2_PATH PREFLIGHT_VERIFICATION_2_SHA256
  BATCH_PLAN_PATH BATCH_PLAN_SHA256 CANDIDATE_QUALITY_PATH
  CANDIDATE_QUALITY_SHA256 CANDIDATE_QUALITY_RESULT
  VERIFICATION_1_QUALITY_PATH VERIFICATION_1_QUALITY_SHA256
  VERIFICATION_1_QUALITY_RESULT VERIFICATION_2_QUALITY_PATH
  VERIFICATION_2_QUALITY_SHA256 VERIFICATION_2_QUALITY_RESULT
  REPEATABILITY_PATH REPEATABILITY_SHA256 REPEATABILITY_RESULT)
set(expected_preflight_keys
  SCHEMA_VERSION RESULT HOST_PROFILE RECORD_PROFILE CONFIGURATION MATH_COMMIT
  BENCHMARK_SHA256 HOST_FINGERPRINT_SHA256 OS_NAME ARCHITECTURE CPU_MODEL
  LOGICAL_CPU_COUNT AFFINITY_CPU AFFINITY_SIBLINGS AC_ONLINE BATTERY_STATUS
  INTEL_PSTATE_MODE SCALING_GOVERNOR INTEL_PSTATE_NO_TURBO LOAD_1M
  PACKAGE_TEMP_MILLIC MAX_LOAD_1M MAX_PACKAGE_TEMP_MILLIC STABLE_WINDOW_SECONDS
  SAMPLE_INTERVAL_SECONDS MAX_WAIT_SECONDS START_UTC END_UTC)
assert_equal("manifest keys" "${expected_manifest_keys}"
  "${VSS_ALGORITHM_MANIFEST_KEYS}")
assert_equal("preflight keys" "${expected_preflight_keys}"
  "${VSS_ALGORITHM_PREFLIGHT_KEYS}")
list(LENGTH VSS_ALGORITHM_MANIFEST_KEYS manifest_key_count)
list(LENGTH VSS_ALGORITHM_PREFLIGHT_KEYS preflight_key_count)
assert_equal("manifest key count" 42 "${manifest_key_count}")
assert_equal("preflight key count" 28 "${preflight_key_count}")

string(REPLACE
  "CASE_01_ID=tri.solve.n16.rhs4\nCASE_01_ITERATIONS=64\n"
  "CASE_01_ITERATIONS=64\nCASE_01_ID=tri.solve.n16.rhs4\n"
  reordered "${canonical}")
expect_invalid_plan(reordered "${reordered}" invalid_record)
string(REPLACE "CASE_01_ID=tri.solve.n16.rhs4\n" "" missing "${canonical}")
expect_invalid_plan(missing "${missing}" invalid_record)
string(REPLACE "CASE_02_ID=tri.solve.n64.rhs4\n"
  "CASE_01_ID=tri.solve.n16.rhs4\n" duplicate "${canonical}")
expect_invalid_plan(duplicate "${duplicate}" invalid_record)
string(REPLACE "CASE_01_ID=" "CASE_01_UNKNOWN=" unknown "${canonical}")
expect_invalid_plan(unknown "${unknown}" invalid_record)
string(REPLACE "CASE_01_ID=tri.solve.n16.rhs4"
  "CASE_01_ID=tri.solve.n64.rhs4" wrong_id "${canonical}")
expect_invalid_plan(wrong_id "${wrong_id}" invalid_record)
foreach(count IN ITEMS 0 3 1048577)
  string(REPLACE "CASE_01_ITERATIONS=64"
    "CASE_01_ITERATIONS=${count}" invalid_count "${canonical}")
  expect_invalid_plan("count_${count}" "${invalid_count}" invalid_record)
endforeach()
string(REPLACE "CASE_01_ITERATIONS=64" "CASE_01_ITERATIONS=128"
  changed_count "${canonical}")
expect_invalid_plan(changed_count "${changed_count}" incompatible)

string(REPLACE "RECORD_PROFILE=performance-test-profile"
  "RECORD_PROFILE=performance-other-profile" stale_profile "${canonical}")
expect_invalid_plan(stale_profile "${stale_profile}" incompatible)
string(REPLACE "CONFIGURATION=Debug" "CONFIGURATION=Release"
  stale_configuration "${canonical}")
expect_invalid_plan(stale_configuration "${stale_configuration}" incompatible)
string(REPLACE "CONFIGURATION=Debug" "CONFIGURATION=Debug;Release"
  unsafe_semicolon_configuration "${canonical}")
expect_invalid_plan(unsafe_semicolon_configuration
  "${unsafe_semicolon_configuration}" invalid_record "Debug;Release")
string(REPLACE "CONFIGURATION=Debug" "CONFIGURATION=Debug=Release"
  unsafe_equals_configuration "${canonical}")
expect_invalid_plan(unsafe_equals_configuration
  "${unsafe_equals_configuration}" invalid_record "Debug=Release")
string(REPLACE
  "MATH_COMMIT=0000000000000000000000000000000000000000"
  "MATH_COMMIT=1111111111111111111111111111111111111111"
  stale_commit "${canonical}")
expect_invalid_plan(stale_commit "${stale_commit}" incompatible)
string(REPLACE
  "BENCHMARK_SHA256=0000000000000000000000000000000000000000000000000000000000000000"
  "BENCHMARK_SHA256=1111111111111111111111111111111111111111111111111111111111111111"
  stale_digest "${canonical}")
expect_invalid_plan(stale_digest "${stale_digest}" incompatible)

string(REPLACE "\n" "\r\n" crlf "${canonical}")
expect_invalid_plan(crlf "${crlf}" invalid_record)
string(REGEX REPLACE "\n$" "" no_final_lf "${canonical}")
expect_invalid_plan(no_final_lf "${no_final_lf}" invalid_record)
expect_invalid_plan(comment "# untrusted comment\n${canonical}" invalid_record)
string(REPLACE "CASE_01_ITERATIONS=64"
  "CASE_01_ITERATIONS=$(false)" shell_text "${canonical}")
expect_invalid_plan(shell_text "${shell_text}" invalid_record)
string(REPEAT "x" 8193 oversized_suffix)
expect_invalid_plan(oversized "${canonical}${oversized_suffix}"
  invalid_record)

set(symlink_path "${work}/symlink.env")
file(CREATE_LINK "${canonical_fixture}" "${symlink_path}" SYMBOLIC
  RESULT symlink_result)
if(NOT symlink_result STREQUAL "0")
  message(FATAL_ERROR "VSSMath0 Task 4 could not create symlink fixture: ${symlink_result}")
endif()
assert_plan("symlink" "${symlink_path}" invalid_record "${candidate}")

find_program(mkfifo_program NAMES mkfifo)
if(mkfifo_program)
  set(fifo_path "${work}/fifo.env")
  execute_process(COMMAND "${mkfifo_program}" "${fifo_path}"
    RESULT_VARIABLE fifo_status ERROR_VARIABLE fifo_error)
  if(NOT fifo_status EQUAL 0)
    message(FATAL_ERROR "VSSMath0 Task 4 could not create FIFO fixture: ${fifo_error}")
  endif()
  assert_plan("FIFO" "${fifo_path}" invalid_record "${candidate}")
else()
  message(FATAL_ERROR "VSSMath0 Task 4 requires mkfifo to test special-file rejection")
endif()

file(SHA256 "${canonical_fixture}" canonical_sha_before)
vss_algorithm_batch_plan_write("${candidate}" "${canonical_fixture}"
  collision_token collision_reason)
assert_equal("existing destination" invalid_record "${collision_token}")
file(SHA256 "${canonical_fixture}" canonical_sha_after)
assert_equal("destination preserved" "${canonical_sha_before}"
  "${canonical_sha_after}")

set(temp_collision "${work}/occupied.env")
file(WRITE "${temp_collision}.tmp" "preserve temporary\n")
vss_algorithm_batch_plan_write("${candidate}" "${temp_collision}"
  collision_token collision_reason)
assert_equal("existing temporary" invalid_record "${collision_token}")
file(READ "${temp_collision}.tmp" preserved_temporary)
assert_equal("temporary preserved" "preserve temporary\n"
  "${preserved_temporary}")

message(STATUS "VSSMath0 Task 4 batch-plan contract tests passed")
