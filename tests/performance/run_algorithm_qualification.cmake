cmake_minimum_required(VERSION 3.28)
include("${CMAKE_CURRENT_LIST_DIR}/qualification_contract.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_session_schema.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_batch_plan.cmake")

set(qualification_keys "${VSS_ALGORITHM_MANIFEST_KEYS}")

foreach(required VSS_PERF_SOURCE_DIR VSS_PERF_OUTPUT_DIR VSS_PERF_CONFIGURATION
                   VSS_PERF_RECORD_PROFILE_ID VSS_PERF_HOST_PROFILE_FILE
                   VSS_PERF_BENCHMARK VSS_PERF_EXPECTED_COMMIT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    if(required STREQUAL "VSS_PERF_HOST_PROFILE_FILE")
      message(FATAL_ERROR
        "VSSMath invalid_record: provide VSS_MATH_BENCHMARK_HOST_PROFILE_FILE; "
        "no host-specific profile is distributed")
    endif()
    message(FATAL_ERROR "VSSMath invalid_record: ${required} is required")
  endif()
endforeach()

if(NOT IS_ABSOLUTE "${VSS_PERF_SOURCE_DIR}" OR
   NOT IS_ABSOLUTE "${VSS_PERF_OUTPUT_DIR}" OR
   NOT IS_ABSOLUTE "${VSS_PERF_HOST_PROFILE_FILE}" OR
   NOT IS_ABSOLUTE "${VSS_PERF_BENCHMARK}")
  message(FATAL_ERROR "VSSMath invalid_record: qualification paths must be absolute")
endif()
if(NOT IS_DIRECTORY "${VSS_PERF_SOURCE_DIR}" OR
   NOT EXISTS "${VSS_PERF_HOST_PROFILE_FILE}" OR
   IS_DIRECTORY "${VSS_PERF_HOST_PROFILE_FILE}" OR
   IS_SYMLINK "${VSS_PERF_HOST_PROFILE_FILE}")
  message(FATAL_ERROR "VSSMath invalid_record: source or host profile is invalid")
endif()
if(EXISTS "${VSS_PERF_OUTPUT_DIR}" AND IS_SYMLINK "${VSS_PERF_OUTPUT_DIR}")
  message(FATAL_ERROR "VSSMath invalid_record: output directory is symlinked")
endif()

vss_contract_require_commit("${VSS_PERF_EXPECTED_COMMIT}" EXPECTED_COMMIT expected_commit_ok expected_commit_reason)
if(NOT expected_commit_ok)
  message(FATAL_ERROR "VSSMath invalid_record: ${expected_commit_reason}")
endif()
if(NOT VSS_PERF_CONFIGURATION MATCHES "^(Debug|Release)$")
  message(FATAL_ERROR "VSSMath invalid_record: configuration is not approved")
endif()
if(VSS_PERF_CONFIGURATION STREQUAL "Debug" AND
   NOT VSS_PERF_RECORD_PROFILE_ID STREQUAL "performance-primary-gcc-debug")
  message(FATAL_ERROR "VSSMath invalid_record: Debug profile mapping is invalid")
endif()
if(VSS_PERF_CONFIGURATION STREQUAL "Release" AND
   NOT VSS_PERF_RECORD_PROFILE_ID STREQUAL "performance-primary-gcc-release")
  message(FATAL_ERROR "VSSMath invalid_record: Release profile mapping is invalid")
endif()
if(NOT VSS_PERF_RECORD_PROFILE_ID MATCHES "^performance-primary-gcc-(debug|release)$")
  message(FATAL_ERROR "VSSMath invalid_record: record profile is not approved")
endif()

execute_process(
  COMMAND git -C "${VSS_PERF_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE actual_commit OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE commit_status)
if(NOT commit_status EQUAL 0 OR NOT actual_commit STREQUAL VSS_PERF_EXPECTED_COMMIT)
  message(FATAL_ERROR "VSSMath invalid_record: source commit differs from expected commit")
endif()
execute_process(
  COMMAND git -C "${VSS_PERF_SOURCE_DIR}" status --porcelain --untracked-files=all
  OUTPUT_VARIABLE status_output OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE status_status)
if(NOT status_status EQUAL 0 OR NOT status_output STREQUAL "")
  message(FATAL_ERROR "VSSMath invalid_record: tracked worktree is dirty")
endif()
execute_process(
  COMMAND git -C "${VSS_PERF_SOURCE_DIR}" diff --cached --quiet --ignore-submodules --
  RESULT_VARIABLE index_status)
if(NOT index_status EQUAL 0)
  message(FATAL_ERROR "VSSMath invalid_record: tracked index is dirty")
endif()

if(NOT EXISTS "${VSS_PERF_BENCHMARK}" OR
   IS_DIRECTORY "${VSS_PERF_BENCHMARK}" OR IS_SYMLINK "${VSS_PERF_BENCHMARK}")
  message(FATAL_ERROR "VSSMath invalid_record: benchmark is missing or symlinked")
endif()
if(DEFINED VSS_PERF_EXPECTED_BENCHMARK_SHA256 AND NOT "${VSS_PERF_EXPECTED_BENCHMARK_SHA256}" STREQUAL "")
  message(FATAL_ERROR "VSSMath invalid_record: caller-supplied benchmark digest is forbidden")
endif()
if(DEFINED VSS_PERF_BATCH_PLAN AND NOT "${VSS_PERF_BATCH_PLAN}" STREQUAL "")
  message(FATAL_ERROR "VSSMath invalid_record: caller-supplied batch plan is forbidden")
endif()
file(SHA256 "${VSS_PERF_BENCHMARK}" actual_benchmark_sha)
set(VSS_PERF_EXPECTED_BENCHMARK_SHA256 "${actual_benchmark_sha}")
include("${VSS_PERF_HOST_PROFILE_FILE}")
if(NOT DEFINED VSS_PERF_HOST_PROFILE_ID OR
   NOT VSS_PERF_HOST_PROFILE_ID MATCHES "^[a-z0-9][a-z0-9._-]*$")
  message(FATAL_ERROR "VSSMath invalid_record: host profile ID is malformed")
endif()
if(NOT DEFINED VSS_PERF_AFFINITY_CPU OR
   NOT VSS_PERF_AFFINITY_CPU MATCHES "^[0-9]+$")
  message(FATAL_ERROR "VSSMath invalid_record: affinity CPU is invalid")
endif()
if(NOT DEFINED VSS_PERF_TASKSET_PATH OR
   NOT EXISTS "${VSS_PERF_TASKSET_PATH}" OR
   IS_DIRECTORY "${VSS_PERF_TASKSET_PATH}" OR
   IS_SYMLINK "${VSS_PERF_TASKSET_PATH}")
  message(FATAL_ERROR "VSSMath invalid_record: taskset tool is unavailable")
endif()
set(taskset "${VSS_PERF_TASKSET_PATH}")
execute_process(COMMAND "${taskset}" --help
  RESULT_VARIABLE taskset_status OUTPUT_QUIET ERROR_QUIET)
if(NOT taskset_status EQUAL 0)
  message(FATAL_ERROR "VSSMath invalid_record: taskset tool is not executable")
endif()

if(NOT EXISTS "${VSS_PERF_OUTPUT_DIR}")
  file(MAKE_DIRECTORY "${VSS_PERF_OUTPUT_DIR}")
endif()
if(NOT IS_DIRECTORY "${VSS_PERF_OUTPUT_DIR}")
  message(FATAL_ERROR "VSSMath invalid_record: output directory is not a directory")
endif()
string(TIMESTAMP session_timestamp "%Y%m%dT%H%M%SZ" UTC)
string(SUBSTRING "${VSS_PERF_EXPECTED_COMMIT}" 0 12 short_commit)
if(DEFINED VSS_PERF_SESSION_ID AND NOT "${VSS_PERF_SESSION_ID}" STREQUAL "")
  set(session_id "${VSS_PERF_SESSION_ID}")
else()
  set(session_id "${session_timestamp}-${short_commit}")
endif()
if(NOT session_id MATCHES "^[0-9]+T[0-9]+Z-[0-9a-f]+$")
  message(FATAL_ERROR "VSSMath invalid_record: session name is invalid")
endif()
set(session "${VSS_PERF_OUTPUT_DIR}/${session_id}")
if(EXISTS "${session}")
  message(FATAL_ERROR "VSSMath qualification session collision: ${session}")
endif()
file(MAKE_DIRECTORY "${session}/candidate")

macro(vss_qualification_fail result reason)
  file(WRITE "${session}/qualification.result" "${result}\n")
  message(FATAL_ERROR "VSSMath ${result}: ${reason}")
endmacro()

set(preflight_script "${CMAKE_CURRENT_LIST_DIR}/check_local_capture_environment.cmake")
if(DEFINED VSS_PERF_PREFLIGHT_SCRIPT AND NOT "${VSS_PERF_PREFLIGHT_SCRIPT}" STREQUAL "")
  set(preflight_script "${VSS_PERF_PREFLIGHT_SCRIPT}")
endif()
set(benchmark_launcher "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_benchmark.cmake")
if(DEFINED VSS_PERF_BENCHMARK_LAUNCHER AND
   NOT "${VSS_PERF_BENCHMARK_LAUNCHER}" STREQUAL "")
  set(benchmark_launcher "${VSS_PERF_BENCHMARK_LAUNCHER}")
endif()
foreach(script IN ITEMS "${preflight_script}" "${benchmark_launcher}")
  if(NOT EXISTS "${script}" OR IS_DIRECTORY "${script}" OR IS_SYMLINK "${script}")
    vss_qualification_fail(invalid_record "qualification helper is unavailable: ${script}")
  endif()
endforeach()

set(test_forward_vars
  VSS_PERF_TEST_EVENT_LOG VSS_PERF_TEST_PREFLIGHT_CANDIDATE_TEMPLATE
  VSS_PERF_TEST_PREFLIGHT_VERIFICATION_1_TEMPLATE
  VSS_PERF_TEST_PREFLIGHT_VERIFICATION_2_TEMPLATE
  VSS_PERF_TEST_PREFLIGHT_FAILURE_TEMPLATE VSS_PERF_TEST_PREFLIGHT_FAILURE_LABEL
  VSS_PERF_TEST_BENCHMARK_FAILURE_LABEL VSS_PERF_TEST_MUTATE_BATCH_PLAN
  VSS_PERF_TEST_CAPTURE_OUTCOME VSS_PERF_TEST_MISSING_RECORD_LABEL
  VSS_PERF_TEST_CANDIDATE_RECORD
  VSS_PERF_TEST_VERIFICATION_1_RECORD VSS_PERF_TEST_VERIFICATION_2_RECORD)

function(run_preflight output_path out_status out_error)
  set(command "${CMAKE_COMMAND}"
    "-DVSS_PERF_HOST_PROFILE_FILE=${VSS_PERF_HOST_PROFILE_FILE}"
    "-DVSS_PERF_PREFLIGHT_OUTPUT=${output_path}"
    "-DVSS_PERF_SOURCE_DIR=${VSS_PERF_SOURCE_DIR}"
    "-DVSS_PERF_EXPECTED_COMMIT=${VSS_PERF_EXPECTED_COMMIT}"
    "-DVSS_PERF_BENCHMARK=${VSS_PERF_BENCHMARK}"
    "-DVSS_PERF_EXPECTED_BENCHMARK_SHA256=${VSS_PERF_EXPECTED_BENCHMARK_SHA256}"
    "-DVSS_PERF_RECORD_PROFILE_ID=${VSS_PERF_RECORD_PROFILE_ID}"
    "-DVSS_PERF_CONFIGURATION=${VSS_PERF_CONFIGURATION}"
    "-DVSS_PERF_HOST_ROOT=/"
    "-DVSS_PERF_TASKSET=${VSS_PERF_TASKSET_PATH}")
  foreach(test_var IN LISTS test_forward_vars)
    if(DEFINED ${test_var})
      list(APPEND command "-D${test_var}=${${test_var}}")
    endif()
  endforeach()
  list(APPEND command -P "${preflight_script}")
  execute_process(COMMAND ${command}
    TIMEOUT 930 RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  set("${out_status}" "${status}" PARENT_SCOPE)
  set("${out_error}" "${output}${error}" PARENT_SCOPE)
endfunction()

function(validate_preflight path label out_ok out_reason)
  set(keys
    SCHEMA_VERSION RESULT HOST_PROFILE RECORD_PROFILE CONFIGURATION MATH_COMMIT
    BENCHMARK_SHA256 HOST_FINGERPRINT_SHA256 OS_NAME ARCHITECTURE CPU_MODEL
    LOGICAL_CPU_COUNT AFFINITY_CPU AFFINITY_SIBLINGS AC_ONLINE BATTERY_STATUS
    INTEL_PSTATE_MODE SCALING_GOVERNOR INTEL_PSTATE_NO_TURBO LOAD_1M
    PACKAGE_TEMP_MILLIC MAX_LOAD_1M MAX_PACKAGE_TEMP_MILLIC
    STABLE_WINDOW_SECONDS SAMPLE_INTERVAL_SECONDS MAX_WAIT_SECONDS START_UTC END_UTC)
  vss_contract_read_kv("${path}" PREFLIGHT "${keys}" ok reason)
  if(NOT ok)
    set("${out_ok}" FALSE PARENT_SCOPE)
    set("${out_reason}" "${label}: ${reason}" PARENT_SCOPE)
    return()
  endif()
  if(NOT PREFLIGHT_SCHEMA_VERSION STREQUAL "1" OR
     NOT PREFLIGHT_RESULT STREQUAL "preflight_pass" OR
     NOT PREFLIGHT_HOST_PROFILE STREQUAL VSS_PERF_HOST_PROFILE_ID OR
     NOT PREFLIGHT_RECORD_PROFILE STREQUAL VSS_PERF_RECORD_PROFILE_ID OR
     NOT PREFLIGHT_CONFIGURATION STREQUAL VSS_PERF_CONFIGURATION OR
     NOT PREFLIGHT_MATH_COMMIT STREQUAL VSS_PERF_EXPECTED_COMMIT OR
     NOT PREFLIGHT_BENCHMARK_SHA256 STREQUAL VSS_PERF_EXPECTED_BENCHMARK_SHA256)
    set("${out_ok}" FALSE PARENT_SCOPE)
    set("${out_reason}" "${label}: identity or result is invalid" PARENT_SCOPE)
    return()
  endif()
  set("${out_ok}" TRUE PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(run_benchmark output_path out_status out_error)
  set(capture_result "${output_path}.capture.result")
  set(command "${taskset}" -c "${VSS_PERF_AFFINITY_CPU}" "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${VSS_PERF_BENCHMARK}"
    "-DVSS_PERF_OUTPUT=${output_path}"
    "-DVSS_PERF_CAPTURE_RESULT=${capture_result}"
    "-DVSS_PERF_FREQUENCY_GOVERNOR=${VSS_PERF_SCALING_GOVERNOR}"
    "-DVSS_PERF_TURBO_STATE=enabled"
    "-DVSS_PERF_RUNNER_LABEL=${VSS_PERF_HOST_PROFILE_ID}")
  if(DEFINED VSS_PERF_BATCH_PLAN AND NOT "${VSS_PERF_BATCH_PLAN}" STREQUAL "")
    list(APPEND command "-DVSS_PERF_BATCH_PLAN=${VSS_PERF_BATCH_PLAN}")
  endif()
  foreach(test_var IN LISTS test_forward_vars)
    if(DEFINED ${test_var})
      list(APPEND command "-D${test_var}=${${test_var}}")
    endif()
  endforeach()
  list(APPEND command -P "${benchmark_launcher}")
  execute_process(COMMAND ${command}
    TIMEOUT 2400 RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  set("${out_status}" "${status}" PARENT_SCOPE)
  set("${out_error}" "${output}${error}" PARENT_SCOPE)
endfunction()

function(validate_record_identity path expected_filename out_token out_reason)
  vss_algorithm_record_load("${path}" FALSE json token reason)
  if(NOT token STREQUAL "record_valid")
    set("${out_token}" "${token}" PARENT_SCOPE)
    set("${out_reason}" "${token}: ${reason}" PARENT_SCOPE)
    return()
  endif()
  string(JSON profile GET "${json}" profile_id)
  string(JSON environment GET "${json}" environment)
  string(JSON math_commit GET "${environment}" math_commit)
  string(JSON configuration GET "${environment}" configuration)
  string(JSON frequency_governor GET "${environment}" frequency_governor)
  string(JSON turbo_state GET "${environment}" turbo_state)
  string(JSON runner_label GET "${environment}" runner_label_or_host)
  get_filename_component(filename "${path}" NAME)
  if(NOT profile STREQUAL VSS_PERF_RECORD_PROFILE_ID OR
     NOT filename STREQUAL expected_filename OR
     NOT math_commit STREQUAL VSS_PERF_EXPECTED_COMMIT OR
     NOT configuration STREQUAL VSS_PERF_CONFIGURATION OR
     NOT frequency_governor STREQUAL VSS_PERF_SCALING_GOVERNOR OR
     NOT turbo_state STREQUAL enabled OR
     NOT runner_label STREQUAL VSS_PERF_HOST_PROFILE_ID)
    set("${out_token}" incompatible PARENT_SCOPE)
    set("${out_reason}" "record identity or Linux metadata is invalid" PARENT_SCOPE)
    return()
  endif()
  set("${out_token}" record_valid PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(validate_executable_digest record out_ok out_reason)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_BENCHMARK=${VSS_PERF_BENCHMARK}"
      "-DVSS_PERF_RECORD=${record}"
      -P "${CMAKE_CURRENT_LIST_DIR}/test_executable_digest.cmake"
    TIMEOUT 120 RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    set("${out_ok}" FALSE PARENT_SCOPE)
    set("${out_reason}" "${output}${error}" PARENT_SCOPE)
    return()
  endif()
  set("${out_ok}" TRUE PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(validate_candidate record out_ok out_reason)
  vss_algorithm_record_load("${record}" TRUE json token reason)
  if(NOT token STREQUAL "record_valid")
    set("${out_ok}" "${token}" PARENT_SCOPE)
    set("${out_reason}" "${reason}" PARENT_SCOPE)
    return()
  endif()
  vss_algorithm_record_quality("${json}" token reason)
  set("${out_ok}" "${token}" PARENT_SCOPE)
  set("${out_reason}" "${reason}" PARENT_SCOPE)
endfunction()

function(validate_verification candidate verification result_path out_token out_error)
  vss_algorithm_record_load("${candidate}" FALSE candidate_json candidate_token candidate_reason)
  vss_algorithm_record_load("${verification}" FALSE verification_json verification_token verification_reason)
  if(NOT candidate_token STREQUAL "record_valid" OR
     NOT verification_token STREQUAL "record_valid")
    set(token invalid_record)
    if(NOT candidate_token STREQUAL "record_valid")
      set(reason "candidate: ${candidate_reason}")
    else()
      set(reason "verification: ${verification_reason}")
    endif()
  else()
    vss_algorithm_record_compatible("${candidate_json}" "${verification_json}"
      TRUE compatibility_token compatibility_reason)
    if(NOT compatibility_token STREQUAL "compatible")
      set(token incompatible)
      set(reason "${compatibility_reason}")
    else()
      vss_algorithm_record_quality("${verification_json}" quality_token quality_reason)
      set(token "${quality_token}")
      set(reason "${quality_reason}")
    endif()
  endif()
  set("${out_token}" "${token}" PARENT_SCOPE)
  set("${out_error}" "${reason}" PARENT_SCOPE)
endfunction()

function(validate_allocation_match candidate verification out_token out_reason)
  vss_algorithm_record_load("${candidate}" FALSE candidate_json candidate_token candidate_reason)
  vss_algorithm_record_load("${verification}" FALSE verification_json verification_token verification_reason)
  if(NOT candidate_token STREQUAL "record_valid" OR
     NOT verification_token STREQUAL "record_valid")
    set("${out_token}" invalid_record PARENT_SCOPE)
    if(NOT candidate_token STREQUAL "record_valid")
      set("${out_reason}" "candidate: ${candidate_reason}" PARENT_SCOPE)
    else()
      set("${out_reason}" "verification: ${verification_reason}" PARENT_SCOPE)
    endif()
    return()
  endif()
  foreach(index RANGE 0 35)
    string(JSON candidate_case GET "${candidate_json}" cases "${index}")
    string(JSON verification_case GET "${verification_json}" cases "${index}")
    string(JSON candidate_allocations GET "${candidate_case}" allocations_per_operation)
    string(JSON verification_allocations GET "${verification_case}" allocations_per_operation)
    if(NOT candidate_allocations STREQUAL verification_allocations)
      string(JSON case_id GET "${candidate_case}" case_id)
      set("${out_token}" allocation_inconsistent PARENT_SCOPE)
      set("${out_reason}" "case ${index} ${case_id} allocations differ" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set("${out_token}" allocation_consistent PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

set(preflight_candidate "${session}/preflight-candidate.env")
set(preflight_1 "${session}/preflight-verification-1.env")
set(preflight_2 "${session}/preflight-verification-2.env")
set(candidate "${session}/candidate/${VSS_PERF_RECORD_PROFILE_ID}.json")
set(verification_1 "${session}/verification-1.json")
set(verification_2 "${session}/verification-2.json")
set(batch_plan "${session}/batch-plan.env")
set(candidate_quality "${session}/candidate-quality.result")
set(verification_1_quality "${session}/verification-1-quality.result")
set(verification_2_quality "${session}/verification-2-quality.result")
set(repeatability_result "${session}/repeatability.result")

run_preflight("${preflight_candidate}" preflight_status preflight_error)
if(NOT preflight_status EQUAL 0 OR NOT EXISTS "${preflight_candidate}")
  vss_qualification_fail(preflight_failed "candidate preflight failed: ${preflight_error}")
endif()
validate_preflight("${preflight_candidate}" candidate-preflight preflight_ok preflight_reason)
if(NOT preflight_ok)
  vss_qualification_fail(preflight_failed "${preflight_reason}")
endif()

run_benchmark("${candidate}" benchmark_status benchmark_error)
if(NOT benchmark_status EQUAL 0 OR NOT EXISTS "${candidate}")
  set(capture_token qualification_incomplete)
  set(capture_result "${candidate}.capture.result")
  if(EXISTS "${capture_result}" AND NOT IS_DIRECTORY "${capture_result}" AND
     NOT IS_SYMLINK "${capture_result}")
    file(READ "${capture_result}" capture_token)
    string(STRIP "${capture_token}" capture_token)
  endif()
  if(NOT capture_token MATCHES "^(invalid_record|allocation_inconsistent|qualification_incomplete|incompatible|unstable_measurement)$")
    set(capture_token qualification_incomplete)
  endif()
  vss_qualification_fail("${capture_token}"
    "candidate benchmark did not produce a record: ${benchmark_error}")
endif()
validate_record_identity("${candidate}" "${VSS_PERF_RECORD_PROFILE_ID}.json"
  record_token record_reason)
if(NOT record_token STREQUAL record_valid)
  vss_qualification_fail("${record_token}" "candidate: ${record_reason}")
endif()
validate_executable_digest("${candidate}" digest_ok digest_reason)
if(NOT digest_ok)
  vss_qualification_fail(invalid_record "candidate digest: ${digest_reason}")
endif()
validate_candidate("${candidate}" candidate_ok candidate_reason)
if(NOT candidate_ok STREQUAL stable_pass)
  vss_qualification_fail("${candidate_ok}" "candidate quality: ${candidate_reason}")
endif()
vss_algorithm_write_result("${candidate_quality}" stable_pass result_ok result_reason)
if(NOT result_ok)
  vss_qualification_fail(invalid_record "candidate quality result: ${result_reason}")
endif()
vss_algorithm_batch_plan_write("${candidate}" "${batch_plan}" plan_token plan_reason)
if(NOT plan_token STREQUAL batch_plan_valid)
  vss_qualification_fail("${plan_token}" "candidate-derived batch plan: ${plan_reason}")
endif()

run_preflight("${preflight_1}" preflight_status preflight_error)
if(NOT preflight_status EQUAL 0 OR NOT EXISTS "${preflight_1}")
  vss_qualification_fail(preflight_failed "verification 1 preflight failed: ${preflight_error}")
endif()
validate_preflight("${preflight_1}" verification-1-preflight preflight_ok preflight_reason)
if(NOT preflight_ok)
  vss_qualification_fail(preflight_failed "${preflight_reason}")
endif()
vss_algorithm_batch_plan_validate("${batch_plan}" "${candidate}" plan_token plan_reason)
if(NOT plan_token STREQUAL batch_plan_valid)
  vss_qualification_fail("${plan_token}" "verification 1 batch plan: ${plan_reason}")
endif()
set(VSS_PERF_BATCH_PLAN "${batch_plan}")
run_benchmark("${verification_1}" benchmark_status benchmark_error)
unset(VSS_PERF_BATCH_PLAN)
if(NOT benchmark_status EQUAL 0 OR NOT EXISTS "${verification_1}")
  set(capture_token qualification_incomplete)
  set(capture_result "${verification_1}.capture.result")
  if(EXISTS "${capture_result}" AND NOT IS_DIRECTORY "${capture_result}" AND
     NOT IS_SYMLINK "${capture_result}")
    file(READ "${capture_result}" capture_token)
    string(STRIP "${capture_token}" capture_token)
  endif()
  if(NOT capture_token MATCHES "^(invalid_record|allocation_inconsistent|qualification_incomplete|incompatible|unstable_measurement)$")
    set(capture_token qualification_incomplete)
  endif()
  vss_qualification_fail("${capture_token}"
    "verification 1 benchmark did not produce a record: ${benchmark_error}")
endif()
validate_record_identity("${verification_1}" verification-1.json record_token record_reason)
if(NOT record_token STREQUAL record_valid)
  vss_qualification_fail("${record_token}" "verification 1: ${record_reason}")
endif()
validate_executable_digest("${verification_1}" digest_ok digest_reason)
if(NOT digest_ok)
  vss_qualification_fail(invalid_record "verification 1 digest: ${digest_reason}")
endif()
validate_verification("${candidate}" "${verification_1}" "${verification_1_quality}"
  verification_1_token verification_error)
if(NOT verification_1_token STREQUAL stable_pass)
  vss_qualification_fail("${verification_1_token}"
    "verification 1 compatibility/quality: ${verification_error}")
endif()
vss_algorithm_write_result("${verification_1_quality}" stable_pass result_ok result_reason)
if(NOT result_ok)
  vss_qualification_fail(invalid_record "verification 1 quality result: ${result_reason}")
endif()
validate_allocation_match("${candidate}" "${verification_1}"
  allocation_token allocation_reason)
if(NOT allocation_token STREQUAL allocation_consistent)
  vss_qualification_fail("${allocation_token}" "verification 1 allocations: ${allocation_reason}")
endif()

run_preflight("${preflight_2}" preflight_status preflight_error)
if(NOT preflight_status EQUAL 0 OR NOT EXISTS "${preflight_2}")
  vss_qualification_fail(preflight_failed "verification 2 preflight failed: ${preflight_error}")
endif()
validate_preflight("${preflight_2}" verification-2-preflight preflight_ok preflight_reason)
if(NOT preflight_ok)
  vss_qualification_fail(preflight_failed "${preflight_reason}")
endif()
vss_algorithm_batch_plan_validate("${batch_plan}" "${candidate}" plan_token plan_reason)
if(NOT plan_token STREQUAL batch_plan_valid)
  vss_qualification_fail("${plan_token}" "verification 2 batch plan: ${plan_reason}")
endif()
set(VSS_PERF_BATCH_PLAN "${batch_plan}")
run_benchmark("${verification_2}" benchmark_status benchmark_error)
unset(VSS_PERF_BATCH_PLAN)
if(NOT benchmark_status EQUAL 0 OR NOT EXISTS "${verification_2}")
  set(capture_token qualification_incomplete)
  set(capture_result "${verification_2}.capture.result")
  if(EXISTS "${capture_result}" AND NOT IS_DIRECTORY "${capture_result}" AND
     NOT IS_SYMLINK "${capture_result}")
    file(READ "${capture_result}" capture_token)
    string(STRIP "${capture_token}" capture_token)
  endif()
  if(NOT capture_token MATCHES "^(invalid_record|allocation_inconsistent|qualification_incomplete|incompatible|unstable_measurement)$")
    set(capture_token qualification_incomplete)
  endif()
  vss_qualification_fail("${capture_token}"
    "verification 2 benchmark did not produce a record: ${benchmark_error}")
endif()
validate_record_identity("${verification_2}" verification-2.json record_token record_reason)
if(NOT record_token STREQUAL record_valid)
  vss_qualification_fail("${record_token}" "verification 2: ${record_reason}")
endif()
validate_executable_digest("${verification_2}" digest_ok digest_reason)
if(NOT digest_ok)
  vss_qualification_fail(invalid_record "verification 2 digest: ${digest_reason}")
endif()
validate_verification("${candidate}" "${verification_2}" "${verification_2_quality}"
  verification_2_token verification_error)
if(NOT verification_2_token STREQUAL stable_pass)
  vss_qualification_fail("${verification_2_token}"
    "verification 2 compatibility/quality: ${verification_error}")
endif()
vss_algorithm_write_result("${verification_2_quality}" stable_pass result_ok result_reason)
if(NOT result_ok)
  vss_qualification_fail(invalid_record "verification 2 quality result: ${result_reason}")
endif()
validate_allocation_match("${candidate}" "${verification_2}"
  allocation_token allocation_reason)
if(NOT allocation_token STREQUAL allocation_consistent)
  vss_qualification_fail("${allocation_token}" "verification 2 allocations: ${allocation_reason}")
endif()
vss_algorithm_repeatability("${candidate}" "${verification_1}" "${verification_2}"
  repeatability_token repeatability_reason)
if(NOT repeatability_token STREQUAL repeatable_pass)
  vss_qualification_fail("${repeatability_token}" "three-capture repeatability: ${repeatability_reason}")
endif()
vss_algorithm_write_result("${repeatability_result}" repeatable_pass result_ok result_reason)
if(NOT result_ok)
  vss_qualification_fail(invalid_record "repeatability result: ${result_reason}")
endif()

vss_contract_read_kv("${preflight_candidate}" CANDIDATE_PREFLIGHT
  "${VSS_ALGORITHM_PREFLIGHT_KEYS}"
  preflight_manifest_ok preflight_manifest_reason)
if(NOT preflight_manifest_ok)
  vss_qualification_fail(invalid_record "candidate preflight manifest: ${preflight_manifest_reason}")
endif()
vss_contract_read_kv("${preflight_1}" VERIFICATION_1_PREFLIGHT
  "${VSS_ALGORITHM_PREFLIGHT_KEYS}" preflight_manifest_ok preflight_manifest_reason)
if(NOT preflight_manifest_ok)
  vss_qualification_fail(invalid_record "verification 1 preflight manifest: ${preflight_manifest_reason}")
endif()
vss_contract_read_kv("${preflight_2}" VERIFICATION_2_PREFLIGHT
  "${VSS_ALGORITHM_PREFLIGHT_KEYS}" preflight_manifest_ok preflight_manifest_reason)
if(NOT preflight_manifest_ok)
  vss_qualification_fail(invalid_record "verification 2 preflight manifest: ${preflight_manifest_reason}")
endif()
set(start_utc "${CANDIDATE_PREFLIGHT_START_UTC}")
if(DEFINED VSS_PERF_TEST_MANIFEST_END_UTC AND NOT "${VSS_PERF_TEST_MANIFEST_END_UTC}" STREQUAL "")
  set(end_utc "${VSS_PERF_TEST_MANIFEST_END_UTC}")
else()
  string(TIMESTAMP end_utc "%Y-%m-%dT%H:%M:%SZ" UTC)
endif()
foreach(timestamp IN ITEMS "${start_utc}" "${end_utc}"
    "${CANDIDATE_PREFLIGHT_START_UTC}" "${CANDIDATE_PREFLIGHT_END_UTC}"
    "${VERIFICATION_1_PREFLIGHT_START_UTC}" "${VERIFICATION_1_PREFLIGHT_END_UTC}"
    "${VERIFICATION_2_PREFLIGHT_START_UTC}" "${VERIFICATION_2_PREFLIGHT_END_UTC}")
  vss_contract_require_utc("${timestamp}" session_timestamp timestamp_ok timestamp_reason)
  if(NOT timestamp_ok)
    vss_qualification_fail(invalid_record "${timestamp_reason}")
  endif()
endforeach()
vss_algorithm_record_load("${candidate}" TRUE candidate_json candidate_token candidate_reason)
vss_algorithm_record_load("${verification_1}" FALSE verification_1_json verification_1_token verification_1_reason)
vss_algorithm_record_load("${verification_2}" FALSE verification_2_json verification_2_token verification_2_reason)
foreach(role IN ITEMS CANDIDATE VERIFICATION_1 VERIFICATION_2)
  if(role STREQUAL "CANDIDATE")
    set(preflight_start "${CANDIDATE_PREFLIGHT_START_UTC}")
    set(preflight_end "${CANDIDATE_PREFLIGHT_END_UTC}")
    set(record_json "${candidate_json}")
    set(label candidate)
  elseif(role STREQUAL "VERIFICATION_1")
    set(preflight_start "${VERIFICATION_1_PREFLIGHT_START_UTC}")
    set(preflight_end "${VERIFICATION_1_PREFLIGHT_END_UTC}")
    set(record_json "${verification_1_json}")
    set(label verification-1)
  else()
    set(preflight_start "${VERIFICATION_2_PREFLIGHT_START_UTC}")
    set(preflight_end "${VERIFICATION_2_PREFLIGHT_END_UTC}")
    set(record_json "${verification_2_json}")
    set(label verification-2)
  endif()
  string(JSON record_environment GET "${record_json}" environment)
  string(JSON record_timestamp GET "${record_environment}" utc_timestamp)
  if(record_timestamp STRLESS preflight_end OR preflight_start STRLESS start_utc OR
     end_utc STRLESS record_timestamp OR end_utc STRLESS preflight_end)
    vss_qualification_fail(invalid_record "${label} UTC role ordering is invalid")
  endif()
endforeach()
if(CANDIDATE_PREFLIGHT_START_UTC STRLESS start_utc OR
   end_utc STRLESS VERIFICATION_2_PREFLIGHT_END_UTC)
  vss_qualification_fail(invalid_record "manifest UTC interval does not contain its evidence")
endif()
string(JSON candidate_environment GET "${candidate_json}" environment)
string(JSON candidate_timestamp GET "${candidate_environment}" utc_timestamp)
string(JSON verification_1_environment GET "${verification_1_json}" environment)
string(JSON verification_1_timestamp GET "${verification_1_environment}" utc_timestamp)
string(JSON verification_2_environment GET "${verification_2_json}" environment)
string(JSON verification_2_timestamp GET "${verification_2_environment}" utc_timestamp)
if(candidate_timestamp STRLESS CANDIDATE_PREFLIGHT_END_UTC OR
   VERIFICATION_1_PREFLIGHT_START_UTC STRLESS candidate_timestamp OR
   verification_1_timestamp STRLESS VERIFICATION_1_PREFLIGHT_END_UTC OR
   VERIFICATION_2_PREFLIGHT_START_UTC STRLESS verification_1_timestamp OR
   verification_2_timestamp STRLESS VERIFICATION_2_PREFLIGHT_END_UTC OR
   end_utc STRLESS verification_2_timestamp)
  vss_qualification_fail(invalid_record "record and preflight role ordering is invalid")
endif()
foreach(path IN ITEMS "${candidate}" "${verification_1}" "${verification_2}"
                      "${preflight_candidate}" "${preflight_1}" "${preflight_2}"
                      "${batch_plan}" "${candidate_quality}"
                      "${verification_1_quality}" "${verification_2_quality}"
                      "${repeatability_result}")
  vss_contract_require_regular("${path}" evidence evidence_ok evidence_reason)
  if(NOT evidence_ok)
    vss_qualification_fail(invalid_record "evidence: ${evidence_reason}")
  endif()
endforeach()

file(SHA256 "${candidate}" candidate_sha)
file(SHA256 "${verification_1}" verification_1_sha)
file(SHA256 "${verification_2}" verification_2_sha)
file(SHA256 "${preflight_candidate}" preflight_candidate_sha)
file(SHA256 "${preflight_1}" preflight_1_sha)
file(SHA256 "${preflight_2}" preflight_2_sha)
file(SHA256 "${batch_plan}" batch_plan_sha)
file(SHA256 "${candidate_quality}" candidate_quality_sha)
file(SHA256 "${verification_1_quality}" verification_1_quality_sha)
file(SHA256 "${verification_2_quality}" verification_2_quality_sha)
file(SHA256 "${repeatability_result}" repeatability_sha)
set(manifest_temp "${session}/qualification.env.tmp")
file(WRITE "${manifest_temp}"
  "SCHEMA_VERSION=2\nRESULT=qualified_pass\nHOST_PROFILE=${VSS_PERF_HOST_PROFILE_ID}\n"
  "RECORD_PROFILE=${VSS_PERF_RECORD_PROFILE_ID}\nCONFIGURATION=${VSS_PERF_CONFIGURATION}\n"
  "MATH_COMMIT=${VSS_PERF_EXPECTED_COMMIT}\nBENCHMARK_SHA256=${VSS_PERF_EXPECTED_BENCHMARK_SHA256}\n"
  "HOST_FINGERPRINT_SHA256=${CANDIDATE_PREFLIGHT_HOST_FINGERPRINT_SHA256}\n"
  "AFFINITY_CPU=${CANDIDATE_PREFLIGHT_AFFINITY_CPU}\n"
  "MAX_LOAD_1M=${CANDIDATE_PREFLIGHT_MAX_LOAD_1M}\n"
  "MAX_PACKAGE_TEMP_MILLIC=${CANDIDATE_PREFLIGHT_MAX_PACKAGE_TEMP_MILLIC}\n"
  "STABLE_WINDOW_SECONDS=${CANDIDATE_PREFLIGHT_STABLE_WINDOW_SECONDS}\n"
  "SAMPLE_INTERVAL_SECONDS=${CANDIDATE_PREFLIGHT_SAMPLE_INTERVAL_SECONDS}\n"
  "MAX_WAIT_SECONDS=${CANDIDATE_PREFLIGHT_MAX_WAIT_SECONDS}\n"
  "START_UTC=${start_utc}\nEND_UTC=${end_utc}\n"
  "CANDIDATE_PATH=candidate/${VSS_PERF_RECORD_PROFILE_ID}.json\n"
  "CANDIDATE_SHA256=${candidate_sha}\n"
  "VERIFICATION_1_PATH=verification-1.json\nVERIFICATION_1_SHA256=${verification_1_sha}\n"
  "VERIFICATION_2_PATH=verification-2.json\nVERIFICATION_2_SHA256=${verification_2_sha}\n"
  "PREFLIGHT_CANDIDATE_PATH=preflight-candidate.env\n"
  "PREFLIGHT_CANDIDATE_SHA256=${preflight_candidate_sha}\n"
  "PREFLIGHT_VERIFICATION_1_PATH=preflight-verification-1.env\n"
  "PREFLIGHT_VERIFICATION_1_SHA256=${preflight_1_sha}\n"
  "PREFLIGHT_VERIFICATION_2_PATH=preflight-verification-2.env\n"
  "PREFLIGHT_VERIFICATION_2_SHA256=${preflight_2_sha}\n"
  "BATCH_PLAN_PATH=batch-plan.env\nBATCH_PLAN_SHA256=${batch_plan_sha}\n"
  "CANDIDATE_QUALITY_PATH=candidate-quality.result\n"
  "CANDIDATE_QUALITY_SHA256=${candidate_quality_sha}\nCANDIDATE_QUALITY_RESULT=stable_pass\n"
  "VERIFICATION_1_QUALITY_PATH=verification-1-quality.result\n"
  "VERIFICATION_1_QUALITY_SHA256=${verification_1_quality_sha}\nVERIFICATION_1_QUALITY_RESULT=stable_pass\n"
  "VERIFICATION_2_QUALITY_PATH=verification-2-quality.result\n"
  "VERIFICATION_2_QUALITY_SHA256=${verification_2_quality_sha}\nVERIFICATION_2_QUALITY_RESULT=stable_pass\n"
  "REPEATABILITY_PATH=repeatability.result\n"
  "REPEATABILITY_SHA256=${repeatability_sha}\nREPEATABILITY_RESULT=repeatable_pass\n")
vss_contract_read_kv("${manifest_temp}" WRITTEN_MANIFEST
  "${VSS_ALGORITHM_MANIFEST_KEYS}" manifest_ok manifest_reason)
if(NOT manifest_ok OR NOT WRITTEN_MANIFEST_SCHEMA_VERSION STREQUAL "2" OR
   NOT WRITTEN_MANIFEST_RESULT STREQUAL "qualified_pass" OR
   NOT WRITTEN_MANIFEST_CANDIDATE_QUALITY_RESULT STREQUAL "stable_pass" OR
   NOT WRITTEN_MANIFEST_VERIFICATION_1_QUALITY_RESULT STREQUAL "stable_pass" OR
   NOT WRITTEN_MANIFEST_VERIFICATION_2_QUALITY_RESULT STREQUAL "stable_pass" OR
   NOT WRITTEN_MANIFEST_REPEATABILITY_RESULT STREQUAL "repeatable_pass")
  file(REMOVE "${manifest_temp}")
  vss_qualification_fail(invalid_record "temporary manifest validation failed: ${manifest_reason}")
endif()
if(DEFINED VSS_PERF_TEST_FAIL_MANIFEST_PUBLICATION AND
   "${VSS_PERF_TEST_FAIL_MANIFEST_PUBLICATION}" STREQUAL "true")
  set(manifest_status "synthetic publication failure requested by test")
else()
  file(RENAME "${manifest_temp}" "${session}/qualification.env"
    NO_REPLACE RESULT manifest_status)
endif()
if(NOT manifest_status STREQUAL "0")
  file(REMOVE "${manifest_temp}")
  vss_qualification_fail(qualification_incomplete
    "could not atomically publish qualification manifest: ${manifest_status}")
endif()
file(REMOVE "${manifest_temp}")
if(EXISTS "${manifest_temp}")
  vss_qualification_fail(qualification_incomplete
    "could not remove temporary qualification manifest")
endif()
file(WRITE "${session}/qualification.result" "qualified_pass\n")
message(STATUS "VSSMath qualified_pass: ${session}")
