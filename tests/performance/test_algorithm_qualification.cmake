cmake_minimum_required(VERSION 3.28)

# Deterministic schema-v2 contract tests for qualification ordering and failure
# preservation. All captures and host evidence are synthetic.
if(DEFINED VSS_PERF_TEST_SOURCE_DIR AND NOT "${VSS_PERF_TEST_SOURCE_DIR}" STREQUAL "")
  set(source_dir "${VSS_PERF_TEST_SOURCE_DIR}")
else()
  set(source_dir "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/support/algorithm_fixtures.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_batch_plan.cmake")
set(orchestrator "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_qualification.cmake")
set(work "${CMAKE_CURRENT_BINARY_DIR}/performance-algorithm-qualification-fixtures")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")

execute_process(
  COMMAND git -C "${source_dir}" rev-parse HEAD
  OUTPUT_VARIABLE expected_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE commit_status)
if(NOT commit_status EQUAL 0 OR NOT expected_commit MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "could not determine the source commit for qualification fixtures")
endif()
string(SUBSTRING "${expected_commit}" 0 12 short_commit)

set(profile "performance-primary-gcc-debug")
set(configuration "Debug")
set(host_profile "synthetic-linux-host-v1")
set(host_profile_file
  "${CMAKE_CURRENT_LIST_DIR}/fixtures/synthetic-qualification-host.cmake")
set(fake_taskset "${work}/taskset")
file(WRITE "${fake_taskset}"
  "#!/bin/sh\n"
  "if [ \"$1\" = --help ]; then exit 0; fi\n"
  "shift 2\n"
  "exec \"$@\"\n")
file(CHMOD "${fake_taskset}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
set(host_profile_fixture "${work}/synthetic-qualification-host.cmake")
file(READ "${host_profile_file}" host_profile_text)
string(REPLACE "set(VSS_PERF_TASKSET_PATH /usr/bin/taskset)"
  "set(VSS_PERF_TASKSET_PATH ${fake_taskset})"
  host_profile_text "${host_profile_text}")
file(WRITE "${host_profile_fixture}" "${host_profile_text}")
set(host_profile_file "${host_profile_fixture}")
set(benchmark "${work}/benchmark")
file(WRITE "${benchmark}" "synthetic benchmark executable\n")
file(CHMOD "${benchmark}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
file(SHA256 "${benchmark}" benchmark_sha)

function(write_record path record_profile record_configuration record_commit
         record_benchmark_sha record_timestamp median allocations correct)
  vss_algorithm_fixture_record(json "${record_profile}"
    "${record_configuration}" "${median}" 64 "${allocations}")
  string(JSON json SET "${json}" environment math_commit "\"${record_commit}\"")
  string(JSON json SET "${json}" environment benchmark_sha256 "\"${record_benchmark_sha}\"")
  string(JSON json SET "${json}" environment frequency_governor "\"powersave\"")
  string(JSON json SET "${json}" environment runner_label_or_host "\"synthetic-linux-host-v1\"")
  string(JSON json SET "${json}" environment utc_timestamp "\"${record_timestamp}\"")
  string(JSON json SET "${json}" environment compile_flags "\"-O3;-DCHANNEL=a|b\"")
  if(NOT correct)
    string(JSON json SET "${json}" cases 0 correctness_passed false)
  endif()
  file(WRITE "${path}" "${json}\n")
endfunction()

function(write_preflight path result start_utc end_utc)
  file(WRITE "${path}"
    "SCHEMA_VERSION=1\nRESULT=${result}\nHOST_PROFILE=${host_profile}\n"
    "RECORD_PROFILE=${profile}\nCONFIGURATION=${configuration}\n"
    "MATH_COMMIT=${expected_commit}\nBENCHMARK_SHA256=${benchmark_sha}\n"
    "HOST_FINGERPRINT_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    "OS_NAME=Linux\nARCHITECTURE=x86_64\nCPU_MODEL=Synthetic test processor\n"
    "LOGICAL_CPU_COUNT=8\nAFFINITY_CPU=4\nAFFINITY_SIBLINGS=0,4\nAC_ONLINE=1\n"
    "BATTERY_STATUS=Charging\nINTEL_PSTATE_MODE=active\nSCALING_GOVERNOR=powersave\n"
    "INTEL_PSTATE_NO_TURBO=0\nLOAD_1M=0.00\nPACKAGE_TEMP_MILLIC=0\n"
    "MAX_LOAD_1M=0.50\nMAX_PACKAGE_TEMP_MILLIC=55000\n"
    "STABLE_WINDOW_SECONDS=60\nSAMPLE_INTERVAL_SECONDS=10\nMAX_WAIT_SECONDS=900\n"
    "START_UTC=${start_utc}\nEND_UTC=${end_utc}\n")
endfunction()

set(valid_record "${work}/valid.json")
set(valid_verification_1 "${work}/valid-verification-1.json")
set(valid_verification_2 "${work}/valid-verification-2.json")
set(timing_record "${work}/timing-regression.json")
set(allocation_record "${work}/allocation-regression.json")
set(invalid_record "${work}/invalid.json")
set(incompatible_record "${work}/incompatible.json")
set(unstable_candidate "${work}/unstable-candidate.json")
set(unstable_verification_1 "${work}/unstable-verification-1.json")
set(unstable_verification_2 "${work}/unstable-verification-2.json")
set(malformed_record "${work}/malformed.json")
write_record("${valid_record}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:01:01Z" 100 2 TRUE)
write_record("${valid_verification_1}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:02:03Z" 100 2 TRUE)
write_record("${valid_verification_2}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:03:05Z" 100 2 TRUE)
write_record("${timing_record}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:02:03Z" 121 2 TRUE)
write_record("${allocation_record}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:02:03Z" 100 3 TRUE)
write_record("${invalid_record}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:01:01Z" 100 2 FALSE)
write_record("${incompatible_record}" "${profile}" "${configuration}"
  "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:02:03Z" 100 2 TRUE)
file(READ "${incompatible_record}" incompatible_record_text)
string(JSON incompatible_record_text SET "${incompatible_record_text}"
  environment compiler_version "\"13.4.0\"")
file(WRITE "${incompatible_record}" "${incompatible_record_text}\n")
foreach(unstable_path IN ITEMS "${unstable_candidate}" "${unstable_verification_1}"
                              "${unstable_verification_2}")
  file(READ "${valid_record}" stable_text)
  string(JSON unstable_case GET "${stable_text}" cases 0)
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
  string(JSON unstable_case SET "${unstable_case}" batch_durations_ns "${unstable_samples}")
  string(JSON unstable_case SET "${unstable_case}" median_batch_ns 100)
  string(JSON unstable_case SET "${unstable_case}" mad_batch_ns 10)
  string(JSON unstable_record_text SET "${stable_text}" cases 0 "${unstable_case}")
  file(WRITE "${unstable_path}" "${unstable_record_text}\n")
endforeach()
file(WRITE "${malformed_record}" "{not-json\n")

set(preflight_candidate_template "${work}/preflight-candidate.env")
set(preflight_verification_1_template "${work}/preflight-verification-1.env")
set(preflight_verification_2_template "${work}/preflight-verification-2.env")
set(preflight_failed_template "${work}/preflight-failed.env")
write_preflight("${preflight_candidate_template}" preflight_pass
  "2026-10-01T00:00:00Z" "2026-10-01T00:01:00Z")
write_preflight("${preflight_verification_1_template}" preflight_pass
  "2026-10-01T00:01:02Z" "2026-10-01T00:02:02Z")
write_preflight("${preflight_verification_2_template}" preflight_pass
  "2026-10-01T00:02:04Z" "2026-10-01T00:03:04Z")
write_preflight("${preflight_failed_template}" preflight_failed
  "2026-10-01T00:01:02Z" "2026-10-01T00:02:02Z")

set(event_script "${work}/fake_preflight.cmake")
file(WRITE "${event_script}"
  "cmake_minimum_required(VERSION 3.28)\n"
  "get_filename_component(_name \"\${VSS_PERF_PREFLIGHT_OUTPUT}\" NAME)\n"
  "file(APPEND \"\${VSS_PERF_TEST_EVENT_LOG}\" \"preflight:\${_name}\\n\")\n"
  "if(\"\${_name}\" STREQUAL \"preflight-candidate.env\")\n"
  "  set(_template \"\${VSS_PERF_TEST_PREFLIGHT_CANDIDATE_TEMPLATE}\")\n"
  "elseif(\"\${_name}\" STREQUAL \"preflight-verification-1.env\")\n"
  "  set(_template \"\${VSS_PERF_TEST_PREFLIGHT_VERIFICATION_1_TEMPLATE}\")\n"
  "else()\n"
  "  set(_template \"\${VSS_PERF_TEST_PREFLIGHT_VERIFICATION_2_TEMPLATE}\")\n"
  "endif()\n"
  "get_filename_component(_session \"\${VSS_PERF_PREFLIGHT_OUTPUT}\" DIRECTORY)\n"
  "if(\"\${_name}\" STREQUAL \"preflight-verification-1.env\" AND\n"
  "   (NOT EXISTS \"\${_session}/candidate-quality.result\" OR\n"
  "    NOT EXISTS \"\${_session}/batch-plan.env\"))\n"
  "  message(FATAL_ERROR \"candidate quality/plan must precede verification preflight\")\n"
  "endif()\n"
  "if(\"\${_name}\" STREQUAL \"preflight-verification-2.env\" AND\n"
  "   (NOT EXISTS \"\${_session}/verification-1-quality.result\" OR\n"
  "    NOT EXISTS \"\${_session}/batch-plan.env\"))\n"
  "  message(FATAL_ERROR \"verification 1 quality must precede verification 2 preflight\")\n"
  "endif()\n"
  "if(\"\${_name}\" STREQUAL \"preflight-verification-1.env\" AND\n"
  "   DEFINED VSS_PERF_TEST_MUTATE_BATCH_PLAN AND\n"
  "   VSS_PERF_TEST_MUTATE_BATCH_PLAN STREQUAL \"true\")\n"
  "  file(READ \"\${_session}/batch-plan.env\" _plan_contents)\n"
  "  string(REPLACE \"CASE_01_ITERATIONS=64\" \"CASE_01_ITERATIONS=128\" _plan_contents \"\${_plan_contents}\")\n"
  "  file(WRITE \"\${_session}/batch-plan.env\" \"\${_plan_contents}\")\n"
  "endif()\n"
  "if(DEFINED VSS_PERF_TEST_PREFLIGHT_FAILURE_LABEL AND\n"
  "   \"\${_name}\" STREQUAL \"\${VSS_PERF_TEST_PREFLIGHT_FAILURE_LABEL}\")\n"
  "  set(_template \"\${VSS_PERF_TEST_PREFLIGHT_FAILURE_TEMPLATE}\")\n"
  "endif()\n"
  "file(COPY_FILE \"\${_template}\" \"\${VSS_PERF_PREFLIGHT_OUTPUT}\")\n"
  "if(DEFINED VSS_PERF_TEST_PREFLIGHT_FAILURE_LABEL AND\n"
  "   \"\${_name}\" STREQUAL \"\${VSS_PERF_TEST_PREFLIGHT_FAILURE_LABEL}\")\n"
  "  message(FATAL_ERROR \"synthetic preflight failure\")\n"
  "endif()\n")

set(benchmark_script "${work}/fake_benchmark_launcher.cmake")
file(WRITE "${benchmark_script}"
  "cmake_minimum_required(VERSION 3.28)\n"
  "get_filename_component(_name \"\${VSS_PERF_OUTPUT}\" NAME)\n"
  "file(APPEND \"\${VSS_PERF_TEST_EVENT_LOG}\" \"benchmark:\${_name}\\n\")\n"
  "if(\"\${_name}\" MATCHES \"^verification-1\\.json$\")\n"
  "  set(_record \"\${VSS_PERF_TEST_VERIFICATION_1_RECORD}\")\n"
  "  if(NOT DEFINED VSS_PERF_BATCH_PLAN OR NOT EXISTS \"\${VSS_PERF_BATCH_PLAN}\")\n"
  "    message(FATAL_ERROR \"verification 1 must use the candidate-derived batch plan\")\n"
  "  endif()\n"
  "  file(SHA256 \"\${VSS_PERF_BATCH_PLAN}\" _plan_sha)\n"
  "  file(APPEND \"\${VSS_PERF_TEST_EVENT_LOG}\" \"batch-plan:\${_plan_sha}\\n\")\n"
  "elseif(\"\${_name}\" MATCHES \"^verification-2\\.json$\")\n"
  "  set(_record \"\${VSS_PERF_TEST_VERIFICATION_2_RECORD}\")\n"
  "  if(NOT DEFINED VSS_PERF_BATCH_PLAN OR NOT EXISTS \"\${VSS_PERF_BATCH_PLAN}\")\n"
  "    message(FATAL_ERROR \"verification 2 must reuse the candidate-derived batch plan\")\n"
  "  endif()\n"
  "  file(SHA256 \"\${VSS_PERF_BATCH_PLAN}\" _plan_sha)\n"
  "  file(APPEND \"\${VSS_PERF_TEST_EVENT_LOG}\" \"batch-plan:\${_plan_sha}\\n\")\n"
  "else()\n"
  "  set(_record \"\${VSS_PERF_TEST_CANDIDATE_RECORD}\")\n"
  "  get_filename_component(_candidate_dir \"\${VSS_PERF_OUTPUT}\" DIRECTORY)\n"
  "  get_filename_component(_candidate_session \"\${_candidate_dir}\" DIRECTORY)\n"
  "  if(EXISTS \"\${_candidate_session}/batch-plan.env\")\n"
  "    message(FATAL_ERROR \"candidate capture must precede its batch plan\")\n"
  "  endif()\n"
  "endif()\n"
  "if(DEFINED VSS_PERF_TEST_BENCHMARK_FAILURE_LABEL AND\n"
  "   \"\${_name}\" STREQUAL \"\${VSS_PERF_TEST_BENCHMARK_FAILURE_LABEL}\")\n"
  "  if(DEFINED VSS_PERF_TEST_CAPTURE_OUTCOME)\n"
  "    file(WRITE \"\${VSS_PERF_CAPTURE_RESULT}\" \"\${VSS_PERF_TEST_CAPTURE_OUTCOME}\\n\")\n"
  "  endif()\n"
  "  message(FATAL_ERROR \"synthetic benchmark interruption\")\n"
  "endif()\n"
  "if(DEFINED VSS_PERF_TEST_MISSING_RECORD_LABEL AND\n"
  "   \"\${_name}\" STREQUAL \"\${VSS_PERF_TEST_MISSING_RECORD_LABEL}\")\n"
  "  return()\n"
  "endif()\n"
  "file(COPY_FILE \"\${_record}\" \"\${VSS_PERF_OUTPUT}\")\n")

function(run_case name expected_success candidate_file verification_1_file
         verification_2_file expected_failure)
  set(case_dir "${work}/${name}")
  file(MAKE_DIRECTORY "${case_dir}")
  set(event_log "${case_dir}/events.log")
  file(WRITE "${event_log}" "")
  set(session_id "20261001T000000Z-${short_commit}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_SOURCE_DIR=${source_dir}"
      "-DVSS_PERF_OUTPUT_DIR=${case_dir}"
      "-DVSS_PERF_CONFIGURATION=${configuration}"
      "-DVSS_PERF_RECORD_PROFILE_ID=${profile}"
      "-DVSS_PERF_HOST_PROFILE_FILE=${host_profile_file}"
      "-DVSS_PERF_BENCHMARK=${benchmark}"
      "-DVSS_PERF_EXPECTED_COMMIT=${expected_commit}"
      "-DVSS_PERF_SESSION_ID=${session_id}"
      "-DVSS_PERF_PREFLIGHT_SCRIPT=${event_script}"
      "-DVSS_PERF_BENCHMARK_LAUNCHER=${benchmark_script}"
      "-DVSS_PERF_TEST_EVENT_LOG=${event_log}"
      "-DVSS_PERF_TEST_PREFLIGHT_CANDIDATE_TEMPLATE=${preflight_candidate_template}"
      "-DVSS_PERF_TEST_PREFLIGHT_VERIFICATION_1_TEMPLATE=${preflight_verification_1_template}"
      "-DVSS_PERF_TEST_PREFLIGHT_VERIFICATION_2_TEMPLATE=${preflight_verification_2_template}"
      "-DVSS_PERF_TEST_PREFLIGHT_FAILURE_TEMPLATE=${preflight_failed_template}"
      "-DVSS_PERF_TEST_MANIFEST_END_UTC=2026-10-01T00:03:06Z"
      "-DVSS_PERF_TEST_CANDIDATE_RECORD=${candidate_file}"
      "-DVSS_PERF_TEST_VERIFICATION_1_RECORD=${verification_1_file}"
      "-DVSS_PERF_TEST_VERIFICATION_2_RECORD=${verification_2_file}"
      "-DVSS_PERF_TEST_PREFLIGHT_FAILURE_LABEL=${expected_failure}"
      "-DVSS_PERF_TEST_BENCHMARK_FAILURE_LABEL=${expected_failure}"
      "-DVSS_PERF_TEST_MUTATE_BATCH_PLAN=${VSS_PERF_TEST_MUTATE_BATCH_PLAN}"
      "-DVSS_PERF_TEST_CAPTURE_OUTCOME=${VSS_PERF_TEST_CAPTURE_OUTCOME}"
      "-DVSS_PERF_TEST_MISSING_RECORD_LABEL=${VSS_PERF_TEST_MISSING_RECORD_LABEL}"
      "-DVSS_PERF_TEST_FAIL_MANIFEST_PUBLICATION=${VSS_PERF_TEST_FAIL_MANIFEST_PUBLICATION}"
      "-DVSS_PERF_BATCH_PLAN=${VSS_PERF_TEST_CALLER_PLAN}"
      -P "${orchestrator}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(expected_success AND NOT status EQUAL 0)
    message(FATAL_ERROR "${name}: expected success, got ${status}\n${output}${error}")
  endif()
  if(NOT expected_success AND status EQUAL 0)
    message(FATAL_ERROR "${name}: expected fail-closed orchestration")
  endif()
  set(session "${case_dir}/${session_id}")
  if(expected_success)
    if(NOT EXISTS "${session}/qualification.env")
      message(FATAL_ERROR "${name}: final qualification manifest is missing")
    endif()
  elseif(EXISTS "${session}/qualification.env")
    message(FATAL_ERROR "${name}: failed session published a final manifest")
  endif()
  set("${name}_status" "${status}" PARENT_SCOPE)
  set("${name}_output" "${output}${error}" PARENT_SCOPE)
  set("${name}_session" "${session}" PARENT_SCOPE)
  set("${name}_events" "${event_log}" PARENT_SCOPE)
endfunction()

run_case(valid TRUE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}" none)
file(READ "${valid_events}" event_text)
string(STRIP "${event_text}" event_text)
file(SHA256 "${valid_session}/batch-plan.env" expected_plan_sha)
set(expected_events "preflight:preflight-candidate.env\nbenchmark:performance-primary-gcc-debug.json\npreflight:preflight-verification-1.env\nbenchmark:verification-1.json\nbatch-plan:${expected_plan_sha}\npreflight:preflight-verification-2.env\nbenchmark:verification-2.json\nbatch-plan:${expected_plan_sha}")
if(NOT event_text STREQUAL expected_events)
  message(FATAL_ERROR "valid: wrong preflight/benchmark order or count:\n${event_text}")
endif()
file(READ "${valid_session}/qualification.env" valid_manifest)
foreach(required_line IN ITEMS
    "RESULT=qualified_pass"
    "CANDIDATE_PATH=candidate/performance-primary-gcc-debug.json"
    "VERIFICATION_1_PATH=verification-1.json"
    "VERIFICATION_2_PATH=verification-2.json"
    "SCHEMA_VERSION=2"
    "BATCH_PLAN_PATH=batch-plan.env"
    "CANDIDATE_QUALITY_RESULT=stable_pass"
    "VERIFICATION_1_QUALITY_RESULT=stable_pass"
    "VERIFICATION_2_QUALITY_RESULT=stable_pass"
    "REPEATABILITY_RESULT=repeatable_pass"
    "START_UTC=2026-10-01T00:00:00Z"
    "END_UTC=2026-10-01T00:03:06Z")
  string(FIND "${valid_manifest}" "${required_line}\n" required_index)
  if(required_index EQUAL -1)
    message(FATAL_ERROR "valid: manifest is missing ${required_line}")
  endif()
endforeach()
foreach(evidence IN ITEMS
    "candidate/performance-primary-gcc-debug.json"
    "verification-1.json" "verification-2.json"
    "preflight-candidate.env" "preflight-verification-1.env"
    "preflight-verification-2.env" "batch-plan.env"
    "candidate-quality.result" "verification-1-quality.result"
    "verification-2-quality.result" "repeatability.result")
  if(NOT EXISTS "${valid_session}/${evidence}")
    message(FATAL_ERROR "valid: evidence file is missing ${evidence}")
  endif()
endforeach()
if(NOT valid_manifest MATCHES "CANDIDATE_SHA256=[0-9a-f]+")
  message(FATAL_ERROR "valid: candidate hash is missing from the manifest")
endif()
foreach(temporary IN ITEMS
    "qualification.env.tmp" "preflight-candidate.env.tmp"
    "preflight-verification-1.env.tmp" "preflight-verification-2.env.tmp")
  if(EXISTS "${valid_session}/${temporary}")
    message(FATAL_ERROR "valid: temporary evidence file remains ${temporary}")
  endif()
endforeach()

run_case(timing FALSE "${valid_record}" "${timing_record}" "${valid_verification_2}" none)
if(NOT timing_output MATCHES "nonrepeatable_timing")
  message(FATAL_ERROR "timing: symmetric repeatability failure was not retained")
endif()

run_case(allocation FALSE "${valid_record}" "${allocation_record}" "${valid_verification_2}" none)
if(NOT allocation_output MATCHES "allocation_inconsistent")
  message(FATAL_ERROR "allocation: allocation inconsistency was not retained")
endif()

run_case(incompatible FALSE "${valid_record}" "${incompatible_record}" "${valid_verification_2}" none)
if(NOT incompatible_output MATCHES "incompatible")
  message(FATAL_ERROR "incompatible: failure classification was not retained")
endif()

run_case(invalid FALSE "${invalid_record}" "${valid_verification_1}" "${valid_verification_2}" none)
if(NOT invalid_output MATCHES "invalid_record")
  message(FATAL_ERROR "invalid: schema/correctness failure was not retained")
endif()
if(EXISTS "${invalid_session}/batch-plan.env")
  message(FATAL_ERROR "invalid candidate quality must not derive a batch plan")
endif()

run_case(candidate-unstable FALSE "${unstable_candidate}" "${valid_verification_1}" "${valid_verification_2}" none)
if(NOT candidate-unstable_output MATCHES "unstable_measurement" OR
   EXISTS "${candidate-unstable_session}/batch-plan.env")
  message(FATAL_ERROR "unstable candidate must fail before deriving a plan")
endif()
run_case(verification-1-unstable FALSE "${valid_record}" "${unstable_verification_1}" "${valid_verification_2}" none)
if(NOT verification-1-unstable_output MATCHES "unstable_measurement" OR
   verification-1-unstable_output MATCHES "preflight-verification-2")
  message(FATAL_ERROR "unstable verification 1 must stop before verification 2")
endif()
run_case(verification-2-unstable FALSE "${valid_record}" "${valid_verification_1}" "${unstable_verification_2}" none)
if(NOT verification-2-unstable_output MATCHES "unstable_measurement" OR
   NOT verification-2-unstable_output MATCHES "compatibility/quality")
  message(FATAL_ERROR "unstable verification 2 quality failure was not retained")
endif()
run_case(malformed FALSE "${malformed_record}" "${valid_verification_1}" "${valid_verification_2}" none)
if(NOT malformed_output MATCHES "invalid_record" OR
   EXISTS "${malformed_session}/batch-plan.env")
  message(FATAL_ERROR "malformed candidate must be invalid and produce no plan")
endif()

set(VSS_PERF_TEST_MUTATE_BATCH_PLAN true)
run_case(stale-plan FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}" none)
unset(VSS_PERF_TEST_MUTATE_BATCH_PLAN)
if(NOT stale-plan_output MATCHES "incompatible" OR
   stale-plan_output MATCHES "benchmark:verification-1.json")
  message(FATAL_ERROR "a stale plan must be rejected before verification capture")
endif()

vss_algorithm_batch_plan_write("${valid_session}/candidate/${profile}.json"
  "${work}/caller-plan.env"
  caller_plan_token caller_plan_reason)
if(NOT caller_plan_token STREQUAL batch_plan_valid)
  message(FATAL_ERROR "could not build caller-plan rejection fixture: ${caller_plan_reason}")
endif()
set(VSS_PERF_TEST_CALLER_PLAN "${work}/caller-plan.env")
run_case(caller-plan FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}" none)
unset(VSS_PERF_TEST_CALLER_PLAN)
if(NOT caller-plan_output MATCHES "caller-supplied batch plan is forbidden")
  message(FATAL_ERROR "caller-supplied plan override was not rejected")
endif()

run_case(preflight-failure FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}"
  preflight-verification-1.env)
if(NOT preflight-failure_output MATCHES "preflight_failed")
  message(FATAL_ERROR "preflight failure was not retained as preflight_failed")
endif()

run_case(benchmark-interruption FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}"
  verification-1.json)
if(NOT benchmark-interruption_output MATCHES "qualification_incomplete")
  message(FATAL_ERROR "benchmark interruption was not retained as a terminal failure")
endif()

set(VSS_PERF_TEST_CAPTURE_OUTCOME unknown_token)
run_case(unknown-token FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}"
  verification-1.json)
unset(VSS_PERF_TEST_CAPTURE_OUTCOME)
if(NOT unknown-token_output MATCHES "qualification_incomplete")
  message(FATAL_ERROR "unknown capture token was not mapped to qualification_incomplete")
endif()

set(VSS_PERF_TEST_MISSING_RECORD_LABEL verification-1.json)
run_case(missing-result FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}" none)
unset(VSS_PERF_TEST_MISSING_RECORD_LABEL)
if(NOT missing-result_output MATCHES "qualification_incomplete")
  message(FATAL_ERROR "successful launcher with no record was not rejected as incomplete")
endif()

set(VSS_PERF_TEST_FAIL_MANIFEST_PUBLICATION true)
run_case(publication-failure FALSE "${valid_record}" "${valid_verification_1}" "${valid_verification_2}" none)
unset(VSS_PERF_TEST_FAIL_MANIFEST_PUBLICATION)
if(NOT publication-failure_output MATCHES "qualification_incomplete" OR
   NOT EXISTS "${publication-failure_session}/qualification.result")
  message(FATAL_ERROR "failed manifest publication did not preserve its failure token")
endif()

set(collision_case "${work}/valid")
set(collision_session "${collision_case}/20261001T000000Z-${short_commit}")
file(SHA256 "${collision_session}/qualification.env" collision_manifest_before)
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_SOURCE_DIR=${source_dir}"
    "-DVSS_PERF_OUTPUT_DIR=${collision_case}"
    "-DVSS_PERF_CONFIGURATION=${configuration}"
    "-DVSS_PERF_RECORD_PROFILE_ID=${profile}"
    "-DVSS_PERF_HOST_PROFILE_FILE=${host_profile_file}"
    "-DVSS_PERF_BENCHMARK=${benchmark}"
    "-DVSS_PERF_EXPECTED_COMMIT=${expected_commit}"
    "-DVSS_PERF_SESSION_ID=20261001T000000Z-${short_commit}"
    -P "${orchestrator}"
  RESULT_VARIABLE second_collision_status OUTPUT_VARIABLE collision_output
  ERROR_VARIABLE collision_error)
if(second_collision_status EQUAL 0 OR
   NOT "${collision_output}${collision_error}" MATCHES "collision")
  message(FATAL_ERROR "collision was not refused")
endif()
file(SHA256 "${collision_session}/qualification.env"
  collision_manifest_after)
if(NOT collision_manifest_before STREQUAL collision_manifest_after)
  message(FATAL_ERROR "collision changed the existing manifest")
endif()

file(WRITE "${source_dir}/.performance-qualification-dirty-fixture" "dirty\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_SOURCE_DIR=${source_dir}"
    "-DVSS_PERF_OUTPUT_DIR=${work}/dirty"
    "-DVSS_PERF_CONFIGURATION=${configuration}"
    "-DVSS_PERF_RECORD_PROFILE_ID=${profile}"
    "-DVSS_PERF_HOST_PROFILE_FILE=${host_profile_file}"
    "-DVSS_PERF_BENCHMARK=${benchmark}"
    "-DVSS_PERF_EXPECTED_COMMIT=${expected_commit}"
    "-DVSS_PERF_SESSION_ID=20261001T000000Z-${short_commit}"
    "-DVSS_PERF_PREFLIGHT_SCRIPT=${event_script}"
    "-DVSS_PERF_BENCHMARK_LAUNCHER=${benchmark_script}"
    "-DVSS_PERF_TEST_EVENT_LOG=${work}/dirty/events.log"
    "-DVSS_PERF_TEST_PREFLIGHT_TEMPLATE=${preflight_template}"
    "-DVSS_PERF_TEST_PREFLIGHT_FAILURE_TEMPLATE=${preflight_failed_template}"
    "-DVSS_PERF_TEST_CANDIDATE_RECORD=${valid_record}"
    "-DVSS_PERF_TEST_VERIFICATION_1_RECORD=${valid_record}"
    "-DVSS_PERF_TEST_VERIFICATION_2_RECORD=${valid_record}"
    -P "${orchestrator}"
  RESULT_VARIABLE dirty_status OUTPUT_VARIABLE dirty_output ERROR_VARIABLE dirty_error)
file(REMOVE "${source_dir}/.performance-qualification-dirty-fixture")
if(dirty_status EQUAL 0 OR NOT "${dirty_output}${dirty_error}" MATCHES "dirty")
  message(FATAL_ERROR "dirty source tree was not refused")
endif()

message(STATUS "VSS-MATH-PERF schema-v2 qualification orchestrator tests passed")
