cmake_minimum_required(VERSION 3.28)

# RED-phase contract test: a baseline may be promoted only from a complete,
# qualified session. The implementation is intentionally absent in Task 2.
if(DEFINED VSS_PERF_TEST_SOURCE_DIR AND NOT "${VSS_PERF_TEST_SOURCE_DIR}" STREQUAL "")
  set(source_dir "${VSS_PERF_TEST_SOURCE_DIR}")
else()
  set(source_dir "${CMAKE_CURRENT_LIST_DIR}/../..")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/support/algorithm_fixtures.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_batch_plan.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_session_schema.cmake")
set(stager "${CMAKE_CURRENT_LIST_DIR}/stage_algorithm_baseline.cmake")
set(work "${CMAKE_CURRENT_BINARY_DIR}/performance-qualified-staging-fixtures")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")

execute_process(
  COMMAND git -C "${source_dir}" rev-parse HEAD
  OUTPUT_VARIABLE expected_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE commit_status)
if(NOT commit_status EQUAL 0 OR NOT expected_commit MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "could not determine the source commit for staging fixtures")
endif()
set(profile performance-primary-gcc-debug)
set(configuration Debug)

set(case_ids
  tri.solve.n16.rhs4 tri.solve.n64.rhs4 tri.solve.n192.rhs4
  lu.factor.n16 lu.factor.n64 lu.factor.n128
  lu.solve.n16.rhs4 lu.solve.n64.rhs4 lu.solve.n128.rhs4
  qr_ls.factor.m32.n16 qr_ls.factor.m128.n64 qr_ls.factor.m256.n128
  qr_ls.solve.m32.n16.rhs4 qr_ls.solve.m128.n64.rhs4 qr_ls.solve.m256.n128.rhs4
  qr_square.factor.n192 qr_square.solve.n192.rhs4
  cholesky.factor.n16 cholesky.factor.n64 cholesky.factor.n128
  cholesky.solve.n16.rhs4 cholesky.solve.n64.rhs4 cholesky.solve.n128.rhs4
  eigen.decompose.n16 eigen.decompose.n64 eigen.decompose.n128
  covariance.propagate.j64x128.c128 differentiation.jacobian.in64.out128
  statistics.mean.n4096 statistics.population_variance.n4096
  statistics.sample_variance.n4096 statistics.population_stddev.n4096
  statistics.sample_stddev.n4096 statistics.rms.n4096
  probability.survival.n4096 probability.log_survival.n4096)

function(write_record path profile configuration commit benchmark_sha timestamp)
  vss_algorithm_fixture_record(json "${profile}" "${configuration}" 100 64 2)
  string(JSON json SET "${json}" environment math_commit "\"${commit}\"")
  string(JSON json SET "${json}" environment benchmark_sha256 "\"${benchmark_sha}\"")
  string(JSON json SET "${json}" environment compile_flags "\"-O3;-DCHANNEL=a|b\"")
  string(JSON json SET "${json}" environment frequency_governor "\"powersave\"")
  string(JSON json SET "${json}" environment runner_label_or_host "\"performance-primary-local-linux-v1\"")
  string(JSON json SET "${json}" environment utc_timestamp "\"${timestamp}\"")
  string(JSON json SET "${json}" environment os_name "\"Linux\"")
  string(JSON json SET "${json}" environment architecture "\"x86_64\"")
  string(JSON json SET "${json}" environment cpu_model
    "\"Synthetic test processor\"")
  string(JSON json SET "${json}" environment logical_cores 8)
  file(WRITE "${path}" "${json}\n")
endfunction()

function(write_preflight path profile configuration commit benchmark_sha result start_utc end_utc)
  file(WRITE "${path}"
    "SCHEMA_VERSION=1\nRESULT=${result}\nHOST_PROFILE=performance-primary-local-linux-v1\n"
    "RECORD_PROFILE=${profile}\nCONFIGURATION=${configuration}\nMATH_COMMIT=${commit}\n"
    "BENCHMARK_SHA256=${benchmark_sha}\n"
    "HOST_FINGERPRINT_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    "OS_NAME=Linux\nARCHITECTURE=x86_64\n"
    "CPU_MODEL=Synthetic test processor\nLOGICAL_CPU_COUNT=8\n"
    "AFFINITY_CPU=4\nAFFINITY_SIBLINGS=0,4\nAC_ONLINE=1\nBATTERY_STATUS=Charging\n"
    "INTEL_PSTATE_MODE=active\nSCALING_GOVERNOR=powersave\nINTEL_PSTATE_NO_TURBO=0\n"
    "LOAD_1M=0.50\nPACKAGE_TEMP_MILLIC=55000\nMAX_LOAD_1M=0.50\n"
    "MAX_PACKAGE_TEMP_MILLIC=55000\nSTABLE_WINDOW_SECONDS=60\nSAMPLE_INTERVAL_SECONDS=10\n"
    "MAX_WAIT_SECONDS=900\nSTART_UTC=${start_utc}\n"
    "END_UTC=${end_utc}\n")
endfunction()

function(write_manifest session profile configuration commit benchmark_sha)
  set(candidate_rel "candidate/${profile}.json")
  set(candidate "${session}/${candidate_rel}")
  set(verification_1 "${session}/verification-1.json")
  set(verification_2 "${session}/verification-2.json")
  set(preflight_candidate "${session}/preflight-candidate.env")
  set(preflight_1 "${session}/preflight-verification-1.env")
  set(preflight_2 "${session}/preflight-verification-2.env")
  set(batch_plan "${session}/batch-plan.env")
  set(candidate_quality "${session}/candidate-quality.result")
  set(verification_1_quality "${session}/verification-1-quality.result")
  set(verification_2_quality "${session}/verification-2-quality.result")
  set(repeatability "${session}/repeatability.result")
  foreach(path IN ITEMS "${candidate}" "${verification_1}" "${verification_2}"
                        "${preflight_candidate}" "${preflight_1}" "${preflight_2}"
                        "${batch_plan}" "${candidate_quality}"
                        "${verification_1_quality}" "${verification_2_quality}"
                        "${repeatability}")
    if(NOT EXISTS "${path}")
      message(FATAL_ERROR "fixture file is missing: ${path}")
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
  file(SHA256 "${repeatability}" repeatability_sha)
  file(WRITE "${session}/qualification.env"
    "SCHEMA_VERSION=2\nRESULT=qualified_pass\nHOST_PROFILE=performance-primary-local-linux-v1\n"
    "RECORD_PROFILE=${profile}\nCONFIGURATION=${configuration}\nMATH_COMMIT=${commit}\n"
    "BENCHMARK_SHA256=${benchmark_sha}\n"
    "HOST_FINGERPRINT_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    "AFFINITY_CPU=4\nMAX_LOAD_1M=0.50\nMAX_PACKAGE_TEMP_MILLIC=55000\n"
    "STABLE_WINDOW_SECONDS=60\nSAMPLE_INTERVAL_SECONDS=10\nMAX_WAIT_SECONDS=900\n"
    "START_UTC=2026-10-01T00:00:00Z\nEND_UTC=2026-10-01T00:03:06Z\n"
    "CANDIDATE_PATH=${candidate_rel}\nCANDIDATE_SHA256=${candidate_sha}\n"
    "VERIFICATION_1_PATH=verification-1.json\nVERIFICATION_1_SHA256=${verification_1_sha}\n"
    "VERIFICATION_2_PATH=verification-2.json\nVERIFICATION_2_SHA256=${verification_2_sha}\n"
    "PREFLIGHT_CANDIDATE_PATH=preflight-candidate.env\nPREFLIGHT_CANDIDATE_SHA256=${preflight_candidate_sha}\n"
    "PREFLIGHT_VERIFICATION_1_PATH=preflight-verification-1.env\nPREFLIGHT_VERIFICATION_1_SHA256=${preflight_1_sha}\n"
    "PREFLIGHT_VERIFICATION_2_PATH=preflight-verification-2.env\nPREFLIGHT_VERIFICATION_2_SHA256=${preflight_2_sha}\n"
    "BATCH_PLAN_PATH=batch-plan.env\nBATCH_PLAN_SHA256=${batch_plan_sha}\n"
    "CANDIDATE_QUALITY_PATH=candidate-quality.result\n"
    "CANDIDATE_QUALITY_SHA256=${candidate_quality_sha}\nCANDIDATE_QUALITY_RESULT=stable_pass\n"
    "VERIFICATION_1_QUALITY_PATH=verification-1-quality.result\n"
    "VERIFICATION_1_QUALITY_SHA256=${verification_1_quality_sha}\nVERIFICATION_1_QUALITY_RESULT=stable_pass\n"
    "VERIFICATION_2_QUALITY_PATH=verification-2-quality.result\n"
    "VERIFICATION_2_QUALITY_SHA256=${verification_2_quality_sha}\nVERIFICATION_2_QUALITY_RESULT=stable_pass\n"
    "REPEATABILITY_PATH=repeatability.result\n"
    "REPEATABILITY_SHA256=${repeatability_sha}\nREPEATABILITY_RESULT=repeatable_pass\n")
endfunction()

function(refresh_manifest session)
  file(READ "${session}/qualification.env" current_manifest)
  string(REGEX MATCH "RECORD_PROFILE=([^\n]+)" _ "${current_manifest}")
  set(profile "${CMAKE_MATCH_1}")
  string(REGEX MATCH "CONFIGURATION=([^\n]+)" _ "${current_manifest}")
  set(configuration "${CMAKE_MATCH_1}")
  string(REGEX MATCH "MATH_COMMIT=([^\n]+)" _ "${current_manifest}")
  set(commit "${CMAKE_MATCH_1}")
  string(REGEX MATCH "BENCHMARK_SHA256=([^\n]+)" _ "${current_manifest}")
  set(benchmark_sha "${CMAKE_MATCH_1}")
  write_manifest("${session}" "${profile}" "${configuration}" "${commit}"
    "${benchmark_sha}")
endfunction()

function(make_session name)
  set(session "${work}/${name}")
  file(MAKE_DIRECTORY "${session}/candidate")
  if(ARGC GREATER 1)
    set(configuration "${ARGV1}")
  else()
    set(configuration Debug)
  endif()
  if(configuration STREQUAL "Release")
    set(profile performance-primary-gcc-release)
  else()
    set(profile performance-primary-gcc-debug)
  endif()
  set(benchmark "${session}/benchmark")
  file(WRITE "${benchmark}" "synthetic benchmark executable\n")
  file(SHA256 "${benchmark}" benchmark_sha)
  write_record("${session}/candidate/${profile}.json" "${profile}" "${configuration}"
    "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:01:01Z")
  write_record("${session}/verification-1.json" "${profile}" "${configuration}"
    "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:02:03Z")
  write_record("${session}/verification-2.json" "${profile}" "${configuration}"
    "${expected_commit}" "${benchmark_sha}" "2026-10-01T00:03:05Z")
  write_preflight("${session}/preflight-candidate.env" "${profile}" "${configuration}"
    "${expected_commit}" "${benchmark_sha}" preflight_pass
    "2026-10-01T00:00:00Z" "2026-10-01T00:01:00Z")
  write_preflight("${session}/preflight-verification-1.env" "${profile}" "${configuration}"
    "${expected_commit}" "${benchmark_sha}" preflight_pass
    "2026-10-01T00:01:02Z" "2026-10-01T00:02:02Z")
  write_preflight("${session}/preflight-verification-2.env" "${profile}" "${configuration}"
    "${expected_commit}" "${benchmark_sha}" preflight_pass
    "2026-10-01T00:02:04Z" "2026-10-01T00:03:04Z")
  vss_algorithm_batch_plan_write("${session}/candidate/${profile}.json"
    "${session}/batch-plan.env" plan_token plan_reason)
  if(NOT plan_token STREQUAL batch_plan_valid)
    message(FATAL_ERROR "test fixture batch plan failed: ${plan_reason}")
  endif()
  file(WRITE "${session}/candidate-quality.result" "stable_pass\n")
  file(WRITE "${session}/verification-1-quality.result" "stable_pass\n")
  file(WRITE "${session}/verification-2-quality.result" "stable_pass\n")
  file(WRITE "${session}/repeatability.result" "repeatable_pass\n")
  write_manifest("${session}" "${profile}" "${configuration}" "${expected_commit}"
    "${benchmark_sha}")
  set("${name}_session" "${session}" PARENT_SCOPE)
endfunction()

function(rewrite_session_record_field session field value)
  foreach(relative_record IN ITEMS
      "candidate/performance-primary-gcc-debug.json"
      "verification-1.json" "verification-2.json")
    set(record_path "${session}/${relative_record}")
    file(READ "${record_path}" record_json)
    if(field STREQUAL "logical_cores")
      string(JSON record_json SET "${record_json}" environment "${field}" ${value})
    else()
      string(JSON record_json SET "${record_json}" environment "${field}"
        "\"${value}\"")
    endif()
    file(WRITE "${record_path}" "${record_json}\n")
  endforeach()
  file(READ "${session}/qualification.env" manifest)
  string(REGEX MATCH "BENCHMARK_SHA256=([^\n]+)" _ "${manifest}")
  write_manifest("${session}" performance-primary-gcc-debug Debug "${expected_commit}"
    "${CMAKE_MATCH_1}")
endfunction()

function(session_fingerprint session out_fingerprint)
  file(GLOB_RECURSE evidence_files LIST_DIRECTORIES FALSE "${session}/*")
  list(SORT evidence_files)
  set(snapshot "")
  foreach(evidence IN LISTS evidence_files)
    file(RELATIVE_PATH evidence_relative "${session}" "${evidence}")
    if(IS_SYMLINK "${evidence}")
      set(evidence_fingerprint symlink)
    else()
      execute_process(COMMAND stat --format=%F -- "${evidence}"
        RESULT_VARIABLE stat_status OUTPUT_VARIABLE file_type ERROR_QUIET)
      string(STRIP "${file_type}" file_type)
      string(TOLOWER "${file_type}" file_type)
      if(stat_status EQUAL 0 AND file_type MATCHES "^regular( empty)? file$")
        file(SHA256 "${evidence}" evidence_fingerprint)
      else()
        set(evidence_fingerprint "special:${file_type}")
      endif()
    endif()
    string(APPEND snapshot "${evidence_relative}=${evidence_fingerprint}\n")
  endforeach()
  string(SHA256 fingerprint "${snapshot}")
  set("${out_fingerprint}" "${fingerprint}" PARENT_SCOPE)
endfunction()

function(run_stage name session expected_status)
  if(ARGC GREATER 3)
    set(source_repo "${ARGV3}")
  else()
    set(source_repo "${work}/${name}-source")
    execute_process(COMMAND git clone --quiet --shared --no-checkout "${source_dir}" "${source_repo}"
      RESULT_VARIABLE clone_status OUTPUT_QUIET ERROR_VARIABLE clone_error)
    if(NOT clone_status EQUAL 0)
      message(FATAL_ERROR "${name}: source fixture clone failed: ${clone_error}")
    endif()
    execute_process(COMMAND git -C "${source_repo}" checkout --quiet "${expected_commit}"
      RESULT_VARIABLE checkout_status OUTPUT_QUIET ERROR_VARIABLE checkout_error)
    if(NOT checkout_status EQUAL 0)
      message(FATAL_ERROR "${name}: source fixture checkout failed: ${checkout_error}")
    endif()
  endif()
  if(EXISTS "${session}/qualification.env")
    file(READ "${session}/qualification.env" manifest)
    string(REGEX MATCH "RECORD_PROFILE=([^\n]+)" _ "${manifest}")
    set(profile "${CMAKE_MATCH_1}")
  else()
    set(profile performance-primary-gcc-debug)
  endif()
  set(destination "${source_repo}/tests/performance/baselines/v2")
  set(destination_record "${destination}/${profile}.json")
  set(temporary_record "${destination}/.${profile}.json.tmp")
  if(name STREQUAL "existing")
    file(MAKE_DIRECTORY "${destination}")
    file(WRITE "${destination_record}" "sentinel\n")
  elseif(name STREQUAL "existing-temp")
    file(MAKE_DIRECTORY "${destination}")
    file(WRITE "${temporary_record}" "temporary sentinel\n")
  elseif(name STREQUAL "destination-symlink")
    file(MAKE_DIRECTORY "${source_repo}/tests/performance/baselines")
    file(MAKE_DIRECTORY "${work}/outside-baseline-destination")
    file(CREATE_LINK "${work}/outside-baseline-destination" "${destination}" SYMBOLIC)
  endif()
  set(destination_existed FALSE)
  if(EXISTS "${destination_record}")
    set(destination_existed TRUE)
    file(SHA256 "${destination_record}" destination_before_sha)
  endif()
  set(temporary_existed FALSE)
  if(EXISTS "${temporary_record}")
    set(temporary_existed TRUE)
    file(SHA256 "${temporary_record}" temporary_before_sha)
  endif()
  session_fingerprint("${session}" session_before)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_SESSION_DIR=${session}"
      "-DVSS_PERF_BASELINE_DIR=${destination}"
      "-DVSS_PERF_SOURCE_DIR=${source_repo}"
      -P "${stager}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if("${expected_status}" STREQUAL "PASS" AND NOT status EQUAL 0)
    message(FATAL_ERROR "${name}: expected success, got ${status}\n${output}${error}")
  endif()
  if("${expected_status}" STREQUAL "FAIL" AND status EQUAL 0)
    message(FATAL_ERROR "${name}: expected fail-closed staging")
  endif()
  session_fingerprint("${session}" session_after)
  if(NOT session_before STREQUAL session_after)
    message(FATAL_ERROR "${name}: staging mutated qualification evidence")
  endif()
  if("${expected_status}" STREQUAL "PASS" AND
     NOT EXISTS "${destination}/${profile}.json")
    message(FATAL_ERROR "${name}: qualified candidate was not staged")
  endif()
  if("${expected_status}" STREQUAL "PASS")
    file(SHA256 "${destination}/${profile}.json" staged_sha)
    file(SHA256 "${session}/candidate/${profile}.json" candidate_sha)
    if(NOT staged_sha STREQUAL candidate_sha)
      message(FATAL_ERROR "${name}: staged baseline bytes differ from candidate")
    endif()
  endif()
  if("${expected_status}" STREQUAL "FAIL" AND destination_existed)
    file(SHA256 "${destination_record}" destination_after_sha)
    if(NOT destination_before_sha STREQUAL destination_after_sha)
      message(FATAL_ERROR "${name}: failed staging changed the destination")
    endif()
  endif()
  if(temporary_existed)
    file(SHA256 "${temporary_record}" temporary_after_sha)
    if(NOT temporary_before_sha STREQUAL temporary_after_sha)
      message(FATAL_ERROR "${name}: failed staging changed its pre-existing temporary")
    endif()
  endif()
endfunction()

make_session(valid)
run_stage(valid "${valid_session}" PASS)

make_session(sequential-debug Debug)
run_stage(sequential-debug "${sequential-debug_session}" PASS)
set(sequential_source "${work}/sequential-debug-source")
make_session(sequential-release Release)
run_stage(sequential-release "${sequential-release_session}" PASS "${sequential_source}")
foreach(staged_profile IN ITEMS performance-primary-gcc-debug performance-primary-gcc-release)
  if(NOT EXISTS "${sequential_source}/tests/performance/baselines/v2/${staged_profile}.json")
    message(FATAL_ERROR "sequential Debug/Release staging lost ${staged_profile}")
  endif()
endforeach()

make_session(hardware-mismatch-model)
rewrite_session_record_field("${hardware-mismatch-model_session}" cpu_model
  "different-cpu")
run_stage(hardware-mismatch-model "${hardware-mismatch-model_session}" FAIL)

make_session(hardware-mismatch-architecture)
rewrite_session_record_field("${hardware-mismatch-architecture_session}" architecture
  "aarch64")
run_stage(hardware-mismatch-architecture "${hardware-mismatch-architecture_session}" FAIL)

make_session(hardware-mismatch-logical-cores)
rewrite_session_record_field("${hardware-mismatch-logical-cores_session}" logical_cores 4)
run_stage(hardware-mismatch-logical-cores
  "${hardware-mismatch-logical-cores_session}" FAIL)

make_session(zero-fingerprint)
file(READ "${zero-fingerprint_session}/qualification.env" zero_fingerprint_manifest)
string(REGEX REPLACE
  "HOST_FINGERPRINT_SHA256=[0-9a-f]+"
  "HOST_FINGERPRINT_SHA256=0000000000000000000000000000000000000000000000000000000000000000"
  zero_fingerprint_manifest "${zero_fingerprint_manifest}")
file(WRITE "${zero-fingerprint_session}/qualification.env" "${zero_fingerprint_manifest}")
foreach(preflight IN ITEMS preflight-candidate.env preflight-verification-1.env
                              preflight-verification-2.env)
  file(READ "${zero-fingerprint_session}/${preflight}" zero_fingerprint_preflight)
  string(REGEX REPLACE
    "HOST_FINGERPRINT_SHA256=[0-9a-f]+"
    "HOST_FINGERPRINT_SHA256=0000000000000000000000000000000000000000000000000000000000000000"
    zero_fingerprint_preflight "${zero_fingerprint_preflight}")
  file(WRITE "${zero-fingerprint_session}/${preflight}"
    "${zero_fingerprint_preflight}")
endforeach()
file(SHA256 "${zero-fingerprint_session}/preflight-candidate.env"
  zero_fingerprint_candidate_sha)
file(SHA256 "${zero-fingerprint_session}/preflight-verification-1.env"
  zero_fingerprint_verification_1_sha)
file(SHA256 "${zero-fingerprint_session}/preflight-verification-2.env"
  zero_fingerprint_verification_2_sha)
string(REGEX REPLACE
  "PREFLIGHT_CANDIDATE_SHA256=[0-9a-f]+"
  "PREFLIGHT_CANDIDATE_SHA256=${zero_fingerprint_candidate_sha}"
  zero_fingerprint_manifest "${zero_fingerprint_manifest}")
string(REGEX REPLACE
  "PREFLIGHT_VERIFICATION_1_SHA256=[0-9a-f]+"
  "PREFLIGHT_VERIFICATION_1_SHA256=${zero_fingerprint_verification_1_sha}"
  zero_fingerprint_manifest "${zero_fingerprint_manifest}")
string(REGEX REPLACE
  "PREFLIGHT_VERIFICATION_2_SHA256=[0-9a-f]+"
  "PREFLIGHT_VERIFICATION_2_SHA256=${zero_fingerprint_verification_2_sha}"
  zero_fingerprint_manifest "${zero_fingerprint_manifest}")
file(WRITE "${zero-fingerprint_session}/qualification.env"
  "${zero_fingerprint_manifest}")
run_stage(zero-fingerprint "${zero-fingerprint_session}" FAIL)

make_session(raw)
file(REMOVE "${raw_session}/qualification.env")
run_stage(raw "${raw_session}" FAIL)

make_session(bad-token)
file(WRITE "${bad-token_session}/verification-1-quality.result" "allocation_inconsistent\n")
file(SHA256 "${bad-token_session}/verification-1-quality.result" bad_token_sha)
file(READ "${bad-token_session}/qualification.env" bad_token_manifest)
string(REGEX REPLACE "VERIFICATION_1_QUALITY_SHA256=[0-9a-f]+"
  "VERIFICATION_1_QUALITY_SHA256=${bad_token_sha}" bad_token_manifest "${bad_token_manifest}")
string(REPLACE "VERIFICATION_1_QUALITY_RESULT=stable_pass"
  "VERIFICATION_1_QUALITY_RESULT=allocation_inconsistent" bad_token_manifest "${bad_token_manifest}")
file(WRITE "${bad-token_session}/qualification.env" "${bad_token_manifest}")
run_stage(bad-token "${bad-token_session}" FAIL)

make_session(bad-preflight)
file(READ "${bad-preflight_session}/preflight-verification-1.env" bad_preflight)
string(REPLACE "RESULT=preflight_pass" "RESULT=preflight_failed"
  bad_preflight "${bad_preflight}")
file(WRITE "${bad-preflight_session}/preflight-verification-1.env" "${bad_preflight}")
file(SHA256 "${bad-preflight_session}/preflight-verification-1.env" bad_preflight_sha)
file(READ "${bad-preflight_session}/qualification.env" bad_preflight_manifest)
string(REGEX REPLACE "PREFLIGHT_VERIFICATION_1_SHA256=[0-9a-f]+"
  "PREFLIGHT_VERIFICATION_1_SHA256=${bad_preflight_sha}" bad_preflight_manifest "${bad_preflight_manifest}")
file(WRITE "${bad-preflight_session}/qualification.env" "${bad_preflight_manifest}")
run_stage(bad-preflight "${bad-preflight_session}" FAIL)

make_session(bad-hash)
file(READ "${bad-hash_session}/qualification.env" bad_hash)
string(REGEX REPLACE "CANDIDATE_SHA256=[0-9a-f]+"
  "CANDIDATE_SHA256=0000000000000000000000000000000000000000000000000000000000000000"
  bad_hash "${bad_hash}")
file(WRITE "${bad-hash_session}/qualification.env" "${bad_hash}")
run_stage(bad-hash "${bad-hash_session}" FAIL)

make_session(path-traversal)
file(READ "${path-traversal_session}/qualification.env" traversal)
string(REPLACE "CANDIDATE_PATH=candidate/performance-primary-gcc-debug.json"
  "CANDIDATE_PATH=../escape.json" traversal "${traversal}")
file(WRITE "${path-traversal_session}/qualification.env" "${traversal}")
run_stage(path-traversal "${path-traversal_session}" FAIL)

run_stage(existing "${valid_session}" FAIL)
run_stage(existing-temp "${valid_session}" FAIL)
run_stage(destination-symlink "${valid_session}" FAIL)

make_session(release Release)
run_stage(release "${release_session}" PASS)

make_session(boundary)
foreach(pair IN ITEMS
    "candidate/${profile}.json|2026-10-01T00:01:00Z"
    "verification-1.json|2026-10-01T00:02:02Z"
    "verification-2.json|2026-10-01T00:03:04Z")
  string(REPLACE "|" ";" fields "${pair}")
  list(GET fields 0 relative)
  list(GET fields 1 timestamp)
  file(READ "${boundary_session}/${relative}" record_text)
  string(JSON record_text SET "${record_text}" environment utc_timestamp "\"${timestamp}\"")
  file(WRITE "${boundary_session}/${relative}" "${record_text}\n")
endforeach()
refresh_manifest("${boundary_session}")
run_stage(boundary "${boundary_session}" PASS)

make_session(role-swap)
file(READ "${role-swap_session}/candidate/${profile}.json" candidate_record)
file(READ "${role-swap_session}/verification-1.json" verification_record)
file(WRITE "${role-swap_session}/candidate/${profile}.json" "${verification_record}")
file(WRITE "${role-swap_session}/verification-1.json" "${candidate_record}")
refresh_manifest("${role-swap_session}")
run_stage(role-swap "${role-swap_session}" FAIL)

make_session(forged-statistics)
file(READ "${forged-statistics_session}/candidate/${profile}.json" forged_record)
string(JSON forged_record SET "${forged_record}" cases 0 median_batch_ns 101)
file(WRITE "${forged-statistics_session}/candidate/${profile}.json" "${forged_record}\n")
refresh_manifest("${forged-statistics_session}")
run_stage(forged-statistics "${forged-statistics_session}" FAIL)

make_session(stale-plan)
file(READ "${stale-plan_session}/batch-plan.env" stale_plan)
string(REPLACE "CASE_01_ITERATIONS=64" "CASE_01_ITERATIONS=128" stale_plan "${stale_plan}")
file(WRITE "${stale-plan_session}/batch-plan.env" "${stale_plan}")
refresh_manifest("${stale-plan_session}")
run_stage(stale-plan "${stale-plan_session}" FAIL)

make_session(legacy-record)
file(READ "${legacy-record_session}/candidate/${profile}.json" legacy_record)
string(JSON legacy_record SET "${legacy_record}" schema_version 1)
file(WRITE "${legacy-record_session}/candidate/${profile}.json" "${legacy_record}\n")
refresh_manifest("${legacy-record_session}")
run_stage(legacy-record "${legacy-record_session}" FAIL)

make_session(missing-key)
file(READ "${missing-key_session}/qualification.env" missing_manifest)
string(REPLACE "REPEATABILITY_RESULT=repeatable_pass\n" "" missing_manifest "${missing_manifest}")
file(WRITE "${missing-key_session}/qualification.env" "${missing_manifest}")
run_stage(missing-key "${missing-key_session}" FAIL)

make_session(unknown-key)
file(APPEND "${unknown-key_session}/qualification.env" "UNREVIEWED_FIELD=unexpected\n")
run_stage(unknown-key "${unknown-key_session}" FAIL)

make_session(reordered-key)
file(READ "${reordered-key_session}/qualification.env" reordered_manifest)
string(REPLACE "SCHEMA_VERSION=2\nRESULT=qualified_pass\n"
  "RESULT=qualified_pass\nSCHEMA_VERSION=2\n" reordered_manifest "${reordered_manifest}")
file(WRITE "${reordered-key_session}/qualification.env" "${reordered_manifest}")
run_stage(reordered-key "${reordered-key_session}" FAIL)

make_session(stale-commit)
file(READ "${stale-commit_session}/qualification.env" stale_manifest)
string(REPLACE "MATH_COMMIT=${expected_commit}"
  "MATH_COMMIT=1111111111111111111111111111111111111111" stale_manifest "${stale_manifest}")
file(WRITE "${stale-commit_session}/qualification.env" "${stale_manifest}")
run_stage(stale-commit "${stale-commit_session}" FAIL)

make_session(symlink-ancestor)
file(RENAME "${symlink-ancestor_session}/candidate"
  "${symlink-ancestor_session}/candidate-held")
file(CREATE_LINK "${symlink-ancestor_session}/candidate-held"
  "${symlink-ancestor_session}/candidate" SYMBOLIC)
run_stage(symlink-ancestor "${symlink-ancestor_session}" FAIL)

make_session(fifo-evidence)
file(REMOVE "${fifo-evidence_session}/verification-1-quality.result")
execute_process(COMMAND mkfifo "${fifo-evidence_session}/verification-1-quality.result"
  RESULT_VARIABLE fifo_status ERROR_VARIABLE fifo_error)
if(NOT fifo_status EQUAL 0)
  message(FATAL_ERROR "could not create deterministic FIFO evidence fixture: ${fifo_error}")
endif()
run_stage(fifo-evidence "${fifo-evidence_session}" FAIL)

message(STATUS "VSS-MATH-PERF schema-v2 baseline staging tests passed")
