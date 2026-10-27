cmake_minimum_required(VERSION 3.28)

set(checker "${CMAKE_CURRENT_LIST_DIR}/check_algorithm_regression.cmake")
set(work "${CMAKE_CURRENT_BINARY_DIR}/performance12-schema-fixtures")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
include("${CMAKE_CURRENT_LIST_DIR}/support/algorithm_fixtures.cmake")

set(ids
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

function(make_record out profile configuration median allocations)
  vss_algorithm_fixture_record(json "${profile}" "${configuration}"
    "${median}" 1 "${allocations}")
  set("${out}" "${json}" PARENT_SCOPE)
endfunction()

function(run_case name profile baseline_json current_json expected_token expected_status)
  set(baseline_path "${work}/${profile}.json")
  set(current_path "${work}/current.json")
  set(result_path "${work}/${name}.result")
  if("${name}" STREQUAL "path-mismatch")
    set(baseline_path "${work}/wrong-name.json")
  endif()
  file(WRITE "${baseline_path}" "${baseline_json}")
  file(WRITE "${current_path}" "${current_json}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_BASELINE=${baseline_path}"
      "-DVSS_PERF_CURRENT=${current_path}"
      "-DVSS_PERF_RESULT=${result_path}"
      "-DVSS_PERF_MAX_PERCENT=120"
      -P "${checker}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT EXISTS "${result_path}")
    message(FATAL_ERROR "${name}: checker did not write a result")
  endif()
  file(READ "${result_path}" token)
  string(STRIP "${token}" token)
  if(NOT "${token}" STREQUAL "${expected_token}")
    message(FATAL_ERROR
      "${name}: expected ${expected_token}, got ${token}\n${output}${error}")
  endif()
  if("${expected_status}" STREQUAL PASS AND NOT status EQUAL 0)
    message(FATAL_ERROR "${name}: expected success\n${output}${error}")
  endif()
  if("${expected_status}" STREQUAL FAIL AND status EQUAL 0)
    message(FATAL_ERROR "${name}: expected failure")
  endif()
endfunction()

make_record(valid_base performance-test-profile Debug 100 2)
make_record(valid_current performance-test-profile Debug 100 2)
run_case(valid performance-test-profile "${valid_base}" "${valid_current}" compatible_pass PASS)

string(REPLACE "\"coverage\":\"off\"," "" missing_key "${valid_base}")
run_case(missing-key performance-test-profile "${missing_key}" "${valid_current}" invalid_record FAIL)

string(REPLACE "\"schema_version\":2,"
  "\"schema_version\":2,\"extra\":1," extra_key "${valid_base}")
run_case(extra-key performance-test-profile "${extra_key}" "${valid_current}" invalid_record FAIL)

string(REPLACE "\"warmup_min_operations\":10"
  "\"warmup_min_operations\":\"10\"" wrong_type "${valid_base}")
run_case(wrong-type performance-test-profile "${wrong_type}" "${valid_current}" invalid_record FAIL)

run_case(path-mismatch wrong-profile "${valid_base}" "${valid_current}" invalid_record FAIL)

string(REPLACE "\"case_id\":\"tri.solve.n16.rhs4\""
  "\"case_id\":\"tri.solve.n64.rhs4\"" duplicate_case "${valid_base}")
run_case(duplicate-case performance-test-profile "${duplicate_case}" "${valid_current}" invalid_record FAIL)

string(REPLACE "\"checksum_hex\":\"0x1p+0\""
  "\"checksum_hex\":\"nan\"" nonfinite "${valid_base}")
run_case(nonfinite-checksum performance-test-profile "${nonfinite}" "${valid_current}" invalid_record FAIL)

string(REPLACE "\"correctness_passed\":true"
  "\"correctness_passed\":false" false_result "${valid_base}")
run_case(false-correctness performance-test-profile "${false_result}" "${valid_current}" invalid_record FAIL)

string(REPLACE "\"clock\":\"steady_clock\""
  "\"clock\":\"system_clock\"" wrong_method "${valid_base}")
run_case(wrong-method performance-test-profile "${wrong_method}" "${valid_current}" incompatible FAIL)

string(REPLACE "\"configuration\":\"Debug\""
  "\"configuration\":\"Release\"" incompatible "${valid_current}")
run_case(incompatible performance-test-profile "${valid_base}" "${incompatible}" incompatible FAIL)

make_record(exact_current performance-test-profile Debug 120 2)
run_case(exact-120 performance-test-profile "${valid_base}" "${exact_current}" compatible_pass PASS)

make_record(timing_current performance-test-profile Debug 121 2)
run_case(timing-regression performance-test-profile "${valid_base}" "${timing_current}" timing_regression FAIL)

make_record(equal_allocation performance-test-profile Debug 100 2)
run_case(equal-allocation performance-test-profile "${valid_base}" "${equal_allocation}" compatible_pass PASS)

make_record(allocation_current performance-test-profile Debug 100 3)
run_case(allocation-regression performance-test-profile "${valid_base}" "${allocation_current}" allocation_regression FAIL)

make_record(zero_base performance-test-profile Debug 0 2)
run_case(zero-median performance-test-profile "${zero_base}" "${valid_current}" invalid_record FAIL)

make_record(large_base performance-test-profile Debug 8000000000000000000 2)
make_record(large_current performance-test-profile Debug 9223372036854775807 2)
run_case(large-integer performance-test-profile "${large_base}" "${large_current}" compatible_pass PASS)

make_record(timing_bad performance-test-profile Debug 121 2)
make_record(timing_good performance-test-profile Debug 100 2)
file(WRITE "${work}/performance-test-profile.json" "${valid_base}")
set(template_good "${work}/fake-good.json")
set(template_bad "${work}/fake-bad.json")
file(WRITE "${template_good}" "${timing_good}")
file(WRITE "${template_bad}" "${timing_bad}")
set(capture_count "${work}/capture-count.txt")
file(WRITE "${capture_count}" "")
set(fake "${work}/fake-benchmark.sh")
file(WRITE "${fake}"
  "#!/bin/sh\n"
  "printf x >> '${capture_count}'\n"
  "test \"$1\" = --output || exit 11\n"
  "test \"$3\" = --benchmark-sha256 || exit 12\n"
  "test \"$5\" = --frequency-governor || exit 13\n"
  "test \"$6\" = fixture-governor || exit 14\n"
  "test \"$7\" = --turbo-state || exit 15\n"
  "test \"$8\" = fixture-turbo || exit 16\n"
  "test \"$9\" = --runner-label || exit 17\n"
  "test \"\${10}\" = fixture-runner || exit 18\n"
  "cp \"${template_bad}\" \"$2\"\n")
file(CHMOD "${fake}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
file(SHA256 "${fake}" fake_digest)
set(digest_field_old "\"benchmark_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"")
set(digest_field_new "\"benchmark_sha256\":\"${fake_digest}\"")
string(REPLACE "${digest_field_old}" "${digest_field_new}"
  valid_base_for_fake "${valid_base}")
string(REPLACE "${digest_field_old}" "${digest_field_new}"
  timing_good_for_fake "${timing_good}")
string(REPLACE "${digest_field_old}" "${digest_field_new}"
  timing_bad_for_fake "${timing_bad}")
file(WRITE "${work}/performance-test-profile.json" "${valid_base_for_fake}")
file(WRITE "${template_good}" "${timing_good_for_fake}")
file(WRITE "${template_bad}" "${timing_bad_for_fake}")
set(orchestrated "${work}/orchestrated")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${fake}"
    "-DVSS_PERF_BASELINE=${work}/performance-test-profile.json"
    "-DVSS_PERF_OUTPUT_DIR=${orchestrated}"
    "-DVSS_PERF_MAX_ATTEMPTS=1"
    -DVSS_PERF_FREQUENCY_GOVERNOR=fixture-governor
    -DVSS_PERF_TURBO_STATE=fixture-turbo
    -DVSS_PERF_RUNNER_LABEL=fixture-runner
    -P "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_regression.cmake"
  RESULT_VARIABLE orchestrated_status)
if(orchestrated_status EQUAL 0)
  message(FATAL_ERROR "one-capture regression accepted a timing regression")
endif()
file(READ "${orchestrated}/summary.txt" orchestrated_token)
string(STRIP "${orchestrated_token}" orchestrated_token)
file(READ "${capture_count}" capture_count_content)
string(LENGTH "${capture_count_content}" capture_count_length)
if(NOT orchestrated_token STREQUAL timing_regression OR
   NOT capture_count_length EQUAL 1 OR
   NOT EXISTS "${orchestrated}/current.json")
  message(FATAL_ERROR "regression orchestration did not capture exactly once")
endif()

set(rejected_attempts "${work}/rejected-attempts")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${fake}"
    "-DVSS_PERF_BASELINE=${work}/performance-test-profile.json"
    "-DVSS_PERF_OUTPUT_DIR=${rejected_attempts}"
    "-DVSS_PERF_MAX_ATTEMPTS=3"
    -DVSS_PERF_FREQUENCY_GOVERNOR=fixture-governor
    -DVSS_PERF_TURBO_STATE=fixture-turbo
    -DVSS_PERF_RUNNER_LABEL=fixture-runner
    -P "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_regression.cmake"
  RESULT_VARIABLE rejected_attempts_status)
if(rejected_attempts_status EQUAL 0 OR
   EXISTS "${rejected_attempts}/current.json")
  message(FATAL_ERROR "regression wrapper accepted retry count other than one")
endif()
file(READ "${capture_count}" capture_count_after_rejection)
string(LENGTH "${capture_count_after_rejection}" capture_count_after_rejection_length)
file(READ "${rejected_attempts}/summary.txt" rejected_attempts_token)
string(STRIP "${rejected_attempts_token}" rejected_attempts_token)
if(NOT capture_count_after_rejection_length EQUAL 1 OR
   NOT rejected_attempts_token STREQUAL invalid_record)
  message(FATAL_ERROR "invalid retry policy was not rejected before capture")
endif()

set(digest_mismatch "${work}/digest-mismatch.sh")
file(WRITE "${digest_mismatch}"
  "#!/bin/sh\ncp \"${template_good}\" \"$2\"\n")
file(CHMOD "${digest_mismatch}"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
set(digest_mismatch_output "${work}/digest-mismatch")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${digest_mismatch}"
    "-DVSS_PERF_BASELINE=${work}/performance-test-profile.json"
    "-DVSS_PERF_OUTPUT_DIR=${digest_mismatch_output}"
    "-DVSS_PERF_MAX_ATTEMPTS=1"
    -DVSS_PERF_FREQUENCY_GOVERNOR=fixture-governor
    -DVSS_PERF_TURBO_STATE=fixture-turbo
    -DVSS_PERF_RUNNER_LABEL=fixture-runner
    -P "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_regression.cmake"
  RESULT_VARIABLE digest_mismatch_status)
if(digest_mismatch_status EQUAL 0)
  message(FATAL_ERROR "orchestrator accepted an executable digest mismatch")
endif()
file(READ "${digest_mismatch_output}/summary.txt" digest_mismatch_token)
string(STRIP "${digest_mismatch_token}" digest_mismatch_token)
if(NOT digest_mismatch_token STREQUAL invalid_record)
  message(FATAL_ERROR
    "orchestrator did not classify executable digest mismatch as invalid_record")
endif()

set(metadata_fake "${work}/metadata-forwarding.sh")
set(metadata_output "${work}/metadata-forwarding.json")
set(metadata_template "${work}/metadata-template.json")
file(WRITE "${metadata_fake}"
  "#!/bin/sh\n"
  "test \"$1\" = --output || exit 11\n"
  "test \"$3\" = --benchmark-sha256 || exit 12\n"
  "test \"$5\" = --frequency-governor || exit 13\n"
  "test \"$6\" = powersave || exit 14\n"
  "test \"$7\" = --turbo-state || exit 15\n"
  "test \"$8\" = enabled || exit 16\n"
  "test \"$9\" = --runner-label || exit 17\n"
  "test \"\${10}\" = performance-primary-local-linux-v1 || exit 18\n"
  "cp \"${metadata_template}\" \"$2\"\n")
file(CHMOD "${metadata_fake}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
file(SHA256 "${metadata_fake}" metadata_fake_sha)
string(REPLACE "${digest_field_old}"
  "\"benchmark_sha256\":\"${metadata_fake_sha}\""
  metadata_record "${valid_base}")
string(REPLACE "\"frequency_governor\":\"performance\""
  "\"frequency_governor\":\"powersave\"" metadata_record "${metadata_record}")
string(REPLACE "\"runner_label_or_host\":\"fixture-runner\""
  "\"runner_label_or_host\":\"performance-primary-local-linux-v1\""
  metadata_record "${metadata_record}")
file(WRITE "${metadata_template}" "${metadata_record}")
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=${metadata_fake}"
    "-DVSS_PERF_OUTPUT=${metadata_output}"
    -DVSS_PERF_FREQUENCY_GOVERNOR=powersave
    -DVSS_PERF_TURBO_STATE=enabled
    -DVSS_PERF_RUNNER_LABEL=performance-primary-local-linux-v1
    -P "${CMAKE_CURRENT_LIST_DIR}/run_algorithm_benchmark.cmake"
  RESULT_VARIABLE metadata_status OUTPUT_VARIABLE metadata_stdout
  ERROR_VARIABLE metadata_stderr)
if(NOT metadata_status EQUAL 0)
  message(FATAL_ERROR "metadata forwarding failed: ${metadata_stdout}${metadata_stderr}")
endif()
file(READ "${metadata_output}" metadata_json)
string(JSON metadata_environment GET "${metadata_json}" environment)
foreach(metadata_pair
    "frequency_governor;powersave"
    "turbo_state;enabled"
    "runner_label_or_host;performance-primary-local-linux-v1")
  list(GET metadata_pair 0 metadata_key)
  list(GET metadata_pair 1 metadata_expected)
  string(JSON metadata_actual GET "${metadata_environment}" "${metadata_key}")
  if(NOT metadata_actual STREQUAL metadata_expected)
    message(FATAL_ERROR "metadata ${metadata_key} was not propagated")
  endif()
endforeach()

message(STATUS "VSSMath schema and regression fixtures passed")
