cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_decimal.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/support/algorithm_fixtures.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")

set(work "${CMAKE_CURRENT_BINARY_DIR}/performance-v2-record-fixtures")
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")

function(require_equal label expected actual)
  if(NOT "${expected}" STREQUAL "${actual}")
    message(FATAL_ERROR
      "VSSMath0 Task 2 ${label}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

function(expect_load label json expected_token filename require_match)
  set(path "${work}/${filename}")
  file(WRITE "${path}" "${json}")
  vss_algorithm_record_load("${path}" "${require_match}" loaded token reason)
  if(NOT token STREQUAL expected_token)
    message(FATAL_ERROR "VSSMath0 Task 2 ${label}: expected ${expected_token}, got ${token}: ${reason}")
  endif()
  require_equal("${label} token" "${expected_token}" "${token}")
  if(token STREQUAL record_valid)
    require_equal("${label} preserves source" "${json}" "${loaded}")
  endif()
endfunction()

function(expect_quality label json expected_token)
  vss_algorithm_record_quality("${json}" token reason)
  require_equal("${label} quality token" "${expected_token}" "${token}")
endfunction()

vss_algorithm_fixture_record(valid performance-test-profile Debug 100 2 3)
expect_load("valid v2 fixture" "${valid}" record_valid performance-test-profile.json TRUE)

# This inventory is deliberately independent from the fixture builder.
set(expected_ids
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
string(JSON case_count LENGTH "${valid}" cases)
list(LENGTH expected_ids expected_count)
require_equal("literal inventory count" "${expected_count}" "${case_count}")
math(EXPR last_case "${expected_count} - 1")
foreach(index RANGE 0 "${last_case}")
  list(GET expected_ids "${index}" expected_id)
  string(JSON actual_id GET "${valid}" cases "${index}" case_id)
  require_equal("literal case order ${index}" "${expected_id}" "${actual_id}")
endforeach()

string(REPLACE "\"schema_version\":2,"
  "\"schema_version\":2,\"unexpected\":true," extra_key "${valid}")
expect_load("extra root key" "${extra_key}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"profile_id\":\"performance-test-profile\"," "" missing_key "${valid}")
expect_load("missing root key" "${missing_key}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"schema_version\":2," "\"schema_version\":\"2\","
  wrong_type "${valid}")
expect_load("wrong root type" "${wrong_type}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"schema_version\":2," "\"schema_version\":1," typed_v1 "${valid}")
expect_load("schema v1" "${typed_v1}" incompatible performance-test-profile.json TRUE)

string(REPLACE "\"method\":{" "\"method\":{\"unexpected\":1,"
  extra_method "${valid}")
expect_load("extra method key" "${extra_method}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"dispersion\":\"integer_mad\"," "" missing_method "${valid}")
expect_load("missing method key" "${missing_method}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"samples\":30" "\"samples\":\"30\"" method_type "${valid}")
expect_load("wrong method type" "${method_type}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"samples\":30" "\"samples\":29" method_value "${valid}")
expect_load("method policy mismatch" "${method_value}" incompatible performance-test-profile.json TRUE)

string(REPLACE "\"environment\":{" "\"environment\":{\"unexpected\":1,"
  extra_environment "${valid}")
expect_load("extra environment key" "${extra_environment}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"coverage\":\"off\"," "" missing_environment "${valid}")
expect_load("missing environment key" "${missing_environment}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"compiler_id\":\"GNU\"" "\"compiler_id\":1"
  environment_type "${valid}")
expect_load("wrong environment type" "${environment_type}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"math_commit\":\"0000000000000000000000000000000000000000\""
  "\"math_commit\":\"bad\"" bad_commit "${valid}")
expect_load("invalid commit identity" "${bad_commit}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"utc_timestamp\":\"2026-01-01T00:00:00Z\""
  "\"utc_timestamp\":\"bad\"" bad_timestamp "${valid}")
expect_load("invalid timestamp" "${bad_timestamp}" invalid_record performance-test-profile.json TRUE)

string(REPLACE "\"case_id\":\"tri.solve.n16.rhs4\""
  "\"case_id\":\"tri.solve.n16.rhs4\",\"unknown\":1" extra_case "${valid}")
expect_load("extra case key" "${extra_case}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"allocations_per_operation\":3," "" missing_case "${valid}")
expect_load("missing case key" "${missing_case}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"correctness_passed\":true" "\"correctness_passed\":false"
  failed_correctness "${valid}")
expect_load("failed correctness" "${failed_correctness}" invalid_record performance-test-profile.json TRUE)

string(REPLACE "\"batch_durations_ns\":[100,100"
  "\"batch_durations_ns\":[0,100" zero_sample "${valid}")
expect_load("zero duration" "${zero_sample}" invalid_record performance-test-profile.json TRUE)
string(REPEAT "100," 28 samples_29)
string(APPEND samples_29 "100")
string(REPEAT "100," 30 samples_31)
string(APPEND samples_31 "100")
string(REPLACE
  "\"batch_durations_ns\":[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]"
  "\"batch_durations_ns\":[${samples_29}]" record_29 "${valid}")
expect_load("29 samples" "${record_29}" invalid_record performance-test-profile.json TRUE)
string(REPLACE
  "\"batch_durations_ns\":[100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]"
  "\"batch_durations_ns\":[${samples_31}]" record_31 "${valid}")
expect_load("31 samples" "${record_31}" invalid_record performance-test-profile.json TRUE)
foreach(bad_sample IN ITEMS -1 1.5 1e2 18446744073709551616)
  string(REPLACE "\"batch_durations_ns\":[100,100"
    "\"batch_durations_ns\":[${bad_sample},100" malformed_sample "${valid}")
  expect_load("invalid sample ${bad_sample}" "${malformed_sample}"
    invalid_record performance-test-profile.json TRUE)
endforeach()
string(REPLACE "\"iterations_per_sample\":2" "\"iterations_per_sample\":3"
  bad_iterations "${valid}")
expect_load("non power-of-two iterations" "${bad_iterations}" invalid_record
  performance-test-profile.json TRUE)
string(REPLACE "\"checksum_hex\":\"0x1p+0\"" "\"checksum_hex\":\"nan\""
  bad_checksum "${valid}")
expect_load("nonfinite checksum" "${bad_checksum}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"checksum_hex\":\"0x1p+0\"" "\"checksum_hex\":\"inf\""
  infinite_checksum "${valid}")
expect_load("infinite checksum" "${infinite_checksum}" invalid_record
  performance-test-profile.json TRUE)

string(JSON forged_median SET "${valid}" cases 0 median_batch_ns 101)
expect_load("forged median" "${forged_median}" invalid_record performance-test-profile.json TRUE)
string(JSON forged_mad SET "${valid}" cases 0 mad_batch_ns 1)
expect_load("forged MAD" "${forged_mad}" invalid_record performance-test-profile.json TRUE)
string(JSON forged_display SET "${valid}" cases 0 median_ns_per_operation 51)
expect_load("forged display" "${forged_display}" invalid_record performance-test-profile.json TRUE)

string(JSON first_case GET "${valid}" cases 0)
string(JSON second_case GET "${valid}" cases 1)
string(JSON reordered SET "${valid}" cases 0 "${second_case}")
string(JSON reordered SET "${reordered}" cases 1 "${first_case}")
expect_load("reordered cases" "${reordered}" invalid_record performance-test-profile.json TRUE)
string(JSON duplicate_case SET "${valid}" cases 0 "${second_case}")
expect_load("duplicate case IDs" "${duplicate_case}" invalid_record performance-test-profile.json TRUE)
set(object_cases "{")
foreach(index RANGE 0 35)
  if(index GREATER 0)
    string(APPEND object_cases ",")
  endif()
  string(JSON item GET "${valid}" cases "${index}")
  string(APPEND object_cases "\"${index}\":${item}")
endforeach()
string(APPEND object_cases "}")
string(JSON cases_as_object SET "${valid}" cases "${object_cases}")
expect_load("case collection must be array" "${cases_as_object}"
  invalid_record performance-test-profile.json TRUE)

string(REPLACE "{\"schema_version\":2,"
  "{\"schema_version\":2,\"schema_version\":2," duplicate_root "${valid}")
expect_load("duplicate root member" "${duplicate_root}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"method\":{" "\"method\":{\"samples\":30,"
  duplicate_method "${valid}")
expect_load("duplicate method member" "${duplicate_method}" invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"environment\":{"
  "\"environment\":{\"compiler_id\":\"GNU\"," duplicate_environment "${valid}")
expect_load("duplicate environment member" "${duplicate_environment}"
  invalid_record performance-test-profile.json TRUE)
string(REPLACE "\"case_id\":\"tri.solve.n16.rhs4\""
  "\"case_id\":\"tri.solve.n16.rhs4\",\"case_id\":\"tri.solve.n16.rhs4\""
  duplicate_case_key "${valid}")
expect_load("duplicate case member" "${duplicate_case_key}" invalid_record
  performance-test-profile.json TRUE)
string(REPLACE "{\"schema_version\":2,"
  "{\"schema_\\u0076ersion\":2,\"schema_version\":2,"
  escaped_duplicate "${valid}")
expect_load("escaped alias duplicate member" "${escaped_duplicate}"
  invalid_record performance-test-profile.json TRUE)

string(REPLACE "\"runner_label_or_host\":\"fixture-runner\""
  "\"runner_label_or_host\":\"brace { } quote \\\" colon : semicolon ; backslash \\\\ schema_version\""
  escaped_string "${valid}")
expect_load("escaped metadata string" "${escaped_string}" record_valid
  performance-test-profile.json TRUE)
string(REPLACE "\"runner_label_or_host\":\"fixture-runner\""
  "\"runner_label_or_host\":\"broken \\q escape\""
  malformed_escape "${valid}")
expect_load("malformed escape" "${malformed_escape}" invalid_record
  performance-test-profile.json TRUE)
expect_load("empty input" "" invalid_record performance-test-profile.json TRUE)
expect_load("trailing data" "${valid} trailing" invalid_record performance-test-profile.json TRUE)
expect_load("JSON comment" "// not JSON\n${valid}" invalid_record performance-test-profile.json TRUE)
expect_load("wrong basename" "${valid}" invalid_record other-name.json TRUE)
expect_load("basename check disabled" "${valid}" record_valid other-name.json FALSE)
string(REPEAT " " 1048577 oversized_padding)
set(oversized "${valid}${oversized_padding}")
expect_load("file over one MiB" "${oversized}" invalid_record
  performance-test-profile.json FALSE)
expect_quality("direct JSON over one MiB" "${oversized}" invalid_record)
vss_algorithm_record_compatible("${valid}" "${oversized}" FALSE token reason)
require_equal("compatibility rejects JSON over one MiB" incompatible "${token}")

vss_algorithm_checksum_finite("0x0p+0" finite)
require_equal("zero checksum" TRUE "${finite}")
vss_algorithm_checksum_finite("-0x1p+0" finite)
require_equal("negative finite checksum" TRUE "${finite}")
vss_algorithm_checksum_finite("0x1p+1024" finite)
require_equal("overflow checksum" FALSE "${finite}")
vss_algorithm_checksum_finite("0x1.fffffffffffffp+1023" finite)
require_equal("largest finite checksum" TRUE "${finite}")
vss_algorithm_checksum_finite("0x1.fffffffffffff1p+1023" finite)
require_equal("just-over-maximum checksum" FALSE "${finite}")
vss_algorithm_checksum_finite("0x1p-1074" finite)
require_equal("minimum subnormal checksum" TRUE "${finite}")
vss_algorithm_checksum_finite("0x1p-1075" finite)
require_equal("underflow checksum" FALSE "${finite}")
vss_algorithm_checksum_finite("0x1p+999999999999999999999999" finite)
require_equal("huge exponent checksum" FALSE "${finite}")

vss_algorithm_fixture_record(quality performance-test-profile Debug 100 2 3)
set(stable_samples "")
set(unstable_samples "")
foreach(i RANGE 1 15)
  string(APPEND stable_samples "95,")
  string(APPEND unstable_samples "94,")
endforeach()
foreach(i RANGE 1 15)
  if(i GREATER 1)
    string(APPEND stable_samples ",")
    string(APPEND unstable_samples ",")
  endif()
  string(APPEND stable_samples "105")
  string(APPEND unstable_samples "106")
endforeach()
string(REGEX REPLACE
  "\"batch_durations_ns\":\\[[0-9,]+\\],\"median_batch_ns\":100,\"mad_batch_ns\":0"
  "\"batch_durations_ns\":[${stable_samples}],\"median_batch_ns\":100,\"mad_batch_ns\":5"
  stable_boundary "${quality}")
expect_quality("95/105 at exact five percent" "${stable_boundary}" stable_pass)
string(REGEX REPLACE
  "\"batch_durations_ns\":\\[[0-9,]+\\],\"median_batch_ns\":100,\"mad_batch_ns\":0"
  "\"batch_durations_ns\":[${unstable_samples}],\"median_batch_ns\":100,\"mad_batch_ns\":6"
  unstable "${quality}")
expect_quality("94/106 above five percent" "${unstable}" unstable_measurement)
vss_algorithm_fixture_record(tiny performance-test-profile Debug 3 4 0)
expect_load("zero ns-per-operation display is valid" "${tiny}" record_valid
  performance-test-profile.json TRUE)
expect_quality("zero display remains quality-valid" "${tiny}" stable_pass)

vss_algorithm_fixture_record(base performance-test-profile Debug 100 2 3)
vss_algorithm_fixture_record(current performance-test-profile Debug 100 2 3)
vss_algorithm_record_compatible("${base}" "${current}" FALSE token reason)
require_equal("matching records compatible" compatible "${token}")
string(REPLACE "\"math_commit\":\"0000000000000000000000000000000000000000\""
  "\"math_commit\":\"1111111111111111111111111111111111111111\""
  different_commit "${current}")
vss_algorithm_record_compatible("${base}" "${different_commit}" FALSE token reason)
require_equal("generic compatibility ignores commit" compatible "${token}")
vss_algorithm_record_compatible("${base}" "${different_commit}" TRUE token reason)
require_equal("qualification compares commit" incompatible "${token}")
string(REPLACE "\"compiler_version\":\"13.3.0\""
  "\"compiler_version\":\"14.1.0\"" different_compiler "${current}")
vss_algorithm_record_compatible("${base}" "${different_compiler}" FALSE token reason)
require_equal("compiler mismatch" incompatible "${token}")
string(JSON different_iterations SET "${current}" cases 0 iterations_per_sample 4)
string(JSON different_iterations SET "${different_iterations}" cases 0
  median_ns_per_operation 25)
vss_algorithm_record_compatible("${base}" "${different_iterations}" FALSE token reason)
require_equal("generic compatibility ignores iteration plan" compatible "${token}")
vss_algorithm_record_compatible("${base}" "${different_iterations}" TRUE token reason)
require_equal("qualification checks iteration plan" incompatible "${token}")

set(regular "${work}/regular.json")
file(WRITE "${regular}" "${valid}")
vss_algorithm_regular_file("${regular}" is_regular reason)
require_equal("regular file accepted" TRUE "${is_regular}")
set(empty "${work}/empty.json")
file(WRITE "${empty}" "")
vss_algorithm_regular_file("${empty}" is_regular reason)
require_equal("empty regular file still classifies as regular" TRUE "${is_regular}")
set(directory "${work}/directory")
file(MAKE_DIRECTORY "${directory}")
vss_algorithm_regular_file("${directory}" is_regular reason)
require_equal("directory rejected" FALSE "${is_regular}")
set(symlink "${work}/symlink.json")
file(CREATE_LINK "${regular}" "${symlink}" SYMBOLIC RESULT link_result)
if(NOT link_result STREQUAL "0")
  message(FATAL_ERROR "Could not create symbolic-link fixture: ${link_result}")
endif()
vss_algorithm_regular_file("${symlink}" is_regular reason)
require_equal("symbolic link rejected" FALSE "${is_regular}")
vss_algorithm_record_load("${symlink}" FALSE loaded token reason)
require_equal("symbolic-link record rejected" invalid_record "${token}")
if(EXISTS "/dev/null")
  vss_algorithm_regular_file("/dev/null" is_regular reason)
  require_equal("device node rejected" FALSE "${is_regular}")
endif()
if(UNIX)
  find_program(mkfifo_executable mkfifo)
  if(mkfifo_executable)
    set(fifo "${work}/record.fifo")
    execute_process(COMMAND "${mkfifo_executable}" "${fifo}"
      RESULT_VARIABLE fifo_status ERROR_VARIABLE fifo_error)
    if(NOT fifo_status EQUAL 0)
      message(FATAL_ERROR "Could not create FIFO fixture: ${fifo_error}")
    endif()
    vss_algorithm_regular_file("${fifo}" is_regular reason)
    require_equal("FIFO rejected" FALSE "${is_regular}")
  endif()
endif()

message(STATUS "VSSMath0 Task 2 record-validation tests passed")
