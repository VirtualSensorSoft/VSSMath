cmake_minimum_required(VERSION 3.28)

foreach(required IN ITEMS VSS_PERF_BENCHMARK VSS_PERF_SOURCE_DIR
    VSS_PERF_TEST_SCRATCH VSS_PERF_PROFILE_ID VSS_PERF_CONFIGURATION)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
if(NOT EXISTS "${VSS_PERF_BENCHMARK}")
  message(FATAL_ERROR "benchmark executable does not exist")
endif()
file(MAKE_DIRECTORY "${VSS_PERF_TEST_SCRATCH}")

include("${VSS_PERF_SOURCE_DIR}/tests/performance/algorithm_record.cmake")
include("${VSS_PERF_SOURCE_DIR}/tests/performance/algorithm_batch_plan.cmake")

file(SHA256 "${VSS_PERF_BENCHMARK}" benchmark_sha256)
execute_process(
  COMMAND git -C "${VSS_PERF_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE git_result
  OUTPUT_VARIABLE math_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
string(LENGTH "${math_commit}" math_commit_length)
if(NOT git_result STREQUAL "0" OR NOT math_commit_length EQUAL 40 OR
   NOT math_commit MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "cannot determine the benchmark source commit")
endif()

set(plan_path "${VSS_PERF_TEST_SCRATCH}/fixed plan.env")
set(valid_output
  "${VSS_PERF_TEST_SCRATCH}/valid record/${VSS_PERF_PROFILE_ID}.json")
set(duplicate_output "${VSS_PERF_TEST_SCRATCH}/duplicate record.json")
set(missing_output "${VSS_PERF_TEST_SCRATCH}/missing record.json")
set(plan_content
  "SCHEMA_VERSION=1\nRECORD_PROFILE=${VSS_PERF_PROFILE_ID}\nCONFIGURATION=${VSS_PERF_CONFIGURATION}\nMATH_COMMIT=${math_commit}\nBENCHMARK_SHA256=${benchmark_sha256}\nCASE_COUNT=36\n")
set(expected_iterations)
foreach(index RANGE 0 35)
  list(GET _vss_algorithm_batch_case_ids "${index}" case_id)
  math(EXPR ordinal "${index} + 1")
  if(ordinal LESS 10)
    set(number "0${ordinal}")
  else()
    set(number "${ordinal}")
  endif()
  math(EXPR parity "${index} % 2")
  set(iterations 1)
  if(index LESS 12 AND index GREATER 0 AND parity EQUAL 1)
    set(iterations 2)
  endif()
  list(APPEND expected_iterations "${iterations}")
  string(APPEND plan_content
    "CASE_${number}_ID=${case_id}\nCASE_${number}_ITERATIONS=${iterations}\n")
endforeach()
file(WRITE "${plan_path}" "${plan_content}")

set(metadata_args
  --benchmark-sha256 "${benchmark_sha256}"
  --frequency-governor unavailable
  --turbo-state unavailable
  --runner-label deterministic-test)

# A duplicated batch plan must be rejected before any benchmark operation.
execute_process(
  COMMAND "${VSS_PERF_BENCHMARK}"
    --output "${duplicate_output}"
    --batch-plan "${plan_path}" --batch-plan "${plan_path}"
    ${metadata_args}
  RESULT_VARIABLE duplicate_result
  OUTPUT_VARIABLE duplicate_stdout
  ERROR_VARIABLE duplicate_stderr
  TIMEOUT 1)
if(duplicate_result STREQUAL "Process terminated due to timeout" OR
   duplicate_result STREQUAL "0" OR
   NOT "${duplicate_stderr}" MATCHES "duplicate --batch-plan" OR
   EXISTS "${duplicate_output}")
  message(FATAL_ERROR
    "duplicate --batch-plan was not rejected before work: ${duplicate_result}\n${duplicate_stdout}\n${duplicate_stderr}")
endif()

# Supplying --batch-plan without its path must fail before the operation loop.
execute_process(
  COMMAND "${VSS_PERF_BENCHMARK}"
    --output "${missing_output}" ${metadata_args} --batch-plan
  RESULT_VARIABLE missing_result
  OUTPUT_VARIABLE missing_stdout
  ERROR_VARIABLE missing_stderr
  TIMEOUT 1)
if(missing_result STREQUAL "Process terminated due to timeout" OR
   missing_result STREQUAL "0" OR EXISTS "${missing_output}")
  message(FATAL_ERROR
    "missing --batch-plan value was not rejected before work: ${missing_result}\n${missing_stdout}\n${missing_stderr}")
endif()

execute_process(
  COMMAND "${VSS_PERF_BENCHMARK}"
    --output "${valid_output}" --batch-plan "${plan_path}"
    ${metadata_args}
  RESULT_VARIABLE capture_result
  OUTPUT_VARIABLE capture_stdout
  ERROR_VARIABLE capture_stderr
  TIMEOUT 240)
if(NOT capture_result STREQUAL "0")
  message(FATAL_ERROR
    "schema-v2 capture failed (${capture_result})\n${capture_stdout}\n${capture_stderr}")
endif()
file(GLOB temporary_paths LIST_DIRECTORIES TRUE "${valid_output}.tmp-*")
if(temporary_paths)
  message(FATAL_ERROR "schema-v2 capture left temporary publication artifacts")
endif()

vss_algorithm_record_load("${valid_output}" TRUE record token reason)
if(NOT token STREQUAL "record_valid")
  message(FATAL_ERROR "producer emitted an invalid v2 record: ${token}: ${reason}")
endif()
string(JSON case_count LENGTH "${record}" cases)
if(NOT case_count EQUAL 36)
  message(FATAL_ERROR "producer emitted ${case_count} cases, expected 36")
endif()
foreach(index RANGE 0 35)
  string(JSON item GET "${record}" cases "${index}")
  string(JSON sample_count LENGTH "${item}" batch_durations_ns)
  string(JSON iterations GET "${item}" iterations_per_sample)
  list(GET expected_iterations "${index}" expected)
  if(NOT sample_count EQUAL 30 OR NOT iterations STREQUAL expected)
    message(FATAL_ERROR
      "case ${index} has ${sample_count} samples and ${iterations} iterations; expected 30 and ${expected}")
  endif()
endforeach()
