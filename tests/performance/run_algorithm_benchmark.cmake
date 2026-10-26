cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")

foreach(required IN ITEMS VSS_PERF_BENCHMARK VSS_PERF_OUTPUT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VSSMath ${required} is required")
  endif()
endforeach()

vss_algorithm_regular_file("${VSS_PERF_BENCHMARK}" benchmark_is_regular
  benchmark_reason)
if(NOT benchmark_is_regular)
  message(FATAL_ERROR "VSSMath benchmark executable is invalid: ${benchmark_reason}")
endif()

get_filename_component(output_path "${VSS_PERF_OUTPUT}" ABSOLUTE)
get_filename_component(output_directory "${output_path}" DIRECTORY)
set(output_ancestor "${output_directory}")
set(output_has_symlink_ancestor FALSE)
while(NOT "${output_ancestor}" STREQUAL "")
  if(IS_SYMLINK "${output_ancestor}")
    set(output_has_symlink_ancestor TRUE)
    break()
  endif()
  get_filename_component(parent_ancestor "${output_ancestor}" DIRECTORY)
  if("${parent_ancestor}" STREQUAL "${output_ancestor}")
    break()
  endif()
  set(output_ancestor "${parent_ancestor}")
endwhile()
if(output_path MATCHES "/tests/performance/baselines(/|$)" OR
   IS_DIRECTORY "${output_path}" OR IS_SYMLINK "${output_path}" OR
   output_has_symlink_ancestor)
  message(FATAL_ERROR "VSSMath output must be a non-canonical raw-record path")
endif()
file(MAKE_DIRECTORY "${output_directory}")

set(batch_plan_args)
if(DEFINED VSS_PERF_BATCH_PLAN AND
   NOT "${VSS_PERF_BATCH_PLAN}" STREQUAL "")
  vss_algorithm_regular_file("${VSS_PERF_BATCH_PLAN}" batch_plan_is_regular
    batch_plan_reason)
  if(NOT batch_plan_is_regular)
    message(FATAL_ERROR "VSSMath batch plan is invalid: ${batch_plan_reason}")
  endif()
  get_filename_component(batch_plan_path "${VSS_PERF_BATCH_PLAN}" ABSOLUTE)
  list(APPEND batch_plan_args --batch-plan "${batch_plan_path}")
endif()

if(DEFINED VSS_PERF_CAPTURE_RESULT AND
   NOT "${VSS_PERF_CAPTURE_RESULT}" STREQUAL "")
  get_filename_component(capture_result "${VSS_PERF_CAPTURE_RESULT}" ABSOLUTE)
  get_filename_component(result_directory "${capture_result}" DIRECTORY)
  if(NOT result_directory STREQUAL output_directory OR
     capture_result STREQUAL output_path OR
     capture_result STREQUAL "${output_path}.stdout.log" OR
     capture_result STREQUAL "${output_path}.stderr.log" OR
     IS_DIRECTORY "${capture_result}" OR IS_SYMLINK "${capture_result}")
    message(FATAL_ERROR
      "VSSMath capture result must be a regular path inside the private output directory")
  endif()
endif()

foreach(log_path IN ITEMS "${output_path}.stdout.log" "${output_path}.stderr.log")
  if(IS_DIRECTORY "${log_path}" OR IS_SYMLINK "${log_path}")
    message(FATAL_ERROR "VSSMath private diagnostic path is not a regular file")
  endif()
endforeach()
foreach(stale_path IN ITEMS "${output_path}" "${output_path}.stdout.log"
    "${output_path}.stderr.log" "${capture_result}")
  if(NOT "${stale_path}" STREQUAL "")
    file(REMOVE "${stale_path}")
  endif()
endforeach()

function(read_linux_value path label out)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}" OR IS_SYMLINK "${path}")
    message(FATAL_ERROR "VSSMath ${label} is missing: ${path}")
  endif()
  file(READ "${path}" value)
  string(STRIP "${value}" value)
  if("${value}" STREQUAL "" OR "${value}" MATCHES "[^A-Za-z0-9._-]")
    message(FATAL_ERROR "VSSMath ${label} is malformed")
  endif()
  set("${out}" "${value}" PARENT_SCOPE)
endfunction()

if(NOT DEFINED VSS_PERF_FREQUENCY_GOVERNOR OR
   "${VSS_PERF_FREQUENCY_GOVERNOR}" STREQUAL "")
  read_linux_value("/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"
    "frequency governor" frequency_governor)
else()
  set(frequency_governor "${VSS_PERF_FREQUENCY_GOVERNOR}")
endif()
if(NOT DEFINED VSS_PERF_TURBO_STATE OR "${VSS_PERF_TURBO_STATE}" STREQUAL "")
  read_linux_value("/sys/devices/system/cpu/intel_pstate/no_turbo"
    "turbo state" turbo_value)
  if(turbo_value STREQUAL "0")
    set(turbo_state enabled)
  elseif(turbo_value STREQUAL "1")
    set(turbo_state disabled)
  else()
    message(FATAL_ERROR "VSSMath turbo state is malformed")
  endif()
else()
  set(turbo_state "${VSS_PERF_TURBO_STATE}")
endif()
if(NOT DEFINED VSS_PERF_RUNNER_LABEL OR "${VSS_PERF_RUNNER_LABEL}" STREQUAL "")
  set(runner_label performance-unqualified-local)
else()
  set(runner_label "${VSS_PERF_RUNNER_LABEL}")
endif()
foreach(metadata_pair
    "frequency governor;${frequency_governor}"
    "turbo state;${turbo_state}"
    "runner label;${runner_label}")
  list(GET metadata_pair 0 metadata_label)
  list(GET metadata_pair 1 metadata_value)
  if("${metadata_value}" STREQUAL "" OR
     "${metadata_value}" MATCHES "[^A-Za-z0-9._-]")
    message(FATAL_ERROR "VSSMath ${metadata_label} is malformed")
  endif()
endforeach()

file(SHA256 "${VSS_PERF_BENCHMARK}" executable_digest)
set(benchmark_command "${VSS_PERF_BENCHMARK}" --output "${output_path}")
list(APPEND benchmark_command ${batch_plan_args}
  --benchmark-sha256 "${executable_digest}"
  --frequency-governor "${frequency_governor}"
  --turbo-state "${turbo_state}"
  --runner-label "${runner_label}")
execute_process(
  COMMAND ${benchmark_command}
  RESULT_VARIABLE benchmark_status
  OUTPUT_VARIABLE benchmark_output
  ERROR_VARIABLE benchmark_error
  TIMEOUT 2220)
file(WRITE "${output_path}.stdout.log" "${benchmark_output}")
file(WRITE "${output_path}.stderr.log" "${benchmark_error}")

if(benchmark_status STREQUAL "0")
  if(EXISTS "${output_path}")
    vss_algorithm_regular_file("${output_path}" output_is_regular output_reason)
  else()
    set(output_is_regular FALSE)
    set(output_reason "benchmark produced no final JSON")
  endif()
  if(output_is_regular)
    set(capture_token capture_pass)
  else()
    set(capture_token invalid_record)
    file(REMOVE "${output_path}")
  endif()
elseif(benchmark_status STREQUAL "2")
  set(capture_token allocation_inconsistent)
elseif(benchmark_status STREQUAL "3" OR
       benchmark_status MATCHES "[Tt]imeout|[Tt]erminat|[Ss]ignal")
  set(capture_token qualification_incomplete)
else()
  set(capture_token invalid_record)
endif()

if(NOT capture_token STREQUAL "capture_pass")
  file(REMOVE "${output_path}")
endif()
if(DEFINED capture_result AND NOT "${capture_result}" STREQUAL "")
  vss_algorithm_write_result("${capture_result}" "${capture_token}"
    result_ok result_reason)
  if(NOT result_ok)
    message(FATAL_ERROR "VSSMath could not write capture result: ${result_reason}")
  endif()
endif()
if(NOT capture_token STREQUAL "capture_pass")
  message(FATAL_ERROR
    "VSSMath benchmark capture failed (${capture_token}, exit ${benchmark_status}): ${benchmark_error}")
endif()
