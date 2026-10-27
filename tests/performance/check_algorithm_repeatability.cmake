cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")

if(NOT DEFINED VSS_PERF_RESULT OR "${VSS_PERF_RESULT}" STREQUAL "")
  message(FATAL_ERROR "VSS_PERF_RESULT is required")
endif()

set(required_inputs VSS_PERF_CANDIDATE VSS_PERF_VERIFICATION_1
  VSS_PERF_VERIFICATION_2)
set(missing_input "")
foreach(input IN LISTS required_inputs)
  if(NOT DEFINED ${input} OR "${${input}}" STREQUAL "")
    set(missing_input "${input}")
    break()
  endif()
endforeach()

if(NOT "${missing_input}" STREQUAL "")
  set(token invalid_record)
  set(reason "${missing_input} is required")
else()
  vss_algorithm_repeatability("${VSS_PERF_CANDIDATE}"
    "${VSS_PERF_VERIFICATION_1}" "${VSS_PERF_VERIFICATION_2}" token reason)
endif()

vss_algorithm_write_result("${VSS_PERF_RESULT}" "${token}" result_written
  write_reason)
if(NOT result_written)
  message(FATAL_ERROR "${write_reason}")
endif()
if(NOT token STREQUAL "repeatable_pass")
  message(FATAL_ERROR "${token}: ${reason}")
endif()
message(STATUS "repeatable_pass: three records are repeatable")
