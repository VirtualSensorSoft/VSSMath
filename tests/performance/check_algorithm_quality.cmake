cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")

if(NOT DEFINED VSS_PERF_RESULT OR "${VSS_PERF_RESULT}" STREQUAL "")
  message(FATAL_ERROR "VSS_PERF_RESULT is required")
endif()

if(NOT DEFINED VSS_PERF_RECORD OR "${VSS_PERF_RECORD}" STREQUAL "")
  set(token invalid_record)
  set(reason "VSS_PERF_RECORD is required")
else()
  _vss_algorithm_comparison_load("${VSS_PERF_RECORD}" FALSE json token reason)
  if(token STREQUAL "record_valid")
    _vss_algorithm_record_quality_validated("${json}" token reason)
  endif()
endif()

vss_algorithm_write_result("${VSS_PERF_RESULT}" "${token}" result_written
  write_reason)
if(NOT result_written)
  message(FATAL_ERROR "${write_reason}")
endif()
if(NOT token STREQUAL "stable_pass")
  message(FATAL_ERROR "${token}: ${reason}")
endif()
message(STATUS "stable_pass: record quality accepted")
