cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_comparison.cmake")

foreach(required IN ITEMS VSS_PERF_BASELINE VSS_PERF_CURRENT
    VSS_PERF_MAX_PERCENT)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VSSMath ${required} is required")
  endif()
endforeach()

if(NOT VSS_PERF_MAX_PERCENT STREQUAL "120")
  set(token invalid_record)
  set(reason "only the reviewed 120 percent threshold is supported")
else()
  vss_algorithm_regression("${VSS_PERF_BASELINE}" "${VSS_PERF_CURRENT}"
    token reason)
endif()

if(DEFINED VSS_PERF_RESULT AND NOT "${VSS_PERF_RESULT}" STREQUAL "")
  vss_algorithm_write_result("${VSS_PERF_RESULT}" "${token}" result_ok
    result_reason)
  if(NOT result_ok)
    message(FATAL_ERROR "VSSMath could not write result: ${result_reason}")
  endif()
endif()

if(token STREQUAL "compatible_pass")
  message(STATUS "VSSMath ${token}")
  return()
endif()
message(FATAL_ERROR "VSSMath ${token}: ${reason}")
