cmake_minimum_required(VERSION 3.28)

set(checker "${CMAKE_CURRENT_LIST_DIR}/../../benchmarks/check_regression.cmake")
set(baseline "${CMAKE_CURRENT_LIST_DIR}/fixtures/baseline.env")
set(current_pass "${CMAKE_CURRENT_LIST_DIR}/fixtures/current-pass.env")
set(current_fail "${CMAKE_CURRENT_LIST_DIR}/fixtures/current-fail.env")

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DVSS_PERF_BASELINE=${baseline}"
          "-DVSS_PERF_CURRENT=${current_pass}"
          "-DVSS_PERF_MAX_PERCENT=120"
          -P "${checker}"
  RESULT_VARIABLE pass_result
  OUTPUT_VARIABLE pass_output
  ERROR_VARIABLE pass_error)
if(NOT pass_result EQUAL 0)
  message(FATAL_ERROR
    "passing performance fixture was rejected:\n${pass_output}${pass_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DVSS_PERF_BASELINE=${baseline}"
          "-DVSS_PERF_CURRENT=${current_fail}"
          "-DVSS_PERF_MAX_PERCENT=120"
          -P "${checker}"
  RESULT_VARIABLE fail_result
  OUTPUT_VARIABLE fail_output
  ERROR_VARIABLE fail_error)
if(fail_result EQUAL 0)
  message(FATAL_ERROR "failing performance fixture was accepted")
endif()
if(NOT "${fail_output}${fail_error}" MATCHES "vector_add_16_ns")
  message(FATAL_ERROR
    "failing fixture did not identify vector_add_16_ns:\n${fail_output}${fail_error}")
endif()

message(STATUS "performance regression checker contract passed")
