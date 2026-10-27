cmake_minimum_required(VERSION 3.28)

foreach(required_variable IN ITEMS VSS_MATH_SOURCE_DIR
                                  VSS_MATH_PRODUCER_BUILD_DIR
                                  VSS_MATH_CONSUMER_SCRATCH)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

set(vss_install "${VSS_MATH_CONSUMER_SCRATCH}/install")
set(vss_evidence "${VSS_MATH_PRODUCER_BUILD_DIR}/acceptance")
file(REMOVE_RECURSE "${vss_install}")
file(MAKE_DIRECTORY "${vss_install}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${VSS_MATH_PRODUCER_BUILD_DIR}"
          --component Development --prefix "${vss_install}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_stdout
  ERROR_VARIABLE install_stderr)
if(NOT install_result EQUAL 0)
  message(STATUS "producer install stdout:\n${install_stdout}")
  message(FATAL_ERROR "producer install failed with exit code ${install_result}: ${install_stderr}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          "-DVSS_MATH_INSTALL_PREFIX=${vss_install}"
          "-DVSS_MATH_SOURCE_DIR=${VSS_MATH_SOURCE_DIR}"
          "-DVSS_MATH_PRODUCER_BUILD_DIR=${VSS_MATH_PRODUCER_BUILD_DIR}"
          "-DVSS_MATH_CONSUMER_SCRATCH=${VSS_MATH_CONSUMER_SCRATCH}/consumer-build"
          "-DVSS_MATH_EVIDENCE_DIR=${vss_evidence}"
          -P "${VSS_MATH_SOURCE_DIR}/cmake/VerifyInstalledArtifact.cmake"
  RESULT_VARIABLE verify_result
  OUTPUT_VARIABLE verify_stdout
  ERROR_VARIABLE verify_stderr)
if(NOT verify_result EQUAL 0)
  message(STATUS "artifact verifier stdout:\n${verify_stdout}")
  message(FATAL_ERROR "artifact verifier failed with exit code ${verify_result}: ${verify_stderr}")
endif()

message(STATUS "installed consumer and package identity checks passed")
