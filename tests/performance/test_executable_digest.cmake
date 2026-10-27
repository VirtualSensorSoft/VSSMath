cmake_minimum_required(VERSION 3.28)

foreach(required VSS_PERF_BENCHMARK VSS_PERF_RECORD)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VSSMath ${required} is required")
  endif()
endforeach()
if(NOT EXISTS "${VSS_PERF_BENCHMARK}")
  message(FATAL_ERROR "VSSMath benchmark executable does not exist")
endif()
if(NOT EXISTS "${VSS_PERF_RECORD}")
  message(FATAL_ERROR "VSSMath benchmark record does not exist")
endif()

file(SHA256 "${VSS_PERF_BENCHMARK}" executable_digest)
file(READ "${VSS_PERF_RECORD}" record)
string(JSON environment GET "${record}" environment)
string(JSON recorded_digest GET "${environment}" benchmark_sha256)
if(NOT "${recorded_digest}" STREQUAL "${executable_digest}")
  message(FATAL_ERROR
    "VSSMath executable digest mismatch: record=${recorded_digest} "
    "executable=${executable_digest}")
endif()
