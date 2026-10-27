set(vss_math_benchmark_warning_flags
  "-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Werror")
string(TOUPPER "${CMAKE_BUILD_TYPE}" vss_math_build_type_upper)
set(vss_math_config_flags
  "${CMAKE_CXX_FLAGS_${vss_math_build_type_upper}}")
string(STRIP "${CMAKE_CXX_FLAGS} ${vss_math_config_flags}"
  vss_math_cmake_flags)
set(vss_math_compiler_options
  "cxx_standard=20;cxx_extensions=off;project_warnings=${vss_math_benchmark_warning_flags};cmake_flags=${vss_math_cmake_flags};sanitizers=off")
set(vss_math_benchmark_command
  "vss_math_benchmark --output performance/current.env --build-type ${CMAKE_BUILD_TYPE}")

function(vss_math_escape_cpp_string output input)
  set(value "${input}")
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  string(REPLACE "\r" "\\r" value "${value}")
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

vss_math_escape_cpp_string(VSS_BENCHMARK_COMPILER_ID
  "${CMAKE_CXX_COMPILER_ID}")
vss_math_escape_cpp_string(VSS_BENCHMARK_COMPILER_VERSION
  "${CMAKE_CXX_COMPILER_VERSION}")
vss_math_escape_cpp_string(VSS_BENCHMARK_BUILD_TYPE
  "${CMAKE_BUILD_TYPE}")
vss_math_escape_cpp_string(VSS_BENCHMARK_COMPILER_OPTIONS
  "${vss_math_compiler_options}")
vss_math_escape_cpp_string(VSS_BENCHMARK_COMMAND
  "${vss_math_benchmark_command}")

set(VSS_MATH_PERFORMANCE_PROFILE_ID "unqualified-current" CACHE STRING
  "Stable profile label embedded in performance records")
if(NOT VSS_MATH_PERFORMANCE_PROFILE_ID MATCHES "^[a-z0-9][a-z0-9._-]*$")
  message(FATAL_ERROR
    "VSS_MATH_PERFORMANCE_PROFILE_ID has invalid syntax: "
    "${VSS_MATH_PERFORMANCE_PROFILE_ID}")
endif()
if((VSS_MATH_PERFORMANCE_PROFILE_ID STREQUAL "performance-primary-gcc-debug" OR
    VSS_MATH_PERFORMANCE_PROFILE_ID STREQUAL "performance-primary-gcc-release") AND
   NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR
    "Reviewed VSS-MATH-PERF primary profiles require the GNU compiler")
endif()

execute_process(
  COMMAND git -C "${PROJECT_SOURCE_DIR}" rev-parse HEAD
  OUTPUT_VARIABLE vss_math_git_commit
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if("${vss_math_git_commit}" STREQUAL "")
  set(vss_math_git_commit "unavailable")
endif()
if("${CMAKE_CXX_COMPILER_TARGET}" STREQUAL "")
  set(vss_math_target_triple "${CMAKE_SYSTEM_PROCESSOR}-unknown-${CMAKE_SYSTEM_NAME}")
else()
  set(vss_math_target_triple "${CMAKE_CXX_COMPILER_TARGET}")
endif()
set(vss_math_sanitizers "off")
if(VSS_MATH_ENABLE_SANITIZERS)
  set(vss_math_sanitizers "address,undefined")
endif()
set(vss_math_coverage "off")
if(VSS_MATH_ENABLE_COVERAGE)
  set(vss_math_coverage "gcc")
endif()
set(vss_math_link_flags "${CMAKE_EXE_LINKER_FLAGS}")
set(vss_math_compile_flags
  "${CMAKE_CXX_FLAGS} ${vss_math_config_flags} ${vss_math_benchmark_warning_flags}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(vss_math_record_build_type "Unspecified")
else()
  set(vss_math_record_build_type "${CMAKE_BUILD_TYPE}")
endif()

vss_math_escape_cpp_string(VSS_BENCHMARK_PROFILE_ID
  "${VSS_MATH_PERFORMANCE_PROFILE_ID}")
vss_math_escape_cpp_string(VSS_BENCHMARK_PACKAGE_VERSION
  "${PROJECT_VERSION}")
vss_math_escape_cpp_string(VSS_BENCHMARK_MATH_COMMIT
  "${vss_math_git_commit}")
vss_math_escape_cpp_string(VSS_BENCHMARK_TARGET_TRIPLE
  "${vss_math_target_triple}")
vss_math_escape_cpp_string(VSS_BENCHMARK_CMAKE_VERSION
  "${CMAKE_VERSION}")
vss_math_escape_cpp_string(VSS_BENCHMARK_COMPILE_FLAGS
  "${vss_math_compile_flags}")
vss_math_escape_cpp_string(VSS_BENCHMARK_LINK_FLAGS
  "${vss_math_link_flags}")
vss_math_escape_cpp_string(VSS_BENCHMARK_SANITIZERS
  "${vss_math_sanitizers}")
vss_math_escape_cpp_string(VSS_BENCHMARK_COVERAGE
  "${vss_math_coverage}")
vss_math_escape_cpp_string(VSS_BENCHMARK_ARCHITECTURE
  "${CMAKE_SYSTEM_PROCESSOR}")
vss_math_escape_cpp_string(VSS_BENCHMARK_BUILD_TYPE
  "${vss_math_record_build_type}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_metadata.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/benchmark_metadata.hpp"
  @ONLY)

add_executable(vss_math_benchmark benchmark_foundation.cpp)
set_target_properties(vss_math_benchmark PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(vss_math_benchmark PRIVATE VirtualSensorSoft::math)
target_include_directories(vss_math_benchmark PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
vss_math_enable_warnings(vss_math_benchmark)
vss_math_enable_clang_tidy(vss_math_benchmark)

add_executable(test_benchmark_standard_library_metadata
  "${PROJECT_SOURCE_DIR}/tests/performance/test_benchmark_standard_library_metadata.cpp")
set_target_properties(test_benchmark_standard_library_metadata PROPERTIES
  CXX_EXTENSIONS OFF)
target_include_directories(test_benchmark_standard_library_metadata PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated")
vss_math_enable_warnings(test_benchmark_standard_library_metadata)
vss_math_enable_clang_tidy(test_benchmark_standard_library_metadata)
add_test(NAME math.benchmark_standard_library_metadata
  COMMAND test_benchmark_standard_library_metadata)

add_custom_target(performance-baseline
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${CMAKE_BINARY_DIR}/performance"
  COMMAND $<TARGET_FILE:vss_math_benchmark>
          --output "${CMAKE_BINARY_DIR}/performance/current.env"
          --build-type "${CMAKE_BUILD_TYPE}"
  DEPENDS vss_math_benchmark
  VERBATIM)

add_executable(vss_math_algorithm_benchmark "${CMAKE_CURRENT_SOURCE_DIR}/../tests/performance/benchmark_algorithms.cpp")
target_sources(vss_math_algorithm_benchmark PRIVATE
  "${PROJECT_SOURCE_DIR}/tests/performance/support/BenchmarkBatchPlan.cpp"
  "${PROJECT_SOURCE_DIR}/tests/performance/support/BenchmarkMeasurement.cpp")
set_target_properties(vss_math_algorithm_benchmark PROPERTIES CXX_EXTENSIONS OFF)
include("${PROJECT_SOURCE_DIR}/cmake/VssMathBenchmarkIsolation.cmake")
vss_math_enforce_benchmark_isolation(vss_math_algorithm_benchmark)
vss_math_disable_benchmark_ipo(vss_math_algorithm_benchmark)
target_link_libraries(vss_math_algorithm_benchmark PRIVATE VirtualSensorSoft::math)
target_include_directories(vss_math_algorithm_benchmark PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/generated"
  "${PROJECT_SOURCE_DIR}/tests/performance")
vss_math_enable_warnings(vss_math_algorithm_benchmark)
vss_math_enable_clang_tidy(vss_math_algorithm_benchmark)
add_test(NAME math.algorithm_benchmark_schema_v2
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_BENCHMARK=$<TARGET_FILE:vss_math_algorithm_benchmark>"
    "-DVSS_PERF_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DVSS_PERF_TEST_SCRATCH=${CMAKE_BINARY_DIR}/algorithm-benchmark-schema-v2-test"
    "-DVSS_PERF_PROFILE_ID=${VSS_MATH_PERFORMANCE_PROFILE_ID}"
    "-DVSS_PERF_CONFIGURATION=${CMAKE_BUILD_TYPE}"
    -P "${PROJECT_SOURCE_DIR}/tests/performance/test_algorithm_benchmark_v2.cmake")
set_tests_properties(math.algorithm_benchmark_schema_v2 PROPERTIES
  RUN_SERIAL TRUE TIMEOUT 300)
add_executable(test_algorithm_benchmark_capture_stub
  "${PROJECT_SOURCE_DIR}/tests/performance/test_algorithm_benchmark_capture_stub.cpp")
set_target_properties(test_algorithm_benchmark_capture_stub PROPERTIES
  CXX_EXTENSIONS OFF)
target_compile_features(test_algorithm_benchmark_capture_stub PRIVATE cxx_std_20)
vss_math_enable_warnings(test_algorithm_benchmark_capture_stub)
vss_math_enable_clang_tidy(test_algorithm_benchmark_capture_stub)
add_test(NAME math.algorithm_benchmark_wrapper
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_CMAKE_COMMAND=${CMAKE_COMMAND}"
    "-DVSS_PERF_STUB=$<TARGET_FILE:test_algorithm_benchmark_capture_stub>"
    "-DVSS_PERF_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DVSS_PERF_SCRATCH=${CMAKE_BINARY_DIR}/algorithm-benchmark-wrapper-test"
    -P "${PROJECT_SOURCE_DIR}/tests/performance/test_algorithm_benchmark_wrapper.cmake")
set_tests_properties(math.algorithm_benchmark_wrapper PROPERTIES RUN_SERIAL TRUE)
add_executable(test_algorithm_ci_diagnostic_stub
  "${PROJECT_SOURCE_DIR}/tests/performance/test_algorithm_ci_diagnostic_stub.cpp")
set_target_properties(test_algorithm_ci_diagnostic_stub PROPERTIES
  CXX_EXTENSIONS OFF)
target_compile_features(test_algorithm_ci_diagnostic_stub PRIVATE cxx_std_20)
vss_math_enable_warnings(test_algorithm_ci_diagnostic_stub)
vss_math_enable_clang_tidy(test_algorithm_ci_diagnostic_stub)
add_test(NAME math.algorithm_ci_diagnostic
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_CMAKE_COMMAND=${CMAKE_COMMAND}"
    "-DVSS_PERF_STUB=$<TARGET_FILE:test_algorithm_ci_diagnostic_stub>"
    "-DVSS_PERF_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DVSS_PERF_SCRATCH=${CMAKE_BINARY_DIR}/algorithm-ci-diagnostic-test"
    -P "${PROJECT_SOURCE_DIR}/tests/performance/test_algorithm_ci_diagnostic.cmake")
set_tests_properties(math.algorithm_ci_diagnostic PROPERTIES RUN_SERIAL TRUE)

set(VSS_MATH_BENCHMARK_FREQUENCY_GOVERNOR "" CACHE STRING
  "Explicit raw algorithm benchmark frequency-governor metadata; empty reads host sysfs")
set(VSS_MATH_BENCHMARK_TURBO_STATE "" CACHE STRING
  "Explicit raw algorithm benchmark turbo metadata; empty reads host sysfs")
set(VSS_MATH_BENCHMARK_RUNNER_LABEL "" CACHE STRING
  "Explicit raw algorithm benchmark runner label; empty uses the local default")
set(vss_math_algorithm_benchmark_metadata_args)
if(NOT "${VSS_MATH_BENCHMARK_FREQUENCY_GOVERNOR}" STREQUAL "")
  list(APPEND vss_math_algorithm_benchmark_metadata_args
    "-DVSS_PERF_FREQUENCY_GOVERNOR=${VSS_MATH_BENCHMARK_FREQUENCY_GOVERNOR}")
endif()
if(NOT "${VSS_MATH_BENCHMARK_TURBO_STATE}" STREQUAL "")
  list(APPEND vss_math_algorithm_benchmark_metadata_args
    "-DVSS_PERF_TURBO_STATE=${VSS_MATH_BENCHMARK_TURBO_STATE}")
endif()
if(NOT "${VSS_MATH_BENCHMARK_RUNNER_LABEL}" STREQUAL "")
  list(APPEND vss_math_algorithm_benchmark_metadata_args
    "-DVSS_PERF_RUNNER_LABEL=${VSS_MATH_BENCHMARK_RUNNER_LABEL}")
endif()
add_custom_target(algorithm-performance-baseline
  COMMAND "${CMAKE_COMMAND}" -E make_directory
          "${CMAKE_BINARY_DIR}/performance"
  COMMAND "${CMAKE_COMMAND}"
          "-DVSS_PERF_BENCHMARK=$<TARGET_FILE:vss_math_algorithm_benchmark>"
          "-DVSS_PERF_OUTPUT=${CMAKE_BINARY_DIR}/performance/performance-current.json"
          ${vss_math_algorithm_benchmark_metadata_args}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/performance/run_algorithm_benchmark.cmake"
  DEPENDS vss_math_algorithm_benchmark
  VERBATIM)

set(VSS_MATH_ALGORITHM_BASELINE "" CACHE FILEPATH "Reviewed performance baseline record used for regression comparison")
set(VSS_MATH_BENCHMARK_HOST_PROFILE_FILE "" CACHE FILEPATH
  "User-provided host profile for opt-in performance qualification; none is distributed")
set(vss_math_qualification_profile "${VSS_MATH_BENCHMARK_HOST_PROFILE_FILE}")
add_custom_target(algorithm-performance-qualification
  COMMAND "${CMAKE_COMMAND}" -E echo
          "VSS-MATH-PERF qualification profile: performance-primary-local-linux-v1"
  COMMAND "${CMAKE_COMMAND}" -E echo
          "Configuration: ${CMAKE_BUILD_TYPE}; affinity CPU: 4"
  COMMAND "${CMAKE_COMMAND}" -E echo
          "Limits: load 0.50; package temperature 55000 mC; stable windows 60/10/900 seconds"
  COMMAND "${CMAKE_COMMAND}" -E echo
          "AC power alone does not guarantee qualification; close workloads and allow cooling."
  COMMAND "${CMAKE_COMMAND}"
          "-DVSS_PERF_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
          "-DVSS_PERF_OUTPUT_DIR=${CMAKE_BINARY_DIR}/performance/qualification"
          "-DVSS_PERF_CONFIGURATION=${CMAKE_BUILD_TYPE}"
          "-DVSS_PERF_RECORD_PROFILE_ID=${VSS_MATH_PERFORMANCE_PROFILE_ID}"
          "-DVSS_PERF_HOST_PROFILE_FILE=${vss_math_qualification_profile}"
          "-DVSS_PERF_BENCHMARK=$<TARGET_FILE:vss_math_algorithm_benchmark>"
          "-DVSS_PERF_EXPECTED_COMMIT=${vss_math_git_commit}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/performance/run_algorithm_qualification.cmake"
  DEPENDS vss_math_algorithm_benchmark
  VERBATIM)
if(NOT "${VSS_MATH_ALGORITHM_BASELINE}" STREQUAL "")
  add_custom_target(algorithm-performance-regression
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_BENCHMARK=$<TARGET_FILE:vss_math_algorithm_benchmark>"
      "-DVSS_PERF_BASELINE=${VSS_MATH_ALGORITHM_BASELINE}"
      "-DVSS_PERF_OUTPUT_DIR=${CMAKE_BINARY_DIR}/performance/regression"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/performance/run_algorithm_regression.cmake"
    DEPENDS vss_math_algorithm_benchmark
    VERBATIM)
  add_test(NAME math.algorithm_performance_regression
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_PERF_BENCHMARK=$<TARGET_FILE:vss_math_algorithm_benchmark>"
      "-DVSS_PERF_BASELINE=${VSS_MATH_ALGORITHM_BASELINE}"
      "-DVSS_PERF_OUTPUT_DIR=${CMAKE_BINARY_DIR}/performance/regression"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/../tests/performance/run_algorithm_regression.cmake")
  set_tests_properties(math.algorithm_performance_regression PROPERTIES
    RUN_SERIAL TRUE LABELS performance)
endif()
