cmake_minimum_required(VERSION 3.28)
project(VirtualSensorMath VERSION 0.1.1 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(cmake/VssMathWarnings.cmake)
include(cmake/VssMathAnalysis.cmake)
include(cmake/VssMathFormatting.cmake)
include(cmake/VssMathCoverage.cmake)
include(cmake/CollectToolVersions.cmake)

option(VSS_MATH_BUILD_DOCS "Build API documentation with normal builds" ON)
option(VSS_MATH_ENABLE_SANITIZERS "Enable address and undefined sanitizers" OFF)
option(VSS_MATH_BUILD_BENCHMARKS "Build performance benchmark executable" OFF)
option(VSS_MATH_ENABLE_CLANG_TIDY
  "Run the centrally selected clang-tidy policy during compilation" OFF)
option(VSS_MATH_ENABLE_FORMAT_CHECK
  "Check the repository C++ formatting with clang-format" OFF)
option(VSS_MATH_ENABLE_COVERAGE
  "Enable GCC coverage instrumentation and report generation" OFF)

set(vss_math_install_cmake_dir
  "${CMAKE_INSTALL_LIBDIR}/cmake/VirtualSensorMath")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/vss/math/version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/vss/math/version.hpp"
  @ONLY)

vss_math_add_format_check()

add_library(vss_math STATIC
  src/Error.cpp
  src/Matrix.cpp
  src/Operations.cpp
  src/covariance/Covariance.cpp
  src/differentiation/FiniteDifference.cpp
  src/probability/ChiSquare.cpp
  src/statistics/Descriptive.cpp
  src/linear/Cholesky.cpp
  src/linear/LU.cpp
  src/linear/QR.cpp
  src/linear/SymmetricEigen.cpp
  src/linear/TriangularSolve.cpp
  src/Tolerance.cpp
  src/Vector.cpp
  src/version.cpp)
add_library(VirtualSensorSoft::math ALIAS vss_math)

set_target_properties(vss_math PROPERTIES
  EXPORT_NAME math
  CXX_EXTENSIONS OFF)

target_compile_features(vss_math PUBLIC cxx_std_20)
target_include_directories(vss_math
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(vss_math PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src")

vss_math_enable_warnings(vss_math)
vss_math_enable_clang_tidy(vss_math)
if(VSS_MATH_ENABLE_SANITIZERS)
  vss_math_enable_sanitizers(vss_math)
endif()
vss_math_enable_coverage(vss_math)
vss_math_add_tool_version_target()

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
  add_subdirectory(tests)
  if(NOT VSS_MATH_ENABLE_SANITIZERS AND NOT VSS_MATH_ENABLE_COVERAGE)
    add_test(NAME math.installed_consumer
      COMMAND "${CMAKE_COMMAND}"
        "-DVSS_MATH_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
        "-DVSS_MATH_PRODUCER_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DVSS_MATH_CONSUMER_SCRATCH=${CMAKE_BINARY_DIR}/installed-consumer"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunInstalledConsumer.cmake")
  endif()
endif()

if(VSS_MATH_BUILD_BENCHMARKS)
  message(STATUS "VSS-MATH-PERF raw benchmark records are not qualified baselines.\nRun target algorithm-performance-qualification under an approved host profile before staging evidence.")
  add_subdirectory(benchmarks)
endif()

if(VSS_MATH_BUILD_DOCS)
  find_package(Doxygen REQUIRED COMPONENTS dot)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    @ONLY)
  add_custom_target(docs ALL
    COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Building VirtualSensorMath API documentation"
    VERBATIM)
  add_dependencies(docs vss_math)
endif()

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VirtualSensorMathConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/VirtualSensorMathConfig.cmake"
  INSTALL_DESTINATION "${vss_math_install_cmake_dir}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/VirtualSensorMathConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMinorVersion)

install(TARGETS vss_math
  EXPORT VirtualSensorMathTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  COMPONENT Development
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT Development
  FILES_MATCHING PATTERN "*.hpp")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/generated/vss/math/version.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/vss/math"
  COMPONENT Development)
install(EXPORT VirtualSensorMathTargets
  FILE VirtualSensorMathTargets.cmake
  NAMESPACE VirtualSensorSoft::
  DESTINATION "${vss_math_install_cmake_dir}"
  COMPONENT Development)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/VirtualSensorMathConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/VirtualSensorMathConfigVersion.cmake"
  DESTINATION "${vss_math_install_cmake_dir}"
  COMPONENT Development)
