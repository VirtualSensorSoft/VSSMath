cmake_minimum_required(VERSION 3.28)

foreach(required_variable IN ITEMS VSS_MATH_INSTALL_PREFIX
                                  VSS_MATH_SOURCE_DIR
                                  VSS_MATH_PRODUCER_BUILD_DIR
                                  VSS_MATH_EVIDENCE_DIR)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

set(vss_install "${VSS_MATH_INSTALL_PREFIX}")
set(vss_consumer_source "${VSS_MATH_SOURCE_DIR}/tests/consumer")
if(DEFINED VSS_MATH_CONSUMER_SCRATCH AND
   NOT "${VSS_MATH_CONSUMER_SCRATCH}" STREQUAL "")
  set(vss_consumer_build "${VSS_MATH_CONSUMER_SCRATCH}")
else()
  set(vss_consumer_build "${VSS_MATH_PRODUCER_BUILD_DIR}/installed-consumer-build")
endif()
set(vss_evidence "${VSS_MATH_EVIDENCE_DIR}")

if(NOT IS_DIRECTORY "${vss_install}")
  message(FATAL_ERROR "installed prefix does not exist: ${vss_install}")
endif()
file(MAKE_DIRECTORY "${vss_evidence}")
cmake_path(IS_PREFIX vss_install "${vss_consumer_build}" NORMALIZE
  vss_install_contains_consumer)
cmake_path(IS_PREFIX vss_consumer_build "${vss_install}" NORMALIZE
  vss_consumer_contains_install)
if(vss_install_contains_consumer OR vss_consumer_contains_install)
  message(FATAL_ERROR
    "consumer build path must not contain or be contained by the install prefix")
endif()
file(REMOVE_RECURSE "${vss_consumer_build}")
file(MAKE_DIRECTORY "${vss_consumer_build}")

function(vss_run description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  file(APPEND "${vss_evidence}/consumer-summary.txt"
    "== ${description} ==\n${stdout}${stderr}\n")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${description} failed with exit code ${result}")
  endif()
endfunction()

function(vss_probe_reject_foreign_target_origin case_name include_path library_path expected_diagnostic)
  set(probe_root "${vss_evidence}/imported-target-provenance/${case_name}")
  set(probe_prefix "${probe_root}/selected-prefix")
  set(probe_config_dir "${probe_prefix}/lib/cmake/VirtualSensorMath")
  set(probe_build "${probe_root}/consumer-build")
  if(EXISTS "${probe_root}" OR IS_SYMLINK "${probe_root}")
    file(REMOVE_RECURSE "${probe_root}")
  endif()
  file(MAKE_DIRECTORY "${probe_config_dir}" "${probe_prefix}/include"
    "${probe_prefix}/lib" "${probe_root}/foreign/include"
    "${probe_root}/foreign/lib")
  file(WRITE "${probe_prefix}/include/probe.hpp" "// selected prefix\n")
  file(WRITE "${probe_root}/foreign/include/probe.hpp" "// foreign prefix\n")
  file(COPY_FILE "${vss_archive}" "${probe_prefix}/lib/libvss_math.a")
  file(COPY_FILE "${vss_archive}" "${probe_root}/foreign/lib/libvss_math.a")
  file(WRITE "${probe_config_dir}/VirtualSensorMathConfig.cmake"
    "set(VirtualSensorMath_VERSION 0.1.0)\n"
    "include(\"${probe_config_dir}/VirtualSensorMathTargets.cmake\")\n")
  file(WRITE "${probe_config_dir}/VirtualSensorMathConfigVersion.cmake"
    "set(PACKAGE_VERSION \"0.1.0\")\n"
    "if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)\n"
    "  set(PACKAGE_VERSION_EXACT TRUE)\n"
    "  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
    "endif()\n")
  file(WRITE "${probe_config_dir}/VirtualSensorMathTargets.cmake"
    "add_library(VirtualSensorSoft::math STATIC IMPORTED)\n"
    "set_target_properties(VirtualSensorSoft::math PROPERTIES\n"
    "  INTERFACE_INCLUDE_DIRECTORIES \"${include_path}\"\n"
    "  IMPORTED_CONFIGURATIONS RELEASE\n"
    "  IMPORTED_LOCATION_RELEASE \"${library_path}\")\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${vss_consumer_source}" -B "${probe_build}"
      "-DCMAKE_PREFIX_PATH=${probe_prefix}"
      "-DVirtualSensorMath_DIR=${probe_config_dir}"
      "-DVSS_EXPECTED_PREFIX=${probe_prefix}"
      -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
      -DCMAKE_FIND_USE_PACKAGE_ROOT_PATH=FALSE
      -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
      -DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE
      -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=FALSE
    RESULT_VARIABLE probe_result
    OUTPUT_VARIABLE probe_stdout
    ERROR_VARIABLE probe_stderr)
  set(probe_output "${probe_stdout}${probe_stderr}")
  file(WRITE "${vss_evidence}/imported-target-provenance-${case_name}.txt"
    "case=${case_name}\n"
    "selected_prefix=${probe_prefix}\n"
    "expected_diagnostic=${expected_diagnostic}\n"
    "result=${probe_result}\n${probe_output}")
  if(probe_result EQUAL 0)
    message(FATAL_ERROR
      "installed consumer accepted foreign ${case_name} origin; expected: "
      "${expected_diagnostic}")
  endif()
  string(REGEX REPLACE "[ \t\r\n]+" " " probe_output_normalized
    "${probe_output}")
  string(FIND "${probe_output_normalized}" "${expected_diagnostic}" diagnostic_index)
  if(diagnostic_index EQUAL -1)
    message(FATAL_ERROR
      "foreign ${case_name} probe failed for an unexpected reason; expected: "
      "${expected_diagnostic}; output: ${probe_output}")
  endif()
endfunction()

set(vss_expected_files
  include/vss/math/Core.hpp
  include/vss/math/Error.hpp
  include/vss/math/Matrix.hpp
  include/vss/math/MatrixView.hpp
  include/vss/math/Operations.hpp
  include/vss/math/Result.hpp
  include/vss/math/Tolerance.hpp
  include/vss/math/Vector.hpp
  include/vss/math/VectorView.hpp
  include/vss/math/covariance/Covariance.hpp
  include/vss/math/differentiation/FiniteDifference.hpp
  include/vss/math/probability/ChiSquare.hpp
  include/vss/math/statistics/Descriptive.hpp
  include/vss/math/linear/Cholesky.hpp
  include/vss/math/linear/FactorizationOptions.hpp
  include/vss/math/linear/LinearSolve.hpp
  include/vss/math/linear/LU.hpp
  include/vss/math/linear/QR.hpp
  include/vss/math/linear/SymmetricEigen.hpp
  include/vss/math/linear/TriangularSolve.hpp
  include/vss/math/version.hpp
  lib/cmake/VirtualSensorMath/VirtualSensorMathConfig.cmake
  lib/cmake/VirtualSensorMath/VirtualSensorMathConfigVersion.cmake
  lib/cmake/VirtualSensorMath/VirtualSensorMathTargets.cmake
  lib/libvss_math.a)
file(GLOB vss_config_fragments LIST_DIRECTORIES false
  RELATIVE "${vss_install}"
  "${vss_install}/lib/cmake/VirtualSensorMath/VirtualSensorMathTargets-*.cmake")
list(LENGTH vss_config_fragments vss_config_fragment_count)
if(NOT vss_config_fragment_count EQUAL 1)
  message(FATAL_ERROR
    "installed package must contain exactly one configuration export fragment")
endif()
list(GET vss_config_fragments 0 vss_config_fragment)
if(NOT vss_config_fragment MATCHES
   "^lib/cmake/VirtualSensorMath/VirtualSensorMathTargets-(debug|release)\\.cmake$")
  message(FATAL_ERROR
    "installed configuration export fragment is not a supported Debug/Release file: "
    "${vss_config_fragment}")
endif()
list(APPEND vss_expected_files "${vss_config_fragment}")
list(SORT vss_expected_files)

file(GLOB_RECURSE vss_installed_files LIST_DIRECTORIES false
  RELATIVE "${vss_install}" "${vss_install}/*")
list(SORT vss_installed_files)
if(NOT vss_installed_files STREQUAL vss_expected_files)
  message(FATAL_ERROR
    "installed file inventory differs from the approved Development allow-list: "
    "${vss_installed_files}")
endif()

string(JOIN "\n" vss_inventory ${vss_installed_files})
file(WRITE "${vss_evidence}/installed-files.txt" "${vss_inventory}\n")
file(WRITE "${vss_evidence}/installed-digests.txt" "")

foreach(path IN LISTS vss_installed_files)
  if(path MATCHES "(^|/)(tests?|benchmarks?|src|\\.git|\\.vscode|\\.idea)(/|$)" OR
     path MATCHES "(^|/)(credentials?|secrets?|.*token.*)(/|$)" OR
     path MATCHES "(^|/).*\\.env($|/)")
    message(FATAL_ERROR "forbidden content in install prefix: ${path}")
  endif()
  file(SHA256 "${vss_install}/${path}" digest)
  file(APPEND "${vss_evidence}/installed-digests.txt"
    "${digest}  ${path}\n")
endforeach()

foreach(path IN LISTS vss_installed_files)
  if(path MATCHES "\\.(cmake|hpp)$")
    file(READ "${vss_install}/${path}" contents)
    string(FIND "${contents}" "${VSS_MATH_SOURCE_DIR}" source_index)
    string(FIND "${contents}" "${VSS_MATH_PRODUCER_BUILD_DIR}" build_index)
    if(NOT source_index EQUAL -1 OR NOT build_index EQUAL -1)
      message(FATAL_ERROR "installed text metadata leaks a source/build path: ${path}")
    endif()
  endif()
endforeach()

set(vss_targets_file
  "${vss_install}/lib/cmake/VirtualSensorMath/VirtualSensorMathTargets.cmake")
file(READ "${vss_targets_file}" vss_targets)
if(NOT vss_targets MATCHES "VirtualSensorSoft::math")
  message(FATAL_ERROR "installed targets file lacks VirtualSensorSoft::math")
endif()

set(vss_archive "${vss_install}/lib/libvss_math.a")
if(vss_config_fragment MATCHES "-release\\.cmake$")
  find_program(vss_strings NAMES strings)
  if(NOT vss_strings)
    message(FATAL_ERROR "strings executable is required for Release archive path scanning")
  endif()
  execute_process(
    COMMAND "${vss_strings}" -a "${vss_archive}"
    RESULT_VARIABLE strings_result
    OUTPUT_VARIABLE archive_strings
    ERROR_VARIABLE strings_error)
  if(NOT strings_result EQUAL 0)
    message(FATAL_ERROR "archive strings scan failed: ${strings_error}")
  endif()
  string(FIND "${archive_strings}" "${VSS_MATH_SOURCE_DIR}" archive_source_index)
  string(FIND "${archive_strings}" "${VSS_MATH_PRODUCER_BUILD_DIR}" archive_build_index)
  if(NOT archive_source_index EQUAL -1 OR NOT archive_build_index EQUAL -1)
    message(FATAL_ERROR "Release archive strings leak a source/build path")
  endif()
  execute_process(COMMAND "${vss_strings}" --version
    OUTPUT_VARIABLE strings_version OUTPUT_STRIP_TRAILING_WHITESPACE)
  file(WRITE "${vss_evidence}/archive-strings-scan.txt"
    "release archive scanned with ${vss_strings}\n")
else()
  set(strings_version "not-executed-for-debug")
  file(WRITE "${vss_evidence}/archive-strings-scan.txt"
    "skipped: archive path scanning is a Release-only check\n")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" --version
  OUTPUT_VARIABLE cmake_version OUTPUT_STRIP_TRAILING_WHITESPACE)
file(WRITE "${vss_evidence}/verifier-tool-versions.txt"
  "${cmake_version}\n${strings_version}\n")

set(vss_foreign_include_root
  "${vss_evidence}/imported-target-provenance/foreign-include")
vss_probe_reject_foreign_target_origin(
  foreign-include
  "${vss_foreign_include_root}/foreign/include"
  "${vss_foreign_include_root}/selected-prefix/lib/libvss_math.a"
  "VSSMath INTERFACE_INCLUDE_DIRECTORIES path resolves outside VSS_EXPECTED_PREFIX")
set(vss_foreign_library_root
  "${vss_evidence}/imported-target-provenance/foreign-library")
vss_probe_reject_foreign_target_origin(
  foreign-library
  "${vss_foreign_library_root}/selected-prefix/include"
  "${vss_foreign_library_root}/foreign/lib/libvss_math.a"
  "VSSMath imported library location resolves outside VSS_EXPECTED_PREFIX")

vss_run("consumer configure"
  "${CMAKE_COMMAND}" -S "${vss_consumer_source}" -B "${vss_consumer_build}"
  "-DCMAKE_PREFIX_PATH=${vss_install}"
  "-DVSS_EXPECTED_PREFIX=${vss_install}"
  "-DVSS_PACKAGE_IDENTITY_FILE=${vss_evidence}/consumer-package-identity.txt"
  -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
  -DCMAKE_FIND_USE_PACKAGE_ROOT_PATH=FALSE
  -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
  -DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE
  -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=FALSE)

vss_run("installed package version compatibility probes"
  "${CMAKE_COMMAND}"
  "-DVSS_MATH_INSTALL_PREFIX=${vss_install}"
  "-DVSS_MATH_EVIDENCE_DIR=${vss_evidence}"
  "-DVSS_MATH_PROBE_SCRATCH=${vss_evidence}/package-version-probes"
  -P "${VSS_MATH_SOURCE_DIR}/cmake/CheckInstalledPackageVersion.cmake")

file(READ "${vss_consumer_build}/CMakeCache.txt" vss_cache)
string(REGEX MATCH "VirtualSensorMath_DIR:PATH=([^\n\r]*)" vss_dir_match
  "${vss_cache}")
if(NOT vss_dir_match)
  message(FATAL_ERROR "consumer cache does not define VirtualSensorMath_DIR")
endif()
set(vss_resolved_dir "${CMAKE_MATCH_1}")
set(vss_install_prefix "${vss_install}")
cmake_path(IS_PREFIX vss_install_prefix "${vss_resolved_dir}" NORMALIZE vss_dir_inside)
if(NOT vss_dir_inside)
  message(FATAL_ERROR
    "consumer resolved VirtualSensorMath outside isolated install: ${vss_resolved_dir}")
endif()
set(vss_package_identity_file "${vss_evidence}/consumer-package-identity.txt")
if(NOT EXISTS "${vss_package_identity_file}")
  message(FATAL_ERROR "consumer did not record the resolved package identity")
endif()
file(READ "${vss_package_identity_file}" vss_package_identity)
string(REGEX MATCH "package_version=([^\n\r]+)" vss_version_match
  "${vss_package_identity}")
if(NOT vss_version_match)
  message(FATAL_ERROR "consumer package identity omitted its version")
endif()
set(vss_package_version "${CMAKE_MATCH_1}")
if(NOT vss_package_version STREQUAL "0.1.0")
  message(FATAL_ERROR "consumer resolved an unexpected package version: ${vss_package_identity}")
endif()
string(REGEX MATCH "config_dir=([^\n\r]+)" vss_identity_dir_match
  "${vss_package_identity}")
if(NOT vss_identity_dir_match)
  message(FATAL_ERROR "consumer package identity omitted its configuration directory")
endif()
set(vss_identity_dir "${CMAKE_MATCH_1}")
if(NOT vss_identity_dir STREQUAL vss_resolved_dir)
  message(FATAL_ERROR "consumer package identity differs from its CMake cache")
endif()
file(APPEND "${vss_package_identity_file}"
  "request=0.1.0 EXACT CONFIG\n"
  "expected_prefix=${vss_install}\n"
  "prefix_provenance=PASS\n")

set(vss_missing_prefix "${vss_evidence}/missing-package-prefix")
if(EXISTS "${vss_missing_prefix}" OR IS_SYMLINK "${vss_missing_prefix}")
  message(FATAL_ERROR "missing-prefix probe path already exists: ${vss_missing_prefix}")
endif()
set(vss_missing_source "${vss_evidence}/missing-prefix-probe-src")
set(vss_missing_build "${vss_evidence}/missing-prefix-probe-build")
file(REMOVE_RECURSE "${vss_missing_source}" "${vss_missing_build}")
file(MAKE_DIRECTORY "${vss_missing_source}")
file(WRITE "${vss_missing_source}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.28)\n"
  "project(VirtualSensorMathMissingPrefixProbe NONE)\n"
  "find_package(VirtualSensorMath 0.1.0 EXACT CONFIG REQUIRED)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${vss_missing_source}" -B "${vss_missing_build}"
          "-DVirtualSensorMath_DIR=${vss_missing_prefix}/lib/cmake/VirtualSensorMath"
          "-DCMAKE_PREFIX_PATH=${vss_missing_prefix}"
          -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
          -DCMAKE_FIND_USE_PACKAGE_ROOT_PATH=FALSE
          -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
          -DCMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH=FALSE
          -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=FALSE
  RESULT_VARIABLE vss_missing_result
  OUTPUT_VARIABLE vss_missing_stdout
  ERROR_VARIABLE vss_missing_stderr)
if("${vss_missing_result}" STREQUAL "0")
  message(FATAL_ERROR "consumer unexpectedly resolved package from the missing-prefix probe")
endif()
string(CONCAT vss_missing_output "${vss_missing_stdout}" "${vss_missing_stderr}")
string(FIND "${vss_missing_output}"
  "Could not find a package configuration file provided by \"VirtualSensorMath\""
  vss_missing_error_index)
if(vss_missing_error_index EQUAL -1)
  message(FATAL_ERROR
    "missing-prefix probe failed for an unexpected reason:\n${vss_missing_output}")
endif()
file(WRITE "${vss_evidence}/missing-prefix-result.txt"
  "request=VirtualSensorMath 0.1.0 EXACT CONFIG\n"
  "selected_prefix=${vss_missing_prefix}\n"
  "selected_config_dir=${vss_missing_prefix}/lib/cmake/VirtualSensorMath\n"
  "user_package_registry=disabled\n"
  "package_root_paths=disabled\n"
  "system_package_registry=disabled\n"
  "CMake_environment_paths=disabled\n"
  "system_search_path=disabled\n"
  "result=package-not-found-as-expected\n")

vss_run("consumer build" "${CMAKE_COMMAND}" --build "${vss_consumer_build}")
if(WIN32)
  set(vss_consumer_executable "${vss_consumer_build}/math_consumer.exe")
else()
  set(vss_consumer_executable "${vss_consumer_build}/math_consumer")
endif()
if(NOT EXISTS "${vss_consumer_executable}")
  message(FATAL_ERROR "consumer executable was not produced: ${vss_consumer_executable}")
endif()
vss_run("consumer execution" "${vss_consumer_executable}")

message(STATUS "installed artifact, archive, and consumer checks passed")
