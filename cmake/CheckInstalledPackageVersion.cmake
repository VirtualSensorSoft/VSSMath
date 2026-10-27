cmake_minimum_required(VERSION 3.28)

foreach(required_variable IN ITEMS VSS_MATH_INSTALL_PREFIX
                                  VSS_MATH_EVIDENCE_DIR
                                  VSS_MATH_PROBE_SCRATCH)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

set(vss_install "${VSS_MATH_INSTALL_PREFIX}")
set(vss_probe_scratch "${VSS_MATH_PROBE_SCRATCH}")
cmake_path(SET vss_expected_probe_scratch NORMALIZE
  "${VSS_MATH_EVIDENCE_DIR}/package-version-probes")
cmake_path(SET vss_probe_scratch NORMALIZE "${VSS_MATH_PROBE_SCRATCH}")
if(NOT vss_probe_scratch STREQUAL vss_expected_probe_scratch)
  message(FATAL_ERROR
    "package-version probes must use the dedicated evidence scratch directory")
endif()
set(vss_package_config_dir "${vss_install}/lib/cmake/VirtualSensorMath")
if(NOT EXISTS "${vss_package_config_dir}/VirtualSensorMathConfig.cmake" OR
   NOT EXISTS "${vss_package_config_dir}/VirtualSensorMathConfigVersion.cmake")
  message(FATAL_ERROR "installed package configuration is incomplete: ${vss_package_config_dir}")
endif()

cmake_path(IS_PREFIX vss_install "${vss_probe_scratch}" NORMALIZE
  vss_install_contains_probes)
cmake_path(IS_PREFIX vss_probe_scratch "${vss_install}" NORMALIZE
  vss_probes_contain_install)
if(vss_install_contains_probes OR vss_probes_contain_install)
  message(FATAL_ERROR "package-version probe directory must be separate from the install prefix")
endif()

file(MAKE_DIRECTORY "${vss_probe_scratch}")

set(vss_package_version_cases
  "0.1.0|EXACT|TRUE"
  "0.1||TRUE"
  "0.0.9||FALSE"
  "0.1.1|EXACT|FALSE"
  "0.2.0||FALSE")

foreach(test_case IN LISTS vss_package_version_cases)
  string(REPLACE "|" ";" fields "${test_case}")
  list(GET fields 0 requested_version)
  list(GET fields 1 exact_argument)
  list(GET fields 2 expected_success)

  set(probe_source "${vss_probe_scratch}/${requested_version}-${expected_success}-src")
  set(probe_build "${vss_probe_scratch}/${requested_version}-${expected_success}-build")
  file(REMOVE_RECURSE "${probe_source}" "${probe_build}")
  file(MAKE_DIRECTORY "${probe_source}")
  file(WRITE "${probe_source}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 3.28)\n"
    "project(VirtualSensorMathVersionProbe NONE)\n"
    "find_package(VirtualSensorMath ${requested_version} ${exact_argument} CONFIG REQUIRED)\n")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${probe_source}" -B "${probe_build}"
            "-DVirtualSensorMath_DIR=${vss_package_config_dir}"
            "-DCMAKE_PREFIX_PATH=${vss_install}"
            -DCMAKE_FIND_USE_PACKAGE_REGISTRY=FALSE
            -DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=FALSE
            -DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=FALSE
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_stdout
    ERROR_VARIABLE configure_stderr)

  if("${configure_result}" STREQUAL "0")
    set(configured TRUE)
  else()
    set(configured FALSE)
  endif()

  if(expected_success AND NOT configured)
    message(FATAL_ERROR
      "package request ${requested_version} ${exact_argument} should configure, but failed:\n"
      "${configure_stdout}\n${configure_stderr}")
  elseif(NOT expected_success AND configured)
    message(FATAL_ERROR
      "package request ${requested_version} ${exact_argument} unexpectedly configured successfully")
  endif()
  message(STATUS
    "package request ${requested_version} ${exact_argument}: ${expected_success}")
endforeach()
