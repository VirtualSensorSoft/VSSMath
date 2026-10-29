cmake_minimum_required(VERSION 3.28)
project(VirtualSensorMathConsumer LANGUAGES CXX)

find_package(VirtualSensorMath 0.1.1 EXACT CONFIG REQUIRED)

add_executable(math_consumer consumer.cpp)
target_compile_features(math_consumer PRIVATE cxx_std_20)
set_target_properties(math_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(math_consumer PRIVATE VirtualSensorSoft::math)

string(CONCAT vss_identity_content
  "package_version=${VirtualSensorMath_VERSION}\n"
  "config_dir=${VirtualSensorMath_DIR}\n")

if(DEFINED VSS_EXPECTED_PREFIX)
  if(NOT IS_DIRECTORY "${VSS_EXPECTED_PREFIX}")
    message(FATAL_ERROR
      "VSS_EXPECTED_PREFIX is not an existing directory: ${VSS_EXPECTED_PREFIX}")
  endif()
  file(REAL_PATH "${VSS_EXPECTED_PREFIX}" vss_expected_prefix_real)
  file(REAL_PATH "${VirtualSensorMath_DIR}" vss_package_config_real)
  cmake_path(IS_PREFIX vss_expected_prefix_real "${vss_package_config_real}"
             NORMALIZE vss_config_from_expected_prefix)
  if(NOT vss_config_from_expected_prefix)
    message(FATAL_ERROR
      "VirtualSensorMath package configuration resolves outside VSS_EXPECTED_PREFIX: "
      "${VirtualSensorMath_DIR} -> ${vss_package_config_real}")
  endif()

  function(vss_require_target_origin kind path base_dir require_directory)
    if("${path}" STREQUAL "" OR "${path}" MATCHES "\\$<")
      message(FATAL_ERROR
        "VSSMath ${kind} path is empty or contains an unresolved generator expression: ${path}")
    endif()
    if(IS_ABSOLUTE "${path}")
      set(candidate "${path}")
    else()
      set(candidate "${base_dir}/${path}")
    endif()
    if(NOT EXISTS "${candidate}")
      message(FATAL_ERROR "VSSMath ${kind} path does not exist: ${candidate}")
    endif()
    if(require_directory AND NOT IS_DIRECTORY "${candidate}")
      message(FATAL_ERROR "VSSMath ${kind} path is not a directory: ${candidate}")
    endif()
    if(NOT require_directory AND IS_DIRECTORY "${candidate}")
      message(FATAL_ERROR "VSSMath ${kind} path is not a file: ${candidate}")
    endif()
    file(REAL_PATH "${candidate}" resolved)
    cmake_path(IS_PREFIX vss_expected_prefix_real "${resolved}" NORMALIZE inside)
    if(NOT inside)
      if(kind STREQUAL "INTERFACE_INCLUDE_DIRECTORIES")
        message(FATAL_ERROR
          "VSSMath INTERFACE_INCLUDE_DIRECTORIES path resolves outside VSS_EXPECTED_PREFIX: "
          "${candidate} -> ${resolved}")
      elseif(kind STREQUAL "imported library location")
        message(FATAL_ERROR
          "VSSMath imported library location resolves outside VSS_EXPECTED_PREFIX: "
          "${candidate} -> ${resolved}")
      else()
        message(FATAL_ERROR
          "VSSMath ${kind} path resolves outside VSS_EXPECTED_PREFIX: "
          "${candidate} -> ${resolved}")
      endif()
    endif()
    set(vss_validated_origin "${resolved}" PARENT_SCOPE)
  endfunction()

  get_target_property(vss_include_directories VirtualSensorSoft::math
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT vss_include_directories OR
     vss_include_directories MATCHES "-NOTFOUND$")
    message(FATAL_ERROR
      "VirtualSensorSoft::math has no INTERFACE_INCLUDE_DIRECTORIES")
  endif()
  set(vss_resolved_include_directories "")
  foreach(vss_include IN LISTS vss_include_directories)
    vss_require_target_origin("INTERFACE_INCLUDE_DIRECTORIES" "${vss_include}"
      "${VirtualSensorMath_DIR}" TRUE)
    list(APPEND vss_resolved_include_directories "${vss_validated_origin}")
  endforeach()

  get_target_property(vss_imported_configurations VirtualSensorSoft::math
    IMPORTED_CONFIGURATIONS)
  set(vss_location_configurations
    ${vss_imported_configurations} Debug Release RelWithDebInfo MinSizeRel NOCONFIG)
  if(CMAKE_CONFIGURATION_TYPES)
    list(APPEND vss_location_configurations ${CMAKE_CONFIGURATION_TYPES})
  endif()
  if(CMAKE_BUILD_TYPE)
    list(APPEND vss_location_configurations "${CMAKE_BUILD_TYPE}")
  endif()
  list(REMOVE_DUPLICATES vss_location_configurations)
  set(vss_location_properties IMPORTED_LOCATION)
  foreach(vss_config IN LISTS vss_location_configurations)
    string(TOUPPER "${vss_config}" vss_config_upper)
    list(APPEND vss_location_properties "IMPORTED_LOCATION_${vss_config_upper}")
  endforeach()
  set(vss_imported_locations "")
  foreach(vss_property IN LISTS vss_location_properties)
    get_target_property(vss_location VirtualSensorSoft::math "${vss_property}")
    if(vss_location AND NOT vss_location MATCHES "-NOTFOUND$")
      list(APPEND vss_imported_locations ${vss_location})
    endif()
  endforeach()
  if(NOT vss_imported_locations)
    message(FATAL_ERROR
      "VirtualSensorSoft::math has no imported library location")
  endif()
  set(vss_resolved_imported_locations "")
  foreach(vss_location IN LISTS vss_imported_locations)
    vss_require_target_origin("imported library location" "${vss_location}"
      "${VirtualSensorMath_DIR}" FALSE)
    list(APPEND vss_resolved_imported_locations "${vss_validated_origin}")
  endforeach()

  list(REMOVE_DUPLICATES vss_resolved_include_directories)
  list(REMOVE_DUPLICATES vss_resolved_imported_locations)
  string(JOIN ";" vss_include_identity ${vss_resolved_include_directories})
  string(JOIN ";" vss_location_identity ${vss_resolved_imported_locations})
  string(APPEND vss_identity_content
    "expected_prefix=${vss_expected_prefix_real}\n"
    "config_dir_realpath=${vss_package_config_real}\n"
    "interface_include_directories=${vss_include_identity}\n"
    "imported_locations=${vss_location_identity}\n"
    "prefix_provenance=PASS\n")
endif()

if(DEFINED VSS_PACKAGE_IDENTITY_FILE AND NOT VSS_PACKAGE_IDENTITY_FILE STREQUAL "")
  file(WRITE "${VSS_PACKAGE_IDENTITY_FILE}" "${vss_identity_content}")
endif()
