include_guard(GLOBAL)

function(vss_math_add_format_check)
  if(NOT VSS_MATH_ENABLE_FORMAT_CHECK)
    return()
  endif()

  find_program(vss_clang_format NAMES clang-format-18 REQUIRED)
  execute_process(
    COMMAND "${vss_clang_format}" --version
    RESULT_VARIABLE vss_clang_format_version_result
    OUTPUT_VARIABLE vss_clang_format_version_output
    ERROR_VARIABLE vss_clang_format_version_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  string(CONCAT vss_clang_format_version
    "${vss_clang_format_version_output}"
    "${vss_clang_format_version_error}")
  if(NOT vss_clang_format_version_result EQUAL 0 OR
     NOT vss_clang_format_version MATCHES "version[ \\t]+18\\.")
    message(FATAL_ERROR
      "VirtualSensorMath requires clang-format major 18; got:\n"
      "${vss_clang_format_version}")
  endif()

  file(GLOB_RECURSE vss_format_files
    CONFIGURE_DEPENDS LIST_DIRECTORIES false
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
    "${PROJECT_SOURCE_DIR}/include/*.hpp.in"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/tests/*.hpp"
    "${PROJECT_SOURCE_DIR}/tests/*.cpp"
    "${PROJECT_SOURCE_DIR}/benchmarks/*.hpp.in"
    "${PROJECT_SOURCE_DIR}/benchmarks/*.cpp")
  list(APPEND vss_format_files
    "${PROJECT_BINARY_DIR}/generated/vss/math/version.hpp")
  list(SORT vss_format_files)
  foreach(vss_format_file IN LISTS vss_format_files)
    if(NOT EXISTS "${vss_format_file}")
      message(FATAL_ERROR
        "format-check source inventory entry is missing: ${vss_format_file}")
    endif()
  endforeach()
  if(NOT vss_format_files)
    message(FATAL_ERROR "format-check source inventory is empty")
  endif()

  add_custom_target(format-check
    COMMAND "${vss_clang_format}" --dry-run --Werror ${vss_format_files}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Checking VirtualSensorMath C++ formatting"
    VERBATIM)
endfunction()
