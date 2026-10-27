function(vss_math_enable_clang_tidy target)
  if(NOT VSS_MATH_ENABLE_CLANG_TIDY)
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR
      "VSS_MATH_ENABLE_CLANG_TIDY requires the Clang C++ compiler; got "
      "${CMAKE_CXX_COMPILER_ID}")
  endif()

  set(VSS_MATH_CLANG_TIDY_EXECUTABLE "" CACHE FILEPATH
    "Verified LLVM 18 clang-tidy executable")
  if(NOT VSS_MATH_CLANG_TIDY_EXECUTABLE)
    find_program(VSS_MATH_CLANG_TIDY_EXECUTABLE
      NAMES clang-tidy-18
      DOC "Verified LLVM 18 clang-tidy executable")
  endif()
  if(NOT VSS_MATH_CLANG_TIDY_EXECUTABLE)
    message(FATAL_ERROR
      "VSS_MATH_ENABLE_CLANG_TIDY requires clang-tidy-18. Set "
      "VSS_MATH_CLANG_TIDY_EXECUTABLE to a verified LLVM 18 executable.")
  endif()

  execute_process(
    COMMAND "${VSS_MATH_CLANG_TIDY_EXECUTABLE}" --version
    RESULT_VARIABLE vss_math_clang_tidy_version_result
    OUTPUT_VARIABLE vss_math_clang_tidy_version
    ERROR_VARIABLE vss_math_clang_tidy_version_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT vss_math_clang_tidy_version_result EQUAL 0)
    message(FATAL_ERROR
      "Unable to run the configured clang-tidy executable: "
      "${VSS_MATH_CLANG_TIDY_EXECUTABLE}\n"
      "${vss_math_clang_tidy_version_error}")
  endif()
  if(NOT vss_math_clang_tidy_version MATCHES "LLVM version[ \t]+18\\.")
    message(FATAL_ERROR
      "VSS_MATH_CLANG_TIDY_EXECUTABLE must report LLVM 18; got:\n"
      "${vss_math_clang_tidy_version}")
  endif()

  set(vss_math_clang_tidy_config "${CMAKE_SOURCE_DIR}/.clang-tidy")
  if(NOT EXISTS "${vss_math_clang_tidy_config}")
    message(FATAL_ERROR
      "The clang-tidy configuration is missing: ${vss_math_clang_tidy_config}")
  endif()
  execute_process(
    COMMAND "${VSS_MATH_CLANG_TIDY_EXECUTABLE}"
            --verify-config
            "--config-file=${vss_math_clang_tidy_config}"
    RESULT_VARIABLE vss_math_clang_tidy_config_result
    OUTPUT_VARIABLE vss_math_clang_tidy_config_output
    ERROR_VARIABLE vss_math_clang_tidy_config_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT vss_math_clang_tidy_config_result EQUAL 0)
    message(FATAL_ERROR
      "clang-tidy rejected ${vss_math_clang_tidy_config}:\n"
      "${vss_math_clang_tidy_config_output}\n"
      "${vss_math_clang_tidy_config_error}")
  endif()

  set(vss_math_header_filter
    "^${CMAKE_SOURCE_DIR}/(include|src|tests|benchmarks)/|^${CMAKE_BINARY_DIR}/generated/")
  set_property(TARGET "${target}" PROPERTY CXX_CLANG_TIDY
    "${VSS_MATH_CLANG_TIDY_EXECUTABLE}"
    "--config-file=${vss_math_clang_tidy_config}"
    "--header-filter=${vss_math_header_filter}")
endfunction()
