include_guard(GLOBAL)

function(vss_math_enable_coverage target)
  if(NOT VSS_MATH_ENABLE_COVERAGE)
    return()
  endif()

  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(FATAL_ERROR
      "VSS_MATH_ENABLE_COVERAGE requires the GNU C++ compiler; got "
      "${CMAKE_CXX_COMPILER_ID}")
  endif()
  if(VSS_MATH_ENABLE_SANITIZERS)
    message(FATAL_ERROR
      "VSS_MATH_ENABLE_COVERAGE cannot be combined with "
      "VSS_MATH_ENABLE_SANITIZERS")
  endif()

  set(VSS_MATH_GCOVR_EXECUTABLE "" CACHE FILEPATH
    "Verified gcovr executable used by coverage-report")
  if(NOT VSS_MATH_GCOVR_EXECUTABLE)
    find_program(vss_math_gcovr_program NAMES gcovr
      DOC "Verified gcovr executable used by coverage-report")
    if(vss_math_gcovr_program)
      set(VSS_MATH_GCOVR_EXECUTABLE "${vss_math_gcovr_program}" CACHE FILEPATH
        "Verified gcovr executable used by coverage-report" FORCE)
    endif()
  endif()
  if(NOT VSS_MATH_GCOVR_EXECUTABLE)
    message(FATAL_ERROR
      "VSS_MATH_ENABLE_COVERAGE requires the gcovr executable")
  endif()

  # GCC otherwise stores source paths relative to the build working directory.
  # Absolute paths let gcovr resolve them even when compilation uses ccache.
  target_compile_options(${target} PRIVATE --coverage -O0 -g -fprofile-abs-path)
  target_link_options(${target} PRIVATE --coverage)

  if(TARGET coverage-report)
    return()
  endif()
  set(vss_math_coverage_directory
    "${PROJECT_BINARY_DIR}/coverage")
  add_custom_target(coverage-report
    COMMAND "${CMAKE_COMMAND}" -E make_directory
      "${vss_math_coverage_directory}"
    COMMAND "${VSS_MATH_GCOVR_EXECUTABLE}"
      --root "${PROJECT_SOURCE_DIR}"
      --filter "${PROJECT_SOURCE_DIR}/include"
      --filter "${PROJECT_SOURCE_DIR}/src"
      --object-directory "${PROJECT_BINARY_DIR}"
      --txt "${vss_math_coverage_directory}/summary.txt"
      --xml "${vss_math_coverage_directory}/coverage.xml"
      --xml-pretty
      --html-details "${vss_math_coverage_directory}/index.html"
    BYPRODUCTS
      "${vss_math_coverage_directory}/summary.txt"
      "${vss_math_coverage_directory}/coverage.xml"
      "${vss_math_coverage_directory}/index.html"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Generating VirtualSensorMath coverage reports"
    VERBATIM)
endfunction()
