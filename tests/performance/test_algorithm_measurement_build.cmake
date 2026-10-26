cmake_minimum_required(VERSION 3.28)

if(NOT IS_DIRECTORY "${VSS_MATH_SOURCE_DIR}" OR
   "${VSS_MATH_MEASUREMENT_BUILD_SCRATCH}" STREQUAL "")
  message(FATAL_ERROR "Measurement build-policy test requires source and scratch paths")
endif()

function(configure_case name expected)
  set(build "${VSS_MATH_MEASUREMENT_BUILD_SCRATCH}/${name}")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${VSS_MATH_SOURCE_DIR}" -B "${build}"
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      -DVSS_MATH_BUILD_DOCS=OFF -DVSS_MATH_BUILD_BENCHMARKS=ON
      -DBUILD_TESTING=ON ${ARGN}
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(expected STREQUAL "reject")
    if(status EQUAL 0 OR NOT "${output}${error}" MATCHES
       "VSS-MATH-PERF measurement requires LTO-free benchmark compilation")
      message(FATAL_ERROR
        "Measurement build policy ${name}: explicit LTO was not rejected by the benchmark guard: ${output}${error}")
    endif()
  elseif(NOT status EQUAL 0)
    message(FATAL_ERROR "Measurement build policy ${name}: ${output}${error}")
  endif()
  set(case_build "${build}" PARENT_SCOPE)
endfunction()

# Each case has its own reusable named directory under the caller's build tree.
configure_case(global_lto reject "-DCMAKE_CXX_FLAGS=-flto")
configure_case(config_lto reject "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -flto=auto")
configure_case(link_lto reject "-DCMAKE_EXE_LINKER_FLAGS=-flto")
configure_case(config_link_lto reject "-DCMAKE_EXE_LINKER_FLAGS_RELEASE=-flto")
configure_case(quoted_lto reject "-DCMAKE_CXX_FLAGS=\"-flto\"")

configure_case(ipo_overridden allow
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE=ON)
file(READ "${case_build}/compile_commands.json" commands)
string(JSON count LENGTH "${commands}")
math(EXPR last "${count} - 1")
set(checked 0)
foreach(index RANGE 0 ${last})
  string(JSON source GET "${commands}" ${index} file)
  if(source MATCHES "/(benchmark_algorithms|BenchmarkMeasurement)\\.cpp$")
    string(JSON command GET "${commands}" ${index} command)
    if(command MATCHES "vss_math_algorithm_benchmark.dir" AND
       command MATCHES "(^|[ \t])-flto([= \t]|$)")
      message(FATAL_ERROR "IPO remained enabled in measured code: ${command}")
    endif()
    string(FIND "${command}" "vss_math_algorithm_benchmark.dir" target_index)
    if(NOT target_index EQUAL -1)
      math(EXPR checked "${checked} + 1")
    endif()
  endif()
endforeach()
if(NOT checked EQUAL 2)
  message(FATAL_ERROR "Measurement build policy did not inspect both measured translation units")
endif()

# The measurement restriction does not prohibit IPO for ordinary library builds.
configure_case(library_only allow -DVSS_MATH_BUILD_BENCHMARKS=OFF
  -DBUILD_TESTING=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
  "-DCMAKE_CXX_FLAGS=-flto")
message(STATUS "VSS-MATH-PERF measurement build policy passed")
