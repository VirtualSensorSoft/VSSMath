include_guard(GLOBAL)

# Per-configuration IPO properties override the generic property. Disable both
# so a caller's CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE cannot fuse callbacks
# with the separately compiled measurement loop.
function(vss_math_disable_benchmark_ipo target)
  set_property(TARGET "${target}" PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set(configurations Debug Release RelWithDebInfo MinSizeRel
    ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
  list(REMOVE_DUPLICATES configurations)
  foreach(configuration IN LISTS configurations)
    string(TOUPPER "${configuration}" upper)
    set_property(TARGET "${target}" PROPERTY
      "INTERPROCEDURAL_OPTIMIZATION_${upper}" FALSE)
  endforeach()
endfunction()

# IPO properties cannot remove explicitly supplied compiler/linker flags.
# Fail closed only for the measured executable, not ordinary package builds.
function(vss_math_enforce_benchmark_isolation target)
  vss_math_disable_benchmark_ipo("${target}")
  set(flag_variables CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS)
  set(configurations ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
  foreach(configuration IN LISTS configurations)
    string(TOUPPER "${configuration}" upper)
    list(APPEND flag_variables
      "CMAKE_CXX_FLAGS_${upper}" "CMAKE_EXE_LINKER_FLAGS_${upper}")
  endforeach()
  foreach(variable IN LISTS flag_variables)
    separate_arguments(flags NATIVE_COMMAND "${${variable}}")
    foreach(flag IN LISTS flags)
      if(flag MATCHES "^(-flto(=.*)?|[-/]GL|[-/]LTCG(:.*)?)$")
        message(FATAL_ERROR
          "VSS-MATH-PERF measurement requires LTO-free benchmark compilation: ${variable} contains explicit LTO flags")
      endif()
    endforeach()
  endforeach()
endfunction()
