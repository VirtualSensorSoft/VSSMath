function(vss_math_enable_warnings target)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR
      "VirtualSensorMath warning policy requires GNU or Clang; got "
      "${CMAKE_CXX_COMPILER_ID}")
  endif()

  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Werror)
endfunction()

function(vss_math_enable_sanitizers target)
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR
      "VirtualSensorMath sanitizers require GNU or Clang; got "
      "${CMAKE_CXX_COMPILER_ID}")
  endif()

  target_compile_options(${target} PRIVATE
    -fsanitize=address,undefined
    -fno-omit-frame-pointer)
  target_link_options(${target} PRIVATE
    -fsanitize=address,undefined
    -fno-omit-frame-pointer)
endfunction()
