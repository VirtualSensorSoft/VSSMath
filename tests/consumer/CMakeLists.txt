cmake_minimum_required(VERSION 3.28)
project(VirtualSensorMathConsumer LANGUAGES CXX)

find_package(VirtualSensorMath 0.1 CONFIG REQUIRED)

add_executable(math_consumer consumer.cpp)
target_compile_features(math_consumer PRIVATE cxx_std_20)
set_target_properties(math_consumer PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(math_consumer PRIVATE VirtualSensorSoft::math)

if(DEFINED VSS_EXPECTED_PREFIX)
  set(vss_expected_prefix "${VSS_EXPECTED_PREFIX}")
  cmake_path(IS_PREFIX vss_expected_prefix "${VirtualSensorMath_DIR}"
             NORMALIZE vss_math_from_expected_prefix)
  if(NOT vss_math_from_expected_prefix)
    message(FATAL_ERROR
      "VirtualSensorMath resolved outside VSS_EXPECTED_PREFIX: ${VirtualSensorMath_DIR}")
  endif()
endif()
