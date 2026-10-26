cmake_minimum_required(VERSION 3.28)

function(vss_math_normalize_value input output)
  set(value "${input}")
  string(REPLACE "\r" " " value "${value}")
  string(REPLACE "\n" " " value "${value}")
  string(REPLACE "\t" " " value "${value}")
  string(REGEX REPLACE " +" " " value "${value}")
  string(STRIP "${value}" value)
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(vss_math_redact_value input output)
  set(value "${input}")
  # Replace the most-specific paths first so nested HOME/build paths retain
  # their dedicated placeholders instead of being consumed by an enclosing
  # repository replacement.
  if(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    string(REPLACE "$ENV{HOME}" "<runner-home>" value "${value}")
  endif()
  if(DEFINED VSS_EVIDENCE_BUILD_DIR AND
     NOT "${VSS_EVIDENCE_BUILD_DIR}" STREQUAL "")
    string(REPLACE "${VSS_EVIDENCE_BUILD_DIR}" "<build>" value
      "${value}")
  endif()
  if(DEFINED VSS_EVIDENCE_SOURCE_DIR AND
     NOT "${VSS_EVIDENCE_SOURCE_DIR}" STREQUAL "")
    string(REPLACE "${VSS_EVIDENCE_SOURCE_DIR}" "<repository>" value
      "${value}")
  endif()
  set(${output} "${value}" PARENT_SCOPE)
endfunction()

function(vss_math_capture_command label)
  set(command ${ARGN})
  execute_process(
    COMMAND ${command}
    RESULT_VARIABLE command_result
    OUTPUT_VARIABLE command_output
    ERROR_VARIABLE command_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT command_result EQUAL 0)
    message(FATAL_ERROR
      "Unable to execute ${label} version command (exit ${command_result}):\n"
      "${command_error}\n${command_output}")
  endif()
  string(CONCAT combined_output "${command_output}" "\n" "${command_error}")
  vss_math_normalize_value("${combined_output}" normalized_output)
  if("${normalized_output}" STREQUAL "")
    message(FATAL_ERROR
      "${label} version command produced no output")
  endif()
  vss_math_redact_value("${normalized_output}" redacted_output)
  set(vss_math_captured_value "${redacted_output}" PARENT_SCOPE)
endfunction()

function(vss_math_collect_tool_versions)
  set(required_fields
    OUTPUT SOURCE_DIR BUILD_DIR COMPILER COMPILER_ID COMPILER_VERSION
    GENERATOR BUILD_TYPE HOST_SYSTEM HOST_PROCESSOR)
  foreach(field IN LISTS required_fields)
    if(NOT DEFINED VSS_EVIDENCE_${field} OR
       "${VSS_EVIDENCE_${field}}" STREQUAL "")
      message(FATAL_ERROR "VSS_EVIDENCE_${field} is required")
    endif()
  endforeach()

  execute_process(
    COMMAND git -C "${VSS_EVIDENCE_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE git_result
    OUTPUT_VARIABLE git_output
    ERROR_VARIABLE git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT git_result EQUAL 0)
    message(FATAL_ERROR
      "Unable to determine commit from VSS_EVIDENCE_SOURCE_DIR:\n"
      "${git_error}")
  endif()
  vss_math_normalize_value("${git_output}" commit)
  if("${commit}" STREQUAL "")
    message(FATAL_ERROR "The commit value is empty")
  endif()

  set(os_value "")
  if(EXISTS "/etc/os-release")
    file(STRINGS "/etc/os-release" os_lines
      REGEX "^PRETTY_NAME=" LIMIT_COUNT 1)
    if(os_lines)
      list(GET os_lines 0 os_line)
      string(REGEX REPLACE "^PRETTY_NAME=\\\"(.*)\\\"$" "\\1"
        os_value "${os_line}")
    endif()
  endif()
  if("${os_value}" STREQUAL "")
    set(os_value "${VSS_EVIDENCE_HOST_SYSTEM}")
  endif()
  vss_math_normalize_value("${os_value}" os_value)

  vss_math_capture_command("CMake" "${CMAKE_COMMAND}" --version)
  set(cmake_value "${vss_math_captured_value}")
  vss_math_normalize_value("${VSS_EVIDENCE_COMPILER}" compiler_value)
  vss_math_redact_value("${compiler_value}" compiler_value)
  vss_math_normalize_value("${VSS_EVIDENCE_COMPILER_VERSION}"
    compiler_version_value)
  vss_math_normalize_value("${VSS_EVIDENCE_GENERATOR}" generator_value)
  vss_math_normalize_value("${VSS_EVIDENCE_BUILD_TYPE}" build_type_value)
  vss_math_normalize_value("${VSS_EVIDENCE_HOST_PROCESSOR}" architecture_value)
  if(DEFINED VSS_EVIDENCE_SANITIZERS AND
     NOT "${VSS_EVIDENCE_SANITIZERS}" STREQUAL "")
    vss_math_normalize_value("${VSS_EVIDENCE_SANITIZERS}" sanitizers_value)
  else()
    set(sanitizers_value "OFF")
  endif()

  set(known_tools doxygen graphviz clang_tidy clang_format gcovr binutils trivy)
  set(requested_tools)
  if(DEFINED VSS_EVIDENCE_TOOLS AND
     NOT "${VSS_EVIDENCE_TOOLS}" STREQUAL "")
    set(requested_tools ${VSS_EVIDENCE_TOOLS})
  endif()
  foreach(tool_name IN LISTS requested_tools)
    list(FIND known_tools tool_index "${tool_name}")
    if(tool_index EQUAL -1)
      message(FATAL_ERROR "Unsupported VSS_EVIDENCE_TOOLS entry: ${tool_name}")
    endif()
  endforeach()

  foreach(tool_name IN LISTS known_tools)
    set(tool_value "not-executed-in-this-job")
    list(FIND requested_tools "${tool_name}" tool_index)
    if(NOT tool_index EQUAL -1)
      if(tool_name STREQUAL "doxygen")
        vss_math_capture_command("doxygen" doxygen --version)
      elseif(tool_name STREQUAL "graphviz")
        vss_math_capture_command("graphviz" dot -V)
      elseif(tool_name STREQUAL "clang_tidy")
        vss_math_capture_command("clang_tidy" clang-tidy-18 --version)
      elseif(tool_name STREQUAL "clang_format")
        vss_math_capture_command("clang_format" clang-format-18 --version)
      elseif(tool_name STREQUAL "gcovr")
        vss_math_capture_command("gcovr" gcovr --version)
      elseif(tool_name STREQUAL "binutils")
        vss_math_capture_command("binutils" strings --version)
      elseif(tool_name STREQUAL "trivy")
        vss_math_capture_command("trivy" trivy --version)
      endif()
      set(tool_value "${vss_math_captured_value}")
    endif()
    set(vss_math_tool_${tool_name} "${tool_value}")
  endforeach()

  get_filename_component(output_directory "${VSS_EVIDENCE_OUTPUT}" DIRECTORY)
  if(NOT "${output_directory}" STREQUAL "")
    file(MAKE_DIRECTORY "${output_directory}")
  endif()
  vss_math_redact_value("${commit}" commit)
  vss_math_redact_value("${os_value}" os_value)
  file(WRITE "${VSS_EVIDENCE_OUTPUT}"
    "commit=${commit}\n"
    "os=${os_value}\n"
    "architecture=${architecture_value}\n"
    "cmake=${cmake_value}\n"
    "compiler=${compiler_value}\n"
    "compiler_version=${compiler_version_value}\n"
    "generator=${generator_value}\n"
    "build_type=${build_type_value}\n"
    "doxygen=${vss_math_tool_doxygen}\n"
    "graphviz=${vss_math_tool_graphviz}\n"
    "clang_tidy=${vss_math_tool_clang_tidy}\n"
    "clang_format=${vss_math_tool_clang_format}\n"
    "gcovr=${vss_math_tool_gcovr}\n"
    "binutils=${vss_math_tool_binutils}\n"
    "trivy=${vss_math_tool_trivy}\n"
    "sanitizers=${sanitizers_value}\n")
endfunction()

if(DEFINED CMAKE_SCRIPT_MODE_FILE)
  vss_math_collect_tool_versions()
else()
  function(vss_math_add_tool_version_target)
    set(VSS_MATH_EVIDENCE_TOOLS "" CACHE STRING
      "Semicolon-separated tools executed in this job")
    mark_as_advanced(VSS_MATH_EVIDENCE_TOOLS)
    if(TARGET math-tool-versions)
      return()
    endif()
    set(vss_math_evidence_output
      "${PROJECT_BINARY_DIR}/acceptance/tool-versions.txt")
    add_custom_target(math-tool-versions
      COMMAND "${CMAKE_COMMAND}"
        "-DVSS_EVIDENCE_OUTPUT=${vss_math_evidence_output}"
        "-DVSS_EVIDENCE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
        "-DVSS_EVIDENCE_BUILD_DIR=${PROJECT_BINARY_DIR}"
        "-DVSS_EVIDENCE_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DVSS_EVIDENCE_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}"
        "-DVSS_EVIDENCE_COMPILER_VERSION=${CMAKE_CXX_COMPILER_VERSION}"
        "-DVSS_EVIDENCE_GENERATOR=${CMAKE_GENERATOR}"
        "-DVSS_EVIDENCE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DVSS_EVIDENCE_HOST_SYSTEM=${CMAKE_HOST_SYSTEM_NAME}"
        "-DVSS_EVIDENCE_HOST_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}"
        "-DVSS_EVIDENCE_SANITIZERS=${VSS_MATH_ENABLE_SANITIZERS}"
        "-DVSS_EVIDENCE_TOOLS=${VSS_MATH_EVIDENCE_TOOLS}"
        -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CollectToolVersions.cmake"
      BYPRODUCTS "${vss_math_evidence_output}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      COMMENT "Collecting VirtualSensorMath tool-version evidence"
      VERBATIM)
  endfunction()
endif()
