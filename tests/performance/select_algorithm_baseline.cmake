cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")

foreach(required IN ITEMS VSS_PERF_BASELINE_DIR VSS_PERF_CURRENT
    VSS_PERF_SELECTED_FILE)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "VSSMath ${required} is required")
  endif()
endforeach()

get_filename_component(baseline_directory "${VSS_PERF_BASELINE_DIR}" ABSOLUTE)
get_filename_component(selected_file "${VSS_PERF_SELECTED_FILE}" ABSOLUTE)
if(NOT IS_DIRECTORY "${baseline_directory}" OR
   IS_SYMLINK "${baseline_directory}")
  message(FATAL_ERROR "VSSMath baseline directory must be a real directory")
endif()
if(IS_DIRECTORY "${selected_file}" OR IS_SYMLINK "${selected_file}")
  message(FATAL_ERROR "VSSMath selected path must be a regular build-tree file")
endif()

vss_algorithm_record_load("${VSS_PERF_CURRENT}" FALSE current_json
  current_token current_reason)
if(NOT current_token STREQUAL "record_valid")
  file(REMOVE "${selected_file}")
  message(FATAL_ERROR
    "VSSMath current record is not valid schema-v2: ${current_token}: ${current_reason}")
endif()

file(GLOB candidates LIST_DIRECTORIES FALSE "${baseline_directory}/*.json")
list(SORT candidates)
set(matches)
foreach(candidate IN LISTS candidates)
  vss_algorithm_record_load("${candidate}" TRUE baseline_json baseline_token
    baseline_reason)
  if(baseline_token STREQUAL "incompatible")
    # Historical schema-v1 records and valid records from another method are
    # ineligible; they are never interpreted using current schema-v2 rules.
    continue()
  elseif(NOT baseline_token STREQUAL "record_valid")
    file(REMOVE "${selected_file}")
    message(FATAL_ERROR
      "VSSMath malformed baseline ${candidate}: ${baseline_reason}")
  endif()
  _vss_algorithm_record_compatible_validated("${baseline_json}"
    "${current_json}" FALSE compatibility_token compatibility_reason)
  if(compatibility_token STREQUAL "compatible")
    list(APPEND matches "${candidate}")
  endif()
endforeach()

list(LENGTH matches match_count)
if(NOT match_count EQUAL 1)
  file(REMOVE "${selected_file}")
  message(FATAL_ERROR
    "VSSMath expected exactly one compatible schema-v2 baseline, found ${match_count}")
endif()

list(GET matches 0 selected)
get_filename_component(selected_directory "${selected_file}" DIRECTORY)
file(MAKE_DIRECTORY "${selected_directory}")
set(temporary "${selected_file}.tmp")
if(EXISTS "${temporary}" OR IS_SYMLINK "${temporary}")
  file(REMOVE "${selected_file}")
  message(FATAL_ERROR "VSSMath selected-path temporary already exists")
endif()
file(WRITE "${temporary}" "${selected}\n")
file(RENAME "${temporary}" "${selected_file}" RESULT rename_result)
if(NOT rename_result STREQUAL "0")
  file(REMOVE "${temporary}" "${selected_file}")
  message(FATAL_ERROR "VSSMath selected path publication failed: ${rename_result}")
endif()
