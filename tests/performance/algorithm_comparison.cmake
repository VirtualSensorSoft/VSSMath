cmake_minimum_required(VERSION 3.28)
include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_decimal.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/algorithm_record.cmake")

function(_vss_algorithm_comparison_schema_v1 path out_is_v1)
  set(is_v1 FALSE)
  vss_algorithm_regular_file("${path}" is_regular file_reason)
  if(is_regular)
    file(SIZE "${path}" size)
    if(size LESS_EQUAL 1048576)
      file(READ "${path}" json)
      string(JSON root_type ERROR_VARIABLE json_error TYPE "${json}")
      if("${json_error}" STREQUAL "NOTFOUND" AND root_type STREQUAL "OBJECT")
        string(JSON version_type ERROR_VARIABLE version_error TYPE "${json}"
          schema_version)
        if("${version_error}" STREQUAL "NOTFOUND" AND
           version_type STREQUAL "NUMBER")
          string(JSON version GET "${json}" schema_version)
          if("${version}" STREQUAL "1")
            set(is_v1 TRUE)
          endif()
        endif()
      endif()
    endif()
  endif()
  set("${out_is_v1}" "${is_v1}" PARENT_SCOPE)
endfunction()

function(_vss_algorithm_comparison_load path require_path_match out_json
    out_token out_reason)
  _vss_algorithm_comparison_schema_v1("${path}" is_v1)
  if(is_v1)
    set("${out_json}" "" PARENT_SCOPE)
    set("${out_token}" incompatible PARENT_SCOPE)
    set("${out_reason}" "schema-v1 record is incompatible with schema-v2" PARENT_SCOPE)
    return()
  endif()
  vss_algorithm_record_load("${path}" "${require_path_match}" json token reason)
  set("${out_json}" "${json}" PARENT_SCOPE)
  set("${out_token}" "${token}" PARENT_SCOPE)
  set("${out_reason}" "${reason}" PARENT_SCOPE)
endfunction()

function(_vss_algorithm_comparison_product value factor_1 factor_2 out_value)
  vss_algorithm_decimal_multiply_small("${value}" "${factor_1}" partial)
  vss_algorithm_decimal_multiply_small("${partial}" "${factor_2}" product)
  set("${out_value}" "${product}" PARENT_SCOPE)
endfunction()

macro(_vss_algorithm_comparison_fail out_token out_reason token reason)
  set("${out_token}" "${token}" PARENT_SCOPE)
  set("${out_reason}" "${reason}" PARENT_SCOPE)
endmacro()

function(_vss_algorithm_comparison_quality json label out_token out_reason)
  _vss_algorithm_record_quality_validated("${json}" token reason)
  if(NOT token STREQUAL "stable_pass")
    set("${out_token}" "${token}" PARENT_SCOPE)
    set("${out_reason}" "${label}: ${reason}" PARENT_SCOPE)
    return()
  endif()
  set("${out_token}" stable_pass PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(_vss_algorithm_comparison_compatible left_json right_json
    qualification label out_token out_reason)
  _vss_algorithm_record_compatible_validated("${left_json}" "${right_json}"
    "${qualification}" token reason)
  if(token STREQUAL "compatible")
    string(JSON left_method GET "${left_json}" method)
    string(JSON right_method GET "${right_json}" method)
    if(NOT left_method STREQUAL right_method)
      set(token incompatible)
      set(reason "method differs")
    endif()
  endif()
  if(NOT token STREQUAL "compatible")
    set("${out_token}" "${token}" PARENT_SCOPE)
    set("${out_reason}" "${label}: ${reason}" PARENT_SCOPE)
    return()
  endif()
  set("${out_token}" compatible PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(vss_algorithm_regression baseline_path current_path out_token
    out_reason)
  _vss_algorithm_comparison_load("${baseline_path}" TRUE baseline_json token
    reason)
  if(NOT token STREQUAL "record_valid")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "baseline: ${reason}")
    return()
  endif()
  _vss_algorithm_comparison_load("${current_path}" FALSE current_json token
    reason)
  if(NOT token STREQUAL "record_valid")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "current: ${reason}")
    return()
  endif()

  _vss_algorithm_comparison_compatible("${baseline_json}" "${current_json}"
    FALSE "records" token reason)
  if(NOT token STREQUAL "compatible")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" incompatible
      "${reason}")
    return()
  endif()

  _vss_algorithm_comparison_quality("${baseline_json}" baseline token reason)
  if(NOT token STREQUAL "stable_pass")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "${reason}")
    return()
  endif()
  _vss_algorithm_comparison_quality("${current_json}" current token reason)
  if(NOT token STREQUAL "stable_pass")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "${reason}")
    return()
  endif()

  # Allocation regressions have precedence over all timing failures.
  foreach(index RANGE 0 35)
    string(JSON baseline_case GET "${baseline_json}" cases "${index}")
    string(JSON current_case GET "${current_json}" cases "${index}")
    string(JSON case_id GET "${baseline_case}" case_id)
    string(JSON baseline_allocations GET "${baseline_case}"
      allocations_per_operation)
    string(JSON current_allocations GET "${current_case}"
      allocations_per_operation)
    vss_algorithm_decimal_compare("${current_allocations}"
      "${baseline_allocations}" allocation_comparison)
    if(allocation_comparison GREATER 0)
      _vss_algorithm_comparison_fail("${out_token}" "${out_reason}"
        allocation_regression
        "case ${index} ${case_id} allocations_per_operation baseline=${baseline_allocations} current=${current_allocations}")
      return()
    endif()
  endforeach()

  foreach(index RANGE 0 35)
    string(JSON baseline_case GET "${baseline_json}" cases "${index}")
    string(JSON current_case GET "${current_json}" cases "${index}")
    string(JSON case_id GET "${baseline_case}" case_id)
    string(JSON baseline_median GET "${baseline_case}" median_batch_ns)
    string(JSON current_median GET "${current_case}" median_batch_ns)
    string(JSON baseline_iterations GET "${baseline_case}"
      iterations_per_sample)
    string(JSON current_iterations GET "${current_case}"
      iterations_per_sample)
    _vss_algorithm_comparison_product("${current_median}"
      "${baseline_iterations}" 100 left_product)
    _vss_algorithm_comparison_product("${baseline_median}"
      "${current_iterations}" 120 right_product)
    vss_algorithm_decimal_compare("${left_product}" "${right_product}"
      timing_comparison)
    if(timing_comparison GREATER 0)
      _vss_algorithm_comparison_fail("${out_token}" "${out_reason}"
        timing_regression
        "case ${index} ${case_id} current_median=${current_median} current_iterations=${current_iterations} baseline_median=${baseline_median} baseline_iterations=${baseline_iterations}; current*baseline_iterations*100=${left_product} > baseline*current_iterations*120=${right_product}")
      return()
    endif()
  endforeach()

  set("${out_token}" compatible_pass PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(vss_algorithm_repeatability candidate_path verification_1_path
    verification_2_path out_token out_reason)
  _vss_algorithm_comparison_load("${candidate_path}" FALSE candidate_json
    token reason)
  if(NOT token STREQUAL "record_valid")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "candidate: ${reason}")
    return()
  endif()
  _vss_algorithm_comparison_load("${verification_1_path}" FALSE
    verification_1_json token reason)
  if(NOT token STREQUAL "record_valid")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "verification_1: ${reason}")
    return()
  endif()
  _vss_algorithm_comparison_load("${verification_2_path}" FALSE
    verification_2_json token reason)
  if(NOT token STREQUAL "record_valid")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
      "verification_2: ${reason}")
    return()
  endif()

  _vss_algorithm_comparison_compatible("${candidate_json}"
    "${verification_1_json}" TRUE "candidate and verification_1" token reason)
  if(NOT token STREQUAL "compatible")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" incompatible
      "${reason}")
    return()
  endif()
  _vss_algorithm_comparison_compatible("${candidate_json}"
    "${verification_2_json}" TRUE "candidate and verification_2" token reason)
  if(NOT token STREQUAL "compatible")
    _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" incompatible
      "${reason}")
    return()
  endif()

  foreach(label_json IN ITEMS candidate_json verification_1_json
      verification_2_json)
    if(label_json STREQUAL "candidate_json")
      set(label candidate)
    elseif(label_json STREQUAL "verification_1_json")
      set(label verification_1)
    else()
      set(label verification_2)
    endif()
    _vss_algorithm_comparison_quality("${${label_json}}" "${label}" token
      reason)
    if(NOT token STREQUAL "stable_pass")
      _vss_algorithm_comparison_fail("${out_token}" "${out_reason}" "${token}"
        "${reason}")
      return()
    endif()
  endforeach()

  # All allocation cases are checked before any timing case.
  foreach(index RANGE 0 35)
    foreach(label_json IN ITEMS candidate_json verification_1_json
        verification_2_json)
      string(JSON item GET "${${label_json}}" cases "${index}")
      string(JSON allocations GET "${item}" allocations_per_operation)
      if(label_json STREQUAL "candidate_json")
        set(candidate_allocations "${allocations}")
      elseif(label_json STREQUAL "verification_1_json")
        set(verification_1_allocations "${allocations}")
      else()
        set(verification_2_allocations "${allocations}")
      endif()
    endforeach()
    string(JSON case_id GET "${candidate_json}" cases "${index}" case_id)
    if(NOT candidate_allocations STREQUAL verification_1_allocations OR
       NOT candidate_allocations STREQUAL verification_2_allocations)
      _vss_algorithm_comparison_fail("${out_token}" "${out_reason}"
        allocation_inconsistent
        "case ${index} ${case_id} allocations_per_operation candidate=${candidate_allocations} verification_1=${verification_1_allocations} verification_2=${verification_2_allocations}")
      return()
    endif()
  endforeach()

  foreach(index RANGE 0 35)
    foreach(label_json IN ITEMS candidate_json verification_1_json
        verification_2_json)
      string(JSON item GET "${${label_json}}" cases "${index}")
      string(JSON median GET "${item}" median_batch_ns)
      if(label_json STREQUAL "candidate_json")
        set(candidate_median "${median}")
      elseif(label_json STREQUAL "verification_1_json")
        set(verification_1_median "${median}")
      else()
        set(verification_2_median "${median}")
      endif()
    endforeach()
    string(JSON case_id GET "${candidate_json}" cases "${index}" case_id)
    set(smallest "${candidate_median}")
    set(largest "${candidate_median}")
    foreach(value IN ITEMS verification_1_median verification_2_median)
      vss_algorithm_decimal_compare("${${value}}" "${smallest}" comparison)
      if(comparison LESS 0)
        set(smallest "${${value}}")
      endif()
      vss_algorithm_decimal_compare("${${value}}" "${largest}" comparison)
      if(comparison GREATER 0)
        set(largest "${${value}}")
      endif()
    endforeach()
    vss_algorithm_decimal_multiply_small("${largest}" 100 left_product)
    vss_algorithm_decimal_multiply_small("${smallest}" 120 right_product)
    vss_algorithm_decimal_compare("${left_product}" "${right_product}"
      timing_comparison)
    if(timing_comparison GREATER 0)
      _vss_algorithm_comparison_fail("${out_token}" "${out_reason}"
        nonrepeatable_timing
        "case ${index} ${case_id} candidate_median=${candidate_median} verification_1_median=${verification_1_median} verification_2_median=${verification_2_median}; largest*100=${left_product} > smallest*120=${right_product}")
      return()
    endif()
  endforeach()

  set("${out_token}" repeatable_pass PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()

function(vss_algorithm_write_result path token out_ok out_reason)
  if("${path}" STREQUAL "")
    set("${out_ok}" FALSE PARENT_SCOPE)
    set("${out_reason}" "result path is empty" PARENT_SCOPE)
    return()
  endif()
  get_filename_component(parent "${path}" DIRECTORY)
  if(NOT "${parent}" STREQUAL "")
    file(MAKE_DIRECTORY "${parent}")
  endif()
  file(WRITE "${path}" "${token}\n")
  set("${out_ok}" TRUE PARENT_SCOPE)
  set("${out_reason}" "" PARENT_SCOPE)
endfunction()
