cmake_minimum_required(VERSION 3.28)

include("${CMAKE_CURRENT_LIST_DIR}/algorithm_decimal.cmake")

function(require_equal label expected actual)
  if(NOT "${expected}" STREQUAL "${actual}")
    message(FATAL_ERROR
      "VSSMath ${label}: expected '${expected}', got '${actual}'")
  endif()
endfunction()

if(DEFINED VSS_ALGORITHM_DECIMAL_INVALID_CASE)
  if(VSS_ALGORITHM_DECIMAL_INVALID_CASE STREQUAL "noncanonical")
    vss_algorithm_decimal_compare("01" "1" comparison)
  elseif(VSS_ALGORITHM_DECIMAL_INVALID_CASE STREQUAL "divide-zero")
    vss_algorithm_decimal_divide_small("241" "0" quotient remainder)
  elseif(VSS_ALGORITHM_DECIMAL_INVALID_CASE STREQUAL "subtract-underflow")
    vss_algorithm_decimal_subtract("1" "2" difference)
  elseif(VSS_ALGORITHM_DECIMAL_INVALID_CASE STREQUAL "factor-out-of-range")
    vss_algorithm_decimal_multiply_small("1" "1048577" product)
  else()
    message(FATAL_ERROR "Unknown decimal invalid-input test case")
  endif()
  message(FATAL_ERROR "VSSMath invalid-input case was not rejected")
endif()

set(maximum "18446744073709551615")
set(maximum_plus_one "18446744073709551616")
vss_algorithm_uint64("0" is_zero_valid)
vss_algorithm_uint64("${maximum}" is_maximum_valid)
vss_algorithm_uint64("${maximum_plus_one}" is_overflow_valid)
vss_algorithm_uint64("-1" is_negative_valid)
require_equal("zero is a uint64" TRUE "${is_zero_valid}")
require_equal("uint64 maximum is valid" TRUE "${is_maximum_valid}")
require_equal("uint64 overflow is invalid" FALSE "${is_overflow_valid}")
require_equal("negative integer is invalid" FALSE "${is_negative_valid}")

vss_algorithm_decimal_compare("9" "10" comparison)
require_equal("decimal comparison does not use numeric coercion" -1 "${comparison}")
vss_algorithm_decimal_compare("${maximum}" "${maximum}" comparison)
require_equal("equal decimal comparison" 0 "${comparison}")
vss_algorithm_decimal_add("${maximum}" "1" sum)
require_equal("decimal addition preserves a uint64 carry" "${maximum_plus_one}" "${sum}")
vss_algorithm_decimal_subtract("${maximum}" "18446744073709551614" difference)
require_equal("decimal subtraction" 1 "${difference}")
vss_algorithm_decimal_multiply_small("${maximum}" "120" product)
require_equal("decimal multiplication preserves wide products"
  "2213609288845146193800" "${product}")
vss_algorithm_decimal_divide_small("241" "8" quotient remainder)
require_equal("small-divisor quotient" 30 "${quotient}")
require_equal("small-divisor remainder" 1 "${remainder}")

set(samples "1;2;3;4;5;6;7;8;9;10;11;12;13;14;15;16;17;18;19;20;21;22;23;24;25;26;27;28;29;30")
set(original_samples "${samples}")
vss_algorithm_statistics("${samples}" median mad)
require_equal("median of 1 through 30" 15 "${median}")
require_equal("MAD of 1 through 30" 7 "${mad}")
require_equal("statistics leaves caller sample order unchanged"
  "${original_samples}" "${samples}")

set(samples)
foreach(sample_index RANGE 1 15)
  list(APPEND samples 95)
endforeach()
foreach(sample_index RANGE 1 15)
  list(APPEND samples 105)
endforeach()
vss_algorithm_statistics("${samples}" median mad)
require_equal("even sample median" 100 "${median}")
require_equal("even sample MAD" 5 "${mad}")

set(samples)
foreach(sample_index RANGE 1 15)
  list(APPEND samples 18446744073709551614 18446744073709551615)
endforeach()
set(original_samples "${samples}")
vss_algorithm_statistics("${samples}" median mad)
require_equal("large alternating sample median"
  18446744073709551614 "${median}")
require_equal("large alternating sample MAD" 0 "${mad}")
require_equal("large statistics leave caller sample order unchanged"
  "${original_samples}" "${samples}")

set(invalid_cases
  "noncanonical|vss_algorithm_decimal_compare"
  "divide-zero|vss_algorithm_decimal_divide_small"
  "subtract-underflow|vss_algorithm_decimal_subtract"
  "factor-out-of-range|vss_algorithm_decimal_multiply_small")
foreach(invalid_case IN LISTS invalid_cases)
  string(REPLACE "|" ";" invalid_fields "${invalid_case}")
  list(GET invalid_fields 0 case_name)
  list(GET invalid_fields 1 expected_context)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DVSS_ALGORITHM_DECIMAL_INVALID_CASE=${case_name}"
      -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE invalid_status
    OUTPUT_VARIABLE invalid_stdout
    ERROR_VARIABLE invalid_stderr)
  if(invalid_status EQUAL 0)
    message(FATAL_ERROR "VSSMath ${case_name}: invalid input was accepted")
  endif()
  if(NOT "${invalid_stdout}${invalid_stderr}" MATCHES "${expected_context}")
    message(FATAL_ERROR
      "VSSMath ${case_name}: failure lacks '${expected_context}' context: "
      "${invalid_stdout}${invalid_stderr}")
  endif()
endforeach()

message(STATUS "VSSMath decimal and statistics tests passed")
