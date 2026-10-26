cmake_minimum_required(VERSION 3.28)
include_guard(GLOBAL)

set(_VSS_ALGORITHM_UINT64_MAX "18446744073709551615")
set(_VSS_ALGORITHM_SMALL_OPERAND_MAX "1048576")

# Validate canonical, unsigned decimal text against the uint64 range.
# Inputs: value is the text to check. Output: out_ok is TRUE or FALSE in the
# caller's scope. This predicate never raises an error for invalid input.
function(vss_algorithm_uint64 value out_ok)
  if(NOT "${value}" MATCHES "^(0|[1-9][0-9]*)$")
    set("${out_ok}" FALSE PARENT_SCOPE)
    return()
  endif()

  string(LENGTH "${value}" value_length)
  string(LENGTH "${_VSS_ALGORITHM_UINT64_MAX}" max_length)
  if(value_length LESS max_length)
    set("${out_ok}" TRUE PARENT_SCOPE)
    return()
  endif()
  if(value_length GREATER max_length)
    set("${out_ok}" FALSE PARENT_SCOPE)
    return()
  endif()

  if("${value}" STRGREATER "${_VSS_ALGORITHM_UINT64_MAX}")
    set("${out_ok}" FALSE PARENT_SCOPE)
  else()
    set("${out_ok}" TRUE PARENT_SCOPE)
  endif()
endfunction()

# Compare two canonical nonnegative decimal strings without numeric coercion.
# Inputs: left and right are canonical digit strings. Output: out_comparison
# is -1, 0, or 1 in the caller's scope. Invalid operands are fatal errors.
function(vss_algorithm_decimal_compare left right out_comparison)
  if(NOT "${left}" MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_compare: left operand is not canonical decimal")
  endif()
  if(NOT "${right}" MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_compare: right operand is not canonical decimal")
  endif()

  string(LENGTH "${left}" left_length)
  string(LENGTH "${right}" right_length)
  if(left_length LESS right_length)
    set(comparison -1)
  elseif(left_length GREATER right_length)
    set(comparison 1)
  elseif("${left}" STRLESS "${right}")
    set(comparison -1)
  elseif("${left}" STRGREATER "${right}")
    set(comparison 1)
  else()
    set(comparison 0)
  endif()

  set("${out_comparison}" "${comparison}" PARENT_SCOPE)
endfunction()

# Add canonical nonnegative decimal strings exactly.
# Inputs: left and right are canonical digit strings. Output: out_value is a
# canonical decimal string in the caller's scope; it may exceed uint64.
function(vss_algorithm_decimal_add left right out_value)
  if(NOT "${left}" MATCHES "^(0|[1-9][0-9]*)$" OR
     NOT "${right}" MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_add: operands must be canonical decimals")
  endif()

  string(LENGTH "${left}" left_length)
  string(LENGTH "${right}" right_length)
  math(EXPR left_index "${left_length} - 1")
  math(EXPR right_index "${right_length} - 1")
  set(carry 0)
  set(result "")

  while(left_index GREATER_EQUAL 0 OR right_index GREATER_EQUAL 0 OR carry GREATER 0)
    if(left_index GREATER_EQUAL 0)
      string(SUBSTRING "${left}" ${left_index} 1 left_digit)
    else()
      set(left_digit 0)
    endif()
    if(right_index GREATER_EQUAL 0)
      string(SUBSTRING "${right}" ${right_index} 1 right_digit)
    else()
      set(right_digit 0)
    endif()

    math(EXPR column_sum "${left_digit} + ${right_digit} + ${carry}")
    math(EXPR digit "${column_sum} % 10")
    math(EXPR carry "${column_sum} / 10")
    string(PREPEND result "${digit}")
    math(EXPR left_index "${left_index} - 1")
    math(EXPR right_index "${right_index} - 1")
  endwhile()

  set("${out_value}" "${result}" PARENT_SCOPE)
endfunction()

# Subtract right from left exactly.
# Inputs: operands are canonical nonnegative decimals and left >= right.
# Output: out_value is a canonical decimal string in the caller's scope.
# Invalid operands or underflow are fatal errors with this operation's context.
function(vss_algorithm_decimal_subtract left right out_value)
  if(NOT "${left}" MATCHES "^(0|[1-9][0-9]*)$" OR
     NOT "${right}" MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_subtract: operands must be canonical decimals")
  endif()

  vss_algorithm_decimal_compare("${left}" "${right}" comparison)
  if(comparison LESS 0)
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_subtract: left operand is smaller than right operand")
  endif()

  string(LENGTH "${left}" left_length)
  string(LENGTH "${right}" right_length)
  math(EXPR left_index "${left_length} - 1")
  math(EXPR right_index "${right_length} - 1")
  set(borrow 0)
  set(result "")

  while(left_index GREATER_EQUAL 0)
    string(SUBSTRING "${left}" ${left_index} 1 left_digit)
    if(right_index GREATER_EQUAL 0)
      string(SUBSTRING "${right}" ${right_index} 1 right_digit)
    else()
      set(right_digit 0)
    endif()

    math(EXPR difference "${left_digit} - ${right_digit} - ${borrow}")
    if(difference LESS 0)
      math(EXPR difference "${difference} + 10")
      set(borrow 1)
    else()
      set(borrow 0)
    endif()
    string(PREPEND result "${difference}")
    math(EXPR left_index "${left_index} - 1")
    math(EXPR right_index "${right_index} - 1")
  endwhile()

  string(REGEX REPLACE "^0+" "" result "${result}")
  if(result STREQUAL "")
    set(result 0)
  endif()
  set("${out_value}" "${result}" PARENT_SCOPE)
endfunction()

# Multiply a canonical decimal string by a bounded small integer.
# Inputs: value is canonical decimal; factor is 0..1048576. Output: out_value
# is exact decimal text, which may exceed uint64. Invalid inputs are fatal.
function(vss_algorithm_decimal_multiply_small value factor out_value)
  if(NOT "${value}" MATCHES "^(0|[1-9][0-9]*)$" OR
     NOT "${factor}" MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_multiply_small: operands must be canonical decimals")
  endif()

  vss_algorithm_decimal_compare(
    "${factor}" "${_VSS_ALGORITHM_SMALL_OPERAND_MAX}" factor_comparison)
  if(factor_comparison GREATER 0)
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_multiply_small: factor exceeds 1048576")
  endif()
  math(EXPR factor_value "${factor}")

  string(LENGTH "${value}" value_length)
  math(EXPR value_index "${value_length} - 1")
  set(carry 0)
  set(result "")
  while(value_index GREATER_EQUAL 0 OR carry GREATER 0)
    if(value_index GREATER_EQUAL 0)
      string(SUBSTRING "${value}" ${value_index} 1 digit_value)
    else()
      set(digit_value 0)
    endif()

    math(EXPR column_product "${digit_value} * ${factor_value} + ${carry}")
    math(EXPR output_digit "${column_product} % 10")
    math(EXPR carry "${column_product} / 10")
    string(PREPEND result "${output_digit}")
    math(EXPR value_index "${value_index} - 1")
  endwhile()

  set("${out_value}" "${result}" PARENT_SCOPE)
endfunction()

# Divide a canonical decimal string by a bounded positive small integer.
# Inputs: value is canonical decimal; divisor is 1..1048576. Outputs:
# out_quotient and out_remainder are canonical decimal strings in caller scope.
# Invalid input or a zero divisor is a fatal error.
function(vss_algorithm_decimal_divide_small value divisor out_quotient out_remainder)
  if(NOT "${value}" MATCHES "^(0|[1-9][0-9]*)$" OR
     NOT "${divisor}" MATCHES "^(0|[1-9][0-9]*)$")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_divide_small: operands must be canonical decimals")
  endif()
  if(divisor STREQUAL "0")
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_divide_small: divisor must be positive")
  endif()

  vss_algorithm_decimal_compare(
    "${divisor}" "${_VSS_ALGORITHM_SMALL_OPERAND_MAX}" divisor_comparison)
  if(divisor_comparison GREATER 0)
    message(FATAL_ERROR
      "VSSMath vss_algorithm_decimal_divide_small: divisor exceeds 1048576")
  endif()
  math(EXPR divisor_value "${divisor}")

  string(LENGTH "${value}" value_length)
  set(remainder_value 0)
  set(quotient "")
  foreach(index RANGE 0 ${value_length})
    if(index EQUAL value_length)
      break()
    endif()
    string(SUBSTRING "${value}" ${index} 1 digit_value)
    math(EXPR remainder_value "${remainder_value} * 10 + ${digit_value}")
    math(EXPR quotient_digit "${remainder_value} / ${divisor_value}")
    math(EXPR remainder_value "${remainder_value} % ${divisor_value}")
    string(APPEND quotient "${quotient_digit}")
  endforeach()

  string(REGEX REPLACE "^0+" "" quotient "${quotient}")
  if(quotient STREQUAL "")
    set(quotient 0)
  endif()
  set("${out_quotient}" "${quotient}" PARENT_SCOPE)
  set("${out_remainder}" "${remainder_value}" PARENT_SCOPE)
endfunction()

# Sort a list of canonical decimal strings without changing the caller's list.
# Input: samples is a CMake list. Output: out_sorted is a sorted copy.
function(_vss_algorithm_decimal_sort samples out_sorted)
  set(sorted_samples)
  foreach(sample IN LISTS samples)
    set(inserted FALSE)
    set(next_samples)
    foreach(existing IN LISTS sorted_samples)
      if(NOT inserted)
        vss_algorithm_decimal_compare("${sample}" "${existing}" comparison)
        if(comparison LESS 0)
          list(APPEND next_samples "${sample}")
          set(inserted TRUE)
        endif()
      endif()
      list(APPEND next_samples "${existing}")
    endforeach()
    if(NOT inserted)
      list(APPEND next_samples "${sample}")
    endif()
    set(sorted_samples "${next_samples}")
  endforeach()
  set("${out_sorted}" "${sorted_samples}" PARENT_SCOPE)
endfunction()

# Calculate the floor-valued median of exactly 30 sorted decimal samples.
# Inputs: samples is a sorted list. Output: out_median is in caller scope.
function(_vss_algorithm_decimal_median_sorted samples out_median)
  list(GET samples 14 lower_middle)
  list(GET samples 15 upper_middle)
  vss_algorithm_decimal_subtract(
    "${upper_middle}" "${lower_middle}" middle_difference)
  vss_algorithm_decimal_divide_small(
    "${middle_difference}" 2 middle_half unused_remainder)
  vss_algorithm_decimal_add("${lower_middle}" "${middle_half}" median)
  set("${out_median}" "${median}" PARENT_SCOPE)
endfunction()

# Compute the exact median and median absolute deviation of 30 positive
# uint64 nanosecond values. Inputs: samples is their CMake list. Outputs:
# out_median and out_mad are canonical decimals in caller scope. The input list
# is unchanged; invalid counts or values are fatal with the function context.
function(vss_algorithm_statistics samples out_median out_mad)
  list(LENGTH samples sample_count)
  if(NOT sample_count EQUAL 30)
    message(FATAL_ERROR
      "VSSMath vss_algorithm_statistics: expected 30 samples, got ${sample_count}")
  endif()

  foreach(sample IN LISTS samples)
    vss_algorithm_uint64("${sample}" sample_is_uint64)
    if(NOT sample_is_uint64 OR sample STREQUAL "0")
      message(FATAL_ERROR
        "VSSMath vss_algorithm_statistics: sample must be a positive uint64 decimal")
    endif()
  endforeach()

  _vss_algorithm_decimal_sort("${samples}" sorted_samples)
  _vss_algorithm_decimal_median_sorted("${sorted_samples}" median)

  set(deviations)
  foreach(sample IN LISTS samples)
    vss_algorithm_decimal_compare("${sample}" "${median}" comparison)
    if(comparison LESS 0)
      vss_algorithm_decimal_subtract("${median}" "${sample}" deviation)
    else()
      vss_algorithm_decimal_subtract("${sample}" "${median}" deviation)
    endif()
    list(APPEND deviations "${deviation}")
  endforeach()
  _vss_algorithm_decimal_sort("${deviations}" sorted_deviations)
  _vss_algorithm_decimal_median_sorted("${sorted_deviations}" mad)

  set("${out_median}" "${median}" PARENT_SCOPE)
  set("${out_mad}" "${mad}" PARENT_SCOPE)
endfunction()
