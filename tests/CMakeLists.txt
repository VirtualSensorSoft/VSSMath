function(vss_math_add_test target source ctest_name)
  add_executable(${target} ${source})
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  target_link_libraries(${target} PRIVATE VirtualSensorSoft::math)
  target_include_directories(${target} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/support")
  vss_math_enable_warnings(${target})
  vss_math_enable_clang_tidy(${target})
  if(VSS_MATH_ENABLE_SANITIZERS)
    vss_math_enable_sanitizers(${target})
  endif()
  vss_math_enable_coverage(${target})
  add_test(NAME ${ctest_name} COMMAND ${target})
endfunction()

function(vss_math_add_algorithm_test target source ctest_name)
  vss_math_add_test(${target} ${source} ${ctest_name})
  if("PRIVATE_DETAIL" IN_LIST ARGN)
    target_include_directories(${target} PRIVATE
      "${PROJECT_SOURCE_DIR}/src")
  endif()
endfunction()

vss_math_add_test(test_version unit/test_version.cpp math.version)
vss_math_add_test(test_status unit/test_status.cpp math.status)
vss_math_add_test(test_result unit/test_result.cpp math.result)
vss_math_add_test(test_array unit/test_array.cpp math.array)
vss_math_add_test(test_tolerance unit/test_tolerance.cpp math.tolerance)
vss_math_add_test(test_platform_contract unit/test_platform_contract.cpp
                  math.platform_contract)
vss_math_add_algorithm_test(test_factorization_options
                             unit/test_factorization_options.cpp
                             math.factorization_options PRIVATE_DETAIL)
vss_math_add_algorithm_test(test_linear_solve unit/test_linear_solve.cpp
                             math.linear_solve PRIVATE_DETAIL)
vss_math_add_algorithm_test(test_triangular_solve
                             algorithms/test_triangular_solve.cpp
                             math.triangular_solve)
vss_math_add_algorithm_test(test_lu algorithms/test_lu.cpp math.lu)
vss_math_add_algorithm_test(test_qr algorithms/test_qr.cpp math.qr)
vss_math_add_algorithm_test(test_cholesky algorithms/test_cholesky.cpp
                             math.cholesky)
vss_math_add_algorithm_test(test_symmetric_eigen
                             algorithms/test_symmetric_eigen.cpp
                             math.symmetric_eigen)
vss_math_add_algorithm_test(test_generalized_symmetric_eigen
                             algorithms/test_generalized_symmetric_eigen.cpp
                             math.generalized_symmetric_eigen)
vss_math_add_algorithm_test(test_rank_aware_least_squares
                             algorithms/test_rank_aware_least_squares.cpp
                             math.rank_aware_least_squares)
vss_math_add_algorithm_test(test_null_space
                             algorithms/test_null_space.cpp
                             math.null_space)
vss_math_add_algorithm_test(test_full_qr
                             algorithms/test_full_qr.cpp
                             math.full_qr)
vss_math_add_algorithm_test(test_covariance
                             algorithms/test_covariance.cpp
                             math.covariance)
vss_math_add_algorithm_test(test_error_models
                             algorithms/test_error_models.cpp
                             math.error_models)
vss_math_add_algorithm_test(test_sensitivity
                             algorithms/test_sensitivity.cpp
                             math.sensitivity)
set_tests_properties(math.sensitivity PROPERTIES
                     LABELS "ci-once-per-full;ci-extended"
                     ENVIRONMENT "VSS_MATH_FULL_SENSITIVITY_ACCEPTANCE=1")
vss_math_add_algorithm_test(test_propagation
                             algorithms/test_propagation.cpp
                             math.propagation)
vss_math_add_algorithm_test(test_sparse algorithms/test_sparse.cpp math.sparse)
vss_math_add_algorithm_test(test_structured algorithms/test_structured.cpp
                             math.structured)
vss_math_add_algorithm_test(test_structured_covariance
                             algorithms/test_structured_covariance.cpp
                             math.structured_covariance)
vss_math_add_algorithm_test(test_iterative algorithms/test_iterative.cpp
                             math.iterative)
vss_math_add_algorithm_test(test_conditioning
                             algorithms/test_conditioning.cpp
                             math.conditioning)
vss_math_add_algorithm_test(test_reduction algorithms/test_reduction.cpp
                             math.reduction)
vss_math_add_algorithm_test(test_diagnostics algorithms/test_diagnostics.cpp
                             math.diagnostics)
vss_math_add_algorithm_test(test_uncertainty_summary
                             algorithms/test_uncertainty_summary.cpp
                             math.uncertainty_summary)
vss_math_add_algorithm_test(test_finite_difference
                             algorithms/test_finite_difference.cpp
                             math.finite_difference)
vss_math_add_algorithm_test(test_polynomial algorithms/test_polynomial.cpp
                             math.polynomial)
vss_math_add_algorithm_test(test_interpolation
                             algorithms/test_interpolation.cpp
                             math.interpolation)
vss_math_add_algorithm_test(test_roots algorithms/test_roots.cpp math.roots)
vss_math_add_algorithm_test(test_integration
                             algorithms/test_integration.cpp
                             math.integration)
vss_math_add_algorithm_test(test_ode algorithms/test_ode.cpp math.ode)
vss_math_add_algorithm_test(test_functions algorithms/test_functions.cpp
                             math.functions)
vss_math_add_algorithm_test(test_scientific_extras
                             algorithms/test_scientific_extras.cpp
                             math.scientific_extras)
vss_math_add_algorithm_test(test_geometry algorithms/test_geometry.cpp
                            math.geometry)
vss_math_add_algorithm_test(test_differentiation_extensions
                             algorithms/test_differentiation_extensions.cpp
                             math.differentiation_extensions)
vss_math_add_algorithm_test(test_descriptive_statistics
                             algorithms/test_descriptive_statistics.cpp
                             math.descriptive_statistics PRIVATE_DETAIL)
vss_math_add_algorithm_test(test_order_statistics
                             algorithms/test_order_statistics.cpp
                             math.order_statistics)
vss_math_add_algorithm_test(test_population_statistics
                             algorithms/test_population_statistics.cpp
                             math.population_statistics)
vss_math_add_algorithm_test(test_robust_statistics
                             algorithms/test_robust_statistics.cpp
                             math.robust_statistics)
vss_math_add_algorithm_test(test_missing_statistics
                             algorithms/test_missing_statistics.cpp
                             math.missing_statistics)
vss_math_add_algorithm_test(test_empirical_statistics
                             algorithms/test_empirical_statistics.cpp
                             math.empirical_statistics)
vss_math_add_algorithm_test(test_dependence_statistics
                             algorithms/test_dependence_statistics.cpp
                             math.dependence_statistics)
vss_math_add_algorithm_test(test_inference_statistics
                             algorithms/test_inference_statistics.cpp
                             math.inference_statistics)
vss_math_add_algorithm_test(test_ratio_estimate
                             algorithms/test_ratio_estimate.cpp
                             math.ratio_estimate)
vss_math_add_algorithm_test(test_estimator_diagnostics
                             algorithms/test_estimator_diagnostics.cpp
                             math.estimator_diagnostics)
vss_math_add_algorithm_test(test_bootstrap_statistics
                             algorithms/test_bootstrap_statistics.cpp
                             math.bootstrap_statistics)
vss_math_add_algorithm_test(test_quantile_state
                             algorithms/test_quantile_state.cpp
                             math.quantile_state)
vss_math_add_algorithm_test(test_weighted_population
                             algorithms/test_weighted_population.cpp
                             math.weighted_population)
vss_math_add_algorithm_test(test_population_moments
                             algorithms/test_moments.cpp
                             math.population_moments)
vss_math_add_algorithm_test(test_moment_state
                             algorithms/test_moment_state.cpp
                             math.moment_state)
vss_math_add_algorithm_test(test_fixed_weighted_third_moment
                             algorithms/test_fixed_weighted_third_moment.cpp
                             math.fixed_weighted_third_moment)
vss_math_add_algorithm_test(test_fixed_weighted_fourth_moment
                             algorithms/test_fixed_weighted_fourth_moment.cpp
                             math.fixed_weighted_fourth_moment)
vss_math_add_algorithm_test(test_cumulants algorithms/test_cumulants.cpp
                             math.cumulants)
vss_math_add_algorithm_test(test_chi_square algorithms/test_chi_square.cpp
                             math.probability PRIVATE_DETAIL)
vss_math_add_algorithm_test(test_special_functions
                             algorithms/test_special_functions.cpp
                             math.probability.special_functions)
vss_math_add_algorithm_test(test_distributions
                             algorithms/test_distributions.cpp
                             math.probability.distributions)
vss_math_add_algorithm_test(test_composite
                             algorithms/test_composite.cpp
                             math.probability.composite)
vss_math_add_algorithm_test(test_nonideal_observations
                             algorithms/test_nonideal_observations.cpp
                             math.probability.nonideal_observations)
vss_math_add_algorithm_test(test_moment_bounds
                             algorithms/test_moment_bounds.cpp
                             math.probability.moment_bounds)
vss_math_add_algorithm_test(test_sampling_design
                             algorithms/test_sampling_design.cpp
                             math.probability.sampling_design)
vss_math_add_algorithm_test(test_likelihood algorithms/test_likelihood.cpp
                             math.probability.likelihood)
vss_math_add_algorithm_test(test_gaussian algorithms/test_gaussian.cpp
                             math.probability.gaussian)
vss_math_add_test(test_error_precedence unit/test_error_precedence.cpp
                  math.error_precedence)
vss_math_add_test(test_numerical_boundaries unit/test_numerical_boundaries.cpp
                  math.numerical_boundaries)
vss_math_add_test(test_reproducibility unit/test_reproducibility.cpp
                  math.reproducibility)
vss_math_add_test(test_vector unit/test_vector.cpp math.vector)
vss_math_add_test(test_matrix unit/test_matrix.cpp math.matrix)
vss_math_add_test(test_tensor_shape unit/test_tensor_shape.cpp math.tensor_shape)
vss_math_add_test(test_tensor unit/test_tensor.cpp math.tensor)
vss_math_add_test(test_tensor_layouts unit/test_tensor_layouts.cpp
                  math.tensor_layouts)
vss_math_add_test(test_tensor_operations unit/test_tensor_operations.cpp
                  math.tensor_arithmetic)
vss_math_add_test(test_tensor_contraction unit/test_tensor_contraction.cpp
                  math.tensor_contraction)
vss_math_add_test(test_packed_symmetric unit/test_packed_symmetric.cpp
                  math.packed_symmetric)
vss_math_add_test(test_scaling unit/test_scaling.cpp math.scaling)
vss_math_add_test(test_structured_products
                  unit/test_structured_products.cpp math.structured_products)
vss_math_add_algorithm_test(test_svd algorithms/test_svd.cpp math.svd)
vss_math_add_algorithm_test(test_ldlt algorithms/test_ldlt.cpp math.ldlt)
vss_math_add_test(test_tensor_reductions unit/test_tensor_reductions.cpp
                  math.tensor_reductions)
vss_math_add_test(test_einsum unit/test_einsum.cpp math.einsum)
vss_math_add_test(test_elementwise_extensions
                  unit/test_elementwise_extensions.cpp
                  math.elementwise_extensions)
vss_math_add_test(test_tensor_products unit/test_tensor_products.cpp
                  math.tensor_products)
vss_math_add_test(test_moments unit/test_moments.cpp math.moments_wrappers)
vss_math_add_test(test_moments_aggregate unit/test_moments_aggregate.cpp
                  math.moments_aggregate)
vss_math_add_algorithm_test(test_moment_diagnostics
                             algorithms/test_moment_diagnostics.cpp
                             math.moment_diagnostics)
vss_math_add_test(test_stream_output unit/test_stream_output.cpp
                  math.stream_output)
vss_math_add_test(test_vector_view unit/test_vector_view.cpp math.vector_view)
vss_math_add_test(test_matrix_view unit/test_matrix_view.cpp math.matrix_view)
vss_math_add_test(test_operations unit/test_operations.cpp math.operations)
vss_math_add_test(test_algebra_properties properties/test_algebra_properties.cpp math.algebra_properties)
vss_math_add_algorithm_test(test_algorithm_properties
                             properties/test_algorithm_properties.cpp
                             math.algorithm_properties)
vss_math_add_algorithm_test(test_algorithm_failures
                             negative/test_algorithm_failures.cpp
                             math.algorithm_failures)
vss_math_add_algorithm_test(test_algorithm_error_precedence
                             negative/test_error_precedence.cpp
                             math.algorithm_error_precedence)
vss_math_add_algorithm_test(test_algorithm_lifecycle
                             negative/test_lifecycle.cpp
                             math.algorithm_lifecycle)
find_package(Threads REQUIRED)
target_link_libraries(test_algorithm_lifecycle PRIVATE Threads::Threads)
add_test(NAME math.performance_gate_contract
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_regression_gate.cmake")
add_test(NAME math.performance_baseline_schema
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_baseline_schema.cmake")
add_test(NAME math.qualified_baseline_staging_policy
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_qualified_baseline_staging.cmake")
add_test(NAME math.local_capture_environment_policy
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_local_capture_environment.cmake")
add_test(NAME math.algorithm_qualification_policy
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_qualification.cmake")
add_test(NAME math.algorithm_decimal
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_decimal.cmake")

add_test(NAME math.algorithm_record_contract
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_record.cmake")
add_test(NAME math.algorithm_comparison
  COMMAND "${CMAKE_COMMAND}" -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_comparison.cmake")
add_test(NAME math.algorithm_baseline_selector
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_PERF_CMAKE_COMMAND=${CMAKE_COMMAND}"
    "-DVSS_PERF_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DVSS_PERF_SCRATCH=${CMAKE_BINARY_DIR}/algorithm-baseline-selector-test"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_baseline_selector.cmake")
add_test(NAME math.algorithm_batch_plan
  COMMAND "${CMAKE_COMMAND}"
          "-DVSS_MATH_BATCH_PLAN_READER=$<TARGET_FILE:test_batch_plan_reader>"
          -P
          "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_batch_plan.cmake")
add_executable(test_batch_plan_reader
  performance/test_batch_plan_reader.cpp
  performance/support/BenchmarkBatchPlan.cpp)
set_target_properties(test_batch_plan_reader PROPERTIES CXX_EXTENSIONS OFF)
target_compile_features(test_batch_plan_reader PRIVATE cxx_std_20)
target_include_directories(test_batch_plan_reader PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/support")
vss_math_enable_warnings(test_batch_plan_reader)
vss_math_enable_clang_tidy(test_batch_plan_reader)
if(VSS_MATH_ENABLE_SANITIZERS)
  vss_math_enable_sanitizers(test_batch_plan_reader)
endif()
vss_math_enable_coverage(test_batch_plan_reader)
add_test(NAME math.algorithm_batch_plan_reader
  COMMAND test_batch_plan_reader
    "${CMAKE_CURRENT_SOURCE_DIR}/performance/fixtures/valid-batch-plan.env")
vss_math_add_test(test_algorithm_measurement
  performance/test_algorithm_measurement.cpp math.algorithm_measurement)
target_sources(test_algorithm_measurement PRIVATE
  performance/support/BenchmarkMeasurement.cpp)
include("${PROJECT_SOURCE_DIR}/cmake/VssMathBenchmarkIsolation.cmake")
vss_math_disable_benchmark_ipo(test_algorithm_measurement)
add_test(NAME math.algorithm_measurement_build_policy
  COMMAND "${CMAKE_COMMAND}"
    "-DVSS_MATH_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DVSS_MATH_MEASUREMENT_BUILD_SCRATCH=${CMAKE_BINARY_DIR}/measurement-build-policy"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/performance/test_algorithm_measurement_build.cmake")
add_test(NAME math.conformance_record_policy
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/policy/test_conformance_record.sh")

add_test(NAME math.analyzer_suppression_policy
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/policy/test_analyzer_suppressions.sh")
add_test(NAME math.numerical_traceability
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/policy/test_numerical_traceability.sh")
add_test(NAME math.documentation_policy
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/policy/test_documentation_policy.sh")

set_property(TEST
  math.performance_baseline_schema
  math.qualified_baseline_staging_policy
  math.algorithm_qualification_policy
  math.algorithm_record_contract
  math.algorithm_comparison
  math.algorithm_baseline_selector
  math.algorithm_batch_plan
  APPEND PROPERTY LABELS ci-extended)

set_property(TEST
  math.qualified_baseline_staging_policy
  math.algorithm_qualification_policy
  math.algorithm_comparison
  APPEND PROPERTY LABELS ci-once-per-full)
