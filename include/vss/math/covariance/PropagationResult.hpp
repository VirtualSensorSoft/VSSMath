#ifndef VSS_MATH_COVARIANCE_PROPAGATION_RESULT_HPP
#define VSS_MATH_COVARIANCE_PROPAGATION_RESULT_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>

namespace vss::math {

/// @brief Identifies the quantity whose uncertainty or moments are estimated.
enum class Estimand : std::uint8_t {
  observed,
  measurement_error,
  estimate,
  calibration,
  process,
  fitted
};

/// @brief Defines how observations contribute to an estimator.
enum class WeightRule : std::uint8_t {
  unweighted,
  empirical,
  fixed_iid,
  frequency,
  precision,
  importance,
  componentwise,
  dependent
};

/// @brief Defines whether an estimator's center is observed or supplied.
enum class CenterRule : std::uint8_t { estimated, known };

/// @brief Identifies an approximation used to produce an estimate.
enum class Approximation : std::uint8_t {
  exact,
  first_order,
  quadratic,
  fourth_amplitude,
  sigma_points,
  monte_carlo
};

/// @brief Records input support and weight mass for one estimate.
struct Support final {
  /// @brief Number of input observations considered.
  Index input;
  /// @brief Number of observations admitted by the declared support rule.
  Index valid;
  /// @brief Number of admitted observations with positive weight.
  Index positive;
  /// @brief Exact total replication count when frequency weights apply.
  std::optional<std::uint64_t> frequency_mass;
  /// @brief Sum of admitted scalar weights when defined.
  std::optional<Scalar> weight_mass;
  /// @brief Sums of normalized weights to powers 2, 3, and 4, when defined.
  std::array<std::optional<Scalar>, 3> normalized_weight_powers;
  /// @brief Effective support count when its estimator premise is defined.
  std::optional<Scalar> effective_count;
  /// @brief Degrees of freedom when defined by the estimator contract.
  std::optional<Scalar> degrees_of_freedom;
  /// @brief Sum of squared scalar weights when the regime defines it.
  std::optional<Scalar> squared_weight_mass = std::nullopt;
};

/// @brief Associates support diagnostics with a selected tensor coordinate.
struct TupleSupport final {
  /// @brief Flattened coordinate tuple whose support was measured.
  Array<Index> coordinates;
  /// @brief Support and weight diagnostics for that coordinate tuple.
  Support support;
};

/// @brief Reports estimator assumptions, support, accuracy and warnings.
struct EstimatorDiagnostics final {
  /// @brief Quantity or parameter targeted by the estimate.
  Estimand estimand;
  /// @brief Weighting interpretation used by the estimator.
  WeightRule weight;
  /// @brief Whether the center was estimated or supplied.
  CenterRule center;
  /// @brief Exact or approximate method used to produce the estimate.
  Approximation approximation;
  /// @brief Global support shared by the estimate when applicable.
  Support global;
  /// @brief Per-coordinate support when support differs across coordinates.
  Array<Support> coordinate_support;
  /// @brief Per-tuple support when support differs across moment tuples.
  Array<TupleSupport> tuple_support;
  /// @brief Retained-spectrum reciprocal condition when available.
  std::optional<Scalar> reciprocal_condition = std::nullopt;
  /// @brief Absolute numerical-error bound in the result norm and units;
  ///        its interpretation is documented in the algorithm premise.
  std::optional<Scalar> numerical_error;
  /// @brief Sampling-error estimate only under a named calibrated model;
  ///        its model is identified in @ref premises.
  std::optional<Scalar> sampling_error;
  /// @brief Effective numerical rank when the algorithm reports one.
  std::optional<Index> rank;
  /// @brief Human-readable algorithm identifier.
  std::string algorithm;
  /// @brief Statistical or numerical premises required for interpretation.
  std::string premises;
  /// @brief Conditions or limitations reported without replacing the value.
  Array<ErrorCode> warnings;
};

/// @brief Classifies the covariance produced by a propagation approximation.
enum class PropagationCovarianceCheck : std::uint8_t {
  positive_semidefinite,
  negative_within_roundoff,
  indefinite_approximation
};

/// @brief Owns a propagated estimate and its covariance diagnostics.
/// @details The covariance is preserved as produced. Some approximation
///          methods can return an indefinite signed result; consumers must
///          validate the actual matrix rather than trust `covariance_check`.
struct PropagationResult final {
  /// @brief Propagated nominal output values.
  Vector mean;
  /// @brief Numerical output covariance, retained without PSD projection.
  Matrix covariance;
  /// @brief Propagation approximation or simulation method.
  Approximation method;
  /// @brief Estimator and method assumptions associated with the result.
  EstimatorDiagnostics diagnostics;
  /// @brief Producer's covariance classification; consumers still revalidate.
  PropagationCovarianceCheck covariance_check{
      PropagationCovarianceCheck::positive_semidefinite};
};

} // namespace vss::math

#endif // VSS_MATH_COVARIANCE_PROPAGATION_RESULT_HPP
