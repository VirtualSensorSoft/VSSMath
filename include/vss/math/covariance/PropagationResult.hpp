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
  Index input;
  Index valid;
  Index positive;
  std::optional<std::uint64_t> frequency_mass;
  std::optional<Scalar> weight_mass;
  std::array<std::optional<Scalar>, 3> normalized_weight_powers;
  std::optional<Scalar> effective_count;
  std::optional<Scalar> degrees_of_freedom;
  std::optional<Scalar> squared_weight_mass = std::nullopt;
};

/// @brief Associates support diagnostics with a selected tensor coordinate.
struct TupleSupport final {
  Array<Index> coordinates;
  Support support;
};

/// @brief Reports estimator assumptions, support, accuracy and warnings.
struct EstimatorDiagnostics final {
  Estimand estimand;
  WeightRule weight;
  CenterRule center;
  Approximation approximation;
  Support global;
  Array<Support> coordinate_support;
  Array<TupleSupport> tuple_support = {};
  std::optional<Scalar> reciprocal_condition = std::nullopt;
  std::optional<Scalar> numerical_error;
  std::optional<Scalar> sampling_error;
  std::optional<Index> rank;
  std::string algorithm;
  std::string premises;
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
  Vector mean;
  Matrix covariance;
  Approximation method;
  EstimatorDiagnostics diagnostics;
  PropagationCovarianceCheck covariance_check{
      PropagationCovarianceCheck::positive_semidefinite};
};

} // namespace vss::math

#endif // VSS_MATH_COVARIANCE_PROPAGATION_RESULT_HPP
