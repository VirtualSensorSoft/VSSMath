#ifndef VSS_MATH_LINEAR_FACTORIZATION_OPTIONS_HPP
#define VSS_MATH_LINEAR_FACTORIZATION_OPTIONS_HPP

#include <cstdint>
#include <vss/math/Core.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math::linear {

/// @brief Absolute and relative thresholds used when classifying numerical
/// rank.
/// @details Both fields are passive configuration. Algorithms validate that
///          supplied values are finite and non-negative when they consume the
///          object; constructing this aggregate does not perform validation.
struct RankTolerance final {
  /// @brief Absolute threshold for a rank decision.
  Scalar absolute{0.0};
  /// @brief Relative threshold for a rank decision.
  Scalar relative{1.0e-12};
};

/// @brief Common options for matrix factorizations.
/// @details The condition limit is an exclusive upper bound for the estimated
///          1-norm condition number: values strictly below it are within the
///          limit, while values meeting or exceeding it are ill-conditioned.
///          Validation is performed by each consuming factorization rather than
///          by this passive aggregate.
struct FactorizationOptions final {
  /// @brief Rank thresholds used by the factorization.
  RankTolerance rankTolerance{};
  /// @brief Exclusive upper bound for the estimated condition number.
  Scalar conditionLimit{1.0e10};
};

/// @brief Options specific to symmetric matrix factorizations.
/// @details The symmetry tolerance is applied before a positive-definiteness
///          decision. All fields remain passive until consumed by an
///          algorithm.
struct SymmetricFactorizationOptions final {
  /// @brief General factorization thresholds.
  FactorizationOptions factorization{};
  /// @brief Tolerance used to compare mirrored matrix entries.
  Tolerance symmetryTolerance{.absolute = 0.0, .relative = 1.0e-12};
};

/// @brief Classifies an estimated matrix condition number.
enum class ConditionClassification : std::uint8_t {
  /// @brief The estimate is strictly below the configured limit.
  WithinLimit,
  /// @brief The estimate meets or exceeds the configured limit.
  IllConditioned
};

/// @brief Reports the condition estimate produced by a factorization.
/// @details This is a value object. Algorithms populate all fields together;
///          a default-constructed report is intentionally not provided.
struct ConditionReport final {
  /// @brief Matrix 1-norm used for the estimate.
  Scalar matrixOneNorm;
  /// @brief Estimated matrix 1-norm condition number.
  Scalar estimatedConditionNumber1;
  /// @brief Condition limit used for classification.
  Scalar conditionLimit;
  /// @brief Classification relative to `conditionLimit`; equality is
  ///        ill-conditioned.
  ConditionClassification classification;
};

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_FACTORIZATION_OPTIONS_HPP
