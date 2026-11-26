#ifndef VSS_MATH_COVARIANCE_SENSITIVITY_HPP
#define VSS_MATH_COVARIANCE_SENSITIVITY_HPP

#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/covariance/Budget.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::covariance {

/// @brief Borrowed set of independent scalar input marginals.
struct IndependentMarginals final {
  /// Borrowed immutable marginal laws, one for each independent input.
  std::span<const probability::Distribution> marginals;
};

/// @brief Sampling and estimator requests for replicated Sobol indices.
struct SensitivitySpec final {
  /// @brief A/B row count in each independent replicate; at least two.
  Index draws{};
  /// @brief Independent replicate count used to estimate Monte Carlo error.
  Index replicates{};
  /// @brief Required output extent, used to preflight all owned result storage.
  Index output_dimension{};
  /// @brief Whether first-order indices are returned.
  bool first_order{true};
  /// @brief Whether total-order indices are returned.
  bool total_order{true};
  /// @brief Marginal sampling accuracy and random-draw budget.
  ProbabilityPolicy sampling{};
};

/// @brief Owns requested first/total Sobol indices and replicate errors.
struct SensitivityResult final {
  /// @brief First-order values, shape outputs by input dimension; 0x0 if
  /// absent.
  Matrix first_order;
  /// @brief Total-order values, shape outputs by input dimension; 0x0 if
  /// absent.
  Matrix total_order;
  /// @brief Replicate standard errors in requested estimator row order.
  Matrix monte_carlo_error;
};

/// @brief Estimates independent-input first- and total-order Sobol indices.
/// @details Uses paired A/B rows and AB_j hybrids, pooled A/B population output
///          variance, Saltelli first-order and Jansen total-order estimators.
///          It never clips finite estimates. Replicate standard errors are the
///          sample standard deviation divided by sqrt(replicates), not a
///          confidence guarantee. All callback evaluations and random draws
///          are preflighted; a failed or dimension-changing callback aborts.
[[nodiscard]] Result<SensitivityResult>
sensitivityIndices(const differentiation::VectorFunction& function,
                   const IndependentMarginals& law, const SensitivitySpec& spec,
                   probability::Rng& rng, Limits limits);

/// @brief Compatibility spelling for the source-pair uncertainty budget.
[[nodiscard]] Result<UncertaintyBudget>
uncertaintyBudget(ConstMatrixView jacobian,
                  std::span<const SourceBlock> sources,
                  std::span<const SourceCrossBlock> cross,
                  DependencePolicy dependence, Limits limits);

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_SENSITIVITY_HPP
