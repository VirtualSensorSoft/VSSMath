#ifndef VSS_MATH_STATISTICS_RATIO_ESTIMATE_HPP
#define VSS_MATH_STATISTICS_RATIO_ESTIMATE_HPP

#include <functional>
#include <optional>
#include <span>
#include <vss/math/Tensor.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Bootstrap.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {

/// @brief Owns a vector-valued result, its Jacobian, and optional Hessian.
/// @details Jacobian rows follow output coordinates and columns follow the
///          flattened joint input. Hessian axes are `[output,input,input]`.
struct VectorDifferential final {
  /// @brief Function value at the supplied nominal input.
  Vector value;
  /// @brief First derivative in output-row/input-column order.
  Matrix jacobian;
  /// @brief Optional second derivative in `[output,input,input]` order.
  std::optional<Tensor> hessian;
};

/// @brief Joint latent-variable model for an observation-weighted ratio.
/// @details Both callbacks receive the same flattened vector ordered as
///          observation values, one scalar weight per row, then shared
///          parameters. They return row-major `rows × components` values and
///          `rows` scalar weights respectively.
struct RatioModel final {
  /// @brief Number of paired observation and weight rows.
  Index rows;
  /// @brief Borrowed observation shape; empty denotes one scalar component.
  std::span<const Index> observation_shape;
  /// @brief Number of trailing shared latent parameters.
  Index parameter_count;
  /// @brief Computes all flattened observation components from the joint input.
  differentiation::VectorFunction observations;
  /// @brief Computes one nonnegative scalar weight per row.
  differentiation::VectorFunction weights;
};

/// @brief Supplies the joint nominal value, covariance and optional moments.
struct JointUncertaintyInput final {
  /// @brief Borrowed latent center in the documented flattened order.
  ConstVectorView center;
  /// @brief Borrowed covariance over all joint coordinates.
  ConstMatrixView covariance;
  /// @brief Optional higher central moments for a separately supported method.
  std::optional<MomentEstimate> higher;
};

/// @brief Selects weighted-ratio propagation and denominator admissibility.
struct PropagationPolicy final {
  /// @brief Approximation requested; this API currently supports first order.
  Approximation method{Approximation::first_order};
  /// @brief The nominal and every derivative stencil denominator must exceed
  /// it.
  Scalar denominator_floor{};
  /// @brief Rank and iteration settings for covariance validation.
  SolvePolicy solve{};
};

/// @brief Callback that produces one whole, paired latent input draw.
using JointSampler = std::function<Result<Vector>(probability::Rng&)>;

/// @brief Selects bounded simulation summaries and invalid-draw handling.
struct SimPolicy final {
  /// @brief Number of joint draws attempted exactly once each.
  Index draws;
  /// @brief Equal attempt batches used for Monte Carlo error; below two omits
  /// it.
  Index batches;
  /// @brief Componentwise empirical probabilities retained in caller order.
  std::span<const Scalar> quantiles;
  /// @brief Reject the first invalid draw or report the retained conditional
  /// law.
  InvalidDraw failure;
  /// @brief Optional output-space empirical central moment request.
  std::optional<MomentRequest> output_central_moments;
  /// @brief A draw is admissible only when its total weight is strictly larger.
  Scalar denominator_floor;
};

/// @brief Owns Monte Carlo estimates and joint-draw support diagnostics.
struct SimulationResult final {
  /// @brief Empirical output mean and population covariance.
  PropagationResult estimate;
  /// @brief Component rows and caller-ordered quantile columns.
  Matrix quantiles;
  /// @brief Batch-means standard errors; empty when not estimable/requested.
  Vector monte_carlo_error;
  /// @brief Accepted joint draws.
  Index valid;
  /// @brief Rejected joint draws; no replacement draw is generated.
  Index rejected;
  /// @brief Optional output-space empirical central moments.
  std::optional<MomentEstimate> central_moments;
};

/// @brief Computes a weighted coordinate mean and exact first two derivatives.
/// @param values Complete paired observation rows, one flattened vector per
/// row.
/// @param weights One finite nonnegative scalar weight per row.
/// @param limits Peak owned element/byte ceilings and callback budget.
/// @return Value, Jacobian and Hessian in `[values,weights]` coordinate order.
/// @details Uses scaled nonnegative weights. The denominator must be positive;
///          all rows, including zero-weight rows, must be finite. Complexity
///          and derivative storage are bounded before output allocation.
[[nodiscard]] Result<VectorDifferential>
weightedRatioDifferential(const SampleView& values, ConstVectorView weights,
                          Limits limits);

/// @brief Propagates a joint covariance through the weighted-ratio map.
/// @details Evaluates both callbacks on one joint center, differentiates the
///          composed quotient with the core finite-difference policy, and
///          propagates the complete covariance including cross blocks. Every
///          stencil must independently satisfy the strict denominator floor.
[[nodiscard]] Result<PropagationResult>
propagateWeightedRatio(const RatioModel& model,
                       const JointUncertaintyInput& uncertainty,
                       const PropagationPolicy& policy, Limits limits);

/// @brief Simulates paired joint inputs and summarizes the retained ratios.
/// @details Calls the sampler once per attempt and never resamples values and
///          weights independently. Conditional summaries use one shared set of
///          accepted draws; invalid samples are never silently repaired.
[[nodiscard]] Result<SimulationResult>
simulateWeightedRatio(const RatioModel& model, const JointSampler& sampler,
                      const SimPolicy& policy, probability::Rng& rng,
                      Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_RATIO_ESTIMATE_HPP
