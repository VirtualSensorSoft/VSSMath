#ifndef VSS_MATH_COVARIANCE_PROPAGATION_HPP
#define VSS_MATH_COVARIANCE_PROPAGATION_HPP

#include <functional>
#include <optional>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math {

using MomentEstimate = statistics::MomentEstimate;
using MomentRequest = statistics::MomentRequest;
using Rng = probability::Rng;

/// @brief Center, Jacobian, and symmetric Hessian of a vector map.
/// @details Views and the tensor are borrowed only for the duration of a
///          propagation call. The Hessian shape is [output,input,input].
struct QuadraticMap final {
  /// @brief Input expansion point; has the same extent as Jacobian columns.
  ConstVectorView at_center;
  /// @brief First derivative with shape [output,input].
  ConstMatrixView jacobian;
  /// @brief Borrowed symmetric second derivative with shape
  ///        [output,input,input].
  std::reference_wrapper<const Tensor> hessian;
};

/// @brief Cubic Taylor map and its nonnegative amplitude.
/// @details `thirds` has shape [output,input,input,input] and contains ordinary
///          third derivatives without factorial scaling.
struct CubicMap final {
  /// @brief Center, Jacobian and Hessian shared with the quadratic surrogate.
  QuadraticMap quadratic;
  /// @brief Borrowed symmetric third derivative with shape
  ///        [output,input,input,input].
  std::reference_wrapper<const Tensor> thirds;
  /// @brief Nonnegative scale for the cubic term in the Taylor map.
  Scalar amplitude{};
};

/// @brief Selects how invalid Monte Carlo draws are handled.
enum class InvalidDraw : std::uint8_t { reject, report_conditional };

/// @brief Synchronously draws one complete joint input vector.
/// @details The callback receives the operation's mutable RNG and must not
///          retain it or any returned view. Its vector is owned by the result.
using JointSampler = std::function<Result<Vector>(Rng&)>;

/// @brief Controls bounded Monte Carlo propagation and requested summaries.
/// @details `draws` counts attempts; rejected attempts are never retried.
///          Batch boundaries are floor(i*draws/batches). Quantile probabilities
///          retain caller order and use linear interpolation at rank
///          (n-1)*p. `denominator_floor` is reserved for ratio simulations.
struct SimPolicy final {
  /// @brief Number of joint draws attempted.
  Index draws{};
  /// @brief Number of contiguous batches used for mean-error estimation.
  Index batches{};
  /// @brief Optional componentwise quantile probabilities in [0,1].
  std::optional<ConstVectorView> quantiles = std::nullopt;
  /// @brief Abort on an eligible invalid draw or report a conditional result.
  InvalidDraw failure{InvalidDraw::reject};
  /// @brief Optional C031 moments of the retained mapped outputs.
  std::optional<MomentRequest> output_central_moments = std::nullopt;
  /// @brief Optional ratio-denominator floor, finite and nonnegative.
  Scalar denominator_floor = 0.0;
};

/// @brief Controls the scaled unscented transform and covariance rank policy.
struct SigmaPointRule final {
  /// @brief Positive spread parameter alpha.
  Scalar alpha{};
  /// @brief Covariance-weight correction parameter beta.
  Scalar beta{};
  /// @brief Secondary spread parameter kappa.
  Scalar kappa{};
  /// @brief PSD rank and conditioning policy for the input covariance.
  SolvePolicy solve;
};

/// @brief Propagation estimate plus exact empirical quantiles and MC error.
struct SimulationResult final {
  /// @brief Mean and population covariance of the retained mapped draws.
  PropagationResult estimate;
  /// @brief Componentwise quantiles; columns follow requested probability
  /// order.
  Matrix quantiles;
  /// @brief Batch-mean standard error, empty when its validity conditions fail.
  Vector monte_carlo_error;
  /// @brief Number of valid draws retained in all summaries.
  Index valid;
  /// @brief Number of attempts omitted by the conditional policy.
  Index rejected;
  /// @brief Optional central moments of exactly the retained sample.
  std::optional<MomentEstimate> central_moments = std::nullopt;
};

namespace covariance {

/// @brief Monte Carlo pushforward through a caller-supplied joint law.
/// @param law Synchronous sampler for one joint input draw.
/// @param function Synchronous map from one draw to its output vector.
/// @param policy Attempt, invalid-draw and summary policy.
/// @param rng Mutable caller-owned random stream; it is not rolled back.
/// @param limits Output, workspace and callback ceilings.
/// @return Owned summaries over all retained outputs.
/// @retval ErrorCode::InsufficientSamples No draw was retained.
/// @retval ErrorCode::DimensionMismatch Input or mapped extents changed.
/// @retval ErrorCode::CallbackFailure A callback failed or threw.
/// @retval ErrorCode::ResourceLimit A preflight ceiling was exceeded.
[[nodiscard]] Result<SimulationResult>
monteCarloPropagate(const JointSampler& law,
                    const differentiation::VectorFunction& function,
                    const SimPolicy& policy, Rng& rng, Limits limits);

/// @brief Propagates a Gaussian input through a quadratic Taylor surrogate.
/// @details The centered Gaussian law is an explicit caller premise. Singular
///          positive-semidefinite covariance is supported; no dense fourth
///          moment tensor is formed.
[[nodiscard]] Result<PropagationResult>
propagateGaussianQuadraticTaylor(const QuadraticMap& map,
                                 ConstMatrixView covariance, Limits limits);

/// @brief Propagates a common-support central-moment estimate through a
///        quadratic Taylor surrogate.
/// @details The estimate must use an estimated common center and provide each
///          nonzero contraction's required moments. Componentwise supports and
///          missing selected tuples are rejected.
[[nodiscard]] Result<PropagationResult>
propagateQuadraticTaylor(const QuadraticMap& map, const MomentEstimate& input,
                         Limits limits);

/// @brief Computes the explicitly truncated fourth-amplitude covariance of a
///        cubic Taylor map; it is not the full nonlinear covariance.
/// @details Uses terms through amplitude order four and preserves signed
///          indefinite approximations. It does not infer a tail bound from
///          moments of order four.
[[nodiscard]] Result<PropagationResult>
propagateFourthOrderCovariance(const CubicMap& map, const MomentEstimate& input,
                               Limits limits);

/// @brief Propagates covariance through a scaled symmetric sigma-point rule.
/// @details Uses a symmetric PSD square root and the center followed by
///          positive/negative coordinate points. Signed output covariance is
///          returned without PSD projection.
[[nodiscard]] Result<PropagationResult>
sigmaPointPropagate(ConstVectorView center, ConstMatrixView covariance,
                    const differentiation::VectorFunction& function,
                    const SigmaPointRule& rule, Limits limits);

} // namespace covariance
} // namespace vss::math

#endif // VSS_MATH_COVARIANCE_PROPAGATION_HPP
