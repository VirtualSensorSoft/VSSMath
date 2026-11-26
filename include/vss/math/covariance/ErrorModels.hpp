#ifndef VSS_MATH_COVARIANCE_ERROR_MODELS_HPP
#define VSS_MATH_COVARIANCE_ERROR_MODELS_HPP

#include <functional>
#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Budget.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::covariance {

/// @brief Evaluates a covariance model at a borrowed state and time.
/// @details The returned matrix owns its storage. The state view is valid only
///          for the duration of the callback and must not be retained.
using ErrorCovarianceEvaluator =
    std::function<Result<Matrix>(ConstVectorView, Scalar)>;

/// @brief Selects either a conditioned model or a block covariance model.
/// @details In block mode, `diagonal` is an independent output variance floor;
///          `loading` maps source coordinates to output coordinates and the
///          source blocks describe their complete covariance. The represented
///          covariance is D + A C A^T. Callback mode requires all block fields
///          to be absent.
struct ErrorCovarianceSpec final {
  /// @brief Optional evaluator selected instead of the block representation.
  ErrorCovarianceEvaluator evaluate;
  /// @brief Optional nonnegative independent output variances.
  std::optional<ConstVectorView> diagonal;
  /// @brief Optional loading matrix mapping concatenated source coordinates.
  std::optional<ConstMatrixView> loading;
  /// @brief Source covariance blocks partitioning loading columns.
  std::span<const SourceBlock> sources;
  /// @brief Declared cross-covariance blocks between source identities.
  std::span<const SourceCrossBlock> cross;
  /// @brief Explicit completeness rule for source dependence.
  DependencePolicy dependence{DependencePolicy::exact_joint};
};

/// @brief Constructs a conditioned or block measurement-error covariance.
/// @details The callback branch invokes the evaluator once after the evaluation
///          budget is checked. The block branch computes D + A C A^T through
///          the source-pair budget API. State and time are validated in both
///          branches; in block mode they do not alter the result. Results are
///          classified without clipping, and materially indefinite covariance
///          inputs fail with `NotPositiveSemidefinite`.
[[nodiscard]] Result<CovarianceResult>
assembleErrorCovariance(const ErrorCovarianceSpec& spec, ConstVectorView state,
                        Scalar time, Limits limits);

/// @brief Subtracts a known independent additive error contribution.
/// @details Both inputs must be finite, symmetric PSD matrices on identical
///          estimator support. The signed difference is retained and may be
///          classified indefinite; negative eigenvalues are never projected
///          away.
[[nodiscard]] Result<CovarianceResult>
subtractKnownErrorCovariance(ConstMatrixView observed,
                             ConstMatrixView knownError, SolvePolicy policy,
                             Limits limits);

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_ERROR_MODELS_HPP
