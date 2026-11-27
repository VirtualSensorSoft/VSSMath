#ifndef VSS_MATH_TENSOR_DECOMPOSITION_HPP
#define VSS_MATH_TENSOR_DECOMPOSITION_HPP

#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {

/// @brief Owning multilinear factors produced by truncated HOSVD.
/// @details `axes[j]` is an `n_j` by `rank_j` matrix whose columns are the
///          retained left singular vectors of the original mode-j unfolding.
///          `core` is the input projected onto all retained axes. `shape`
///          preserves the original tensor extents. `residual_norm` is the
///          Frobenius norm of the actual reconstruction error, not a bound
///          and not an optimality certificate. All members own their data.
struct TuckerFactors final {
  /// @brief Tensor projected onto all retained mode subspaces.
  Tensor core;
  /// @brief Retained orthonormal mode bases in original axis order.
  Array<Matrix> axes;
  /// @brief Original input shape, retained for reconstruction.
  TensorShape shape;
  /// @brief Frobenius norm of input minus the Tucker reconstruction.
  Scalar residual_norm;
};

/// @brief Computes truncated higher-order singular-value decomposition.
/// @param input Borrowed finite tensor; it is not modified.
/// @param ranks One requested multilinear rank per input axis.
/// @param policy SVD rank and convergence policy, applied independently to
///               each original mode unfolding.
/// @param limits Maximum operation-owned element and byte peak.
/// @return Tucker factors and the measured reconstruction residual, or a
///         failure with no partial factors. Failures include
///         `DimensionMismatch` for rank-count mismatch, `IndexOutOfRange` for
///         a rank above its mode extent, `InvalidDomain` for a zero rank on a
///         nonempty mode, `InvalidTolerance`, `NonFiniteInput`,
///         `NonFiniteResult`, `NoConvergence`, and `ResourceLimit`.
/// @throws std::bad_alloc if the planned storage is admitted but unavailable.
/// @details Mode-j unfolding rows correspond to axis j and columns enumerate
///          all other axes in original row-major order. The independent compact
///          SVD is applied to each original unfolding. HOSVD is not a globally
///          optimal fixed-rank approximation; in exact arithmetic its error is
///          bounded by the square root of the sum of discarded singular-value
///          tail energies. For nonempty tensors all requested ranks are
///          positive; a zero-sized tensor permits zero rank only on empty
///          axes. A rank-zero tensor (scalar) uses empty ranks and axes.
/// @note Complexity is O(sum_j D*min(n_j,D/n_j)) arithmetic for D input
///       elements. Peak storage preflight includes each unfolding, the full
///       thin-SVD basis/workspace, retained axes, core contractions, the
///       reconstruction used for residual measurement, and simultaneously
///       live intermediates. Inputs are borrowed and are not charged.
[[nodiscard]] Result<TuckerFactors> tuckerHOSVD(const Tensor& input,
                                                std::span<const Extent> ranks,
                                                SolvePolicy policy,
                                                Limits limits);

/// @brief Reconstructs a tensor from Tucker core and mode factors.
/// @param factors Borrowed, fully owning Tucker factors; they are not modified.
/// @param limits Maximum operation-owned element and byte peak.
/// @return An owning tensor with the original shape, or a failure without a
///         partial tensor. Inconsistent shapes return `DimensionMismatch`,
///         invalid residual provenance returns `InvalidDomain`, non-finite
///         factors/core return `NonFiniteInput`, non-finite reconstructed
///         values return `NonFiniteResult`, and insufficient storage returns
///         `ResourceLimit`.
/// @throws std::bad_alloc if the planned storage is admitted but unavailable.
/// @details Validates all public factor shapes and finite values before
///          allocating output. `residual_norm` must be finite and nonnegative
///          but is provenance only and is never trusted to compute output.
[[nodiscard]] Result<Tensor> reconstructTucker(const TuckerFactors& factors,
                                               Limits limits);

} // namespace vss::math

#endif // VSS_MATH_TENSOR_DECOMPOSITION_HPP
