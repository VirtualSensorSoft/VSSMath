#ifndef VSS_MATH_COVARIANCE_CONDITIONING_HPP
#define VSS_MATH_COVARIANCE_CONDITIONING_HPP

#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::covariance {

/// @brief Classifies a covariance or covariance-derived result.
enum class CovarianceKind : std::uint8_t {
  /// @brief Every retained output eigenvalue is positive and within the floor.
  positive_definite,
  /// @brief At least one output direction is excluded by the rank cutoff.
  semidefinite_singular,
  /// @brief The supplied matrix has a materially negative eigenvalue.
  indefinite,
  /// @brief Full positive rank is present but below the condition floor.
  ill_conditioned
};

/// @brief Owned covariance result and its spectral diagnostics.
struct CovarianceResult final {
  /// @brief Symmetric covariance matrix in the caller's coordinate order.
  Matrix covariance;
  /// @brief Spectral class of the returned covariance.
  CovarianceKind classification{CovarianceKind::semidefinite_singular};
  /// @brief Number of positive eigenvalues strictly above `cutoff`.
  Index rank{};
  /// @brief Rank threshold in the original covariance units.
  Scalar cutoff{};
  /// @brief Retained minimum-to-maximum eigenvalue ratio for full rank only.
  std::optional<Scalar> reciprocal_condition;
};

/// @brief Reusable covariance whitening map and support diagnostics.
/// @details `map` is the symmetric inverse square root on retained
///          eigendirections and zero on excluded directions. The input value
///          is accepted only when its Euclidean residual outside
///          `support_basis` is within the documented arithmetic allowance.
struct Whitening final {
  /// @brief Owned symmetric inverse-square-root map in original coordinates.
  Matrix map;
  /// @brief The whitened value supplied to `whitenSupported`.
  Vector value;
  /// @brief Number of retained positive covariance eigendirections.
  Index rank{};
  /// @brief Inclusive rank cutoff in the original covariance units.
  Scalar cutoff{};
  /// @brief Euclidean norm of the input component outside retained support.
  Scalar null_residual{};
  /// @brief Orthonormal retained covariance eigenvectors, one per column.
  Matrix support_basis;
};

/// @brief Computes the symmetric Schur complement A - B D⁺ Bᵀ.
/// @param aa Finite square leading block A; borrowed and unchanged.
/// @param ab Finite cross block B with shape A.rows() by D.rows().
/// @param bb Finite square conditioning block D; it may be indefinite.
/// @param policy Rank, retained-condition, and eigensolver policies.
/// @param limits Peak element and byte limits for bounded scratch and output.
/// @return The owned symmetric Schur complement, or a failure without a
///         partial matrix.
/// @throws std::bad_alloc if storage within the supplied limits cannot be
///         allocated.
[[nodiscard]] Result<Matrix> schurComplement(ConstMatrixView aa,
                                             ConstMatrixView ab,
                                             ConstMatrixView bb,
                                             SolvePolicy policy, Limits limits);

/// @brief Conditions selected covariance coordinates on given coordinates.
/// @details The complete joint input is validated as finite, symmetric and
///          positive semidefinite before selecting blocks. Result coordinates
///          follow `keep` order. The returned rank and cutoff describe the
///          conditional covariance spectrum.
/// @param covariance Finite square joint covariance; borrowed and unchanged.
/// @param keep Unique selected coordinate indices.
/// @param given Unique conditioning indices, disjoint from `keep`.
/// @param policy Rank and eigensolver policies.
/// @param limits Peak element and byte limits for bounded scratch and output.
/// @return Owned conditional covariance and spectral classification.
[[nodiscard]] Result<CovarianceResult>
conditionalCovariance(ConstMatrixView covariance, std::span<const Index> keep,
                      std::span<const Index> given, SolvePolicy policy,
                      Limits limits);

/// @brief Whitens a vector only when it lies in the covariance support.
/// @details Forms the symmetric map Q_r diag(1/sqrt(lambda_r)) Q_r^T from
///          retained positive covariance eigenpairs. `reject` fails on rank
///          loss; both supported-subspace policies preserve the projection
///          residual and reject vectors outside support instead of fabricating
///          a finite pseudometric. Zero-rank support admits only the exact zero
///          vector.
/// @param covariance Finite square covariance, borrowed and unchanged.
/// @param value Finite vector with one value per covariance coordinate.
/// @param policy Rank cutoff, retained condition floor and sweep cap.
/// @param limits Peak scratch and output limits.
/// @return Owned map, whitened vector and support diagnostics, or a failure
///         without partial output.
[[nodiscard]] Result<Whitening> whitenSupported(ConstMatrixView covariance,
                                                ConstVectorView value,
                                                SolvePolicy policy,
                                                Limits limits);

/// @brief Computes the squared whitened norm for a supported residual.
/// @details Rejects a vector outside the factor's retained support. The result
///          is `||factor.map * value||_2^2`; no projection is silently applied.
/// @param factor Owned whitening map and orthonormal support basis.
/// @param value Finite residual in the factor's original coordinate system.
/// @return Finite squared Mahalanobis distance or a dimension/support/range
///         failure.
[[nodiscard]] Result<Scalar> mahalanobisSquared(const Whitening& factor,
                                                ConstVectorView value);

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_CONDITIONING_HPP
