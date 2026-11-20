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
  positive_definite,
  semidefinite_singular,
  indefinite,
  ill_conditioned
};

/// @brief Owned covariance result and its spectral diagnostics.
struct CovarianceResult final {
  Matrix covariance;
  CovarianceKind classification{CovarianceKind::semidefinite_singular};
  Index rank{};
  Scalar cutoff{};
  std::optional<Scalar> reciprocal_condition;
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

} // namespace vss::math::covariance

#endif // VSS_MATH_COVARIANCE_CONDITIONING_HPP
