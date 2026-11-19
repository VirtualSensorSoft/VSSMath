#ifndef VSS_MATH_PACKED_SYMMETRIC_HPP
#define VSS_MATH_PACKED_SYMMETRIC_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>

namespace vss::math {

/// @brief Owning canonical storage for a fully symmetric rank-2 through rank-4
/// tensor.
/// @details Entries are ordered lexicographically by nondecreasing coordinate
///          tuples. For order @c k and dimension @c d, storage contains
///          choose(d+k-1,k) values rather than d^k dense values. The public
///          aggregate may be constructed directly; every operation validates
///          its metadata and storage length before accessing coefficients.
struct PackedSymmetric final {
  /// @brief Tensor order, restricted to 2, 3, or 4.
  Index order{};
  /// @brief Equal positive extent of every tensor axis.
  Index dimension{};
  /// @brief Canonical values for sorted nondecreasing coordinate tuples.
  Vector lexicographic_values{};
};

/// @brief Packs a symmetric dense tensor after checking all permutation orbits.
/// @param input Borrowed finite tensor with equal positive extents on all axes.
/// @param tolerance Absolute-plus-relative symmetry comparison tolerance.
/// @param limits Caps packed scalar elements and output bytes.
/// @return Canonical owning storage or a diagnostic failure.
/// @throws std::bad_alloc if permitted output allocation fails.
[[nodiscard]] Result<PackedSymmetric>
packSymmetric(const Tensor& input, Tolerance tolerance, Limits limits);

/// @brief Expands canonical symmetric storage into its full dense tensor.
/// @param input Borrowed packed aggregate, validated before coefficient access.
/// @param limits Caps dense output elements and bytes.
/// @return An owning dense tensor or a diagnostic failure.
/// @throws std::bad_alloc if permitted output allocation fails.
[[nodiscard]] Result<Tensor> unpackSymmetric(const PackedSymmetric& input,
                                             Limits limits);

/// @brief Computes the full dense inner product without materializing either
/// tensor.
/// @param lhs Left packed aggregate.
/// @param rhs Right packed aggregate with matching order and dimension.
/// @return The orbit-weighted contraction or a diagnostic failure.
[[nodiscard]] Result<Scalar> symmetricContraction(const PackedSymmetric& lhs,
                                                  const PackedSymmetric& rhs);

} // namespace vss::math

#endif // VSS_MATH_PACKED_SYMMETRIC_HPP
