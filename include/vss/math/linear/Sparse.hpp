#ifndef VSS_MATH_LINEAR_SPARSE_HPP
#define VSS_MATH_LINEAR_SPARSE_HPP

#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math::linear {

/// @brief One coordinate/value entry in a sparse matrix assembly input.
struct Triplet final {
  /// @brief Zero-based row coordinate.
  Index row;
  /// @brief Zero-based column coordinate.
  Index column;
  /// @brief Finite value to add at the coordinate.
  Scalar value;
};

/// @brief Non-owning immutable input range of sparse matrix triplets.
using TripletView = std::span<const Triplet>;

/// @brief Canonical compressed sparse row representation of a real matrix.
/// @details Arrays are immutable owning values. Public fields permit aggregate
///          construction, so numerical operations validate the CSR invariants
///          before using them. Every row has strictly increasing column
///          indices and every stored value is finite and nonzero.
struct CSR final {
  /// @brief Number of matrix rows.
  Extent rows;
  /// @brief Number of matrix columns.
  Extent columns;
  /// @brief Row offsets, of length `rows + 1`, with first zero and last nnz.
  Array<Index> row_offsets;
  /// @brief Strictly increasing column index within each row.
  Array<Index> column_indices;
  /// @brief Finite, nonzero values corresponding to column indices.
  Array<Scalar> values;
};

/// @brief Builds canonical CSR by sorting and coalescing coordinate triplets.
/// @param rows Matrix row count.
/// @param columns Matrix column count.
/// @param entries Borrowed triplets; the input is not modified or retained.
/// @param limits Maximum output/workspace elements and bytes.
/// @return Canonical CSR, or a diagnostic for invalid dimensions/coordinates,
///         non-finite input or result, or exhausted resource limits.
/// @details Coordinates are sorted lexicographically. Duplicate values are
///          summed using scaled compensated arithmetic; exact-zero sums are
///          omitted. Empty rows and zero-sized dimensions are supported.
///          Complexity is O(z log z + rows) time and O(z + rows) storage for
///          z input entries.
[[nodiscard]] Result<CSR> sparseCSRFromTriplets(Extent rows, Extent columns,
                                                TripletView entries,
                                                Limits limits);

/// @brief Computes a deterministic sparse matrix-vector product.
/// @param matrix Canonical CSR matrix. Its invariants are checked at runtime.
/// @param vector Borrowed vector whose size equals `matrix.columns`.
/// @param limits Maximum output and workspace resources.
/// @return An owning result vector, or a diagnostic for malformed CSR,
///         incompatible shape, non-finite input/result, or resource limits.
/// @details Rows are evaluated in increasing order. Each dot product uses
///          exponent-scaled compensated accumulation, allowing finite
///          products that overflow in isolation to cancel before the final
///          result is rounded. Complexity is O(columns + nnz + rows + 1).
[[nodiscard]] Result<Vector> apply(const CSR& matrix, ConstVectorView vector,
                                   Limits limits);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_SPARSE_HPP
