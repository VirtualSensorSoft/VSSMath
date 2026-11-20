#ifndef VSS_MATH_LINEAR_STRUCTURED_PRODUCTS_HPP
#define VSS_MATH_LINEAR_STRUCTURED_PRODUCTS_HPP

#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {

/// @brief Square matrix in compact row-major band storage.
/// @details `bands` has `order` rows and `lower + 1 + upper` columns;
///          A(i,j) is stored at bands(i, lower + j - i). Unused edge cells
///          are required to be finite zero.
struct BandMatrix final {
  Index order;
  Index lower;
  Index upper;
  Matrix bands;
};

/// @brief Direct sum of finite square matrices in caller-specified order.
struct BlockDiagonal final {
  Array<Matrix> blocks;
};

namespace linear {

/// @brief Diagonal operator represented by its owned diagonal vector.
struct DiagonalMatrix final {
  Vector diagonal;
};

/// @brief Dense tile at one zero-based block-grid coordinate.
struct MatrixBlock final {
  Index block_row;
  Index block_column;
  Matrix value;
};

/// @brief Rectangular sparse grid of dense matrix blocks.
struct BlockMatrix final {
  Array<Index> row_extents;
  Array<Index> column_extents;
  Array<MatrixBlock> blocks;
};

/// @brief Applies a diagonal operator without dense expansion.
[[nodiscard]] Result<Vector> apply(const DiagonalMatrix& matrix,
                                   ConstVectorView vector, Limits limits);

/// @brief Applies a validated general block-grid matrix without dense
/// expansion.
[[nodiscard]] Result<Vector> apply(const BlockMatrix& matrix,
                                   ConstVectorView vector, Limits limits);

} // namespace linear
} // namespace vss::math

namespace vss::math {

/// @brief Forms the block Kronecker product in row-major matrix order.
/// @param lhs Borrowed finite left matrix.
/// @param rhs Borrowed finite right matrix.
/// @param limits Output element and byte ceilings.
/// @return An owning Kronecker product or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Matrix> kronecker(ConstMatrixView lhs, ConstMatrixView rhs,
                                       Limits limits);

/// @brief Forms the columnwise Kronecker (Khatri--Rao) matrix product.
/// @param lhs Borrowed finite left matrix.
/// @param rhs Borrowed finite right matrix with the same column count.
/// @param limits Output element and byte ceilings.
/// @return An owning columnwise product or a diagnostic failure.
/// @throws std::bad_alloc if permitted result storage cannot be allocated.
[[nodiscard]] Result<Matrix> khatriRao(ConstMatrixView lhs, ConstMatrixView rhs,
                                       Limits limits);

/// @brief Computes the compensated sum of a square matrix's diagonal.
/// @param value Borrowed finite square matrix; the empty 0x0 trace is zero.
/// @return The trace or a shape, finite-input, or range diagnostic.
[[nodiscard]] Result<Scalar> trace(ConstMatrixView value);

} // namespace vss::math

#endif // VSS_MATH_LINEAR_STRUCTURED_PRODUCTS_HPP
