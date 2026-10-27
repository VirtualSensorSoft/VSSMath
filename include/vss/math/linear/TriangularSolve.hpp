#ifndef VSS_MATH_LINEAR_TRIANGULAR_SOLVE_HPP
#define VSS_MATH_LINEAR_TRIANGULAR_SOLVE_HPP

#include <cstdint>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace vss::math::linear {

/// @brief Selects the stored triangular part of a factor.
enum class Triangle : std::uint8_t {
  /// @brief Use entries on and below the diagonal.
  Lower,
  /// @brief Use entries on and above the diagonal.
  Upper
};

/// @brief Selects whether a triangular factor is used directly or transposed.
enum class TransposeMode : std::uint8_t {
  /// @brief Solve with the factor as supplied.
  None,
  /// @brief Solve with the transpose of the factor.
  Transpose
};

/// @brief Selects whether stored diagonal entries are used or treated as one.
enum class DiagonalMode : std::uint8_t {
  /// @brief Read and validate the stored diagonal entries.
  Explicit,
  /// @brief Treat every diagonal entry as one without reading it.
  Unit
};

/// @brief Options controlling one triangular solve.
/// @details The rank tolerance is passive until the solve validates and
///          consumes it. Unit-diagonal mode does not inspect stored diagonal
///          values.
// NOLINTBEGIN(misc-non-private-member-variables-in-classes): this is a
// normative public aggregate whose fields are part of the configured API.
struct TriangularSolveOptions final {
  /// @brief Selects the stored lower or upper triangle.
  explicit TriangularSolveOptions(Triangle selectedTriangle) noexcept
      : triangle(selectedTriangle) {}

  /// @brief Stored triangle selected for the factor.
  Triangle triangle;
  /// @brief Whether to use the factor directly or transpose it.
  TransposeMode transpose{TransposeMode::None};
  /// @brief Whether to use explicit or unit diagonal entries.
  DiagonalMode diagonal{DiagonalMode::Explicit};
  /// @brief Absolute and relative singularity thresholds.
  RankTolerance rankTolerance{};
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

/// @brief Solves a triangular system with one right-hand-side vector.
/// @param factor Non-empty square triangular factor view.
/// @param rightHandSide Right-hand-side vector with one entry per factor row.
/// @param options Triangle, transpose, diagonal, and rank-tolerance policy.
/// @return An owning solution outcome without a condition report, or a failure
///         status according to the documented validation precedence.
/// @note Complexity: O(n²) arithmetic and O(n) result storage.
[[nodiscard]] Result<SolveOutcome<Vector>>
solveTriangular(ConstMatrixView factor, ConstVectorView rightHandSide,
                TriangularSolveOptions options);

/// @brief Solves a triangular system with multiple right-hand-side columns.
/// @param factor Non-empty square triangular factor view.
/// @param rightHandSides Matrix whose rows match the factor and whose columns
///        are processed in increasing order.
/// @param options Triangle, transpose, diagonal, and rank-tolerance policy.
/// @return An owning solution matrix outcome without a condition report, or a
///         failure status according to the documented validation precedence.
/// @note Complexity: O(n² * columns) arithmetic and O(n * columns) storage.
[[nodiscard]] Result<SolveOutcome<Matrix>>
solveTriangular(ConstMatrixView factor, ConstMatrixView rightHandSides,
                TriangularSolveOptions options);

} // namespace vss::math::linear

#endif // VSS_MATH_LINEAR_TRIANGULAR_SOLVE_HPP
