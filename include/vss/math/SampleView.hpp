#ifndef VSS_MATH_SAMPLE_VIEW_HPP
#define VSS_MATH_SAMPLE_VIEW_HPP

#include <cstdint>
#include <span>
#include <vss/math/MatrixView.hpp>

namespace vss::math {

/// @brief Borrowed row population with flattened coordinate metadata.
/// @details Rows are observations and columns are flattened coordinates in
///          row-major order. An empty shape denotes a scalar observation;
///          otherwise its positive extents must multiply to the column count.
///          An empty validity mask means every cell is observed. A nonempty
///          mask has one 0/1 byte per row-major cell. The view owns no storage.
struct SampleView final {
  /// @brief N by d rows of observations in row-major coordinate order.
  ConstMatrixView rows;
  /// @brief Positive logical coordinate extents; empty means scalar d=1.
  std::span<const Index> observation_shape;
  /// @brief Optional row-major cell mask; empty means all cells are observed.
  std::span<const std::uint8_t> valid;
};

} // namespace vss::math

#endif // VSS_MATH_SAMPLE_VIEW_HPP
