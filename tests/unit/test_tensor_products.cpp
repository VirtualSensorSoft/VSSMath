#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Scalar;
using vss::math::Tensor;
using vss::math::TensorLimits;
using vss::math::TensorShape;

Tensor makeTensor(std::span<const Extent> extents,
                  std::span<const Scalar> values) {
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor = Tensor::create(shape.value(), 0.0, TensorLimits{64U});
  VSS_CHECK(tensor.hasValue());
  VSS_CHECK(tensor.value().size() == values.size());
  for (Extent index = 0U; index < values.size(); ++index) {
    tensor.value().data()[index] = values[index];
  }
  return std::move(tensor).value();
}

} // namespace

int main() {
  using vss::math::contract;
  using vss::math::outer;

  const std::array<Extent, 1> vector_shape{2U};
  const std::array<Scalar, 2> lhs_values{1.0, 2.0};
  const std::array<Scalar, 2> rhs_values{3.0, 4.0};
  const Tensor lhs = makeTensor(vector_shape, lhs_values);
  const Tensor rhs = makeTensor(vector_shape, rhs_values);

  const auto product = outer(lhs, rhs, TensorLimits{4U});
  VSS_CHECK(product.hasValue());
  const std::array<Extent, 2> expected_product_shape{2U, 2U};
  VSS_CHECK(product.value().shape().extents()[0] == expected_product_shape[0]);
  VSS_CHECK(product.value().shape().extents()[1] == expected_product_shape[1]);
  VSS_CHECK(product.value().at(std::array<Extent, 2>{0U, 0U}) == 3.0);
  VSS_CHECK(product.value().at(std::array<Extent, 2>{0U, 1U}) == 4.0);
  VSS_CHECK(product.value().at(std::array<Extent, 2>{1U, 0U}) == 6.0);
  VSS_CHECK(product.value().at(std::array<Extent, 2>{1U, 1U}) == 8.0);

  const std::array<Extent, 1> lhs_axis{0U};
  const std::array<Extent, 1> rhs_axis{0U};
  const auto dot = contract(lhs, rhs, lhs_axis, rhs_axis, TensorLimits{1U});
  VSS_CHECK(dot.hasValue());
  VSS_CHECK(dot.value().shape().rank() == 0U);
  VSS_CHECK(dot.value().data()[0] == 11.0);

  const auto no_axes = contract(lhs, rhs, std::span<const Extent>{},
                                std::span<const Extent>{}, TensorLimits{4U});
  VSS_CHECK(no_axes.hasValue());
  VSS_CHECK(no_axes.value().shape().rank() == 2U);
  for (Extent index = 0U; index < product.value().size(); ++index) {
    VSS_CHECK(no_axes.value().data()[index] == product.value().data()[index]);
  }

  const auto over_budget = outer(lhs, rhs, TensorLimits{3U});
  VSS_CHECK(!over_budget.hasValue());
  VSS_CHECK(over_budget.status().code() == ErrorCode::ResourceLimit);

  const auto repeated =
      contract(lhs, lhs, std::array<Extent, 2>{0U, 0U},
               std::array<Extent, 2>{0U, 0U}, TensorLimits{1U});
  VSS_CHECK(!repeated.hasValue());
  VSS_CHECK(repeated.status().code() == ErrorCode::InvalidExtent);

  const auto lhs_out_of_range =
      contract(lhs, rhs, std::array<Extent, 1>{1U}, rhs_axis, TensorLimits{4U});
  VSS_CHECK(!lhs_out_of_range.hasValue());
  VSS_CHECK(lhs_out_of_range.status().code() == ErrorCode::IndexOutOfRange);

  const auto axis_count_mismatch =
      contract(lhs, rhs, lhs_axis, std::span<const Extent>{}, TensorLimits{4U});
  VSS_CHECK(!axis_count_mismatch.hasValue());
  VSS_CHECK(axis_count_mismatch.status().code() ==
            ErrorCode::DimensionMismatch);

  const auto extent_mismatch =
      contract(lhs,
               makeTensor(std::array<Extent, 1>{3U},
                          std::array<Scalar, 3>{1.0, 2.0, 3.0}),
               lhs_axis, rhs_axis, TensorLimits{4U});
  VSS_CHECK(!extent_mismatch.hasValue());
  VSS_CHECK(extent_mismatch.status().code() == ErrorCode::DimensionMismatch);

  const Tensor left_matrix =
      makeTensor(std::array<Extent, 2>{2U, 3U},
                 std::array<Scalar, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const Tensor right_matrix =
      makeTensor(std::array<Extent, 2>{3U, 4U},
                 std::array<Scalar, 12>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                        9.0, 10.0, 11.0, 12.0});
  const auto matrix_product =
      contract(left_matrix, right_matrix, std::array<Extent, 1>{1U},
               std::array<Extent, 1>{0U}, TensorLimits{8U});
  VSS_CHECK(matrix_product.hasValue());
  VSS_CHECK(matrix_product.value().shape().rank() == 2U);
  VSS_CHECK(matrix_product.value().shape().extent(0U) == 2U);
  VSS_CHECK(matrix_product.value().shape().extent(1U) == 4U);
  VSS_CHECK(matrix_product.value().at(std::array<Extent, 2>{0U, 0U}) == 38.0);
  VSS_CHECK(matrix_product.value().at(std::array<Extent, 2>{0U, 3U}) == 56.0);
  VSS_CHECK(matrix_product.value().at(std::array<Extent, 2>{1U, 0U}) == 83.0);
  VSS_CHECK(matrix_product.value().at(std::array<Extent, 2>{1U, 3U}) == 128.0);
  const Tensor paired_axes_left =
      makeTensor(std::array<Extent, 2>{2U, 3U},
                 std::array<Scalar, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const Tensor paired_axes_right =
      makeTensor(std::array<Extent, 2>{3U, 2U},
                 std::array<Scalar, 6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
  const auto reordered_full_contraction = contract(
      paired_axes_left, paired_axes_right, std::array<Extent, 2>{1U, 0U},
      std::array<Extent, 2>{0U, 1U}, TensorLimits{1U});
  VSS_CHECK(reordered_full_contraction.hasValue());
  VSS_CHECK(reordered_full_contraction.value().shape().rank() == 0U);
  VSS_CHECK(reordered_full_contraction.value().data()[0] == 86.0);
  const auto contraction_over_budget =
      contract(left_matrix, right_matrix, std::array<Extent, 1>{1U},
               std::array<Extent, 1>{0U}, TensorLimits{7U});
  VSS_CHECK(!contraction_over_budget.hasValue());
  VSS_CHECK(contraction_over_budget.status().code() ==
            ErrorCode::ResourceLimit);

  const Tensor scalar =
      makeTensor(std::span<const Extent>{}, std::array<Scalar, 1>{2.0});
  const auto scalar_product = outer(scalar, lhs, TensorLimits{2U});
  VSS_CHECK(scalar_product.hasValue());
  VSS_CHECK(scalar_product.value().shape().extents().size() == 1U);
  VSS_CHECK(scalar_product.value().data()[0] == 2.0);
  VSS_CHECK(scalar_product.value().data()[1] == 4.0);

  const auto scalar_contract =
      contract(scalar, lhs, std::span<const Extent>{},
               std::span<const Extent>{}, TensorLimits{2U});
  VSS_CHECK(scalar_contract.hasValue());
  VSS_CHECK(scalar_contract.value().shape().rank() == 1U);
  VSS_CHECK(scalar_contract.value().data()[1] == 4.0);

  const Tensor zero_extent =
      makeTensor(std::array<Extent, 1>{0U}, std::span<const Scalar>{});
  const auto zero_outer = outer(scalar, zero_extent, TensorLimits{0U});
  VSS_CHECK(zero_outer.hasValue());
  VSS_CHECK(zero_outer.value().size() == 0U);

  const Tensor max_scalar =
      makeTensor(std::span<const Extent>{},
                 std::array<Scalar, 1>{std::numeric_limits<Scalar>::max()});
  const auto non_finite_result = outer(max_scalar, scalar, TensorLimits{1U});
  VSS_CHECK(!non_finite_result.hasValue());
  VSS_CHECK(non_finite_result.status().code() == ErrorCode::NonFiniteResult);
  const Tensor nan_scalar = makeTensor(
      std::span<const Extent>{},
      std::array<Scalar, 1>{std::numeric_limits<Scalar>::quiet_NaN()});
  const auto non_finite_input = outer(nan_scalar, scalar, TensorLimits{1U});
  VSS_CHECK(!non_finite_input.hasValue());
  VSS_CHECK(non_finite_input.status().code() == ErrorCode::NonFiniteInput);

  const auto overflow_shape = TensorShape::create(
      std::array<Extent, 2>{std::numeric_limits<Extent>::max(), 2U});
  VSS_CHECK(!overflow_shape.hasValue());
  VSS_CHECK(overflow_shape.status().code() == ErrorCode::SizeOverflow);
}
