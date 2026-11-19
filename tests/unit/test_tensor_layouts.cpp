#include "Test.hpp"

#include <array>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/TensorViews.hpp>

namespace {

using namespace vss::math;

Tensor makeTensor(std::span<const Extent> extents,
                  std::span<const Scalar> values) {
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor = Tensor::create(shape.value(), 0.0, TensorLimits{values.size()});
  VSS_CHECK(tensor.hasValue());
  VSS_CHECK(tensor.value().size() == values.size());
  for (Extent i = 0; i < values.size(); ++i) {
    tensor.value().data()[i] = values[i];
  }
  return std::move(tensor).value();
}

void checkValues(const Tensor& tensor, std::span<const Scalar> expected) {
  VSS_CHECK(tensor.size() == expected.size());
  for (Extent i = 0; i < expected.size(); ++i) {
    VSS_CHECK(tensor.data()[i] == expected[i]);
  }
}

} // namespace

int main() {
  const std::array<Extent, 2> extents{2, 3};
  const std::array<Scalar, 6> values{0, 1, 2, 3, 4, 5};
  const auto input = makeTensor(extents, values);
  const Limits ample{64, 4096, 64};

  const std::array<Extent, 2> begin{0, 1};
  const std::array<Extent, 2> slice_extents{2, 2};
  const std::array<Extent, 2> steps{1, 1};
  const auto sliced = slice(input, begin, slice_extents, steps, ample);
  VSS_CHECK(sliced.hasValue());
  VSS_CHECK(sliced.value().shape().extents()[0] == 2);
  VSS_CHECK(sliced.value().shape().extents()[1] == 2);
  checkValues(sliced.value(), std::array<Scalar, 4>{1, 2, 4, 5});

  const auto square = makeTensor(std::array<Extent, 2>{2, 2},
                                 std::array<Scalar, 4>{1, 2, 3, 4});
  const auto specified_slice = slice(square, std::array<Extent, 2>{0, 1},
                                     std::array<Extent, 2>{2, 1}, steps, ample);
  VSS_CHECK(specified_slice.hasValue());
  checkValues(specified_slice.value(), std::array<Scalar, 2>{2, 4});

  const std::array<Extent, 2> stepped_begin{0, 0};
  const std::array<Extent, 2> stepped_extent{2, 2};
  const std::array<Extent, 2> stepped_step{1, 2};
  const auto stepped =
      slice(input, stepped_begin, stepped_extent, stepped_step, ample);
  VSS_CHECK(stepped.hasValue());
  checkValues(stepped.value(), std::array<Scalar, 4>{0, 2, 3, 5});

  const auto flat = reshape(input, std::array<Extent, 1>{6}, ample);
  VSS_CHECK(flat.hasValue());
  checkValues(flat.value(), values);
  VSS_CHECK(!reshape(input, std::array<Extent, 1>{5}, ample).hasValue());
  VSS_CHECK(reshape(input, std::array<Extent, 1>{6}, Limits{5, 4096, 64})
                .status()
                .code() == ErrorCode::ResourceLimit);

  const auto transposed = permuteAxes(input, std::array<Index, 2>{1, 0}, ample);
  VSS_CHECK(transposed.hasValue());
  VSS_CHECK(transposed.value().shape().extents()[0] == 3);
  VSS_CHECK(transposed.value().shape().extents()[1] == 2);
  checkValues(transposed.value(), std::array<Scalar, 6>{0, 3, 1, 4, 2, 5});

  VSS_CHECK(
      slice(input, begin, slice_extents, std::array<Extent, 2>{1, 0}, ample)
          .status()
          .code() == ErrorCode::InvalidDomain);
  VSS_CHECK(
      slice(input, std::array<Extent, 2>{2, 0}, slice_extents, steps, ample)
          .status()
          .code() == ErrorCode::IndexOutOfRange);
  VSS_CHECK(slice(input, begin, slice_extents, steps, Limits{3, 4096, 64})
                .status()
                .code() == ErrorCode::ResourceLimit);
  VSS_CHECK(
      permuteAxes(input, std::array<Index, 2>{0, 0}, ample).status().code() ==
      ErrorCode::InvalidDomain);
  VSS_CHECK(
      permuteAxes(input, std::array<Index, 2>{0, 2}, ample).status().code() ==
      ErrorCode::IndexOutOfRange);

  const auto rank_three = makeTensor(std::array<Extent, 3>{2, 1, 3},
                                     std::array<Scalar, 6>{1, 2, 3, 4, 5, 6});
  const auto permuted =
      permuteAxes(rank_three, std::array<Index, 3>{2, 1, 0}, ample);
  VSS_CHECK(permuted.hasValue());
  checkValues(permuted.value(), std::array<Scalar, 6>{1, 4, 2, 5, 3, 6});

  const auto scalar =
      makeTensor(std::span<const Extent>{}, std::array<Scalar, 1>{7});
  const auto scalar_copy = permuteAxes(scalar, std::span<const Index>{}, ample);
  VSS_CHECK(scalar_copy.hasValue());
  VSS_CHECK(scalar_copy.value().data()[0] == 7);
  const auto scalar_reshape = reshape(scalar, std::span<const Extent>{}, ample);
  VSS_CHECK(scalar_reshape.hasValue());
  VSS_CHECK(scalar_reshape.value().data()[0] == 7);

  const auto empty_input =
      makeTensor(std::array<Extent, 2>{0, 3}, std::span<const Scalar>{});
  const auto empty_slice = slice(empty_input, std::array<Extent, 2>{0, 0},
                                 std::array<Extent, 2>{0, 2}, steps, ample);
  VSS_CHECK(empty_slice.hasValue());
  VSS_CHECK(empty_slice.value().size() == 0);
  const auto flattened_empty =
      reshape(empty_input, std::array<Extent, 1>{0}, ample);
  VSS_CHECK(flattened_empty.hasValue());
  VSS_CHECK(flattened_empty.value().size() == 0);
  VSS_CHECK(slice(input, begin, std::array<Extent, 1>{2}, steps, ample)
                .status()
                .code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(reshape(scalar, std::array<Extent, 1>{2}, ample).status().code() ==
            ErrorCode::DimensionMismatch);
}
