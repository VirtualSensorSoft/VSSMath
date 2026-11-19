#include "Test.hpp"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/PackedSymmetric.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Tolerance.hpp>

namespace {

using namespace vss::math;

Tensor makeTensor(std::span<const Extent> extents,
                  std::span<const Scalar> values) {
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor = Tensor::create(shape.value(), 0.0, TensorLimits{values.size()});
  VSS_CHECK(tensor.hasValue());
  VSS_CHECK(tensor.value().size() == values.size());
  const std::span<Scalar> tensorValues(tensor.value().data(),
                                       tensor.value().size());
  for (Extent i = 0; i < values.size(); ++i) {
    tensorValues[i] = values[i];
  }
  return std::move(tensor).value();
}

void checkTensorEqual(const Tensor& lhs, const Tensor& rhs) {
  VSS_CHECK(lhs.shape().extents().size() == rhs.shape().extents().size());
  for (Extent axis = 0; axis < lhs.shape().rank(); ++axis) {
    VSS_CHECK(lhs.shape().extent(axis) == rhs.shape().extent(axis));
  }
  VSS_CHECK(lhs.size() == rhs.size());
  const std::span<const Scalar> lhsValues(lhs.data(), lhs.size());
  const std::span<const Scalar> rhsValues(rhs.data(), rhs.size());
  for (Extent i = 0; i < lhs.size(); ++i) {
    VSS_CHECK(lhsValues[i] == rhsValues[i]);
  }
}

} // namespace

int main() try {
  const Limits ample{
      .elements = 20'000, .bytes = 1'000'000, .evaluations = 100'000};
  const auto matrix = makeTensor(std::array<Extent, 2>{2, 2},
                                 std::array<Scalar, 4>{1, 2, 2, 4});
  const auto packed_result =
      packSymmetric(matrix, Tolerance{.absolute = 0, .relative = 0}, ample);
  VSS_CHECK(packed_result.hasValue());
  const auto& packed = packed_result.value();
  VSS_CHECK(packed.order == 2);
  VSS_CHECK(packed.dimension == 2);
  VSS_CHECK(packed.lexicographic_values.size() == 3);
  VSS_CHECK(packed.lexicographic_values.at(0) == 1);
  VSS_CHECK(packed.lexicographic_values.at(1) == 2);
  VSS_CHECK(packed.lexicographic_values.at(2) == 4);
  const auto contraction = symmetricContraction(packed, packed);
  VSS_CHECK(contraction.hasValue());
  VSS_CHECK(contraction.value() == 25);
  const auto restored = unpackSymmetric(packed, ample);
  VSS_CHECK(restored.hasValue());
  checkTensorEqual(restored.value(), matrix);

  auto asymmetric = makeTensor(std::array<Extent, 2>{2, 2},
                               std::array<Scalar, 4>{1, 2, 3, 4});
  VSS_CHECK(
      packSymmetric(asymmetric, Tolerance{.absolute = 0, .relative = 0}, ample)
          .status()
          .code() == ErrorCode::NotSymmetric);
  std::span<Scalar>(asymmetric.data(), asymmetric.size())[2] = 2.0 + 1e-8;
  VSS_CHECK(packSymmetric(asymmetric,
                          Tolerance{.absolute = 1e-7, .relative = 0}, ample)
                .hasValue());

  VSS_CHECK(
      unpackSymmetric(packed, Limits{1, 1'000'000, 100'000}).status().code() ==
      ErrorCode::ResourceLimit);
  VSS_CHECK(symmetricContraction(packed, PackedSymmetric{2, 1, Vector(1)})
                .status()
                .code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(unpackSymmetric(PackedSymmetric{2, 2, Vector(2)}, ample)
                .status()
                .code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(unpackSymmetric(PackedSymmetric{1, 2, Vector(1)}, ample)
                .status()
                .code() == ErrorCode::InvalidExtent);
  VSS_CHECK(
      packSymmetric(matrix, Tolerance{.absolute = -1, .relative = 0}, ample)
          .status()
          .code() == ErrorCode::InvalidTolerance);
  auto nonfinite = makeTensor(
      std::array<Extent, 2>{2, 2},
      std::array<Scalar, 4>{1, 2, 2, std::numeric_limits<Scalar>::infinity()});
  VSS_CHECK(
      packSymmetric(nonfinite, Tolerance{.absolute = 0, .relative = 0}, ample)
          .status()
          .code() == ErrorCode::NonFiniteInput);

  for (const Index dimension : {1U, 2U}) {
    for (const Index order : {2U, 3U, 4U}) {
      std::vector<Extent> shape(order, dimension);
      Extent denseCount = 1U;
      for (Index axis = 0U; axis < order; ++axis) {
        denseCount *= dimension;
      }
      const Extent packedCount = dimension == 1U ? 1U : (order + 1U);
      std::vector<Scalar> values(denseCount, 1.0);
      const auto dense = makeTensor(shape, values);
      const auto canonical =
          packSymmetric(dense, Tolerance{.absolute = 0, .relative = 0}, ample);
      VSS_CHECK(canonical.hasValue());
      VSS_CHECK(canonical.value().lexicographic_values.size() == packedCount);
    }
  }

  std::vector<Extent> rank4_shape(4, 10U);
  std::vector<Scalar> all_ones(10'000U, 1.0);
  const auto rank4 = makeTensor(rank4_shape, all_ones);
  const auto packed_rank4 =
      packSymmetric(rank4, Tolerance{.absolute = 0, .relative = 0}, ample);
  VSS_CHECK(packed_rank4.hasValue());
  VSS_CHECK(packed_rank4.value().lexicographic_values.size() == 715);

  std::vector<Scalar> mixed_values(16U);
  for (Extent i = 0U; i < mixed_values.size(); ++i) {
    Index ones = 0U;
    Extent coordinate = i;
    std::array<Index, 4> tuple{};
    const std::span<Index> tupleValues(tuple);
    for (Index axis = 4U; axis > 0U; --axis) {
      tupleValues[axis - 1U] = coordinate % 2U;
      coordinate /= 2U;
    }
    std::ranges::sort(tuple);
    ones = std::accumulate(tuple.begin(), tuple.end(), Index{0});
    mixed_values[i] = static_cast<Scalar>(ones + 1U);
  }
  const auto mixed =
      makeTensor(std::array<Extent, 4>{2, 2, 2, 2}, mixed_values);
  const auto mixed_packed =
      packSymmetric(mixed, Tolerance{.absolute = 0, .relative = 0}, ample);
  VSS_CHECK(mixed_packed.hasValue());
  const auto mixed_restored = unpackSymmetric(mixed_packed.value(), ample);
  VSS_CHECK(mixed_restored.hasValue());
  checkTensorEqual(mixed_restored.value(), mixed);

  const auto rank3 = makeTensor(std::array<Extent, 3>{2, 2, 2},
                                std::array<Scalar, 8>{1, 2, 2, 3, 2, 3, 3, 4});
  const auto rank3_packed =
      packSymmetric(rank3, Tolerance{.absolute = 0, .relative = 0}, ample);
  VSS_CHECK(rank3_packed.hasValue());
  const auto packed_sum =
      symmetricContraction(rank3_packed.value(), rank3_packed.value());
  VSS_CHECK(packed_sum.hasValue());
  Scalar dense_sum = 0;
  const std::span<const Scalar> rank3Values(rank3.data(), rank3.size());
  for (Extent i = 0; i < rank3.size(); ++i) {
    dense_sum += rank3Values[i] * rank3Values[i];
  }
  VSS_CHECK(packed_sum.value() == dense_sum);
  VSS_CHECK(packed_sum.value() == 56);
} catch (const std::exception& exception) {
  std::cerr << exception.what() << '\n';
  return 1;
}
