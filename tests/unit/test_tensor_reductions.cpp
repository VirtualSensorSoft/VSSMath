#include "Test.hpp"

#include <array>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>

namespace {

using namespace vss::math;

std::span<const Scalar> values(const Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

Tensor makeTensor(std::span<const Extent> extents,
                  std::span<const Scalar> values) {
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor = Tensor::create(shape.value(), 0.0, TensorLimits{values.size()});
  VSS_CHECK(tensor.hasValue());
  VSS_CHECK(tensor.value().size() == values.size());
  const std::span<Scalar> output{tensor.value().data(), tensor.value().size()};
  for (Extent i = 0; i < values.size(); ++i) {
    output[i] = values[i];
  }
  return std::move(tensor).value();
}

} // namespace

int main() {
  const Limits ample{.elements = 128, .bytes = 8192, .evaluations = 1024};
  const auto matrix = makeTensor(std::array<Extent, 2>{2, 2},
                                 std::array<Scalar, 4>{1, 2, 3, 4});
  const auto sum =
      reduceTensor(matrix, std::array<Index, 1>{0}, ReductionKind::sum, ample);
  VSS_CHECK(sum.hasValue());
  VSS_CHECK(sum.value().shape().extents()[0] == 2);
  VSS_CHECK(values(sum.value())[0] == 4);
  VSS_CHECK(values(sum.value())[1] == 6);
  VSS_CHECK(reduceTensor(matrix, std::array<Index, 2>{0, 0}, ReductionKind::sum,
                         ample)
                .status()
                .code() == ErrorCode::InvalidDomain);
  VSS_CHECK(
      reduceTensor(matrix, std::array<Index, 1>{2}, ReductionKind::sum, ample)
          .status()
          .code() == ErrorCode::IndexOutOfRange);
  const auto identity = reduceTensor(matrix, std::span<const Index>{},
                                     ReductionKind::maximum, ample);
  VSS_CHECK(identity.hasValue());
  VSS_CHECK(values(identity.value())[0] == 1);
  VSS_CHECK(values(identity.value())[3] == 4);
  VSS_CHECK(
      reduceTensor(matrix, std::array<Index, 1>{0}, ReductionKind::sum,
                   Limits{.elements = 128, .bytes = 8192, .evaluations = 3})
          .status()
          .code() == ErrorCode::ResourceLimit);

  const auto empty_reduced =
      makeTensor(std::array<Extent, 3>{2, 0, 3}, std::span<const Scalar>{});
  const auto empty_sum = reduceTensor(empty_reduced, std::array<Index, 1>{1},
                                      ReductionKind::sum, ample);
  VSS_CHECK(empty_sum.hasValue());
  VSS_CHECK(empty_sum.value().shape().extents()[0] == 2);
  VSS_CHECK(empty_sum.value().shape().extents()[1] == 3);
  for (Extent i = 0; i < empty_sum.value().size(); ++i) {
    VSS_CHECK(values(empty_sum.value())[i] == 0);
  }
  const auto empty_product = reduceTensor(
      empty_reduced, std::array<Index, 1>{1}, ReductionKind::product, ample);
  VSS_CHECK(empty_product.hasValue());
  for (Extent i = 0; i < empty_product.value().size(); ++i) {
    VSS_CHECK(values(empty_product.value())[i] == 1);
  }
  const auto empty_norm = reduceTensor(empty_reduced, std::array<Index, 1>{1},
                                       ReductionKind::l2, ample);
  VSS_CHECK(empty_norm.hasValue());
  for (Extent i = 0; i < empty_norm.value().size(); ++i) {
    VSS_CHECK(values(empty_norm.value())[i] == 0);
  }
  VSS_CHECK(reduceTensor(empty_reduced, std::array<Index, 1>{1},
                         ReductionKind::mean, ample)
                .status()
                .code() == ErrorCode::InsufficientSamples);

  const auto no_outputs =
      makeTensor(std::array<Extent, 2>{0, 2}, std::span<const Scalar>{});
  const auto empty_mean = reduceTensor(no_outputs, std::array<Index, 1>{1},
                                       ReductionKind::mean, ample);
  VSS_CHECK(empty_mean.hasValue());
  VSS_CHECK(empty_mean.value().size() == 0);
}
