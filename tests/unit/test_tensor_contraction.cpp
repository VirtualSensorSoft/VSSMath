#include "Test.hpp"

#include <array>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>

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

void checkValues(const Tensor& tensor, std::span<const Extent> shape,
                 std::span<const Scalar> expected) {
  VSS_CHECK(tensor.shape().extents().size() == shape.size());
  for (Extent i = 0; i < shape.size(); ++i) {
    VSS_CHECK(tensor.shape().extents()[i] == shape[i]);
  }
  VSS_CHECK(tensor.size() == expected.size());
  const std::span<const Scalar> tensorValues(tensor.data(), tensor.size());
  for (Extent i = 0; i < expected.size(); ++i) {
    VSS_CHECK(tensorValues[i] == expected[i]);
  }
}

} // namespace

int main() {
  const auto vector_left =
      makeTensor(std::array<Extent, 1>{2}, std::array<Scalar, 2>{2, 3});
  const auto vector_right =
      makeTensor(std::array<Extent, 1>{3}, std::array<Scalar, 3>{5, 7, 11});
  const auto outer = vss::math::outer(vector_left, vector_right);
  VSS_CHECK(outer.hasValue());
  checkValues(outer.value(), std::array<Extent, 2>{2, 3},
              std::array<Scalar, 6>{10, 14, 22, 15, 21, 33});
  VSS_CHECK(vss::math::outer(vector_left, vector_right, TensorLimits{5})
                .status()
                .code() == ErrorCode::ResourceLimit);

  const auto matrix = makeTensor(std::array<Extent, 2>{2, 2},
                                 std::array<Scalar, 4>{1, 2, 3, 4});
  const auto full = vss::math::contract(
      matrix, matrix, std::array<Index, 2>{0, 1}, std::array<Index, 2>{0, 1});
  VSS_CHECK(full.hasValue());
  VSS_CHECK(full.value().shape().rank() == 0);
  VSS_CHECK(std::span<const Scalar>(full.value().data(), full.value().size())
                .front() == 30);

  const auto other = makeTensor(std::array<Extent, 2>{2, 2},
                                std::array<Scalar, 4>{5, 6, 7, 8});
  const auto product = vss::math::contract(
      matrix, other, std::array<Index, 1>{1}, std::array<Index, 1>{0});
  VSS_CHECK(product.hasValue());
  checkValues(product.value(), std::array<Extent, 2>{2, 2},
              std::array<Scalar, 4>{19, 22, 43, 50});

  const auto vector =
      makeTensor(std::array<Extent, 1>{2}, std::array<Scalar, 2>{5, 6});
  const auto matrix_vector = vss::math::contract(
      matrix, vector, std::array<Index, 1>{1}, std::array<Index, 1>{0});
  VSS_CHECK(matrix_vector.hasValue());
  checkValues(matrix_vector.value(), std::array<Extent, 1>{2},
              std::array<Scalar, 2>{17, 39});

  const auto cube = makeTensor(std::array<Extent, 3>{2, 2, 2},
                               std::array<Scalar, 8>{1, 2, 3, 4, 5, 6, 7, 8});
  const auto ones = makeTensor(std::array<Extent, 3>{2, 2, 2},
                               std::array<Scalar, 8>{1, 1, 1, 1, 1, 1, 1, 1});
  const auto multi = vss::math::contract(cube, ones, std::array<Index, 2>{1, 2},
                                         std::array<Index, 2>{0, 1});
  VSS_CHECK(multi.hasValue());
  checkValues(multi.value(), std::array<Extent, 2>{2, 2},
              std::array<Scalar, 4>{10, 10, 26, 26});

  VSS_CHECK(vss::math::contract(matrix, vector, std::array<Index, 1>{2},
                                std::array<Index, 1>{0})
                .status()
                .code() == ErrorCode::IndexOutOfRange);
  VSS_CHECK(vss::math::contract(cube, ones, std::array<Index, 2>{1, 1},
                                std::array<Index, 2>{0, 1})
                .status()
                .code() == ErrorCode::InvalidExtent);
}
