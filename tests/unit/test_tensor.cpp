#include "Test.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>

int main() {
  using vss::math::ErrorCode;
  using vss::math::Tensor;
  using vss::math::TensorLimits;
  using vss::math::TensorShape;

  auto shape = TensorShape::create(std::array<std::size_t, 2>{2U, 3U});
  VSS_CHECK(shape.hasValue());
  const TensorLimits limits{8U};
  auto tensor = Tensor::create(shape.value(), 0.0, limits);
  VSS_CHECK(tensor.hasValue());
  tensor.value().at(std::array<std::size_t, 2>{1U, 2U}) = 7.0;
  VSS_CHECK(tensor.value().at(std::array<std::size_t, 2>{1U, 2U}) == 7.0);
  VSS_CHECK(tensor.value().size() == 6U);

  auto tooLarge = Tensor::create(shape.value(), 0.0, TensorLimits{5U});
  VSS_CHECK(!tooLarge.hasValue());
  VSS_CHECK(tooLarge.status().code() == ErrorCode::ResourceLimit);

  bool rankRejected = false;
  try {
    static_cast<void>(tensor.value().at(std::array<std::size_t, 1>{1U}));
  } catch (const std::invalid_argument&) {
    rankRejected = true;
  }
  VSS_CHECK(rankRejected);

  bool boundsRejected = false;
  try {
    static_cast<void>(tensor.value().at(std::array<std::size_t, 2>{2U, 0U}));
  } catch (const std::out_of_range&) {
    boundsRejected = true;
  }
  VSS_CHECK(boundsRejected);

  auto scalarShape = TensorShape::create(std::array<std::size_t, 0>{});
  VSS_CHECK(scalarShape.hasValue());
  auto scalar = Tensor::create(scalarShape.value(), 4.0, limits);
  VSS_CHECK(scalar.hasValue());
  VSS_CHECK(scalar.value().at(std::array<std::size_t, 0>{}) == 4.0);

  auto emptyShape = TensorShape::create(std::array<std::size_t, 1>{0U});
  VSS_CHECK(emptyShape.hasValue());
  auto empty = Tensor::create(emptyShape.value(), 0.0, limits);
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().size() == 0U);

  Tensor copy = tensor.value();
  copy.at(std::array<std::size_t, 2>{1U, 2U}) = -1.0;
  VSS_CHECK(tensor.value().at(std::array<std::size_t, 2>{1U, 2U}) == 7.0);
  const Tensor moved = std::move(copy);
  VSS_CHECK(moved.size() == 6U);
}
