#include "Test.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/NumericalError.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Vector.hpp>

namespace {

using vss::math::ErrorCode;
using vss::math::Extent;
using vss::math::Scalar;
using vss::math::Tensor;
using vss::math::TensorLimits;
using vss::math::TensorShape;

template <std::size_t N>
Tensor makeTensor(Extent extent, std::array<Scalar, N> values) {
  const std::array<Extent, 1> extents{extent};
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor = Tensor::create(shape.value(), 0.0, TensorLimits{16U});
  VSS_CHECK(tensor.hasValue());
  const std::span<const Scalar> inputValues{values};
  const std::span<Scalar> tensorValues{tensor.value().data(),
                                       tensor.value().size()};
  for (Extent index = 0U; index < inputValues.size(); ++index) {
    tensorValues[index] = inputValues[index];
  }
  return std::move(tensor).value();
}

} // namespace

int main() {
  using vss::math::Matrix;
  using vss::math::Vector;

  const std::array<Scalar, 2> vector_values_a{1.0, 2.0};
  const std::array<Scalar, 2> vector_values_b{3.0, 4.0};
  const Vector vector_a{std::span<const Scalar>{vector_values_a}};
  const Vector vector_b{std::span<const Scalar>{vector_values_b}};
  const Vector vector_sum = vector_a + vector_b;
  VSS_CHECK(vector_sum.at(0U) == 4.0);
  VSS_CHECK(vector_sum.at(1U) == 6.0);
  VSS_CHECK((2.0 * vector_a).at(1U) == 4.0);
  VSS_CHECK((vector_b - vector_a).at(0U) == 2.0);
  VSS_CHECK(vector_a.at(0U) == 1.0);

  bool vector_mismatch_threw = false;
  try {
    static_cast<void>(vector_a + Vector{1U});
  } catch (const vss::math::NumericalError& error) {
    vector_mismatch_threw = error.code() == ErrorCode::DimensionMismatch;
  }
  VSS_CHECK(vector_mismatch_threw);

  Matrix matrix_a{1U, 2U};
  matrix_a.at(0U, 0U) = 1.0;
  matrix_a.at(0U, 1U) = 2.0;
  Matrix matrix_b{1U, 2U};
  matrix_b.at(0U, 0U) = 3.0;
  matrix_b.at(0U, 1U) = 4.0;
  const Matrix matrix_sum = matrix_a + matrix_b;
  VSS_CHECK(matrix_sum.at(0U, 0U) == 4.0);
  VSS_CHECK(matrix_sum.at(0U, 1U) == 6.0);
  const auto checked_matrix_difference =
      vss::math::subtract(matrix_b.view(), matrix_a.view());
  VSS_CHECK(checked_matrix_difference.hasValue());
  VSS_CHECK(checked_matrix_difference.value().at(0U, 0U) == 2.0);
  VSS_CHECK(checked_matrix_difference.value().at(0U, 1U) == 2.0);
  const auto checked_matrix_scale = vss::math::scale(matrix_a.view(), 2.0);
  VSS_CHECK(checked_matrix_scale.hasValue());
  VSS_CHECK(checked_matrix_scale.value().at(0U, 1U) == 4.0);
  VSS_CHECK((matrix_a * 3.0).at(0U, 1U) == 6.0);
  VSS_CHECK((2.0 * matrix_b).at(0U, 0U) == 6.0);
  VSS_CHECK(matrix_a.at(0U, 0U) == 1.0);

  Matrix matrix_c{2U, 1U};
  matrix_c.at(0U, 0U) = 5.0;
  matrix_c.at(1U, 0U) = 6.0;
  const auto rejected_matrix_sum =
      vss::math::add(matrix_a.view(), matrix_c.view());
  VSS_CHECK(!rejected_matrix_sum.hasValue());
  VSS_CHECK(rejected_matrix_sum.status().code() ==
            ErrorCode::DimensionMismatch);
  bool matrix_mismatch_threw = false;
  try {
    static_cast<void>(matrix_a + matrix_c);
  } catch (const vss::math::NumericalError& error) {
    matrix_mismatch_threw = error.code() == ErrorCode::DimensionMismatch;
  }
  VSS_CHECK(matrix_mismatch_threw);
  const Matrix matrix_product = matrix_a * matrix_c;
  VSS_CHECK(matrix_product.at(0U, 0U) == 17.0);
  const std::array<Scalar, 1> matrix_vector_values{1.0};
  const Vector matrix_vector =
      matrix_c * Vector{std::span<const Scalar>{matrix_vector_values}};
  VSS_CHECK(matrix_vector.at(0U) == 5.0);
  VSS_CHECK(matrix_vector.at(1U) == 6.0);

  const std::array<Extent, 1> tensor_extent{2U};
  const Tensor tensor_a =
      makeTensor(tensor_extent[0], std::array<Scalar, 2>{1.0, 2.0});
  const Tensor tensor_b =
      makeTensor(tensor_extent[0], std::array<Scalar, 2>{3.0, 4.0});
  const TensorLimits limits{16U};
  const auto checked_sum = vss::math::add(tensor_a, tensor_b, limits);
  VSS_CHECK(checked_sum.hasValue());
  VSS_CHECK(checked_sum.value().at(std::array<Extent, 1>{0U}) == 4.0);
  VSS_CHECK(checked_sum.value().at(std::array<Extent, 1>{1U}) == 6.0);
  VSS_CHECK((tensor_a - tensor_b).at(std::array<Extent, 1>{1U}) == -2.0);
  VSS_CHECK((tensor_a * 2.0).at(std::array<Extent, 1>{0U}) == 2.0);
  VSS_CHECK((2.0 * tensor_b).at(std::array<Extent, 1>{1U}) == 8.0);
  VSS_CHECK(tensor_a.at(std::array<Extent, 1>{0U}) == 1.0);

  const Tensor tensor_mismatch = makeTensor(1U, std::array<Scalar, 1>{7.0});
  const auto rejected = vss::math::add(tensor_a, tensor_mismatch, limits);
  VSS_CHECK(!rejected.hasValue());
  VSS_CHECK(rejected.status().code() == ErrorCode::DimensionMismatch);
  bool tensor_mismatch_threw = false;
  try {
    static_cast<void>(tensor_a + tensor_mismatch);
  } catch (const vss::math::NumericalError& error) {
    tensor_mismatch_threw = error.code() == ErrorCode::DimensionMismatch;
  }
  VSS_CHECK(tensor_mismatch_threw);
}
