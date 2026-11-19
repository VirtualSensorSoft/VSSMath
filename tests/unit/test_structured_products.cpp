#include "Test.hpp"

#include <array>
#include <limits>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/linear/StructuredProducts.hpp>

int main() {
  using namespace vss::math;
  const Limits ample{.elements = 128, .bytes = 8192, .evaluations = 1024};
  Matrix left{2, 1};
  left.at(0, 0) = 1;
  left.at(1, 0) = 2;
  Matrix right{2, 1};
  right.at(0, 0) = 3;
  right.at(1, 0) = 4;

  const auto kronecker = vss::math::kronecker(left.view(), right.view(), ample);
  VSS_CHECK(kronecker.hasValue());
  VSS_CHECK(kronecker.value().rows() == 4);
  VSS_CHECK(kronecker.value().columns() == 1);
  const std::array<Scalar, 4> product_values{3, 4, 6, 8};
  const std::span<const Scalar> expected_values{product_values};
  for (Index i = 0; i < product_values.size(); ++i) {
    VSS_CHECK(kronecker.value().at(i, 0) == expected_values[i]);
  }
  VSS_CHECK(
      vss::math::kronecker(left.view(), right.view(), Limits{1, 8192, 1024})
          .status()
          .code() == ErrorCode::ResourceLimit);

  const auto khatri = vss::math::khatriRao(left.view(), right.view(), ample);
  VSS_CHECK(khatri.hasValue());
  VSS_CHECK(khatri.value().rows() == 4);
  VSS_CHECK(khatri.value().columns() == 1);
  for (Index i = 0; i < product_values.size(); ++i) {
    VSS_CHECK(khatri.value().at(i, 0) == expected_values[i]);
  }
  Matrix different_columns{2, 2};
  VSS_CHECK(vss::math::khatriRao(left.view(), different_columns.view(), ample)
                .status()
                .code() == ErrorCode::DimensionMismatch);

  Matrix values{2, 2};
  values.at(0, 0) = 1;
  values.at(0, 1) = 2;
  values.at(1, 0) = 3;
  values.at(1, 1) = 4;
  const auto diagonal = vss::math::trace(values.view());
  VSS_CHECK(diagonal.hasValue());
  VSS_CHECK(diagonal.value() == 5);
  Matrix nonsquare{2, 1};
  VSS_CHECK(vss::math::trace(nonsquare.view()).status().code() ==
            ErrorCode::DimensionMismatch);
  Matrix empty{0, 0};
  VSS_CHECK(vss::math::trace(empty.view()).value() == 0);

  Matrix nonfinite{1, 1};
  nonfinite.at(0, 0) = std::numeric_limits<Scalar>::infinity();
  VSS_CHECK(vss::math::kronecker(nonfinite.view(), left.view(), ample)
                .status()
                .code() == ErrorCode::NonFiniteInput);
}
