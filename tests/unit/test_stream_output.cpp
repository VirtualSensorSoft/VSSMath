#include "Test.hpp"

#include <array>
#include <iomanip>
#include <ostream>
#include <span>
#include <sstream>
#include <streambuf>
#include <string>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MultivariateMoments.hpp>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/ThirdMoment.hpp>
#include <vss/math/Vector.hpp>

namespace {
using vss::math::Extent;
using vss::math::Scalar;

class CountingBuffer final : public std::stringbuf {
public:
  int sync() override {
    ++sync_count;
    return std::stringbuf::sync();
  }
  int sync_count{0};
};

vss::math::Tensor makeTensor() {
  const std::array<Extent, 2U> extents{2U, 3U};
  const std::array<Scalar, 6U> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  auto shape = vss::math::TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor = vss::math::Tensor::create(
      shape.value(), 0.0, vss::math::TensorLimits{values.size()});
  VSS_CHECK(tensor.hasValue());
  std::span<Scalar> storage{tensor.value().data(), tensor.value().size()};
  for (Extent index = 0U; index < values.size(); ++index) {
    storage[index] = values[index];
  }
  return std::move(tensor).value();
}
} // namespace

int main() {
  using vss::math::FirstMoment;
  using vss::math::Matrix;
  using vss::math::MultivariateMoments;
  using vss::math::PrintOptions;
  using vss::math::Tensor;
  using vss::math::Vector;

  const Tensor tensor = makeTensor();
  std::ostringstream full;
  full << tensor;
  VSS_CHECK(full.str() == "Tensor(shape=[2,3], values=[1,2,3,4,5,6])");

  std::ostringstream bounded;
  PrintOptions bounded_options;
  bounded_options.maxElements = 2U;
  vss::math::print(bounded, tensor, bounded_options);
  VSS_CHECK(bounded.str() ==
            "Tensor(shape=[2,3], values=[1,2, ... (4 omitted)])");

  Vector vector(2U);
  vector.at(0U) = 7.0;
  vector.at(1U) = 8.0;
  Matrix matrix(2U, 3U);
  for (Extent index = 0U; index < 6U; ++index) {
    matrix.at(index / 3U, index % 3U) = static_cast<Scalar>(index + 1U);
  }
  std::ostringstream standard_containers;
  standard_containers << vector << ' ' << matrix << ' ' << tensor.shape();
  VSS_CHECK(standard_containers.str() ==
            "Vector(shape=[2], values=[7,8]) "
            "Matrix(shape=[2,3], values=[1,2,3,4,5,6]) "
            "TensorShape(extents=[2,3], elements=6)");

  auto empty_aggregate = MultivariateMoments::create(1U);
  VSS_CHECK(empty_aggregate.hasValue());
  std::ostringstream absent;
  absent << empty_aggregate.value();
  VSS_CHECK(absent.str() ==
            "MultivariateMoments(dimension=1, first=absent, second=absent, "
            "third=absent, fourth=absent)");

  const std::array<Extent, 1U> scalar_extent{1U};
  auto scalar_shape = vss::math::TensorShape::create(scalar_extent);
  VSS_CHECK(scalar_shape.hasValue());
  auto scalar_tensor =
      Tensor::create(scalar_shape.value(), 3.5, vss::math::TensorLimits{1U});
  VSS_CHECK(scalar_tensor.hasValue());
  auto first = FirstMoment::create(scalar_tensor.value());
  VSS_CHECK(first.hasValue());
  std::ostringstream moment_output;
  moment_output << first.value();
  VSS_CHECK(moment_output.str() == "FirstMoment(shape=[1], values=[3.5])");

  const std::array<Extent, 2U> second_extents{1U, 1U};
  const std::array<Extent, 3U> third_extents{1U, 1U, 1U};
  const std::array<Extent, 4U> fourth_extents{1U, 1U, 1U, 1U};
  auto second_shape = vss::math::TensorShape::create(second_extents);
  auto third_shape = vss::math::TensorShape::create(third_extents);
  auto fourth_shape = vss::math::TensorShape::create(fourth_extents);
  VSS_CHECK(second_shape && third_shape && fourth_shape);
  auto second_tensor =
      Tensor::create(second_shape.value(), 2.0, vss::math::TensorLimits{1U});
  auto third_tensor =
      Tensor::create(third_shape.value(), 0.0, vss::math::TensorLimits{1U});
  auto fourth_tensor =
      Tensor::create(fourth_shape.value(), 4.0, vss::math::TensorLimits{1U});
  VSS_CHECK(second_tensor && third_tensor && fourth_tensor);
  auto second = vss::math::SecondMoment::create(second_tensor.value());
  auto third = vss::math::ThirdMoment::create(third_tensor.value());
  auto fourth = vss::math::FourthMoment::create(fourth_tensor.value());
  VSS_CHECK(second && third && fourth);
  std::ostringstream all_moments;
  all_moments << first.value() << ' ' << second.value() << ' ' << third.value()
              << ' ' << fourth.value();
  VSS_CHECK(all_moments.str() == "FirstMoment(shape=[1], values=[3.5]) "
                                 "SecondMoment(shape=[1,1], values=[2]) "
                                 "ThirdMoment(shape=[1,1,1], values=[0]) "
                                 "FourthMoment(shape=[1,1,1,1], values=[4])");

  auto populated = MultivariateMoments::create(1U);
  VSS_CHECK(populated.hasValue());
  VSS_CHECK(populated.value().setFirstMoment(first.value()).ok());
  VSS_CHECK(populated.value().setSecondMoment(second.value()).ok());
  VSS_CHECK(populated.value().setThirdMoment(third.value()).ok());
  VSS_CHECK(populated.value().setFourthMoment(fourth.value()).ok());
  std::ostringstream aggregate_output;
  aggregate_output << populated.value();
  VSS_CHECK(aggregate_output.str().find("first=FirstMoment(") !=
            std::string::npos);
  VSS_CHECK(aggregate_output.str().find("fourth=FourthMoment(") !=
            std::string::npos);

  std::ostringstream formatted;
  formatted << std::hex << std::setprecision(3) << std::setfill('#');
  const auto flags = formatted.flags();
  const auto precision = formatted.precision();
  const auto fill = formatted.fill();
  formatted << matrix;
  VSS_CHECK(formatted.flags() == flags);
  VSS_CHECK(formatted.precision() == precision);
  VSS_CHECK(formatted.fill() == fill);

  std::ostringstream failed;
  failed.setstate(std::ios::failbit);
  failed << tensor;
  VSS_CHECK(failed.fail());

  CountingBuffer buffer;
  std::ostream non_flushing(&buffer);
  non_flushing << tensor;
  VSS_CHECK(buffer.sync_count == 0);
}
