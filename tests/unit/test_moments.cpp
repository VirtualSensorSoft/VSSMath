#include "Test.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/ThirdMoment.hpp>
#include <vss/math/Tolerance.hpp>

namespace {
using vss::math::Extent;
using vss::math::Scalar;
using vss::math::Tensor;
static_assert(std::is_same_v<
              decltype(std::declval<const vss::math::FirstMoment&>().tensor()),
              const Tensor&>);

template <std::size_t Rank, std::size_t Count>
Tensor makeTensor(const std::array<Extent, Rank>& extents,
                  const std::array<Scalar, Count>& values) {
  auto shape = vss::math::TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor =
      Tensor::create(shape.value(), 0.0, vss::math::TensorLimits{Count});
  VSS_CHECK(tensor.hasValue());
  auto storage =
      std::span<Scalar>{tensor.value().data(), tensor.value().size()};
  const std::span<const Scalar> input_values{values};
  VSS_CHECK(storage.size() == values.size());
  for (Extent index = 0U; index < values.size(); ++index) {
    storage[index] = input_values[index];
  }
  return std::move(tensor).value();
}

template <class ResultType>
void expectFailure(const ResultType& result, vss::math::ErrorCode code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == code);
}
} // namespace

int main() {
  using vss::math::ErrorCode;
  using vss::math::FirstMoment;
  using vss::math::FourthMoment;
  using vss::math::SecondMoment;
  using vss::math::ThirdMoment;
  using vss::math::Tolerance;

  const auto first_scalar = makeTensor<1, 1>({1U}, {0.0});
  const auto second_scalar = makeTensor<2, 1>({1U, 1U}, {0.0});
  const auto third_scalar = makeTensor<3, 1>({1U, 1U, 1U}, {0.0});
  const auto fourth_scalar = makeTensor<4, 1>({1U, 1U, 1U, 1U}, {0.0});

  auto first = FirstMoment::create(first_scalar, Tolerance{});
  auto second = SecondMoment::create(second_scalar, Tolerance{});
  auto third = ThirdMoment::create(third_scalar, Tolerance{});
  auto fourth = FourthMoment::create(fourth_scalar, Tolerance{});
  VSS_CHECK(first && second && third && fourth);
  VSS_CHECK(first.value().tensor().shape().rank() == 1U);
  VSS_CHECK(second.value().tensor().shape().rank() == 2U);
  VSS_CHECK(third.value().tensor().shape().rank() == 3U);
  VSS_CHECK(fourth.value().tensor().shape().rank() == 4U);
  const auto first_data = std::span<const Scalar>{
      first.value().tensor().data(), first.value().tensor().size()};
  const auto second_data = std::span<const Scalar>{
      second.value().tensor().data(), second.value().tensor().size()};
  const auto third_data = std::span<const Scalar>{
      third.value().tensor().data(), third.value().tensor().size()};
  const auto fourth_data = std::span<const Scalar>{
      fourth.value().tensor().data(), fourth.value().tensor().size()};
  VSS_CHECK(first_data[0] == 0.0);
  VSS_CHECK(second_data[0] == 0.0);
  VSS_CHECK(third_data[0] == 0.0);
  VSS_CHECK(fourth_data[0] == 0.0);

  auto symmetric_second_tensor =
      makeTensor<2, 4>({2U, 2U}, {1.0, 2.0, 2.0, 3.0});
  auto symmetric_second = SecondMoment::create(symmetric_second_tensor);
  VSS_CHECK(symmetric_second.hasValue());
  std::span<Scalar>{symmetric_second_tensor.data(),
                    symmetric_second_tensor.size()}[0] = 99.0;
  const auto stored_second =
      std::span<const Scalar>{symmetric_second.value().tensor().data(),
                              symmetric_second.value().tensor().size()};
  VSS_CHECK(stored_second[0] == 1.0);

  auto symmetric_third = ThirdMoment::create(
      makeTensor<3, 8>({2U, 2U, 2U}, {1.0, 2.0, 2.0, 3.0, 2.0, 3.0, 3.0, 4.0}),
      Tolerance{});
  auto symmetric_fourth = FourthMoment::create(
      makeTensor<4, 16>({2U, 2U, 2U, 2U},
                        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0}),
      Tolerance{});
  VSS_CHECK(symmetric_third.hasValue());
  VSS_CHECK(symmetric_fourth.hasValue());

  expectFailure(
      FirstMoment::create(makeTensor<2, 1>({1U, 1U}, {0.0}), Tolerance{}),
      ErrorCode::InvalidExtent);
  expectFailure(
      SecondMoment::create(makeTensor<3, 1>({1U, 1U, 1U}, {0.0}), Tolerance{}),
      ErrorCode::InvalidExtent);
  expectFailure(
      ThirdMoment::create(makeTensor<2, 1>({1U, 1U}, {0.0}), Tolerance{}),
      ErrorCode::InvalidExtent);
  expectFailure(
      FourthMoment::create(makeTensor<3, 1>({1U, 1U, 1U}, {0.0}), Tolerance{}),
      ErrorCode::InvalidExtent);

  expectFailure(
      SecondMoment::create(makeTensor<2, 2>({1U, 2U}, {1.0, 2.0}), Tolerance{}),
      ErrorCode::DimensionMismatch);
  expectFailure(ThirdMoment::create(makeTensor<3, 2>({1U, 1U, 2U}, {1.0, 2.0}),
                                    Tolerance{}),
                ErrorCode::DimensionMismatch);
  expectFailure(
      FourthMoment::create(makeTensor<4, 2>({1U, 1U, 1U, 2U}, {1.0, 2.0}),
                           Tolerance{}),
      ErrorCode::DimensionMismatch);
  expectFailure(FirstMoment::create(makeTensor<1, 0>({0U}, {}), Tolerance{}),
                ErrorCode::InvalidExtent);

  const Scalar infinity = std::numeric_limits<Scalar>::infinity();
  expectFailure(
      FirstMoment::create(makeTensor<1, 1>({1U}, {infinity}), Tolerance{}),
      ErrorCode::NonFiniteInput);
  expectFailure(
      SecondMoment::create(makeTensor<2, 1>({1U, 1U}, {infinity}), Tolerance{}),
      ErrorCode::NonFiniteInput);
  expectFailure(ThirdMoment::create(makeTensor<3, 1>({1U, 1U, 1U}, {infinity}),
                                    Tolerance{}),
                ErrorCode::NonFiniteInput);
  expectFailure(
      FourthMoment::create(makeTensor<4, 1>({1U, 1U, 1U, 1U}, {infinity}),
                           Tolerance{}),
      ErrorCode::NonFiniteInput);

  expectFailure(
      SecondMoment::create(makeTensor<2, 4>({2U, 2U}, {1.0, 2.0, 3.0, 4.0}),
                           Tolerance{}),
      ErrorCode::NotSymmetric);
  expectFailure(
      ThirdMoment::create(makeTensor<3, 8>({2U, 2U, 2U}, {1.0, 2.0, 2.0, 3.0,
                                                          2.0, 3.0, 4.0, 4.0}),
                          Tolerance{}),
      ErrorCode::NotSymmetric);
  auto asymmetric_fourth = makeTensor<4, 16>(
      {2U, 2U, 2U, 2U}, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0});
  expectFailure(FourthMoment::create(asymmetric_fourth, Tolerance{}),
                ErrorCode::NotSymmetric);

  auto invalid_tolerance =
      SecondMoment::create(second_scalar, Tolerance{-1.0, 0.0});
  expectFailure(invalid_tolerance, ErrorCode::InvalidTolerance);
}
