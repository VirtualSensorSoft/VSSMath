#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <iostream>
#include <span>
#include <utility>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/MultivariateMoments.hpp>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/ThirdMoment.hpp>

namespace {
using vss::math::Extent;
using vss::math::Scalar;
using vss::math::Tensor;

template <std::size_t Rank, std::size_t Count>
Tensor makeTensor(const std::array<Extent, Rank>& extents,
                  const std::array<Scalar, Count>& values) {
  auto shape = vss::math::TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto tensor =
      Tensor::create(shape.value(), 0.0, vss::math::TensorLimits{Count});
  VSS_CHECK(tensor.hasValue());
  const std::span<Scalar> storage{tensor.value().data(), tensor.value().size()};
  VSS_CHECK(storage.size() == values.size());
  std::ranges::copy(values, storage.begin());
  return std::move(tensor).value();
}

template <class ResultType>
void expectError(const ResultType& result, vss::math::ErrorCode code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == code);
}
} // namespace

int main() try {
  using vss::math::ErrorCode;
  using vss::math::FirstMoment;
  using vss::math::FourthMoment;
  using vss::math::MultivariateMoments;
  using vss::math::SecondMoment;
  using vss::math::ThirdMoment;

  expectError(MultivariateMoments::create(0U), ErrorCode::InvalidExtent);
  auto aggregate_result = MultivariateMoments::create(1U);
  VSS_CHECK(aggregate_result.hasValue());
  auto& aggregate = aggregate_result.value();
  VSS_CHECK(aggregate.dimension() == 1U);
  VSS_CHECK(aggregate.getFirstMoment() == nullptr);
  VSS_CHECK(aggregate.getSecondMoment() == nullptr);
  VSS_CHECK(aggregate.getThirdMoment() == nullptr);
  VSS_CHECK(aggregate.getFourthMoment() == nullptr);
  expectError(aggregate.componentStandardDeviation(0U),
              ErrorCode::InvalidDomain);
  expectError(aggregate.componentStandardDeviation(1U),
              ErrorCode::IndexOutOfRange);

  auto variance = SecondMoment::create(makeTensor<2, 1>({1U, 1U}, {4.0}));
  auto third = ThirdMoment::create(makeTensor<3, 1>({1U, 1U, 1U}, {0.0}));
  auto fourth =
      FourthMoment::create(makeTensor<4, 1>({1U, 1U, 1U, 1U}, {48.0}));
  VSS_CHECK(variance && third && fourth);
  VSS_CHECK(aggregate.setSecondMoment(variance.value()).ok());
  VSS_CHECK(aggregate.setThirdMoment(third.value()).ok());
  VSS_CHECK(aggregate.setFourthMoment(fourth.value()).ok());

  // A standardized quantity must not silently assume a missing mean.
  expectError(aggregate.componentSkewness(0U), ErrorCode::InvalidDomain);
  expectError(aggregate.componentKurtosis(0U), ErrorCode::InvalidDomain);
  expectError(aggregate.componentExcessKurtosis(0U), ErrorCode::InvalidDomain);

  auto mean = FirstMoment::create(makeTensor<1, 1>({1U}, {12.0}));
  VSS_CHECK(mean.hasValue());
  VSS_CHECK(aggregate.setFirstMoment(mean.value()).ok());
  const auto standard_deviation = aggregate.componentStandardDeviation(0U);
  const auto skewness = aggregate.componentSkewness(0U);
  const auto kurtosis = aggregate.componentKurtosis(0U);
  const auto excess = aggregate.componentExcessKurtosis(0U);
  VSS_CHECK(standard_deviation && standard_deviation.value() == 2.0);
  VSS_CHECK(skewness && skewness.value() == 0.0);
  VSS_CHECK(kurtosis && kurtosis.value() == 3.0);
  VSS_CHECK(excess && excess.value() == 0.0);

  auto two_component_mean =
      FirstMoment::create(makeTensor<1, 2>({2U}, {1.0, 2.0}));
  VSS_CHECK(two_component_mean.hasValue());
  const auto mismatch = aggregate.setFirstMoment(two_component_mean.value());
  VSS_CHECK(!mismatch.ok());
  VSS_CHECK(mismatch.code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(aggregate.getFirstMoment() != nullptr);
  const std::array<vss::math::Index, 1U> first_component{0U};
  VSS_CHECK(aggregate.getFirstMoment()->tensor().at(first_component) == 12.0);
  expectError(aggregate.componentStandardDeviation(1U),
              ErrorCode::IndexOutOfRange);

  aggregate.clearFirstMoment();
  VSS_CHECK(aggregate.getFirstMoment() == nullptr);
  expectError(aggregate.componentSkewness(0U), ErrorCode::InvalidDomain);

  auto zero_variance = SecondMoment::create(makeTensor<2, 1>({1U, 1U}, {0.0}));
  VSS_CHECK(zero_variance.hasValue());
  VSS_CHECK(aggregate.setSecondMoment(zero_variance.value()).ok());
  VSS_CHECK(aggregate.componentStandardDeviation(0U).value() == 0.0);
  auto zero_mean = FirstMoment::create(makeTensor<1, 1>({1U}, {0.0}));
  VSS_CHECK(zero_mean.hasValue());
  VSS_CHECK(aggregate.setFirstMoment(zero_mean.value()).ok());
  expectError(aggregate.componentSkewness(0U), ErrorCode::InvalidDomain);
  expectError(aggregate.componentKurtosis(0U), ErrorCode::InvalidDomain);

  auto large_aggregate_result = MultivariateMoments::create(1U);
  VSS_CHECK(large_aggregate_result.hasValue());
  auto large_mean = FirstMoment::create(makeTensor<1, 1>({1U}, {1.0}));
  auto large_variance =
      SecondMoment::create(makeTensor<2, 1>({1U, 1U}, {1.0e250}));
  auto large_third =
      ThirdMoment::create(makeTensor<3, 1>({1U, 1U, 1U}, {1.0e300}));
  auto large_fourth =
      FourthMoment::create(makeTensor<4, 1>({1U, 1U, 1U, 1U}, {1.0e300}));
  VSS_CHECK(large_mean && large_variance && large_third && large_fourth);
  auto& large_aggregate = large_aggregate_result.value();
  VSS_CHECK(large_aggregate.setFirstMoment(large_mean.value()).ok());
  VSS_CHECK(large_aggregate.setSecondMoment(large_variance.value()).ok());
  VSS_CHECK(large_aggregate.setThirdMoment(large_third.value()).ok());
  VSS_CHECK(large_aggregate.setFourthMoment(large_fourth.value()).ok());
  VSS_CHECK(large_aggregate.componentSkewness(0U).value() > 0.0);
  VSS_CHECK(large_aggregate.componentKurtosis(0U).value() > 0.0);
} catch (const std::exception& exception) {
  std::cerr << "Unexpected exception: " << exception.what() << '\n';
  return 1;
}
