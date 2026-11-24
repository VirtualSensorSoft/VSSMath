#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Weighted.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent rowCount = rows.size();
  const Extent columnCount = rowCount == 0U ? 0U : rows.begin()->size();
  Matrix result(rowCount, columnCount);
  Extent row = 0U;
  for (const auto& values : rows) {
    VSS_CHECK(values.size() == columnCount);
    Extent column = 0U;
    for (const Scalar value : values) {
      result.at(row, column++) = value;
    }
    ++row;
  }
  return result;
}

Vector vector(std::span<const Scalar> values) { return Vector(values); }

SampleView sample(const Matrix& rows, std::span<const Index> shape = {}) {
  return {.rows = rows.view(), .observation_shape = shape, .valid = {}};
}

WeightSpec fixed(const Vector& weights) {
  return {.rule = WeightRule::fixed_iid,
          .rowWeights = weights.view(),
          .componentWeights = std::nullopt,
          .frequencyCounts = {}};
}

MomentRequest denseThird() {
  return {.orders = 0x04U,
          .representation = MomentRepresentation::dense,
          .selected = {},
          .contraction = std::nullopt};
}

Limits ample() {
  return {.elements = std::numeric_limits<Extent>::max(),
          .bytes = std::numeric_limits<Extent>::max(),
          .evaluations = 0U};
}

void close(Scalar actual, Scalar expected) {
  const Scalar tolerance =
      1.0e-12 + 1.0e-10 * std::max(std::abs(actual), std::abs(expected));
  if (!(std::abs(actual - expected) <= tolerance)) {
    std::cerr << "expected " << expected << ", got " << actual << '\n';
    VSS_CHECK(false);
  }
}

struct ExactFraction final {
  std::int64_t numerator;
  std::int64_t denominator;
};

ExactFraction referenceMixedThird(const std::array<Scalar, 4U>& x,
                                  const std::array<Scalar, 4U>& y,
                                  const std::array<Scalar, 4U>& weights) {
  std::int64_t mass = 0;
  std::int64_t weightedX = 0;
  std::int64_t weightedY = 0;
  for (Index i = 0U; i < 4U; ++i) {
    const auto w = static_cast<std::int64_t>(weights.at(i));
    mass += w;
    weightedX += w * static_cast<std::int64_t>(x.at(i));
    weightedY += w * static_cast<std::int64_t>(y.at(i));
  }
  std::int64_t numerator = 0;
  for (Index i = 0U; i < 4U; ++i) {
    const auto w = static_cast<std::int64_t>(weights.at(i));
    const std::int64_t dx =
        std::int64_t{10} * static_cast<std::int64_t>(x.at(i)) - weightedX;
    const std::int64_t dy =
        std::int64_t{10} * static_cast<std::int64_t>(y.at(i)) - weightedY;
    numerator += w * dx * dx * dy;
  }
  std::int64_t e3 = 0;
  for (Index i = 0U; i < 4U; ++i) {
    for (Index j = i + 1U; j < 4U; ++j) {
      for (Index k = j + 1U; k < 4U; ++k) {
        e3 += static_cast<std::int64_t>(weights.at(i)) *
              static_cast<std::int64_t>(weights.at(j)) *
              static_cast<std::int64_t>(weights.at(k));
      }
    }
  }
  VSS_CHECK(mass == 10);
  return {.numerator = numerator, .denominator = std::int64_t{60} * e3};
}

void exactFiniteLawEnumeration() {
  const Vector weights = vector(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  const std::array<Scalar, 3U> atoms{-1.0, 0.0, 2.0};
  const std::array<Scalar, 4U> rawWeights{1.0, 2.0, 3.0, 4.0};
  std::int64_t numeratorSum = 0;
  constexpr std::int64_t denominator = std::int64_t{60} * 50;
  for (Index a = 0U; a < 3U; ++a) {
    for (Index b = 0U; b < 3U; ++b) {
      for (Index c = 0U; c < 3U; ++c) {
        for (Index d = 0U; d < 3U; ++d) {
          const std::array<Scalar, 4U> u{atoms.at(a), atoms.at(b), atoms.at(c),
                                         atoms.at(d)};
          Matrix rows(4U, 2U);
          for (Index i = 0U; i < 4U; ++i) {
            rows.at(i, 0U) = u.at(i);
            rows.at(i, 1U) = u.at(i) * u.at(i);
          }
          const auto result = unbiasedFixedWeightedThirdMoment(
              sample(rows, std::array<Index, 1U>{2U}), fixed(weights),
              denseThird(), ample());
          VSS_CHECK(result);
          const Scalar actual = result.value().terms.view()[0U].values.at(1U);
          std::array<Scalar, 4U> square{};
          std::ranges::transform(u, square.begin(),
                                 [](Scalar value) { return value * value; });
          const auto reference = referenceMixedThird(u, square, rawWeights);
          const long double exact =
              static_cast<long double>(reference.numerator) /
              static_cast<long double>(reference.denominator);
          VSS_CHECK(std::abs(static_cast<long double>(actual) - exact) <=
                    1.0e-12L + 1.0e-10L * std::abs(exact));
          VSS_CHECK(reference.denominator == denominator);
          numeratorSum += reference.numerator;
        }
      }
    }
  }
  VSS_CHECK(numeratorSum * std::int64_t{27} ==
            std::int64_t{46} * 81 * denominator);
}

void fullRankMixedFiniteLawEnumeration() {
  const Vector weights = vector(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  const std::array<std::array<Scalar, 2U>, 3U> atoms{
      std::array<Scalar, 2U>{0.0, 0.0}, {1.0, 0.0}, {0.0, 2.0}};
  std::int64_t weightedNumerator = 0;
  std::int64_t swappedWeightedNumerator = 0;
  constexpr std::int64_t denominator = std::int64_t{60} * 50;
  for (Index a = 0U; a < 3U; ++a) {
    for (Index b = 0U; b < 3U; ++b) {
      for (Index c = 0U; c < 3U; ++c) {
        for (Index d = 0U; d < 3U; ++d) {
          const std::array<Index, 4U> state{a, b, c, d};
          std::array<Scalar, 4U> x{};
          std::array<Scalar, 4U> y{};
          Matrix rows(4U, 2U);
          std::int64_t probabilityNumerator = 1;
          for (Index i = 0U; i < 4U; ++i) {
            const auto stateIndex = state.at(i);
            x.at(i) = atoms.at(stateIndex).at(0U);
            y.at(i) = atoms.at(stateIndex).at(1U);
            rows.at(i, 0U) = x.at(i);
            rows.at(i, 1U) = y.at(i);
            if (stateIndex == 0U) {
              probabilityNumerator *= 3;
            } else if (stateIndex == 1U) {
              probabilityNumerator *= 2;
            }
          }
          const auto result = unbiasedFixedWeightedThirdMoment(
              sample(rows, std::array<Index, 1U>{2U}), fixed(weights),
              denseThird(), ample());
          VSS_CHECK(result);
          const auto reference =
              referenceMixedThird(x, y, {1.0, 2.0, 3.0, 4.0});
          const long double exact =
              static_cast<long double>(reference.numerator) /
              static_cast<long double>(reference.denominator);
          const Scalar actual = result.value().terms.view()[0U].values.at(1U);
          VSS_CHECK(std::abs(static_cast<long double>(actual) - exact) <=
                    1.0e-12L + 1.0e-10L * std::abs(exact));
          weightedNumerator += probabilityNumerator * reference.numerator;
          const auto swapped = referenceMixedThird(y, x, {1.0, 2.0, 3.0, 4.0});
          close(result.value().terms.view()[0U].values.at(3U),
                static_cast<Scalar>(swapped.numerator) /
                    static_cast<Scalar>(swapped.denominator));
          swappedWeightedNumerator += probabilityNumerator * swapped.numerator;
        }
      }
    }
  }
  VSS_CHECK(weightedNumerator * std::int64_t{27} ==
            std::int64_t{-1296} * denominator);
  VSS_CHECK(swappedWeightedNumerator * std::int64_t{27} ==
            std::int64_t{-5184} * denominator);
}

void equalAndUnequalRankOneWeights() {
  const Matrix observations = matrix({{1.0}, {2.0}, {4.0}, {8.0}, {16.0}});
  const Vector equal(std::array<Scalar, 5U>{1.0, 1.0, 1.0, 1.0, 1.0});
  const auto scalar = unbiasedFixedWeightedThirdMoment(
      sample(observations), fixed(equal), denseThird(), ample());
  VSS_CHECK(scalar);
  close(scalar.value().terms.view()[0U].values.at(0U), 3007.0 / 10.0);

  const Matrix rankOne = matrix({{1.0, 2.0, -3.0},
                                 {2.0, 4.0, -6.0},
                                 {4.0, 8.0, -12.0},
                                 {8.0, 16.0, -24.0},
                                 {16.0, 32.0, -48.0}});
  const std::array<Index, 1U> shape{3U};
  for (const Scalar factor : {1.0, 0x1p500}) {
    const Vector unequal(std::array<Scalar, 5U>{
        2.0 * factor, 3.0 * factor, 4.0 * factor, 5.0 * factor, 6.0 * factor});
    const auto mixed = unbiasedFixedWeightedThirdMoment(
        sample(rankOne, shape), fixed(unequal), denseThird(), ample());
    VSS_CHECK(mixed);
    close(mixed.value().terms.view()[0U].values.at(5U), -6.0 * 4940.0 / 29.0);
  }
}

void layoutsAndContraction() {
  const Matrix rows = matrix({{0.0, 1.0}, {1.0, 0.0}, {2.0, 3.0}, {4.0, 2.0}});
  const Vector weights = vector(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  const std::array<Index, 3U> mixed{0U, 0U, 1U};
  const std::array<Index, 3U> permuted{0U, 1U, 0U};
  const std::array<std::span<const Index>, 2U> selections{
      std::span<const Index>{mixed}, std::span<const Index>{permuted}};
  const MomentRequest selected{.orders = 0x04U,
                               .representation = MomentRepresentation::selected,
                               .selected = selections,
                               .contraction = std::nullopt};
  const auto picked =
      unbiasedFixedWeightedThirdMoment(sample(rows, std::array<Index, 1U>{2U}),
                                       fixed(weights), selected, ample());
  VSS_CHECK(picked);
  VSS_CHECK(picked.value().terms.view()[0U].flattened_tuples.view().size() ==
            6U);
  close(picked.value().terms.view()[0U].values.at(0U),
        picked.value().terms.view()[0U].values.at(1U));
  const Matrix map = matrix({{2.0, -1.0}});
  const MomentRequest contracted{.orders = 0x04U,
                                 .representation =
                                     MomentRepresentation::contraction,
                                 .selected = {},
                                 .contraction = map.view()};
  const auto mapped =
      unbiasedFixedWeightedThirdMoment(sample(rows, std::array<Index, 1U>{2U}),
                                       fixed(weights), contracted, ample());
  VSS_CHECK(mapped);
  const Matrix transformed = matrix({{-1.0}, {2.0}, {1.0}, {6.0}});
  const auto direct = unbiasedFixedWeightedThirdMoment(
      sample(transformed), fixed(weights), denseThird(), ample());
  VSS_CHECK(direct);
  close(mapped.value().terms.view()[0U].values.at(0U),
        direct.value().terms.view()[0U].values.at(0U));
}

void scaleAndExtremeRange() {
  for (const int exponent : {8, 32, 128}) {
    const Scalar t = std::scalbn(1.0, -exponent);
    const Vector weights = vector(std::array<Scalar, 3U>{1.0, t, t});
    const Matrix rows = matrix({{0.0}, {1.0}, {1.0}});
    const auto base = unbiasedFixedWeightedThirdMoment(
        sample(rows), fixed(weights), denseThird(), ample());
    const Vector scaledWeights =
        vector(std::array<Scalar, 3U>{8.0, 8.0 * t, 8.0 * t});
    const auto scaled = unbiasedFixedWeightedThirdMoment(
        sample(rows), fixed(scaledWeights), denseThird(), ample());
    VSS_CHECK(base && scaled);
    const Scalar expected = (1.0 - (2.0 * t)) / (3.0 * t);
    close(base.value().terms.view()[0U].values.at(0U), expected);
    close(base.value().terms.view()[0U].values.at(0U),
          scaled.value().terms.view()[0U].values.at(0U));
    VSS_CHECK(base.value().diagnostics.weight == WeightRule::fixed_iid);
    VSS_CHECK(base.value().diagnostics.numerical_error.has_value());
  }

  const Scalar b = std::scalbn(1.0, -350);
  const Scalar t = std::scalbn(1.0, -128);
  const Vector weights = vector(std::array<Scalar, 3U>{1.0, t, t});
  const Matrix rows = matrix({{0.0}, {b}, {b}});
  const auto tiny = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(weights), denseThird(), ample());
  VSS_CHECK(tiny);
  const Scalar expected = std::scalbn(1.0, -922) / 3.0;
  VSS_CHECK(tiny.value().terms.view()[0U].values.at(0U) != 0.0);
  VSS_CHECK(std::abs(tiny.value().terms.view()[0U].values.at(0U) / expected -
                     1.0) < 1.0e-9);

  const Scalar denormal = std::numeric_limits<Scalar>::denorm_min();
  const Vector extremeWeights =
      vector(std::array<Scalar, 3U>{1.0, denormal, denormal});
  const Matrix extremeRows = matrix({{0.0}, {1.0}, {1.0}});
  const auto overflow = unbiasedFixedWeightedThirdMoment(
      sample(extremeRows), fixed(extremeWeights), denseThird(), ample());
  VSS_CHECK(!overflow);
  VSS_CHECK(overflow.status().code() == ErrorCode::NonFiniteResult);
}

void validationAndLimits() {
  const Matrix rows = matrix({{1.0}, {2.0}, {4.0}});
  const Vector weights = vector(std::array<Scalar, 3U>{1.0, 2.0, 3.0});
  auto wrongOrder = denseThird();
  wrongOrder.orders = 0x02U;
  const auto wrong = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(weights), wrongOrder, ample());
  VSS_CHECK(!wrong && wrong.status().code() == ErrorCode::InvalidDomain);
  const auto limited = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(weights), denseThird(), Limits{1U, 8U, 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const Vector twoPositive = vector(std::array<Scalar, 3U>{1.0, 1.0, 0.0});
  const auto insufficient = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(twoPositive), denseThird(), ample());
  VSS_CHECK(!insufficient &&
            insufficient.status().code() == ErrorCode::InsufficientSamples);
  const Vector negativeWeights = vector(std::array<Scalar, 3U>{1.0, -1.0, 2.0});
  const auto negative = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(negativeWeights), denseThird(), ample());
  VSS_CHECK(!negative && negative.status().code() == ErrorCode::InvalidDomain);
  const Vector nonfiniteWeights = vector(std::array<Scalar, 3U>{
      1.0, std::numeric_limits<Scalar>::infinity(), 2.0});
  const auto nonfinite = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(nonfiniteWeights), denseThird(), ample());
  VSS_CHECK(!nonfinite &&
            nonfinite.status().code() == ErrorCode::NonFiniteInput);
  const Vector wrongCount = vector(std::array<Scalar, 2U>{1.0, 1.0});
  const auto mismatched = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(wrongCount), denseThird(), ample());
  VSS_CHECK(!mismatched &&
            mismatched.status().code() == ErrorCode::DimensionMismatch);
  const Matrix inactiveNonfinite =
      matrix({{1.0}, {std::numeric_limits<Scalar>::quiet_NaN()}, {4.0}});
  const Vector zeroWeight = vector(std::array<Scalar, 3U>{1.0, 0.0, 1.0});
  const auto nonfiniteZeroRow = unbiasedFixedWeightedThirdMoment(
      sample(inactiveNonfinite), fixed(zeroWeight), denseThird(), ample());
  VSS_CHECK(!nonfiniteZeroRow &&
            nonfiniteZeroRow.status().code() == ErrorCode::NonFiniteInput);

  const Matrix cancellation = matrix({{-1.0e100}, {0.0}, {1.0e100}});
  const Vector equalWeights = vector(std::array<Scalar, 3U>{1.0, 1.0, 1.0});
  const auto uncertified = unbiasedFixedWeightedThirdMoment(
      sample(cancellation), fixed(equalWeights), denseThird(), ample());
  VSS_CHECK(!uncertified &&
            uncertified.status().code() == ErrorCode::IllConditioned);

  const int original = std::fegetround();
  VSS_CHECK(std::fesetround(FE_DOWNWARD) == 0);
  const auto unsupported = unbiasedFixedWeightedThirdMoment(
      sample(rows), fixed(weights), denseThird(), ample());
  VSS_CHECK(!unsupported &&
            unsupported.status().code() == ErrorCode::IllConditioned);
  VSS_CHECK(std::fesetround(original) == 0);
}

} // namespace

int main() try {
  exactFiniteLawEnumeration();
  fullRankMixedFiniteLawEnumeration();
  equalAndUnequalRankOneWeights();
  layoutsAndContraction();
  scaleAndExtremeRange();
  validationAndLimits();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
