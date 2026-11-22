#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
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

class Fraction final {
  std::int64_t numerator_{0};
  std::int64_t denominator_{1};

public:
  Fraction() = default;
  Fraction(std::int64_t n, std::int64_t d = 1)
      : numerator_(n), denominator_(d) {
    VSS_CHECK(d != 0);
    if (denominator_ < 0) {
      numerator_ = -numerator_;
      denominator_ = -denominator_;
    }
    const auto divisor =
        std::gcd(numerator_ < 0 ? -numerator_ : numerator_, denominator_);
    numerator_ /= divisor;
    denominator_ /= divisor;
  }

  [[nodiscard]] std::int64_t numerator() const noexcept { return numerator_; }
  [[nodiscard]] std::int64_t denominator() const noexcept {
    return denominator_;
  }
  [[nodiscard]] long double value() const {
    return static_cast<long double>(numerator_) /
           static_cast<long double>(denominator_);
  }
};

Fraction operator+(Fraction a, Fraction b) {
  const auto common = std::gcd(a.denominator(), b.denominator());
  return {a.numerator() * (b.denominator() / common) +
              b.numerator() * (a.denominator() / common),
          a.denominator() * (b.denominator() / common)};
}

Fraction operator-(Fraction a, Fraction b) {
  return a + Fraction{-b.numerator(), b.denominator()};
}

Fraction operator*(Fraction a, Fraction b) {
  const auto left = std::gcd(a.numerator() < 0 ? -a.numerator() : a.numerator(),
                             b.denominator());
  const auto right = std::gcd(
      b.numerator() < 0 ? -b.numerator() : b.numerator(), a.denominator());
  return {(a.numerator() / left) * (b.numerator() / right),
          (a.denominator() / right) * (b.denominator() / left)};
}

Fraction operator/(Fraction a, Fraction b) {
  VSS_CHECK(b.numerator() != 0);
  return a * Fraction{b.denominator(), b.numerator()};
}

using Row = std::array<Fraction, 2U>;
using Tuple = std::array<Index, 4U>;

Fraction sum(std::span<const Fraction> values) {
  Fraction result{};
  for (const auto value : values) {
    result = result + value;
  }
  return result;
}

Fraction exactFourthReference(std::span<const Row> rows,
                              std::span<const Fraction> rawWeights,
                              Tuple tuple) {
  const Fraction mass = sum(rawWeights);
  std::array<Fraction, 2U> mean{};
  for (Index coordinate = 0U; coordinate < 2U; ++coordinate) {
    for (Index row = 0U; row < rows.size(); ++row) {
      mean.at(coordinate) = mean.at(coordinate) +
                            rawWeights[row] * rows[row].at(coordinate) / mass;
    }
  }
  std::array<Row, 8U> centered{};
  VSS_CHECK(rows.size() <= centered.size());
  for (Index row = 0U; row < rows.size(); ++row) {
    for (Index coordinate = 0U; coordinate < 2U; ++coordinate) {
      centered.at(row).at(coordinate) =
          rows[row].at(coordinate) - mean.at(coordinate);
    }
  }
  Fraction m4{};
  for (Index row = 0U; row < rows.size(); ++row) {
    auto product = rawWeights[row] / mass;
    for (const Index coordinate : tuple) {
      product = product * centered.at(row).at(coordinate);
    }
    m4 = m4 + product;
  }
  auto m2 = [&](Index a, Index b) {
    Fraction value{};
    for (Index row = 0U; row < rows.size(); ++row) {
      value = value + rawWeights[row] / mass * centered.at(row).at(a) *
                          centered.at(row).at(b);
    }
    return value;
  };
  const auto [a, b, c, d] = tuple;
  const Fraction q =
      (m2(a, b) * m2(c, d) + m2(a, c) * m2(b, d) + m2(a, d) * m2(b, c)) /
      Fraction{3};

  Fraction s2{};
  Fraction s3{};
  Fraction s4{};
  for (const auto raw : rawWeights) {
    const auto p = raw / mass;
    s2 = s2 + p * p;
    s3 = s3 + p * p * p;
    s4 = s4 + p * p * p * p;
  }
  const Fraction A =
      Fraction{1} - Fraction{4} * s2 + Fraction{6} * s3 - Fraction{3} * s4;
  const Fraction B = Fraction{6} * s2 - Fraction{9} * s2 * s2 -
                     Fraction{6} * s3 + Fraction{9} * s4;
  const Fraction C = s2 - Fraction{2} * s3 + s4;
  const Fraction D = Fraction{1} - Fraction{3} * s2 + Fraction{3} * s2 * s2 +
                     Fraction{2} * s3 - Fraction{3} * s4;
  return (D * m4 - B * q) / (A * D - B * C);
}

Matrix matrix(std::span<const Row> rows) {
  Matrix result(rows.size(), 2U);
  for (Index row = 0U; row < rows.size(); ++row) {
    result.at(row, 0U) = static_cast<Scalar>(rows[row].at(0U).value());
    result.at(row, 1U) = static_cast<Scalar>(rows[row].at(1U).value());
  }
  return result;
}

SampleView sample(const Matrix& rows) {
  static constexpr std::array<Index, 1U> shape{2U};
  return {.rows = rows.view(), .observation_shape = shape, .valid = {}};
}

SampleView scalarSample(const Matrix& rows) {
  return {.rows = rows.view(), .observation_shape = {}, .valid = {}};
}

WeightSpec fixed(const Vector& weights) {
  return {.rule = WeightRule::fixed_iid,
          .rowWeights = weights.view(),
          .componentWeights = std::nullopt,
          .frequencyCounts = {}};
}

Limits ample() {
  return {.elements = std::numeric_limits<Extent>::max(),
          .bytes = std::numeric_limits<Extent>::max(),
          .evaluations = 0U};
}

MomentRequest denseFourth() {
  return {.orders = 0x08U,
          .representation = MomentRepresentation::dense,
          .selected = {},
          .contraction = std::nullopt};
}

void close(Scalar actual, long double expected) {
  const long double tolerance =
      1.0e-12L +
      1.0e-10L * std::max(std::abs(expected),
                          std::abs(static_cast<long double>(actual)));
  if (!(std::abs(static_cast<long double>(actual) - expected) <= tolerance)) {
    std::cerr << "expected " << static_cast<double>(expected) << ", got "
              << actual << '\n';
    VSS_CHECK(false);
  }
}

Scalar at(const MomentEstimate& result, Tuple tuple, Extent dimension = 2U) {
  const Extent index =
      (((tuple.at(0U) * dimension + tuple.at(1U)) * dimension + tuple.at(2U)) *
           dimension +
       tuple.at(3U));
  return result.terms.view()[0U].values.at(index);
}

void unequalWeightExactEnumeration() {
  const std::array<Fraction, 4U> weights{Fraction{1}, Fraction{2}, Fraction{3},
                                         Fraction{4}};
  const Vector vectorWeights(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  const std::array<Row, 3U> atoms{Row{Fraction{-1}, Fraction{1}},
                                  Row{Fraction{0}, Fraction{0}},
                                  Row{Fraction{2}, Fraction{4}}};
  const std::array<Tuple, 3U> checkedTuples{
      Tuple{0U, 0U, 0U, 1U}, Tuple{0U, 0U, 1U, 1U}, Tuple{0U, 1U, 1U, 1U}};
  std::array<Fraction, 3U> expectedSums{};
  for (Index encoded = 0U; encoded < 81U; ++encoded) {
    Index state = encoded;
    std::array<Row, 4U> rows{};
    for (auto& row : rows) {
      row = atoms.at(state % 3U);
      state /= 3U;
    }
    const Matrix observations = matrix(rows);
    const auto result = unbiasedFixedWeightedFourthMoment(
        sample(observations), fixed(vectorWeights), denseFourth(), ample());
    VSS_CHECK(result);
    for (Index checked = 0U; checked < checkedTuples.size(); ++checked) {
      const auto exact =
          exactFourthReference(rows, weights, checkedTuples.at(checked));
      close(at(result.value(), checkedTuples.at(checked)), exact.value());
      expectedSums.at(checked) =
          expectedSums.at(checked) + exact / Fraction{81};
    }
  }
  close(static_cast<Scalar>(expectedSums[0U].value()), 112.0L / 27.0L);
  close(static_cast<Scalar>(expectedSums[1U].value()), 146.0L / 27.0L);
  close(static_cast<Scalar>(expectedSums[2U].value()), 208.0L / 27.0L);
}

void fullRankUnequalLawEnumeration() {
  const std::array<Fraction, 4U> weights{Fraction{1}, Fraction{2}, Fraction{3},
                                         Fraction{4}};
  const std::array<Fraction, 3U> probabilities{Fraction{1, 2}, Fraction{1, 3},
                                               Fraction{1, 6}};
  const std::array<Row, 3U> atoms{Row{Fraction{0}, Fraction{0}},
                                  Row{Fraction{1}, Fraction{0}},
                                  Row{Fraction{0}, Fraction{2}}};
  const Vector vectorWeights(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  Fraction expected{};
  for (Index encoded = 0U; encoded < 81U; ++encoded) {
    Index state = encoded;
    std::array<Row, 4U> rows{};
    Fraction probability{1};
    for (auto& row : rows) {
      const Index atom = state % 3U;
      state /= 3U;
      row = atoms.at(atom);
      probability = probability * probabilities.at(atom);
    }
    const Matrix observations = matrix(rows);
    const auto result = unbiasedFixedWeightedFourthMoment(
        sample(observations), fixed(vectorWeights), denseFourth(), ample());
    VSS_CHECK(result);
    const auto estimate =
        exactFourthReference(rows, weights, Tuple{0U, 0U, 1U, 1U});
    close(at(result.value(), Tuple{0U, 0U, 1U, 1U}), estimate.value());
    expected = expected + probability * estimate;
  }
  close(static_cast<Scalar>(expected.value()), 2.0L / 27.0L);

  // Independently verify the nuisance Q of the population covariance. The
  // corrected public result remains the fourth central moment only.
  std::array<Fraction, 2U> mean{};
  for (Index atom = 0U; atom < atoms.size(); ++atom) {
    for (Index coordinate = 0U; coordinate < 2U; ++coordinate) {
      mean.at(coordinate) =
          mean.at(coordinate) +
          probabilities.at(atom) * atoms.at(atom).at(coordinate);
    }
  }
  std::array<std::array<Fraction, 2U>, 2U> covariance{};
  for (Index atom = 0U; atom < atoms.size(); ++atom) {
    for (Index a = 0U; a < 2U; ++a) {
      for (Index b = 0U; b < 2U; ++b) {
        covariance.at(a).at(b) =
            covariance.at(a).at(b) + probabilities.at(atom) *
                                         (atoms.at(atom).at(a) - mean.at(a)) *
                                         (atoms.at(atom).at(b) - mean.at(b));
      }
    }
  }
  const auto q = (covariance[0U][0U] * covariance[1U][1U] +
                  covariance[0U][1U] * covariance[1U][0U] +
                  covariance[0U][1U] * covariance[1U][0U]) /
                 Fraction{3};
  close(static_cast<Scalar>(q.value()), 4.0L / 81.0L);
}

void equalWeightsAndRankOneMixedTuple() {
  const std::array<Row, 5U> rows{
      Row{Fraction{1}, Fraction{2}}, Row{Fraction{2}, Fraction{4}},
      Row{Fraction{4}, Fraction{8}}, Row{Fraction{8}, Fraction{16}},
      Row{Fraction{16}, Fraction{32}}};
  const Matrix observations = matrix(rows);
  const Vector weights(std::array<Scalar, 5U>{1.0, 1.0, 1.0, 1.0, 1.0});
  const auto result = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), denseFourth(), ample());
  VSS_CHECK(result);
  const auto exact = exactFourthReference(
      rows,
      std::array<Fraction, 5U>{Fraction{1}, Fraction{1}, Fraction{1},
                               Fraction{1}, Fraction{1}},
      Tuple{0U, 0U, 0U, 0U});
  close(at(result.value(), Tuple{0U, 0U, 0U, 0U}), exact.value());
  close(at(result.value(), Tuple{0U, 0U, 0U, 0U}), 19251.0L / 5.0L);
  close(at(result.value(), Tuple{0U, 0U, 1U, 1U}), 4.0L * exact.value());
}

void validationAndWeightScale() {
  const std::array<Row, 4U> rows{
      Row{Fraction{0}, Fraction{0}}, Row{Fraction{1}, Fraction{1}},
      Row{Fraction{2}, Fraction{4}}, Row{Fraction{4}, Fraction{16}}};
  const Matrix observations = matrix(rows);
  Vector weights(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  const auto base = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), denseFourth(), ample());
  VSS_CHECK(base);
  for (Index i = 0U; i < weights.size(); ++i) {
    weights.at(i) *= 1.0e200;
  }
  const auto scaled = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), denseFourth(), ample());
  VSS_CHECK(scaled);
  for (Index i = 0U; i < base.value().terms.view()[0U].values.size(); ++i) {
    close(scaled.value().terms.view()[0U].values.at(i),
          base.value().terms.view()[0U].values.at(i));
  }

  auto wrongOrder = denseFourth();
  wrongOrder.orders = 0x04U;
  const auto wrong = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), wrongOrder, ample());
  VSS_CHECK(!wrong && wrong.status().code() == ErrorCode::InvalidDomain);
  const auto limited = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), denseFourth(), Limits{1U, 8U, 0U});
  VSS_CHECK(!limited && limited.status().code() == ErrorCode::ResourceLimit);
  const Vector threePositive(std::array<Scalar, 4U>{1.0, 1.0, 1.0, 0.0});
  const auto insufficient = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(threePositive), denseFourth(), ample());
  VSS_CHECK(!insufficient &&
            insufficient.status().code() == ErrorCode::InsufficientSamples);
  const Vector nearlyDominant(
      std::array<Scalar, 4U>{1.0, 1.0e-12, 1.0e-12, 1.0e-12});
  const auto illConditioned = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(nearlyDominant), denseFourth(), ample());
  VSS_CHECK(!illConditioned &&
            illConditioned.status().code() == ErrorCode::IllConditioned);
}

void selectedComponentAndContractionLayouts() {
  const std::array<Row, 4U> rows{
      Row{Fraction{0}, Fraction{2}}, Row{Fraction{1}, Fraction{-1}},
      Row{Fraction{3}, Fraction{4}}, Row{Fraction{5}, Fraction{1}}};
  const Matrix observations = matrix(rows);
  const Vector weights(std::array<Scalar, 4U>{1.0, 2.0, 3.0, 4.0});
  const auto dense = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), denseFourth(), ample());
  VSS_CHECK(dense);

  const std::array<Index, 4U> tupleA{0U, 0U, 1U, 1U};
  const std::array<Index, 4U> tupleB{0U, 1U, 1U, 0U};
  const std::array<std::span<const Index>, 2U> selectedTuples{
      std::span<const Index>{tupleA}, std::span<const Index>{tupleB}};
  const MomentRequest selected{.orders = 0x08U,
                               .representation = MomentRepresentation::selected,
                               .selected = selectedTuples,
                               .contraction = std::nullopt};
  const auto selectedResult = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), selected, ample());
  VSS_CHECK(selectedResult);
  VSS_CHECK(
      selectedResult.value().terms.view()[0U].flattened_tuples.view().size() ==
      8U);
  close(selectedResult.value().terms.view()[0U].values.at(0U),
        at(dense.value(), tupleA));
  close(selectedResult.value().terms.view()[0U].values.at(1U),
        at(dense.value(), tupleB));

  auto component = denseFourth();
  component.representation = MomentRepresentation::component;
  const auto componentResult = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), component, ample());
  VSS_CHECK(componentResult);
  for (Index coordinate = 0U; coordinate < 2U; ++coordinate) {
    const Tuple diagonal{coordinate, coordinate, coordinate, coordinate};
    close(componentResult.value().terms.view()[0U].values.at(coordinate),
          at(dense.value(), diagonal));
  }

  Matrix map(1U, 2U);
  map.at(0U, 0U) = 2.0;
  map.at(0U, 1U) = -1.0;
  auto contracted = denseFourth();
  contracted.representation = MomentRepresentation::contraction;
  contracted.contraction = map.view();
  const auto mapped = unbiasedFixedWeightedFourthMoment(
      sample(observations), fixed(weights), contracted, ample());
  VSS_CHECK(mapped);
  Matrix transformed(rows.size(), 1U);
  for (Index row = 0U; row < rows.size(); ++row) {
    transformed.at(row, 0U) =
        2.0 * observations.at(row, 0U) - observations.at(row, 1U);
  }
  const auto direct = unbiasedFixedWeightedFourthMoment(
      scalarSample(transformed), fixed(weights), denseFourth(), ample());
  VSS_CHECK(direct);
  close(mapped.value().terms.view()[0U].values.at(0U),
        direct.value().terms.view()[0U].values.at(0U));
}

} // namespace

int main() try {
  unequalWeightExactEnumeration();
  fullRankUnequalLawEnumeration();
  equalWeightsAndRankOneMixedTuple();
  validationAndWeightScale();
  selectedComponentAndContractionLayouts();
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
