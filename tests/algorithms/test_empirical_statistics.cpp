#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Empirical.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

Vector vector(std::initializer_list<Scalar> values) {
  return Vector(std::span<const Scalar>(values.begin(), values.size()));
}

Matrix matrix(std::initializer_list<std::initializer_list<Scalar>> rows) {
  const Extent n = rows.size();
  const Extent d = n == 0U ? 0U : rows.begin()->size();
  Matrix result(n, d, 0.0);
  Extent i = 0U;
  for (const auto& row : rows) {
    VSS_CHECK(row.size() == d);
    Extent j = 0U;
    for (const Scalar value : row) {
      result.at(i, j++) = value;
    }
    ++i;
  }
  return result;
}

Limits ample() {
  return {.elements = std::numeric_limits<std::size_t>::max(),
          .bytes = std::numeric_limits<std::size_t>::max(),
          .evaluations = std::numeric_limits<std::uint64_t>::max()};
}

void close(Scalar value, Scalar expected) {
  VSS_CHECK(std::abs(value - expected) < 1.0e-12);
}

void oneDimensionalHistogramsUseSpecifiedEdgesAndMass() {
  const Vector values = vector({0.0, 0.5, 1.0, -1.0, 2.0});
  const Vector edges = vector({0.0, 0.5, 1.0});
  const auto counts = histogram(values.view(), edges.view(),
                                HistogramNormalization::counts, ample());
  VSS_CHECK(counts.hasValue());
  close(counts.value().bins.at(0U), 1.0);
  close(counts.value().bins.at(1U), 2.0);
  close(counts.value().underflow, 1.0);
  close(counts.value().overflow, 1.0);
  close(counts.value().accepted, 3.0);

  const Vector weightedValues = vector({0.25, 0.75, 1.0});
  const Vector weights = vector({1.0, 2.0, 3.0});
  const Vector weightedEdges = vector({0.0, 0.5, 1.0});
  const auto weighted = weightedHistogram(
      weightedValues.view(), weights.view(), weightedEdges.view(),
      HistogramNormalization::counts, ample());
  VSS_CHECK(weighted.hasValue());
  close(weighted.value().bins.at(0U), 1.0);
  close(weighted.value().bins.at(1U), 5.0);
  const auto mass = weightedHistogram(weightedValues.view(), weights.view(),
                                      weightedEdges.view(),
                                      HistogramNormalization::mass, ample());
  VSS_CHECK(mass.hasValue());
  close(mass.value().bins.at(0U), 1.0 / 6.0);
  close(mass.value().bins.at(1U), 5.0 / 6.0);
  const auto density = weightedHistogram(
      weightedValues.view(), weights.view(), weightedEdges.view(),
      HistogramNormalization::density, ample());
  VSS_CHECK(density.hasValue());
  close(density.value().bins.at(0U), 1.0 / 3.0);
  close(density.value().bins.at(1U), 5.0 / 3.0);
}

void multidimensionalHistogramUsesRowMajorBinsAndPreflights() {
  const Matrix rows = matrix({{0.25, 0.25}, {0.75, 0.75}, {1.0, 1.0}});
  const std::array<Vector, 2U> edges{vector({0.0, 0.5, 1.0}),
                                     vector({0.0, 0.5, 1.0})};
  const WeightSpec unweighted{.rule = WeightRule::unweighted,
                              .rowWeights = std::nullopt,
                              .componentWeights = std::nullopt,
                              .frequencyCounts = {}};
  const SampleView x{.rows = rows.view(), .observation_shape = {}, .valid = {}};
  const auto result = histogramND(x, edges, unweighted,
                                  HistogramNormalization::counts, ample());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().extents.view().size() == 2U);
  VSS_CHECK(result.value().extents.view()[0U] == 2U);
  VSS_CHECK(result.value().extents.view()[1U] == 2U);
  close(result.value().bins.at(0U), 1.0);
  close(result.value().bins.at(1U), 0.0);
  close(result.value().bins.at(2U), 0.0);
  close(result.value().bins.at(3U), 2.0);

  Limits capped = ample();
  capped.elements = 3U;
  const auto denied =
      histogramND(x, edges, unweighted, HistogramNormalization::counts, capped);
  VSS_CHECK(!denied.hasValue());
  VSS_CHECK(denied.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  oneDimensionalHistogramsUseSpecifiedEdgesAndMass();
  multidimensionalHistogramUsesRowMajorBinsAndPreflights();
}
