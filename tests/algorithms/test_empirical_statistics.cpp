#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Empirical.hpp>
#include <vss/math/statistics/Weighted.hpp>

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

void productKernelsSupportDiagonalBandwidthsAndReflection() {
  const Matrix observations = matrix({{0.0}});
  const Matrix evaluation = matrix({{0.0}, {1.0}});
  const SampleView sample{
      .rows = observations.view(), .observation_shape = {}, .valid = {}};
  const SampleView points{
      .rows = evaluation.view(), .observation_shape = {}, .valid = {}};
  const WeightSpec unweighted{.rule = WeightRule::unweighted,
                              .rowWeights = std::nullopt,
                              .componentWeights = std::nullopt,
                              .frequencyCounts = {}};
  const Vector bandwidth = vector({1.0});
  const KernelSpec gaussian{.kernel = KernelKind::gaussian,
                            .bandwidth = bandwidth.view(),
                            .reflection_bounds = std::nullopt};
  const auto normal =
      kernelDensityEstimate(sample, unweighted, points, gaussian, ample());
  VSS_CHECK(normal.hasValue());
  close(normal.value().at(0U), 1.0 / std::sqrt(2.0 * std::numbers::pi));
  close(normal.value().at(1U),
        std::exp(-0.5) / std::sqrt(2.0 * std::numbers::pi));
  const Vector rowWeights = vector({1.0});
  const WeightSpec weighted{.rule = WeightRule::empirical,
                            .rowWeights = rowWeights.view(),
                            .componentWeights = std::nullopt,
                            .frequencyCounts = {}};
  const auto weightedDensity =
      kernelDensityEstimate(sample, weighted, points, gaussian, ample());
  VSS_CHECK(weightedDensity.hasValue());
  close(weightedDensity.value().at(0U), normal.value().at(0U));

  const Matrix epanEvaluation = matrix({{0.0}, {-1.0}, {1.0}});
  const SampleView epanPoints{
      .rows = epanEvaluation.view(), .observation_shape = {}, .valid = {}};
  const KernelSpec epan{.kernel = KernelKind::epanechnikov,
                        .bandwidth = bandwidth.view(),
                        .reflection_bounds = std::nullopt};
  const auto compact =
      kernelDensityEstimate(sample, unweighted, epanPoints, epan, ample());
  VSS_CHECK(compact.hasValue());
  close(compact.value().at(0U), 0.75);
  close(compact.value().at(1U), 0.0);
  close(compact.value().at(2U), 0.0);

  const Matrix observations2d = matrix({{0.0, 0.0}});
  const Matrix evaluation2d = matrix({{0.0, 0.0}});
  const SampleView sample2d{
      .rows = observations2d.view(), .observation_shape = {}, .valid = {}};
  const SampleView points2d{
      .rows = evaluation2d.view(), .observation_shape = {}, .valid = {}};
  const Vector diagonal = vector({2.0, 4.0});
  const KernelSpec diagonalEpan{.kernel = KernelKind::epanechnikov,
                                .bandwidth = diagonal.view(),
                                .reflection_bounds = std::nullopt};
  const auto product = kernelDensityEstimate(sample2d, unweighted, points2d,
                                             diagonalEpan, ample());
  VSS_CHECK(product.hasValue());
  close(product.value().at(0U), (0.75 / 2.0) * (0.75 / 4.0));

  const Matrix bounds = matrix({{0.0, 1.0}});
  const KernelSpec reflected{.kernel = KernelKind::gaussian,
                             .bandwidth = bandwidth.view(),
                             .reflection_bounds = bounds.view()};
  const Index gridCount = 100U;
  Matrix grid(gridCount + 1U, 1U, 0.0);
  for (Index i = 0U; i <= gridCount; ++i) {
    grid.at(i, 0U) = static_cast<Scalar>(i) / gridCount;
  }
  const SampleView gridView{
      .rows = grid.view(), .observation_shape = {}, .valid = {}};
  const ProbabilityPolicy accuracy{.iteration = {.abs = 1.0e-12,
                                                 .rel = 1.0e-12,
                                                 .max_iterations = 4096U,
                                                 .max_evaluations = 4096U,
                                                 .bracket = std::nullopt},
                                   .max_draws = 1U,
                                   .abs_log_error = 1.0e-12};
  const auto reflectedDensity = kernelDensityEstimate(
      sample, unweighted, gridView, reflected, accuracy, ample());
  VSS_CHECK(reflectedDensity.hasValue());
  long double integral = 0.0L;
  for (Index i = 1U; i < gridCount; ++i) {
    integral += reflectedDensity.value().at(i);
  }
  integral += 0.5L * (reflectedDensity.value().at(0U) +
                      reflectedDensity.value().at(gridCount));
  integral /= gridCount;
  VSS_CHECK(std::abs(integral - 1.0L) < 2.0e-5L);

  ProbabilityPolicy capped = accuracy;
  capped.iteration.max_evaluations = 1U;
  const Vector broadBandwidth = vector({10.0});
  const KernelSpec broad{.kernel = KernelKind::gaussian,
                         .bandwidth = broadBandwidth.view(),
                         .reflection_bounds = bounds.view()};
  const auto noConvergence =
      kernelDensityEstimate(sample, unweighted, points, broad, capped, ample());
  VSS_CHECK(!noConvergence.hasValue());
  VSS_CHECK(noConvergence.status().code() == ErrorCode::NoConvergence);
  Limits outputCapped = ample();
  outputCapped.elements = 1U;
  const auto outputLimit =
      kernelDensityEstimate(sample, unweighted, points, gaussian, outputCapped);
  VSS_CHECK(!outputLimit.hasValue());
  VSS_CHECK(outputLimit.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  oneDimensionalHistogramsUseSpecifiedEdgesAndMass();
  multidimensionalHistogramUsesRowMajorBinsAndPreflights();
  productKernelsSupportDiagonalBandwidthsAndReflection();
}
