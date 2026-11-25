#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/statistics/Missing.hpp>

using namespace vss::math;
using namespace vss::math::statistics;

namespace {

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

MissingPolicy selection(MissingMode mode) {
  return {.mode = mode,
          .policy = {},
          .minimum_support = 1U,
          .initial_mean = std::nullopt,
          .initial_covariance = std::nullopt};
}

void selectedSamplesPreserveMaskProvenance() {
  const Matrix values = matrix({{1.0, 10.0}, {3.0, 30.0}, {5.0, 50.0}});
  const std::array<std::uint8_t, 6U> mask{1U, 1U, 1U, 0U, 1U, 1U};
  const SampleView x{
      .rows = values.view(), .observation_shape = {}, .valid = mask};

  const auto complete =
      sampleWithMissingPolicy(x, selection(MissingMode::complete_row), ample());
  VSS_CHECK(complete.hasValue());
  VSS_CHECK(complete.value().data.rows() == 2U);
  VSS_CHECK(complete.value().source_rows.view()[0U] == 0U);
  VSS_CHECK(complete.value().source_rows.view()[1U] == 2U);
  VSS_CHECK(complete.value().mask.view()[0U] == 1U);

  const auto pairwise =
      sampleWithMissingPolicy(x, selection(MissingMode::pairwise), ample());
  VSS_CHECK(pairwise.hasValue());
  VSS_CHECK(pairwise.value().data.rows() == 3U);
  VSS_CHECK(pairwise.value().data.at(1U, 1U) == 0.0);
  VSS_CHECK(pairwise.value().mask.view()[3U] == 0U);
  VSS_CHECK(pairwise.value().source_rows.view()[2U] == 2U);

  const auto reject =
      sampleWithMissingPolicy(x, selection(MissingMode::reject), ample());
  VSS_CHECK(!reject.hasValue());
  VSS_CHECK(reject.status().code() == ErrorCode::InvalidDomain);
  const auto emSelection =
      sampleWithMissingPolicy(x, selection(MissingMode::gaussian_em), ample());
  VSS_CHECK(!emSelection.hasValue());
  VSS_CHECK(emSelection.status().code() == ErrorCode::InvalidDomain);
}

void empiricalCovarianceModesRetainTheirSupportRules() {
  const Matrix values = matrix({{1.0, 1.0},
                                {3.0, std::numeric_limits<Scalar>::quiet_NaN()},
                                {5.0, 5.0}});
  const std::array<std::uint8_t, 6U> mask{1U, 1U, 1U, 0U, 1U, 1U};
  const SampleView x{
      .rows = values.view(), .observation_shape = {}, .valid = mask};
  auto completePolicy = selection(MissingMode::complete_row);
  completePolicy.minimum_support = 2U;
  const auto complete = covarianceWithMissing(x, completePolicy, ample());
  VSS_CHECK(complete.hasValue());
  VSS_CHECK(std::abs(complete.value().value.at(0U, 0U) - 4.0) < 1.0e-12);
  VSS_CHECK(std::abs(complete.value().value.at(0U, 1U) - 4.0) < 1.0e-12);
  VSS_CHECK(complete.value().diagnostics.global.valid == 2U);

  const Matrix disjoint = matrix({{0.0, 0.0, 0.0},
                                  {1.0, 1.0, 0.0},
                                  {0.0, 0.0, 0.0},
                                  {1.0, 0.0, 1.0},
                                  {0.0, 0.0, 1.0},
                                  {0.0, 1.0, 0.0}});
  const std::array<std::uint8_t, 18U> disjointMask{
      1U, 1U, 0U, 1U, 1U, 0U, 1U, 0U, 1U, 1U, 0U, 1U, 0U, 1U, 1U, 0U, 1U, 1U};
  auto pairwisePolicy = selection(MissingMode::pairwise);
  pairwisePolicy.minimum_support = 2U;
  const auto pairwise =
      covarianceWithMissing(SampleView{.rows = disjoint.view(),
                                       .observation_shape = {},
                                       .valid = disjointMask},
                            pairwisePolicy, ample());
  VSS_CHECK(pairwise.hasValue());
  VSS_CHECK(pairwise.value().diagnostics.tuple_support.view().size() == 6U);
  bool hasPsdWarning = false;
  for (const ErrorCode warning : pairwise.value().diagnostics.warnings.view()) {
    hasPsdWarning =
        hasPsdWarning || warning == ErrorCode::NotPositiveSemidefinite;
  }
  VSS_CHECK(hasPsdWarning);

  const auto strict =
      covarianceWithMissing(x, selection(MissingMode::reject), ample());
  VSS_CHECK(!strict.hasValue());
  VSS_CHECK(strict.status().code() == ErrorCode::InvalidDomain);
}

void gaussianEmRequiresIdentifiableConvergedModel() {
  const Matrix values = matrix({{-1.0}, {1.0}});
  auto policy = selection(MissingMode::gaussian_em);
  policy.minimum_support = 2U;
  policy.initial_mean = Vector(1U, 2.0);
  policy.initial_covariance = matrix({{4.0}});
  policy.policy.iteration = {.abs = 1.0e-10,
                             .rel = 1.0e-10,
                             .max_iterations = 1U,
                             .max_evaluations = 100U,
                             .bracket = std::nullopt};
  const auto oneStep = covarianceWithMissing(
      SampleView{.rows = values.view(), .observation_shape = {}, .valid = {}},
      policy, ample());
  VSS_CHECK(!oneStep.hasValue());
  VSS_CHECK(oneStep.status().code() == ErrorCode::NoConvergence);

  policy.policy.iteration.max_iterations = 20U;
  const auto fitted = covarianceWithMissing(
      SampleView{.rows = values.view(), .observation_shape = {}, .valid = {}},
      policy, ample());
  VSS_CHECK(fitted.hasValue());
  VSS_CHECK(std::abs(fitted.value().value.at(0U, 0U) - 1.0) < 1.0e-9);
  VSS_CHECK(fitted.value().diagnostics.global.valid == 2U);

  const Matrix withEmptyRow =
      matrix({{-1.0}, {1.0}, {std::numeric_limits<Scalar>::quiet_NaN()}});
  const std::array<std::uint8_t, 3U> emptyRowMask{1U, 1U, 0U};
  const auto emptyRowFit =
      covarianceWithMissing(SampleView{.rows = withEmptyRow.view(),
                                       .observation_shape = {},
                                       .valid = emptyRowMask},
                            policy, ample());
  VSS_CHECK(emptyRowFit.hasValue());
  VSS_CHECK(std::abs(emptyRowFit.value().value.at(0U, 0U) -
                     fitted.value().value.at(0U, 0U)) < 1.0e-12);

  const Matrix separated = matrix({{1.0, 0.0}, {0.0, 2.0}});
  const std::array<std::uint8_t, 4U> separatedMask{1U, 0U, 0U, 1U};
  policy.initial_mean = Vector(2U, 0.0);
  policy.initial_covariance = matrix({{1.0, 0.0}, {0.0, 1.0}});
  const auto unidentifiable =
      covarianceWithMissing(SampleView{.rows = separated.view(),
                                       .observation_shape = {},
                                       .valid = separatedMask},
                            policy, ample());
  VSS_CHECK(!unidentifiable.hasValue());
  VSS_CHECK(unidentifiable.status().code() == ErrorCode::RankDeficient);

  const std::array<std::uint8_t, 4U> absentMask{1U, 0U, 1U, 0U};
  const auto absent = covarianceWithMissing(SampleView{.rows = separated.view(),
                                                       .observation_shape = {},
                                                       .valid = absentMask},
                                            policy, ample());
  VSS_CHECK(!absent.hasValue());
  VSS_CHECK(absent.status().code() == ErrorCode::InsufficientSamples);

  policy.initial_mean = Vector(1U, 0.0);
  policy.initial_covariance = matrix({{1.0}});
  Limits capped = ample();
  capped.evaluations = 1U;
  const auto exhausted = covarianceWithMissing(
      SampleView{.rows = values.view(), .observation_shape = {}, .valid = {}},
      policy, capped);
  VSS_CHECK(!exhausted.hasValue());
  VSS_CHECK(exhausted.status().code() == ErrorCode::NoConvergence);
}

} // namespace

int main() {
  selectedSamplesPreserveMaskProvenance();
  empiricalCovarianceModesRetainTheirSupportRules();
  gaussianEmRequiresIdentifiableConvergedModel();
}
