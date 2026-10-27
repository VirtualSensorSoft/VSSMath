#include "Test.hpp"

#include <detail/SolveOutcomeAccess.hpp>
#include <optional>
#include <type_traits>
#include <utility>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LinearSolve.hpp>

namespace {

using vss::math::Vector;
using vss::math::linear::ConditionClassification;
using vss::math::linear::ConditionReport;
using vss::math::linear::SolveOutcome;
using vss::math::linear::detail::SolveOutcomeAccess;

static_assert(!std::is_default_constructible_v<SolveOutcome<Vector>>);
static_assert(std::is_copy_constructible_v<SolveOutcome<Vector>>);
static_assert(std::is_copy_assignable_v<SolveOutcome<Vector>>);
static_assert(std::is_nothrow_move_constructible_v<SolveOutcome<Vector>>);
static_assert(std::is_nothrow_move_assignable_v<SolveOutcome<Vector>>);

} // namespace

int main() {
  auto withoutReport =
      SolveOutcomeAccess<Vector>::make(Vector(2U, 1.0), std::nullopt);
  VSS_CHECK(withoutReport.solution().size() == 2U);
  VSS_CHECK(withoutReport.solution().at(0U) == 1.0);
  VSS_CHECK(!withoutReport.conditionReport().has_value());

  const ConditionReport conditionReportValue{
      .matrixOneNorm = 2.0,
      .estimatedConditionNumber1 = 3.0,
      .conditionLimit = 1.0e10,
      .classification = ConditionClassification::WithinLimit};
  auto withReport = SolveOutcomeAccess<Vector>::make(
      Vector(2U, 2.0), std::optional<ConditionReport>{conditionReportValue});
  VSS_CHECK(withReport.solution().size() == 2U);
  VSS_CHECK(withReport.solution().at(0U) == 2.0);
  const auto& report = withReport.conditionReport();
  VSS_CHECK(report.has_value());
  if (!report.has_value()) {
    return 1;
  }
  VSS_CHECK(report->matrixOneNorm == 2.0);
  VSS_CHECK(report->estimatedConditionNumber1 == 3.0);
  VSS_CHECK(report->conditionLimit == 1.0e10);
  VSS_CHECK(report->classification == ConditionClassification::WithinLimit);

  auto copy = withReport;
  VSS_CHECK(copy.solution().at(0U) == 2.0);
  VSS_CHECK(copy.solution().size() == withReport.solution().size());
  VSS_CHECK(copy.solution().data() != withReport.solution().data());
  const auto& copiedReport = copy.conditionReport();
  VSS_CHECK(copiedReport.has_value());
  if (!copiedReport.has_value()) {
    return 1;
  }
  VSS_CHECK(copiedReport->estimatedConditionNumber1 == 3.0);

  auto moved = std::move(copy);
  VSS_CHECK(moved.solution().size() == 2U);
  VSS_CHECK(moved.solution().at(0U) == 2.0);
  const auto& movedReport = moved.conditionReport();
  VSS_CHECK(movedReport.has_value());
  if (!movedReport.has_value()) {
    return 1;
  }
  VSS_CHECK(movedReport->classification ==
            ConditionClassification::WithinLimit);
}
