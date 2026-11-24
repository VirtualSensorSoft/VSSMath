#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vss/math/probability/SamplingDesign.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

void latinHypercubeStrataAndReplay() {
  using LatinHypercube = Result<SampleDesign> (*)(Index, Index, Rng&, Limits);
  static_assert(std::is_same_v<decltype(&latinHypercube), LatinHypercube>);

  Rng first(42U);
  Rng replay(42U);
  const auto design = latinHypercube(2U, 4U, first, Limits{64U, 4096U, 0U});
  const auto repeated = latinHypercube(2U, 4U, replay, Limits{64U, 4096U, 0U});
  VSS_CHECK(design.hasValue() && repeated.hasValue());
  if (!design || !repeated) {
    return;
  }
  VSS_CHECK(design.value().points.rows() == 4U);
  VSS_CHECK(design.value().points.columns() == 2U);
  VSS_CHECK(design.value().algorithm == "latin-hypercube-fisher-yates-v1");
  VSS_CHECK(design.value().provenance.has_value());
  if (design.value().provenance) {
    const auto& provenance = *design.value().provenance;
    VSS_CHECK(provenance.kind == SamplingDesignKind::latin_hypercube);
    VSS_CHECK(provenance.algorithm_version == 1U);
    VSS_CHECK(provenance.coordinate_bits == 52U);
    VSS_CHECK(provenance.direction_bits == 0U);
    VSS_CHECK(!provenance.digital_shift);
    VSS_CHECK(!provenance.seed.has_value());
    VSS_CHECK(provenance.direction_table_sha256.empty());
    VSS_CHECK(provenance.first_index == 0U);
    VSS_CHECK(provenance.dimension == 2U);
    VSS_CHECK(provenance.count == 4U);
  }

  for (Index column = 0U; column < 2U; ++column) {
    std::array<bool, 4> seen{};
    for (Index row = 0U; row < 4U; ++row) {
      const Scalar value = design.value().points.at(row, column);
      VSS_CHECK(value >= 0.0 && value < 1.0);
      const auto stratum = static_cast<Index>(value * 4.0);
      VSS_CHECK(stratum < seen.size());
      if (stratum < seen.size()) {
        VSS_CHECK(!seen[stratum]);
        seen[stratum] = true;
      }
      VSS_CHECK(value == repeated.value().points.at(row, column));
    }
    for (const bool occupied : seen) {
      VSS_CHECK(occupied);
    }
  }
}

void validationAndLimitPreflight() {
  Rng invalidRng(9U);
  Rng untouchedRng(9U);
  const auto zeroCount =
      latinHypercube(2U, 0U, invalidRng, Limits{64U, 4096U, 0U});
  VSS_CHECK(!zeroCount.hasValue());
  VSS_CHECK(zeroCount.status().code() == ErrorCode::InvalidExtent);
  const auto capped = latinHypercube(2U, 4U, invalidRng, Limits{7U, 4096U, 0U});
  VSS_CHECK(!capped.hasValue());
  VSS_CHECK(capped.status().code() == ErrorCode::ResourceLimit);

  const auto afterInvalid =
      latinHypercube(2U, 4U, invalidRng, Limits{64U, 4096U, 0U});
  const auto untouched =
      latinHypercube(2U, 4U, untouchedRng, Limits{64U, 4096U, 0U});
  VSS_CHECK(afterInvalid.hasValue() && untouched.hasValue());
  if (afterInvalid && untouched) {
    for (Index i = 0U; i < afterInvalid.value().points.size(); ++i) {
      VSS_CHECK(afterInvalid.value().points.data()[i] ==
                untouched.value().points.data()[i]);
    }
  }
}
} // namespace

int main() {
  latinHypercubeStrataAndReplay();
  validationAndLimitPreflight();
}

// VSS-MATH-TRACEABILITY: C106 CASE-C106.latin-hypercube-strata
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-resource-limit
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-zero-count
