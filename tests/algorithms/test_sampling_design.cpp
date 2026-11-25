#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/probability/Distribution.hpp>
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
  if (design.value().provenance.has_value()) {
    const auto& provenance = design.value().provenance.value();
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
        VSS_CHECK(!seen.at(stratum));
        seen.at(stratum) = true;
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
    for (Index row = 0U; row < afterInvalid.value().points.rows(); ++row) {
      for (Index column = 0U; column < afterInvalid.value().points.columns();
           ++column) {
        VSS_CHECK(afterInvalid.value().points.at(row, column) ==
                  untouched.value().points.at(row, column));
      }
    }
  }
}

void sobolPrefixRangeAndProvenance() {
  using SobolPrefix =
      Result<SampleDesign> (*)(Index, Index, std::uint64_t, bool, Limits);
  using SobolRange = Result<SampleDesign> (*)(Index, std::uint64_t, Index,
                                              std::uint64_t, bool, Limits);
  static_assert(std::is_same_v<decltype(static_cast<SobolPrefix>(&sobolDesign)),
                               SobolPrefix>);
  static_assert(
      std::is_same_v<decltype(static_cast<SobolRange>(&sobolDesignRange)),
                     SobolRange>);
  const auto prefix = sobolDesign(4U, 2U, 77U, false, Limits{64U, 16384U, 0U});
  VSS_CHECK(prefix.hasValue());
  if (!prefix) {
    return;
  }
  constexpr std::array<std::array<Scalar, 4>, 4> expected{{
      {{0.0, 0.0, 0.0, 0.0}},
      {{0.5, 0.5, 0.5, 0.5}},
      {{0.75, 0.25, 0.25, 0.25}},
      {{0.25, 0.75, 0.75, 0.75}},
  }};
  VSS_CHECK(prefix.value().points.rows() == expected.size());
  VSS_CHECK(prefix.value().points.columns() == expected.front().size());
  for (Index row = 0U; row < expected.size(); ++row) {
    for (Index column = 0U; column < expected.at(row).size(); ++column) {
      VSS_CHECK(prefix.value().points.at(row, column) ==
                expected.at(row).at(column));
    }
  }
  VSS_CHECK(prefix.value().provenance.has_value());
  if (prefix.value().provenance.has_value()) {
    const auto& provenance = *prefix.value().provenance;
    VSS_CHECK(provenance.kind == SamplingDesignKind::sobol);
    VSS_CHECK(provenance.coordinate_bits == 52U);
    VSS_CHECK(provenance.direction_bits == 64U);
    VSS_CHECK(!provenance.digital_shift);
    VSS_CHECK(!provenance.seed.has_value());
    VSS_CHECK(
        provenance.direction_table_sha256 ==
        "68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441");
    VSS_CHECK(provenance.first_index == 0U);
    VSS_CHECK(provenance.dimension == 4U);
    VSS_CHECK(provenance.count == 4U);
  }

  const auto shifted = sobolDesign(2U, 2U, 42U, true, Limits{64U, 16384U, 0U});
  constexpr std::array<std::array<std::uint64_t, 2>, 4> shiftedExpected{{
      {{377685245215468ULL, 1706775315665171ULL}},
      {{2629485058900716ULL, 3958575129350419ULL}},
      {{3755384965743340ULL, 580875408822547ULL}},
      {{1503585152058092ULL, 2832675222507795ULL}},
  }};
  VSS_CHECK(shifted.hasValue());
  if (shifted) {
    for (Index row = 0U; row < shiftedExpected.size(); ++row) {
      for (Index column = 0U; column < shiftedExpected.at(row).size();
           ++column) {
        const auto actual = static_cast<std::uint64_t>(
            shifted.value().points.at(row, column) * 0x1p52);
        VSS_CHECK(actual == shiftedExpected.at(row).at(column));
      }
    }
    if (shifted.value().provenance.has_value()) {
      const auto& provenance = shifted.value().provenance.value();
      VSS_CHECK(provenance.seed == 42U);
      VSS_CHECK(provenance.digital_shift);
    } else {
      VSS_CHECK(false);
    }
  }

  const auto shiftedPrefix =
      sobolDesign(3U, 4U, 42U, true, Limits{64U, 131072U, 0U});
  const auto shiftedRange =
      sobolDesignRange(3U, 11U, 5U, 42U, true, Limits{32U, 65536U, 0U});
  const auto shiftedReplay =
      sobolDesignRange(3U, 11U, 5U, 42U, true, Limits{32U, 65536U, 0U});
  const auto changedSeed =
      sobolDesignRange(3U, 11U, 5U, 43U, true, Limits{32U, 65536U, 0U});
  VSS_CHECK(shiftedPrefix.hasValue() && shiftedRange.hasValue() &&
            shiftedReplay.hasValue() && changedSeed.hasValue());
  if (shiftedPrefix && shiftedRange && shiftedReplay && changedSeed) {
    for (Index row = 0U; row < 5U; ++row) {
      for (Index column = 0U; column < 3U; ++column) {
        VSS_CHECK(shiftedRange.value().points.at(row, column) ==
                  shiftedPrefix.value().points.at(row + 11U, column));
        VSS_CHECK(shiftedRange.value().points.at(row, column) ==
                  shiftedReplay.value().points.at(row, column));
      }
    }
    bool changed = false;
    for (Index row = 0U; row < 5U; ++row) {
      for (Index column = 0U; column < 3U; ++column) {
        changed = changed || shiftedRange.value().points.at(row, column) !=
                                 changedSeed.value().points.at(row, column);
      }
    }
    VSS_CHECK(changed);
  }

  for (const Index dimension : {4U, 8U}) {
    const auto full = sobolDesign(dimension, 4U, 42U, false,
                                  Limits{dimension * 16U, 131072U, 0U});
    const auto first = sobolDesignRange(dimension, 0U, 8U, 42U, false,
                                        Limits{dimension * 16U, 131072U, 0U});
    const auto second = sobolDesignRange(dimension, 8U, 8U, 42U, false,
                                         Limits{dimension * 16U, 131072U, 0U});
    const auto interior = sobolDesignRange(dimension, 3U, 6U, 42U, false,
                                           Limits{dimension * 8U, 65536U, 0U});
    VSS_CHECK(full.hasValue() && first.hasValue() && second.hasValue() &&
              interior.hasValue());
    if (full && first && second && interior &&
        first.value().provenance.has_value() &&
        second.value().provenance.has_value() &&
        interior.value().provenance.has_value()) {
      for (Index row = 0U; row < 8U; ++row) {
        for (Index column = 0U; column < dimension; ++column) {
          VSS_CHECK(first.value().points.at(row, column) ==
                    full.value().points.at(row, column));
          VSS_CHECK(second.value().points.at(row, column) ==
                    full.value().points.at(row + 8U, column));
        }
      }
      for (Index row = 0U; row < 6U; ++row) {
        for (Index column = 0U; column < dimension; ++column) {
          VSS_CHECK(interior.value().points.at(row, column) ==
                    full.value().points.at(row + 3U, column));
        }
      }
      VSS_CHECK(first.value().provenance.value().first_index == 0U);
      VSS_CHECK(second.value().provenance.value().first_index == 8U);
      VSS_CHECK(interior.value().provenance.value().first_index == 3U);
    }
  }
}

void sobolEdgesAndFullTable() {
  constexpr auto maxIndex = std::numeric_limits<std::uint64_t>::max();
  const auto last =
      sobolDesignRange(1U, maxIndex, 1U, 0U, false, Limits{1U, 8192U, 0U});
  VSS_CHECK(last.hasValue());
  if (last) {
    const auto coordinate =
        static_cast<std::uint64_t>(last.value().points.at(0U, 0U) * 0x1p52);
    VSS_CHECK(coordinate == 0U);
  }
  const auto overflow =
      sobolDesignRange(1U, maxIndex, 2U, 0U, false, Limits{2U, 8192U, 0U});
  VSS_CHECK(!overflow.hasValue());
  VSS_CHECK(overflow.status().code() == ErrorCode::IndexOutOfRange);
  const auto zero =
      sobolDesignRange(1U, 0U, 0U, 0U, false, Limits{1U, 8192U, 0U});
  VSS_CHECK(!zero.hasValue());
  VSS_CHECK(zero.status().code() == ErrorCode::InvalidExtent);
  const auto invalidDimension =
      sobolDesign(0U, 0U, 0U, false, Limits{1U, 8192U, 0U});
  VSS_CHECK(!invalidDimension.hasValue());
  VSS_CHECK(invalidDimension.status().code() == ErrorCode::InvalidExtent);
  const auto unsupportedDimension =
      sobolDesign(21202U, 0U, 0U, false, Limits{21202U, 20000000U, 0U});
  VSS_CHECK(!unsupportedDimension.hasValue());
  VSS_CHECK(unsupportedDimension.status().code() == ErrorCode::InvalidDomain);
  const auto unsupportedBits =
      sobolDesign(1U, 53U, 0U, false, Limits{1U, 8192U, 0U});
  VSS_CHECK(!unsupportedBits.hasValue());
  VSS_CHECK(unsupportedBits.status().code() == ErrorCode::InvalidDomain);
  const auto elementLimit =
      sobolDesign(2U, 2U, 0U, false, Limits{7U, 16384U, 0U});
  VSS_CHECK(!elementLimit.hasValue());
  VSS_CHECK(elementLimit.status().code() == ErrorCode::ResourceLimit);
  const auto byteLimit =
      sobolDesignRange(2U, 0U, 1U, 0U, false, Limits{2U, 1024U, 0U});
  VSS_CHECK(!byteLimit.hasValue());
  VSS_CHECK(byteLimit.status().code() == ErrorCode::ResourceLimit);
  const auto extentOverflow =
      sobolDesignRange(2U, 0U, std::numeric_limits<Index>::max(), 0U, false,
                       Limits{std::numeric_limits<std::size_t>::max(),
                              std::numeric_limits<std::size_t>::max(), 0U});
  VSS_CHECK(!extentOverflow.hasValue());
  VSS_CHECK(extentOverflow.status().code() == ErrorCode::SizeOverflow);

  // Requesting all dimensions validates every pinned table row and boundary.
  const auto allDimensions =
      sobolDesign(21201U, 0U, 0U, false,
                  Limits{21201U, std::size_t{16U} * 1024U * 1024U, 0U});
  VSS_CHECK(allDimensions.hasValue());
  if (allDimensions) {
    VSS_CHECK(allDimensions.value().points.rows() == 1U);
    VSS_CHECK(allDimensions.value().points.columns() == 21201U);
  }
}
} // namespace

int main() {
  latinHypercubeStrataAndReplay();
  validationAndLimitPreflight();
  sobolPrefixRangeAndProvenance();
  sobolEdgesAndFullTable();
}

// VSS-MATH-TRACEABILITY: C106 CASE-C106.latin-hypercube-strata
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-resource-limit
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-zero-count
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.carry
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.multidimensional
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.digital-shift
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.chunk-replay
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-chunk-prefix
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-interior
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-shift-replay
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-overflow
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-dimension
// VSS-MATH-TRACEABILITY: C106 CASE-C106.sobol.range-resource-limit
