#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/probability/SamplingDesign.hpp>

namespace vss::math::probability {
namespace {

constexpr std::uint64_t coordinateCount = std::uint64_t{1U} << 52U;
constexpr Scalar inverseCoordinateCount = 0x1p-52;
constexpr char latinAlgorithm[] = "latin-hypercube-fisher-yates-v1";

bool addBytes(std::size_t value, std::size_t& total) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - total) {
    return false;
  }
  total += value;
  return true;
}

std::uint64_t bounded(Rng& rng, std::uint64_t bound) noexcept {
  const std::uint64_t threshold = (std::uint64_t{0U} - bound) % bound;
  for (;;) {
    const std::uint64_t value = rng.next();
    if (value >= threshold) {
      return value % bound;
    }
  }
}

Scalar openUniform(Rng& rng) noexcept {
  const auto highBits = static_cast<Scalar>(rng.next() >> 12U);
  Scalar value = (highBits + 0.5) * inverseCoordinateCount;
  if (value >= 1.0) {
    value = std::nextafter(1.0, 0.0);
  }
  if (value <= 0.0) {
    value = std::nextafter(0.0, 1.0);
  }
  return value;
}

} // namespace

Result<SampleDesign> latinHypercube(Index dimensions, Index count, Rng& rng,
                                    Limits limits) {
  if (dimensions == 0U || count == 0U) {
    return Result<SampleDesign>::failure(
        ErrorCode::InvalidExtent,
        "Latin hypercube dimensions and count must be positive");
  }
  if (count > coordinateCount) {
    return Result<SampleDesign>::failure(
        ErrorCode::InvalidDomain,
        "Latin hypercube count exceeds 52-bit coordinate resolution");
  }
  if (dimensions > std::numeric_limits<Index>::max() / count) {
    return Result<SampleDesign>::failure(ErrorCode::SizeOverflow,
                                         "design extent overflows Index");
  }
  const Index outputElements = dimensions * count;
  if (outputElements > limits.elements) {
    return Result<SampleDesign>::failure(ErrorCode::ResourceLimit,
                                         "design exceeds element limit");
  }

  std::size_t peakBytes = 0U;
  if (outputElements >
          std::numeric_limits<std::size_t>::max() / sizeof(Scalar) ||
      count > std::numeric_limits<std::size_t>::max() / sizeof(Index) ||
      !addBytes(outputElements * sizeof(Scalar), peakBytes) ||
      !addBytes(count * sizeof(Index), peakBytes) ||
      !addBytes(sizeof(SampleDesign) + sizeof(SamplingDesignProvenance),
                peakBytes) ||
      !addBytes(sizeof(latinAlgorithm), peakBytes) ||
      peakBytes > limits.bytes) {
    return Result<SampleDesign>::failure(
        ErrorCode::ResourceLimit,
        "design output and workspace exceed byte limit");
  }

  try {
    SampleDesign design{Matrix(count, dimensions), 0U, latinAlgorithm,
                        SamplingDesignProvenance{
                            SamplingDesignKind::latin_hypercube, 1U, 52U, 0U,
                            false, std::nullopt, "", 0U, dimensions, count}};
    std::vector<Index> permutation(count);
    for (Index column = 0U; column < dimensions; ++column) {
      std::iota(permutation.begin(), permutation.end(), Index{0U});
      for (Index remaining = count; remaining > 1U; --remaining) {
        const auto selected = static_cast<Index>(
            bounded(rng, static_cast<std::uint64_t>(remaining)));
        std::swap(permutation[remaining - 1U], permutation[selected]);
      }
      for (Index row = 0U; row < count; ++row) {
        const Index stratum = permutation[row];
        const long double fraction =
            (static_cast<long double>(stratum) + openUniform(rng)) /
            static_cast<long double>(count);
        Scalar value = static_cast<Scalar>(fraction);
        const Scalar lower =
            static_cast<Scalar>(static_cast<long double>(stratum) /
                                static_cast<long double>(count));
        const Scalar upper =
            static_cast<Scalar>((static_cast<long double>(stratum) + 1.0L) /
                                static_cast<long double>(count));
        if (value < lower) {
          value = lower;
        }
        if (value >= upper) {
          value = std::nextafter(upper, lower);
        }
        if (!(value >= lower && value < upper)) {
          return Result<SampleDesign>::failure(
              ErrorCode::InvalidDomain,
              "a Latin hypercube stratum has no representable interior point");
        }
        design.points.at(row, column) = value;
      }
    }
    return Result<SampleDesign>::success(std::move(design));
  } catch (const std::length_error& error) {
    return Result<SampleDesign>::failure(ErrorCode::SizeOverflow, error.what());
  } catch (const std::bad_alloc&) {
    return Result<SampleDesign>::failure(ErrorCode::ResourceLimit,
                                         "design allocation failed");
  }
}

} // namespace vss::math::probability
