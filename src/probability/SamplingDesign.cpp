#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/SamplingDesign.hpp>

namespace vss::math::probability {
namespace {

constexpr std::uint64_t coordinateCount = std::uint64_t{1U} << 52U;
constexpr Scalar inverseCoordinateCount = 0x1p-52;
constexpr std::string_view latinAlgorithm = "latin-hypercube-fisher-yates-v1";
constexpr std::string_view sobolAlgorithm = "sobol-gray-code-joe-kuo-d6-v1";
constexpr std::string_view directionTableSha256 =
    "68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441";
const std::array<std::string_view, 58U> directionTableChunks{
#include "detail/SobolJoeKuoD6.inc"
};
constexpr Index maximumSobolDimension = 21201U;
constexpr std::size_t directionBits = 64U;

struct DirectionTableCursor final {
  std::size_t chunk{0U};
  std::size_t offset{0U};
};

bool atEnd(const DirectionTableCursor& cursor) noexcept {
  return cursor.chunk >= directionTableChunks.size();
}

char peek(const DirectionTableCursor& cursor) noexcept {
  return directionTableChunks.at(cursor.chunk).at(cursor.offset);
}

void advance(DirectionTableCursor& cursor) noexcept {
  ++cursor.offset;
  if (cursor.offset == directionTableChunks.at(cursor.chunk).size()) {
    ++cursor.chunk;
    cursor.offset = 0U;
  }
}

void skipWhitespace(DirectionTableCursor& cursor) noexcept {
  while (!atEnd(cursor) && (peek(cursor) == ' ' || peek(cursor) == '\t' ||
                            peek(cursor) == '\r' || peek(cursor) == '\n')) {
    advance(cursor);
  }
}

bool skipLine(DirectionTableCursor& cursor) noexcept {
  while (!atEnd(cursor) && peek(cursor) != '\n') {
    advance(cursor);
  }
  if (atEnd(cursor)) {
    return false;
  }
  advance(cursor);
  return true;
}

bool addBytes(std::size_t value, std::size_t& total) noexcept {
  if (value > std::numeric_limits<std::size_t>::max() - total) {
    return false;
  }
  total += value;
  return true;
}

bool multiplyBytes(std::size_t count, std::size_t width,
                   std::size_t& bytes) noexcept {
  if (width != 0U && count > std::numeric_limits<std::size_t>::max() / width) {
    return false;
  }
  bytes = count * width;
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

bool readUnsigned(DirectionTableCursor& cursor, std::uint64_t& value) noexcept {
  skipWhitespace(cursor);
  if (atEnd(cursor) || peek(cursor) < '0' || peek(cursor) > '9') {
    return false;
  }
  value = 0U;
  while (!atEnd(cursor) && peek(cursor) >= '0' && peek(cursor) <= '9') {
    const auto digit = static_cast<std::uint64_t>(peek(cursor) - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
      return false;
    }
    value = value * 10U + digit;
    advance(cursor);
  }
  return true;
}

bool loadDirections(Index dimensions,
                    std::vector<std::array<std::uint64_t, directionBits>>&
                        directions) noexcept {
  for (std::size_t bit = 0U; bit < directionBits; ++bit) {
    directions.at(0U).at(bit) = std::uint64_t{1U} << (63U - bit);
  }
  if (dimensions == 1U) {
    return true;
  }

  DirectionTableCursor cursor{};
  if (!skipLine(cursor)) {
    return false;
  }
  for (Index dimension = 2U; dimension <= dimensions; ++dimension) {
    std::uint64_t rowDimension = 0U;
    std::uint64_t degree = 0U;
    std::uint64_t polynomial = 0U;
    if (!readUnsigned(cursor, rowDimension) || !readUnsigned(cursor, degree) ||
        !readUnsigned(cursor, polynomial) || rowDimension != dimension ||
        degree == 0U || degree >= directionBits ||
        polynomial >= (std::uint64_t{1U} << degree)) {
      return false;
    }

    std::array<std::uint64_t, directionBits> initial{};
    for (std::size_t index = 0U; index < degree; ++index) {
      std::uint64_t numerator = 0U;
      if (!readUnsigned(cursor, numerator) || (numerator & 1U) == 0U ||
          (index + 1U < directionBits &&
           numerator >= (std::uint64_t{1U} << (index + 1U)))) {
        return false;
      }
      initial.at(index) = numerator;
    }
    auto& values = directions.at(dimension - 1U);
    for (std::size_t index = 0U; index < degree; ++index) {
      values.at(index) = initial.at(index) << (63U - index);
    }
    for (auto index = static_cast<std::size_t>(degree); index < directionBits;
         ++index) {
      std::uint64_t value =
          values.at(index - degree) ^ (values.at(index - degree) >> degree);
      for (std::size_t coefficient = 1U; coefficient < degree; ++coefficient) {
        const auto bit = static_cast<unsigned>(degree - 1U - coefficient);
        if (((polynomial >> bit) & 1U) != 0U) {
          value ^= values.at(index - coefficient);
        }
      }
      values.at(index) = value;
    }
  }
  if (dimensions == maximumSobolDimension) {
    skipWhitespace(cursor);
    if (!atEnd(cursor)) {
      return false;
    }
  }
  return true;
}

bool checkedSobolBytes(Index outputElements, std::size_t workElements,
                       std::size_t metadataBytes, std::size_t limit,
                       std::size_t& total) noexcept {
  std::size_t outputBytes = 0U;
  std::size_t workBytes = 0U;
  total = 0U;
  return multiplyBytes(outputElements, sizeof(Scalar), outputBytes) &&
         multiplyBytes(workElements, sizeof(std::uint64_t), workBytes) &&
         addBytes(outputBytes, total) && addBytes(workBytes, total) &&
         addBytes(sizeof(SamplingDesignProvenance), total) &&
         addBytes(sizeof(SampleDesign), total) &&
         addBytes(metadataBytes, total) && total <= limit;
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
      !addBytes(latinAlgorithm.size() + 1U, peakBytes) ||
      peakBytes > limits.bytes) {
    return Result<SampleDesign>::failure(
        ErrorCode::ResourceLimit,
        "design output and workspace exceed byte limit");
  }

  try {
    SampleDesign design{
        Matrix(count, dimensions), 0U, std::string(latinAlgorithm),
        SamplingDesignProvenance{SamplingDesignKind::latin_hypercube, 1U, 52U,
                                 0U, false, std::nullopt, "", 0U, dimensions,
                                 count}};
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
        auto value = static_cast<Scalar>(fraction);
        const auto lower =
            static_cast<Scalar>(static_cast<long double>(stratum) /
                                static_cast<long double>(count));
        const auto upper =
            static_cast<Scalar>((static_cast<long double>(stratum) + 1.0L) /
                                static_cast<long double>(count));
        if (value < lower) {
          value = lower;
        }
        if (value >= upper) {
          value = std::nextafter(upper, lower);
        }
        if (value < lower || value >= upper) {
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

Result<SampleDesign> sobolDesignRange(Index dimensions,
                                      std::uint64_t firstIndex, Index count,
                                      std::uint64_t seed, bool scramble,
                                      Limits limits) {
  if (dimensions == 0U || count == 0U) {
    return Result<SampleDesign>::failure(
        ErrorCode::InvalidExtent, "Sobol dimension and count must be positive");
  }
  if (dimensions > maximumSobolDimension) {
    return Result<SampleDesign>::failure(
        ErrorCode::InvalidDomain,
        "Sobol dimension exceeds the pinned Joe-Kuo D6 table");
  }
  if constexpr (sizeof(Index) > sizeof(std::uint64_t)) {
    if (count > std::numeric_limits<std::uint64_t>::max()) {
      return Result<SampleDesign>::failure(
          ErrorCode::SizeOverflow, "Sobol count does not fit the index domain");
    }
  }
  const auto count64 = static_cast<std::uint64_t>(count);
  if (count64 - 1U > std::numeric_limits<std::uint64_t>::max() - firstIndex) {
    return Result<SampleDesign>::failure(ErrorCode::IndexOutOfRange,
                                         "Sobol range last index overflows");
  }
  if (dimensions > std::numeric_limits<Index>::max() / count) {
    return Result<SampleDesign>::failure(ErrorCode::SizeOverflow,
                                         "Sobol output extent overflows Index");
  }
  const Index outputElements = dimensions * count;
  if (outputElements > limits.elements) {
    return Result<SampleDesign>::failure(ErrorCode::ResourceLimit,
                                         "Sobol output exceeds element limit");
  }
  if (dimensions > std::numeric_limits<std::size_t>::max() / directionBits) {
    return Result<SampleDesign>::failure(ErrorCode::SizeOverflow,
                                         "Sobol direction workspace overflows");
  }
  const std::size_t directionElements = dimensions * directionBits;
  if (directionElements >
      std::numeric_limits<std::size_t>::max() - (scramble ? dimensions : 0U)) {
    return Result<SampleDesign>::failure(ErrorCode::SizeOverflow,
                                         "Sobol workspace extent overflows");
  }
  const std::size_t workElements =
      directionElements + (scramble ? dimensions : 0U);
  std::size_t peakBytes = 0U;
  if (!checkedSobolBytes(outputElements, workElements,
                         sobolAlgorithm.size() + directionTableSha256.size() +
                             2U,
                         limits.bytes, peakBytes)) {
    return Result<SampleDesign>::failure(
        ErrorCode::ResourceLimit,
        "Sobol output and workspace exceed byte limit");
  }

  try {
    Matrix points(count, dimensions);
    std::vector<std::array<std::uint64_t, directionBits>> directions(
        dimensions);
    if (!loadDirections(dimensions, directions)) {
      return Result<SampleDesign>::failure(
          ErrorCode::InvalidDomain,
          "pinned Joe-Kuo D6 direction table failed integrity checks");
    }
    std::vector<std::uint64_t> shifts;
    if (scramble) {
      shifts.resize(dimensions);
      Rng generator(seed);
      for (auto& shift : shifts) {
        shift = generator.next() >> 12U;
      }
    }

    for (Index row = 0U; row < count; ++row) {
      const std::uint64_t sequenceIndex =
          firstIndex + static_cast<std::uint64_t>(row);
      const std::uint64_t gray = sequenceIndex ^ (sequenceIndex >> 1U);
      for (Index column = 0U; column < dimensions; ++column) {
        std::uint64_t coordinate = 0U;
        std::uint64_t grayBits = gray;
        std::size_t bit = 0U;
        while (grayBits != 0U) {
          if ((grayBits & 1U) != 0U) {
            coordinate ^= directions.at(column).at(bit);
          }
          grayBits >>= 1U;
          ++bit;
        }
        coordinate >>= 12U;
        if (scramble) {
          coordinate ^= shifts[column];
        }
        points.at(row, column) = static_cast<Scalar>(coordinate) * 0x1p-52;
      }
    }

    SamplingDesignProvenance provenance{
        SamplingDesignKind::sobol,
        1U,
        52U,
        64U,
        scramble,
        scramble ? std::optional<std::uint64_t>(seed) : std::nullopt,
        std::string(directionTableSha256),
        firstIndex,
        dimensions,
        count};
    return Result<SampleDesign>::success(
        SampleDesign{std::move(points), scramble ? seed : 0U,
                     std::string(sobolAlgorithm), std::move(provenance)});
  } catch (const std::length_error& error) {
    return Result<SampleDesign>::failure(ErrorCode::SizeOverflow, error.what());
  } catch (const std::bad_alloc&) {
    return Result<SampleDesign>::failure(ErrorCode::ResourceLimit,
                                         "Sobol allocation failed");
  }
}

Result<SampleDesign> sobolDesign(Index dimensions, Index log2Count,
                                 std::uint64_t seed, bool scramble,
                                 Limits limits) {
  if (log2Count > 52U) {
    return Result<SampleDesign>::failure(
        ErrorCode::InvalidDomain,
        "Sobol prefix bit count must be in the range [0, 52]");
  }
  const std::uint64_t count64 = std::uint64_t{1U} << log2Count;
  if constexpr (sizeof(Index) < sizeof(std::uint64_t)) {
    if (count64 > std::numeric_limits<Index>::max()) {
      return Result<SampleDesign>::failure(
          ErrorCode::SizeOverflow,
          "Sobol prefix count does not fit the Index representation");
    }
  }
  const auto count = static_cast<Index>(count64);
  return sobolDesignRange(dimensions, 0U, count, seed, scramble, limits);
}

} // namespace vss::math::probability
