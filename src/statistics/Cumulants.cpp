#include "MomentInterval.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Cumulants.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {
namespace {

using detail::MomentCertificate;
using detail::MomentInterval;

[[nodiscard]] Result<MomentEstimate> failure(ErrorCode code,
                                             const char* message) {
  return Result<MomentEstimate>::failure(code, message);
}

[[nodiscard]] bool addSize(Extent left, Extent right, Extent& result) {
  if (right > std::numeric_limits<Extent>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] bool multiplySize(Extent left, Extent right, Extent& result) {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool shapeDimension(const MomentEstimate& source,
                                  Extent& dimension) {
  dimension = 1U;
  for (const Index extent : source.observation_shape.view()) {
    if (extent == 0U || !multiplySize(dimension, extent, dimension)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool addStorage(Extent count, Extent elementSize, Extent& total) {
  Extent bytes = 0U;
  return multiplySize(count, elementSize, bytes) &&
         addSize(total, bytes, total);
}

[[nodiscard]] bool tupleCount(Extent dimension, Index order,
                              MomentRepresentation layout, Extent& count) {
  if (layout == MomentRepresentation::component) {
    count = dimension;
    return true;
  }
  if (layout == MomentRepresentation::dense) {
    count = 1U;
    for (Index axis = 0U; axis < order; ++axis) {
      if (!multiplySize(count, dimension, count)) {
        return false;
      }
    }
    return true;
  }
  if (layout == MomentRepresentation::packed) {
    // Number of nondecreasing tuples with repetition: C(d + q - 1, q).
    Extent numerator = 1U;
    for (Index axis = 1U; axis <= order; ++axis) {
      Extent factor = dimension + axis - 1U;
      if (factor < dimension) {
        return false;
      }
      Extent divisor = axis;
      const Extent common = std::gcd(factor, divisor);
      factor /= common;
      divisor /= common;
      const Extent commonProduct = std::gcd(numerator, divisor);
      numerator /= commonProduct;
      divisor /= commonProduct;
      if (divisor != 1U || !multiplySize(numerator, factor, numerator)) {
        return false;
      }
    }
    count = numerator;
    return true;
  }
  return false;
}

[[nodiscard]] bool chooseCount(Extent n, Extent k, Extent& result) {
  if (k > n) {
    result = 0U;
    return true;
  }
  k = std::min(k, n - k);
  Extent value = 1U;
  for (Extent index = 1U; index <= k; ++index) {
    Extent numerator = n - k + index;
    Extent denominator = index;
    const Extent first = std::gcd(numerator, denominator);
    numerator /= first;
    denominator /= first;
    const Extent second = std::gcd(value, denominator);
    value /= second;
    denominator /= second;
    if (denominator != 1U || !multiplySize(value, numerator, value)) {
      return false;
    }
  }
  result = value;
  return true;
}

struct Entry final {
  Index order;
  std::array<Index, 4U> coordinates{};
  Scalar value;
};

void canonicalize(std::array<Index, 4U>& coordinates, Index order) {
  for (Index index = 1U; index < order; ++index) {
    const Index value = coordinates.at(index);
    Index position = index;
    while (position > 0U && coordinates.at(position - 1U) > value) {
      coordinates.at(position) = coordinates.at(position - 1U);
      --position;
    }
    coordinates.at(position) = value;
  }
}

[[nodiscard]] bool closeEnough(Scalar left, Scalar right) {
  return std::abs(left - right) <=
         1.0e-12 + 1.0e-10 * std::max(std::abs(left), std::abs(right));
}

[[nodiscard]] bool sameKey(const Entry& left, Index order,
                           const std::array<Index, 4U>& coordinates) {
  return left.order == order && left.coordinates == coordinates;
}

class ExactFourth final {
public:
  void add(Scalar value, bool subtract) {
    const auto decoded = decode(value);
    const std::array<std::uint32_t, 2U> significand{
        static_cast<std::uint32_t>(decoded.significand),
        static_cast<std::uint32_t>(decoded.significand >> 32U)};
    addInteger(significand, decoded.exponent, decoded.negative != subtract);
  }

  void addProduct(Scalar left, Scalar right, bool subtract) {
    const auto a = decode(left);
    const auto b = decode(right);
    std::array<std::uint32_t, 4U> product{};
    multiply(a.significand, b.significand, product);
    addInteger(product, a.exponent + b.exponent,
               (a.negative != b.negative) != subtract);
  }

  [[nodiscard]] bool rounded(Scalar& value) const {
    const bool negative = less(positive_, negative_);
    const Big& magnitude = negative ? subtract(negative_, positive_)
                                    : subtract(positive_, negative_);
    const Index top = highestBit(magnitude);
    if (top == noBit) {
      value = 0.0;
      return true;
    }
    const Index shift = std::max<Index>(top > 52U ? top - 52U : 0U, 1074U);
    std::uint64_t significand = shiftedLow(magnitude, shift);
    if (shift != 0U && testBit(magnitude, shift - 1U) &&
        (anyBelow(magnitude, shift - 1U) || (significand & 1U) != 0U)) {
      ++significand;
    }
    const double roundedMagnitude = std::scalbn(
        static_cast<double>(significand), static_cast<int>(shift) - 2148);
    if (!std::isfinite(roundedMagnitude)) {
      return false;
    }
    value = negative ? -roundedMagnitude : roundedMagnitude;
    return true;
  }

private:
  // Exact products of binary64 inputs span exponents -2148 through 2047;
  // 136 32-bit limbs include headroom for summing the four signed terms.
  static constexpr Index limbCount = 136U;
  static constexpr Index noBit = std::numeric_limits<Index>::max();
  using Big = std::array<std::uint32_t, limbCount>;
  struct Binary final {
    std::uint64_t significand;
    int exponent;
    bool negative;
  };

  [[nodiscard]] static Binary decode(double value) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponentBits = static_cast<unsigned>((bits >> 52U) & 0x7ffU);
    const auto fraction = bits & ((std::uint64_t{1U} << 52U) - 1U);
    if (exponentBits == 0U) {
      return {fraction, -1074, (bits >> 63U) != 0U};
    }
    return {fraction | (std::uint64_t{1U} << 52U),
            static_cast<int>(exponentBits) - 1075, (bits >> 63U) != 0U};
  }

  static void multiply(std::uint64_t left, std::uint64_t right,
                       std::array<std::uint32_t, 4U>& result) {
    const std::array<std::uint32_t, 2U> a{
        static_cast<std::uint32_t>(left),
        static_cast<std::uint32_t>(left >> 32U)};
    const std::array<std::uint32_t, 2U> b{
        static_cast<std::uint32_t>(right),
        static_cast<std::uint32_t>(right >> 32U)};
    for (Index i = 0U; i < 2U; ++i) {
      std::uint64_t carry = 0U;
      for (Index j = 0U; j < 2U; ++j) {
        const std::uint64_t current =
            static_cast<std::uint64_t>(a.at(i)) * b.at(j) + result.at(i + j) +
            carry;
        result.at(i + j) = static_cast<std::uint32_t>(current);
        carry = current >> 32U;
      }
      Index position = i + 2U;
      while (carry != 0U && position < result.size()) {
        const auto current =
            static_cast<std::uint64_t>(result.at(position)) + carry;
        result.at(position) = static_cast<std::uint32_t>(current);
        ++position;
        carry = current >> 32U;
      }
    }
  }

  template <Index Size>
  void addInteger(const std::array<std::uint32_t, Size>& value, int exponent,
                  bool negative) {
    if (exponent < -2148) {
      return;
    }
    Big& destination = negative ? negative_ : positive_;
    const Index offset = static_cast<Index>(exponent) + 2148U;
    for (Index limb = 0U; limb < Size; ++limb) {
      for (Index bit = 0U; bit < 32U; ++bit) {
        if (((value.at(limb) >> bit) & 1U) == 0U) {
          continue;
        }
        Index position = offset + limb * 32U + bit;
        while (position < limbCount * 32U) {
          const Index word = position / 32U;
          const auto mask = std::uint32_t{1U} << (position % 32U);
          if ((destination.at(word) & mask) == 0U) {
            destination.at(word) |= mask;
            break;
          }
          destination.at(word) &= ~mask;
          ++position;
        }
      }
    }
  }

  [[nodiscard]] static bool less(const Big& left, const Big& right) {
    for (Index index = limbCount; index > 0U; --index) {
      if (left.at(index - 1U) != right.at(index - 1U)) {
        return left.at(index - 1U) < right.at(index - 1U);
      }
    }
    return false;
  }

  [[nodiscard]] static Big subtract(const Big& left, const Big& right) {
    Big result{};
    std::uint64_t borrow = 0U;
    for (Index index = 0U; index < limbCount; ++index) {
      const std::uint64_t subtrahend =
          static_cast<std::uint64_t>(right.at(index)) + borrow;
      const std::uint64_t minuend = left.at(index);
      result.at(index) = static_cast<std::uint32_t>(minuend - subtrahend);
      borrow = minuend < subtrahend ? 1U : 0U;
    }
    return result;
  }

  [[nodiscard]] static Index highestBit(const Big& value) {
    for (Index word = limbCount; word > 0U; --word) {
      if (value.at(word - 1U) != 0U) {
        return (word - 1U) * 32U + (31U - static_cast<Index>(std::countl_zero(
                                              value.at(word - 1U))));
      }
    }
    return noBit;
  }

  [[nodiscard]] static bool testBit(const Big& value, Index bit) {
    return ((value.at(bit / 32U) >> (bit % 32U)) & 1U) != 0U;
  }

  [[nodiscard]] static bool anyBelow(const Big& value, Index limit) {
    for (Index bit = 0U; bit < limit; ++bit) {
      if (testBit(value, bit)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] static std::uint64_t shiftedLow(const Big& value, Index shift) {
    std::uint64_t result = 0U;
    for (Index bit = 0U; bit < 64U && shift + bit < limbCount * 32U; ++bit) {
      if (testBit(value, shift + bit)) {
        result |= std::uint64_t{1U} << bit;
      }
    }
    return result;
  }

  Big positive_{};
  Big negative_{};
};

[[nodiscard]] bool exactFourth(Scalar fourth, Scalar left0, Scalar left1,
                               Scalar left2, Scalar right0, Scalar right1,
                               Scalar right2, bool forward, Scalar& value,
                               Scalar& roundingError) {
  ExactFourth exact;
  exact.add(fourth, false);
  for (Index pairing = 0U; pairing < 3U; ++pairing) {
    Scalar left = left2;
    Scalar right = right2;
    if (pairing == 0U) {
      left = left0;
      right = right0;
    } else if (pairing == 1U) {
      left = left1;
      right = right1;
    }
    exact.addProduct(left, right, forward);
  }
  if (!exact.rounded(value)) {
    return false;
  }
  roundingError =
      value == 0.0
          ? std::numeric_limits<Scalar>::denorm_min()
          : std::max(std::numeric_limits<Scalar>::denorm_min(),
                     std::nextafter(std::abs(value) *
                                        std::numeric_limits<Scalar>::epsilon(),
                                    std::numeric_limits<Scalar>::infinity()));
  return true;
}

[[nodiscard]] bool supportEqual(const Support& left, const Support& right) {
  return left.input == right.input && left.valid == right.valid &&
         left.positive == right.positive &&
         left.frequency_mass == right.frequency_mass &&
         left.weight_mass == right.weight_mass &&
         left.normalized_weight_powers == right.normalized_weight_powers &&
         left.effective_count == right.effective_count &&
         left.degrees_of_freedom == right.degrees_of_freedom &&
         left.squared_weight_mass == right.squared_weight_mass;
}

[[nodiscard]] bool compatibleSupport(const EstimatorDiagnostics& diagnostics,
                                     const std::array<Index, 4U>& tuple,
                                     Index order) {
  const auto coordinateSupports = diagnostics.coordinate_support.view();
  const auto tupleSupports = diagnostics.tuple_support.view();
  if (!coordinateSupports.empty()) {
    if (tuple.at(0U) >= coordinateSupports.size()) {
      return false;
    }
    for (Index axis = 1U; axis < order; ++axis) {
      if (tuple.at(axis) >= coordinateSupports.size() ||
          !supportEqual(coordinateSupports[tuple.at(0U)],
                        coordinateSupports[tuple.at(axis)])) {
        return false;
      }
    }
  }
  const Support* first = nullptr;
  for (const auto& item : tupleSupports) {
    const auto coordinates = item.coordinates.view();
    if (coordinates.size() != order) {
      continue;
    }
    bool matches = true;
    for (Index axis = 0U; axis < order; ++axis) {
      matches = matches && coordinates[axis] == tuple.at(axis);
    }
    if (!matches) {
      continue;
    }
    if (first == nullptr) {
      first = &item.support;
    } else if (!supportEqual(*first, item.support)) {
      return false;
    }
  }
  if (first != nullptr && !coordinateSupports.empty()) {
    return supportEqual(*first, coordinateSupports[tuple.at(0U)]);
  }
  return true;
}

[[nodiscard]] std::array<Index, 4U> tupleFor(Index ordinal, Index order,
                                             Extent dimension,
                                             MomentRepresentation layout) {
  std::array<Index, 4U> tuple{};
  if (layout == MomentRepresentation::component) {
    for (Index axis = 0U; axis < order; ++axis) {
      tuple.at(axis) = ordinal;
    }
    return tuple;
  }
  if (layout == MomentRepresentation::dense) {
    for (Index axis = order; axis > 0U; --axis) {
      tuple.at(axis - 1U) = ordinal % dimension;
      ordinal /= dimension;
    }
    return tuple;
  }
  for (Index axis = 0U; axis < order; ++axis) {
    const Index remaining = order - axis - 1U;
    Index coordinate = axis == 0U ? 0U : tuple.at(axis - 1U);
    for (; coordinate < dimension; ++coordinate) {
      Extent suffix = 0U;
      if (!chooseCount(dimension - coordinate + remaining - 1U, remaining,
                       suffix)) {
        return {};
      }
      if (ordinal < suffix) {
        break;
      }
      ordinal -= suffix;
    }
    tuple.at(axis) = coordinate;
  }
  return tuple;
}

[[nodiscard]] bool validRequest(const MomentRequest& request,
                                Extent dimension) {
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U ||
      request.contraction.has_value()) {
    return false;
  }
  if (request.representation == MomentRepresentation::contraction) {
    return false;
  }
  if (request.representation != MomentRepresentation::selected) {
    return request.selected.empty();
  }
  if (request.selected.empty()) {
    return false;
  }
  Index selectedOrder = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) != 0U) {
      if (selectedOrder != 0U) {
        return false;
      }
      selectedOrder = order;
    }
  }
  for (const auto tuple : request.selected) {
    if (tuple.size() != selectedOrder) {
      return false;
    }
    for (const Index coordinate : tuple) {
      if (coordinate >= dimension) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool collectEntries(const MomentEstimate& source,
                                  Extent dimension,
                                  std::vector<Entry>& entries) {
  for (const auto& term : source.terms.view()) {
    if (term.order < 1U || term.order > 4U ||
        term.layout == MomentRepresentation::contraction) {
      return false;
    }
    Extent count = 0U;
    if (term.layout == MomentRepresentation::selected) {
      Extent expectedTupleSize = 0U;
      if (!multiplySize(term.values.size(), term.order, expectedTupleSize) ||
          term.flattened_tuples.view().size() != expectedTupleSize) {
        return false;
      }
      count = term.values.size();
    } else {
      if (!term.flattened_tuples.view().empty() ||
          !tupleCount(dimension, term.order, term.layout, count) ||
          count != term.values.size()) {
        return false;
      }
    }
    for (Index ordinal = 0U; ordinal < count; ++ordinal) {
      Entry entry{.order = term.order, .coordinates = {}, .value = 0.0};
      if (term.layout == MomentRepresentation::selected) {
        for (Index axis = 0U; axis < term.order; ++axis) {
          entry.coordinates.at(axis) =
              term.flattened_tuples.view()[ordinal * term.order + axis];
          if (entry.coordinates.at(axis) >= dimension) {
            return false;
          }
        }
      } else {
        entry.coordinates =
            tupleFor(ordinal, term.order, dimension, term.layout);
      }
      if (!std::isfinite(term.values.at(ordinal))) {
        return false;
      }
      entry.value = term.values.at(ordinal);
      canonicalize(entry.coordinates, term.order);
      entries.push_back(entry);
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry& left, const Entry& right) {
              if (left.order != right.order) {
                return left.order < right.order;
              }
              if (left.coordinates != right.coordinates) {
                return left.coordinates < right.coordinates;
              }
              return left.value < right.value;
            });
  for (Index index = 1U; index < entries.size(); ++index) {
    if (sameKey(entries[index], entries[index - 1U].order,
                entries[index - 1U].coordinates) &&
        !closeEnough(entries[index].value, entries[index - 1U].value)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] const Entry* findEntry(const std::vector<Entry>& entries,
                                     Index order,
                                     std::array<Index, 4U> coordinates) {
  canonicalize(coordinates, order);
  const auto found = std::lower_bound(entries.begin(), entries.end(),
                                      std::pair{order, coordinates},
                                      [](const Entry& entry, const auto& key) {
                                        if (entry.order != key.first) {
                                          return entry.order < key.first;
                                        }
                                        return entry.coordinates < key.second;
                                      });
  return found != entries.end() && sameKey(*found, order, coordinates)
             ? &*found
             : nullptr;
}

[[nodiscard]] Result<MomentEstimate> convert(const MomentEstimate& source,
                                             const MomentRequest& request,
                                             Limits limits, bool forward) {
  Extent dimension = 0U;
  if (!shapeDimension(source, dimension) || !validRequest(request, dimension)) {
    return failure(ErrorCode::InvalidDomain,
                   "invalid moment shape or conversion request");
  }
  if (forward && source.diagnostics.center != CenterRule::estimated) {
    return failure(ErrorCode::InvalidDomain,
                   "central moments must be centered on the estimated mean");
  }
  if (source.population_mean.has_value() &&
      source.population_mean->size() != dimension) {
    return failure(ErrorCode::DimensionMismatch,
                   "population mean does not match moment coordinates");
  }
  if (source.population_mean.has_value()) {
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      if (!std::isfinite(source.population_mean->at(coordinate))) {
        return failure(ErrorCode::NonFiniteInput,
                       "population mean metadata contains a non-finite value");
      }
    }
  }
  if (forward && (request.orders & 0x01U) != 0U &&
      !source.population_mean.has_value()) {
    return failure(ErrorCode::InvalidDomain,
                   "first cumulant requires the population mean");
  }

  const Extent shapeCount = source.observation_shape.view().size();
  Extent estimatedBytes = 0U;
  if (!multiplySize(shapeCount, sizeof(Index), estimatedBytes) ||
      estimatedBytes > limits.bytes) {
    return failure(ErrorCode::ResourceLimit,
                   "moment conversion exceeds configured limits");
  }

  std::vector<Entry> entries;
  Extent inputCount = 0U;
  for (const auto& term : source.terms.view()) {
    if (!addSize(inputCount, term.values.size(), inputCount)) {
      return failure(ErrorCode::SizeOverflow, "moment input size overflow");
    }
  }
  Extent workspaceBytes = 0U;
  if (!multiplySize(inputCount, sizeof(Entry), workspaceBytes)) {
    return failure(ErrorCode::SizeOverflow, "moment workspace size overflow");
  }
  Extent outputElements = 0U;
  Extent outputBytes = 0U;
  Extent selectedTupleElements = 0U;
  Index outputTermCount = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    Extent count = request.representation == MomentRepresentation::selected
                       ? request.selected.size()
                       : 0U;
    if (request.representation != MomentRepresentation::selected &&
        !tupleCount(dimension, order, request.representation, count)) {
      return failure(ErrorCode::SizeOverflow, "output moment size overflow");
    }
    if (!addSize(outputElements, count, outputElements) ||
        !addStorage(count, sizeof(Scalar), outputBytes)) {
      return failure(ErrorCode::SizeOverflow, "output moment size overflow");
    }
    if (request.representation == MomentRepresentation::selected) {
      Extent tupleElements = 0U;
      if (!multiplySize(order, request.selected.size(), tupleElements) ||
          !addSize(selectedTupleElements, tupleElements,
                   selectedTupleElements) ||
          !addStorage(tupleElements, sizeof(Index), outputBytes)) {
        return failure(ErrorCode::SizeOverflow,
                       "selected tuple metadata size overflow");
      }
    }
    ++outputTermCount;
  }
  const bool mayCopyMean =
      source.population_mean.has_value() ||
      (!forward &&
       std::any_of(source.terms.view().begin(), source.terms.view().end(),
                   [](const MomentTerm& term) { return term.order == 1U; }));
  if (mayCopyMean && !addSize(outputElements, dimension, outputElements)) {
    return failure(ErrorCode::SizeOverflow, "population mean size overflow");
  }
  Extent peakOutputBytes = 0U;
  if (!multiplySize(outputBytes, 4U, peakOutputBytes) ||
      !addSize(peakOutputBytes, workspaceBytes, peakOutputBytes) ||
      !addStorage(shapeCount, 2U * sizeof(Index), peakOutputBytes) ||
      !addStorage(outputTermCount, 3U * sizeof(MomentTerm), peakOutputBytes) ||
      !addSize(peakOutputBytes,
               sizeof(ExactFourth) + 4U * sizeof(MomentInterval),
               peakOutputBytes) ||
      !addStorage(dimension, sizeof(Scalar), peakOutputBytes) ||
      !addStorage(source.diagnostics.coordinate_support.view().size(),
                  sizeof(Support), peakOutputBytes) ||
      !addStorage(source.diagnostics.tuple_support.view().size(),
                  sizeof(TupleSupport), peakOutputBytes) ||
      !addStorage(source.diagnostics.warnings.view().size(), sizeof(ErrorCode),
                  peakOutputBytes) ||
      !addStorage(source.diagnostics.algorithm.size() +
                      source.diagnostics.premises.size(),
                  sizeof(char), peakOutputBytes) ||
      !addSize(peakOutputBytes, 128U, peakOutputBytes)) {
    return failure(ErrorCode::SizeOverflow,
                   "moment conversion peak storage size overflow");
  }
  for (const auto& support : source.diagnostics.tuple_support.view()) {
    if (!addStorage(support.coordinates.view().size(), sizeof(Index),
                    peakOutputBytes)) {
      return failure(ErrorCode::SizeOverflow, "support metadata size overflow");
    }
  }
  Extent outputIndexElements = 0U;
  if (!addSize(shapeCount, selectedTupleElements, outputIndexElements)) {
    return failure(ErrorCode::SizeOverflow,
                   "moment index metadata size overflow");
  }
  if (outputElements > limits.elements ||
      outputIndexElements > limits.elements || peakOutputBytes > limits.bytes) {
    return failure(ErrorCode::ResourceLimit,
                   "moment conversion workspace exceeds configured limits");
  }
  entries.reserve(inputCount);
  if (!collectEntries(source, dimension, entries)) {
    return failure(ErrorCode::InvalidDomain,
                   "moment terms are malformed, non-finite, or inconsistent");
  }

  MomentEstimate output{.observation_shape = source.observation_shape,
                        .terms = {},
                        .diagnostics = source.diagnostics,
                        .population_mean = source.population_mean};
  std::vector<MomentTerm> outputTerms;
  Scalar maximumError = 0.0;
  output.diagnostics.algorithm =
      forward ? "central-moments-to-cumulants" : "cumulants-to-central-moments";
  if (!forward && !output.population_mean.has_value()) {
    const Entry* mean = findEntry(entries, 1U, {});
    if (mean != nullptr) {
      // A full first-order term is needed to reconstruct all mean coordinates.
      std::vector<Scalar> values(dimension);
      bool complete = true;
      for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
        const std::array<Index, 4U> key{coordinate, 0U, 0U, 0U};
        const auto* item = findEntry(entries, 1U, key);
        if (item == nullptr) {
          complete = false;
          break;
        }
        values[coordinate] = item->value;
      }
      if (complete) {
        output.population_mean.emplace(values);
      }
    }
  }
  if (!forward && source.population_mean.has_value()) {
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      const std::array<Index, 4U> key{coordinate, 0U, 0U, 0U};
      const auto* item = findEntry(entries, 1U, key);
      if (item != nullptr &&
          item->value != source.population_mean->at(coordinate)) {
        return failure(
            ErrorCode::InvalidDomain,
            "first cumulant conflicts with population mean metadata");
      }
    }
  }

  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    Extent count = request.representation == MomentRepresentation::selected
                       ? request.selected.size()
                       : 0U;
    if (request.representation != MomentRepresentation::selected &&
        !tupleCount(dimension, order, request.representation, count)) {
      return failure(ErrorCode::SizeOverflow, "output moment size overflow");
    }
    Extent bytes = 0U;
    if (count > limits.elements ||
        !multiplySize(count, sizeof(Scalar), bytes) || bytes > limits.bytes) {
      return failure(ErrorCode::ResourceLimit,
                     "moment output exceeds configured limits");
    }
    std::vector<Scalar> values;
    values.reserve(count);
    for (Index outputIndex = 0U; outputIndex < count; ++outputIndex) {
      const auto tuple =
          request.representation == MomentRepresentation::selected
              ? [&] {
                  std::array<Index, 4U> result{};
                  const auto selected = request.selected[outputIndex];
                  std::copy(selected.begin(), selected.end(), result.begin());
                  return result;
                }()
              : tupleFor(outputIndex, order, dimension, request.representation);
      Scalar value = 0.0;
      if (order == 1U && !forward) {
        value = 0.0;
      } else if (forward && order == 1U) {
        if (!source.population_mean.has_value()) {
          return failure(ErrorCode::InvalidDomain,
                         "first cumulant requires the population mean");
        }
        value = source.population_mean->at(tuple.at(0U));
      } else {
        const auto* sourceTerm = findEntry(entries, order, tuple);
        if (sourceTerm == nullptr) {
          return failure(ErrorCode::InvalidDomain,
                         "a requested moment dependency is missing");
        }
        value = sourceTerm->value;
        if (order == 4U) {
          MomentInterval result = MomentInterval::point(value);
          std::array<Scalar, 3U> leftValues{};
          std::array<Scalar, 3U> rightValues{};
          for (Index firstPair = 0U; firstPair < 3U; ++firstPair) {
            std::array<Index, 4U> leftPair{};
            std::array<Index, 4U> rightPair{};
            if (firstPair == 0U) {
              leftPair = {tuple.at(0U), tuple.at(1U), 0U, 0U};
              rightPair = {tuple.at(2U), tuple.at(3U), 0U, 0U};
            } else if (firstPair == 1U) {
              leftPair = {tuple.at(0U), tuple.at(2U), 0U, 0U};
              rightPair = {tuple.at(1U), tuple.at(3U), 0U, 0U};
            } else {
              leftPair = {tuple.at(0U), tuple.at(3U), 0U, 0U};
              rightPair = {tuple.at(1U), tuple.at(2U), 0U, 0U};
            }
            const auto* left = findEntry(entries, 2U, leftPair);
            const auto* right = findEntry(entries, 2U, rightPair);
            if (left == nullptr || right == nullptr ||
                !compatibleSupport(source.diagnostics, tuple, 4U) ||
                !compatibleSupport(source.diagnostics, leftPair, 2U) ||
                !compatibleSupport(source.diagnostics, rightPair, 2U)) {
              return failure(ErrorCode::InvalidDomain,
                             "a fourth-order pairing dependency is missing or "
                             "incompatible");
            }
            leftValues.at(firstPair) = left->value;
            rightValues.at(firstPair) = right->value;
            const auto product = MomentInterval::point(left->value) *
                                 MomentInterval::point(right->value);
            result = forward ? result - product : result + product;
          }
          Scalar absoluteError = 0.0;
          const auto certificate =
              detail::certifyMoment(result, value, absoluteError);
          if (certificate != MomentCertificate::accepted) {
            if (!exactFourth(
                    sourceTerm->value, leftValues.at(0U), leftValues.at(1U),
                    leftValues.at(2U), rightValues.at(0U), rightValues.at(1U),
                    rightValues.at(2U), forward, value, absoluteError)) {
              return failure(ErrorCode::NonFiniteResult,
                             "exact cumulant conversion is not representable");
            }
          }
          maximumError = std::max(maximumError, absoluteError);
        }
      }
      if (!std::isfinite(value)) {
        return failure(ErrorCode::NonFiniteResult,
                       "cumulant conversion produced a non-finite value");
      }
      if (!compatibleSupport(source.diagnostics, tuple, order)) {
        return failure(ErrorCode::InvalidDomain,
                       "moment coordinates have incompatible support");
      }
      values.push_back(value);
    }
    MomentTerm term{.order = order,
                    .layout = request.representation,
                    .flattened_tuples = {},
                    .values = Vector(values)};
    if (request.representation == MomentRepresentation::selected) {
      std::vector<Index> flattened;
      flattened.reserve(order * request.selected.size());
      for (const auto selected : request.selected) {
        flattened.insert(flattened.end(), selected.begin(), selected.end());
      }
      auto tupleArray = Array<Index>::copy(flattened, limits);
      if (!tupleArray) {
        return failure(ErrorCode::ResourceLimit,
                       "selected tuple metadata exceeds configured limits");
      }
      term.flattened_tuples = std::move(tupleArray).value();
    }
    outputTerms.push_back(std::move(term));
  }
  Extent termBytes = 0U;
  if (!multiplySize(outputTerms.size(), sizeof(MomentTerm), termBytes) ||
      termBytes > limits.bytes || outputTerms.size() > limits.elements) {
    return failure(ErrorCode::ResourceLimit,
                   "moment term metadata exceeds configured limits");
  }
  auto copiedTerms = Array<MomentTerm>::copy(outputTerms, limits);
  if (!copiedTerms) {
    return failure(ErrorCode::ResourceLimit,
                   "moment term metadata exceeds configured limits");
  }
  output.terms = std::move(copiedTerms).value();
  if (maximumError > 0.0) {
    output.diagnostics.numerical_error = maximumError;
  }
  return Result<MomentEstimate>::success(std::move(output));
}

class ExactProductSum final {
public:
  static constexpr Index maximumFactors = 9U;

  [[nodiscard]] bool add(std::span<const Scalar> factors) {
    if (factors.empty() || factors.size() > maximumFactors) {
      return false;
    }
    std::array<std::uint32_t, 16U> product{};
    product.at(0U) = 1U;
    Index used = 1U;
    int exponent = 0;
    bool negative = false;
    for (const Scalar factor : factors) {
      if (!std::isfinite(factor)) {
        return false;
      }
      const auto decoded = decode(factor);
      if (decoded.significand == 0U) {
        return true;
      }
      exponent += decoded.exponent;
      negative = negative != decoded.negative;
      const std::array<std::uint32_t, 2U> multiplier{
          static_cast<std::uint32_t>(decoded.significand),
          static_cast<std::uint32_t>(decoded.significand >> 32U)};
      std::array<std::uint32_t, 16U> next{};
      for (Index left = 0U; left < used; ++left) {
        std::uint64_t carry = 0U;
        for (Index right = 0U; right < multiplier.size(); ++right) {
          const Index position = left + right;
          const std::uint64_t current =
              static_cast<std::uint64_t>(product.at(left)) *
                  multiplier.at(right) +
              next.at(position) + carry;
          next.at(position) = static_cast<std::uint32_t>(current);
          carry = current >> 32U;
        }
        Index position = left + multiplier.size();
        while (carry != 0U) {
          if (position >= next.size()) {
            return false;
          }
          const std::uint64_t current = next.at(position) + carry;
          next.at(position) = static_cast<std::uint32_t>(current);
          carry = current >> 32U;
          ++position;
        }
      }
      product = next;
      used = std::min<Index>(used + 2U, product.size());
      while (used > 1U && product.at(used - 1U) == 0U) {
        --used;
      }
    }
    return addInteger(product, used, exponent, negative);
  }

  [[nodiscard]] bool rounded(Scalar& value) const {
    const bool negative = less(positive_, negative_);
    const Big& larger = negative ? negative_ : positive_;
    const Big& smaller = negative ? positive_ : negative_;
    const Big magnitude = subtract(larger, smaller);
    const Index top = highestBit(magnitude);
    if (top == noBit) {
      value = 0.0;
      return true;
    }
    constexpr Index subnormalShift = 9666U - 1074U;
    const Index shift = std::max(top > 52U ? top - 52U : 0U, subnormalShift);
    std::uint64_t significand = shiftedLow(magnitude, shift);
    if (shift != 0U && testBit(magnitude, shift - 1U) &&
        (anyBelow(magnitude, shift - 1U) || (significand & 1U) != 0U)) {
      ++significand;
    }
    const Scalar roundedMagnitude = std::scalbn(
        static_cast<Scalar>(significand), static_cast<int>(shift) - 9666);
    if (!std::isfinite(roundedMagnitude)) {
      return false;
    }
    value = negative ? -roundedMagnitude : roundedMagnitude;
    return true;
  }

private:
  static constexpr int minimumExponent = -9666;
  static constexpr Index limbCount = 600U;
  static constexpr Index noBit = std::numeric_limits<Index>::max();
  using Big = std::array<std::uint32_t, limbCount>;
  struct Binary final {
    std::uint64_t significand;
    int exponent;
    bool negative;
  };

  [[nodiscard]] static Binary decode(Scalar value) {
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto exponentBits = static_cast<unsigned>((bits >> 52U) & 0x7ffU);
    const auto fraction = bits & ((std::uint64_t{1U} << 52U) - 1U);
    if (exponentBits == 0U) {
      return {fraction, -1074, (bits >> 63U) != 0U};
    }
    return {fraction | (std::uint64_t{1U} << 52U),
            static_cast<int>(exponentBits) - 1075, (bits >> 63U) != 0U};
  }

  [[nodiscard]] bool addInteger(const std::array<std::uint32_t, 16U>& value,
                                Index size, int exponent, bool negative) {
    if (exponent < minimumExponent) {
      return false;
    }
    Big& destination = negative ? negative_ : positive_;
    const auto bitOffset = static_cast<Index>(exponent - minimumExponent);
    const Index wordOffset = bitOffset / 32U;
    const auto bitShift = static_cast<unsigned>(bitOffset % 32U);
    const auto addWord = [&destination](Index position, std::uint32_t addend) {
      while (addend != 0U) {
        if (position >= destination.size()) {
          return false;
        }
        const std::uint64_t sum =
            static_cast<std::uint64_t>(destination.at(position)) + addend;
        destination.at(position) = static_cast<std::uint32_t>(sum);
        addend = static_cast<std::uint32_t>(sum >> 32U);
        ++position;
      }
      return true;
    };
    for (Index index = 0U; index < size; ++index) {
      const Index position = wordOffset + index;
      if (bitShift == 0U) {
        if (!addWord(position, value.at(index))) {
          return false;
        }
        continue;
      }
      const auto low = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(value.at(index)) << bitShift);
      const auto high =
          static_cast<std::uint32_t>(value.at(index) >> (32U - bitShift));
      if (!addWord(position, low) || !addWord(position + 1U, high)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] static bool less(const Big& left, const Big& right) {
    for (Index index = limbCount; index > 0U; --index) {
      if (left.at(index - 1U) != right.at(index - 1U)) {
        return left.at(index - 1U) < right.at(index - 1U);
      }
    }
    return false;
  }

  [[nodiscard]] static Big subtract(const Big& left, const Big& right) {
    Big result{};
    std::uint64_t borrow = 0U;
    for (Index index = 0U; index < limbCount; ++index) {
      const std::uint64_t subtrahend =
          static_cast<std::uint64_t>(right.at(index)) + borrow;
      const std::uint64_t minuend = left.at(index);
      result.at(index) = static_cast<std::uint32_t>(minuend - subtrahend);
      borrow = minuend < subtrahend ? 1U : 0U;
    }
    return result;
  }

  [[nodiscard]] static Index highestBit(const Big& value) {
    for (Index word = limbCount; word > 0U; --word) {
      const auto current = value.at(word - 1U);
      if (current != 0U) {
        return (word - 1U) * 32U +
               (31U - static_cast<Index>(std::countl_zero(current)));
      }
    }
    return noBit;
  }

  [[nodiscard]] static bool testBit(const Big& value, Index bit) {
    return ((value.at(bit / 32U) >> (bit % 32U)) & 1U) != 0U;
  }

  [[nodiscard]] static bool anyBelow(const Big& value, Index limit) {
    for (Index bit = 0U; bit < limit; ++bit) {
      if (testBit(value, bit)) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] static std::uint64_t shiftedLow(const Big& value, Index shift) {
    std::uint64_t result = 0U;
    for (Index bit = 0U; bit < 64U && shift + bit < limbCount * 32U; ++bit) {
      if (testBit(value, shift + bit)) {
        result |= std::uint64_t{1U} << bit;
      }
    }
    return result;
  }

  Big positive_{};
  Big negative_{};
};

[[nodiscard]] bool validIndependentRequest(const MomentRequest& request,
                                           Extent inputDimension,
                                           Extent outputDimension) {
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U) {
    return false;
  }
  if (request.representation == MomentRepresentation::contraction) {
    return request.contraction.has_value() && request.selected.empty() &&
           request.contraction->rows() == outputDimension &&
           request.contraction->columns() == inputDimension;
  }
  if (request.contraction.has_value()) {
    return false;
  }
  if (request.representation == MomentRepresentation::selected) {
    if (request.selected.empty()) {
      return false;
    }
    Index selectedOrder = 0U;
    for (Index order = 1U; order <= 4U; ++order) {
      if ((request.orders & (1U << (order - 1U))) != 0U) {
        if (selectedOrder != 0U) {
          return false;
        }
        selectedOrder = order;
      }
    }
    for (const auto tuple : request.selected) {
      if (tuple.size() != selectedOrder) {
        return false;
      }
      for (const Index coordinate : tuple) {
        if (coordinate >= outputDimension) {
          return false;
        }
      }
    }
    return true;
  }
  return request.selected.empty() &&
         (request.representation == MomentRepresentation::component ||
          request.representation == MomentRepresentation::packed ||
          request.representation == MomentRepresentation::dense);
}

[[nodiscard]] bool sameShape(const MomentEstimate& left,
                             const MomentEstimate& right) {
  const auto a = left.observation_shape.view();
  const auto b = right.observation_shape.view();
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

[[nodiscard]] bool addPopulationMeanEntries(const MomentEstimate& input,
                                            Extent dimension,
                                            std::vector<Entry>& entries) {
  if (!input.population_mean.has_value()) {
    return true;
  }
  if (input.population_mean->size() != dimension) {
    return false;
  }
  for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
    const Scalar mean = input.population_mean->at(coordinate);
    if (!std::isfinite(mean)) {
      return false;
    }
    const std::array<Index, 4U> key{coordinate, 0U, 0U, 0U};
    const Entry* existing = findEntry(entries, 1U, key);
    if (existing != nullptr && existing->value != mean) {
      return false;
    }
    if (existing == nullptr) {
      entries.push_back(Entry{.order = 1U, .coordinates = key, .value = mean});
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry& left, const Entry& right) {
              if (left.order != right.order) {
                return left.order < right.order;
              }
              if (left.coordinates != right.coordinates) {
                return left.coordinates < right.coordinates;
              }
              return left.value < right.value;
            });
  return true;
}

[[nodiscard]] Result<MomentEstimate>
independentCumulantSum(std::span<const MomentEstimate> inputs,
                       ConstVectorView coefficients,
                       const MomentRequest& request, Limits limits) {
  if (inputs.empty()) {
    return failure(ErrorCode::InsufficientSamples,
                   "independent sum requires at least one input");
  }
  if (inputs.size() != coefficients.size()) {
    return failure(ErrorCode::DimensionMismatch,
                   "input and coefficient counts differ");
  }
  Extent dimension = 0U;
  if (!shapeDimension(inputs.front(), dimension)) {
    return failure(ErrorCode::InvalidDomain,
                   "independent sum input shape is invalid");
  }
  const bool contracted =
      request.representation == MomentRepresentation::contraction;
  if (contracted && !request.contraction.has_value()) {
    return failure(ErrorCode::InvalidDomain,
                   "independent sum contraction is missing");
  }
  const Extent outputDimension =
      contracted ? request.contraction->rows() : dimension;
  if (!validIndependentRequest(request, dimension, outputDimension)) {
    return failure(ErrorCode::InvalidDomain,
                   "independent sum request is malformed");
  }
  if (contracted && outputDimension == 0U) {
    return failure(ErrorCode::InvalidExtent,
                   "contraction output dimension must be positive");
  }
  for (Index row = 0U; row < outputDimension && contracted; ++row) {
    for (Index column = 0U; column < dimension; ++column) {
      if (!std::isfinite(request.contraction->at(row, column))) {
        return failure(ErrorCode::NonFiniteInput,
                       "contraction map contains a non-finite value");
      }
    }
  }
  for (Index input = 0U; input < inputs.size(); ++input) {
    if (!sameShape(inputs.front(), inputs[input])) {
      return failure(ErrorCode::DimensionMismatch,
                     "independent sum input shapes differ");
    }
    if (!std::isfinite(coefficients.at(input))) {
      return failure(ErrorCode::NonFiniteInput,
                     "independent sum coefficient is non-finite");
    }
  }

  Index conventionIndex = inputs.size();
  Extent inputValues = 0U;
  Extent meanElements = 0U;
  std::vector<Extent> perInputValues(inputs.size(), 0U);
  for (Index input = 0U; input < inputs.size(); ++input) {
    if (coefficients.at(input) == 0.0) {
      continue;
    }
    if (conventionIndex == inputs.size()) {
      conventionIndex = input;
    } else if (inputs[input].diagnostics.center !=
               inputs[conventionIndex].diagnostics.center) {
      return failure(ErrorCode::InvalidDomain,
                     "independent inputs use incompatible center conventions");
    }
    if (!addSize(meanElements, dimension, meanElements)) {
      return failure(ErrorCode::SizeOverflow,
                     "independent sum mean workspace overflows");
    }
    for (const auto& term : inputs[input].terms.view()) {
      if (!addSize(inputValues, term.values.size(), inputValues) ||
          !addSize(perInputValues.at(input), term.values.size(),
                   perInputValues.at(input))) {
        return failure(ErrorCode::SizeOverflow,
                       "independent sum input size overflows");
      }
    }
    if (inputs[input].population_mean.has_value() &&
        (!addSize(inputValues, dimension, inputValues) ||
         !addSize(perInputValues.at(input), dimension,
                  perInputValues.at(input)))) {
      return failure(ErrorCode::SizeOverflow,
                     "independent sum mean size overflows");
    }
  }

  Extent outputValues = 0U;
  Extent outputIndexValues = 0U;
  Index outputTermCount = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    Extent count = request.representation == MomentRepresentation::selected
                       ? request.selected.size()
                       : 0U;
    const MomentRepresentation countLayout =
        contracted ? MomentRepresentation::dense : request.representation;
    if (request.representation != MomentRepresentation::selected &&
        !tupleCount(outputDimension, order, countLayout, count)) {
      return failure(ErrorCode::SizeOverflow,
                     "independent sum output size overflows");
    }
    if (!addSize(outputValues, count, outputValues)) {
      return failure(ErrorCode::SizeOverflow,
                     "independent sum output count overflows");
    }
    if (request.representation == MomentRepresentation::selected) {
      Extent tupleSize = 0U;
      if (!multiplySize(order, request.selected.size(), tupleSize) ||
          !addSize(outputIndexValues, tupleSize, outputIndexValues)) {
        return failure(ErrorCode::SizeOverflow,
                       "independent sum selected index size overflows");
      }
    }
    ++outputTermCount;
  }
  if (outputValues > limits.elements || outputIndexValues > limits.elements) {
    return failure(ErrorCode::ResourceLimit,
                   "independent sum output exceeds element limits");
  }
  const Extent outputShapeSize =
      contracted ? 1U : inputs.front().observation_shape.view().size();
  Extent peakBytes = 0U;
  if (!addStorage(inputValues, sizeof(Entry), peakBytes) ||
      !addStorage(inputs.size(), sizeof(std::vector<Entry>), peakBytes) ||
      !addStorage(inputs.size(), sizeof(std::vector<Scalar>) + sizeof(Extent),
                  peakBytes) ||
      !addStorage(meanElements, sizeof(Scalar), peakBytes) ||
      !addStorage(outputValues, 3U * sizeof(Scalar), peakBytes) ||
      !addStorage(outputIndexValues, 3U * sizeof(Index), peakBytes) ||
      !addStorage(outputTermCount, 3U * sizeof(MomentTerm), peakBytes) ||
      !addSize(peakBytes, sizeof(ExactProductSum), peakBytes) ||
      !addStorage(outputDimension, 2U * sizeof(Scalar), peakBytes) ||
      !addStorage(outputShapeSize, 2U * sizeof(Index), peakBytes) ||
      !addSize(peakBytes, 256U, peakBytes) || peakBytes > limits.bytes) {
    return failure(ErrorCode::ResourceLimit,
                   "independent sum workspace exceeds byte limits");
  }

  std::vector<std::vector<Entry>> indexedInputs(inputs.size());
  std::vector<std::vector<Scalar>> means(inputs.size());
  bool meansComplete = true;
  bool allZero = true;
  for (Index input = 0U; input < inputs.size(); ++input) {
    const Scalar coefficient = coefficients.at(input);
    if (coefficient == 0.0) {
      continue;
    }
    allZero = false;
    auto& entries = indexedInputs.at(input);
    entries.reserve(perInputValues.at(input));
    if (!collectEntries(inputs[input], dimension, entries) ||
        !addPopulationMeanEntries(inputs[input], dimension, entries)) {
      return failure(ErrorCode::InvalidDomain,
                     "independent sum input terms or mean are malformed");
    }
    auto& inputMean = means.at(input);
    inputMean.resize(dimension);
    bool complete = true;
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      const std::array<Index, 4U> key{coordinate, 0U, 0U, 0U};
      const Entry* mean = findEntry(entries, 1U, key);
      if (mean == nullptr) {
        complete = false;
        break;
      }
      inputMean.at(coordinate) = mean->value;
    }
    if (!complete) {
      inputMean.clear();
      meansComplete = false;
    }
  }
  if (allZero) {
    meansComplete = true;
  }

  std::vector<Index> outputShape;
  if (contracted) {
    outputShape.push_back(outputDimension);
  } else {
    const auto shape = inputs.front().observation_shape.view();
    outputShape.assign(shape.begin(), shape.end());
  }
  auto copiedShape = Array<Index>::copy(outputShape, limits);
  if (!copiedShape) {
    return failure(ErrorCode::ResourceLimit,
                   "independent sum output shape exceeds limits");
  }
  const auto& convention = conventionIndex == inputs.size()
                               ? inputs.front().diagnostics
                               : inputs[conventionIndex].diagnostics;
  const Support noSupport{.input = 0U,
                          .valid = 0U,
                          .positive = 0U,
                          .frequency_mass = std::nullopt,
                          .weight_mass = std::nullopt,
                          .normalized_weight_powers = {},
                          .effective_count = std::nullopt,
                          .degrees_of_freedom = std::nullopt,
                          .squared_weight_mass = std::nullopt};
  EstimatorDiagnostics diagnostics{
      .estimand = convention.estimand,
      .weight = WeightRule::unweighted,
      .center = conventionIndex == inputs.size() ? CenterRule::estimated
                                                 : convention.center,
      .approximation = Approximation::exact,
      .global = noSupport,
      .coordinate_support = {},
      .tuple_support = {},
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "independent-cumulant-sum",
      .premises =
          "input records are mutually independent; operand supports and "
          "sampling diagnostics are not combined",
      .warnings = {}};
  MomentEstimate output{.observation_shape = std::move(copiedShape).value(),
                        .terms = {},
                        .diagnostics = std::move(diagnostics),
                        .population_mean = std::nullopt};

  Scalar maximumNumericalError = 0.0;
  const auto recordRoundingBound = [&maximumNumericalError](Scalar value) {
    const Scalar bound =
        value == 0.0 ? std::numeric_limits<Scalar>::denorm_min()
                     : std::max(std::numeric_limits<Scalar>::denorm_min(),
                                std::nextafter(
                                    std::abs(value) *
                                        std::numeric_limits<Scalar>::epsilon(),
                                    std::numeric_limits<Scalar>::infinity()));
    maximumNumericalError = std::max(maximumNumericalError, bound);
  };
  std::vector<Scalar> outputMean(outputDimension, 0.0);
  bool outputMeanComplete = contracted || meansComplete;
  for (Index outputCoordinate = 0U; outputCoordinate < outputDimension;
       ++outputCoordinate) {
    ExactProductSum sum;
    bool coordinateComplete = true;
    for (Index input = 0U; input < inputs.size(); ++input) {
      const Scalar coefficient = coefficients.at(input);
      if (coefficient == 0.0) {
        continue;
      }
      for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
        Scalar mapCoefficient = 1.0;
        if (contracted) {
          mapCoefficient =
              request.contraction->at(outputCoordinate, coordinate);
        } else if (coordinate != outputCoordinate) {
          continue;
        }
        if (mapCoefficient == 0.0) {
          continue;
        }
        if (means.at(input).empty()) {
          coordinateComplete = false;
          break;
        }
        const std::array<Scalar, 3U> factors{coefficient, mapCoefficient,
                                             means.at(input).at(coordinate)};
        if (!sum.add(factors)) {
          return failure(ErrorCode::NonFiniteResult,
                         "independent sum mean exceeds exact range");
        }
      }
      if (!coordinateComplete) {
        break;
      }
    }
    if (!coordinateComplete) {
      outputMeanComplete = false;
      continue;
    }
    if (!sum.rounded(outputMean.at(outputCoordinate))) {
      return failure(ErrorCode::NonFiniteResult,
                     "independent sum mean is not representable");
    }
    recordRoundingBound(outputMean.at(outputCoordinate));
  }
  if (outputMeanComplete) {
    output.population_mean.emplace(outputMean);
  }

  std::vector<MomentTerm> outputTerms;
  outputTerms.reserve(outputTermCount);
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    const MomentRepresentation outputLayout =
        contracted ? MomentRepresentation::dense : request.representation;
    Extent count = request.representation == MomentRepresentation::selected
                       ? request.selected.size()
                       : 0U;
    if (request.representation != MomentRepresentation::selected &&
        !tupleCount(outputDimension, order, outputLayout, count)) {
      return failure(ErrorCode::SizeOverflow,
                     "independent sum output size overflows");
    }
    std::vector<Scalar> values;
    values.reserve(count);
    for (Index outputIndex = 0U; outputIndex < count; ++outputIndex) {
      std::array<Index, 4U> outputTuple{};
      if (request.representation == MomentRepresentation::selected) {
        const auto selected = request.selected[outputIndex];
        std::copy(selected.begin(), selected.end(), outputTuple.begin());
      } else {
        outputTuple =
            tupleFor(outputIndex, order, outputDimension, outputLayout);
      }
      ExactProductSum sum;
      if (contracted) {
        Extent sourceTupleCount = 0U;
        if (!tupleCount(dimension, order, MomentRepresentation::dense,
                        sourceTupleCount)) {
          return failure(ErrorCode::SizeOverflow,
                         "contraction source tuple count overflows");
        }
        for (Index sourceOrdinal = 0U; sourceOrdinal < sourceTupleCount;
             ++sourceOrdinal) {
          const auto sourceTuple = tupleFor(sourceOrdinal, order, dimension,
                                            MomentRepresentation::dense);
          std::array<Scalar, 4U> mapCoefficients{};
          bool structuralZero = false;
          for (Index axis = 0U; axis < order; ++axis) {
            mapCoefficients.at(axis) = request.contraction->at(
                outputTuple.at(axis), sourceTuple.at(axis));
            structuralZero = structuralZero || mapCoefficients.at(axis) == 0.0;
          }
          if (structuralZero) {
            continue;
          }
          for (Index input = 0U; input < inputs.size(); ++input) {
            const Scalar coefficient = coefficients.at(input);
            if (coefficient == 0.0) {
              continue;
            }
            const Entry* term =
                findEntry(indexedInputs.at(input), order, sourceTuple);
            if (term == nullptr || !compatibleSupport(inputs[input].diagnostics,
                                                      sourceTuple, order)) {
              return failure(ErrorCode::InvalidDomain,
                             "a contracted independent-sum dependency is "
                             "missing or incompatible");
            }
            std::array<Scalar, ExactProductSum::maximumFactors> factors{};
            Index factorCount = 0U;
            for (Index power = 0U; power < order; ++power) {
              factors.at(factorCount++) = coefficient;
            }
            for (Index axis = 0U; axis < order; ++axis) {
              factors.at(factorCount++) = mapCoefficients.at(axis);
            }
            factors.at(factorCount++) = term->value;
            if (!sum.add(std::span<const Scalar>(factors).first(factorCount))) {
              return failure(ErrorCode::NonFiniteResult,
                             "contracted independent sum exceeds exact range");
            }
          }
        }
      } else {
        for (Index input = 0U; input < inputs.size(); ++input) {
          const Scalar coefficient = coefficients.at(input);
          if (coefficient == 0.0) {
            continue;
          }
          const Entry* term =
              findEntry(indexedInputs.at(input), order, outputTuple);
          if (term == nullptr || !compatibleSupport(inputs[input].diagnostics,
                                                    outputTuple, order)) {
            return failure(ErrorCode::InvalidDomain,
                           "an independent-sum moment dependency is missing "
                           "or incompatible");
          }
          std::array<Scalar, ExactProductSum::maximumFactors> factors{};
          Index factorCount = 0U;
          for (Index power = 0U; power < order; ++power) {
            factors.at(factorCount++) = coefficient;
          }
          factors.at(factorCount++) = term->value;
          if (!sum.add(std::span<const Scalar>(factors).first(factorCount))) {
            return failure(ErrorCode::NonFiniteResult,
                           "independent cumulant sum exceeds exact range");
          }
        }
      }
      Scalar value = 0.0;
      if (!sum.rounded(value) || !std::isfinite(value)) {
        return failure(ErrorCode::NonFiniteResult,
                       "independent cumulant sum is not representable");
      }
      recordRoundingBound(value);
      values.push_back(value);
    }
    MomentTerm term{.order = order,
                    .layout = outputLayout,
                    .flattened_tuples = {},
                    .values = Vector(values)};
    if (request.representation == MomentRepresentation::selected) {
      std::vector<Index> flattened;
      flattened.reserve(order * request.selected.size());
      for (const auto tuple : request.selected) {
        flattened.insert(flattened.end(), tuple.begin(), tuple.end());
      }
      auto tupleArray = Array<Index>::copy(flattened, limits);
      if (!tupleArray) {
        return failure(ErrorCode::ResourceLimit,
                       "selected tuple metadata exceeds limits");
      }
      term.flattened_tuples = std::move(tupleArray).value();
    }
    outputTerms.push_back(std::move(term));
  }
  auto copiedTerms = Array<MomentTerm>::copy(outputTerms, limits);
  if (!copiedTerms) {
    return failure(ErrorCode::ResourceLimit,
                   "independent sum term metadata exceeds limits");
  }
  output.terms = std::move(copiedTerms).value();
  if (maximumNumericalError > 0.0) {
    output.diagnostics.numerical_error = maximumNumericalError;
  }
  return Result<MomentEstimate>::success(std::move(output));
}

using SelectedTupleList = std::vector<std::vector<Index>>;

[[nodiscard]] Result<SelectedTupleList>
contractionDependencies(const MomentRequest& request, Index order,
                        Extent inputDimension, Limits limits) {
  if (!request.contraction.has_value()) {
    return Result<SelectedTupleList>::failure(ErrorCode::InvalidDomain,
                                              "contraction map is missing");
  }
  const auto& map = request.contraction.value();
  Extent outputCount = 0U;
  Extent sourceCount = 0U;
  if (!tupleCount(map.rows(), order, MomentRepresentation::dense,
                  outputCount) ||
      !tupleCount(inputDimension, order, MomentRepresentation::dense,
                  sourceCount)) {
    return Result<SelectedTupleList>::failure(
        ErrorCode::SizeOverflow, "contraction tuple count overflows");
  }
  if (outputCount > limits.elements) {
    return Result<SelectedTupleList>::failure(
        ErrorCode::ResourceLimit, "contraction output exceeds element limits");
  }

  std::set<std::array<Index, 4U>> unique;
  const auto maximumTuples =
      limits.bytes / (sizeof(std::array<Index, 4U>) + 3U * sizeof(void*) + 32U);
  for (Index outputOrdinal = 0U; outputOrdinal < outputCount; ++outputOrdinal) {
    const auto outputTuple =
        tupleFor(outputOrdinal, order, map.rows(), MomentRepresentation::dense);
    for (Index sourceOrdinal = 0U; sourceOrdinal < sourceCount;
         ++sourceOrdinal) {
      auto sourceTuple = tupleFor(sourceOrdinal, order, inputDimension,
                                  MomentRepresentation::dense);
      bool needed = true;
      for (Index axis = 0U; axis < order; ++axis) {
        if (map.at(outputTuple.at(axis), sourceTuple.at(axis)) == 0.0) {
          needed = false;
          break;
        }
      }
      if (!needed) {
        continue;
      }
      canonicalize(sourceTuple, order);
      unique.insert(sourceTuple);
      if (unique.size() > limits.elements || unique.size() > maximumTuples) {
        return Result<SelectedTupleList>::failure(
            ErrorCode::ResourceLimit,
            "contraction dependency tuples exceed configured limits");
      }
    }
  }
  SelectedTupleList result;
  result.reserve(unique.size());
  for (const auto& tuple : unique) {
    result.emplace_back(tuple.begin(), tuple.begin() + order);
  }
  Extent tupleElements = 0U;
  Extent vectorBytes = 0U;
  Extent setBytes = 0U;
  Extent totalBytes = 0U;
  if (!multiplySize(result.size(), order, tupleElements) ||
      !multiplySize(result.size(),
                    sizeof(std::vector<Index>) + order * sizeof(Index),
                    vectorBytes) ||
      !multiplySize(unique.size(),
                    sizeof(std::array<Index, 4U>) + 3U * sizeof(void*) + 32U,
                    setBytes) ||
      !addSize(vectorBytes, setBytes, totalBytes) ||
      tupleElements > limits.elements || totalBytes > limits.bytes) {
    return Result<SelectedTupleList>::failure(
        ErrorCode::ResourceLimit,
        "contraction dependency storage exceeds configured limits");
  }
  return Result<SelectedTupleList>::success(std::move(result));
}

[[nodiscard]] SelectedTupleList
fourthOrderPairTuples(std::span<const std::span<const Index>> fourthTuples) {
  std::set<std::array<Index, 2U>> unique;
  for (const auto tuple : fourthTuples) {
    if (tuple.size() != 4U) {
      continue;
    }
    unique.insert(
        {std::min(tuple[0U], tuple[1U]), std::max(tuple[0U], tuple[1U])});
    unique.insert(
        {std::min(tuple[0U], tuple[2U]), std::max(tuple[0U], tuple[2U])});
    unique.insert(
        {std::min(tuple[0U], tuple[3U]), std::max(tuple[0U], tuple[3U])});
  }
  SelectedTupleList result;
  result.reserve(unique.size());
  for (const auto& tuple : unique) {
    result.emplace_back(tuple.begin(), tuple.end());
  }
  return result;
}

[[nodiscard]] Result<MomentEstimate>
convertCentralOrder(const MomentEstimate& input, const MomentRequest& base,
                    Index order, Extent inputDimension, bool contracted,
                    Limits limits, SelectedTupleList* selectedStorage) {
  MomentRequest conversionRequest = base;
  conversionRequest.orders = static_cast<std::uint8_t>(1U << (order - 1U));
  std::vector<std::span<const Index>> selectedViews;
  if (contracted) {
    auto dependencies =
        contractionDependencies(base, order, inputDimension, limits);
    if (!dependencies) {
      return Result<MomentEstimate>::failure(
          dependencies.status().code(),
          std::string(dependencies.status().message()));
    }
    *selectedStorage = std::move(dependencies).value();
  } else if (base.representation == MomentRepresentation::selected &&
             order == 2U && (base.orders & 0x08U) != 0U) {
    *selectedStorage = fourthOrderPairTuples(base.selected);
  } else if (base.representation == MomentRepresentation::selected) {
    conversionRequest.selected = base.selected;
  }
  if (contracted || (base.representation == MomentRepresentation::selected &&
                     order == 2U && (base.orders & 0x08U) != 0U)) {
    selectedViews.reserve(selectedStorage->size());
    for (const auto& tuple : *selectedStorage) {
      selectedViews.emplace_back(tuple);
    }
    if (selectedViews.empty()) {
      MomentEstimate empty{.observation_shape = input.observation_shape,
                           .terms = {},
                           .diagnostics = input.diagnostics,
                           .population_mean = input.population_mean};
      return Result<MomentEstimate>::success(std::move(empty));
    }
    conversionRequest.representation = MomentRepresentation::selected;
    conversionRequest.selected = selectedViews;
    conversionRequest.contraction = std::nullopt;
  }
  return centralMomentsToCumulants(input, conversionRequest, limits);
}

[[nodiscard]] Result<MomentEstimate>
mergeMomentEstimates(const MomentEstimate& first, const MomentEstimate& second,
                     Limits limits) {
  std::vector<MomentTerm> terms;
  const auto firstTerms = first.terms.view();
  const auto secondTerms = second.terms.view();
  Extent termCount = 0U;
  if (!addSize(firstTerms.size(), secondTerms.size(), termCount) ||
      termCount > limits.elements ||
      termCount > limits.bytes / sizeof(MomentTerm)) {
    return failure(ErrorCode::ResourceLimit,
                   "merged moment terms exceed configured limits");
  }
  terms.reserve(termCount);
  terms.insert(terms.end(), firstTerms.begin(), firstTerms.end());
  terms.insert(terms.end(), secondTerms.begin(), secondTerms.end());
  auto ownedTerms = Array<MomentTerm>::copy(terms, limits);
  if (!ownedTerms) {
    return failure(ErrorCode::ResourceLimit,
                   "merged moment terms exceed configured limits");
  }
  MomentEstimate result{.observation_shape = first.observation_shape,
                        .terms = std::move(ownedTerms).value(),
                        .diagnostics = first.diagnostics,
                        .population_mean = first.population_mean};
  if (second.diagnostics.numerical_error.has_value()) {
    const Scalar secondError = second.diagnostics.numerical_error.value();
    result.diagnostics.numerical_error =
        result.diagnostics.numerical_error.has_value()
            ? std::max(result.diagnostics.numerical_error.value(), secondError)
            : secondError;
  }
  return Result<MomentEstimate>::success(std::move(result));
}

} // namespace

Result<MomentEstimate> centralMomentsToCumulants(const MomentEstimate& moments,
                                                 const MomentRequest& request,
                                                 Limits limits) {
  return convert(moments, request, limits, true);
}

Result<MomentEstimate>
cumulantsToCentralMoments(const MomentEstimate& cumulants,
                          const MomentRequest& request, Limits limits) {
  return convert(cumulants, request, limits, false);
}

Result<MomentEstimate>
cumulantsOfIndependentSum(std::span<const MomentEstimate> inputs,
                          ConstVectorView coefficients,
                          const MomentRequest& request, Limits limits) {
  return independentCumulantSum(inputs, coefficients, request, limits);
}

Result<MomentEstimate> weightedSum(ConstVectorView coefficients,
                                   std::span<const MomentEstimate> inputs,
                                   const MomentRequest& request,
                                   Limits limits) {
  if (inputs.empty()) {
    return failure(ErrorCode::InsufficientSamples,
                   "weighted sum requires at least one input");
  }
  if (inputs.size() != coefficients.size()) {
    return failure(ErrorCode::DimensionMismatch,
                   "input and coefficient counts differ");
  }
  Extent inputDimension = 0U;
  if (!shapeDimension(inputs.front(), inputDimension)) {
    return failure(ErrorCode::InvalidDomain,
                   "weighted sum input shape is invalid");
  }
  const bool contracted =
      request.representation == MomentRepresentation::contraction;
  if (contracted && !request.contraction.has_value()) {
    return failure(ErrorCode::InvalidDomain,
                   "weighted sum contraction is missing");
  }
  const Extent outputDimension =
      contracted ? request.contraction->rows() : inputDimension;
  if (!validIndependentRequest(request, inputDimension, outputDimension)) {
    return failure(ErrorCode::InvalidDomain,
                   "weighted sum request is malformed");
  }
  bool allZero = true;
  for (Index input = 0U; input < inputs.size(); ++input) {
    if (!sameShape(inputs.front(), inputs[input])) {
      return failure(ErrorCode::DimensionMismatch,
                     "weighted sum input shapes differ");
    }
    if (!std::isfinite(coefficients.at(input))) {
      return failure(ErrorCode::NonFiniteInput,
                     "weighted sum coefficient is non-finite");
    }
    allZero = allZero && coefficients.at(input) == 0.0;
  }

  MomentRequest internalRequest = request;
  const bool selectedFourth =
      request.representation == MomentRepresentation::selected &&
      (request.orders & 0x08U) != 0U;
  if (!selectedFourth &&
      request.representation != MomentRepresentation::selected &&
      (request.orders & 0x08U) != 0U) {
    internalRequest.orders = static_cast<std::uint8_t>(request.orders | 0x02U);
  }
  std::vector<MomentEstimate> cumulantInputs;
  cumulantInputs.reserve(inputs.size());
  const auto conversionOrders = static_cast<std::uint8_t>(
      internalRequest.orders | (selectedFourth ? 0x02U : 0U));
  for (Index inputIndex = 0U; inputIndex < inputs.size(); ++inputIndex) {
    if (coefficients.at(inputIndex) == 0.0 || allZero) {
      MomentEstimate empty{.observation_shape =
                               inputs[inputIndex].observation_shape,
                           .terms = {},
                           .diagnostics = inputs[inputIndex].diagnostics,
                           .population_mean = std::nullopt};
      cumulantInputs.push_back(std::move(empty));
      continue;
    }

    std::optional<MomentEstimate> converted;
    for (Index order = 1U; order <= 4U; ++order) {
      if ((conversionOrders & (1U << (order - 1U))) == 0U) {
        continue;
      }
      SelectedTupleList selectedStorage;
      auto orderResult = convertCentralOrder(inputs[inputIndex], request, order,
                                             inputDimension, contracted, limits,
                                             &selectedStorage);
      if (!orderResult) {
        return Result<MomentEstimate>::failure(
            orderResult.status().code(),
            std::string(orderResult.status().message()));
      }
      if (!converted.has_value()) {
        converted.emplace(std::move(orderResult).value());
      } else {
        auto merged =
            mergeMomentEstimates(*converted, orderResult.value(), limits);
        if (!merged) {
          return merged;
        }
        converted.emplace(std::move(merged).value());
      }
    }
    if (converted.has_value()) {
      cumulantInputs.push_back(std::move(converted).value());
    } else {
      return failure(ErrorCode::InvalidDomain,
                     "weighted sum has no requested cumulant orders");
    }
  }

  std::optional<MomentEstimate> summedCumulants;
  if (selectedFourth) {
    MomentRequest pairRequest = request;
    pairRequest.orders = 0x02U;
    const SelectedTupleList pairTuples =
        fourthOrderPairTuples(request.selected);
    std::vector<std::span<const Index>> pairViews;
    pairViews.reserve(pairTuples.size());
    for (const auto& tuple : pairTuples) {
      pairViews.emplace_back(tuple);
    }
    pairRequest.selected = pairViews;
    auto secondOrder = independentCumulantSum(cumulantInputs, coefficients,
                                              pairRequest, limits);
    if (!secondOrder) {
      return secondOrder;
    }
    auto fourthOrder =
        independentCumulantSum(cumulantInputs, coefficients, request, limits);
    if (!fourthOrder) {
      return fourthOrder;
    }
    auto combined =
        mergeMomentEstimates(secondOrder.value(), fourthOrder.value(), limits);
    if (!combined) {
      return combined;
    }
    summedCumulants.emplace(std::move(combined).value());
  } else {
    auto sum = independentCumulantSum(cumulantInputs, coefficients,
                                      internalRequest, limits);
    if (!sum) {
      return sum;
    }
    summedCumulants.emplace(std::move(sum).value());
  }

  MomentRequest outputRequest = request;
  if (contracted) {
    outputRequest.representation = MomentRepresentation::dense;
    outputRequest.selected = {};
    outputRequest.contraction = std::nullopt;
  }
  return cumulantsToCentralMoments(*summedCumulants, outputRequest, limits);
}

} // namespace vss::math::statistics
