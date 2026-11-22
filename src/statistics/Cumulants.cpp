#include "MomentInterval.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
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
            static_cast<std::uint64_t>(a.at(i)) * b.at(j) + result.at(i + j) + carry;
        result.at(i + j) = static_cast<std::uint32_t>(current);
        carry = current >> 32U;
      }
      Index position = i + 2U;
      while (carry != 0U && position < result.size()) {
        const auto current = static_cast<std::uint64_t>(result.at(position)) + carry;
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
        return (word - 1U) * 32U +
               (31U - static_cast<Index>(std::countl_zero(value.at(word - 1U))));
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
            if (!exactFourth(sourceTerm->value, leftValues.at(0U), leftValues.at(1U),
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

} // namespace vss::math::statistics
