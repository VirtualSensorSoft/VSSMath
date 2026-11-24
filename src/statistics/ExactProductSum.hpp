#pragma once

#include "MomentInterval.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vss/math/Core.hpp>

namespace vss::math::statistics::detail {

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

  [[nodiscard]] BinaryEndpoint endpoint() const {
    const bool negative = less(positive_, negative_);
    const Big& larger = negative ? negative_ : positive_;
    const Big& smaller = negative ? positive_ : negative_;
    const Big magnitude = subtract(larger, smaller);
    const Index top = highestBit(magnitude);
    if (top == noBit) {
      return {};
    }
    const Index shift = top > 52U ? top - 52U : 0U;
    std::uint64_t significand = shiftedLow(magnitude, shift);
    if (shift != 0U && testBit(magnitude, shift - 1U) &&
        (anyBelow(magnitude, shift - 1U) || (significand & 1U) != 0U)) {
      ++significand;
    }
    const double signedSignificand = negative
                                         ? -static_cast<double>(significand)
                                         : static_cast<double>(significand);
    return BinaryEndpoint::normalized(
        signedSignificand, static_cast<int>(shift) + minimumExponent);
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

} // namespace vss::math::statistics::detail
