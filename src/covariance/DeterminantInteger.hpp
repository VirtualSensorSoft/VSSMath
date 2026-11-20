#ifndef VSS_MATH_DETERMINANT_INTEGER_HPP
#define VSS_MATH_DETERMINANT_INTEGER_HPP

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>

namespace vss::math::covariance::detail {

// Private signed base-2^32 integers for fraction-free elimination. Each
// operation allocates at most its result's limb count. The determinant caller
// checks a bound for the entire elimination before constructing any integers.
class DeterminantInteger final {
public:
  using Limb = std::uint32_t;
  static constexpr Index limbBits = std::numeric_limits<Limb>::digits;

  DeterminantInteger() = default;

  DeterminantInteger(std::uint64_t magnitude, Index shift, bool negative)
      : negative_(negative && magnitude != 0) {
    if (magnitude != 0) {
      const Index size = static_cast<Index>(std::bit_width(magnitude)) + shift;
      limbs_.resize((size - 1) / limbBits + 1);
      for (Index bit = 0; magnitude != 0; ++bit, magnitude >>= 1) {
        if ((magnitude & 1U) != 0) {
          set(bit + shift);
        }
      }
    }
  }

  [[nodiscard]] bool zero() const noexcept { return limbs_.empty(); }
  [[nodiscard]] bool negative() const noexcept { return negative_; }
  [[nodiscard]] Index bits() const noexcept {
    return zero() ? 0
                  : (limbs_.size() - 1) * limbBits +
                        static_cast<Index>(std::bit_width(limbs_.back()));
  }
  [[nodiscard]] bool bit(Index position) const noexcept {
    return (limb(position / limbBits) >> (position % limbBits) & 1U) != 0;
  }
  [[nodiscard]] bool anyBelow(Index end) const noexcept {
    const Index whole = end / limbBits;
    for (Index index = 0; index < whole && index < limbs_.size(); ++index) {
      if (limbs_[index] != 0) {
        return true;
      }
    }
    const Limb mask = (Limb{1} << (end % limbBits)) - 1U;
    return (limb(whole) & mask) != 0;
  }
  [[nodiscard]] std::uint64_t leading(Index begin) const noexcept {
    std::uint64_t result = 0;
    for (Index index = bits(); index > begin; --index) {
      result = (result << 1U) | static_cast<std::uint64_t>(bit(index - 1));
    }
    return result;
  }
  void negate() noexcept { negative_ = !zero() && !negative_; }

  [[nodiscard]] static DeterminantInteger
  multiply(const DeterminantInteger& left, const DeterminantInteger& right) {
    if (left.zero() || right.zero()) {
      return {};
    }
    DeterminantInteger result;
    result.limbs_.resize(left.limbs_.size() + right.limbs_.size());
    for (Index i = 0; i < left.limbs_.size(); ++i) {
      std::uint64_t carry = 0;
      for (Index j = 0; j < right.limbs_.size(); ++j) {
        const auto product =
            static_cast<std::uint64_t>(left.limbs_[i]) * right.limbs_[j] +
            result.limbs_[i + j] + carry;
        result.limbs_[i + j] = static_cast<Limb>(product);
        carry = product >> limbBits;
      }
      result.limbs_[i + right.limbs_.size()] = static_cast<Limb>(carry);
    }
    result.negative_ = left.negative_ != right.negative_;
    result.normalize();
    return result;
  }

  [[nodiscard]] static DeterminantInteger subtract(DeterminantInteger left,
                                                   DeterminantInteger right) {
    right.negate();
    if (left.zero()) {
      return right;
    }
    if (right.zero()) {
      return left;
    }
    if (left.negative_ != right.negative_) {
      if (left.compareShifted(right, 0) < 0) {
        std::swap(left, right);
      }
      left.subtractShifted(right, 0);
      return left;
    }
    const Index width = std::max(left.limbs_.size(), right.limbs_.size());
    DeterminantInteger result;
    result.limbs_.resize(width + 1);
    result.negative_ = left.negative_;
    std::uint64_t carry = 0;
    for (Index index = 0; index < width; ++index) {
      const auto sum = static_cast<std::uint64_t>(left.limb(index)) +
                       right.limb(index) + carry;
      result.limbs_[index] = static_cast<Limb>(sum);
      carry = sum >> limbBits;
    }
    result.limbs_[width] = static_cast<Limb>(carry);
    result.normalize();
    return result;
  }

  // Return false if the fraction-free elimination invariant is violated.
  bool divideExactly(const DeterminantInteger& divisor) {
    if (divisor.zero()) {
      return false;
    }
    if (zero()) {
      return true;
    }
    if (bits() < divisor.bits()) {
      return false;
    }
    DeterminantInteger quotient;
    const Index maximumShift = bits() - divisor.bits();
    quotient.limbs_.resize(maximumShift / limbBits + 1);
    quotient.negative_ = negative_ != divisor.negative_;
    for (Index position = maximumShift + 1; position > 0; --position) {
      const Index shift = position - 1;
      if (compareShifted(divisor, shift) >= 0) {
        subtractShifted(divisor, shift);
        quotient.set(shift);
      }
    }
    if (!zero()) {
      return false;
    }
    quotient.normalize();
    *this = std::move(quotient);
    return true;
  }

private:
  [[nodiscard]] Limb limb(Index index) const noexcept {
    return index < limbs_.size() ? limbs_[index] : Limb{0};
  }
  [[nodiscard]] Limb shiftedLimb(Index index, Index shift) const noexcept {
    const Index whole = shift / limbBits;
    const Index part = shift % limbBits;
    if (index < whole) {
      return 0;
    }
    const Index source = index - whole;
    Limb result = limb(source) << part;
    if (source > 0 && part != 0) {
      result |= limb(source - 1) >> (limbBits - part);
    }
    return result;
  }
  [[nodiscard]] int compareShifted(const DeterminantInteger& right,
                                   Index shift) const noexcept {
    const Index rightBits = right.zero() ? 0 : right.bits() + shift;
    if (bits() != rightBits) {
      return bits() < rightBits ? -1 : 1;
    }
    for (Index end = limbs_.size(); end > 0; --end) {
      const Index index = end - 1;
      const Limb rhs = right.shiftedLimb(index, shift);
      if (limbs_[index] != rhs) {
        return limbs_[index] < rhs ? -1 : 1;
      }
    }
    return 0;
  }
  void subtractShifted(const DeterminantInteger& right, Index shift) noexcept {
    std::uint64_t borrow = 0;
    for (Index index = shift / limbBits; index < limbs_.size(); ++index) {
      const std::uint64_t lhs = limbs_[index];
      const auto rhs =
          static_cast<std::uint64_t>(right.shiftedLimb(index, shift)) + borrow;
      limbs_[index] = static_cast<Limb>(lhs - rhs);
      borrow = static_cast<std::uint64_t>(lhs < rhs);
    }
    normalize();
  }
  void normalize() noexcept {
    while (!limbs_.empty() && limbs_.back() == 0) {
      limbs_.pop_back();
    }
    if (zero()) {
      negative_ = false;
    }
  }
  void set(Index position) noexcept {
    limbs_[position / limbBits] |= Limb{1} << (position % limbBits);
  }

  std::vector<Limb> limbs_;
  bool negative_{};
};

} // namespace vss::math::covariance::detail

#endif
