#ifndef VSS_MATH_STATISTICS_MOMENT_INTERVAL_HPP
#define VSS_MATH_STATISTICS_MOMENT_INTERVAL_HPP

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>

namespace vss::math::statistics::detail {

// Normalized binary significands keep exponent range independent of Scalar.
// The estimator's bounded operation order keeps exponents within int range.
class BinaryEndpoint final {
public:
  BinaryEndpoint() = default;
  BinaryEndpoint(double fraction, int exponent)
      : fraction_(fraction), exponent_(exponent) {}

  static BinaryEndpoint normalized(double fraction, int exponent = 0) {
    if (fraction == 0.0) {
      return {};
    }
    int adjustment = 0;
    const double normalized = std::frexp(fraction, &adjustment);
    return {normalized, exponent + adjustment};
  }

  [[nodiscard]] double rounded() const {
    return std::scalbn(fraction_, exponent_);
  }

  [[nodiscard]] double fraction() const { return fraction_; }
  [[nodiscard]] int exponent() const { return exponent_; }

  [[nodiscard]] BinaryEndpoint negated() const {
    return {-fraction_, exponent_};
  }

  friend bool operator<(BinaryEndpoint a, BinaryEndpoint b) {
    if (a.fraction_ <= 0.0 && b.fraction_ >= 0.0) {
      return a.fraction_ < b.fraction_;
    }
    if (a.fraction_ >= 0.0 && b.fraction_ <= 0.0) {
      return false;
    }
    if (a.exponent_ == b.exponent_) {
      return a.fraction_ < b.fraction_;
    }
    return a.fraction_ > 0.0 ? a.exponent_ < b.exponent_
                             : a.exponent_ > b.exponent_;
  }

private:
  double fraction_{};
  int exponent_{};
};

enum class Bound : std::uint8_t { lower, upper };

inline BinaryEndpoint enclose(double fraction, int exponent, Bound direction) {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  return BinaryEndpoint::normalized(
      std::nextafter(fraction,
                     direction == Bound::lower ? -infinity : infinity),
      exponent);
}

inline BinaryEndpoint sumBound(BinaryEndpoint a, BinaryEndpoint b,
                               Bound direction) {
  if (a.fraction() == 0.0) {
    return b;
  }
  if (b.fraction() == 0.0) {
    return a;
  }
  if (a.exponent() < b.exponent()) {
    std::swap(a, b);
  }
  const int shift = a.exponent() - b.exponent();
  if (shift > std::numeric_limits<double>::digits + 2) {
    // The smaller value is less than a step even at a binade boundary.
    const bool widens =
        direction == Bound::lower ? b.fraction() < 0.0 : b.fraction() > 0.0;
    return widens ? enclose(a.fraction(), a.exponent(), direction) : a;
  }
  const double aligned = std::scalbn(b.fraction(), -shift);
  const double sum = a.fraction() + aligned;
  // Alignment is exact in this range. A rounded zero can only be exact.
  return sum == 0.0 ? BinaryEndpoint{} : enclose(sum, a.exponent(), direction);
}

inline BinaryEndpoint productBound(BinaryEndpoint a, BinaryEndpoint b,
                                   Bound direction) {
  if (a.fraction() == 0.0 || b.fraction() == 0.0) {
    return {};
  }
  return enclose(a.fraction() * b.fraction(), a.exponent() + b.exponent(),
                 direction);
}

// Precondition: denominator is strictly positive.
inline BinaryEndpoint quotientBound(BinaryEndpoint numerator,
                                    BinaryEndpoint denominator,
                                    Bound direction) {
  if (numerator.fraction() == 0.0) {
    return {};
  }
  return enclose(numerator.fraction() / denominator.fraction(),
                 numerator.exponent() - denominator.exponent(), direction);
}

class MomentInterval final {
public:
  MomentInterval() = default;
  MomentInterval(BinaryEndpoint lower, BinaryEndpoint upper)
      : lower_(lower), upper_(upper) {}

  [[nodiscard]] BinaryEndpoint lower() const { return lower_; }
  [[nodiscard]] BinaryEndpoint upper() const { return upper_; }

  static MomentInterval point(double value) {
    const auto endpoint = BinaryEndpoint::normalized(value);
    return {endpoint, endpoint};
  }

  static bool supportsCurrentRounding() {
    static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::radix == 2);
    return std::fegetround() == FE_TONEAREST;
  }

  friend MomentInterval operator+(MomentInterval a, MomentInterval b) {
    return {sumBound(a.lower_, b.lower_, Bound::lower),
            sumBound(a.upper_, b.upper_, Bound::upper)};
  }

  friend MomentInterval operator-(MomentInterval a, MomentInterval b) {
    return a + MomentInterval{b.upper_.negated(), b.lower_.negated()};
  }

  friend MomentInterval operator*(MomentInterval a, MomentInterval b) {
    const std::array lowerBounds{
        productBound(a.lower_, b.lower_, Bound::lower),
        productBound(a.lower_, b.upper_, Bound::lower),
        productBound(a.upper_, b.lower_, Bound::lower),
        productBound(a.upper_, b.upper_, Bound::lower)};
    const std::array upperBounds{
        productBound(a.lower_, b.lower_, Bound::upper),
        productBound(a.lower_, b.upper_, Bound::upper),
        productBound(a.upper_, b.lower_, Bound::upper),
        productBound(a.upper_, b.upper_, Bound::upper)};
    return {*std::ranges::min_element(lowerBounds,
                                      [](auto x, auto y) { return x < y; }),
            *std::ranges::max_element(upperBounds,
                                      [](auto x, auto y) { return x < y; })};
  }

  // Precondition: zero < denominator.lower().
  [[nodiscard]] MomentInterval
  divideByPositive(MomentInterval denominator) const {
    const BinaryEndpoint zero{};
    return {quotientBound(
                lower_, lower_ < zero ? denominator.lower_ : denominator.upper_,
                Bound::lower),
            quotientBound(
                upper_, upper_ < zero ? denominator.upper_ : denominator.lower_,
                Bound::upper)};
  }

private:
  BinaryEndpoint lower_{};
  BinaryEndpoint upper_{};
};

enum class MomentCertificate : std::uint8_t {
  accepted,
  illConditioned,
  overflow
};

inline MomentCertificate certifyMoment(MomentInterval range, double& value,
                                       double& absoluteError) {
  const double lower = range.lower().rounded();
  const double upper = range.upper().rounded();
  if (std::isinf(lower) && lower == upper) {
    return MomentCertificate::overflow;
  }
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    return MomentCertificate::illConditioned;
  }
  const double candidate = std::midpoint(lower, upper);
  const auto point = BinaryEndpoint::normalized(candidate);
  const auto left = sumBound(point, range.lower().negated(), Bound::upper);
  const auto right = sumBound(range.upper(), point.negated(), Bound::upper);
  const auto error = std::max(left, right);
  const BinaryEndpoint zero{};
  BinaryEndpoint minMagnitude = zero;
  if (zero < range.lower()) {
    minMagnitude = range.lower();
  } else if (range.upper() < zero) {
    minMagnitude = range.upper().negated();
  }
  const auto relative = productBound(
      minMagnitude, BinaryEndpoint::normalized(std::nextafter(1.0e-10, 0.0)),
      Bound::lower);
  const auto allowed =
      sumBound(BinaryEndpoint::normalized(std::nextafter(1.0e-12, 0.0)),
               relative, Bound::lower);
  if (allowed < error) {
    return MomentCertificate::illConditioned;
  }
  value = candidate;
  const double roundedError = error.rounded();
  absoluteError = BinaryEndpoint::normalized(roundedError) < error
                      ? std::nextafter(roundedError,
                                       std::numeric_limits<double>::infinity())
                      : roundedError;
  return MomentCertificate::accepted;
}

} // namespace vss::math::statistics::detail

#endif
