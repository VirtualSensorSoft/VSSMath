#ifndef VSS_MATH_STATISTICS_MOMENT_INTERVAL_HPP
#define VSS_MATH_STATISTICS_MOMENT_INTERVAL_HPP

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <limits>
#include <numeric>
#include <ranges>

namespace vss::math::statistics::detail {

// Normalized binary significands keep exponent range independent of Scalar.
// The estimator's bounded operation order keeps exponents within int range.
struct BinaryEndpoint final {
  double fraction{};
  int exponent{};

  static BinaryEndpoint normalized(double fraction, int exponent = 0) {
    if (fraction == 0.0) {
      return {};
    }
    int adjustment = 0;
    const double normalized = std::frexp(fraction, &adjustment);
    return {normalized, exponent + adjustment};
  }

  [[nodiscard]] double rounded() const {
    return std::scalbn(fraction, exponent);
  }

  [[nodiscard]] BinaryEndpoint negated() const { return {-fraction, exponent}; }

  friend bool operator<(BinaryEndpoint a, BinaryEndpoint b) {
    if (a.fraction <= 0.0 && b.fraction >= 0.0) {
      return a.fraction < b.fraction;
    }
    if (a.fraction >= 0.0 && b.fraction <= 0.0) {
      return false;
    }
    if (a.exponent == b.exponent) {
      return a.fraction < b.fraction;
    }
    return a.fraction > 0.0 ? a.exponent < b.exponent : a.exponent > b.exponent;
  }
};

enum class Bound { lower, upper };

inline BinaryEndpoint enclose(double fraction, int exponent, Bound direction) {
  constexpr double infinity = std::numeric_limits<double>::infinity();
  return BinaryEndpoint::normalized(
      std::nextafter(fraction,
                     direction == Bound::lower ? -infinity : infinity),
      exponent);
}

inline BinaryEndpoint sumBound(BinaryEndpoint a, BinaryEndpoint b,
                               Bound direction) {
  if (a.fraction == 0.0) {
    return b;
  }
  if (b.fraction == 0.0) {
    return a;
  }
  if (a.exponent < b.exponent) {
    std::swap(a, b);
  }
  const int shift = a.exponent - b.exponent;
  if (shift > std::numeric_limits<double>::digits + 2) {
    // The smaller value is less than a step even at a binade boundary.
    const bool widens =
        direction == Bound::lower ? b.fraction < 0.0 : b.fraction > 0.0;
    return widens ? enclose(a.fraction, a.exponent, direction) : a;
  }
  const double aligned = std::scalbn(b.fraction, -shift);
  const double sum = a.fraction + aligned;
  // Alignment is exact in this range. A rounded zero can only be exact.
  return sum == 0.0 ? BinaryEndpoint{} : enclose(sum, a.exponent, direction);
}

inline BinaryEndpoint productBound(BinaryEndpoint a, BinaryEndpoint b,
                                   Bound direction) {
  if (a.fraction == 0.0 || b.fraction == 0.0) {
    return {};
  }
  return enclose(a.fraction * b.fraction, a.exponent + b.exponent, direction);
}

// Precondition: denominator is strictly positive.
inline BinaryEndpoint quotientBound(BinaryEndpoint numerator,
                                    BinaryEndpoint denominator,
                                    Bound direction) {
  if (numerator.fraction == 0.0) {
    return {};
  }
  return enclose(numerator.fraction / denominator.fraction,
                 numerator.exponent - denominator.exponent, direction);
}

struct MomentInterval final {
  BinaryEndpoint lower{};
  BinaryEndpoint upper{};

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
    return {sumBound(a.lower, b.lower, Bound::lower),
            sumBound(a.upper, b.upper, Bound::upper)};
  }

  friend MomentInterval operator-(MomentInterval a, MomentInterval b) {
    return a + MomentInterval{b.upper.negated(), b.lower.negated()};
  }

  friend MomentInterval operator*(MomentInterval a, MomentInterval b) {
    const std::array lowerBounds{productBound(a.lower, b.lower, Bound::lower),
                                 productBound(a.lower, b.upper, Bound::lower),
                                 productBound(a.upper, b.lower, Bound::lower),
                                 productBound(a.upper, b.upper, Bound::lower)};
    const std::array upperBounds{productBound(a.lower, b.lower, Bound::upper),
                                 productBound(a.lower, b.upper, Bound::upper),
                                 productBound(a.upper, b.lower, Bound::upper),
                                 productBound(a.upper, b.upper, Bound::upper)};
    return {*std::ranges::min_element(lowerBounds,
                                      [](auto x, auto y) { return x < y; }),
            *std::ranges::max_element(upperBounds,
                                      [](auto x, auto y) { return x < y; })};
  }

  // Precondition: zero < denominator.lower.
  [[nodiscard]] MomentInterval
  divideByPositive(MomentInterval denominator) const {
    const BinaryEndpoint zero{};
    return {quotientBound(lower,
                          lower < zero ? denominator.lower : denominator.upper,
                          Bound::lower),
            quotientBound(upper,
                          upper < zero ? denominator.upper : denominator.lower,
                          Bound::upper)};
  }
};

enum class MomentCertificate { accepted, illConditioned, overflow };

inline MomentCertificate certifyMoment(MomentInterval range, double& value,
                                       double& absoluteError) {
  const double lower = range.lower.rounded();
  const double upper = range.upper.rounded();
  if (std::isinf(lower) && lower == upper) {
    return MomentCertificate::overflow;
  }
  if (!std::isfinite(lower) || !std::isfinite(upper)) {
    return MomentCertificate::illConditioned;
  }
  const double candidate = std::midpoint(lower, upper);
  const auto point = BinaryEndpoint::normalized(candidate);
  const auto left = sumBound(point, range.lower.negated(), Bound::upper);
  const auto right = sumBound(range.upper, point.negated(), Bound::upper);
  const auto error = std::max(left, right);
  const BinaryEndpoint zero{};
  const auto minMagnitude =
      zero < range.lower ? range.lower
                         : (range.upper < zero ? range.upper.negated() : zero);
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
