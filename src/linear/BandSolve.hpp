#ifndef VSS_MATH_BAND_SOLVE_HPP
#define VSS_MATH_BAND_SOLVE_HPP

#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/StructuredProducts.hpp>

namespace vss::math::linear::detail {
namespace band_detail {

inline Wide up(Wide value) {
  return value == 0
             ? 0
             : std::nextafter(value, std::numeric_limits<Wide>::infinity());
}

inline Wide roundingError(Wide result, Wide a, Wide b) {
  if (a == 0 || b == 0) {
    return 0; // fma adds an exact zero to its third operand
  }
  return std::max(
      std::nextafter(result, std::numeric_limits<Wide>::infinity()) - result,
      result - std::nextafter(result, -std::numeric_limits<Wide>::infinity()));
}

} // namespace band_detail

// Absence means that the compact path did not establish a full-rank solve.
// Release all compact allocations before attempting the independent fallback.
inline std::optional<LinearEstimate> tryCompactBand(const BandMatrix& band,
                                                    ConstMatrixView rhs,
                                                    SolvePolicy policy,
                                                    Limits limits) {
  using band_detail::roundingError;
  using band_detail::up;
  const Index n = band.order;
  const Index upper = band.lower + band.upper;
  if (upper >= std::numeric_limits<Index>::max() - band.lower) {
    return std::nullopt;
  }
  const Index width = upper + band.lower + 1;
  StorageBudget budget(limits);
  if (!budget.claim<Wide>(n, width) || !budget.claim<Wide>(n, band.lower) ||
      !budget.claim<Index>(n) || !budget.claim<Wide>(n) ||
      !budget.claim<Scalar>(n, rhs.columns()) ||
      !budget.claim<Scalar>(rhs.columns())) {
    return std::nullopt;
  }

  Scalar maximum = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < band.bands.columns(); ++j) {
      maximum = std::max(maximum, std::abs(band.bands.at(i, j)));
    }
  }
  if (maximum == 0) {
    return std::nullopt;
  }
  const int exponent = std::ilogb(maximum);
  const auto scaled = [&](Scalar value) {
    return std::scalbn(static_cast<Wide>(value), -exponent);
  };
  std::vector<Wide> factor(n * width);
  std::vector<Wide> errors(n * band.lower);
  std::vector<Index> pivots(n);
  std::vector<Wide> work(n);
  const auto entry = [&](Index i, Index j) -> Wide& {
    const Index slot = j < i ? band.lower - (i - j) : band.lower + (j - i);
    return factor[i * width + slot];
  };
  const auto first = [&](Index i) { return i - std::min(i, band.lower); };
  const auto last = [&](Index i, Index distance) {
    return i + std::min(n - 1 - i, distance);
  };
  const auto original = [&](Index i, Index j) {
    const Index slot = j < i ? band.lower - (i - j) : band.lower + (j - i);
    return scaled(band.bands.at(i, slot));
  };
  Wide norm_squared = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = first(i); j <= last(i, band.upper); ++j) {
      const Wide value = original(i, j);
      entry(i, j) = value;
      norm_squared = up(norm_squared + up(value * value));
    }
  }
  const Wide upper_bound = up(std::sqrt(norm_squared));
  for (Index k = 0; k < n; ++k) {
    const Index bottom = last(k, band.lower);
    const Index right = last(k, upper);
    Index pivot = k;
    for (Index i = k + 1; i <= bottom; ++i) {
      if (std::abs(entry(i, k)) > std::abs(entry(pivot, k))) {
        pivot = i;
      }
    }
    if (entry(pivot, k) == 0) {
      return std::nullopt;
    }
    pivots[k] = pivot;
    if (pivot != k) {
      for (Index j = k; j <= right; ++j) {
        std::swap(entry(k, j), entry(pivot, j));
      }
    }
    for (Index i = k + 1; i <= bottom; ++i) {
      const Wide previous = entry(i, k);
      const Wide multiplier = previous / entry(k, k);
      const Wide eliminated = std::fma(-multiplier, entry(k, k), previous);
      Wide error = up(std::abs(eliminated) +
                      roundingError(eliminated, multiplier, entry(k, k)));
      entry(i, k) = multiplier;
      for (Index j = k + 1; j <= right; ++j) {
        auto& target = entry(i, j);
        target = std::fma(-multiplier, entry(k, j), target);
        error = up(error + roundingError(target, multiplier, entry(k, j)));
      }
      errors[k * band.lower + (i - k - 1)] = error;
    }
  }
  for (Index k = n; k-- > 0;) {
    for (Index i = k + 1; i <= last(k, band.lower); ++i) {
      work[i] = up(up(work[i] + errors[k * band.lower + (i - k - 1)]) +
                   up(std::abs(entry(i, k)) * work[k]));
    }
    std::swap(work[k], work[pivots[k]]);
  }
  const Wide root_n = up(std::sqrt(static_cast<Wide>(n)));
  const Wide factor_error = up(root_n * *std::ranges::max_element(work));
  std::ranges::fill(work, 1);
  for (Index k = 0; k < n; ++k) {
    std::swap(work[k], work[pivots[k]]);
    for (Index i = k + 1; i <= last(k, band.lower); ++i) {
      work[i] = up(work[i] + up(std::abs(entry(i, k)) * work[k]));
    }
  }
  for (Index k = n; k-- > 0;) {
    for (Index j = k + 1; j <= last(k, upper); ++j) {
      work[k] = up(work[k] + up(std::abs(entry(k, j)) * work[j]));
    }
    work[k] = up(work[k] / std::abs(entry(k, k)));
  }
  const Wide inverse_bound = up(root_n * *std::ranges::max_element(work));
  const Wide lower_bound =
      std::nextafter(std::nextafter(1 / inverse_bound, Wide{0}) - factor_error,
                     -std::numeric_limits<Wide>::infinity());
  const Wide cutoff = scaled(policy.rank_abs);
  if (!(lower_bound > cutoff) || up(policy.rank_rel * upper_bound) > cutoff ||
      up(policy.reciprocal_condition_floor * upper_bound) > lower_bound) {
    return std::nullopt;
  }

  LinearEstimate result{.parameters = Matrix(n, rhs.columns()),
                        .covariance = std::nullopt,
                        .identified_basis = std::nullopt,
                        .residual_norms = Vector(rhs.columns()),
                        .rank = n,
                        .rank_cutoff = policy.rank_abs};
  for (Index col = 0; col < rhs.columns(); ++col) {
    for (Index i = 0; i < n; ++i) {
      work[i] = scaled(rhs.at(i, col));
    }
    for (Index k = 0; k < n; ++k) {
      std::swap(work[k], work[pivots[k]]);
      for (Index i = k + 1; i <= last(k, band.lower); ++i) {
        work[i] = std::fma(-entry(i, k), work[k], work[i]);
      }
    }
    for (Index k = n; k-- > 0;) {
      for (Index j = k + 1; j <= last(k, upper); ++j) {
        work[k] = std::fma(-entry(k, j), work[j], work[k]);
      }
      work[k] /= entry(k, k);
      if (!representable(work[k])) {
        return std::nullopt;
      }
      result.parameters.at(k, col) = static_cast<Scalar>(work[k]);
    }
    Wide norm = 0;
    Wide residual_bound = 0;
    Wide solution_norm = 0;
    for (Index i = 0; i < n; ++i) {
      Wide residual = -scaled(rhs.at(i, col));
      Wide error = 0;
      for (Index j = first(i); j <= last(i, band.upper); ++j) {
        const Wide a = original(i, j);
        const Wide x = result.parameters.at(j, col);
        residual = std::fma(a, x, residual);
        error = up(error + roundingError(residual, a, x));
      }
      norm = std::hypot(norm, residual);
      residual_bound =
          up(std::hypot(residual_bound, up(std::abs(residual) + error)));
      solution_norm = std::hypot(solution_norm, result.parameters.at(i, col));
    }
    const Wide envelope = 1.0e-12L + 1.0e-10L * solution_norm;
    const Wide original_norm = std::scalbn(norm, exponent);
    if (up(residual_bound / lower_bound) > envelope ||
        !representable(original_norm)) {
      return std::nullopt;
    }
    result.residual_norms.at(col) = static_cast<Scalar>(original_norm);
  }
  return result;
}

} // namespace vss::math::linear::detail
#endif
