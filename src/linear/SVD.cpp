#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {
namespace {

using detail::Wide;

class WorkMatrix final {
public:
  WorkMatrix(Index rows, Index cols)
      : rows_(rows), cols_(cols), data_(rows * cols) {}

  [[nodiscard]] Index rows() const { return rows_; }
  [[nodiscard]] Index columns() const { return cols_; }
  [[nodiscard]] Wide& at(Index row, Index col) {
    return data_.at((row * cols_) + col);
  }
  [[nodiscard]] Wide at(Index row, Index col) const {
    return data_.at((row * cols_) + col);
  }
  void identity() {
    for (Index i = 0; i < std::min(rows_, cols_); ++i) {
      at(i, i) = 1;
    }
  }

private:
  Index rows_;
  Index cols_;
  std::vector<Wide> data_;
};

struct Bidiagonal final {
  std::vector<Wide> diagonal;
  std::vector<Wide> super;
};

struct Rotation final {
  Wide cosine;
  Wide sine;
  Wide norm;
};

Rotation givens(Wide first, Wide second) {
  const Wide norm = std::hypot(first, second);
  if (norm == 0) {
    return {.cosine = 1, .sine = 0, .norm = 0};
  }
  return {.cosine = first / norm, .sine = second / norm, .norm = norm};
}

void rotateColumns(WorkMatrix& factor, Index i, Index j, Rotation rotation) {
  for (Index row = 0; row < factor.rows(); ++row) {
    const Wide first = factor.at(row, i);
    const Wide second = factor.at(row, j);
    factor.at(row, i) = (rotation.cosine * first) + (rotation.sine * second);
    factor.at(row, j) = (rotation.cosine * second) - (rotation.sine * first);
  }
}

Wide makeReflector(std::span<Wide> direction) {
  Wide norm = 0;
  for (const Wide value : direction) {
    norm = std::hypot(norm, value);
  }
  if (norm == 0) {
    return 0;
  }
  const Wide target = -std::copysign(norm, direction.front());
  const Wide leading = direction.front() - target;
  direction.front() = leading;
  const Wide reflectorNorm = norm * std::sqrt(2 * (std::abs(leading) / norm));
  for (auto& entry : direction) {
    entry /= reflectorNorm;
  }
  return target;
}

void reflectLeft(WorkMatrix& matrix, Index fromRow, Index fromCol,
                 std::span<const Wide> direction) {
  for (Index col = fromCol; col < matrix.columns(); ++col) {
    Wide dot = 0;
    for (Index row = fromRow; row < matrix.rows(); ++row) {
      dot = std::fma(direction[row - fromRow], matrix.at(row, col), dot);
    }
    for (Index row = fromRow; row < matrix.rows(); ++row) {
      matrix.at(row, col) =
          std::fma(-2 * dot, direction[row - fromRow], matrix.at(row, col));
    }
  }
}

void reflectRight(WorkMatrix& matrix, Index fromRow, Index fromCol,
                  std::span<const Wide> direction) {
  for (Index row = fromRow; row < matrix.rows(); ++row) {
    Wide dot = 0;
    for (Index col = fromCol; col < matrix.columns(); ++col) {
      dot = std::fma(direction[col - fromCol], matrix.at(row, col), dot);
    }
    for (Index col = fromCol; col < matrix.columns(); ++col) {
      matrix.at(row, col) =
          std::fma(-2 * dot, direction[col - fromCol], matrix.at(row, col));
    }
  }
}

void reduceToBidiagonal(WorkMatrix& work, WorkMatrix& left, WorkMatrix& right,
                        Bidiagonal& band) {
  std::vector<Wide> scratch(work.rows());
  const std::span<Wide> storage{scratch};
  for (Index j = 0; j < work.columns(); ++j) {
    const auto column = storage.first(work.rows() - j);
    for (Index row = j; row < work.rows(); ++row) {
      column[row - j] = work.at(row, j);
    }
    band.diagonal.at(j) = makeReflector(column);
    reflectLeft(work, j, j + 1, column);
    for (Index row = j; row < work.rows(); ++row) {
      work.at(row, j) = column[row - j];
    }

    if (j + 1 < work.columns()) {
      const auto tail = storage.first(work.columns() - j - 1);
      for (Index col = j + 1; col < work.columns(); ++col) {
        tail[col - j - 1] = work.at(j, col);
      }
      band.super.at(j) = makeReflector(tail);
      reflectRight(work, j + 1, j + 1, tail);
      reflectRight(right, 0, j + 1, tail);
    }
  }

  for (Index end = work.columns(); end > 0; --end) {
    const auto j = end - 1;
    const auto column = storage.first(work.rows() - j);
    for (Index row = j; row < work.rows(); ++row) {
      column[row - j] = work.at(row, j);
    }
    reflectLeft(left, j, j, column);
  }
}

// For an exactly zero diagonal, rotations can chase its coupling out of the
// active block without dividing by that diagonal. These rotations also update
// the singular-vector factors.
void splitZeroRow(Bidiagonal& band, WorkMatrix& left, Index zero, Index last) {
  Wide bulge = std::exchange(band.super.at(zero), 0);
  for (Index row = zero + 1; row <= last; ++row) {
    const auto rotation = givens(band.diagonal.at(row), -bulge);
    rotateColumns(left, zero, row, rotation);
    band.diagonal.at(row) = rotation.norm;
    if (row < last) {
      bulge = rotation.sine * band.super.at(row);
      band.super.at(row) *= rotation.cosine;
    }
  }
}

void splitZeroColumn(Bidiagonal& band, WorkMatrix& right, Index first,
                     Index last) {
  Wide bulge = std::exchange(band.super.at(last - 1), 0);
  for (Index end = last; end > first; --end) {
    const Index col = end - 1;
    const auto rotation = givens(band.diagonal.at(col), bulge);
    rotateColumns(right, col, last, rotation);
    band.diagonal.at(col) = rotation.norm;
    if (col > first) {
      bulge = -rotation.sine * band.super.at(col - 1);
      band.super.at(col - 1) *= rotation.cosine;
    }
  }
}

// Use the eigenvalue nearest the bottom diagonal of the trailing 2x2 block
// of B^T B. The matrix itself is never squared. Skip the shift when resolving
// a small singular direction needs a zero-shift step.
Wide trailingShift(const Bidiagonal& band, Index first, Index last) {
  const Wide x = band.diagonal.at(last - 1);
  const Wide y = band.diagonal.at(last);
  const Wide edge = band.super.at(last - 1);
  const Wide previous = last - 1 > first ? band.super.at(last - 2) : 0;
  const Wide a = (x * x) + (previous * previous);
  const Wide c = (y * y) + (edge * edge);
  const Wide b = x * edge;
  if (b == 0) {
    return c;
  }
  const Wide delta = (a - c) / 2;
  return std::max(
      Wide{0},
      c - (b * (b / (delta + std::copysign(std::hypot(delta, b), delta)))));
}

void chaseBulge(Bidiagonal& band, WorkMatrix& left, WorkMatrix& right,
                Index first, Index last) {
  const Wide leading = band.diagonal.at(first);
  const Wide edge = band.super.at(first);
  const Wide proposed = trailingShift(band, first, last);
  const Wide shift =
      proposed * std::numeric_limits<Wide>::epsilon() > leading * leading
          ? 0
          : proposed;
  Wide x = shift == 0 ? leading : (leading * leading) - shift;
  Wide y = shift == 0 ? edge : leading * edge;
  for (Index i = first; i < last; ++i) {
    const auto rightRotation = givens(x, y);
    rotateColumns(right, i, i + 1, rightRotation);
    if (i > first) {
      band.super.at(i - 1) = rightRotation.norm;
    }

    const Wide d = band.diagonal.at(i);
    const Wide e = band.super.at(i);
    const Wide next = band.diagonal.at(i + 1);
    const auto leftRotation =
        givens((rightRotation.cosine * d) + (rightRotation.sine * e),
               rightRotation.sine * next);
    rotateColumns(left, i, i + 1, leftRotation);

    const Wide upper = (rightRotation.cosine * e) - (rightRotation.sine * d);
    const Wide lower = rightRotation.cosine * next;
    band.diagonal.at(i) = leftRotation.norm;
    band.super.at(i) =
        (leftRotation.cosine * upper) + (leftRotation.sine * lower);
    band.diagonal.at(i + 1) =
        (leftRotation.cosine * lower) - (leftRotation.sine * upper);
    if (i + 1 < last) {
      x = band.super.at(i);
      y = leftRotation.sine * band.super.at(i + 1);
      band.super.at(i + 1) *= leftRotation.cosine;
    }
  }
}

bool diagonalize(Bidiagonal& band, WorkMatrix& left, WorkMatrix& right,
                 Index remaining) {
  constexpr Wide epsilon = std::numeric_limits<Wide>::epsilon();
  Index end = band.diagonal.size();
  while (end > 1) {
    const Index last = end - 1;
    Index first = last;
    while (first > 0) {
      const Wide threshold = epsilon * (std::abs(band.diagonal.at(first - 1)) +
                                        std::abs(band.diagonal.at(first)));
      if (std::abs(band.super.at(first - 1)) <= threshold) {
        band.super.at(first - 1) = 0;
        break;
      }
      --first;
    }
    if (first == last) {
      --end;
      continue;
    }
    if (remaining == 0) {
      return false;
    }
    --remaining;
    Index zero = first;
    while (zero < last && band.diagonal.at(zero) != 0) {
      ++zero;
    }
    if (zero < last) {
      splitZeroRow(band, left, zero, last);
    } else if (band.diagonal.at(last) == 0) {
      splitZeroColumn(band, right, first, last);
    } else {
      chaseBulge(band, left, right, first, last);
    }
  }
  return true;
}

bool claimStorage(detail::StorageBudget& budget, Index tall, Index thin) {
  return budget.claim<Wide>(tall, thin) && budget.claim<Wide>(tall, thin) &&
         budget.claim<Wide>(thin, thin) && budget.claim<Wide>(tall) &&
         budget.claim<Wide>(thin, 2) && budget.claim<Index>(tall) &&
         budget.claim<Index>(thin, 2) && budget.claim<Scalar>(tall, thin) &&
         budget.claim<Scalar>(thin, thin) && budget.claim<Scalar>(thin);
}

// Direct construction for signed permuted diagonals. The singular values
// are input entries and exact comparisons also work at a zero iteration cap.
bool directFactors(const WorkMatrix& work, WorkMatrix& left, WorkMatrix& right,
                   Bidiagonal& band) {
  const auto missing = work.rows();
  std::vector<Index> columnRows(work.columns(), missing);
  std::vector<Index> used(work.rows(), 0);
  for (Index col = 0; col < work.columns(); ++col) {
    for (Index row = 0; row < work.rows(); ++row) {
      if (work.at(row, col) == 0) {
        continue;
      }
      if (columnRows.at(col) != missing || used.at(row) != 0) {
        return false;
      }
      columnRows.at(col) = row;
      used.at(row) = 1;
    }
  }
  Index unused = 0;
  for (Index col = 0; col < work.columns(); ++col) {
    auto row = columnRows.at(col);
    if (row == missing) {
      while (used.at(unused) != 0) {
        ++unused;
      }
      row = unused++;
    } else {
      band.diagonal.at(col) = work.at(row, col);
    }
    left.at(row, col) = 1;
  }
  right.identity();
  return true;
}

void absoluteSpectrum(Bidiagonal& band, WorkMatrix& right) {
  for (Index j = 0; j < band.diagonal.size(); ++j) {
    if (band.diagonal.at(j) < 0) {
      for (Index row = 0; row < right.rows(); ++row) {
        right.at(row, j) = -right.at(row, j);
      }
    }
    band.diagonal.at(j) = std::abs(band.diagonal.at(j));
  }
}

Wide gramDefect(const WorkMatrix& factor) {
  Wide defect = 0;
  for (Index i = 0; i < factor.columns(); ++i) {
    for (Index j = 0; j < factor.columns(); ++j) {
      Wide entry = i == j ? -1 : 0;
      for (Index row = 0; row < factor.rows(); ++row) {
        entry = std::fma(factor.at(row, i), factor.at(row, j), entry);
      }
      defect = std::hypot(defect, entry);
    }
  }
  return defect;
}

Scalar tallEntry(ConstMatrixView input, bool transpose, Index tallAxis,
                 Index thinAxis) {
  return transpose ? input.at(thinAxis, tallAxis)
                   : input.at(tallAxis, thinAxis);
}

Wide reconstructionError(ConstMatrixView input, bool transpose, int exponent,
                         const WorkMatrix& left, const WorkMatrix& right,
                         const Bidiagonal& band, Wide largest) {
  Wide residual = 0;
  Wide norm = 0;
  for (Index row = 0; row < left.rows(); ++row) {
    for (Index col = 0; col < right.rows(); ++col) {
      const Wide original = std::scalbn(
          static_cast<Wide>(tallEntry(input, transpose, row, col)), -exponent);
      Wide rebuilt = 0;
      for (Index i = 0; i < right.columns(); ++i) {
        rebuilt = std::fma(left.at(row, i) * band.diagonal.at(i),
                           right.at(col, i), rebuilt);
      }
      residual = std::hypot(residual, original - rebuilt);
      norm = std::hypot(norm, original);
    }
  }
  const Wide roundoff = (32 * std::numeric_limits<Wide>::epsilon()) *
                        static_cast<Wide>(left.rows());
  const Wide leftDefect = gramDefect(left) + roundoff;
  const Wide rightDefect = gramDefect(right) + roundoff;
  if (std::max(leftDefect, rightDefect) >= 0.5L) {
    return std::numeric_limits<Wide>::infinity();
  }
  // Distance to the polar orthogonal factor is at most the Gram defect
  // in operator norm. Add both corrections before applying Weyl's bound.
  const Wide polarError =
      largest * ((leftDefect * (1 + rightDefect)) + rightDefect);
  return (residual + polarError) + (roundoff * norm);
}

Index exactRankUpper(ConstMatrixView input) {
  Index nonzero_rows = 0;
  Index nonzero_cols = 0;
  Index pivot_row = 0;
  Index pivot_col = 0;
  for (Index i = 0; i < input.rows(); ++i) {
    for (Index j = 0; j < input.columns(); ++j) {
      if (input.at(i, j) != 0) {
        ++nonzero_rows;
        pivot_row = i;
        pivot_col = j;
        break;
      }
    }
  }
  for (Index j = 0; j < input.columns(); ++j) {
    for (Index i = 0; i < input.rows(); ++i) {
      if (input.at(i, j) != 0) {
        ++nonzero_cols;
        break;
      }
    }
  }
  Index rank_upper = std::min(nonzero_rows, nonzero_cols);
  // A product and its FMA residual uniquely represent the exact
  // product of two binary64 values in the supported wide format.
  const auto product = [](Wide a, Wide b) {
    const Wide p = a * b;
    return std::pair{p, std::fma(a, b, -p)};
  };
  bool rank_one = rank_upper > 1;
  for (Index i = 0; rank_one && i < input.rows(); ++i) {
    for (Index j = 0; j < input.columns(); ++j) {
      if (product(input.at(i, j), input.at(pivot_row, pivot_col)) !=
          product(input.at(i, pivot_col), input.at(pivot_row, j))) {
        rank_one = false;
        break;
      }
    }
  }
  if (rank_one) {
    rank_upper = 1;
  }
  return rank_upper;
}

Result<SVD> makeResult(ConstMatrixView input, const WorkMatrix& left,
                       const WorkMatrix& right, const Bidiagonal& band,
                       const SolvePolicy& policy, bool exact, int exponent,
                       Wide* spectral_error = nullptr) {
  const auto count = band.diagonal.size();
  std::vector<Index> order(count);
  std::iota(order.begin(), order.end(), Index{0});
  std::ranges::sort(order, [&band](Index a, Index b) {
    const auto first = band.diagonal.at(a);
    const auto second = band.diagonal.at(b);
    return first > second || (first == second && a < b);
  });
  const Wide largest = count == 0 ? 0 : band.diagonal.at(order.front());
  const Wide absolute =
      std::scalbn(static_cast<Wide>(policy.rank_abs), -exponent);
  const Wide relative = policy.rank_rel;
  const Wide cutoff = std::max(static_cast<Wide>(policy.rank_abs),
                               relative * std::scalbn(largest, exponent));
  const bool transpose = input.rows() < input.columns();
  const Wide error = exact ? 0
                           : reconstructionError(input, transpose, exponent,
                                                 left, right, band, largest);
  if (spectral_error != nullptr) {
    *spectral_error = std::scalbn(error, exponent);
    if (!std::isfinite(*spectral_error)) {
      return Result<SVD>::failure(ErrorCode::NoConvergence,
                                  "SVD error bound is not finite");
    }
  }
  if (!detail::representable(cutoff)) {
    return Result<SVD>::failure(ErrorCode::NonFiniteResult,
                                "SVD cutoff overflow");
  }
  const Index rank_upper = exact ? count : exactRankUpper(input);
  SVD result{.left = Matrix(input.rows(), count),
             .singular = Vector(count),
             .right_transpose = Matrix(count, input.columns()),
             .rank = 0,
             .cutoff = static_cast<Scalar>(cutoff)};
  for (Index i = 0; i < count; ++i) {
    const auto axis = order.at(i);
    const Wide value = i < rank_upper ? band.diagonal.at(axis) : 0;
    const Wide originalValue = std::scalbn(value, exponent);
    if (!detail::representable(originalValue)) {
      return Result<SVD>::failure(ErrorCode::NonFiniteResult,
                                  "SVD spectrum overflow");
    }
    const Wide absoluteGap = value - absolute;
    const Wide relativeGap = std::fma(-relative, largest, value);
    if (spectral_error == nullptr && !exact && i < rank_upper &&
        std::abs(std::min(absoluteGap, relativeGap)) <=
            (1 + relative) * error) {
      return Result<SVD>::failure(ErrorCode::NoConvergence,
                                  "SVD cutoff could not be certified");
    }
    if (absoluteGap > 0 && relativeGap > 0) {
      ++result.rank;
    }
    result.singular.at(i) = static_cast<Scalar>(originalValue);
    for (Index row = 0; row < input.rows(); ++row) {
      result.left.at(row, i) = static_cast<Scalar>(
          transpose ? right.at(row, axis) : left.at(row, axis));
    }
    for (Index coordinate = 0; coordinate < input.columns(); ++coordinate) {
      result.right_transpose.at(i, coordinate) = static_cast<Scalar>(
          transpose ? left.at(coordinate, axis) : right.at(coordinate, axis));
    }
  }
  return Result<SVD>::success(std::move(result));
}

template <class Finish>
auto decomposeSpectrum(ConstMatrixView input, SolvePolicy policy, Limits limits,
                       Finish finish) {
  using Output = decltype(finish(std::declval<const WorkMatrix&>(),
                                 std::declval<const WorkMatrix&>(),
                                 std::declval<const Bidiagonal&>(), true, 0));
  const bool transpose = input.rows() < input.columns();
  const auto thin = std::min(input.rows(), input.columns());
  const auto tall = std::max(input.rows(), input.columns());
  detail::StorageBudget budget(limits);
  if (!claimStorage(budget, tall, thin)) {
    return Output::failure(ErrorCode::ResourceLimit,
                           "SVD workspace exceeds limits");
  }

  Scalar maximum = 0;
  for (Index row = 0; row < input.rows(); ++row) {
    for (Index col = 0; col < input.columns(); ++col) {
      const Scalar value = input.at(row, col);
      if (!std::isfinite(value)) {
        return Output::failure(ErrorCode::NonFiniteInput,
                               "Nonfinite SVD input");
      }
      maximum = std::max(maximum, std::abs(value));
    }
  }

  WorkMatrix work(tall, thin);
  for (Index row = 0; row < tall; ++row) {
    for (Index col = 0; col < thin; ++col) {
      work.at(row, col) = tallEntry(input, transpose, row, col);
    }
  }
  WorkMatrix left(tall, thin);
  WorkMatrix right(thin, thin);
  Bidiagonal band{.diagonal = std::vector<Wide>(thin),
                  .super = std::vector<Wide>(thin - 1)};
  if (directFactors(work, left, right, band)) {
    absoluteSpectrum(band, right);
    return finish(left, right, band, true, 0);
  }
  if (policy.iteration.max_iterations == 0) {
    return Output::failure(ErrorCode::NoConvergence, "SVD needs iteration");
  }
  const int exponent = std::ilogb(maximum);
  for (Index row = 0; row < tall; ++row) {
    for (Index col = 0; col < thin; ++col) {
      auto& entry = work.at(row, col);
      const Wide scaled = std::scalbn(entry, -exponent);
      if (entry != 0 && scaled == 0) {
        return Output::failure(ErrorCode::NoConvergence,
                               "SVD scaling lost a nonzero entry");
      }
      entry = scaled;
    }
  }
  left.identity();
  right.identity();
  reduceToBidiagonal(work, left, right, band);
  if (!diagonalize(band, left, right, policy.iteration.max_iterations)) {
    return Output::failure(ErrorCode::NoConvergence,
                           "SVD iteration cap reached");
  }
  absoluteSpectrum(band, right);
  return finish(left, right, band, false, exponent);
}

} // namespace

Result<SVD> factorSVD(ConstMatrixView input, SolvePolicy policy,
                      Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return Result<SVD>::failure(ErrorCode::InvalidTolerance,
                                "Invalid SVD policy");
  }
  if (std::min(input.rows(), input.columns()) == 0) {
    return Result<SVD>::success({.left = Matrix(input.rows(), 0),
                                 .singular = Vector(),
                                 .right_transpose = Matrix(0, input.columns()),
                                 .rank = 0,
                                 .cutoff = policy.rank_abs});
  }
  return decomposeSpectrum(
      input, policy, limits,
      [&](const WorkMatrix& left, const WorkMatrix& right,
          const Bidiagonal& band, bool exact, int exponent) {
        return makeResult(input, left, right, band, policy, exact, exponent);
      });
}

Result<detail::BoundedSVD>
detail::boundedSVD(ConstMatrixView input, SolvePolicy policy, Limits limits) {
  using Output = Result<BoundedSVD>;
  if (input.empty()) {
    auto factors = factorSVD(input, policy, limits);
    if (!factors.hasValue()) {
      return Output::failure(factors.status().code(),
                             std::string(factors.status().message()));
    }
    return Output::success(
        {.factors = std::move(factors).value(), .error = 0, .rank_upper = 0});
  }
  policy.rank_abs = 0;
  policy.rank_rel = 0;
  return decomposeSpectrum(
      input, policy, limits,
      [&](const WorkMatrix& left, const WorkMatrix& right,
          const Bidiagonal& band, bool exact, int exponent) -> Output {
        Wide error = 0;
        auto factors = makeResult(input, left, right, band, policy, exact,
                                  exponent, &error);
        if (!factors.hasValue()) {
          return Output::failure(factors.status().code(),
                                 std::string(factors.status().message()));
        }
        if (exact) {
          const Index rank = factors.value().rank;
          return Output::success({.factors = std::move(factors).value(),
                                  .error = 0,
                                  .rank_upper = rank});
        }
        const Index rank_upper = exactRankUpper(input);
        // Enclose the conversion of the wide singular values to the public
        // type.
        error += std::numeric_limits<Scalar>::epsilon() *
                     static_cast<Wide>(factors.value().singular.at(0)) +
                 std::numeric_limits<Scalar>::denorm_min();
        return Output::success({.factors = std::move(factors).value(),
                                .error = error,
                                .rank_upper = rank_upper});
      });
}

Status detail::certifyNonsingularSpectrum(ConstMatrixView input,
                                          SolvePolicy policy, Limits limits) {
  const bool checkRank = policy.rank == RankPolicy::reject &&
                         (policy.rank_abs != 0 || policy.rank_rel != 0);
  if (!checkRank && policy.reciprocal_condition_floor == 0) {
    return Status::success();
  }
  return decomposeSpectrum(
      input, policy, limits,
      [&](const WorkMatrix& left, const WorkMatrix& right,
          const Bidiagonal& band, bool exact, int exponent) {
        const auto [minimum, maximum] =
            std::ranges::minmax_element(band.diagonal);
        const Wide smallest = *minimum;
        const Wide largest = *maximum;
        const Wide error =
            exact ? 0
                  : reconstructionError(input, false, exponent, left, right,
                                        band, largest);
        const auto comparisonError = [&](Wide ratio) {
          return (1 + ratio) *
                 (error +
                  (exact ? 0
                         : 4 * std::numeric_limits<Wide>::epsilon() * largest));
        };
        if (checkRank) {
          const Wide absolute =
              std::scalbn(static_cast<Wide>(policy.rank_abs), -exponent);
          const Wide relative = policy.rank_rel;
          const Wide gap = std::min(smallest - absolute,
                                    std::fma(-relative, largest, smallest));
          const Wide allowance = comparisonError(relative);
          if (gap <= -allowance) {
            return Status::failure(ErrorCode::RankDeficient,
                                   "determinant spectrum fails strict rank");
          }
          if (!(gap > allowance)) {
            return Status::failure(ErrorCode::NoConvergence,
                                   "determinant rank is unresolved");
          }
        }
        const Wide floor = policy.reciprocal_condition_floor;
        if (floor != 0) {
          const Wide gap = std::fma(-floor, largest, smallest);
          const Wide allowance = comparisonError(floor);
          if (gap < -allowance) {
            return Status::failure(
                ErrorCode::IllConditioned,
                "determinant spectrum fails condition floor");
          }
          if (!(gap >= allowance)) {
            return Status::failure(ErrorCode::NoConvergence,
                                   "determinant condition is unresolved");
          }
        }
        return Status::success();
      });
}

} // namespace vss::math::linear
