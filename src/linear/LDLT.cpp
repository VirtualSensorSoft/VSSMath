#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {
namespace {

using detail::Wide;

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool checkedSquare(Index size, Index& count) noexcept {
  if (size != 0U && size > std::numeric_limits<Index>::max() / size) {
    return false;
  }
  count = size * size;
  return true;
}

bool validInput(ConstMatrixView input) noexcept {
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      if (!std::isfinite(input.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool withinSymmetryAllowance(ConstMatrixView input, Scalar maximum) noexcept {
  const Wide scale = static_cast<Wide>(maximum);
  const Wide dimension = static_cast<Wide>(std::max<Index>(1U, input.rows()));
  const Wide allowance =
      64.0L * dimension *
      static_cast<Wide>(std::numeric_limits<Scalar>::epsilon());
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = row + 1U; column < input.columns(); ++column) {
      if (scale == 0.0L) {
        continue;
      }
      const Wide lower = static_cast<Wide>(input.at(row, column)) / scale;
      const Wide upper = static_cast<Wide>(input.at(column, row)) / scale;
      if (std::abs(lower - upper) > allowance) {
        return false;
      }
    }
  }
  return true;
}

bool preflight(Index size, Limits limits) {
  if (size == 0U) {
    return true;
  }
  Index square = 0U;
  if (!checkedSquare(size, square)) {
    return false;
  }

  // Rank certification owns the canonical input plus the same workspace and
  // output categories claimed by factorSVD for a square matrix.
  detail::StorageBudget rankBudget(limits);
  if (!rankBudget.claim<Scalar>(size, size) ||
      !rankBudget.claim<Wide>(size, size) ||
      !rankBudget.claim<Wide>(size, size) ||
      !rankBudget.claim<Wide>(size, size) || !rankBudget.claim<Wide>(size) ||
      !rankBudget.claim<Wide>(size, 2U) || !rankBudget.claim<Index>(size) ||
      !rankBudget.claim<Index>(size, 2U) ||
      !rankBudget.claim<Scalar>(size, size) ||
      !rankBudget.claim<Scalar>(size, size) ||
      !rankBudget.claim<Scalar>(size)) {
    return false;
  }

  // Elimination peaks at the work matrix, two returned matrices, temporary
  // pivot lists and their immutable owning Array results.
  detail::StorageBudget factorBudget(limits);
  return factorBudget.claim<Scalar>(size, size) &&
         factorBudget.claim<Scalar>(size, size) &&
         factorBudget.claim<Scalar>(size, size) &&
         factorBudget.claim<Index>(size) &&
         factorBudget.claim<std::uint8_t>(size) &&
         factorBudget.claim<Index>(size) &&
         factorBudget.claim<std::uint8_t>(size);
}

bool reducedSvdLimits(Index size, Limits limits, Limits& reduced) noexcept {
  Index square = 0U;
  if (!checkedSquare(size, square) || square > limits.elements ||
      square > limits.bytes / sizeof(Scalar)) {
    return false;
  }
  reduced = {.elements = limits.elements - square,
             .bytes = limits.bytes - (square * sizeof(Scalar)),
             .evaluations = limits.evaluations};
  return true;
}

void swapActiveSymmetric(Matrix& work, Matrix& lower,
                         std::vector<Index>& permutation, Index active,
                         Index first, Index second) {
  if (first == second) {
    return;
  }
  const Index size = work.rows();
  for (Index column = active; column < size; ++column) {
    std::swap(work.at(first, column), work.at(second, column));
  }
  for (Index row = active; row < size; ++row) {
    std::swap(work.at(row, first), work.at(row, second));
  }
  for (Index column = 0U; column < active; ++column) {
    std::swap(lower.at(first, column), lower.at(second, column));
  }
  std::swap(permutation.at(first), permutation.at(second));
}

bool solvePivotBlock(Wide a, Wide b, Wide c, Wide first, Wide second,
                     Wide& solutionFirst, Wide& solutionSecond) noexcept {
  const Wide scale = std::max({std::abs(a), std::abs(b), std::abs(c)});
  if (scale == 0.0L) {
    return false;
  }
  const Wide aa = a / scale;
  const Wide bb = b / scale;
  const Wide cc = c / scale;
  const Wide rr = first / scale;
  const Wide ss = second / scale;

  if (std::abs(aa) >= std::abs(bb)) {
    if (aa == 0.0L) {
      return false;
    }
    const Wide multiplier = bb / aa;
    const Wide schur = std::fma(-multiplier, bb, cc);
    if (schur == 0.0L) {
      return false;
    }
    solutionSecond = std::fma(-multiplier, rr, ss) / schur;
    solutionFirst = std::fma(-bb, solutionSecond, rr) / aa;
  } else {
    const Wide multiplier = aa / bb;
    const Wide schur = std::fma(-multiplier, cc, bb);
    if (schur == 0.0L) {
      return false;
    }
    solutionSecond = std::fma(-multiplier, ss, rr) / schur;
    solutionFirst = std::fma(-cc, solutionSecond, ss) / bb;
  }
  return std::isfinite(solutionFirst) && std::isfinite(solutionSecond);
}

void findColumnMaximum(const Matrix& work, Index active, Index& pivot,
                       Scalar& maximum) noexcept {
  pivot = active;
  maximum = 0.0;
  for (Index row = active + 1U; row < work.rows(); ++row) {
    const Scalar magnitude = std::abs(work.at(row, active));
    if (magnitude > maximum) {
      maximum = magnitude;
      pivot = row;
    }
  }
}

bool chooseOneByOne(const Matrix& work, Index active, Index offDiagonalIndex,
                    Scalar columnMaximum, Scalar alpha,
                    Index& selected) noexcept {
  const Scalar diagonal = std::abs(work.at(active, active));
  if (diagonal >= alpha * columnMaximum) {
    selected = active;
    return true;
  }

  Scalar rowMaximum = 0.0;
  for (Index column = active; column < work.columns(); ++column) {
    if (column != offDiagonalIndex) {
      rowMaximum =
          std::max(rowMaximum, std::abs(work.at(offDiagonalIndex, column)));
    }
  }
  if (rowMaximum == 0.0) {
    selected = active;
    return true;
  }
  if (diagonal >= alpha * columnMaximum * (columnMaximum / rowMaximum)) {
    selected = active;
    return true;
  }
  if (std::abs(work.at(offDiagonalIndex, offDiagonalIndex)) >=
      alpha * rowMaximum) {
    selected = offDiagonalIndex;
    return true;
  }
  return false;
}

bool finiteMatrix(const Matrix& matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool certifiedReconstruction(ConstMatrixView input, const Matrix& lower,
                             const Matrix& block,
                             const std::vector<Index>& perm,
                             const std::vector<std::uint8_t>& widths,
                             int exponent) noexcept {
  Wide residualNorm = 0.0L;
  Wide inputNorm = 0.0L;
  Wide absoluteProductNorm = 0.0L;
  Index blockStart = 0U;
  const Index size = input.rows();

  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      Wide reconstructed = 0.0L;
      Wide absoluteProduct = 0.0L;
      blockStart = 0U;
      for (const auto width : widths) {
        for (Index left = 0U; left < width; ++left) {
          for (Index right = 0U; right < width; ++right) {
            const Wide term =
                static_cast<Wide>(lower.at(row, blockStart + left)) *
                static_cast<Wide>(
                    block.at(blockStart + left, blockStart + right)) *
                static_cast<Wide>(lower.at(column, blockStart + right));
            reconstructed += term;
            absoluteProduct += std::abs(term);
          }
        }
        blockStart += width;
      }

      const Scalar originalMidpoint = std::midpoint(
          input.at(perm[row], perm[column]), input.at(perm[column], perm[row]));
      const Wide expected =
          std::scalbn(static_cast<Wide>(originalMidpoint), -exponent);
      if (!std::isfinite(reconstructed) || !std::isfinite(expected) ||
          !std::isfinite(absoluteProduct)) {
        return false;
      }
      residualNorm = std::hypot(residualNorm, reconstructed - expected);
      inputNorm = std::hypot(inputNorm, expected);
      absoluteProductNorm = std::hypot(absoluteProductNorm, absoluteProduct);
    }
  }

  const Wide tolerance =
      128.0L * static_cast<Wide>(std::max<Index>(1U, size)) *
      static_cast<Wide>(std::numeric_limits<Scalar>::epsilon()) *
      (inputNorm + absoluteProductNorm);
  return std::isfinite(residualNorm) && std::isfinite(tolerance) &&
         residualNorm <= tolerance;
}

} // namespace

Result<LDLT> factorLDLT(ConstMatrixView input, SolvePolicy policy,
                        Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return Result<LDLT>::failure(ErrorCode::InvalidTolerance,
                                 "LDLT numerical policy is invalid");
  }
  if (input.rows() != input.columns()) {
    return Result<LDLT>::failure(ErrorCode::DimensionMismatch,
                                 "LDLT requires a square matrix");
  }
  if (!validInput(input)) {
    return Result<LDLT>::failure(ErrorCode::NonFiniteInput,
                                 "LDLT input must be finite");
  }

  const Index size = input.rows();
  if (size == 0U) {
    return Result<LDLT>::success({.unit_lower = Matrix{},
                                  .block_d = Matrix{},
                                  .permutation = Array<Index>{},
                                  .pivot_sizes = Array<std::uint8_t>{},
                                  .rank = 0U});
  }
  if (!preflight(size, limits)) {
    return Result<LDLT>::failure(ErrorCode::ResourceLimit,
                                 "LDLT workspace exceeds configured limits");
  }

  Scalar maximum = 0.0;
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      maximum = std::max(maximum, std::abs(input.at(row, column)));
    }
  }
  if (!withinSymmetryAllowance(input, maximum)) {
    return Result<LDLT>::failure(
        ErrorCode::NotSymmetric,
        "LDLT input is outside the symmetry allowance");
  }

  Matrix work(size, size);
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column <= row; ++column) {
      const Scalar value =
          std::midpoint(input.at(row, column), input.at(column, row));
      work.at(row, column) = value;
      work.at(column, row) = value;
    }
  }

  Limits rankLimits{};
  if (!reducedSvdLimits(size, limits, rankLimits)) {
    return Result<LDLT>::failure(
        ErrorCode::ResourceLimit,
        "LDLT rank workspace exceeds configured limits");
  }
  Index rank = 0U;
  {
    auto spectrum = factorSVD(work.view(), policy, rankLimits);
    if (!spectrum.hasValue()) {
      return copyFailure<LDLT>(spectrum.status());
    }
    rank = spectrum.value().rank;
  }

  const int exponent = maximum == 0.0 ? 0 : std::ilogb(maximum);
  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      const Scalar original = work.at(row, column);
      const Scalar scaled = std::scalbn(original, -exponent);
      if (original != 0.0 && scaled == 0.0) {
        return Result<LDLT>::failure(
            ErrorCode::NoConvergence,
            "LDLT power-of-two scaling lost a nonzero entry");
      }
      work.at(row, column) = scaled;
    }
  }

  Matrix lower(size, size, 0.0);
  Matrix block(size, size, 0.0);
  std::vector<Index> permutation(size);
  std::vector<std::uint8_t> widths;
  widths.reserve(size);
  for (Index index = 0U; index < size; ++index) {
    lower.at(index, index) = 1.0;
    permutation.at(index) = index;
  }

  constexpr Scalar alpha = (1.0 + 4.1231056256176605498) / 8.0;
  Index active = 0U;
  while (active < size) {
    Index offDiagonalIndex = active;
    Scalar columnMaximum = 0.0;
    findColumnMaximum(work, active, offDiagonalIndex, columnMaximum);

    Index oneByOneIndex = active;
    const bool useOneByOne = chooseOneByOne(
        work, active, offDiagonalIndex, columnMaximum, alpha, oneByOneIndex);
    if (useOneByOne) {
      swapActiveSymmetric(work, lower, permutation, active, active,
                          oneByOneIndex);
      const Scalar pivot = work.at(active, active);
      block.at(active, active) = pivot;
      for (Index row = active + 1U; row < size; ++row) {
        const Scalar value = work.at(row, active);
        if (pivot == 0.0) {
          if (value != 0.0) {
            return Result<LDLT>::failure(
                ErrorCode::NoConvergence,
                "LDLT selected an unusable zero singleton pivot");
          }
          lower.at(row, active) = 0.0;
        } else {
          lower.at(row, active) = value / pivot;
          if (!std::isfinite(lower.at(row, active))) {
            return Result<LDLT>::failure(
                ErrorCode::NonFiniteResult,
                "LDLT lower factor is not representable");
          }
        }
      }
      for (Index row = active + 1U; row < size; ++row) {
        for (Index column = row; column < size; ++column) {
          const Wide update = static_cast<Wide>(work.at(row, column)) -
                              static_cast<Wide>(lower.at(row, active)) *
                                  static_cast<Wide>(work.at(column, active));
          const Scalar stored = static_cast<Scalar>(update);
          if (!std::isfinite(update) || !std::isfinite(stored)) {
            return Result<LDLT>::failure(ErrorCode::NonFiniteResult,
                                         "LDLT trailing update overflowed");
          }
          work.at(row, column) = stored;
          work.at(column, row) = stored;
        }
      }
      widths.push_back(1U);
      ++active;
      continue;
    }

    if (active + 1U >= size) {
      return Result<LDLT>::failure(ErrorCode::NoConvergence,
                                   "LDLT could not select a pivot block");
    }
    swapActiveSymmetric(work, lower, permutation, active, active + 1U,
                        offDiagonalIndex);
    const Wide a = work.at(active, active);
    const Wide b = work.at(active, active + 1U);
    const Wide c = work.at(active + 1U, active + 1U);
    block.at(active, active) = static_cast<Scalar>(a);
    block.at(active, active + 1U) = static_cast<Scalar>(b);
    block.at(active + 1U, active) = static_cast<Scalar>(b);
    block.at(active + 1U, active + 1U) = static_cast<Scalar>(c);

    for (Index row = active + 2U; row < size; ++row) {
      Wide first = 0.0L;
      Wide second = 0.0L;
      if (!solvePivotBlock(a, b, c, work.at(row, active),
                           work.at(row, active + 1U), first, second)) {
        return Result<LDLT>::failure(ErrorCode::NoConvergence,
                                     "LDLT 2x2 pivot solve was unresolved");
      }
      const Scalar storedFirst = static_cast<Scalar>(first);
      const Scalar storedSecond = static_cast<Scalar>(second);
      if (!std::isfinite(first) || !std::isfinite(second) ||
          !std::isfinite(storedFirst) || !std::isfinite(storedSecond)) {
        return Result<LDLT>::failure(ErrorCode::NonFiniteResult,
                                     "LDLT lower factor is not representable");
      }
      lower.at(row, active) = storedFirst;
      lower.at(row, active + 1U) = storedSecond;
    }

    for (Index row = active + 2U; row < size; ++row) {
      for (Index column = row; column < size; ++column) {
        const Wide rowFirst = lower.at(row, active);
        const Wide rowSecond = lower.at(row, active + 1U);
        const Wide columnFirst = lower.at(column, active);
        const Wide columnSecond = lower.at(column, active + 1U);
        const Wide firstProduct = std::fma(a, columnFirst, b * columnSecond);
        const Wide secondProduct = std::fma(b, columnFirst, c * columnSecond);
        const Wide update =
            std::fma(-rowFirst, firstProduct,
                     std::fma(-rowSecond, secondProduct, work.at(row, column)));
        const Scalar stored = static_cast<Scalar>(update);
        if (!std::isfinite(update) || !std::isfinite(stored)) {
          return Result<LDLT>::failure(ErrorCode::NonFiniteResult,
                                       "LDLT trailing update overflowed");
        }
        work.at(row, column) = stored;
        work.at(column, row) = stored;
      }
    }
    widths.push_back(2U);
    active += 2U;
  }

  if (!finiteMatrix(lower) || !finiteMatrix(block) ||
      !certifiedReconstruction(input, lower, block, permutation, widths,
                               exponent)) {
    return Result<LDLT>::failure(
        ErrorCode::NoConvergence,
        "LDLT factors did not meet the reconstruction residual bound");
  }

  for (Index row = 0U; row < size; ++row) {
    for (Index column = 0U; column < size; ++column) {
      const Wide original =
          std::scalbn(static_cast<Wide>(block.at(row, column)), exponent);
      if (!detail::representable(original)) {
        return Result<LDLT>::failure(ErrorCode::NonFiniteResult,
                                     "LDLT block factor is not representable");
      }
      const Scalar stored = static_cast<Scalar>(original);
      if (block.at(row, column) != 0.0 && stored == 0.0) {
        return Result<LDLT>::failure(ErrorCode::NonFiniteResult,
                                     "LDLT block factor underflowed");
      }
      block.at(row, column) = stored;
    }
  }

  const auto permutationArray = Array<Index>::copy(permutation, limits);
  if (!permutationArray.hasValue()) {
    return copyFailure<LDLT>(permutationArray.status());
  }
  const auto widthsArray = Array<std::uint8_t>::copy(widths, limits);
  if (!widthsArray.hasValue()) {
    return copyFailure<LDLT>(widthsArray.status());
  }
  return Result<LDLT>::success(
      {.unit_lower = std::move(lower),
       .block_d = std::move(block),
       .permutation = std::move(permutationArray).value(),
       .pivot_sizes = std::move(widthsArray).value(),
       .rank = rank});
}

} // namespace vss::math::linear
