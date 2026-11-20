#include "DenseWorkspace.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math::linear {
namespace {

using detail::Wide;

[[nodiscard]] bool validRankPolicy(RankPolicy rank) {
  switch (rank) {
  case RankPolicy::reject:
  case RankPolicy::supported_subspace:
  case RankPolicy::minimum_norm:
    return true;
  }
  return false;
}

[[nodiscard]] bool finiteInput(ConstMatrixView input) {
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      if (!std::isfinite(input.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool withinStorageLimits(Extent rows, Extent columns,
                                       Limits limits) {
  detail::StorageBudget budget(limits);
  return budget.claim<Scalar>(rows, rows) &&
         budget.claim<Scalar>(rows, columns) &&
         budget.claim<Wide>(rows, rows) && budget.claim<Wide>(rows, columns) &&
         budget.claim<Index>(columns) && budget.claim<Index>(columns);
}

[[nodiscard]] Result<FullQR> copyFailure(const Status& status) {
  return Result<FullQR>::failure(status.code(), std::string{status.message()});
}

} // namespace

Result<FullQR> factorFullQR(ConstMatrixView input, QRColumnOrder order,
                            SolvePolicy policy, Limits limits) {
  if (order != QRColumnOrder::original && order != QRColumnOrder::pivoted) {
    return Result<FullQR>::failure(ErrorCode::InvalidDomain,
                                   "Unknown full QR column order");
  }
  if (!detail::validNumericalPolicy(policy)) {
    return Result<FullQR>::failure(ErrorCode::InvalidTolerance,
                                   "Invalid full QR rank policy");
  }
  if (!validRankPolicy(policy.rank)) {
    return Result<FullQR>::failure(ErrorCode::InvalidDomain,
                                   "Unknown full QR rank policy");
  }
  if (!finiteInput(input)) {
    return Result<FullQR>::failure(ErrorCode::NonFiniteInput,
                                   "Full QR input contains a nonfinite value");
  }

  const Extent rows = input.rows();
  const Extent columns = input.columns();
  if (!withinStorageLimits(rows, columns, limits)) {
    return Result<FullQR>::failure(ErrorCode::ResourceLimit,
                                   "Full QR workspace exceeds limits");
  }

  Index rank = 0U;
  Scalar cutoff = 0.0;
  {
    const auto spectrum = factorSVD(input, policy, limits);
    if (!spectrum.hasValue()) {
      return copyFailure(spectrum.status());
    }
    rank = spectrum.value().rank;
    cutoff = spectrum.value().cutoff;
  }

  const Index workSize = rows * columns;
  const Index qSize = rows * rows;
  std::vector<Wide> work(workSize, 0.0L);
  std::vector<Wide> qWork(qSize, 0.0L);
  std::vector<Index> permutation(columns);
  std::iota(permutation.begin(), permutation.end(), Index{0U});

  Scalar maximum = 0.0;
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < columns; ++column) {
      maximum = std::max(maximum, std::abs(input.at(row, column)));
    }
  }
  int exponent = 0;
  if (maximum != 0.0) {
    static_cast<void>(std::frexp(maximum, &exponent));
  }

  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < columns; ++column) {
      work.at((row * columns) + column) =
          std::scalbn(static_cast<Wide>(input.at(row, column)), -exponent);
    }
  }
  for (Index axis = 0U; axis < rows; ++axis) {
    qWork.at((axis * rows) + axis) = 1.0L;
  }

  const Index steps = std::min(rows, columns);
  for (Index step = 0U; step < steps; ++step) {
    if (order == QRColumnOrder::pivoted) {
      Index pivot = step;
      Wide largestNorm = -1.0L;
      for (Index column = step; column < columns; ++column) {
        Wide norm = 0.0L;
        for (Index row = step; row < rows; ++row) {
          norm = std::hypot(norm, work.at((row * columns) + column));
        }
        if (norm > largestNorm ||
            (norm == largestNorm && permutation[column] < permutation[pivot])) {
          largestNorm = norm;
          pivot = column;
        }
      }
      if (pivot != step) {
        for (Index row = 0U; row < rows; ++row) {
          std::swap(work.at((row * columns) + step),
                    work.at((row * columns) + pivot));
        }
        std::swap(permutation[step], permutation[pivot]);
      }
    }

    Wide norm = 0.0L;
    for (Index row = step; row < rows; ++row) {
      norm = std::hypot(norm, work.at((row * columns) + step));
    }
    if (norm == 0.0L) {
      for (Index row = step + 1U; row < rows; ++row) {
        work.at((row * columns) + step) = 0.0L;
      }
      continue;
    }

    const Wide alpha = work.at((step * columns) + step);
    const Wide beta = -std::copysign(norm, alpha);
    const Wide leading = alpha - beta;
    const Wide tau = (beta - alpha) / beta;
    for (Index row = step + 1U; row < rows; ++row) {
      work.at((row * columns) + step) /= leading;
    }
    work.at((step * columns) + step) = beta;

    for (Index column = step + 1U; column < columns; ++column) {
      Wide dot = work.at((step * columns) + column);
      for (Index row = step + 1U; row < rows; ++row) {
        dot = std::fma(work.at((row * columns) + step),
                       work.at((row * columns) + column), dot);
      }
      const Wide factor = tau * dot;
      work.at((step * columns) + column) -= factor;
      for (Index row = step + 1U; row < rows; ++row) {
        work.at((row * columns) + column) =
            std::fma(-factor, work.at((row * columns) + step),
                     work.at((row * columns) + column));
      }
    }

    for (Index row = 0U; row < rows; ++row) {
      Wide dot = qWork.at((row * rows) + step);
      for (Index column = step + 1U; column < rows; ++column) {
        dot = std::fma(qWork.at((row * rows) + column),
                       work.at((column * columns) + step), dot);
      }
      const Wide factor = tau * dot;
      qWork.at((row * rows) + step) -= factor;
      for (Index column = step + 1U; column < rows; ++column) {
        qWork.at((row * rows) + column) =
            std::fma(-factor, work.at((column * columns) + step),
                     qWork.at((row * rows) + column));
      }
    }
  }

  Matrix q(rows, rows, 0.0);
  Matrix r(rows, columns, 0.0);
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < rows; ++column) {
      const Wide value = qWork.at((row * rows) + column);
      if (!detail::representable(value)) {
        return Result<FullQR>::failure(ErrorCode::NonFiniteResult,
                                       "Full QR orthogonal factor overflow");
      }
      q.at(row, column) = static_cast<Scalar>(value);
    }
  }
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = row; column < columns; ++column) {
      const Wide value =
          std::scalbn(work.at((row * columns) + column), exponent);
      if (!detail::representable(value)) {
        return Result<FullQR>::failure(ErrorCode::NonFiniteResult,
                                       "Full QR upper factor overflow");
      }
      r.at(row, column) = static_cast<Scalar>(value);
    }
  }

  auto ownedPermutation = Array<Index>::copy(permutation, limits);
  if (!ownedPermutation.hasValue()) {
    return copyFailure(ownedPermutation.status());
  }
  return Result<FullQR>::success(
      {.orthogonal = std::move(q),
       .upper = std::move(r),
       .permutation = std::move(ownedPermutation).value(),
       .rank = rank,
       .cutoff = cutoff});
}

} // namespace vss::math::linear
