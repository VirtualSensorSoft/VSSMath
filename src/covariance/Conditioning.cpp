#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedMultiply(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (lhs != 0U && rhs > std::numeric_limits<Extent>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool checkedAdd(Extent lhs, Extent rhs, Extent& result) noexcept {
  if (rhs > std::numeric_limits<Extent>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool budget(Extent m, Extent n, Limits limits) noexcept {
  Extent mm = 0U;
  Extent nn = 0U;
  Extent mn = 0U;
  Extent elements = 0U;
  Extent temporary = 0U;
  if (!checkedMultiply(m, m, mm) || !checkedMultiply(n, n, nn) ||
      !checkedMultiply(m, n, mn) || !checkedAdd(mm, nn, temporary) ||
      !checkedAdd(temporary, mn, elements) ||
      !checkedAdd(elements, mm, temporary) ||
      !checkedAdd(temporary, nn, elements) ||
      !checkedAdd(elements, mn, temporary) ||
      !checkedAdd(temporary, nn, elements)) {
    return false;
  }
  return elements <= limits.elements &&
         elements <= limits.bytes / sizeof(Scalar);
}

bool finite(ConstMatrixView value) noexcept {
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      if (!std::isfinite(value.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

bool finite(ConstVectorView value) noexcept {
  for (Index index = 0U; index < value.size(); ++index) {
    if (!std::isfinite(value.at(index))) {
      return false;
    }
  }
  return true;
}

Scalar maximumMagnitude(ConstMatrixView value) noexcept {
  Scalar scale = 0.0;
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      scale = std::max(scale, std::abs(value.at(row, column)));
    }
  }
  return scale;
}

bool validPolicy(const SolvePolicy& policy) noexcept {
  return std::isfinite(policy.rank_abs) && policy.rank_abs >= 0.0 &&
         std::isfinite(policy.rank_rel) && policy.rank_rel >= 0.0 &&
         policy.rank_rel < 1.0 &&
         std::isfinite(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor >= 0.0 &&
         policy.reciprocal_condition_floor <= 1.0 &&
         std::isfinite(policy.iteration.abs) && policy.iteration.abs >= 0.0 &&
         std::isfinite(policy.iteration.rel) && policy.iteration.rel >= 0.0 &&
         (policy.rank == RankPolicy::reject ||
          policy.rank == RankPolicy::supported_subspace ||
          policy.rank == RankPolicy::minimum_norm);
}

Scalar symmetryAllowance(Extent dimension) noexcept {
  return 64.0 * static_cast<Scalar>(std::max<Extent>(1U, dimension)) *
         std::numeric_limits<Scalar>::epsilon();
}

bool symmetrize(ConstMatrixView input, Matrix& output) {
  const Scalar scale = maximumMagnitude(input);
  output = Matrix(input.rows(), input.columns(), 0.0);
  if (scale == 0.0) {
    return true;
  }
  const Scalar allowance = symmetryAllowance(input.rows());
  for (Index row = 0U; row < input.rows(); ++row) {
    output.at(row, row) = input.at(row, row);
    for (Index column = row + 1U; column < input.columns(); ++column) {
      const Scalar upper = input.at(row, column) / scale;
      const Scalar lower = input.at(column, row) / scale;
      if (std::abs(upper - lower) > allowance) {
        return false;
      }
      const Scalar average =
          std::midpoint(input.at(row, column), input.at(column, row));
      output.at(row, column) = average;
      output.at(column, row) = average;
    }
  }
  return true;
}

struct Spectrum final {
  linear::SymmetricEigenDecomposition decomposition;
  Scalar scale;
  Scalar cutoffScaled;
  Extent rank;
  Scalar minimumRetained;
  Scalar maximumRetained;
};

Result<Spectrum> spectrum(ConstMatrixView input, const SolvePolicy& policy) {
  const Extent n = input.rows();
  const Scalar scale = maximumMagnitude(input);
  Matrix scaled(n, n, 0.0);
  if (scale != 0.0) {
    for (Index row = 0U; row < n; ++row) {
      for (Index column = 0U; column < n; ++column) {
        scaled.at(row, column) = input.at(row, column) / scale;
      }
    }
  }
  linear::SymmetricEigenOptions options{};
  options.rankTolerance = {.absolute = 0.0, .relative = 0.0};
  options.symmetryTolerance = {.absolute = 0.0,
                               .relative = symmetryAllowance(n)};
  options.maximumSweeps = policy.iteration.max_iterations;
  if (options.maximumSweeps == 0U) {
    bool diagonal = true;
    for (Index row = 0U; row < n; ++row) {
      for (Index column = row + 1U; column < n; ++column) {
        diagonal = diagonal && scaled.at(row, column) == 0.0;
      }
    }
    if (!diagonal) {
      return failure<Spectrum>(
          ErrorCode::NoConvergence,
          "zero sweep budget cannot certify this spectrum");
    }
    options.maximumSweeps = 1U;
  }
  auto eigen = linear::decomposeSymmetricEigen(scaled.view(), options);
  if (!eigen.hasValue()) {
    return Result<Spectrum>::failure(eigen.status().code(),
                                     std::string(eigen.status().message()));
  }
  Scalar largest = 0.0;
  for (Index index = 0U; index < n; ++index) {
    largest =
        std::max(largest, std::abs(eigen.value().eigenvalues().at(index)));
  }
  const long double absCutoff =
      scale == 0.0 ? (policy.rank_abs == 0.0
                          ? 0.0L
                          : std::numeric_limits<long double>::infinity())
                   : static_cast<long double>(policy.rank_abs) /
                         static_cast<long double>(scale);
  const long double relativeCutoff =
      static_cast<long double>(policy.rank_rel) * largest;
  const long double cutoff = std::max(absCutoff, relativeCutoff);
  Extent rank = 0U;
  Scalar minimumRetained = std::numeric_limits<Scalar>::infinity();
  Scalar maximumRetained = 0.0;
  for (Index index = 0U; index < n; ++index) {
    const Scalar magnitude = std::abs(eigen.value().eigenvalues().at(index));
    if (static_cast<long double>(magnitude) > cutoff) {
      ++rank;
      minimumRetained = std::min(minimumRetained, magnitude);
      maximumRetained = std::max(maximumRetained, magnitude);
    }
  }
  const Scalar cutoffScaled =
      cutoff > static_cast<long double>(std::numeric_limits<Scalar>::max())
          ? std::numeric_limits<Scalar>::infinity()
          : static_cast<Scalar>(cutoff);
  return Result<Spectrum>::success(Spectrum{std::move(eigen).value(), scale,
                                            cutoffScaled, rank, minimumRetained,
                                            maximumRetained});
}

bool compatibleWithDiscarded(ConstMatrixView cross, const Spectrum& spec,
                             ConstMatrixView selected,
                             bool covarianceCompatibility) {
  Scalar crossScale = maximumMagnitude(cross);
  if (crossScale == 0.0) {
    return true;
  }
  Scalar selectedScale = maximumMagnitude(selected);
  const long double allowance = static_cast<long double>(symmetryAllowance(
                                    cross.rows() + cross.columns())) *
                                std::max(crossScale, selectedScale);
  for (Index k = 0U; k < spec.decomposition.size(); ++k) {
    const Scalar lambda = spec.decomposition.eigenvalues().at(k);
    if (std::abs(lambda) > spec.cutoffScaled) {
      continue;
    }
    for (Index row = 0U; row < cross.rows(); ++row) {
      long double projection = 0.0L;
      for (Index column = 0U; column < cross.columns(); ++column) {
        projection +=
            static_cast<long double>(cross.at(row, column) / crossScale) *
            spec.decomposition.eigenvectors().at(column, k);
      }
      long double permitted = allowance / crossScale;
      if (covarianceCompatibility && lambda > 0.0) {
        const long double eigenvalue =
            static_cast<long double>(spec.scale) * lambda;
        const long double variance = std::max(0.0, selected.at(row, row));
        permitted += std::sqrt(eigenvalue * variance) / crossScale;
      }
      if (std::abs(projection) > permitted) {
        return false;
      }
    }
  }
  return true;
}

Result<Matrix> schurImpl(ConstMatrixView aa, ConstMatrixView ab,
                         ConstMatrixView bb, const SolvePolicy& policy,
                         Limits limits, bool unsupportedIsPsdFailure) {
  if (!validPolicy(policy)) {
    return failure<Matrix>(ErrorCode::InvalidTolerance,
                           "Schur solve policy is invalid");
  }
  if (aa.rows() != aa.columns() || bb.rows() != bb.columns() ||
      ab.rows() != aa.rows() || ab.columns() != bb.rows()) {
    return failure<Matrix>(ErrorCode::DimensionMismatch,
                           "Schur block dimensions are incompatible");
  }
  if (!budget(aa.rows(), bb.rows(), limits)) {
    return failure<Matrix>(ErrorCode::ResourceLimit,
                           "Schur scratch exceeds the configured limits");
  }
  if (!finite(aa) || !finite(ab) || !finite(bb)) {
    return failure<Matrix>(ErrorCode::NonFiniteInput,
                           "Schur input contains a non-finite value");
  }
  Matrix symmetricA;
  Matrix symmetricD;
  if (!symmetrize(aa, symmetricA) || !symmetrize(bb, symmetricD)) {
    return failure<Matrix>(ErrorCode::NotSymmetric,
                           "Schur symmetric block failed symmetry validation");
  }
  const Extent n = bb.rows();
  Matrix output = symmetricA;
  if (n == 0U || aa.rows() == 0U) {
    return Result<Matrix>::success(std::move(output));
  }
  auto spec = spectrum(symmetricD.view(), policy);
  if (!spec.hasValue()) {
    return Result<Matrix>::failure(spec.status().code(),
                                   std::string(spec.status().message()));
  }
  const bool rankLoss = spec.value().rank < n;
  if (rankLoss && policy.rank == RankPolicy::reject) {
    return failure<Matrix>(ErrorCode::RankDeficient,
                           "Schur conditioning block is rank deficient");
  }
  if (spec.value().rank > 0U && policy.reciprocal_condition_floor > 0.0 &&
      spec.value().minimumRetained / spec.value().maximumRetained <
          policy.reciprocal_condition_floor) {
    return failure<Matrix>(ErrorCode::IllConditioned,
                           "Schur retained spectrum is ill-conditioned");
  }
  if (rankLoss && policy.rank == RankPolicy::supported_subspace &&
      !compatibleWithDiscarded(ab, spec.value(), aa, unsupportedIsPsdFailure)) {
    return failure<Matrix>(unsupportedIsPsdFailure
                               ? ErrorCode::NotPositiveSemidefinite
                               : ErrorCode::InvalidDomain,
                           "Schur cross block loads a discarded direction");
  }

  const Scalar bScale = maximumMagnitude(ab);
  if (bScale == 0.0 || spec.value().rank == 0U) {
    return Result<Matrix>::success(std::move(output));
  }
  const long double correctionScale =
      static_cast<long double>(bScale) * bScale / spec.value().scale;
  if (!std::isfinite(correctionScale)) {
    return failure<Matrix>(ErrorCode::NonFiniteResult,
                           "Schur correction is not representable");
  }
  const auto& eigen = spec.value().decomposition;
  for (Index row = 0U; row < aa.rows(); ++row) {
    for (Index column = row; column < aa.rows(); ++column) {
      long double sum = 0.0L;
      long double compensation = 0.0L;
      for (Index k = 0U; k < n; ++k) {
        const Scalar lambda = eigen.eigenvalues().at(k);
        if (std::abs(lambda) <= spec.value().cutoffScaled) {
          continue;
        }
        long double left = 0.0L;
        long double right = 0.0L;
        for (Index inner = 0U; inner < n; ++inner) {
          const long double q = eigen.eigenvectors().at(inner, k);
          left += static_cast<long double>(ab.at(row, inner) / bScale) * q;
          right += static_cast<long double>(ab.at(column, inner) / bScale) * q;
        }
        const long double term = left * right / lambda;
        const long double adjusted = term - compensation;
        const long double next = sum + adjusted;
        compensation = (next - sum) - adjusted;
        sum = next;
      }
      const long double correction = correctionScale * sum;
      const long double result =
          static_cast<long double>(symmetricA.at(row, column)) - correction;
      if (!std::isfinite(result) ||
          std::abs(result) > std::numeric_limits<Scalar>::max()) {
        return failure<Matrix>(ErrorCode::NonFiniteResult,
                               "Schur complement is not representable");
      }
      const Scalar rounded = static_cast<Scalar>(result);
      output.at(row, column) = rounded;
      output.at(column, row) = rounded;
    }
  }
  return Result<Matrix>::success(std::move(output));
}

} // namespace

Result<Matrix> schurComplement(ConstMatrixView aa, ConstMatrixView ab,
                               ConstMatrixView bb, SolvePolicy policy,
                               Limits limits) {
  return schurImpl(aa, ab, bb, policy, limits, false);
}

Result<CovarianceResult> conditionalCovariance(ConstMatrixView covariance,
                                               std::span<const Index> keep,
                                               std::span<const Index> given,
                                               SolvePolicy policy,
                                               Limits limits) {
  if (!validPolicy(policy)) {
    return failure<CovarianceResult>(
        ErrorCode::InvalidTolerance,
        "conditional covariance policy is invalid");
  }
  const Extent dimension = covariance.rows();
  if (dimension != covariance.columns()) {
    return failure<CovarianceResult>(ErrorCode::DimensionMismatch,
                                     "joint covariance must be square");
  }
  if (keep.empty()) {
    return failure<CovarianceResult>(
        ErrorCode::InsufficientSamples,
        "conditional covariance needs kept coordinates");
  }
  if (!budget(dimension, dimension, limits)) {
    return failure<CovarianceResult>(ErrorCode::ResourceLimit,
                                     "joint covariance scratch exceeds limits");
  }
  if (!finite(covariance)) {
    return failure<CovarianceResult>(
        ErrorCode::NonFiniteInput,
        "joint covariance contains non-finite values");
  }
  std::vector<bool> selected(dimension, false);
  for (const Index index : keep) {
    if (index >= dimension) {
      return failure<CovarianceResult>(ErrorCode::IndexOutOfRange,
                                       "kept covariance index is out of range");
    }
    if (selected[index]) {
      return failure<CovarianceResult>(
          ErrorCode::InvalidDomain, "kept covariance indices must be unique");
    }
    selected[index] = true;
  }
  for (const Index index : given) {
    if (index >= dimension) {
      return failure<CovarianceResult>(
          ErrorCode::IndexOutOfRange,
          "conditioning covariance index is out of range");
    }
    if (selected[index]) {
      return failure<CovarianceResult>(
          ErrorCode::InvalidDomain,
          "kept and conditioned indices must be disjoint");
    }
    selected[index] = true;
  }
  std::vector<bool> seen(dimension, false);
  for (const Index index : given) {
    if (seen[index]) {
      return failure<CovarianceResult>(
          ErrorCode::InvalidDomain,
          "conditioning covariance indices must be unique");
    }
    seen[index] = true;
  }

  Matrix joint;
  if (!symmetrize(covariance, joint)) {
    return failure<CovarianceResult>(
        ErrorCode::NotSymmetric, "joint covariance failed symmetry validation");
  }
  if (dimension > 0U) {
    auto jointSpectrum = spectrum(joint.view(), policy);
    if (!jointSpectrum.hasValue()) {
      return Result<CovarianceResult>::failure(
          jointSpectrum.status().code(),
          std::string(jointSpectrum.status().message()));
    }
    const Scalar allowance = symmetryAllowance(dimension);
    for (Index index = 0U; index < dimension; ++index) {
      if (jointSpectrum.value().decomposition.eigenvalues().at(index) <
          -allowance) {
        return failure<CovarianceResult>(
            ErrorCode::NotPositiveSemidefinite,
            "joint covariance is not positive semidefinite");
      }
    }
  }

  const Extent m = keep.size();
  const Extent n = given.size();
  Matrix aa(m, m, 0.0);
  Matrix ab(m, n, 0.0);
  Matrix bb(n, n, 0.0);
  for (Index row = 0U; row < m; ++row) {
    for (Index column = 0U; column < m; ++column) {
      aa.at(row, column) = joint.at(keep[row], keep[column]);
    }
    for (Index column = 0U; column < n; ++column) {
      ab.at(row, column) = joint.at(keep[row], given[column]);
    }
  }
  for (Index row = 0U; row < n; ++row) {
    for (Index column = 0U; column < n; ++column) {
      bb.at(row, column) = joint.at(given[row], given[column]);
    }
  }
  const auto schur =
      schurImpl(aa.view(), ab.view(), bb.view(), policy, limits, true);
  if (!schur.hasValue()) {
    return Result<CovarianceResult>::failure(
        schur.status().code(), std::string(schur.status().message()));
  }
  Matrix conditional = std::move(schur).value();
  const Scalar scale = maximumMagnitude(conditional.view());
  Extent rank = 0U;
  Scalar cutoff = policy.rank_abs;
  Scalar reciprocal = 1.0;
  if (m > 0U && scale > 0.0) {
    auto resultSpectrum = spectrum(conditional.view(), policy);
    if (!resultSpectrum.hasValue()) {
      return Result<CovarianceResult>::failure(
          resultSpectrum.status().code(),
          std::string(resultSpectrum.status().message()));
    }
    const Scalar psdAllowance = symmetryAllowance(m);
    for (Index index = 0U; index < m; ++index) {
      if (resultSpectrum.value().decomposition.eigenvalues().at(index) <
          -psdAllowance) {
        return failure<CovarianceResult>(
            ErrorCode::NotPositiveSemidefinite,
            "conditional covariance is not positive semidefinite");
      }
    }
    rank = 0U;
    Scalar minimumPositive = std::numeric_limits<Scalar>::infinity();
    Scalar maximumPositive = 0.0;
    for (Index index = 0U; index < m; ++index) {
      const Scalar eigenvalue =
          resultSpectrum.value().decomposition.eigenvalues().at(index);
      if (eigenvalue > resultSpectrum.value().cutoffScaled) {
        ++rank;
        minimumPositive = std::min(minimumPositive, eigenvalue);
        maximumPositive = std::max(maximumPositive, eigenvalue);
      }
    }
    const long double cutoffValue =
        static_cast<long double>(scale) * resultSpectrum.value().cutoffScaled;
    cutoff = cutoffValue > std::numeric_limits<Scalar>::max()
                 ? std::numeric_limits<Scalar>::infinity()
                 : static_cast<Scalar>(cutoffValue);
    if (rank == m) {
      reciprocal = minimumPositive / maximumPositive;
    }
  }
  CovarianceKind classification = CovarianceKind::semidefinite_singular;
  std::optional<Scalar> reportedReciprocal;
  if (rank == m) {
    reportedReciprocal = reciprocal;
    classification = reciprocal < policy.reciprocal_condition_floor
                         ? CovarianceKind::ill_conditioned
                         : CovarianceKind::positive_definite;
  }
  return Result<CovarianceResult>::success(
      CovarianceResult{std::move(conditional), classification, rank, cutoff,
                       reportedReciprocal});
}

Result<Whitening> whitenSupported(ConstMatrixView covariance,
                                  ConstVectorView value, SolvePolicy policy,
                                  Limits limits) {
  if (!validPolicy(policy)) {
    return failure<Whitening>(ErrorCode::InvalidTolerance,
                              "whitening solve policy is invalid");
  }
  if (covariance.rows() != covariance.columns() ||
      value.size() != covariance.rows()) {
    return failure<Whitening>(
        ErrorCode::DimensionMismatch,
        "whitening covariance and vector dimensions differ");
  }
  const Extent n = covariance.rows();
  if (!budget(n, n, limits)) {
    return failure<Whitening>(ErrorCode::ResourceLimit,
                              "whitening scratch exceeds configured limits");
  }
  if (!finite(covariance) || !finite(value)) {
    return failure<Whitening>(ErrorCode::NonFiniteInput,
                              "whitening input contains a non-finite value");
  }
  Matrix symmetricCovariance;
  if (!symmetrize(covariance, symmetricCovariance)) {
    return failure<Whitening>(
        ErrorCode::NotSymmetric,
        "whitening covariance failed symmetry validation");
  }
  auto spec = spectrum(symmetricCovariance.view(), policy);
  if (!spec.hasValue()) {
    return Result<Whitening>::failure(spec.status().code(),
                                      std::string(spec.status().message()));
  }
  const auto& eigen = spec.value().decomposition;
  const Scalar psdAllowance = symmetryAllowance(n);
  Extent rank = 0U;
  Scalar minimumPositive = std::numeric_limits<Scalar>::infinity();
  Scalar maximumPositive = 0.0;
  for (Index index = 0U; index < n; ++index) {
    const Scalar eigenvalue = eigen.eigenvalues().at(index);
    if (eigenvalue < -psdAllowance) {
      return failure<Whitening>(
          ErrorCode::NotPositiveSemidefinite,
          "whitening covariance is not positive semidefinite");
    }
    if (eigenvalue > spec.value().cutoffScaled) {
      ++rank;
      minimumPositive = std::min(minimumPositive, eigenvalue);
      maximumPositive = std::max(maximumPositive, eigenvalue);
    }
  }
  if (rank < n && policy.rank == RankPolicy::reject) {
    return failure<Whitening>(ErrorCode::RankDeficient,
                              "whitening covariance is rank deficient");
  }
  if (rank > 0U &&
      minimumPositive / maximumPositive < policy.reciprocal_condition_floor) {
    return failure<Whitening>(
        ErrorCode::IllConditioned,
        "whitening retained covariance is ill-conditioned");
  }

  long double inputNorm = 0.0L;
  for (Index index = 0U; index < n; ++index) {
    inputNorm =
        std::hypot(inputNorm, static_cast<long double>(value.at(index)));
  }
  long double nullResidual = 0.0L;
  if (rank < n) {
    if (rank == 0U) {
      for (Index index = 0U; index < n; ++index) {
        if (value.at(index) != 0.0) {
          return failure<Whitening>(
              ErrorCode::InvalidDomain,
              "zero-rank whitening requires the exact zero vector");
        }
      }
    } else {
      for (Index row = 0U; row < n; ++row) {
        long double projection = 0.0L;
        for (Index k = 0U; k < n; ++k) {
          if (!(eigen.eigenvalues().at(k) > spec.value().cutoffScaled)) {
            continue;
          }
          long double coefficient = 0.0L;
          for (Index inner = 0U; inner < n; ++inner) {
            coefficient +=
                static_cast<long double>(eigen.eigenvectors().at(inner, k)) *
                value.at(inner);
          }
          projection +=
              static_cast<long double>(eigen.eigenvectors().at(row, k)) *
              coefficient;
        }
        nullResidual = std::hypot(
            nullResidual, static_cast<long double>(value.at(row)) - projection);
      }
      const long double supportAllowance =
          static_cast<long double>(symmetryAllowance(n)) * inputNorm;
      if (nullResidual > supportAllowance) {
        return failure<Whitening>(
            ErrorCode::InvalidDomain,
            "whitening vector has a component outside covariance support");
      }
    }
  }
  if (nullResidual > std::numeric_limits<Scalar>::max()) {
    return failure<Whitening>(ErrorCode::NonFiniteResult,
                              "whitening null residual is not representable");
  }

  Matrix map(n, n, 0.0);
  Matrix support(n, rank, 0.0);
  Extent destination = 0U;
  for (Index k = 0U; k < n; ++k) {
    const Scalar eigenvalue = eigen.eigenvalues().at(k);
    if (!(eigenvalue > spec.value().cutoffScaled)) {
      continue;
    }
    const long double originalEigenvalue =
        static_cast<long double>(spec.value().scale) * eigenvalue;
    const long double inverseRoot = 1.0L / std::sqrt(originalEigenvalue);
    for (Index row = 0U; row < n; ++row) {
      support.at(row, destination) = eigen.eigenvectors().at(row, k);
      for (Index column = row; column < n; ++column) {
        const long double next =
            static_cast<long double>(map.at(row, column)) +
            static_cast<long double>(eigen.eigenvectors().at(row, k)) *
                eigen.eigenvectors().at(column, k) * inverseRoot;
        if (!std::isfinite(next) ||
            std::abs(next) > std::numeric_limits<Scalar>::max()) {
          return failure<Whitening>(ErrorCode::NonFiniteResult,
                                    "whitening map is not representable");
        }
        map.at(row, column) = static_cast<Scalar>(next);
        map.at(column, row) = map.at(row, column);
      }
    }
    ++destination;
  }
  Vector whitened(n, 0.0);
  for (Index row = 0U; row < n; ++row) {
    long double total = 0.0L;
    for (Index column = 0U; column < n; ++column) {
      total += static_cast<long double>(map.at(row, column)) * value.at(column);
    }
    if (!std::isfinite(total) ||
        std::abs(total) > std::numeric_limits<Scalar>::max()) {
      return failure<Whitening>(ErrorCode::NonFiniteResult,
                                "whitened vector is not representable");
    }
    whitened.at(row) = static_cast<Scalar>(total);
  }
  const long double cutoff =
      std::max(static_cast<long double>(policy.rank_abs),
               static_cast<long double>(policy.rank_rel) * spec.value().scale *
                   maximumPositive);
  const Scalar reportedCutoff = cutoff > std::numeric_limits<Scalar>::max()
                                    ? std::numeric_limits<Scalar>::infinity()
                                    : static_cast<Scalar>(cutoff);
  return Result<Whitening>::success(
      Whitening{std::move(map), std::move(whitened), rank, reportedCutoff,
                static_cast<Scalar>(nullResidual), std::move(support)});
}

Result<Scalar> mahalanobisSquared(const Whitening& factor,
                                  ConstVectorView value) {
  const Extent n = factor.map.rows();
  if (factor.map.columns() != n || value.size() != n ||
      factor.support_basis.rows() != n ||
      factor.support_basis.columns() != factor.rank || factor.rank > n) {
    return failure<Scalar>(ErrorCode::DimensionMismatch,
                           "Mahalanobis factor and vector dimensions differ");
  }
  if (!finite(factor.map.view()) || !finite(factor.support_basis.view()) ||
      !finite(value)) {
    return failure<Scalar>(ErrorCode::NonFiniteInput,
                           "Mahalanobis input contains a non-finite value");
  }
  long double inputNorm = 0.0L;
  for (Index index = 0U; index < n; ++index) {
    inputNorm =
        std::hypot(inputNorm, static_cast<long double>(value.at(index)));
  }
  if (factor.rank == 0U) {
    for (Index index = 0U; index < n; ++index) {
      if (value.at(index) != 0.0) {
        return failure<Scalar>(
            ErrorCode::InvalidDomain,
            "zero-rank Mahalanobis factor requires the exact zero vector");
      }
    }
  } else if (factor.rank < n) {
    long double residualNorm = 0.0L;
    for (Index row = 0U; row < n; ++row) {
      long double projection = 0.0L;
      for (Index k = 0U; k < factor.rank; ++k) {
        long double coefficient = 0.0L;
        for (Index column = 0U; column < n; ++column) {
          coefficient +=
              static_cast<long double>(factor.support_basis.at(column, k)) *
              value.at(column);
        }
        projection +=
            static_cast<long double>(factor.support_basis.at(row, k)) *
            coefficient;
      }
      residualNorm = std::hypot(
          residualNorm, static_cast<long double>(value.at(row)) - projection);
    }
    if (residualNorm >
        static_cast<long double>(symmetryAllowance(n)) * inputNorm) {
      return failure<Scalar>(
          ErrorCode::InvalidDomain,
          "Mahalanobis vector has a component outside factor support");
    }
  }
  long double whitenedNorm = 0.0L;
  for (Index row = 0U; row < n; ++row) {
    long double component = 0.0L;
    for (Index column = 0U; column < n; ++column) {
      component += static_cast<long double>(factor.map.at(row, column)) *
                   value.at(column);
    }
    if (!std::isfinite(component)) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "Mahalanobis whitening is not finite");
    }
    whitenedNorm = std::hypot(whitenedNorm, component);
  }
  const long double squared = whitenedNorm * whitenedNorm;
  if (!std::isfinite(squared) || squared > std::numeric_limits<Scalar>::max()) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "squared Mahalanobis distance is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(squared));
}

} // namespace vss::math::covariance
