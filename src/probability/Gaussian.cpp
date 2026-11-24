#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <string>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/probability/DistributionMoments.hpp>
#include <vss/math/probability/Gaussian.hpp>
#include <vss/math/probability/Special.hpp>

namespace vss::math::probability {

struct JointDistribution::State final {
  JointFamily family{JointFamily::normal};
  Vector location;
  Matrix scale;
  Matrix basis;
  Vector eigenvalues;
  Index rank{};
  Scalar degreesOfFreedom{};
  Scalar logPseudoDeterminant{};
};

namespace {

bool withinLimits(Extent dimension, Limits limits) {
  if (dimension == 0U) {
    return limits.bytes >= sizeof(Scalar);
  }
  if (dimension > std::numeric_limits<Extent>::max() / dimension) {
    return false;
  }
  const Extent square = dimension * dimension;
  if (square > (std::numeric_limits<Extent>::max() - 2U * dimension) / 2U) {
    return false;
  }
  const Extent elements = 2U * square + 2U * dimension;
  return elements <= limits.elements &&
         elements <= limits.bytes / sizeof(Scalar);
}

Result<JointDistribution> failed(ErrorCode code, const char* message) {
  return Result<JointDistribution>::failure(code, message);
}

linear::SymmetricEigenOptions eigenOptions(SolvePolicy policy) {
  const auto sweeps =
      std::clamp<Index>(policy.iteration.max_iterations, 1U, 1024U);
  return {.rankTolerance = {.absolute = policy.rank_abs,
                            .relative = policy.rank_rel},
          .symmetryTolerance = {.absolute = policy.iteration.abs,
                                .relative = policy.iteration.rel},
          .maximumSweeps = sweeps};
}

Result<std::shared_ptr<const JointDistribution::State>>
buildState(JointFamily family, ConstVectorView location, ConstMatrixView scale,
           Scalar degreesOfFreedom, SolvePolicy policy, Limits limits) {
  const Extent dimension = location.size();
  if (scale.rows() != dimension || scale.columns() != dimension) {
    return Result<std::shared_ptr<const JointDistribution::State>>::failure(
        ErrorCode::DimensionMismatch,
        "joint location and scale dimensions must agree");
  }
  if (!withinLimits(dimension, limits)) {
    return Result<std::shared_ptr<const JointDistribution::State>>::failure(
        ErrorCode::ResourceLimit, "joint law exceeds configured limits");
  }
  if (family == JointFamily::student_t &&
      (!std::isfinite(degreesOfFreedom) || degreesOfFreedom <= 0.0)) {
    return Result<std::shared_ptr<const JointDistribution::State>>::failure(
        ErrorCode::InvalidDomain,
        "multivariate Student-t degrees of freedom must be positive");
  }
  for (Index i = 0U; i < dimension; ++i) {
    if (!std::isfinite(location.at(i))) {
      return Result<std::shared_ptr<const JointDistribution::State>>::failure(
          ErrorCode::NonFiniteInput, "joint location must be finite");
    }
    for (Index j = 0U; j < dimension; ++j) {
      if (!std::isfinite(scale.at(i, j))) {
        return Result<std::shared_ptr<const JointDistribution::State>>::failure(
            ErrorCode::NonFiniteInput, "joint scale must be finite");
      }
    }
  }
  auto state = std::make_shared<JointDistribution::State>();
  state->family = family;
  state->location = Vector(dimension);
  state->scale = Matrix(dimension, dimension);
  for (Index i = 0U; i < dimension; ++i) {
    state->location.at(i) = location.at(i);
    for (Index j = 0U; j < dimension; ++j) {
      state->scale.at(i, j) = scale.at(i, j);
    }
  }
  state->degreesOfFreedom = degreesOfFreedom;
  if (dimension == 0U) {
    state->basis = Matrix(0U, 0U);
    state->eigenvalues = Vector(0U);
    state->rank = 0U;
    state->logPseudoDeterminant = 0.0;
    return Result<std::shared_ptr<const JointDistribution::State>>::success(
        std::move(state));
  }
  auto decomposition =
      linear::decomposeSymmetricEigen(scale, eigenOptions(policy));
  if (!decomposition) {
    return Result<std::shared_ptr<const JointDistribution::State>>::failure(
        decomposition.status().code(),
        std::string(decomposition.status().message()));
  }
  const auto& values = decomposition.value().eigenvalues();
  const auto& vectors = decomposition.value().eigenvectors();
  const Scalar largest =
      std::max(0.0, decomposition.value().maximumEigenvalue());
  const Scalar cutoff = std::max(policy.rank_abs, policy.rank_rel * largest);
  Index rank = 0U;
  Scalar logDet = 0.0;
  for (Index i = 0U; i < dimension; ++i) {
    if (values.at(i) < -cutoff) {
      return Result<std::shared_ptr<const JointDistribution::State>>::failure(
          ErrorCode::NotPositiveSemidefinite,
          "joint scale matrix must be positive semidefinite");
    }
    if (values.at(i) > cutoff) {
      ++rank;
      logDet += std::log(values.at(i));
    }
  }
  if (rank < dimension && policy.rank == RankPolicy::reject) {
    return Result<std::shared_ptr<const JointDistribution::State>>::failure(
        ErrorCode::RankDeficient,
        "joint scale is rank deficient under the requested rank policy");
  }
  state->basis = Matrix(dimension, rank, 0.0);
  state->eigenvalues = Vector(rank, 0.0);
  Index destination = 0U;
  for (Index source = 0U; source < dimension; ++source) {
    if (values.at(source) <= cutoff) {
      continue;
    }
    state->eigenvalues.at(destination) = values.at(source);
    for (Index row = 0U; row < dimension; ++row) {
      state->basis.at(row, destination) = vectors.at(row, source);
    }
    ++destination;
  }
  state->rank = rank;
  state->logPseudoDeterminant = logDet;
  return Result<std::shared_ptr<const JointDistribution::State>>::success(
      std::move(state));
}

} // namespace

namespace detail {

class JointDistributionFactory final {
public:
  static Result<JointDistribution>
  create(JointFamily family, ConstVectorView location, ConstMatrixView scale,
         Scalar degreesOfFreedom, SolvePolicy policy, Limits limits) {
    auto state =
        buildState(family, location, scale, degreesOfFreedom, policy, limits);
    if (!state) {
      return Result<JointDistribution>::failure(
          state.status().code(), std::string(state.status().message()));
    }
    return Result<JointDistribution>::success(
        JointDistribution(std::move(state).value()));
  }
};

class JointDistributionEngine final {
public:
  static Result<Scalar> logLikelihood(const JointDistribution& law,
                                      ConstVectorView value,
                                      ProbabilityPolicy policy);
  static Result<JointDistribution>
  conditional(const JointDistribution& law, std::span<const Index> keep,
              std::span<const Index> observed, ConstVectorView values,
              SolvePolicy policy, Limits limits);
  static Result<Vector> sample(const JointDistribution& law, Rng& rng,
                               ProbabilityPolicy policy, Limits limits);
};

class JointMomentsEngine final {
public:
  static Result<Vector> mean(const JointDistribution& law, Limits limits);
  static Result<Matrix> covariance(const JointDistribution& law, Limits limits);
};

} // namespace detail

JointDistribution::JointDistribution(
    std::shared_ptr<const State> state) noexcept
    : state_(std::move(state)) {}

JointFamily JointDistribution::family() const noexcept {
  return state_ ? state_->family : JointFamily::normal;
}
Extent JointDistribution::dimension() const noexcept {
  return state_ ? state_->location.size() : 0U;
}
Index JointDistribution::supportRank() const noexcept {
  return state_ ? state_->rank : 0U;
}
Scalar JointDistribution::degreesOfFreedom() const noexcept {
  return state_ ? state_->degreesOfFreedom : 0.0;
}
ConstVectorView JointDistribution::location() const noexcept {
  static const Vector empty;
  return state_ ? state_->location.view() : empty.view();
}
ConstMatrixView JointDistribution::scale() const noexcept {
  static const Matrix empty;
  return state_ ? state_->scale.view() : empty.view();
}

Result<JointDistribution> multivariateNormal(ConstVectorView location,
                                             ConstMatrixView covariance,
                                             SolvePolicy policy,
                                             Limits limits) {
  return detail::JointDistributionFactory::create(
      JointFamily::normal, location, covariance, 0.0, policy, limits);
}

Result<JointDistribution> multivariateStudentT(ConstVectorView location,
                                               ConstMatrixView scale,
                                               Scalar degreesOfFreedom,
                                               SolvePolicy policy,
                                               Limits limits) {
  return detail::JointDistributionFactory::create(
      JointFamily::student_t, location, scale, degreesOfFreedom, policy,
      limits);
}

Result<Scalar>
detail::JointDistributionEngine::logLikelihood(const JointDistribution& law,
                                               ConstVectorView value,
                                               ProbabilityPolicy policy) {
  if (!law.state_) {
    return Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                   "joint distribution is moved from");
  }
  const auto& state = *law.state_;
  if (value.size() != state.location.size()) {
    return Result<Scalar>::failure(ErrorCode::DimensionMismatch,
                                   "joint sample dimension does not match law");
  }
  if (!std::isfinite(policy.iteration.abs) ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.abs <= 0.0 ||
      !std::isfinite(policy.abs_log_error) || policy.iteration.rel <= 0.0 ||
      policy.abs_log_error <= 0.0 || policy.iteration.max_evaluations == 0U) {
    return Result<Scalar>::failure(ErrorCode::InvalidTolerance,
                                   "joint likelihood policy is invalid");
  }
  Vector residual(state.location.size(), 0.0);
  long double norm = 0.0L;
  for (Index i = 0U; i < value.size(); ++i) {
    if (!std::isfinite(value.at(i))) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteInput,
                                     "joint sample must be finite");
    }
    residual.at(i) = value.at(i) - state.location.at(i);
    norm = std::hypot(norm, static_cast<long double>(residual.at(i)));
  }
  long double q = 0.0L;
  long double nullResidual = 0.0L;
  for (Index row = 0U; row < state.location.size(); ++row) {
    long double projection = 0.0L;
    for (Index k = 0U; k < state.rank; ++k) {
      long double coefficient = 0.0L;
      for (Index column = 0U; column < state.location.size(); ++column) {
        coefficient += static_cast<long double>(state.basis.at(column, k)) *
                       residual.at(column);
      }
      if (row == 0U) {
        q += coefficient * coefficient / state.eigenvalues.at(k);
      }
      projection +=
          static_cast<long double>(state.basis.at(row, k)) * coefficient;
    }
    nullResidual = std::hypot(
        nullResidual, static_cast<long double>(residual.at(row)) - projection);
  }
  const long double supportTolerance =
      policy.iteration.abs + policy.iteration.rel * norm;
  if (nullResidual > supportTolerance) {
    return Result<Scalar>::success(-std::numeric_limits<Scalar>::infinity());
  }
  if (state.rank == 0U) {
    return Result<Scalar>::success(0.0);
  }
  long double result = 0.0L;
  if (state.family == JointFamily::normal) {
    result =
        -0.5L * state.rank * std::log(2.0L * std::numbers::pi_v<long double>) -
        0.5L * state.logPseudoDeterminant - 0.5L * q;
  } else {
    if (state.degreesOfFreedom > std::numeric_limits<Scalar>::max()) {
      return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                     "Student-t degrees of freedom overflow");
    }
    const Scalar rankValue = static_cast<Scalar>(state.rank);
    auto first = logGamma((state.degreesOfFreedom + rankValue) / 2.0, policy);
    auto second = logGamma(state.degreesOfFreedom / 2.0, policy);
    if (!first || !second) {
      const Status& status = !first ? first.status() : second.status();
      return Result<Scalar>::failure(status.code(),
                                     std::string(status.message()));
    }
    result = first.value() - second.value() -
             0.5L * state.rank *
                 std::log(state.degreesOfFreedom * std::numbers::pi_v<Scalar>) -
             0.5L * state.logPseudoDeterminant -
             0.5L * (state.degreesOfFreedom + rankValue) *
                 std::log1p(q / state.degreesOfFreedom);
  }
  if (!std::isfinite(result) ||
      std::abs(result) > std::numeric_limits<Scalar>::max()) {
    return Result<Scalar>::failure(ErrorCode::NonFiniteResult,
                                   "joint log likelihood is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(result));
}

Result<Vector> detail::JointMomentsEngine::mean(const JointDistribution& law,
                                                Limits limits) {
  if (!law.state_) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "joint distribution is moved from");
  }
  if (law.state_->location.size() > limits.elements ||
      law.state_->location.size() > limits.bytes / sizeof(Scalar)) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "joint mean exceeds configured limits");
  }
  return Result<Vector>::success(law.state_->location);
}

Result<Matrix>
detail::JointMomentsEngine::covariance(const JointDistribution& law,
                                       Limits limits) {
  if (!law.state_) {
    return Result<Matrix>::failure(ErrorCode::InvalidDomain,
                                   "joint distribution is moved from");
  }
  const auto& state = *law.state_;
  if (state.family == JointFamily::student_t && state.degreesOfFreedom <= 2.0) {
    return Result<Matrix>::failure(ErrorCode::InvalidDomain,
                                   "Student-t covariance does not exist");
  }
  if (state.scale.size() > limits.elements ||
      state.scale.size() > limits.bytes / sizeof(Scalar)) {
    return Result<Matrix>::failure(
        ErrorCode::ResourceLimit, "joint covariance exceeds configured limits");
  }
  Matrix result(state.scale);
  if (state.family == JointFamily::student_t) {
    const Scalar factor =
        state.degreesOfFreedom / (state.degreesOfFreedom - 2.0);
    for (Index row = 0U; row < result.rows(); ++row) {
      for (Index column = 0U; column < result.columns(); ++column) {
        const long double next =
            static_cast<long double>(result.at(row, column)) * factor;
        if (!std::isfinite(next) ||
            std::abs(next) > std::numeric_limits<Scalar>::max()) {
          return Result<Matrix>::failure(ErrorCode::NonFiniteResult,
                                         "Student-t covariance overflowed");
        }
        result.at(row, column) = static_cast<Scalar>(next);
      }
    }
  }
  return Result<Matrix>::success(std::move(result));
}

Result<Vector>
detail::JointDistributionEngine::sample(const JointDistribution& law, Rng& rng,
                                        ProbabilityPolicy policy,
                                        Limits limits) {
  if (!law.state_) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "joint distribution is moved from");
  }
  const auto& state = *law.state_;
  const std::uint64_t words = state.rank;
  if (policy.max_draws < words) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "joint normal sample exceeds draw limit");
  }
  if (state.location.size() > limits.elements ||
      state.location.size() > limits.bytes / sizeof(Scalar)) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "joint sample exceeds output limits");
  }
  Vector standard(state.rank, 0.0);
  if (state.rank > 0U) {
    auto standardNormal = normal(0.0, 1.0);
    if (!standardNormal) {
      return Result<Vector>::failure(
          standardNormal.status().code(),
          std::string(standardNormal.status().message()));
    }
    for (Index index = 0U; index < state.rank; ++index) {
      auto draw =
          vss::math::probability::sample(standardNormal.value(), rng, policy);
      if (!draw) {
        return Result<Vector>::failure(draw.status().code(),
                                       std::string(draw.status().message()));
      }
      standard.at(index) = draw.value();
    }
  }
  Vector result(state.location);
  for (Index row = 0U; row < state.location.size(); ++row) {
    long double value = state.location.at(row);
    for (Index k = 0U; k < state.rank; ++k) {
      value += static_cast<long double>(state.basis.at(row, k)) *
               std::sqrt(state.eigenvalues.at(k)) * standard.at(k);
    }
    if (!std::isfinite(value) ||
        std::abs(value) > std::numeric_limits<Scalar>::max()) {
      return Result<Vector>::failure(
          ErrorCode::NonFiniteResult,
          "joint normal sample is not representable");
    }
    result.at(row) = static_cast<Scalar>(value);
  }
  return Result<Vector>::success(std::move(result));
}

Result<Scalar> logLikelihood(const JointDistribution& law,
                             ConstVectorView jointDraw,
                             ProbabilityPolicy policy) {
  return detail::JointDistributionEngine::logLikelihood(law, jointDraw, policy);
}

Result<Vector> mean(const JointDistribution& law, ProbabilityPolicy,
                    Limits limits) {
  return detail::JointMomentsEngine::mean(law, limits);
}

Result<Matrix> covariance(const JointDistribution& law, ProbabilityPolicy,
                          Limits limits) {
  return detail::JointMomentsEngine::covariance(law, limits);
}

Result<Vector> multivariateNormalSample(const JointDistribution& law, Rng& rng,
                                        ProbabilityPolicy policy,
                                        Limits limits) {
  if (law.family() != JointFamily::normal) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "normal sampling requires a normal law");
  }
  return detail::JointDistributionEngine::sample(law, rng, policy, limits);
}

Result<JointDistribution>
conditionalGaussian(const JointDistribution& law, std::span<const Index> keep,
                    std::span<const Index> observed, ConstVectorView values,
                    SolvePolicy policy, Limits limits) {
  if (!law.state_) {
    return failed(ErrorCode::InvalidDomain, "joint distribution is moved from");
  }
  if (law.state_->family != JointFamily::normal) {
    return failed(ErrorCode::InvalidDomain,
                  "Gaussian conditioning requires a normal joint law");
  }
  const Index dimension = law.state_->location.size();
  if (values.size() != observed.size()) {
    return failed(ErrorCode::DimensionMismatch,
                  "observed coordinate values have the wrong length");
  }
  if (keep.size() > limits.elements || observed.size() > limits.elements ||
      !withinLimits(dimension, limits)) {
    return failed(ErrorCode::ResourceLimit,
                  "conditional Gaussian exceeds configured limits");
  }
  std::vector<bool> seen(dimension, false);
  for (const Index index : keep) {
    if (index >= dimension) {
      return failed(ErrorCode::IndexOutOfRange, "kept index is out of range");
    }
    if (seen[index]) {
      return failed(ErrorCode::InvalidDomain, "kept indices must be unique");
    }
    seen[index] = true;
  }
  for (const Index index : observed) {
    if (index >= dimension) {
      return failed(ErrorCode::IndexOutOfRange,
                    "observed index is out of range");
    }
    if (seen[index]) {
      return failed(ErrorCode::InvalidDomain,
                    "kept and observed indices must be unique and disjoint");
    }
    seen[index] = true;
  }
  for (Index i = 0U; i < values.size(); ++i) {
    if (!std::isfinite(values.at(i))) {
      return failed(ErrorCode::NonFiniteInput,
                    "observed values must be finite");
    }
  }
  const auto& source = *law.state_;
  Vector conditionalMean(keep.size(), 0.0);
  Matrix conditionalScale(keep.size(), keep.size(), 0.0);
  for (Index i = 0U; i < keep.size(); ++i) {
    conditionalMean.at(i) = source.location.at(keep[i]);
    for (Index j = 0U; j < keep.size(); ++j) {
      conditionalScale.at(i, j) = source.scale.at(keep[i], keep[j]);
    }
  }
  if (!observed.empty()) {
    Matrix observedScale(observed.size(), observed.size(), 0.0);
    Vector residual(observed.size(), 0.0);
    for (Index i = 0U; i < observed.size(); ++i) {
      residual.at(i) = values.at(i) - source.location.at(observed[i]);
      for (Index j = 0U; j < observed.size(); ++j) {
        observedScale.at(i, j) = source.scale.at(observed[i], observed[j]);
      }
    }
    auto eigen = linear::decomposeSymmetricEigen(observedScale.view(),
                                                 eigenOptions(policy));
    if (!eigen) {
      return Result<JointDistribution>::failure(
          eigen.status().code(), std::string(eigen.status().message()));
    }
    const auto eigenvalues = eigen.value().eigenvalues();
    const auto eigenvectors = eigen.value().eigenvectors();
    const Scalar cutoff = std::max(
        policy.rank_abs,
        policy.rank_rel * std::max(0.0, eigen.value().maximumEigenvalue()));
    Index observedRank = 0U;
    for (Index k = 0U; k < observed.size(); ++k) {
      if (eigenvalues.at(k) > cutoff) {
        ++observedRank;
      }
    }
    if (observedRank < observed.size() && policy.rank == RankPolicy::reject) {
      return failed(ErrorCode::RankDeficient,
                    "observed covariance is rank deficient under policy");
    }
    Vector solved(observed.size(), 0.0);
    long double residualNorm = 0.0L;
    long double nullNorm = 0.0L;
    for (Index i = 0U; i < observed.size(); ++i) {
      residualNorm =
          std::hypot(residualNorm, static_cast<long double>(residual.at(i)));
    }
    for (Index k = 0U; k < observed.size(); ++k) {
      const Scalar eigenvalue = eigenvalues.at(k);
      if (eigenvalue < -cutoff) {
        return failed(ErrorCode::NotPositiveSemidefinite,
                      "observed covariance is not positive semidefinite");
      }
      long double coefficient = 0.0L;
      for (Index row = 0U; row < observed.size(); ++row) {
        coefficient += static_cast<long double>(eigenvectors.at(row, k)) *
                       residual.at(row);
      }
      if (eigenvalue > cutoff) {
        coefficient /= eigenvalue;
        for (Index row = 0U; row < observed.size(); ++row) {
          solved.at(row) += static_cast<Scalar>(
              static_cast<long double>(eigenvectors.at(row, k)) * coefficient);
        }
      } else {
        nullNorm = std::hypot(nullNorm, coefficient);
      }
    }
    if (nullNorm > policy.iteration.abs + policy.iteration.rel * residualNorm) {
      return failed(ErrorCode::InvalidDomain,
                    "observed values are inconsistent with singular support");
    }
    Matrix observedPseudoInverse(observed.size(), observed.size(), 0.0);
    for (Index k = 0U; k < observed.size(); ++k) {
      if (eigenvalues.at(k) <= cutoff) {
        continue;
      }
      for (Index row = 0U; row < observed.size(); ++row) {
        for (Index column = 0U; column < observed.size(); ++column) {
          observedPseudoInverse.at(row, column) += eigenvectors.at(row, k) *
                                                   eigenvectors.at(column, k) /
                                                   eigenvalues.at(k);
        }
      }
    }
    Matrix solvedCross(observed.size(), keep.size(), 0.0);
    for (Index row = 0U; row < observed.size(); ++row) {
      for (Index column = 0U; column < keep.size(); ++column) {
        long double value = 0.0L;
        for (Index inner = 0U; inner < observed.size(); ++inner) {
          value +=
              static_cast<long double>(observedPseudoInverse.at(row, inner)) *
              source.scale.at(observed[inner], keep[column]);
        }
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<Scalar>::max()) {
          return failed(ErrorCode::NonFiniteResult,
                        "conditional cross solve is not representable");
        }
        solvedCross.at(row, column) = static_cast<Scalar>(value);
      }
    }
    for (Index i = 0U; i < keep.size(); ++i) {
      long double shift = 0.0L;
      for (Index j = 0U; j < observed.size(); ++j) {
        shift +=
            static_cast<long double>(source.scale.at(keep[i], observed[j])) *
            solved.at(j);
      }
      const long double next = conditionalMean.at(i) + shift;
      if (!std::isfinite(next) ||
          std::abs(next) > std::numeric_limits<Scalar>::max()) {
        return failed(ErrorCode::NonFiniteResult,
                      "conditional mean is not representable");
      }
      conditionalMean.at(i) = static_cast<Scalar>(next);
      for (Index j = 0U; j < keep.size(); ++j) {
        long double reduction = 0.0L;
        for (Index a = 0U; a < observed.size(); ++a) {
          reduction +=
              static_cast<long double>(source.scale.at(keep[i], observed[a])) *
              solvedCross.at(a, j);
        }
        const long double updatedScale = conditionalScale.at(i, j) - reduction;
        if (!std::isfinite(updatedScale) ||
            std::abs(updatedScale) > std::numeric_limits<Scalar>::max()) {
          return failed(ErrorCode::NonFiniteResult,
                        "conditional scale is not representable");
        }
        conditionalScale.at(i, j) = static_cast<Scalar>(updatedScale);
      }
    }
  }
  auto locationView = conditionalMean.view();
  auto scaleView = conditionalScale.view();
  return detail::JointDistributionFactory::create(
      JointFamily::normal, locationView, scaleView, 0.0, policy, limits);
}

} // namespace vss::math::probability
