#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/covariance/Structured.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {
namespace {
template <class T> Result<T> fail(ErrorCode c, const char* m) {
  return Result<T>::failure(c, m);
}
bool mul(Index a, Index b, Index& out) {
  if (a && b > std::numeric_limits<Index>::max() / a)
    return false;
  out = a * b;
  return true;
}
bool add(Index a, Index b, Index& out) {
  if (b > std::numeric_limits<Index>::max() - a)
    return false;
  out = a + b;
  return true;
}
bool validPolicy(SolvePolicy p) {
  return std::isfinite(p.rank_abs) && p.rank_abs >= 0.0 &&
         std::isfinite(p.rank_rel) && p.rank_rel >= 0.0 &&
         std::isfinite(p.reciprocal_condition_floor) &&
         p.reciprocal_condition_floor >= 0.0 &&
         p.reciprocal_condition_floor <= 1.0 &&
         (p.rank == RankPolicy::reject ||
          p.rank == RankPolicy::supported_subspace ||
          p.rank == RankPolicy::minimum_norm);
}
bool finite(ConstVectorView v) {
  for (Index i = 0; i < v.size(); ++i)
    if (!std::isfinite(v.at(i)))
      return false;
  return true;
}
bool finite(ConstMatrixView m) {
  for (Index i = 0; i < m.rows(); ++i)
    for (Index j = 0; j < m.columns(); ++j)
      if (!std::isfinite(m.at(i, j)))
        return false;
  return true;
}
bool empty(ConstVectorView v) { return v.size() == 0; }

Result<Vector> applyRaw(const StructuredCovariance& s, ConstVectorView x,
                        Limits lim) {
  if (static_cast<unsigned>(s.kind) >
      static_cast<unsigned>(StructureKind::gaussian_kernel))
    return fail<Vector>(ErrorCode::InvalidDomain,
                        "unknown covariance structure");
  if (x.size() != s.dimension)
    return fail<Vector>(ErrorCode::DimensionMismatch,
                        "structured covariance vector dimension mismatch");
  if (!finite(x))
    return fail<Vector>(ErrorCode::NonFiniteInput,
                        "structured covariance vector is non-finite");
  if (s.dimension > lim.elements || s.dimension > lim.bytes / sizeof(Scalar))
    return fail<Vector>(ErrorCode::ResourceLimit,
                        "structured covariance output exceeds limits");
  if (s.kind == StructureKind::kronecker) {
    if (!s.diagonal_or_lags.empty() || s.low_rank_or_points || s.kernel_scale ||
        s.kernel_length)
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "inactive Kronecker covariance fields are populated");
    Index dimension = 1U;
    std::vector<Index> extents;
    extents.reserve(s.factors.view().size());
    for (const Matrix& factor : s.factors.view()) {
      if (factor.rows() != factor.columns() || !finite(factor.view()) ||
          !mul(dimension, factor.rows(), dimension))
        return fail<Vector>(ErrorCode::InvalidDomain,
                            "invalid Kronecker covariance payload");
      extents.push_back(factor.rows());
    }
    if (dimension != s.dimension ||
        (s.factors.view().empty() && s.dimension != 1U))
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "Kronecker factors disagree with stored dimension");
    Index scratch = 0U;
    if (!mul(s.dimension, 2U, scratch) || scratch > lim.elements ||
        scratch > lim.bytes / sizeof(Scalar))
      return fail<Vector>(ErrorCode::ResourceLimit,
                          "Kronecker mode-product buffers exceed limits");
    Vector current(s.dimension, 0.0), next(s.dimension, 0.0);
    for (Index i = 0U; i < s.dimension; ++i)
      current.data()[i] = x.at(i);
    Index suffix_stride = s.dimension;
    for (Index axis = 0U; axis < extents.size(); ++axis) {
      const Index extent = extents[axis];
      if (extent == 0U)
        return Result<Vector>::success(Vector{});
      suffix_stride /= extent;
      const Index outer_count = s.dimension / (extent * suffix_stride);
      const Matrix& factor = s.factors.view()[axis];
      for (Index outer = 0U; outer < outer_count; ++outer) {
        const Index base = outer * extent * suffix_stride;
        for (Index row = 0U; row < extent; ++row) {
          for (Index inner = 0U; inner < suffix_stride; ++inner) {
            long double sum = 0.0L;
            for (Index column = 0U; column < extent; ++column)
              sum += static_cast<long double>(factor.at(row, column)) *
                     current.at(base + column * suffix_stride + inner);
            if (!std::isfinite(sum) ||
                std::abs(sum) > std::numeric_limits<Scalar>::max())
              return fail<Vector>(ErrorCode::NonFiniteResult,
                                  "Kronecker mode product is not representable");
            next.data()[base + row * suffix_stride + inner] =
                static_cast<Scalar>(sum);
          }
        }
      }
      std::swap(current, next);
    }
    return Result<Vector>::success(std::move(current));
  }
  Vector y(s.dimension, 0.0);
  switch (s.kind) {
  case StructureKind::diagonal_low_rank: {
    if (s.diagonal_or_lags.size() != s.dimension ||
        !finite(s.diagonal_or_lags.view()))
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "invalid diagonal covariance payload");
    for (Index i = 0; i < s.dimension; ++i)
      y.data()[i] = s.diagonal_or_lags.at(i) * x.at(i);
    if (s.low_rank_or_points) {
      const auto& u = *s.low_rank_or_points;
      if (u.rows() != s.dimension || !finite(u.view()))
        return fail<Vector>(ErrorCode::InvalidDomain,
                            "invalid low-rank covariance payload");
      for (Index j = 0; j < u.columns(); ++j) {
        long double dot = 0;
        for (Index i = 0; i < s.dimension; ++i)
          dot += static_cast<long double>(u.at(i, j)) * x.at(i);
        for (Index i = 0; i < s.dimension; ++i)
          y.data()[i] =
              static_cast<Scalar>(static_cast<long double>(y.at(i)) +
                                  static_cast<long double>(u.at(i, j)) * dot);
      }
    }
    break;
  }
  case StructureKind::toeplitz:
    if (s.diagonal_or_lags.size() != s.dimension ||
        !finite(s.diagonal_or_lags.view()))
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "invalid Toeplitz covariance payload");
    for (Index i = 0; i < s.dimension; ++i) {
      long double sum = 0;
      for (Index j = 0; j < s.dimension; ++j)
        sum += static_cast<long double>(
                   s.diagonal_or_lags.at(i > j ? i - j : j - i)) *
               x.at(j);
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max())
        return fail<Vector>(ErrorCode::NonFiniteResult,
                            "Toeplitz action is not representable");
      y.data()[i] = static_cast<Scalar>(sum);
    }
    break;
  case StructureKind::block: {
    Index off = 0;
    for (const auto& b : s.factors.view()) {
      if (b.rows() != b.columns() || b.rows() > s.dimension - off ||
          !finite(b.view()))
        return fail<Vector>(ErrorCode::InvalidDomain,
                            "invalid block covariance payload");
      for (Index i = 0; i < b.rows(); ++i) {
        long double sum = 0;
        for (Index j = 0; j < b.columns(); ++j)
          sum += static_cast<long double>(b.at(i, j)) * x.at(off + j);
        if (!std::isfinite(sum) ||
            std::abs(sum) > std::numeric_limits<Scalar>::max())
          return fail<Vector>(ErrorCode::NonFiniteResult,
                              "block action is not representable");
        y.data()[off + i] = static_cast<Scalar>(sum);
      }
      off += b.rows();
    }
    if (off != s.dimension)
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "block dimensions disagree with stored dimension");
    break;
  }
  case StructureKind::kronecker: {
    if (s.factors.view().empty()) {
      if (s.dimension != 1)
        return fail<Vector>(ErrorCode::InvalidDomain,
                            "empty Kronecker covariance must be scalar");
      y.data()[0] = x.at(0);
      break;
    }
    Index n = 1;
    for (const auto& f : s.factors.view()) {
      if (f.rows() != f.columns() || !finite(f.view()) || !mul(n, f.rows(), n))
        return fail<Vector>(ErrorCode::InvalidDomain,
                            "invalid Kronecker covariance payload");
    }
    if (n != s.dimension)
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "Kronecker factors disagree with stored dimension");
    for (Index row = 0; row < n; ++row) {
      long double sum = 0;
      for (Index col = 0; col < n; ++col) {
        long double a = 1;
        Index rr = row, cc = col;
        for (auto it = s.factors.view().rbegin(); it != s.factors.view().rend();
             ++it) {
          const Index d = it->rows();
          a *= it->at(rr % d, cc % d);
          rr /= d;
          cc /= d;
        }
        sum += a * x.at(col);
      }
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max())
        return fail<Vector>(ErrorCode::NonFiniteResult,
                            "Kronecker action is not representable");
      y.data()[row] = static_cast<Scalar>(sum);
    }
    break;
  }
  case StructureKind::gaussian_kernel: {
    if (!s.low_rank_or_points || !s.kernel_scale || !s.kernel_length ||
        s.low_rank_or_points->rows() != s.dimension ||
        s.low_rank_or_points->columns() == 0 || *s.kernel_scale < 0 ||
        !(*s.kernel_length > 0) || !std::isfinite(*s.kernel_scale) ||
        !std::isfinite(*s.kernel_length) ||
        !finite(s.low_rank_or_points->view()))
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "invalid Gaussian covariance payload");
    const auto& p = *s.low_rank_or_points;
    const long double ell = *s.kernel_length;
    for (Index i = 0; i < s.dimension; ++i) {
      long double sum = 0;
      for (Index j = 0; j < s.dimension; ++j) {
        long double d2 = 0;
        for (Index k = 0; k < p.columns(); ++k) {
          const long double d =
              static_cast<long double>(p.at(i, k)) - p.at(j, k);
          const long double z = d / ell;
          d2 += z * z;
        }
        sum += static_cast<long double>(*s.kernel_scale) *
               std::exp(-0.5L * d2) * x.at(j);
      }
      if (!std::isfinite(sum) ||
          std::abs(sum) > std::numeric_limits<Scalar>::max())
        return fail<Vector>(ErrorCode::NonFiniteResult,
                            "Gaussian covariance action is not representable");
      y.data()[i] = static_cast<Scalar>(sum);
    }
    break;
  }
  }
  for (Index i = 0; i < y.size(); ++i)
    if (!std::isfinite(y.at(i)))
      return fail<Vector>(ErrorCode::NonFiniteResult,
                          "structured covariance result is non-finite");
  return Result<Vector>::success(std::move(y));
}

Result<linear::LinearEstimate> pcgDiagonal(const StructuredCovariance& s,
                                           ConstMatrixView rhs,
                                           SolvePolicy policy, Limits limits) {
  if (s.kind != StructureKind::diagonal_low_rank || s.low_rank_or_points ||
      s.diagonal_or_lags.size() != s.dimension || policy.rank_rel != 0.0)
    return fail<linear::LinearEstimate>(
        ErrorCode::NoConvergence, "PCG currently requires diagonal covariance "
                                  "and an absolute rank cutoff");
  Index output = 0U, work = 0U;
  if (!mul(s.dimension, rhs.columns(), output) || !mul(s.dimension, 4U, work) ||
      !add(output, work, work) || !add(work, rhs.columns(), work) ||
      work > limits.elements || work > limits.bytes / sizeof(Scalar))
    return fail<linear::LinearEstimate>(ErrorCode::ResourceLimit,
                                        "PCG workspace exceeds limits");
  Scalar lo = std::numeric_limits<Scalar>::infinity(), hi = 0.0;
  for (Index i = 0; i < s.dimension; ++i) {
    const Scalar d = s.diagonal_or_lags.at(i);
    if (!std::isfinite(d))
      return fail<linear::LinearEstimate>(ErrorCode::NonFiniteInput,
                                          "diagonal covariance is non-finite");
    lo = std::min(lo, d);
    hi = std::max(hi, d);
  }
  if (!(lo > policy.rank_abs) || !(hi > 0.0) ||
      static_cast<long double>(lo) / hi < policy.reciprocal_condition_floor)
    return fail<linear::LinearEstimate>(
        ErrorCode::NoConvergence,
        "diagonal covariance does not certify retained full-rank SPD support");
  if (policy.iteration.max_iterations == 0U ||
      !std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0)
    return fail<linear::LinearEstimate>(ErrorCode::InvalidTolerance,
                                        "PCG convergence policy is invalid");
  linear::LinearEstimate out{.parameters =
                                 Matrix(s.dimension, rhs.columns(), 0.0),
                             .covariance = std::nullopt,
                             .identified_basis = std::nullopt,
                             .residual_norms = Vector(rhs.columns(), 0.0),
                             .rank = s.dimension,
                             .rank_cutoff = policy.rank_abs};
  std::uint64_t evaluations = 0U;
  const auto cap =
      std::min(limits.evaluations, policy.iteration.max_evaluations);
  for (Index c = 0; c < rhs.columns(); ++c) {
    long double norm_b = 0.0L;
    Vector residual(s.dimension, 0.0), z(s.dimension, 0.0),
        direction(s.dimension, 0.0), product(s.dimension, 0.0);
    long double rz = 0.0L;
    for (Index i = 0; i < s.dimension; ++i)
      norm_b = std::hypot(norm_b, static_cast<long double>(rhs.at(i, c)));
    const long double target =
        std::max(static_cast<long double>(policy.iteration.abs),
                 static_cast<long double>(policy.iteration.rel) * norm_b);
    for (Index i = 0; i < s.dimension; ++i) {
      residual.data()[i] = rhs.at(i, c);
      z.data()[i] = residual.at(i) / s.diagonal_or_lags.at(i);
      direction.data()[i] = z.at(i);
      rz += static_cast<long double>(residual.at(i)) * z.at(i);
    }
    long double true_norm = norm_b;
    bool converged = true_norm <= target;
    for (Index step = 0; !converged && step < policy.iteration.max_iterations;
         ++step) {
      if (evaluations >= cap)
        return fail<linear::LinearEstimate>(ErrorCode::ResourceLimit,
                                            "PCG evaluation budget exhausted");
      ++evaluations;
      long double curvature = 0.0L;
      for (Index i = 0; i < s.dimension; ++i) {
        product.data()[i] = s.diagonal_or_lags.at(i) * direction.at(i);
        curvature += static_cast<long double>(direction.at(i)) * product.at(i);
      }
      if (!(curvature > 0.0L) || !(rz > 0.0L))
        return fail<linear::LinearEstimate>(
            ErrorCode::NoConvergence, "PCG encountered non-positive curvature");
      const long double alpha = rz / curvature;
      for (Index i = 0; i < s.dimension; ++i) {
        const long double value =
            static_cast<long double>(out.parameters.at(i, c)) +
            alpha * direction.at(i);
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<Scalar>::max())
          return fail<linear::LinearEstimate>(
              ErrorCode::NonFiniteResult, "PCG iterate is not representable");
        out.parameters.at(i, c) = static_cast<Scalar>(value);
        residual.data()[i] = static_cast<Scalar>(
            static_cast<long double>(residual.at(i)) - alpha * product.at(i));
      }
      if (evaluations >= cap)
        return fail<linear::LinearEstimate>(
            ErrorCode::ResourceLimit, "PCG true-residual budget exhausted");
      ++evaluations;
      true_norm = 0.0L;
      long double next_rz = 0.0L;
      for (Index i = 0; i < s.dimension; ++i) {
        const long double r =
            static_cast<long double>(s.diagonal_or_lags.at(i)) *
                out.parameters.at(i, c) -
            rhs.at(i, c);
        true_norm = std::hypot(true_norm, r);
        z.data()[i] = residual.at(i) / s.diagonal_or_lags.at(i);
        next_rz += static_cast<long double>(residual.at(i)) * z.at(i);
      }
      if (true_norm <= target) {
        converged = true;
        break;
      }
      const long double beta = next_rz / rz;
      for (Index i = 0; i < s.dimension; ++i)
        direction.data()[i] =
            static_cast<Scalar>(z.at(i) + beta * direction.at(i));
      rz = next_rz;
    }
    if (!converged)
      return fail<linear::LinearEstimate>(
          ErrorCode::NoConvergence, "PCG did not satisfy its independently "
                                    "recomputed residual tolerance");
    out.residual_norms.data()[c] = static_cast<Scalar>(true_norm);
  }
  return Result<linear::LinearEstimate>::success(std::move(out));
}
} // namespace

Result<StructuredCovariance> structuredCovariance(const StructuredSpec& s,
                                                  SolvePolicy p, Limits lim) {
  if (!validPolicy(p))
    return fail<StructuredCovariance>(
        ErrorCode::InvalidTolerance, "structured covariance policy is invalid");
  if (static_cast<unsigned>(s.kind) >
      static_cast<unsigned>(StructureKind::gaussian_kernel))
    return fail<StructuredCovariance>(ErrorCode::InvalidDomain,
                                      "unknown covariance structure");
  Index n = s.diagonal_or_lags.size();
  Vector diag;
  std::optional<Matrix> auxiliary;
  std::vector<Matrix> factors;
  std::optional<Scalar> scale, length;
  const auto noExtra = [&] {
    return !s.low_rank_or_points && s.factors.empty() && !s.kernel_scale &&
           !s.kernel_length;
  };
  Index preflight = 0U;
  if (s.kind == StructureKind::diagonal_low_rank) {
    preflight = n;
    if (s.low_rank_or_points) {
      Index count = 0U;
      if (!mul(s.low_rank_or_points->rows(), s.low_rank_or_points->columns(),
               count) ||
          !add(preflight, count, preflight))
        return fail<StructuredCovariance>(ErrorCode::SizeOverflow,
                                          "low-rank payload size overflows");
    }
  } else if (s.kind == StructureKind::toeplitz) {
    Index square = 0U;
    if (!mul(n, n, square) || !mul(square, 6U, preflight) ||
        !add(preflight, n, preflight))
      return fail<StructuredCovariance>(ErrorCode::SizeOverflow,
                                        "Toeplitz certificate size overflows");
  } else if (s.kind == StructureKind::gaussian_kernel && s.low_rank_or_points) {
    if (!mul(s.low_rank_or_points->rows(), s.low_rank_or_points->columns(),
             preflight))
      return fail<StructuredCovariance>(ErrorCode::SizeOverflow,
                                        "Gaussian point storage overflows");
  } else if (s.kind == StructureKind::block ||
             s.kind == StructureKind::kronecker) {
    Index copied = 0U;
    Index largest_square = 0U;
    for (const Matrix& factor : s.factors) {
      Index square = 0U;
      if (!mul(factor.rows(), factor.rows(), square) ||
          !add(copied, factor.size(), copied))
        return fail<StructuredCovariance>(ErrorCode::SizeOverflow,
                                          "factor payload size overflows");
      largest_square = std::max(largest_square, square);
    }
    Index validation = 0U;
    if (!mul(largest_square, 6U, validation) ||
        !add(copied, validation, preflight))
      return fail<StructuredCovariance>(ErrorCode::SizeOverflow,
                                        "factor certificate size overflows");
  }
  if (preflight > lim.elements || preflight > lim.bytes / sizeof(Scalar))
    return fail<StructuredCovariance>(ErrorCode::ResourceLimit,
                                      "construction output and simultaneous "
                                      "certificate workspace exceed limits");
  if (s.kind == StructureKind::diagonal_low_rank) {
    if (!s.factors.empty() || s.kernel_scale || s.kernel_length)
      return fail<StructuredCovariance>(
          ErrorCode::InvalidDomain, "inactive covariance fields must be empty");
    if (!finite(s.diagonal_or_lags))
      return fail<StructuredCovariance>(ErrorCode::NonFiniteInput,
                                        "diagonal covariance is non-finite");
    for (Index i = 0; i < n; ++i)
      if (s.diagonal_or_lags.at(i) < 0)
        return fail<StructuredCovariance>(ErrorCode::NotPositiveSemidefinite,
                                          "diagonal covariance is negative");
    if (s.low_rank_or_points) {
      if (s.low_rank_or_points->rows() != n || !finite(*s.low_rank_or_points))
        return fail<StructuredCovariance>(ErrorCode::DimensionMismatch,
                                          "low-rank factor shape is invalid");
      auxiliary.emplace(n, s.low_rank_or_points->columns());
      for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < auxiliary->columns(); ++j)
          auxiliary->at(i, j) = s.low_rank_or_points->at(i, j);
    }
  } else if (s.kind == StructureKind::toeplitz) {
    if (!noExtra())
      return fail<StructuredCovariance>(
          ErrorCode::InvalidDomain, "inactive covariance fields must be empty");
    if (!finite(s.diagonal_or_lags))
      return fail<StructuredCovariance>(ErrorCode::NonFiniteInput,
                                        "Toeplitz lags are non-finite");
    if (n) {
      Index elems = 0;
      if (!mul(n, n, elems) || elems > lim.elements ||
          elems > lim.bytes / sizeof(Scalar))
        return fail<StructuredCovariance>(
            ErrorCode::ResourceLimit,
            "Toeplitz PSD certificate exceeds limits");
      Matrix dense(n, n, 0.0);
      for (Index i = 0; i < n; ++i)
        for (Index j = 0; j < n; ++j)
          dense.at(i, j) = s.diagonal_or_lags.at(i > j ? i - j : j - i);
      auto cert = analyzeCovariance(dense.view());
      if (!cert.hasValue())
        return fail<StructuredCovariance>(cert.status().code(),
                                          "Toeplitz PSD certification failed");
      if (cert.value().classification() ==
          CovarianceClassification::NotPositiveSemidefinite)
        return fail<StructuredCovariance>(ErrorCode::NotPositiveSemidefinite,
                                          "Toeplitz covariance is not PSD");
    }
  } else if (s.kind == StructureKind::block ||
             s.kind == StructureKind::kronecker) {
    if (!empty(s.diagonal_or_lags) || s.low_rank_or_points || s.kernel_scale ||
        s.kernel_length)
      return fail<StructuredCovariance>(
          ErrorCode::InvalidDomain, "inactive covariance fields must be empty");
    n = s.kind == StructureKind::block ? 0U : 1U;
    for (const auto& f : s.factors) {
      if (f.rows() != f.columns() || !finite(f.view()))
        return fail<StructuredCovariance>(
            ErrorCode::DimensionMismatch,
            "covariance factors must be finite and square");
      if (s.kind == StructureKind::block) {
        if (!add(n, f.rows(), n))
          return fail<StructuredCovariance>(
              ErrorCode::SizeOverflow, "block covariance dimension overflows");
      } else if (!mul(n, f.rows(), n))
        return fail<StructuredCovariance>(
            ErrorCode::SizeOverflow,
            "Kronecker covariance dimension overflows");
      if (f.rows()) {
        auto cert = analyzeCovariance(f.view());
        if (!cert.hasValue() ||
            cert.value().classification() ==
                CovarianceClassification::NotPositiveSemidefinite)
          return fail<StructuredCovariance>(ErrorCode::NotPositiveSemidefinite,
                                            "covariance factor is not PSD");
      }
      factors.push_back(f);
    }
  } else {
    if (!empty(s.diagonal_or_lags) || !s.low_rank_or_points ||
        !s.factors.empty() || !s.kernel_scale || !s.kernel_length)
      return fail<StructuredCovariance>(
          ErrorCode::InvalidDomain,
          "Gaussian kernel fields are incomplete or contain inactive values");
    if (s.low_rank_or_points->columns() == 0 ||
        !finite(*s.low_rank_or_points) || !std::isfinite(*s.kernel_scale) ||
        *s.kernel_scale < 0 || !std::isfinite(*s.kernel_length) ||
        !(*s.kernel_length > 0))
      return fail<StructuredCovariance>(
          ErrorCode::InvalidDomain, "Gaussian kernel parameters are invalid");
    n = s.low_rank_or_points->rows();
    auxiliary.emplace(n, s.low_rank_or_points->columns());
    for (Index i = 0; i < n; ++i)
      for (Index j = 0; j < auxiliary->columns(); ++j)
        auxiliary->at(i, j) = s.low_rank_or_points->at(i, j);
    scale = s.kernel_scale;
    length = s.kernel_length;
  }
  Index payload = 0;
  if (!add(n, auxiliary ? auxiliary->size() : 0, payload))
    return fail<StructuredCovariance>(
        ErrorCode::SizeOverflow, "structured covariance payload overflows");
  for (const auto& f : factors)
    if (!add(payload, f.size(), payload))
      return fail<StructuredCovariance>(
          ErrorCode::SizeOverflow, "structured covariance payload overflows");
  if (!add(payload,
           s.kind == StructureKind::diagonal_low_rank ||
                   s.kind == StructureKind::toeplitz
               ? n
               : 0,
           payload) ||
      payload > lim.elements || payload > lim.bytes / sizeof(Scalar))
    return fail<StructuredCovariance>(
        ErrorCode::ResourceLimit,
        "structured covariance payload exceeds limits");
  if (s.kind == StructureKind::diagonal_low_rank ||
      s.kind == StructureKind::toeplitz) {
    diag = Vector(n, 0.0);
    for (Index i = 0; i < n; ++i)
      diag.data()[i] = s.diagonal_or_lags.at(i);
  }
  auto copied = Array<Matrix>::copy(factors, lim);
  if (!copied.hasValue())
    return fail<StructuredCovariance>(copied.status().code(),
                                      "covariance factors exceed limits");
  return Result<StructuredCovariance>::success(
      StructuredCovariance{s.kind, n, std::move(diag), std::move(auxiliary),
                           std::move(copied).value(), scale, length});
}

Result<StructuredCovariance>
kroneckerCovariance(std::span<const Matrix> factors, SolvePolicy p,
                    Limits lim) {
  const Vector empty_vector;
  return structuredCovariance(
      StructuredSpec{StructureKind::kronecker, empty_vector.view(),
                     std::nullopt, factors, std::nullopt, std::nullopt},
      p, lim);
}

Result<Vector> applyStructuredCovariance(const StructuredCovariance& s,
                                         ConstVectorView x, Limits lim) {
  return applyRaw(s, x, lim);
}

Result<linear::LinearEstimate>
solveStructuredCovariance(const StructuredCovariance& s, ConstMatrixView rhs,
                          StructuredSolveKind mode, SolvePolicy p, Limits lim) {
  if (!validPolicy(p))
    return fail<linear::LinearEstimate>(
        ErrorCode::InvalidTolerance,
        "structured covariance solve policy is invalid");
  if (mode != StructuredSolveKind::factorized &&
      mode != StructuredSolveKind::iterative)
    return fail<linear::LinearEstimate>(ErrorCode::InvalidDomain,
                                        "unknown structured solve mode");
  if (rhs.rows() != s.dimension)
    return fail<linear::LinearEstimate>(
        ErrorCode::DimensionMismatch, "structured covariance RHS row mismatch");
  if (!finite(rhs))
    return fail<linear::LinearEstimate>(
        ErrorCode::NonFiniteInput, "structured covariance RHS is non-finite");
  if (mode == StructuredSolveKind::iterative)
    return pcgDiagonal(s, rhs, p, lim);
  if (s.dimension == 0U) {
    return Result<linear::LinearEstimate>::success(
        linear::LinearEstimate{.parameters = Matrix(0U, rhs.columns()),
                               .covariance = std::nullopt,
                               .identified_basis = std::nullopt,
                               .residual_norms = Vector(rhs.columns(), 0.0),
                               .rank = 0U,
                               .rank_cutoff = p.rank_abs});
  }
  Index square = 0, out = 0;
  if (!mul(s.dimension, s.dimension, square) ||
      !mul(s.dimension, rhs.columns(), out))
    return fail<linear::LinearEstimate>(ErrorCode::SizeOverflow,
                                        "structured solve size overflows");
  Index required = 0U, rows_work = 0U, bytes = 0U, index_bytes = 0U;
  if (!mul(square, 5U, required) || !add(required, out, required) ||
      !mul(s.dimension, 8U, rows_work) || !add(required, rows_work, required) ||
      !mul(rhs.columns(), 2U, rows_work) ||
      !add(required, rows_work, required) ||
      !mul(required, sizeof(Scalar), bytes) ||
      !mul(s.dimension, sizeof(Index), index_bytes) ||
      !add(bytes, index_bytes, bytes) || required > lim.elements ||
      bytes > lim.bytes)
    return fail<linear::LinearEstimate>(
        ErrorCode::ResourceLimit,
        "factorized structured solve and eigensolver peak exceed limits");
  Matrix dense(s.dimension, s.dimension, 0.0);
  Vector unit(s.dimension, 0.0);
  for (Index j = 0; j < s.dimension; ++j) {
    unit.data()[j] = 1.0;
    auto col = applyRaw(s, unit.view(),
                        Limits{.elements = s.dimension,
                               .bytes = s.dimension * sizeof(Scalar),
                               .evaluations = lim.evaluations});
    unit.data()[j] = 0.0;
    if (!col.hasValue())
      return fail<linear::LinearEstimate>(
          col.status().code(),
          "structured operator application failed during factorization");
    for (Index i = 0; i < s.dimension; ++i)
      dense.at(i, j) = col.value().at(i);
  }
  for (Index i = 0; i < s.dimension; ++i)
    for (Index j = 0; j < i; ++j) {
      const Scalar v = 0.5 * dense.at(i, j) + 0.5 * dense.at(j, i);
      dense.at(i, j) = v;
      dense.at(j, i) = v;
    }
  auto eig = linear::decomposeSymmetricEigen(dense.view());
  if (!eig.hasValue())
    return fail<linear::LinearEstimate>(
        eig.status().code(), "structured covariance spectrum failed");
  if (eig.value().definiteness() ==
      linear::PositiveDefiniteness::NotPositiveSemidefinite)
    return fail<linear::LinearEstimate>(ErrorCode::NotPositiveSemidefinite,
                                        "represented covariance is not PSD");
  const auto values = eig.value().eigenvalues();
  const auto q = eig.value().eigenvectors();
  Scalar max =
      values.empty() ? 0.0 : std::max(0.0, values.at(values.size() - 1));
  const long double tw = std::max(static_cast<long double>(p.rank_abs),
                                  static_cast<long double>(p.rank_rel) * max);
  if (!std::isfinite(tw) || tw > std::numeric_limits<Scalar>::max())
    return fail<linear::LinearEstimate>(
        ErrorCode::NonFiniteResult,
        "structured rank cutoff is not representable");
  Scalar tau = static_cast<Scalar>(tw);
  Index rank = 0;
  Scalar minret = max;
  for (Index k = 0; k < values.size(); ++k)
    if (values.at(k) > tau) {
      ++rank;
      minret = std::min(minret, values.at(k));
    }
  if (p.rank == RankPolicy::reject && rank != s.dimension) {
    return fail<linear::LinearEstimate>(
        ErrorCode::RankDeficient, "structured covariance is rank deficient");
  }
  if (rank && max > 0.0 &&
      static_cast<long double>(minret) / max < p.reciprocal_condition_floor) {
    return fail<linear::LinearEstimate>(
        ErrorCode::IllConditioned,
        "structured covariance fails condition floor");
  }
  const bool deficient = rank < s.dimension;
  Index basis_elems = deficient ? s.dimension * rank : 0;
  Index total = 0;
  if (!add(out, rhs.columns(), total) || !add(total, basis_elems, total) ||
      !add(total, square, total) || total > lim.elements ||
      total > lim.bytes / sizeof(Scalar))
    return fail<linear::LinearEstimate>(
        ErrorCode::ResourceLimit, "structured solve workspace exceeds limits");
  linear::LinearEstimate result{
      .parameters = Matrix(s.dimension, rhs.columns(), 0.0),
      .covariance = std::nullopt,
      .identified_basis =
          deficient ? std::optional<Matrix>(Matrix(s.dimension, rank, 0.0))
                    : std::nullopt,
      .residual_norms = Vector(rhs.columns(), 0.0),
      .rank = rank,
      .rank_cutoff = tau};
  Index bc = 0;
  for (Index k = 0; k < values.size(); ++k)
    if (values.at(k) > tau) {
      if (deficient) {
        for (Index i = 0; i < s.dimension; ++i)
          result.identified_basis->at(i, bc) = q.at(i, k);
        ++bc;
      }
      for (Index col = 0; col < rhs.columns(); ++col) {
        long double proj = 0;
        for (Index i = 0; i < s.dimension; ++i)
          proj += static_cast<long double>(q.at(i, k)) * rhs.at(i, col);
        proj /= values.at(k);
        for (Index i = 0; i < s.dimension; ++i)
          result.parameters.at(i, col) = static_cast<Scalar>(
              static_cast<long double>(result.parameters.at(i, col)) +
              q.at(i, k) * proj);
      }
    }
  for (Index col = 0; col < rhs.columns(); ++col) {
    Vector x(s.dimension, 0.0);
    for (Index i = 0; i < s.dimension; ++i)
      x.data()[i] = result.parameters.at(i, col);
    auto ax = applyRaw(s, x.view(), lim);
    if (!ax.hasValue())
      return fail<linear::LinearEstimate>(ax.status().code(),
                                          "structured residual action failed");
    long double norm = 0;
    for (Index i = 0; i < s.dimension; ++i)
      norm = std::hypot(norm, static_cast<long double>(ax.value().at(i)) -
                                  rhs.at(i, col));
    if (!std::isfinite(norm) || norm > std::numeric_limits<Scalar>::max())
      return fail<linear::LinearEstimate>(
          ErrorCode::NonFiniteResult,
          "structured residual is not representable");
    result.residual_norms.data()[col] = static_cast<Scalar>(norm);
  }
  return Result<linear::LinearEstimate>::success(std::move(result));
}
} // namespace vss::math::covariance
