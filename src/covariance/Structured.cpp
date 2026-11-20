#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <linear/DenseWorkspace.hpp>
#include <linear/PCG.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/Structured.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>

namespace vss::math::covariance {
namespace {

using linear::LinearEstimate;
using linear::detail::EvaluationBudget;
using linear::detail::representable;
using linear::detail::Wide;
using WideVector = std::vector<Wide>;

constexpr Wide epsilon = std::numeric_limits<Scalar>::epsilon();
constexpr Wide wide_epsilon = std::numeric_limits<Wide>::epsilon();
constexpr Wide infinity = std::numeric_limits<Wide>::infinity();

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

template <class T> Result<T> forward(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

template <class T> bool reserve(Limits& limits, Index rows, Index columns = 1) {
  if (columns == 0 || rows == 0) {
    return true;
  }
  if (rows >
      std::min({limits.elements, limits.bytes / sizeof(T),
                static_cast<Index>(std::numeric_limits<std::ptrdiff_t>::max()) /
                    sizeof(T)}) /
          columns) {
    return false;
  }
  const auto size = rows * columns;
  limits.elements -= size;
  limits.bytes -= size * sizeof(T);
  return true;
}

bool finite(ConstVectorView input) {
  for (Index i = 0; i < input.size(); ++i) {
    if (!std::isfinite(input.at(i))) {
      return false;
    }
  }
  return true;
}

bool finite(ConstMatrixView input) {
  for (Index i = 0; i < input.rows(); ++i) {
    for (Index j = 0; j < input.columns(); ++j) {
      if (!std::isfinite(input.at(i, j))) {
        return false;
      }
    }
  }
  return true;
}

Index bandwidth(ConstVectorView lags) {
  Index size = lags.size();
  while (size > 1 && lags.at(size - 1) == 0) {
    --size;
  }
  return size == 0 ? 0 : size - 1;
}

Wide symmetricEntry(const Matrix& factor, Index i, Index j) {
  return (static_cast<Wide>(factor.at(i, j)) + factor.at(j, i)) / 2;
}

Result<Index> validate(const StructuredSpec& spec) {
  const auto lags = spec.diagonal_or_lags;
  const bool kernel = spec.kind == StructureKind::gaussian_kernel;
  const bool low_rank = spec.kind == StructureKind::diagonal_low_rank;
  const bool product = spec.kind == StructureKind::kronecker;
  const bool factors = product || spec.kind == StructureKind::block;
  if (!low_rank && !kernel && !factors &&
      spec.kind != StructureKind::toeplitz) {
    return fail<Index>(ErrorCode::InvalidDomain,
                       "unknown covariance structure");
  }
  if ((!low_rank && spec.kind != StructureKind::toeplitz && !lags.empty()) ||
      (!low_rank && !kernel && spec.low_rank_or_points) ||
      (!factors && !spec.factors.empty()) ||
      (!kernel && (spec.kernel_scale || spec.kernel_length))) {
    return fail<Index>(ErrorCode::InvalidDomain,
                       "inactive covariance fields are populated");
  }
  if (!finite(lags) ||
      (spec.low_rank_or_points && !finite(*spec.low_rank_or_points)) ||
      (spec.kernel_scale && !std::isfinite(*spec.kernel_scale)) ||
      (spec.kernel_length && !std::isfinite(*spec.kernel_length))) {
    return fail<Index>(ErrorCode::NonFiniteInput,
                       "structured covariance data are non-finite");
  }
  if (kernel) {
    if (!spec.low_rank_or_points || spec.low_rank_or_points->columns() == 0 ||
        !spec.kernel_scale || *spec.kernel_scale < 0 || !spec.kernel_length ||
        *spec.kernel_length <= 0) {
      return fail<Index>(ErrorCode::InvalidDomain,
                         "invalid Gaussian kernel parameters");
    }
    return Result<Index>::success(spec.low_rank_or_points->rows());
  }
  if (!factors) {
    if (low_rank && spec.low_rank_or_points &&
        spec.low_rank_or_points->rows() != lags.size()) {
      return fail<Index>(ErrorCode::DimensionMismatch,
                         "low rank factor height differs from diagonal");
    }
    const Index diagonals =
        low_rank ? lags.size() : std::min<Index>(1, lags.size());
    for (Index i = 0; i < diagonals; ++i) {
      if (lags.at(i) < 0) {
        return fail<Index>(ErrorCode::NotPositiveSemidefinite,
                           "covariance diagonal is negative");
      }
    }
    return Result<Index>::success(lags.size());
  }

  const bool zero_product =
      product && std::ranges::any_of(spec.factors, [](const Matrix& factor) {
        return factor.rows() == 0;
      });
  Index n = product && !zero_product ? 1 : 0;
  for (const auto& factor : spec.factors) {
    const Index order = factor.rows();
    if (order != factor.columns()) {
      return fail<Index>(ErrorCode::DimensionMismatch,
                         "covariance factor must be square");
    }
    if (!finite(factor.view())) {
      return fail<Index>(ErrorCode::NonFiniteInput,
                         "covariance factor is non-finite");
    }
    Wide scale = 0;
    for (const auto value : std::span(factor.data(), factor.size())) {
      scale = std::max(scale, std::abs(static_cast<Wide>(value)));
    }
    const Wide allowance = 64 * epsilon * static_cast<Wide>(order) * scale;
    for (Index i = 0; i < order; ++i) {
      for (Index j = 0; j < i; ++j) {
        if (std::abs(static_cast<Wide>(factor.at(i, j)) - factor.at(j, i)) >
            allowance) {
          return fail<Index>(ErrorCode::NotSymmetric,
                             "covariance factor is asymmetric");
        }
      }
    }
    if (product && !zero_product) {
      if (n > std::numeric_limits<Index>::max() / order) {
        return fail<Index>(ErrorCode::SizeOverflow,
                           "Kronecker order overflows");
      }
      n *= order;
    } else if (!product) {
      if (order > std::numeric_limits<Index>::max() - n) {
        return fail<Index>(ErrorCode::SizeOverflow,
                           "block covariance order overflows");
      }
      n += order;
    }
  }
  return Result<Index>::success(n);
}

StructuredSpec borrow(const StructuredCovariance& value) {
  return {
      .kind = value.kind,
      .diagonal_or_lags = value.diagonal_or_lags.view(),
      .low_rank_or_points =
          value.low_rank_or_points
              ? std::optional<ConstMatrixView>{value.low_rank_or_points->view()}
              : std::nullopt,
      .factors = value.factors.view(),
      .kernel_scale = value.kernel_scale,
      .kernel_length = value.kernel_length};
}

template <class Term> Wide sum(Index count, const Term& term) {
  Wide total = 0;
  Wide correction = 0;
  for (Index i = 0; i < count; ++i) {
    const Wide value = term(i);
    const Wide next = total + value;
    correction += std::abs(total) >= std::abs(value) ? (total - next) + value
                                                     : (value - next) + total;
    total = next;
  }
  return total + correction;
}

Wide norm(std::span<const Wide> values) {
  Wide result = 0;
  for (const Wide value : values) {
    result = std::hypot(result, value);
  }
  return result;
}

Wide kernelEntry(const StructuredSpec& spec, Index i, Index j) {
  const Matrix empty_points;
  const auto points = spec.low_rank_or_points.value_or(empty_points.view());
  const Wide length = spec.kernel_length.value_or(1);
  const Wide distance = sum(points.columns(), [&](Index k) {
    const Wide scaled =
        (static_cast<Wide>(points.at(i, k)) - points.at(j, k)) / length;
    return scaled * scaled;
  });
  return static_cast<Wide>(spec.kernel_scale.value_or(0)) *
         std::exp(-distance / 2);
}

template <class Entry>
void axisProduct(std::span<const Wide> input, std::span<Wide> output,
                 Index order, Index stride, const Entry& entry) {
  const Index slab = order * stride;
  for (Index base = 0; base < input.size(); base += slab) {
    for (Index row = 0; row < order; ++row) {
      for (Index inner = 0; inner < stride; ++inner) {
        output[base + row * stride + inner] = sum(order, [&](Index col) {
          return entry(row, col) * input[base + col * stride + inner];
        });
      }
    }
  }
}

class Action final {
public:
  Action(StructuredSpec spec, Index order, std::span<Wide> scratch)
      : spec_(spec), n_(order), bandwidth_(bandwidth(spec.diagonal_or_lags)),
        scratch_(scratch) {}

  static Index scratchSize(const StructuredSpec& spec, Index n) {
    if (n == 0) {
      return 0;
    }
    if (spec.kind == StructureKind::kronecker && spec.factors.size() > 1) {
      return n;
    }
    return spec.kind == StructureKind::diagonal_low_rank &&
                   spec.low_rank_or_points
               ? spec.low_rank_or_points->columns()
               : 0;
  }

  void operator()(std::span<const Wide> input, std::span<Wide> output) const {
    if (n_ == 0) {
      return;
    }
    switch (spec_.kind) {
    case StructureKind::diagonal_low_rank: {
      const auto d = spec_.diagonal_or_lags;
      if (spec_.low_rank_or_points) {
        const auto u = *spec_.low_rank_or_points;
        for (Index j = 0; j < u.columns(); ++j) {
          scratch_[j] = sum(n_, [&](Index i) { return u.at(i, j) * input[i]; });
        }
        for (Index i = 0; i < n_; ++i) {
          output[i] = sum(u.columns() + 1, [&](Index j) {
            return j == u.columns() ? d.at(i) * input[i]
                                    : u.at(i, j) * scratch_[j];
          });
        }
      } else {
        for (Index i = 0; i < n_; ++i) {
          output[i] = d.at(i) * input[i];
        }
      }
      return;
    }
    case StructureKind::toeplitz:
      for (Index i = 0; i < n_; ++i) {
        const Index first = i - std::min(i, bandwidth_);
        const Index last = i + std::min(n_ - i - 1, bandwidth_);
        output[i] = sum(last - first + 1, [&](Index offset) {
          const Index j = first + offset;
          return spec_.diagonal_or_lags.at(i < j ? j - i : i - j) * input[j];
        });
      }
      return;
    case StructureKind::block: {
      Index first = 0;
      for (const auto& factor : spec_.factors) {
        for (Index i = 0; i < factor.rows(); ++i) {
          output[first + i] = sum(factor.rows(), [&](Index j) {
            return symmetricEntry(factor, i, j) * input[first + j];
          });
        }
        first += factor.rows();
      }
      return;
    }
    case StructureKind::kronecker:
      kronecker(input, output);
      return;
    case StructureKind::gaussian_kernel:
      for (Index i = 0; i < n_; ++i) {
        output[i] = sum(
            n_, [&](Index j) { return kernelEntry(spec_, i, j) * input[j]; });
      }
      return;
    }
  }

private:
  void kronecker(std::span<const Wide> source, std::span<Wide> output) const {
    Index stride = n_;
    std::int64_t exponent = 0;
    for (Index i = 0; i < spec_.factors.size(); ++i) {
      const auto& factor = spec_.factors[i];
      stride /= factor.rows();
      const auto target =
          ((spec_.factors.size() - i) % 2 == 1) ? output : scratch_;
      axisProduct(source, target, factor.rows(), stride,
                  [&](Index row, Index col) {
                    return symmetricEntry(factor, row, col);
                  });
      Wide scale = 0;
      for (const Wide value : target) {
        scale = std::max(scale, std::abs(value));
      }
      if (scale != 0) {
        const int shift = std::ilogb(scale);
        exponent += shift;
        for (Wide& value : target) {
          value = std::scalbn(value, -shift);
        }
      }
      source = target;
    }
    if (spec_.factors.empty()) {
      output.front() = source.front();
      return;
    }
    for (Wide& value : output) {
      if (value != 0 && exponent != 0) {
        const auto shift =
            std::clamp(exponent, static_cast<std::int64_t>(-32768),
                       static_cast<std::int64_t>(32768));
        value = std::scalbn(value, static_cast<int>(shift));
      }
    }
  }

  StructuredSpec spec_;
  Index n_;
  Index bandwidth_;
  std::span<Wide> scratch_;
};

struct Mode final {
  Wide value{};
  Wide lower{};
  Wide upper{};
};

Mode exact(Wide value) { return {value, value, value}; }

Mode enclosed(Wide value, Wide error) {
  if (error == 0) {
    return exact(value);
  }
  return {value, std::nextafter(value - error, -infinity),
          std::nextafter(value + error, infinity)};
}

struct Spectrum final {
  std::vector<Mode> values;
  Matrix basis; // An empty basis denotes the coordinate basis.
};

Wide basisEntry(const Spectrum& spectrum, Index coordinate, Index mode) {
  if (spectrum.basis.empty()) {
    return coordinate == mode ? 1 : 0;
  }
  return spectrum.basis.at(coordinate, mode);
}

struct Bounds final {
  Wide lower{};
  Wide upper{};
};

template <class Term> Wide frobenius(Index n, const Term& term) {
  Wide result = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      result = std::hypot(result, term(i, j));
    }
  }
  return result;
}

template <class Entry, class Image>
Wide symmetricError(ConstMatrixView basis, std::span<const Mode> values,
                    const Entry& entry, const Image& image, Wide action_error) {
  const Index n = basis.rows();
  const Wide gram = frobenius(n, [&](Index i, Index j) {
    return sum(n,
               [&](Index k) {
                 return static_cast<Wide>(basis.at(k, i)) * basis.at(k, j);
               }) -
           (i == j ? 1 : 0);
  });
  const Wide roundoff = 32 * wide_epsilon * static_cast<Wide>(n);
  const Wide delta = gram + roundoff * static_cast<Wide>(n);
  if (delta >= 0.5L) {
    return infinity;
  }
  const Wide residual = frobenius(n, [&](Index i, Index j) {
    return image(i, j) - basis.at(i, j) * values[j].value;
  });
  Wide largest = 0;
  for (const Mode mode : values) {
    largest = std::max(largest, std::abs(mode.value));
  }
  const Wide matrix_norm = frobenius(n, entry);
  const Wide polar = delta / (1 + std::sqrt(1 - delta));
  return std::nextafter(
      (residual + action_error + roundoff * matrix_norm + 2 * polar * largest) /
          std::sqrt(1 - delta),
      infinity);
}

bool reserveEigenScratch(Limits limits, Index n) {
  // Jacobi working copy, computed and sorted bases, values and sort indices.
  for (Index matrix = 0; matrix < 3; ++matrix) {
    if (!reserve<Scalar>(limits, n, n)) {
      return false;
    }
  }
  return reserve<Scalar>(limits, n, 2) && reserve<Index>(limits, n, 2);
}

template <class Entry>
Result<Spectrum> symmetricSpectrum(Index n, const Entry& entry,
                                   SolvePolicy policy, Limits limits) {
  if (!reserve<Mode>(limits, n)) {
    return fail<Spectrum>(ErrorCode::ResourceLimit,
                          "covariance spectrum exceeds limits");
  }
  bool diagonal = true;
  Wide scale = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j <= i; ++j) {
      const Wide value = entry(i, j);
      diagonal = diagonal && (i == j || value == 0);
      scale = std::max(scale, std::abs(value));
    }
  }
  Spectrum result{.values = std::vector<Mode>(n), .basis = {}};
  if (diagonal) {
    for (Index i = 0; i < n; ++i) {
      result.values[i] = exact(entry(i, i));
    }
    return Result<Spectrum>::success(std::move(result));
  }
  if (policy.iteration.max_iterations == 0) {
    return fail<Spectrum>(ErrorCode::NoConvergence,
                          "covariance eigensystem requires iterations");
  }
  if (!reserve<Scalar>(limits, n, n) || !reserveEigenScratch(limits, n)) {
    return fail<Spectrum>(ErrorCode::ResourceLimit,
                          "covariance eigensystem exceeds limits");
  }
  const int exponent = std::ilogb(scale);
  Matrix scaled(n, n);
  Wide conversion_error = 0;
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j <= i; ++j) {
      const Wide value = std::scalbn(entry(i, j), -exponent);
      const auto converted = static_cast<Scalar>(value);
      conversion_error = std::hypot(conversion_error, value - converted);
      if (i != j) {
        conversion_error = std::hypot(conversion_error, value - converted);
      }
      scaled.at(i, j) = converted;
      scaled.at(j, i) = converted;
    }
  }
  auto eigen = linear::decomposeSymmetricEigen(
      scaled.view(), {.rankTolerance = {.absolute = 0, .relative = 0},
                      .symmetryTolerance = {.absolute = 0, .relative = 0},
                      .maximumSweeps = policy.iteration.max_iterations});
  if (!eigen.hasValue()) {
    return forward<Spectrum>(eigen.status());
  }
  result.basis = Matrix(n, n);
  for (Index i = 0; i < n; ++i) {
    result.values[i] = exact(eigen.value().eigenvalues().at(i));
    for (Index j = 0; j < n; ++j) {
      result.basis.at(j, i) = eigen.value().eigenvectors().at(j, i);
    }
  }
  const Wide error = symmetricError(
      result.basis.view(), result.values,
      [&](Index i, Index j) { return scaled.at(i, j); },
      [&](Index i, Index j) {
        return sum(n, [&](Index k) {
          return static_cast<Wide>(scaled.at(i, k)) * result.basis.at(k, j);
        });
      },
      conversion_error);
  if (!std::isfinite(error)) {
    return fail<Spectrum>(ErrorCode::NoConvergence,
                          "covariance spectrum error is unbounded");
  }
  for (Mode& value : result.values) {
    value = enclosed(std::scalbn(value.value, exponent),
                     std::scalbn(error, exponent));
  }
  Index pivot = 0;
  for (Index i = 1; i < n; ++i) {
    if (std::abs(entry(i, i)) > std::abs(entry(pivot, pivot))) {
      pivot = i;
    }
  }
  const auto exactProduct = [](Wide lhs, Wide rhs) {
    const Wide product = lhs * rhs;
    return std::pair{product, std::fma(lhs, rhs, -product)};
  };
  const Wide leading = entry(pivot, pivot);
  bool rank_one = leading > 0;
  for (Index i = 0; rank_one && i < n; ++i) {
    for (Index j = 0; j <= i; ++j) {
      if (exactProduct(leading, entry(i, j)) !=
          exactProduct(entry(i, pivot), entry(pivot, j))) {
        rank_one = false;
        break;
      }
    }
  }
  if (rank_one && result.values.back().lower > 0) {
    for (Index i = 0; i + 1 < n; ++i) {
      result.values[i] = exact(0);
    }
  }
  return Result<Spectrum>::success(std::move(result));
}

Status certifyPsd(std::span<const Mode> values) {
  Wide scale = 0;
  for (const Mode mode : values) {
    scale = std::max({scale, std::abs(mode.lower), std::abs(mode.upper)});
  }
  const Wide slack = 64 * epsilon * static_cast<Wide>(values.size()) * scale;
  for (const Mode mode : values) {
    if (mode.upper < -slack) {
      return Status::failure(ErrorCode::NotPositiveSemidefinite,
                             "covariance has a negative eigenvalue");
    }
    if (mode.lower < -slack) {
      return Status::failure(ErrorCode::NoConvergence,
                             "covariance definiteness is unresolved");
    }
  }
  return Status::success();
}

Mode nonnegative(Mode mode) {
  return {std::max<Wide>(0, mode.value), std::max<Wide>(0, mode.lower),
          std::max<Wide>(0, mode.upper)};
}

Bounds extremes(std::span<const Mode> values) {
  Bounds result{infinity, 0};
  for (const Mode mode : values) {
    result.lower = std::min(result.lower, mode.lower);
    result.upper = std::max(result.upper, mode.upper);
  }
  return result;
}

Result<Spectrum> lowRankSpectrum(const StructuredSpec& spec, SolvePolicy policy,
                                 Limits limits) {
  const Index n = spec.diagonal_or_lags.size();
  const Matrix empty_updates;
  const auto updates = spec.low_rank_or_points.value_or(empty_updates.view());
  if (updates.columns() > std::numeric_limits<Index>::max() - n) {
    return fail<Spectrum>(ErrorCode::SizeOverflow,
                          "low rank factor width overflows");
  }
  const Index width = n + updates.columns();
  if (!reserve<Scalar>(limits, n, width) || !reserve<Mode>(limits, n)) {
    return fail<Spectrum>(ErrorCode::ResourceLimit,
                          "low rank factor exceeds limits");
  }
  Matrix factor(n, width);
  Wide scale = 0;
  for (Index i = 0; i < n; ++i) {
    scale = std::max(scale,
                     std::sqrt(static_cast<Wide>(spec.diagonal_or_lags.at(i))));
    for (Index j = 0; j < updates.columns(); ++j) {
      scale = std::max(scale, std::abs(static_cast<Wide>(updates.at(i, j))));
    }
  }
  const int exponent = scale == 0 ? 0 : std::ilogb(scale);
  Wide conversion_error = 0;
  for (Index i = 0; i < n; ++i) {
    const Wide root = std::scalbn(
        std::sqrt(static_cast<Wide>(spec.diagonal_or_lags.at(i))), -exponent);
    factor.at(i, i) = static_cast<Scalar>(root);
    const Wide root_error =
        std::abs(root - factor.at(i, i)) +
        (std::fma(root, root,
                  -std::scalbn(static_cast<Wide>(spec.diagonal_or_lags.at(i)),
                               -2 * exponent)) == 0
             ? 0
             : wide_epsilon * root);
    conversion_error = std::hypot(conversion_error, root_error);
    for (Index j = 0; j < updates.columns(); ++j) {
      const Wide value =
          std::scalbn(static_cast<Wide>(updates.at(i, j)), -exponent);
      factor.at(i, n + j) = static_cast<Scalar>(value);
      conversion_error =
          std::hypot(conversion_error, value - factor.at(i, n + j));
    }
  }
  auto bounded = linear::detail::boundedSVD(factor.view(), policy, limits);
  if (!bounded.hasValue()) {
    return forward<Spectrum>(bounded.status());
  }
  const auto& decomposition = bounded.value();
  const Wide error = decomposition.error + conversion_error;
  Spectrum result{.values = std::vector<Mode>(n),
                  .basis = std::move(bounded.value().factors.left)};
  const Index possible_rank =
      conversion_error == 0
          ? decomposition.rank_upper
          : static_cast<Index>(
                std::count_if(std::span(factor.data(), factor.size()).begin(),
                              std::span(factor.data(), factor.size()).end(),
                              [](Scalar value) { return value != 0; }));
  for (Index i = 0; i < std::min(n, possible_rank); ++i) {
    const Wide value = std::scalbn(
        static_cast<Wide>(decomposition.factors.singular.at(i)), exponent);
    const Wide width_error = std::scalbn(error, exponent);
    const Wide lower = std::max<Wide>(0, value - width_error);
    const Wide upper = value + width_error;
    result.values[i] =
        width_error == 0
            ? exact(value * value)
            : Mode{value * value, std::nextafter(lower * lower, Wide{0}),
                   std::nextafter(upper * upper, infinity)};
  }
  return Result<Spectrum>::success(std::move(result));
}

bool repeatedPoints(ConstMatrixView points) {
  for (Index i = 1; i < points.rows(); ++i) {
    for (Index j = 0; j < points.columns(); ++j) {
      if (points.at(i, j) != points.at(0, j)) {
        return false;
      }
    }
  }
  return true;
}

Result<Spectrum> projectedSpectrum(const StructuredSpec& spec, Index n,
                                   SolvePolicy policy, Limits limits) {
  if (!reserve<Wide>(limits, n, n) || !reserve<Wide>(limits, n, n + 1) ||
      !reserve<Wide>(limits, Action::scratchSize(spec, n)) ||
      !reserve<Scalar>(limits, n, n)) {
    return fail<Spectrum>(ErrorCode::ResourceLimit,
                          "Krylov projection exceeds limits");
  }
  WideVector basis(n * n);
  WideVector images(n * n);
  WideVector work(n);
  WideVector scratch(Action::scratchSize(spec, n));
  const Action apply(spec, n, scratch);
  const auto column = [n](WideVector& values, Index j) {
    return std::span(values).subspan(j * n, n);
  };
  Index coordinate = 0;
  work.front() = 1;
  for (Index j = 0; j < n; ++j) {
    Wide length = 0;
    while (true) {
      const Wide initial = norm(work);
      for (Index pass = 0; pass < 2; ++pass) {
        for (Index k = 0; k < j; ++k) {
          const auto q = column(basis, k);
          const Wide projection =
              sum(n, [&](Index i) { return q[i] * work[i]; });
          for (Index i = 0; i < n; ++i) {
            work[i] = std::fma(-q[i], projection, work[i]);
          }
        }
      }
      length = norm(work);
      if (length > 64 * wide_epsilon * static_cast<Wide>(n) * initial) {
        break;
      }
      if (coordinate == n) {
        return fail<Spectrum>(ErrorCode::NoConvergence,
                              "Krylov basis lost an independent direction");
      }
      std::ranges::fill(work, 0);
      work[coordinate++] = 1;
    }
    auto q = column(basis, j);
    auto image = column(images, j);
    for (Index i = 0; i < n; ++i) {
      q[i] = work[i] / length;
    }
    apply(q, image);
    std::ranges::copy(image, work.begin());
  }

  const auto projected = [&](Index i, Index j) {
    const auto qi = column(basis, i);
    const auto qj = column(basis, j);
    const auto si = column(images, i);
    const auto sj = column(images, j);
    return sum(n, [&](Index k) { return (qi[k] * sj[k] + si[k] * qj[k]) / 2; });
  };
  auto modes = symmetricSpectrum(n, projected, policy, limits);
  if (!modes.hasValue()) {
    return modes;
  }
  Spectrum result{.values = std::move(modes.value().values),
                  .basis = Matrix(n, n)};
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      result.basis.at(i, j) = static_cast<Scalar>(sum(n, [&](Index k) {
        return column(basis, k)[i] * basisEntry(modes.value(), k, j);
      }));
    }
  }
  const auto original = [&](Index i, Index j) {
    return spec.kind == StructureKind::toeplitz
               ? static_cast<Wide>(
                     spec.diagonal_or_lags.at(i < j ? j - i : i - j))
               : kernelEntry(spec, i, j);
  };
  const Wide kernel_width =
      spec.kind == StructureKind::gaussian_kernel && spec.low_rank_or_points
          ? 1 + static_cast<Wide>(spec.low_rank_or_points->columns())
          : 1;
  const Wide operator_norm = frobenius(n, original);
  const Wide formation_error = frobenius(n, [&](Index i, Index j) {
    return sum(n,
               [&](Index k) {
                 return column(basis, k)[i] * basisEntry(modes.value(), k, j);
               }) -
           result.basis.at(i, j);
  });
  const Wide action_error =
      operator_norm *
      (formation_error +
       128 * wide_epsilon * static_cast<Wide>(n) * kernel_width);
  const Wide error = symmetricError(
      result.basis.view(), result.values, original,
      [&](Index i, Index j) {
        return sum(n, [&](Index k) {
          return column(images, k)[i] * basisEntry(modes.value(), k, j);
        });
      },
      action_error);
  if (!std::isfinite(error)) {
    return fail<Spectrum>(ErrorCode::NoConvergence,
                          "projected covariance spectrum error is unbounded");
  }
  for (Mode& mode : result.values) {
    mode = enclosed(mode.value, error);
  }
  return Result<Spectrum>::success(std::move(result));
}

Result<Spectrum> singleSpectrum(const StructuredSpec& spec, Index n,
                                SolvePolicy policy, Limits limits) {
  if (n == 0) {
    return Result<Spectrum>::success({{}, {}});
  }
  if (spec.kind == StructureKind::diagonal_low_rank &&
      spec.low_rank_or_points && spec.low_rank_or_points->columns() != 0) {
    return lowRankSpectrum(spec, policy, limits);
  }
  const auto kernel_scale = spec.kernel_scale.value_or(0);
  if ((spec.kind == StructureKind::toeplitz &&
       bandwidth(spec.diagonal_or_lags) != 0) ||
      (spec.kind == StructureKind::gaussian_kernel && kernel_scale != 0 &&
       spec.low_rank_or_points && !repeatedPoints(*spec.low_rank_or_points))) {
    return projectedSpectrum(spec, n, policy, limits);
  }
  if (!reserve<Mode>(limits, n)) {
    return fail<Spectrum>(ErrorCode::ResourceLimit,
                          "covariance modes exceed limits");
  }
  Spectrum result{.values = std::vector<Mode>(n), .basis = {}};
  if (spec.kind == StructureKind::gaussian_kernel && kernel_scale != 0) {
    if (!reserve<Scalar>(limits, n)) {
      return fail<Spectrum>(ErrorCode::ResourceLimit,
                            "kernel rank-one basis exceeds limits");
    }
    result.values.front() =
        exact(static_cast<Wide>(kernel_scale) * static_cast<Wide>(n));
    result.basis =
        Matrix(n, 1, static_cast<Scalar>(1 / std::sqrt(static_cast<Wide>(n))));
  } else if (spec.kind != StructureKind::gaussian_kernel) {
    for (Index i = 0; i < n; ++i) {
      result.values[i] = exact(spec.diagonal_or_lags.at(
          spec.kind == StructureKind::toeplitz ? 0 : i));
    }
  }
  return Result<Spectrum>::success(std::move(result));
}

template <class Value>
Wide scaledProduct(Index count, const Value& value, int rounding = 0) {
  Wide mantissa = 1;
  std::int64_t exponent = 0;
  for (Index i = 0; i < count; ++i) {
    const Wide v = value(i);
    if (v == 0) {
      return 0;
    }
    int shift = 0;
    const Wide fraction = std::frexp(v, &shift);
    const Wide product = mantissa * fraction;
    const Wide error = std::fma(mantissa, fraction, -product);
    mantissa = (rounding > 0 && error > 0) || (rounding < 0 && error < 0)
                   ? std::nextafter(product, rounding > 0 ? infinity : Wide{0})
                   : product;
    int normalization = 0;
    mantissa = std::frexp(mantissa, &normalization);
    exponent += shift + normalization;
  }
  if (exponent > std::numeric_limits<Wide>::max_exponent) {
    return infinity;
  }
  if (exponent < std::numeric_limits<Wide>::min_exponent -
                     std::numeric_limits<Wide>::digits) {
    return rounding > 0 ? std::numeric_limits<Wide>::denorm_min() : 0;
  }
  return std::scalbn(mantissa, static_cast<int>(exponent));
}

Result<std::vector<Spectrum>>
factorSpectra(const StructuredSpec& spec, SolvePolicy policy, Limits& limits) {
  using Output = Result<std::vector<Spectrum>>;
  if (!reserve<Spectrum>(limits, spec.factors.size())) {
    return fail<std::vector<Spectrum>>(
        ErrorCode::ResourceLimit, "covariance factor metadata exceeds limits");
  }
  std::vector<Spectrum> factors;
  factors.reserve(spec.factors.size());
  for (const auto& factor : spec.factors) {
    const auto n = factor.rows();
    auto spectrum = symmetricSpectrum(
        n, [&](Index i, Index j) { return symmetricEntry(factor, i, j); },
        policy, limits);
    if (!spectrum.hasValue()) {
      return forward<std::vector<Spectrum>>(spectrum.status());
    }
    auto& modes = spectrum.value();
    if (const auto psd = certifyPsd(modes.values); !psd.ok()) {
      return forward<std::vector<Spectrum>>(psd);
    }
    if (!reserve<Mode>(limits, n) ||
        !reserve<Scalar>(limits, modes.basis.rows(), modes.basis.columns())) {
      return fail<std::vector<Spectrum>>(
          ErrorCode::ResourceLimit,
          "retained covariance factors exceed limits");
    }
    for (Mode& mode : modes.values) {
      mode = nonnegative(mode);
    }
    factors.push_back(std::move(spectrum).value());
  }
  return Output::success(std::move(factors));
}

Mode productMode(std::span<const Spectrum> factors, Index n, Index coordinate) {
  const auto compute = [&](Wide Mode::*field, int rounding) {
    Index stride = n;
    return scaledProduct(
        factors.size(),
        [&](Index axis) {
          const auto& factor = factors[axis];
          stride /= factor.values.size();
          const Index component = (coordinate / stride) % factor.values.size();
          return factor.values[component].*field;
        },
        rounding);
  };
  return {compute(&Mode::value, 0), compute(&Mode::lower, -1),
          compute(&Mode::upper, 1)};
}

void rotateProduct(std::span<const Spectrum> factors, WideVector& values,
                   WideVector& work, bool transpose) {
  Index stride = values.size();
  for (const auto& factor : factors) {
    stride /= factor.values.size();
    if (factor.basis.empty()) {
      continue;
    }
    axisProduct(values, work, factor.values.size(), stride,
                [&](Index row, Index col) {
                  const Index source_row = transpose ? col : row;
                  const Index source_col = transpose ? row : col;
                  return factor.basis.at(source_row, source_col);
                });
    values.swap(work);
  }
}

Wide productBasis(std::span<const Spectrum> factors, Index dimension,
                  Index coordinate, Index mode) {
  Index stride = dimension;
  return scaledProduct(factors.size(), [&](Index axis) {
    const auto& factor = factors[axis];
    const Index order = factor.values.size();
    stride /= order;
    return basisEntry(factor, (coordinate / stride) % order,
                      (mode / stride) % order);
  });
}

Result<LinearEstimate> factorizedSolve(const StructuredSpec& spec, Index n,
                                       ConstMatrixView rhs, SolvePolicy policy,
                                       Limits limits) {
  const bool is_product = spec.kind == StructureKind::kronecker;
  const bool is_block = spec.kind == StructureKind::block;
  std::vector<Spectrum> factors;
  if (is_product || is_block) {
    auto factored = factorSpectra(spec, policy, limits);
    if (!factored.hasValue()) {
      return forward<LinearEstimate>(factored.status());
    }
    factors = std::move(factored).value();
  } else {
    if (!reserve<Spectrum>(limits, 1)) {
      return fail<LinearEstimate>(ErrorCode::ResourceLimit,
                                  "covariance factor record exceeds limits");
    }
    auto spectrum = singleSpectrum(spec, n, policy, limits);
    if (!spectrum.hasValue()) {
      return forward<LinearEstimate>(spectrum.status());
    }
    if (const auto psd = certifyPsd(spectrum.value().values); !psd.ok()) {
      return forward<LinearEstimate>(psd);
    }
    if (!reserve<Mode>(limits, n) ||
        !reserve<Scalar>(limits, spectrum.value().basis.rows(),
                         spectrum.value().basis.columns())) {
      return fail<LinearEstimate>(
          ErrorCode::ResourceLimit,
          "retained covariance eigensystem exceeds limits");
    }
    for (Mode& mode : spectrum.value().values) {
      mode = nonnegative(mode);
    }
    factors.push_back(std::move(spectrum).value());
  }

  if (!reserve<Mode>(limits, n) || !reserve<std::uint8_t>(limits, n)) {
    return fail<LinearEstimate>(ErrorCode::ResourceLimit,
                                "covariance cutoff workspace exceeds limits");
  }
  std::vector<Mode> modes(n);
  if (is_product) {
    for (Index i = 0; i < n; ++i) {
      modes[i] = productMode(factors, n, i);
    }
  } else {
    auto destination = modes.begin();
    for (const auto& factor : factors) {
      destination = std::ranges::copy(factor.values, destination).out;
    }
  }
  Mode maximum{};
  for (const Mode m : modes) {
    maximum = {std::max(maximum.value, m.value),
               std::max(maximum.lower, m.lower),
               std::max(maximum.upper, m.upper)};
  }
  const auto cutoff = [&](Wide value) {
    return policy.rank_rel == 0
               ? static_cast<Wide>(policy.rank_abs)
               : std::max<Wide>(policy.rank_abs, policy.rank_rel * value);
  };
  const Wide tau = cutoff(maximum.value);
  const Wide lower_tau = cutoff(maximum.lower);
  const Wide upper_tau = cutoff(maximum.upper);
  if (!representable(tau)) {
    return fail<LinearEstimate>(ErrorCode::NonFiniteResult,
                                "covariance cutoff is not representable");
  }
  Wide smallest_lower = infinity;
  Wide smallest_upper = infinity;
  std::vector<std::uint8_t> retained(n);
  Index rank = 0;
  for (Index i = 0; i < n; ++i) {
    if (modes[i].lower > upper_tau) {
      retained[i] = 1;
      ++rank;
      smallest_lower = std::min(smallest_lower, modes[i].lower);
      smallest_upper = std::min(smallest_upper, modes[i].upper);
    } else if (modes[i].upper > lower_tau) {
      return fail<LinearEstimate>(
          ErrorCode::NoConvergence,
          "covariance cutoff intersects a spectral bound");
    }
  }
  if (policy.rank == RankPolicy::reject && rank != n) {
    return fail<LinearEstimate>(ErrorCode::RankDeficient,
                                "covariance rank is deficient");
  }
  if (rank != 0 && policy.reciprocal_condition_floor != 0) {
    if (smallest_upper < policy.reciprocal_condition_floor * maximum.lower) {
      return fail<LinearEstimate>(ErrorCode::IllConditioned,
                                  "covariance fails condition floor");
    }
    if (smallest_lower < policy.reciprocal_condition_floor * maximum.upper) {
      return fail<LinearEstimate>(ErrorCode::NoConvergence,
                                  "covariance condition floor is unresolved");
    }
  }

  const Index p = rhs.columns();
  const Index workspace = p == 0 ? 0 : n;
  const bool deficient = rank != n;
  if (!reserve<Scalar>(limits, n, p) || !reserve<Scalar>(limits, p) ||
      (deficient && !reserve<Scalar>(limits, n, rank)) ||
      !reserve<Wide>(limits, workspace, 2) ||
      (p != 0 && !reserve<Wide>(limits, Action::scratchSize(spec, n)))) {
    return fail<LinearEstimate>(
        ErrorCode::ResourceLimit,
        "covariance solve output and workspace exceed limits");
  }
  LinearEstimate result{.parameters = Matrix(n, p),
                        .covariance = std::nullopt,
                        .identified_basis = deficient
                                                ? std::optional{Matrix(n, rank)}
                                                : std::nullopt,
                        .residual_norms = Vector(p),
                        .rank = rank,
                        .rank_cutoff = static_cast<Scalar>(tau)};
  if (n == 0) {
    return Result<LinearEstimate>::success(std::move(result));
  }
  if (auto& basis_output = result.identified_basis; basis_output) {
    Index basis_column = 0;
    if (is_product) {
      for (Index mode = 0; mode < n; ++mode) {
        if (retained[mode] == 0) {
          continue;
        }
        for (Index i = 0; i < n; ++i) {
          basis_output->at(i, basis_column) =
              static_cast<Scalar>(productBasis(factors, n, i, mode));
        }
        ++basis_column;
      }
    } else {
      Index first = 0;
      for (const auto& factor : factors) {
        const Index order = factor.values.size();
        for (Index mode = 0; mode < order; ++mode) {
          if (retained[first + mode] == 0) {
            continue;
          }
          for (Index i = 0; i < order; ++i) {
            basis_output->at(first + i, basis_column) =
                static_cast<Scalar>(basisEntry(factor, i, mode));
          }
          ++basis_column;
        }
        first += order;
      }
    }
  }
  WideVector values(workspace);
  WideVector work(workspace);
  WideVector scratch(p == 0 ? 0 : Action::scratchSize(spec, n));
  const Action apply(spec, n, scratch);
  for (Index col = 0; col < p; ++col) {
    if (is_product) {
      for (Index i = 0; i < n; ++i) {
        values[i] = rhs.at(i, col);
      }
      rotateProduct(factors, values, work, true);
      for (Index i = 0; i < n; ++i) {
        values[i] = retained[i] != 0 ? values[i] / modes[i].value : 0;
      }
      rotateProduct(factors, values, work, false);
    } else {
      Index first = 0;
      for (const auto& factor : factors) {
        const Index order = factor.values.size();
        if (factor.basis.empty()) {
          for (Index i = 0; i < order; ++i) {
            values[first + i] =
                retained[first + i] != 0
                    ? rhs.at(first + i, col) / modes[first + i].value
                    : 0;
          }
          first += order;
          continue;
        }
        const Index basis_modes = factor.basis.columns();
        for (Index mode = 0; mode < basis_modes; ++mode) {
          work[first + mode] = retained[first + mode] != 0
                                   ? sum(order,
                                         [&](Index i) {
                                           return basisEntry(factor, i, mode) *
                                                  rhs.at(first + i, col);
                                         }) /
                                         modes[first + mode].value
                                   : 0;
        }
        for (Index i = 0; i < order; ++i) {
          values[first + i] = sum(basis_modes, [&](Index mode) {
            return retained[first + mode] != 0
                       ? basisEntry(factor, i, mode) * work[first + mode]
                       : 0;
          });
        }
        first += order;
      }
    }
    for (Index i = 0; i < n; ++i) {
      if (!representable(values[i])) {
        return fail<LinearEstimate>(ErrorCode::NonFiniteResult,
                                    "covariance solution is not representable");
      }
      result.parameters.at(i, col) = static_cast<Scalar>(values[i]);
      values[i] = result.parameters.at(i, col);
    }
    apply(values, work);
    for (Index i = 0; i < n; ++i) {
      work[i] -= rhs.at(i, col);
    }
    const Wide residual = norm(work);
    if (!representable(residual)) {
      return fail<LinearEstimate>(ErrorCode::NonFiniteResult,
                                  "covariance residual is not representable");
    }
    result.residual_norms.at(col) = static_cast<Scalar>(residual);
  }
  return Result<LinearEstimate>::success(std::move(result));
}

Result<Bounds> jacobiBounds(const StructuredSpec& spec,
                            std::span<Wide> diagonal, SolvePolicy policy,
                            Limits limits, Mode& maximum) {
  const Index n = diagonal.size();
  Bounds bounds{infinity, 0};
  const Wide slack = 64 * wide_epsilon * static_cast<Wide>(n);
  switch (spec.kind) {
  case StructureKind::diagonal_low_rank: {
    Wide highest_diagonal = 0;
    for (Index i = 0; i < n; ++i) {
      const Wide value = spec.diagonal_or_lags.at(i);
      diagonal[i] = value;
      bounds.lower = std::min(bounds.lower, value);
      highest_diagonal = std::max(highest_diagonal, value);
    }
    Wide update = 0;
    if (spec.low_rank_or_points) {
      const auto u = *spec.low_rank_or_points;
      for (Index i = 0; i < n; ++i) {
        const Wide energy = sum(u.columns(), [&](Index k) {
          const Wide value = u.at(i, k);
          return value * value;
        });
        diagonal[i] += energy;
        update += energy;
      }
    }
    bounds.upper = (highest_diagonal + update) * (1 + slack);
    if (update == 0) {
      maximum = exact(highest_diagonal);
      bounds.upper = highest_diagonal;
    }
    break;
  }
  case StructureKind::toeplitz:
  case StructureKind::gaussian_kernel: {
    const bool toeplitz = spec.kind == StructureKind::toeplitz;
    const Index width = toeplitz ? bandwidth(spec.diagonal_or_lags) : n - 1;
    const Wide d =
        toeplitz ? spec.diagonal_or_lags.at(0) : spec.kernel_scale.value_or(0);
    const Wide kernel_width =
        spec.low_rank_or_points
            ? 1 + static_cast<Wide>(spec.low_rank_or_points->columns())
            : 1;
    std::ranges::fill(diagonal, d);
    for (Index i = 0; i < n; ++i) {
      const Index first = i - std::min(i, width);
      const Index last = i + std::min(n - i - 1, width);
      const Wide radius = sum(last - first + 1, [&](Index offset) {
        const Index j = first + offset;
        if (i == j) {
          return Wide{0};
        }
        return toeplitz ? std::abs(static_cast<Wide>(
                              spec.diagonal_or_lags.at(i < j ? j - i : i - j)))
                        : kernelEntry(spec, i, j);
      });
      const Wide error = slack * kernel_width * (d + radius);
      bounds.lower = std::min(bounds.lower, d - radius - error);
      bounds.upper = std::max(bounds.upper, d + radius + error);
    }
    if (width == 0 || n == 1) {
      bounds = {d, d};
      maximum = exact(d);
    }
    break;
  }
  case StructureKind::block:
  case StructureKind::kronecker: {
    auto decompositions = factorSpectra(spec, policy, limits);
    if (!decompositions.hasValue()) {
      return forward<Bounds>(decompositions.status());
    }
    const auto& factors = decompositions.value();
    if (spec.kind == StructureKind::block) {
      Index first = 0;
      for (Index k = 0; k < factors.size(); ++k) {
        const auto& factor = factors[k];
        for (Index i = 0; i < factor.values.size(); ++i) {
          diagonal[first++] = spec.factors[k].at(i, i);
        }
        for (const Mode m : factor.values) {
          bounds.lower = std::min(bounds.lower, m.lower);
          maximum = {std::max(maximum.value, m.value),
                     std::max(maximum.lower, m.lower),
                     std::max(maximum.upper, m.upper)};
        }
      }
    } else {
      const auto factorMaximum = [&](Index i, Wide Mode::*field) {
        Wide largest = 0;
        for (const Mode mode : factors[i].values) {
          largest = std::max(largest, mode.*field);
        }
        return largest;
      };
      maximum = {
          scaledProduct(
              factors.size(),
              [&](Index i) { return factorMaximum(i, &Mode::value); }),
          scaledProduct(
              factors.size(),
              [&](Index i) { return factorMaximum(i, &Mode::lower); }, -1),
          scaledProduct(
              factors.size(),
              [&](Index i) { return factorMaximum(i, &Mode::upper); }, 1)};
      bounds.lower = scaledProduct(
          factors.size(),
          [&](Index i) { return extremes(factors[i].values).lower; }, -1);
      for (Index i = 0; i < n; ++i) {
        Index stride = n;
        diagonal[i] = scaledProduct(spec.factors.size(), [&](Index k) {
          const auto& factor = spec.factors[k];
          stride /= factor.rows();
          const Index local = (i / stride) % factor.rows();
          return factor.at(local, local);
        });
      }
    }
    bounds.upper = maximum.upper;
    break;
  }
  }
  return Result<Bounds>::success(bounds);
}

Result<Wide> refineMaximum(const Action& action, Index n, Wide upper,
                           Wide lower, Wide arithmetic_scale, Index rounds,
                           Limits limits, EvaluationBudget& evaluations) {
  if (!reserve<Wide>(limits, n, 3)) {
    return fail<Wide>(ErrorCode::ResourceLimit,
                      "spectral refinement workspace exceeds limits");
  }
  WideVector vector(n);
  WideVector next(n);
  WideVector row_sums(n);
  // Divide by a power of two that bounds the exact operator norm.
  const int exponent = std::ilogb(upper) + 1;
  const Wide normalized_upper = std::scalbn(upper, -exponent);
  Wide lower_bound = std::scalbn(lower, -exponent);
  Wide upper_bound = normalized_upper;
  const Wide envelope = 64 * epsilon * static_cast<Wide>(n) * normalized_upper;
  const auto apply_scaled = [&] {
    action(vector, next);
    for (Wide& value : next) {
      value = std::scalbn(value, -exponent);
    }
  };
  Index power = 1;
  for (Index round = 0; round < rounds; ++round) {
    if (power == std::numeric_limits<Index>::max() ||
        power + 1 > evaluations.remaining() / n) {
      return fail<Wide>(ErrorCode::ResourceLimit,
                        "spectral operator evaluations exceed limits");
    }
    if (!evaluations.charge(n * (power + 1))) {
      return fail<Wide>(ErrorCode::ResourceLimit,
                        "spectral operator evaluations exceed limits");
    }
    const Wide unit_error = 32 * wide_epsilon * arithmetic_scale *
                            static_cast<Wide>(n) * normalized_upper;
    std::ranges::fill(row_sums, 0);
    for (Index coordinate = 0; coordinate < n; ++coordinate) {
      std::ranges::fill(vector, 0);
      vector[coordinate] = 1;
      Wide error = 0;
      for (Index k = 0; k < power; ++k) {
        const Wide length = norm(vector);
        apply_scaled();
        error = std::nextafter(normalized_upper * error + unit_error * length,
                               infinity);
        vector.swap(next);
      }
      for (Index i = 0; i < n; ++i) {
        row_sums[i] =
            std::nextafter(row_sums[i] + std::abs(vector[i]) + error, infinity);
      }
      const Wide length = norm(vector);
      if (std::isfinite(length) && length > 0) {
        for (Wide& value : vector) {
          value /= length;
        }
        apply_scaled();
        const Wide quotient =
            sum(n, [&](Index i) { return vector[i] * next[i]; });
        lower_bound = std::max(lower_bound, quotient - 4 * unit_error);
      }
    }
    const Wide maximum_sum = *std::ranges::max_element(row_sums);
    if (!std::isfinite(maximum_sum)) {
      break;
    }
    const Wide root = std::pow(maximum_sum, 1 / static_cast<Wide>(power));
    upper_bound = std::min(upper_bound, root + 8 * wide_epsilon * root);
    if (lower_bound <= upper_bound && upper_bound - lower_bound <= envelope) {
      return Result<Wide>::success(
          std::scalbn(std::midpoint(lower_bound, upper_bound), exponent));
    }
    if (power > std::numeric_limits<Index>::max() / 2) {
      break;
    }
    power *= 2;
  }
  return fail<Wide>(ErrorCode::NoConvergence,
                    "largest covariance eigenvalue interval remains too wide");
}

Result<LinearEstimate> iterativeSolve(const StructuredSpec& spec, Index n,
                                      ConstMatrixView rhs, SolvePolicy policy,
                                      Limits limits) {
  const Index p = rhs.columns();
  if (!reserve<Scalar>(limits, n, p) || !reserve<Scalar>(limits, p) ||
      !reserve<Wide>(limits, n) ||
      !reserve<Wide>(limits, Action::scratchSize(spec, n))) {
    return fail<LinearEstimate>(ErrorCode::ResourceLimit,
                                "iterative covariance storage exceeds limits");
  }
  if (n == 0) {
    return Result<LinearEstimate>::success({.parameters = Matrix(0, p),
                                            .covariance = std::nullopt,
                                            .identified_basis = std::nullopt,
                                            .residual_norms = Vector(p),
                                            .rank = 0,
                                            .rank_cutoff = policy.rank_abs});
  }
  WideVector diagonal(n);
  WideVector scratch(Action::scratchSize(spec, n));
  const Action apply(spec, n, scratch);
  Mode maximum{};
  auto certificate = jacobiBounds(spec, diagonal, policy, limits, maximum);
  if (!certificate.hasValue()) {
    return forward<LinearEstimate>(certificate.status());
  }
  const Bounds bounds = certificate.value();
  const Wide relative_ceiling =
      policy.rank_rel == 0 ? 0 : policy.rank_rel * bounds.upper;
  if (!std::isfinite(bounds.upper) ||
      !(bounds.lower > std::max<Wide>(policy.rank_abs, relative_ceiling)) ||
      bounds.lower / bounds.upper < policy.reciprocal_condition_floor ||
      std::ranges::any_of(diagonal, [](Wide value) {
        return !std::isfinite(value) || value <= 0;
      })) {
    return fail<LinearEstimate>(
        ErrorCode::NoConvergence,
        "iterative covariance solve cannot certify full SPD support");
  }
  EvaluationBudget evaluations(
      std::min<Index>(limits.evaluations, policy.iteration.max_evaluations));
  Wide cutoff = policy.rank_abs;
  if (relative_ceiling > cutoff) {
    const Wide envelope = 64 * epsilon * static_cast<Wide>(n) * bounds.upper;
    if (maximum.upper == 0 || maximum.upper - maximum.lower > envelope) {
      const Wide arithmetic_scale =
          spec.low_rank_or_points
              ? 1 + static_cast<Wide>(spec.low_rank_or_points->columns())
              : 1;
      const auto refined =
          refineMaximum(apply, n, bounds.upper, bounds.lower, arithmetic_scale,
                        policy.iteration.max_iterations, limits, evaluations);
      if (!refined.hasValue()) {
        return forward<LinearEstimate>(refined.status());
      }
      maximum.value = refined.value();
    }
    cutoff = std::max<Wide>(cutoff, policy.rank_rel * maximum.value);
  }
  if (!representable(cutoff)) {
    return fail<LinearEstimate>(
        ErrorCode::NonFiniteResult,
        "iterative covariance cutoff is not representable");
  }
  if (p != 0 && !reserve<Wide>(limits, n, 5)) {
    return fail<LinearEstimate>(ErrorCode::ResourceLimit,
                                "PCG vectors exceed limits");
  }
  LinearEstimate result{.parameters = Matrix(n, p),
                        .covariance = std::nullopt,
                        .identified_basis = std::nullopt,
                        .residual_norms = Vector(p),
                        .rank = n,
                        .rank_cutoff = static_cast<Scalar>(cutoff)};
  if (p == 0) {
    return Result<LinearEstimate>::success(std::move(result));
  }
  WideVector x(n);
  WideVector r(n);
  WideVector z(n);
  WideVector direction(n);
  WideVector image(n);
  const auto operator_action = [&](std::span<const Wide> input,
                                   std::span<Wide> output) {
    apply(input, output);
    return Status::success();
  };
  const auto precondition = [&](std::span<const Wide> input,
                                std::span<Wide> output) {
    for (Index i = 0; i < input.size(); ++i) {
      output[i] = input[i] / diagonal[i];
    }
    return Status::success();
  };
  for (Index col = 0; col < p; ++col) {
    std::ranges::fill(x, 0);
    const auto report = linear::detail::runPcg(
        [&](Index i) { return rhs.at(i, col); }, operator_action, precondition,
        policy.iteration, evaluations,
        {.solution = x,
         .residual = r,
         .preconditioned = z,
         .direction = direction,
         .image = image});
    if (!report.hasValue()) {
      return forward<LinearEstimate>(report.status());
    }
    if (!report.value().converged) {
      return fail<LinearEstimate>(
          ErrorCode::NoConvergence,
          "PCG did not meet its true-residual tolerance");
    }
    for (Index i = 0; i < n; ++i) {
      result.parameters.at(i, col) = static_cast<Scalar>(x[i]);
    }
    result.residual_norms.at(col) =
        static_cast<Scalar>(report.value().residual_norm);
  }
  return Result<LinearEstimate>::success(std::move(result));
}

Status checkPolicy(SolvePolicy policy) {
  if (!linear::detail::validNumericalPolicy(policy)) {
    return Status::failure(ErrorCode::InvalidTolerance,
                           "invalid structured covariance tolerance");
  }
  if (policy.rank != RankPolicy::reject &&
      policy.rank != RankPolicy::supported_subspace &&
      policy.rank != RankPolicy::minimum_norm) {
    return Status::failure(ErrorCode::InvalidDomain,
                           "invalid covariance rank policy");
  }
  return Status::success();
}

Result<Index> validateOwned(const StructuredCovariance& covariance,
                            const StructuredSpec& spec) {
  auto size = validate(spec);
  if (size.hasValue() && size.value() != covariance.dimension) {
    return fail<Index>(
        ErrorCode::DimensionMismatch,
        "stored covariance dimension does not match its payload");
  }
  return size;
}

} // namespace

Result<StructuredCovariance> structuredCovariance(const StructuredSpec& spec,
                                                  SolvePolicy policy,
                                                  Limits limits) {
  if (const auto status = checkPolicy(policy); !status.ok()) {
    return forward<StructuredCovariance>(status);
  }
  const auto dimension = validate(spec);
  if (!dimension.hasValue()) {
    return forward<StructuredCovariance>(dimension.status());
  }
  const Index n = dimension.value();
  Limits copy_budget = limits;
  if (!reserve<Scalar>(copy_budget, spec.diagonal_or_lags.size()) ||
      (spec.low_rank_or_points &&
       !reserve<Scalar>(copy_budget, spec.low_rank_or_points->rows(),
                        spec.low_rank_or_points->columns())) ||
      !reserve<Matrix>(copy_budget, spec.factors.size(), 2)) {
    return fail<StructuredCovariance>(
        ErrorCode::ResourceLimit,
        "owned covariance payload exceeds storage limits");
  }
  for (const auto& factor : spec.factors) {
    if (!reserve<Scalar>(copy_budget, factor.size(), 2)) {
      return fail<StructuredCovariance>(
          ErrorCode::ResourceLimit, "copied covariance factors exceed limits");
    }
  }
  if (spec.kind == StructureKind::toeplitz &&
      bandwidth(spec.diagonal_or_lags) != 0) {
    auto spectrum = symmetricSpectrum(
        n,
        [&](Index i, Index j) {
          return static_cast<Wide>(
              spec.diagonal_or_lags.at(i < j ? j - i : i - j));
        },
        policy, limits);
    if (!spectrum.hasValue()) {
      return forward<StructuredCovariance>(spectrum.status());
    }
    if (const auto status = certifyPsd(spectrum.value().values); !status.ok()) {
      return forward<StructuredCovariance>(status);
    }
  }
  for (const auto& factor : spec.factors) {
    auto spectrum = symmetricSpectrum(
        factor.rows(),
        [&](Index i, Index j) { return symmetricEntry(factor, i, j); }, policy,
        limits);
    if (!spectrum.hasValue()) {
      return forward<StructuredCovariance>(spectrum.status());
    }
    if (const auto status = certifyPsd(spectrum.value().values); !status.ok()) {
      return forward<StructuredCovariance>(status);
    }
  }

  std::vector<Matrix> canonical;
  canonical.reserve(spec.factors.size());
  for (const auto& factor : spec.factors) {
    Matrix matrix(factor.rows(), factor.columns());
    for (Index i = 0; i < factor.rows(); ++i) {
      for (Index j = 0; j <= i; ++j) {
        const auto value = static_cast<Scalar>(symmetricEntry(factor, i, j));
        matrix.at(i, j) = value;
        matrix.at(j, i) = value;
      }
    }
    canonical.push_back(std::move(matrix));
  }
  auto factors = Array<Matrix>::copy(canonical, limits);
  if (!factors.hasValue()) {
    return forward<StructuredCovariance>(factors.status());
  }
  Vector lags(spec.diagonal_or_lags.size());
  for (Index i = 0; i < lags.size(); ++i) {
    lags.at(i) = spec.diagonal_or_lags.at(i);
  }
  std::optional<Matrix> auxiliary;
  if (spec.low_rank_or_points) {
    const auto values = *spec.low_rank_or_points;
    auto& payload = auxiliary.emplace(values.rows(), values.columns());
    for (Index i = 0; i < values.rows(); ++i) {
      for (Index j = 0; j < values.columns(); ++j) {
        payload.at(i, j) = values.at(i, j);
      }
    }
  }
  return Result<StructuredCovariance>::success(
      {spec.kind, n, std::move(lags), std::move(auxiliary),
       std::move(factors).value(), spec.kernel_scale, spec.kernel_length});
}

Result<StructuredCovariance>
kroneckerCovariance(std::span<const Matrix> factors, SolvePolicy policy,
                    Limits limits) {
  const Vector empty;
  return structuredCovariance({StructureKind::kronecker, empty.view(),
                               std::nullopt, factors, std::nullopt,
                               std::nullopt},
                              policy, limits);
}

Result<Vector> applyStructuredCovariance(const StructuredCovariance& covariance,
                                         ConstVectorView value, Limits limits) {
  const auto spec = borrow(covariance);
  const auto size = validateOwned(covariance, spec);
  if (!size.hasValue()) {
    return forward<Vector>(size.status());
  }
  const Index n = size.value();
  if (value.size() != n) {
    return fail<Vector>(ErrorCode::DimensionMismatch,
                        "structured covariance vector size differs");
  }
  if (!finite(value)) {
    return fail<Vector>(ErrorCode::NonFiniteInput,
                        "structured covariance vector is non-finite");
  }
  if (!reserve<Scalar>(limits, n) || !reserve<Wide>(limits, n, 2) ||
      !reserve<Wide>(limits, Action::scratchSize(spec, n))) {
    return fail<Vector>(ErrorCode::ResourceLimit,
                        "structured covariance action exceeds limits");
  }
  WideVector input(n);
  WideVector output(n);
  WideVector scratch(Action::scratchSize(spec, n));
  for (Index i = 0; i < n; ++i) {
    input[i] = value.at(i);
  }
  Action(spec, n, scratch)(input, output);
  Vector result(n);
  for (Index i = 0; i < n; ++i) {
    if (!representable(output[i])) {
      return fail<Vector>(ErrorCode::NonFiniteResult,
                          "structured covariance action is not representable");
    }
    result.at(i) = static_cast<Scalar>(output[i]);
  }
  return Result<Vector>::success(std::move(result));
}

Result<LinearEstimate>
solveStructuredCovariance(const StructuredCovariance& covariance,
                          ConstMatrixView rhs, StructuredSolveKind mode,
                          SolvePolicy policy, Limits limits) {
  if (const auto status = checkPolicy(policy); !status.ok()) {
    return forward<LinearEstimate>(status);
  }
  if (mode != StructuredSolveKind::factorized &&
      mode != StructuredSolveKind::iterative) {
    return fail<LinearEstimate>(ErrorCode::InvalidDomain,
                                "invalid structured solve mode");
  }
  const auto spec = borrow(covariance);
  const auto size = validateOwned(covariance, spec);
  if (!size.hasValue()) {
    return forward<LinearEstimate>(size.status());
  }
  if (size.value() != rhs.rows()) {
    return fail<LinearEstimate>(ErrorCode::DimensionMismatch,
                                "covariance RHS height differs from order");
  }
  if (!finite(rhs)) {
    return fail<LinearEstimate>(ErrorCode::NonFiniteInput,
                                "structured covariance RHS is non-finite");
  }
  return mode == StructuredSolveKind::factorized
             ? factorizedSolve(spec, size.value(), rhs, policy, limits)
             : iterativeSolve(spec, size.value(), rhs, policy, limits);
}

} // namespace vss::math::covariance
