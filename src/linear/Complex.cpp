#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/linear/Complex.hpp>

namespace vss::math::linear {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedProduct(Extent a, Extent b, Extent& result) noexcept {
  if (b != 0U && a > std::numeric_limits<Extent>::max() / b) {
    return false;
  }
  result = a * b;
  return true;
}

Status validateView(ComplexMatrixView matrix) {
  if (matrix.rows == 0U || matrix.columns == 0U) {
    return Status::success();
  }
  Extent row_offset = 0U;
  Extent column_offset = 0U;
  if (!checkedProduct(matrix.rows - 1U, matrix.row_stride, row_offset) ||
      !checkedProduct(matrix.columns - 1U, matrix.column_stride,
                      column_offset) ||
      row_offset > std::numeric_limits<Extent>::max() - column_offset) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "complex matrix view offset overflows");
  }
  const Extent last = row_offset + column_offset;
  if (last >= matrix.values.size()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "complex matrix view exceeds its backing span");
  }
  for (Index row = 0U; row < matrix.rows; ++row) {
    for (Index column = 0U; column < matrix.columns; ++column) {
      const Complex value =
          matrix
              .values[row * matrix.row_stride + column * matrix.column_stride];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "complex matrix contains a non-finite part");
      }
    }
  }
  return Status::success();
}

template <class T> Result<T> copyFailure(const Status& status) {
  return Result<T>::failure(status.code(), std::string(status.message()));
}

bool workspaceAllowed(Extent elements, std::size_t copies,
                      Limits limits) noexcept;

Result<ComplexMatrix> makeMatrix(Extent rows, Extent columns,
                                 std::vector<Complex>&& values, Limits limits) {
  Extent count = 0U;
  if (!checkedProduct(rows, columns, count)) {
    return fail<ComplexMatrix>(ErrorCode::SizeOverflow,
                               "complex matrix output extent overflows");
  }
  if (count > limits.elements || count > limits.bytes / sizeof(Complex)) {
    return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                               "complex matrix output exceeds limits");
  }
  if (!workspaceAllowed(count, 2U, limits)) {
    return fail<ComplexMatrix>(
        ErrorCode::ResourceLimit,
        "complex matrix construction workspace exceeds limits");
  }
  auto array = Array<Complex>::copy(values, limits);
  if (!array) {
    return copyFailure<ComplexMatrix>(array.status());
  }
  return Result<ComplexMatrix>::success(
      ComplexMatrix{rows, columns, std::move(array).value()});
}

Complex at(ComplexMatrixView matrix, Index row, Index column) noexcept {
  return matrix.values[row * matrix.row_stride + column * matrix.column_stride];
}

using Dense = std::vector<Complex>;
using RealDense = std::vector<Scalar>;

constexpr std::size_t kTallSvdWorkspaceCopies = 20U;
constexpr std::size_t kTallSvdSquareWorkspaceCopies = 4U;
constexpr std::size_t kSvdPeakWorkspaceCopies = 24U;
constexpr std::size_t kHermitianWorkspaceCopies = 12U;
constexpr std::size_t kHermitianVectorWorkspaceCopies = 12U;

bool validPolicy(SolvePolicy policy) noexcept {
  return std::isfinite(policy.iteration.abs) &&
         std::isfinite(policy.iteration.rel) && policy.iteration.abs >= 0.0 &&
         policy.iteration.rel >= 0.0 && policy.iteration.max_iterations > 0U &&
         std::isfinite(policy.rank_abs) && policy.rank_abs >= 0.0 &&
         std::isfinite(policy.rank_rel) && policy.rank_rel >= 0.0 &&
         std::isfinite(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor >= 0.0 &&
         policy.reciprocal_condition_floor <= 1.0 &&
         (policy.rank == RankPolicy::reject ||
          policy.rank == RankPolicy::supported_subspace ||
          policy.rank == RankPolicy::minimum_norm);
}

Scalar maxComponent(const Dense& a) noexcept {
  Scalar scale = 0.0;
  for (const Complex z : a) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
      return std::numeric_limits<Scalar>::infinity();
    }
    scale = std::max({scale, std::abs(z.real()), std::abs(z.imag())});
  }
  return scale;
}

Complex scaledDivide(Complex numerator, Complex denominator) noexcept {
  const Scalar real = denominator.real();
  const Scalar imaginary = denominator.imag();
  if (std::abs(real) >= std::abs(imaginary)) {
    const Scalar ratio = imaginary / real;
    const Scalar divisor = 1.0 + ratio * ratio;
    const Scalar a = numerator.real() / real;
    const Scalar b = numerator.imag() / real;
    return {(a + b * ratio) / divisor, (b - a * ratio) / divisor};
  }
  const Scalar ratio = real / imaginary;
  const Scalar divisor = 1.0 + ratio * ratio;
  const Scalar a = numerator.real() / imaginary;
  const Scalar b = numerator.imag() / imaginary;
  return {(a * ratio + b) / divisor, (b * ratio - a) / divisor};
}

Scalar rescaleProduct(Scalar leftScale, Scalar rightScale,
                      long double normalizedProduct) noexcept {
  if (normalizedProduct == 0.0L || leftScale == 0.0 || rightScale == 0.0) {
    return 0.0;
  }
  int leftExponent = 0;
  int rightExponent = 0;
  int productExponent = 0;
  const long double leftMantissa = std::frexp(leftScale, &leftExponent);
  const long double rightMantissa = std::frexp(rightScale, &rightExponent);
  const long double productMantissa =
      std::frexp(normalizedProduct, &productExponent);
  const long double value =
      std::ldexp(leftMantissa * rightMantissa * productMantissa,
                 leftExponent + rightExponent + productExponent);
  return static_cast<Scalar>(value);
}

Scalar normComplexScaled(const Dense& a, Scalar divisor) noexcept {
  if (divisor == 0.0) {
    return 0.0;
  }
  if (!std::isfinite(divisor)) {
    return std::numeric_limits<Scalar>::infinity();
  }
  Scalar scale = 0.0;
  Scalar sum = 1.0;
  for (const Complex z : a) {
    for (const Scalar component : {std::abs(z.real()), std::abs(z.imag())}) {
      const Scalar value = component / divisor;
      if (!std::isfinite(value)) {
        return std::numeric_limits<Scalar>::infinity();
      }
      if (value == 0.0) {
        continue;
      }
      if (scale < value) {
        const Scalar ratio = scale / value;
        sum = 1.0 + sum * ratio * ratio;
        scale = value;
      } else {
        const Scalar ratio = value / scale;
        sum += ratio * ratio;
      }
    }
  }
  return scale == 0.0 ? 0.0 : scale * std::sqrt(sum);
}

Scalar maxComponent(ComplexMatrixView a) noexcept {
  Scalar scale = 0.0;
  for (Index row = 0U; row < a.rows; ++row) {
    for (Index column = 0U; column < a.columns; ++column) {
      const Complex z = at(a, row, column);
      if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
        return std::numeric_limits<Scalar>::infinity();
      }
      scale = std::max({scale, std::abs(z.real()), std::abs(z.imag())});
    }
  }
  return scale;
}

long double normViewScaled(ComplexMatrixView a, Scalar divisor) noexcept {
  if (divisor == 0.0) {
    return 0.0L;
  }
  long double sum = 0.0L;
  for (Index row = 0U; row < a.rows; ++row) {
    for (Index column = 0U; column < a.columns; ++column) {
      const Complex z = at(a, row, column) / divisor;
      sum += static_cast<long double>(z.real()) * z.real() +
             static_cast<long double>(z.imag()) * z.imag();
    }
  }
  return std::sqrt(sum);
}

long double solveBackwardError(ComplexMatrixView a, ComplexMatrixView b,
                               const Dense& x, Index rhs,
                               bool relativeToRightHandSide = false) noexcept {
  const Scalar aScale = maxComponent(a);
  const Scalar xScale = maxComponent(x);
  const Scalar bScale = maxComponent(b);
  if (!std::isfinite(aScale) || !std::isfinite(xScale) ||
      !std::isfinite(bScale)) {
    return std::numeric_limits<long double>::infinity();
  }
  if (aScale == 0.0 || xScale == 0.0) {
    return bScale == 0.0 ? 0.0L : 1.0L;
  }

  int aExponent = 0;
  int xExponent = 0;
  int bExponent = 0;
  const long double aMantissa = std::frexp(aScale, &aExponent);
  const long double xMantissa = std::frexp(xScale, &xExponent);
  const long double bMantissa = std::frexp(bScale, &bExponent);
  const int productExponent = aExponent + xExponent;
  const int commonExponent = std::max(productExponent, bExponent);
  const long double alpha =
      std::ldexp(aMantissa * xMantissa, productExponent - commonExponent);
  const long double beta =
      bScale == 0.0 ? 0.0L : std::ldexp(bMantissa, bExponent - commonExponent);

  long double residualSum = 0.0L;
  for (Index row = 0U; row < a.rows; ++row) {
    for (Index column = 0U; column < rhs; ++column) {
      long double real = 0.0L;
      long double imaginary = 0.0L;
      for (Index inner = 0U; inner < a.columns; ++inner) {
        const Complex av = at(a, row, inner) / aScale;
        const Complex xv = x[inner * rhs + column] / xScale;
        real += static_cast<long double>(av.real()) * xv.real() -
                static_cast<long double>(av.imag()) * xv.imag();
        imaginary += static_cast<long double>(av.real()) * xv.imag() +
                     static_cast<long double>(av.imag()) * xv.real();
      }
      const Complex bv =
          bScale == 0.0 ? Complex{} : at(b, row, column) / bScale;
      real = alpha * real - beta * bv.real();
      imaginary = alpha * imaginary - beta * bv.imag();
      residualSum += real * real + imaginary * imaginary;
    }
  }
  const long double denominator =
      relativeToRightHandSide
          ? beta * normViewScaled(b, bScale)
          : alpha * normViewScaled(a, aScale) * normComplexScaled(x, xScale) +
                beta * normViewScaled(b, bScale);
  return denominator == 0.0L ? std::sqrt(residualSum)
                             : std::sqrt(residualSum) / denominator;
}

Scalar normalizedDifference(const Dense& a, const Dense& b) noexcept {
  const Scalar divisor = std::max(maxComponent(a), maxComponent(b));
  if (divisor == 0.0) {
    return 0.0;
  }
  if (!std::isfinite(divisor)) {
    return std::numeric_limits<Scalar>::infinity();
  }
  Scalar scale = 0.0;
  Scalar sum = 1.0;
  for (Index i = 0U; i < a.size(); ++i) {
    const Complex difference = a[i] / divisor - b[i] / divisor;
    for (const Scalar component :
         {std::abs(difference.real()), std::abs(difference.imag())}) {
      if (component == 0.0) {
        continue;
      }
      if (scale < component) {
        const Scalar ratio = scale / component;
        sum = 1.0 + sum * ratio * ratio;
        scale = component;
      } else {
        const Scalar ratio = component / scale;
        sum += ratio * ratio;
      }
    }
  }
  const Scalar numerator = scale == 0.0 ? 0.0 : scale * std::sqrt(sum);
  const Scalar denominator = normComplexScaled(a, divisor);
  return denominator == 0.0 ? numerator : numerator / denominator;
}

Scalar normComplex(const Dense& a) noexcept {
  Scalar scale = 0.0;
  Scalar sum = 1.0;
  for (const Complex z : a) {
    for (const Scalar component : {std::abs(z.real()), std::abs(z.imag())}) {
      if (!std::isfinite(component)) {
        return std::numeric_limits<Scalar>::infinity();
      }
      if (component == 0.0) {
        continue;
      }
      if (scale < component) {
        const Scalar ratio = scale / component;
        sum = 1.0 + sum * ratio * ratio;
        scale = component;
      } else {
        const Scalar ratio = component / scale;
        sum += ratio * ratio;
      }
    }
  }
  return scale == 0.0 ? 0.0 : scale * std::sqrt(sum);
}

bool workspaceAllowed(Extent elements, std::size_t copies,
                      Limits limits) noexcept {
  if (elements > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
    return false;
  }
  const std::size_t bytes = elements * sizeof(Complex);
  return copies == 0U || bytes <= limits.bytes / copies;
}

bool workspaceAllowedCombined(
    Limits limits,
    std::initializer_list<std::pair<Extent, std::size_t>> buffers) noexcept {
  std::size_t bytes = 0U;
  for (const auto& [elements, copies] : buffers) {
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(Complex)) {
      return false;
    }
    const std::size_t one = elements * sizeof(Complex);
    if (bytes > limits.bytes ||
        (copies != 0U && one > (limits.bytes - bytes) / copies)) {
      return false;
    }
    bytes += one * copies;
  }
  return bytes <= limits.bytes;
}

Dense copyDense(ComplexMatrixView view) {
  Dense result(view.rows * view.columns);
  for (Index row = 0U; row < view.rows; ++row) {
    for (Index column = 0U; column < view.columns; ++column) {
      result[row * view.columns + column] = at(view, row, column);
    }
  }
  return result;
}

// Applies a complex Householder reflector H=I-beta*v*v* from the left.
void applyLeft(Dense& a, Index rows, Index columns, Index firstRow,
               Index firstColumn, const Dense& v, Scalar beta) {
  const Index length = rows - firstRow;
  for (Index column = firstColumn; column < columns; ++column) {
    Complex projection{};
    for (Index i = 0U; i < length; ++i) {
      projection += std::conj(v[i]) * a[(firstRow + i) * columns + column];
    }
    projection *= beta;
    for (Index i = 0U; i < length; ++i) {
      a[(firstRow + i) * columns + column] -= v[i] * projection;
    }
  }
}

void applyRight(Dense& a, Index rows, Index columns, Index firstRow,
                Index firstColumn, const Dense& v, Scalar beta) {
  const Index length = columns - firstColumn;
  for (Index row = firstRow; row < rows; ++row) {
    Complex projection{};
    for (Index j = 0U; j < length; ++j) {
      projection += a[row * columns + firstColumn + j] * v[j];
    }
    projection *= beta;
    for (Index j = 0U; j < length; ++j) {
      a[row * columns + firstColumn + j] -= projection * std::conj(v[j]);
    }
  }
}

bool reflector(const Dense& x, Dense& v, Scalar& beta, Complex& alpha) {
  const Index length = x.size();
  Scalar scale = 0.0;
  for (Index i = 0U; i < length; ++i) {
    const Complex z = x[i];
    scale = std::max({scale, std::abs(z.real()), std::abs(z.imag())});
  }
  if (scale == 0.0) {
    v.assign(length, Complex{});
    beta = 0.0;
    alpha = {};
    return true;
  }
  Scalar sumSquares = 0.0;
  for (Index i = 0U; i < length; ++i) {
    const Complex z = x[i] / scale;
    sumSquares += std::norm(z);
  }
  const Scalar scaledNorm = std::sqrt(sumSquares);
  const Complex first = x[0] / scale;
  const Scalar firstMagnitude = std::hypot(first.real(), first.imag());
  const Complex direction =
      firstMagnitude == 0.0 ? Complex{-1.0, 0.0} : -first / firstMagnitude;
  const Complex scaledAlpha = direction * scaledNorm;
  v.resize(length);
  for (Index i = 0U; i < length; ++i) {
    v[i] = x[i] / scale;
  }
  v[0] -= scaledAlpha;
  Scalar vnorm2 = 0.0;
  for (const Complex z : v) {
    vnorm2 += std::norm(z);
  }
  if (vnorm2 == 0.0 || !std::isfinite(vnorm2)) {
    return false;
  }
  beta = 2.0 / vnorm2;
  alpha = scaledAlpha * scale;
  return std::isfinite(alpha.real()) && std::isfinite(alpha.imag()) &&
         std::isfinite(beta);
}

void multiplyRightByReflector(Dense& q, Index order, Index first,
                              const Dense& v, Scalar beta) {
  applyRight(q, order, order, 0U, first, v, beta);
}

// One-sided real Jacobi is used as the real bidiagonal SVD kernel.
bool realJacobiSvd(RealDense& b, Index rows, Index columns, RealDense& right,
                   Index maximumSweeps) {
  right.assign(columns * columns, 0.0);
  for (Index i = 0U; i < columns; ++i) {
    right[i * columns + i] = 1.0;
  }
  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  for (Index sweep = 0U; sweep < maximumSweeps; ++sweep) {
    bool changed = false;
    for (Index p = 0U; p < columns; ++p) {
      for (Index q = p + 1U; q < columns; ++q) {
        long double aa = 0.0L;
        long double bb = 0.0L;
        long double ab = 0.0L;
        for (Index i = 0U; i < rows; ++i) {
          const Scalar x = b[i * columns + p];
          const Scalar y = b[i * columns + q];
          aa += static_cast<long double>(x) * x;
          bb += static_cast<long double>(y) * y;
          ab += static_cast<long double>(x) * y;
        }
        if (ab == 0.0L || std::abs(ab) <= 8.0L * epsilon * std::sqrt(aa * bb)) {
          continue;
        }
        const long double tau = (bb - aa) / (2.0L * ab);
        const long double t =
            std::copysign(1.0L / (std::abs(tau) + std::hypot(1.0L, tau)), tau);
        const Scalar c = static_cast<Scalar>(1.0L / std::sqrt(1.0L + t * t));
        const Scalar s = static_cast<Scalar>(t * c);
        for (Index i = 0U; i < rows; ++i) {
          const Scalar x = b[i * columns + p];
          const Scalar y = b[i * columns + q];
          b[i * columns + p] = c * x - s * y;
          b[i * columns + q] = s * x + c * y;
        }
        for (Index i = 0U; i < columns; ++i) {
          const Scalar x = right[i * columns + p];
          const Scalar y = right[i * columns + q];
          right[i * columns + p] = c * x - s * y;
          right[i * columns + q] = s * x + c * y;
        }
        changed = true;
      }
    }
    if (!changed) {
      return true;
    }
  }
  return false;
}

struct TallSvd final {
  Dense left;
  std::vector<Scalar> singular;
  Dense right_adjoint;
};

Result<TallSvd> tallSvd(ComplexMatrixView input, SolvePolicy policy,
                        Limits limits) {
  const Index m = input.rows;
  const Index n = input.columns;
  Extent mn = 0U;
  Extent nn = 0U;
  if (!checkedProduct(m, n, mn) || !checkedProduct(n, n, nn)) {
    return fail<TallSvd>(ErrorCode::SizeOverflow,
                         "complex SVD workspace extent overflows");
  }
  if (!workspaceAllowedCombined(limits,
                                {{mn, kTallSvdWorkspaceCopies},
                                 {nn, kTallSvdSquareWorkspaceCopies}})) {
    return fail<TallSvd>(ErrorCode::ResourceLimit,
                         "complex SVD workspace exceeds limits");
  }
  Dense b = copyDense(input);
  Scalar scale = maxComponent(b);
  if (scale != 0.0) {
    for (Complex& z : b) {
      z /= scale;
    }
  }
  Dense leftReflectors(mn, Complex{});
  std::vector<Scalar> leftBetas(n, 0.0);
  Dense right(nn, Complex{});
  for (Index i = 0U; i < n; ++i) {
    right[i * n + i] = 1.0;
  }
  for (Index k = 0U; k < n; ++k) {
    Dense v;
    Scalar beta = 0.0;
    Complex alpha{};
    Dense column(m - k);
    for (Index i = 0U; i < m - k; ++i) {
      column[i] = b[(k + i) * n + k];
    }
    if (!reflector(column, v, beta, alpha)) {
      return fail<TallSvd>(ErrorCode::NoConvergence,
                           "complex SVD left reflector failed");
    }
    leftBetas[k] = beta;
    for (Index i = 0U; i < v.size(); ++i) {
      leftReflectors[k * m + i] = v[i];
    }
    applyLeft(b, m, n, k, k, v, beta);
    b[k * n + k] = alpha;
    for (Index i = k + 1U; i < m; ++i) {
      b[i * n + k] = {};
    }
    if (k + 1U < n) {
      const Index length = n - k - 1U;
      Dense rowConjugate(length);
      for (Index j = 0U; j < length; ++j) {
        rowConjugate[j] = std::conj(b[k * n + k + 1U + j]);
      }
      if (!reflector(rowConjugate, v, beta, alpha)) {
        return fail<TallSvd>(ErrorCode::NoConvergence,
                             "complex SVD right reflector failed");
      }
      applyRight(b, m, n, k, k + 1U, v, beta);
      applyRight(right, n, n, 0U, k + 1U, v, beta);
      b[k * n + k + 1U] = std::conj(alpha);
      for (Index j = k + 2U; j < n; ++j) {
        b[k * n + j] = {};
      }
    }
  }
  // Transfer diagonal phases into the accumulated unitary factors so the
  // bidiagonal problem sent to the real SVD kernel has real entries.
  Dense rowPhase(m, Complex{1.0, 0.0});
  Dense columnPhase(n, Complex{1.0, 0.0});
  const auto unitPhase = [](Complex value) {
    const Scalar magnitude = std::hypot(value.real(), value.imag());
    return magnitude == 0.0 ? Complex{1.0, 0.0} : value / magnitude;
  };
  if (n > 0U) {
    rowPhase[0] = unitPhase(b[0]);
    for (Index k = 0U; k + 1U < n; ++k) {
      columnPhase[k + 1U] =
          rowPhase[k] * std::conj(unitPhase(b[k * n + k + 1U]));
      rowPhase[k + 1U] =
          unitPhase(b[(k + 1U) * n + k + 1U]) * columnPhase[k + 1U];
    }
  }
  for (Index i = 0U; i < m; ++i) {
    for (Index j = 0U; j < n; ++j) {
      b[i * n + j] *= std::conj(rowPhase[i]) * columnPhase[j];
    }
  }
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      right[i * n + j] *= columnPhase[j];
    }
  }
  RealDense bidiagonal(mn, 0.0);
  for (Index i = 0U; i < mn; ++i) {
    bidiagonal[i] = b[i].real();
  }
  RealDense rightReal;
  if (!realJacobiSvd(bidiagonal, m, n, rightReal,
                     policy.iteration.max_iterations)) {
    return fail<TallSvd>(ErrorCode::NoConvergence,
                         "real bidiagonal SVD did not converge");
  }
  std::vector<Scalar> sigma(n, 0.0);
  for (Index j = 0U; j < n; ++j) {
    long double norm2 = 0.0L;
    for (Index i = 0U; i < m; ++i) {
      const Scalar x = bidiagonal[i * n + j];
      norm2 += static_cast<long double>(x) * x;
    }
    sigma[j] = static_cast<Scalar>(std::sqrt(norm2)) * scale;
  }
  std::vector<Index> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&sigma](Index lhs, Index rhs) {
    return sigma[lhs] > sigma[rhs];
  });
  Dense sortedLeft(mn);
  Dense sortedRight(nn);
  std::vector<Scalar> sortedSigma(n);
  for (Index j = 0U; j < n; ++j) {
    const Index source = order[j];
    sortedSigma[j] = sigma[source];
    const Scalar singularScaled = scale == 0.0 ? 0.0 : sigma[source] / scale;
    for (Index i = 0U; i < m; ++i) {
      if (singularScaled == 0.0) {
        sortedLeft[i * n + j] = {};
      } else {
        const Complex ub{bidiagonal[i * n + source] / singularScaled, 0.0};
        sortedLeft[i * n + j] = ub * rowPhase[i];
      }
    }
    for (Index i = 0U; i < n; ++i) {
      // The real Jacobi right vectors diagonalize the reduced bidiagonal.
      Complex value{};
      for (Index q = 0U; q < n; ++q) {
        value += right[i * n + q] * rightReal[q * n + source];
      }
      sortedRight[j * n + i] = std::conj(value);
    }
    if (singularScaled == 0.0) {
      bool completed = false;
      for (Index basis = 0U; basis < m && !completed; ++basis) {
        Dense candidate(m, Complex{});
        candidate[basis] = rowPhase[basis];
        for (Index previous = 0U; previous < j; ++previous) {
          Complex projection{};
          for (Index i = 0U; i < m; ++i) {
            projection +=
                std::conj(sortedLeft[i * n + previous]) * candidate[i];
          }
          for (Index i = 0U; i < m; ++i) {
            candidate[i] -= sortedLeft[i * n + previous] * projection;
          }
        }
        const Scalar candidateNorm = normComplex(candidate);
        if (candidateNorm > 64.0 * std::numeric_limits<Scalar>::epsilon()) {
          for (Index i = 0U; i < m; ++i) {
            sortedLeft[i * n + j] = candidate[i] / candidateNorm;
          }
          completed = true;
        }
      }
      if (!completed) {
        return fail<TallSvd>(ErrorCode::NoConvergence,
                             "complex SVD could not complete a null column");
      }
    }
  }
  for (Index k = n; k-- > 0U;) {
    Dense v(m - k);
    for (Index i = 0U; i < v.size(); ++i) {
      v[i] = leftReflectors[k * m + i];
    }
    if (leftBetas[k] != 0.0) {
      applyLeft(sortedLeft, m, n, k, 0U, v, leftBetas[k]);
    }
  }
  return Result<TallSvd>::success(TallSvd{
      std::move(sortedLeft), std::move(sortedSigma), std::move(sortedRight)});
}

Scalar frobeniusNorm(const Dense& a) noexcept { return normComplex(a); }

Result<ComplexEigen> eigenImpl(ComplexMatrixView matrix,
                               Tolerance hermitianTolerance, SolvePolicy policy,
                               Limits limits) {
  const Status view_status = validateView(matrix);
  if (!view_status.ok()) {
    return copyFailure<ComplexEigen>(view_status);
  }
  if (matrix.rows != matrix.columns) {
    return fail<ComplexEigen>(ErrorCode::DimensionMismatch,
                              "Hermitian eigensolve requires a square matrix");
  }
  const Status tolerance_status = validate(hermitianTolerance);
  if (!tolerance_status.ok()) {
    return copyFailure<ComplexEigen>(tolerance_status);
  }
  if (!validPolicy(policy)) {
    return fail<ComplexEigen>(ErrorCode::InvalidTolerance,
                              "invalid Hermitian eigensolver policy");
  }
  const Index n = matrix.rows;
  Extent nn = 0U;
  if (!checkedProduct(n, n, nn)) {
    return fail<ComplexEigen>(ErrorCode::SizeOverflow,
                              "Hermitian eigensolver extent overflows");
  }
  const Extent eigenElements =
      nn + (n <= (std::numeric_limits<Extent>::max() - nn) / 2U ? 2U * n : 0U);
  if (n > (std::numeric_limits<Extent>::max() - nn) / 2U ||
      eigenElements > limits.elements ||
      !workspaceAllowedCombined(limits,
                                {{nn, kHermitianWorkspaceCopies},
                                 {n, kHermitianVectorWorkspaceCopies}})) {
    return fail<ComplexEigen>(ErrorCode::ResourceLimit,
                              "Hermitian eigensolver workspace exceeds limits");
  }
  Dense original = copyDense(matrix);
  Scalar scale = maxComponent(original);
  Dense h(nn, Complex{});
  Dense defect(nn, Complex{});
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      const Complex a = original[i * n + j];
      const Complex adjoint = std::conj(original[j * n + i]);
      defect[i * n + j] =
          scale == 0.0 ? Complex{} : a / scale - adjoint / scale;
      // Halving before addition prevents overflow for finite same-sign parts.
      h[i * n + j] = 0.5 * a + 0.5 * adjoint;
    }
  }
  const Scalar h_norm = normComplexScaled(original, scale);
  const Scalar defect_norm = frobeniusNorm(defect);
  const long double threshold =
      (scale == 0.0
           ? static_cast<long double>(hermitianTolerance.absolute)
           : static_cast<long double>(hermitianTolerance.absolute) / scale) +
      static_cast<long double>(hermitianTolerance.relative) * h_norm;
  if (static_cast<long double>(defect_norm) > threshold) {
    return fail<ComplexEigen>(
        ErrorCode::NotSymmetric,
        "matrix exceeds the Hermitian admission tolerance");
  }
  if (n == 0U) {
    return Result<ComplexEigen>::success(ComplexEigen{
        Vector{}, ComplexMatrix{0U, 0U, Array<Complex>{}}, Vector{}});
  }
  if (scale != 0.0) {
    for (Complex& z : h) {
      z /= scale;
    }
  }
  const Dense hermitian_scaled = h;
  // Reduce the Hermitian matrix to a real symmetric tridiagonal form by
  // complex Householder similarities while accumulating the unitary basis.
  Dense q(nn, Complex{});
  for (Index i = 0U; i < n; ++i) {
    q[i * n + i] = 1.0;
    h[i * n + i] = {h[i * n + i].real(), 0.0};
  }
  for (Index k = 0U; k + 2U < n; ++k) {
    Dense x(n - k - 1U);
    for (Index i = 0U; i < x.size(); ++i) {
      x[i] = h[(k + 1U + i) * n + k];
    }
    Dense v;
    Scalar beta = 0.0;
    Complex alpha{};
    if (!reflector(x, v, beta, alpha)) {
      return fail<ComplexEigen>(ErrorCode::NoConvergence,
                                "Hermitian Householder reduction failed");
    }
    if (beta != 0.0) {
      applyLeft(h, n, n, k + 1U, k + 1U, v, beta);
      applyRight(h, n, n, k + 1U, k + 1U, v, beta);
      multiplyRightByReflector(q, n, k + 1U, v, beta);
    }
    h[(k + 1U) * n + k] = alpha;
    h[k * n + k + 1U] = std::conj(alpha);
    for (Index i = k + 2U; i < n; ++i) {
      h[i * n + k] = {};
      h[k * n + i] = {};
    }
  }
  Dense phase(n, Complex{1.0, 0.0});
  for (Index i = 1U; i < n; ++i) {
    const Complex off = h[i * n + (i - 1U)];
    const Scalar magnitude = std::hypot(off.real(), off.imag());
    phase[i] =
        magnitude == 0.0 ? phase[i - 1U] : phase[i - 1U] * off / magnitude;
  }
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      h[i * n + j] *= std::conj(phase[i]) * phase[j];
    }
    for (Index row = 0U; row < n; ++row) {
      q[row * n + i] *= phase[i];
    }
  }
  std::vector<Scalar> diagonal(n, 0.0);
  std::vector<Scalar> offDiagonal(n, 0.0);
  RealDense realVectors(nn, 0.0);
  for (Index i = 0U; i < n; ++i) {
    diagonal[i] = h[i * n + i].real();
    offDiagonal[i] = i == 0U ? 0.0 : h[i * n + i - 1U].real();
    realVectors[i * n + i] = 1.0;
    for (Index j = 0U; j < n; ++j) {
      const Scalar value = h[i * n + j].real();
      if (!std::isfinite(value)) {
        return fail<ComplexEigen>(
            ErrorCode::NonFiniteResult,
            "Hermitian reduction produced non-finite data");
      }
    }
  }
  for (Index i = 1U; i < n; ++i) {
    offDiagonal[i - 1U] = offDiagonal[i];
  }
  offDiagonal[n - 1U] = 0.0;
  // Implicitly shifted tridiagonal QR/QL iteration. The per-eigenvalue
  // iteration count is bounded by SolvePolicy::iteration.max_iterations.
  const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
  for (Index l = 0U; l < n; ++l) {
    Index iterations = 0U;
    while (true) {
      Index m = l;
      for (; m + 1U < n; ++m) {
        if (std::abs(offDiagonal[m]) <=
            epsilon * (std::abs(diagonal[m]) + std::abs(diagonal[m + 1U]))) {
          break;
        }
      }
      if (m == l) {
        break;
      }
      if (iterations++ >= policy.iteration.max_iterations) {
        return fail<ComplexEigen>(ErrorCode::NoConvergence,
                                  "Hermitian tridiagonal QR did not converge");
      }
      const Scalar g0 =
          (diagonal[l + 1U] - diagonal[l]) / (2.0 * offDiagonal[l]);
      Scalar g = diagonal[m] - diagonal[l] +
                 offDiagonal[l] / (g0 + std::copysign(std::hypot(g0, 1.0), g0));
      Scalar sine = 1.0;
      Scalar cosine = 1.0;
      Scalar shift = 0.0;
      for (Index ii = m; ii-- > l;) {
        const Scalar f = sine * offDiagonal[ii];
        const Scalar b = cosine * offDiagonal[ii];
        Scalar radius = std::hypot(f, g);
        offDiagonal[ii + 1U] = radius;
        if (radius == 0.0) {
          diagonal[ii + 1U] -= shift;
          offDiagonal[m] = 0.0;
          break;
        }
        sine = f / radius;
        cosine = g / radius;
        g = diagonal[ii + 1U] - shift;
        radius = (diagonal[ii] - g) * sine + 2.0 * cosine * b;
        shift = sine * radius;
        diagonal[ii + 1U] = g + shift;
        g = cosine * radius - b;
        for (Index row = 0U; row < n; ++row) {
          const Scalar upper = realVectors[row * n + ii + 1U];
          const Scalar lower = realVectors[row * n + ii];
          realVectors[row * n + ii + 1U] = sine * lower + cosine * upper;
          realVectors[row * n + ii] = cosine * lower - sine * upper;
        }
      }
      diagonal[l] -= shift;
      offDiagonal[l] = g;
      offDiagonal[m] = 0.0;
    }
  }
  std::vector<Index> eigenOrder(n);
  std::iota(eigenOrder.begin(), eigenOrder.end(), 0U);
  std::stable_sort(eigenOrder.begin(), eigenOrder.end(),
                   [&diagonal](Index lhs, Index rhs) {
                     return diagonal[lhs] < diagonal[rhs];
                   });
  Vector values(n);
  Vector residuals(n);
  Dense vectors(nn, Complex{});
  for (Index j = 0U; j < n; ++j) {
    const Index source = eigenOrder[j];
    const Scalar lambda_scaled = diagonal[source];
    const Scalar lambda = lambda_scaled * scale;
    if (!std::isfinite(lambda)) {
      return fail<ComplexEigen>(ErrorCode::NonFiniteResult,
                                "Hermitian eigenvalue is not representable");
    }
    values.at(j) = lambda;
    for (Index i = 0U; i < n; ++i) {
      Complex value{};
      for (Index k = 0U; k < n; ++k) {
        value += q[i * n + k] * realVectors[k * n + source];
      }
      vectors[i * n + j] = value;
    }
    Index phaseIndex = 0U;
    Scalar phaseMagnitude = 0.0;
    for (Index i = 0U; i < n; ++i) {
      const Complex value = vectors[i * n + j];
      const Scalar magnitude = std::hypot(value.real(), value.imag());
      if (magnitude > phaseMagnitude) {
        phaseMagnitude = magnitude;
        phaseIndex = i;
      }
    }
    if (phaseMagnitude > 0.0) {
      const Complex phaseValue = vectors[phaseIndex * n + j] / phaseMagnitude;
      for (Index i = 0U; i < n; ++i) {
        vectors[i * n + j] *= std::conj(phaseValue);
      }
    }
    Dense r(n);
    for (Index i = 0U; i < n; ++i) {
      Complex value{};
      for (Index k = 0U; k < n; ++k) {
        value += hermitian_scaled[i * n + k] * vectors[k * n + j];
      }
      value -= lambda_scaled * vectors[i * n + j];
      r[i] = value;
    }
    const Scalar residual_scaled = normComplex(r);
    const Scalar residual = residual_scaled * scale;
    if (!std::isfinite(residual)) {
      return fail<ComplexEigen>(
          ErrorCode::NonFiniteResult,
          "Hermitian eigen residual is not representable");
    }
    residuals.at(j) = residual;
    const Scalar denominator = std::max(1.0, frobeniusNorm(hermitian_scaled));
    const Scalar eta = std::max({policy.iteration.abs, policy.iteration.rel,
                                 256.0 * static_cast<Scalar>(2U * n + 1U) *
                                     std::numeric_limits<Scalar>::epsilon()});
    if (residual_scaled / denominator > eta) {
      return fail<ComplexEigen>(ErrorCode::NoConvergence,
                                "Hermitian eigen residual exceeds policy");
    }
  }
  Dense orthogonality(nn, Complex{});
  for (Index i = 0U; i < n; ++i) {
    for (Index j = 0U; j < n; ++j) {
      Complex product{};
      for (Index row = 0U; row < n; ++row) {
        product += std::conj(vectors[row * n + i]) * vectors[row * n + j];
      }
      orthogonality[i * n + j] =
          product - (i == j ? Complex{1.0, 0.0} : Complex{});
    }
  }
  if (normComplex(orthogonality) >
      std::max({policy.iteration.abs, policy.iteration.rel,
                256.0 * static_cast<Scalar>(2U * n + 1U) *
                    std::numeric_limits<Scalar>::epsilon()})) {
    return fail<ComplexEigen>(ErrorCode::NoConvergence,
                              "Hermitian eigenvectors are not orthonormal");
  }
  Dense ownedQ = vectors;
  auto qArray = Array<Complex>::copy(ownedQ, limits);
  if (!qArray) {
    return copyFailure<ComplexEigen>(qArray.status());
  }
  return Result<ComplexEigen>::success(ComplexEigen{
      std::move(values), ComplexMatrix{n, n, std::move(qArray).value()},
      std::move(residuals)});
}

} // namespace

Result<ComplexMatrix> multiply(ComplexMatrixView a, ComplexMatrixView b,
                               Limits limits) {
  const Status a_status = validateView(a);
  if (!a_status.ok()) {
    return copyFailure<ComplexMatrix>(a_status);
  }
  const Status b_status = validateView(b);
  if (!b_status.ok()) {
    return copyFailure<ComplexMatrix>(b_status);
  }
  if (a.columns != b.rows) {
    return fail<ComplexMatrix>(ErrorCode::DimensionMismatch,
                               "complex product dimensions do not conform");
  }
  Extent count = 0U;
  if (!checkedProduct(a.rows, b.columns, count)) {
    return fail<ComplexMatrix>(ErrorCode::SizeOverflow,
                               "complex product output extent overflows");
  }
  if (count > limits.elements || count > limits.bytes / sizeof(Complex)) {
    return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                               "complex product output exceeds limits");
  }
  if (!workspaceAllowed(count, 2U, limits)) {
    return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                               "complex product workspace exceeds limits");
  }
  std::vector<Complex> output(count, Complex{});
  for (Index row = 0U; row < a.rows; ++row) {
    for (Index column = 0U; column < b.columns; ++column) {
      Scalar leftScale = 0.0;
      Scalar rightScale = 0.0;
      for (Index inner = 0U; inner < a.columns; ++inner) {
        const Complex lhs = at(a, row, inner);
        const Complex rhs = at(b, inner, column);
        leftScale =
            std::max({leftScale, std::abs(lhs.real()), std::abs(lhs.imag())});
        rightScale =
            std::max({rightScale, std::abs(rhs.real()), std::abs(rhs.imag())});
      }
      long double real_sum = 0.0L;
      long double real_correction = 0.0L;
      long double imaginary_sum = 0.0L;
      long double imaginary_correction = 0.0L;
      for (Index inner = 0U; inner < a.columns; ++inner) {
        const Complex lhs = at(a, row, inner);
        const Complex rhs = at(b, inner, column);
        const Complex left = leftScale == 0.0 ? Complex{} : lhs / leftScale;
        const Complex right = rightScale == 0.0 ? Complex{} : rhs / rightScale;
        const long double real_term =
            static_cast<long double>(left.real()) * right.real() -
            static_cast<long double>(left.imag()) * right.imag();
        const long double real_adjusted = real_term - real_correction;
        const long double real_next = real_sum + real_adjusted;
        real_correction = (real_next - real_sum) - real_adjusted;
        real_sum = real_next;
        const long double imaginary_term =
            static_cast<long double>(left.real()) * right.imag() +
            static_cast<long double>(left.imag()) * right.real();
        const long double imaginary_adjusted =
            imaginary_term - imaginary_correction;
        const long double imaginary_next = imaginary_sum + imaginary_adjusted;
        imaginary_correction =
            (imaginary_next - imaginary_sum) - imaginary_adjusted;
        imaginary_sum = imaginary_next;
      }
      const Scalar real = rescaleProduct(leftScale, rightScale, real_sum);
      const Scalar imaginary =
          rescaleProduct(leftScale, rightScale, imaginary_sum);
      if (!std::isfinite(real) || !std::isfinite(imaginary)) {
        return fail<ComplexMatrix>(ErrorCode::NonFiniteResult,
                                   "complex product result is not finite");
      }
      output[row * b.columns + column] = {real, imaginary};
    }
  }
  return makeMatrix(a.rows, b.columns, std::move(output), limits);
}

Result<ComplexMatrix> conjugateTranspose(ComplexMatrixView matrix,
                                         Limits limits) {
  const Status input_status = validateView(matrix);
  if (!input_status.ok()) {
    return copyFailure<ComplexMatrix>(input_status);
  }
  Extent count = 0U;
  if (!checkedProduct(matrix.rows, matrix.columns, count)) {
    return fail<ComplexMatrix>(ErrorCode::SizeOverflow,
                               "complex adjoint output extent overflows");
  }
  if (count > limits.elements || count > limits.bytes / sizeof(Complex)) {
    return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                               "complex adjoint output exceeds limits");
  }
  if (!workspaceAllowed(count, 2U, limits)) {
    return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                               "complex adjoint workspace exceeds limits");
  }
  std::vector<Complex> output(count);
  for (Index row = 0U; row < matrix.rows; ++row) {
    for (Index column = 0U; column < matrix.columns; ++column) {
      output[column * matrix.rows + row] = std::conj(at(matrix, row, column));
    }
  }
  return makeMatrix(matrix.columns, matrix.rows, std::move(output), limits);
}

Result<ComplexEigen> eigenHermitian(ComplexMatrixView matrix,
                                    SolvePolicy policy, Limits limits) {
  return eigenImpl(
      matrix, Tolerance{0.0, 64.0 * std::numeric_limits<Scalar>::epsilon()},
      policy, limits);
}

Result<ComplexEigen> eigenHermitian(ComplexMatrixView matrix,
                                    Tolerance hermitian_tolerance,
                                    SolvePolicy policy, Limits limits) {
  return eigenImpl(matrix, hermitian_tolerance, policy, limits);
}

Result<ComplexSVD> factorComplexSVD(ComplexMatrixView matrix,
                                    SolvePolicy policy, Limits limits) {
  const Status input_status = validateView(matrix);
  if (!input_status.ok()) {
    return copyFailure<ComplexSVD>(input_status);
  }
  if (!validPolicy(policy)) {
    return fail<ComplexSVD>(ErrorCode::InvalidTolerance,
                            "invalid complex SVD policy");
  }
  const Index m = matrix.rows;
  const Index n = matrix.columns;
  const Index k = std::min(m, n);
  Extent mn = 0U;
  if (!checkedProduct(m, n, mn)) {
    return fail<ComplexSVD>(ErrorCode::SizeOverflow,
                            "complex SVD input extent overflows");
  }
  Extent uk = 0U;
  Extent kn = 0U;
  if (!checkedProduct(m, k, uk) || !checkedProduct(k, n, kn)) {
    return fail<ComplexSVD>(ErrorCode::SizeOverflow,
                            "complex SVD output extent overflows");
  }
  Extent outputs = 0U;
  if (uk > std::numeric_limits<Extent>::max() - kn ||
      uk + kn > std::numeric_limits<Extent>::max() - k) {
    return fail<ComplexSVD>(ErrorCode::SizeOverflow,
                            "complex SVD output size overflows");
  }
  outputs = uk + kn + k;
  if (outputs > limits.elements ||
      !workspaceAllowedCombined(limits, {{mn, kSvdPeakWorkspaceCopies}})) {
    return fail<ComplexSVD>(ErrorCode::ResourceLimit,
                            "complex SVD outputs exceed limits");
  }
  if (k == 0U) {
    auto left = Array<Complex>::copy({}, limits);
    auto right = Array<Complex>::copy({}, limits);
    if (!left || !right) {
      return fail<ComplexSVD>(ErrorCode::ResourceLimit,
                              "empty complex SVD output exceeds limits");
    }
    return Result<ComplexSVD>::success(
        ComplexSVD{ComplexMatrix{m, 0U, std::move(left).value()}, Vector{},
                   ComplexMatrix{0U, n, std::move(right).value()}, 0U});
  }

  Result<TallSvd> tall = fail<TallSvd>(ErrorCode::NoConvergence,
                                       "complex SVD was not initialized");
  Dense transposed;
  if (m >= n) {
    tall = tallSvd(matrix, policy, limits);
  } else {
    transposed.resize(mn);
    for (Index i = 0U; i < m; ++i) {
      for (Index j = 0U; j < n; ++j) {
        transposed[j * m + i] = std::conj(at(matrix, i, j));
      }
    }
    const ComplexMatrixView adjoint{transposed, n, m, m, 1U};
    tall = tallSvd(adjoint, policy, limits);
  }
  if (!tall) {
    return copyFailure<ComplexSVD>(tall.status());
  }

  Dense left(uk, Complex{});
  Dense right(kn, Complex{});
  std::vector<Scalar> singular = tall.value().singular;
  if (m >= n) {
    left = tall.value().left;
    right = tall.value().right_adjoint;
  } else {
    // A* = U2 S V2* implies A = V2 S U2*.
    for (Index i = 0U; i < m; ++i) {
      for (Index j = 0U; j < m; ++j) {
        left[i * k + j] = std::conj(tall.value().right_adjoint[j * m + i]);
      }
    }
    for (Index i = 0U; i < k; ++i) {
      for (Index j = 0U; j < n; ++j) {
        right[i * n + j] = std::conj(tall.value().left[j * k + i]);
      }
    }
  }
  Vector singularValues(k);
  for (Index i = 0U; i < k; ++i) {
    if (!std::isfinite(singular[i])) {
      return fail<ComplexSVD>(ErrorCode::NonFiniteResult,
                              "complex singular value is not finite");
    }
    singularValues.at(i) = singular[i];
  }
  const Scalar largest = singular.empty() ? 0.0 : singular.front();
  const long double cutoff =
      std::max(static_cast<long double>(policy.rank_abs),
               static_cast<long double>(policy.rank_rel) * largest);
  Extent rank = 0U;
  for (const Scalar value : singular) {
    if (static_cast<long double>(value) > cutoff) {
      ++rank;
    }
  }

  // Certify the returned thin factors against the original input and both
  // unitary Gram identities before publishing any result.
  Dense reconstruction(mn, Complex{});
  for (Index i = 0U; i < m; ++i) {
    for (Index j = 0U; j < n; ++j) {
      Complex value{};
      for (Index q = 0U; q < k; ++q) {
        value += left[i * k + q] * singular[q] * right[q * n + j];
      }
      reconstruction[i * n + j] = value;
    }
  }
  Dense source = copyDense(matrix);
  const Scalar normalizedResidual =
      normalizedDifference(source, reconstruction);
  Dense gramLeft(k * k, Complex{});
  Dense gramRight(k * k, Complex{});
  for (Index i = 0U; i < k; ++i) {
    for (Index j = 0U; j < k; ++j) {
      Complex ul{};
      for (Index q = 0U; q < m; ++q) {
        ul += std::conj(left[q * k + i]) * left[q * k + j];
      }
      Complex vr{};
      for (Index q = 0U; q < n; ++q) {
        vr += right[i * n + q] * std::conj(right[j * n + q]);
      }
      gramLeft[i * k + j] = ul - (i == j ? Complex{1.0, 0.0} : Complex{});
      gramRight[i * k + j] = vr - (i == j ? Complex{1.0, 0.0} : Complex{});
    }
  }
  const Scalar eta = std::max({policy.iteration.abs, policy.iteration.rel,
                               256.0 * static_cast<Scalar>(m + n + 1U) *
                                   std::numeric_limits<Scalar>::epsilon()});
  if (!std::isfinite(normalizedResidual) || normalizedResidual > eta ||
      normComplex(gramLeft) > eta || normComplex(gramRight) > eta) {
    return fail<ComplexSVD>(ErrorCode::NoConvergence,
                            "complex SVD residual or orthogonality failed");
  }
  auto leftArray = Array<Complex>::copy(left, limits);
  auto rightArray = Array<Complex>::copy(right, limits);
  if (!leftArray || !rightArray) {
    return fail<ComplexSVD>(ErrorCode::ResourceLimit,
                            "complex SVD factor output exceeds limits");
  }
  return Result<ComplexSVD>::success(
      ComplexSVD{ComplexMatrix{m, k, std::move(leftArray).value()},
                 std::move(singularValues),
                 ComplexMatrix{k, n, std::move(rightArray).value()}, rank});
}

Result<ComplexMatrix> solveComplex(ComplexMatrixView a, ComplexMatrixView b,
                                   SolvePolicy policy, Limits limits) {
  const Status a_status = validateView(a);
  if (!a_status.ok()) {
    return copyFailure<ComplexMatrix>(a_status);
  }
  const Status b_status = validateView(b);
  if (!b_status.ok()) {
    return copyFailure<ComplexMatrix>(b_status);
  }
  if (a.rows != a.columns || b.rows != a.rows) {
    return fail<ComplexMatrix>(
        ErrorCode::DimensionMismatch,
        "complex solve requires square A and matching B rows");
  }
  if (!validPolicy(policy)) {
    return fail<ComplexMatrix>(ErrorCode::InvalidTolerance,
                               "invalid complex solve policy");
  }
  const Index n = a.rows;
  const Index rhs = b.columns;
  const bool estimateCondition = policy.reciprocal_condition_floor > 0.0;
  Extent nn = 0U;
  Extent nrhs = 0U;
  if (!checkedProduct(n, n, nn) || !checkedProduct(n, rhs, nrhs)) {
    return fail<ComplexMatrix>(ErrorCode::SizeOverflow,
                               "complex solve workspace extent overflows");
  }
  if (nrhs > limits.elements ||
      !workspaceAllowedCombined(
          limits, {{nn, estimateCondition ? 2U : 1U}, {nrhs, 6U}, {n, 4U}})) {
    return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                               "complex solve workspace exceeds limits");
  }
  if (n == 0U) {
    return makeMatrix(0U, rhs, {}, limits);
  }
  Dense lu = copyDense(a);
  const Scalar scale = maxComponent(lu);
  if (scale == 0.0) {
    if (policy.rank == RankPolicy::reject) {
      return fail<ComplexMatrix>(ErrorCode::Singular,
                                 "complex solve matrix is zero");
    }
    // The minimum-norm solution for a zero operator is zero. The supported
    // subspace policy accepts only a zero right-hand side.
    Dense zero(nrhs, Complex{});
    for (const Complex value : copyDense(b)) {
      if (value != Complex{} && policy.rank == RankPolicy::supported_subspace) {
        return fail<ComplexMatrix>(
            ErrorCode::RankDeficient,
            "right-hand side is outside the supported range");
      }
    }
    return makeMatrix(n, rhs, std::move(zero), limits);
  }
  for (Complex& value : lu) {
    value /= scale;
  }
  long double matrixOneNorm = 0.0L;
  Dense inverse;
  if (estimateCondition) {
    for (Index column = 0U; column < n; ++column) {
      long double columnSum = 0.0L;
      for (Index row = 0U; row < n; ++row) {
        const Complex value = lu[row * n + column];
        columnSum += std::hypot(static_cast<long double>(value.real()),
                                static_cast<long double>(value.imag()));
      }
      matrixOneNorm = std::max(matrixOneNorm, columnSum);
    }
    inverse.assign(nn, Complex{});
    for (Index i = 0U; i < n; ++i) {
      inverse[i * n + i] = 1.0;
    }
  }
  Dense solution = copyDense(b);
  for (Complex& value : solution) {
    value /= scale;
    if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
      return fail<ComplexMatrix>(
          ErrorCode::NonFiniteResult,
          "scaled complex right-hand side is not finite");
    }
  }
  bool deficient = false;
  const long double cutoff =
      std::max(static_cast<long double>(policy.rank_abs) / scale,
               static_cast<long double>(policy.rank_rel) * maxComponent(lu));
  for (Index column = 0U; column < n; ++column) {
    Index pivotRow = column;
    Scalar pivotMagnitude = 0.0;
    for (Index row = column; row < n; ++row) {
      const Complex value = lu[row * n + column];
      const Scalar magnitude = std::hypot(value.real(), value.imag());
      if (magnitude > pivotMagnitude) {
        pivotMagnitude = magnitude;
        pivotRow = row;
      }
    }
    if (static_cast<long double>(pivotMagnitude) <= cutoff) {
      deficient = true;
      break;
    }
    if (pivotRow != column) {
      for (Index j = 0U; j < n; ++j) {
        std::swap(lu[column * n + j], lu[pivotRow * n + j]);
      }
      for (Index j = 0U; j < rhs; ++j) {
        std::swap(solution[column * rhs + j], solution[pivotRow * rhs + j]);
      }
      if (estimateCondition) {
        for (Index j = 0U; j < n; ++j) {
          std::swap(inverse[column * n + j], inverse[pivotRow * n + j]);
        }
      }
    }
    const Complex diagonal = lu[column * n + column];
    for (Index row = column + 1U; row < n; ++row) {
      const Complex multiplier = scaledDivide(lu[row * n + column], diagonal);
      lu[row * n + column] = multiplier;
      for (Index j = column + 1U; j < n; ++j) {
        lu[row * n + j] -= multiplier * lu[column * n + j];
      }
      for (Index j = 0U; j < rhs; ++j) {
        solution[row * rhs + j] -= multiplier * solution[column * rhs + j];
      }
      if (estimateCondition) {
        for (Index j = 0U; j < n; ++j) {
          inverse[row * n + j] -= multiplier * inverse[column * n + j];
        }
      }
    }
  }
  if (deficient) {
    if (policy.rank == RankPolicy::reject) {
      return fail<ComplexMatrix>(ErrorCode::RankDeficient,
                                 "complex solve matrix is rank deficient");
    }
    if (!workspaceAllowedCombined(limits, {{nn, 26U}, {nrhs, 3U}})) {
      return fail<ComplexMatrix>(ErrorCode::ResourceLimit,
                                 "rank-deficient complex solve exceeds limits");
    }
    Dense{}.swap(lu);
    Dense{}.swap(inverse);
    Dense{}.swap(solution);
    auto decomposition = factorComplexSVD(a, policy, limits);
    if (!decomposition) {
      return copyFailure<ComplexMatrix>(decomposition.status());
    }
    const auto& svd = decomposition.value();
    Dense result(n * rhs, Complex{});
    const Scalar largest =
        svd.singular.size() == 0U ? 0.0 : svd.singular.at(0U);
    const long double threshold =
        std::max(static_cast<long double>(policy.rank_abs),
                 static_cast<long double>(policy.rank_rel) * largest);
    for (Index q = 0U; q < svd.singular.size(); ++q) {
      const Scalar sigma = svd.singular.at(q);
      if (static_cast<long double>(sigma) <= threshold) {
        continue;
      }
      for (Index column = 0U; column < rhs; ++column) {
        Complex projection{};
        for (Index row = 0U; row < n; ++row) {
          projection += std::conj(svd.left.values.view()[row * n + q]) *
                        at(b, row, column);
        }
        projection /= sigma;
        if (!std::isfinite(projection.real()) ||
            !std::isfinite(projection.imag())) {
          return fail<ComplexMatrix>(
              ErrorCode::NonFiniteResult,
              "complex minimum-norm projection is not representable");
        }
        for (Index row = 0U; row < n; ++row) {
          result[row * rhs + column] +=
              std::conj(svd.right_adjoint.values.view()[q * n + row]) *
              projection;
          if (!std::isfinite(result[row * rhs + column].real()) ||
              !std::isfinite(result[row * rhs + column].imag())) {
            return fail<ComplexMatrix>(
                ErrorCode::NonFiniteResult,
                "complex minimum-norm solution is not representable");
          }
        }
      }
    }
    if (policy.rank == RankPolicy::supported_subspace) {
      const Scalar eta = std::max({policy.iteration.abs, policy.iteration.rel,
                                   256.0 * static_cast<Scalar>(n + 1U) *
                                       std::numeric_limits<Scalar>::epsilon()});
      if (solveBackwardError(a, b, result, rhs, true) > eta) {
        return fail<ComplexMatrix>(
            ErrorCode::RankDeficient,
            "right-hand side is outside the supported range");
      }
    }
    return makeMatrix(n, rhs, std::move(result), limits);
  }
  for (Index row = n; row-- > 0U;) {
    for (Index column = 0U; column < rhs; ++column) {
      Complex value = solution[row * rhs + column];
      for (Index j = row + 1U; j < n; ++j) {
        value -= lu[row * n + j] * solution[j * rhs + column];
      }
      solution[row * rhs + column] = scaledDivide(value, lu[row * n + row]);
      if (!std::isfinite(solution[row * rhs + column].real()) ||
          !std::isfinite(solution[row * rhs + column].imag())) {
        return fail<ComplexMatrix>(
            ErrorCode::NonFiniteResult,
            "complex solve produced a non-finite solution");
      }
    }
    if (estimateCondition) {
      for (Index column = 0U; column < n; ++column) {
        Complex value = inverse[row * n + column];
        for (Index j = row + 1U; j < n; ++j) {
          value -= lu[row * n + j] * inverse[j * n + column];
        }
        inverse[row * n + column] = scaledDivide(value, lu[row * n + row]);
        if (!std::isfinite(inverse[row * n + column].real()) ||
            !std::isfinite(inverse[row * n + column].imag())) {
          return fail<ComplexMatrix>(
              ErrorCode::IllConditioned,
              "complex solve condition estimate is not representable");
        }
      }
    }
  }
  if (estimateCondition) {
    long double inverseOneNorm = 0.0L;
    for (Index column = 0U; column < n; ++column) {
      long double columnSum = 0.0L;
      for (Index row = 0U; row < n; ++row) {
        const Complex value = inverse[row * n + column];
        columnSum += std::hypot(static_cast<long double>(value.real()),
                                static_cast<long double>(value.imag()));
      }
      inverseOneNorm = std::max(inverseOneNorm, columnSum);
    }
    const long double reciprocalCondition =
        1.0L / (matrixOneNorm * inverseOneNorm);
    if (reciprocalCondition < policy.reciprocal_condition_floor) {
      return fail<ComplexMatrix>(
          ErrorCode::IllConditioned,
          "complex solve reciprocal condition is below policy");
    }
  }
  const long double backwardError = solveBackwardError(a, b, solution, rhs);
  const Scalar eta = std::max({policy.iteration.abs, policy.iteration.rel,
                               256.0 * static_cast<Scalar>(n + 1U) *
                                   std::numeric_limits<Scalar>::epsilon()});
  if (!std::isfinite(backwardError) || backwardError > eta) {
    return fail<ComplexMatrix>(ErrorCode::NoConvergence,
                               "complex solve residual exceeds policy");
  }
  return makeMatrix(n, rhs, std::move(solution), limits);
}

} // namespace vss::math::linear
