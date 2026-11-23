#include <algorithm>
#include <array>
#include <cmath>
#include <detail/CompensatedSum.hpp>
#include <limits>
#include <new>
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
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Diagnostics.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/statistics/MomentDiagnostics.hpp>
#include <vss/math/statistics/Moments.hpp>
#include <vss/math/statistics/Population.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

class LongCompensatedSum final {
public:
  void add(long double value) noexcept {
    const long double next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }

  [[nodiscard]] long double total() const noexcept {
    return sum_ + correction_;
  }

private:
  long double sum_{};
  long double correction_{};
};

bool checkedAdd(Extent left, Extent right, Extent& output) noexcept {
  if (right > std::numeric_limits<Extent>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

bool checkedMultiply(Extent left, Extent right, Extent& output) noexcept {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

bool addProduct(Extent left, Extent right, Extent& total) noexcept {
  Extent product = 0U;
  return checkedMultiply(left, right, product) &&
         checkedAdd(total, product, total);
}

bool validPolicy(const SolvePolicy& policy) noexcept {
  return std::isfinite(policy.rank_abs) && policy.rank_abs >= 0.0 &&
         std::isfinite(policy.rank_rel) && policy.rank_rel >= 0.0 &&
         std::isfinite(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor >= 0.0 &&
         policy.reciprocal_condition_floor <= 1.0 &&
         std::isfinite(policy.iteration.abs) && policy.iteration.abs >= 0.0 &&
         std::isfinite(policy.iteration.rel) && policy.iteration.rel >= 0.0 &&
         (policy.rank == RankPolicy::reject ||
          policy.rank == RankPolicy::supported_subspace ||
          policy.rank == RankPolicy::minimum_norm);
}

bool supportEqual(const Support& left, const Support& right) noexcept {
  return left.input == right.input && left.valid == right.valid &&
         left.positive == right.positive &&
         left.frequency_mass == right.frequency_mass &&
         left.weight_mass == right.weight_mass &&
         left.normalized_weight_powers == right.normalized_weight_powers &&
         left.effective_count == right.effective_count &&
         left.degrees_of_freedom == right.degrees_of_freedom &&
         left.squared_weight_mass == right.squared_weight_mass;
}

Result<Extent> momentDimension(const MomentEstimate& moments) {
  Extent dimension = 1U;
  for (const Index extent : moments.observation_shape.view()) {
    if (extent == 0U) {
      return fail<Extent>(ErrorCode::InsufficientSamples,
                          "moment table has an empty observation shape");
    }
    if (!checkedMultiply(dimension, extent, dimension)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "moment observation shape overflows");
    }
  }
  return Result<Extent>::success(dimension);
}

Result<Extent> sampleDimension(const SampleView& sample) {
  Extent dimension = 1U;
  for (const Index extent : sample.observation_shape) {
    if (extent == 0U) {
      return fail<Extent>(ErrorCode::InvalidExtent,
                          "sample observation extents must be positive");
    }
    if (!checkedMultiply(dimension, extent, dimension)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "sample observation shape overflows");
    }
  }
  if (sample.rows.columns() != dimension) {
    return fail<Extent>(ErrorCode::DimensionMismatch,
                        "sample shape does not match its row columns");
  }
  return Result<Extent>::success(dimension);
}

Result<const MomentTerm*> findTerm(const MomentEstimate& moments, Index order) {
  const MomentTerm* found = nullptr;
  for (const auto& term : moments.terms.view()) {
    if (term.order != order) {
      continue;
    }
    if (found != nullptr) {
      return fail<const MomentTerm*>(ErrorCode::InvalidDomain,
                                     "moment order is duplicated");
    }
    found = &term;
  }
  if (found == nullptr) {
    return fail<const MomentTerm*>(ErrorCode::InsufficientSamples,
                                   "a required moment order is absent");
  }
  return Result<const MomentTerm*>::success(found);
}

Result<Extent> binomial(Extent n, Index k) {
  if (k > n) {
    return Result<Extent>::success(0U);
  }
  k = std::min<Index>(k, n - k);
  Extent result = 1U;
  for (Index i = 1U; i <= k; ++i) {
    Extent numerator = n - k + i;
    Extent denominator = i;
    const Extent firstGcd = std::gcd(result, denominator);
    result /= firstGcd;
    denominator /= firstGcd;
    const Extent secondGcd = std::gcd(numerator, denominator);
    numerator /= secondGcd;
    denominator /= secondGcd;
    if (denominator != 1U || !checkedMultiply(result, numerator, result)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "packed moment index overflows");
    }
  }
  return Result<Extent>::success(result);
}

Result<std::array<Index, 4U>> sortedTuple(std::span<const Index> tuple) {
  std::array<Index, 4U> sorted{};
  if (tuple.empty() || tuple.size() > sorted.size()) {
    return fail<std::array<Index, 4U>>(
        ErrorCode::InvalidExtent,
        "moment tuple order must be between one and four");
  }
  for (Index index = 0U; index < tuple.size(); ++index) {
    sorted.at(index) = tuple[index];
  }
  for (Index left = 0U; left < tuple.size(); ++left) {
    for (Index right = left + 1U; right < tuple.size(); ++right) {
      if (sorted.at(right) < sorted.at(left)) {
        std::swap(sorted.at(left), sorted.at(right));
      }
    }
  }
  return Result<std::array<Index, 4U>>::success(sorted);
}

bool nextTuplePermutation(std::array<Index, 4U>& tuple, Index size) {
  if (size < 2U) {
    return false;
  }
  Index pivot = size - 1U;
  while (pivot > 0U && tuple.at(pivot - 1U) >= tuple.at(pivot)) {
    --pivot;
  }
  if (pivot == 0U) {
    return false;
  }
  Index successor = size - 1U;
  while (tuple.at(successor) <= tuple.at(pivot - 1U)) {
    --successor;
  }
  std::swap(tuple.at(pivot - 1U), tuple.at(successor));
  Index left = pivot;
  Index right = size - 1U;
  while (left < right) {
    std::swap(tuple.at(left), tuple.at(right));
    ++left;
    --right;
  }
  return true;
}

void setSymmetricEntry(Matrix& matrix, Index first, Index second,
                       Scalar value) {
  matrix.at(second, first) = value;
}

Status validateMomentTermShapes(const MomentEstimate& moments,
                                Extent dimension) {
  std::array<bool, 5U> seenOrders{};
  for (const MomentTerm& term : moments.terms.view()) {
    if (term.order == 0U || term.order > 4U || seenOrders.at(term.order)) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "moment table has an invalid or duplicate order");
    }
    seenOrders.at(term.order) = true;
    Extent expectedValues = 0U;
    switch (term.layout) {
    case MomentRepresentation::component:
      expectedValues = dimension;
      break;
    case MomentRepresentation::dense:
      expectedValues = 1U;
      for (Index axis = 0U; axis < term.order; ++axis) {
        if (!checkedMultiply(expectedValues, dimension, expectedValues)) {
          return Status::failure(ErrorCode::SizeOverflow,
                                 "dense moment shape overflows");
        }
      }
      break;
    case MomentRepresentation::packed: {
      Extent combinations = 0U;
      if (!checkedAdd(dimension, term.order - 1U, combinations)) {
        return Status::failure(ErrorCode::SizeOverflow,
                               "packed moment shape overflows");
      }
      const auto count = binomial(combinations, term.order);
      if (!count) {
        return Status::failure(count.status().code(),
                               "packed moment shape is not representable");
      }
      expectedValues = count.value();
      break;
    }
    case MomentRepresentation::selected: {
      Extent flattenedCount = 0U;
      if (!checkedMultiply(term.values.size(), term.order, flattenedCount) ||
          flattenedCount != term.flattened_tuples.view().size()) {
        return Status::failure(ErrorCode::DimensionMismatch,
                               "selected moment metadata has the wrong shape");
      }
      const auto flattened = term.flattened_tuples.view();
      for (const Index coordinate : flattened) {
        if (coordinate >= dimension) {
          return Status::failure(ErrorCode::IndexOutOfRange,
                                 "selected moment coordinate is out of range");
        }
      }
      for (Index left = 0U; left < term.values.size(); ++left) {
        const auto leftTuple = flattened.subspan(left * term.order, term.order);
        for (Index right = left + 1U; right < term.values.size(); ++right) {
          const auto rightTuple =
              flattened.subspan(right * term.order, term.order);
          if (std::ranges::equal(leftTuple, rightTuple)) {
            return Status::failure(ErrorCode::InvalidDomain,
                                   "selected moment tuple is duplicated");
          }
        }
      }
      expectedValues = term.values.size();
      break;
    }
    case MomentRepresentation::contraction:
      return Status::failure(ErrorCode::InvalidDomain,
                             "contracted moments must be stored as dense");
    default:
      return Status::failure(ErrorCode::InvalidDomain,
                             "moment term layout is unknown");
    }
    if (term.values.size() != expectedValues ||
        (term.layout != MomentRepresentation::selected &&
         !term.flattened_tuples.view().empty())) {
      return Status::failure(ErrorCode::DimensionMismatch,
                             "moment term values have the wrong shape");
    }
    for (Index index = 0U; index < term.values.size(); ++index) {
      if (!std::isfinite(term.values.at(index))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "moment table contains a non-finite value");
      }
    }
  }
  return Status::success();
}

Result<Extent> packedIndex(std::span<const Index> tuple, Extent dimension) {
  if (tuple.empty() || tuple.size() > 4U) {
    return fail<Extent>(ErrorCode::InvalidExtent,
                        "packed moment tuple order is invalid");
  }
  const auto sortedResult = sortedTuple(tuple);
  if (!sortedResult) {
    return fail<Extent>(sortedResult.status().code(),
                        "packed moment tuple is invalid");
  }
  std::array<Index, 4U> sorted = sortedResult.value();
  Extent rank = 0U;
  Index minimum = 0U;
  for (Index axis = 0U; axis < tuple.size(); ++axis) {
    const Index coordinate = sorted.at(axis);
    if (coordinate >= dimension || coordinate < minimum) {
      return fail<Extent>(ErrorCode::IndexOutOfRange,
                          "packed moment tuple is outside its shape");
    }
    const Index remaining = tuple.size() - axis - 1U;
    Extent firstN = 0U;
    Extent secondN = 0U;
    if (!checkedAdd(dimension - minimum, remaining, firstN) ||
        !checkedAdd(dimension - coordinate, remaining, secondN)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "packed moment rank overflows");
    }
    const auto first = binomial(firstN, remaining + 1U);
    const auto second = binomial(secondN, remaining + 1U);
    if (!first || !second || first.value() < second.value() ||
        !checkedAdd(rank, first.value() - second.value(), rank)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "packed moment rank overflows");
    }
    minimum = coordinate;
  }
  return Result<Extent>::success(rank);
}

Result<Scalar> tupleValue(const MomentEstimate& moments,
                          std::span<const Index> tuple, Extent dimension) {
  const auto termResult = findTerm(moments, tuple.size());
  if (!termResult) {
    return fail<Scalar>(termResult.status().code(),
                        "required moment order is unavailable");
  }
  const MomentTerm& term = *termResult.value();
  const auto values = term.values.view();
  if (term.layout == MomentRepresentation::component) {
    if (!std::ranges::all_of(tuple, [&](Index coordinate) {
          return coordinate == tuple.front();
        })) {
      return fail<Scalar>(ErrorCode::InsufficientSamples,
                          "component moments omit a cross-coordinate term");
    }
    if (tuple.front() >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "component moments have the wrong shape");
    }
    return Result<Scalar>::success(values.at(tuple.front()));
  }
  if (term.layout == MomentRepresentation::dense) {
    Extent linear = 0U;
    for (const Index coordinate : tuple) {
      if (coordinate >= dimension ||
          !checkedMultiply(linear, dimension, linear) ||
          !checkedAdd(linear, coordinate, linear)) {
        return fail<Scalar>(ErrorCode::IndexOutOfRange,
                            "dense moment tuple is outside its shape");
      }
    }
    if (linear >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "dense moment order has an inconsistent size");
    }
    return Result<Scalar>::success(values.at(linear));
  }
  if (term.layout == MomentRepresentation::packed) {
    const auto index = packedIndex(tuple, dimension);
    if (!index) {
      return fail<Scalar>(index.status().code(),
                          "packed moment tuple index is invalid");
    }
    if (index.value() >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "packed moment order has an inconsistent size");
    }
    return Result<Scalar>::success(values.at(index.value()));
  }
  if (term.layout == MomentRepresentation::selected) {
    const auto flattened = term.flattened_tuples.view();
    if (tuple.size() != term.order ||
        flattened.size() != values.size() * tuple.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "selected moment metadata is inconsistent");
    }
    for (Index position = 0U; position < values.size(); ++position) {
      const auto stored =
          flattened.subspan(position * tuple.size(), tuple.size());
      if (std::ranges::equal(stored, tuple)) {
        return Result<Scalar>::success(values.at(position));
      }
    }
    return fail<Scalar>(ErrorCode::InsufficientSamples,
                        "required selected moment tuple is absent");
  }
  return fail<Scalar>(ErrorCode::InvalidDomain,
                      "moment representation is not valid for a source table");
}

Result<Support> supportForTuple(const MomentEstimate& moments,
                                std::span<const Index> tuple,
                                Extent dimension) {
  Support result = moments.diagnostics.global;
  bool found = false;
  for (const auto& entry : moments.diagnostics.tuple_support.view()) {
    const auto coordinates = entry.coordinates.view();
    if (coordinates.size() == tuple.size() &&
        std::ranges::equal(coordinates, tuple)) {
      if (found && !supportEqual(result, entry.support)) {
        return fail<Support>(ErrorCode::InvalidDomain,
                             "tuple has conflicting support metadata");
      }
      result = entry.support;
      found = true;
    }
  }
  if (found || moments.diagnostics.coordinate_support.view().empty()) {
    return Result<Support>::success(result);
  }
  const auto coordinateSupport = moments.diagnostics.coordinate_support.view();
  if (coordinateSupport.size() != dimension) {
    return fail<Support>(ErrorCode::DimensionMismatch,
                         "coordinate support has the wrong dimension");
  }
  bool first = true;
  for (const Index coordinate : tuple) {
    if (coordinate >= dimension) {
      return fail<Support>(ErrorCode::IndexOutOfRange,
                           "tuple support index is outside its shape");
    }
    if (first) {
      result = coordinateSupport[coordinate];
      first = false;
    } else if (!supportEqual(result, coordinateSupport[coordinate])) {
      return fail<Support>(ErrorCode::InvalidDomain,
                           "tuple combines incompatible support records");
    }
  }
  return Result<Support>::success(result);
}

struct SymmetricValue final {
  Scalar value;
  bool symmetric;
};

Result<SymmetricValue> symmetricValue(const MomentEstimate& moments,
                                      std::span<const Index> tuple,
                                      Extent dimension, Tolerance tolerance,
                                      std::optional<Support>& commonSupport) {
  const auto sortedResult = sortedTuple(tuple);
  if (!sortedResult) {
    return fail<SymmetricValue>(sortedResult.status().code(),
                                "moment tuple is invalid");
  }
  std::array<Index, 4U> sorted = sortedResult.value();
  const auto support = supportForTuple(moments, tuple, dimension);
  if (!support) {
    return fail<SymmetricValue>(support.status().code(),
                                "moment support is invalid");
  }
  if (commonSupport.has_value() &&
      !supportEqual(*commonSupport, support.value())) {
    return fail<SymmetricValue>(ErrorCode::InvalidDomain,
                                "moment terms use incompatible support");
  }
  commonSupport = support.value();
  LongCompensatedSum sum;
  Scalar reference = 0.0;
  bool first = true;
  bool symmetric = true;
  Index permutations = 0U;
  while (true) {
    const std::span<const Index> permutation(sorted.data(), tuple.size());
    const auto permutationSupport =
        supportForTuple(moments, permutation, dimension);
    if (!permutationSupport) {
      return fail<SymmetricValue>(permutationSupport.status().code(),
                                  "moment permutation support is invalid");
    }
    if (!supportEqual(support.value(), permutationSupport.value())) {
      return fail<SymmetricValue>(ErrorCode::InvalidDomain,
                                  "moment permutations use different support");
    }
    const auto value = tupleValue(moments, permutation, dimension);
    if (!value) {
      return fail<SymmetricValue>(value.status().code(),
                                  "required moment permutation is absent");
    }
    if (!std::isfinite(value.value())) {
      return fail<SymmetricValue>(ErrorCode::NonFiniteInput,
                                  "moment table contains a non-finite value");
    }
    if (first) {
      reference = value.value();
      first = false;
    } else {
      const Scalar bound =
          tolerance.absolute +
          tolerance.relative *
              std::max(std::abs(reference), std::abs(value.value()));
      symmetric = symmetric && std::abs(value.value() - reference) <= bound;
    }
    sum.add(value.value());
    ++permutations;
    if (!nextTuplePermutation(sorted, tuple.size())) {
      break;
    }
  }
  const long double average =
      sum.total() / static_cast<long double>(permutations);
  if (!std::isfinite(average) ||
      std::abs(average) > std::numeric_limits<Scalar>::max()) {
    return fail<SymmetricValue>(ErrorCode::NonFiniteResult,
                                "symmetrized moment is not representable");
  }
  return Result<SymmetricValue>::success(
      SymmetricValue{static_cast<Scalar>(average), symmetric});
}

Result<Scalar> getSymmetricMoment(const MomentEstimate& moments,
                                  std::span<const Index> tuple,
                                  Extent dimension, Tolerance tolerance,
                                  std::optional<Support>& commonSupport) {
  const auto value =
      symmetricValue(moments, tuple, dimension, tolerance, commonSupport);
  if (!value) {
    return fail<Scalar>(value.status().code(),
                        "symmetric moment is unavailable");
  }
  return Result<Scalar>::success(value.value().value);
}

Result<Scalar> normalizedAbsoluteTolerance(Scalar absolute, Scalar scale) {
  if (absolute == 0.0 || scale == 0.0) {
    return Result<Scalar>::success(absolute);
  }
  const long double ratio = static_cast<long double>(absolute) / scale;
  return Result<Scalar>::success(ratio > std::numeric_limits<Scalar>::max()
                                     ? std::numeric_limits<Scalar>::max()
                                     : static_cast<Scalar>(ratio));
}

Result<Scalar> restoreStatisticScale(long double normalized, Scalar scale,
                                     int scalePower) {
  if (normalized == 0.0L) {
    return Result<Scalar>::success(0.0);
  }
  if (!(scale > 0.0) || !std::isfinite(scale) || !std::isfinite(normalized)) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "scaled statistic has invalid normalized inputs");
  }
  int scaleExponent = 0;
  int normalizedExponent = 0;
  const long double scaleMantissa =
      std::frexp(static_cast<long double>(scale), &scaleExponent);
  long double mantissa = std::frexp(normalized, &normalizedExponent);
  for (int power = 0; power < scalePower; ++power) {
    mantissa *= scaleMantissa;
  }
  const int exponent = normalizedExponent + scalePower * scaleExponent;
  const long double restored = std::scalbn(mantissa, exponent);
  if (!std::isfinite(restored) ||
      std::abs(restored) > std::numeric_limits<Scalar>::max() ||
      restored == 0.0L || static_cast<Scalar>(restored) == 0.0) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "scaled statistic is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(restored));
}

Result<linear::SymmetricEigenDecomposition>
decomposePsd(Matrix& matrix, Scalar scale, SolvePolicy policy) {
  linear::SymmetricEigenOptions options{};
  const auto absolute = normalizedAbsoluteTolerance(policy.rank_abs, scale);
  if (!absolute) {
    return fail<linear::SymmetricEigenDecomposition>(
        absolute.status().code(), "PSD absolute tolerance cannot be scaled");
  }
  options.rankTolerance = {.absolute = absolute.value(),
                           .relative = policy.rank_rel};
  options.symmetryTolerance = {.absolute = 0.0, .relative = 0.0};
  options.maximumSweeps = policy.iteration.max_iterations;
  if (options.maximumSweeps == 0U) {
    for (Index row = 0U; row < matrix.rows(); ++row) {
      for (Index column = row + 1U; column < matrix.columns(); ++column) {
        if (matrix.at(row, column) != 0.0) {
          return fail<linear::SymmetricEigenDecomposition>(
              ErrorCode::NoConvergence,
              "zero sweep budget cannot certify a non-diagonal matrix");
        }
      }
    }
    options.maximumSweeps = 1U;
  }
  return linear::decomposeSymmetricEigen(matrix.view(), options);
}

Result<Extent> gramOrder(Extent dimension) {
  Extent square = 0U;
  Extent product = 0U;
  Extent order = 0U;
  if (dimension == std::numeric_limits<Extent>::max() ||
      !checkedMultiply(dimension, dimension + 1U, product)) {
    return fail<Extent>(ErrorCode::SizeOverflow,
                        "degree-two Gram order overflows");
  }
  product /= 2U;
  if (!checkedAdd(1U, dimension, order) || !checkedAdd(order, product, order) ||
      !checkedMultiply(order, order, square)) {
    return fail<Extent>(ErrorCode::SizeOverflow,
                        "degree-two Gram size overflows");
  }
  return Result<Extent>::success(order);
}

Result<bool> withinBudget(Extent dimension, Extent gramSize, Limits limits) {
  Extent square = 0U;
  Extent scalarWorkspace = 0U;
  // The eigensolver owns a working matrix and eigenvectors alongside the
  // caller's Gram. The remaining terms cover the covariance and basis scales.
  if (!checkedMultiply(dimension, dimension, square) ||
      !checkedMultiply(3U, gramSize, scalarWorkspace) ||
      !addProduct(4U, square, scalarWorkspace) ||
      !addProduct(8U, gramSize, scalarWorkspace) ||
      !addProduct(8U, dimension, scalarWorkspace) ||
      scalarWorkspace > limits.elements ||
      scalarWorkspace > limits.bytes / sizeof(Scalar)) {
    return Result<bool>::success(false);
  }
  return Result<bool>::success(true);
}

Status validateSample(const SampleView& sample, Extent& dimension) {
  const auto dimensionResult = sampleDimension(sample);
  if (!dimensionResult) {
    return Status::failure(dimensionResult.status().code(),
                           "sample dimensions are invalid");
  }
  dimension = dimensionResult.value();
  if (dimension == 0U || sample.rows.rows() == 0U) {
    return Status::failure(ErrorCode::InsufficientSamples,
                           "sample must contain rows and coordinates");
  }
  Extent cellCount = 0U;
  if (!checkedMultiply(sample.rows.rows(), dimension, cellCount)) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "sample cell count overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != cellCount) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "sample validity mask has the wrong size");
  }
  for (Index cell = 0U; cell < cellCount; ++cell) {
    if (!sample.valid.empty()) {
      if (sample.valid[cell] > 1U) {
        return Status::failure(
            ErrorCode::InvalidDomain,
            "sample validity mask contains a non-boolean value");
      }
      if (sample.valid[cell] == 0U) {
        return Status::failure(ErrorCode::InvalidDomain,
                               "Mardia statistics require complete rows");
      }
    }
    const Index row = cell / dimension;
    const Index column = cell % dimension;
    if (!std::isfinite(sample.rows.at(row, column))) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "sample contains a non-finite value");
    }
  }
  return Status::success();
}

Result<bool> sampleWorkspace(Extent observations, Extent dimension,
                             Limits limits) {
  Extent nd = 0U;
  Extent dd = 0U;
  Extent total = 0U;
  if (!checkedMultiply(observations, dimension, nd) ||
      !checkedMultiply(dimension, dimension, dd) ||
      !addProduct(3U, dd, total) || !checkedAdd(total, nd, total) ||
      !addProduct(4U, dimension, total) || total > limits.elements ||
      total > limits.bytes / sizeof(Scalar)) {
    return Result<bool>::success(false);
  }
  return Result<bool>::success(true);
}

Result<Matrix> whitenedRows(const SampleView& sample, SolvePolicy policy,
                            Limits limits) {
  Extent dimension = 0U;
  const auto valid = validateSample(sample, dimension);
  if (!valid.ok()) {
    return fail<Matrix>(valid.code(), "Mardia sample validation failed");
  }
  if (!validPolicy(policy)) {
    return fail<Matrix>(ErrorCode::InvalidTolerance,
                        "Mardia solve policy is invalid");
  }
  if (sample.rows.rows() <= dimension) {
    return fail<Matrix>(ErrorCode::InsufficientSamples,
                        "Mardia statistics require N greater than dimension");
  }
  const auto budget = sampleWorkspace(sample.rows.rows(), dimension, limits);
  if (!budget || !budget.value()) {
    return fail<Matrix>(ErrorCode::ResourceLimit,
                        "Mardia workspace exceeds configured limits");
  }
  const auto covarianceResult = covarianceMatrix(
      sample, CenterSpec{},
      PopulationPolicy{.normalization = VarianceMode::empirical,
                       .missingness = MissingMode::reject,
                       .estimand = Estimand::observed},
      limits);
  if (!covarianceResult) {
    return fail<Matrix>(covarianceResult.status().code(),
                        "Mardia population covariance failed");
  }
  SolvePolicy fullRankPolicy = policy;
  fullRankPolicy.rank = RankPolicy::reject;
  const auto inverseRoot = covariance::inverseSquareRoot(
      covarianceResult.value().value.view(), fullRankPolicy, limits);
  if (!inverseRoot) {
    return fail<Matrix>(inverseRoot.status().code(),
                        "Mardia whitening requires full-rank covariance");
  }
  if (inverseRoot.value().rank != dimension) {
    return fail<Matrix>(ErrorCode::RankDeficient,
                        "Mardia covariance is rank deficient");
  }

  const auto meanEstimate = estimateRawMoments(
      sample,
      WeightSpec{.rule = WeightRule::unweighted,
                 .rowWeights = std::nullopt,
                 .componentWeights = std::nullopt,
                 .frequencyCounts = {}},
      MomentRequest{.orders = 0x01U,
                    .representation = MomentRepresentation::dense,
                    .selected = {},
                    .contraction = std::nullopt},
      limits);
  if (!meanEstimate) {
    return fail<Matrix>(meanEstimate.status().code(),
                        "Mardia mean estimation failed");
  }
  if (!meanEstimate.value().population_mean.has_value()) {
    return fail<Matrix>(ErrorCode::InsufficientSamples,
                        "Mardia mean estimation failed");
  }
  const Vector& populationMean = meanEstimate.value().population_mean.value();
  Matrix whitened(sample.rows.rows(), dimension, 0.0);
  for (Index row = 0U; row < sample.rows.rows(); ++row) {
    for (Index output = 0U; output < dimension; ++output) {
      detail::CompensatedSum sum;
      for (Index input = 0U; input < dimension; ++input) {
        sum.add(inverseRoot.value().covariance.at(output, input) *
                (sample.rows.at(row, input) - populationMean.at(input)));
      }
      if (!std::isfinite(sum.total()) ||
          std::abs(sum.total()) > std::numeric_limits<Scalar>::max()) {
        return fail<Matrix>(ErrorCode::NonFiniteResult,
                            "Mardia whitened row is not representable");
      }
      whitened.at(row, output) = sum.total();
    }
  }
  return Result<Matrix>::success(std::move(whitened));
}

Result<Scalar> mardia(const SampleView& sample, SolvePolicy policy,
                      Limits limits, bool skewness) {
  auto whitenedResult = whitenedRows(sample, policy, limits);
  if (!whitenedResult) {
    return Result<Scalar>::failure(
        whitenedResult.status().code(),
        std::string(whitenedResult.status().message()));
  }
  const Matrix& whitened = whitenedResult.value();
  LongCompensatedSum total;
  LongCompensatedSum absoluteTotal;
  const Index rows = whitened.rows();
  const Extent dimension = whitened.columns();
  Scalar scale = 0.0;
  for (Index row = 0U; row < rows; ++row) {
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      scale = std::max(scale, std::abs(whitened.at(row, coordinate)));
    }
  }
  if (scale == 0.0) {
    return fail<Scalar>(ErrorCode::RankDeficient,
                        "Mardia whitening produced only zero coordinates");
  }
  for (Index left = 0U; left < rows; ++left) {
    const Index rightBegin = skewness ? 0U : left;
    const Index rightEnd = skewness ? rows : left + 1U;
    for (Index right = rightBegin; right < rightEnd; ++right) {
      LongCompensatedSum dot;
      for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
        dot.add(
            (static_cast<long double>(whitened.at(left, coordinate)) / scale) *
            (static_cast<long double>(whitened.at(right, coordinate)) / scale));
      }
      const long double value = skewness
                                    ? dot.total() * dot.total() * dot.total()
                                    : dot.total() * dot.total();
      if (!std::isfinite(value)) {
        return fail<Scalar>(ErrorCode::NonFiniteResult,
                            "Mardia summand is not finite");
      }
      total.add(value);
      absoluteTotal.add(std::abs(value));
      if (!std::isfinite(absoluteTotal.total())) {
        return fail<Scalar>(ErrorCode::NonFiniteResult,
                            "Mardia absolute accumulation is not finite");
      }
    }
  }
  long double result = total.total();
  if (skewness) {
    result /= static_cast<long double>(rows) * rows;
    const long double roundoff = 128.0L *
                                 std::numeric_limits<Scalar>::epsilon() *
                                 std::max(absoluteTotal.total(), 1.0L);
    if (result < 0.0L && std::abs(result) <= roundoff) {
      result = 0.0L;
    }
  } else {
    result /= static_cast<long double>(rows);
  }
  if (result < 0.0L || !std::isfinite(result)) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "Mardia normalized statistic is invalid");
  }
  return restoreStatisticScale(result, scale, skewness ? 6 : 4);
}

} // namespace

Result<MomentValidity> validateMoments(const MomentEstimate& moments,
                                       SolvePolicy policy, Limits limits) {
  const auto dimensionResult = momentDimension(moments);
  if (!dimensionResult) {
    return fail<MomentValidity>(dimensionResult.status().code(),
                                "moment validation shape is invalid");
  }
  const Extent dimension = dimensionResult.value();
  if (dimension == 0U) {
    return fail<MomentValidity>(ErrorCode::InsufficientSamples,
                                "moment validation requires coordinates");
  }
  if (!validPolicy(policy)) {
    return fail<MomentValidity>(ErrorCode::InvalidTolerance,
                                "moment validation policy is invalid");
  }
  if (moments.diagnostics.algorithm.find("central") == std::string::npos) {
    return fail<MomentValidity>(ErrorCode::InvalidDomain,
                                "moment validation requires central moments");
  }
  const Status termStatus = validateMomentTermShapes(moments, dimension);
  if (!termStatus.ok()) {
    return fail<MomentValidity>(termStatus.code(),
                                "moment table structure is invalid");
  }
  if (!std::isfinite(policy.iteration.abs) || policy.iteration.abs < 0.0 ||
      !std::isfinite(policy.iteration.rel) || policy.iteration.rel < 0.0) {
    return fail<MomentValidity>(ErrorCode::InvalidTolerance,
                                "moment symmetry tolerance is invalid");
  }
  const auto gramOrderResult = gramOrder(dimension);
  if (!gramOrderResult) {
    return fail<MomentValidity>(gramOrderResult.status().code(),
                                "moment Gram order is not representable");
  }
  const auto budget = withinBudget(dimension, gramOrderResult.value(), limits);
  if (!budget || !budget.value()) {
    return fail<MomentValidity>(ErrorCode::ResourceLimit,
                                "moment Gram workspace exceeds limits");
  }

  try {
    const Tolerance symmetryTolerance{.absolute = policy.iteration.abs,
                                      .relative = policy.iteration.rel};
    bool symmetric = true;
    std::optional<Support> commonSupport;
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      const std::array<Index, 1U> tuple{coordinate};
      const auto value = symmetricValue(moments, tuple, dimension,
                                        symmetryTolerance, commonSupport);
      if (!value) {
        return fail<MomentValidity>(value.status().code(),
                                    "first central moments are incomplete");
      }
      symmetric = symmetric && value.value().symmetric;
    }
    for (Index first = 0U; first < dimension; ++first) {
      for (Index second = first; second < dimension; ++second) {
        const std::array<Index, 2U> tuple{first, second};
        const auto value = symmetricValue(moments, tuple, dimension,
                                          symmetryTolerance, commonSupport);
        if (!value) {
          return fail<MomentValidity>(value.status().code(),
                                      "second moments are incomplete");
        }
        symmetric = symmetric && value.value().symmetric;
      }
    }
    for (Index first = 0U; first < dimension; ++first) {
      for (Index second = first; second < dimension; ++second) {
        for (Index third = second; third < dimension; ++third) {
          const std::array<Index, 3U> tuple{first, second, third};
          const auto value = symmetricValue(moments, tuple, dimension,
                                            symmetryTolerance, commonSupport);
          if (!value) {
            return fail<MomentValidity>(value.status().code(),
                                        "third moments are incomplete");
          }
          symmetric = symmetric && value.value().symmetric;
        }
      }
    }
    for (Index first = 0U; first < dimension; ++first) {
      for (Index second = first; second < dimension; ++second) {
        for (Index third = second; third < dimension; ++third) {
          for (Index fourth = third; fourth < dimension; ++fourth) {
            const std::array<Index, 4U> tuple{first, second, third, fourth};
            const auto value = symmetricValue(moments, tuple, dimension,
                                              symmetryTolerance, commonSupport);
            if (!value) {
              return fail<MomentValidity>(value.status().code(),
                                          "fourth moments are incomplete");
            }
            symmetric = symmetric && value.value().symmetric;
          }
        }
      }
    }

    Matrix second(dimension, dimension, 0.0);
    Scalar secondScale = 0.0;
    for (Index row = 0U; row < dimension; ++row) {
      for (Index column = 0U; column < dimension; ++column) {
        const std::array<Index, 2U> tuple{row, column};
        const auto value = getSymmetricMoment(moments, tuple, dimension,
                                              symmetryTolerance, commonSupport);
        if (!value) {
          return fail<MomentValidity>(value.status().code(),
                                      "second-moment matrix is incomplete");
        }
        second.at(row, column) = value.value();
        secondScale = std::max(secondScale, std::abs(value.value()));
      }
    }
    if (secondScale == 0.0) {
      secondScale = 1.0;
    }
    for (Index row = 0U; row < dimension; ++row) {
      for (Index column = 0U; column < dimension; ++column) {
        second.at(row, column) /= secondScale;
      }
    }
    auto secondSpectrum = decomposePsd(second, secondScale, policy);
    if (!secondSpectrum) {
      return fail<MomentValidity>(secondSpectrum.status().code(),
                                  "second-moment PSD analysis failed");
    }

    std::vector<Scalar> coordinateScales(dimension, 1.0);
    Scalar largestDiagonal = 0.0;
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      largestDiagonal =
          std::max(largestDiagonal,
                   std::abs(second.at(coordinate, coordinate) * secondScale));
    }
    const Scalar fallbackScale =
        largestDiagonal > 0.0 ? std::sqrt(largestDiagonal) : 1.0;
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      const Scalar variance = second.at(coordinate, coordinate) * secondScale;
      coordinateScales.at(coordinate) =
          variance == 0.0 ? fallbackScale : std::sqrt(std::abs(variance));
      if (!(coordinateScales.at(coordinate) > 0.0) ||
          !std::isfinite(coordinateScales.at(coordinate))) {
        return fail<MomentValidity>(ErrorCode::NonFiniteResult,
                                    "moment coordinate scale is invalid");
      }
    }

    const Extent gramOrder = gramOrderResult.value();
    Matrix gram(gramOrder, gramOrder, 0.0);
    struct Monomial final {
      std::array<Index, 2U> coordinates;
      Index degree;
    };
    std::vector<Monomial> monomials;
    monomials.reserve(gramOrder);
    monomials.push_back(Monomial{.coordinates = {}, .degree = 0U});
    for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
      monomials.push_back(
          Monomial{.coordinates = {coordinate, 0U}, .degree = 1U});
    }
    for (Index first = 0U; first < dimension; ++first) {
      for (Index secondCoordinate = first; secondCoordinate < dimension;
           ++secondCoordinate) {
        monomials.push_back(
            Monomial{.coordinates = {first, secondCoordinate}, .degree = 2U});
      }
    }
    if (monomials.size() != gramOrder) {
      return fail<MomentValidity>(ErrorCode::InvalidExtent,
                                  "degree-two monomial count is inconsistent");
    }
    for (Index row = 0U; row < gramOrder; ++row) {
      for (Index column = row; column < gramOrder; ++column) {
        const Monomial& lhs = monomials.at(row);
        const Monomial& rhs = monomials.at(column);
        std::array<Index, 4U> tuple{};
        Index order = 0U;
        for (Index axis = 0U; axis < lhs.degree; ++axis) {
          tuple.at(order++) = lhs.coordinates.at(axis);
        }
        for (Index axis = 0U; axis < rhs.degree; ++axis) {
          tuple.at(order++) = rhs.coordinates.at(axis);
        }
        Scalar moment = 1.0;
        if (order > 0U) {
          const auto value = getSymmetricMoment(
              moments, std::span<const Index>(tuple.data(), order), dimension,
              symmetryTolerance, commonSupport);
          if (!value) {
            return fail<MomentValidity>(
                value.status().code(),
                "degree-two Gram needs a missing moment");
          }
          moment = value.value();
        }
        long double normalized = moment;
        for (Index axis = 0U; axis < lhs.degree; ++axis) {
          normalized /= coordinateScales.at(lhs.coordinates.at(axis));
        }
        for (Index axis = 0U; axis < rhs.degree; ++axis) {
          normalized /= coordinateScales.at(rhs.coordinates.at(axis));
        }
        if (!std::isfinite(normalized) ||
            std::abs(normalized) > std::numeric_limits<Scalar>::max() ||
            (normalized != 0.0L && static_cast<Scalar>(normalized) == 0.0)) {
          return fail<MomentValidity>(
              ErrorCode::NonFiniteResult,
              "normalized Gram entry is not representable");
        }
        const Scalar entry = static_cast<Scalar>(normalized);
        gram.at(row, column) = entry;
        if (row != column) {
          setSymmetricEntry(gram, row, column, entry);
        }
      }
    }
    Scalar gramScale = 0.0;
    for (Index row = 0U; row < gramOrder; ++row) {
      for (Index column = 0U; column < gramOrder; ++column) {
        gramScale = std::max(gramScale, std::abs(gram.at(row, column)));
      }
    }
    if (gramScale == 0.0) {
      gramScale = 1.0;
    }
    for (Index row = 0U; row < gramOrder; ++row) {
      for (Index column = 0U; column < gramOrder; ++column) {
        gram.at(row, column) /= gramScale;
      }
    }
    auto gramSpectrum = decomposePsd(gram, gramScale, policy);
    if (!gramSpectrum) {
      return fail<MomentValidity>(gramSpectrum.status().code(),
                                  "degree-two Gram PSD analysis failed");
    }
    const auto normalizedSecondTolerance =
        normalizedAbsoluteTolerance(policy.rank_abs, secondScale);
    const auto normalizedGramTolerance =
        normalizedAbsoluteTolerance(policy.rank_abs, gramScale);
    if (!normalizedSecondTolerance || !normalizedGramTolerance) {
      return fail<MomentValidity>(ErrorCode::NonFiniteResult,
                                  "PSD tolerance cannot be normalized");
    }
    return Result<MomentValidity>::success(MomentValidity{
        .symmetric = symmetric,
        .second_psd = secondSpectrum.value().definiteness() !=
                      linear::PositiveDefiniteness::NotPositiveSemidefinite,
        .degree_two_moment_gram_psd =
            gramSpectrum.value().definiteness() !=
            linear::PositiveDefiniteness::NotPositiveSemidefinite,
        .sufficient_for_model = false,
        .symmetry_tolerance_absolute = policy.iteration.abs,
        .symmetry_tolerance_relative = policy.iteration.rel,
        .second_scale = secondScale,
        .second_minimum_eigenvalue_normalized =
            secondSpectrum.value().minimumEigenvalue(),
        .second_psd_tolerance_normalized = std::max(
            normalizedSecondTolerance.value(),
            policy.rank_rel *
                std::max(std::abs(secondSpectrum.value().maximumEigenvalue()),
                         std::abs(secondSpectrum.value().minimumEigenvalue()))),
        .gram_scale = gramScale,
        .gram_minimum_eigenvalue_normalized =
            gramSpectrum.value().minimumEigenvalue(),
        .gram_psd_tolerance_normalized = std::max(
            normalizedGramTolerance.value(),
            policy.rank_rel *
                std::max(std::abs(gramSpectrum.value().maximumEigenvalue()),
                         std::abs(gramSpectrum.value().minimumEigenvalue())))});
  } catch (const std::bad_alloc&) {
    return fail<MomentValidity>(ErrorCode::ResourceLimit,
                                "moment validation allocation failed");
  }
}

Result<Scalar> mardiaSkewness(const SampleView& sample, SolvePolicy policy,
                              Limits limits) {
  return mardia(sample, policy, limits, true);
}

Result<Scalar> mardiaKurtosis(const SampleView& sample, SolvePolicy policy,
                              Limits limits) {
  return mardia(sample, policy, limits, false);
}

} // namespace vss::math::statistics
