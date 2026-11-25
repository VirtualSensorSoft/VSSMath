#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
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
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Empirical.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

Result<Extent> validateEdges(std::span<const Vector> edges, Limits limits) {
  if (edges.empty()) {
    return fail<Extent>(ErrorCode::InvalidExtent,
                        "a histogram requires at least one axis");
  }
  Extent bins = 1U;
  for (const Vector& axis : edges) {
    if (axis.size() < 2U) {
      return fail<Extent>(ErrorCode::InvalidExtent,
                          "each histogram axis requires at least two edges");
    }
    for (Extent i = 0U; i < axis.size(); ++i) {
      if (!std::isfinite(axis.at(i))) {
        return fail<Extent>(ErrorCode::NonFiniteInput,
                            "histogram edges must be finite");
      }
      if (i != 0U && !(axis.at(i - 1U) < axis.at(i))) {
        return fail<Extent>(ErrorCode::InvalidDomain,
                            "histogram edges must be strictly increasing");
      }
    }
    const Extent axisBins = axis.size() - 1U;
    if (bins > std::numeric_limits<Extent>::max() / axisBins) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "histogram bin-count product overflows");
    }
    bins *= axisBins;
  }
  if (bins > limits.elements ||
      bins > limits.bytes / (sizeof(Scalar) + sizeof(long double))) {
    return fail<Extent>(ErrorCode::ResourceLimit,
                        "histogram bins exceed configured limits");
  }
  return Result<Extent>::success(bins);
}

Result<std::vector<long double>> rowMasses(const WeightSpec& weights,
                                           Extent rows) {
  std::vector<long double> masses(rows, 1.0L);
  if (weights.rule == WeightRule::unweighted) {
    if (weights.rowWeights || weights.componentWeights ||
        !weights.frequencyCounts.empty()) {
      return fail<std::vector<long double>>(
          ErrorCode::InvalidDomain,
          "unweighted histogram cannot receive weight arrays");
    }
    return Result<std::vector<long double>>::success(std::move(masses));
  }
  if (weights.rule == WeightRule::componentwise ||
      weights.rule == WeightRule::precision || weights.componentWeights) {
    return fail<std::vector<long double>>(
        ErrorCode::InvalidDomain,
        "histograms require one nonnegative scalar mass per row");
  }
  if (!weights.frequencyCounts.empty()) {
    if (weights.rule != WeightRule::frequency || weights.rowWeights ||
        weights.frequencyCounts.size() != rows) {
      return fail<std::vector<long double>>(
          ErrorCode::DimensionMismatch,
          "frequency counts must be the sole row mass and match the sample");
    }
    for (Extent i = 0U; i < rows; ++i) {
      masses[i] = static_cast<long double>(weights.frequencyCounts[i]);
    }
    return Result<std::vector<long double>>::success(std::move(masses));
  }
  if (!weights.rowWeights || weights.rowWeights->size() != rows) {
    return fail<std::vector<long double>>(
        ErrorCode::DimensionMismatch,
        "weighted histogram requires one row weight per sample row");
  }
  for (Extent i = 0U; i < rows; ++i) {
    const Scalar value = weights.rowWeights->at(i);
    if (!std::isfinite(value)) {
      return fail<std::vector<long double>>(ErrorCode::NonFiniteInput,
                                            "histogram weights must be finite");
    }
    if (value < 0.0) {
      return fail<std::vector<long double>>(
          ErrorCode::InvalidDomain, "histogram weights must be nonnegative");
    }
    masses[i] = value;
  }
  return Result<std::vector<long double>>::success(std::move(masses));
}

Result<Scalar> representable(long double value, const char* message) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<Scalar>::max()) {
    return fail<Scalar>(ErrorCode::NonFiniteResult, message);
  }
  return Result<Scalar>::success(static_cast<Scalar>(value));
}

Result<Histogram> buildHistogram(const SampleView& sample,
                                 std::span<const Vector> edges,
                                 const WeightSpec& weights,
                                 HistogramNormalization mode, Limits limits) {
  if (mode != HistogramNormalization::counts &&
      mode != HistogramNormalization::mass &&
      mode != HistogramNormalization::density) {
    return fail<Histogram>(ErrorCode::InvalidDomain,
                           "unsupported histogram normalization");
  }
  const Extent rows = sample.rows.rows();
  const Extent dimensions = sample.rows.columns();
  if (rows == 0U || dimensions == 0U) {
    return fail<Histogram>(ErrorCode::InsufficientSamples,
                           "histogram requires nonempty observations");
  }
  if (!sample.valid.empty() &&
      (rows > std::numeric_limits<Extent>::max() / dimensions ||
       sample.valid.size() != rows * dimensions)) {
    return fail<Histogram>(
        ErrorCode::DimensionMismatch,
        "histogram mask must have one entry per sample cell");
  }
  if (edges.size() != dimensions) {
    return fail<Histogram>(
        ErrorCode::DimensionMismatch,
        "histogram edge axes must match the observation dimension");
  }
  if (rows > std::numeric_limits<Extent>::max() / dimensions) {
    return fail<Histogram>(ErrorCode::SizeOverflow,
                           "sample cell count overflows");
  }
  for (const std::uint8_t bit : sample.valid) {
    if (bit > 1U) {
      return fail<Histogram>(
          ErrorCode::InvalidDomain,
          "histogram validity mask entries must be zero or one");
    }
    if (bit == 0U) {
      return fail<Histogram>(ErrorCode::InvalidDomain,
                             "histograms require complete observations");
    }
  }
  auto binCount = validateEdges(edges, limits);
  if (!binCount) {
    return fail<Histogram>(binCount.status().code(),
                           std::string(binCount.status().message()));
  }
  auto masses = rowMasses(weights, rows);
  if (!masses) {
    return fail<Histogram>(masses.status().code(),
                           std::string(masses.status().message()));
  }
  std::size_t requiredBytes = 0U;
  std::size_t requiredElements = binCount.value();
  const auto addStorage = [&](Extent count, std::size_t elementSize,
                              std::size_t& total) {
    if (count > std::numeric_limits<std::size_t>::max() / elementSize) {
      return false;
    }
    const std::size_t bytes = count * elementSize;
    if (bytes > limits.bytes - total) {
      return false;
    }
    total += bytes;
    return true;
  };
  if (!addStorage(binCount.value(), sizeof(long double), requiredBytes) ||
      !addStorage(binCount.value(), sizeof(Scalar), requiredBytes) ||
      !addStorage(edges.size(), sizeof(Index), requiredBytes) ||
      !addStorage(edges.size(), sizeof(Vector), requiredBytes) ||
      !addStorage(rows, sizeof(long double), requiredBytes)) {
    return fail<Histogram>(ErrorCode::ResourceLimit,
                           "histogram workspace exceeds configured limits");
  }
  for (const Vector& axis : edges) {
    if (axis.size() > limits.elements - requiredElements) {
      return fail<Histogram>(ErrorCode::ResourceLimit,
                             "histogram edge copies exceed element limits");
    }
    requiredElements += axis.size();
    if (!addStorage(axis.size(), sizeof(Scalar), requiredBytes)) {
      return fail<Histogram>(ErrorCode::ResourceLimit,
                             "histogram edge copies exceed byte limits");
    }
  }
  if (requiredElements > limits.elements) {
    return fail<Histogram>(
        ErrorCode::ResourceLimit,
        "histogram output exceeds configured element limits");
  }
  auto extents = std::vector<Index>{};
  extents.reserve(dimensions);
  for (const Vector& axis : edges) {
    extents.push_back(axis.size() - 1U);
  }
  std::vector<long double> binMass(binCount.value(), 0.0L);
  long double underflow = 0.0L;
  long double overflow = 0.0L;
  long double accepted = 0.0L;
  for (Extent row = 0U; row < rows; ++row) {
    bool below = false;
    bool above = false;
    Extent flattened = 0U;
    for (Extent axis = 0U; axis < dimensions; ++axis) {
      const Scalar value = sample.rows.at(row, axis);
      if (!std::isfinite(value)) {
        return fail<Histogram>(ErrorCode::NonFiniteInput,
                               "histogram observations must be finite");
      }
      const Vector& axisEdges = edges[axis];
      if (value < axisEdges.at(0U)) {
        below = true;
        continue;
      }
      if (value > axisEdges.at(axisEdges.size() - 1U)) {
        above = true;
        continue;
      }
      Extent first = 0U;
      Extent last = axisEdges.size();
      while (first < last) {
        const Extent middle = first + (last - first) / 2U;
        if (value < axisEdges.at(middle)) {
          last = middle;
        } else {
          first = middle + 1U;
        }
      }
      const Extent index = value == axisEdges.at(axisEdges.size() - 1U)
                               ? axisEdges.size() - 2U
                               : first - 1U;
      flattened = flattened * extents[axis] + index;
    }
    if (below) {
      underflow += masses.value()[row];
    } else if (above) {
      overflow += masses.value()[row];
    } else {
      accepted += masses.value()[row];
      binMass[flattened] += masses.value()[row];
    }
  }
  if (mode != HistogramNormalization::counts && accepted <= 0.0L) {
    return fail<Histogram>(
        ErrorCode::InvalidDomain,
        "normalized histogram has no positive in-range mass");
  }
  Vector output(binCount.value(), 0.0);
  for (Extent flat = 0U; flat < binCount.value(); ++flat) {
    long double value = binMass[flat];
    if (mode != HistogramNormalization::counts) {
      value /= accepted;
    }
    if (mode == HistogramNormalization::density && value > 0.0L) {
      Extent remainder = flat;
      long double logVolume = 0.0L;
      for (Extent axis = dimensions; axis-- > 0U;) {
        const Extent bin = remainder % extents[axis];
        remainder /= extents[axis];
        const long double width =
            static_cast<long double>(edges[axis].at(bin + 1U)) -
            static_cast<long double>(edges[axis].at(bin));
        logVolume += std::log(width);
      }
      const long double logDensity = std::log(value) - logVolume;
      if (logDensity > std::log(static_cast<long double>(
                           std::numeric_limits<Scalar>::max())) ||
          logDensity < std::log(static_cast<long double>(
                           std::numeric_limits<Scalar>::denorm_min()))) {
        return fail<Histogram>(ErrorCode::NonFiniteResult,
                               "histogram density is not representable");
      }
      value = std::exp(logDensity);
    }
    auto scalar = representable(value, "histogram bin is not representable");
    if (!scalar) {
      return fail<Histogram>(scalar.status().code(),
                             std::string(scalar.status().message()));
    }
    output.at(flat) = scalar.value();
  }
  auto extentsArray = Array<Index>::copy(extents, limits);
  auto edgesArray = Array<Vector>::copy(edges, limits);
  auto under =
      representable(underflow, "histogram underflow is not representable");
  auto over =
      representable(overflow, "histogram overflow is not representable");
  auto acceptedValue =
      representable(accepted, "histogram accepted mass is not representable");
  if (!extentsArray) {
    return fail<Histogram>(extentsArray.status().code(),
                           std::string(extentsArray.status().message()));
  }
  if (!edgesArray) {
    return fail<Histogram>(edgesArray.status().code(),
                           std::string(edgesArray.status().message()));
  }
  if (!under) {
    return fail<Histogram>(under.status().code(),
                           std::string(under.status().message()));
  }
  if (!over) {
    return fail<Histogram>(over.status().code(),
                           std::string(over.status().message()));
  }
  if (!acceptedValue) {
    return fail<Histogram>(acceptedValue.status().code(),
                           std::string(acceptedValue.status().message()));
  }
  return Result<Histogram>::success(
      Histogram{.extents = std::move(extentsArray).value(),
                .edges = std::move(edgesArray).value(),
                .bins = std::move(output),
                .underflow = under.value(),
                .overflow = over.value(),
                .accepted = acceptedValue.value()});
}

Status validateDensitySample(const SampleView& sample, bool allowEmpty) {
  const Extent rows = sample.rows.rows();
  const Extent dimensions = sample.rows.columns();
  if (dimensions == 0U || (!allowEmpty && rows == 0U)) {
    return Status::failure(ErrorCode::InsufficientSamples,
                           "KDE requires positive observation dimensions and "
                           "nonempty training data");
  }
  if (rows > std::numeric_limits<Extent>::max() / dimensions) {
    return Status::failure(ErrorCode::SizeOverflow,
                           "KDE sample cell count overflows");
  }
  if (!sample.valid.empty() && sample.valid.size() != rows * dimensions) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "KDE validity mask must match sample cells");
  }
  for (const std::uint8_t bit : sample.valid) {
    if (bit > 1U) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "KDE validity mask entries must be zero or one");
    }
    if (bit == 0U) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "KDE requires complete observations");
    }
  }
  for (Extent i = 0U; i < rows; ++i) {
    for (Extent j = 0U; j < dimensions; ++j) {
      if (!std::isfinite(sample.rows.at(i, j))) {
        return Status::failure(ErrorCode::NonFiniteInput,
                               "KDE sample values must be finite");
      }
    }
  }
  return Status::success();
}

Status validateProbabilityAccuracy(const ProbabilityPolicy& accuracy) {
  if (!std::isfinite(accuracy.iteration.abs) ||
      !std::isfinite(accuracy.iteration.rel) || accuracy.iteration.abs < 0.0 ||
      accuracy.iteration.rel < 0.0 || !std::isfinite(accuracy.abs_log_error) ||
      accuracy.abs_log_error < 0.0) {
    return Status::failure(
        ErrorCode::InvalidTolerance,
        "KDE accuracy tolerances must be finite and nonnegative");
  }
  if (accuracy.iteration.max_iterations == 0U ||
      accuracy.iteration.max_evaluations == 0U) {
    return Status::failure(ErrorCode::InvalidTolerance,
                           "KDE image work caps must be positive");
  }
  return Status::success();
}

long double kernelLog(KernelKind kind, long double coordinate,
                      long double bandwidth) {
  const long double standardized = coordinate / bandwidth;
  if (kind == KernelKind::gaussian) {
    if (!std::isfinite(standardized)) {
      return -std::numeric_limits<long double>::infinity();
    }
    return -0.5L * standardized * standardized -
           0.5L * std::log(2.0L * std::numbers::pi_v<long double>) -
           std::log(bandwidth);
  }
  if (std::abs(standardized) >= 1.0L) {
    return -std::numeric_limits<long double>::infinity();
  }
  return std::log(0.75L * (1.0L - standardized * standardized)) -
         std::log(bandwidth);
}

Result<long double>
reflectedAxis(KernelKind kind, long double observation, long double evaluation,
              long double lower, long double upper, long double bandwidth,
              long double absoluteTolerance, Index iterationCap,
              std::uint64_t evaluationCap, std::uint64_t& evaluations) {
  const long double length = upper - lower;
  const long double period = 2.0L * length;
  long double sum = 0.0L;
  if (kind == KernelKind::gaussian) {
    if (!(absoluteTolerance > 0.0L)) {
      return fail<long double>(
          ErrorCode::InvalidTolerance,
          "reflected Gaussian KDE requires a positive absolute tail tolerance");
    }
    Index maximum = 0U;
    bool bounded = false;
    for (Index k = 1U; k <= iterationCap; ++k) {
      const long double distance = 2.0L * k * length;
      const long double exponent =
          -0.5L * distance * distance / (bandwidth * bandwidth);
      const long double first =
          std::exp(exponent) /
          (std::sqrt(2.0L * std::numbers::pi_v<long double>) * bandwidth);
      const long double ratio = std::exp(
          -0.5L *
          ((distance + period) * (distance + period) - distance * distance) /
          (bandwidth * bandwidth));
      const long double tail =
          ratio >= 1.0L ? std::numeric_limits<long double>::infinity()
                        : 4.0L * first / (1.0L - ratio);
      if (tail <= absoluteTolerance) {
        maximum = k;
        bounded = true;
        break;
      }
    }
    if (!bounded) {
      return fail<long double>(ErrorCode::NoConvergence,
                               "Gaussian reflected KDE tail bound did not "
                               "converge within the iteration cap");
    }
    if (maximum >
        static_cast<Index>(std::numeric_limits<std::uint64_t>::max() / 4U)) {
      return fail<long double>(
          ErrorCode::ResourceLimit,
          "Gaussian reflected KDE image count is not representable");
    }
    const std::uint64_t terms =
        2U * (2U * static_cast<std::uint64_t>(maximum) + 1U);
    if (terms > evaluationCap - evaluations) {
      return fail<long double>(ErrorCode::NoConvergence,
                               "Gaussian reflected KDE image budget exhausted");
    }
    for (std::int64_t k = -static_cast<std::int64_t>(maximum);
         k <= static_cast<std::int64_t>(maximum); ++k) {
      const long double shift = static_cast<long double>(k) * period;
      sum += std::exp(
          kernelLog(kind, evaluation - observation - shift, bandwidth));
      sum += std::exp(kernelLog(
          kind, evaluation - (2.0L * lower - observation + shift), bandwidth));
      evaluations += 2U;
    }
    return Result<long double>::success(sum);
  }

  const long double directLower =
      std::ceil((evaluation - observation - bandwidth) / period);
  const long double directUpper =
      std::floor((evaluation - observation + bandwidth) / period);
  const long double mirror = 2.0L * lower - observation;
  const long double reflectedLower =
      std::ceil((evaluation - mirror - bandwidth) / period);
  const long double reflectedUpper =
      std::floor((evaluation - mirror + bandwidth) / period);
  const long double imageCount =
      std::max(0.0L, directUpper - directLower + 1.0L) +
      std::max(0.0L, reflectedUpper - reflectedLower + 1.0L);
  if (!std::isfinite(imageCount) || imageCount > evaluationCap - evaluations ||
      std::abs(directLower) > std::numeric_limits<std::int64_t>::max() ||
      std::abs(directUpper) >= std::numeric_limits<std::int64_t>::max() ||
      std::abs(reflectedLower) > std::numeric_limits<std::int64_t>::max() ||
      std::abs(reflectedUpper) >= std::numeric_limits<std::int64_t>::max()) {
    return fail<long double>(
        ErrorCode::NoConvergence,
        "Epanechnikov reflected KDE image budget is not met");
  }
  const auto accumulate = [&](long double center, long double first,
                              long double last) {
    if (last < first) {
      return;
    }
    for (auto k = static_cast<std::int64_t>(first);
         k <= static_cast<std::int64_t>(last); ++k) {
      const long double shift = static_cast<long double>(k) * period;
      sum +=
          std::exp(kernelLog(kind, evaluation - (center + shift), bandwidth));
      ++evaluations;
    }
  };
  accumulate(observation, directLower, directUpper);
  accumulate(mirror, reflectedLower, reflectedUpper);
  return Result<long double>::success(sum);
}

Result<Vector> evaluateKde(const SampleView& observations,
                           const WeightSpec& weights,
                           const SampleView& evaluation,
                           const KernelSpec& kernel,
                           const ProbabilityPolicy& accuracy, Limits limits) {
  const Status accuracyStatus = validateProbabilityAccuracy(accuracy);
  if (!accuracyStatus.ok()) {
    return fail<Vector>(accuracyStatus.code(),
                        std::string(accuracyStatus.message()));
  }
  const Status observationStatus = validateDensitySample(observations, false);
  if (!observationStatus.ok()) {
    return fail<Vector>(observationStatus.code(),
                        std::string(observationStatus.message()));
  }
  const Status evaluationStatus = validateDensitySample(evaluation, true);
  if (!evaluationStatus.ok()) {
    return fail<Vector>(evaluationStatus.code(),
                        std::string(evaluationStatus.message()));
  }
  const Extent n = observations.rows.rows();
  const Extent dimensions = observations.rows.columns();
  const Extent points = evaluation.rows.rows();
  if (evaluation.rows.columns() != dimensions) {
    return fail<Vector>(ErrorCode::DimensionMismatch,
                        "KDE training and evaluation dimensions differ");
  }
  std::size_t requiredBytes = 0U;
  const auto addWorkspace = [&](Extent count, std::size_t elementSize) {
    if (count > std::numeric_limits<std::size_t>::max() / elementSize) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(count) * elementSize;
    if (bytes > limits.bytes - requiredBytes) {
      return false;
    }
    requiredBytes += bytes;
    return true;
  };
  if (points > limits.elements || !addWorkspace(points, sizeof(Scalar)) ||
      !addWorkspace(n, sizeof(long double)) ||
      !addWorkspace(n, sizeof(long double)) ||
      !addWorkspace(dimensions, sizeof(long double)) ||
      !addWorkspace(dimensions, sizeof(long double)) ||
      !addWorkspace(dimensions, sizeof(long double)) ||
      !addWorkspace(dimensions, sizeof(long double))) {
    return fail<Vector>(ErrorCode::ResourceLimit,
                        "KDE output and workspace exceed configured limits");
  }
  if (kernel.kernel != KernelKind::gaussian &&
      kernel.kernel != KernelKind::epanechnikov) {
    return fail<Vector>(ErrorCode::InvalidDomain,
                        "unsupported KDE kernel kind");
  }
  if (kernel.bandwidth.size() != 1U && kernel.bandwidth.size() != dimensions) {
    return fail<Vector>(
        ErrorCode::DimensionMismatch,
        "KDE bandwidth must be scalar or diagonal by coordinate");
  }
  std::vector<long double> bandwidths(dimensions, 0.0L);
  long double logMaximumProduct = 0.0L;
  std::vector<long double> logAxisMaximum(dimensions, 0.0L);
  for (Extent j = 0U; j < dimensions; ++j) {
    const Scalar h =
        kernel.bandwidth.at(kernel.bandwidth.size() == 1U ? 0U : j);
    if (!std::isfinite(h) || h <= 0.0) {
      return fail<Vector>(ErrorCode::InvalidDomain,
                          "KDE bandwidths must be finite and positive");
    }
    bandwidths[j] = h;
    const long double maximum =
        (kernel.kernel == KernelKind::gaussian
             ? 1.0L / std::sqrt(2.0L * std::numbers::pi_v<long double>)
             : 0.75L) /
        h;
    logAxisMaximum[j] = std::log(maximum);
    logMaximumProduct += logAxisMaximum[j];
  }
  const bool reflected = kernel.reflection_bounds.has_value();
  if (reflected && (kernel.reflection_bounds->rows() != dimensions ||
                    kernel.reflection_bounds->columns() != 2U)) {
    return fail<Vector>(ErrorCode::DimensionMismatch,
                        "KDE reflection bounds must have shape d-by-2");
  }
  std::vector<long double> lower(dimensions, 0.0L);
  std::vector<long double> upper(dimensions, 0.0L);
  if (reflected) {
    for (Extent j = 0U; j < dimensions; ++j) {
      lower[j] = kernel.reflection_bounds->at(j, 0U);
      upper[j] = kernel.reflection_bounds->at(j, 1U);
      if (!std::isfinite(lower[j]) || !std::isfinite(upper[j]) ||
          !(lower[j] < upper[j])) {
        return fail<Vector>(
            ErrorCode::InvalidDomain,
            "KDE reflection intervals must be finite and nonempty");
      }
      for (Extent i = 0U; i < n; ++i) {
        const long double value = observations.rows.at(i, j);
        if (value < lower[j] || value > upper[j]) {
          return fail<Vector>(
              ErrorCode::InvalidDomain,
              "reflected KDE observations must lie within bounds");
        }
      }
    }
  }
  auto rowWeights = rowMasses(weights, n);
  if (!rowWeights) {
    return fail<Vector>(rowWeights.status().code(),
                        std::string(rowWeights.status().message()));
  }
  long double maximumWeight = 0.0L;
  for (const long double weight : rowWeights.value()) {
    maximumWeight = std::max(maximumWeight, weight);
  }
  if (!(maximumWeight > 0.0L)) {
    return fail<Vector>(ErrorCode::InvalidDomain,
                        "KDE requires positive total row weight");
  }
  long double scaledMass = 0.0L;
  for (const long double weight : rowWeights.value()) {
    scaledMass += weight / maximumWeight;
  }
  Vector output(points, 0.0);
  std::uint64_t imageEvaluations = 0U;
  const std::uint64_t imageCap = std::min<std::uint64_t>(
      accuracy.iteration.max_evaluations, limits.evaluations);
  for (Extent point = 0U; point < points; ++point) {
    bool outside = false;
    if (reflected) {
      for (Extent j = 0U; j < dimensions; ++j) {
        const long double value = evaluation.rows.at(point, j);
        outside = outside || value < lower[j] || value > upper[j];
      }
    }
    if (outside) {
      continue;
    }
    std::vector<long double> logTerms;
    logTerms.reserve(n);
    long double maximumLog = -std::numeric_limits<long double>::infinity();
    for (Extent row = 0U; row < n; ++row) {
      long double logValue = std::log(rowWeights.value()[row] / maximumWeight) -
                             std::log(scaledMass);
      if (rowWeights.value()[row] == 0.0L) {
        logTerms.push_back(-std::numeric_limits<long double>::infinity());
        continue;
      }
      for (Extent j = 0U; j < dimensions; ++j) {
        const long double observedValue = observations.rows.at(row, j);
        const long double queryValue = evaluation.rows.at(point, j);
        long double axisDensity = 0.0L;
        if (!reflected) {
          axisDensity = std::exp(kernelLog(
              kernel.kernel, queryValue - observedValue, bandwidths[j]));
        } else {
          const long double logTolerance =
              std::log(static_cast<long double>(accuracy.iteration.abs)) -
              std::log(static_cast<long double>(dimensions)) -
              logMaximumProduct + logAxisMaximum[j];
          const long double axisTolerance =
              logTolerance <
                      std::log(std::numeric_limits<long double>::denorm_min())
                  ? 0.0L
                  : std::exp(logTolerance);
          auto reflectedValue = reflectedAxis(
              kernel.kernel, observedValue, queryValue, lower[j], upper[j],
              bandwidths[j], axisTolerance, accuracy.iteration.max_iterations,
              imageCap, imageEvaluations);
          if (!reflectedValue) {
            return fail<Vector>(reflectedValue.status().code(),
                                std::string(reflectedValue.status().message()));
          }
          axisDensity = reflectedValue.value();
        }
        if (!(axisDensity > 0.0L)) {
          logValue = -std::numeric_limits<long double>::infinity();
          break;
        }
        logValue += std::log(axisDensity);
      }
      logTerms.push_back(logValue);
      maximumLog = std::max(maximumLog, logValue);
    }
    if (maximumLog == -std::numeric_limits<long double>::infinity()) {
      continue;
    }
    long double scaledDensity = 0.0L;
    for (const long double term : logTerms) {
      if (std::isfinite(term)) {
        scaledDensity += std::exp(term - maximumLog);
      }
    }
    const long double density = std::exp(maximumLog) * scaledDensity;
    if (!std::isfinite(density) ||
        density > std::numeric_limits<Scalar>::max()) {
      return fail<Vector>(ErrorCode::NonFiniteResult,
                          "KDE density is not representable");
    }
    output.at(point) = static_cast<Scalar>(density);
  }
  return Result<Vector>::success(std::move(output));
}

} // namespace

Result<Histogram> histogram(ConstVectorView values, ConstVectorView edges,
                            HistogramNormalization mode, Limits limits) {
  if (values.size() > limits.elements ||
      values.size() > limits.bytes / (sizeof(Scalar) + sizeof(long double)) ||
      edges.size() > limits.elements ||
      edges.size() > limits.bytes / sizeof(Scalar)) {
    return fail<Histogram>(ErrorCode::ResourceLimit,
                           "histogram input staging exceeds configured limits");
  }
  Matrix rows(values.size(), 1U, 0.0);
  for (Extent i = 0U; i < values.size(); ++i) {
    rows.at(i, 0U) = values.at(i);
  }
  Vector axis(edges.size(), 0.0);
  for (Extent i = 0U; i < edges.size(); ++i) {
    axis.at(i) = edges.at(i);
  }
  const std::array<Vector, 1U> axes{std::move(axis)};
  const WeightSpec weights{.rule = WeightRule::unweighted,
                           .rowWeights = std::nullopt,
                           .componentWeights = std::nullopt,
                           .frequencyCounts = {}};
  return buildHistogram(
      SampleView{.rows = rows.view(), .observation_shape = {}, .valid = {}},
      axes, weights, mode, limits);
}

Result<Histogram> weightedHistogram(ConstVectorView values,
                                    ConstVectorView weights,
                                    ConstVectorView edges,
                                    HistogramNormalization mode,
                                    Limits limits) {
  if (values.size() > limits.elements ||
      values.size() > limits.bytes / (sizeof(Scalar) + sizeof(long double)) ||
      weights.size() > limits.elements ||
      weights.size() > limits.bytes / sizeof(Scalar) ||
      edges.size() > limits.elements ||
      edges.size() > limits.bytes / sizeof(Scalar)) {
    return fail<Histogram>(
        ErrorCode::ResourceLimit,
        "weighted histogram input staging exceeds configured limits");
  }
  Matrix rows(values.size(), 1U, 0.0);
  for (Extent i = 0U; i < values.size(); ++i) {
    rows.at(i, 0U) = values.at(i);
  }
  Vector axis(edges.size(), 0.0);
  Vector rowWeights(weights.size(), 0.0);
  for (Extent i = 0U; i < edges.size(); ++i) {
    axis.at(i) = edges.at(i);
  }
  for (Extent i = 0U; i < weights.size(); ++i) {
    rowWeights.at(i) = weights.at(i);
  }
  const std::array<Vector, 1U> axes{std::move(axis)};
  const WeightSpec weightSpec{.rule = WeightRule::empirical,
                              .rowWeights = rowWeights.view(),
                              .componentWeights = std::nullopt,
                              .frequencyCounts = {}};
  return buildHistogram(
      SampleView{.rows = rows.view(), .observation_shape = {}, .valid = {}},
      axes, weightSpec, mode, limits);
}

Result<Histogram> histogramND(const SampleView& values,
                              std::span<const Vector> edges,
                              const WeightSpec& weights,
                              HistogramNormalization mode, Limits limits) {
  return buildHistogram(values, edges, weights, mode, limits);
}

Result<Vector> kernelDensityEstimate(const SampleView& observations,
                                     const WeightSpec& weights,
                                     const SampleView& evaluation,
                                     const KernelSpec& kernel, Limits limits) {
  const ProbabilityPolicy accuracy{.iteration = {.abs = 1.0e-13,
                                                 .rel = 1.0e-11,
                                                 .max_iterations = 4096U,
                                                 .max_evaluations = 4096U,
                                                 .bracket = std::nullopt},
                                   .max_draws = 1U,
                                   .abs_log_error = 1.0e-11};
  return evaluateKde(observations, weights, evaluation, kernel, accuracy,
                     limits);
}

Result<Vector>
kernelDensityEstimate(const SampleView& observations, const WeightSpec& weights,
                      const SampleView& evaluation, const KernelSpec& kernel,
                      ProbabilityPolicy accuracy, Limits limits) {
  return evaluateKde(observations, weights, evaluation, kernel, accuracy,
                     limits);
}

} // namespace vss::math::statistics
