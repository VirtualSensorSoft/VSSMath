#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/statistics/Empirical.hpp>

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
  if (sample.valid.size() != 0U &&
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
      const auto begin = axisEdges.data();
      const auto end = begin + axisEdges.size();
      const auto upper = std::upper_bound(begin, end, value);
      const Extent index = value == axisEdges.at(axisEdges.size() - 1U)
                               ? axisEdges.size() - 2U
                               : static_cast<Extent>(upper - begin - 1);
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
  if (!extentsArray || !edgesArray || !under || !over || !acceptedValue) {
    const Status& status = !extentsArray ? extentsArray.status()
                           : !edgesArray ? edgesArray.status()
                           : !under      ? under.status()
                           : !over       ? over.status()
                                         : acceptedValue.status();
    return fail<Histogram>(status.code(), std::string(status.message()));
  }
  return Result<Histogram>::success(
      Histogram{.extents = std::move(extentsArray).value(),
                .edges = std::move(edgesArray).value(),
                .bins = std::move(output),
                .underflow = under.value(),
                .overflow = over.value(),
                .accepted = acceptedValue.value()});
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

} // namespace vss::math::statistics
