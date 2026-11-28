#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorViews.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/signal/LaggedMoments.hpp>
#include <vss/math/signal/Series.hpp>
#include <vss/math/statistics/MomentState.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::signal {
namespace {

template <class T> Result<T> fail(ErrorCode code, std::string message) {
  return Result<T>::failure(code, std::move(message));
}

bool checkedAdd(Index left, Index right, Index& result) noexcept {
  if (right > std::numeric_limits<Index>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(Index left, Index right, Index& result) noexcept {
  if (left != 0U && right > std::numeric_limits<Index>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

ErrorCode validateGrid(const SeriesView& series) {
  const Index count = series.times().size();
  if (series.rows().rows() != count || series.rows().columns() == 0U ||
      (!series.valid().empty() && series.valid().size() != count)) {
    return ErrorCode::DimensionMismatch;
  }
  if (!series.uniformStep().has_value()) {
    return ErrorCode::InvalidDomain;
  }
  const Scalar step = series.uniformStep().value();
  if (!std::isfinite(step) || step <= 0.0) {
    return ErrorCode::InvalidDomain;
  }
  for (const std::uint8_t flag : series.valid()) {
    if (flag > 1U) {
      return ErrorCode::InvalidDomain;
    }
  }
  for (Index row = 0U; row < count; ++row) {
    const Scalar time = series.times().at(row);
    if (!std::isfinite(time)) {
      return ErrorCode::NonFiniteInput;
    }
    if (row > 0U && !(time > series.times().at(row - 1U))) {
      return ErrorCode::InvalidDomain;
    }
  }

  if (count > 1U) {
    using Wide = long double;
    const Wide origin = static_cast<Wide>(series.times().at(0U));
    const Wide wideStep = static_cast<Wide>(step);
    for (Index row = 1U; row < count; ++row) {
      const Wide elapsed = static_cast<Wide>(series.times().at(row)) - origin;
      const Wide expected = static_cast<Wide>(row) * wideStep;
      const Wide scale = std::max({wideStep, std::abs(elapsed), expected});
      const Wide tolerance =
          8.0L * static_cast<Wide>(std::numeric_limits<Scalar>::epsilon()) *
          scale;
      if (!std::isfinite(elapsed) || !std::isfinite(expected) ||
          std::abs(elapsed - expected) > tolerance) {
        return ErrorCode::InvalidDomain;
      }
    }
  }
  return ErrorCode::None;
}

struct AnchorRange final {
  Index first{};
  Index last{};
  Index count{};
};

Result<AnchorRange> commonAnchors(std::span<const std::int64_t> lags,
                                  Index rows) {
  if (rows == 0U || lags.empty()) {
    return fail<AnchorRange>(ErrorCode::InsufficientSamples,
                             "lagged moment has no rows or requested lags");
  }
  Index first = 0U;
  Index last = rows - 1U;
  for (const std::int64_t lag : lags) {
    if (lag >= 0) {
      const auto shift = static_cast<std::uint64_t>(lag);
      if (shift >= rows) {
        return fail<AnchorRange>(ErrorCode::InsufficientSamples,
                                 "a requested lag has no common anchor");
      }
      last = std::min(last, rows - 1U - static_cast<Index>(shift));
      continue;
    }
    const auto shift = std::uint64_t{0} - static_cast<std::uint64_t>(lag);
    if (shift >= rows) {
      return fail<AnchorRange>(ErrorCode::InsufficientSamples,
                               "a requested lag has no common anchor");
    }
    first = std::max(first, static_cast<Index>(shift));
  }
  if (first > last) {
    return fail<AnchorRange>(ErrorCode::InsufficientSamples,
                             "requested lags have no common anchor");
  }
  return Result<AnchorRange>::success(
      {.first = first, .last = last, .count = last - first + 1U});
}

Index shiftedRow(Index anchor, std::int64_t lag) noexcept {
  if (lag >= 0) {
    return anchor + static_cast<Index>(static_cast<std::uint64_t>(lag));
  }
  const auto magnitude = std::uint64_t{0} - static_cast<std::uint64_t>(lag);
  return anchor - static_cast<Index>(magnitude);
}

struct TupleKey final {
  std::array<Index, 4U> coordinates{};
  Index order{};

  friend bool operator==(const TupleKey&, const TupleKey&) noexcept = default;
};

struct TupleHash final {
  [[nodiscard]] std::size_t operator()(const TupleKey& key) const noexcept {
    std::size_t value = static_cast<std::size_t>(key.order) + 0x9e3779b9U;
    for (Index i = 0U; i < key.order; ++i) {
      value ^= static_cast<std::size_t>(key.coordinates.at(i)) + 0x9e3779b9U +
               (value << 6U) + (value >> 2U);
    }
    return value;
  }
};

using MomentMap = std::unordered_map<TupleKey, Scalar, TupleHash>;
using MomentMaps = std::array<MomentMap, 5U>;

template <class Visitor>
bool forEachTuple(const statistics::MomentTerm& term, Index dimension,
                  Visitor visitor) {
  const Index order = term.order;
  const Index count = term.values.size();
  if (order == 0U || order > 4U) {
    return false;
  }
  if (term.layout == statistics::MomentRepresentation::selected) {
    Index expected = 0U;
    if (!checkedMultiply(count, order, expected) ||
        term.flattened_tuples.view().size() != expected) {
      return false;
    }
    for (Index i = 0U; i < count; ++i) {
      TupleKey key{};
      key.order = order;
      for (Index axis = 0U; axis < order; ++axis) {
        key.coordinates.at(axis) =
            term.flattened_tuples.view()[(i * order) + axis];
      }
      if (!visitor(i, key)) {
        return false;
      }
    }
    return true;
  }

  if (dimension == 0U) {
    return false;
  }
  if (term.layout == statistics::MomentRepresentation::component) {
    if (count != dimension) {
      return false;
    }
    for (Index i = 0U; i < count; ++i) {
      TupleKey key{};
      key.order = order;
      std::ranges::fill(std::span<Index>{key.coordinates}.first(order), i);
      if (!visitor(i, key)) {
        return false;
      }
    }
    return true;
  }

  if (term.layout == statistics::MomentRepresentation::dense) {
    for (Index i = 0U; i < count; ++i) {
      TupleKey key{};
      key.order = order;
      Index value = i;
      for (Index axis = order; axis > 0U; --axis) {
        key.coordinates.at(axis - 1U) = value % dimension;
        value /= dimension;
      }
      if (value != 0U || !visitor(i, key)) {
        return false;
      }
    }
    return true;
  }

  if (term.layout != statistics::MomentRepresentation::packed) {
    return false;
  }
  std::array<Index, 4U> tuple{};
  for (Index i = 0U; i < count; ++i) {
    const TupleKey key{.coordinates = tuple, .order = order};
    if (!visitor(i, key)) {
      return false;
    }
    if (i + 1U == count) {
      continue;
    }
    Index axis = order;
    while (axis > 0U && tuple.at(axis - 1U) + 1U >= dimension) {
      --axis;
    }
    if (axis == 0U) {
      return false;
    }
    --axis;
    ++tuple.at(axis);
    std::ranges::fill(
        std::span<Index>{tuple}.subspan(axis + 1U, order - axis - 1U),
        tuple.at(axis));
  }
  return true;
}

bool collectMomentMaps(const MomentEstimate& estimate, MomentMaps& maps) {
  const Index dimension = estimate.population_mean.has_value()
                              ? estimate.population_mean->size()
                              : 0U;
  for (const statistics::MomentTerm& term : estimate.terms.view()) {
    auto& map = maps.at(term.order);
    if (!forEachTuple(term, dimension, [&](Index i, const TupleKey& key) {
          map.try_emplace(key, term.values.at(i));
          return true;
        })) {
      return false;
    }
  }
  return true;
}

struct Scaled final {
  Scalar fraction{};
  int exponent{};
};

Scaled scaled(Scalar value) noexcept {
  if (value == 0.0) {
    return {};
  }
  int exponent = 0;
  return {.fraction = std::frexp(value, &exponent), .exponent = exponent};
}

Scaled multiply(Scaled left, Scaled right) noexcept {
  if (left.fraction == 0.0 || right.fraction == 0.0) {
    return {};
  }
  int adjustment = 0;
  const Scalar fraction =
      std::frexp(left.fraction * right.fraction, &adjustment);
  return {.fraction = fraction,
          .exponent = left.exponent + right.exponent + adjustment};
}

class ScaledSum final {
public:
  void add(Scaled value) noexcept {
    if (value.fraction == 0.0) {
      return;
    }
    if (!initialized_) {
      sum_ = value.fraction;
      exponent_ = value.exponent;
      initialized_ = true;
      return;
    }
    if (value.exponent > exponent_) {
      const int shift = exponent_ - value.exponent;
      sum_ = std::scalbn(sum_, shift);
      correction_ = std::scalbn(correction_, shift);
      exponent_ = value.exponent;
    }
    const Scalar addend =
        std::scalbn(value.fraction, value.exponent - exponent_);
    const Scalar total = sum_ + addend;
    if (std::abs(sum_) >= std::abs(addend)) {
      correction_ += (sum_ - total) + addend;
    } else {
      correction_ += (addend - total) + sum_;
    }
    sum_ = total;
  }

  [[nodiscard]] bool toScalar(Scalar& value) const noexcept {
    const Scalar total = sum_ + correction_;
    value = std::scalbn(total, exponent_);
    return std::isfinite(value);
  }

  [[nodiscard]] bool toMean(Index count, Scalar& value) const noexcept {
    const Scalar total = sum_ + correction_;
    const auto divisor = static_cast<Scalar>(count);
    value = std::scalbn(total / divisor, exponent_);
    return std::isfinite(value);
  }

private:
  Scalar sum_{};
  Scalar correction_{};
  int exponent_{};
  bool initialized_{};
};

Result<Scalar> rawMoment(const TupleKey& tuple, const MomentMaps& central,
                         const Vector& mean) {
  ScaledSum total;
  const Index combinations = Index{1U} << tuple.order;
  for (Index subset = 0U; subset < combinations; ++subset) {
    TupleKey centralKey{};
    Scaled product{.fraction = 1.0, .exponent = 0};
    for (Index axis = 0U; axis < tuple.order; ++axis) {
      if ((subset & (Index{1U} << axis)) != 0U) {
        centralKey.coordinates.at(centralKey.order) =
            tuple.coordinates.at(axis);
        ++centralKey.order;
      } else {
        product =
            multiply(product, scaled(mean.at(tuple.coordinates.at(axis))));
      }
    }
    if (centralKey.order > 0U) {
      const auto found = central.at(centralKey.order).find(centralKey);
      if (found == central.at(centralKey.order).end()) {
        return fail<Scalar>(ErrorCode::InsufficientSamples,
                            "raw moment requires an unavailable central term");
      }
      product = multiply(product, scaled(found->second));
    }
    total.add(product);
  }
  Scalar value = 0.0;
  if (!total.toScalar(value)) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "known-center moment is not representable");
  }
  return Result<Scalar>::success(value);
}

Result<MomentEstimate> publishEstimate(MomentEstimate estimate,
                                       const MomentRequest& request,
                                       CenterRule center, Index candidateCount,
                                       Limits limits) {
  estimate.diagnostics.center = center;
  estimate.diagnostics.global.input = candidateCount;
  estimate.diagnostics.global.positive = estimate.diagnostics.global.valid;
  estimate.diagnostics.algorithm = "finite_aligned_lagged_joint_moments";
  estimate.diagnostics.premises =
      center == CenterRule::known
          ? "finite empirical common-pair support on an identical uniform "
            "grid; no stationarity assumption inferred; known center is zero"
          : "finite empirical common-pair support on an identical uniform "
            "grid; no stationarity assumption inferred";

  if (center == CenterRule::estimated) {
    return Result<MomentEstimate>::success(std::move(estimate));
  }
  if (!estimate.population_mean.has_value()) {
    return fail<MomentEstimate>(ErrorCode::InsufficientSamples,
                                "lagged population mean is unavailable");
  }

  MomentMaps maps{};
  if (!collectMomentMaps(estimate, maps)) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "central moment output layout is invalid");
  }
  const Index dimension = estimate.population_mean->size();
  std::vector<statistics::MomentTerm> outputTerms;
  outputTerms.reserve(estimate.terms.view().size());
  for (const statistics::MomentTerm& source : estimate.terms.view()) {
    if ((request.orders & (1U << (source.order - 1U))) == 0U) {
      continue;
    }
    Vector values(source.values.size());
    const bool valid =
        forEachTuple(source, dimension, [&](Index i, const TupleKey& tuple) {
          auto raw = rawMoment(tuple, maps, estimate.population_mean.value());
          if (!raw) {
            return false;
          }
          values.at(i) = raw.value();
          return true;
        });
    if (!valid) {
      return fail<MomentEstimate>(
          ErrorCode::NonFiniteResult,
          "known-center moment conversion failed or is not representable");
    }
    outputTerms.push_back({.order = source.order,
                           .layout = source.layout,
                           .flattened_tuples = source.flattened_tuples,
                           .values = std::move(values)});
  }
  auto terms = Array<statistics::MomentTerm>::copy(outputTerms, limits);
  if (!terms) {
    return fail<MomentEstimate>(terms.status().code(),
                                "moment output exceeds limits");
  }
  estimate.terms = std::move(terms.value());
  return Result<MomentEstimate>::success(std::move(estimate));
}

} // namespace

Result<MomentEstimate> laggedHigherMoments(const SeriesView& x,
                                           const SeriesView& y,
                                           const LagMomentSpec& spec,
                                           Limits limits) {
  try {
    if (spec.center != CenterRule::estimated &&
        spec.center != CenterRule::known) {
      return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                  "unsupported lagged moment center rule");
    }
    const ErrorCode xStatus = validateGrid(x);
    if (xStatus != ErrorCode::None) {
      return fail<MomentEstimate>(xStatus, "x series has an invalid grid");
    }
    const ErrorCode yStatus = validateGrid(y);
    if (yStatus != ErrorCode::None) {
      return fail<MomentEstimate>(yStatus, "y series has an invalid grid");
    }
    if (x.times().size() != y.times().size() ||
        x.uniformStep() != y.uniformStep()) {
      return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                  "lagged series require equal aligned grids");
    }
    for (Index row = 0U; row < x.times().size(); ++row) {
      if (x.times().at(row) != y.times().at(row)) {
        return fail<MomentEstimate>(
            ErrorCode::InvalidDomain,
            "lagged series time axes must be identical");
      }
    }
    if (spec.lags.size() != spec.components.size()) {
      return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                  "each lag requires one component list");
    }
    if (spec.lags.empty()) {
      return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                  "at least one signed lag is required");
    }

    Index combinedDimension = 0U;
    if (!checkedAdd(x.rows().columns(), y.rows().columns(),
                    combinedDimension)) {
      return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                  "combined series dimension overflows");
    }
    Index stackedDimension = 0U;
    for (const IndexList components : spec.components) {
      if (components.empty()) {
        return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                    "each lag needs a selected component");
      }
      if (!checkedAdd(stackedDimension, components.size(), stackedDimension)) {
        return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                    "stacked observation dimension overflows");
      }
      for (const Index component : components) {
        if (component >= combinedDimension) {
          return fail<MomentEstimate>(ErrorCode::IndexOutOfRange,
                                      "selected component is outside x/y");
        }
      }
    }

    const auto rangeResult = commonAnchors(spec.lags, x.times().size());
    if (!rangeResult) {
      return fail<MomentEstimate>(rangeResult.status().code(),
                                  std::string(rangeResult.status().message()));
    }
    const AnchorRange range = rangeResult.value();
    Index scratchBytes = 0U;
    if (!checkedMultiply(stackedDimension, sizeof(Scalar), scratchBytes) ||
        stackedDimension > limits.elements || scratchBytes > limits.bytes) {
      return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                  "lagged observation scratch exceeds limits");
    }
    Limits stateLimits = limits;
    stateLimits.elements -= stackedDimension;
    stateLimits.bytes -= scratchBytes;

    std::array<Index, 1U> shapeStorage{stackedDimension};
    const std::span<const Index> shape =
        stackedDimension == 1U ? std::span<const Index>{}
                               : std::span<const Index>{shapeStorage};
    if (spec.center == CenterRule::known &&
        spec.request.representation ==
            statistics::MomentRepresentation::selected) {
      if (spec.request.orders == 0U || (spec.request.orders & 0xf0U) != 0U ||
          std::popcount(spec.request.orders) != 1U ||
          spec.request.selected.empty() ||
          spec.request.contraction.has_value()) {
        return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                    "selected moment request is malformed");
      }
      const Index order =
          static_cast<Index>(std::countr_zero(spec.request.orders)) + 1U;
      Index selectedCoordinates = 0U;
      for (const IndexList tuple : spec.request.selected) {
        if (tuple.size() != order) {
          return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                      "selected tuple order is inconsistent");
        }
        for (const Index coordinate : tuple) {
          if (coordinate >= stackedDimension) {
            return fail<MomentEstimate>(
                ErrorCode::IndexOutOfRange,
                "selected tuple is outside stacked data");
          }
        }
        if (!checkedAdd(selectedCoordinates, tuple.size(),
                        selectedCoordinates)) {
          return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                      "selected tuple storage overflows");
        }
      }
      const Index selectedCount = spec.request.selected.size();
      Index workingElements = 0U;
      Index workingBytes = 0U;
      Index accumulatorBytes = 0U;
      Index outputElements = 0U;
      Index outputBytes = 0U;
      Index outputValueCopies = 0U;
      Index outputIndexCopies = 0U;
      Index outputValueBytes = 0U;
      Index outputIndexBytes = 0U;
      if (!checkedAdd(stackedDimension, selectedCount, workingElements) ||
          !checkedMultiply(workingElements - stackedDimension,
                           sizeof(ScaledSum), accumulatorBytes) ||
          !checkedAdd(scratchBytes, accumulatorBytes, workingBytes) ||
          !checkedMultiply(selectedCount, 2U, outputValueCopies) ||
          !checkedMultiply(selectedCoordinates, 3U, outputIndexCopies) ||
          !checkedAdd(outputValueCopies, outputIndexCopies, outputElements) ||
          !checkedAdd(outputElements, 2U, outputElements) ||
          !checkedMultiply(outputValueCopies, sizeof(Scalar),
                           outputValueBytes) ||
          !checkedMultiply(outputIndexCopies, sizeof(Index),
                           outputIndexBytes) ||
          !checkedAdd(outputValueBytes, outputIndexBytes, outputBytes) ||
          !checkedAdd(outputBytes, 2U * sizeof(statistics::MomentTerm),
                      outputBytes)) {
        return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                    "selected moment resource size overflows");
      }
      if (workingElements > limits.elements / 4U ||
          workingBytes > limits.bytes / 4U ||
          outputElements > limits.elements / 4U ||
          outputBytes > limits.bytes / 4U) {
        return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                    "selected lagged moment exceeds limits");
      }

      const Limits meanStateLimits{.elements = limits.elements / 4U,
                                   .bytes = limits.bytes / 4U,
                                   .evaluations = limits.evaluations};
      const Limits resultLimits{.elements = limits.elements / 4U,
                                .bytes = limits.bytes / 4U,
                                .evaluations = limits.evaluations};
      const Limits termLimits{.elements = limits.elements / 4U,
                              .bytes = limits.bytes / 4U,
                              .evaluations = limits.evaluations};
      const MomentRequest meanRequest{
          .orders = 0b0001U,
          .representation = statistics::MomentRepresentation::component,
          .selected = {},
          .contraction = std::nullopt};
      auto meanStateResult =
          statistics::makeOnlineMoments(shape, meanRequest, meanStateLimits);
      if (!meanStateResult) {
        return fail<MomentEstimate>(
            meanStateResult.status().code(),
            std::string(meanStateResult.status().message()));
      }
      auto meanState = std::move(meanStateResult.value());
      Vector observation(stackedDimension);
      std::vector<ScaledSum> sums(selectedCount);
      std::vector<Index> flattened;
      flattened.reserve(selectedCoordinates);
      for (const IndexList tuple : spec.request.selected) {
        flattened.insert(flattened.end(), tuple.begin(), tuple.end());
      }
      Index support = 0U;
      for (Index anchor = range.first;; ++anchor) {
        bool admitted = true;
        for (Index lagIndex = 0U; lagIndex < spec.lags.size() && admitted;
             ++lagIndex) {
          const Index xRow = shiftedRow(anchor, spec.lags[lagIndex]);
          for (const Index component : spec.components[lagIndex]) {
            const bool fromX = component < x.rows().columns();
            const Index row = fromX ? xRow : anchor;
            const auto mask = fromX ? x.valid() : y.valid();
            if (!mask.empty() && mask[row] == 0U) {
              admitted = false;
              break;
            }
          }
        }
        if (admitted) {
          Index cursor = 0U;
          for (Index lagIndex = 0U; lagIndex < spec.lags.size(); ++lagIndex) {
            const Index xRow = shiftedRow(anchor, spec.lags[lagIndex]);
            for (const Index component : spec.components[lagIndex]) {
              const bool fromX = component < x.rows().columns();
              const Scalar value =
                  fromX ? x.rows().at(xRow, component)
                        : y.rows().at(anchor, component - x.rows().columns());
              if (!std::isfinite(value)) {
                return fail<MomentEstimate>(
                    ErrorCode::NonFiniteInput,
                    "a selected lagged observation is not finite");
              }
              observation.at(cursor++) = value;
            }
          }
          const Status added =
              statistics::add(meanState, observation.view(), 1.0);
          if (!added.ok()) {
            return fail<MomentEstimate>(added.code(),
                                        std::string(added.message()));
          }
          Index tupleIndex = 0U;
          for (const IndexList tuple : spec.request.selected) {
            Scaled product{.fraction = 1.0, .exponent = 0};
            for (const Index coordinate : tuple) {
              product = multiply(product, scaled(observation.at(coordinate)));
            }
            sums.at(tupleIndex++).add(product);
          }
          ++support;
        }
        if (anchor == range.last) {
          break;
        }
      }
      if (support == 0U) {
        return fail<MomentEstimate>(ErrorCode::InsufficientSamples,
                                    "no complete common lagged rows remain");
      }
      auto meanEstimate =
          statistics::finalize(meanState, WeightRule::unweighted, resultLimits);
      if (!meanEstimate) {
        return fail<MomentEstimate>(
            meanEstimate.status().code(),
            std::string(meanEstimate.status().message()));
      }
      Vector values(selectedCount);
      for (Index i = 0U; i < selectedCount; ++i) {
        Scalar value = 0.0;
        if (!sums.at(i).toMean(support, value)) {
          return fail<MomentEstimate>(
              ErrorCode::NonFiniteResult,
              "known-center moment is not representable");
        }
        values.at(i) = value;
      }
      auto ownedIndices = Array<Index>::copy(flattened, termLimits);
      if (!ownedIndices) {
        return fail<MomentEstimate>(ownedIndices.status().code(),
                                    "selected tuple output exceeds limits");
      }
      statistics::MomentTerm term{
          .order = order,
          .layout = statistics::MomentRepresentation::selected,
          .flattened_tuples = std::move(ownedIndices.value()),
          .values = std::move(values)};
      const std::array<statistics::MomentTerm, 1U> outputTerms{std::move(term)};
      auto ownedTerms =
          Array<statistics::MomentTerm>::copy(outputTerms, termLimits);
      if (!ownedTerms) {
        return fail<MomentEstimate>(ownedTerms.status().code(),
                                    "selected moment output exceeds limits");
      }
      MomentEstimate result = std::move(meanEstimate.value());
      result.terms = std::move(ownedTerms.value());
      result.diagnostics.center = CenterRule::known;
      result.diagnostics.global.input = range.count;
      result.diagnostics.global.valid = support;
      result.diagnostics.global.positive = support;
      result.diagnostics.algorithm = "finite_aligned_lagged_joint_moments";
      result.diagnostics.premises =
          "finite empirical common-pair support on an identical uniform grid; "
          "no stationarity assumption inferred; known center is zero";
      return Result<MomentEstimate>::success(std::move(result));
    }

    if (spec.request.representation !=
            statistics::MomentRepresentation::selected ||
        spec.center == CenterRule::estimated) {
      MomentRequest stateRequest = spec.request;
      if (spec.center == CenterRule::known) {
        if (stateRequest.orders == 0U || (stateRequest.orders & 0xf0U) != 0U) {
          return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                      "moment order mask is invalid");
        }
        const auto highest =
            static_cast<Index>(std::bit_width(stateRequest.orders));
        stateRequest.orders = static_cast<std::uint8_t>((1U << highest) - 1U);
      }
      auto stateResult =
          statistics::makeOnlineMoments(shape, stateRequest, stateLimits);
      if (!stateResult) {
        return fail<MomentEstimate>(
            stateResult.status().code(),
            std::string(stateResult.status().message()));
      }
      statistics::OnlineMoments state = std::move(stateResult.value());
      Vector observation(stackedDimension);
      Index support = 0U;
      for (Index anchor = range.first;; ++anchor) {
        bool admitted = true;
        for (Index lagIndex = 0U; lagIndex < spec.lags.size() && admitted;
             ++lagIndex) {
          const Index xRow = shiftedRow(anchor, spec.lags[lagIndex]);
          for (const Index component : spec.components[lagIndex]) {
            const bool fromX = component < x.rows().columns();
            const Index row = fromX ? xRow : anchor;
            const auto mask = fromX ? x.valid() : y.valid();
            if (!mask.empty() && mask[row] == 0U) {
              admitted = false;
              break;
            }
          }
        }
        if (admitted) {
          Index cursor = 0U;
          for (Index lagIndex = 0U; lagIndex < spec.lags.size(); ++lagIndex) {
            const Index xRow = shiftedRow(anchor, spec.lags[lagIndex]);
            for (const Index component : spec.components[lagIndex]) {
              const bool fromX = component < x.rows().columns();
              const Scalar value =
                  fromX ? x.rows().at(xRow, component)
                        : y.rows().at(anchor, component - x.rows().columns());
              if (!std::isfinite(value)) {
                return fail<MomentEstimate>(
                    ErrorCode::NonFiniteInput,
                    "a selected lagged observation is not finite");
              }
              observation.at(cursor++) = value;
            }
          }
          const Status added = statistics::add(state, observation.view(), 1.0);
          if (!added.ok()) {
            return fail<MomentEstimate>(added.code(),
                                        std::string(added.message()));
          }
          ++support;
        }
        if (anchor == range.last) {
          break;
        }
      }
      if (support == 0U) {
        return fail<MomentEstimate>(ErrorCode::InsufficientSamples,
                                    "no complete common lagged rows remain");
      }
      auto estimate =
          statistics::finalize(state, WeightRule::unweighted, stateLimits);
      if (!estimate) {
        return fail<MomentEstimate>(estimate.status().code(),
                                    std::string(estimate.status().message()));
      }
      auto published =
          publishEstimate(std::move(estimate.value()), spec.request,
                          spec.center, range.count, limits);
      if (!published) {
        return published;
      }
      published.value().diagnostics.global.valid = support;
      published.value().diagnostics.global.positive = support;
      return published;
    }

    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "unsupported lagged moment request");
  } catch (const std::bad_alloc&) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "lagged moment workspace allocation failed");
  } catch (const std::length_error&) {
    return fail<MomentEstimate>(
        ErrorCode::ResourceLimit,
        "lagged moment workspace size is not representable");
  }
}

} // namespace vss::math::signal
