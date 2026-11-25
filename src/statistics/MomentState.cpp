#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
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
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/MomentState.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {
namespace {

using Key = std::array<Index, 5U>;

struct ScaledValue final {
  Scalar fraction{};
  int exponent{};
};

ScaledValue scaled(Scalar value) {
  if (value == 0.0) {
    return {};
  }
  int exponent = 0;
  const Scalar fraction = std::frexp(value, &exponent);
  return {.fraction = fraction, .exponent = exponent};
}

ScaledValue scaledMultiply(ScaledValue left, ScaledValue right) {
  if (left.fraction == 0.0 || right.fraction == 0.0) {
    return {};
  }
  int adjustment = 0;
  const Scalar fraction =
      std::frexp(left.fraction * right.fraction, &adjustment);
  return {.fraction = fraction,
          .exponent = left.exponent + right.exponent + adjustment};
}

class ScaledAccumulator final {
public:
  void add(ScaledValue value) noexcept {
    if (value.fraction == 0.0) {
      return;
    }
    if (!initialized_) {
      exponent_ = value.exponent;
      initialized_ = true;
    } else if (value.exponent > exponent_) {
      const Scalar factor = std::scalbn(1.0, exponent_ - value.exponent);
      sum_ *= factor;
      correction_ *= factor;
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

  [[nodiscard]] ScaledValue value() const noexcept {
    const Scalar total = sum_ + correction_;
    if (!initialized_ || total == 0.0) {
      return {};
    }
    int adjustment = 0;
    const Scalar fraction = std::frexp(total, &adjustment);
    return {.fraction = fraction, .exponent = exponent_ + adjustment};
  }

private:
  Scalar sum_{};
  Scalar correction_{};
  int exponent_{};
  bool initialized_{};
};

bool narrow(ScaledValue value, Scalar& result) {
  result = std::scalbn(value.fraction, value.exponent);
  return std::isfinite(result);
}

bool checkedAdd(Index left, Index right, Index& result) {
  if (left > std::numeric_limits<Index>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool checkedMultiply(Index left, Index right, Index& result) {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

ErrorCode preflightRequest(Index dimension, const MomentRequest& request,
                           Limits limits) {
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U) {
    return ErrorCode::InvalidDomain;
  }
  const MomentRepresentation layout =
      request.representation == MomentRepresentation::contraction
          ? MomentRepresentation::dense
          : request.representation;
  Index outputCount = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    Index count = 0U;
    if (layout == MomentRepresentation::selected) {
      for (const auto tuple : request.selected) {
        if (tuple.size() != order) {
          return ErrorCode::InvalidDomain;
        }
        if (!checkedAdd(count, 1U, count)) {
          return ErrorCode::SizeOverflow;
        }
      }
    } else if (layout == MomentRepresentation::component) {
      count = dimension;
    } else if (layout == MomentRepresentation::dense) {
      count = 1U;
      for (Index axis = 0U; axis < order; ++axis) {
        if (!checkedMultiply(count, dimension, count)) {
          return ErrorCode::SizeOverflow;
        }
      }
    } else if (layout == MomentRepresentation::packed) {
      // Compute C(d + order - 1, order), cancelling the small denominator
      // before multiplication so the checked result is exact.
      if (dimension > std::numeric_limits<Index>::max() - (order - 1U)) {
        return ErrorCode::SizeOverflow;
      }
      const Index n = dimension + order - 1U;
      std::array<Index, 4U> factors{};
      for (Index i = 0U; i < order; ++i) {
        factors.at(i) = n - order + 1U + i;
      }
      Index denominator = 1U;
      for (Index i = 2U; i <= order; ++i) {
        denominator *= i;
      }
      for (Index i = 0U; i < order && denominator > 1U; ++i) {
        const Index divisor = std::gcd(factors.at(i), denominator);
        factors.at(i) /= divisor;
        denominator /= divisor;
      }
      if (denominator != 1U) {
        return ErrorCode::SizeOverflow;
      }
      count = 1U;
      for (Index i = 0U; i < order; ++i) {
        if (!checkedMultiply(count, factors.at(i), count)) {
          return ErrorCode::SizeOverflow;
        }
      }
    } else {
      return ErrorCode::InvalidDomain;
    }
    if (!checkedAdd(outputCount, count, outputCount)) {
      return ErrorCode::SizeOverflow;
    }
  }
  if (outputCount == 0U || outputCount > limits.elements) {
    return outputCount == 0U ? ErrorCode::InvalidDomain
                             : ErrorCode::ResourceLimit;
  }
  Index candidateCount = 0U;
  Index candidateBytes = 0U;
  if (!checkedMultiply(outputCount, 16U, candidateCount) ||
      !checkedMultiply(candidateCount, sizeof(Key), candidateBytes)) {
    return ErrorCode::SizeOverflow;
  }
  return candidateBytes <= limits.bytes ? ErrorCode::None
                                        : ErrorCode::ResourceLimit;
}

Status fail(ErrorCode code, const char* message) {
  return Status::failure(code, message);
}

Index dimensions(std::span<const Index> shape, bool& valid) {
  valid = true;
  if (shape.empty()) {
    return 1U;
  }
  Index result = 1U;
  for (const Index extent : shape) {
    if (extent == 0U) {
      valid = false;
      return 0U;
    }
    if (result > std::numeric_limits<Index>::max() / extent) {
      valid = false;
      return 0U;
    }
    result *= extent;
  }
  return result;
}

std::vector<Key> requestedKeys(Index d, const MomentRequest& request,
                               ErrorCode& error) {
  std::vector<Key> result;
  error = ErrorCode::None;
  if (d == 0U || request.orders == 0U || (request.orders & 0xf0U) != 0U) {
    error = ErrorCode::InvalidDomain;
    return result;
  }
  const MomentRepresentation layout =
      request.representation == MomentRepresentation::contraction
          ? MomentRepresentation::dense
          : request.representation;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    if (layout == MomentRepresentation::selected) {
      for (const auto tuple : request.selected) {
        if (tuple.size() != order) {
          error = ErrorCode::InvalidDomain;
          return {};
        }
        Key key{};
        key.at(0) = order;
        for (Index axis = 0U; axis < order; ++axis) {
          if (tuple[axis] >= d) {
            error = ErrorCode::IndexOutOfRange;
            return {};
          }
          key.at(axis + 1U) = tuple[axis];
        }
        result.push_back(key);
      }
      continue;
    }
    if (layout == MomentRepresentation::component) {
      for (Index i = 0U; i < d; ++i) {
        Key key{};
        key.at(0) = order;
        for (Index axis = 0U; axis < order; ++axis) {
          key.at(axis + 1U) = i;
        }
        result.push_back(key);
      }
      continue;
    }
    if (layout == MomentRepresentation::dense) {
      Index count = 1U;
      for (Index i = 0U; i < order; ++i) {
        if (count > std::numeric_limits<Index>::max() / d) {
          error = ErrorCode::SizeOverflow;
          return {};
        }
        count *= d;
      }
      for (Index n = 0U; n < count; ++n) {
        Key key{};
        key.at(0) = order;
        Index value = n;
        for (Index axis = order; axis > 0U; --axis) {
          key.at(axis) = value % d;
          value /= d;
        }
        result.push_back(key);
      }
      continue;
    }
    if (layout == MomentRepresentation::packed) {
      Key key{};
      key.at(0) = order;
      std::array<Index, 4U> coordinates{};
      bool hasNext = true;
      while (hasNext) {
        for (Index axis = 0U; axis < order; ++axis) {
          key.at(axis + 1U) = coordinates.at(axis);
        }
        result.push_back(key);
        Index axis = order;
        while (axis > 0U && coordinates.at(axis - 1U) == d - 1U) {
          --axis;
        }
        if (axis == 0U) {
          hasNext = false;
        } else {
          const Index next = coordinates.at(axis - 1U) + 1U;
          coordinates.at(axis - 1U) = next;
          for (Index following = axis; following < order; ++following) {
            coordinates.at(following) = next;
          }
        }
      }
      continue;
    }
    error = ErrorCode::InvalidDomain;
    return {};
  }
  if (result.empty()) {
    error = ErrorCode::InvalidDomain;
  }
  return result;
}

std::vector<Key> closure(const std::vector<Key>& requested) {
  std::vector<Key> result;
  for (const Key& key : requested) {
    const Index order = key.at(0);
    const Index masks = 1U << order;
    for (Index mask = 0U; mask < masks; ++mask) {
      Key child{};
      Index next = 0U;
      for (Index axis = 0U; axis < order; ++axis) {
        if ((mask & (1U << axis)) == 0U) {
          child.at(++next) = key.at(axis + 1U);
        }
      }
      child.at(0) = next;
      if (next > 1U) {
        result.push_back(child);
      }
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

Index findKey(const OnlineMoments& state, const Key& key) {
  const auto orders = state.tuple_orders.view();
  const auto indices = state.tuple_indices.view();
  Index offset = 0U;
  for (Index i = 0U; i < orders.size(); ++i) {
    if (orders[i] == key.at(0)) {
      bool same = true;
      for (Index j = 0U; j < key.at(0); ++j) {
        same = same && indices[offset + j] == key.at(j + 1U);
      }
      if (same) {
        return i;
      }
    }
    offset += orders[i];
  }
  return std::numeric_limits<Index>::max();
}

Key keyAt(const OnlineMoments& state, Index index) {
  Key key{};
  const auto orders = state.tuple_orders.view();
  const auto indices = state.tuple_indices.view();
  Index offset = 0U;
  for (Index i = 0U; i < index; ++i) {
    offset += orders[i];
  }
  key.at(0) = orders[index];
  for (Index i = 0U; i < key.at(0); ++i) {
    key.at(i + 1U) = indices[offset + i];
  }
  return key;
}

bool validOnlineState(const OnlineMoments& state) {
  bool inputValid = false;
  bool outputValid = false;
  const Index inputD = dimensions(state.shape.view(), inputValid);
  const Index outputD = dimensions(state.output_shape.view(), outputValid);
  if (!inputValid || !outputValid || inputD != state.input_dimension ||
      outputD != state.mean.size() || state.anchor.size() != outputD ||
      state.mean_offset.size() != outputD ||
      state.scratch_delta.size() != outputD ||
      state.scratch_mean.size() != outputD ||
      state.scratch_observation.size() != outputD ||
      state.centered_sums.size() != state.scratch_sums.size() ||
      state.tuple_orders.view().size() != state.centered_sums.size() ||
      state.count < state.positive_count || !std::isfinite(state.mass) ||
      state.mass < 0.0 || state.stored_request.orders == 0U ||
      (state.stored_request.orders & 0xf0U) != 0U ||
      state.stored_request.contraction.has_value() !=
          (state.stored_request.representation ==
           MomentRepresentation::contraction) ||
      (state.mass == 0.0) != (state.positive_count == 0U)) {
    return false;
  }
  const auto allFinite = [](const Vector& values) {
    for (Index i = 0U; i < values.size(); ++i) {
      if (!std::isfinite(values.at(i))) {
        return false;
      }
    }
    return true;
  };
  if (!allFinite(state.mean) || !allFinite(state.anchor) ||
      !allFinite(state.mean_offset) || !allFinite(state.centered_sums)) {
    return false;
  }
  for (Index i = 0U; i < state.mean.size(); ++i) {
    if (!std::isfinite(state.anchor.at(i) + state.mean_offset.at(i))) {
      return false;
    }
  }
  for (const Scalar power : state.raw_weight_powers) {
    if (!std::isfinite(power) || power < 0.0 ||
        power > 1.0 + 8.0 * std::numeric_limits<Scalar>::epsilon()) {
      return false;
    }
  }
  const auto orders = state.tuple_orders.view();
  const auto indices = state.tuple_indices.view();
  Index expectedIndices = 0U;
  for (const Index order : orders) {
    if (order < 2U || order > 4U ||
        expectedIndices > std::numeric_limits<Index>::max() - order) {
      return false;
    }
    expectedIndices += order;
  }
  if (expectedIndices != indices.size()) {
    return false;
  }
  Index offset = 0U;
  Key previous{};
  for (Index i = 0U; i < orders.size(); ++i) {
    Key key{};
    key.at(0) = orders[i];
    for (Index axis = 0U; axis < orders[i]; ++axis) {
      if (indices[offset + axis] >= outputD) {
        return false;
      }
      key.at(axis + 1U) = indices[offset + axis];
    }
    if (i > 0U && !(previous < key)) {
      return false;
    }
    previous = key;
    offset += orders[i];
  }
  if (state.stored_request.contraction.has_value()) {
    const Matrix& map = *state.stored_request.contraction;
    if (map.rows() != outputD || map.columns() != inputD) {
      return false;
    }
    for (Index row = 0U; row < map.rows(); ++row) {
      for (Index column = 0U; column < map.columns(); ++column) {
        if (!std::isfinite(map.at(row, column))) {
          return false;
        }
      }
    }
  } else if (inputD != outputD) {
    return false;
  }
  if (state.stored_request.representation == MomentRepresentation::selected) {
    for (const auto& tuple : state.stored_request.selected.view()) {
      const Index order = tuple.view().size();
      if (order == 0U || order > 4U ||
          (state.stored_request.orders & (1U << (order - 1U))) == 0U) {
        return false;
      }
      for (const Index coordinate : tuple.view()) {
        if (coordinate >= outputD) {
          return false;
        }
      }
    }
  }
  return true;
}

Status validateObservation(const OnlineMoments& state,
                           ConstVectorView observation, Scalar weight) {
  if (!validOnlineState(state)) {
    return fail(ErrorCode::InvalidDomain, "online state structure is invalid");
  }
  bool valid = false;
  const Index shapeDimension = dimensions(state.shape.view(), valid);
  const Index d = state.input_dimension;
  if (!valid || shapeDimension != d || observation.size() != d) {
    return fail(ErrorCode::DimensionMismatch, "observation shape mismatch");
  }
  if (!std::isfinite(weight) || weight < 0.0) {
    return fail(ErrorCode::InvalidDomain,
                "moment weight must be finite and nonnegative");
  }
  for (Index i = 0U; i < d; ++i) {
    if (!std::isfinite(observation.at(i))) {
      return fail(ErrorCode::NonFiniteInput,
                  "moment observation is not finite");
    }
  }
  return Status::success();
}

Result<OnlineMoments> makeOnlineFromOwned(const Array<Index>& shape,
                                          const OwnedMomentRequest& request,
                                          Limits limits) {
  bool valid = false;
  const Index inputDimension = dimensions(shape.view(), valid);
  if (!valid) {
    return Result<OnlineMoments>::failure(ErrorCode::InvalidExtent,
                                          "invalid observation shape");
  }
  MomentRequest borrowed{.orders = request.orders,
                         .representation = request.representation,
                         .selected = {},
                         .contraction = std::nullopt};
  std::vector<std::span<const Index>> selected;
  for (const auto& tuple : request.selected.view()) {
    selected.push_back(tuple.view());
  }
  borrowed.selected = selected;
  ErrorCode error{};
  const Index d = request.contraction.has_value() ? request.contraction->rows()
                                                  : inputDimension;
  auto outputs = requestedKeys(d, borrowed, error);
  if (error != ErrorCode::None) {
    return Result<OnlineMoments>::failure(error, "invalid moment request");
  }
  auto keys = closure(outputs);
  std::size_t indexCount = 0U;
  for (const auto& key : keys) {
    if (!checkedAdd(indexCount, key.at(0), indexCount)) {
      return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                            "online tuple storage overflows");
    }
  }
  Index candidateCount = 0U;
  Index outputKeyCount = 0U;
  Index selectedElements = 0U;
  Index mapElements = 0U;
  if (!checkedMultiply(outputs.size(), 16U, candidateCount) ||
      !checkedMultiply(outputs.size(), 5U, outputKeyCount)) {
    return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                          "online layout workspace overflows");
  }
  for (const auto& tuple : request.selected.view()) {
    if (!checkedAdd(selectedElements, tuple.view().size(), selectedElements)) {
      return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                            "selected tuple extent overflows");
    }
  }
  if (request.contraction.has_value() &&
      !checkedMultiply(request.contraction->rows(),
                       request.contraction->columns(), mapElements)) {
    return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                          "contraction map size overflows");
  }
  const Index outputDimension = request.contraction.has_value()
                                    ? request.contraction->rows()
                                    : inputDimension;
  const auto indexEntries = static_cast<Index>(indexCount);
  Index elementBudget = 0U;
  Index byteBudget = 0U;
  const auto include = [&](Index count, Index elementBytes) {
    Index bytes = 0U;
    return checkedAdd(elementBudget, count, elementBudget) &&
           checkedMultiply(count, elementBytes, bytes) &&
           checkedAdd(byteBudget, bytes, byteBudget);
  };
  // Include both request/shape inputs and the state-owned copies that coexist
  // while the factory constructs its result, plus its peak closure workspace.
  Index shapeCopies = 0U;
  Index selectedEntries = 0U;
  Index selectedCopies = 0U;
  Index mapCopies = 0U;
  Index tupleMetadata = 0U;
  Index outputWorkspace = 0U;
  Index closureWorkspace = 0U;
  Index momentValues = 0U;
  Index coordinateValues = 0U;
  Index stateValues = 0U;
  const Index outputShapeSize =
      request.contraction.has_value() ? 1U : shape.view().size();
  if (!checkedMultiply(shape.view().size(), 2U, shapeCopies) ||
      !checkedAdd(request.selected.view().size(), selectedElements,
                  selectedEntries) ||
      !checkedMultiply(selectedEntries, 2U, selectedCopies) ||
      !checkedMultiply(mapElements, 2U, mapCopies) ||
      !checkedMultiply(keys.size(), 3U, tupleMetadata) ||
      !checkedAdd(tupleMetadata, indexEntries, tupleMetadata) ||
      !checkedMultiply(outputKeyCount, 2U, outputWorkspace) ||
      !checkedMultiply(candidateCount, 5U, closureWorkspace) ||
      !checkedMultiply(keys.size(), 2U, momentValues) ||
      !checkedMultiply(outputDimension, 6U, coordinateValues) ||
      !checkedAdd(momentValues, coordinateValues, stateValues)) {
    return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                          "online layout accounting overflows");
  }
  if (!include(shapeCopies, sizeof(Index)) ||
      !include(outputShapeSize, sizeof(Index)) ||
      !include(tupleMetadata, sizeof(Index)) ||
      !include(selectedCopies, sizeof(Index)) ||
      !include(mapCopies, sizeof(Scalar)) ||
      !include(outputWorkspace, sizeof(Index)) ||
      !include(closureWorkspace, sizeof(Index)) ||
      !include(stateValues, sizeof(Scalar)) ||
      elementBudget > limits.elements || byteBudget > limits.bytes) {
    return Result<OnlineMoments>::failure(
        ErrorCode::ResourceLimit, "online moment layout exceeds limits");
  }
  std::vector<Index> orders;
  std::vector<Index> indices;
  orders.reserve(keys.size());
  indices.reserve(indexCount);
  for (const auto& key : keys) {
    orders.push_back(key.at(0));
    for (Index axis = 0U; axis < key.at(0); ++axis) {
      indices.push_back(key.at(axis + 1U));
    }
  }
  auto ownedOrders = Array<Index>::copy(orders, limits);
  auto ownedIndices = Array<Index>::copy(indices, limits);
  auto ownedMean = Vector(d, 0.0);
  auto sums = Vector(keys.size(), 0.0);
  auto scratch = Vector(keys.size(), 0.0);
  auto delta = Vector(d, 0.0);
  auto newMean = Vector(d, 0.0);
  auto projectedObservation = Vector(d, 0.0);
  std::vector<Index> outputExtents;
  if (request.contraction.has_value()) {
    outputExtents.push_back(d);
  } else {
    outputExtents.assign(shape.view().begin(), shape.view().end());
  }
  auto outputShape = Array<Index>::copy(outputExtents, limits);
  if (!ownedOrders || !ownedIndices || !outputShape) {
    return Result<OnlineMoments>::failure(
        ErrorCode::ResourceLimit, "online moment metadata exceeds limits");
  }
  return Result<OnlineMoments>::success(
      OnlineMoments{.shape = shape,
                    .output_shape = std::move(outputShape).value(),
                    .input_dimension = inputDimension,
                    .tuple_orders = std::move(ownedOrders).value(),
                    .tuple_indices = std::move(ownedIndices).value(),
                    .centered_sums = std::move(sums),
                    .scratch_sums = std::move(scratch),
                    .mean = std::move(ownedMean),
                    .anchor = Vector(d, 0.0),
                    .mean_offset = Vector(d, 0.0),
                    .scratch_delta = std::move(delta),
                    .scratch_mean = std::move(newMean),
                    .scratch_observation = std::move(projectedObservation),
                    .mass = 0.0,
                    .raw_weight_powers = {},
                    .count = 0U,
                    .positive_count = 0U,
                    .all_unit_weights = true,
                    .stored_request = request,
                    .limits = limits});
}

MomentRequest borrowedRequest(const OwnedMomentRequest& request,
                              std::vector<std::span<const Index>>& selected) {
  for (const auto& tuple : request.selected.view()) {
    selected.push_back(tuple.view());
  }
  return {.orders = request.orders,
          .representation = request.representation,
          .selected = selected,
          .contraction = std::nullopt};
}

void copyPrefix(const RollingMoments& rolling, bool front, Index slot,
                OnlineMoments& target) {
  static_cast<void>(reset(target));
  const Index d = target.mean.size();
  const Index q = target.centered_sums.size();
  const auto& means = front ? rolling.front_means : rolling.back_means;
  const auto& anchors = front ? rolling.front_anchors : rolling.back_anchors;
  const auto& offsets = front ? rolling.front_offsets : rolling.back_offsets;
  const auto& sums = front ? rolling.front_sums : rolling.back_sums;
  const auto& masses = front ? rolling.front_masses : rolling.back_masses;
  const auto& positives =
      front ? rolling.front_positive_counts : rolling.back_positive_counts;
  const auto& units =
      front ? rolling.front_unit_flags : rolling.back_unit_flags;
  const auto& powers =
      front ? rolling.front_weight_powers : rolling.back_weight_powers;
  for (Index i = 0U; i < d; ++i) {
    target.mean.at(i) = means.at(slot * d + i);
    target.anchor.at(i) = anchors.at(slot * d + i);
    target.mean_offset.at(i) = offsets.at(slot * d + i);
  }
  for (Index i = 0U; i < q; ++i) {
    target.centered_sums.at(i) = sums.at(slot * q + i);
  }
  target.mass = masses.at(slot);
  target.count = slot + 1U;
  target.positive_count = static_cast<Index>(positives.at(slot));
  target.all_unit_weights = units.at(slot) != 0.0;
  for (Index i = 0U; i < 3U; ++i) {
    target.raw_weight_powers.at(i) = powers.at(slot * 3U + i);
  }
}

void storePrefix(RollingMoments& rolling, bool front, Index slot,
                 const OnlineMoments& source) {
  const Index d = source.mean.size();
  const Index q = source.centered_sums.size();
  auto& means = front ? rolling.front_means : rolling.back_means;
  auto& anchors = front ? rolling.front_anchors : rolling.back_anchors;
  auto& offsets = front ? rolling.front_offsets : rolling.back_offsets;
  auto& sums = front ? rolling.front_sums : rolling.back_sums;
  auto& masses = front ? rolling.front_masses : rolling.back_masses;
  auto& positives =
      front ? rolling.front_positive_counts : rolling.back_positive_counts;
  auto& units = front ? rolling.front_unit_flags : rolling.back_unit_flags;
  auto& powers =
      front ? rolling.front_weight_powers : rolling.back_weight_powers;
  for (Index i = 0U; i < d; ++i) {
    means.at(slot * d + i) = source.mean.at(i);
    anchors.at(slot * d + i) = source.anchor.at(i);
    offsets.at(slot * d + i) = source.mean_offset.at(i);
  }
  for (Index i = 0U; i < q; ++i) {
    sums.at(slot * q + i) = source.centered_sums.at(i);
  }
  masses.at(slot) = source.mass;
  positives.at(slot) = static_cast<Scalar>(source.positive_count);
  units.at(slot) = source.all_unit_weights ? 1.0 : 0.0;
  for (Index i = 0U; i < 3U; ++i) {
    powers.at(slot * 3U + i) = source.raw_weight_powers.at(i);
  }
}

bool validRollingState(const RollingMoments& state) {
  if (state.capacity == 0U || !validOnlineState(state.work) ||
      !validOnlineState(state.work2) ||
      state.input_dimension != state.work.input_dimension ||
      state.front_count > state.capacity || state.back_count > state.capacity ||
      state.front_count >
          std::numeric_limits<Index>::max() - state.back_count ||
      state.front_count + state.back_count != state.count ||
      state.count > state.capacity ||
      state.capacity >
          std::numeric_limits<Index>::max() / state.input_dimension ||
      state.capacity >
          std::numeric_limits<Index>::max() / state.work.mean.size()) {
    return false;
  }
  const Index inputEntries = state.capacity * state.input_dimension;
  const Index outputEntries = state.capacity * state.work.mean.size();
  const Index sumEntries = state.capacity * state.work.centered_sums.size();
  return state.observations.size() == inputEntries &&
         state.weights.size() == state.capacity &&
         state.front_observations.size() == inputEntries &&
         state.front_weights.size() == state.capacity &&
         state.front_means.size() == outputEntries &&
         state.front_anchors.size() == outputEntries &&
         state.front_offsets.size() == outputEntries &&
         state.front_sums.size() == sumEntries &&
         state.front_masses.size() == state.capacity &&
         state.front_positive_counts.size() == state.capacity &&
         state.front_unit_flags.size() == state.capacity &&
         state.front_weight_powers.size() == state.capacity * 3U &&
         state.back_means.size() == outputEntries &&
         state.back_anchors.size() == outputEntries &&
         state.back_offsets.size() == outputEntries &&
         state.back_sums.size() == sumEntries &&
         state.back_masses.size() == state.capacity &&
         state.back_positive_counts.size() == state.capacity &&
         state.back_unit_flags.size() == state.capacity &&
         state.back_weight_powers.size() == state.capacity * 3U &&
         state.tuple_orders.view().size() ==
             state.work.tuple_orders.view().size() &&
         std::equal(state.tuple_orders.view().begin(),
                    state.tuple_orders.view().end(),
                    state.work.tuple_orders.view().begin()) &&
         state.tuple_indices.view().size() ==
             state.work.tuple_indices.view().size() &&
         std::equal(state.tuple_indices.view().begin(),
                    state.tuple_indices.view().end(),
                    state.work.tuple_indices.view().begin());
}

Result<MomentEstimate> makeEstimate(const OnlineMoments& state, WeightRule rule,
                                    Limits limits) {
  if (!validOnlineState(state)) {
    return Result<MomentEstimate>::failure(ErrorCode::InvalidDomain,
                                           "online state structure is invalid");
  }
  if (state.count == 0U || state.mass <= 0.0) {
    return Result<MomentEstimate>::failure(ErrorCode::InsufficientSamples,
                                           "no positive-weight observations");
  }
  if (rule != WeightRule::empirical && rule != WeightRule::unweighted) {
    return Result<MomentEstimate>::failure(ErrorCode::InvalidDomain,
                                           "unsupported online weight rule");
  }
  if (rule == WeightRule::unweighted && !state.all_unit_weights) {
    return Result<MomentEstimate>::failure(
        ErrorCode::InvalidDomain, "unweighted state contains non-unit weights");
  }
  std::vector<std::span<const Index>> selected;
  const MomentRequest request = borrowedRequest(state.stored_request, selected);
  bool valid = false;
  const Index d = dimensions(state.output_shape.view(), valid);
  if (!valid || preflightRequest(d, request, state.limits) != ErrorCode::None) {
    return Result<MomentEstimate>::failure(
        ErrorCode::InvalidDomain, "stored online request is malformed");
  }
  ErrorCode error{};
  auto outputs = requestedKeys(d, request, error);
  if (error != ErrorCode::None) {
    return Result<MomentEstimate>::failure(
        ErrorCode::InvalidDomain, "stored online request is malformed");
  }
  Index flatEntries = 0U;
  Index termCount = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) != 0U) {
      ++termCount;
    }
  }
  if (request.representation == MomentRepresentation::selected) {
    for (const Key& key : outputs) {
      if (!checkedAdd(flatEntries, key.at(0), flatEntries)) {
        return Result<MomentEstimate>::failure(
            ErrorCode::SizeOverflow, "selected result layout overflows");
      }
    }
  }
  Index valueEntries = 0U;
  Index flatCopies = 0U;
  Index keyEntries = 0U;
  Index termBytes = 0U;
  Index termEntries = 0U;
  Index meanBytes = 0U;
  Index shapeBytes = 0U;
  Index valueBytes = 0U;
  Index flatBytes = 0U;
  Index keysBytes = 0U;
  Index resultBytes = 0U;
  Index resultElements = 0U;
  if (!checkedMultiply(outputs.size(), 3U, valueEntries) ||
      !checkedMultiply(flatEntries, 3U, flatCopies) ||
      !checkedMultiply(outputs.size(), 10U, keyEntries) ||
      !checkedMultiply(termCount, 2U * sizeof(MomentTerm), termBytes) ||
      !checkedMultiply(termCount, 2U, termEntries) ||
      !checkedMultiply(valueEntries, sizeof(Scalar), valueBytes) ||
      !checkedMultiply(flatCopies, sizeof(Index), flatBytes) ||
      !checkedMultiply(keyEntries, sizeof(Index), keysBytes) ||
      !checkedMultiply(d, sizeof(Scalar), meanBytes) ||
      !checkedMultiply(state.output_shape.view().size(), sizeof(Index),
                       shapeBytes) ||
      !checkedAdd(valueBytes, flatBytes, resultBytes) ||
      !checkedAdd(resultBytes, keysBytes, resultBytes) ||
      !checkedAdd(resultBytes, termBytes, resultBytes) ||
      !checkedAdd(resultBytes, meanBytes, resultBytes) ||
      !checkedAdd(resultBytes, shapeBytes, resultBytes)) {
    return Result<MomentEstimate>::failure(ErrorCode::SizeOverflow,
                                           "moment estimate storage overflows");
  }
  if (!checkedAdd(valueEntries, flatCopies, resultElements) ||
      !checkedAdd(resultElements, keyEntries, resultElements) ||
      !checkedAdd(resultElements, termEntries, resultElements) ||
      !checkedAdd(resultElements, d, resultElements) ||
      !checkedAdd(resultElements, state.output_shape.view().size(),
                  resultElements) ||
      resultElements > limits.elements || resultBytes > limits.bytes) {
    return Result<MomentEstimate>::failure(ErrorCode::ResourceLimit,
                                           "moment estimate exceeds limits");
  }
  std::vector<MomentTerm> built;
  Index current = std::numeric_limits<Index>::max();
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    std::vector<Key> orderKeys;
    for (const Key& key : outputs) {
      if (key.at(0) == order) {
        orderKeys.push_back(key);
      }
    }
    std::vector<Scalar> values;
    std::vector<Index> flat;
    values.reserve(orderKeys.size());
    for (const Key& key : orderKeys) {
      if (key.at(0) == 1U) {
        values.push_back(0.0);
        if (request.representation == MomentRepresentation::selected) {
          flat.push_back(key.at(1));
        }
        continue;
      }
      current = findKey(state, key);
      if (current == std::numeric_limits<Index>::max()) {
        return Result<MomentEstimate>::failure(ErrorCode::InvalidDomain,
                                               "online closure is incomplete");
      }
      values.push_back(state.centered_sums.at(current));
      if (request.representation == MomentRepresentation::selected) {
        for (Index axis = 0U; axis < order; ++axis) {
          flat.push_back(key.at(axis + 1U));
        }
      }
    }
    auto ownedValues = Vector(values);
    auto ownedFlat = Array<Index>::copy(flat, limits);
    built.push_back(MomentTerm{
        .order = order,
        .layout = request.representation == MomentRepresentation::contraction
                      ? MomentRepresentation::dense
                      : request.representation,
        .flattened_tuples = std::move(ownedFlat).value(),
        .values = std::move(ownedValues)});
  }
  auto ownedTerms = Array<MomentTerm>::copy(built, limits);
  auto ownedShape = Array<Index>::copy(state.output_shape.view(), limits);
  auto populationMean = Vector(state.mean);
  Support support{.input = state.count,
                  .valid = state.count,
                  .positive = state.positive_count,
                  .frequency_mass = std::nullopt,
                  .weight_mass = state.mass,
                  .normalized_weight_powers = {},
                  .effective_count = std::nullopt,
                  .degrees_of_freedom = std::nullopt,
                  .squared_weight_mass = std::nullopt};
  const long double squaredMass = static_cast<long double>(state.mass) *
                                  state.mass * state.raw_weight_powers.at(0U);
  if (std::isfinite(squaredMass) &&
      squaredMass <= std::numeric_limits<Scalar>::max()) {
    support.squared_weight_mass = static_cast<Scalar>(squaredMass);
  }
  support.normalized_weight_powers[0] = state.raw_weight_powers.at(0);
  support.normalized_weight_powers[1] = state.raw_weight_powers.at(1);
  support.normalized_weight_powers[2] = state.raw_weight_powers.at(2);
  EstimatorDiagnostics diagnostics{
      .estimand = Estimand::observed,
      .weight = rule,
      .center = CenterRule::estimated,
      .approximation = Approximation::exact,
      .global = support,
      .coordinate_support = {},
      .tuple_support = {},
      .reciprocal_condition = std::nullopt,
      .numerical_error = std::nullopt,
      .sampling_error = std::nullopt,
      .rank = std::nullopt,
      .algorithm = "online-central-moments",
      .premises = "empirical weighted population moments; common support",
      .warnings = {}};
  if (!ownedTerms || !ownedShape) {
    return Result<MomentEstimate>::failure(
        ErrorCode::ResourceLimit, "moment estimate metadata exceeds limits");
  }
  return Result<MomentEstimate>::success(
      MomentEstimate{.observation_shape = std::move(ownedShape).value(),
                     .terms = std::move(ownedTerms).value(),
                     .diagnostics = std::move(diagnostics),
                     .population_mean = std::move(populationMean)});
}

} // namespace

Result<OnlineMoments> makeOnlineMoments(std::span<const Index> shape,
                                        const MomentRequest& request,
                                        Limits limits) {
  bool valid = false;
  const Index inputDimension = dimensions(shape, valid);
  if (!valid) {
    return Result<OnlineMoments>::failure(ErrorCode::InvalidExtent,
                                          "invalid observation shape");
  }
  Index outputDimension = inputDimension;
  if (request.representation == MomentRepresentation::contraction) {
    if (!request.contraction.has_value()) {
      return Result<OnlineMoments>::failure(ErrorCode::InvalidDomain,
                                            "contraction request has no map");
    }
    const ConstMatrixView map = *request.contraction;
    if (map.columns() != inputDimension || map.rows() == 0U) {
      return Result<OnlineMoments>::failure(
          ErrorCode::DimensionMismatch,
          "contraction map shape is incompatible");
    }
    Index mapElements = 0U;
    if (!checkedMultiply(map.rows(), map.columns(), mapElements)) {
      return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                            "contraction map size overflows");
    }
    if (mapElements > limits.elements ||
        mapElements > limits.bytes / sizeof(Scalar)) {
      return Result<OnlineMoments>::failure(ErrorCode::ResourceLimit,
                                            "contraction map exceeds limits");
    }
    outputDimension = map.rows();
  } else if (request.contraction.has_value()) {
    return Result<OnlineMoments>::failure(
        ErrorCode::InvalidDomain,
        "contraction map requires contraction representation");
  }
  const ErrorCode preflight =
      preflightRequest(outputDimension, request, limits);
  if (preflight != ErrorCode::None) {
    return Result<OnlineMoments>::failure(
        preflight,
        "online moment request exceeds validation or resource limits");
  }
  Index selectedElements = 0U;
  for (const auto tuple : request.selected) {
    if (!checkedAdd(selectedElements, tuple.size(), selectedElements)) {
      return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                            "selected tuple extent overflows");
    }
  }
  Index selectedBytes = 0U;
  if (!checkedMultiply(selectedElements, sizeof(Index), selectedBytes)) {
    return Result<OnlineMoments>::failure(ErrorCode::SizeOverflow,
                                          "selected tuple storage overflows");
  }
  if (selectedBytes > limits.bytes) {
    return Result<OnlineMoments>::failure(ErrorCode::ResourceLimit,
                                          "selected tuples exceed limits");
  }
  std::vector<Array<Index>> selected;
  selected.reserve(request.selected.size());
  for (const auto tuple : request.selected) {
    auto owned = Array<Index>::copy(tuple, limits);
    if (!owned) {
      return Result<OnlineMoments>::failure(owned.status().code(),
                                            "selected tuple exceeds limits");
    }
    selected.push_back(std::move(owned).value());
  }
  auto ownedSelected = Array<Array<Index>>::copy(selected, limits);
  auto ownedShape = Array<Index>::copy(shape, limits);
  if (!ownedSelected || !ownedShape) {
    return Result<OnlineMoments>::failure(ErrorCode::ResourceLimit,
                                          "online request exceeds limits");
  }
  std::optional<Matrix> ownedMap;
  if (request.contraction.has_value()) {
    const ConstMatrixView map = *request.contraction;
    Matrix copy(map.rows(), map.columns());
    for (Index row = 0U; row < map.rows(); ++row) {
      for (Index column = 0U; column < map.columns(); ++column) {
        const Scalar value = map.at(row, column);
        if (!std::isfinite(value)) {
          return Result<OnlineMoments>::failure(
              ErrorCode::NonFiniteInput, "contraction map is not finite");
        }
        copy.at(row, column) = value;
      }
    }
    ownedMap = std::move(copy);
  }
  const OwnedMomentRequest ownedRequest{
      .orders = request.orders,
      .representation = request.representation,
      .selected = std::move(ownedSelected).value(),
      .contraction = std::move(ownedMap)};
  return makeOnlineFromOwned(ownedShape.value(), ownedRequest, limits);
}

Status add(OnlineMoments& state, ConstVectorView observation, Scalar weight) {
  Status input = validateObservation(state, observation, weight);
  if (!input.ok()) {
    return input;
  }
  if (state.count == std::numeric_limits<Index>::max()) {
    return fail(ErrorCode::SizeOverflow, "online observation count overflow");
  }
  const Index d = state.mean.size();
  if (weight == 0.0) {
    ++state.count;
    state.all_unit_weights = false;
    return Status::success();
  }
  const Scalar oldMass = state.mass;
  const Scalar total = state.mass + weight;
  if (!std::isfinite(total) || total <= 0.0) {
    return fail(ErrorCode::NonFiniteResult,
                "online weight mass is not representable");
  }
  const Scalar ratio = weight / total;
  const VectorView scratch =
      VectorView::create(state.scratch_sums.data() == nullptr
                             ? std::span<Scalar>{}
                             : std::span<Scalar>{state.scratch_sums.data(),
                                                 state.scratch_sums.size()},
                         state.scratch_sums.size())
          .value();
  if (state.stored_request.contraction.has_value()) {
    const Matrix& map = *state.stored_request.contraction;
    for (Index row = 0U; row < d; ++row) {
      Scalar value = 0.0;
      for (Index column = 0U; column < state.input_dimension; ++column) {
        value += map.at(row, column) * observation.at(column);
      }
      if (!std::isfinite(value)) {
        return fail(ErrorCode::NonFiniteResult,
                    "projected online observation is not finite");
      }
      state.scratch_observation.at(row) = value;
    }
  } else {
    for (Index i = 0U; i < d; ++i) {
      state.scratch_observation.at(i) = observation.at(i);
    }
  }
  for (Index i = 0U; i < d; ++i) {
    if (oldMass == 0.0) {
      state.scratch_delta.at(i) = 0.0;
      state.scratch_mean.at(i) = 0.0;
    } else {
      state.scratch_delta.at(i) =
          (state.scratch_observation.at(i) - state.anchor.at(i)) -
          state.mean_offset.at(i);
      state.scratch_mean.at(i) =
          state.mean_offset.at(i) + ratio * state.scratch_delta.at(i);
    }
    if (!std::isfinite(state.scratch_delta.at(i)) ||
        !std::isfinite(state.scratch_mean.at(i)) ||
        !std::isfinite(state.anchor.at(i) + state.scratch_mean.at(i))) {
      return fail(ErrorCode::NonFiniteResult,
                  "online mean update is not representable");
    }
  }
  for (Index entry = 0U; entry < state.centered_sums.size(); ++entry) {
    const Key key = keyAt(state, entry);
    ScaledAccumulator sum;
    const Index order = key.at(0);
    for (Index mask = 0U; mask < (1U << order); ++mask) {
      Key child{};
      Index childOrder = 0U;
      ScaledValue shift = scaled(1.0);
      for (Index axis = 0U; axis < order; ++axis) {
        if ((mask & (1U << axis)) != 0U) {
          shift = scaledMultiply(shift, scaled(-ratio));
          shift = scaledMultiply(
              shift, scaled(state.scratch_delta.at(key.at(axis + 1U))));
        } else {
          child.at(++childOrder) = key.at(axis + 1U);
        }
      }
      child.at(0) = childOrder;
      if (childOrder == 0U) {
        sum.add(scaledMultiply(shift, scaled(oldMass / total)));
      } else if (childOrder > 1U) {
        const Index childIndex = findKey(state, child);
        ScaledValue term = scaledMultiply(shift, scaled(oldMass / total));
        term = scaledMultiply(term, scaled(state.centered_sums.at(childIndex)));
        sum.add(term);
      }
    }
    ScaledValue point = scaled(ratio);
    for (Index axis = 0U; axis < order; ++axis) {
      const Index coordinate = key.at(axis + 1U);
      const Scalar residual = oldMass == 0.0
                                  ? 0.0
                                  : (state.scratch_observation.at(coordinate) -
                                     state.anchor.at(coordinate)) -
                                        state.scratch_mean.at(coordinate);
      point = scaledMultiply(point, scaled(residual));
    }
    sum.add(point);
    Scalar value = 0.0;
    if (!narrow(sum.value(), value)) {
      return fail(ErrorCode::NonFiniteResult,
                  "online centered moment overflow");
    }
    scratch.at(entry) = value;
  }
  for (Index i = 0U; i < d; ++i) {
    if (oldMass == 0.0) {
      state.anchor.at(i) = state.scratch_observation.at(i);
      state.mean_offset.at(i) = 0.0;
      state.mean.at(i) = state.anchor.at(i);
    } else {
      state.mean_offset.at(i) = state.scratch_mean.at(i);
      state.mean.at(i) = state.anchor.at(i) + state.mean_offset.at(i);
    }
  }
  for (Index i = 0U; i < state.centered_sums.size(); ++i) {
    state.centered_sums.at(i) = scratch.at(i);
  }
  state.mass = total;
  ++state.count;
  ++state.positive_count;
  state.all_unit_weights = state.all_unit_weights && weight == 1.0;
  for (Index power = 2U; power <= 4U; ++power) {
    const Scalar oldRatio = oldMass / total;
    Scalar nextPower = state.raw_weight_powers.at(power - 2U) *
                       std::pow(oldRatio, static_cast<Scalar>(power));
    nextPower += std::pow(weight / total, static_cast<Scalar>(power));
    state.raw_weight_powers.at(power - 2U) = nextPower;
  }
  return Status::success();
}

Status merge(OnlineMoments& destination, const OnlineMoments& source) {
  if (!validOnlineState(destination) || !validOnlineState(source)) {
    return fail(ErrorCode::InvalidDomain, "online state structure is invalid");
  }
  if (&destination == &source) {
    return fail(ErrorCode::InvalidDomain,
                "cannot merge an online state with itself");
  }
  if (destination.input_dimension != source.input_dimension ||
      destination.output_shape.view().size() !=
          source.output_shape.view().size() ||
      !std::equal(destination.output_shape.view().begin(),
                  destination.output_shape.view().end(),
                  source.output_shape.view().begin()) ||
      destination.shape.view().size() != source.shape.view().size() ||
      !std::equal(destination.shape.view().begin(),
                  destination.shape.view().end(),
                  source.shape.view().begin()) ||
      destination.stored_request.orders != source.stored_request.orders ||
      destination.stored_request.representation !=
          source.stored_request.representation ||
      destination.stored_request.selected.view().size() !=
          source.stored_request.selected.view().size() ||
      destination.tuple_orders.view().size() !=
          source.tuple_orders.view().size() ||
      !std::equal(destination.tuple_orders.view().begin(),
                  destination.tuple_orders.view().end(),
                  source.tuple_orders.view().begin()) ||
      destination.tuple_indices.view().size() !=
          source.tuple_indices.view().size() ||
      !std::equal(destination.tuple_indices.view().begin(),
                  destination.tuple_indices.view().end(),
                  source.tuple_indices.view().begin())) {
    return fail(ErrorCode::DimensionMismatch,
                "online moment states have incompatible layouts");
  }
  for (Index tuple = 0U;
       tuple < destination.stored_request.selected.view().size(); ++tuple) {
    const auto left = destination.stored_request.selected.view()[tuple].view();
    const auto right = source.stored_request.selected.view()[tuple].view();
    if (left.size() != right.size() ||
        !std::equal(left.begin(), left.end(), right.begin())) {
      return fail(ErrorCode::DimensionMismatch,
                  "online moment selectors differ");
    }
  }
  const auto& leftMap = destination.stored_request.contraction;
  const auto& rightMap = source.stored_request.contraction;
  if (leftMap.has_value() != rightMap.has_value()) {
    return fail(ErrorCode::DimensionMismatch,
                "online contraction policies differ");
  }
  if (leftMap.has_value()) {
    if (leftMap->rows() != rightMap->rows() ||
        leftMap->columns() != rightMap->columns()) {
      return fail(ErrorCode::DimensionMismatch,
                  "online contraction maps differ");
    }
    for (Index row = 0U; row < leftMap->rows(); ++row) {
      for (Index column = 0U; column < leftMap->columns(); ++column) {
        if (leftMap->at(row, column) != rightMap->at(row, column)) {
          return fail(ErrorCode::DimensionMismatch,
                      "online contraction maps differ");
        }
      }
    }
  }
  if (source.count > std::numeric_limits<Index>::max() - destination.count) {
    return fail(ErrorCode::SizeOverflow, "online observation count overflow");
  }
  if (source.positive_count >
      std::numeric_limits<Index>::max() - destination.positive_count) {
    return fail(ErrorCode::SizeOverflow, "online positive count overflow");
  }
  if (source.mass == 0.0) {
    destination.count += source.count;
    destination.all_unit_weights =
        destination.all_unit_weights && source.all_unit_weights;
    return Status::success();
  }
  if (destination.mass == 0.0) {
    if (source.mean.size() != destination.mean.size() ||
        source.centered_sums.size() != destination.centered_sums.size()) {
      return fail(ErrorCode::InvalidDomain,
                  "online state storage is malformed");
    }
    for (Index i = 0U; i < destination.mean.size(); ++i) {
      destination.mean.at(i) = source.mean.at(i);
      destination.anchor.at(i) = source.anchor.at(i);
      destination.mean_offset.at(i) = source.mean_offset.at(i);
    }
    for (Index i = 0U; i < destination.centered_sums.size(); ++i) {
      destination.centered_sums.at(i) = source.centered_sums.at(i);
    }
    destination.mass = source.mass;
    destination.count += source.count;
    destination.positive_count += source.positive_count;
    destination.all_unit_weights =
        destination.all_unit_weights && source.all_unit_weights;
    destination.raw_weight_powers = source.raw_weight_powers;
    return Status::success();
  }
  const Scalar total = destination.mass + source.mass;
  if (!std::isfinite(total)) {
    return fail(ErrorCode::NonFiniteResult, "merged weight mass overflow");
  }
  const Scalar leftRatio = destination.mass / total;
  const Scalar rightRatio = source.mass / total;
  const Index d = destination.mean.size();
  for (Index i = 0U; i < d; ++i) {
    destination.scratch_delta.at(i) =
        (source.anchor.at(i) - destination.anchor.at(i)) +
        (source.mean_offset.at(i) - destination.mean_offset.at(i));
    const bool keepDestinationAnchor = destination.mass >= source.mass;
    destination.scratch_mean.at(i) =
        keepDestinationAnchor ? destination.mean_offset.at(i) +
                                    rightRatio * destination.scratch_delta.at(i)
                              : source.mean_offset.at(i) -
                                    leftRatio * destination.scratch_delta.at(i);
    const Scalar mergedAnchor = destination.mass >= source.mass
                                    ? destination.anchor.at(i)
                                    : source.anchor.at(i);
    if (!std::isfinite(destination.scratch_delta.at(i)) ||
        !std::isfinite(destination.scratch_mean.at(i)) ||
        !std::isfinite(mergedAnchor + destination.scratch_mean.at(i))) {
      return fail(ErrorCode::NonFiniteResult,
                  "merged online mean is not representable");
    }
  }
  const VectorView scratch =
      VectorView::create(std::span<Scalar>{destination.scratch_sums.data(),
                                           destination.scratch_sums.size()},
                         destination.scratch_sums.size())
          .value();
  for (Index entry = 0U; entry < destination.centered_sums.size(); ++entry) {
    const Key key = keyAt(destination, entry);
    ScaledAccumulator sum;
    const Index order = key.at(0);
    for (Index mask = 0U; mask < (1U << order); ++mask) {
      for (Index side = 0U; side < 2U; ++side) {
        const auto& block = side == 0U ? destination : source;
        const Scalar ratio = side == 0U ? leftRatio : rightRatio;
        ScaledValue shift = scaled(1.0);
        Key child{};
        Index childOrder = 0U;
        for (Index axis = 0U; axis < order; ++axis) {
          if ((mask & (1U << axis)) != 0U) {
            const Scalar signedRatio = side == 0U ? -rightRatio : leftRatio;
            shift = scaledMultiply(shift, scaled(signedRatio));
            shift = scaledMultiply(
                shift, scaled(destination.scratch_delta.at(key.at(axis + 1U))));
          } else {
            child.at(++childOrder) = key.at(axis + 1U);
          }
        }
        child.at(0) = childOrder;
        Scalar moment = 0.0;
        if (childOrder == 0U) {
          moment = 1.0;
        } else if (childOrder > 1U) {
          const Index childIndex = findKey(block, child);
          moment = block.centered_sums.at(childIndex);
        }
        ScaledValue term = scaledMultiply(scaled(ratio), shift);
        term = scaledMultiply(term, scaled(moment));
        sum.add(term);
      }
    }
    Scalar value = 0.0;
    if (!narrow(sum.value(), value)) {
      return fail(ErrorCode::NonFiniteResult,
                  "merged centered moment overflow");
    }
    scratch.at(entry) = value;
  }
  for (Index i = 0U; i < d; ++i) {
    if (destination.mass < source.mass) {
      destination.anchor.at(i) = source.anchor.at(i);
    }
    destination.mean_offset.at(i) = destination.scratch_mean.at(i);
    destination.mean.at(i) =
        destination.anchor.at(i) + destination.mean_offset.at(i);
  }
  for (Index i = 0U; i < destination.centered_sums.size(); ++i) {
    destination.centered_sums.at(i) = scratch.at(i);
  }
  destination.mass = total;
  destination.count += source.count;
  destination.positive_count += source.positive_count;
  destination.all_unit_weights =
      destination.all_unit_weights && source.all_unit_weights;
  for (Index power = 2U; power <= 4U; ++power) {
    const auto exponent = static_cast<Scalar>(power);
    destination.raw_weight_powers.at(power - 2U) =
        std::pow(leftRatio, exponent) *
            destination.raw_weight_powers.at(power - 2U) +
        std::pow(rightRatio, exponent) *
            source.raw_weight_powers.at(power - 2U);
  }
  return Status::success();
}

Result<MomentEstimate> finalize(const OnlineMoments& state, WeightRule rule,
                                Limits limits) {
  return makeEstimate(state, rule, limits);
}

Status reset(OnlineMoments& state) {
  if (!validOnlineState(state)) {
    return fail(ErrorCode::InvalidDomain, "online state structure is invalid");
  }
  for (Index i = 0U; i < state.mean.size(); ++i) {
    state.mean.at(i) = 0.0;
    state.anchor.at(i) = 0.0;
    state.mean_offset.at(i) = 0.0;
  }
  for (Index i = 0U; i < state.centered_sums.size(); ++i) {
    state.centered_sums.at(i) = 0.0;
    state.scratch_sums.at(i) = 0.0;
  }
  state.mass = 0.0;
  state.raw_weight_powers = {};
  state.count = 0U;
  state.positive_count = 0U;
  state.all_unit_weights = true;
  for (Index i = 0U; i < state.scratch_delta.size(); ++i) {
    state.scratch_delta.at(i) = 0.0;
    state.scratch_mean.at(i) = 0.0;
  }
  return Status::success();
}

Result<RollingMoments> makeRollingMoments(std::span<const Index> shape,
                                          Index capacity,
                                          const MomentRequest& request,
                                          Limits limits) {
  if (capacity == 0U) {
    return Result<RollingMoments>::failure(ErrorCode::InvalidExtent,
                                           "rolling capacity must be positive");
  }
  auto emptyOnline = makeOnlineMoments(shape, request, limits);
  if (!emptyOnline) {
    return Result<RollingMoments>::failure(emptyOnline.status().code(),
                                           "rolling layout exceeds limits");
  }
  const Index d = emptyOnline.value().mean.size();
  const Index inputD = emptyOnline.value().input_dimension;
  Index observationElements = 0U;
  if (!checkedMultiply(capacity, inputD, observationElements)) {
    return Result<RollingMoments>::failure(
        ErrorCode::SizeOverflow, "rolling observation storage overflows");
  }
  const Index q = emptyOnline.value().centered_sums.size();
  Index stackCoordinateEntries = 0U;
  Index stackPerEntry = 0U;
  Index stackEntries = 0U;
  Index observationBuffers = 0U;
  Index weightBuffers = 0U;
  Index rollingScalars = 0U;
  if (!checkedMultiply(d, 3U, stackCoordinateEntries) ||
      !checkedAdd(stackCoordinateEntries, q, stackPerEntry) ||
      !checkedAdd(stackPerEntry, 6U, stackPerEntry) ||
      !checkedMultiply(capacity, stackPerEntry, stackEntries) ||
      !checkedMultiply(stackEntries, 2U, stackEntries) ||
      !checkedMultiply(observationElements, 2U, observationBuffers) ||
      !checkedMultiply(capacity, 2U, weightBuffers) ||
      !checkedAdd(stackEntries, observationBuffers, rollingScalars) ||
      !checkedAdd(rollingScalars, weightBuffers, rollingScalars)) {
    return Result<RollingMoments>::failure(
        ErrorCode::SizeOverflow, "rolling aggregate storage overflows");
  }
  Index selectedElements = 0U;
  for (const auto& tuple : emptyOnline.value().stored_request.selected.view()) {
    if (!checkedAdd(selectedElements, tuple.view().size(), selectedElements)) {
      return Result<RollingMoments>::failure(ErrorCode::SizeOverflow,
                                             "rolling request size overflows");
    }
  }
  Index mapElements = 0U;
  const auto& contraction = emptyOnline.value().stored_request.contraction;
  if (contraction.has_value() &&
      !checkedMultiply(contraction->rows(), contraction->columns(),
                       mapElements)) {
    return Result<RollingMoments>::failure(
        ErrorCode::SizeOverflow, "rolling contraction size overflows");
  }
  const Index selectedCount =
      emptyOnline.value().stored_request.selected.view().size();
  Index oneWorkScalar = 0U;
  Index onlineMetadata = 0U;
  Index rollingMetadata = 0U;
  Index twoMomentValues = 0U;
  Index twoWorkScalars = 0U;
  Index twoWorkMetadata = 0U;
  Index totalScalars = 0U;
  Index totalMetadata = 0U;
  Index totalElements = 0U;
  Index scalarBytes = 0U;
  Index metadataBytes = 0U;
  Index totalBytes = 0U;
  if (!checkedMultiply(d, 6U, oneWorkScalar) ||
      !checkedMultiply(q, 2U, twoMomentValues) ||
      !checkedAdd(oneWorkScalar, twoMomentValues, oneWorkScalar) ||
      !checkedAdd(oneWorkScalar, mapElements, oneWorkScalar) ||
      !checkedAdd(emptyOnline.value().shape.view().size(),
                  emptyOnline.value().output_shape.view().size(),
                  onlineMetadata) ||
      !checkedAdd(onlineMetadata, q, onlineMetadata) ||
      !checkedAdd(onlineMetadata,
                  emptyOnline.value().tuple_indices.view().size(),
                  onlineMetadata) ||
      !checkedAdd(onlineMetadata, selectedCount, onlineMetadata) ||
      !checkedAdd(onlineMetadata, selectedElements, onlineMetadata) ||
      !checkedAdd(emptyOnline.value().shape.view().size(),
                  emptyOnline.value().output_shape.view().size(),
                  rollingMetadata) ||
      !checkedAdd(rollingMetadata, q, rollingMetadata) ||
      !checkedAdd(rollingMetadata,
                  emptyOnline.value().tuple_indices.view().size(),
                  rollingMetadata) ||
      !checkedAdd(rollingMetadata, selectedCount, rollingMetadata) ||
      !checkedAdd(rollingMetadata, selectedElements, rollingMetadata) ||
      !checkedMultiply(oneWorkScalar, 2U, twoWorkScalars) ||
      !checkedMultiply(onlineMetadata, 2U, twoWorkMetadata) ||
      !checkedAdd(rollingScalars, twoWorkScalars, totalScalars) ||
      !checkedAdd(totalScalars, mapElements, totalScalars) ||
      !checkedAdd(twoWorkMetadata, rollingMetadata, totalMetadata) ||
      !checkedAdd(totalScalars, totalMetadata, totalElements) ||
      !checkedMultiply(totalScalars, sizeof(Scalar), scalarBytes) ||
      !checkedMultiply(totalMetadata, sizeof(Index), metadataBytes) ||
      !checkedAdd(scalarBytes, metadataBytes, totalBytes)) {
    return Result<RollingMoments>::failure(ErrorCode::SizeOverflow,
                                           "rolling total storage overflows");
  }
  if (totalElements > limits.elements || totalBytes > limits.bytes) {
    return Result<RollingMoments>::failure(ErrorCode::ResourceLimit,
                                           "rolling window exceeds limits");
  }
  auto ownedWeights = Vector(capacity, 0.0);
  auto ownedObservations = Vector(observationElements, 0.0);
  auto frontWeights = Vector(capacity, 0.0);
  auto frontObservations = Vector(observationElements, 0.0);
  auto frontMeans = Vector(capacity * d, 0.0);
  auto frontAnchors = Vector(capacity * d, 0.0);
  auto frontOffsets = Vector(capacity * d, 0.0);
  auto frontSums = Vector(capacity * q, 0.0);
  auto frontMasses = Vector(capacity, 0.0);
  auto frontPositives = Vector(capacity, 0.0);
  auto frontUnits = Vector(capacity, 1.0);
  auto frontPowers = Vector(capacity * 3U, 0.0);
  auto backMeans = Vector(capacity * d, 0.0);
  auto backAnchors = Vector(capacity * d, 0.0);
  auto backOffsets = Vector(capacity * d, 0.0);
  auto backSums = Vector(capacity * q, 0.0);
  auto backMasses = Vector(capacity, 0.0);
  auto backPositives = Vector(capacity, 0.0);
  auto backUnits = Vector(capacity, 1.0);
  auto backPowers = Vector(capacity * 3U, 0.0);
  auto secondWork = makeOnlineFromOwned(
      emptyOnline.value().shape, emptyOnline.value().stored_request, limits);
  if (!secondWork) {
    return Result<RollingMoments>::failure(
        secondWork.status().code(), "rolling scratch state exceeds limits");
  }
  auto ownedShape = Array<Index>::copy(shape, limits);
  auto ownedOutputShape =
      Array<Index>::copy(emptyOnline.value().output_shape.view(), limits);
  auto ownedOrders =
      Array<Index>::copy(emptyOnline.value().tuple_orders.view(), limits);
  auto ownedIndices =
      Array<Index>::copy(emptyOnline.value().tuple_indices.view(), limits);
  if (!ownedShape || !ownedOutputShape || !ownedOrders || !ownedIndices) {
    return Result<RollingMoments>::failure(ErrorCode::ResourceLimit,
                                           "rolling metadata exceeds limits");
  }
  OwnedMomentRequest storedRequest = emptyOnline.value().stored_request;
  return Result<RollingMoments>::success(
      RollingMoments{.shape = std::move(ownedShape).value(),
                     .output_shape = std::move(ownedOutputShape).value(),
                     .input_dimension = inputD,
                     .tuple_orders = std::move(ownedOrders).value(),
                     .tuple_indices = std::move(ownedIndices).value(),
                     .observations = std::move(ownedObservations),
                     .weights = std::move(ownedWeights),
                     .front_observations = std::move(frontObservations),
                     .front_weights = std::move(frontWeights),
                     .front_means = std::move(frontMeans),
                     .front_anchors = std::move(frontAnchors),
                     .front_offsets = std::move(frontOffsets),
                     .front_sums = std::move(frontSums),
                     .front_masses = std::move(frontMasses),
                     .front_positive_counts = std::move(frontPositives),
                     .front_unit_flags = std::move(frontUnits),
                     .front_weight_powers = std::move(frontPowers),
                     .back_means = std::move(backMeans),
                     .back_anchors = std::move(backAnchors),
                     .back_offsets = std::move(backOffsets),
                     .back_sums = std::move(backSums),
                     .back_masses = std::move(backMasses),
                     .back_positive_counts = std::move(backPositives),
                     .back_unit_flags = std::move(backUnits),
                     .back_weight_powers = std::move(backPowers),
                     .work = std::move(emptyOnline).value(),
                     .work2 = std::move(secondWork).value(),
                     .capacity = capacity,
                     .front_count = 0U,
                     .back_count = 0U,
                     .count = 0U,
                     .stored_request = std::move(storedRequest),
                     .limits = limits});
}

Status add(RollingMoments& state, ConstVectorView observation, Scalar weight) {
  if (!validRollingState(state)) {
    return fail(ErrorCode::InvalidDomain, "rolling state structure is invalid");
  }
  const Index d = state.input_dimension;
  if (observation.size() != d) {
    return fail(ErrorCode::DimensionMismatch,
                "rolling observation shape mismatch");
  }
  if (!std::isfinite(weight) || weight < 0.0) {
    return fail(ErrorCode::InvalidDomain,
                "rolling weight must be finite and nonnegative");
  }
  for (Index i = 0U; i < d; ++i) {
    if (!std::isfinite(observation.at(i))) {
      return fail(ErrorCode::NonFiniteInput,
                  "rolling observation is not finite");
    }
  }
  Index stagedFrontCount = state.front_count;
  Index stagedBackCount = state.back_count;
  const bool full = state.count == state.capacity;
  if (full && stagedFrontCount == 0U) {
    for (Index remaining = stagedBackCount; remaining > 0U; --remaining) {
      const Index backSlot = remaining - 1U;
      const auto item = state.observations.view().subview(backSlot * d, d);
      if (!item) {
        return fail(ErrorCode::InvalidDomain,
                    "rolling back stack is malformed");
      }
      static_cast<void>(reset(state.work));
      Status singleton =
          add(state.work, item.value(), state.weights.at(backSlot));
      if (!singleton.ok()) {
        return singleton;
      }
      if (stagedFrontCount > 0U) {
        copyPrefix(state, true, stagedFrontCount - 1U, state.work2);
        const Status combined = merge(state.work, state.work2);
        if (!combined.ok()) {
          return combined;
        }
      }
      for (Index i = 0U; i < d; ++i) {
        state.front_observations.at(stagedFrontCount * d + i) =
            item.value().at(i);
      }
      state.front_weights.at(stagedFrontCount) = state.weights.at(backSlot);
      storePrefix(state, true, stagedFrontCount, state.work);
      ++stagedFrontCount;
    }
    stagedBackCount = 0U;
  }
  if (full) {
    --stagedFrontCount;
  }
  if (stagedBackCount > 0U) {
    copyPrefix(state, false, stagedBackCount - 1U, state.work);
  } else {
    static_cast<void>(reset(state.work));
  }
  Status added = add(state.work, observation, weight);
  if (!added.ok()) {
    return added;
  }
  for (Index i = 0U; i < d; ++i) {
    state.observations.at(stagedBackCount * d + i) = observation.at(i);
  }
  state.weights.at(stagedBackCount) = weight;
  storePrefix(state, false, stagedBackCount, state.work);
  ++stagedBackCount;
  state.front_count = stagedFrontCount;
  state.back_count = stagedBackCount;
  if (!full) {
    ++state.count;
  }
  return Status::success();
}

Status evict(RollingMoments& state) {
  if (!validRollingState(state)) {
    return fail(ErrorCode::InvalidDomain, "rolling state structure is invalid");
  }
  if (state.count == 0U) {
    return fail(ErrorCode::InsufficientSamples,
                "cannot evict an empty rolling state");
  }
  if (state.front_count == 0U) {
    const Index d = state.work.mean.size();
    Index stagedFrontCount = 0U;
    for (Index remaining = state.back_count; remaining > 0U; --remaining) {
      const Index backSlot = remaining - 1U;
      const auto item = state.observations.view().subview(backSlot * d, d);
      if (!item) {
        return fail(ErrorCode::InvalidDomain,
                    "rolling back stack is malformed");
      }
      static_cast<void>(reset(state.work));
      Status singleton =
          add(state.work, item.value(), state.weights.at(backSlot));
      if (!singleton.ok()) {
        return singleton;
      }
      if (stagedFrontCount > 0U) {
        copyPrefix(state, true, stagedFrontCount - 1U, state.work2);
        const Status combined = merge(state.work, state.work2);
        if (!combined.ok()) {
          return combined;
        }
      }
      for (Index i = 0U; i < d; ++i) {
        state.front_observations.at(stagedFrontCount * d + i) =
            item.value().at(i);
      }
      state.front_weights.at(stagedFrontCount) = state.weights.at(backSlot);
      storePrefix(state, true, stagedFrontCount, state.work);
      ++stagedFrontCount;
    }
    state.front_count = stagedFrontCount;
    state.back_count = 0U;
  }
  --state.front_count;
  --state.count;
  return Status::success();
}

Result<MomentEstimate> finalize(const RollingMoments& state, WeightRule rule,
                                Limits limits) {
  if (!validRollingState(state)) {
    return Result<MomentEstimate>::failure(
        ErrorCode::InvalidDomain, "rolling state structure is invalid");
  }
  if (state.count == 0U) {
    return Result<MomentEstimate>::failure(ErrorCode::InsufficientSamples,
                                           "no positive-weight observations");
  }
  if (state.front_count > 0U) {
    copyPrefix(state, true, state.front_count - 1U, state.work);
  } else {
    copyPrefix(state, false, state.back_count - 1U, state.work);
  }
  if (state.front_count > 0U && state.back_count > 0U) {
    copyPrefix(state, false, state.back_count - 1U, state.work2);
    const Status combined = merge(state.work, state.work2);
    if (!combined.ok()) {
      return Result<MomentEstimate>::failure(combined.code(),
                                             "rolling aggregate merge failed");
    }
  }
  return makeEstimate(state.work, rule, limits);
}

Status reset(RollingMoments& state) {
  if (!validRollingState(state)) {
    return fail(ErrorCode::InvalidDomain, "rolling state structure is invalid");
  }
  for (Index i = 0U; i < state.weights.size(); ++i) {
    state.weights.at(i) = 0.0;
    state.front_weights.at(i) = 0.0;
  }
  for (Index i = 0U; i < state.observations.size(); ++i) {
    state.observations.at(i) = 0.0;
    state.front_observations.at(i) = 0.0;
  }
  for (Index i = 0U; i < state.front_means.size(); ++i) {
    state.front_means.at(i) = 0.0;
    state.front_anchors.at(i) = 0.0;
    state.front_offsets.at(i) = 0.0;
    state.back_means.at(i) = 0.0;
    state.back_anchors.at(i) = 0.0;
    state.back_offsets.at(i) = 0.0;
  }
  for (Index i = 0U; i < state.front_sums.size(); ++i) {
    state.front_sums.at(i) = 0.0;
    state.back_sums.at(i) = 0.0;
  }
  for (Index i = 0U; i < state.capacity; ++i) {
    state.front_masses.at(i) = 0.0;
    state.front_positive_counts.at(i) = 0.0;
    state.front_unit_flags.at(i) = 1.0;
    state.back_masses.at(i) = 0.0;
    state.back_positive_counts.at(i) = 0.0;
    state.back_unit_flags.at(i) = 1.0;
  }
  for (Index i = 0U; i < state.front_weight_powers.size(); ++i) {
    state.front_weight_powers.at(i) = 0.0;
    state.back_weight_powers.at(i) = 0.0;
  }
  static_cast<void>(reset(state.work));
  static_cast<void>(reset(state.work2));
  state.front_count = 0U;
  state.back_count = 0U;
  state.count = 0U;
  return Status::success();
}

} // namespace vss::math::statistics
