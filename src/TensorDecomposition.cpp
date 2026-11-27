#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorDecomposition.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace vss::math {
namespace {

using Wide = long double;

std::span<const Scalar> values(const Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

std::span<Scalar> values(Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

struct Footprint final {
public:
  [[nodiscard]] static Footprint invalid() {
    Footprint result;
    result.valid_ = false;
    return result;
  }

  [[nodiscard]] std::size_t elements() const noexcept { return elements_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool valid() const noexcept { return valid_; }
  void invalidate() noexcept { valid_ = false; }

  template <class T> void add(std::size_t count) { add(count, sizeof(T)); }

  void add(std::size_t count, std::size_t width) {
    if (width != 0 && count > static_cast<std::size_t>(
                                  std::numeric_limits<std::ptrdiff_t>::max()) /
                                  width) {
      valid_ = false;
      return;
    }
    if (!valid_ ||
        count > std::numeric_limits<std::size_t>::max() - elements_ ||
        (width != 0 &&
         count > (std::numeric_limits<std::size_t>::max() - bytes_) / width)) {
      valid_ = false;
      return;
    }
    elements_ += count;
    bytes_ += count * width;
  }

  void merge(const Footprint& other) {
    if (!other.valid() ||
        other.elements() >
            std::numeric_limits<std::size_t>::max() - elements_ ||
        other.bytes() > std::numeric_limits<std::size_t>::max() - bytes_) {
      valid_ = false;
      return;
    }
    elements_ += other.elements();
    bytes_ += other.bytes();
  }

  void maximizeWith(const Footprint& other) {
    if (!other.valid()) {
      invalidate();
      return;
    }
    elements_ = std::max(elements_, other.elements());
    bytes_ = std::max(bytes_, other.bytes());
  }

private:
  std::size_t elements_{};
  std::size_t bytes_{};
  bool valid_{true};
};

bool multiply(std::size_t first, std::size_t second, std::size_t& result) {
  if (second != 0 && first > std::numeric_limits<std::size_t>::max() / second) {
    return false;
  }
  result = first * second;
  return true;
}

bool addChecked(std::size_t first, std::size_t second, std::size_t& result) {
  if (first > std::numeric_limits<std::size_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool addProduct(Footprint& footprint, std::size_t first, std::size_t second,
                std::size_t width) {
  std::size_t count{};
  if (!multiply(first, second, count)) {
    footprint.invalidate();
    return false;
  }
  footprint.add(count, width);
  return footprint.valid();
}

void takePeak(Footprint& peak, const Footprint& candidate) {
  if (!candidate.valid()) {
    peak.invalidate();
    return;
  }
  peak.maximizeWith(candidate);
}

Footprint svdPeak(Index rows, Index columns) {
  const Index tall = std::max(rows, columns);
  const Index thin = std::min(rows, columns);
  Footprint result;
  std::size_t tallThin{};
  std::size_t thinSquared{};
  if (!multiply(tall, thin, tallThin) || !multiply(thin, thin, thinSquared)) {
    result.invalidate();
    return result;
  }
  // This mirrors the existing SVD preflight: work and left Wide matrices,
  // right Wide matrix, vector/band scratch, index ordering, and all owning
  // Scalar outputs, including untruncated thin factors.
  addProduct(result, 2, tallThin, sizeof(Wide));
  result.add(thinSquared, sizeof(Wide));
  result.add(tall, sizeof(Wide));
  addProduct(result, 2, thin, sizeof(Wide));
  result.add(tall, sizeof(Index));
  addProduct(result, 2, thin, sizeof(Index));
  result.add(tallThin, sizeof(Scalar));
  result.add(thinSquared, sizeof(Scalar));
  result.add(thin, sizeof(Scalar));
  return result;
}

bool fits(const Footprint& footprint, const Limits& limits) {
  return footprint.valid() && footprint.elements() <= limits.elements &&
         footprint.bytes() <= limits.bytes;
}

bool addAll(Footprint& target, const Footprint& other) {
  target.merge(other);
  return target.valid();
}

Footprint factorPayload(std::span<const Extent> extents,
                        std::span<const Extent> ranks, std::size_t rankCount) {
  Footprint result;
  for (Index axis = 0; axis < rankCount; ++axis) {
    addProduct(result, extents[axis], ranks[axis], sizeof(Scalar));
  }
  return result;
}

bool tensorElementCount(std::span<const Extent> extents, std::size_t& count) {
  count = 1;
  for (const auto extent : extents) {
    if (!multiply(count, extent, count)) {
      return false;
    }
  }
  return true;
}

bool coreElementCount(std::span<const Extent> ranks, std::size_t& count) {
  return tensorElementCount(ranks, count);
}

bool validPolicy(const SolvePolicy& policy) {
  const auto nonnegative = [](Scalar value) {
    return std::isfinite(value) && value >= 0;
  };
  return nonnegative(policy.rank_abs) && nonnegative(policy.rank_rel) &&
         policy.rank_rel < 1 &&
         nonnegative(policy.reciprocal_condition_floor) &&
         policy.reciprocal_condition_floor <= 1 &&
         nonnegative(policy.iteration.abs) && nonnegative(policy.iteration.rel);
}

void addShapeAndCoordinates(Footprint& footprint, std::size_t rank,
                            std::size_t shapeCopies = 0) {
  footprint.add(rank, sizeof(Index));
  addProduct(footprint, rank, shapeCopies, sizeof(Extent));
}

Footprint reconstructionPeak(std::span<const Extent> outputShape,
                             std::span<const Extent> ranks, bool chargeCore) {
  const std::size_t rank = outputShape.size();
  Footprint peak;
  std::vector<Extent> current(ranks.begin(), ranks.end());
  std::size_t currentCount{};
  if (!tensorElementCount(current, currentCount)) {
    peak.invalidate();
    return peak;
  }
  for (Index axis = 0; axis < rank; ++axis) {
    std::vector<Extent> next = current;
    next[axis] = outputShape[axis];
    std::size_t nextCount{};
    if (!tensorElementCount(next, nextCount)) {
      peak.invalidate();
      return peak;
    }
    Footprint live;
    if (axis != 0 || chargeCore) {
      live.add(currentCount, sizeof(Scalar));
      live.add(rank, sizeof(Extent)); // current tensor shape
    }
    live.add(nextCount, sizeof(Scalar));
    addProduct(live, rank, 2, sizeof(Extent)); // shape plus Tensor's owned copy
    addShapeAndCoordinates(live, rank);
    addProduct(live, rank, 2, sizeof(Index)); // ranks and current extents
    live.add(rank, sizeof(Extent));           // currentExtents workspace
    takePeak(peak, live);
    current = std::move(next);
    currentCount = nextCount;
  }
  if (rank == 0 && chargeCore) {
    peak.add(1, sizeof(Scalar));
  }
  return peak;
}

Footprint hosvdPeak(std::span<const Extent> extents,
                    std::span<const Extent> ranks) {
  const std::size_t rank = extents.size();
  std::size_t dataCount{};
  std::size_t coreCount{};
  if (!tensorElementCount(extents, dataCount) ||
      !coreElementCount(ranks, coreCount)) {
    return Footprint::invalid();
  }
  const auto allFactors = factorPayload(extents, ranks, rank);
  if (!allFactors.valid()) {
    return Footprint::invalid();
  }

  Footprint peak;
  std::size_t priorFactorElements{};
  for (Index axis = 0; axis < rank && dataCount != 0; ++axis) {
    const std::size_t modeColumns = dataCount / extents[axis];
    std::size_t newFactor{};
    if (!multiply(extents[axis], ranks[axis], newFactor) ||
        !addChecked(priorFactorElements, newFactor, priorFactorElements)) {
      return Footprint::invalid();
    }
    Footprint live;
    live.add(dataCount, sizeof(Scalar));           // unfolding
    live.add(priorFactorElements, sizeof(Scalar)); // retained axes incl. new
    live.add(rank, sizeof(Matrix)); // reserved Matrix handles in source vector
    live.add(rank, sizeof(Index));  // unfolding coordinates
    addAll(live, svdPeak(extents[axis], modeColumns));
    takePeak(peak, live);

    Footprint completionLive;
    completionLive.add(dataCount, sizeof(Scalar));
    completionLive.add(priorFactorElements, sizeof(Scalar));
    completionLive.add(rank, sizeof(Matrix));
    completionLive.add(rank, sizeof(Index));
    completionLive.add(extents[axis], sizeof(Wide));
    Footprint outputOnly;
    const Index tall = std::max(extents[axis], modeColumns);
    const Index narrow = std::min(extents[axis], modeColumns);
    std::size_t tallNarrow{};
    std::size_t narrowSquared{};
    if (!multiply(tall, narrow, tallNarrow) ||
        !multiply(narrow, narrow, narrowSquared)) {
      return Footprint::invalid();
    }
    outputOnly.add(tallNarrow, sizeof(Scalar));
    outputOnly.add(narrowSquared, sizeof(Scalar));
    outputOnly.add(narrow, sizeof(Scalar));
    addAll(completionLive, outputOnly);
    takePeak(peak, completionLive);
  }

  // Sequential core projections retain the old intermediate only while the
  // next, smaller tensor is filled. The original input remains borrowed.
  std::vector<Extent> current(extents.begin(), extents.end());
  std::size_t currentCount = dataCount;
  std::size_t coreContractionPeak{};
  for (Index axis = 0; axis < rank; ++axis) {
    std::vector<Extent> next = current;
    next[axis] = ranks[axis];
    std::size_t nextCount{};
    if (!tensorElementCount(next, nextCount)) {
      return Footprint::invalid();
    }
    std::size_t liveCount{};
    if (!addChecked(axis == 0 ? 0 : currentCount, nextCount, liveCount)) {
      return Footprint::invalid();
    }
    coreContractionPeak = std::max(coreContractionPeak, liveCount);
    current = std::move(next);
    currentCount = nextCount;
  }
  Footprint coreLive;
  addAll(coreLive, allFactors);
  coreLive.add(rank, sizeof(Matrix));
  coreLive.add(coreContractionPeak, sizeof(Scalar));
  addShapeAndCoordinates(coreLive, rank, 5);
  takePeak(peak, coreLive);

  // Packaging Array<Matrix> copies the axis matrices while the source vector
  // is still alive. Include both matrix payloads and both handle arrays.
  Footprint packageLive;
  addProduct(packageLive, allFactors.elements(), 2, sizeof(Scalar));
  addProduct(packageLive, rank, 2, sizeof(Matrix));
  packageLive.add(coreCount, sizeof(Scalar));
  addProduct(packageLive, rank, 2, sizeof(Extent)); // core and output shapes
  takePeak(peak, packageLive);

  // The final factors remain owned while reconstructTucker materializes the
  // approximation and its contraction intermediates for the residual.
  const Footprint reconstruction = reconstructionPeak(extents, ranks, false);
  Footprint residualLive;
  addAll(residualLive, allFactors);
  residualLive.add(rank, sizeof(Matrix));
  residualLive.add(coreCount, sizeof(Scalar));
  addProduct(residualLive, rank, 2, sizeof(Extent)); // factor and core shapes
  addAll(residualLive, reconstruction);
  takePeak(peak, residualLive);

  if (rank == 0) {
    Footprint scalarLive;
    scalarLive.add(1, sizeof(Scalar)); // copied core
    scalarLive.add(1, sizeof(Scalar)); // scalar reconstruction
    takePeak(peak, scalarLive);
  }
  if (dataCount == 0) {
    Footprint emptyLive;
    addAll(emptyLive, allFactors);
    addProduct(emptyLive, rank, 2, sizeof(Matrix));
    addProduct(emptyLive, rank, 3, sizeof(Extent));
    emptyLive.add(rank, sizeof(Index));
    emptyLive.add(coreCount, sizeof(Scalar));
    takePeak(peak, emptyLive);
  }
  return peak;
}

Status validateRanks(const Tensor& input, std::span<const Extent> ranks) {
  if (ranks.size() != input.shape().rank()) {
    return Status::failure(ErrorCode::DimensionMismatch,
                           "Tucker rank count must match tensor rank");
  }
  const bool empty = input.size() == 0;
  for (Index axis = 0; axis < ranks.size(); ++axis) {
    const auto extent = input.shape().extent(axis);
    if (ranks[axis] > extent) {
      return Status::failure(ErrorCode::IndexOutOfRange,
                             "Tucker rank exceeds its mode extent");
    }
    if (!empty && ranks[axis] == 0) {
      return Status::failure(ErrorCode::InvalidDomain,
                             "Nonempty tensors require positive Tucker ranks");
    }
    if (empty && extent != 0 && ranks[axis] == 0) {
      return Status::failure(
          ErrorCode::InvalidDomain,
          "Zero Tucker rank is permitted only for an empty mode");
    }
  }
  return Status::success();
}

Status validateFinite(const Tensor& tensor) {
  for (const Scalar value : values(tensor)) {
    if (!std::isfinite(value)) {
      return Status::failure(ErrorCode::NonFiniteInput,
                             "Tucker input contains a non-finite value");
    }
  }
  return Status::success();
}

Index rowMajorOffset(std::span<const Extent> extents,
                     std::span<const Index> coordinates) {
  Index offset = 0;
  for (Index axis = 0; axis < extents.size(); ++axis) {
    offset = (offset * extents[axis]) + coordinates[axis];
  }
  return offset;
}

void decodeCoordinates(Index linear, std::span<const Extent> extents,
                       std::span<Index> coordinates) {
  for (Index axis = extents.size(); axis > 0; --axis) {
    const Index current = axis - 1;
    coordinates[current] = linear % extents[current];
    linear /= extents[current];
  }
}

Result<Tensor> expandFactors(const Tensor& core,
                             std::span<const Extent> outputExtents,
                             std::span<const Matrix> axes,
                             const Limits& limits) {
  if (outputExtents.empty()) {
    const auto shape = TensorShape::create(outputExtents);
    if (!shape.hasValue()) {
      return Result<Tensor>::failure(shape.status().code(),
                                     std::string{shape.status().message()});
    }
    return Tensor::create(shape.value(), values(core).front(),
                          TensorLimits{static_cast<Extent>(limits.elements)});
  }
  std::vector<Extent> currentExtents(core.shape().extents().begin(),
                                     core.shape().extents().end());
  std::optional<Tensor> current;
  const Tensor* source = &core;
  std::vector<Index> coordinates(outputExtents.size());
  for (Index axis = 0; axis < outputExtents.size(); ++axis) {
    const Extent modeRank = currentExtents[axis];
    if (outputExtents[axis] == 0) {
      currentExtents[axis] = 0;
      const auto shape = TensorShape::create(currentExtents);
      if (!shape.hasValue()) {
        return Result<Tensor>::failure(shape.status().code(),
                                       std::string{shape.status().message()});
      }
      auto next =
          Tensor::create(shape.value(), 0.0,
                         TensorLimits{static_cast<Extent>(limits.elements)});
      if (!next.hasValue()) {
        return next;
      }
      current.emplace(std::move(next).value());
      source = &*current;
      continue;
    }
    const auto sourceValues = values(*source);
    currentExtents[axis] = outputExtents[axis];
    const auto shape = TensorShape::create(currentExtents);
    if (!shape.hasValue()) {
      return Result<Tensor>::failure(shape.status().code(),
                                     std::string{shape.status().message()});
    }
    auto next = Tensor::create(
        shape.value(), 0.0, TensorLimits{static_cast<Extent>(limits.elements)});
    if (!next.hasValue()) {
      return next;
    }
    for (Index output = 0; output < next.value().size(); ++output) {
      decodeCoordinates(output, currentExtents, coordinates);
      const Index outputCoordinate = coordinates[axis];
      Wide sum = 0;
      for (Index component = 0; component < modeRank; ++component) {
        coordinates[axis] = component;
        sum = std::fma(
            static_cast<Wide>(axes[axis].at(outputCoordinate, component)),
            static_cast<Wide>(sourceValues[rowMajorOffset(
                source->shape().extents(), coordinates)]),
            sum);
      }
      auto converted = static_cast<Scalar>(sum);
      if (!std::isfinite(converted)) {
        return Result<Tensor>::failure(ErrorCode::NonFiniteResult,
                                       "Tucker reconstruction is non-finite");
      }
      values(next.value())[output] = converted;
      coordinates[axis] = outputCoordinate;
    }
    current.emplace(std::move(next).value());
    source = &*current;
  }
  if (!current.has_value()) {
    return Result<Tensor>::failure(ErrorCode::InvalidDomain,
                                   "Tucker reconstruction produced no data");
  }
  return Result<Tensor>::success(std::move(current.value()));
}

bool completeOrthonormalColumns(Matrix& factor, Index firstColumn) {
  const Index rows = factor.rows();
  const Index columns = factor.columns();
  std::vector<Wide> candidate(rows);
  const Wide threshold = 64 * std::numeric_limits<Scalar>::epsilon() *
                         std::sqrt(static_cast<Wide>(rows));
  Index filled = firstColumn;
  for (Index basis = 0; basis < rows && filled < columns; ++basis) {
    std::ranges::fill(candidate, Wide{0});
    candidate[basis] = 1;
    for (int pass = 0; pass < 2; ++pass) {
      for (Index column = 0; column < filled; ++column) {
        Wide projection = 0;
        for (Index row = 0; row < rows; ++row) {
          projection = std::fma(static_cast<Wide>(factor.at(row, column)),
                                candidate[row], projection);
        }
        for (Index row = 0; row < rows; ++row) {
          candidate[row] =
              std::fma(-projection, static_cast<Wide>(factor.at(row, column)),
                       candidate[row]);
        }
      }
    }
    Wide norm = 0;
    for (const auto value : candidate) {
      norm = std::hypot(norm, value);
    }
    if (norm <= threshold) {
      continue;
    }
    for (Index row = 0; row < rows; ++row) {
      factor.at(row, filled) = static_cast<Scalar>(candidate[row] / norm);
    }
    ++filled;
  }
  return filled == columns;
}

Scalar frobeniusDifference(const Tensor& first, const Tensor& second,
                           bool& representable) {
  const auto firstValues = values(first);
  const auto secondValues = values(second);
  Wide scale = 0;
  Wide sumSquares = 1;
  for (Extent i = 0; i < first.size(); ++i) {
    const Wide magnitude =
        std::abs(static_cast<Wide>(firstValues[i]) - secondValues[i]);
    if (magnitude == 0) {
      continue;
    }
    if (scale < magnitude) {
      const Wide ratio = scale / magnitude;
      sumSquares = 1 + (sumSquares * ratio * ratio);
      scale = magnitude;
    } else {
      const Wide ratio = magnitude / scale;
      sumSquares += ratio * ratio;
    }
  }
  const Wide norm = scale == 0 ? 0 : scale * std::sqrt(sumSquares);
  const auto result = static_cast<Scalar>(norm);
  representable = std::isfinite(result);
  return result;
}

} // namespace

Result<Tensor> reconstructTucker(const TuckerFactors& factors, Limits limits) {
  const auto extents = factors.shape.extents();
  const auto rank = extents.size();
  if (!std::isfinite(factors.residual_norm) || factors.residual_norm < 0) {
    return Result<Tensor>::failure(
        ErrorCode::InvalidDomain,
        "Tucker residual provenance must be finite and nonnegative");
  }
  if (factors.axes.view().size() != rank ||
      factors.core.shape().rank() != rank) {
    return Result<Tensor>::failure(
        ErrorCode::DimensionMismatch,
        "Tucker factor record has inconsistent rank metadata");
  }
  std::vector<Extent> ranks(rank);
  const auto axes = factors.axes.view();
  bool empty = false;
  for (Index axis = 0; axis < rank; ++axis) {
    const auto& matrix = axes[axis];
    const auto modeExtent = extents[axis];
    const auto modeRank = matrix.columns();
    if (matrix.rows() != modeExtent || modeRank > modeExtent ||
        factors.core.shape().extent(axis) != modeRank) {
      return Result<Tensor>::failure(
          ErrorCode::DimensionMismatch,
          "Tucker axis and core extents are inconsistent");
    }
    if (modeExtent == 0) {
      empty = true;
      if (modeRank != 0) {
        return Result<Tensor>::failure(
            ErrorCode::DimensionMismatch,
            "Empty tensor axis must have zero Tucker rank");
      }
    } else if (modeRank == 0) {
      return Result<Tensor>::failure(
          ErrorCode::DimensionMismatch,
          "Nonempty tensor axis has zero Tucker rank");
    }
    ranks[axis] = modeRank;
    for (Index row = 0; row < matrix.rows(); ++row) {
      for (Index column = 0; column < matrix.columns(); ++column) {
        if (!std::isfinite(matrix.at(row, column))) {
          return Result<Tensor>::failure(
              ErrorCode::NonFiniteInput,
              "Tucker factor contains a non-finite value");
        }
      }
    }
  }
  for (const Scalar value : values(factors.core)) {
    if (!std::isfinite(value)) {
      return Result<Tensor>::failure(ErrorCode::NonFiniteInput,
                                     "Tucker core contains a non-finite value");
    }
  }
  if (empty) {
    Footprint emptyPeak;
    emptyPeak.add(rank, sizeof(Index)); // validated per-mode ranks
    addProduct(emptyPeak, rank, 2, sizeof(Extent));
    if (!fits(emptyPeak, limits)) {
      return Result<Tensor>::failure(
          ErrorCode::ResourceLimit,
          "Empty Tucker output exceeds storage limits");
    }
    const auto shape = TensorShape::create(extents);
    if (!shape.hasValue()) {
      return Result<Tensor>::failure(shape.status().code(),
                                     std::string{shape.status().message()});
    }
    return Tensor::create(shape.value(), 0.0,
                          TensorLimits{static_cast<Extent>(limits.elements)});
  }
  const Footprint peak = reconstructionPeak(extents, ranks, false);
  if (!fits(peak, limits)) {
    return Result<Tensor>::failure(
        ErrorCode::ResourceLimit,
        "Tucker reconstruction exceeds storage limits");
  }
  return expandFactors(factors.core, extents, axes, limits);
}

Result<TuckerFactors> tuckerHOSVD(const Tensor& input,
                                  std::span<const Extent> ranks,
                                  SolvePolicy policy, Limits limits) {
  const Status rankStatus = validateRanks(input, ranks);
  if (!rankStatus.ok()) {
    return Result<TuckerFactors>::failure(rankStatus.code(),
                                          std::string{rankStatus.message()});
  }
  if (!validPolicy(policy)) {
    return Result<TuckerFactors>::failure(ErrorCode::InvalidTolerance,
                                          "Invalid Tucker SVD policy");
  }
  const Status finiteStatus = validateFinite(input);
  if (!finiteStatus.ok()) {
    return Result<TuckerFactors>::failure(finiteStatus.code(),
                                          std::string{finiteStatus.message()});
  }
  const auto extents = input.shape().extents();
  const auto rank = extents.size();
  const auto inputValues = values(input);
  const Footprint peak = hosvdPeak(extents, ranks);
  if (!fits(peak, limits)) {
    return Result<TuckerFactors>::failure(
        ErrorCode::ResourceLimit, "Tucker HOSVD exceeds storage limits");
  }

  std::vector<Matrix> axes;
  axes.reserve(rank);
  std::vector<Index> coordinates(rank);
  if (input.size() == 0) {
    for (Index mode = 0; mode < rank; ++mode) {
      Matrix axisFactor(extents[mode], ranks[mode]);
      for (Index column = 0; column < ranks[mode]; ++column) {
        axisFactor.at(column, column) = 1.0;
      }
      axes.push_back(std::move(axisFactor));
    }
  }
  for (Index mode = 0; mode < rank && input.size() != 0; ++mode) {
    const Index rows = extents[mode];
    const Index columns = input.size() / rows;
    Matrix unfolding(rows, columns);
    for (Index column = 0; column < columns; ++column) {
      Index remainder = column;
      for (Index axis = rank; axis > 0; --axis) {
        const Index coordinateAxis = axis - 1;
        if (coordinateAxis == mode) {
          continue;
        }
        coordinates[coordinateAxis] = remainder % extents[coordinateAxis];
        remainder /= extents[coordinateAxis];
      }
      for (Index row = 0; row < rows; ++row) {
        coordinates[mode] = row;
        unfolding.at(row, column) =
            inputValues[rowMajorOffset(extents, coordinates)];
      }
    }
    std::size_t priorFactors = 0;
    for (const auto& factor : axes) {
      priorFactors += factor.size();
    }
    const std::size_t newFactorSize = rows * ranks[mode];
    Footprint external;
    external.add(input.size(), sizeof(Scalar)); // unfolding
    external.add(priorFactors + newFactorSize, sizeof(Scalar));
    external.add(rank, sizeof(Matrix));
    external.add(rank, sizeof(Index));
    const Limits svdLimits{.elements = limits.elements - external.elements(),
                           .bytes = limits.bytes - external.bytes(),
                           .evaluations = limits.evaluations};
    Matrix axisFactor(rows, ranks[mode]);
    const auto decomposition =
        linear::factorSVD(unfolding.view(), policy, svdLimits);
    if (!decomposition.hasValue()) {
      return Result<TuckerFactors>::failure(
          decomposition.status().code(),
          std::string{decomposition.status().message()});
    }
    const Index available = decomposition.value().left.columns();
    const Index copied = std::min(ranks[mode], available);
    for (Index row = 0; row < rows; ++row) {
      for (Index column = 0; column < copied; ++column) {
        axisFactor.at(row, column) = decomposition.value().left.at(row, column);
      }
    }
    if (copied < ranks[mode] &&
        !completeOrthonormalColumns(axisFactor, copied)) {
      return Result<TuckerFactors>::failure(
          ErrorCode::NoConvergence,
          "Could not complete the requested Tucker mode basis");
    }
    axes.push_back(std::move(axisFactor));
  }

  std::vector<Extent> currentExtents(extents.begin(), extents.end());
  std::optional<Tensor> current;
  const Tensor* source = &input;
  for (Index mode = 0; mode < rank && input.size() != 0; ++mode) {
    std::vector<Extent> nextExtents = currentExtents;
    nextExtents[mode] = ranks[mode];
    const auto shape = TensorShape::create(nextExtents);
    if (!shape.hasValue()) {
      return Result<TuckerFactors>::failure(
          shape.status().code(), std::string{shape.status().message()});
    }
    auto next = Tensor::create(
        shape.value(), 0.0, TensorLimits{static_cast<Extent>(limits.elements)});
    if (!next.hasValue()) {
      return Result<TuckerFactors>::failure(
          next.status().code(), std::string{next.status().message()});
    }
    const auto previousValues = values(*source);
    auto nextValues = values(next.value());
    for (Index output = 0; output < nextValues.size(); ++output) {
      decodeCoordinates(output, nextExtents, coordinates);
      const Index outputCoordinate = coordinates[mode];
      Wide sum = 0;
      for (Index component = 0; component < extents[mode]; ++component) {
        coordinates[mode] = component;
        sum = std::fma(
            static_cast<Wide>(axes[mode].at(component, outputCoordinate)),
            static_cast<Wide>(
                previousValues[rowMajorOffset(currentExtents, coordinates)]),
            sum);
      }
      auto converted = static_cast<Scalar>(sum);
      if (!std::isfinite(converted)) {
        return Result<TuckerFactors>::failure(
            ErrorCode::NonFiniteResult, "Tucker core projection is non-finite");
      }
      nextValues[output] = converted;
      coordinates[mode] = outputCoordinate;
    }
    current.emplace(std::move(next).value());
    source = &*current;
    currentExtents = std::move(nextExtents);
  }
  if (rank == 0) {
    auto scalarCore =
        Tensor::create(input.shape(), inputValues[0],
                       TensorLimits{static_cast<Extent>(limits.elements)});
    if (!scalarCore.hasValue()) {
      return Result<TuckerFactors>::failure(
          scalarCore.status().code(),
          std::string{scalarCore.status().message()});
    }
    current.emplace(std::move(scalarCore).value());
  }
  if (input.size() == 0 && rank > 0) {
    const auto shape = TensorShape::create(ranks);
    if (!shape.hasValue()) {
      return Result<TuckerFactors>::failure(
          shape.status().code(), std::string{shape.status().message()});
    }
    auto emptyCore = Tensor::create(
        shape.value(), 0.0, TensorLimits{static_cast<Extent>(limits.elements)});
    if (!emptyCore.hasValue()) {
      return Result<TuckerFactors>::failure(
          emptyCore.status().code(), std::string{emptyCore.status().message()});
    }
    current.emplace(std::move(emptyCore).value());
  }
  if (!current.has_value()) {
    return Result<TuckerFactors>::failure(
        ErrorCode::InvalidDomain, "Tucker decomposition produced no core");
  }

  auto storedAxes = Array<Matrix>::copy(axes, limits);
  if (!storedAxes.hasValue()) {
    return Result<TuckerFactors>::failure(
        storedAxes.status().code(), std::string{storedAxes.status().message()});
  }
  auto originalShape = TensorShape::create(extents);
  if (!originalShape.hasValue()) {
    return Result<TuckerFactors>::failure(
        originalShape.status().code(),
        std::string{originalShape.status().message()});
  }
  TuckerFactors result{.core = std::move(current.value()),
                       .axes = std::move(storedAxes).value(),
                       .shape = std::move(originalShape).value(),
                       .residual_norm = 0};
  axes.clear();
  if (input.size() != 0) {
    const auto approximation = reconstructTucker(result, limits);
    if (!approximation.hasValue()) {
      return Result<TuckerFactors>::failure(
          approximation.status().code(),
          std::string{approximation.status().message()});
    }
    bool representable = false;
    result.residual_norm =
        frobeniusDifference(input, approximation.value(), representable);
    if (!representable) {
      return Result<TuckerFactors>::failure(ErrorCode::NonFiniteResult,
                                            "Tucker residual is non-finite");
    }
  }
  return Result<TuckerFactors>::success(std::move(result));
}

} // namespace vss::math
