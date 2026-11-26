#include "vss/math/Array.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Matrix.hpp"
#include "vss/math/MatrixView.hpp"
#include "vss/math/Result.hpp"
#include "vss/math/Tensor.hpp"
#include "vss/math/TensorLimits.hpp"
#include "vss/math/TensorShape.hpp"
#include "vss/math/covariance/PropagationResult.hpp"
#include "vss/math/statistics/Moments.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <span>
#include <statistics/ExactProductSum.hpp>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/AffineMaps.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math {
namespace {
template <class T> Result<T> fail(ErrorCode c, const char* m) {
  return Result<T>::failure(c, m);
}
bool product(Extent a, Extent b, Extent& out) {
  if ((a != 0U) && b > std::numeric_limits<Extent>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}
bool within(Extent n, Limits l) {
  return n <= l.elements && n <= l.bytes / sizeof(Scalar);
}

const statistics::MomentTerm* termFor(const MomentEstimate& e, Index order) {
  for (const auto& t : e.terms.view()) {
    if (t.order == order) {
      return &t;
    }
  }
  return nullptr;
}

bool sameSupport(const Support& a, const Support& b) {
  return a.input == b.input && a.valid == b.valid && a.positive == b.positive &&
         a.frequency_mass == b.frequency_mass &&
         a.weight_mass == b.weight_mass &&
         a.normalized_weight_powers == b.normalized_weight_powers &&
         a.effective_count == b.effective_count &&
         a.degrees_of_freedom == b.degrees_of_freedom &&
         a.squared_weight_mass == b.squared_weight_mass;
}

bool commonSupport(const MomentEstimate& source) {
  const auto& global = source.diagnostics.global;
  return std::ranges::all_of(source.diagnostics.coordinate_support.view(),
                             [&](const auto& support) {
                               return sameSupport(global, support);
                             }) &&
         std::ranges::all_of(source.diagnostics.tuple_support.view(),
                             [&](const auto& record) {
                               return sameSupport(global, record.support);
                             });
}

Result<Scalar> sourceValue(const statistics::MomentTerm& term,
                           std::span<const Index> tuple, Extent dimension) {
  const auto values = term.values.view();
  const Index order = term.order;
  if (order == 0U || order > 4U) {
    return fail<Scalar>(ErrorCode::InvalidDomain,
                        "moment order is outside the supported range");
  }
  if (tuple.size() != order) {
    return fail<Scalar>(ErrorCode::DimensionMismatch,
                        "moment tuple rank mismatch");
  }
  if (term.layout == statistics::MomentRepresentation::component) {
    if (!std::all_of(tuple.begin(), tuple.end(),
                     [&](Index i) { return i == tuple.front(); })) {
      return fail<Scalar>(ErrorCode::InvalidDomain,
                          "component-only source omits a mixed moment");
    }
    if (tuple.front() >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "component moment storage is inconsistent");
    }
    return Result<Scalar>::success(values.at(tuple.front()));
  }
  if (term.layout == statistics::MomentRepresentation::dense) {
    Extent index = 0;
    for (Index const axis : tuple) {
      if (axis >= dimension || !product(index, dimension, index) ||
          index > std::numeric_limits<Extent>::max() - axis) {
        return fail<Scalar>(ErrorCode::IndexOutOfRange,
                            "dense moment tuple is invalid");
      }
      index += axis;
    }
    if (index >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "dense moment storage is inconsistent");
    }
    return Result<Scalar>::success(values.at(index));
  }
  if (term.layout == statistics::MomentRepresentation::selected) {
    const auto flat = term.flattened_tuples.view();
    if (((order != 0U) &&
         values.size() > std::numeric_limits<Extent>::max() / order) ||
        flat.size() != values.size() * order) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "selected moment tuple metadata is inconsistent");
    }
    for (Extent row = 0; row < values.size(); ++row) {
      bool same = true;
      for (Index axis = 0; axis < order; ++axis) {
        const Index target = tuple[axis];
        Index matches = 0;
        for (Index candidate = 0; candidate < order; ++candidate) {
          if (flat[row * order + candidate] == target) {
            ++matches;
          }
        }
        Index expected = 0;
        for (Index const candidate : tuple) {
          if (candidate == target) {
            ++expected;
          }
        }
        if (matches != expected) {
          same = false;
          break;
        }
      }
      if (same) {
        return Result<Scalar>::success(values.at(row));
      }
    }
    return fail<Scalar>(
        ErrorCode::InvalidDomain,
        "required source moment is absent from sparse representation");
  }
  if (term.layout == statistics::MomentRepresentation::packed) {
    std::array<Index, 4U> sorted{};
    for (Index i = 0; i < order; ++i) {
      const Index value = tuple[i];
      Index position = i;
      while (position > 0U && sorted.at(position - 1U) > value) {
        sorted.at(position) = sorted.at(position - 1U);
        --position;
      }
      sorted.at(position) = value;
    }
    Extent rank = 0;
    Extent minimum = 0;
    for (Index axis = 0; axis < order; ++axis) {
      const Extent coordinate = sorted.at(axis);
      const Extent remaining = order - axis - 1U;
      if (coordinate >= dimension || coordinate < minimum) {
        return fail<Scalar>(ErrorCode::IndexOutOfRange,
                            "packed tuple coordinate is invalid");
      }
      // Lexicographic combinations-with-repetition rank.
      for (Extent candidate = minimum; candidate < coordinate; ++candidate) {
        Extent count = 1;
        for (Extent j = 1; j <= remaining; ++j) {
          if (count > std::numeric_limits<Extent>::max() /
                          (dimension - candidate + j - 1U)) {
            return fail<Scalar>(ErrorCode::SizeOverflow,
                                "packed tuple rank overflows");
          }
          count = count * (dimension - candidate + j - 1U) / j;
        }
        if (rank > std::numeric_limits<Extent>::max() - count) {
          return fail<Scalar>(ErrorCode::SizeOverflow,
                              "packed tuple rank overflows");
        }
        rank += count;
      }
      minimum = coordinate;
    }
    if (rank >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "packed moment storage is inconsistent");
    }
    return Result<Scalar>::success(values.at(rank));
  }
  return fail<Scalar>(ErrorCode::InvalidDomain,
                      "unsupported source moment layout");
}
} // namespace

namespace {
void decodeLinearIndex(Extent linear, const TensorShape& shape,
                       std::vector<Index>& indices) {
  for (Extent axis = shape.rank(); axis > 0U;) {
    --axis;
    const Extent extent = shape.extent(axis);
    indices.at(axis) = linear % extent;
    linear /= extent;
  }
}
} // namespace

Result<Tensor> applyAffine(const Tensor& value, ConstMatrixView linear,
                           ConstVectorView offset,
                           std::span<const Extent> outputShape, Limits limits) {
  if (linear.columns() != value.size() || linear.rows() != offset.size()) {
    return fail<Tensor>(ErrorCode::DimensionMismatch,
                        "affine map dimensions do not match input and offset");
  }
  Extent shaped = 1;
  if (outputShape.empty()) {
    shaped = 1;
  } else {
    for (Extent const e : outputShape) {
      if (e == 0U) {
        return fail<Tensor>(ErrorCode::InvalidExtent,
                            "output shape extents must be positive");
      }
      if (!product(shaped, e, shaped)) {
        return fail<Tensor>(ErrorCode::SizeOverflow,
                            "output shape product overflows");
      }
    }
  }
  if (shaped != linear.rows()) {
    return fail<Tensor>(ErrorCode::DimensionMismatch,
                        "output shape does not match affine row count");
  }
  if (!within(linear.rows(), limits)) {
    return fail<Tensor>(ErrorCode::ResourceLimit,
                        "affine output exceeds limits");
  }
  std::vector<Index> inputIndex(value.shape().rank());
  for (Extent i = 0; i < value.size(); ++i) {
    decodeLinearIndex(i, value.shape(), inputIndex);
    if (!isFinite(value.at(inputIndex))) {
      return fail<Tensor>(ErrorCode::NonFiniteInput,
                          "affine input is non-finite");
    }
  }
  for (Extent i = 0; i < linear.rows(); ++i) {
    if (!isFinite(offset.at(i))) {
      return fail<Tensor>(ErrorCode::NonFiniteInput,
                          "affine offset is non-finite");
    }
    for (Extent j = 0; j < linear.columns(); ++j) {
      if (!isFinite(linear.at(i, j))) {
        return fail<Tensor>(ErrorCode::NonFiniteInput,
                            "affine matrix is non-finite");
      }
    }
  }
  auto shape = TensorShape::create(outputShape);
  if (!shape) {
    return fail<Tensor>(shape.status().code(),
                        "affine output shape is invalid");
  }
  auto output =
      Tensor::create(shape.value(), 0.0, TensorLimits(limits.elements));
  if (!output) {
    return fail<Tensor>(output.status().code(),
                        "affine output allocation exceeds limits");
  }
  std::vector<Index> outputIndex(output.value().shape().rank());
  for (Extent i = 0; i < linear.rows(); ++i) {
    Scalar sum = offset.at(i);
    for (Extent j = 0; j < linear.columns(); ++j) {
      decodeLinearIndex(j, value.shape(), inputIndex);
      sum = std::fma(linear.at(i, j), value.at(inputIndex), sum);
    }
    if (!isFinite(sum)) {
      return fail<Tensor>(ErrorCode::NonFiniteResult,
                          "affine result is non-finite");
    }
    decodeLinearIndex(i, output.value().shape(), outputIndex);
    output.value().at(outputIndex) = sum;
  }
  return Result<Tensor>::success(std::move(output).value());
}

Result<MomentEstimate> transformMoments(const MomentEstimate& source,
                                        ConstMatrixView linear,
                                        ConstVectorView offset,
                                        const MomentRequest& request,
                                        Limits limits) {
  const Extent inputDimension =
      source.observation_shape.view().empty() ? 1U : [&] {
        Extent p = 1;
        for (Extent const e : source.observation_shape.view()) {
          if (!product(p, e, p)) {
            return Extent{0};
          }
        }
        return p;
      }();
  if (inputDimension == 0U || linear.columns() != inputDimension ||
      linear.rows() == 0U || offset.size() != linear.rows()) {
    return fail<MomentEstimate>(
        ErrorCode::DimensionMismatch,
        "affine moment map has incompatible dimensions");
  }
  const bool contract =
      request.representation == statistics::MomentRepresentation::contraction;
  if (request.orders == 0U || ((request.orders & 0xf0U) != 0U) ||
      (contract != request.contraction.has_value()) ||
      (contract && !request.selected.empty())) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "moment transform request is invalid");
  }
  const bool central =
      source.diagnostics.algorithm.find("central") != std::string::npos;
  const bool cumulant =
      source.diagnostics.algorithm.find("cumulant") != std::string::npos;
  if (!central && !cumulant) {
    return fail<MomentEstimate>(
        ErrorCode::InvalidDomain,
        "affine transform requires central moments or cumulants");
  }
  if (!commonSupport(source)) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "affine transformation requires common support "
                                "across contributing coordinates");
  }
  if (request.representation == statistics::MomentRepresentation::selected &&
      std::popcount(request.orders) != 1) {
    return fail<MomentEstimate>(
        ErrorCode::InvalidDomain,
        "selected layout requests one moment order at a time");
  }
  Extent outputDimension = linear.rows();
  if (contract) {
    const auto projection = request.contraction.value();
    if (projection.rows() == 0U || projection.columns() != linear.rows()) {
      return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                  "moment contraction has incompatible shape");
    }
    outputDimension = projection.rows();
  }
  Extent mapElements = 0U;
  if (!product(outputDimension, inputDimension, mapElements) ||
      mapElements > limits.elements ||
      outputDimension > limits.elements - mapElements ||
      mapElements > limits.bytes / sizeof(Scalar) ||
      outputDimension > limits.bytes / sizeof(Scalar) - mapElements) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "affine map and offset exceed limits");
  }
  for (Extent i = 0; i < linear.rows(); ++i) {
    if (!isFinite(offset.at(i))) {
      return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                  "affine offset is non-finite");
    }
    for (Extent j = 0; j < linear.columns(); ++j) {
      if (!isFinite(linear.at(i, j))) {
        return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                    "affine matrix is non-finite");
      }
    }
  }
  Matrix map(outputDimension, inputDimension, 0.0);
  Vector shift(outputDimension, 0.0);
  if (contract) {
    const auto projection = request.contraction.value();
    for (Extent i = 0; i < projection.rows(); ++i) {
      for (Extent k = 0; k < projection.columns(); ++k) {
        if (!isFinite(projection.at(i, k))) {
          return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                      "moment contraction is non-finite");
        }
        shift.at(i) = std::fma(projection.at(i, k), offset.at(k), shift.at(i));
        for (Extent j = 0; j < inputDimension; ++j) {
          map.at(i, j) =
              std::fma(projection.at(i, k), linear.at(k, j), map.at(i, j));
        }
      }
    }
  } else {
    for (Extent i = 0; i < outputDimension; ++i) {
      shift.at(i) = offset.at(i);
      for (Extent j = 0; j < inputDimension; ++j) {
        map.at(i, j) = linear.at(i, j);
      }
    }
  }
  for (Extent i = 0; i < outputDimension; ++i) {
    if (!isFinite(shift.at(i))) {
      return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                  "composed affine offset is non-finite");
    }
    for (Extent j = 0; j < inputDimension; ++j) {
      if (!isFinite(map.at(i, j))) {
        return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                    "composed affine matrix is non-finite");
      }
    }
  }
  const auto outputLayout = contract ? statistics::MomentRepresentation::dense
                                     : request.representation;
  if (!source.population_mean ||
      source.population_mean->size() != inputDimension) {
    return fail<MomentEstimate>(
        ErrorCode::InvalidDomain,
        "source population mean is required for affine transformation");
  }
  Vector mean(outputDimension);
  for (Extent i = 0; i < outputDimension; ++i) {
    Scalar s = shift.at(i);
    for (Extent j = 0; j < inputDimension; ++j) {
      if (!isFinite(source.population_mean->at(j))) {
        return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                    "source population mean is non-finite");
      }
      s = std::fma(map.at(i, j), source.population_mean->at(j), s);
    }
    if (!isFinite(s)) {
      return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                  "transformed mean is non-finite");
    }
    mean.at(i) = s;
  }
  std::vector<statistics::MomentTerm> terms;
  Extent total = outputDimension;
  for (Index order = 1; order <= 4; ++order) {
    if ((request.orders & (1U << (order - 1U))) != 0U) {
      Extent count = 0;
      if (outputLayout == statistics::MomentRepresentation::component) {
        count = outputDimension;
      } else if (outputLayout == statistics::MomentRepresentation::selected) {
        for (auto tuple : request.selected) {
          if (tuple.size() == order) {
            ++count;
          }
        }
        if (count == 0U) {
          return fail<MomentEstimate>(
              ErrorCode::InvalidDomain,
              "selected request has no tuples for requested order");
        }
      } else if (outputLayout == statistics::MomentRepresentation::packed) {
        count = 1;
        for (Index i = 1; i <= order; ++i) {
          if (count >
              std::numeric_limits<Extent>::max() / (outputDimension + i - 1U)) {
            return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                        "packed output count overflows");
          }
          count = count * (outputDimension + i - 1U) / i;
        }
      } else {
        count = 1;
        for (Index i = 0; i < order; ++i) {
          if (!product(count, outputDimension, count)) {
            return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                        "dense output count overflows");
          }
        }
      }
      const Extent fixedWorkspace = mapElements + outputDimension;
      if (fixedWorkspace > limits.elements ||
          total > limits.elements - fixedWorkspace ||
          count > limits.elements - fixedWorkspace - total ||
          fixedWorkspace > limits.bytes / sizeof(Scalar) ||
          total > limits.bytes / sizeof(Scalar) - fixedWorkspace ||
          count > limits.bytes / sizeof(Scalar) - fixedWorkspace - total) {
        return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                    "moment output exceeds limits");
      }
      total += count;
      const auto* sourceTerm = termFor(source, order);
      if (order > 1U && (sourceTerm == nullptr)) {
        return fail<MomentEstimate>(
            ErrorCode::InvalidDomain,
            "source is missing a requested moment order");
      }
      Extent work = count;
      for (Index k = 0; k < order; ++k) {
        if (!product(work, inputDimension, work)) {
          return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                      "moment transform work overflows");
        }
      }
      if (work > limits.elements) {
        return fail<MomentEstimate>(
            ErrorCode::ResourceLimit,
            "moment transform arithmetic work exceeds element budget");
      }
      std::vector<Scalar> vals;
      std::vector<Index> tuple(order, 0U);
      std::vector<Index> flat;
      vals.reserve(count);
      if (outputLayout == statistics::MomentRepresentation::selected) {
        flat.reserve(count * order);
      }
      auto emit = [&]() -> Status {
        if (outputLayout == statistics::MomentRepresentation::selected) {
          flat.insert(flat.end(), tuple.begin(), tuple.end());
        }
        if (order == 1U) {
          vals.push_back(cumulant ? mean.at(tuple.front()) : 0.0);
          return Status::success();
        }
        statistics::detail::ExactProductSum sum;
        Extent combinations = 1;
        for (Index k = 0; k < order; ++k) {
          combinations *= inputDimension;
        }
        std::array<Index, 4U> sourceTuple{};
        for (Extent linearIndex = 0; linearIndex < combinations;
             ++linearIndex) {
          Extent q = linearIndex;
          std::array<Scalar, 5U> factors{};
          for (Index axis = order; axis > 0; --axis) {
            const Index j = q % inputDimension;
            q /= inputDimension;
            sourceTuple.at(axis - 1U) = j;
            factors.at(axis) = map.at(tuple[axis - 1U], j);
          }
          if (std::ranges::any_of(
                  std::span<const Scalar>(factors).subspan(1U, order),
                  [](Scalar x) { return x == 0.0; })) {
            continue;
          }
          auto x = sourceValue(*sourceTerm,
                               std::span<const Index>(sourceTuple).first(order),
                               inputDimension);
          if (!x) {
            return Status::failure(
                ErrorCode::InvalidDomain,
                "source does not contain all moments required by affine map");
          }
          factors[0U] = x.value();
          if (!sum.add(std::span<const Scalar>(factors).first(order + 1U))) {
            return Status::failure(ErrorCode::NonFiniteResult,
                                   "moment product exceeds accumulator range");
          }
        }
        Scalar transformed = 0.0;
        if (!sum.rounded(transformed)) {
          return Status::failure(ErrorCode::NonFiniteResult,
                                 "transformed moment is non-finite");
        }
        vals.push_back(transformed);
        return Status::success();
      };
      if (request.representation ==
          statistics::MomentRepresentation::selected) {
        for (auto selected : request.selected) {
          if (selected.size() == order) {
            if (std::ranges::any_of(
                    selected, [&](Index i) { return i >= outputDimension; })) {
              return fail<MomentEstimate>(
                  ErrorCode::IndexOutOfRange,
                  "selected output tuple is out of range");
            }
            std::copy(selected.begin(), selected.end(), tuple.begin());
            auto status = emit();
            if (!status.ok()) {
              return fail<MomentEstimate>(status.code(),
                                          "selected affine moment failed");
            }
          }
        }
      } else {
        for (Extent position = 0; position < count; ++position) {
          auto status = emit();
          if (!status.ok()) {
            return fail<MomentEstimate>(status.code(),
                                        "affine moment evaluation failed");
          }
          bool advanced = false;
          for (Index axis = order; axis > 0;) {
            --axis;
            const Index minimum =
                outputLayout == statistics::MomentRepresentation::packed &&
                        axis != 0U
                    ? tuple[axis - 1U]
                    : 0U;
            if (tuple[axis] + 1U < outputDimension &&
                tuple[axis] + 1U >= minimum) {
              ++tuple[axis];
              for (Index j = axis + 1; j < order; ++j) {
                tuple[j] =
                    outputLayout == statistics::MomentRepresentation::packed
                        ? tuple[axis]
                        : 0U;
              }
              advanced = true;
              break;
            }
          }
          if (!advanced && position + 1 < count) {
            return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                        "moment tuple enumeration failed");
          }
        }
      }
      auto valueVector = Vector(vals);
      auto tupleArray = Array<Index>::copy(flat, limits);
      if (!tupleArray) {
        return fail<MomentEstimate>(tupleArray.status().code(),
                                    "moment tuple storage exceeds limits");
      }
      terms.push_back(statistics::MomentTerm{.order = order,
                                             .layout = outputLayout,
                                             .flattened_tuples =
                                                 std::move(tupleArray).value(),
                                             .values = std::move(valueVector)});
    }
  }
  auto termArray = Array<statistics::MomentTerm>::copy(terms, limits);
  if (!termArray) {
    return fail<MomentEstimate>(termArray.status().code(),
                                "moment result exceeds limits");
  }
  std::array<Extent, 1U> outputShape{outputDimension};
  auto shapeArray = Array<Extent>::copy(outputShape, limits);
  if (!shapeArray) {
    return fail<MomentEstimate>(shapeArray.status().code(),
                                "moment shape exceeds limits");
  }
  auto diagnostics = source.diagnostics;
  diagnostics.algorithm += "; affine map";
  diagnostics.coordinate_support = {};
  diagnostics.tuple_support = {};
  return Result<MomentEstimate>::success(
      MomentEstimate{.observation_shape = std::move(shapeArray).value(),
                     .terms = std::move(termArray).value(),
                     .diagnostics = std::move(diagnostics),
                     .population_mean = std::move(mean)});
}
} // namespace vss::math
