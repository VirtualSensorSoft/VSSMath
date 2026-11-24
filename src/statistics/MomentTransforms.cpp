#include "ExactProductSum.hpp"
#include "MomentInterval.hpp"

#include <algorithm>
#include <array>
#include <bit>
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
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {
namespace {

using detail::ExactProductSum;

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool checkedMultiply(Extent left, Extent right, Extent& result) noexcept {
  if (left != 0U && right > std::numeric_limits<Extent>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

bool checkedAdd(Extent left, Extent right, Extent& result) noexcept {
  if (right > std::numeric_limits<Extent>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
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

bool tupleRequested(const MomentRequest& request, Index order,
                    std::span<const Index> tuple) {
  if (request.representation != MomentRepresentation::selected) {
    return true;
  }
  return std::ranges::any_of(request.selected, [&](auto selected) {
    return selected.size() == order && std::ranges::equal(selected, tuple);
  });
}

Result<Extent> momentDimension(const MomentEstimate& source) {
  const auto shape = source.observation_shape.view();
  if (shape.empty()) {
    return Result<Extent>::success(1U);
  }
  Extent dimension = 1U;
  for (const Index extent : shape) {
    if (extent == 0U) {
      return Result<Extent>::success(0U);
    }
    if (!checkedMultiply(dimension, extent, dimension)) {
      return fail<Extent>(ErrorCode::InvalidExtent,
                          "moment estimate has an invalid observation shape");
    }
  }
  return Result<Extent>::success(dimension);
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
                          "packed moment tuple count overflows");
    }
  }
  return Result<Extent>::success(result);
}

Result<Extent> combinationsWithRepetition(Extent dimension, Index order) {
  if (dimension == 0U) {
    return Result<Extent>::success(0U);
  }
  Extent n = 0U;
  if (!checkedAdd(dimension, order - 1U, n)) {
    return fail<Extent>(ErrorCode::SizeOverflow,
                        "packed moment tuple count overflows");
  }
  return binomial(n, order);
}

Result<Extent> packedTupleIndex(std::span<const Index> tuple,
                                Extent dimension) {
  if (tuple.empty()) {
    return fail<Extent>(ErrorCode::InvalidExtent,
                        "packed moment tuple has zero order");
  }
  std::vector<Index> sorted(tuple.begin(), tuple.end());
  std::ranges::sort(sorted);
  Extent rank = 0U;
  Index minimum = 0U;
  for (Index axis = 0U; axis < sorted.size(); ++axis) {
    const Index coordinate = sorted[axis];
    if (coordinate >= dimension || coordinate < minimum) {
      return fail<Extent>(ErrorCode::IndexOutOfRange,
                          "packed moment tuple is outside its shape");
    }
    const Index remaining = sorted.size() - axis - 1U;
    Extent firstN = 0U;
    Extent secondN = 0U;
    if (!checkedAdd(dimension - minimum, remaining, firstN) ||
        !checkedAdd(dimension - coordinate, remaining, secondN)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "packed moment tuple rank overflows");
    }
    auto first = binomial(firstN, remaining + 1U);
    auto second = binomial(secondN, remaining + 1U);
    if (!first || !second || first.value() < second.value() ||
        !checkedAdd(rank, first.value() - second.value(), rank)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "packed moment tuple rank overflows");
    }
    minimum = coordinate;
  }
  return Result<Extent>::success(rank);
}

Result<Scalar> finiteProduct(std::span<const Scalar> factors) {
  Scalar fraction = 1.0;
  int exponent = 0;
  for (const Scalar factor : factors) {
    if (!std::isfinite(factor)) {
      return fail<Scalar>(ErrorCode::NonFiniteResult,
                          "contracted moment product is non-finite");
    }
    if (factor == 0.0) {
      return Result<Scalar>::success(0.0);
    }
    int factorExponent = 0;
    const Scalar factorFraction = std::frexp(factor, &factorExponent);
    int adjustment = 0;
    fraction = std::frexp(fraction * factorFraction, &adjustment);
    exponent += factorExponent + adjustment;
  }
  const Scalar result = std::scalbn(fraction, exponent);
  if (!std::isfinite(result)) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "contracted moment product is not representable");
  }
  return Result<Scalar>::success(result);
}

bool validRepresentation(MomentRepresentation representation) noexcept {
  return representation == MomentRepresentation::component ||
         representation == MomentRepresentation::selected ||
         representation == MomentRepresentation::packed ||
         representation == MomentRepresentation::dense ||
         representation == MomentRepresentation::contraction;
}

Result<std::vector<std::vector<Index>>>
enumerateTuples(Extent dimension, Index order, MomentRepresentation layout,
                std::span<const std::span<const Index>> selected,
                Limits limits) {
  Extent count = 0U;
  if (layout == MomentRepresentation::component) {
    count = dimension;
  } else if (layout == MomentRepresentation::selected) {
    count = selected.size();
  } else if (layout == MomentRepresentation::packed) {
    const auto combinations = combinationsWithRepetition(dimension, order);
    if (!combinations) {
      return fail<std::vector<std::vector<Index>>>(
          combinations.status().code(),
          "packed tuple count is not representable");
    }
    count = combinations.value();
  } else {
    count = 1U;
    for (Index i = 0U; i < order; ++i) {
      if (!checkedMultiply(count, dimension, count)) {
        return fail<std::vector<std::vector<Index>>>(
            ErrorCode::SizeOverflow, "dense tuple count overflows");
      }
    }
  }
  Extent tupleElements = 0U;
  Extent tupleHeaders = 0U;
  Extent tuplePayloadBytes = 0U;
  Extent tupleWorkspaceBytes = 0U;
  if (!checkedMultiply(count, order, tupleElements) ||
      count > limits.elements || tupleElements > limits.elements ||
      !checkedMultiply(count, sizeof(std::vector<Index>), tupleHeaders) ||
      !checkedMultiply(tupleElements, sizeof(Index), tuplePayloadBytes) ||
      !checkedAdd(tupleHeaders, tuplePayloadBytes, tupleWorkspaceBytes)) {
    return fail<std::vector<std::vector<Index>>>(
        ErrorCode::ResourceLimit, "moment tuples exceed configured limits");
  }
  if (tupleWorkspaceBytes > limits.bytes) {
    return fail<std::vector<std::vector<Index>>>(
        ErrorCode::ResourceLimit, "moment tuple workspace exceeds byte limits");
  }
  try {
    std::vector<std::vector<Index>> tuples;
    tuples.reserve(count);
    if (layout == MomentRepresentation::component) {
      for (Index component = 0U; component < dimension; ++component) {
        tuples.emplace_back(order, component);
      }
    } else if (layout == MomentRepresentation::selected) {
      for (const auto tuple : selected) {
        if (tuple.size() != order) {
          return fail<std::vector<std::vector<Index>>>(
              ErrorCode::DimensionMismatch,
              "selected tuple length does not match its order");
        }
        if (std::ranges::any_of(tuple, [dimension](Index value) {
              return value >= dimension;
            })) {
          return fail<std::vector<std::vector<Index>>>(
              ErrorCode::IndexOutOfRange,
              "selected tuple coordinate is outside the dimension");
        }
        tuples.emplace_back(tuple.begin(), tuple.end());
      }
    } else {
      std::vector<Index> tuple(order, 0U);
      const bool packed = layout == MomentRepresentation::packed;
      while (true) {
        tuples.push_back(tuple);
        bool advanced = false;
        for (Index axis = order; axis > 0U;) {
          --axis;
          if (tuple[axis] + 1U < dimension) {
            ++tuple[axis];
            const Index next = packed ? tuple[axis] : 0U;
            for (Index suffix = axis + 1U; suffix < order; ++suffix) {
              tuple[suffix] = next;
            }
            advanced = true;
            break;
          }
        }
        if (!advanced) {
          break;
        }
      }
    }
    return Result<std::vector<std::vector<Index>>>::success(std::move(tuples));
  } catch (const std::bad_alloc&) {
    return fail<std::vector<std::vector<Index>>>(
        ErrorCode::ResourceLimit, "moment tuple workspace allocation failed");
  }
}

const MomentTerm* findTerm(const MomentEstimate& source, Index order) {
  for (const auto& term : source.terms.view()) {
    if (term.order == order) {
      return &term;
    }
  }
  return nullptr;
}

Result<Scalar> termValue(const MomentEstimate& source, Index order,
                         std::span<const Index> tuple, Extent dimension) {
  const MomentTerm* term = findTerm(source, order);
  if (term == nullptr) {
    return fail<Scalar>(ErrorCode::InsufficientSamples,
                        "a required source moment order is absent");
  }
  const auto values = term->values.view();
  if (term->layout == MomentRepresentation::component) {
    if (tuple.size() != order || tuple.empty() ||
        !std::ranges::all_of(
            tuple, [&](Index value) { return value == tuple.front(); })) {
      return fail<Scalar>(ErrorCode::InsufficientSamples,
                          "component moments do not contain the requested "
                          "mixed tuple");
    }
    if (tuple.front() >= values.size()) {
      return fail<Scalar>(ErrorCode::IndexOutOfRange,
                          "component moment index is outside its values");
    }
    return Result<Scalar>::success(values.at(tuple.front()));
  }
  if (term->layout == MomentRepresentation::selected) {
    const auto flattened = term->flattened_tuples.view();
    if (flattened.size() != values.size() * order) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "selected moment tuple metadata is inconsistent");
    }
    for (Index position = 0U; position < values.size(); ++position) {
      const auto stored = flattened.subspan(position * order, order);
      if (std::ranges::equal(stored, tuple)) {
        return Result<Scalar>::success(values.at(position));
      }
    }
    return fail<Scalar>(ErrorCode::InsufficientSamples,
                        "requested source tuple was not stored");
  }
  if (term->layout == MomentRepresentation::dense) {
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
                          "dense moment values do not match their shape");
    }
    return Result<Scalar>::success(values.at(linear));
  }
  if (term->layout == MomentRepresentation::packed) {
    auto position = packedTupleIndex(tuple, dimension);
    if (!position) {
      return fail<Scalar>(position.status().code(),
                          "packed source tuple index is invalid");
    }
    if (position.value() >= values.size()) {
      return fail<Scalar>(ErrorCode::DimensionMismatch,
                          "packed moments do not match their dimension");
    }
    return Result<Scalar>::success(values.at(position.value()));
  }
  return fail<Scalar>(ErrorCode::InvalidDomain,
                      "source moment representation is unsupported");
}

Result<Support> supportForTuple(const MomentEstimate& source,
                                std::span<const Index> tuple,
                                Extent dimension) {
  Support recorded = source.diagnostics.global;
  bool hasRecorded = false;
  for (const auto& item : source.diagnostics.tuple_support.view()) {
    const auto coordinates = item.coordinates.view();
    if (coordinates.size() == tuple.size() &&
        std::ranges::equal(coordinates, tuple)) {
      if (hasRecorded && !supportEqual(recorded, item.support)) {
        return fail<Support>(ErrorCode::InvalidDomain,
                             "source tuple has conflicting support records");
      }
      if (!hasRecorded) {
        recorded = item.support;
        hasRecorded = true;
      }
    }
  }
  if (hasRecorded) {
    return Result<Support>::success(recorded);
  }
  const auto coordinateSupport = source.diagnostics.coordinate_support.view();
  if (coordinateSupport.empty()) {
    return Result<Support>::success(source.diagnostics.global);
  }
  if (coordinateSupport.size() != dimension) {
    return fail<Support>(ErrorCode::DimensionMismatch,
                         "source coordinate support has the wrong shape");
  }
  Support common = source.diagnostics.global;
  bool hasCommon = false;
  for (const Index coordinate : tuple) {
    if (coordinate >= dimension) {
      return fail<Support>(ErrorCode::IndexOutOfRange,
                           "support tuple coordinate is outside its shape");
    }
    const Support& current = coordinateSupport[coordinate];
    if (hasCommon && !supportEqual(common, current)) {
      return fail<Support>(ErrorCode::InvalidDomain,
                           "moment tuple combines different supports");
    }
    if (!hasCommon) {
      common = current;
      hasCommon = true;
    }
  }
  return Result<Support>::success(common);
}

Result<MomentEstimate> buildTransform(const MomentEstimate& source,
                                      Extent outputDimension,
                                      std::span<const Index> components,
                                      const MomentRequest& request,
                                      Limits limits) {
  const bool contract =
      request.representation == MomentRepresentation::contraction;
  const auto sourceDimension = momentDimension(source);
  if (!sourceDimension) {
    return fail<MomentEstimate>(sourceDimension.status().code(),
                                "source moment shape is invalid");
  }
  if (!source.diagnostics.coordinate_support.view().empty() &&
      source.diagnostics.coordinate_support.view().size() !=
          sourceDimension.value()) {
    return fail<MomentEstimate>(
        ErrorCode::DimensionMismatch,
        "source coordinate support has the wrong shape");
  }
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U ||
      !validRepresentation(request.representation)) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "moment transform request is invalid");
  }
  if (request.representation == MomentRepresentation::contraction) {
    if (!request.contraction.has_value()) {
      return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                  "contraction layout requires a map");
    }
    if (request.contraction->columns() != components.size()) {
      return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                  "contraction map input dimension is wrong");
    }
    outputDimension = request.contraction->rows();
    for (Index row = 0U; row < request.contraction->rows(); ++row) {
      for (Index column = 0U; column < request.contraction->columns();
           ++column) {
        if (!std::isfinite(request.contraction->at(row, column))) {
          return fail<MomentEstimate>(
              ErrorCode::NonFiniteInput,
              "contraction map contains a non-finite value");
        }
      }
    }
  } else if (request.contraction.has_value()) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "contraction map requires contraction layout");
  }
  if (request.representation == MomentRepresentation::selected &&
      std::popcount(request.orders) != 1) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "selected representation requests one order");
  }
  try {
    std::vector<MomentTerm> outputTerms;
    std::vector<TupleSupport> transformedTupleSupport;
    for (Index order = 1U; order <= 4U; ++order) {
      if ((request.orders & (1U << (order - 1U))) == 0U) {
        continue;
      }
      auto outputTuples =
          enumerateTuples(outputDimension, order, request.representation,
                          request.selected, limits);
      if (!outputTuples) {
        return fail<MomentEstimate>(outputTuples.status().code(),
                                    "output moment tuple generation failed");
      }
      std::optional<std::vector<std::vector<Index>>> inputTuples;
      if (contract) {
        auto generated = enumerateTuples(
            components.size(), order, MomentRepresentation::dense, {}, limits);
        if (!generated) {
          return fail<MomentEstimate>(generated.status().code(),
                                      "contraction input tuple count exceeds "
                                      "limits");
        }
        inputTuples.emplace(std::move(generated).value());
      }
      std::vector<Scalar> values;
      values.reserve(outputTuples.value().size());
      for (const auto& outputTuple : outputTuples.value()) {
        vss::math::detail::CompensatedSum sum;
        if (!contract) {
          std::vector<Index> sourceTuple;
          sourceTuple.reserve(order);
          for (const Index coordinate : outputTuple) {
            if (coordinate >= components.size()) {
              return fail<MomentEstimate>(ErrorCode::IndexOutOfRange,
                                          "requested output tuple is out of "
                                          "range");
            }
            sourceTuple.push_back(components[coordinate]);
          }
          auto value = termValue(source, order, sourceTuple,
                                 momentDimension(source).value());
          if (!value) {
            return fail<MomentEstimate>(
                value.status().code(),
                "required source moment tuple is absent");
          }
          values.push_back(value.value());
          continue;
        }
        const auto map = *request.contraction;
        std::optional<Support> outputSupport;
        for (const auto& inputTuple : *inputTuples) {
          std::array<Scalar, 4U> coefficientFactors{};
          bool zeroCoefficient = false;
          for (Index axis = 0U; axis < order; ++axis) {
            coefficientFactors.at(axis) =
                map.at(outputTuple[axis], inputTuple[axis]);
            zeroCoefficient =
                zeroCoefficient || coefficientFactors.at(axis) == 0.0;
          }
          if (zeroCoefficient) {
            continue;
          }
          std::vector<Index> sourceTuple;
          sourceTuple.reserve(order);
          for (const Index coordinate : inputTuple) {
            sourceTuple.push_back(components[coordinate]);
          }
          auto inputSupport =
              supportForTuple(source, sourceTuple, sourceDimension.value());
          if (!inputSupport) {
            return fail<MomentEstimate>(
                inputSupport.status().code(),
                "contraction input support is incompatible");
          }
          if (outputSupport.has_value() &&
              !supportEqual(*outputSupport, inputSupport.value())) {
            return fail<MomentEstimate>(
                ErrorCode::InvalidDomain,
                "contraction combines different support records");
          }
          outputSupport = inputSupport.value();
          auto value =
              termValue(source, order, sourceTuple, sourceDimension.value());
          if (!value) {
            return fail<MomentEstimate>(value.status().code(),
                                        "contraction requires an absent source "
                                        "moment tuple");
          }
          std::array<Scalar, 5U> factors{};
          std::copy_n(coefficientFactors.begin(), order, factors.begin());
          factors.at(order) = value.value();
          auto product = finiteProduct(
              std::span<const Scalar>{factors.data(), order + 1U});
          if (!product) {
            return fail<MomentEstimate>(
                product.status().code(),
                "contracted moment product is not finite");
          }
          sum.add(product.value());
        }
        if (!std::isfinite(sum.total())) {
          return fail<MomentEstimate>(ErrorCode::NonFiniteResult,
                                      "contracted moment is not finite");
        }
        if (outputSupport.has_value() &&
            !supportEqual(*outputSupport, source.diagnostics.global)) {
          auto coordinates = Array<Index>::copy(outputTuple, limits);
          if (!coordinates) {
            return fail<MomentEstimate>(
                coordinates.status().code(),
                "contracted tuple support exceeds limits");
          }
          transformedTupleSupport.push_back(
              TupleSupport{.coordinates = std::move(coordinates).value(),
                           .support = *outputSupport});
        }
        values.push_back(sum.total());
      }
      Array<Index> flattened;
      const MomentRepresentation layout =
          contract ? MomentRepresentation::dense : request.representation;
      if (request.representation == MomentRepresentation::selected &&
          !contract) {
        std::vector<Index> indices;
        for (const auto& tuple : outputTuples.value()) {
          indices.insert(indices.end(), tuple.begin(), tuple.end());
        }
        auto copied = Array<Index>::copy(indices, limits);
        if (!copied) {
          return fail<MomentEstimate>(copied.status().code(),
                                      "selected tuple metadata exceeds limits");
        }
        flattened = std::move(copied).value();
      }
      auto ownedValues = Vector(std::span<const Scalar>{values});
      outputTerms.push_back(MomentTerm{.order = order,
                                       .layout = layout,
                                       .flattened_tuples = std::move(flattened),
                                       .values = std::move(ownedValues)});
    }
    auto terms = Array<MomentTerm>::copy(outputTerms, limits);
    if (!terms) {
      return fail<MomentEstimate>(terms.status().code(),
                                  "transformed moment output exceeds limits");
    }
    const std::array<Index, 1U> shapeValues{outputDimension};
    auto shape = Array<Index>::copy(shapeValues, limits);
    if (!shape) {
      return fail<MomentEstimate>(shape.status().code(),
                                  "transformed moment shape exceeds limits");
    }
    EstimatorDiagnostics diagnostics = source.diagnostics;
    diagnostics.algorithm =
        (contract ? "contracted_" : "marginal_") + source.diagnostics.algorithm;
    diagnostics.premises += "; support and estimand preserved by coordinate "
                            "selection/linear map";
    if (!contract && !source.diagnostics.coordinate_support.view().empty()) {
      std::vector<Support> supports;
      supports.reserve(components.size());
      for (const Index component : components) {
        supports.push_back(
            source.diagnostics.coordinate_support.view()[component]);
      }
      auto copied = Array<Support>::copy(supports, limits);
      if (!copied) {
        return fail<MomentEstimate>(copied.status().code(),
                                    "coordinate support exceeds limits");
      }
      diagnostics.coordinate_support = std::move(copied).value();
    } else if (contract) {
      if (outputDimension > limits.elements) {
        return fail<MomentEstimate>(
            ErrorCode::ResourceLimit,
            "contracted coordinate support exceeds limits");
      }
      std::vector<Support> supportsForOutput;
      supportsForOutput.reserve(outputDimension);
      for (Index row = 0U; row < outputDimension; ++row) {
        std::optional<Support> outputSupport;
        for (Index column = 0U; column < components.size(); ++column) {
          if (request.contraction->at(row, column) == 0.0) {
            continue;
          }
          const std::array<Index, 1U> inputCoordinate{components[column]};
          auto inputSupport =
              supportForTuple(source, inputCoordinate, sourceDimension.value());
          if (!inputSupport) {
            return fail<MomentEstimate>(
                inputSupport.status().code(),
                "contracted coordinate support is incompatible");
          }
          if (outputSupport.has_value() &&
              !supportEqual(*outputSupport, inputSupport.value())) {
            return fail<MomentEstimate>(
                ErrorCode::InvalidDomain,
                "contracted coordinate combines different supports");
          }
          outputSupport = inputSupport.value();
        }
        supportsForOutput.push_back(
            outputSupport.value_or(source.diagnostics.global));
      }
      auto copied = Array<Support>::copy(supportsForOutput, limits);
      if (!copied) {
        return fail<MomentEstimate>(
            copied.status().code(),
            "contracted coordinate support exceeds limits");
      }
      diagnostics.coordinate_support = std::move(copied).value();
      auto tupleCopied =
          Array<TupleSupport>::copy(transformedTupleSupport, limits);
      if (!tupleCopied) {
        return fail<MomentEstimate>(tupleCopied.status().code(),
                                    "contracted tuple support exceeds limits");
      }
      diagnostics.tuple_support = std::move(tupleCopied).value();
    }

    if (!contract) {
      std::vector<TupleSupport> remappedSupports;
      const auto sourceSupports = source.diagnostics.tuple_support.view();
      for (const auto& item : sourceSupports) {
        const auto oldTuple = item.coordinates.view();
        const Index order = oldTuple.size();
        if (order == 0U || order > 4U ||
            (request.orders & (1U << (order - 1U))) == 0U) {
          continue;
        }
        std::vector<Index> newTuple;
        newTuple.reserve(order);
        bool fullySelected = true;
        for (const Index oldCoordinate : oldTuple) {
          if (oldCoordinate >= sourceDimension.value()) {
            return fail<MomentEstimate>(
                ErrorCode::DimensionMismatch,
                "source tuple support coordinate is outside its shape");
          }
          const auto position = std::ranges::find(components, oldCoordinate);
          if (position == components.end()) {
            fullySelected = false;
            break;
          }
          newTuple.push_back(static_cast<Index>(position - components.begin()));
        }
        if (!fullySelected || !tupleRequested(request, order, newTuple)) {
          continue;
        }
        auto coordinates = Array<Index>::copy(newTuple, limits);
        if (!coordinates) {
          return fail<MomentEstimate>(coordinates.status().code(),
                                      "remapped tuple support exceeds limits");
        }
        remappedSupports.push_back(
            TupleSupport{.coordinates = std::move(coordinates).value(),
                         .support = item.support});
      }
      auto copied = Array<TupleSupport>::copy(remappedSupports, limits);
      if (!copied) {
        return fail<MomentEstimate>(copied.status().code(),
                                    "remapped tuple support exceeds limits");
      }
      diagnostics.tuple_support = std::move(copied).value();
    }
    auto mean = source.population_mean;
    if (mean.has_value() && !contract) {
      std::vector<Scalar> selectedMean;
      selectedMean.reserve(components.size());
      for (const Index component : components) {
        if (component >= mean->size()) {
          return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                      "population mean does not match source "
                                      "dimension");
        }
        selectedMean.push_back(mean->at(component));
      }
      mean.emplace(std::span<const Scalar>{selectedMean});
    } else if (contract && mean.has_value()) {
      if (mean->size() != sourceDimension.value()) {
        return fail<MomentEstimate>(
            ErrorCode::DimensionMismatch,
            "population mean does not match contraction input");
      }
      std::vector<Scalar> transformedMean(outputDimension, 0.0);
      for (Index row = 0U; row < outputDimension; ++row) {
        vss::math::detail::CompensatedSum sum;
        for (Index column = 0U; column < components.size(); ++column) {
          sum.add(request.contraction->at(row, column) *
                  mean->at(components[column]));
        }
        if (!std::isfinite(sum.total())) {
          return fail<MomentEstimate>(
              ErrorCode::NonFiniteResult,
              "contracted population mean is not finite");
        }
        transformedMean[row] = sum.total();
      }
      mean.emplace(std::span<const Scalar>{transformedMean});
    }
    return Result<MomentEstimate>::success(
        MomentEstimate{.observation_shape = std::move(shape).value(),
                       .terms = std::move(terms).value(),
                       .diagnostics = std::move(diagnostics),
                       .population_mean = std::move(mean)});
  } catch (const std::bad_alloc&) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "moment transformation allocation failed");
  }
}

Result<Scalar> rawToCentral(const MomentEstimate& source,
                            std::span<const Index> tuple, Extent dimension) {
  const Index order = tuple.size();
  const auto referenceSupport = supportForTuple(source, tuple, dimension);
  if (!referenceSupport) {
    return fail<Scalar>(referenceSupport.status().code(),
                        "raw-to-central conversion has invalid support");
  }
  const auto requireCompatibleSupport =
      [&](std::span<const Index> lowerTuple) -> Result<bool> {
    const auto lowerSupport = supportForTuple(source, lowerTuple, dimension);
    if (!lowerSupport) {
      return fail<bool>(lowerSupport.status().code(),
                        "raw-to-central conversion has invalid lower support");
    }
    return Result<bool>::success(
        supportEqual(referenceSupport.value(), lowerSupport.value()));
  };
  if (order == 1U) {
    return Result<Scalar>::success(0.0);
  }
  const auto meanFor = [&](Index coordinate) -> Result<Scalar> {
    if (source.population_mean.has_value()) {
      if (coordinate >= source.population_mean->size()) {
        return fail<Scalar>(ErrorCode::DimensionMismatch,
                            "population mean shape differs from moments");
      }
      return Result<Scalar>::success(source.population_mean->at(coordinate));
    }
    const std::array<Index, 1U> index{coordinate};
    return termValue(source, 1U, index, dimension);
  };
  auto raw = termValue(source, order, tuple, dimension);
  if (!raw) {
    return raw;
  }
  long double central = raw.value();
  if (order == 2U) {
    for (const Index coordinate : tuple) {
      const std::array<Index, 1U> singleton{coordinate};
      const auto compatible = requireCompatibleSupport(singleton);
      if (!compatible) {
        return fail<Scalar>(
            compatible.status().code(),
            "raw-to-central conversion has invalid mean support");
      }
      if (!compatible.value()) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "raw-to-central conversion mixes incompatible supports");
      }
    }
    auto left = meanFor(tuple[0]);
    auto right = meanFor(tuple[1]);
    if (!left || !right) {
      return fail<Scalar>(ErrorCode::InsufficientSamples,
                          "raw-to-central conversion requires the mean");
    }
    central -= static_cast<long double>(left.value()) * right.value();
  } else if (order == 3U) {
    std::array<Scalar, 3U> means{};
    std::array<Scalar, 3U> second{};
    for (Index i = 0U; i < 3U; ++i) {
      const std::array<Index, 1U> singleton{tuple[i]};
      const auto compatible = requireCompatibleSupport(singleton);
      if (!compatible) {
        return fail<Scalar>(
            compatible.status().code(),
            "raw-to-central conversion has invalid mean support");
      }
      if (!compatible.value()) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "raw-to-central conversion mixes incompatible supports");
      }
      auto m = meanFor(tuple[i]);
      if (!m) {
        return fail<Scalar>(ErrorCode::InsufficientSamples,
                            "raw-to-central conversion requires the mean");
      }
      means.at(i) = m.value();
    }
    for (Index omitted = 0U; omitted < 3U; ++omitted) {
      const std::array<Index, 2U> pair{tuple[(omitted + 1U) % 3U],
                                       tuple[(omitted + 2U) % 3U]};
      const auto compatible = requireCompatibleSupport(pair);
      if (!compatible) {
        return fail<Scalar>(
            compatible.status().code(),
            "raw-to-central conversion has invalid pair support");
      }
      if (!compatible.value()) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "raw-to-central conversion mixes incompatible supports");
      }
      auto value = termValue(source, 2U, pair, dimension);
      if (!value) {
        return fail<Scalar>(ErrorCode::InsufficientSamples,
                            "raw-to-central conversion requires all lower "
                            "orders");
      }
      second.at(omitted) = value.value();
      central -=
          static_cast<long double>(means.at(omitted)) * second.at(omitted);
    }
    central += 2.0L * means[0] * means[1] * means[2];
  } else if (order == 4U) {
    std::array<Scalar, 4U> means{};
    for (Index i = 0U; i < 4U; ++i) {
      const std::array<Index, 1U> singleton{tuple[i]};
      const auto compatible = requireCompatibleSupport(singleton);
      if (!compatible) {
        return fail<Scalar>(
            compatible.status().code(),
            "raw-to-central conversion has invalid mean support");
      }
      if (!compatible.value()) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "raw-to-central conversion mixes incompatible supports");
      }
      auto m = meanFor(tuple[i]);
      if (!m) {
        return fail<Scalar>(ErrorCode::InsufficientSamples,
                            "raw-to-central conversion requires the mean");
      }
      means.at(i) = m.value();
    }
    for (Index omitted = 0U; omitted < 4U; ++omitted) {
      std::array<Index, 3U> triple{};
      Index position = 0U;
      for (Index i = 0U; i < 4U; ++i) {
        if (i != omitted) {
          triple.at(position++) = tuple[i];
        }
      }
      const auto compatible = requireCompatibleSupport(triple);
      if (!compatible) {
        return fail<Scalar>(
            compatible.status().code(),
            "raw-to-central conversion has invalid triple support");
      }
      if (!compatible.value()) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "raw-to-central conversion mixes incompatible supports");
      }
      auto value = termValue(source, 3U, triple, dimension);
      if (!value) {
        return fail<Scalar>(ErrorCode::InsufficientSamples,
                            "raw-to-central conversion requires all lower "
                            "orders");
      }
      central -= static_cast<long double>(means.at(omitted)) * value.value();
    }
    constexpr std::array<std::array<Index, 2U>, 6U> pairs{{{{0U, 1U}},
                                                           {{0U, 2U}},
                                                           {{0U, 3U}},
                                                           {{1U, 2U}},
                                                           {{1U, 3U}},
                                                           {{2U, 3U}}}};
    for (const auto pair : pairs) {
      std::array<Index, 2U> remaining{};
      Index position = 0U;
      for (Index i = 0U; i < 4U; ++i) {
        if (i != pair.at(0U) && i != pair.at(1U)) {
          remaining.at(position++) = tuple[i];
        }
      }
      const auto compatible = requireCompatibleSupport(remaining);
      if (!compatible) {
        return fail<Scalar>(
            compatible.status().code(),
            "raw-to-central conversion has invalid pair support");
      }
      if (!compatible.value()) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "raw-to-central conversion mixes incompatible supports");
      }
      auto value = termValue(source, 2U, remaining, dimension);
      if (!value) {
        return fail<Scalar>(ErrorCode::InsufficientSamples,
                            "raw-to-central conversion requires all lower "
                            "orders");
      }
      central += static_cast<long double>(means.at(pair.at(0U))) *
                 means.at(pair.at(1U)) * value.value();
    }
    central -= 3.0L * means[0] * means[1] * means[2] * means[3];
  }
  if (!std::isfinite(central) ||
      std::abs(central) >
          static_cast<long double>(std::numeric_limits<Scalar>::max())) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "central moment conversion is not representable");
  }
  return Result<Scalar>::success(static_cast<Scalar>(central));
}

Result<Scalar> scaledStandardize(Scalar moment,
                                 std::span<const Scalar> deviations) {
  if (moment == 0.0) {
    return Result<Scalar>::success(0.0);
  }
  int exponent = 0;
  Scalar fraction = std::frexp(moment, &exponent);
  for (const Scalar deviation : deviations) {
    if (!(deviation > 0.0) || !std::isfinite(deviation)) {
      return fail<Scalar>(ErrorCode::InvalidDomain,
                          "standardization requires positive finite scales");
    }
    int scaleExponent = 0;
    const Scalar scaleFraction = std::frexp(deviation, &scaleExponent);
    int adjustment = 0;
    fraction = std::frexp(fraction / scaleFraction, &adjustment);
    exponent += adjustment - scaleExponent;
  }
  const Scalar result = std::scalbn(fraction, exponent);
  if (!std::isfinite(result)) {
    return fail<Scalar>(ErrorCode::NonFiniteResult,
                        "standardized moment is not representable");
  }
  return Result<Scalar>::success(result);
}

struct ProjectedMoment final {
  detail::BinaryEndpoint value;
  Support support;
};

Result<Extent> projectedTupleCount(Extent dimension, Index order) {
  Extent count = 1U;
  for (Index axis = 0U; axis < order; ++axis) {
    if (!checkedMultiply(count, dimension, count)) {
      return fail<Extent>(ErrorCode::SizeOverflow,
                          "projected dense tuple count overflows");
    }
  }
  return Result<Extent>::success(count);
}

void decodeTuple(Extent ordinal, Extent dimension, Index order,
                 std::array<Index, 4U>& tuple) {
  for (Index axis = order; axis > 0U;) {
    --axis;
    tuple.at(axis) = ordinal % dimension;
    ordinal /= dimension;
  }
}

Result<ProjectedMoment>
projectedMoment(const MomentEstimate& source, ConstMatrixView map,
                Extent inputDimension, Index outputOrder,
                std::span<const Index> outputTuple, Limits limits) {
  const auto sourceTupleCount =
      projectedTupleCount(inputDimension, outputOrder);
  if (!sourceTupleCount) {
    return fail<ProjectedMoment>(sourceTupleCount.status().code(),
                                 "projected source tuple count is invalid");
  }
  if (sourceTupleCount.value() > limits.elements) {
    return fail<ProjectedMoment>(
        ErrorCode::ResourceLimit,
        "projected dependencies exceed element limits");
  }
  ExactProductSum sum;
  std::optional<Support> commonSupport;
  for (Extent ordinal = 0U; ordinal < sourceTupleCount.value(); ++ordinal) {
    std::array<Index, 4U> sourceTuple{};
    decodeTuple(ordinal, inputDimension, outputOrder, sourceTuple);
    std::array<Scalar, 4U> coefficients{};
    bool structuralZero = false;
    for (Index axis = 0U; axis < outputOrder; ++axis) {
      coefficients.at(axis) =
          map.at(outputTuple.subspan(axis).front(), sourceTuple.at(axis));
      structuralZero = structuralZero || coefficients.at(axis) == 0.0;
    }
    if (structuralZero) {
      continue;
    }
    auto sourceValue = termValue(
        source, outputOrder,
        std::span<const Index>(sourceTuple).first(outputOrder), inputDimension);
    if (!sourceValue) {
      return fail<ProjectedMoment>(
          ErrorCode::InvalidDomain,
          "projected standardized moment has a missing source term");
    }
    auto support = supportForTuple(
        source, std::span<const Index>(sourceTuple).first(outputOrder),
        inputDimension);
    if (!support) {
      return fail<ProjectedMoment>(ErrorCode::InvalidDomain,
                                   "projected source supports are invalid");
    }
    if (commonSupport.has_value() &&
        !supportEqual(*commonSupport, support.value())) {
      return fail<ProjectedMoment>(
          ErrorCode::InvalidDomain,
          "projected standardized terms combine incompatible supports");
    }
    commonSupport = support.value();
    std::array<Scalar, 5U> factors{};
    factors[0U] = sourceValue.value();
    for (Index axis = 0U; axis < outputOrder; ++axis) {
      factors.at(axis + 1U) = coefficients.at(axis);
    }
    if (!sum.add(std::span<const Scalar>(factors).first(outputOrder + 1U))) {
      return fail<ProjectedMoment>(ErrorCode::NonFiniteResult,
                                   "projected moment exceeds exact range");
    }
  }
  if (!commonSupport.has_value()) {
    return fail<ProjectedMoment>(ErrorCode::InvalidDomain,
                                 "projected moment has no contributing terms");
  }
  return Result<ProjectedMoment>::success(
      ProjectedMoment{.value = sum.endpoint(), .support = *commonSupport});
}

Result<MomentEstimate> standardizedProjection(const MomentEstimate& source,
                                              const MomentRequest& request,
                                              Extent inputDimension,
                                              Limits limits) {
  if (source.diagnostics.algorithm.find("central") == std::string::npos ||
      source.diagnostics.center == CenterRule::known ||
      !request.contraction.has_value() || !request.selected.empty() ||
      request.orders == 0U || (request.orders & 0xf0U) != 0U ||
      request.representation != MomentRepresentation::contraction) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "projected standardization request is invalid");
  }
  const ConstMatrixView map = request.contraction.value();
  if (map.rows() == 0U || map.columns() != inputDimension) {
    return fail<MomentEstimate>(ErrorCode::DimensionMismatch,
                                "standardization map has incompatible shape");
  }
  for (Index row = 0U; row < map.rows(); ++row) {
    for (Index column = 0U; column < map.columns(); ++column) {
      if (!std::isfinite(map.at(row, column))) {
        return fail<MomentEstimate>(ErrorCode::NonFiniteInput,
                                    "standardization map is not finite");
      }
    }
  }

  Extent outputValueCount = 0U;
  for (Index order = 1U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    auto count = projectedTupleCount(map.rows(), order);
    if (!count ||
        !checkedAdd(outputValueCount, count.value(), outputValueCount)) {
      return fail<MomentEstimate>(ErrorCode::SizeOverflow,
                                  "projected output tuple count overflows");
    }
    if (count.value() > limits.elements) {
      return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                  "projected moments exceed element limits");
    }
  }
  if (map.rows() > limits.elements || outputValueCount > limits.elements) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "projected result exceeds element limits");
  }
  Extent dependencyWork = 0U;
  auto varianceDependencies = projectedTupleCount(inputDimension, 2U);
  Extent allVarianceDependencies = 0U;
  if (!varianceDependencies ||
      !checkedMultiply(map.rows(), varianceDependencies.value(),
                       allVarianceDependencies) ||
      allVarianceDependencies > limits.elements) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "projected variance work exceeds limits");
  }
  dependencyWork = allVarianceDependencies;
  for (Index order = 2U; order <= 4U; ++order) {
    if ((request.orders & (1U << (order - 1U))) == 0U) {
      continue;
    }
    auto sourceCount = projectedTupleCount(inputDimension, order);
    auto outputCount = projectedTupleCount(map.rows(), order);
    Extent orderWork = 0U;
    if (!sourceCount || !outputCount ||
        !checkedMultiply(sourceCount.value(), outputCount.value(), orderWork) ||
        !checkedAdd(dependencyWork, orderWork, dependencyWork) ||
        dependencyWork > limits.elements) {
      return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                  "projected dependency work exceeds limits");
    }
  }
  Extent valueBytes = 0U;
  Extent meanBytes = 0U;
  Extent warningBytes = 0U;
  Extent premiseBytes = 0U;
  Extent termBytes = 0U;
  const auto outputTermCount =
      static_cast<Extent>(std::popcount(request.orders));
  Extent peakBytes = 0U;
  if (!checkedMultiply(outputValueCount, 2U * sizeof(Scalar), valueBytes) ||
      !checkedMultiply(map.rows(),
                       (2U * sizeof(Scalar)) +
                           (2U * sizeof(detail::BinaryEndpoint)) +
                           (2U * sizeof(Support)),
                       meanBytes) ||
      !checkedMultiply(source.diagnostics.warnings.view().size(),
                       sizeof(ErrorCode), warningBytes) ||
      !checkedAdd(source.diagnostics.premises.size(), 512U, premiseBytes) ||
      !checkedMultiply(outputTermCount, 2U * sizeof(MomentTerm), termBytes) ||
      !checkedAdd(valueBytes, meanBytes, peakBytes) ||
      !checkedAdd(peakBytes, termBytes, peakBytes) ||
      !checkedAdd(peakBytes, warningBytes, peakBytes) ||
      !checkedAdd(peakBytes, premiseBytes, peakBytes) ||
      !checkedAdd(peakBytes,
                  (2U * sizeof(Index)) + sizeof(ExactProductSum) + 512U,
                  peakBytes) ||
      peakBytes > limits.bytes) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "projected workspace exceeds byte limits");
  }

  try {
    std::vector<detail::BinaryEndpoint> variances;
    std::vector<Support> varianceSupports;
    variances.reserve(map.rows());
    varianceSupports.reserve(map.rows());
    for (Index coordinate = 0U; coordinate < map.rows(); ++coordinate) {
      const std::array<Index, 2U> diagonal{coordinate, coordinate};
      auto variance =
          projectedMoment(source, map, inputDimension, 2U, diagonal, limits);
      if (!variance) {
        return fail<MomentEstimate>(variance.status().code(),
                                    "projected variance is unavailable");
      }
      if (!(variance.value().value.fraction() > 0.0)) {
        return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                    "projected variance must be positive");
      }
      variances.push_back(variance.value().value);
      varianceSupports.push_back(variance.value().support);
    }

    std::vector<MomentTerm> outputTerms;
    outputTerms.reserve(outputTermCount);
    for (Index order = 1U; order <= 4U; ++order) {
      if ((request.orders & (1U << (order - 1U))) == 0U) {
        continue;
      }
      auto count = projectedTupleCount(map.rows(), order);
      if (!count) {
        return fail<MomentEstimate>(count.status().code(),
                                    "projected output tuple count is invalid");
      }
      std::vector<Scalar> values;
      values.reserve(count.value());
      for (Extent ordinal = 0U; ordinal < count.value(); ++ordinal) {
        std::array<Index, 4U> tuple{};
        decodeTuple(ordinal, map.rows(), order, tuple);
        Scalar value = 0.0;
        if (order > 1U) {
          auto moment = projectedMoment(
              source, map, inputDimension, order,
              std::span<const Index>(tuple).first(order), limits);
          if (!moment) {
            return fail<MomentEstimate>(moment.status().code(),
                                        "projected moment is unavailable");
          }
          if (!supportEqual(moment.value().support,
                            varianceSupports[tuple[0U]])) {
            return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                        "projected moment support differs from "
                                        "its coordinate variance support");
          }
          for (Index axis = 1U; axis < order; ++axis) {
            if (!supportEqual(moment.value().support,
                              varianceSupports.at(tuple.at(axis)))) {
              return fail<MomentEstimate>(
                  ErrorCode::InvalidDomain,
                  "projected coordinates have incompatible supports");
            }
          }
          if (moment.value().value.fraction() != 0.0) {
            double fraction = moment.value().value.fraction();
            int exponent = moment.value().value.exponent();
            for (Index axis = 0U; axis < order; ++axis) {
              double varianceFraction = variances.at(tuple.at(axis)).fraction();
              int varianceExponent = variances.at(tuple.at(axis)).exponent();
              if ((varianceExponent % 2) != 0) {
                varianceFraction *= 2.0;
                --varianceExponent;
              }
              int deviationExponent = 0;
              const double deviationFraction =
                  std::frexp(std::sqrt(varianceFraction), &deviationExponent);
              int adjustment = 0;
              fraction = std::frexp(fraction / deviationFraction, &adjustment);
              exponent +=
                  adjustment - (varianceExponent / 2 + deviationExponent);
            }
            value = std::scalbn(fraction, exponent);
            if (!std::isfinite(value) || value == 0.0) {
              return fail<MomentEstimate>(
                  ErrorCode::NonFiniteResult,
                  "standardized projected moment is not representable");
            }
          }
        }
        values.push_back(value);
      }
      outputTerms.push_back(
          MomentTerm{.order = order,
                     .layout = MomentRepresentation::dense,
                     .flattened_tuples = {},
                     .values = Vector(std::span<const Scalar>{values})});
    }
    auto terms = Array<MomentTerm>::copy(outputTerms, limits);
    if (!terms) {
      return fail<MomentEstimate>(terms.status().code(),
                                  "projected output exceeds configured limits");
    }
    const std::array<Index, 1U> outputShape{map.rows()};
    auto shape = Array<Index>::copy(outputShape, limits);
    if (!shape) {
      return fail<MomentEstimate>(shape.status().code(),
                                  "projected shape exceeds configured limits");
    }
    auto outputSupports = Array<Support>::copy(varianceSupports, limits);
    if (!outputSupports) {
      return fail<MomentEstimate>(
          outputSupports.status().code(),
          "projected support exceeds configured limits");
    }
    auto warnings =
        Array<ErrorCode>::copy(source.diagnostics.warnings.view(), limits);
    if (!warnings) {
      return fail<MomentEstimate>(
          warnings.status().code(),
          "projected warnings exceed configured limits");
    }
    EstimatorDiagnostics diagnostics{
        .estimand = source.diagnostics.estimand,
        .weight = source.diagnostics.weight,
        .center = source.diagnostics.center,
        .approximation = source.diagnostics.approximation,
        .global = source.diagnostics.global,
        .coordinate_support = std::move(outputSupports).value(),
        .tuple_support = {},
        .reciprocal_condition = source.diagnostics.reciprocal_condition,
        .numerical_error = std::nullopt,
        .sampling_error = std::nullopt,
        .rank = source.diagnostics.rank,
        .algorithm = "standardized_projected_central_moments",
        .premises = source.diagnostics.premises +
                    "; exact linear projection standardized by positive "
                    "projected variances",
        .warnings = std::move(warnings).value()};
    return Result<MomentEstimate>::success(
        MomentEstimate{.observation_shape = std::move(shape).value(),
                       .terms = std::move(terms).value(),
                       .diagnostics = std::move(diagnostics),
                       .population_mean = Vector(map.rows(), 0.0)});
  } catch (const std::bad_alloc&) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "projected standardization allocation failed");
  }
}

} // namespace

Result<MomentEstimate> marginalMoments(const MomentEstimate& source,
                                       std::span<const Index> components,
                                       const MomentRequest& request,
                                       Limits limits) {
  const auto sourceDimension = momentDimension(source);
  if (!sourceDimension) {
    return fail<MomentEstimate>(sourceDimension.status().code(),
                                "source moment shape is invalid");
  }
  std::vector<Index> seen;
  try {
    seen.reserve(components.size());
    for (const Index component : components) {
      if (component >= sourceDimension.value()) {
        return fail<MomentEstimate>(ErrorCode::IndexOutOfRange,
                                    "selected component is out of range");
      }
      if (std::ranges::find(seen, component) != seen.end()) {
        return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                    "selected components must be unique");
      }
      seen.push_back(component);
    }
  } catch (const std::bad_alloc&) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "component validation allocation failed");
  }
  return buildTransform(source, components.size(), components, request, limits);
}

Result<MomentEstimate> standardizedMoments(const MomentEstimate& source,
                                           const MomentRequest& request,
                                           Limits limits) {
  const auto dimensionResult = momentDimension(source);
  if (!dimensionResult) {
    return fail<MomentEstimate>(dimensionResult.status().code(),
                                "source moment shape is invalid");
  }
  const Extent dimension = dimensionResult.value();
  const auto& algorithm = source.diagnostics.algorithm;
  const bool centralInput = algorithm.find("central") != std::string::npos;
  const bool rawInput = algorithm.find("raw") != std::string::npos;
  if (!centralInput && !rawInput) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "source moment kind is not identifiable");
  }
  if (source.diagnostics.center == CenterRule::known) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "standardization requires mean-centered "
                                "moments");
  }
  if (request.orders == 0U || (request.orders & 0xf0U) != 0U ||
      !validRepresentation(request.representation)) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "standardized moment request is invalid");
  }
  if (request.contraction.has_value() ||
      request.representation == MomentRepresentation::contraction) {
    if (request.representation == MomentRepresentation::contraction &&
        request.contraction.has_value()) {
      return standardizedProjection(source, request, dimension, limits);
    }
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "standardization does not accept a linear "
                                "contraction");
  }
  if (request.representation == MomentRepresentation::selected &&
      std::popcount(request.orders) != 1) {
    return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                "selected representation requests one order");
  }

  try {
    std::vector<std::optional<Scalar>> deviations(dimension);
    const auto deviationFor = [&](Index coordinate) -> Result<Scalar> {
      if (coordinate >= dimension) {
        return fail<Scalar>(
            ErrorCode::IndexOutOfRange,
            "standardized tuple coordinate is outside its shape");
      }
      if (deviations[coordinate].has_value()) {
        return Result<Scalar>::success(*deviations[coordinate]);
      }
      const std::array<Index, 2U> diagonal{coordinate, coordinate};
      auto variance = centralInput ? termValue(source, 2U, diagonal, dimension)
                                   : rawToCentral(source, diagonal, dimension);
      if (!variance) {
        return fail<Scalar>(variance.status().code(),
                            "standardization requires diagonal second moments");
      }
      if (!(variance.value() > 0.0)) {
        return fail<Scalar>(
            ErrorCode::InvalidDomain,
            "standardization requires positive coordinate variance");
      }
      deviations[coordinate] = std::sqrt(variance.value());
      return Result<Scalar>::success(*deviations[coordinate]);
    };

    std::vector<MomentTerm> outputTerms;
    for (Index order = 1U; order <= 4U; ++order) {
      if ((request.orders & (1U << (order - 1U))) == 0U) {
        continue;
      }
      auto tuples = enumerateTuples(dimension, order, request.representation,
                                    request.selected, limits);
      if (!tuples) {
        return fail<MomentEstimate>(tuples.status().code(),
                                    "standardized tuple generation failed");
      }
      std::vector<Scalar> values;
      values.reserve(tuples.value().size());
      for (const auto& tuple : tuples.value()) {
        const auto tupleSupport = supportForTuple(source, tuple, dimension);
        if (!tupleSupport) {
          return fail<MomentEstimate>(tupleSupport.status().code(),
                                      "standardized tuple support is invalid");
        }
        for (const Index coordinate : tuple) {
          const std::array<Index, 2U> diagonal{coordinate, coordinate};
          const auto scaleSupport =
              supportForTuple(source, diagonal, dimension);
          if (!scaleSupport) {
            return fail<MomentEstimate>(
                scaleSupport.status().code(),
                "standardized scale support is invalid");
          }
          if (!supportEqual(tupleSupport.value(), scaleSupport.value())) {
            return fail<MomentEstimate>(ErrorCode::InvalidDomain,
                                        "standardized moment and coordinate "
                                        "scales use incompatible supports");
          }
        }
        Scalar central = 0.0;
        if (order > 1U) {
          auto moment = centralInput
                            ? termValue(source, order, tuple, dimension)
                            : rawToCentral(source, tuple, dimension);
          if (!moment) {
            return fail<MomentEstimate>(
                moment.status().code(),
                "standardization requires source "
                "moments through every requested order");
          }
          central = moment.value();
        }
        std::vector<Scalar> scales;
        scales.reserve(order);
        for (const Index coordinate : tuple) {
          auto deviation = deviationFor(coordinate);
          if (!deviation) {
            return fail<MomentEstimate>(deviation.status().code(),
                                        "standardization requires a defined "
                                        "coordinate scale");
          }
          scales.push_back(deviation.value());
        }
        auto standardized = scaledStandardize(central, scales);
        if (!standardized) {
          return fail<MomentEstimate>(
              standardized.status().code(),
              "standardized moment is not representable");
        }
        values.push_back(standardized.value());
      }
      Array<Index> flattened;
      if (request.representation == MomentRepresentation::selected) {
        std::vector<Index> indices;
        for (const auto& tuple : tuples.value()) {
          indices.insert(indices.end(), tuple.begin(), tuple.end());
        }
        auto copied = Array<Index>::copy(indices, limits);
        if (!copied) {
          return fail<MomentEstimate>(copied.status().code(),
                                      "standardized tuple metadata exceeds "
                                      "limits");
        }
        flattened = std::move(copied).value();
      }
      outputTerms.push_back(
          MomentTerm{.order = order,
                     .layout = request.representation,
                     .flattened_tuples = std::move(flattened),
                     .values = Vector(std::span<const Scalar>{values})});
    }
    auto terms = Array<MomentTerm>::copy(outputTerms, limits);
    if (!terms) {
      return fail<MomentEstimate>(terms.status().code(),
                                  "standardized moment output exceeds limits");
    }
    auto shape = Array<Index>::copy(source.observation_shape.view(), limits);
    if (!shape) {
      return fail<MomentEstimate>(shape.status().code(),
                                  "standardized moment shape exceeds limits");
    }
    EstimatorDiagnostics diagnostics = source.diagnostics;
    diagnostics.algorithm = "standardized_central_moments";
    diagnostics.premises += "; central moments standardized by positive "
                            "component standard deviations";
    return Result<MomentEstimate>::success(
        MomentEstimate{.observation_shape = std::move(shape).value(),
                       .terms = std::move(terms).value(),
                       .diagnostics = std::move(diagnostics),
                       .population_mean = Vector(dimension, 0.0)});
  } catch (const std::bad_alloc&) {
    return fail<MomentEstimate>(ErrorCode::ResourceLimit,
                                "standardization allocation failed");
  }
}

} // namespace vss::math::statistics
