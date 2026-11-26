#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Error.hpp>
#include <vss/math/covariance/Sensitivity.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool add(Index left, Index right, Index& result) noexcept {
  if (left > std::numeric_limits<Index>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

bool multiply(Index left, Index right, Index& result) noexcept {
  if (right != 0U && left > std::numeric_limits<Index>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

Result<Vector> evaluate(const differentiation::VectorFunction& function,
                        ConstVectorView input, Index outputDimension) {
  try {
    auto output = function(input);
    if (!output) {
      const ErrorCode code =
          output.status().code() == ErrorCode::DimensionMismatch
              ? ErrorCode::DimensionMismatch
              : ErrorCode::CallbackFailure;
      return fail<Vector>(code, "sensitivity function callback failed");
    }
    if (output.value().size() != outputDimension) {
      return fail<Vector>(ErrorCode::DimensionMismatch,
                          "sensitivity function output dimension changed");
    }
    for (Index i = 0U; i < outputDimension; ++i) {
      if (!std::isfinite(output.value().at(i))) {
        return fail<Vector>(
            ErrorCode::NonFiniteResult,
            "sensitivity callback returned a non-finite output");
      }
    }
    return output;
  } catch (...) {
    return fail<Vector>(ErrorCode::CallbackFailure,
                        "sensitivity function callback threw an exception");
  }
}

Result<Scalar> draw(const IndependentMarginals& law, Index coordinate,
                    probability::Rng& rng, ProbabilityPolicy policy) {
  return probability::sample(law.marginals[coordinate], rng, policy);
}

} // namespace

Result<SensitivityResult>
sensitivityIndices(const differentiation::VectorFunction& function,
                   const IndependentMarginals& law, const SensitivitySpec& spec,
                   probability::Rng& rng, Limits limits) {
  const Index dimension = law.marginals.size();
  if (!function || dimension == 0U || spec.draws < 2U || spec.replicates < 2U ||
      spec.output_dimension == 0U || (!spec.first_order && !spec.total_order) ||
      spec.sampling.max_draws == 0U) {
    return fail<SensitivityResult>(
        ErrorCode::InvalidDomain,
        "sensitivity request is incomplete or invalid");
  }

  Index rows = 0U;
  Index callbackCount = 0U;
  Index sampleRows = 0U;
  Index sampleCount = 0U;
  if (!multiply(spec.replicates, spec.draws, rows) ||
      !add(dimension, 2U, callbackCount) ||
      !multiply(rows, callbackCount, callbackCount) ||
      !multiply(rows, 2U, sampleRows) ||
      !multiply(sampleRows, dimension, sampleCount)) {
    return fail<SensitivityResult>(ErrorCode::SizeOverflow,
                                   "sensitivity work count overflows");
  }
  if (callbackCount > limits.evaluations) {
    return fail<SensitivityResult>(ErrorCode::ResourceLimit,
                                   "sensitivity callback limit exceeded");
  }

  Index matrixElements = 0U;
  Index vectorElements = 0U;
  Index longDoubleElements = 0U;
  Index scalarElements = 0U;
  Index matrixBytes = 0U;
  Index accumulatorBytes = 0U;
  Index requiredBytes = 0U;
  Index requiredElements = 0U;
  if (!multiply(spec.output_dimension, dimension, matrixElements) ||
      !multiply(matrixElements, 6U, longDoubleElements) ||
      !multiply(spec.output_dimension, 3U, vectorElements) ||
      !multiply(dimension, 6U, scalarElements) ||
      !add(scalarElements, vectorElements, scalarElements) ||
      !multiply(matrixElements, 3U, vectorElements) ||
      !add(scalarElements, vectorElements, scalarElements) ||
      !multiply(spec.output_dimension, 2U, vectorElements) ||
      !add(longDoubleElements, vectorElements, longDoubleElements) ||
      !multiply(longDoubleElements, sizeof(long double), accumulatorBytes) ||
      !multiply(scalarElements, sizeof(Scalar), matrixBytes) ||
      !add(accumulatorBytes, matrixBytes, requiredBytes) ||
      !add(longDoubleElements, scalarElements, requiredElements)) {
    return fail<SensitivityResult>(ErrorCode::SizeOverflow,
                                   "sensitivity workspace size overflows");
  }
  if (requiredElements > limits.elements || requiredBytes > limits.bytes) {
    return fail<SensitivityResult>(ErrorCode::ResourceLimit,
                                   "sensitivity workspace exceeds limits");
  }
  (void)
      sampleCount; // checked above; scalar marginal draws are bounded by rows.

  std::vector<long double> outputMean(spec.output_dimension, 0.0L);
  std::vector<long double> outputM2(spec.output_dimension, 0.0L);
  std::vector<long double> firstSquares(matrixElements, 0.0L);
  std::vector<long double> totalSquares(matrixElements, 0.0L);
  std::vector<long double> firstMean(matrixElements, 0.0L);
  std::vector<long double> firstM2(matrixElements, 0.0L);
  std::vector<long double> totalMean(matrixElements, 0.0L);
  std::vector<long double> totalM2(matrixElements, 0.0L);
  Vector a(dimension);
  Vector b(dimension);
  Vector hybrid(dimension);
  Index evaluations = 0U;

  for (Index replicate = 0U; replicate < spec.replicates; ++replicate) {
    std::fill(outputMean.begin(), outputMean.end(), 0.0L);
    std::fill(outputM2.begin(), outputM2.end(), 0.0L);
    std::fill(firstSquares.begin(), firstSquares.end(), 0.0L);
    std::fill(totalSquares.begin(), totalSquares.end(), 0.0L);
    Index pooledCount = 0U;
    for (Index row = 0U; row < spec.draws; ++row) {
      for (Index input = 0U; input < dimension; ++input) {
        auto drawA = draw(law, input, rng, spec.sampling);
        if (!drawA) {
          return Result<SensitivityResult>::failure(
              drawA.status().code(), std::string(drawA.status().message()));
        }
        auto drawB = draw(law, input, rng, spec.sampling);
        if (!drawB) {
          return Result<SensitivityResult>::failure(
              drawB.status().code(), std::string(drawB.status().message()));
        }
        a.at(input) = drawA.value();
        b.at(input) = drawB.value();
      }

      ++evaluations;
      auto outputA = evaluate(function, a.view(), spec.output_dimension);
      if (!outputA) {
        return Result<SensitivityResult>::failure(
            outputA.status().code(), std::string(outputA.status().message()));
      }
      ++evaluations;
      auto outputB = evaluate(function, b.view(), spec.output_dimension);
      if (!outputB) {
        return Result<SensitivityResult>::failure(
            outputB.status().code(), std::string(outputB.status().message()));
      }
      for (Index output = 0U; output < spec.output_dimension; ++output) {
        const long double values[] = {outputA.value().at(output),
                                      outputB.value().at(output)};
        Index coordinateCount = pooledCount;
        for (const long double value : values) {
          ++coordinateCount;
          const long double delta = value - outputMean[output];
          outputMean[output] += delta / coordinateCount;
          outputM2[output] += delta * (value - outputMean[output]);
        }
      }
      pooledCount += 2U;

      for (Index input = 0U; input < dimension; ++input) {
        for (Index coordinate = 0U; coordinate < dimension; ++coordinate) {
          hybrid.at(coordinate) =
              coordinate == input ? b.at(coordinate) : a.at(coordinate);
        }
        ++evaluations;
        auto outputHybrid =
            evaluate(function, hybrid.view(), spec.output_dimension);
        if (!outputHybrid) {
          return Result<SensitivityResult>::failure(
              outputHybrid.status().code(),
              std::string(outputHybrid.status().message()));
        }
        for (Index output = 0U; output < spec.output_dimension; ++output) {
          const Index flat = output * dimension + input;
          const long double differenceA =
              static_cast<long double>(outputA.value().at(output)) -
              outputHybrid.value().at(output);
          const long double differenceB =
              static_cast<long double>(outputB.value().at(output)) -
              outputHybrid.value().at(output);
          totalSquares[flat] += differenceA * differenceA;
          firstSquares[flat] += differenceB * differenceB;
        }
      }
    }

    for (Index output = 0U; output < spec.output_dimension; ++output) {
      const long double variance = outputM2[output] / pooledCount;
      if (!std::isfinite(variance) || variance <= 0.0L) {
        return fail<SensitivityResult>(
            ErrorCode::InvalidDomain,
            "sensitivity output variance is zero or invalid");
      }
      for (Index input = 0U; input < dimension; ++input) {
        const Index flat = output * dimension + input;
        const long double first =
            1.0L - firstSquares[flat] / spec.draws / (2.0L * variance);
        const long double total =
            totalSquares[flat] / spec.draws / (2.0L * variance);
        const long double firstDelta = first - firstMean[flat];
        firstMean[flat] += firstDelta / (replicate + 1U);
        firstM2[flat] += firstDelta * (first - firstMean[flat]);
        const long double totalDelta = total - totalMean[flat];
        totalMean[flat] += totalDelta / (replicate + 1U);
        totalM2[flat] += totalDelta * (total - totalMean[flat]);
      }
    }
  }
  if (evaluations != callbackCount) {
    return fail<SensitivityResult>(ErrorCode::NonFiniteResult,
                                   "sensitivity callback accounting mismatch");
  }

  const Index errorRows = spec.output_dimension *
                          ((spec.first_order && spec.total_order) ? 2U : 1U);
  Matrix firstResult(spec.first_order ? spec.output_dimension : 0U,
                     spec.first_order ? dimension : 0U);
  Matrix totalResult(spec.total_order ? spec.output_dimension : 0U,
                     spec.total_order ? dimension : 0U);
  Matrix errors(errorRows, dimension);
  for (Index output = 0U; output < spec.output_dimension; ++output) {
    for (Index input = 0U; input < dimension; ++input) {
      const Index flat = output * dimension + input;
      const long double firstError =
          std::sqrt(firstM2[flat] / (spec.replicates - 1U) / spec.replicates);
      const long double totalError =
          std::sqrt(totalM2[flat] / (spec.replicates - 1U) / spec.replicates);
      const long double first = firstMean[flat];
      const long double total = totalMean[flat];
      if (!std::isfinite(first) || !std::isfinite(total) ||
          !std::isfinite(firstError) || !std::isfinite(totalError) ||
          std::abs(first) > std::numeric_limits<Scalar>::max() ||
          std::abs(total) > std::numeric_limits<Scalar>::max() ||
          firstError > std::numeric_limits<Scalar>::max() ||
          totalError > std::numeric_limits<Scalar>::max()) {
        return fail<SensitivityResult>(
            ErrorCode::NonFiniteResult,
            "sensitivity result is outside scalar range");
      }
      if (spec.first_order) {
        firstResult.at(output, input) = static_cast<Scalar>(first);
        errors.at(output, input) = static_cast<Scalar>(firstError);
      }
      if (spec.total_order) {
        totalResult.at(output, input) = static_cast<Scalar>(total);
        const Index row =
            spec.first_order ? spec.output_dimension + output : output;
        errors.at(row, input) = static_cast<Scalar>(totalError);
      }
    }
  }
  return Result<SensitivityResult>::success(
      {std::move(firstResult), std::move(totalResult), std::move(errors)});
}

Result<UncertaintyBudget>
uncertaintyBudget(ConstMatrixView jacobian,
                  std::span<const SourceBlock> sources,
                  std::span<const SourceCrossBlock> cross,
                  DependencePolicy dependence, Limits limits) {
  return uncertaintyContributionBudget(jacobian, sources, cross, dependence,
                                       limits);
}

} // namespace vss::math::covariance
