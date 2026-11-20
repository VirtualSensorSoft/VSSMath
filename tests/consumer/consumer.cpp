#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MultivariateMoments.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Conditioning.hpp>
#include <vss/math/covariance/Diagnostics.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/covariance/Structured.hpp>
#include <vss/math/covariance/UncertaintySummary.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/Reduction.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Sparse.hpp>
#include <vss/math/linear/Structured.hpp>
#include <vss/math/version.hpp>

namespace {

bool require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "math_consumer: " << message << '\n';
    return false;
  }
  return true;
}

template <class ResultType>
bool requireError(const ResultType& result, vss::math::ErrorCode code,
                  const char* message) {
  return require(!result.hasValue() && result.status().code() == code, message);
}

} // namespace

int main() {
  using namespace vss::math;

  const std::array<Extent, 2> tensorExtents{2U, 2U};
  const auto tensorShape = TensorShape::create(tensorExtents);
  if (!require(tensorShape.hasValue(), "tensor shape construction failed")) {
    return 1;
  }
  auto tensor = Tensor::create(tensorShape.value(), 0.0, TensorLimits{4U});
  if (!require(tensor.hasValue(), "tensor allocation failed")) {
    return 1;
  }
  tensor.value().at(std::array<Index, 2>{1U, 1U}) = 5.0;
  if (!require(tensor.value().at(std::array<Index, 2>{1U, 1U}) == 5.0,
               "installed tensor access failed")) {
    return 1;
  }

  const Limits limits{.elements = 64U, .bytes = 4096U, .evaluations = 64U};
  {
    const Vector input(std::array{2.0, 3.0});
    const Vector diagonal(std::array{4.0, 5.0});
    const auto action =
        linear::apply(linear::DiagonalMatrix{diagonal}, input.view(), limits);
    const auto structured = covariance::structuredCovariance(
        {covariance::StructureKind::diagonal_low_rank,
         diagonal.view(),
         std::nullopt,
         {},
         std::nullopt,
         std::nullopt},
        SolvePolicy{}, limits);
    if (!require(
            action.hasValue() && action.value().at(0) == 8 &&
                structured.hasValue(),
            "installed structured construction or diagonal action failed")) {
      return 1;
    }
    const auto applied = covariance::applyStructuredCovariance(
        structured.value(), input.view(), limits);
    if (!require(applied.hasValue() && applied.value().at(1) == 15,
                 "installed structured covariance action failed")) {
      return 1;
    }
    const auto sparse = linear::sparseCSRFromTriplets(
        2, 2, std::array{linear::Triplet{0, 1, 4}}, limits);
    if (!require(sparse.hasValue(), "installed sparse construction failed")) {
      return 1;
    }
    const auto sparse_product =
        linear::apply(sparse.value(), input.view(), limits);
    if (!require(sparse_product.hasValue() &&
                     sparse_product.value().at(0) == 12,
                 "installed sparse action failed")) {
      return 1;
    }
  }
  const auto tensorProduct = hadamard(tensor.value(), tensor.value(), limits);
  if (!require(tensorProduct.hasValue() &&
                   tensorProduct.value().at(std::array<Index, 2>{1U, 1U}) ==
                       25.0,
               "installed tensor Hadamard API failed")) {
    return 1;
  }
  const auto tensorQuotient = divide(tensor.value(), 5.0, limits);
  if (!require(tensorQuotient.hasValue() &&
                   tensorQuotient.value().at(std::array<Index, 2>{1U, 1U}) ==
                       1.0,
               "installed tensor division API failed")) {
    return 1;
  }
  const std::array<Extent, 2> broadcastShape{2U, 2U};
  const std::array<Index, 2> broadcastAxes{0U, 1U};
  const auto broadcast =
      broadcastTo(tensor.value(), broadcastShape, broadcastAxes, limits);
  if (!require(broadcast.hasValue() &&
                   broadcast.value().at(std::array<Index, 2>{1U, 1U}) == 5.0,
               "installed tensor broadcast API failed")) {
    return 1;
  }
  const auto transformed = transformElements(
      tensor.value(),
      [](Scalar value) -> Result<Scalar> {
        return Result<Scalar>::success(value + 1.0);
      },
      limits);
  if (!require(transformed.hasValue() &&
                   transformed.value().at(std::array<Index, 2>{1U, 1U}) == 6.0,
               "installed tensor transform API failed")) {
    return 1;
  }

  Matrix matrix(2U, 2U);
  matrix.at(0U, 0U) = 1.0;
  matrix.at(0U, 1U) = 2.0;
  matrix.at(1U, 0U) = 3.0;
  matrix.at(1U, 1U) = 4.0;
  const auto matrixProduct = hadamard(matrix.view(), matrix.view(), limits);
  if (!require(matrixProduct.hasValue() &&
                   matrixProduct.value().at(1U, 1U) == 16.0,
               "installed matrix Hadamard API failed")) {
    return 1;
  }
  const auto matrixNorm = frobenius(matrix.view());
  if (!require(matrixNorm.hasValue() && matrixNorm.value() > 5.47 &&
                   matrixNorm.value() < 5.48,
               "installed matrix Frobenius API failed")) {
    return 1;
  }
  const auto matrixInner = frobeniusInner(matrix.view(), matrix.view());
  if (!require(matrixInner.hasValue() && matrixInner.value() == 30.0,
               "installed matrix Frobenius inner-product API failed")) {
    return 1;
  }
  const auto tensorNorm = frobenius(tensor.value());
  const auto tensorInner = frobeniusInner(tensor.value(), tensor.value());
  if (!require(tensorNorm.hasValue() && tensorInner.hasValue() &&
                   tensorInner.value() == 25.0,
               "installed tensor Frobenius APIs failed")) {
    return 1;
  }

  Vector vector(2U);
  vector.at(0U) = 5.0;
  vector.at(1U) = 6.0;
  const auto vectorProduct = hadamard(vector.view(), vector.view(), limits);
  if (!require(vectorProduct.hasValue() &&
                   vectorProduct.value().at(0U) == 25.0 &&
                   vectorProduct.value().at(1U) == 36.0,
               "installed vector Hadamard API failed")) {
    return 1;
  }

  PrintOptions printOptions;
  printOptions.maxElements = 2U;
  std::ostringstream formatted;
  print(formatted, tensor.value(), printOptions);
  if (!require(formatted.str().find("2 omitted") != std::string::npos,
               "installed bounded tensor printing failed")) {
    return 1;
  }
  const std::array<Extent, 1> scalarExtents{1U};
  const auto scalarShape = TensorShape::create(scalarExtents);
  if (!require(scalarShape.hasValue(), "scalar shape construction failed")) {
    return 1;
  }
  auto scalarTensor =
      Tensor::create(scalarShape.value(), 12.0, TensorLimits{1U});
  if (!require(scalarTensor.hasValue(), "scalar tensor allocation failed")) {
    return 1;
  }
  auto firstMoment = FirstMoment::create(scalarTensor.value());
  auto moments = MultivariateMoments::create(1U);
  if (!require(firstMoment.hasValue() && moments.hasValue(),
               "installed moment API construction failed")) {
    return 1;
  }
  if (!require(moments.value().setFirstMoment(firstMoment.value()).ok(),
               "installed moment setter failed")) {
    return 1;
  }
  formatted << vector << matrix << tensor.value() << tensorShape.value()
            << firstMoment.value() << moments.value();
  if (!require(formatted.str().find("MultivariateMoments(") !=
                   std::string::npos,
               "installed stream insertion symbols failed to link")) {
    return 1;
  }

  const auto result = multiply(matrix.view(), vector.view());
  if (!require(result.hasValue(), "matrix-vector multiplication failed")) {
    return 1;
  }
  if (!require(result.value().size() == 2U, "unexpected result size")) {
    return 1;
  }
  if (!require(result.value().at(0U) == 17.0 && result.value().at(1U) == 39.0,
               "unexpected matrix-vector result")) {
    return 1;
  }

  Matrix diagonal(2U, 2U);
  diagonal.at(0U, 0U) = 2.0;
  diagonal.at(1U, 1U) = 3.0;
  Vector diagonalInput(2U);
  diagonalInput.at(0U) = 2.0;
  diagonalInput.at(1U) = 3.0;
  const auto diagonalResult = multiply(diagonal.view(), diagonalInput.view());
  if (!require(diagonalResult.hasValue(),
               "diagonal matrix-vector multiplication failed")) {
    return 1;
  }
  if (!require(diagonalResult.value().at(0U) == 4.0 &&
                   diagonalResult.value().at(1U) == 9.0,
               "unexpected diagonal matrix-vector result")) {
    return 1;
  }

  Vector rightHandSide(2U);
  rightHandSide.at(0U) = 4.0;
  rightHandSide.at(1U) = 9.0;
  const auto factorized = vss::math::linear::factorizeLU(diagonal.view());
  if (!require(factorized.hasValue(), "LU factorization failed")) {
    return 1;
  }
  const auto solved = factorized.value().solve(rightHandSide.view());
  if (!require(solved.hasValue(), "LU solve failed")) {
    return 1;
  }
  if (!require(solved.value().solution().at(0U) == 2.0 &&
                   solved.value().solution().at(1U) == 3.0,
               "unexpected LU solution")) {
    return 1;
  }
  if (!require(solved.value().conditionReport().has_value(),
               "LU solve did not return condition diagnostics")) {
    return 1;
  }

  Matrix generalizedA(2U, 2U);
  generalizedA.at(0U, 0U) = 2.0;
  generalizedA.at(1U, 1U) = 12.0;
  Matrix generalizedB(2U, 2U);
  generalizedB.at(0U, 0U) = 1.0;
  generalizedB.at(1U, 1U) = 3.0;
  SolvePolicy generalizedPolicy{};
  generalizedPolicy.iteration.max_iterations = 64U;
  const auto generalized = linear::factorGeneralizedSymmetric(
      generalizedA.view(), generalizedB.view(), generalizedPolicy,
      Limits{.elements = 100'000U,
             .bytes = std::size_t{8U} * 1024U * 1024U,
             .evaluations = 0U});
  if (!require(generalized.hasValue(),
               "installed generalized eigen API failed")) {
    return 1;
  }
  if (!require(std::abs(generalized.value().values.at(0U) - 2.0) < 1.0e-12 &&
                   std::abs(generalized.value().values.at(1U) - 4.0) < 1.0e-12,
               "installed generalized eigen values are incorrect")) {
    return 1;
  }

  const std::array<Index, 2> indices{4U, 9U};
  const auto ownedIndices = Array<Index>::copy(indices, limits);
  if (!require(ownedIndices.hasValue() &&
                   ownedIndices.value().view().size() == indices.size() &&
                   ownedIndices.value().view()[1] == 9U,
               "installed immutable Array copy failed")) {
    return 1;
  }

  Matrix exchange(2U, 2U);
  exchange.at(0U, 1U) = 1.0;
  exchange.at(1U, 0U) = 1.0;
  SolvePolicy decompositionPolicy;
  decompositionPolicy.iteration.max_iterations = 64U;
  const auto indefiniteFactor =
      linear::factorLDLT(exchange.view(), decompositionPolicy, limits);
  if (!require(indefiniteFactor.hasValue() &&
                   indefiniteFactor.value().rank == 2U &&
                   indefiniteFactor.value().pivot_sizes.view().size() == 1U &&
                   indefiniteFactor.value().pivot_sizes.view()[0] == 2U,
               "installed symmetric-indefinite LDLT factorization failed")) {
    return 1;
  }

  Matrix singular(2U, 2U);
  const auto singularFactor = linear::factorizeLU(singular.view());
  if (!require(singularFactor.hasValue() && singularFactor.value().isSingular(),
               "zero matrix did not produce a singular LU diagnostic")) {
    return 1;
  }
  Vector nanVector(2U);
  nanVector.at(0U) = std::numeric_limits<Scalar>::quiet_NaN();
  nanVector.at(1U) = 1.0;
  if (!requireError(singularFactor.value().solve(nanVector.view()),
                    ErrorCode::NonFiniteInput,
                    "singular vector solve did not prioritize NaN input")) {
    return 1;
  }
  Matrix infiniteMatrix(2U, 1U);
  infiniteMatrix.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  infiniteMatrix.at(1U, 0U) = 1.0;
  if (!requireError(singularFactor.value().solve(infiniteMatrix.view()),
                    ErrorCode::NonFiniteInput,
                    "singular matrix solve did not prioritize infinity")) {
    return 1;
  }
  Matrix emptyColumns(2U, 0U);
  if (!requireError(singularFactor.value().solve(emptyColumns.view()),
                    ErrorCode::InvalidExtent,
                    "singular matrix solve accepted an empty RHS")) {
    return 1;
  }

  Matrix illConditionedInput(2U, 2U);
  illConditionedInput.at(0U, 0U) = 1.0;
  illConditionedInput.at(1U, 1U) = 1.0e-8;
  linear::FactorizationOptions factorizationOptions;
  factorizationOptions.conditionLimit = 2.0;
  const auto illConditionedFactor =
      linear::factorizeLU(illConditionedInput.view(), factorizationOptions);
  if (!require(
          illConditionedFactor.hasValue() &&
              illConditionedFactor.value().conditionReport().classification ==
                  linear::ConditionClassification::IllConditioned,
          "configured ill-conditioned LU factorization did not succeed")) {
    return 1;
  }
  if (!requireError(
          illConditionedFactor.value().solve(nanVector.view()),
          ErrorCode::NonFiniteInput,
          "ill-conditioned vector solve did not prioritize NaN input")) {
    return 1;
  }
  if (!requireError(
          illConditionedFactor.value().solve(infiniteMatrix.view()),
          ErrorCode::NonFiniteInput,
          "ill-conditioned matrix solve did not prioritize infinity")) {
    return 1;
  }

  Matrix rankAwareDesign(1U, 2U, 1.0);
  Matrix rankAwareObservations(1U, 1U, 4.0);
  SolvePolicy rankAwarePolicy;
  rankAwarePolicy.iteration.max_iterations = 64U;
  rankAwarePolicy.rank = RankPolicy::minimum_norm;
  const auto rankAware = linear::leastSquaresMinNorm(
      rankAwareDesign.view(), rankAwareObservations.view(), rankAwarePolicy,
      limits);
  if (!require(rankAware.hasValue() && rankAware.value().rank == 1U &&
                   rankAware.value().identified_basis.has_value() &&
                   std::abs(rankAware.value().parameters.at(0U, 0U) - 2.0) <
                       1.0e-12 &&
                   std::abs(rankAware.value().parameters.at(1U, 0U) - 2.0) <
                       1.0e-12 &&
                   rankAware.value().residual_norms.at(0U) < 1.0e-12,
               "installed rank-aware minimum-norm least-squares API failed")) {
    return 1;
  }
  const auto nullBasis =
      linear::nullSpace(rankAwareDesign.view(), rankAwarePolicy, limits);
  if (!require(nullBasis.hasValue() && nullBasis.value().rows() == 2U &&
                   nullBasis.value().columns() == 1U &&
                   std::abs(nullBasis.value().at(0U, 0U) +
                            nullBasis.value().at(1U, 0U)) < 1.0e-12 &&
                   std::abs(std::hypot(nullBasis.value().at(0U, 0U),
                                       nullBasis.value().at(1U, 0U)) -
                            1.0) < 1.0e-12,
               "installed right-null-space API failed")) {
    return 1;
  }

  Matrix fullQrInput(3U, 2U);
  fullQrInput.at(0U, 0U) = 1.0;
  fullQrInput.at(1U, 1U) = 2.0;
  fullQrInput.at(2U, 0U) = 1.0;
  fullQrInput.at(2U, 1U) = 1.0;
  SolvePolicy fullQrPolicy{};
  fullQrPolicy.iteration.max_iterations = 128U;
  const auto fullQr = linear::factorFullQR(
      fullQrInput.view(), linear::QRColumnOrder::pivoted, fullQrPolicy, limits);
  if (!require(fullQr.hasValue() && fullQr.value().orthogonal.rows() == 3U &&
                   fullQr.value().orthogonal.columns() == 3U &&
                   fullQr.value().upper.rows() == 3U &&
                   fullQr.value().upper.columns() == 2U &&
                   fullQr.value().permutation.view()[0U] == 1U &&
                   fullQr.value().rank == 2U,
               "installed full QR API failed")) {
    return 1;
  }

  Matrix jointCovariance(2U, 2U);
  jointCovariance.at(0U, 0U) = 4.0;
  jointCovariance.at(0U, 1U) = 3.0;
  jointCovariance.at(1U, 0U) = 3.0;
  jointCovariance.at(1U, 1U) = 9.0;
  const std::array<Index, 1U> kept{0U};
  const std::array<Index, 1U> conditioned{1U};
  const auto conditional = covariance::conditionalCovariance(
      jointCovariance.view(), kept, conditioned, decompositionPolicy, limits);
  if (!require(conditional.hasValue() && conditional.value().rank == 1U &&
                   std::abs(conditional.value().covariance.at(0U, 0U) - 3.0) <
                       1.0e-12,
               "installed conditional covariance API failed")) {
    return 1;
  }
  const Vector whiteningInput{std::array<Scalar, 2U>{2.0, 3.0}};
  const auto whitening =
      covariance::whitenSupported(jointCovariance.view(), whiteningInput.view(),
                                  decompositionPolicy, limits);
  const auto mahalanobis =
      whitening.hasValue()
          ? covariance::mahalanobisSquared(whitening.value(),
                                           whiteningInput.view())
          : Result<Scalar>::failure(ErrorCode::InvalidDomain,
                                    "whitening consumer setup failed");
  if (!require(whitening.hasValue() && mahalanobis.hasValue() &&
                   std::abs(mahalanobis.value() - (4.0 / 3.0)) < 1.0e-12,
               "installed covariance whitening APIs failed")) {
    return 1;
  }

  Matrix glsDesign(2U, 1U, 1.0);
  Matrix glsObservations(2U, 1U);
  glsObservations.at(0U, 0U) = 1.0;
  glsObservations.at(1U, 0U) = 3.0;
  Matrix glsCovariance(2U, 2U);
  glsCovariance.at(0U, 0U) = 4.0;
  glsCovariance.at(0U, 1U) = 1.0;
  glsCovariance.at(1U, 0U) = 1.0;
  glsCovariance.at(1U, 1U) = 1.0;
  SolvePolicy glsPolicy = decompositionPolicy;
  glsPolicy.rank = RankPolicy::supported_subspace;
  const auto gls = linear::generalizedLinearEstimate(
      glsDesign.view(), glsObservations.view(), glsCovariance.view(), glsPolicy,
      limits);
  if (!require(gls.hasValue() && gls.value().covariance.has_value() &&
                   std::abs(gls.value().parameters.at(0U, 0U) - 3.0) <
                       1.0e-12 &&
                   std::abs(gls.value().covariance->at(0U, 0U) - 1.0) < 1.0e-12,
               "installed known-covariance GLS API failed")) {
    return 1;
  }

  Matrix reductionInput(2U, 2U);
  reductionInput.at(0U, 0U) = 3.0;
  const Limits reductionLimits{
      .elements = 4096U, .bytes = 262144U, .evaluations = 64U};
  const auto reduced = linear::lowRankApproximation(reductionInput.view(), 1U,
                                                    glsPolicy, reductionLimits);
  if (!require(reduced.hasValue() && reduced.value().singular.at(0U) == 3.0 &&
                   reduced.value().residual_frobenius == 0.0,
               "installed low-rank approximation API failed")) {
    return 1;
  }
  Matrix sampleRows(4U, 2U);
  sampleRows.at(0U, 0U) = 2.0;
  sampleRows.at(1U, 0U) = -2.0;
  sampleRows.at(2U, 1U) = 1.0;
  sampleRows.at(3U, 1U) = -1.0;
  const std::array<Index, 1U> sampleShape{2U};
  const SampleView sampleView{
      .rows = sampleRows.view(), .observation_shape = sampleShape, .valid = {}};
  const auto principal =
      linear::pca(sampleView, 1U, true, glsPolicy, reductionLimits);
  if (!require(principal.hasValue() &&
                   std::abs(principal.value().explained_variance.at(0U) -
                            (8.0 / 3.0)) < 1.0e-12,
               "installed centered PCA API failed")) {
    return 1;
  }

  Matrix propagatedCovariance(2U, 2U);
  propagatedCovariance.at(0U, 0U) = 4.0;
  propagatedCovariance.at(1U, 1U) = 9.0;
  const PropagationResult propagated{
      .mean = Vector{std::array<Scalar, 2U>{2.0, -3.0}},
      .covariance = propagatedCovariance,
      .method = Approximation::exact,
      .diagnostics = {},
      .covariance_check = PropagationCovarianceCheck::positive_semidefinite};
  const auto uncertainty = covariance::summarizeUncertainty(
      propagated, 2.0, std::nullopt, std::nullopt, reductionLimits);
  if (!require(uncertainty.hasValue() &&
                   uncertainty.value().standard.at(0U) == 2.0 &&
                   uncertainty.value().expanded.at(1U) == 6.0,
               "installed uncertainty summary API failed")) {
    return 1;
  }

  const std::array<Index, 1U> marginalIndex{1U};
  const auto marginal = covariance::marginalCovariance(
      jointCovariance.view(), marginalIndex, glsPolicy, reductionLimits);
  const auto inverseRoot = covariance::inverseSquareRoot(
      jointCovariance.view(), glsPolicy, reductionLimits);
  if (!require(marginal.hasValue() && marginal.value().at(0U, 0U) == 9.0 &&
                   inverseRoot.hasValue() && inverseRoot.value().rank == 2U,
               "installed covariance diagnostic APIs failed")) {
    return 1;
  }

  const auto jointDeterminant = covariance::determinant(
      jointCovariance.view(), glsPolicy, reductionLimits);
  const auto jointLogDeterminant = covariance::logAbsoluteDeterminant(
      jointCovariance.view(), glsPolicy, reductionLimits);
  if (!require(jointDeterminant.hasValue() && jointLogDeterminant.hasValue() &&
                   std::abs(jointDeterminant.value() - 27.0) < 1.0e-12 &&
                   std::abs(jointLogDeterminant.value() - std::log(27.0)) <
                       1.0e-12,
               "installed determinant diagnostic APIs failed")) {
    return 1;
  }

  Matrix leftMap(1U, 1U, 2.0);
  Matrix sourceCross(1U, 1U, 0.5);
  Matrix rightMap(1U, 1U, 3.0);
  const auto transformedCross = covariance::crossCovarianceFromMaps(
      leftMap.view(), sourceCross.view(), rightMap.view(), reductionLimits);
  if (!require(transformedCross.hasValue() &&
                   transformedCross.value().at(0U, 0U) == 3.0,
               "installed cross-covariance transformation API failed")) {
    return 1;
  }

  std::cout << "VirtualSensorMath " << version::string << " consumer passed\n";
  return 0;
}
