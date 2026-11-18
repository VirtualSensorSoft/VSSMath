#include "Test.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorElementwise.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorOperations.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/Vector.hpp>

namespace {

using namespace vss::math;

std::span<Scalar> elements(Tensor& value) {
  return {value.data(), value.size()};
}

std::span<const Scalar> elements(const Tensor& value) {
  return {value.data(), value.size()};
}

std::span<Scalar> elements(Matrix& value) {
  return {value.data(), value.size()};
}

Tensor tensor(std::span<const Extent> extents, std::span<const Scalar> values) {
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  auto result = Tensor::create(shape.value(), 0.0, TensorLimits{values.size()});
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().size() == values.size());
  std::ranges::copy(values, elements(result.value()).begin());
  return std::move(result).value();
}

bool close(Scalar actual, Scalar expected, Scalar tolerance = 1.0e-12) {
  return std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected));
}

void checkCode(const auto& result, ErrorCode code) {
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == code);
}

void testHadamardAndDivide() {
  const Limits limits{.elements = 64U, .bytes = 4096U, .evaluations = 64U};
  const std::array<Scalar, 2> va{2.0, -3.0};
  const std::array<Scalar, 2> vb{4.0, 5.0};
  const Vector a{va};
  const Vector b{vb};
  const auto vectorProduct = hadamard(a.view(), b.view(), limits);
  VSS_CHECK(vectorProduct.hasValue());
  VSS_CHECK(vectorProduct.value().at(0U) == 8.0);
  VSS_CHECK(vectorProduct.value().at(1U) == -15.0);
  checkCode(hadamard(a.view(), Vector{1U}.view(), limits),
            ErrorCode::DimensionMismatch);
  Vector nonfiniteVector{1U};
  nonfiniteVector.at(0U) = std::numeric_limits<Scalar>::infinity();
  checkCode(hadamard(nonfiniteVector.view(), nonfiniteVector.view(), limits),
            ErrorCode::NonFiniteInput);
  checkCode(
      hadamard(a.view(), b.view(),
               Limits{.elements = 1U, .bytes = 4096U, .evaluations = 64U}),
      ErrorCode::ResourceLimit);
  checkCode(hadamard(a.view(), b.view(),
                     Limits{.elements = 64U, .bytes = 15U, .evaluations = 64U}),
            ErrorCode::ResourceLimit);
  Vector hugeVector{1U};
  Vector factorVector{1U};
  hugeVector.at(0U) = std::numeric_limits<Scalar>::max();
  factorVector.at(0U) = 2.0;
  checkCode(hadamard(hugeVector.view(), factorVector.view(), limits),
            ErrorCode::NonFiniteResult);

  Matrix ma{2U, 2U};
  Matrix mb{2U, 2U};
  const std::array<Scalar, 4> mva{2.0, -3.0, 0.0, 4.0};
  const std::array<Scalar, 4> mvb{4.0, 5.0, 6.0, -2.0};
  std::ranges::copy(mva, elements(ma).begin());
  std::ranges::copy(mvb, elements(mb).begin());
  const auto matrixProduct = hadamard(ma.view(), mb.view(), limits);
  VSS_CHECK(matrixProduct.hasValue());
  VSS_CHECK(matrixProduct.value().at(0U, 0U) == 8.0);
  VSS_CHECK(matrixProduct.value().at(0U, 1U) == -15.0);
  VSS_CHECK(matrixProduct.value().at(1U, 0U) == 0.0);
  VSS_CHECK(matrixProduct.value().at(1U, 1U) == -8.0);
  Matrix badShape{1U, 4U};
  checkCode(hadamard(ma.view(), badShape.view(), limits),
            ErrorCode::DimensionMismatch);
  Matrix badMatrix{1U, 1U};
  badMatrix.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  checkCode(hadamard(badMatrix.view(), badMatrix.view(), limits),
            ErrorCode::NonFiniteInput);
  checkCode(
      hadamard(ma.view(), mb.view(),
               Limits{.elements = 1U, .bytes = 4096U, .evaluations = 64U}),
      ErrorCode::ResourceLimit);
  checkCode(hadamard(ma.view(), mb.view(),
                     Limits{.elements = 64U, .bytes = 31U, .evaluations = 64U}),
            ErrorCode::ResourceLimit);

  const std::array<Extent, 2> shape2{2U, 2U};
  const std::array<Scalar, 4> taValues{-2.0, -2.0, -2.0, -2.0};
  const std::array<Scalar, 4> tbValues{3.0, 3.0, 3.0, 3.0};
  const Tensor ta = tensor(shape2, taValues);
  const Tensor tb = tensor(shape2, tbValues);
  const auto tensorProduct = hadamard(ta, tb, limits);
  VSS_CHECK(tensorProduct.hasValue());
  VSS_CHECK(std::ranges::all_of(elements(tensorProduct.value()),
                                [](Scalar value) { return value == -6.0; }));
  const std::array<Extent, 3> shape3{2U, 1U, 2U};
  const std::array<Scalar, 4> threeA{2.0, 2.0, 2.0, 2.0};
  const std::array<Scalar, 4> threeB{-3.0, -3.0, -3.0, -3.0};
  const auto product3 =
      hadamard(tensor(shape3, threeA), tensor(shape3, threeB), limits);
  VSS_CHECK(product3.hasValue());
  VSS_CHECK(product3.value().shape().rank() == 3U);
  checkCode(hadamard(ta,
                     tensor(std::array<Extent, 1>{4U},
                            std::array<Scalar, 4>{1, 1, 1, 1}),
                     limits),
            ErrorCode::DimensionMismatch);
  const std::array<Scalar, 4> badTensorValues{
      1.0, 2.0, std::numeric_limits<Scalar>::infinity(), 4.0};
  checkCode(hadamard(ta, tensor(shape2, badTensorValues), limits),
            ErrorCode::NonFiniteInput);
  checkCode(
      hadamard(ta, tb,
               Limits{.elements = 1U, .bytes = 4096U, .evaluations = 64U}),
      ErrorCode::ResourceLimit);
  checkCode(hadamard(ta, tb,
                     Limits{.elements = 64U, .bytes = 47U, .evaluations = 64U}),
            ErrorCode::ResourceLimit);
  checkCode(
      hadamard(
          tensor(std::array<Extent, 1>{1U},
                 std::array<Scalar, 1>{std::numeric_limits<Scalar>::max()}),
          tensor(std::array<Extent, 1>{1U}, std::array<Scalar, 1>{2.0}),
          limits),
      ErrorCode::NonFiniteResult);

  const std::array<Scalar, 3> divideValues{2.0, -4.0, 0.0};
  const Tensor divInput = tensor(std::array<Extent, 1>{3U}, divideValues);
  const auto quotient = divide(divInput, 2.0, limits);
  VSS_CHECK(quotient.hasValue());
  VSS_CHECK(elements(quotient.value())[0] == 1.0);
  VSS_CHECK(elements(quotient.value())[1] == -2.0);
  VSS_CHECK(elements(quotient.value())[2] == 0.0);
  const std::array<Scalar, 1> scalarValue{-6.0};
  const auto scalarQuotient = divide(tensor({}, scalarValue), -3.0, limits);
  VSS_CHECK(scalarQuotient.hasValue());
  VSS_CHECK(scalarQuotient.value().shape().rank() == 0U);
  VSS_CHECK(elements(scalarQuotient.value())[0] == 2.0);
  const Tensor empty = tensor(std::array<Extent, 3>{2U, 0U, 3U}, {});
  const auto emptyQuotient = divide(empty, 2.0, limits);
  VSS_CHECK(emptyQuotient.hasValue());
  VSS_CHECK(emptyQuotient.value().shape().extent(0U) == 2U);
  VSS_CHECK(emptyQuotient.value().shape().extent(1U) == 0U);
  VSS_CHECK(emptyQuotient.value().shape().extent(2U) == 3U);
  checkCode(divide(empty, std::numeric_limits<Scalar>::infinity(), limits),
            ErrorCode::NonFiniteInput);
  checkCode(divide(tensor(std::array<Extent, 1>{1U},
                          std::array<Scalar, 1>{
                              std::numeric_limits<Scalar>::quiet_NaN()}),
                   2.0, limits),
            ErrorCode::NonFiniteInput);
  checkCode(divide(divInput, 0.0, limits), ErrorCode::InvalidDomain);
  checkCode(divide(divInput, -0.0, limits), ErrorCode::InvalidDomain);
  checkCode(divide(divInput, 2.0,
                   Limits{.elements = 2U, .bytes = 4096U, .evaluations = 64U}),
            ErrorCode::ResourceLimit);
  checkCode(divide(divInput, 2.0,
                   Limits{.elements = 64U, .bytes = 31U, .evaluations = 64U}),
            ErrorCode::ResourceLimit);
  const Scalar max = std::numeric_limits<Scalar>::max();
  checkCode(
      divide(tensor(std::array<Extent, 1>{1U}, std::array<Scalar, 1>{max}), 0.5,
             limits),
      ErrorCode::NonFiniteResult);
  const Scalar least = std::numeric_limits<Scalar>::denorm_min();
  const auto reciprocal =
      divide(tensor(std::array<Extent, 1>{1U}, std::array<Scalar, 1>{least}),
             least, limits);
  VSS_CHECK(reciprocal.hasValue() && elements(reciprocal.value())[0] == 1.0);
  const auto tie =
      divide(tensor(std::array<Extent, 1>{1U}, std::array<Scalar, 1>{least}),
             2.0, limits);
  VSS_CHECK(tie.hasValue() && elements(tie.value())[0] == 0.0);
  VSS_CHECK(!std::signbit(elements(tie.value())[0]));
  const std::array<Extent, 2> overflowShape{std::numeric_limits<Extent>::max(),
                                            2U};
  const auto invalidShape = TensorShape::create(overflowShape);
  VSS_CHECK(!invalidShape.hasValue());
  VSS_CHECK(invalidShape.status().code() == ErrorCode::SizeOverflow);
}

void testFrobeniusOperations() {
  Matrix matrix{2U, 2U};
  matrix.at(0U, 0U) = 2.0;
  matrix.at(0U, 1U) = -3.0;
  matrix.at(1U, 0U) = 0.0;
  matrix.at(1U, 1U) = 4.0;
  const auto matrixNorm = frobenius(matrix.view());
  VSS_CHECK(matrixNorm.hasValue() &&
            close(matrixNorm.value(), std::sqrt(29.0)));
  Matrix largeNormInput{1U, 2U};
  largeNormInput.at(0U, 0U) = 3.0e200;
  largeNormInput.at(0U, 1U) = 4.0e200;
  VSS_CHECK(close(frobenius(largeNormInput.view()).value(), 5.0e200, 1.0e-15));
  Matrix emptyNormInput{0U, 2U};
  VSS_CHECK(frobenius(emptyNormInput.view()).value() == 0.0);
  const auto matrixInner = frobeniusInner(matrix.view(), matrix.view());
  VSS_CHECK(matrixInner.hasValue() && matrixInner.value() == 29.0);
  Matrix left{2U, 2U};
  Matrix right{2U, 2U};
  const std::array<Scalar, 4> lv{1.0, 2.0, 3.0, 4.0};
  const std::array<Scalar, 4> rv{5.0, 6.0, 7.0, 8.0};
  std::ranges::copy(lv, elements(left).begin());
  std::ranges::copy(rv, elements(right).begin());
  VSS_CHECK(frobeniusInner(left.view(), right.view()).value() == 70.0);
  Matrix negativeLeft{1U, 2U};
  Matrix negativeRight{1U, 2U};
  negativeLeft.at(0U, 0U) = -1.0;
  negativeLeft.at(0U, 1U) = 2.0;
  negativeRight.at(0U, 0U) = 3.0;
  negativeRight.at(0U, 1U) = -4.0;
  VSS_CHECK(frobeniusInner(negativeLeft.view(), negativeRight.view()).value() ==
            -11.0);
  checkCode(frobeniusInner(left.view(), Matrix{1U, 4U}.view()),
            ErrorCode::DimensionMismatch);
  Matrix empty{0U, 3U};
  const auto emptyInner = frobeniusInner(empty.view(), empty.view());
  VSS_CHECK(emptyInner.hasValue() && emptyInner.value() == 0.0 &&
            !std::signbit(emptyInner.value()));
  Matrix infinite{1U, 1U};
  infinite.at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  checkCode(frobenius(infinite.view()), ErrorCode::NonFiniteInput);
  checkCode(frobeniusInner(infinite.view(), infinite.view()),
            ErrorCode::NonFiniteInput);
  Matrix large{1U, 1U};
  Matrix twice{1U, 1U};
  large.at(0U, 0U) = std::numeric_limits<Scalar>::max();
  twice.at(0U, 0U) = 2.0;
  checkCode(frobeniusInner(large.view(), twice.view()),
            ErrorCode::NonFiniteResult);
  Matrix cancellationA{1U, 2U};
  Matrix cancellationB{1U, 2U};
  cancellationA.at(0U, 0U) = std::numeric_limits<Scalar>::max();
  cancellationA.at(0U, 1U) = std::numeric_limits<Scalar>::max();
  cancellationB.at(0U, 0U) = 2.0;
  cancellationB.at(0U, 1U) = -2.0;
  const auto cancellation =
      frobeniusInner(cancellationA.view(), cancellationB.view());
  VSS_CHECK(cancellation.hasValue() && cancellation.value() == 0.0 &&
            !std::signbit(cancellation.value()));

  const std::array<Extent, 2> shape{2U, 2U};
  const std::array<Scalar, 4> values{1.0, 2.0, 3.0, 4.0};
  const std::array<Scalar, 4> other{5.0, 6.0, 7.0, 8.0};
  const Tensor a = tensor(shape, values);
  const Tensor b = tensor(shape, other);
  VSS_CHECK(frobenius(a).value() == 5.4772255750516612);
  const Tensor emptyTensor = tensor(std::array<Extent, 3>{2U, 0U, 3U}, {});
  VSS_CHECK(frobenius(emptyTensor).value() == 0.0);
  VSS_CHECK(frobenius(tensor({}, std::array<Scalar, 1>{-3.0})).value() == 3.0);
  checkCode(frobenius(tensor(std::array<Extent, 1>{1U},
                             std::array<Scalar, 1>{
                                 std::numeric_limits<Scalar>::infinity()})),
            ErrorCode::NonFiniteInput);
  VSS_CHECK(frobeniusInner(a, b).value() == 70.0);
  const auto tensorCancellation = frobeniusInner(
      tensor(std::array<Extent, 1>{2U},
             std::array<Scalar, 2>{std::numeric_limits<Scalar>::max(),
                                   std::numeric_limits<Scalar>::max()}),
      tensor(std::array<Extent, 1>{2U}, std::array<Scalar, 2>{2.0, -2.0}));
  VSS_CHECK(tensorCancellation.hasValue() &&
            tensorCancellation.value() == 0.0 &&
            !std::signbit(tensorCancellation.value()));
  const auto emptyTensorInner = frobeniusInner(emptyTensor, emptyTensor);
  VSS_CHECK(emptyTensorInner.hasValue() && emptyTensorInner.value() == 0.0 &&
            !std::signbit(emptyTensorInner.value()));
  checkCode(frobeniusInner(a, tensor(std::array<Extent, 2>{1U, 4U}, values)),
            ErrorCode::DimensionMismatch);
  checkCode(frobenius(tensor(
                std::array<Extent, 1>{2U},
                std::array<Scalar, 2>{std::numeric_limits<Scalar>::max(),
                                      std::numeric_limits<Scalar>::max()})),
            ErrorCode::NonFiniteResult);
  checkCode(
      frobeniusInner(
          tensor({}, std::array<Scalar, 1>{std::numeric_limits<Scalar>::max()}),
          tensor({}, std::array<Scalar, 1>{2.0})),
      ErrorCode::NonFiniteResult);
  const auto rankZero = frobeniusInner(tensor({}, std::array<Scalar, 1>{3.0}),
                                       tensor({}, std::array<Scalar, 1>{-4.0}));
  VSS_CHECK(rankZero.hasValue() && rankZero.value() == -12.0);
  VSS_CHECK(frobenius(tensor({}, std::array<Scalar, 1>{-3.0})).value() == 3.0);
  const Scalar least = std::numeric_limits<Scalar>::denorm_min();
  const auto tiny = frobeniusInner(
      tensor(std::array<Extent, 1>{1U}, std::array<Scalar, 1>{least}),
      tensor(std::array<Extent, 1>{1U}, std::array<Scalar, 1>{least}));
  VSS_CHECK(tiny.hasValue() && tiny.value() == 0.0);
  const auto mixed = frobeniusInner(
      tensor(std::array<Extent, 1>{3U},
             std::array<Scalar, 3>{1.0e150, 1.0, -1.0e150}),
      tensor(std::array<Extent, 1>{3U}, std::array<Scalar, 3>{1.0, 1.0, 1.0}));
  VSS_CHECK(mixed.hasValue() && close(mixed.value(), 1.0, 1.0e-12));
}

void testTensorElementwise() {
  const Limits limits{.elements = 32U, .bytes = 4096U, .evaluations = 32U};
  const Tensor input =
      tensor(std::array<Extent, 1>{2U}, std::array<Scalar, 2>{1.0, 2.0});
  const std::array<Extent, 2> destination{2U, 3U};
  const std::array<Index, 1> map{0U};
  const auto broadcast = broadcastTo(input, destination, map, limits);
  VSS_CHECK(broadcast.hasValue());
  const std::array<Scalar, 6> expected{1, 1, 1, 2, 2, 2};
  VSS_CHECK(std::ranges::equal(elements(broadcast.value()), expected));
  const std::array<Index, 1> badAxis{2U};
  checkCode(broadcastTo(input, destination, badAxis, limits),
            ErrorCode::IndexOutOfRange);
  const std::array<Extent, 2> incompatible{3U, 3U};
  checkCode(broadcastTo(input, incompatible, map, limits),
            ErrorCode::DimensionMismatch);
  checkCode(
      broadcastTo(input, destination, map,
                  Limits{.elements = 1U, .bytes = 4096U, .evaluations = 32U}),
      ErrorCode::ResourceLimit);
  checkCode(
      broadcastTo(input, destination, map,
                  Limits{.elements = 32U, .bytes = 63U, .evaluations = 32U}),
      ErrorCode::ResourceLimit);
  checkCode(
      broadcastTo(input, destination, map,
                  Limits{.elements = 32U, .bytes = 64U, .evaluations = 32U}),
      ErrorCode::ResourceLimit);
  checkCode(
      broadcastTo(input, destination, map,
                  Limits{.elements = 32U, .bytes = 72U, .evaluations = 32U}),
      ErrorCode::ResourceLimit);
  const std::array<Scalar, 2> nonfiniteValues{
      1.0, std::numeric_limits<Scalar>::infinity()};
  checkCode(broadcastTo(tensor(std::array<Extent, 1>{2U}, nonfiniteValues),
                        destination, map, limits),
            ErrorCode::NonFiniteInput);
  const std::array<Index, 2> duplicateMap{0U, 0U};
  checkCode(broadcastTo(tensor(std::array<Extent, 2>{1U, 1U},
                               std::array<Scalar, 1>{3.0}),
                        destination, duplicateMap, limits),
            ErrorCode::InvalidDomain);
  checkCode(broadcastTo(input, destination, {}, limits),
            ErrorCode::DimensionMismatch);
  const Tensor scalar = tensor({}, std::array<Scalar, 1>{7.0});
  const auto scalarBroadcast = broadcastTo(scalar, destination, {}, limits);
  VSS_CHECK(scalarBroadcast.hasValue());
  VSS_CHECK(std::ranges::all_of(elements(scalarBroadcast.value()),
                                [](Scalar value) { return value == 7.0; }));

  std::vector<Scalar> observed;
  const UnaryFunction square = [&observed](Scalar x) -> Result<Scalar> {
    observed.push_back(x);
    return Result<Scalar>::success(x * x);
  };
  const auto transformed = transformElements(input, square, limits);
  VSS_CHECK(transformed.hasValue());
  VSS_CHECK(observed.size() == 2U && observed[0] == 1.0 && observed[1] == 2.0);
  VSS_CHECK(elements(transformed.value())[0] == 1.0 &&
            elements(transformed.value())[1] == 4.0);
  std::array<Index, 1> calls{0U};
  const UnaryFunction limited = [&calls](Scalar x) -> Result<Scalar> {
    ++calls.front();
    return Result<Scalar>::success(x);
  };
  checkCode(transformElements(
                input, limited,
                Limits{.elements = 1U, .bytes = 4096U, .evaluations = 32U}),
            ErrorCode::ResourceLimit);
  VSS_CHECK(calls.front() == 0U);
  checkCode(transformElements(
                input, limited,
                Limits{.elements = 32U, .bytes = 4096U, .evaluations = 1U}),
            ErrorCode::ResourceLimit);
  VSS_CHECK(calls.front() == 0U);
  const UnaryFunction failure =
      [calls = Index{0U}](Scalar x) mutable -> Result<Scalar> {
    if (calls++ == 1U) {
      return Result<Scalar>::failure(ErrorCode::InvalidDomain, "callback");
    }
    return Result<Scalar>::success(x);
  };
  checkCode(transformElements(input, failure, limits),
            ErrorCode::InvalidDomain);
  const UnaryFunction nonfinite = [](Scalar) -> Result<Scalar> {
    return Result<Scalar>::success(std::numeric_limits<Scalar>::infinity());
  };
  checkCode(transformElements(input, nonfinite, limits),
            ErrorCode::NonFiniteResult);
  const UnaryFunction throws = [](Scalar) -> Result<Scalar> {
    throw std::runtime_error("failure");
  };
  checkCode(transformElements(input, throws, limits),
            ErrorCode::CallbackFailure);
}

} // namespace

int main() {
  testHadamardAndDivide();
  testFrobeniusOperations();
  testTensorElementwise();
}
