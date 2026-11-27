#include "Test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <utility>
#include <vss/math/Array.hpp>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorDecomposition.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/linear/SolvePolicy.hpp>

namespace {

using namespace vss::math;

constexpr Limits ample{
    .elements = 1'000'000, .bytes = 64'000'000, .evaluations = 0};

std::span<const Scalar> tensorValues(const Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

std::span<Scalar> tensorValues(Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}

SolvePolicy policy() {
  SolvePolicy result{};
  result.rank_abs = 1e-14;
  result.rank_rel = 1e-12;
  result.iteration.max_iterations = 256;
  result.iteration.abs = 1e-12;
  result.iteration.rel = 1e-12;
  return result;
}

Tensor makeTensor(std::span<const Extent> extents,
                  std::span<const Scalar> values) {
  const auto shape = TensorShape::create(extents);
  VSS_CHECK(shape.hasValue());
  const auto tensor =
      Tensor::create(shape.value(), 0.0, TensorLimits{values.size()});
  VSS_CHECK(tensor.hasValue());
  VSS_CHECK(tensor.value().size() == values.size());
  auto result = std::move(tensor).value();
  for (Extent i = 0; i < values.size(); ++i) {
    tensorValues(result)[i] = values[i];
  }
  return result;
}

void checkTensor(const Tensor& actual, std::span<const Extent> shape,
                 std::span<const Scalar> values, Scalar tolerance = 1e-10) {
  VSS_CHECK(actual.shape().rank() == shape.size());
  VSS_CHECK(actual.size() == values.size());
  for (Extent axis = 0; axis < shape.size(); ++axis) {
    VSS_CHECK(actual.shape().extent(axis) == shape[axis]);
  }
  for (Extent i = 0; i < values.size(); ++i) {
    if (std::abs(tensorValues(actual)[i] - values[i]) > tolerance) {
      std::cerr << "tensor mismatch at " << i << ": got "
                << tensorValues(actual)[i] << ", expected " << values[i]
                << '\n';
    }
    VSS_CHECK(std::abs(tensorValues(actual)[i] - values[i]) <= tolerance);
  }
}

void fullRankRectangularRoundTrip() {
  const auto input =
      makeTensor(std::array<Extent, 3>{2, 3, 2},
                 std::array<Scalar, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  const auto decomposition =
      tuckerHOSVD(input, std::array<Extent, 3>{2, 3, 2}, policy(), ample);
  if (!decomposition.hasValue()) {
    std::cerr << "full-rank HOSVD failed: " << decomposition.status().message()
              << '\n';
  }
  VSS_CHECK(decomposition.hasValue());
  VSS_CHECK(decomposition.value().axes.view().size() == 3);
  const auto rebuilt = reconstructTucker(decomposition.value(), ample);
  VSS_CHECK(rebuilt.hasValue());
  checkTensor(rebuilt.value(), std::array<Extent, 3>{2, 3, 2},
              std::array<Scalar, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  VSS_CHECK(decomposition.value().residual_norm < 1e-10);
}

void fullRankAboveThinSvdExtent() {
  std::array<Scalar, 20> values{};
  const auto valueView = std::span<Scalar>{values};
  for (Index i = 0; i < valueView.size(); ++i) {
    valueView[i] = static_cast<Scalar>(i) - 7.0;
  }
  const auto input = makeTensor(std::array<Extent, 3>{5, 2, 2}, values);
  const auto factors =
      tuckerHOSVD(input, std::array<Extent, 3>{5, 2, 2}, policy(), ample);
  VSS_CHECK(factors.hasValue());
  VSS_CHECK(factors.value().axes.view()[0].columns() == 5);
  const auto rebuilt = reconstructTucker(factors.value(), ample);
  VSS_CHECK(rebuilt.hasValue());
  checkTensor(rebuilt.value(), std::array<Extent, 3>{5, 2, 2}, values);
}

void unequalRankOneOuterProduct() {
  // Independent direct outer-product fixture: 2 x 3 x 2, multilinear rank 1.
  const auto input = makeTensor(
      std::array<Extent, 3>{2, 3, 2},
      std::array<Scalar, 12>{10, 15, 20, 30, 30, 45, 20, 30, 40, 60, 60, 90});
  const auto factors =
      tuckerHOSVD(input, std::array<Extent, 3>{1, 1, 1}, policy(), ample);
  VSS_CHECK(factors.hasValue());
  const auto rebuilt = reconstructTucker(factors.value(), ample);
  VSS_CHECK(rebuilt.hasValue());
  checkTensor(
      rebuilt.value(), std::array<Extent, 3>{2, 3, 2},
      std::array<Scalar, 12>{10, 15, 20, 30, 30, 45, 20, 30, 40, 60, 60, 90},
      1e-9);
  VSS_CHECK(factors.value().residual_norm < 1e-9);
}

void modePermutationRoundTrip() {
  const auto input =
      makeTensor(std::array<Extent, 3>{2, 3, 2},
                 std::array<Scalar, 12>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  std::array<Scalar, 12> permuted_values{};
  auto permuted_view = std::span<Scalar>{permuted_values};
  for (Index first = 0; first < 2; ++first) {
    for (Index second = 0; second < 2; ++second) {
      for (Index third = 0; third < 3; ++third) {
        const Index output = ((first * 2 + second) * 3) + third;
        const Index source = ((second * 3 + third) * 2) + first;
        permuted_view[output] = tensorValues(input)[source];
      }
    }
  }
  const auto permuted =
      makeTensor(std::array<Extent, 3>{2, 2, 3}, permuted_values);
  const auto factors =
      tuckerHOSVD(permuted, std::array<Extent, 3>{2, 2, 3}, policy(), ample);
  VSS_CHECK(factors.hasValue());
  const auto rebuilt = reconstructTucker(factors.value(), ample);
  VSS_CHECK(rebuilt.hasValue());
  checkTensor(rebuilt.value(), std::array<Extent, 3>{2, 2, 3}, permuted_values);
}

void truncatedOrthogonalComponentsAndRepeatedSpectrum() {
  const auto orthogonal =
      makeTensor(std::array<Extent, 3>{2, 2, 2},
                 std::array<Scalar, 8>{3, 0, 0, 0, 0, 0, 0, 4});
  const auto truncated =
      tuckerHOSVD(orthogonal, std::array<Extent, 3>{1, 1, 1}, policy(), ample);
  VSS_CHECK(truncated.hasValue());
  const auto rebuilt = reconstructTucker(truncated.value(), ample);
  VSS_CHECK(rebuilt.hasValue());
  Scalar squared_error = 0;
  for (Extent i = 0; i < orthogonal.size(); ++i) {
    const Scalar difference =
        tensorValues(orthogonal)[i] - tensorValues(rebuilt.value())[i];
    squared_error = std::fma(difference, difference, squared_error);
  }
  VSS_CHECK(std::abs(std::sqrt(squared_error) - 3.0) < 1e-10);
  VSS_CHECK(std::abs(truncated.value().residual_norm -
                     std::sqrt(squared_error)) < 1e-10);
  // HOSVD's actual residual must respect the documented square-root tail
  // bound. Each of the three unfoldings discards a singular value of 3.
  VSS_CHECK(squared_error <= 27.0 + 1e-10);
  // Equal singular values make individual bases non-unique; check the actual
  // approximation error, not a particular basis orientation.
  const auto repeated =
      makeTensor(std::array<Extent, 3>{2, 2, 2},
                 std::array<Scalar, 8>{1, 0, 0, 0, 0, 0, 0, 1});
  const auto tied =
      tuckerHOSVD(repeated, std::array<Extent, 3>{1, 1, 1}, policy(), ample);
  VSS_CHECK(tied.hasValue());
  const auto tied_rebuilt = reconstructTucker(tied.value(), ample);
  VSS_CHECK(tied_rebuilt.hasValue());
  Scalar tied_error_squared = 0;
  for (Extent i = 0; i < repeated.size(); ++i) {
    const Scalar difference =
        tensorValues(repeated)[i] - tensorValues(tied_rebuilt.value())[i];
    tied_error_squared = std::fma(difference, difference, tied_error_squared);
  }
  VSS_CHECK(std::sqrt(tied_error_squared) <= std::sqrt(3.0) + 1e-10);
}

void scalarEmptyAndInvalidInputs() {
  const auto scalar =
      makeTensor(std::span<const Extent>{}, std::array<Scalar, 1>{-2.5});
  const auto scalar_factors =
      tuckerHOSVD(scalar, std::span<const Extent>{}, policy(), ample);
  VSS_CHECK(scalar_factors.hasValue());
  VSS_CHECK(scalar_factors.value().axes.view().empty());
  const auto scalar_rebuilt = reconstructTucker(scalar_factors.value(), ample);
  VSS_CHECK(scalar_rebuilt.hasValue());
  VSS_CHECK(tensorValues(scalar_rebuilt.value())[0] == -2.5);
  VSS_CHECK(scalar_factors.value().residual_norm == 0.0);

  const auto empty =
      makeTensor(std::array<Extent, 3>{2, 0, 3}, std::span<const Scalar>{});
  const auto empty_factors =
      tuckerHOSVD(empty, std::array<Extent, 3>{1, 0, 2}, policy(), ample);
  VSS_CHECK(empty_factors.hasValue());
  const auto empty_rebuilt = reconstructTucker(empty_factors.value(), ample);
  VSS_CHECK(empty_rebuilt.hasValue());
  VSS_CHECK(empty_rebuilt.value().shape().extent(0) == 2);
  VSS_CHECK(empty_rebuilt.value().shape().extent(1) == 0);
  VSS_CHECK(empty_rebuilt.value().shape().extent(2) == 3);
  VSS_CHECK(empty_factors.value().residual_norm == 0.0);

  const auto nonempty = makeTensor(std::array<Extent, 2>{2, 2},
                                   std::array<Scalar, 4>{1, 2, 3, 4});
  VSS_CHECK(tuckerHOSVD(nonempty, std::array<Extent, 1>{1}, policy(), ample)
                .status()
                .code() == ErrorCode::DimensionMismatch);
  VSS_CHECK(tuckerHOSVD(nonempty, std::array<Extent, 2>{0, 1}, policy(), ample)
                .status()
                .code() == ErrorCode::InvalidDomain);
  VSS_CHECK(tuckerHOSVD(nonempty, std::array<Extent, 2>{3, 1}, policy(), ample)
                .status()
                .code() == ErrorCode::IndexOutOfRange);
  auto nonfinite = nonempty;
  tensorValues(nonfinite)[0] = std::numeric_limits<Scalar>::infinity();
  VSS_CHECK(tuckerHOSVD(nonfinite, std::array<Extent, 2>{1, 1}, policy(), ample)
                .status()
                .code() == ErrorCode::NonFiniteInput);
  VSS_CHECK(tuckerHOSVD(nonempty, std::array<Extent, 2>{1, 1}, policy(),
                        Limits{.elements = 0, .bytes = 0, .evaluations = 0})
                .status()
                .code() == ErrorCode::ResourceLimit);

  const auto valid =
      tuckerHOSVD(nonempty, std::array<Extent, 2>{1, 1}, policy(), ample);
  VSS_CHECK(valid.hasValue());
  auto bad_provenance = valid.value();
  bad_provenance.residual_norm = std::numeric_limits<Scalar>::infinity();
  VSS_CHECK(reconstructTucker(bad_provenance, ample).status().code() ==
            ErrorCode::InvalidDomain);

  const Matrix wrong_axis(1, 1);
  const auto wrong_axes =
      Array<Matrix>::copy(std::array<Matrix, 1>{wrong_axis}, ample);
  VSS_CHECK(wrong_axes.hasValue());
  const TuckerFactors malformed{.core = valid.value().core,
                                .axes = wrong_axes.value(),
                                .shape = valid.value().shape,
                                .residual_norm = 0.0};
  VSS_CHECK(reconstructTucker(malformed, ample).status().code() ==
            ErrorCode::DimensionMismatch);

  std::array<Matrix, 2> nonfinite_axes{valid.value().axes.view()[0],
                                       valid.value().axes.view()[1]};
  nonfinite_axes[0].at(0U, 0U) = std::numeric_limits<Scalar>::infinity();
  const auto nonfinite_array = Array<Matrix>::copy(nonfinite_axes, ample);
  VSS_CHECK(nonfinite_array.hasValue());
  const TuckerFactors nonfinite_factor{.core = valid.value().core,
                                       .axes = nonfinite_array.value(),
                                       .shape = valid.value().shape,
                                       .residual_norm = 0.0};
  VSS_CHECK(reconstructTucker(nonfinite_factor, ample).status().code() ==
            ErrorCode::NonFiniteInput);
}

void exactPeakElementAndByteCaps() {
  std::array<Scalar, 4096> values{};
  const auto valueView = std::span<Scalar>{values};
  for (Index i = 0; i < 16; ++i) {
    valueView[i * 273] = static_cast<Scalar>(i + 1);
  }
  const auto input = makeTensor(std::array<Extent, 3>{16, 16, 16}, values);
  const auto ranks = std::array<Extent, 3>{8, 8, 8};
  const auto succeedsWithElements = [&](std::size_t cap) {
    return tuckerHOSVD(
        input, ranks, policy(),
        Limits{.elements = cap, .bytes = 64'000'000, .evaluations = 0});
  };
  std::size_t low = 0;
  std::size_t high = 1'000'000;
  while (low < high) {
    const auto middle = low + ((high - low) / 2);
    const auto attempt = succeedsWithElements(middle);
    if (attempt.hasValue()) {
      high = middle;
    } else {
      VSS_CHECK(attempt.status().code() == ErrorCode::ResourceLimit);
      low = middle + 1;
    }
  }
  VSS_CHECK(succeedsWithElements(low).hasValue());
  VSS_CHECK(low > 0);
  const auto below_elements = succeedsWithElements(low - 1);
  VSS_CHECK(!below_elements.hasValue());
  VSS_CHECK(below_elements.status().code() == ErrorCode::ResourceLimit);

  const auto succeedsWithBytes = [&](std::size_t cap) {
    return tuckerHOSVD(
        input, ranks, policy(),
        Limits{.elements = 1'000'000, .bytes = cap, .evaluations = 0});
  };
  low = 0;
  high = 64'000'000;
  while (low < high) {
    const auto middle = low + ((high - low) / 2);
    const auto attempt = succeedsWithBytes(middle);
    if (attempt.hasValue()) {
      high = middle;
    } else {
      VSS_CHECK(attempt.status().code() == ErrorCode::ResourceLimit);
      low = middle + 1;
    }
  }
  VSS_CHECK(succeedsWithBytes(low).hasValue());
  VSS_CHECK(low > 0);
  const auto below_bytes = succeedsWithBytes(low - 1);
  VSS_CHECK(!below_bytes.hasValue());
  VSS_CHECK(below_bytes.status().code() == ErrorCode::ResourceLimit);
}

} // namespace

int main() {
  fullRankRectangularRoundTrip();
  fullRankAboveThinSvdExtent();
  unequalRankOneOuterProduct();
  modePermutationRoundTrip();
  truncatedOrthogonalComponentsAndRepeatedSpectrum();
  scalarEmptyAndInvalidInputs();
  exactPeakElementAndByteCaps();
}

// VSS-MATH-TRACEABILITY: C105 CASE-C105.tucker-full-rank-reconstruction
// VSS-MATH-TRACEABILITY: C105 CASE-C105.tucker-truncated-error-bound
// VSS-MATH-TRACEABILITY: C105 CASE-C105.peak-live-workspace
