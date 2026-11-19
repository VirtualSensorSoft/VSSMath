#include "Test.hpp"

#include <array>
#include <limits>
#include <span>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/TensorOperations.hpp>

int main() {
  using namespace vss::math;
  const Limits ample{128, 8192, 1024};
  const std::array<Scalar, 4> a{1, 2, 3, 4};
  const std::array<Scalar, 4> b{5, 6, 7, 8};
  const std::array<Index, 2> extents{2, 2};
  const std::array<Index, 2> strides{2, 1};
  const TensorView left{a, extents, strides};
  const TensorView right{b, extents, strides};

  const std::array<Index, 2> left_labels{0, 1};
  const std::array<Index, 2> right_labels{1, 2};
  const std::array<IndexList, 2> labels{left_labels, right_labels};
  const std::array<TensorView, 2> terms{left, right};
  const std::array<Index, 2> output_labels{0, 2};
  const auto product = einsum(terms, labels, output_labels, ample);
  VSS_CHECK(product.hasValue());
  const std::array<Scalar, 4> expected{19, 22, 43, 50};
  for (Extent i = 0; i < expected.size(); ++i) {
    VSS_CHECK(product.value().data()[i] == expected[i]);
  }
  VSS_CHECK(einsum(terms, labels, output_labels, Limits{1, 8192, 1024})
                .status()
                .code() == ErrorCode::ResourceLimit);
  VSS_CHECK(einsum(terms, labels, output_labels, Limits{128, 8192, 1})
                .status()
                .code() == ErrorCode::ResourceLimit);

  const std::array<Index, 2> diagonal_labels{7, 7};
  const std::array<IndexList, 1> diagonal_index{diagonal_labels};
  const std::array<TensorView, 1> diagonal_term{left};
  const auto diagonal =
      einsum(diagonal_term, diagonal_index, std::span<const Index>{}, ample);
  VSS_CHECK(diagonal.hasValue());
  VSS_CHECK(diagonal.value().shape().rank() == 0);
  VSS_CHECK(diagonal.value().data()[0] == 5);

  VSS_CHECK(
      einsum(terms, labels, std::array<Index, 1>{9}, ample).status().code() ==
      ErrorCode::IndexOutOfRange);
  VSS_CHECK(einsum(terms, labels, std::array<Index, 2>{0, 0}, ample)
                .status()
                .code() == ErrorCode::InvalidDomain);

  const std::array<Index, 2> mismatched_extents{3, 2};
  const std::array<Scalar, 6> mismatch_values{1, 2, 3, 4, 5, 6};
  const TensorView mismatched{mismatch_values, mismatched_extents, strides};
  const std::array<TensorView, 2> mismatch_terms{left, mismatched};
  VSS_CHECK(
      einsum(mismatch_terms, labels, output_labels, ample).status().code() ==
      ErrorCode::DimensionMismatch);

  const std::array<Index, 1> short_labels{0};
  const std::array<IndexList, 2> wrong_rank_labels{short_labels, right_labels};
  VSS_CHECK(
      einsum(terms, wrong_rank_labels, output_labels, ample).status().code() ==
      ErrorCode::DimensionMismatch);

  const std::array<Index, 2> invalid_strides{3, 1};
  const TensorView invalid_view{a, extents, invalid_strides};
  const std::array<TensorView, 1> invalid_terms{invalid_view};
  const std::array<IndexList, 1> invalid_labels{left_labels};
  VSS_CHECK(einsum(invalid_terms, invalid_labels, left_labels, ample)
                .status()
                .code() == ErrorCode::InvalidExtent);

  const std::array<Scalar, 4> nonfinite_values{
      1, 2, 3, std::numeric_limits<Scalar>::infinity()};
  const TensorView nonfinite{nonfinite_values, extents, strides};
  const std::array<TensorView, 1> nonfinite_terms{nonfinite};
  VSS_CHECK(
      einsum(nonfinite_terms, diagonal_index, std::span<const Index>{}, ample)
          .status()
          .code() == ErrorCode::NonFiniteInput);
}
