#include "Test.hpp"

#include <array>
#include <limits>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/Sparse.hpp>

namespace {

using namespace vss::math;

Limits limits() {
  return {.elements = 4096U, .bytes = 262144U, .evaluations = 0U};
}

void coalescesDuplicatesAndDropsExactCancellation() {
  // VSS-MATH-TRACEABILITY: SP-01
  const std::array<linear::Triplet, 5U> entries{{{1U, 2U, 4.0},
                                                 {0U, 1U, 1.0},
                                                 {0U, 1U, 2.0},
                                                 {1U, 0U, 3.0},
                                                 {0U, 1U, -3.0}}};
  const auto result = linear::sparseCSRFromTriplets(2U, 3U, entries, limits());
  VSS_CHECK(result.hasValue());
  const auto offsets = result.value().row_offsets.view();
  const auto columns = result.value().columns_by_entry.view();
  const auto values = result.value().values.view();
  VSS_CHECK(offsets.size() == 3U && offsets[0U] == 0U && offsets[1U] == 0U &&
            offsets[2U] == 2U);
  VSS_CHECK(columns[0U] == 0U && columns[1U] == 2U);
  VSS_CHECK(values.at(0U) == 3.0 && values.at(1U) == 4.0);
}

void acceptsEmptyRowsAndZeroRowMatrices() {
  // VSS-MATH-TRACEABILITY: SP-02
  const std::array<linear::Triplet, 1U> entries{{{2U, 1U, 3.0}}};
  const auto sparse = linear::sparseCSRFromTriplets(4U, 2U, entries, limits());
  VSS_CHECK(sparse.hasValue());
  const Vector input(std::array<Scalar, 2U>{2.0, 2.0});
  const auto output = linear::apply(sparse.value(), input.view(), limits());
  VSS_CHECK(output.hasValue());
  VSS_CHECK(output.value().size() == 4U);
  VSS_CHECK(output.value().at(0U) == 0.0 && output.value().at(1U) == 0.0 &&
            output.value().at(2U) == 6.0 && output.value().at(3U) == 0.0);

  const auto empty = linear::sparseCSRFromTriplets(0U, 2U, {}, limits());
  VSS_CHECK(empty.hasValue());
  VSS_CHECK(empty.value().row_offsets.view().size() == 1U);
  const auto empty_result =
      linear::apply(empty.value(), input.view(), limits());
  VSS_CHECK(empty_result.hasValue() && empty_result.value().empty());
}

void validatesCanonicalCSRBeforeApplying() {
  // VSS-MATH-TRACEABILITY: SP-03
  const std::array<Index, 2U> offsets{{0U, 1U}};
  const std::array<Index, 1U> columns{{0U}};
  const std::array<Scalar, 1U> stored_zero{{0.0}};
  const auto row_offsets = Array<Index>::copy(offsets, limits());
  const auto column_indices = Array<Index>::copy(columns, limits());
  const Vector values(stored_zero);
  VSS_CHECK(row_offsets.hasValue() && column_indices.hasValue());
  const linear::CSR malformed{1U, 1U, row_offsets.value(),
                              column_indices.value(), values};
  const Vector input(std::array<Scalar, 1U>{1.0});
  const auto result = linear::apply(malformed, input.view(), limits());
  VSS_CHECK(!result.hasValue());
  VSS_CHECK(result.status().code() == ErrorCode::InvalidDomain);

  const Vector wrong_size(std::array<Scalar, 2U>{1.0, 2.0});
  const auto valid = linear::sparseCSRFromTriplets(
      1U, 1U, std::span<const linear::Triplet>{}, limits());
  VSS_CHECK(valid.hasValue());
  const auto shape = linear::apply(valid.value(), wrong_size.view(), limits());
  VSS_CHECK(!shape.hasValue());
  VSS_CHECK(shape.status().code() == ErrorCode::DimensionMismatch);
}

void coalescedEntryActsAsDenseMatrixEntry() {
  // VSS-MATH-TRACEABILITY: SP-01
  const std::array<linear::Triplet, 2U> entries{{{0U, 0U, 1.0}, {0U, 0U, 2.0}}};
  const auto sparse = linear::sparseCSRFromTriplets(1U, 1U, entries, limits());
  VSS_CHECK(sparse.hasValue());
  const Vector input(std::array<Scalar, 1U>{2.0});
  const auto result = linear::apply(sparse.value(), input.view(), limits());
  VSS_CHECK(result.hasValue());
  VSS_CHECK(result.value().at(0U) == 6.0);
}

void rejectsMalformedInputsAndEnforcesLimits() {
  // VSS-MATH-TRACEABILITY: SP-05
  const std::array<linear::Triplet, 1U> bad_index{{{2U, 0U, 1.0}}};
  const auto index = linear::sparseCSRFromTriplets(2U, 1U, bad_index, limits());
  VSS_CHECK(!index.hasValue());
  VSS_CHECK(index.status().code() == ErrorCode::IndexOutOfRange);

  const std::array<linear::Triplet, 1U> bad_value{
      {{0U, 0U, std::numeric_limits<Scalar>::infinity()}}};
  const auto nonfinite =
      linear::sparseCSRFromTriplets(1U, 1U, bad_value, limits());
  VSS_CHECK(!nonfinite.hasValue());
  VSS_CHECK(nonfinite.status().code() == ErrorCode::NonFiniteInput);

  const auto resource = linear::sparseCSRFromTriplets(
      4U, 4U, std::span<const linear::Triplet>{},
      Limits{.elements = 1U, .bytes = 8U, .evaluations = 0U});
  VSS_CHECK(!resource.hasValue());
  VSS_CHECK(resource.status().code() == ErrorCode::ResourceLimit);
}

void accountsForOverlappingConstructionPeak() {
  const std::array<linear::Triplet, 1U> entry{{{0U, 0U, 2.0}}};
  const auto exact = linear::sparseCSRFromTriplets(
      1U, 1U, entry, Limits{.elements = 8U, .bytes = 64U, .evaluations = 0U});
  VSS_CHECK(exact.hasValue());

  const auto one_element_short = linear::sparseCSRFromTriplets(
      1U, 1U, entry, Limits{.elements = 7U, .bytes = 64U, .evaluations = 0U});
  VSS_CHECK(!one_element_short.hasValue());
  VSS_CHECK(one_element_short.status().code() == ErrorCode::ResourceLimit);

  const auto one_byte_short = linear::sparseCSRFromTriplets(
      1U, 1U, entry, Limits{.elements = 8U, .bytes = 63U, .evaluations = 0U});
  VSS_CHECK(!one_byte_short.hasValue());
  VSS_CHECK(one_byte_short.status().code() == ErrorCode::ResourceLimit);
}

void scaledRowActionDefersOverflowUntilFinalResult() {
  // VSS-MATH-TRACEABILITY: SP-04
  const Scalar maximum = std::numeric_limits<Scalar>::max();
  const std::array<linear::Triplet, 2U> cancelling{
      {{0U, 0U, maximum}, {0U, 1U, maximum}}};
  const auto sparse =
      linear::sparseCSRFromTriplets(1U, 2U, cancelling, limits());
  VSS_CHECK(sparse.hasValue());
  const Vector cancellation(std::array<Scalar, 2U>{2.0, -2.0});
  const auto exact_zero =
      linear::apply(sparse.value(), cancellation.view(), limits());
  VSS_CHECK(exact_zero.hasValue());
  VSS_CHECK(exact_zero.value().at(0U) == 0.0);

  const Vector overflow(std::array<Scalar, 2U>{2.0, 2.0});
  const auto nonfinite =
      linear::apply(sparse.value(), overflow.view(), limits());
  VSS_CHECK(!nonfinite.hasValue());
  VSS_CHECK(nonfinite.status().code() == ErrorCode::NonFiniteResult);
}

} // namespace

int main() {
  coalescesDuplicatesAndDropsExactCancellation();
  coalescedEntryActsAsDenseMatrixEntry();
  acceptsEmptyRowsAndZeroRowMatrices();
  rejectsMalformedInputsAndEnforcesLimits();
  validatesCanonicalCSRBeforeApplying();
  scaledRowActionDefersOverflowUntilFinalResult();
  accountsForOverlappingConstructionPeak();
  return 0;
}
