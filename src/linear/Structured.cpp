#include "BandSolve.hpp"
#include "DenseWorkspace.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/Decomposition.hpp>
#include <vss/math/linear/RankAware.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/linear/Structured.hpp>
#include <vss/math/linear/StructuredProducts.hpp>

namespace vss::math::linear {
namespace {

using detail::Wide;

template <class T> Result<T> failure(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool addFits(Extent lhs, Extent rhs) noexcept {
  return rhs <= std::numeric_limits<Extent>::max() - lhs;
}

bool multiplyFits(Extent lhs, Extent rhs) noexcept {
  return lhs == 0U || rhs <= std::numeric_limits<Extent>::max() / lhs;
}

bool finite(ConstMatrixView matrix) noexcept {
  for (Index row = 0U; row < matrix.rows(); ++row) {
    for (Index column = 0U; column < matrix.columns(); ++column) {
      if (!std::isfinite(matrix.at(row, column))) {
        return false;
      }
    }
  }
  return true;
}

Result<Scalar> scaledProducts(Index count, const auto& pair_at) {
  int common_exponent = std::numeric_limits<int>::min();
  for (Index i = 0U; i < count; ++i) {
    const auto [lhs, rhs] = pair_at(i);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    static_cast<void>(std::frexp(lhs, &lhs_exponent));
    static_cast<void>(std::frexp(rhs, &rhs_exponent));
    common_exponent = std::max(common_exponent, lhs_exponent + rhs_exponent);
  }
  if (common_exponent == std::numeric_limits<int>::min()) {
    return Result<Scalar>::success(0.0);
  }
  Scalar sum = 0.0;
  Scalar correction = 0.0;
  const auto add = [&sum, &correction](Scalar value) {
    const Scalar next = sum + value;
    if (!std::isfinite(next)) {
      return false;
    }
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value
                                                   : (value - next) + sum;
    sum = next;
    return std::isfinite(correction);
  };
  for (Index i = 0U; i < count; ++i) {
    const auto [lhs, rhs] = pair_at(i);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const Scalar lhs_fraction = std::frexp(lhs, &lhs_exponent);
    const Scalar rhs_fraction = std::frexp(rhs, &rhs_exponent);
    const int exponent = lhs_exponent + rhs_exponent;
    const Scalar product = lhs_fraction * rhs_fraction;
    const Scalar product_error = std::fma(lhs_fraction, rhs_fraction, -product);
    const int shift = exponent - common_exponent;
    if (!add(std::scalbn(product, shift)) ||
        !add(std::scalbn(product_error, shift))) {
      return failure<Scalar>(ErrorCode::NonFiniteResult,
                             "structured row accumulation overflowed");
    }
  }
  const Scalar result = std::scalbn(sum + correction, common_exponent);
  if (!std::isfinite(result)) {
    return failure<Scalar>(ErrorCode::NonFiniteResult,
                           "structured row result is not representable");
  }
  return Result<Scalar>::success(result);
}

Result<Wide> scaledProductsWide(Index count, const auto& pair_at) {
  int common_exponent = std::numeric_limits<int>::min();
  for (Index i = 0U; i < count; ++i) {
    const auto [lhs, rhs] = pair_at(i);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    static_cast<void>(std::frexp(lhs, &lhs_exponent));
    static_cast<void>(std::frexp(rhs, &rhs_exponent));
    common_exponent = std::max(common_exponent, lhs_exponent + rhs_exponent);
  }
  if (common_exponent == std::numeric_limits<int>::min()) {
    return Result<Wide>::success(0.0L);
  }
  Scalar sum = 0.0;
  Scalar correction = 0.0;
  const auto add = [&sum, &correction](Scalar value) {
    const Scalar next = sum + value;
    if (!std::isfinite(next)) {
      return false;
    }
    correction += std::abs(sum) >= std::abs(value) ? (sum - next) + value
                                                   : (value - next) + sum;
    sum = next;
    return std::isfinite(correction);
  };
  for (Index i = 0U; i < count; ++i) {
    const auto [lhs, rhs] = pair_at(i);
    if (lhs == 0.0 || rhs == 0.0) {
      continue;
    }
    int lhs_exponent = 0;
    int rhs_exponent = 0;
    const Scalar lhs_fraction = std::frexp(lhs, &lhs_exponent);
    const Scalar rhs_fraction = std::frexp(rhs, &rhs_exponent);
    const int shift = lhs_exponent + rhs_exponent - common_exponent;
    const Scalar product = lhs_fraction * rhs_fraction;
    const Scalar error = std::fma(lhs_fraction, rhs_fraction, -product);
    if (!add(std::scalbn(product, shift)) || !add(std::scalbn(error, shift))) {
      return failure<Wide>(ErrorCode::NoConvergence,
                           "wide scaled product accumulation was unresolved");
    }
  }
  const Wide result =
      std::scalbn(static_cast<Wide>(sum + correction), common_exponent);
  if (!std::isfinite(result)) {
    return failure<Wide>(ErrorCode::NonFiniteResult,
                         "wide scaled product accumulation overflowed");
  }
  return Result<Wide>::success(result);
}

bool leastSquaresPeakFits(Extent rows, Extent columns, Extent rhs_columns,
                          Limits limits) {
  const Extent thin = std::min(rows, columns);
  Extent parameters = 0U;
  Extent basis = 0U;
  if (!multiplyFits(columns, rhs_columns) || !multiplyFits(columns, thin)) {
    return false;
  }
  parameters = columns * rhs_columns;
  basis = columns * thin;
  Extent output_elements = 0U;
  if (!addFits(parameters, basis) ||
      !addFits(parameters + basis, rhs_columns) ||
      !addFits(parameters + basis + rhs_columns, thin)) {
    return false;
  }
  output_elements = parameters + basis + rhs_columns + thin;
  Extent output_bytes = 0U;
  if (!multiplyFits(parameters + basis + rhs_columns, sizeof(Scalar)) ||
      !multiplyFits(thin, sizeof(Wide)) ||
      !addFits((parameters + basis + rhs_columns) * sizeof(Scalar),
               thin * sizeof(Wide))) {
    return false;
  }
  output_bytes =
      (parameters + basis + rhs_columns) * sizeof(Scalar) + thin * sizeof(Wide);

  if (thin == 0U) {
    return output_elements <= limits.elements && output_bytes <= limits.bytes;
  }
  const Extent tall = std::max(rows, columns);
  Extent tall_thin = 0U;
  Extent thin_squared = 0U;
  if (!multiplyFits(tall, thin) || !multiplyFits(thin, thin) ||
      !multiplyFits(thin, 2U)) {
    return false;
  }
  tall_thin = tall * thin;
  thin_squared = thin * thin;
  Extent factor_elements = 0U;
  const std::array factor_parts = {
      tall_thin, tall_thin, thin_squared, tall,         2U * thin,
      tall,      2U * thin, tall_thin,    thin_squared, thin};
  for (const Extent part : factor_parts) {
    if (!addFits(factor_elements, part)) {
      return false;
    }
    factor_elements += part;
  }
  Extent factor_bytes = 0U;
  Extent wide_elements = 0U;
  Extent scalar_elements = 0U;
  const std::array wide_parts = {tall_thin, tall_thin, thin_squared, tall,
                                 2U * thin};
  for (const Extent part : wide_parts) {
    if (!addFits(wide_elements, part)) {
      return false;
    }
    wide_elements += part;
  }
  const std::array scalar_parts = {tall_thin, thin_squared, thin};
  for (const Extent part : scalar_parts) {
    if (!addFits(scalar_elements, part)) {
      return false;
    }
    scalar_elements += part;
  }
  if (!multiplyFits(wide_elements, sizeof(Wide)) ||
      !multiplyFits(scalar_elements, sizeof(Scalar)) ||
      !addFits(wide_elements * sizeof(Wide),
               scalar_elements * sizeof(Scalar))) {
    return false;
  }
  factor_bytes =
      wide_elements * sizeof(Wide) + scalar_elements * sizeof(Scalar);
  if (!addFits(output_elements, factor_elements) ||
      !addFits(output_bytes, factor_bytes) ||
      output_elements + factor_elements > limits.elements ||
      output_bytes + factor_bytes > limits.bytes) {
    return false;
  }
  const Extent residual_scratch = rows;
  if (!multiplyFits(rows, sizeof(Scalar))) {
    return false;
  }
  const Extent residual_bytes = rows * sizeof(Scalar);
  return output_elements <= limits.elements &&
         residual_scratch <= limits.elements - output_elements &&
         output_bytes <= limits.bytes &&
         residual_bytes <= limits.bytes - output_bytes;
}

struct AugmentedSystem final {
  Matrix design;
  Matrix observations;
  int input_scale_exponent;
};

struct ScaledProduct final {
  Scalar fraction;
  int exponent;
};

Result<AugmentedSystem> denseAugmented(ConstMatrixView design,
                                       ConstMatrixView observations,
                                       const LinearMap& penalty, Scalar lambda,
                                       Limits limits) {
  const Extent penalty_rows = lambda > 0.0 ? penalty.rows : 0U;
  Extent rows = 0U;
  Extent design_elements = 0U;
  Extent observation_elements = 0U;
  if (!addFits(design.rows(), penalty_rows) ||
      !multiplyFits(design.rows() + penalty_rows, design.columns()) ||
      !multiplyFits(design.rows() + penalty_rows, observations.columns())) {
    return failure<AugmentedSystem>(ErrorCode::SizeOverflow,
                                    "augmented least-squares shape overflows");
  }
  rows = design.rows() + penalty_rows;
  design_elements = rows * design.columns();
  observation_elements = rows * observations.columns();
  Extent product_count = 0U;
  if (penalty_rows != 0U && !multiplyFits(penalty_rows, design.columns())) {
    return failure<AugmentedSystem>(ErrorCode::SizeOverflow,
                                    "penalty image shape overflows");
  }
  product_count = penalty_rows * design.columns();
  Extent callback_elements = 0U;
  if (penalty_rows != 0U && design.columns() != 0U) {
    if (!addFits(design.columns(), penalty_rows)) {
      return failure<AugmentedSystem>(ErrorCode::SizeOverflow,
                                      "penalty callback workspace overflows");
    }
    callback_elements = design.columns() + penalty_rows;
  }
  if (!addFits(design_elements, observation_elements) ||
      !addFits(design_elements + observation_elements, product_count) ||
      !addFits(design_elements + observation_elements + product_count,
               callback_elements)) {
    return failure<AugmentedSystem>(ErrorCode::SizeOverflow,
                                    "augmented system storage overflows");
  }
  const Extent total_elements = design_elements + observation_elements +
                                product_count + callback_elements;
  Extent scalar_elements = 0U;
  if (!addFits(design_elements, observation_elements) ||
      !addFits(design_elements + observation_elements, callback_elements)) {
    return failure<AugmentedSystem>(ErrorCode::SizeOverflow,
                                    "augmented scalar storage overflows");
  }
  scalar_elements = design_elements + observation_elements + callback_elements;
  if (total_elements > limits.elements ||
      !multiplyFits(scalar_elements, sizeof(Scalar)) ||
      !multiplyFits(product_count, sizeof(ScaledProduct)) ||
      !addFits(scalar_elements * sizeof(Scalar),
               product_count * sizeof(ScaledProduct)) ||
      scalar_elements * sizeof(Scalar) + product_count * sizeof(ScaledProduct) >
          limits.bytes) {
    return failure<AugmentedSystem>(
        ErrorCode::ResourceLimit,
        "augmented least-squares storage exceeds limits");
  }
  const Extent augmented_rows = rows;
  const Extent augmented_reserved = design_elements + observation_elements;
  const Extent augmented_bytes = augmented_reserved * sizeof(Scalar);
  if (augmented_reserved > limits.elements || augmented_bytes > limits.bytes ||
      !leastSquaresPeakFits(
          augmented_rows, design.columns(), observations.columns(),
          Limits{.elements = limits.elements - augmented_reserved,
                 .bytes = limits.bytes - augmented_bytes,
                 .evaluations = limits.evaluations})) {
    return failure<AugmentedSystem>(
        ErrorCode::ResourceLimit,
        "regularized solve and its required workspace exceed limits");
  }
  std::vector<ScaledProduct> products(product_count, {0.0, 0});
  const Scalar lambda_scale = std::sqrt(lambda);
  Vector basis(design.columns(), 0.0);
  for (Index column = 0U; penalty_rows != 0U && design.columns() != 0U &&
                          column < design.columns();
       ++column) {
    std::fill_n(basis.data(), basis.size(), 0.0);
    basis.at(column) = 1.0;
    Result<Vector> action = [&]() {
      try {
        return penalty.apply(basis.view());
      } catch (...) {
        return Result<Vector>::failure(ErrorCode::CallbackFailure,
                                       "penalty callback threw an exception");
      }
    }();
    if (!action.hasValue()) {
      return Result<AugmentedSystem>::failure(
          action.status().code(), std::string(action.status().message()));
    }
    if (action.value().size() != penalty.rows) {
      return failure<AugmentedSystem>(
          ErrorCode::DimensionMismatch,
          "penalty callback returned an incorrect extent");
    }
    for (Index row = 0U; row < penalty_rows; ++row) {
      const Scalar value = action.value().at(row);
      if (!std::isfinite(value)) {
        return failure<AugmentedSystem>(
            ErrorCode::NonFiniteInput,
            "penalty callback returned a non-finite value");
      }
      if (value != 0.0 && lambda_scale != 0.0) {
        int value_exponent = 0;
        int scale_exponent = 0;
        const Scalar value_fraction = std::frexp(value, &value_exponent);
        const Scalar scale_fraction = std::frexp(lambda_scale, &scale_exponent);
        products[row * design.columns() + column] = {
            value_fraction * scale_fraction, value_exponent + scale_exponent};
      }
    }
  }

  int common_exponent = std::numeric_limits<int>::min();
  const auto observe = [&common_exponent](Scalar value) {
    if (value != 0.0) {
      int exponent = 0;
      static_cast<void>(std::frexp(value, &exponent));
      common_exponent = std::max(common_exponent, exponent);
    }
  };
  for (Index row = 0U; row < design.rows(); ++row) {
    for (Index column = 0U; column < design.columns(); ++column) {
      observe(design.at(row, column));
    }
  }
  for (Index row = 0U; row < observations.rows(); ++row) {
    for (Index column = 0U; column < observations.columns(); ++column) {
      observe(observations.at(row, column));
    }
  }
  for (const auto& product : products) {
    if (product.fraction != 0.0) {
      common_exponent = std::max(common_exponent, product.exponent);
    }
  }
  if (common_exponent == std::numeric_limits<int>::min() ||
      (common_exponent >= -500 && common_exponent <= 500)) {
    common_exponent = 0;
  }
  Matrix augmented(rows, design.columns(), 0.0);
  Matrix augmented_rhs(rows, observations.columns(), 0.0);
  const auto scaleData = [&](Scalar value, Scalar& destination) {
    destination = std::scalbn(value, -common_exponent);
    return std::isfinite(destination) &&
           std::scalbn(destination, common_exponent) == value;
  };
  for (Index row = 0; row < design.rows(); ++row) {
    for (Index column = 0; column < design.columns(); ++column) {
      if (!scaleData(design.at(row, column), augmented.at(row, column))) {
        return failure<AugmentedSystem>(
            ErrorCode::NoConvergence, "augmented scaling loses design entries");
      }
    }
  }
  for (Index row = 0; row < observations.rows(); ++row) {
    for (Index column = 0; column < observations.columns(); ++column) {
      if (!scaleData(observations.at(row, column),
                     augmented_rhs.at(row, column))) {
        return failure<AugmentedSystem>(
            ErrorCode::NoConvergence,
            "augmented scaling loses observation entries");
      }
    }
  }
  for (Index row = 0; row < penalty_rows; ++row) {
    for (Index column = 0; column < design.columns(); ++column) {
      const auto& product = products.at(row * design.columns() + column);
      const int shift = product.exponent - common_exponent;
      const Scalar entry = std::scalbn(product.fraction, shift);
      if (!std::isfinite(entry) ||
          std::scalbn(entry, -shift) != product.fraction) {
        return failure<AugmentedSystem>(
            ErrorCode::NoConvergence,
            "augmented scaling loses penalty entries");
      }
      augmented.at(design.rows() + row, column) = entry;
    }
  }
  return Result<AugmentedSystem>::success(
      {std::move(augmented), std::move(augmented_rhs), common_exponent});
}

} // namespace

Result<Vector> apply(const DiagonalMatrix& matrix, ConstVectorView vector,
                     Limits limits) {
  const auto diagonal = matrix.diagonal.view();
  if (vector.size() != diagonal.size()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "diagonal order must equal vector size");
  }
  if (diagonal.size() > limits.elements ||
      !multiplyFits(diagonal.size(), sizeof(Scalar)) ||
      diagonal.size() * sizeof(Scalar) > limits.bytes) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "diagonal result exceeds configured limits");
  }
  for (Index i = 0U; i < vector.size(); ++i) {
    if (!std::isfinite(diagonal.at(i)) || !std::isfinite(vector.at(i))) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "diagonal product contains non-finite input");
    }
  }
  Vector output(diagonal.size(), 0.0);
  for (Index i = 0U; i < diagonal.size(); ++i) {
    const Scalar product = diagonal.at(i) * vector.at(i);
    if (!std::isfinite(product)) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "diagonal product is not representable");
    }
    output.at(i) = product;
  }
  return Result<Vector>::success(std::move(output));
}

Result<Vector> apply(const BlockMatrix& matrix, ConstVectorView vector,
                     Limits limits) {
  const auto row_extents = matrix.row_extents.view();
  const auto column_extents = matrix.column_extents.view();
  const auto blocks = matrix.blocks.view();
  Extent rows = 0U;
  Extent columns = 0U;
  for (const Extent extent : row_extents) {
    if (extent == 0U) {
      return failure<Vector>(ErrorCode::InvalidDomain,
                             "block row extents must be positive");
    }
    if (!addFits(rows, extent)) {
      return failure<Vector>(ErrorCode::SizeOverflow,
                             "block row extent sum overflows");
    }
    rows += extent;
  }
  for (const Extent extent : column_extents) {
    if (extent == 0U) {
      return failure<Vector>(ErrorCode::InvalidDomain,
                             "block column extents must be positive");
    }
    if (!addFits(columns, extent)) {
      return failure<Vector>(ErrorCode::SizeOverflow,
                             "block column extent sum overflows");
    }
    columns += extent;
  }
  if (vector.size() != columns) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "block matrix column total must match vector size");
  }
  if (rows > limits.elements || !multiplyFits(rows, sizeof(Scalar)) ||
      rows * sizeof(Scalar) > limits.bytes) {
    return failure<Vector>(ErrorCode::ResourceLimit,
                           "block matrix output exceeds configured limits");
  }
  for (Index column = 0U; column < vector.size(); ++column) {
    if (!std::isfinite(vector.at(column))) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "block matrix input vector is non-finite");
    }
  }

  Index previous_row = 0U;
  Index previous_column = 0U;
  bool has_previous = false;
  Extent row_terms = 0U;
  Extent max_terms = 0U;
  for (Index i = 0U; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    if (block.block_row >= row_extents.size() ||
        block.block_column >= column_extents.size()) {
      return failure<Vector>(ErrorCode::IndexOutOfRange,
                             "block grid coordinate is out of range");
    }
    if (has_previous && (block.block_row < previous_row ||
                         (block.block_row == previous_row &&
                          block.block_column <= previous_column))) {
      return failure<Vector>(ErrorCode::InvalidDomain,
                             "block entries must be unique and sorted");
    }
    if (block.value.rows() != row_extents[block.block_row] ||
        block.value.columns() != column_extents[block.block_column]) {
      return failure<Vector>(ErrorCode::DimensionMismatch,
                             "block matrix tile shape disagrees with metadata");
    }
    if (!finite(block.value.view())) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "block matrix tile contains non-finite values");
    }
    if (!has_previous || block.block_row != previous_row) {
      row_terms = 0U;
    }
    if (!addFits(row_terms, column_extents[block.block_column])) {
      return failure<Vector>(ErrorCode::SizeOverflow,
                             "block row scratch extent overflows");
    }
    row_terms += column_extents[block.block_column];
    previous_row = block.block_row;
    previous_column = block.block_column;
    has_previous = true;
    max_terms = std::max(max_terms, row_terms);
  }

  const Extent row_blocks = row_extents.size();
  const Extent column_blocks = column_extents.size();
  if (row_blocks == std::numeric_limits<Extent>::max() ||
      column_blocks == std::numeric_limits<Extent>::max()) {
    return failure<Vector>(ErrorCode::SizeOverflow,
                           "block prefix extent overflows");
  }
  const Extent row_prefix_size = row_blocks + 1U;
  const Extent column_prefix_size = column_blocks + 1U;
  Extent metadata_elements = row_prefix_size;
  const std::array metadata_parts = {column_prefix_size, row_blocks,
                                     row_blocks};
  for (const Extent part : metadata_parts) {
    if (!addFits(metadata_elements, part)) {
      return failure<Vector>(ErrorCode::SizeOverflow,
                             "block metadata extent overflows");
    }
    metadata_elements += part;
  }
  Extent scratch_elements = 0U;
  if (!multiplyFits(max_terms, 2U) || !addFits(rows, metadata_elements) ||
      !addFits(rows + metadata_elements, 2U * max_terms)) {
    return failure<Vector>(ErrorCode::SizeOverflow,
                           "block action workspace extent overflows");
  }
  scratch_elements = rows + metadata_elements + 2U * max_terms;
  Extent metadata_bytes = 0U;
  Extent output_bytes = 0U;
  Extent scratch_bytes = 0U;
  if (!multiplyFits(metadata_elements, sizeof(Index)) ||
      !multiplyFits(rows, sizeof(Scalar)) ||
      !multiplyFits(max_terms, 2U * sizeof(Scalar))) {
    return failure<Vector>(ErrorCode::SizeOverflow,
                           "block action workspace byte size overflows");
  }
  metadata_bytes = metadata_elements * sizeof(Index);
  output_bytes = rows * sizeof(Scalar);
  scratch_bytes = 2U * max_terms * sizeof(Scalar);
  if (scratch_elements > limits.elements ||
      !addFits(metadata_bytes, output_bytes) ||
      !addFits(metadata_bytes + output_bytes, scratch_bytes) ||
      metadata_bytes + output_bytes + scratch_bytes > limits.bytes) {
    return failure<Vector>(
        ErrorCode::ResourceLimit,
        "block matrix row workspace exceeds configured limits");
  }

  std::vector<Index> row_prefix(row_prefix_size, 0U);
  std::vector<Index> column_prefix(column_prefix_size, 0U);
  std::vector<Index> block_begin(row_blocks, blocks.size());
  std::vector<Index> block_end(row_blocks, blocks.size());
  for (Index row = 0U; row < row_blocks; ++row) {
    row_prefix[row + 1U] = row_prefix[row] + row_extents[row];
  }
  for (Index column = 0U; column < column_blocks; ++column) {
    column_prefix[column + 1U] = column_prefix[column] + column_extents[column];
  }
  for (Index i = 0U; i < blocks.size(); ++i) {
    const Index row = blocks[i].block_row;
    if (block_begin[row] == blocks.size()) {
      block_begin[row] = i;
    }
    block_end[row] = i + 1U;
  }

  std::vector<Scalar> row_values(max_terms, 0.0);
  std::vector<Scalar> row_inputs(max_terms, 0.0);
  Vector output(rows, 0.0);
  Index block_row = 0U;
  const auto row_action = [&](Index global_row) -> Result<Scalar> {
    while (block_row < row_extents.size() &&
           global_row >= row_prefix[block_row + 1U]) {
      ++block_row;
    }
    const Index local_row = global_row - row_prefix[block_row];
    const Index first = block_begin[block_row];
    const Index last = block_end[block_row];
    Extent term_count = 0U;
    for (Index block_index = first; block_index < last; ++block_index) {
      term_count += column_extents[blocks[block_index].block_column];
    }
    Index ordinal = 0U;
    for (Index block_index = first; block_index < last; ++block_index) {
      const auto& block = blocks[block_index];
      const Index column = block.block_column;
      const Extent width = column_extents[column];
      for (Index local_column = 0U; local_column < width; ++local_column) {
        row_values[ordinal] = block.value.at(local_row, local_column);
        row_inputs[ordinal] = vector.at(column_prefix[column] + local_column);
        ++ordinal;
      }
    }
    const auto pair_at = [&](Index term) {
      return std::pair{row_values[term], row_inputs[term]};
    };
    return scaledProducts(term_count, pair_at);
  };
  for (Index row = 0U; row < rows; ++row) {
    const auto value = row_action(row);
    if (!value.hasValue()) {
      return failure<Vector>(value.status().code(),
                             "block matrix product failed");
    }
    output.at(row) = value.value();
  }
  return Result<Vector>::success(std::move(output));
}

Result<Vector> orthogonalProjection(ConstMatrixView basis,
                                    ConstVectorView vector, SolvePolicy policy,
                                    Limits limits) {
  if (vector.size() != basis.rows()) {
    return failure<Vector>(ErrorCode::DimensionMismatch,
                           "projection vector length must equal basis rows");
  }
  for (Index i = 0U; i < vector.size(); ++i) {
    if (!std::isfinite(vector.at(i))) {
      return failure<Vector>(ErrorCode::NonFiniteInput,
                             "projection input contains a non-finite value");
    }
  }
  const Extent thin = std::min(basis.rows(), basis.columns());
  if (!addFits(basis.rows(), thin) ||
      !multiplyFits(basis.rows(), sizeof(Scalar)) ||
      !multiplyFits(thin, sizeof(Wide)) ||
      !addFits(basis.rows() * sizeof(Scalar), thin * sizeof(Wide)) ||
      basis.rows() + thin > limits.elements ||
      basis.rows() * sizeof(Scalar) + thin * sizeof(Wide) > limits.bytes) {
    return failure<Vector>(
        ErrorCode::ResourceLimit,
        "projection output and coefficient workspace exceed limits");
  }
  const Extent reserved_elements = basis.rows() + thin;
  const Extent reserved_bytes =
      basis.rows() * sizeof(Scalar) + thin * sizeof(Wide);
  auto decomposition =
      factorSVD(basis, policy,
                Limits{.elements = limits.elements - reserved_elements,
                       .bytes = limits.bytes - reserved_bytes,
                       .evaluations = limits.evaluations});
  if (!decomposition.hasValue()) {
    return failure<Vector>(decomposition.status().code(),
                           "projection basis SVD failed");
  }
  const auto& svd = decomposition.value();
  if (policy.rank == RankPolicy::reject &&
      svd.rank < std::min(basis.rows(), basis.columns())) {
    return failure<Vector>(ErrorCode::RankDeficient,
                           "projection basis is rank deficient");
  }
  if (svd.rank == basis.rows()) {
    Vector copy(vector.size(), 0.0);
    for (Index i = 0U; i < vector.size(); ++i) {
      copy.at(i) = vector.at(i);
    }
    return Result<Vector>::success(std::move(copy));
  }
  Vector output(basis.rows(), 0.0);
  if (svd.rank == 0U) {
    return Result<Vector>::success(std::move(output));
  }
  std::vector<Wide> coefficients(svd.rank, 0.0L);
  for (Index mode = 0U; mode < svd.rank; ++mode) {
    const auto value = scaledProductsWide(basis.rows(), [&](Index row) {
      return std::pair{svd.left.at(row, mode), vector.at(row)};
    });
    if (!value.hasValue()) {
      return failure<Vector>(value.status().code(),
                             "projection coefficient accumulation failed");
    }
    coefficients[mode] = value.value();
  }
  for (Index row = 0U; row < basis.rows(); ++row) {
    Wide value = 0.0L;
    Wide correction = 0.0L;
    for (Index mode = 0U; mode < svd.rank; ++mode) {
      const Wide term =
          static_cast<Wide>(svd.left.at(row, mode)) * coefficients[mode];
      const Wide next = value + term;
      correction += std::abs(value) >= std::abs(term) ? (value - next) + term
                                                      : (term - next) + value;
      value = next;
      if (!std::isfinite(value) || !std::isfinite(correction)) {
        return failure<Vector>(ErrorCode::NonFiniteResult,
                               "projected coordinate is not representable");
      }
    }
    value += correction;
    if (!detail::representable(value) ||
        (value != 0.0L && static_cast<Scalar>(value) == 0.0)) {
      return failure<Vector>(ErrorCode::NonFiniteResult,
                             "projected coordinate is not representable");
    }
    output.at(row) = static_cast<Scalar>(value);
  }
  return Result<Vector>::success(std::move(output));
}

Result<LinearEstimate>
regularizedLeastSquares(ConstMatrixView design, ConstMatrixView observations,
                        const LinearMap& penalty, Scalar lambda,
                        SolvePolicy policy, Limits limits) {
  if (!std::isfinite(lambda)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteInput,
                                   "regularization weight is non-finite");
  }
  if (lambda < 0.0) {
    return failure<LinearEstimate>(ErrorCode::InvalidDomain,
                                   "regularization weight must be nonnegative");
  }
  if (design.rows() != observations.rows()) {
    return failure<LinearEstimate>(ErrorCode::DimensionMismatch,
                                   "design and observations row counts differ");
  }
  if (!finite(design) || !finite(observations)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteInput,
                                   "least-squares input is non-finite");
  }
  if (!detail::validNumericalPolicy(policy) ||
      (policy.rank != RankPolicy::reject &&
       policy.rank != RankPolicy::supported_subspace &&
       policy.rank != RankPolicy::minimum_norm)) {
    return failure<LinearEstimate>(ErrorCode::InvalidTolerance,
                                   "regularized solve policy is invalid");
  }
  if (lambda > 0.0 && penalty.columns != design.columns()) {
    return failure<LinearEstimate>(ErrorCode::DimensionMismatch,
                                   "penalty columns do not match design");
  }
  if (lambda == 0.0 || penalty.rows == 0U || design.columns() == 0U) {
    return leastSquaresMinNorm(design, observations, policy, limits);
  }
  if (!penalty.apply) {
    return failure<LinearEstimate>(ErrorCode::CallbackFailure,
                                   "active penalty map has no apply callback");
  }
  if (design.columns() >
      std::min(limits.evaluations,
               static_cast<std::uint64_t>(policy.iteration.max_evaluations))) {
    return failure<LinearEstimate>(
        ErrorCode::ResourceLimit,
        "penalty callback evaluation cap is too small");
  }
  auto augmented =
      denseAugmented(design, observations, penalty, lambda, limits);
  if (!augmented.hasValue()) {
    return Result<LinearEstimate>::failure(
        augmented.status().code(), std::string(augmented.status().message()));
  }
  const Extent augmented_elements =
      augmented.value().design.rows() * augmented.value().design.columns();
  const Extent rhs_elements = augmented.value().observations.rows() *
                              augmented.value().observations.columns();
  const Wide scaled_absolute =
      std::scalbn(static_cast<Wide>(policy.rank_abs),
                  -augmented.value().input_scale_exponent);
  if (!detail::representable(scaled_absolute) ||
      (policy.rank_abs != 0.0 && static_cast<Scalar>(scaled_absolute) == 0.0)) {
    return failure<LinearEstimate>(ErrorCode::NoConvergence,
                                   "scaled absolute rank cutoff is unresolved");
  }
  SolvePolicy scaled_policy = policy;
  scaled_policy.rank_abs = static_cast<Scalar>(scaled_absolute);
  const Limits solve_limits{
      .elements = limits.elements - augmented_elements - rhs_elements,
      .bytes =
          limits.bytes - (augmented_elements + rhs_elements) * sizeof(Scalar),
      .evaluations = limits.evaluations - design.columns()};
  auto estimate = leastSquaresMinNorm(augmented.value().design.view(),
                                      augmented.value().observations.view(),
                                      scaled_policy, solve_limits);
  if (!estimate.hasValue()) {
    return failure<LinearEstimate>(estimate.status().code(),
                                   "augmented least-squares solve failed");
  }
  LinearEstimate result = std::move(estimate).value();
  const Wide restored_cutoff =
      std::scalbn(static_cast<Wide>(result.rank_cutoff),
                  augmented.value().input_scale_exponent);
  if (!detail::representable(restored_cutoff) ||
      (result.rank_cutoff != 0.0 &&
       static_cast<Scalar>(restored_cutoff) == 0.0)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                   "reported rank cutoff is not representable");
  }
  result.rank_cutoff = static_cast<Scalar>(restored_cutoff);
  for (Index column = 0U; column < observations.columns(); ++column) {
    Wide norm = 0;
    for (Index row = 0U; row < observations.rows(); ++row) {
      const auto residual =
          scaledProductsWide(design.columns() + 1U, [&](Index term) {
            if (term == design.columns()) {
              return std::pair{1.0, -observations.at(row, column)};
            }
            return std::pair{design.at(row, term),
                             result.parameters.at(term, column)};
          });
      if (!residual.hasValue()) {
        return failure<LinearEstimate>(
            residual.status().code(),
            "original-data residual accumulation failed");
      }
      norm = std::hypot(norm, residual.value());
    }
    if (!detail::representable(norm)) {
      return failure<LinearEstimate>(
          ErrorCode::NonFiniteResult,
          "original-data residual norm is not representable");
    }
    result.residual_norms.at(column) = static_cast<Scalar>(norm);
  }
  return Result<LinearEstimate>::success(std::move(result));
}

Result<LinearEstimate> solveBanded(const BandMatrix& band, ConstMatrixView rhs,
                                   SolvePolicy policy, Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return failure<LinearEstimate>(ErrorCode::InvalidTolerance,
                                   "band solve policy is invalid");
  }
  if (policy.rank != RankPolicy::reject &&
      policy.rank != RankPolicy::supported_subspace &&
      policy.rank != RankPolicy::minimum_norm) {
    return failure<LinearEstimate>(ErrorCode::InvalidDomain,
                                   "band rank policy is invalid");
  }
  if ((band.order == 0U && (band.lower != 0U || band.upper != 0U)) ||
      (band.order != 0U &&
       (band.lower >= band.order || band.upper >= band.order))) {
    return failure<LinearEstimate>(ErrorCode::InvalidDomain,
                                   "bandwidth must be smaller than order");
  }
  if (!addFits(band.lower, 1U) || !addFits(band.lower + 1U, band.upper)) {
    return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                   "band storage width overflows");
  }
  const Extent width = band.order == 0U ? 0U : band.lower + 1U + band.upper;
  if (band.bands.rows() != band.order || band.bands.columns() != width) {
    return failure<LinearEstimate>(ErrorCode::DimensionMismatch,
                                   "compact band shape is inconsistent");
  }
  if (rhs.rows() != band.order) {
    return failure<LinearEstimate>(
        ErrorCode::DimensionMismatch,
        "band right-hand side has the wrong row count");
  }
  if (!finite(band.bands.view()) || !finite(rhs)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteInput,
                                   "band solve input is non-finite");
  }
  for (Index row = 0U; row < band.order; ++row) {
    for (Index slot = 0U; slot < width; ++slot) {
      const Index distance =
          slot < band.lower ? band.lower - slot : slot - band.lower;
      const bool unused =
          slot < band.lower ? row < distance : distance > band.order - 1U - row;
      if (unused && band.bands.at(row, slot) != 0.0) {
        return failure<LinearEstimate>(ErrorCode::InvalidDomain,
                                       "unused band padding must be zero");
      }
    }
  }

  if (band.lower == 0U && band.upper == 0U) {
    const Index n = band.order;
    const Index p = rhs.columns();
    Scalar largest = 0.0;
    for (Index i = 0U; i < n; ++i) {
      largest = std::max(largest, std::abs(band.bands.at(i, 0U)));
    }
    const Wide cutoff_wide = std::max(static_cast<Wide>(policy.rank_abs),
                                      static_cast<Wide>(policy.rank_rel) *
                                          static_cast<Wide>(largest));
    if (!std::isfinite(cutoff_wide) ||
        cutoff_wide > static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
      return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                     "band cutoff is not representable");
    }
    const auto cutoff = static_cast<Scalar>(cutoff_wide);
    Index rank = 0U;
    Scalar smallest = largest;
    for (Index i = 0U; i < n; ++i) {
      const Scalar singular = std::abs(band.bands.at(i, 0U));
      if (singular > cutoff) {
        ++rank;
        smallest = std::min(smallest, singular);
      }
    }
    if (policy.rank == RankPolicy::reject && rank != n) {
      return failure<LinearEstimate>(ErrorCode::RankDeficient,
                                     "diagonal band is rank deficient");
    }
    if (rank != 0U && policy.reciprocal_condition_floor != 0.0 &&
        static_cast<Wide>(smallest) / largest <
            static_cast<Wide>(policy.reciprocal_condition_floor)) {
      return failure<LinearEstimate>(ErrorCode::IllConditioned,
                                     "diagonal band fails condition floor");
    }
    Extent params = 0U;
    Extent basis = 0U;
    if (!multiplyFits(n, p) || (rank < n && !multiplyFits(n, rank))) {
      return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                     "diagonal-band result shape overflows");
    }
    params = n * p;
    basis = rank < n ? n * rank : 0U;
    if (!addFits(params, basis) || !addFits(params + basis, p) ||
        !addFits(params + basis + p, n)) {
      return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                     "diagonal-band workspace overflows");
    }
    const Extent elements = params + basis + p + n;
    if (elements > limits.elements || !multiplyFits(elements, sizeof(Scalar)) ||
        elements * sizeof(Scalar) > limits.bytes) {
      return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                     "diagonal-band outputs exceed limits");
    }
    LinearEstimate result{.parameters = Matrix(n, p, 0.0),
                          .covariance = std::nullopt,
                          .identified_basis =
                              rank < n ? std::optional{Matrix(n, rank, 0.0)}
                                       : std::nullopt,
                          .residual_norms = Vector(p, 0.0),
                          .rank = rank,
                          .rank_cutoff = cutoff};
    Index basis_column = 0U;
    for (Index i = 0U; i < n; ++i) {
      const Scalar diagonal = band.bands.at(i, 0U);
      if (!(std::abs(diagonal) > cutoff)) {
        continue;
      }
      if (result.identified_basis) {
        result.identified_basis->at(i, basis_column++) = 1.0;
      }
      for (Index column = 0U; column < p; ++column) {
        const Wide value =
            static_cast<Wide>(rhs.at(i, column)) / static_cast<Wide>(diagonal);
        if (!std::isfinite(value) ||
            std::abs(value) >
                static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
          return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                         "diagonal-band solution overflows");
        }
        result.parameters.at(i, column) = static_cast<Scalar>(value);
      }
    }
    for (Index column = 0U; column < p; ++column) {
      Vector residual(n, 0.0);
      for (Index row = 0U; row < n; ++row) {
        const Wide value =
            static_cast<Wide>(band.bands.at(row, 0U)) *
                static_cast<Wide>(result.parameters.at(row, column)) -
            static_cast<Wide>(rhs.at(row, column));
        if (!std::isfinite(value) ||
            std::abs(value) >
                static_cast<Wide>(std::numeric_limits<Scalar>::max())) {
          return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                         "diagonal-band residual overflows");
        }
        residual.at(row) = static_cast<Scalar>(value);
      }
      const auto norm = norm2(residual.view(), OperationOptions{});
      if (!norm.hasValue()) {
        return failure<LinearEstimate>(norm.status().code(),
                                       "diagonal-band residual norm failed");
      }
      result.residual_norms.at(column) = norm.value();
    }
    return Result<LinearEstimate>::success(std::move(result));
  }

  if (auto compact = detail::tryCompactBand(band, rhs, policy, limits)) {
    return Result<LinearEstimate>::success(std::move(*compact));
  }

  Extent dense_elements = 0U;
  if (!multiplyFits(band.order, band.order)) {
    return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                   "dense band fallback shape overflows");
  }
  dense_elements = band.order * band.order;
  if (dense_elements > limits.elements ||
      !multiplyFits(dense_elements, sizeof(Scalar)) ||
      dense_elements * sizeof(Scalar) > limits.bytes) {
    return failure<LinearEstimate>(
        ErrorCode::ResourceLimit,
        "certified dense band fallback exceeds limits");
  }
  const Limits remaining{.elements = limits.elements - dense_elements,
                         .bytes =
                             limits.bytes - dense_elements * sizeof(Scalar),
                         .evaluations = limits.evaluations};
  if (!leastSquaresPeakFits(band.order, band.order, rhs.columns(), remaining)) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "dense band solve workspace exceeds limits");
  }
  Matrix dense(band.order, band.order, 0.0);
  for (Index row = 0U; row < band.order; ++row) {
    for (Index slot = 0U; slot < width; ++slot) {
      if (slot < band.lower) {
        const Index distance = band.lower - slot;
        if (row >= distance) {
          dense.at(row, row - distance) = band.bands.at(row, slot);
        }
      } else {
        const Index distance = slot - band.lower;
        if (distance <= band.order - 1U - row) {
          dense.at(row, row + distance) = band.bands.at(row, slot);
        }
      }
    }
  }
  return leastSquaresMinNorm(dense.view(), rhs, policy, remaining);
}

Result<LinearEstimate> solveBlockDiagonal(const BlockDiagonal& matrix,
                                          ConstMatrixView rhs,
                                          SolvePolicy policy, Limits limits) {
  if (!detail::validNumericalPolicy(policy)) {
    return failure<LinearEstimate>(ErrorCode::InvalidTolerance,
                                   "block solve policy is invalid");
  }
  if (policy.rank != RankPolicy::reject &&
      policy.rank != RankPolicy::supported_subspace &&
      policy.rank != RankPolicy::minimum_norm) {
    return failure<LinearEstimate>(ErrorCode::InvalidDomain,
                                   "block rank policy is invalid");
  }
  const auto blocks = matrix.blocks.view();
  Extent order = 0U;
  for (const auto& block : blocks) {
    if (block.rows() != block.columns()) {
      return failure<LinearEstimate>(ErrorCode::DimensionMismatch,
                                     "each direct-sum block must be square");
    }
    if (!finite(block.view())) {
      return failure<LinearEstimate>(ErrorCode::NonFiniteInput,
                                     "direct-sum block is non-finite");
    }
    if (!addFits(order, block.rows())) {
      return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                     "direct-sum order overflows");
    }
    order += block.rows();
  }
  if (rhs.rows() != order) {
    return failure<LinearEstimate>(
        ErrorCode::DimensionMismatch,
        "right-hand side row count differs from blocks");
  }
  if (!finite(rhs)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteInput,
                                   "block solve right-hand side is non-finite");
  }

  Limits remaining = limits;
  const auto reserve = [&](Index count, Index size) {
    if (count > remaining.elements || count > remaining.bytes / size) {
      return false;
    }
    remaining.elements -= count;
    remaining.bytes -= count * size;
    return true;
  };
  if (!reserve(blocks.size(), sizeof(detail::BoundedSVD))) {
    return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                   "block factor metadata exceeds limits");
  }
  std::vector<detail::BoundedSVD> factors;
  factors.reserve(blocks.size());
  Wide maximum = 0;
  Wide maximum_lower = 0;
  Wide maximum_upper = 0;
  for (const auto& block : blocks) {
    auto bounded = detail::boundedSVD(block.view(), policy, remaining);
    if (!bounded.hasValue()) {
      return Result<LinearEstimate>::failure(
          bounded.status().code(), std::string(bounded.status().message()));
    }
    const auto& factor = bounded.value();
    const auto& svd = factor.factors;
    if (!reserve(svd.singular.size(), sizeof(Scalar)) ||
        !reserve(svd.left.size(), sizeof(Scalar)) ||
        !reserve(svd.right_transpose.size(), sizeof(Scalar))) {
      return failure<LinearEstimate>(ErrorCode::ResourceLimit,
                                     "block factors exceed limits");
    }
    if (factor.rank_upper > 0) {
      const Wide value = svd.singular.at(0);
      maximum = std::max(maximum, value);
      maximum_lower = std::max(maximum_lower, value - factor.error);
      maximum_upper = std::max(maximum_upper, value + factor.error);
    }
    factors.push_back(std::move(bounded).value());
  }
  const auto cutoff_at = [&](Wide largest) {
    return std::max(static_cast<Wide>(policy.rank_abs),
                    policy.rank_rel * largest);
  };
  const Wide cutoff = cutoff_at(maximum);
  const Wide low_cutoff = cutoff_at(maximum_lower);
  const Wide high_cutoff = cutoff_at(maximum_upper);
  if (!detail::representable(cutoff)) {
    return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                   "global block cutoff is not representable");
  }
  Index rank = 0;
  Index largest_block = 0;
  Wide smallest_lower = maximum_upper;
  Wide smallest_upper = maximum_upper;
  for (auto& factor : factors) {
    auto& svd = factor.factors;
    largest_block = std::max(largest_block, svd.singular.size());
    svd.rank = 0;
    for (Index mode = 0; mode < factor.rank_upper; ++mode) {
      const Wide value = svd.singular.at(mode);
      const Wide lower = value - factor.error;
      const Wide upper = value + factor.error;
      if (lower > high_cutoff) {
        ++svd.rank;
        ++rank;
        smallest_lower = std::min(smallest_lower, lower);
        smallest_upper = std::min(smallest_upper, upper);
      } else if (upper > low_cutoff) {
        return failure<LinearEstimate>(
            ErrorCode::NoConvergence,
            "global cutoff intersects a singular value bound");
      }
    }
  }
  if (policy.rank == RankPolicy::reject && rank != order) {
    return failure<LinearEstimate>(ErrorCode::RankDeficient,
                                   "direct-sum spectrum is rank deficient");
  }
  if (rank != 0U && policy.reciprocal_condition_floor > 0) {
    if (smallest_upper < policy.reciprocal_condition_floor * maximum_lower) {
      return failure<LinearEstimate>(ErrorCode::IllConditioned,
                                     "block solve fails condition floor");
    }
    if (smallest_lower < policy.reciprocal_condition_floor * maximum_upper) {
      return failure<LinearEstimate>(
          ErrorCode::NoConvergence,
          "block solve cannot certify condition floor");
    }
  }
  const bool deficient = rank < order;
  if (!multiplyFits(order, rhs.columns()) ||
      (deficient && !multiplyFits(order, rank))) {
    return failure<LinearEstimate>(ErrorCode::SizeOverflow,
                                   "block solve output shape overflows");
  }
  if (!reserve(order * rhs.columns(), sizeof(Scalar)) ||
      (deficient && !reserve(order * rank, sizeof(Scalar))) ||
      !reserve(rhs.columns(), sizeof(Scalar)) ||
      !reserve(rhs.columns() == 0 ? 0 : largest_block, sizeof(Wide))) {
    return failure<LinearEstimate>(
        ErrorCode::ResourceLimit,
        "block solve outputs exceed configured limits");
  }
  LinearEstimate result{.parameters = Matrix(order, rhs.columns()),
                        .covariance = std::nullopt,
                        .identified_basis =
                            deficient ? std::optional{Matrix(order, rank)}
                                      : std::nullopt,
                        .residual_norms = Vector(rhs.columns()),
                        .rank = rank,
                        .rank_cutoff = static_cast<Scalar>(cutoff)};
  std::vector<Wide> coefficients(rhs.columns() == 0 ? 0 : largest_block);
  Index offset = 0;
  Index basis_offset = 0;
  for (const auto& factor : factors) {
    const auto& svd = factor.factors;
    const auto n = svd.singular.size();
    if (auto& basis_output = result.identified_basis; basis_output) {
      for (Index mode = 0; mode < svd.rank; ++mode) {
        for (Index i = 0; i < n; ++i) {
          basis_output->at(offset + i, basis_offset + mode) =
              svd.right_transpose.at(mode, i);
        }
      }
      basis_offset += svd.rank;
    }
    for (Index column = 0; column < rhs.columns(); ++column) {
      for (Index mode = 0; mode < svd.rank; ++mode) {
        const auto projection = scaledProductsWide(n, [&](Index i) {
          return std::pair{svd.left.at(i, mode), rhs.at(offset + i, column)};
        });
        if (!projection.hasValue()) {
          return failure<LinearEstimate>(projection.status().code(),
                                         "block solve projection failed");
        }
        coefficients[mode] = projection.value() / svd.singular.at(mode);
      }
      for (Index i = 0; i < n; ++i) {
        Wide value = 0;
        for (Index mode = 0; mode < svd.rank; ++mode) {
          value = std::fma(svd.right_transpose.at(mode, i), coefficients[mode],
                           value);
        }
        if (!detail::representable(value)) {
          return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                         "block solution is not representable");
        }
        result.parameters.at(offset + i, column) = static_cast<Scalar>(value);
      }
    }
    offset += n;
  }
  for (Index column = 0; column < rhs.columns(); ++column) {
    Wide norm = 0;
    offset = 0;
    for (const auto& block : blocks) {
      for (Index row = 0; row < block.rows(); ++row) {
        const auto residual =
            scaledProductsWide(block.columns() + 1, [&](Index i) {
              if (i == block.columns()) {
                return std::pair{1.0, -rhs.at(offset + row, column)};
              }
              return std::pair{block.at(row, i),
                               result.parameters.at(offset + i, column)};
            });
        if (!residual.hasValue()) {
          return failure<LinearEstimate>(residual.status().code(),
                                         "block residual accumulation failed");
        }
        norm = std::hypot(norm, residual.value());
      }
      offset += block.rows();
    }
    if (!detail::representable(norm)) {
      return failure<LinearEstimate>(ErrorCode::NonFiniteResult,
                                     "block residual is not representable");
    }
    result.residual_norms.at(column) = static_cast<Scalar>(norm);
  }
  return Result<LinearEstimate>::success(std::move(result));
}

} // namespace vss::math::linear
