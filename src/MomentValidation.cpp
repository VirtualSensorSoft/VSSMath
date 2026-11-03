#include "MomentValidation.hpp"

#include <algorithm>
#include <array>
#include <span>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/Tolerance.hpp>

namespace vss::math::detail {
namespace {
Status failure(ErrorCode code, const char* message) {
  return Status::failure(code, message);
}

std::span<const Scalar> values(const Tensor& tensor) {
  return {tensor.data(), tensor.size()};
}
} // namespace

Status validateMomentTensor(const Tensor& tensor, Extent order,
                            Tolerance tolerance) {
  Status tolerance_status = validate(tolerance);
  if (!tolerance_status.ok()) {
    return tolerance_status;
  }
  if (order == 0U || order > 4U || tensor.shape().rank() != order) {
    return failure(ErrorCode::InvalidExtent,
                   "moment tensor rank does not match its order");
  }

  const Extent dimension = tensor.shape().extent(0U);
  if (dimension == 0U) {
    return failure(ErrorCode::InvalidExtent,
                   "moment tensor component extent must be positive");
  }
  for (Index axis = 1U; axis < order; ++axis) {
    if (tensor.shape().extent(axis) != dimension) {
      return failure(ErrorCode::DimensionMismatch,
                     "moment tensor extents must all match");
    }
  }
  if (tensor.size() != tensor.shape().elementCount()) {
    return failure(ErrorCode::InvalidExtent,
                   "moment tensor storage does not match its shape");
  }

  const std::span<const Scalar> elements = values(tensor);
  if (!std::ranges::all_of(elements,
                           [](Scalar value) { return isFinite(value); })) {
    return failure(ErrorCode::NonFiniteInput,
                   "moment tensor contains a non-finite value");
  }

  std::array<Index, 4U> permutation{0U, 1U, 2U, 3U};
  const std::span<Index> active_permutation{permutation.data(), order};
  bool has_permutation = true;
  while (has_permutation) {
    for (Index linear = 0U; linear < elements.size(); ++linear) {
      std::array<Index, 4U> coordinates{};
      const std::span<Index> active_coordinates{coordinates.data(), order};
      Index remainder = linear;
      for (Index axis = order; axis > 0U; --axis) {
        const Index component = axis - 1U;
        active_coordinates[component] = remainder % dimension;
        remainder /= dimension;
      }

      Index permuted_linear = 0U;
      for (Index axis = 0U; axis < order; ++axis) {
        permuted_linear = (permuted_linear * dimension) +
                          active_coordinates[active_permutation[axis]];
      }
      if (!nearlyEqual(elements[linear], elements[permuted_linear],
                       tolerance)) {
        return failure(ErrorCode::NotSymmetric,
                       "moment tensor is not permutation-symmetric");
      }
    }
    has_permutation = std::next_permutation(active_permutation.begin(),
                                            active_permutation.end());
  }

  return Status::success();
}
} // namespace vss::math::detail
