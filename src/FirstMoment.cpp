#include "MomentValidation.hpp"

#include <string>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {
FirstMoment::FirstMoment(Tensor tensor) noexcept : tensor_(std::move(tensor)) {}

Result<FirstMoment> FirstMoment::create(const Tensor& tensor,
                                        Tolerance tolerance) {
  const Status status = detail::validateMomentTensor(tensor, 1U, tolerance);
  if (!status.ok()) {
    return Result<FirstMoment>::failure(status.code(),
                                        std::string{status.message()});
  }
  return Result<FirstMoment>::success(FirstMoment{tensor});
}

const Tensor& FirstMoment::tensor() const noexcept { return tensor_; }
} // namespace vss::math
