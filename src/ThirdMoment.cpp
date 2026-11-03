#include "MomentValidation.hpp"

#include <string>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/ThirdMoment.hpp>

namespace vss::math {
ThirdMoment::ThirdMoment(Tensor tensor) noexcept : tensor_(std::move(tensor)) {}

Result<ThirdMoment> ThirdMoment::create(const Tensor& tensor,
                                        Tolerance tolerance) {
  const Status status = detail::validateMomentTensor(tensor, 3U, tolerance);
  if (!status.ok()) {
    return Result<ThirdMoment>::failure(status.code(),
                                        std::string{status.message()});
  }
  return Result<ThirdMoment>::success(ThirdMoment{tensor});
}

const Tensor& ThirdMoment::tensor() const noexcept { return tensor_; }
} // namespace vss::math
