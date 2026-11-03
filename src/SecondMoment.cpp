#include "MomentValidation.hpp"

#include <string>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SecondMoment.hpp>

namespace vss::math {
SecondMoment::SecondMoment(Tensor tensor) noexcept
    : tensor_(std::move(tensor)) {}

Result<SecondMoment> SecondMoment::create(const Tensor& tensor,
                                          Tolerance tolerance) {
  const Status status = detail::validateMomentTensor(tensor, 2U, tolerance);
  if (!status.ok()) {
    return Result<SecondMoment>::failure(status.code(),
                                         std::string{status.message()});
  }
  return Result<SecondMoment>::success(SecondMoment{tensor});
}

const Tensor& SecondMoment::tensor() const noexcept { return tensor_; }
} // namespace vss::math
