#include "MomentValidation.hpp"

#include <string>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/Result.hpp>

namespace vss::math {
FourthMoment::FourthMoment(Tensor tensor) noexcept
    : tensor_(std::move(tensor)) {}

Result<FourthMoment> FourthMoment::create(const Tensor& tensor,
                                          Tolerance tolerance) {
  const Status status = detail::validateMomentTensor(tensor, 4U, tolerance);
  if (!status.ok()) {
    return Result<FourthMoment>::failure(status.code(),
                                         std::string{status.message()});
  }
  return Result<FourthMoment>::success(FourthMoment{tensor});
}

const Tensor& FourthMoment::tensor() const noexcept { return tensor_; }
} // namespace vss::math
