#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/TensorLimits.hpp>

namespace vss::math {

Status TensorLimits::validate(Extent requested) const {
  if (requested > maxElements_) {
    return Status::failure(ErrorCode::ResourceLimit,
                           "tensor element count exceeds configured limit");
  }
  return Status::success();
}

} // namespace vss::math
