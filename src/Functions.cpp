#include <cmath>
#include <new>
#include <stdexcept>
#include <vss/math/Error.hpp>
#include <vss/math/Functions.hpp>

namespace vss::math {

Result<Vector> exponentialSmooth(ConstVectorView values, Scalar alpha,
                                 Scalar initial, Limits limits) {
  if (!std::isfinite(alpha) || !std::isfinite(initial)) {
    return Result<Vector>::failure(ErrorCode::NonFiniteInput,
                                   "smoothing parameters must be finite");
  }
  if (alpha < 0.0 || alpha > 1.0) {
    return Result<Vector>::failure(ErrorCode::InvalidDomain,
                                   "smoothing alpha must be in [0,1]");
  }
  for (Index i = 0U; i < values.size(); ++i) {
    if (!std::isfinite(values.at(i))) {
      return Result<Vector>::failure(ErrorCode::NonFiniteInput,
                                     "smoothing input must be finite");
    }
  }
  if (values.size() > limits.elements ||
      values.size() > limits.bytes / sizeof(Scalar)) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "smoothing output exceeds resource limits");
  }
  try {
    Vector output(values.size());
    Scalar state = initial;
    for (Index i = 0U; i < values.size(); ++i) {
      const Scalar observation = values.at(i);
      state = alpha == 0.0 ? state
                           : alpha == 1.0 ? observation
                                          : std::lerp(state, observation, alpha);
      if (!std::isfinite(state)) {
        return Result<Vector>::failure(
            ErrorCode::NonFiniteResult,
            "smoothed state is not representable as a finite scalar");
      }
      output.at(i) = state;
    }
    return Result<Vector>::success(std::move(output));
  } catch (const std::bad_alloc&) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "smoothing output allocation failed");
  } catch (const std::length_error&) {
    return Result<Vector>::failure(ErrorCode::ResourceLimit,
                                   "smoothing output extent is unsupported");
  }
}

} // namespace vss::math
