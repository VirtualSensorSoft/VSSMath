#include <stdexcept>
#include <string>
#include <utility>
#include <vss/math/Error.hpp>
#include <vss/math/NumericalError.hpp>

namespace vss::math {

NumericalError::NumericalError(ErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code) {
  if (code_ == ErrorCode::None) {
    throw std::invalid_argument("NumericalError cannot carry ErrorCode::None");
  }
}

ErrorCode NumericalError::code() const noexcept { return code_; }

} // namespace vss::math
