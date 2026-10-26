#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vss/math/Error.hpp>

namespace vss::math {

Status::Status(ErrorCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

Status Status::success() noexcept { return Status(ErrorCode::None, {}); }

Status Status::failure(ErrorCode code, std::string message) {
  if (code == ErrorCode::None) {
    throw std::invalid_argument(
        "a failed status requires a non-None error code");
  }
  if (message.empty()) {
    throw std::invalid_argument("a failed status requires a non-empty message");
  }
  return Status(code, std::move(message));
}

bool Status::ok() const noexcept { return code_ == ErrorCode::None; }

ErrorCode Status::code() const noexcept { return code_; }

std::string_view Status::message() const noexcept { return message_; }

} // namespace vss::math
