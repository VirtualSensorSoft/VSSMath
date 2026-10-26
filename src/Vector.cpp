#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math {
namespace {

void checkIndex(Index index, Extent size) {
  if (index >= size) {
    throw std::out_of_range("IndexOutOfRange: vector index " +
                            std::to_string(index));
  }
}

} // namespace

Vector::Vector(Extent size, Scalar value) : values_(size, value) {}

Vector::Vector(std::vector<Scalar> values) : values_(std::move(values)) {}

Vector::Vector(Vector&& other) noexcept : values_(std::move(other.values_)) {
  other.values_.clear();
}

Vector& Vector::operator=(Vector&& other) noexcept {
  if (this != &other) {
    values_ = std::move(other.values_);
    other.values_.clear();
  }
  return *this;
}

Extent Vector::size() const noexcept {
  return static_cast<Extent>(values_.size());
}

bool Vector::empty() const noexcept { return values_.empty(); }

Scalar* Vector::data() noexcept { return values_.data(); }

const Scalar* Vector::data() const noexcept { return values_.data(); }

Scalar& Vector::at(Index index) {
  checkIndex(index, size());
  return values_[index];
}

const Scalar& Vector::at(Index index) const {
  checkIndex(index, size());
  return values_[index];
}

void Vector::fill(Scalar value) {
  std::fill(values_.begin(), values_.end(), value);
}

VectorView Vector::view() noexcept { return VectorView(data(), size(), 1U); }

ConstVectorView Vector::view() const noexcept {
  return ConstVectorView(data(), size(), 1U);
}

} // namespace vss::math
