#ifndef VSS_MATH_TENSOR_HPP
#define VSS_MATH_TENSOR_HPP

#include <iosfwd>
#include <span>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/TensorLimits.hpp>
#include <vss/math/TensorShape.hpp>

namespace vss::math {
/// @brief Owns row-major scalar values with an arbitrary checked rank.
class Tensor final {
public:
  /// @brief Creates filled storage when shape and resource limit permit it.
  [[nodiscard]] static Result<Tensor>
  create(const TensorShape& shape, Scalar fill, const TensorLimits& limits);
  /// @brief Deep-copies the shape and scalar storage.
  Tensor(const Tensor&) = default;
  /// @brief Replaces this value with a deep copy.
  Tensor& operator=(const Tensor&) = default;
  /// @brief Transfers the shape and owned storage.
  Tensor(Tensor&&) noexcept = default;
  /// @brief Replaces this value by transferring owned storage.
  Tensor& operator=(Tensor&&) noexcept = default;
  /// @brief Releases the owned scalar storage.
  ~Tensor() = default;
  /// @brief Returns the immutable owned shape.
  [[nodiscard]] const TensorShape& shape() const noexcept;
  /// @brief Returns the number of stored elements.
  [[nodiscard]] Extent size() const noexcept;
  /// @brief Returns mutable checked access for a rank-matched index.
  [[nodiscard]] Scalar& at(std::span<const Index> indices);
  /// @brief Returns immutable checked access for a rank-matched index.
  [[nodiscard]] const Scalar& at(std::span<const Index> indices) const;
  /// @brief Returns mutable contiguous row-major storage.
  [[nodiscard]] Scalar* data() noexcept;
  /// @brief Returns immutable contiguous row-major storage.
  [[nodiscard]] const Scalar* data() const noexcept;

private:
  Tensor(TensorShape shape, std::vector<Scalar> values) noexcept;
  [[nodiscard]] Index linearIndex(std::span<const Index> indices) const;
  TensorShape shape_;
  std::vector<Scalar> values_;
};
} // namespace vss::math

namespace vss::math {
/// @brief Writes a bounded textual representation without flushing the stream.
std::ostream& print(std::ostream& stream, const Tensor& tensor,
                    const PrintOptions& options);
/// @brief Writes a Tensor using default print options without flushing.
std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
} // namespace vss::math

#endif // VSS_MATH_TENSOR_HPP
