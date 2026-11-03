#include <algorithm>
#include <ios>
#include <ostream>
#include <span>
#include <string_view>
#include <vss/math/FirstMoment.hpp>
#include <vss/math/FourthMoment.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MultivariateMoments.hpp>
#include <vss/math/PrintOptions.hpp>
#include <vss/math/SecondMoment.hpp>
#include <vss/math/Tensor.hpp>
#include <vss/math/TensorShape.hpp>
#include <vss/math/ThirdMoment.hpp>
#include <vss/math/Vector.hpp>

namespace vss::math {
namespace {
class StreamStateGuard final {
public:
  explicit StreamStateGuard(std::ostream& stream)
      : stream_(stream), flags_(stream.flags()), precision_(stream.precision()),
        fill_(stream.fill()), width_(stream.width()) {}

  StreamStateGuard(const StreamStateGuard&) = delete;
  StreamStateGuard& operator=(const StreamStateGuard&) = delete;

  ~StreamStateGuard() noexcept {
    try {
      stream_.flags(flags_);
      stream_.precision(precision_);
      stream_.fill(fill_);
      stream_.width(width_);
    } catch (...) {
      // Preserve the stream's original exception during stack unwinding.
    }
  }

private:
  std::ostream& stream_;
  std::ios_base::fmtflags flags_;
  std::streamsize precision_;
  char fill_;
  std::streamsize width_;
};

void writeExtents(std::ostream& stream, std::span<const Extent> extents) {
  stream << '[';
  for (Extent axis = 0U; axis < extents.size(); ++axis) {
    if (axis != 0U) {
      stream << ',';
    }
    stream << extents[axis];
  }
  stream << ']';
}

void writeValues(std::ostream& stream, const Scalar* data, Extent count,
                 const PrintOptions& options) {
  const Extent shown = std::min(count, options.maxElements);
  stream << '[';
  for (Extent index = 0U; index < shown; ++index) {
    if (index != 0U) {
      stream << ',';
    }
    stream << data[index];
  }
  if (shown < count) {
    if (shown != 0U) {
      stream << ", ";
    }
    stream << "... (" << (count - shown) << " omitted)";
  }
  stream << ']';
}

std::ostream& printNamedTensor(std::ostream& stream, std::string_view name,
                               const Tensor& tensor,
                               const PrintOptions& options) {
  const StreamStateGuard guard{stream};
  stream << name << "(shape=";
  writeExtents(stream, tensor.shape().extents());
  stream << ", values=";
  writeValues(stream, tensor.data(), tensor.size(), options);
  stream << ')';
  return stream;
}
} // namespace

std::ostream& print(std::ostream& stream, const Vector& vector,
                    const PrintOptions& options) {
  const StreamStateGuard guard{stream};
  stream << "Vector(shape=[" << vector.size() << "], values=";
  writeValues(stream, vector.data(), vector.size(), options);
  stream << ')';
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Vector& vector) {
  return print(stream, vector, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const Matrix& matrix,
                    const PrintOptions& options) {
  const StreamStateGuard guard{stream};
  stream << "Matrix(shape=[" << matrix.rows() << ',' << matrix.columns()
         << "], values=";
  writeValues(stream, matrix.data(), matrix.size(), options);
  stream << ')';
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const Matrix& matrix) {
  return print(stream, matrix, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const Tensor& tensor,
                    const PrintOptions& options) {
  return printNamedTensor(stream, "Tensor", tensor, options);
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
  return print(stream, tensor, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const TensorShape& shape) {
  const StreamStateGuard guard{stream};
  stream << "TensorShape(extents=";
  writeExtents(stream, shape.extents());
  stream << ", elements=" << shape.elementCount() << ')';
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const TensorShape& shape) {
  return print(stream, shape);
}

std::ostream& print(std::ostream& stream, const FirstMoment& moment,
                    const PrintOptions& options) {
  return printNamedTensor(stream, "FirstMoment", moment.tensor(), options);
}

std::ostream& operator<<(std::ostream& stream, const FirstMoment& moment) {
  return print(stream, moment, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const SecondMoment& moment,
                    const PrintOptions& options) {
  return printNamedTensor(stream, "SecondMoment", moment.tensor(), options);
}

std::ostream& operator<<(std::ostream& stream, const SecondMoment& moment) {
  return print(stream, moment, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const ThirdMoment& moment,
                    const PrintOptions& options) {
  return printNamedTensor(stream, "ThirdMoment", moment.tensor(), options);
}

std::ostream& operator<<(std::ostream& stream, const ThirdMoment& moment) {
  return print(stream, moment, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const FourthMoment& moment,
                    const PrintOptions& options) {
  return printNamedTensor(stream, "FourthMoment", moment.tensor(), options);
}

std::ostream& operator<<(std::ostream& stream, const FourthMoment& moment) {
  return print(stream, moment, PrintOptions{});
}

std::ostream& print(std::ostream& stream, const MultivariateMoments& moments,
                    const PrintOptions& options) {
  const StreamStateGuard guard{stream};
  stream << "MultivariateMoments(dimension=" << moments.dimension();
  stream << ", first=";
  if (const auto* moment = moments.getFirstMoment()) {
    print(stream, *moment, options);
  } else {
    stream << "absent";
  }
  stream << ", second=";
  if (const auto* moment = moments.getSecondMoment()) {
    print(stream, *moment, options);
  } else {
    stream << "absent";
  }
  stream << ", third=";
  if (const auto* moment = moments.getThirdMoment()) {
    print(stream, *moment, options);
  } else {
    stream << "absent";
  }
  stream << ", fourth=";
  if (const auto* moment = moments.getFourthMoment()) {
    print(stream, *moment, options);
  } else {
    stream << "absent";
  }
  stream << ')';
  return stream;
}

std::ostream& operator<<(std::ostream& stream,
                         const MultivariateMoments& moments) {
  return print(stream, moments, PrintOptions{});
}

} // namespace vss::math
