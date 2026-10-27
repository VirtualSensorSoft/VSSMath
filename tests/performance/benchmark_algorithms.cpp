#include "benchmark_metadata.hpp"
#include "support/BenchmarkBatchPlan.hpp"
#include "support/BenchmarkMeasurement.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/covariance/Covariance.hpp>
#include <vss/math/differentiation/FiniteDifference.hpp>
#include <vss/math/linear/Cholesky.hpp>
#include <vss/math/linear/LU.hpp>
#include <vss/math/linear/LinearSolve.hpp>
#include <vss/math/linear/QR.hpp>
#include <vss/math/linear/SymmetricEigen.hpp>
#include <vss/math/linear/TriangularSolve.hpp>
#include <vss/math/probability/ChiSquare.hpp>
#include <vss/math/statistics/Descriptive.hpp>

namespace {

using vss::math::ConstVectorView;
using vss::math::Extent;
using vss::math::Index;
using vss::math::Matrix;
using vss::math::Scalar;
using vss::math::Vector;

constexpr std::uint64_t kSeed = 0x5653534d415448ULL;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> measuring{false};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<std::size_t> allocationCount{0U};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> allocationOverflow{false};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
volatile Scalar sink = 0.0;

void* allocate(std::size_t size) {
  const std::size_t actual = size == 0U ? 1U : size;
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory,misc-const-correctness)
  void* pointer = std::malloc(actual);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  if (measuring.load(std::memory_order_relaxed)) {
    std::size_t observed = allocationCount.load(std::memory_order_relaxed);
    for (;;) {
      if (observed == std::numeric_limits<std::size_t>::max()) {
        allocationOverflow.store(true, std::memory_order_relaxed);
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
        std::free(pointer);
        throw std::bad_alloc();
      }
      if (allocationCount.compare_exchange_weak(observed, observed + 1U,
                                                std::memory_order_relaxed,
                                                std::memory_order_relaxed)) {
        break;
      }
    }
  }
  return pointer;
}

} // namespace

void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void* operator new(std::size_t size, const std::nothrow_t& tag) noexcept {
  static_cast<void>(tag);
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t& tag) noexcept {
  static_cast<void>(tag);
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* pointer) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t size) noexcept {
  static_cast<void>(size);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t size) noexcept {
  static_cast<void>(size);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t& tag) noexcept {
  static_cast<void>(tag);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t& tag) noexcept {
  static_cast<void>(tag);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}

namespace {

using Metric = vss::math::benchmark::BatchMetric;
using Probe = vss::math::benchmark::Probe;

constexpr std::array<std::string_view, 36> kCaseIds{
    "tri.solve.n16.rhs4",
    "tri.solve.n64.rhs4",
    "tri.solve.n192.rhs4",
    "lu.factor.n16",
    "lu.factor.n64",
    "lu.factor.n128",
    "lu.solve.n16.rhs4",
    "lu.solve.n64.rhs4",
    "lu.solve.n128.rhs4",
    "qr_ls.factor.m32.n16",
    "qr_ls.factor.m128.n64",
    "qr_ls.factor.m256.n128",
    "qr_ls.solve.m32.n16.rhs4",
    "qr_ls.solve.m128.n64.rhs4",
    "qr_ls.solve.m256.n128.rhs4",
    "qr_square.factor.n192",
    "qr_square.solve.n192.rhs4",
    "cholesky.factor.n16",
    "cholesky.factor.n64",
    "cholesky.factor.n128",
    "cholesky.solve.n16.rhs4",
    "cholesky.solve.n64.rhs4",
    "cholesky.solve.n128.rhs4",
    "eigen.decompose.n16",
    "eigen.decompose.n64",
    "eigen.decompose.n128",
    "covariance.propagate.j64x128.c128",
    "differentiation.jacobian.in64.out128",
    "statistics.mean.n4096",
    "statistics.population_variance.n4096",
    "statistics.sample_variance.n4096",
    "statistics.population_stddev.n4096",
    "statistics.sample_stddev.n4096",
    "statistics.rms.n4096",
    "probability.survival.n4096",
    "probability.log_survival.n4096"};

class MeasurementContext final {
public:
  explicit MeasurementContext(
      const std::optional<vss::math::benchmark::BatchPlan>& plan)
      : plan_(plan) {}

  [[nodiscard]] std::optional<std::uint32_t>
  fixedIterations(std::string_view caseId) {
    if (nextCase_ >= kCaseIds.size() || kCaseIds.at(nextCase_) != caseId) {
      throw std::logic_error(
          "benchmark case order differs from frozen inventory");
    }
    const std::optional<std::uint32_t> result =
        plan_ ? std::optional<std::uint32_t>{plan_->iterations.at(nextCase_)}
              : std::nullopt;
    ++nextCase_;
    return result;
  }

  void requireComplete() const {
    if (nextCase_ != kCaseIds.size()) {
      throw std::logic_error("benchmark did not consume all frozen cases");
    }
  }

private:
  std::optional<vss::math::benchmark::BatchPlan> plan_;
  std::size_t nextCase_{0U};
};

void neumaierAdd(Scalar value, Scalar& sum, Scalar& correction) {
  const Scalar next = sum + value;
  if (std::abs(sum) >= std::abs(value)) {
    correction += (sum - next) + value;
  } else {
    correction += (value - next) + sum;
  }
  sum = next;
}

Scalar neumaierFinish(Scalar sum, Scalar correction) {
  return sum + correction;
}

Scalar checksumValue(Scalar value) { return value; }

Scalar checksumValue(const Matrix& value) {
  Scalar result = 0.0;
  Scalar correction = 0.0;
  for (Index row = 0U; row < value.rows(); ++row) {
    for (Index column = 0U; column < value.columns(); ++column) {
      neumaierAdd(value.at(row, column), result, correction);
    }
  }
  return neumaierFinish(result, correction);
}

Scalar
checksumValue(const vss::math::linear::SymmetricEigenDecomposition& value) {
  Scalar result = 0.0;
  Scalar correction = 0.0;
  const auto eigenvalues = value.eigenvalues();
  for (Index index = 0U; index < eigenvalues.size(); ++index) {
    neumaierAdd(eigenvalues.at(index), result, correction);
  }
  return neumaierFinish(result, correction);
}

Scalar checksumPermutation(std::span<const Index> permutation) {
  Scalar result = 0.0;
  Scalar correction = 0.0;
  for (const Index value : permutation) {
    neumaierAdd(static_cast<Scalar>(value), result, correction);
  }
  return neumaierFinish(result, correction);
}

Scalar checksumValue(const vss::math::linear::LUFactorization& value) {
  return checksumValue(value.lowerFactor()) +
         checksumValue(value.upperFactor()) +
         checksumPermutation(value.permutation());
}

Scalar checksumValue(const vss::math::linear::QRFactorization& value) {
  return checksumValue(value.upperFactor()) +
         checksumPermutation(value.permutation());
}

Scalar checksumValue(const vss::math::linear::CholeskyFactorization& value) {
  return checksumValue(value.lowerFactor());
}

Scalar checksumValue(const vss::math::differentiation::JacobianResult& value) {
  return checksumValue(value.values());
}
template <typename Value>
Scalar checksumValue(const vss::math::linear::SolveOutcome<Value>& value) {
  return checksumValue(value.solution());
}

template <typename Value> Probe checked(vss::math::Result<Value> result) {
  if (!result.hasValue()) {
    return Probe{.passed = false, .checksum = 0.0};
  }
  const Scalar checksum = checksumValue(result.value());
  return Probe{.passed = std::isfinite(checksum), .checksum = checksum};
}

template <typename Callable>
Metric measure(std::string caseId, Callable& callable,
               MeasurementContext& context) {
  const std::optional<std::uint32_t> fixedIterations =
      context.fixedIterations(caseId);
  const vss::math::benchmark::Operation operation = [&callable] {
    return callable();
  };
  const vss::math::benchmark::ClockNow clock = [] {
    const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    if (nanoseconds < 0) {
      throw std::overflow_error("steady_clock timestamp is negative");
    }
    return static_cast<std::uint64_t>(nanoseconds);
  };
  const vss::math::benchmark::AllocationScope allocations{
      .begin =
          [] {
            allocationCount.store(0U, std::memory_order_relaxed);
            allocationOverflow.store(false, std::memory_order_relaxed);
            measuring.store(true, std::memory_order_release);
          },
      .end =
          [] {
            measuring.store(false, std::memory_order_release);
            if (allocationOverflow.exchange(false, std::memory_order_relaxed)) {
              throw std::overflow_error("allocation counter overflowed");
            }
            return static_cast<std::uint64_t>(
                allocationCount.load(std::memory_order_relaxed));
          }};
  const vss::math::benchmark::Consume consume = [](double checksum) {
    sink = checksum;
    std::atomic_signal_fence(std::memory_order_seq_cst);
  };
  return vss::math::benchmark::measureCase(std::move(caseId), operation, clock,
                                           allocations, consume,
                                           fixedIterations);
}

std::mt19937_64 generator(std::uint64_t salt) {
  static_cast<void>(salt);
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp): fixed contract seed
  return std::mt19937_64{kSeed};
}

Scalar signedUnit(std::mt19937_64& source) {
  const std::uint64_t raw = source();
  const Scalar unit = static_cast<Scalar>(raw >> 11U) * 0x1.0p-53;
  return (2.0 * unit) - 1.0;
}

Vector makeVector(Extent size, std::uint64_t salt) {
  auto source = generator(salt);
  Vector result(size);
  for (Index index = 0U; index < size; ++index) {
    result.at(index) = signedUnit(source);
  }
  return result;
}

Vector makeStatisticsVector() {
  auto source = generator(17000U);
  Vector result(4096U);
  for (Index index = 0U; index < result.size(); ++index) {
    result.at(index) = 1.0e6 + (1000.0 * signedUnit(source)) +
                       static_cast<Scalar>(index % 17U);
  }
  return result;
}

Matrix product(const Matrix& matrix, const Matrix& rhs) {
  const auto result = vss::math::multiply(matrix.view(), rhs.view());
  if (!result.hasValue()) {
    throw std::runtime_error("benchmark input matrix product failed");
  }
  return result.value();
}

Matrix makeRhs(Extent rows, std::uint64_t salt) {
  auto source = generator(salt);
  Matrix result(rows, 4U);
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < 4U; ++column) {
      result.at(row, column) = signedUnit(source);
    }
  }
  return result;
}

Matrix makeLu(Extent size, std::uint64_t salt) {
  auto source = generator(salt);
  Matrix result(size, size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    Scalar sum = 0.0;
    Scalar correction = 0.0;
    for (Index column = 0U; column < size; ++column) {
      if (row == column) {
        continue;
      }
      const Scalar value =
          signedUnit(source) / (64.0 * static_cast<Scalar>(size));
      result.at(row, column) = value;
      neumaierAdd(std::abs(value), sum, correction);
    }
    result.at(row, row) =
        neumaierFinish(sum, correction) + 2.0 +
        (static_cast<Scalar>(row + 1U) / static_cast<Scalar>(size));
  }
  return result;
}

Matrix makeQr(Extent rows, Extent columns, std::uint64_t salt) {
  auto source = generator(salt);
  Matrix result(rows, columns, 0.0);
  for (Index row = 0U; row < rows; ++row) {
    for (Index column = 0U; column < columns; ++column) {
      if (row < columns && row == column) {
        result.at(row, column) = 1.0 + (static_cast<Scalar>(column + 1U) /
                                        (2.0 * static_cast<Scalar>(columns)));
      } else {
        result.at(row, column) =
            signedUnit(source) / (32.0 * static_cast<Scalar>(columns));
      }
    }
  }
  return result;
}

Matrix makeQrSquare(Extent size, std::uint64_t salt) {
  constexpr Extent lowerSize = 64U;
  constexpr Extent upperSize = 128U;
  if (size != upperSize + lowerSize) {
    throw std::runtime_error("unexpected square QR size");
  }
  auto source = generator(salt);
  Matrix result(size, size, 0.0);
  for (Index row = 0U; row < upperSize; ++row) {
    for (Index column = 0U; column < upperSize; ++column) {
      result.at(row, column) =
          row == column
              ? 2.0 + (static_cast<Scalar>(row + 1U) /
                       static_cast<Scalar>(upperSize))
              : signedUnit(source) / (64.0 * static_cast<Scalar>(size));
    }
  }
  for (Index row = 0U; row < lowerSize; ++row) {
    for (Index column = 0U; column < upperSize; ++column) {
      const Scalar value =
          row == column
              ? 0.05
              : signedUnit(source) / (64.0 * static_cast<Scalar>(size));
      result.at(upperSize + row, column) = value;
      result.at(column, upperSize + row) = value;
    }
    result.at(upperSize + row, upperSize + row) = -1.0;
  }
  return result;
}

Matrix makeEigen(Extent size) {
  Matrix result(size, size, 0.0);
  const Scalar normalization = std::sqrt(static_cast<Scalar>(size));
  for (Index row = 0U; row < size; ++row) {
    const Scalar uRow = (row % 2U == 0U ? 1.0 : -1.0) / normalization;
    for (Index column = 0U; column < size; ++column) {
      const Scalar uColumn = (column % 2U == 0U ? 1.0 : -1.0) / normalization;
      result.at(row, column) =
          (row == column ? 1.0 + (static_cast<Scalar>(row + 1U) /
                                  static_cast<Scalar>(size))
                         : 0.0) +
          (0.1 * uRow * uColumn);
    }
  }
  return result;
}

Matrix makeCovarianceJacobian() {
  auto source = generator(14000U);
  Matrix result(64U, 128U, 0.0);
  for (Index row = 0U; row < result.rows(); ++row) {
    for (Index column = 0U; column < result.columns(); ++column) {
      result.at(row, column) =
          row == column ? 1.0 : signedUnit(source) / (128.0 * 64.0);
    }
  }
  return result;
}

Matrix makeDifferentiationMatrix() {
  auto source = generator(16000U);
  Matrix result(128U, 64U, 0.0);
  for (Index row = 0U; row < result.rows(); ++row) {
    for (Index column = 0U; column < result.columns(); ++column) {
      result.at(row, column) = signedUnit(source) / 64.0;
      if (row == column) {
        result.at(row, column) += 1.0;
      }
    }
  }
  return result;
}

Matrix makeLower(Extent size, std::uint64_t salt) {
  auto source = generator(salt);
  Matrix result(size, size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    result.at(row, row) =
        2.0 + (static_cast<Scalar>(row + 1U) / static_cast<Scalar>(size));
    for (Index column = 0U; column < row; ++column) {
      result.at(row, column) =
          signedUnit(source) / (16.0 * static_cast<Scalar>(size));
    }
  }
  return result;
}

Matrix transpose(const Matrix& input) {
  Matrix result(input.columns(), input.rows());
  for (Index row = 0U; row < input.rows(); ++row) {
    for (Index column = 0U; column < input.columns(); ++column) {
      // NOLINTNEXTLINE(readability-suspicious-call-argument)
      result.at(column, row) = input.at(row, column);
    }
  }
  return result;
}

Matrix makeCholeskyFactor(Extent size, std::uint64_t salt) {
  auto source = generator(salt);
  Matrix result(size, size, 0.0);
  for (Index row = 0U; row < size; ++row) {
    result.at(row, row) = 1.0 + (static_cast<Scalar>(row + 1U) /
                                 (2.0 * static_cast<Scalar>(size)));
    for (Index column = 0U; column < row; ++column) {
      result.at(row, column) =
          signedUnit(source) / (64.0 * static_cast<Scalar>(size));
    }
  }
  return result;
}

Matrix makePositiveDefinite(Extent size, std::uint64_t salt) {
  const Matrix lower = makeCholeskyFactor(size, salt);
  return product(lower, transpose(lower));
}

std::string readSystemValue(const char* path, std::string_view prefix,
                            std::string fallback) {
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (line.starts_with(prefix)) {
      line.erase(0U, prefix.size());
      if (line.size() >= 2U && line.front() == '"' && line.back() == '"') {
        line = line.substr(1U, line.size() - 2U);
      }
      return line;
    }
  }
  return fallback;
}

std::uint64_t physicalMemoryBytes() {
  std::ifstream input("/proc/meminfo");
  std::string key;
  std::uint64_t kibibytes = 0U;
  while (input >> key) {
    if (key == "MemTotal:") {
      input >> kibibytes;
      return kibibytes * 1024U;
    }
    std::string ignored;
    std::getline(input, ignored);
  }
  return 0U;
}

std::string utcTimestamp() {
  const auto now =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &now);
#else
  gmtime_r(&now, &utc);
#endif
  std::ostringstream output;
  output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return output.str();
}

std::string jsonEscape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped += character;
      break;
    }
  }
  return escaped;
}

void writeJsonString(std::ostream& output, std::string_view value) {
  output << '"' << jsonEscape(value) << '"';
}

void writeMetric(std::ostream& output, const Metric& metric, bool first) {
  if (!first) {
    output << ',';
  }
  output << "{\"case_id\":";
  writeJsonString(output, metric.caseId);
  output << ",\"iterations_per_sample\":" << metric.iterationsPerSample
         << ",\"batch_durations_ns\":[";
  for (std::size_t index = 0U; index < metric.batchDurationsNs.size();
       ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << metric.batchDurationsNs.at(index);
  }
  output << "],\"median_batch_ns\":" << metric.medianBatchNs
         << ",\"mad_batch_ns\":" << metric.madBatchNs
         << ",\"median_ns_per_operation\":" << metric.medianNsPerOperation
         << ",\"allocations_per_operation\":" << metric.allocationsPerOperation
         << R"(,"checksum_hex":")";
  output << std::hexfloat << metric.checksum << std::defaultfloat;
  output << R"(","correctness_passed":true})";
}

void addTriangular(std::vector<Metric>& metrics, Extent size,
                   MeasurementContext& context) {
  const Matrix matrix = makeLower(size, size);
  const Matrix expected = makeRhs(size, size + 1000U);
  const Matrix rhs = product(matrix, expected);
  const auto operation = [&matrix, &rhs] {
    return checked(vss::math::linear::solveTriangular(
        matrix.view(), rhs.view(),
        vss::math::linear::TriangularSolveOptions{
            vss::math::linear::Triangle::Lower}));
  };
  metrics.push_back(measure("tri.solve.n" + std::to_string(size) + ".rhs4",
                            operation, context));
}
void addLuFactor(std::vector<Metric>& metrics, Extent size,
                 MeasurementContext& context) {
  const Matrix matrix = makeLu(size, size + 2000U);
  const auto operation = [&matrix] {
    return checked(vss::math::linear::factorizeLU(matrix.view()));
  };
  metrics.push_back(
      measure("lu.factor.n" + std::to_string(size), operation, context));
}

void addLuSolve(std::vector<Metric>& metrics, Extent size,
                MeasurementContext& context) {
  const Matrix matrix = makeLu(size, size + 3000U);
  const Matrix expected = makeRhs(size, size + 4000U);
  const Matrix rhs = product(matrix, expected);
  auto factorization = vss::math::linear::factorizeLU(matrix.view());
  if (!factorization.hasValue()) {
    throw std::runtime_error("LU benchmark setup failed");
  }
  const auto factor = std::move(factorization).value();
  const auto operation = [&factor, &rhs] {
    return checked(factor.solve(rhs.view()));
  };
  metrics.push_back(measure("lu.solve.n" + std::to_string(size) + ".rhs4",
                            operation, context));
}
void addQrFactor(std::vector<Metric>& metrics, Extent rows, Extent columns,
                 MeasurementContext& context) {
  const Matrix matrix = makeQr(rows, columns, rows + columns + 5000U);
  const auto operation = [&matrix] {
    return checked(vss::math::linear::factorizeQR(matrix.view()));
  };
  metrics.push_back(measure("qr_ls.factor.m" + std::to_string(rows) + ".n" +
                                std::to_string(columns),
                            operation, context));
}

void addQrSolve(std::vector<Metric>& metrics, Extent rows, Extent columns,
                MeasurementContext& context) {
  const Matrix matrix = makeQr(rows, columns, rows + columns + 6000U);
  const Matrix expected = makeRhs(columns, rows + columns + 7000U);
  const Matrix rhs = product(matrix, expected);
  auto factorization = vss::math::linear::factorizeQR(matrix.view());
  if (!factorization.hasValue()) {
    throw std::runtime_error("QR benchmark setup failed");
  }
  const auto factor = std::move(factorization).value();
  const auto operation = [&factor, &rhs] {
    return checked(factor.leastSquares(rhs.view()));
  };
  metrics.push_back(measure("qr_ls.solve.m" + std::to_string(rows) + ".n" +
                                std::to_string(columns) + ".rhs4",
                            operation, context));
}
void addQrSquare(std::vector<Metric>& metrics, Extent size,
                 MeasurementContext& context) {
  const Matrix matrix = makeQrSquare(size, size + 8000U);
  const Matrix expected = makeRhs(size, size + 9000U);
  const Matrix rhs = product(matrix, expected);
  auto factorization = vss::math::linear::factorizeQR(matrix.view());
  if (!factorization.hasValue()) {
    throw std::runtime_error("square QR benchmark setup failed");
  }
  const auto factor = std::move(factorization).value();
  const auto factorOperation = [&matrix] {
    return checked(vss::math::linear::factorizeQR(matrix.view()));
  };
  const auto solveOperation = [&factor, &rhs] {
    return checked(factor.leastSquares(rhs.view()));
  };
  metrics.push_back(measure("qr_square.factor.n" + std::to_string(size),
                            factorOperation, context));
  metrics.push_back(
      measure("qr_square.solve.n" + std::to_string(size) + ".rhs4",
              solveOperation, context));
}
void addCholeskyFactor(std::vector<Metric>& metrics, Extent size,
                       MeasurementContext& context) {
  const Matrix matrix = makePositiveDefinite(size, size + 10000U);
  const auto operation = [&matrix] {
    return checked(vss::math::linear::factorizeCholesky(matrix.view()));
  };
  metrics.push_back(
      measure("cholesky.factor.n" + std::to_string(size), operation, context));
}

void addCholeskySolve(std::vector<Metric>& metrics, Extent size,
                      MeasurementContext& context) {
  const Matrix matrix = makePositiveDefinite(size, size + 11000U);
  const Matrix expected = makeRhs(size, size + 12000U);
  const Matrix rhs = product(matrix, expected);
  auto decomposition = vss::math::linear::factorizeCholesky(matrix.view());
  if (!decomposition.hasValue()) {
    throw std::runtime_error("Cholesky benchmark setup failed");
  }
  const auto factor = std::move(decomposition).value();
  const auto operation = [&factor, &rhs] {
    return checked(factor.solve(rhs.view()));
  };
  metrics.push_back(measure("cholesky.solve.n" + std::to_string(size) + ".rhs4",
                            operation, context));
}
void addEigen(std::vector<Metric>& metrics, Extent size,
              MeasurementContext& context) {
  const Matrix matrix = makeEigen(size);
  const auto operation = [&matrix] {
    return checked(vss::math::linear::decomposeSymmetricEigen(matrix.view()));
  };
  metrics.push_back(
      measure("eigen.decompose.n" + std::to_string(size), operation, context));
}

void addCovariance(std::vector<Metric>& metrics, MeasurementContext& context) {
  const Matrix covariance = makePositiveDefinite(128U, 14000U);

  const Matrix jacobian = makeCovarianceJacobian();
  const auto operation = [&jacobian, &covariance] {
    return checked(vss::math::covariance::propagateCovariance(
        jacobian.view(), covariance.view()));
  };
  metrics.push_back(
      measure("covariance.propagate.j64x128.c128", operation, context));
}

void addDifferentiation(std::vector<Metric>& metrics,
                        MeasurementContext& context) {
  const Vector parameters = makeVector(64U, 16000U);
  const Matrix matrix = makeDifferentiationMatrix();
  std::vector<vss::math::differentiation::ParameterStep> steps;
  steps.reserve(parameters.size());
  for (Index index = 0U; index < parameters.size(); ++index) {
    steps.emplace_back(1.0);
  }
  const vss::math::differentiation::VectorFunction function =
      [&matrix](ConstVectorView input) {
        Vector output(matrix.rows(), 0.0);
        for (Index row = 0U; row < matrix.rows(); ++row) {
          Scalar sum = 0.0;
          Scalar correction = 0.0;
          for (Index column = 0U; column < matrix.columns(); ++column) {
            neumaierAdd(matrix.at(row, column) * input.at(column), sum,
                        correction);
          }
          output.at(row) = neumaierFinish(sum, correction);
        }
        return vss::math::Result<Vector>::success(std::move(output));
      };
  const auto operation = [&function, &parameters, &steps] {
    return checked(vss::math::differentiation::numericalJacobian(
        function, parameters.view(), steps));
  };
  metrics.push_back(
      measure("differentiation.jacobian.in64.out128", operation, context));
}
void addStatistics(std::vector<Metric>& metrics, MeasurementContext& context) {
  const Vector values = makeStatisticsVector();
  const auto add = [&metrics, &values, &context](std::string caseId,
                                                 auto function) {
    const auto operation = [&values, function] {
      return checked(function(values.view()));
    };
    metrics.push_back(measure(std::move(caseId), operation, context));
  };
  add("statistics.mean.n4096", vss::math::statistics::mean);
  add("statistics.population_variance.n4096",
      vss::math::statistics::populationVariance);
  add("statistics.sample_variance.n4096",
      vss::math::statistics::sampleVariance);
  add("statistics.population_stddev.n4096",
      vss::math::statistics::populationStandardDeviation);
  add("statistics.sample_stddev.n4096",
      vss::math::statistics::sampleStandardDeviation);
  add("statistics.rms.n4096", vss::math::statistics::rootMeanSquare);
}

void addProbability(std::vector<Metric>& metrics, MeasurementContext& context) {
  std::vector<Scalar> values;
  std::vector<Extent> degrees;
  values.reserve(4096U);
  degrees.reserve(4096U);
  for (Index index = 0U; index < 4096U; ++index) {
    values.push_back(static_cast<Scalar>(index % 200U) + 0.25);
    degrees.push_back(1U + (index % 10000U));
  }
  const auto measureProbability = [&values, &degrees](bool logarithm) {
    Scalar sum = 0.0;
    Scalar correction = 0.0;
    for (std::size_t index = 0U; index < values.size(); ++index) {
      const auto result =
          logarithm ? vss::math::probability::chiSquareLogSurvival(
                          values[index], degrees[index])
                    : vss::math::probability::chiSquareSurvival(values[index],
                                                                degrees[index]);
      if (!result.hasValue()) {
        return Probe{.passed = false, .checksum = 0.0};
      }
      neumaierAdd(result.value(), sum, correction);
    }
    const Scalar checksum = neumaierFinish(sum, correction);
    return Probe{.passed = std::isfinite(checksum), .checksum = checksum};
  };
  const auto survival = [&measureProbability] {
    return measureProbability(false);
  };
  const auto logSurvival = [&measureProbability] {
    return measureProbability(true);
  };
  metrics.push_back(measure("probability.survival.n4096", survival, context));
  metrics.push_back(
      measure("probability.log_survival.n4096", logSurvival, context));
}
std::string serializeRecord(const std::vector<Metric>& metrics,
                            std::string_view benchmarkSha256,
                            std::string_view frequencyGovernor,
                            std::string_view turboState,
                            std::string_view runnerLabel) {
  if (metrics.size() != 36U) {
    throw std::runtime_error("benchmark did not produce exactly 36 cases");
  }
  std::ostringstream output;
  output << "{\n  \"schema_version\": 2,\n  \"profile_id\": ";
  writeJsonString(output,
                  std::string(vss::math::benchmark_metadata::profileId));
  output << ",\n  \"environment\": {\n";
  const auto writeField = [](std::ostream& stream, std::string_view key,
                             std::string_view value, bool& first) {
    if (!first) {
      stream << ",\n";
    }
    first = false;
    stream << "    ";
    writeJsonString(stream, key);
    stream << ": ";
    writeJsonString(stream, value);
  };
  bool first = true;
  writeField(output, "math_commit", vss::math::benchmark_metadata::mathCommit,
             first);
  writeField(output, "package_version",
             vss::math::benchmark_metadata::packageVersion, first);
  writeField(output, "benchmark_sha256", benchmarkSha256, first);
  writeField(output, "configuration", vss::math::benchmark_metadata::buildType,
             first);
  writeField(output, "compiler_id", vss::math::benchmark_metadata::compilerId,
             first);
  writeField(output, "compiler_version",
             vss::math::benchmark_metadata::compilerVersion, first);
  writeField(output, "target_triple",
             vss::math::benchmark_metadata::targetTriple, first);
  writeField(output, "stdlib_id", vss::math::benchmark_metadata::stdlibId,
             first);
  writeField(output, "stdlib_version",
             vss::math::benchmark_metadata::stdlibVersion, first);
  writeField(output, "cmake_version",
             vss::math::benchmark_metadata::cmakeVersion, first);
  writeField(output, "cxx_standard", "20", first);
  writeField(output, "compile_flags",
             vss::math::benchmark_metadata::compileFlags, first);
  writeField(output, "link_flags", vss::math::benchmark_metadata::linkFlags,
             first);
  writeField(output, "sanitizers", vss::math::benchmark_metadata::sanitizers,
             first);
  writeField(output, "coverage", vss::math::benchmark_metadata::coverage,
             first);
  writeField(output, "os_name",
             readSystemValue("/etc/os-release", "PRETTY_NAME=", "unavailable"),
             first);
  writeField(output, "os_version",
             readSystemValue("/etc/os-release", "VERSION_ID=", "unavailable"),
             first);
  writeField(output, "kernel",
             readSystemValue("/proc/sys/kernel/osrelease", "", "unavailable"),
             first);
  writeField(output, "architecture",
             vss::math::benchmark_metadata::architecture, first);
  writeField(output, "cpu_vendor",
             readSystemValue("/proc/cpuinfo", "vendor_id\t: ", "unavailable"),
             first);
  writeField(output, "cpu_model",
             readSystemValue("/proc/cpuinfo", "model name\t: ", "unavailable"),
             first);
  writeField(output, "cpu_reported_frequency",
             readSystemValue("/proc/cpuinfo", "cpu MHz\t\t: ", "unavailable"),
             first);
  writeField(output, "frequency_governor", frequencyGovernor, first);
  writeField(output, "turbo_state", turboState, first);
  writeField(output, "runner_image", "native-local", first);
  writeField(output, "runner_label_or_host", runnerLabel, first);
  writeField(output, "utc_timestamp", utcTimestamp(), first);
  const unsigned int hardwareConcurrency = std::thread::hardware_concurrency();
  output << ",\n    \"logical_cores\": " << std::max(1U, hardwareConcurrency)
         << ",\n    \"physical_memory_bytes\": " << physicalMemoryBytes()
         << "\n  },\n  \"method\": {\n"
            "    \"seed\": \"0x5653534d415448\",\n"
            "    \"warmup_min_operations\": 10,\n"
            "    \"warmup_min_duration_ns\": 100000000,\n"
            "    \"target_batch_duration_ns\": 20000000,\n"
            "    \"samples\": 30,\n"
            "    \"batch_limit\": 1048576,\n"
            "    \"clock\": \"steady_clock\",\n"
            "    \"statistic\": \"integer_median\",\n"
            "    \"dispersion\": \"integer_mad\",\n"
            "    \"allocation_scope\": \"executable_local\"\n"
            "  },\n  \"cases\": [\n";
  for (std::size_t index = 0U; index < metrics.size(); ++index) {
    writeMetric(output, metrics[index], index == 0U);
    output << '\n';
  }
  output << "  ]\n}\n";
  if (!output) {
    throw std::runtime_error("unable to write benchmark output");
  }
  return output.str();
}

void publishRecord(const std::string& outputPath, std::string_view contents) {
  const std::filesystem::path outputFile(outputPath);
  const std::filesystem::path outputDirectory = outputFile.parent_path();
  if (!outputDirectory.empty()) {
    std::error_code error;
    std::filesystem::create_directories(outputDirectory, error);
    if (error) {
      throw std::runtime_error("unable to create benchmark output directory");
    }
  }

  const auto timestamp =
      std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path temporaryDirectory;
  bool temporaryDirectoryCreated = false;
  for (unsigned int attempt = 0U; attempt < 100U; ++attempt) {
    temporaryDirectory = outputFile;
    temporaryDirectory +=
        ".tmp-" + std::to_string(timestamp) + "-" + std::to_string(attempt);
    std::error_code createError;
    temporaryDirectoryCreated =
        std::filesystem::create_directory(temporaryDirectory, createError);
    if (temporaryDirectoryCreated) {
      break;
    }
    if (createError && createError != std::errc::file_exists) {
      throw std::runtime_error("unable to create fresh benchmark temporary");
    }
  }
  if (!temporaryDirectoryCreated) {
    throw std::runtime_error("unable to allocate a fresh benchmark temporary");
  }

  const auto discardTemporaryDirectory = [&temporaryDirectory]() noexcept {
    std::error_code removeError;
    std::filesystem::remove_all(temporaryDirectory, removeError);
  };
  try {
    const std::filesystem::path temporary = temporaryDirectory / "record.json";
    std::ofstream output(temporary,
                         std::ios::binary | std::ios::out | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("unable to open benchmark temporary");
    }
    output << contents;
    output.close();
    if (!output) {
      throw std::runtime_error("unable to write benchmark temporary");
    }

    std::error_code renameError;
    std::filesystem::rename(temporary, outputFile, renameError);
    if (renameError) {
      throw std::runtime_error("unable to atomically publish benchmark output");
    }
  } catch (...) {
    discardTemporaryDirectory();
    throw;
  }
  discardTemporaryDirectory();
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string outputPath;
    std::string batchPlanPath;
    std::string benchmarkSha256;
    std::string frequencyGovernor;
    std::string turboState;
    std::string runnerLabel;
    bool outputSeen = false;
    bool batchPlanSeen = false;
    bool benchmarkShaSeen = false;
    bool frequencyGovernorSeen = false;
    bool turboStateSeen = false;
    bool runnerLabelSeen = false;
    const std::span<char*> arguments{argv, static_cast<std::size_t>(argc)};
    for (std::size_t index = 1U; index < arguments.size(); ++index) {
      const std::string argument(arguments[index]);
      if (argument == "--output" && index + 1U < arguments.size()) {
        if (outputSeen) {
          throw std::invalid_argument("duplicate --output");
        }
        outputSeen = true;
        outputPath = arguments[++index];
      } else if (argument == "--batch-plan") {
        if (batchPlanSeen) {
          throw std::invalid_argument("duplicate --batch-plan");
        }
        batchPlanSeen = true;
        if (index + 1U >= arguments.size() ||
            std::string_view(arguments[index + 1U]).starts_with("--")) {
          throw std::invalid_argument("--batch-plan requires a path");
        }
        batchPlanPath = arguments[++index];
      } else if (argument == "--benchmark-sha256" &&
                 index + 1U < arguments.size()) {
        if (benchmarkShaSeen) {
          throw std::invalid_argument("duplicate --benchmark-sha256");
        }
        benchmarkShaSeen = true;
        benchmarkSha256 = arguments[++index];
      } else if (argument == "--frequency-governor" &&
                 index + 1U < arguments.size()) {
        if (frequencyGovernorSeen) {
          throw std::invalid_argument("duplicate --frequency-governor");
        }
        frequencyGovernorSeen = true;
        frequencyGovernor = arguments[++index];
      } else if (argument == "--turbo-state" && index + 1U < arguments.size()) {
        if (turboStateSeen) {
          throw std::invalid_argument("duplicate --turbo-state");
        }
        turboStateSeen = true;
        turboState = arguments[++index];
      } else if (argument == "--runner-label" &&
                 index + 1U < arguments.size()) {
        if (runnerLabelSeen) {
          throw std::invalid_argument("duplicate --runner-label");
        }
        runnerLabelSeen = true;
        runnerLabel = arguments[++index];
      } else {
        throw std::invalid_argument("usage: vss_math_algorithm_benchmark "
                                    "--output PATH [--batch-plan PATH] "
                                    "--benchmark-sha256 HEX "
                                    "--frequency-governor NAME --turbo-state "
                                    "STATE --runner-label LABEL");
      }
    }
    const auto validMetadata = [](std::string_view value) {
      return !value.empty() &&
             value.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNO"
                                     "PQRSTUVWXYZ0123456789._-") ==
                 std::string_view::npos;
    };
    const bool validDigest =
        benchmarkSha256.size() == 64U &&
        std::all_of(benchmarkSha256.begin(), benchmarkSha256.end(),
                    [](char character) {
                      return (character >= '0' && character <= '9') ||
                             (character >= 'a' && character <= 'f');
                    });
    if (outputPath.empty() || !validDigest || !frequencyGovernorSeen ||
        !turboStateSeen || !runnerLabelSeen ||
        !validMetadata(frequencyGovernor) || !validMetadata(turboState) ||
        !validMetadata(runnerLabel)) {
      throw std::invalid_argument(
          "usage: vss_math_algorithm_benchmark --output PATH "
          "[--batch-plan PATH] "
          "--benchmark-sha256 HEX "
          "--frequency-governor NAME --turbo-state STATE --runner-label LABEL");
    }
    std::optional<vss::math::benchmark::BatchPlan> batchPlan;
    if (batchPlanSeen) {
      const vss::math::benchmark::PlanIdentity identity{
          .profile = std::string(vss::math::benchmark_metadata::profileId),
          .configuration =
              std::string(vss::math::benchmark_metadata::buildType),
          .commit = std::string(vss::math::benchmark_metadata::mathCommit),
          .benchmarkSha256 = benchmarkSha256};
      batchPlan = vss::math::benchmark::readBatchPlan(batchPlanPath, identity);
    }
    MeasurementContext measurementContext(batchPlan);
    std::vector<Metric> metrics;
    metrics.reserve(36U);
    addTriangular(metrics, 16U, measurementContext);
    addTriangular(metrics, 64U, measurementContext);
    addTriangular(metrics, 192U, measurementContext);
    addLuFactor(metrics, 16U, measurementContext);
    addLuFactor(metrics, 64U, measurementContext);
    addLuFactor(metrics, 128U, measurementContext);
    addLuSolve(metrics, 16U, measurementContext);
    addLuSolve(metrics, 64U, measurementContext);
    addLuSolve(metrics, 128U, measurementContext);
    addQrFactor(metrics, 32U, 16U, measurementContext);
    addQrFactor(metrics, 128U, 64U, measurementContext);
    addQrFactor(metrics, 256U, 128U, measurementContext);
    addQrSolve(metrics, 32U, 16U, measurementContext);
    addQrSolve(metrics, 128U, 64U, measurementContext);
    addQrSolve(metrics, 256U, 128U, measurementContext);
    addQrSquare(metrics, 192U, measurementContext);
    addCholeskyFactor(metrics, 16U, measurementContext);
    addCholeskyFactor(metrics, 64U, measurementContext);
    addCholeskyFactor(metrics, 128U, measurementContext);
    addCholeskySolve(metrics, 16U, measurementContext);
    addCholeskySolve(metrics, 64U, measurementContext);
    addCholeskySolve(metrics, 128U, measurementContext);
    addEigen(metrics, 16U, measurementContext);
    addEigen(metrics, 64U, measurementContext);
    addEigen(metrics, 128U, measurementContext);
    addCovariance(metrics, measurementContext);
    addDifferentiation(metrics, measurementContext);
    addStatistics(metrics, measurementContext);
    addProbability(metrics, measurementContext);
    measurementContext.requireComplete();
    const std::string record = serializeRecord(
        metrics, benchmarkSha256, frequencyGovernor, turboState, runnerLabel);
    publishRecord(outputPath, record);
  } catch (const vss::math::benchmark::MeasurementError& exception) {
    measuring.store(false, std::memory_order_release);
    std::cerr << "vss_math_algorithm_benchmark: " << exception.what() << '\n';
    switch (exception.failure()) {
    case vss::math::benchmark::MeasurementFailure::invalid_record:
      return 1;
    case vss::math::benchmark::MeasurementFailure::allocation_inconsistent:
      return 2;
    case vss::math::benchmark::MeasurementFailure::qualification_incomplete:
      return 3;
    }
    return 1;
  } catch (const std::exception& exception) {
    measuring.store(false, std::memory_order_release);
    std::cerr << "vss_math_algorithm_benchmark: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
