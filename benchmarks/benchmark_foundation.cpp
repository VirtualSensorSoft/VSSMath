#include "benchmark_metadata.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Operations.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>

namespace {

constexpr std::uint64_t seed = 0x5653534d415448ULL;
constexpr std::size_t warmups = 10U;
constexpr std::size_t measurements = 30U;
// Benchmark-only process-local allocation instrumentation.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<bool> measuring{false};
// Benchmark-only process-local allocation instrumentation.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
std::atomic<std::size_t> allocation_count{0U};
// Benchmark-only sink prevents optimization of measured results.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
volatile vss::math::Scalar sink = 0.0;

void* allocate(std::size_t size) {
  const std::size_t actual_size = size == 0U ? 1U : size;
  // Benchmark allocation hook measures allocations without entering the
  // installed library.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  if (void* pointer = std::malloc(actual_size); pointer != nullptr) {
    if (measuring.load(std::memory_order_relaxed)) {
      allocation_count.fetch_add(1U, std::memory_order_relaxed);
    }
    return pointer;
  }
  throw std::bad_alloc();
}

} // namespace

void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void* operator new(std::size_t size, const std::nothrow_t& nothrow) noexcept {
  static_cast<void>(nothrow);
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t& nothrow) noexcept {
  static_cast<void>(nothrow);
  try {
    return allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void operator delete(void* pointer) noexcept {
  // Paired benchmark allocation hook.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  // Paired benchmark allocation hook.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t size) noexcept {
  static_cast<void>(size);
  // Paired benchmark allocation hook.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t size) noexcept {
  static_cast<void>(size);
  // Paired benchmark allocation hook.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t& nothrow) noexcept {
  static_cast<void>(nothrow);
  // Paired benchmark allocation hook.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t& nothrow) noexcept {
  static_cast<void>(nothrow);
  // Paired benchmark allocation hook.
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc,cppcoreguidelines-owning-memory)
  std::free(pointer);
}

namespace {

struct Metric final {
  std::string name;
  long long nanoseconds;
  std::size_t allocations;
};

vss::math::Scalar checksum(const vss::math::Result<vss::math::Scalar>& result) {
  return result.value();
}

vss::math::Scalar checksum(const vss::math::Result<vss::math::Vector>& result) {
  vss::math::Scalar value = 0.0;
  for (vss::math::Index index = 0U; index < result.value().size(); ++index) {
    value += result.value().at(index);
  }
  return value;
}

vss::math::Scalar checksum(const vss::math::Result<vss::math::Matrix>& result) {
  vss::math::Scalar value = 0.0;
  for (vss::math::Index row = 0U; row < result.value().rows(); ++row) {
    for (vss::math::Index column = 0U; column < result.value().columns();
         ++column) {
      value += result.value().at(row, column);
    }
  }
  return value;
}

template <class Callable>
Metric measure(std::string name, Callable&& callable) {
  for (std::size_t iteration = 0U; iteration < warmups; ++iteration) {
    const auto result = std::forward<Callable>(callable)();
    if (!result.hasValue()) {
      throw std::runtime_error("warm-up operation failed");
    }
  }

  std::vector<long long> elapsed;
  std::vector<std::size_t> allocations;
  elapsed.reserve(measurements);
  allocations.reserve(measurements);
  for (std::size_t iteration = 0U; iteration < measurements; ++iteration) {
    allocation_count.store(0U, std::memory_order_relaxed);
    measuring.store(true, std::memory_order_release);
    const auto start = std::chrono::steady_clock::now();
    const auto result = std::forward<Callable>(callable)();
    const auto finish = std::chrono::steady_clock::now();
    measuring.store(false, std::memory_order_release);
    if (!result.hasValue()) {
      throw std::runtime_error("measured operation failed");
    }
    sink = checksum(result);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    elapsed.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finish - start)
            .count());
    allocations.push_back(allocation_count.load(std::memory_order_relaxed));
  }

  std::sort(elapsed.begin(), elapsed.end());
  std::sort(allocations.begin(), allocations.end());
  const long long elapsed_median =
      elapsed[14U] + (elapsed[15U] - elapsed[14U]) / 2LL;
  const std::size_t allocation_median =
      allocations[14U] + (allocations[15U] - allocations[14U]) / 2U;
  return Metric{std::move(name), elapsed_median, allocation_median};
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

vss::math::Vector makeVector(vss::math::Extent size) {
  // The fixed seed is recorded for reproducible benchmark inputs.
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  vss::math::Vector result(size);
  for (vss::math::Index index = 0U; index < size; ++index) {
    result.at(index) = distribution(generator);
  }
  return result;
}

vss::math::Matrix makeMatrix(vss::math::Extent size) {
  // The fixed seed is recorded for reproducible benchmark inputs.
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  vss::math::Matrix result(size, size);
  for (vss::math::Index row = 0U; row < size; ++row) {
    for (vss::math::Index column = 0U; column < size; ++column) {
      result.at(row, column) = distribution(generator);
    }
  }
  return result;
}

void writeMetric(std::ofstream& output, const Metric& metric) {
  output << metric.name << "_ns=" << metric.nanoseconds << '\n';
  output << metric.name << "_allocations=" << metric.allocations << '\n';
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::string output_path;
    std::string requested_build_type;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(
          argv[index]); // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic):
                        // index is bounded by argc.
      if (argument == "--output" && index + 1 < argc) {
        // The loop and option guard bound this argument access by argc.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        output_path = argv[++index];
      } else if (argument == "--build-type" && index + 1 < argc) {
        // The loop and option guard bound this argument access by argc.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        requested_build_type = argv[++index];
      } else {
        throw std::invalid_argument(
            "usage: benchmark_foundation --output PATH --build-type TYPE");
      }
    }
    if (output_path.empty() || requested_build_type.empty()) {
      throw std::invalid_argument(
          "usage: benchmark_foundation --output PATH --build-type TYPE");
    }
    if (requested_build_type != vss::math::benchmark_metadata::buildType) {
      throw std::invalid_argument(
          "requested build type does not match compiled metadata");
    }

    std::vector<Metric> metrics;
    metrics.reserve(12U);
    for (const auto size : {16U, 256U, 4096U}) {
      const auto lhs = makeVector(size);
      const auto rhs = makeVector(size);
      metrics.push_back(
          measure("vector_add_" + std::to_string(size), [&lhs, &rhs] {
            return vss::math::add(lhs.view(), rhs.view());
          }));
      metrics.push_back(measure("dot_" + std::to_string(size), [&lhs, &rhs] {
        return vss::math::dot(lhs.view(), rhs.view());
      }));
    }
    for (const auto size : {16U, 128U, 512U}) {
      const auto matrix = makeMatrix(size);
      const auto vector = makeVector(size);
      metrics.push_back(
          measure("matrix_vector_" + std::to_string(size), [&matrix, &vector] {
            return vss::math::multiply(matrix.view(), vector.view());
          }));
    }
    for (const auto size : {16U, 64U, 128U}) {
      const auto lhs = makeMatrix(size);
      const auto rhs = makeMatrix(size);
      metrics.push_back(
          measure("matrix_multiply_" + std::to_string(size), [&lhs, &rhs] {
            return vss::math::multiply(lhs.view(), rhs.view());
          }));
    }

    std::ofstream output(output_path);
    if (!output) {
      throw std::runtime_error("unable to open benchmark output: " +
                               output_path);
    }
    output << "schema=1\n";
    output << "os="
           << readSystemValue("/etc/os-release", "PRETTY_NAME=", "unknown-os")
           << '\n';
    output << "cpu="
           << readSystemValue("/proc/cpuinfo", "model name\t: ", "unknown-cpu")
           << '\n';
    output << "compiler=" << vss::math::benchmark_metadata::compilerId << ' '
           << vss::math::benchmark_metadata::compilerVersion << '\n';
    output << "build_type=" << vss::math::benchmark_metadata::buildType << '\n';
    output << "compiler_options="
           << vss::math::benchmark_metadata::compilerOptions << '\n';
    output << "command=" << vss::math::benchmark_metadata::benchmarkCommand
           << '\n';
    output << "seed=0x5653534d415448\n";
    output << "warmups=" << warmups << '\n';
    output << "measurements=" << measurements << '\n';
    vss::math::Scalar total_checksum = 0.0;
    for (const auto& metric : metrics) {
      writeMetric(output, metric);
      total_checksum += static_cast<vss::math::Scalar>(metric.nanoseconds);
      total_checksum += static_cast<vss::math::Scalar>(metric.allocations);
    }
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    output << "checksum=" << total_checksum << '\n';
    if (!output) {
      throw std::runtime_error("unable to write benchmark output: " +
                               output_path);
    }
  } catch (const std::exception& exception) {
    measuring.store(false, std::memory_order_release);
    std::cerr << "benchmark_foundation: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
