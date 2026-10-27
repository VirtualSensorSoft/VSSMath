#include "Test.hpp"
#include "support/BenchmarkBatchPlan.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#endif

namespace {

using vss::math::benchmark::BatchPlan;
using vss::math::benchmark::PlanIdentity;
using vss::math::benchmark::readBatchPlan;

constexpr std::array<const char*, 36> kCaseIds{
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

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    const auto root = std::filesystem::temp_directory_path();
    const auto ticks =
        std::chrono::steady_clock::now().time_since_epoch().count();
    for (std::uint32_t attempt = 0U; attempt < 100U; ++attempt) {
      path_ = root / ("vss-math-batch-plan-" + std::to_string(ticks) + "-" +
                      std::to_string(attempt));
      std::error_code error;
      if (std::filesystem::create_directory(path_, error)) {
        return;
      }
      if (error && error != std::errc::file_exists) {
        throw std::runtime_error("could not create temporary test directory");
      }
    }
    throw std::runtime_error("could not choose temporary test directory");
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  TemporaryDirectory(TemporaryDirectory&&) = delete;
  TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

[[nodiscard]] PlanIdentity identity() {
  return {
      .profile = "performance-test-profile",
      .configuration = "Debug",
      .commit = "0000000000000000000000000000000000000000",
      .benchmarkSha256 =
          "0000000000000000000000000000000000000000000000000000000000000000"};
}

[[nodiscard]] std::string readBytes(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open test fixture");
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void writeBytes(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not create test plan");
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("could not write test plan");
  }
}

void expectInvalid(const std::filesystem::path& path,
                   const std::string& contents, const PlanIdentity& expected) {
  writeBytes(path, contents);
  VSS_CHECK_THROWS_AS(readBatchPlan(path, expected), std::invalid_argument);
}

} // namespace

int main(int argc, char** argv) {
  try {
    const std::span<char*> arguments(argv, static_cast<std::size_t>(argc));
    if ((arguments.size() == 3U || arguments.size() == 4U) &&
        std::string_view(arguments[1]) == "--validate") {
      PlanIdentity expected = identity();
      if (arguments.size() == 4U) {
        expected.configuration = arguments[3];
      }
      try {
        static_cast<void>(readBatchPlan(arguments[2], expected));
      } catch (const std::invalid_argument&) {
        return 1;
      }
      return 0;
    }
    VSS_CHECK(arguments.size() == 2U);
    const std::filesystem::path fixture_path(arguments[1]);
    const std::string canonical = readBytes(fixture_path);
    const PlanIdentity expected = identity();
    const BatchPlan plan = readBatchPlan(fixture_path, expected);
    VSS_CHECK(plan.iterations.size() == kCaseIds.size());
    for (const std::uint32_t iterations : plan.iterations) {
      VSS_CHECK(iterations == 64U);
    }
    for (const char* unsafe_configuration :
         {"Debug=Release", "Debug;Release"}) {
      PlanIdentity unsafe_identity = expected;
      unsafe_identity.configuration = unsafe_configuration;
      VSS_CHECK_THROWS_AS(readBatchPlan(fixture_path, unsafe_identity),
                          std::invalid_argument);
    }

    const TemporaryDirectory temporary;
    const auto invalid_path = temporary.path() / "invalid.env";

    std::string reordered = canonical;
    const std::string first_id = "CASE_01_ID=tri.solve.n16.rhs4\n";
    const std::string first_count = "CASE_01_ITERATIONS=64\n";
    const std::size_t first_id_at = reordered.find(first_id);
    VSS_CHECK(first_id_at != std::string::npos);
    reordered.replace(first_id_at, first_id.size() + first_count.size(),
                      first_count + first_id);
    expectInvalid(invalid_path, reordered, expected);

    std::string missing = canonical;
    missing.erase(missing.find(first_id), first_id.size());
    expectInvalid(invalid_path, missing, expected);

    std::string duplicate = canonical;
    const std::string second_id = "CASE_02_ID=tri.solve.n64.rhs4\n";
    duplicate.replace(duplicate.find(second_id), second_id.size(), first_id);
    expectInvalid(invalid_path, duplicate, expected);

    std::string unknown = canonical;
    unknown.replace(unknown.find(first_id), std::string("CASE_01_ID=").size(),
                    "CASE_01_UNKNOWN=");
    expectInvalid(invalid_path, unknown, expected);

    std::string wrong_id = canonical;
    wrong_id.replace(wrong_id.find("tri.solve.n16.rhs4"),
                     std::string("tri.solve.n16.rhs4").size(),
                     "tri.solve.n64.rhs4");
    expectInvalid(invalid_path, wrong_id, expected);

    for (const char* count : {"0", "3", "1048577"}) {
      std::string invalid_count = canonical;
      invalid_count.replace(invalid_count.find(first_count), first_count.size(),
                            std::string("CASE_01_ITERATIONS=") + count + "\n");
      expectInvalid(invalid_path, invalid_count, expected);
    }

    // A standalone reader cannot compare this count with a candidate; Task 6's
    // CMake orchestrator performs that candidate-bound validation before
    // launch.
    std::string other_valid_count = canonical;
    other_valid_count.replace(other_valid_count.find(first_count),
                              first_count.size(), "CASE_01_ITERATIONS=128\n");
    writeBytes(invalid_path, other_valid_count);
    const BatchPlan changed_plan = readBatchPlan(invalid_path, expected);
    VSS_CHECK(changed_plan.iterations[0] == 128U);

    for (const std::string& invalid_header :
         {std::string("RECORD_PROFILE=performance-other-profile"),
          std::string("CONFIGURATION=Release"),
          std::string("MATH_COMMIT=1111111111111111111111111111111111111111"),
          std::string(
              "BENCHMARK_SHA256="
              "1111111111111111111111111111111111111111111111111111111111"
              "111111")}) {
      std::string stale = canonical;
      const std::size_t equals_at = invalid_header.find('=');
      const std::string key = invalid_header.substr(0U, equals_at + 1U);
      const std::size_t start = stale.find(key);
      VSS_CHECK(start != std::string::npos);
      const std::size_t end = stale.find('\n', start);
      stale.replace(start, end - start, invalid_header);
      expectInvalid(invalid_path, stale, expected);
    }

    std::string crlf = canonical;
    for (std::size_t index = 0U;
         (index = crlf.find('\n', index)) != std::string::npos; index += 2U) {
      crlf.insert(index, "\r");
    }
    expectInvalid(invalid_path, crlf, expected);

    std::string no_final_lf = canonical;
    no_final_lf.pop_back();
    expectInvalid(invalid_path, no_final_lf, expected);
    expectInvalid(invalid_path, "# comment\n" + canonical, expected);

    std::string shell_text = canonical;
    shell_text.replace(shell_text.find(first_count), first_count.size(),
                       "CASE_01_ITERATIONS=$(false)\n");
    expectInvalid(invalid_path, shell_text, expected);

    std::string oversized = canonical;
    oversized.append(8193U, 'x');
    expectInvalid(invalid_path, oversized, expected);

    const auto symlink_path = temporary.path() / "symlink.env";
    std::filesystem::create_symlink(fixture_path, symlink_path);
    VSS_CHECK_THROWS_AS(readBatchPlan(symlink_path, expected),
                        std::invalid_argument);

#if defined(__unix__) || defined(__APPLE__)
    const auto fifo_path = temporary.path() / "fifo.env";
    VSS_CHECK(::mkfifo(fifo_path.c_str(), 0600) == 0);
    VSS_CHECK_THROWS_AS(readBatchPlan(fifo_path, expected),
                        std::invalid_argument);
#endif

    return 0;
  } catch (const std::exception&) {
    return 2;
  }
}
