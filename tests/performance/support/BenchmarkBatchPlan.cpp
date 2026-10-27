#include "BenchmarkBatchPlan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace vss::math::benchmark {
namespace {

constexpr std::size_t kMaximumPlanBytes = 8192U;
constexpr std::size_t kPlanLineCount = 78U;
constexpr std::uint32_t kMaximumIterations = 1048576U;
constexpr std::array<std::string_view, 6> kHeaderKeys{
    "SCHEMA_VERSION", "RECORD_PROFILE",   "CONFIGURATION",
    "MATH_COMMIT",    "BENCHMARK_SHA256", "CASE_COUNT"};
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

[[noreturn]] void rejectPlan(const char* reason) {
  throw std::invalid_argument(reason);
}

[[nodiscard]] bool isLowerHex(std::string_view value,
                              std::size_t expected_size) noexcept {
  return value.size() == expected_size &&
         std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] bool isProfile(std::string_view value) noexcept {
  if (value.empty()) {
    return false;
  }
  const char first = value.front();
  const bool begins_with_lowercase = first >= 'a' && first <= 'z';
  const bool begins_with_digit = first >= '0' && first <= '9';
  if (!begins_with_lowercase && !begins_with_digit) {
    return false;
  }
  return std::ranges::all_of(value, [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == '-';
  });
}

[[nodiscard]] bool isConfiguration(std::string_view value) noexcept {
  return !value.empty() && std::ranges::all_of(value, [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == '-';
  });
}

[[nodiscard]] std::string readPlanBytes(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, error);
  if (error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    rejectPlan("batch plan must be a regular non-symlink file");
  }

  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > kMaximumPlanBytes) {
    rejectPlan("batch plan is unreadable or exceeds 8192 bytes");
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    rejectPlan("batch plan could not be opened");
  }

  std::string contents;
  contents.reserve(static_cast<std::size_t>(size));
  char byte = '\0';
  while (input.get(byte)) {
    if (contents.size() == kMaximumPlanBytes) {
      rejectPlan("batch plan exceeds 8192 bytes");
    }
    const auto unsigned_byte = static_cast<unsigned char>(byte);
    if (byte != '\n' && (unsigned_byte < 0x20U || unsigned_byte > 0x7eU)) {
      rejectPlan("batch plan is not printable ASCII with LF line endings");
    }
    contents.push_back(byte);
  }
  if (input.bad()) {
    rejectPlan("batch plan could not be read completely");
  }
  if (contents.empty() || contents.back() != '\n') {
    rejectPlan("batch plan must end with LF");
  }
  return contents;
}

[[nodiscard]] std::uint32_t parseIterations(std::string_view value) {
  if (value.empty()) {
    rejectPlan("batch-plan iteration count is not unsigned decimal");
  }
  std::uint32_t iterations = 0U;
  for (const char character : value) {
    if (character < '0' || character > '9') {
      rejectPlan("batch-plan iteration count is not unsigned decimal");
    }
    const auto digit = static_cast<std::uint32_t>(character - '0');
    if (iterations > kMaximumIterations / 10U ||
        (iterations == kMaximumIterations / 10U &&
         digit > kMaximumIterations % 10U)) {
      rejectPlan(
          "batch-plan iteration count is outside the supported powers of two");
    }
    iterations = (iterations * 10U) + digit;
  }
  if (std::to_string(iterations) != value || iterations == 0U ||
      iterations > kMaximumIterations ||
      (iterations & (iterations - 1U)) != 0U) {
    rejectPlan(
        "batch-plan iteration count is outside the supported powers of two");
  }
  return iterations;
}

[[nodiscard]] std::string caseKey(std::size_t case_index,
                                  std::string_view suffix) {
  const std::size_t ordinal = case_index + 1U;
  const std::string number =
      ordinal < 10U ? "0" + std::to_string(ordinal) : std::to_string(ordinal);
  return "CASE_" + number + "_" + std::string(suffix);
}

void validateIdentity(const PlanIdentity& expected) {
  if (!isProfile(expected.profile) ||
      !isConfiguration(expected.configuration) ||
      !isLowerHex(expected.commit, 40U) ||
      !isLowerHex(expected.benchmarkSha256, 64U)) {
    rejectPlan("compiled batch-plan identity is invalid");
  }
}

[[nodiscard]] std::string_view nextValue(const std::string& contents,
                                         std::size_t& offset,
                                         std::string_view expected_key) {
  const std::size_t newline = contents.find('\n', offset);
  if (newline == std::string::npos) {
    rejectPlan("batch plan has a missing line or final LF");
  }
  const std::string_view line =
      std::string_view(contents).substr(offset, newline - offset);
  offset = newline + 1U;
  const std::size_t separator = line.find('=');
  if (separator == std::string_view::npos ||
      line.substr(0U, separator) != expected_key ||
      separator + 1U == line.size()) {
    rejectPlan("batch plan has an invalid, reordered, duplicate, or empty key");
  }
  return line.substr(separator + 1U);
}

} // namespace

BatchPlan readBatchPlan(const std::filesystem::path& path,
                        const PlanIdentity& expected) {
  validateIdentity(expected);
  const std::string contents = readPlanBytes(path);
  BatchPlan plan{};
  std::size_t offset = 0U;

  const std::array<std::string_view, 6> expected_header_values{
      "1",
      expected.profile,
      expected.configuration,
      expected.commit,
      expected.benchmarkSha256,
      "36"};
  for (std::size_t index = 0U; index < kHeaderKeys.size(); ++index) {
    if (nextValue(contents, offset, kHeaderKeys.at(index)) !=
        expected_header_values.at(index)) {
      rejectPlan("batch-plan schema, identity, or case count does not match");
    }
  }

  for (std::size_t index = 0U; index < kCaseIds.size(); ++index) {
    if (nextValue(contents, offset, caseKey(index, "ID")) !=
        kCaseIds.at(index)) {
      rejectPlan(
          "batch-plan case ID or order does not match the frozen inventory");
    }
    plan.iterations.at(index) = parseIterations(
        nextValue(contents, offset, caseKey(index, "ITERATIONS")));
  }
  if (offset != contents.size() ||
      kPlanLineCount != 6U + (2U * kCaseIds.size())) {
    rejectPlan("batch plan has trailing or missing data");
  }
  return plan;
}

} // namespace vss::math::benchmark
