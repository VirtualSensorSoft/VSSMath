#ifndef VSS_MATH_BENCHMARK_BATCH_PLAN_HPP
#define VSS_MATH_BENCHMARK_BATCH_PLAN_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace vss::math::benchmark {

/**
 * @brief Identifies the benchmark build that may consume a batch plan.
 *
 * The values must match the plan's profile, safe configuration token
 * (ASCII letters, digits, dot, underscore, or hyphen), full source commit,
 * and executable SHA-256. This private test-support type does not bind counts
 * to a candidate record; the qualification orchestrator performs that check.
 */
struct PlanIdentity {
  std::string profile;
  std::string configuration;
  std::string commit;
  std::string benchmarkSha256;
};

/**
 * @brief Holds one bounded iteration count for each frozen benchmark case.
 */
struct BatchPlan {
  std::array<std::uint32_t, 36> iterations{};
};

/**
 * @brief Reads a restricted batch plan without executing file contents.
 *
 * @param path Plan file to read; it must be a regular, non-symlink file.
 * @param expected Identity compiled into the consuming benchmark executable.
 * @return The validated 36 case iteration counts in frozen case order.
 * @throws std::invalid_argument if the file, grammar, bounds, or identity is
 * invalid or the plan is larger than 8192 bytes.
 *
 * The caller retains path ownership. This parser performs no benchmark work
 * and does not compare iterations with a candidate record.
 */
[[nodiscard]] BatchPlan readBatchPlan(const std::filesystem::path& path,
                                      const PlanIdentity& expected);

} // namespace vss::math::benchmark

#endif // VSS_MATH_BENCHMARK_BATCH_PLAN_HPP
