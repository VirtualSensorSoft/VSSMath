#include "BenchmarkMeasurement.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace vss::math::benchmark {
namespace {

constexpr std::uint64_t kMinimumWarmupOperations = 10U;
constexpr std::uint64_t kMinimumWarmupNs = 100000000U;
constexpr std::uint64_t kTargetBatchNs = 20000000U;
constexpr std::uint32_t kMaximumIterations = 1048576U;
constexpr std::uint64_t kCaseDeadlineNs = 60000000000U;

/** @brief Borrows and monitors one clock without overflowing timestamp sums. */
class ClockMonitor final {
public:
  explicit ClockMonitor(const ClockNow& clock)
      : clock_(clock), start_(clock_()), last_(start_) {}
  ~ClockMonitor() = default;
  ClockMonitor(const ClockMonitor&) = delete;
  ClockMonitor& operator=(const ClockMonitor&) = delete;
  ClockMonitor(ClockMonitor&&) = delete;
  ClockMonitor& operator=(ClockMonitor&&) = delete;

  /** @brief Reads a non-reversing timestamp and checks the elapsed deadline. */
  std::uint64_t read() {
    const std::uint64_t now = clock_();
    if (now < last_) {
      throw MeasurementError(MeasurementFailure::invalid_record,
                             "measurement clock reversed");
    }
    last_ = now;
    if (now - start_ >= kCaseDeadlineNs) {
      throw MeasurementError(MeasurementFailure::qualification_incomplete,
                             "measurement case deadline elapsed");
    }
    return now;
  }

private:
  const ClockNow& clock_;
  const std::uint64_t start_;
  std::uint64_t last_;
};

/** @brief Disables borrowed allocation instrumentation on every scoped exit. */
class CountedScope final {
public:
  explicit CountedScope(const AllocationScope& hooks) : hooks_(hooks) {
    try {
      hooks_.begin();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~CountedScope() { cleanup(); }
  CountedScope(const CountedScope&) = delete;
  CountedScope& operator=(const CountedScope&) = delete;
  CountedScope(CountedScope&&) = delete;
  CountedScope& operator=(CountedScope&&) = delete;

  /** @brief Ends counting; the hook disables its state before reporting errors.
   */
  std::uint64_t finish() {
    active_ = false;
    return hooks_.end();
  }

private:
  void cleanup() noexcept {
    if (active_) {
      active_ = false;
      try {
        static_cast<void>(hooks_.end());
      } catch (...) {
        // Preserve the original failure after the hook disables its counter.
        return;
      }
    }
  }

  const AllocationScope& hooks_;
  bool active_{true};
};

/** @brief Checks and consumes exactly one unchanged operation result. */
double runOperation(const Operation& operation, const Consume& consume,
                    ClockMonitor& clock) {
  static_cast<void>(clock.read());
  std::atomic_signal_fence(std::memory_order_seq_cst);
  const Probe probe = operation();
  std::atomic_signal_fence(std::memory_order_seq_cst);
  if (!probe.passed || !std::isfinite(probe.checksum)) {
    throw MeasurementError(MeasurementFailure::invalid_record,
                           "measurement correctness or checksum failed");
  }
  consume(probe.checksum);
  std::atomic_signal_fence(std::memory_order_seq_cst);
  static_cast<void>(clock.read());
  return probe.checksum;
}

/** @brief Holds one complete timed batch after counter cleanup and validation.
 */
struct TimedBatch {
  std::uint64_t durationNs;
  double checksum;
};

/** @brief Uses the identical counted operation body for calibration and
 * samples. */
TimedBatch runBatch(std::uint32_t iterations, std::uint64_t expectedAllocations,
                    const Operation& operation, const Consume& consume,
                    const AllocationScope& allocations, ClockMonitor& clock) {
  CountedScope scope(allocations);
  const std::uint64_t start = clock.read();
  double checksum = 0.0;
  for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
    checksum = runOperation(operation, consume, clock);
  }
  const std::uint64_t finish = clock.read();
  const std::uint64_t totalAllocations = scope.finish();
  const std::uint64_t duration = finish - start;
  if (duration == 0U) {
    throw MeasurementError(MeasurementFailure::invalid_record,
                           "measurement batch duration is zero");
  }
  if (totalAllocations % iterations != 0U ||
      totalAllocations / iterations != expectedAllocations) {
    throw MeasurementError(MeasurementFailure::allocation_inconsistent,
                           "measurement batch allocations disagree");
  }
  return {.durationNs = duration, .checksum = checksum};
}

/** @brief Calculates the even-sample integer median on a fixed-size copy. */
std::uint64_t median(std::array<std::uint64_t, 30> samples) {
  std::ranges::sort(samples);
  return samples.at(14U) + ((samples.at(15U) - samples.at(14U)) / 2U);
}

} // namespace

MeasurementError::MeasurementError(MeasurementFailure failure,
                                   const std::string& message)
    : std::runtime_error(message), failure_(failure) {}

MeasurementFailure MeasurementError::failure() const noexcept {
  return failure_;
}

BatchMetric measureCase(std::string caseId, const Operation& operation,
                        const ClockNow& clock,
                        const AllocationScope& allocations,
                        const Consume& consume,
                        std::optional<std::uint32_t> fixedIterations) {
  try {
    if (caseId.empty() || !operation || !clock || !allocations.begin ||
        !allocations.end || !consume ||
        (fixedIterations &&
         (*fixedIterations == 0U || *fixedIterations > kMaximumIterations ||
          (*fixedIterations & (*fixedIterations - 1U)) != 0U))) {
      throw MeasurementError(MeasurementFailure::invalid_record,
                             "measurement arguments are invalid");
    }

    BatchMetric metric{.caseId = std::move(caseId),
                       .iterationsPerSample = 0U,
                       .batchDurationsNs = {},
                       .medianBatchNs = 0U,
                       .madBatchNs = 0U,
                       .medianNsPerOperation = 0U,
                       .allocationsPerOperation = 0U,
                       .checksum = 0.0};
    ClockMonitor monitor(clock);
    {
      CountedScope scope(allocations);
      metric.checksum = runOperation(operation, consume, monitor);
      metric.allocationsPerOperation = scope.finish();
    }

    const std::uint64_t warmupStart = monitor.read();
    std::uint64_t warmed = 0U;
    std::uint64_t now = warmupStart;
    while (warmed < kMinimumWarmupOperations ||
           now - warmupStart < kMinimumWarmupNs) {
      metric.checksum = runOperation(operation, consume, monitor);
      if (warmed == std::numeric_limits<std::uint64_t>::max()) {
        throw MeasurementError(MeasurementFailure::invalid_record,
                               "measurement warm-up counter overflowed");
      }
      ++warmed;
      now = monitor.read();
    }

    std::uint32_t iterations = fixedIterations.value_or(1U);
    if (!fixedIterations) {
      for (;;) {
        const TimedBatch batch =
            runBatch(iterations, metric.allocationsPerOperation, operation,
                     consume, allocations, monitor);
        if (batch.durationNs >= kTargetBatchNs) {
          break;
        }
        if (iterations == kMaximumIterations) {
          throw MeasurementError(MeasurementFailure::invalid_record,
                                 "measurement calibration exhausted its cap");
        }
        iterations *= 2U;
      }
    }
    metric.iterationsPerSample = iterations;
    for (auto& duration : metric.batchDurationsNs) {
      const TimedBatch batch =
          runBatch(iterations, metric.allocationsPerOperation, operation,
                   consume, allocations, monitor);
      duration = batch.durationNs;
      metric.checksum = batch.checksum;
    }
    metric.medianBatchNs = median(metric.batchDurationsNs);
    std::array<std::uint64_t, 30> deviations{};
    for (std::size_t index = 0U; index < deviations.size(); ++index) {
      const std::uint64_t duration = metric.batchDurationsNs.at(index);
      deviations.at(index) = duration >= metric.medianBatchNs
                                 ? duration - metric.medianBatchNs
                                 : metric.medianBatchNs - duration;
    }
    metric.madBatchNs = median(deviations);
    metric.medianNsPerOperation = metric.medianBatchNs / iterations;
    return metric;
  } catch (const MeasurementError&) {
    throw;
  } catch (const std::exception&) {
    throw MeasurementError(MeasurementFailure::invalid_record,
                           "measurement callback failed");
  } catch (...) {
    throw MeasurementError(MeasurementFailure::invalid_record,
                           "measurement callback failed");
  }
}

} // namespace vss::math::benchmark
