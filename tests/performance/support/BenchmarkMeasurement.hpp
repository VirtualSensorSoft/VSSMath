#ifndef VSS_MATH_BENCHMARK_MEASUREMENT_HPP
#define VSS_MATH_BENCHMARK_MEASUREMENT_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace vss::math::benchmark {

/** @brief Result of one unchanged benchmark operation and correctness check. */
struct Probe {
  bool passed; ///< True only when the operation's numerical checks pass.
  double
      checksum; ///< Existing checksum; every successful value must be finite.
};

/** @brief Preconstructed callback performing one operation and its checks. */
using Operation = std::function<Probe()>;

/**
 * @brief Preconstructed, allocation-free monotonic nanosecond clock callback.
 *
 * Returns unsigned timestamps from one origin throughout measureCase. A clock
 * adapter must report conversion/overflow errors by throwing, never by
 * wrapping.
 */
using ClockNow = std::function<std::uint64_t()>;

/**
 * @brief Borrows allocation instrumentation owned by the caller.
 *
 * Both hooks must be prepared before measurement and perform no allocations.
 * begin resets and enables the counter. end disables it before returning the
 * exact total or throwing (including overflow). end must also be safe after
 * a partially failed begin. The adapter detects counter overflow before wrap.
 * Hooks and their captured state must outlive measureCase; simultaneous or
 * reentrant use of the same instrumentation is unsupported.
 */
struct AllocationScope {
  std::function<void()> begin;        ///< Reset and enable counting.
  std::function<std::uint64_t()> end; ///< Disable and read the counter.
};

/**
 * @brief Allocation-free callback consuming every successful operation result.
 *
 * The caller supplies its existing checksum sink, rather than changing the
 * operation or accumulating a new numerical checksum.
 */
using Consume = std::function<void(double)>;

/** @brief Fail-closed measurement outcomes, matching qualification policy. */
enum class MeasurementFailure : std::uint8_t {
  invalid_record, ///< Bad input, operation, clock, or counter overflow.
  allocation_inconsistent, ///< Batch allocations disagree with the untimed
                           ///< check.
  qualification_incomplete ///< The 60-second per-case deadline elapsed.
};

/** @brief Private measurement exception carrying a stable failure category. */
class MeasurementError : public std::runtime_error {
public:
  /** @brief Owns the failure category and copies the diagnostic message.
   * @param failure Stable qualification failure category.
   * @param message Diagnostic without protected host identifiers.
   */
  MeasurementError(MeasurementFailure failure, const std::string& message);

  /** @brief Returns the stored failure category without changing state. */
  [[nodiscard]] MeasurementFailure failure() const noexcept;

private:
  MeasurementFailure failure_;
};

/**
 * @brief Owns complete batch samples and exact integer summary statistics.
 *
 * Timings represent amortized batch work, not direct single-call latency.
 * Original capture order is retained. The checksum is the last successful
 * operation's existing checksum; it is not a sum over operations.
 */
struct BatchMetric {
  std::string caseId;                ///< Caller-supplied case ID.
  std::uint32_t iterationsPerSample; ///< Power of two, 1..1048576.
  std::array<std::uint64_t, 30>
      batchDurationsNs;                  ///< Positive ns in capture order.
  std::uint64_t medianBatchNs;           ///< Exact integer median in ns.
  std::uint64_t madBatchNs;              ///< Integer median absolute deviation.
  std::uint64_t medianNsPerOperation;    ///< Floor of median / iterations.
  std::uint64_t allocationsPerOperation; ///< Untimed and timed quotient.
  double checksum;                       ///< Last successful finite checksum.
};

/**
 * @brief Measures one case using the approved warm-up and batch policy.
 *
 * @param caseId Owned case label, moved into the returned result.
 * @param operation Borrowed, preconstructed operation/correctness callback.
 * @param clock Borrowed monotonic clock in nanoseconds; it must not allocate.
 * @param allocations Borrowed counter hooks obeying AllocationScope's contract.
 * @param consume Borrowed, allocation-free consumer of each successful
 * checksum.
 * @param fixedIterations Verification count, or no value to calibrate 1,2,4,...
 * @return All 30 samples, exact median/MAD, matching allocations, last
 * checksum.
 * @throws MeasurementError for invalid callbacks/counts, correctness/checksum
 * failure, callback exceptions, clock reversal, zero batch duration, allocation
 * disagreement, calibration exhaustion, or elapsed per-case deadline.
 *
 * Runs one separately counted untimed operation, warms for at least 10
 * operations AND 100000000 ns, and calibrates to at least 20000000 ns unless
 * fixedIterations is supplied. Counts are powers of two up to 1048576. Samples
 * and statistics use fixed-size storage; counting is disabled on every exit.
 * Checks the 60000000000 ns deadline between operations. A stuck callback is
 * bounded by the orchestrator's separate capture timeout, not this clock.
 *
 * No callback or instrumentation ownership transfers. Concurrent calls need
 * independent callback/counter state. Errors publish no partial metric. Sample
 * computation introduces no numerical tolerance or workload change.
 */
[[nodiscard]] BatchMetric
measureCase(std::string caseId, const Operation& operation,
            const ClockNow& clock, const AllocationScope& allocations,
            const Consume& consume,
            std::optional<std::uint32_t> fixedIterations);

} // namespace vss::math::benchmark

#endif // VSS_MATH_BENCHMARK_MEASUREMENT_HPP
