#include "Test.hpp"
#include "support/BenchmarkMeasurement.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

namespace {

using vss::math::benchmark::AllocationScope;
using vss::math::benchmark::BatchMetric;
using vss::math::benchmark::ClockNow;
using vss::math::benchmark::Consume;
using vss::math::benchmark::measureCase;
using vss::math::benchmark::MeasurementError;
using vss::math::benchmark::MeasurementFailure;
using vss::math::benchmark::Operation;
using vss::math::benchmark::Probe;

enum class EventKind : std::uint8_t { clock, begin, operation, consume, end };
enum class OperationFault : std::uint8_t {
  none,
  correctness,
  infinity,
  nan,
  exception
};

struct Event {
  EventKind kind;
  bool counting;
};

constexpr std::array<std::uint64_t, 30> kSampleDurations{
    105U, 95U, 105U, 95U, 105U, 95U, 105U, 95U, 105U, 95U,
    105U, 95U, 105U, 95U, 105U, 95U, 105U, 95U, 105U, 95U,
    105U, 95U, 105U, 95U, 105U, 95U, 105U, 95U, 105U, 95U};

// Inject only the environmental boundaries. measureCase supplies the real
// warm-up, batching, correctness, cleanup, and statistics behavior under test.
struct Harness {
  std::uint64_t nowNs{0U};
  std::uint64_t warmupStepNs{10000000U};
  std::uint64_t timedStepNs{5000000U};
  std::uint64_t singleOperationAllocations{3U};
  std::uint64_t allocationTotal{0U};
  std::uint64_t operations{0U};
  std::uint64_t warmups{0U};
  std::uint64_t consumed{0U};
  std::size_t begins{0U};
  std::size_t ends{0U};
  std::size_t clockReads{0U};
  std::size_t reverseClockAt{0U};
  std::size_t throwClockAt{0U};
  std::size_t throwBeginAt{0U};
  std::size_t throwEndAt{0U};
  std::size_t overflowEndAt{0U};
  std::size_t changeAllocationAt{0U};
  std::uint64_t allocationAdjustment{0U};
  std::uint64_t faultOperationAt{0U};
  std::uint64_t throwConsumeAt{0U};
  OperationFault fault{OperationFault::none};
  bool useSampleDurations{false};
  bool prepared{false};
  bool counting{false};
  std::array<std::uint64_t, 64> scopeOperations{};
  std::array<Event, 8192> events{};
  std::size_t eventCount{0U};
};

void logEvent(Harness& harness, EventKind kind) {
  if (harness.eventCount < harness.events.size()) {
    harness.events.at(harness.eventCount) = {.kind = kind,
                                             .counting = harness.counting};
    ++harness.eventCount;
  }
}

std::uint64_t readClock(Harness& harness) {
  ++harness.clockReads;
  logEvent(harness, EventKind::clock);
  if (harness.clockReads == harness.throwClockAt) {
    throw std::runtime_error("injected clock failure");
  }
  if (harness.clockReads == harness.reverseClockAt) {
    VSS_CHECK(harness.nowNs > 0U);
    return harness.nowNs - 1U;
  }
  return harness.nowNs;
}

Probe performOperation(Harness& harness) {
  logEvent(harness, EventKind::operation);
  ++harness.operations;
  std::uint64_t step = harness.warmupStepNs;
  if (harness.counting) {
    const std::size_t scope = harness.begins - 1U;
    ++harness.scopeOperations.at(scope);
    if (scope == 0U) {
      step = 1U;
    } else if (harness.useSampleDurations) {
      step = kSampleDurations.at(scope - 1U);
    } else {
      step = harness.timedStepNs;
    }
    if (harness.allocationTotal > std::numeric_limits<std::uint64_t>::max() -
                                      harness.singleOperationAllocations) {
      throw std::overflow_error("injected counter overflow");
    }
    harness.allocationTotal += harness.singleOperationAllocations;
  } else {
    ++harness.warmups;
  }
  VSS_CHECK(harness.nowNs <= std::numeric_limits<std::uint64_t>::max() - step);
  harness.nowNs += step;
  if (harness.operations == harness.faultOperationAt) {
    switch (harness.fault) {
    case OperationFault::correctness:
      return {.passed = false, .checksum = 0.0};
    case OperationFault::infinity:
      return {.passed = true,
              .checksum = std::numeric_limits<double>::infinity()};
    case OperationFault::nan:
      return {.passed = true,
              .checksum = std::numeric_limits<double>::quiet_NaN()};
    case OperationFault::exception:
      throw std::runtime_error("injected operation failure");
    case OperationFault::none:
      break;
    }
  }
  return {.passed = true, .checksum = static_cast<double>(harness.operations)};
}

void beginCounting(Harness& harness) {
  VSS_CHECK(harness.prepared);
  VSS_CHECK(!harness.counting);
  harness.counting = true;
  harness.allocationTotal = 0U;
  ++harness.begins;
  logEvent(harness, EventKind::begin);
  if (harness.begins == harness.throwBeginAt) {
    throw std::runtime_error("injected begin failure");
  }
}

std::uint64_t endCounting(Harness& harness) {
  VSS_CHECK(harness.counting);
  harness.counting = false;
  ++harness.ends;
  logEvent(harness, EventKind::end);
  if (harness.ends == harness.overflowEndAt) {
    throw std::overflow_error("injected counter overflow");
  }
  if (harness.ends == harness.throwEndAt) {
    throw std::runtime_error("injected end failure");
  }
  if (harness.ends == harness.changeAllocationAt) {
    return harness.allocationTotal + harness.allocationAdjustment;
  }
  return harness.allocationTotal;
}

void consumeResult(Harness& harness, double checksum) {
  logEvent(harness, EventKind::consume);
  ++harness.consumed;
  VSS_CHECK(checksum == static_cast<double>(harness.operations));
  if (harness.consumed == harness.throwConsumeAt) {
    throw std::runtime_error("injected consume failure");
  }
}

BatchMetric run(Harness& harness,
                std::optional<std::uint32_t> fixedIterations = std::nullopt) {
  const Operation operation = [&harness] { return performOperation(harness); };
  const ClockNow clock = [&harness] { return readClock(harness); };
  const AllocationScope scope{
      .begin = [&harness] { beginCounting(harness); },
      .end = [&harness] { return endCounting(harness); }};
  const Consume consume = [&harness](double checksum) {
    consumeResult(harness, checksum);
  };
  harness.prepared = true;
  return measureCase("tri.solve.n16.rhs4", operation, clock, scope, consume,
                     fixedIterations);
}

void expectFailure(Harness& harness, MeasurementFailure expected,
                   std::optional<std::uint32_t> fixedIterations = 1U) {
  try {
    static_cast<void>(run(harness, fixedIterations));
    VSS_CHECK(false);
  } catch (const MeasurementError& error) {
    VSS_CHECK(error.failure() == expected);
  }
  VSS_CHECK(!harness.counting);
  VSS_CHECK(harness.begins == harness.ends);
}

void testWarmupMinimums() {
  Harness short_duration;
  short_duration.warmupStepNs = 9000000U;
  static_cast<void>(run(short_duration, 1U));
  VSS_CHECK(short_duration.warmups == 12U);

  Harness few_operations;
  few_operations.warmupStepNs = 20000000U;
  static_cast<void>(run(few_operations, 1U));
  VSS_CHECK(few_operations.warmups == 10U);
}

void testCalibration() {
  Harness harness;
  const BatchMetric metric = run(harness);
  VSS_CHECK(metric.caseId == "tri.solve.n16.rhs4");
  VSS_CHECK(metric.iterationsPerSample == 4U);
  VSS_CHECK(harness.scopeOperations.at(0U) == 1U);
  VSS_CHECK(harness.scopeOperations.at(1U) == 1U);
  VSS_CHECK(harness.scopeOperations.at(2U) == 2U);
  VSS_CHECK(harness.scopeOperations.at(3U) == 4U);
  VSS_CHECK(harness.begins == 34U);
  VSS_CHECK(harness.ends == 34U);
  VSS_CHECK(harness.warmups == 10U);
  VSS_CHECK(harness.operations == 138U);
  VSS_CHECK(harness.consumed == 138U);
  VSS_CHECK(metric.checksum == 138.0);
  VSS_CHECK(metric.medianBatchNs == 20000000U);
  VSS_CHECK(metric.madBatchNs == 0U);
  VSS_CHECK(metric.medianNsPerOperation == 5000000U);
  VSS_CHECK(metric.allocationsPerOperation == 3U);
  for (std::size_t index = 4U; index < harness.begins; ++index) {
    VSS_CHECK(harness.scopeOperations.at(index) == 4U);
  }
  for (std::size_t index = 0U; index < harness.eventCount; ++index) {
    if (harness.events.at(index).kind == EventKind::operation) {
      VSS_CHECK(harness.events.at(index + 1U).kind == EventKind::consume);
    }
  }
  VSS_CHECK(!harness.counting);
}

void testLargeCaseAndFixedPlan() {
  Harness large_case;
  large_case.timedStepNs = 25000000U;
  const BatchMetric large_metric = run(large_case);
  VSS_CHECK(large_metric.iterationsPerSample == 1U);
  VSS_CHECK(large_case.begins == 32U);
  VSS_CHECK(large_case.consumed == 42U);

  Harness fixed_case;
  const BatchMetric fixed_metric = run(fixed_case, 64U);
  VSS_CHECK(fixed_metric.iterationsPerSample == 64U);
  VSS_CHECK(fixed_case.begins == 31U);
  VSS_CHECK(fixed_case.consumed == 1931U);
  for (std::size_t index = 1U; index < fixed_case.begins; ++index) {
    VSS_CHECK(fixed_case.scopeOperations.at(index) == 64U);
  }
}

void testStatisticsAndClockBounds() {
  Harness harness;
  harness.useSampleDurations = true;
  const BatchMetric metric = run(harness, 1U);
  VSS_CHECK(metric.batchDurationsNs == kSampleDurations);
  VSS_CHECK(metric.medianBatchNs == 100U);
  VSS_CHECK(metric.madBatchNs == 5U);
  VSS_CHECK(metric.medianNsPerOperation == 100U);
  VSS_CHECK(harness.begins == 31U);
  VSS_CHECK(harness.consumed == 41U);
  VSS_CHECK(metric.checksum == 41.0);

  Harness near_limit;
  near_limit.nowNs = std::numeric_limits<std::uint64_t>::max() - 1000000000U;
  const BatchMetric near_metric = run(near_limit, 1U);
  VSS_CHECK(near_metric.medianBatchNs == 5000000U);
}

void testInvalidTiming() {
  Harness cap;
  cap.timedStepNs = 1U;
  expectFailure(cap, MeasurementFailure::invalid_record, std::nullopt);
  VSS_CHECK(cap.begins == 22U);
  VSS_CHECK(cap.scopeOperations.at(21U) == 1048576U);

  Harness reversed;
  reversed.nowNs = 1000U;
  reversed.reverseClockAt = 2U;
  expectFailure(reversed, MeasurementFailure::invalid_record);

  Harness zero;
  zero.timedStepNs = 0U;
  expectFailure(zero, MeasurementFailure::invalid_record);

  Harness warmup_deadline;
  warmup_deadline.warmupStepNs = 60000000000U;
  expectFailure(warmup_deadline, MeasurementFailure::qualification_incomplete);

  Harness timed_deadline;
  timed_deadline.timedStepNs = 60000000000U;
  expectFailure(timed_deadline, MeasurementFailure::qualification_incomplete);
}

void testCorrectnessAndExceptionCleanup() {
  for (const OperationFault fault :
       {OperationFault::correctness, OperationFault::infinity,
        OperationFault::nan, OperationFault::exception}) {
    for (const std::uint64_t operation : {1U, 2U, 12U}) {
      Harness harness;
      harness.fault = fault;
      harness.faultOperationAt = operation;
      expectFailure(harness, MeasurementFailure::invalid_record);
      VSS_CHECK(harness.consumed == operation - 1U);
    }
  }
  Harness calibration_failure;
  calibration_failure.fault = OperationFault::correctness;
  calibration_failure.faultOperationAt = 12U;
  expectFailure(calibration_failure, MeasurementFailure::invalid_record,
                std::nullopt);

  Harness clock_exception;
  clock_exception.throwClockAt = 2U;
  expectFailure(clock_exception, MeasurementFailure::invalid_record);

  Harness consume_exception;
  consume_exception.throwConsumeAt = 12U;
  expectFailure(consume_exception, MeasurementFailure::invalid_record);

  Harness begin_exception;
  begin_exception.throwBeginAt = 2U;
  expectFailure(begin_exception, MeasurementFailure::invalid_record);

  Harness end_exception;
  end_exception.throwEndAt = 2U;
  expectFailure(end_exception, MeasurementFailure::invalid_record);

  Harness cleanup_exception;
  cleanup_exception.timedStepNs = 60000000000U;
  cleanup_exception.throwEndAt = 2U;
  expectFailure(cleanup_exception,
                MeasurementFailure::qualification_incomplete);
}

void testAllocationFailures() {
  Harness nondivisible;
  nondivisible.changeAllocationAt = 2U;
  nondivisible.allocationAdjustment = 1U;
  expectFailure(nondivisible, MeasurementFailure::allocation_inconsistent, 4U);

  Harness changed_quotient;
  changed_quotient.changeAllocationAt = 2U;
  changed_quotient.allocationAdjustment = 1U;
  expectFailure(changed_quotient, MeasurementFailure::allocation_inconsistent);

  Harness calibration_quotient;
  calibration_quotient.changeAllocationAt = 2U;
  calibration_quotient.allocationAdjustment = 1U;
  expectFailure(calibration_quotient,
                MeasurementFailure::allocation_inconsistent, std::nullopt);

  Harness end_overflow;
  end_overflow.overflowEndAt = 2U;
  expectFailure(end_overflow, MeasurementFailure::invalid_record);

  Harness accumulation_overflow;
  accumulation_overflow.singleOperationAllocations =
      std::numeric_limits<std::uint64_t>::max();
  expectFailure(accumulation_overflow, MeasurementFailure::invalid_record, 2U);
}

void testInvalidArguments() {
  for (const std::uint32_t count : {0U, 3U, 1048577U}) {
    Harness harness;
    expectFailure(harness, MeasurementFailure::invalid_record, count);
    VSS_CHECK(harness.operations == 0U);
  }
  Harness maximum_count;
  maximum_count.fault = OperationFault::correctness;
  maximum_count.faultOperationAt = 1U;
  expectFailure(maximum_count, MeasurementFailure::invalid_record, 1048576U);
  VSS_CHECK(maximum_count.operations == 1U);
  const Operation operation = [] {
    return Probe{.passed = true, .checksum = 1.0};
  };
  const ClockNow clock = [] { return 0U; };
  const AllocationScope scope{.begin = [] {}, .end = [] { return 0U; }};
  const Consume consume = [](double) {};
  VSS_CHECK_THROWS_AS(measureCase("case", {}, clock, scope, consume, 1U),
                      MeasurementError);
  VSS_CHECK_THROWS_AS(measureCase("case", operation, {}, scope, consume, 1U),
                      MeasurementError);
  VSS_CHECK_THROWS_AS(measureCase("case", operation, clock, {}, consume, 1U),
                      MeasurementError);
  VSS_CHECK_THROWS_AS(measureCase("case", operation, clock, scope, {}, 1U),
                      MeasurementError);
  VSS_CHECK_THROWS_AS(measureCase("", operation, clock, scope, consume, 1U),
                      MeasurementError);
}

} // namespace

int main() {
  try {
    testWarmupMinimums();
    testCalibration();
    testLargeCaseAndFixedPlan();
    testStatisticsAndClockBounds();
    testInvalidTiming();
    testCorrectnessAndExceptionCleanup();
    testAllocationFailures();
    testInvalidArguments();
  } catch (const std::exception& error) {
    std::cerr << "unexpected measurement test failure: " << error.what()
              << '\n';
    return 1;
  }
  return 0;
}
