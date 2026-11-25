#ifndef VSS_MATH_STATISTICS_BOOTSTRAP_HPP
#define VSS_MATH_STATISTICS_BOOTSTRAP_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::statistics {

/// @brief Selects IID or fixed-length block resampling.
enum class BootstrapKind : std::uint8_t { iid, moving_block, circular_block };

/// @brief Selects how empirical endpoint quantiles form an interval.
enum class QuantileInterval : std::uint8_t { percentile, basic };

/// @brief Selects whether invalid replicate statistics fail or are omitted.
enum class InvalidDraw : std::uint8_t { reject, report_conditional };

/// @brief Declares bootstrap design, confidence, and invalid-draw handling.
struct BootstrapPolicy final {
  /// @brief IID, non-wrapping moving-block, or wrapping circular-block draws.
  BootstrapKind kind{BootstrapKind::iid};
  /// @brief Block length; one for IID, otherwise in [1, sample row count].
  Index block_length{1U};
  /// @brief Number of attempted resampled populations.
  Index draws{};
  /// @brief Central interval coverage in (0,1).
  Scalar confidence{};
  /// @brief Percentile endpoints or their reflection about the original value.
  QuantileInterval interval{QuantileInterval::percentile};
  /// @brief Fail on the first invalid statistic or report the valid-draw law.
  InvalidDraw invalid{InvalidDraw::reject};
};

/// @brief Computes a vector statistic from one borrowed complete population.
using Statistic = std::function<Result<Vector>(const SampleView&)>;

/// @brief Owns bootstrap replicates, coordinate intervals, and Monte Carlo
///        standard errors.
struct BootstrapResult final {
  /// @brief Retained statistic vectors as rows in original draw order.
  Matrix estimates;
  /// @brief One row per statistic coordinate, columns lower then upper.
  Matrix intervals;
  /// @brief Replicates omitted under `report_conditional`.
  Index rejected{};
  /// @brief Coordinate standard errors of the replicate mean when defined.
  std::optional<Vector> monte_carlo_error;
};

/// @brief Evaluates a caller-supplied statistic under a bounded bootstrap.
/// @details Resamples complete rows, preserving every within-row coordinate
///          relationship. IID mode draws each row independently. Moving-block
///          and circular-block modes append chronological blocks and truncate
///          only the final block. Basic intervals evaluate the original
///          statistic before consuming random numbers. Callbacks borrow their
///          sample only until return. Conditional results describe the
///          retained valid-replicate law and include the rejected count.
[[nodiscard]] Result<BootstrapResult>
bootstrapEstimate(const SampleView& sample, const Statistic& statistic,
                  const BootstrapPolicy& policy, probability::Rng& rng,
                  Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_BOOTSTRAP_HPP
