#ifndef VSS_MATH_PROBABILITY_SAMPLING_DESIGN_HPP
#define VSS_MATH_PROBABILITY_SAMPLING_DESIGN_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::probability {

/// @brief Identifies the deterministic construction used for a design.
enum class SamplingDesignKind : std::uint8_t { sobol, latin_hypercube };

/// @brief Owned metadata needed to interpret and reproduce a design.
struct SamplingDesignProvenance final {
  /// Sampling construction whose exact version is recorded below.
  SamplingDesignKind kind{SamplingDesignKind::latin_hypercube};
  /// Version of the coordinate and permutation algorithms.
  std::uint32_t algorithm_version{1U};
  /// Number of most-significant coordinate bits converted to Scalar.
  std::uint32_t coordinate_bits{52U};
  /// Number of direction bits used by a Sobol sequence; zero for LHS.
  std::uint32_t direction_bits{0U};
  /// Whether a fixed seed-derived digital shift was applied.
  bool digital_shift{false};
  /// Seed used to derive a Sobol shift; absent for unscrambled Sobol and LHS.
  std::optional<std::uint64_t> seed;
  /// SHA-256 identity of the direction table; empty for LHS.
  std::string direction_table_sha256;
  /// Absolute first sequence index; zero for prefix and LHS designs.
  std::uint64_t first_index{0U};
  /// Number of coordinates in each point.
  Index dimension{0U};
  /// Number of generated points.
  Index count{0U};
};

/// @brief Owns points from one reproducible sampling design.
struct SampleDesign final {
  /// Owning row-major matrix with one sampled point per row.
  Matrix points;
  /// Seed of a seeded Sobol design, or zero when no design seed is applicable.
  std::uint64_t seed{0U};
  /// Stable algorithm name and version string.
  std::string algorithm;
  /// Reproducibility metadata populated for every successful design.
  std::optional<SamplingDesignProvenance> provenance{std::nullopt};
};

/// @brief Generate a randomized Latin hypercube in the unit cube.
/// @details Each of the `count` rows is a point with `dimensions` coordinates.
///          Every output column contains exactly one point in each equal-width
///          stratum. A caller-owned generator supplies each stratum jitter and
///          each Fisher–Yates permutation; this function never seeds it.
///          Validation and resource preflight complete before the first draw.
/// @param dimensions Positive number of unit-cube dimensions.
/// @param count Positive number of points/strata.
/// @param rng Caller-owned reproducible generator; it is advanced only after
///            all input and resource checks pass.
/// @param limits Output element and peak-byte limits; callback evaluations
///               are not used.
/// @return An owning design with versioned provenance, or a diagnostic error.
[[nodiscard]] Result<SampleDesign> latinHypercube(Index dimensions, Index count,
                                                  Rng& rng, Limits limits);

/// @brief Generate a power-of-two Sobol prefix from pinned Joe–Kuo D6 data.
[[nodiscard]] Result<SampleDesign> sobolDesign(Index dimensions,
                                               Index log2Count,
                                               std::uint64_t seed,
                                               bool scramble, Limits limits);

/// @brief Generate an indexed Sobol range without walking its prefix.
[[nodiscard]] Result<SampleDesign>
sobolDesignRange(Index dimensions, std::uint64_t firstIndex, Index count,
                 std::uint64_t seed, bool scramble, Limits limits);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_SAMPLING_DESIGN_HPP
