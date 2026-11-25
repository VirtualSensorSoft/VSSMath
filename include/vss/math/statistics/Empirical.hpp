#ifndef VSS_MATH_STATISTICS_EMPIRICAL_HPP
#define VSS_MATH_STATISTICS_EMPIRICAL_HPP

#include <cstdint>
#include <optional>
#include <span>
#include <vss/math/Array.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/covariance/PropagationResult.hpp>
#include <vss/math/statistics/Weighted.hpp>

namespace vss::math::statistics {

/// @brief Selects the contents returned by an empirical histogram.
enum class HistogramNormalization : std::uint8_t { counts, mass, density };

/// @brief Selects the one-dimensional kernel used by product KDE.
enum class KernelKind : std::uint8_t { gaussian, epanechnikov };

/// @brief Owns row-major histogram bins and their axis edge definitions.
struct Histogram final {
  /// @brief Number of bins along each observation axis.
  Array<Index> extents;
  /// @brief Deep copies of the strictly increasing axis edges.
  Array<Vector> edges;
  /// @brief Flattened bins in row-major axis order.
  Vector bins;
  /// @brief Mass of rows below an edge; ND rows are counted once.
  Scalar underflow{};
  /// @brief Mass of rows above an edge, excluding rows already underflowed.
  Scalar overflow{};
  /// @brief Total mass admitted to in-range bins.
  Scalar accepted{};
};

/// @brief Declares product-kernel bandwidths and optional reflecting bounds.
struct KernelSpec final {
  /// @brief Gaussian or compact-support Epanechnikov kernel.
  KernelKind kernel;
  /// @brief One bandwidth broadcast to all coordinates or one per coordinate.
  ConstVectorView bandwidth;
  /// @brief Optional d-by-2 lower/upper reflecting interval bounds.
  std::optional<ConstMatrixView> reflection_bounds;
};

/// @brief Computes a one-dimensional unweighted histogram.
[[nodiscard]] Result<Histogram> histogram(ConstVectorView values,
                                          ConstVectorView edges,
                                          HistogramNormalization mode,
                                          Limits limits);

/// @brief Computes a one-dimensional row-weighted histogram.
[[nodiscard]] Result<Histogram> weightedHistogram(ConstVectorView values,
                                                  ConstVectorView weights,
                                                  ConstVectorView edges,
                                                  HistogramNormalization mode,
                                                  Limits limits);

/// @brief Computes a bounded multidimensional weighted histogram.
[[nodiscard]] Result<Histogram> histogramND(const SampleView& values,
                                            std::span<const Vector> edges,
                                            const WeightSpec& weights,
                                            HistogramNormalization mode,
                                            Limits limits);

/// @brief Evaluates a weighted product-kernel density estimate.
[[nodiscard]] Result<Vector>
kernelDensityEstimate(const SampleView& observations, const WeightSpec& weights,
                      const SampleView& evaluation, const KernelSpec& kernel,
                      Limits limits);

/// @brief Evaluates a reflected product KDE to a requested Gaussian tail bound.
[[nodiscard]] Result<Vector>
kernelDensityEstimate(const SampleView& observations, const WeightSpec& weights,
                      const SampleView& evaluation, const KernelSpec& kernel,
                      ProbabilityPolicy accuracy, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_EMPIRICAL_HPP
