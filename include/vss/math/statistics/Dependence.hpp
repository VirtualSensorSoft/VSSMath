#ifndef VSS_MATH_STATISTICS_DEPENDENCE_HPP
#define VSS_MATH_STATISTICS_DEPENDENCE_HPP

#include <vss/math/Core.hpp>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math::statistics {

/// @brief Computes Spearman rank correlation with average ranks for ties.
/// @details Signed zeros compare equal. Inputs must be finite, equal length and
///          contain at least two values with nonzero rank variance. The
///          operation sorts O(N) records and honors element, byte and
///          evaluation limits before allocating workspace.
[[nodiscard]] Result<Scalar>
spearmanCorrelation(ConstVectorView x, ConstVectorView y, Limits limits);

/// @brief Computes Kendall's tau-b using exact unordered-pair enumeration.
/// @details Concordant and discordant pairs form the numerator; pairs tied
///          only in one variable enter the corresponding tau-b denominator
///          factor. Joint ties enter neither factor. A zero denominator is an
///          invalid domain. Pair comparisons are bounded by `limits`.
[[nodiscard]] Result<Scalar> kendallTau(ConstVectorView x, ConstVectorView y,
                                        Limits limits);

/// @brief Computes the biased empirical distance correlation.
/// @details Each input is a complete row population; their flattened
///          coordinate dimensions may differ, but row counts must match.
///          The implementation uses O(N) workspace and O(N^2(d_x+d_y))
///          distance work, recomputing distances rather than storing N-by-N
///          matrices. Constant populations have undefined correlation.
[[nodiscard]] Result<Scalar>
distanceCorrelation(const SampleView& x, const SampleView& y, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_DEPENDENCE_HPP
