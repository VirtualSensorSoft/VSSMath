#ifndef VSS_MATH_STATISTICS_MOMENT_DIAGNOSTICS_HPP
#define VSS_MATH_STATISTICS_MOMENT_DIAGNOSTICS_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {

/// @brief Necessary finite-order consistency checks for supplied moments.
/// @details `symmetric`, `second_psd`, and
///          `degree_two_moment_gram_psd` are independent diagnostics.
///          Minimum eigenvalues are reported in normalized matrix units;
///          multiplying by the accompanying scale recovers the matrix-unit
///          eigenvalue. Tolerances use the same normalized units. A successful
///          result is not proof that a truncated moment table has a unique or
///          realizable probability distribution.
struct MomentValidity final {
  /// @brief Whether all supplied order-one through order-four entries are
  ///        invariant under coordinate permutations within policy tolerance.
  bool symmetric;
  /// @brief Whether the symmetrized second-moment matrix is positive
  ///        semidefinite within the recorded tolerance.
  bool second_psd;
  /// @brief Whether the centered degree-two moment Gram is positive
  ///        semidefinite within the recorded tolerance.
  bool degree_two_moment_gram_psd;
  /// @brief Always false for a generic finite moment table; necessary checks
  ///        do not establish full distributional realizability.
  bool sufficient_for_model;
  /// @brief Absolute tolerance used for permutation-symmetry comparisons.
  Scalar symmetry_tolerance_absolute;
  /// @brief Relative tolerance used for permutation-symmetry comparisons.
  Scalar symmetry_tolerance_relative;
  /// @brief Maximum absolute entry used to normalize the second-moment matrix;
  ///        one when that matrix is identically zero.
  Scalar second_scale;
  /// @brief Minimum eigenvalue of the normalized second-moment matrix.
  Scalar second_minimum_eigenvalue_normalized;
  /// @brief PSD cutoff applied to the normalized second-moment spectrum.
  Scalar second_psd_tolerance_normalized;
  /// @brief Maximum absolute entry used to normalize the degree-two Gram;
  ///        one when that matrix is identically zero.
  Scalar gram_scale;
  /// @brief Minimum eigenvalue of the normalized degree-two moment Gram.
  Scalar gram_minimum_eigenvalue_normalized;
  /// @brief PSD cutoff applied to the normalized Gram spectrum.
  Scalar gram_psd_tolerance_normalized;
};

/// @brief Checks permutation symmetry and necessary PSD conditions through
///        fourth central order.
/// @param moments Complete central moments through order four on compatible
///        support; the source is borrowed and unchanged.
/// @param policy Symmetry tolerances, PSD rank cutoffs and eigensolver sweep
///        limit. Solve-only fields are ignored.
/// @param limits Peak owned and workspace element and byte limits.
/// @return Independent necessary-condition diagnostics, or a typed error when
///         required moments are missing or the input is malformed.
/// @retval ErrorCode::InsufficientSamples A required moment tuple is absent.
/// @retval ErrorCode::ResourceLimit The Gram workspace exceeds limits.
/// @retval ErrorCode::NoConvergence A non-diagonal PSD check has a zero sweep
///         budget or the eigensolver reaches its sweep limit.
/// @note For dimension d, the degree-two Gram is of order
///       `1 + d + d(d+1)/2`; eigen analysis costs O(d^6) time and O(d^4)
///       workspace. No generic realizability claim is made.
[[nodiscard]] Result<MomentValidity>
validateMoments(const MomentEstimate& moments, SolvePolicy policy,
                Limits limits);

/// @brief Computes Mardia's multivariate sample skewness statistic.
/// @param sample Complete finite, unweighted observations in rows.
/// @param policy Full-rank covariance cutoff, conditioning and eigensolver
///        sweep policy; rank-deficient whitening is rejected.
/// @param limits Peak owned and workspace element and byte limits.
/// @return Mardia's descriptive b_(1,d), or a typed validation or numerical
///         failure. No asymptotic p-value is implied.
/// @retval ErrorCode::InsufficientSamples The sample count does not exceed the
///         flattened observation dimension.
/// @retval ErrorCode::RankDeficient The empirical population covariance is not
///         full numerical rank.
/// @retval ErrorCode::NoConvergence The covariance eigensolver reaches its
///         configured sweep limit.
/// @note Time complexity is O(N^2 d + d^3); owned workspace is O(Nd + d^2).
[[nodiscard]] Result<Scalar> mardiaSkewness(const SampleView& sample,
                                            SolvePolicy policy, Limits limits);

/// @brief Computes Mardia's multivariate sample kurtosis statistic.
/// @param sample Complete finite, unweighted observations in rows.
/// @param policy Full-rank covariance cutoff, conditioning and eigensolver
///        sweep policy; rank-deficient whitening is rejected.
/// @param limits Peak owned and workspace element and byte limits.
/// @return Mardia's descriptive b_(2,d), or a typed validation or numerical
///         failure. No asymptotic p-value is implied.
/// @retval ErrorCode::InsufficientSamples The sample count does not exceed the
///         flattened observation dimension.
/// @retval ErrorCode::RankDeficient The empirical population covariance is not
///         full numerical rank.
/// @retval ErrorCode::NoConvergence The covariance eigensolver reaches its
///         configured sweep limit.
/// @note Time complexity is O(N d + d^3); owned workspace is O(Nd + d^2).
[[nodiscard]] Result<Scalar> mardiaKurtosis(const SampleView& sample,
                                            SolvePolicy policy, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_MOMENT_DIAGNOSTICS_HPP
