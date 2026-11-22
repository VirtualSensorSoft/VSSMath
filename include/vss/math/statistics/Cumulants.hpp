#ifndef VSS_MATH_STATISTICS_CUMULANTS_HPP
#define VSS_MATH_STATISTICS_CUMULANTS_HPP

#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/statistics/Moments.hpp>

namespace vss::math::statistics {

/// @brief Converts central moments about the population mean to cumulants.
/// @param moments Owned central moment terms with compatible support.
/// @param request Requested output orders and tuple representation. A new
///        contraction map is not accepted; convert already-contracted terms
///        in their declared output coordinates instead.
/// @param limits Maximum output and simultaneously required workspace.
/// @return Owned cumulant terms with the input shape and classification
///         metadata, or a typed validation, resource, or numerical error.
/// @details The first cumulant is the stored population mean. Orders two and
///          three equal the corresponding central moments. The fourth
///          cumulant subtracts all three pairings of the second central
///          moment. Required source entries must be present; missing mixed
///          terms are never substituted with zero.
[[nodiscard]] Result<MomentEstimate>
centralMomentsToCumulants(const MomentEstimate& moments,
                          const MomentRequest& request, Limits limits);

/// @brief Converts cumulants to central moments about the population mean.
/// @param cumulants Owned cumulant terms with compatible support.
/// @param request Requested output orders and tuple representation. A new
///        contraction map is not accepted; convert already-contracted terms
///        in their declared output coordinates instead.
/// @param limits Maximum output and simultaneously required workspace.
/// @return Owned central moment terms with exact zero first central moments
///         and preserved population-mean metadata when available.
/// @details Orders two and three equal their cumulants. The fourth central
///          moment adds all three covariance pairings. A complete first
///          cumulant term may supply missing population-mean metadata.
[[nodiscard]] Result<MomentEstimate>
cumulantsToCentralMoments(const MomentEstimate& cumulants,
                          const MomentRequest& request, Limits limits);

/// @brief Adds cumulant records for mutually independent random vectors.
/// @param inputs Borrowed cumulant estimates with the same observation shape.
/// @param coefficients One finite scalar multiplier per independent input.
/// @param request Requested output orders and optional coordinate contraction.
/// @param limits Maximum output and simultaneously required workspace.
/// @return The cumulants of the weighted sum, or a typed validation, resource,
///         or numerical error.
/// @details The caller asserts mutual independence between input records.
///          Reusing the same record is valid only when it represents
///          independent draws. Order k is combined as the sum of each input
///          cumulant multiplied by its coefficient to the kth power. A zero
///          coefficient requires a matching shape but no numeric terms.
///          Independent operands may have different sample supports; the
///          result therefore does not infer shared support or sampling error.
///          A contraction is applied multilinearly to every requested
///          cumulant order; exact zero map entries remove their corresponding
///          input dependencies. Contracted output terms use dense layout.
[[nodiscard]] Result<MomentEstimate>
cumulantsOfIndependentSum(std::span<const MomentEstimate> inputs,
                          ConstVectorView coefficients,
                          const MomentRequest& request, Limits limits);

/// @brief Computes central moments of a weighted sum of independent inputs.
/// @param coefficients One finite scalar multiplier per independent input.
/// @param inputs Borrowed mean-centered central-moment estimates.
/// @param request Requested output orders and optional coordinate contraction.
/// @param limits Maximum output and simultaneously required workspace.
/// @return The central moments of the weighted independent sum, or a typed
///         validation, resource, or numerical error.
/// @details Converts the required input terms to cumulants, applies
///          @ref cumulantsOfIndependentSum, and converts the result back to
///          central moments. The caller asserts mutual independence. When
///          fourth-order output is requested, second-order cumulants are also
///          computed internally for the three inverse-conversion pairings,
///          including when only selected fourth-order tuples are requested.
///          An optional contraction is applied to input cumulants before the
///          independent sum; exact zero map entries do not require the
///          corresponding input terms.
[[nodiscard]] Result<MomentEstimate>
weightedSum(ConstVectorView coefficients,
            std::span<const MomentEstimate> inputs,
            const MomentRequest& request, Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_CUMULANTS_HPP
