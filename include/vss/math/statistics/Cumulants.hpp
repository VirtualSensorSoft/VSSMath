#ifndef VSS_MATH_STATISTICS_CUMULANTS_HPP
#define VSS_MATH_STATISTICS_CUMULANTS_HPP

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

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_CUMULANTS_HPP
