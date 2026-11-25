#ifndef VSS_MATH_STATISTICS_DIAGNOSTICS_HPP
#define VSS_MATH_STATISTICS_DIAGNOSTICS_HPP

#include <functional>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/statistics/Inference.hpp>

namespace vss::math::statistics {

/// @brief Legacy scalar-family goodness-of-fit refit request.
struct GoodnessSpec final {
  /// @brief Return only a statistic and leave inferential fields absent.
  bool statistic_only{};
  /// @brief Number of parametric replicates for calibrated calls.
  Index replicates{};
  /// @brief Sampling design metadata; `draws` must equal `replicates`.
  BootstrapPolicy calibration{};
  /// @brief Optional fitted scalar parameters in the family's constructor
  /// order.
  std::function<Result<Vector>(ConstVectorView)> refit_parameters;
};

/// @brief Goodness-of-fit calibration with an owning refitted law callback.
struct GoodnessCalibration final {
  /// @brief Return only the statistic and omit inferential fields.
  bool statistic_only{};
  /// @brief Number of parametric replicates for calibrated calls.
  Index replicates{};
  /// @brief Optional procedure that refits and returns a complete scalar law.
  std::function<Result<probability::Distribution>(ConstVectorView)> refit_law;
};

/// @brief Computes Akaike's information criterion, -2 log L + 2k.
[[nodiscard]] Result<Scalar> aic(Scalar logLikelihood, Index parameters);

/// @brief Computes BIC, -2 log L + k log(n), for independent count n>0.
[[nodiscard]] Result<Scalar> bic(Scalar logLikelihood, Index parameters,
                                 Index independentCount);

/// @brief Computes scalar Kolmogorov–Smirnov discrepancy and calibration.
[[nodiscard]] Result<TestResult> ks(const SampleView& sample,
                                    const probability::Distribution& law,
                                    const GoodnessSpec& specification,
                                    probability::Rng& rng, Limits limits);

/// @brief Computes scalar Anderson–Darling discrepancy and calibration.
[[nodiscard]] Result<TestResult>
andersonDarling(const SampleView& sample, const probability::Distribution& law,
                const GoodnessSpec& specification, probability::Rng& rng,
                Limits limits);

/// @brief Computes KS with an explicit complete-law refit procedure.
[[nodiscard]] Result<TestResult> ks(const SampleView& sample,
                                    const probability::Distribution& law,
                                    const GoodnessCalibration& calibration,
                                    probability::Rng& rng, Limits limits);

/// @brief Computes AD with an explicit complete-law refit procedure.
[[nodiscard]] Result<TestResult>
andersonDarling(const SampleView& sample, const probability::Distribution& law,
                const GoodnessCalibration& calibration, probability::Rng& rng,
                Limits limits);

} // namespace vss::math::statistics

#endif // VSS_MATH_STATISTICS_DIAGNOSTICS_HPP
