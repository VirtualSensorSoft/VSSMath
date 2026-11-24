#ifndef VSS_MATH_PROBABILITY_GAUSSIAN_HPP
#define VSS_MATH_PROBABILITY_GAUSSIAN_HPP

#include <memory>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math::probability {

enum class JointFamily : std::uint8_t { normal, student_t };

namespace detail {
class JointDistributionFactory;
class JointDistributionEngine;
class LikelihoodEngine;
class JointMomentsEngine;
} // namespace detail

/// @brief Immutable owning multivariate normal or Student-t law.
/// @details The retained covariance/scale eigenspace defines intrinsic
///          support. Views borrow from this object's immutable shared state.
class JointDistribution final {
public:
  struct State;
  JointDistribution() = delete;
  JointDistribution(const JointDistribution&) noexcept = default;
  JointDistribution& operator=(const JointDistribution&) noexcept = default;
  JointDistribution(JointDistribution&&) noexcept = default;
  JointDistribution& operator=(JointDistribution&&) noexcept = default;
  ~JointDistribution() = default;

  [[nodiscard]] JointFamily family() const noexcept;
  [[nodiscard]] Extent dimension() const noexcept;
  [[nodiscard]] Index supportRank() const noexcept;
  [[nodiscard]] Scalar degreesOfFreedom() const noexcept;
  [[nodiscard]] ConstVectorView location() const noexcept;
  [[nodiscard]] ConstMatrixView scale() const noexcept;

private:
  friend class detail::JointDistributionFactory;
  friend class detail::JointDistributionEngine;
  friend class detail::LikelihoodEngine;
  friend class detail::JointMomentsEngine;
  friend Result<JointDistribution>
  conditionalGaussian(const JointDistribution& law, std::span<const Index> keep,
                      std::span<const Index> observed, ConstVectorView values,
                      SolvePolicy policy, Limits limits);
  explicit JointDistribution(std::shared_ptr<const State> state) noexcept;
  std::shared_ptr<const State> state_;
};

[[nodiscard]] Result<JointDistribution>
multivariateNormal(ConstVectorView location, ConstMatrixView covariance,
                   SolvePolicy policy, Limits limits);
[[nodiscard]] Result<JointDistribution>
multivariateStudentT(ConstVectorView location, ConstMatrixView scale,
                     Scalar degreesOfFreedom, SolvePolicy policy,
                     Limits limits);
[[nodiscard]] Result<Vector> mean(const JointDistribution& law,
                                  ProbabilityPolicy policy, Limits limits);
[[nodiscard]] Result<Matrix> covariance(const JointDistribution& law,
                                        ProbabilityPolicy policy,
                                        Limits limits);
[[nodiscard]] Result<JointDistribution>
conditionalGaussian(const JointDistribution& law, std::span<const Index> keep,
                    std::span<const Index> observed, ConstVectorView values,
                    SolvePolicy policy, Limits limits);
[[nodiscard]] Result<Vector>
multivariateNormalSample(const JointDistribution& law, Rng& rng,
                         ProbabilityPolicy policy, Limits limits);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_GAUSSIAN_HPP
