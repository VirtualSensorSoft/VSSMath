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
  /// Copy the immutable joint law by sharing its retained state.
  JointDistribution(const JointDistribution&) noexcept = default;
  /// Assign the immutable joint law by sharing its retained state.
  JointDistribution& operator=(const JointDistribution&) noexcept = default;
  /// Transfer the immutable joint law state; the source becomes empty.
  JointDistribution(JointDistribution&&) noexcept = default;
  /// Transfer the immutable joint law state; the source becomes empty.
  JointDistribution& operator=(JointDistribution&&) noexcept = default;
  /// Destroy this joint-law handle and release its shared state when unused.
  ~JointDistribution() = default;

  /// Return whether the law is multivariate normal or Student-t.
  [[nodiscard]] JointFamily family() const noexcept;
  /// Return the ambient dimension of the represented random vector.
  [[nodiscard]] Extent dimension() const noexcept;
  /// Return the numerical rank of the covariance or scale support.
  [[nodiscard]] Index supportRank() const noexcept;
  /// Return Student-t degrees of freedom, or positive infinity for normal.
  [[nodiscard]] Scalar degreesOfFreedom() const noexcept;
  /// Borrow the immutable location vector from the retained law state.
  [[nodiscard]] ConstVectorView location() const noexcept;
  /// Borrow the immutable covariance or scale matrix from the retained state.
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
[[nodiscard]] Result<JointDistribution>
conditionalGaussian(const JointDistribution& law, std::span<const Index> keep,
                    std::span<const Index> observed, ConstVectorView values,
                    SolvePolicy policy, Limits limits);
[[nodiscard]] Result<Vector>
multivariateNormalSample(const JointDistribution& law, Rng& rng,
                         ProbabilityPolicy policy, Limits limits);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_GAUSSIAN_HPP
