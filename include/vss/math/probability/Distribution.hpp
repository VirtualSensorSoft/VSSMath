#ifndef VSS_MATH_PROBABILITY_DISTRIBUTION_HPP
#define VSS_MATH_PROBABILITY_DISTRIBUTION_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/SampleView.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/probability/LogTailPair.hpp>

namespace vss::math::probability {

enum class Family : std::uint8_t {
  normal,
  student_t,
  laplace,
  logistic,
  lognormal,
  gamma,
  beta,
  weibull,
  skew_normal,
  uniform,
  triangular,
  truncated_normal,
  bernoulli,
  binomial,
  poisson,
  negative_binomial,
  chi_square,
  fisher_f,
  mixture,
  finite_convolution,
  empirical,
  truncated,
  multivariate_normal,
  multivariate_student_t,
  empirical_joint
};

enum class Measure : std::uint8_t {
  real_density,
  integer_mass,
  atom_mass,
  multivariate_density,
  subspace_density,
  joint_atom_mass
};

/// @brief Read-only locations and masses for a stored finite atom table.
/// @details The spans are aligned by index and borrowed from their owning
///          immutable distribution. Both are empty when no explicit finite
///          atom table is available.
struct FiniteAtomView final {
  /// Sorted finite atom locations.
  std::span<const Scalar> locations;
  /// Non-negative masses aligned with @ref locations.
  std::span<const Scalar> masses;
};

namespace detail {
class DistributionFactory;
class DistributionEngine;
} // namespace detail

/// @brief Immutable, owning representation of a scalar probability law.
/// @details Factories copy borrowed inputs. Copies share immutable state, and
///          moved-from values may only be destroyed or assigned; operations
///          on them return InvalidDomain. No public default constructor is
///          provided.
class Distribution final {
public:
  Distribution() = delete;
  /// Copy an immutable law by sharing its retained state.
  Distribution(const Distribution&) noexcept = default;
  /// Assign an immutable law by sharing its retained state.
  Distribution& operator=(const Distribution&) noexcept = default;
  /// Transfer the immutable law state; the source becomes empty.
  Distribution(Distribution&&) noexcept = default;
  /// Transfer the immutable law state; the source becomes empty.
  Distribution& operator=(Distribution&&) noexcept = default;
  /// Destroy this law handle and release its shared state when unused.
  ~Distribution() = default;

  /// Return the scalar law family represented by this value.
  [[nodiscard]] Family family() const noexcept;
  /// Return whether evaluation uses a density, mass, or mixed measure.
  [[nodiscard]] Measure measure() const noexcept;
  /// Return family-specific parameters in canonical order.
  [[nodiscard]] std::span<const Scalar> parameters() const noexcept;
  /// Return the explicitly stored finite atoms, when available.
  /// @details Empirical and materialized finite-convolution laws expose their
  ///          atom table. An empty view does not imply a continuous law.
  [[nodiscard]] FiniteAtomView finiteAtoms() const noexcept;

private:
  struct State;
  friend class detail::DistributionFactory;
  friend class detail::DistributionEngine;
  explicit Distribution(std::shared_ptr<const State> state) noexcept;
  std::shared_ptr<const State> state_;
};

/// @brief A scalar law and its non-negative mixture weight.
struct WeightedLaw final {
  /// Immutable component law included in a mixture.
  Distribution law;
  /// Unnormalized non-negative component weight.
  Scalar weight;
};

/// @brief Density/mass and both tails returned by one law evaluation.
struct TailValues final {
  /// Probability density or probability mass at the queried value.
  Scalar density_or_mass{};
  /// Natural logarithm of the density or mass.
  Scalar log_density_or_mass{};
  /// Cumulative probability at the queried value.
  Scalar cdf{};
  /// Complementary probability at the queried value.
  Scalar survival{};
  /// Logarithms of the cumulative and complementary probabilities.
  LogTailPair log_tails{};
  /// Whether the queried value belongs to the law's support.
  bool on_support{};
};

/// @brief Small deterministic pseudo-random generator for probability sampling.
class Rng final {
public:
  /// Initialize the generator from a reproducible 64-bit seed.
  explicit Rng(std::uint64_t seed) noexcept;
  /// Return the next 64-bit value and advance the generator state.
  [[nodiscard]] std::uint64_t next() noexcept;

private:
  std::array<std::uint64_t, 4> state_{};
};

[[nodiscard]] Result<Distribution> normal(Scalar mean, Scalar sd);
[[nodiscard]] Result<Distribution> studentT(Scalar nu, Scalar location,
                                            Scalar scale);
[[nodiscard]] Result<Distribution> laplace(Scalar location, Scalar scale);
[[nodiscard]] Result<Distribution> logistic(Scalar location, Scalar scale);
[[nodiscard]] Result<Distribution> lognormal(Scalar logMean, Scalar logSd);
[[nodiscard]] Result<Distribution> gamma(Scalar shape, Scalar scale);
[[nodiscard]] Result<Distribution> beta(Scalar shapeA, Scalar shapeB);
[[nodiscard]] Result<Distribution> weibull(Scalar shape, Scalar scale);
[[nodiscard]] Result<Distribution> skewNormal(Scalar location, Scalar scale,
                                              Scalar shape);
[[nodiscard]] Result<Distribution> uniform(Scalar lower, Scalar upper);
[[nodiscard]] Result<Distribution> triangular(Scalar lower, Scalar mode,
                                              Scalar upper);
[[nodiscard]] Result<Distribution> truncatedNormal(Scalar mean, Scalar sd,
                                                   Scalar lower, Scalar upper);
[[nodiscard]] Result<Distribution> bernoulli(Scalar success);
[[nodiscard]] Result<Distribution> binomial(std::uint64_t trials,
                                            Scalar success);
[[nodiscard]] Result<Distribution> poisson(Scalar rate);
[[nodiscard]] Result<Distribution> negativeBinomial(Scalar successes,
                                                    Scalar successProbability);
[[nodiscard]] Result<Distribution> chiSquare(Scalar degreesOfFreedom);
[[nodiscard]] Result<Distribution> fisherF(Scalar numeratorDof,
                                           Scalar denominatorDof);
[[nodiscard]] Result<Distribution>
empirical(ConstVectorView observations, ConstVectorView masses, Limits limits);
[[nodiscard]] Result<Distribution> mixture(std::span<const WeightedLaw> terms,
                                           ProbabilityPolicy policy);
[[nodiscard]] Result<Distribution> mixture(std::span<const WeightedLaw> terms,
                                           ProbabilityPolicy policy,
                                           Limits limits);
[[nodiscard]] Result<Distribution>
finiteConvolution(std::span<const Distribution> independentTerms,
                  ProbabilityPolicy policy);
[[nodiscard]] Result<Distribution>
finiteConvolution(std::span<const Distribution> independentTerms,
                  ProbabilityPolicy policy, Limits limits);
[[nodiscard]] Result<TailValues> evaluate(const Distribution& law, Scalar x,
                                          ProbabilityPolicy policy);
[[nodiscard]] Result<Scalar>
quantile(const Distribution& law, Scalar probability, ProbabilityPolicy policy);
[[nodiscard]] Result<Scalar> sample(const Distribution& law, Rng& rng,
                                    ProbabilityPolicy policy);
[[nodiscard]] Result<Scalar> rawMoment(const Distribution& law, Index order,
                                       ProbabilityPolicy policy);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_DISTRIBUTION_HPP
