#ifndef VSS_MATH_SCIENTIFIC_EXTRAS_HPP
#define VSS_MATH_SCIENTIFIC_EXTRAS_HPP

#include <vss/math/Limits.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/linear/FactorizationOptions.hpp>
#include <vss/math/linear/SolvePolicy.hpp>
#include <vss/math/probability/Composite.hpp>
#include <vss/math/probability/Distribution.hpp>

namespace vss::math {

enum class MultiplicityRule : unsigned char {
  bonferroni,
  holm,
  benjamini_hochberg
};

[[nodiscard]] Result<Matrix>
matrixExponential(Matrix matrix, SolvePolicy policy, Limits limits);
[[nodiscard]] Result<Vector>
adjustPValues(ConstVectorView pValues, MultiplicityRule rule, Limits limits);
[[nodiscard]] Result<Scalar>
numericalEntropy(const probability::Distribution& law,
                 probability::IntervalSupport truncation,
                 ProbabilityPolicy policy);
[[nodiscard]] Result<Scalar> numericalKLDivergence(
    const probability::Distribution& p, const probability::Distribution& q,
    probability::IntervalSupport support, ProbabilityPolicy policy);

} // namespace vss::math

#endif // VSS_MATH_SCIENTIFIC_EXTRAS_HPP
