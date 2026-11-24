#ifndef VSS_MATH_PROBABILITY_SPECIAL_HPP
#define VSS_MATH_PROBABILITY_SPECIAL_HPP

#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/LogTailPair.hpp>

namespace vss::math::probability {

/// @brief Computes the natural logarithm of the gamma function for x > 0.
/// @param x Positive finite argument.
/// @param policy Positive accuracy tolerances and bounded iteration settings.
/// @return log(Gamma(x)), or a domain, range, or convergence failure.
/// @note Uses a Lanczos approximation with positive-real argument reduction.
[[nodiscard]] Result<Scalar> logGamma(Scalar x, ProbabilityPolicy policy);

/// @brief Computes regularized lower and upper incomplete-gamma tails.
/// @param shape Positive finite shape parameter.
/// @param x Non-negative argument; positive infinity is an exact endpoint.
/// @param policy Accuracy and work limits.
/// @return Logarithms of P(shape,x) and Q(shape,x), respectively.
[[nodiscard]] Result<LogTailPair> incompleteGamma(Scalar shape, Scalar x,
                                                  ProbabilityPolicy policy);

/// @brief Computes regularized lower and upper incomplete-beta tails.
/// @param a,b Positive finite shape parameters.
/// @param x Finite argument in [0,1].
/// @param policy Accuracy and work limits.
/// @return Logarithms of I_x(a,b) and 1-I_x(a,b), respectively.
[[nodiscard]] Result<LogTailPair> incompleteBeta(Scalar a, Scalar b, Scalar x,
                                                 ProbabilityPolicy policy);

/// @brief Computes log(sum(exp(terms[i]))) without overflowing intermediate
///        exponentials.
/// @param terms Borrowed finite values or negative infinities; the view may be
///        empty.
/// @return The stable logarithmic sum, with negative infinity for zero mass.
[[nodiscard]] Result<Scalar> logSumExp(ConstVectorView terms);

} // namespace vss::math::probability

#endif // VSS_MATH_PROBABILITY_SPECIAL_HPP
