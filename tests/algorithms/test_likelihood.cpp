#include "Test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/ProbabilityPolicy.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/VectorView.hpp>
#include <vss/math/probability/Distribution.hpp>
#include <vss/math/probability/Likelihood.hpp>

namespace {
using namespace vss::math;
using namespace vss::math::probability;

ProbabilityPolicy policy() {
  return {{1.0e-13, 1.0e-13, 1000U, 10000U, std::nullopt}, 10000U, 2.0e-9};
}

void independentBernoulliLikelihood() {
  const auto law = bernoulli(0.25);
  VSS_CHECK(law.hasValue());
  const std::array<Scalar, 2> values{1.0, 1.0};
  const auto view = ConstVectorView::create(values, values.size());
  VSS_CHECK(view.hasValue());
  if (law && view) {
    const auto result = logLikelihood(law.value(), view.value(), policy());
    VSS_CHECK(result.hasValue());
    if (result) {
      VSS_CHECK(std::abs(result.value() - std::log(1.0 / 16.0)) < 1.0e-14);
    }
  }
}

void invalidAndZeroLikelihoodInputs() {
  const auto law = normal(0.0, 1.0);
  VSS_CHECK(law.hasValue());
  const std::array<Scalar, 1> nonFinite{
      std::numeric_limits<Scalar>::infinity()};
  const auto view = ConstVectorView::create(nonFinite, nonFinite.size());
  VSS_CHECK(view.hasValue());
  if (law && view) {
    VSS_CHECK(
        logLikelihood(law.value(), view.value(), policy()).status().code() ==
        ErrorCode::NonFiniteInput);
    const auto empty = ConstVectorView::create({}, 0U);
    VSS_CHECK(empty.hasValue());
    if (empty) {
      VSS_CHECK(logLikelihood(law.value(), empty.value(), policy()).value() ==
                0.0);
    }
  }
}

} // namespace

int main() {
  independentBernoulliLikelihood();
  invalidAndZeroLikelihoodInputs();
}

// VSS-MATH-TRACEABILITY: C060 CASE-C060.loglikelihood-independent
