#ifndef VSS_MATH_DIFFERENTIATION_FINITE_DIFFERENCE_HPP
#define VSS_MATH_DIFFERENTIATION_FINITE_DIFFERENCE_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vss/math/Core.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/MatrixView.hpp>
#include <vss/math/Result.hpp>
#include <vss/math/Tolerance.hpp>
#include <vss/math/Vector.hpp>
#include <vss/math/VectorView.hpp>

namespace vss::math::differentiation {

namespace detail {
struct DerivativeCheckAccess;
} // namespace detail

/// @brief Scalar callback evaluated at one immutable parameter vector.
/// @details The view is valid only during the callback invocation and must not
/// be retained. A failed result is wrapped as CallbackFailure; exceptions
/// propagate unchanged.
using ScalarFunction = std::function<Result<Scalar>(ConstVectorView)>;

/// @brief Vector callback evaluated at one immutable parameter vector.
/// @details The view is valid only during the callback invocation and must not
/// be retained. The base result fixes the Jacobian output extent.
using VectorFunction = std::function<Result<Vector>(ConstVectorView)>;

/// @brief Second-order finite-difference stencil selected for one parameter.
enum class DifferenceFormula : std::uint8_t {
  /// @brief Uses samples at both sides of the base value.
  Central,
  /// @brief Uses two samples above the base value.
  Forward,
  /// @brief Uses two samples below the base value.
  Backward
};

/// @brief Scale and optional bounds controlling one parameter step.
/// @details Public operations validate finite, strictly positive scales and
/// explicit steps, finite ordered bounds, and base containment before calling
/// a user callback.
// NOLINTBEGIN(misc-non-private-member-variables-in-classes): this is a
// normative public aggregate whose fields are part of the configured API.
struct ParameterStep final {
  /// @brief Constructs a step configuration with no explicit step or bounds.
  /// @param parameterScale Positive scale used by the default proposal.
  explicit ParameterStep(Scalar parameterScale) noexcept;

  /// @brief Scale used by the default step proposal.
  Scalar scale;
  /// @brief Optional caller-specified finite positive step.
  std::optional<Scalar> explicitStep;
  /// @brief Optional finite lower bound for the parameter.
  std::optional<Scalar> lowerBound;
  /// @brief Optional finite upper bound for the parameter.
  std::optional<Scalar> upperBound;
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

class DerivativeCheckResult;

/// @brief Owns one numerical gradient and stencil evidence.
/// @details Values, effective steps, and formulas are deep-owned. Moving
/// transfers storage and leaves the source valid but unspecified.
class GradientResult final {
public:
  /// @brief Default construction is unavailable.
  GradientResult() = delete;
  /// @brief Deep-copies a gradient result.
  GradientResult(const GradientResult&) = default;
  /// @brief Replaces this result with a deep copy.
  GradientResult& operator=(const GradientResult&) = default;
  /// @brief Transfers owned storage.
  GradientResult(GradientResult&&) noexcept;
  /// @brief Replaces this result by transferring owned storage.
  GradientResult& operator=(GradientResult&&) noexcept;
  /// @brief Destroys owned gradient evidence.
  ~GradientResult() = default;

  /// @brief Returns the numerical derivative for each parameter.
  [[nodiscard]] const Vector& values() const noexcept { return values_; }
  /// @brief Returns the accepted effective step for each parameter.
  [[nodiscard]] const Vector& effectiveSteps() const noexcept {
    return effectiveSteps_;
  }
  /// @brief Returns the selected formula for each parameter.
  [[nodiscard]] std::span<const DifferenceFormula> formulas() const noexcept {
    return formulas_;
  }

private:
  GradientResult(Vector values, Vector effectiveSteps,
                 std::vector<DifferenceFormula> formulas) noexcept;

  friend Result<GradientResult>
  numericalGradient(const ScalarFunction& function, ConstVectorView parameters,
                    std::span<const ParameterStep> steps);
  friend Result<DerivativeCheckResult>
  checkGradient(ConstVectorView analyticGradient,
                const GradientResult& numericalGradient,
                Tolerance comparisonTolerance);

  Vector values_;
  Vector effectiveSteps_;
  std::vector<DifferenceFormula> formulas_;
};

/// @brief Owns one numerical Jacobian and stencil evidence.
/// @details Values are row-major with output rows and parameter columns.
/// Effective steps and formulas are indexed by parameter.
class JacobianResult final {
public:
  /// @brief Default construction is unavailable.
  JacobianResult() = delete;
  /// @brief Deep-copies a Jacobian result.
  JacobianResult(const JacobianResult&) = default;
  /// @brief Replaces this result with a deep copy.
  JacobianResult& operator=(const JacobianResult&) = default;
  /// @brief Transfers owned storage.
  JacobianResult(JacobianResult&&) noexcept;
  /// @brief Replaces this result by transferring owned storage.
  JacobianResult& operator=(JacobianResult&&) noexcept;
  /// @brief Destroys owned Jacobian evidence.
  ~JacobianResult() = default;

  /// @brief Returns the row-major output-by-parameter Jacobian.
  [[nodiscard]] const Matrix& values() const noexcept { return values_; }
  /// @brief Returns the accepted effective step for each parameter.
  [[nodiscard]] const Vector& effectiveSteps() const noexcept {
    return effectiveSteps_;
  }
  /// @brief Returns the selected formula for each parameter.
  [[nodiscard]] std::span<const DifferenceFormula> formulas() const noexcept {
    return formulas_;
  }

private:
  JacobianResult(Matrix values, Vector effectiveSteps,
                 std::vector<DifferenceFormula> formulas) noexcept;

  friend Result<JacobianResult>
  numericalJacobian(const VectorFunction& function, ConstVectorView parameters,
                    std::span<const ParameterStep> steps);
  friend Result<DerivativeCheckResult>
  checkJacobian(ConstMatrixView analyticJacobian,
                const JacobianResult& numericalJacobian,
                Tolerance comparisonTolerance);

  Matrix values_;
  Vector effectiveSteps_;
  std::vector<DifferenceFormula> formulas_;
};

/// @brief Owns analytic/numerical derivative comparison evidence.
/// @details Matrices are row-major and have equal extents. Gradient checks use
/// one row; Jacobian checks preserve output-row by parameter-column ordering.
class DerivativeCheckResult final {
public:
  /// @brief Default construction is unavailable.
  DerivativeCheckResult() = delete;
  /// @brief Deep-copies comparison evidence.
  DerivativeCheckResult(const DerivativeCheckResult&) = default;
  /// @brief Replaces this result with a deep copy.
  DerivativeCheckResult& operator=(const DerivativeCheckResult&) = default;
  /// @brief Transfers owned comparison evidence.
  DerivativeCheckResult(DerivativeCheckResult&&) noexcept;
  /// @brief Replaces this result by transferring owned evidence.
  DerivativeCheckResult& operator=(DerivativeCheckResult&&) noexcept;
  /// @brief Destroys owned comparison evidence.
  ~DerivativeCheckResult() = default;

  /// @brief Returns the row extent of the comparison matrices.
  [[nodiscard]] Extent rows() const noexcept { return analyticValues_.rows(); }
  /// @brief Returns the column extent of the comparison matrices.
  [[nodiscard]] Extent columns() const noexcept {
    return analyticValues_.columns();
  }
  /// @brief Returns copied analytic derivative values.
  [[nodiscard]] const Matrix& analyticValues() const noexcept {
    return analyticValues_;
  }
  /// @brief Returns copied numerical derivative values.
  [[nodiscard]] const Matrix& numericalValues() const noexcept {
    return numericalValues_;
  }
  /// @brief Returns per-entry absolute errors.
  [[nodiscard]] const Matrix& absoluteErrors() const noexcept {
    return absoluteErrors_;
  }
  /// @brief Returns per-entry relative errors.
  [[nodiscard]] const Matrix& relativeErrors() const noexcept {
    return relativeErrors_;
  }
  /// @brief Returns numerical effective steps for each parameter.
  [[nodiscard]] const Vector& effectiveSteps() const noexcept {
    return effectiveSteps_;
  }
  /// @brief Returns whether one entry passed the comparison tolerance.
  /// @throws std::out_of_range when either index is outside the matrix.
  [[nodiscard]] bool passed(Index row, Index column) const;
  /// @brief Reports whether every comparison entry passed.
  [[nodiscard]] bool allPassed() const noexcept { return allPassed_; }

private:
  friend struct detail::DerivativeCheckAccess;

  DerivativeCheckResult(Matrix analyticValues, Matrix numericalValues,
                        Matrix absoluteErrors, Matrix relativeErrors,
                        Vector effectiveSteps,
                        std::vector<std::uint8_t> passed) noexcept;

  friend Result<DerivativeCheckResult>
  checkGradient(ConstVectorView analyticGradient,
                const GradientResult& numericalGradient,
                Tolerance comparisonTolerance);
  friend Result<DerivativeCheckResult>
  checkJacobian(ConstMatrixView analyticJacobian,
                const JacobianResult& numericalJacobian,
                Tolerance comparisonTolerance);

  Matrix analyticValues_;
  Matrix numericalValues_;
  Matrix absoluteErrors_;
  Matrix relativeErrors_;
  Vector effectiveSteps_;
  std::vector<std::uint8_t> passed_;
  bool allPassed_;
};

/// @brief Computes a deterministic scalar finite-difference gradient.
/// @details Validation precedes the base callback. Formula selection is
/// central, forward, then backward according to finite bounds and representable
/// sample points. Samples are evaluated in base, parameter-increasing,
/// positive-then-negative order. Cost is O(n) callback evaluations and O(n)
/// owned result storage.
[[nodiscard]] Result<GradientResult>
numericalGradient(const ScalarFunction& function, ConstVectorView parameters,
                  std::span<const ParameterStep> steps);

/// @brief Computes a deterministic vector-valued finite-difference Jacobian.
/// @details The base output fixes the positive row extent and the result is
/// row-major. Callback failures, changed output extent, and non-finite values
/// publish no partial derivative. Cost is O(m*n) arithmetic and O(m*n) storage.
[[nodiscard]] Result<JacobianResult>
numericalJacobian(const VectorFunction& function, ConstVectorView parameters,
                  std::span<const ParameterStep> steps);

/// @brief Compares an analytic gradient with a numerical gradient.
/// @details Tolerance is validated before derivative data. Relative error uses
/// the maximum analytic magnitude, numerical magnitude, and smallest normal.
[[nodiscard]] Result<DerivativeCheckResult>
checkGradient(ConstVectorView analyticGradient,
              const GradientResult& numericalGradient,
              Tolerance comparisonTolerance);

/// @brief Compares an analytic Jacobian with a numerical Jacobian.
/// @details Tolerance is validated before derivative data; row-major
/// orientation and the pass-mask ordering are preserved.
[[nodiscard]] Result<DerivativeCheckResult>
checkJacobian(ConstMatrixView analyticJacobian,
              const JacobianResult& numericalJacobian,
              Tolerance comparisonTolerance);

} // namespace vss::math::differentiation

#endif // VSS_MATH_DIFFERENTIATION_FINITE_DIFFERENCE_HPP
