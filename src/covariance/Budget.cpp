#include "vss/math/covariance/Budget.hpp"

#include "vss/math/Array.hpp"
#include "vss/math/Core.hpp"
#include "vss/math/Error.hpp"
#include "vss/math/Limits.hpp"
#include "vss/math/Matrix.hpp"
#include "vss/math/MatrixView.hpp"
#include "vss/math/Result.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vss/math/covariance/Covariance.hpp>

namespace vss::math::covariance {
namespace {

template <class T> Result<T> fail(ErrorCode code, const char* message) {
  return Result<T>::failure(code, message);
}

bool product(Extent left, Extent right, Extent& result) noexcept {
  if (right != 0U && left > std::numeric_limits<Extent>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

bool add(Extent left, Extent right, Extent& result) noexcept {
  if (left > std::numeric_limits<Extent>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

class WideSum final {
public:
  void add(long double value) noexcept {
    const long double next = sum_ + value;
    if (std::abs(sum_) >= std::abs(value)) {
      correction_ += (sum_ - next) + value;
    } else {
      correction_ += (value - next) + sum_;
    }
    sum_ = next;
  }

  [[nodiscard]] long double total() const noexcept {
    return sum_ + correction_;
  }

private:
  long double sum_{};
  long double correction_{};
};

struct CrossReference final {
  const SourceCrossBlock* block;
  std::uint64_t low;
  std::uint64_t high;
};

const SourceBlock* findSource(std::span<const SourceBlock> sources,
                              std::uint64_t id) noexcept {
  for (const auto& source : sources) {
    if (source.id == id) {
      return &source;
    }
  }
  return nullptr;
}

} // namespace

Result<UncertaintyBudget>
uncertaintyContributionBudget(ConstMatrixView jacobian,
                              std::span<const SourceBlock> sources,
                              std::span<const SourceCrossBlock> cross,
                              DependencePolicy dependence, Limits limits) {
  const Index m = jacobian.rows();
  const Index d = jacobian.columns();
  if (dependence != DependencePolicy::exact_joint &&
      dependence != DependencePolicy::declared_independent_blocks) {
    return fail<UncertaintyBudget>(ErrorCode::InvalidDomain,
                                   "dependence policy is invalid");
  }
  if (![&]() {
        for (Index i = 0U; i < m; ++i) {
          for (Index j = 0U; j < d; ++j) {
            if (!std::isfinite(jacobian.at(i, j))) {
              return false;
            }
          }
        }
        return true;
      }()) {
    return fail<UncertaintyBudget>(ErrorCode::NonFiniteInput,
                                   "Jacobian contains a non-finite value");
  }

  Extent pairProduct = 0U;
  Extent m2 = 0U;
  Extent d2 = 0U;
  Extent md = 0U;
  if (sources.size() == std::numeric_limits<Extent>::max() ||
      !product(sources.size(), sources.size() + 1U, pairProduct) ||
      !product(m, m, m2) || !product(d, d, d2) || !product(m, d, md)) {
    return fail<UncertaintyBudget>(ErrorCode::SizeOverflow,
                                   "uncertainty budget size overflows");
  }
  const Extent pairCount = pairProduct / 2U;
  Extent termScalars = 0U;
  if (!product(pairCount, m2, termScalars) ||
      termScalars > std::numeric_limits<Extent>::max() / 2U) {
    return fail<UncertaintyBudget>(ErrorCode::SizeOverflow,
                                   "uncertainty budget storage overflows");
  }
  Extent eigenElements = 0U;
  if (!product(d2, 3U, eigenElements)) {
    return fail<UncertaintyBudget>(ErrorCode::SizeOverflow,
                                   "joint covariance workspace size overflows");
  }
  Extent scalarElements = 0U;
  if (m2 > std::numeric_limits<Extent>::max() / 2U ||
      !add(eigenElements, 2U * m2, scalarElements) ||
      !add(scalarElements, md, scalarElements) ||
      !add(scalarElements, 2U * termScalars, scalarElements) ||
      scalarElements > limits.elements) {
    return fail<UncertaintyBudget>(ErrorCode::ResourceLimit,
                                   "uncertainty budget exceeds element limits");
  }
  Extent matrixBytes = 0U;
  Extent coordinateBytes = 0U;
  Extent orderBytes = 0U;
  Extent crossBytes = 0U;
  Extent entryBytes = 0U;
  Extent metadataBytes = 0U;
  Extent requiredBytes = 0U;
  if (!product(scalarElements, sizeof(Scalar), matrixBytes) ||
      !product(d, sizeof(Index), coordinateBytes) ||
      !product(sources.size(), sizeof(Index), orderBytes) ||
      !product(cross.size(), sizeof(CrossReference), crossBytes) ||
      !product(pairCount, sizeof(BudgetEntry), entryBytes) ||
      entryBytes > std::numeric_limits<Extent>::max() / 2U ||
      !add(coordinateBytes, orderBytes, metadataBytes) ||
      !add(metadataBytes, crossBytes, metadataBytes) ||
      !add(metadataBytes, 2U * entryBytes, metadataBytes) ||
      !add(matrixBytes, metadataBytes, requiredBytes)) {
    return fail<UncertaintyBudget>(ErrorCode::SizeOverflow,
                                   "uncertainty budget byte count overflows");
  }
  if (requiredBytes > limits.bytes) {
    return fail<UncertaintyBudget>(ErrorCode::ResourceLimit,
                                   "uncertainty budget exceeds byte limits");
  }

  std::vector<Index> coordinateOwner(d, sources.size());
  for (Index i = 0U; i < sources.size(); ++i) {
    const auto& source = sources[i];
    if (source.coordinates.empty()) {
      return fail<UncertaintyBudget>(
          ErrorCode::InvalidDomain,
          "source coordinate blocks must be nonempty");
    }
    for (Index j = 0U; j < i; ++j) {
      if (sources[j].id == source.id) {
        return fail<UncertaintyBudget>(ErrorCode::InvalidDomain,
                                       "source identifiers must be unique");
      }
    }
    if (source.covariance.rows() != source.coordinates.size() ||
        source.covariance.columns() != source.coordinates.size()) {
      return fail<UncertaintyBudget>(ErrorCode::DimensionMismatch,
                                     "source covariance has the wrong extent");
    }
    for (const Index coordinate : source.coordinates) {
      if (coordinate >= d) {
        return fail<UncertaintyBudget>(
            ErrorCode::IndexOutOfRange,
            "source coordinate is outside the Jacobian");
      }
      if (coordinateOwner[coordinate] != sources.size()) {
        return fail<UncertaintyBudget>(ErrorCode::InvalidDomain,
                                       "source coordinate blocks overlap");
      }
      coordinateOwner[coordinate] = i;
    }
    for (Index row = 0U; row < source.covariance.rows(); ++row) {
      for (Index column = 0U; column < source.covariance.columns(); ++column) {
        if (!std::isfinite(source.covariance.at(row, column))) {
          return fail<UncertaintyBudget>(ErrorCode::NonFiniteInput,
                                         "source covariance is non-finite");
        }
      }
    }
  }
  if (std::ranges::any_of(coordinateOwner, [&](Index owner) {
        return owner == sources.size();
      })) {
    return fail<UncertaintyBudget>(
        ErrorCode::InvalidDomain,
        "source coordinates do not partition the Jacobian");
  }
  if ((d == 0U && !sources.empty()) || (d != 0U && sources.empty())) {
    return fail<UncertaintyBudget>(
        ErrorCode::InvalidDomain,
        "empty source partition disagrees with Jacobian dimension");
  }

  std::vector<CrossReference> crossReferences;
  crossReferences.reserve(cross.size());
  for (const auto& block : cross) {
    if (block.left == block.right) {
      return fail<UncertaintyBudget>(
          ErrorCode::InvalidDomain,
          "cross block cannot refer to one source twice");
    }
    const auto* left = findSource(sources, block.left);
    const auto* right = findSource(sources, block.right);
    if (left == nullptr || right == nullptr) {
      return fail<UncertaintyBudget>(ErrorCode::InvalidDomain,
                                     "cross block refers to an unknown source");
    }
    if (block.covariance.rows() != left->coordinates.size() ||
        block.covariance.columns() != right->coordinates.size()) {
      return fail<UncertaintyBudget>(
          ErrorCode::DimensionMismatch,
          "cross covariance has the wrong orientation or extent");
    }
    for (Index row = 0U; row < block.covariance.rows(); ++row) {
      for (Index column = 0U; column < block.covariance.columns(); ++column) {
        const Scalar value = block.covariance.at(row, column);
        if (!std::isfinite(value)) {
          return fail<UncertaintyBudget>(ErrorCode::NonFiniteInput,
                                         "cross covariance is non-finite");
        }
        if (dependence == DependencePolicy::declared_independent_blocks &&
            value != 0.0) {
          return fail<UncertaintyBudget>(
              ErrorCode::InvalidDomain,
              "independent sources cannot have nonzero cross covariance");
        }
      }
    }
    const auto low = std::min(block.left, block.right);
    const auto high = std::max(block.left, block.right);
    for (const auto& previous : crossReferences) {
      if (previous.low == low && previous.high == high) {
        return fail<UncertaintyBudget>(
            ErrorCode::InvalidDomain,
            "unordered source-pair blocks must be unique");
      }
    }
    crossReferences.push_back({&block, low, high});
  }

  if (dependence == DependencePolicy::exact_joint &&
      crossReferences.size() != pairCount - sources.size()) {
    return fail<UncertaintyBudget>(
        ErrorCode::InvalidDomain,
        "exact joint dependence requires every cross block");
  }

  Matrix joint(d, d, 0.0);
  for (const auto& source : sources) {
    for (Index i = 0U; i < source.coordinates.size(); ++i) {
      for (Index j = 0U; j < source.coordinates.size(); ++j) {
        joint.at(source.coordinates[i], source.coordinates[j]) =
            source.covariance.at(i, j);
      }
    }
  }
  for (const auto& reference : crossReferences) {
    const auto& block = *reference.block;
    const auto* left = findSource(sources, block.left);
    const auto* right = findSource(sources, block.right);
    for (Index i = 0U; i < left->coordinates.size(); ++i) {
      for (Index j = 0U; j < right->coordinates.size(); ++j) {
        const Scalar value = block.covariance.at(i, j);
        joint.at(left->coordinates[i], right->coordinates[j]) = value;
        joint.at(right->coordinates[j], left->coordinates[i]) = value;
      }
    }
  }
  if (d != 0U) {
    const auto jointDiagnostics = analyzeCovariance(joint.view());
    if (!jointDiagnostics.hasValue()) {
      return Result<UncertaintyBudget>::failure(
          jointDiagnostics.status().code(),
          std::string(jointDiagnostics.status().message()));
    }
    if (jointDiagnostics.value().classification() ==
        CovarianceClassification::NotPositiveSemidefinite) {
      return fail<UncertaintyBudget>(
          ErrorCode::NotPositiveSemidefinite,
          "assembled joint covariance is not positive semidefinite");
    }
  }

  std::vector<Index> order(sources.size());
  for (Index i = 0U; i < order.size(); ++i) {
    order[i] = i;
  }
  std::ranges::sort(order, [&](Index lhs, Index rhs) {
    return sources[lhs].id < sources[rhs].id;
  });
  std::vector<BudgetEntry> terms;
  terms.reserve(pairCount);
  for (Index first = 0U; first < order.size(); ++first) {
    for (Index second = first; second < order.size(); ++second) {
      const auto& left = sources[order[first]];
      const auto& right = sources[order[second]];
      Matrix contribution(m, m, 0.0);
      for (Index a = 0U; a < m; ++a) {
        for (Index b = a; b < m; ++b) {
          WideSum forward;
          for (Index i = 0U; i < left.coordinates.size(); ++i) {
            for (Index j = 0U; j < right.coordinates.size(); ++j) {
              Scalar blockValue = 0.0;
              if (left.id == right.id) {
                const Index row =
                    std::max(left.coordinates[i], left.coordinates[j]);
                const Index column =
                    std::min(left.coordinates[i], left.coordinates[j]);
                blockValue = joint.at(row, column);
              } else {
                for (const auto& reference : crossReferences) {
                  if (reference.low == left.id && reference.high == right.id) {
                    const auto& sourceCross = *reference.block;
                    blockValue = sourceCross.left == left.id
                                     ? sourceCross.covariance.at(i, j)
                                     : sourceCross.covariance.at(j, i);
                    break;
                  }
                }
              }
              forward.add(static_cast<long double>(
                              jacobian.at(a, left.coordinates[i])) *
                          blockValue * jacobian.at(b, right.coordinates[j]));
            }
          }
          long double value = forward.total();
          if (left.id != right.id) {
            WideSum reverse;
            for (Index i = 0U; i < right.coordinates.size(); ++i) {
              for (Index j = 0U; j < left.coordinates.size(); ++j) {
                Scalar blockValue = 0.0;
                for (const auto& reference : crossReferences) {
                  if (reference.low == left.id && reference.high == right.id) {
                    const auto& sourceCross = *reference.block;
                    blockValue = sourceCross.left == left.id
                                     ? sourceCross.covariance.at(j, i)
                                     : sourceCross.covariance.at(i, j);
                    break;
                  }
                }
                reverse.add(static_cast<long double>(
                                jacobian.at(a, right.coordinates[i])) *
                            blockValue * jacobian.at(b, left.coordinates[j]));
              }
            }
            value += reverse.total();
          }
          if (!std::isfinite(value) ||
              std::abs(value) > std::numeric_limits<Scalar>::max()) {
            return fail<UncertaintyBudget>(
                ErrorCode::NonFiniteResult,
                "source-pair contribution is not finite");
          }
          contribution.at(a, b) = static_cast<Scalar>(value);
          contribution.at(b, a) = static_cast<Scalar>(value);
        }
      }
      terms.push_back(BudgetEntry{.left = left.id,
                                  .right = right.id,
                                  .contribution = std::move(contribution)});
    }
  }

  Matrix total(m, m, 0.0);
  for (Index a = 0U; a < m; ++a) {
    for (Index b = a; b < m; ++b) {
      WideSum sum;
      for (const auto& term : terms) {
        sum.add(term.contribution.at(a, b));
      }
      if (!std::isfinite(sum.total()) ||
          std::abs(sum.total()) > std::numeric_limits<Scalar>::max()) {
        return fail<UncertaintyBudget>(
            ErrorCode::NonFiniteResult,
            "total uncertainty budget is not finite");
      }
      total.at(a, b) = static_cast<Scalar>(sum.total());
      total.at(b, a) = static_cast<Scalar>(sum.total());
    }
  }
  if (d != 0U) {
    auto direct = propagateCovariance(jacobian, joint.view());
    if (!direct.hasValue()) {
      return Result<UncertaintyBudget>::failure(
          direct.status().code(), std::string(direct.status().message()));
    }
    for (Index a = 0U; a < m; ++a) {
      for (Index b = 0U; b < m; ++b) {
        const Scalar expected = direct.value().at(a, b);
        const Scalar actual = total.at(a, b);
        const Scalar tolerance =
            1.0e-12 +
            (1.0e-10 * std::max(std::abs(expected), std::abs(actual)));
        if (std::abs(expected - actual) > tolerance) {
          return fail<UncertaintyBudget>(
              ErrorCode::NonFiniteResult,
              "budget terms disagree with joint propagation");
        }
      }
    }
  }
  auto ownedTerms =
      Array<BudgetEntry>::copy(std::span<const BudgetEntry>(terms), limits);
  if (!ownedTerms.hasValue()) {
    return Result<UncertaintyBudget>::failure(
        ownedTerms.status().code(), std::string(ownedTerms.status().message()));
  }
  return Result<UncertaintyBudget>::success(UncertaintyBudget{
      .covariance = std::move(total), .terms = std::move(ownedTerms).value()});
}

} // namespace vss::math::covariance
