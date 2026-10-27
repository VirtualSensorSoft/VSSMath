#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vss/math/Core.hpp>
#include <vss/math/Error.hpp>
#include <vss/math/Matrix.hpp>
#include <vss/math/Vector.hpp>

static_assert(sizeof(vss::math::Scalar) == 8U);
static_assert(std::numeric_limits<vss::math::Scalar>::is_iec559);
static_assert(std::numeric_limits<vss::math::Scalar>::radix == 2);
static_assert(std::is_same_v<vss::math::Index, std::size_t>);
static_assert(std::is_same_v<vss::math::Extent, std::size_t>);
static_assert(std::is_nothrow_destructible_v<vss::math::Status>);
static_assert(std::is_nothrow_destructible_v<vss::math::Vector>);
static_assert(std::is_nothrow_destructible_v<vss::math::Matrix>);
static_assert(std::is_nothrow_move_constructible_v<vss::math::Vector>);
static_assert(std::is_nothrow_move_constructible_v<vss::math::Matrix>);

static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::None) == 0U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::InvalidExtent) ==
              1U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::SizeOverflow) ==
              2U);
static_assert(
    static_cast<std::uint8_t>(vss::math::ErrorCode::IndexOutOfRange) == 3U);
static_assert(
    static_cast<std::uint8_t>(vss::math::ErrorCode::DimensionMismatch) == 4U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::NonFiniteInput) ==
              5U);
static_assert(
    static_cast<std::uint8_t>(vss::math::ErrorCode::NonFiniteResult) == 6U);
static_assert(
    static_cast<std::uint8_t>(vss::math::ErrorCode::InvalidTolerance) == 7U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::Singular) == 8U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::RankDeficient) ==
              9U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::IllConditioned) ==
              10U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::NotSymmetric) ==
              11U);
static_assert(static_cast<std::uint8_t>(
                  vss::math::ErrorCode::NotPositiveDefinite) == 12U);
static_assert(static_cast<std::uint8_t>(
                  vss::math::ErrorCode::NotPositiveSemidefinite) == 13U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::InvalidDomain) ==
              14U);
static_assert(static_cast<std::uint8_t>(
                  vss::math::ErrorCode::InsufficientSamples) == 15U);
static_assert(
    static_cast<std::uint8_t>(vss::math::ErrorCode::CallbackFailure) == 16U);
static_assert(static_cast<std::uint8_t>(vss::math::ErrorCode::NoConvergence) ==
              17U);

int main() { return 0; }
