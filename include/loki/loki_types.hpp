#pragma once

#include <array>
#include <complex>
#include <cstddef>

#include <xtensor/xtensor.hpp>

using SizeType        = std::size_t;
using FloatType       = double;
using IndexType       = std::ptrdiff_t;
using ComplexType     = std::complex<float>;
using ParamLimitType  = std::array<FloatType, 2>;
using SuggestionTypeF = xt::xtensor<float, 3>;
using SuggestionTypeD = xt::xtensor<double, 3>;