#ifndef _SCTL_MATH_UTILS_
#define _SCTL_MATH_UTILS_

#include <cmath>
#include <ostream>

#include "sctl/common.hpp"

namespace sctl {


template <class ValueType> static inline constexpr ValueType pow_integer_exp(ValueType b, Long e) {
  return (e > 0) ? ((e & 1) ? b : ValueType(1)) * pow_integer_exp(b*b, e>>1) : ValueType(1);
}
template <class Real, class ExpType> class pow_wrapper {
  public:
    static Real pow(Real b, ExpType e) {
      return (Real)std::pow(b, e);
    }
};
template <class ValueType> class pow_wrapper<ValueType,Long> {
  public:
    static constexpr ValueType pow(ValueType b, Long e) {
      return (e > 0) ? pow_integer_exp(b, e) : 1/pow_integer_exp(b, -e);
    }
};

template <Long e, class ValueType> inline constexpr ValueType pow(ValueType b) {
  return (e > 0) ? pow_integer_exp<ValueType>(b, e) : 1/pow_integer_exp<ValueType>(b, -e);
}


template <class Real, Integer bits = sizeof(Real)*8> struct GetSigBits {
  static constexpr Integer value() {
    return ((Real)( pow<bits>((Real)0.5) + 1 ) == (Real)1 ? GetSigBits<Real, bits - 1 >::value() : bits);
  }
};

template <class Real> struct GetSigBits<Real,0> {
  static constexpr Integer value() {
    return 0;
  }
};

template <class Real> inline constexpr Integer significant_bits() {
  return GetSigBits<Real>::value();
}

template <class Real> inline constexpr Real machine_eps() {
  return pow<-GetSigBits<Real>::value()-1,Real>(2);
}





template <class Real> Real atoreal(const char* str);

template <class Real> inline constexpr Real const_pi() { return (Real)3.1415926535897932384626433832795028841L; }

template <class Real> inline constexpr Real const_e() { return (Real)2.7182818284590452353602874713526624977L; }

template <class Real> inline Real fabs(const Real a) { return (Real)std::fabs(a); }

template <class Real> inline Real round(const Real a) { return (Real)std::round(a); }

template <class Real> inline Real floor(const Real a) { return (Real)std::floor(a); }

template <class Real> inline Real sqrt(const Real a) { return (Real)std::sqrt(a); }

template <class Real> inline Real sin(const Real a) { return (Real)std::sin(a); }

template <class Real> inline Real cos(const Real a) { return (Real)std::cos(a); }

template <class Real> inline Real acos(const Real a) { return (Real)std::acos(a); }

template <class Real> inline Real exp(const Real a) { return (Real)std::exp(a); }

template <class Real> inline Real log(const Real a) { return (Real)std::log(a); }

template <class Real, class ExpType> inline Real pow(const Real b, const ExpType e);



#ifdef SCTL_QUAD_T
class QuadReal {
  typedef SCTL_QUAD_T QuadRealType;
  public:

    QuadReal() = default;
    constexpr QuadReal(const QuadReal& v) = default;
    QuadReal& operator=(const QuadReal&) = default;
    ~QuadReal() = default;

    template <class ValueType> constexpr QuadReal(ValueType v) : val((QuadRealType)v) {}
    template <class ValueType> explicit constexpr operator ValueType() const { return (ValueType)val; }


    QuadReal& operator+=(const QuadReal& x) { val += x.val; return *this; }
    QuadReal& operator-=(const QuadReal& x) { val -= x.val; return *this; }
    QuadReal& operator*=(const QuadReal& x) { val *= x.val; return *this; }
    QuadReal& operator/=(const QuadReal& x) { val /= x.val; return *this; }

    constexpr QuadReal operator+(const QuadReal& x) const { return QuadReal(val + x.val); }
    constexpr QuadReal operator-(const QuadReal& x) const { return QuadReal(val - x.val); }
    constexpr QuadReal operator*(const QuadReal& x) const { return QuadReal(val * x.val); }
    constexpr QuadReal operator/(const QuadReal& x) const { return QuadReal(val / x.val); }

    constexpr QuadReal operator-() const { return QuadReal(-val); }

    constexpr bool operator< (const QuadReal& x) const { return val <  x.val; }
    constexpr bool operator> (const QuadReal& x) const { return val >  x.val; }
    constexpr bool operator!=(const QuadReal& x) const { return val != x.val; }
    constexpr bool operator==(const QuadReal& x) const { return val == x.val; }
    constexpr bool operator<=(const QuadReal& x) const { return val <= x.val; }
    constexpr bool operator>=(const QuadReal& x) const { return val >= x.val; }


    constexpr friend QuadReal operator+(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) + b; }
    constexpr friend QuadReal operator-(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) - b; }
    constexpr friend QuadReal operator*(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) * b; }
    constexpr friend QuadReal operator/(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) / b; }

    constexpr friend bool operator< (const QuadRealType& a, const QuadReal& b) { return QuadReal(a) <  b; }
    constexpr friend bool operator> (const QuadRealType& a, const QuadReal& b) { return QuadReal(a) >  b; }
    constexpr friend bool operator!=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) != b; }
    constexpr friend bool operator==(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) == b; }
    constexpr friend bool operator<=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) <= b; }
    constexpr friend bool operator>=(const QuadRealType& a, const QuadReal& b) { return QuadReal(a) >= b; }

    friend QuadReal round(const QuadReal& x) {
      #ifdef __SIZEOF_INT128__
      return (__int128)((x+(QuadReal)0.5).val)-(x<(QuadReal)-0.5);
      #else
      return (int64_t)((x+(QuadReal)0.5).val)-(x<(QuadReal)-0.5);
      #endif
    }

    friend QuadReal floor(const QuadReal& x) {
      #ifdef __SIZEOF_INT128__
      return (__int128)(x.val)-(x<0);
      #else
      return (int64_t)(x.val)-(x<0);
      #endif
    }

  private:
    QuadRealType val;
};

inline std::ostream& operator<<(std::ostream& output, const QuadReal& x);
inline std::istream& operator>>(std::istream& inputstream, QuadReal& x);
#endif

}  // end namespace

#endif  //_SCTL_MATH_UTILS_HPP_
