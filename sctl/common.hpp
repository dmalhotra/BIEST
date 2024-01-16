#ifndef _SCTL_COMMON_HPP_
#define _SCTL_COMMON_HPP_

// fall-back to double if ValueType is not defined
#ifdef SCTL_DOUBLE
using ValueType = double;
#elif defined SCTL_FLOAT
using ValueType = float;
#else
#error "Must define either SCTL_DOUBLE or SCTL_FLOAT."
#endif // ValueType

// Profiling parameters
#ifndef SCTL_PROFILE
#define SCTL_PROFILE -1 // Granularity level
#endif

#if defined(__AVX512__) || defined(__AVX512F__)
  #define SCTL_ALIGN_BYTES 64
#elif defined(__AVX__)
  #define SCTL_ALIGN_BYTES 32
#elif defined(__SSE__) || defined(__ARM_NEON)
  #define SCTL_ALIGN_BYTES 16
#else
  #define SCTL_ALIGN_BYTES 8
#endif

// Parameters for memory manager
#ifndef SCTL_MEM_ALIGN
#define SCTL_MEM_ALIGN (64 > SCTL_ALIGN_BYTES ? 64 : SCTL_ALIGN_BYTES)
#endif
#ifndef SCTL_GLOBAL_MEM_BUFF
#define SCTL_GLOBAL_MEM_BUFF 1024LL * 0LL  // in MB
#endif

// Define NULL
#ifndef NULL
#define NULL 0
#endif

#include <cstddef>
#include <cstdint>
namespace sctl {
typedef long Integer;  // bounded numbers < 32k
typedef int64_t Long;  // problem size
}

#include <iostream>

#define SCTL_WARN(msg)                                         \
  do {                                                          \
    std::cerr << "\n\033[1;31mWarning:\033[0m " << msg << '\n'; \
  } while (0)

#define SCTL_ERROR(msg)                                      \
  do {                                                        \
    std::cerr << "\n\033[1;31mError:\033[0m " << msg << '\n'; \
    abort();                                                  \
  } while (0)

#define SCTL_ASSERT_MSG(cond, msg) \
  do {                              \
    if (!(cond)) SCTL_ERROR(msg);  \
  } while (0)

#define SCTL_ASSERT(cond)                                                                                      \
  do {                                                                                                          \
    if (!(cond)) {                                                                                              \
      fprintf(stderr, "\n%s:%d: %s: Assertion `%s' failed.\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, #cond); \
      abort();                                                                                                  \
    }                                                                                                           \
  } while (0)

#define SCTL_UNUSED(x) (void)(x)  // to ignore unused variable warning.

namespace sctl {
#ifdef SCTL_MEMDEBUG
template <class ValueType> class ConstIterator;
template <class ValueType> class Iterator;
template <class ValueType, Long DIM> class StaticArray;
#else
template <typename ValueType> using Iterator = ValueType*;
template <typename ValueType> using ConstIterator = const ValueType*;
template <typename ValueType, Long DIM> using StaticArray = ValueType[DIM];
#endif
}

#endif  //_SCTL_COMMON_HPP_
