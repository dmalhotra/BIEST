// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _BIEST_TEST_UTILS_HPP_
#define _BIEST_TEST_UTILS_HPP_

#include <biest.hpp>
#include <cstdio>
#include <string>

namespace biest_test {

  using Real = double;
  constexpr sctl::Integer COORD_DIM = 3;

  /**
   * Records the results of the checks of one test program.
   */
  class Results {
    public:

    /**
     * Prints the check and records a failure if err > tol or err is NaN.
     */
    void Check(const char* name, const Real err, const Real tol) {
      const bool pass = (err <= tol);
      std::printf("%s  %-72s %.3e  (tol %.0e)\n", (pass ? "PASS" : "FAIL"), name, (double)err, (double)tol);
      if (!pass) failures++;
    }

    /**
     * Prints the number of failed checks and returns the exit code of the test program.
     */
    int ExitCode() const {
      std::printf("%d check(s) failed\n", failures);
      return (failures ? 1 : 0);
    }

    private:

    int failures = 0;
  };

  inline Real MaxAbs(const sctl::Vector<Real>& A) {
    Real max_val = 0;
    for (const auto& a : A) max_val = std::max<Real>(max_val, sctl::fabs<Real>(a));
    return max_val;
  }

  /**
   * Returns max|A - B| / max|B|.
   */
  inline Real MaxRelDiff(const sctl::Vector<Real>& A, const sctl::Vector<Real>& B) {
    SCTL_ASSERT(A.Dim() == B.Dim());
    Real max_err = 0;
    for (sctl::Long i = 0; i < A.Dim(); i++) max_err = std::max<Real>(max_err, sctl::fabs<Real>(A[i] - B[i]));
    return max_err / MaxAbs(B);
  }

}

#endif // _BIEST_TEST_UTILS_HPP_
