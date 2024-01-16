#ifndef _SCTL_MAT_HPP_
#define _SCTL_MAT_HPP_

namespace biest {

template <class Real, sctl::Integer N1, sctl::Integer N2, bool own_data = true> class Mat {
 public:
  Mat() {
    static_assert(own_data,"A data pointer must be provided when own_data=false.");
    iter_ = buff;
  }

  Mat(sctl::Iterator<Real> src_iter) { Init(src_iter); }
  Mat(sctl::ConstIterator<Real> src_iter) { ConstInit(src_iter); }

  Mat(const Mat &M) { ConstInit(M.begin()); }
  template <bool own_data_> Mat(Mat<Real, N1, N2, own_data_> &M) { Init(M.begin()); }
  template <bool own_data_> Mat(const Mat<Real, N1, N2, own_data_> &M) { ConstInit(M.begin()); }

  Mat &operator=(const Mat &M) {
    auto src_iter = M.begin();
    for (sctl::Integer i = 0; i < N1 * N2; i++) this->begin()[i] = src_iter[i];
    return *this;
  }

  template <bool own_data_> Mat &operator=(const Mat<Real, N1, N2, own_data_> &M) {
    auto src_iter = M.begin();
    for (sctl::Integer i = 0; i < N1 * N2; i++) this->begin()[i] = src_iter[i];
    return *this;
  }

  sctl::Integer Dim0() const { return N1; }
  sctl::Integer Dim1() const { return N2; }

  sctl::Iterator<Real> begin() { return iter_; }
  sctl::ConstIterator<Real> begin() const { return iter_; }

  Mat<Real, N1, N2> operator*(const Real &s) const {
    Mat<Real, N1, N2> M0;
    const auto &M1 = *this;

    for (sctl::Integer i1 = 0; i1 < N1; i1++) {
      for (sctl::Integer i2 = 0; i2 < N2; i2++) {
        M0[i1][i2] = M1[i1][i2] * s;
      }
    }
    return M0;
  }

  template <sctl::Integer N3, bool own_data_> Mat<Real, N1, N3> operator*(const Mat<Real, N2, N3, own_data_> &M2) const {
    Mat<Real, N1, N3> M0;
    const auto &M1 = *this;
    for (sctl::Integer i1 = 0; i1 < N1; i1++) {
      for (sctl::Integer i3 = 0; i3 < N3; i3++) {
        Real v = 0;
        for (sctl::Integer i2 = 0; i2 < N2; i2++) {
          v += M1[i1][i2] * M2[i2][i3];
        }
        M0[i1][i3] = v;
      }
    }
    return M0;
  }

  Mat<Real, N1, N2> operator+(const Mat<Real, N1, N2> &M2) const {
    Mat<Real, N1, N2> M0;
    const auto &M1 = *this;

    for (sctl::Integer i1 = 0; i1 < N1; i1++) {
      for (sctl::Integer i2 = 0; i2 < N2; i2++) {
        M0[i1][i2] = M1[i1][i2] + M2[i1][i2];
      }
    }
    return M0;
  }

  Mat<Real, N1, N2> operator-(const Mat<Real, N1, N2> &M2) const {
    Mat<Real, N1, N2> M0;
    const auto &M1 = *this;

    for (sctl::Integer i1 = 0; i1 < N1; i1++) {
      for (sctl::Integer i2 = 0; i2 < N2; i2++) {
        M0[i1][i2] = M1[i1][i2] - M2[i1][i2];
      }
    }
    return M0;
  }

  sctl::Iterator<Real> operator[](sctl::Integer i) {
    #ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(i < N1);
    #endif
    return iter_ + i * N2;
  }

  sctl::ConstIterator<Real> operator[](sctl::Integer i) const {
    #ifdef SCTL_MEMDEBUG
    SCTL_ASSERT(i < N1);
    #endif
    return iter_ + i * N2;
  }

  Mat<Real, N2, N1> Transpose() const {
    Mat<Real, N2, N1> M0;
    const auto &M1 = *this;
    for (sctl::Integer i1 = 0; i1 < N1; i1++) {
      for (sctl::Integer i2 = 0; i2 < N2; i2++) {
        M0[i2][i1] = M1[i1][i2];
      }
    }
    return M0;
  }

  Real Trace() const {
    Real sum = 0;
    const auto &M1 = *this;
    static_assert(N1 == N2,"Cannot compute trace of non-square matrix.");
    for (sctl::Integer i = 0; i < N1; i++) sum += M1[i][i];
    return sum;
  }

  bool OwnData() const { return own_data; }

 private:

  void ConstInit(sctl::ConstIterator<Real> src_iter) {
    iter_ = buff;
    static_assert(own_data,"Data must be modifiable when own_data=false.");
    for (sctl::Integer i = 0; i < N1 * N2; i++) this->begin()[i] = src_iter[i];
  }

  void Init(sctl::Iterator<Real> src_iter) {
    if (own_data) {
      iter_ = buff;
      for (sctl::Integer i = 0; i < N1 * N2; i++) this->begin()[i] = src_iter[i];
    } else {
      iter_ = src_iter;
      #ifdef SCTL_MEMDEBUG
      if (N1 && N2) {
        SCTL_UNUSED(src_iter[0]);
        SCTL_UNUSED(src_iter[N1 * N2 - 1]);
      }
      #endif
    }
  }

  sctl::Iterator<Real> iter_;
  sctl::StaticArray<Real, own_data ? N1 * N2 : 0> buff;
};

template <class Real, sctl::Integer N1, sctl::Integer N2, bool own_data> Mat<Real, N1, N2> operator*(Real s, const Mat<Real, N1, N2, own_data> &M1) {
  Mat<Real, N1, N2> M0;
  for (sctl::Integer i1 = 0; i1 < N1; i1++) {
    for (sctl::Integer i2 = 0; i2 < N2; i2++) {
      M0[i1][i2] = M1[i1][i2] * s;
    }
  }
  return M0;
}

template <class Real, sctl::Integer N1, sctl::Integer N2, bool own_data> std::ostream &operator<<(std::ostream &output, const Mat<Real, N1, N2, own_data> &M) {
  std::ios::fmtflags f(std::cout.flags());
  output << std::fixed << std::setprecision(4) << std::setiosflags(std::ios::left);
  for (sctl::Long i = 0; i < N1; i++) {
    for (sctl::Long j = 0; j < N2; j++) {
      float f = ((float)M[i][j]);
      if (sctl::fabs<Real>(f) < 1e-25) f = 0;
      output << std::setw(10) << ((double)f) << ' ';
    }
    output << ";\n";
  }
  std::cout.flags(f);
  return output;
}

}

#endif  //_SCTL_MAT_HPP_
