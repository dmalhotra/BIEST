// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <sctl.hpp>
#include <omp.h>
#include <functional>
#include <map>
#include <memory>
#include <tuple>

namespace biest {

// The 2D transforms of the periodic grids are done as 1D transforms of their rows and of their columns, a few at a
// time by each thread, from loops of all threads: a 2D transform by FFTW on many threads splits its work into teams
// of fewer threads, and libgomp ends the threads that a team smaller than the one before does not use, to create them
// again for the next larger team.
namespace detail_surfop {

  /**
   * Returns the FFT of howmany 1D transforms of n points of the given type, on one thread. It is set up on the first
   * call and kept until the program ends, for all callers and all threads: the execution of a plan is thread safe.
   */
  template <class Real> const sctl::FFT<Real>& CachedFFT(const sctl::FFT_Type type, const sctl::Long howmany, const sctl::Long n) {
    using Key = std::tuple<sctl::Integer, sctl::Long, sctl::Long>;
    static auto* plans = new std::map<Key, std::unique_ptr<sctl::FFT<Real>>>(); // never freed: a plan may be in use until the program ends
    const sctl::FFT<Real>* fft;
    #pragma omp critical(BIEST_CachedFFT)
    {
      auto& plan = (*plans)[Key((sctl::Integer)type, howmany, n)];
      if (!plan) {
        plan.reset(new sctl::FFT<Real>());
        sctl::StaticArray<sctl::Long,1> dim{n};
        plan->Setup(type, howmany, sctl::Vector<sctl::Long>(1, dim, false), 1);
      }
      fft = plan.get();
    }
    return *fft;
  }

  constexpr sctl::Long FFT_ROWS = 8; // rows transformed together, by one thread
  constexpr sctl::Long FFT_COLS = 4; // columns transformed together, by one thread: 4 complex values fill 64 bytes
  constexpr sctl::Long LOOP_POINTS = 8192; // the fewest points in all for which the transforms use all threads

  /**
   * Calls body(b0, b1) for the blocks b0 to b1 of the Nb, divided evenly among the threads of a loop of 1D transforms
   * of n points in all: all threads if n is at least LOOP_POINTS, else one, and one inside a parallel region; never a
   * team of other sizes (see above). With one thread, on the calling thread without a parallel region, whose team
   * libgomp would allocate and free.
   */
  template <class Body> void ForBlocks(const sctl::Long Nb, const sctl::Long n, const Body& body) {
    const sctl::Integer nt = (omp_in_parallel() || n < LOOP_POINTS ? 1 : omp_get_max_threads());
    if (nt == 1) {
      body((sctl::Long)0, Nb);
      return;
    }
    #pragma omp parallel num_threads(nt)
    {
      const sctl::Integer tid = omp_get_thread_num(), nt_ = omp_get_num_threads();
      body(Nb * tid / nt_, Nb * (tid + 1) / nt_);
    }
  }

  /**
   * Returns Q rounded up to a multiple of FFT_COLS: the leading dimension, in complex values, of the arrays whose
   * columns are transformed.
   */
  inline sctl::Long PaddedCols(const sctl::Long Q) {
    return (Q + FFT_COLS - 1) / FFT_COLS * FFT_COLS;
  }

  /**
   * Fourier transform of R rows of n real values, FFT_ROWS rows at a time by each thread: in(r) returns an iterator to
   * the values of row r, and out(r, c) gets the first Qm unnormalized Fourier coefficients of row r, as an iterator c
   * to complex values.
   */
  template <class Real, class In, class Out> void RowsR2C(const sctl::Long R, const sctl::Long n, const sctl::Long Qm, const In& in, const Out& out) {
    const sctl::Long Q = n / 2 + 1, Nb = (R + FFT_ROWS - 1) / FFT_ROWS;
    SCTL_ASSERT(Qm <= Q);
    const sctl::FFT<Real>& fft = CachedFFT<Real>(sctl::FFT_Type::R2C, FFT_ROWS, n);
    ForBlocks(Nb, R * n, [n,Q,R,&in,&out,&fft](const sctl::Long b0, const sctl::Long b1) {
      sctl::ScratchBuf<Real> X_buf(FFT_ROWS * n), C_buf(FFT_ROWS * Q * 2);
      sctl::Vector<Real> X(X_buf), C(C_buf);
      for (sctl::Long b = b0; b < b1; b++) {
        const sctl::Long r0 = b * FFT_ROWS, nr = std::min(FFT_ROWS, R - r0);
        for (sctl::Long i = 0; i < FFT_ROWS; i++) {
          if (i < nr) {
            const auto x = in(r0 + i);
            for (sctl::Long j = 0; j < n; j++) X[i * n + j] = x[j];
          } else {
            for (sctl::Long j = 0; j < n; j++) X[i * n + j] = 0;
          }
        }
        fft.Execute(X, C, false);
        for (sctl::Long i = 0; i < nr; i++) out(r0 + i, (sctl::ConstIterator<Real>)C.begin() + i * Q * 2);
      }
    });
  }

  /**
   * Inverse Fourier transform of R rows of n real values, FFT_ROWS rows at a time by each thread: in(r, c) sets the
   * first Qm unnormalized Fourier coefficients of row r, as an iterator c to complex values, the others being zero,
   * and out(r, x) gets the values of row r, as an iterator x.
   */
  template <class Real, class In, class Out> void RowsC2R(const sctl::Long R, const sctl::Long n, const sctl::Long Qm, const In& in, const Out& out) {
    const sctl::Long Q = n / 2 + 1, Nb = (R + FFT_ROWS - 1) / FFT_ROWS;
    SCTL_ASSERT(Qm <= Q);
    const sctl::FFT<Real>& fft = CachedFFT<Real>(sctl::FFT_Type::C2R, FFT_ROWS, n);
    ForBlocks(Nb, R * n, [Q,n,R,&in,Qm,&out,&fft](const sctl::Long b0, const sctl::Long b1) {
      sctl::ScratchBuf<Real> C_buf(FFT_ROWS * Q * 2), X_buf(FFT_ROWS * n);
      sctl::Vector<Real> C(C_buf), X(X_buf);
      for (sctl::Long b = b0; b < b1; b++) {
        const sctl::Long r0 = b * FFT_ROWS, nr = std::min(FFT_ROWS, R - r0);
        for (sctl::Long i = 0; i < FFT_ROWS; i++) {
          const sctl::Iterator<Real> c = C.begin() + i * Q * 2;
          if (i < nr) in(r0 + i, c);
          else for (sctl::Long q = 0; q < 2 * Qm; q++) c[q] = 0;
          for (sctl::Long q = 2 * Qm; q < 2 * Q; q++) c[q] = 0;
        }
        fft.Execute(C, X, false);
        for (sctl::Long i = 0; i < nr; i++) out(r0 + i, (sctl::ConstIterator<Real>)X.begin() + i * n);
      }
    });
  }

  /**
   * Transform along the columns of dof blocks of rows, FFT_COLS columns at a time by each thread. Of column q < Qm of
   * block k, the Nt0 complex values A[(k*Nt0 + t)*ld + q] are transformed, op(c1, c0, q) sets from these Fourier
   * coefficients c0 the Nt1 coefficients c1 (iterators to complex values), and their inverse transform is written to
   * B[(k*Nt1 + t)*ld + q]; the columns from Qm to ld are set to zero. The leading dimension ld is PaddedCols(Qm), and B
   * may be A if Nt1 = Nt0.
   */
  template <class Real, class Op> void Columns(const sctl::Iterator<Real> B, const sctl::Long Nt1, const sctl::ConstIterator<Real> A, const sctl::Long Nt0, const sctl::Long dof, const sctl::Long ld, const sctl::Long Qm, const Op& op) {
    constexpr sctl::Long CB = FFT_COLS;
    SCTL_ASSERT(ld == PaddedCols(Qm));
    const sctl::Long Nb = ld / CB; // blocks of columns of each block of rows
    const sctl::FFT<Real>& fwd = CachedFFT<Real>(sctl::FFT_Type::C2C, CB, Nt0);
    const sctl::FFT<Real>& inv = CachedFFT<Real>(sctl::FFT_Type::C2C_INV, CB, Nt1);
    ForBlocks(dof * Nb, dof * Qm * (Nt0 + Nt1), [Nt0,Nt1,Nb,A,ld,Qm,&op,B,&fwd,&inv](const sctl::Long task0, const sctl::Long task1) {
      sctl::ScratchBuf<Real> a_buf(CB * Nt0 * 2), ahat_buf(CB * Nt0 * 2), bhat_buf(CB * Nt1 * 2), b_buf(CB * Nt1 * 2);
      sctl::Vector<Real> a(a_buf), ahat(ahat_buf), bhat(bhat_buf), b(b_buf);
      for (sctl::Long task = task0; task < task1; task++) {
        const sctl::Long k = task / Nb, q0 = (task % Nb) * CB;
        for (sctl::Long t = 0; t < Nt0; t++) {
          const sctl::ConstIterator<Real> src = A + ((k * Nt0 + t) * ld + q0) * 2;
          for (sctl::Long j = 0; j < CB; j++) {
            a[(j * Nt0 + t) * 2 + 0] = src[j * 2 + 0];
            a[(j * Nt0 + t) * 2 + 1] = src[j * 2 + 1];
          }
        }
        fwd.Execute(a, ahat, false);
        for (sctl::Long j = 0; j < CB; j++) {
          const sctl::Iterator<Real> c1 = bhat.begin() + j * Nt1 * 2;
          if (q0 + j < Qm) op(c1, (sctl::ConstIterator<Real>)ahat.begin() + j * Nt0 * 2, q0 + j);
          else for (sctl::Long t = 0; t < 2 * Nt1; t++) c1[t] = 0;
        }
        inv.Execute(bhat, b, false);
        for (sctl::Long t = 0; t < Nt1; t++) {
          const sctl::Iterator<Real> dst = B + ((k * Nt1 + t) * ld + q0) * 2;
          for (sctl::Long j = 0; j < CB; j++) {
            dst[j * 2 + 0] = b[(j * Nt1 + t) * 2 + 0];
            dst[j * 2 + 1] = b[(j * Nt1 + t) * 2 + 1];
          }
        }
      }
    });
  }

  /**
   * Multiply the normalized 2D Fourier coefficients of dof fields on a periodic Nt x Np grid: Y is the inverse
   * transform of the coefficients of X, each times the complex factor that factor(f, t, q) sets f to, for the toroidal
   * frequency t (from -Nt/2 to Nt/2) and the poloidal frequency q (from 0 to Np/2). Y may be X.
   */
  template <class Real, class Factor> void FourierMultiply(const sctl::Iterator<Real> Y, const sctl::ConstIterator<Real> X, const sctl::Long dof, const sctl::Long Nt, const sctl::Long Np, const Factor& factor) {
    const sctl::Long Q = Np / 2 + 1, ld = PaddedCols(Q);
    sctl::ScratchBuf<Real> A_buf(dof * Nt * ld * 2); // the Fourier coefficients along the rows
    const sctl::Iterator<Real> A = A_buf.begin();
    RowsR2C<Real>(dof * Nt, Np, Q, [X,Np](const sctl::Long r) { return X + r * Np; }, [A,ld,Q](const sctl::Long r, const sctl::ConstIterator<Real> c) {
      for (sctl::Long q = 0; q < 2 * Q; q++) A[r * ld * 2 + q] = c[q];
      for (sctl::Long q = 2 * Q; q < 2 * ld; q++) A[r * ld * 2 + q] = 0;
    });
    const Real scale = 1 / (Real)(Nt * Np); // FFT is unnormalized
    Columns<Real>(A, Nt, A, Nt, dof, ld, Q, [&factor,Nt,scale](const sctl::Iterator<Real> c1, const sctl::ConstIterator<Real> c0, const sctl::Long q) {
      for (sctl::Long t = 0; t < Nt; t++) {
        Real f[2];
        factor(f, t - (t > Nt / 2 ? Nt : 0), q);
        const Real real = c0[t * 2 + 0] * scale, imag = c0[t * 2 + 1] * scale;
        c1[t * 2 + 0] = real * f[0] - imag * f[1];
        c1[t * 2 + 1] = real * f[1] + imag * f[0];
      }
    });
    RowsC2R<Real>(dof * Nt, Np, Q, [A,ld,Q](const sctl::Long r, const sctl::Iterator<Real> c) {
      for (sctl::Long q = 0; q < 2 * Q; q++) c[q] = A[r * ld * 2 + q];
    }, [Y,Np](const sctl::Long r, const sctl::ConstIterator<Real> x) {
      for (sctl::Long j = 0; j < Np; j++) Y[r * Np + j] = x[j];
    });
  }

}

template <class Real> SurfaceOp<Real>::SurfaceOp(const sctl::Comm& comm, sctl::Long Nt, sctl::Long Np, sctl::Integer NFP, sctl::Integer tor_orient) {
  Init(comm, Nt, Np, NFP, tor_orient);
}

template <class Real> SurfaceOp<Real>::SurfaceOp(const SurfaceOp& Op) {
  Init(Op.comm_, Op.Nt_, Op.Np_, Op.NFP_, Op.tor_orient_);
}

template <class Real> SurfaceOp<Real>& SurfaceOp<Real>::operator=(const SurfaceOp& Op) {
  Init(Op.comm_, Op.Nt_, Op.Np_, Op.NFP_, Op.tor_orient_);
  return *this;
}

template <class Real> void SurfaceOp<Real>::Upsample(const sctl::Vector<Real>& X0_, sctl::Long Nt0, sctl::Long Np0, sctl::Vector<Real>& X1_, sctl::Long Nt1, sctl::Long Np1) {
  assert(X0_.Dim() % (Nt0 * Np0) == 0);
  const sctl::Long dof = X0_.Dim() / (Nt0 * Np0);
  if (Nt1 == Nt0 && Np1 == Np0) { // the same grid: the interpolant at the grid points is X0
    if (X1_.Dim() != X0_.Dim()) X1_.ReInit(X0_.Dim());
    if (X0_.Dim() && X1_.begin() != X0_.begin()) sctl::omp_par::memcpy(X1_.begin(), X0_.begin(), X0_.Dim());
    return;
  }

  { // the rows, then the columns of each component, then the rows; A and B: the Fourier coefficients along the rows,
    // before and after the transforms of the columns, the first Qm of each row
    const sctl::Long Q0 = Np0 / 2 + 1, Q1 = Np1 / 2 + 1, Qm = std::min(Q0, Q1), ld = detail_surfop::PaddedCols(Qm);
    sctl::ScratchBuf<Real> A_buf(dof * Nt0 * ld * 2), B_buf(dof * Nt1 * ld * 2);
    const sctl::Iterator<Real> A = A_buf.begin(), B = B_buf.begin();
    const sctl::ConstIterator<Real> X0 = X0_.begin();
    detail_surfop::RowsR2C<Real>(dof * Nt0, Np0, Qm, [X0,Np0](const sctl::Long r) { return X0 + r * Np0; }, [A,ld,Qm](const sctl::Long r, const sctl::ConstIterator<Real> c) {
      for (sctl::Long q = 0; q < 2 * Qm; q++) A[r * ld * 2 + q] = c[q];
      for (sctl::Long q = 2 * Qm; q < 2 * ld; q++) A[r * ld * 2 + q] = 0;
    });

    const Real scale = 1 / (Real)(Nt0 * Np0); // FFT is unnormalized
    const sctl::Long Ntt = std::min(Nt0, Nt1);
    detail_surfop::Columns<Real>(B, Nt1, A, Nt0, dof, ld, Qm, [Np0,Q0,Q1,Qm,Np1,Nt1,Ntt,Nt0,scale](const sctl::Iterator<Real> c1, const sctl::ConstIterator<Real> c0, const sctl::Long q) {
      Real scale__ = 1;
      if (Np0%2==0 && Q0 < Q1 && q == Qm-1) scale__ = 0.5;
      if (Np1%2==0 && Q1 < Q0 && q == Qm-1) scale__ = 2.0;
      for (sctl::Long t = 0; t < 2 * Nt1; t++) c1[t] = 0;
      for (sctl::Long t = 0; t <= Ntt / 2; t++) {
        Real scale_ = 1;
        if (Nt0%2==0 && Nt0 < Nt1 && t == Ntt/2) scale_ = 0.5;
        if (Nt1%2==0 && Nt1 < Nt0 && t == Ntt/2) scale_ = 2.0;
        c1[t * 2 + 0] = scale * scale_ * scale__ * c0[t * 2 + 0];
        c1[t * 2 + 1] = scale * scale_ * scale__ * c0[t * 2 + 1];
      }
      for (sctl::Long t = 0; t < Ntt / 2; t++) {
        Real scale_ = 1;
        if (Nt0%2==0 && Nt0 < Nt1 && t == Ntt/2-1) scale_ = 0.5;
        if (Nt1%2==0 && Nt1 < Nt0 && t == Ntt/2-1) scale_ = 2.0;
        c1[(Nt1 - t - 1) * 2 + 0] = scale * scale_ * scale__ * c0[(Nt0 - t - 1) * 2 + 0];
        c1[(Nt1 - t - 1) * 2 + 1] = scale * scale_ * scale__ * c0[(Nt0 - t - 1) * 2 + 1];
      }
    });

    if (X1_.Dim() != dof * Nt1 * Np1) X1_.ReInit(dof * Nt1 * Np1);
    const sctl::Iterator<Real> X1 = X1_.begin();
    detail_surfop::RowsC2R<Real>(dof * Nt1, Np1, Qm, [B,ld,Qm](const sctl::Long r, const sctl::Iterator<Real> c) {
      for (sctl::Long q = 0; q < 2 * Qm; q++) c[q] = B[r * ld * 2 + q];
    }, [X1,Np1](const sctl::Long r, const sctl::ConstIterator<Real> x) {
      for (sctl::Long j = 0; j < Np1; j++) X1[r * Np1 + j] = x[j];
    });
  }

  { // Floating-point correction
    sctl::Long Ut = Nt1 / Nt0;
    sctl::Long Up = Np1 / Np0;
    if (Nt1 == Nt0 * Ut && Np1 == Np0 * Up) { // rows t of field k, without a parallel region of one thread (see sctl::omp_par::one_thread)
      sctl::omp_par::parallel_for(dof * Nt0, [Nt0,Np0,&X1_,Nt1,Ut,Np1,Up,&X0_](const sctl::Long kt) {
        const sctl::Long k = kt / Nt0, t = kt % Nt0;
        for (sctl::Long p = 0; p < Np0; p++) {
          X1_[(k * Nt1 + t * Ut) * Np1 + p * Up] = X0_[(k * Nt0 + t) * Np0 + p];
        }
      });
    }
  }
}

template <class Real> void SurfaceOp<Real>::Grad2D(sctl::Vector<Real>& dX, const sctl::Vector<Real>& X) const {
  const sctl::Long dof = X.Dim() / (Nt_ * Np_);
  assert(X.Dim() == dof * Nt_ * Np_);
  if (dX.Dim() != dof * 2 * Nt_ * Np_) dX.ReInit(dof * 2 * Nt_ * Np_);
  if (!dof) return;

  // The derivative along the rows from their transforms; along the columns, from the transforms of the columns of
  // the transforms of the rows. A: the Fourier coefficients along the rows
  const sctl::Long Nt = Nt_, Np = Np_, N = Nt * Np, Q = Np / 2 + 1, ld = detail_surfop::PaddedCols(Q);
  sctl::ScratchBuf<Real> A_buf(dof * Nt * ld * 2);
  const sctl::Iterator<Real> A = A_buf.begin(), dX_ = dX.begin();
  const sctl::ConstIterator<Real> X_ = X.begin();
  detail_surfop::RowsR2C<Real>(dof * Nt, Np, Q, [X_,Np](const sctl::Long r) { return X_ + r * Np; }, [A,ld,Q](const sctl::Long r, const sctl::ConstIterator<Real> c) {
    for (sctl::Long q = 0; q < 2 * Q; q++) A[r * ld * 2 + q] = c[q];
    for (sctl::Long q = 2 * Q; q < 2 * ld; q++) A[r * ld * 2 + q] = 0;
  });

  const Real sp = (2 * sctl::const_pi<Real>()) / (Real)Np; // FFT is unnormalized
  detail_surfop::RowsC2R<Real>(dof * Nt, Np, Q, [A,ld,Q,sp](const sctl::Long r, const sctl::Iterator<Real> c) {
    for (sctl::Long p = 0; p < Q; p++) {
      const Real real = A[(r * ld + p) * 2 + 0] * sp, imag = A[(r * ld + p) * 2 + 1] * sp;
      c[p * 2 + 0] = imag * p;
      c[p * 2 + 1] = -real * p;
    }
  }, [dX_,Nt,Np,N](const sctl::Long r, const sctl::ConstIterator<Real> x) {
    const sctl::Long k = r / Nt, t = r % Nt;
    for (sctl::Long j = 0; j < Np; j++) dX_[(k * 2 + 1) * N + t * Np + j] = x[j];
  });

  const Real st = (2 * sctl::const_pi<Real>()) * NFP_ / (Real)(Nt * Np); // FFT is unnormalized; t in [0,1/NFP) on one of NFP field periods
  detail_surfop::Columns<Real>(A, Nt, A, Nt, dof, ld, Q, [Nt,st](const sctl::Iterator<Real> c1, const sctl::ConstIterator<Real> c0, const sctl::Long q) {
    for (sctl::Long t = 0; t < Nt; t++) {
      const Real tt = (Real)(t - (t > Nt / 2 ? Nt : 0));
      const Real real = c0[t * 2 + 0] * st, imag = c0[t * 2 + 1] * st;
      c1[t * 2 + 0] = imag * tt;
      c1[t * 2 + 1] = -real * tt;
    }
  });
  detail_surfop::RowsC2R<Real>(dof * Nt, Np, Q, [A,ld,Q](const sctl::Long r, const sctl::Iterator<Real> c) {
    for (sctl::Long q = 0; q < 2 * Q; q++) c[q] = A[r * ld * 2 + q];
  }, [dX_,Nt,Np,N](const sctl::Long r, const sctl::ConstIterator<Real> x) {
    const sctl::Long k = r / Nt, t = r % Nt;
    for (sctl::Long j = 0; j < Np; j++) dX_[(k * 2 + 0) * N + t * Np + j] = x[j];
  });
}

template <class Real> Real SurfaceOp<Real>::SurfNormalAreaElem(sctl::Vector<Real>* normal, sctl::Vector<Real>* area_elem, const sctl::Vector<Real>& dX, const sctl::Vector<Real>* X) const {
  if (normal != nullptr && normal->Dim() != COORD_DIM * Nt_ * Np_) {
    normal->ReInit(COORD_DIM * Nt_ * Np_);
  }
  if (area_elem != nullptr && area_elem->Dim() != Nt_ * Np_) {
    area_elem->ReInit(Nt_ * Np_);
  }

  sctl::Long N = Nt_ * Np_;
  Real scal = 1 / (Real)(NFP_ * N); // the weights of the full torus, on one of NFP field periods
  if (normal != nullptr && area_elem != nullptr) {
    sctl::omp_par::parallel_for(N, [&dX,N,area_elem,scal,normal](const sctl::Long i) {
      sctl::StaticArray<Real, COORD_DIM> xt{dX[0*N+i], dX[2*N+i], dX[4*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xp{dX[1*N+i], dX[3*N+i], dX[5*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xn;
      (*area_elem)[i] = compute_area_elem(xn, xt, xp) * scal;
      (*normal)[0*N+i] = xn[0];
      (*normal)[1*N+i] = xn[1];
      (*normal)[2*N+i] = xn[2];
    });
  } else if (normal != nullptr) {
    sctl::omp_par::parallel_for(N, [&dX,N,normal](const sctl::Long i) {
      sctl::StaticArray<Real, COORD_DIM> xt{dX[0*N+i], dX[2*N+i], dX[4*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xp{dX[1*N+i], dX[3*N+i], dX[5*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xn;
      compute_area_elem(xn, xt, xp);
      (*normal)[0*N+i] = xn[0];
      (*normal)[1*N+i] = xn[1];
      (*normal)[2*N+i] = xn[2];
    });
  } else if (area_elem != nullptr) {
    sctl::omp_par::parallel_for(N, [&dX,N,area_elem,scal](const sctl::Long i) {
      sctl::StaticArray<Real, COORD_DIM> xt{dX[0*N+i], dX[2*N+i], dX[4*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xp{dX[1*N+i], dX[3*N+i], dX[5*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xn;
      (*area_elem)[i] = compute_area_elem(xn, xt, xp) * scal;
    });
  }

  Real normal_orient = 0;
  if (X != nullptr && normal != nullptr) { // Set Orientation: at the point farthest from the z-axis the outward normal points away from it, and one of NFP field periods has such a point
    const auto R2 = [X,N](const sctl::Long j) { return (*X)[0*N+j] * (*X)[0*N+j] + (*X)[1*N+j] * (*X)[1*N+j]; };
    sctl::Long idx = 0;
    for (sctl::Long j = 0; j < N; j++) {
      if (R2(idx) < R2(j)) idx = j;
    }
    if ((*normal)[0*N+idx] * (*X)[0*N+idx] + (*normal)[1*N+idx] * (*X)[1*N+idx] < 0) {
      normal_orient = -1;
      (*normal) *= normal_orient;
    } else {
      normal_orient = 1;
    }
  }
  return normal_orient;
}

template <class Real> void SurfaceOp<Real>::SurfCurl(sctl::Vector<Real>& CurlF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& normal, const sctl::Vector<Real>& F) const {
  // -n . curl(F) = div(n x F), from the components (n x F) . dX/du_j of n x F along the tangent vectors (see SurfDiv)
  const sctl::Long N = Nt_ * Np_, dof = F.Dim() / (COORD_DIM * N);
  SCTL_ASSERT(F.Dim() == dof * COORD_DIM * N);
  SCTL_ASSERT(normal.Dim() == COORD_DIM * N);
  sctl::ScratchBuf<Real> C_buf(4 * N), A_buf(dof * 2 * N); // in scratch memory: see SurfGrad
  sctl::Vector<Real> C(C_buf), A(A_buf);
  LapCoeff(C, dX);
  sctl::omp_par::parallel_for(N, [&normal,&F,&A,&dX,N,dof](const sctl::Long i) {
    const Real n0 = normal[0*N+i], n1 = normal[1*N+i], n2 = normal[2*N+i];
    for (sctl::Long k = 0; k < dof; k++) {
      const Real f0 = F[(k*COORD_DIM+0)*N+i], f1 = F[(k*COORD_DIM+1)*N+i], f2 = F[(k*COORD_DIM+2)*N+i];
      const Real v0 = n1 * f2 - n2 * f1, v1 = n2 * f0 - n0 * f2, v2 = n0 * f1 - n1 * f0;
      for (sctl::Integer d = 0; d < 2; d++) A[(k*2+d)*N+i] = v0 * dX[(0*2+d)*N+i] + v1 * dX[(1*2+d)*N+i] + v2 * dX[(2*2+d)*N+i];
    }
  });
  CovDiv(CurlF, C, A);
}

template <class Real> void SurfaceOp<Real>::SurfGrad(sctl::Vector<Real>& GradFvec, const sctl::Vector<Real>& dXvec, const sctl::Vector<Real>& Fvec) const {
  auto inv2x2 = [](sctl::Tensor<Real, true, 2, 2> M) {
    sctl::Tensor<Real, true, 2, 2> Mout;
    Real oodet = 1 / (M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0));
    Mout(0, 0) = M(1, 1) * oodet;
    Mout(0, 1) = -M(0, 1) * oodet;
    Mout(1, 0) = -M(1, 0) * oodet;
    Mout(1, 1) = M(0, 0) * oodet;
    return Mout;
  };
  SCTL_ASSERT(dXvec.Dim() == 2 * COORD_DIM * Nt_ * Np_);
  const sctl::Long N = Nt_ * Np_, dof = Fvec.Dim() / N;
  assert(Fvec.Dim() == dof * N);

  // dFvec in scratch memory, like the temporaries of the other surface operators: a new allocation of this size is new
  // memory for the threads to fault in, at every call
  sctl::ScratchBuf<Real> dFvec_buf(dof * 2 * N);
  sctl::Vector<Real> dFvec(dFvec_buf);
  Grad2D(dFvec, Fvec);

  if (GradFvec.Dim() != dof * COORD_DIM * N) GradFvec.ReInit(dof * COORD_DIM * N);
  sctl::omp_par::parallel_for(N, [&dXvec,&inv2x2,&dFvec,&GradFvec,N,dof](const sctl::Long i) { // the components (k*2+d)*N+i of dXvec and dFvec, and (k*COORD_DIM+l)*N+i of GradFvec
    sctl::Tensor<Real, true, 3, 2> dX_du;
    for (sctl::Integer l = 0; l < COORD_DIM; l++) {
      for (sctl::Integer d = 0; d < 2; d++) dX_du(l, d) = dXvec[(l*2+d)*N+i];
    }
    const auto du_dX = inv2x2(dX_du.RotateLeft() * dX_du) * dX_du.RotateLeft(); // RotateLeft: the transpose
    for (sctl::Long k = 0; k < dof; k++) {
      sctl::Tensor<Real, true, 1, 2> dF_du;
      for (sctl::Integer d = 0; d < 2; d++) dF_du(0, d) = dFvec[(k*2+d)*N+i];
      const sctl::Tensor<Real, true, 1, 3> GradF = dF_du * du_dX;
      for (sctl::Integer l = 0; l < COORD_DIM; l++) GradFvec[(k*COORD_DIM+l)*N+i] = GradF(0, l);
    }
  });
}

template <class Real> void SurfaceOp<Real>::SurfDiv(sctl::Vector<Real>& DivF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const {
  // From the components F . dX/du_j of F along the tangent vectors: dot products, which repeat from one field period to
  // the next, unlike the Cartesian components of F
  const sctl::Long N = Nt_ * Np_, dof = F.Dim() / (COORD_DIM * N);
  SCTL_ASSERT(F.Dim() == dof * COORD_DIM * N);
  sctl::ScratchBuf<Real> C_buf(4 * N), A_buf(dof * 2 * N); // in scratch memory: see SurfGrad
  sctl::Vector<Real> C(C_buf), A(A_buf);
  LapCoeff(C, dX);
  sctl::omp_par::parallel_for(N, [&F,&dX,&A,dof,N](const sctl::Long i) {
    for (sctl::Long k = 0; k < dof; k++) {
      for (sctl::Integer d = 0; d < 2; d++) {
        Real a = 0;
        for (sctl::Integer l = 0; l < COORD_DIM; l++) a += F[(k*COORD_DIM+l)*N+i] * dX[(l*2+d)*N+i];
        A[(k*2+d)*N+i] = a;
      }
    }
  });
  CovDiv(DivF, C, A);
}

template <class Real> void SurfaceOp<Real>::SurfLap(sctl::Vector<Real>& LapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const {
  const sctl::Long N = Nt_ * Np_;
  sctl::ScratchBuf<Real> C_buf(4 * N), dF_buf(2 * F.Dim()); // in scratch memory: see SurfGrad
  sctl::Vector<Real> C(C_buf), dF(dF_buf);
  LapCoeff(C, dX);
  Grad2D(dF, F);
  CovDiv(LapF, C, dF);
}

template <class Real> void SurfaceOp<Real>::Div2D(sctl::Vector<Real>& DivA, const sctl::Vector<Real>& A) const {
  const sctl::Long Nt = Nt_, Np = Np_, N = Nt * Np, dof = A.Dim() / (2 * N);
  SCTL_ASSERT(A.Dim() == dof * 2 * N);
  if (DivA.Dim() != dof * N) DivA.ReInit(dof * N);
  if (!dof) return;

  // As in Grad2D: T, the Fourier coefficients along the rows of the toroidal components, and P, the derivatives along
  // the rows of the poloidal components; then the derivatives of T along the columns, plus P, transformed back
  const sctl::Long Q = Np / 2 + 1, ld = detail_surfop::PaddedCols(Q);
  sctl::ScratchBuf<Real> T_buf(dof * Nt * ld * 2), P_buf(dof * Nt * ld * 2);
  const sctl::Iterator<Real> T = T_buf.begin(), P = P_buf.begin(), DivA_ = DivA.begin();
  const sctl::ConstIterator<Real> A_ = A.begin();
  const Real sp = (2 * sctl::const_pi<Real>()) / (Real)Np; // FFT is unnormalized
  detail_surfop::RowsR2C<Real>(dof * 2 * Nt, Np, Q, [A_,Np](const sctl::Long r) { return A_ + r * Np; }, [T,P,ld,Q,Nt,sp](const sctl::Long r, const sctl::ConstIterator<Real> c) {
    const sctl::Long k = r / (2 * Nt), d = (r / Nt) % 2, t = r % Nt;
    const sctl::Iterator<Real> row = (d == 0 ? T : P) + (k * Nt + t) * ld * 2;
    if (d == 0) {
      for (sctl::Long q = 0; q < 2 * Q; q++) row[q] = c[q];
    } else {
      for (sctl::Long p = 0; p < Q; p++) {
        row[p * 2 + 0] = c[p * 2 + 1] * sp * p;
        row[p * 2 + 1] = -c[p * 2 + 0] * sp * p;
      }
    }
    for (sctl::Long q = 2 * Q; q < 2 * ld; q++) row[q] = 0;
  });

  const Real st = (2 * sctl::const_pi<Real>()) * NFP_ / (Real)(Nt * Np); // FFT is unnormalized; t in [0,1/NFP) on one of NFP field periods
  detail_surfop::Columns<Real>(T, Nt, T, Nt, dof, ld, Q, [Nt,st](const sctl::Iterator<Real> c1, const sctl::ConstIterator<Real> c0, const sctl::Long q) {
    for (sctl::Long t = 0; t < Nt; t++) {
      const Real tt = (Real)(t - (t > Nt / 2 ? Nt : 0));
      const Real real = c0[t * 2 + 0] * st, imag = c0[t * 2 + 1] * st;
      c1[t * 2 + 0] = imag * tt;
      c1[t * 2 + 1] = -real * tt;
    }
  });
  detail_surfop::RowsC2R<Real>(dof * Nt, Np, Q, [T,P,ld,Q](const sctl::Long r, const sctl::Iterator<Real> c) {
    for (sctl::Long q = 0; q < 2 * Q; q++) c[q] = T[r * ld * 2 + q] + P[r * ld * 2 + q];
  }, [DivA_,Np](const sctl::Long r, const sctl::ConstIterator<Real> x) {
    for (sctl::Long j = 0; j < Np; j++) DivA_[r * Np + j] = x[j];
  });
}

template <class Real> void SurfaceOp<Real>::LapCoeff(sctl::Vector<Real>& C, const sctl::Vector<Real>& dX) const {
  const sctl::Long N = Nt_ * Np_;
  SCTL_ASSERT(dX.Dim() == COORD_DIM * 2 * N);
  if (C.Dim() != 4 * N) C.ReInit(4 * N);
  sctl::omp_par::parallel_for(N, [&dX,&C,N](const sctl::Long i) {
    Real gtt = 0, gtp = 0, gpp = 0;
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      const Real xt = dX[(k*2+0)*N+i], xp = dX[(k*2+1)*N+i];
      gtt += xt * xt;
      gtp += xt * xp;
      gpp += xp * xp;
    }
    const Real sqrt_g = sctl::sqrt<Real>(gtt * gpp - gtp * gtp), inv_sqrt_g = 1 / sqrt_g;
    C[0*N+i] = gpp * inv_sqrt_g; // sqrt(g) g^tt
    C[1*N+i] = -gtp * inv_sqrt_g; // sqrt(g) g^tp
    C[2*N+i] = gtt * inv_sqrt_g; // sqrt(g) g^pp
    C[3*N+i] = sqrt_g;
  });
}

template <class Real> void SurfaceOp<Real>::CovDiv(sctl::Vector<Real>& DivA, const sctl::Vector<Real>& C, const sctl::Vector<Real>& A) const {
  const sctl::Long N = Nt_ * Np_, dof = A.Dim() / (2 * N);
  SCTL_ASSERT(A.Dim() == dof * 2 * N);
  SCTL_ASSERT(C.Dim() == 4 * N);
  sctl::ScratchBuf<Real> a_buf(dof * 2 * N); // the contravariant components times sqrt(g), in scratch memory: see SurfGrad
  sctl::Vector<Real> a(a_buf);
  sctl::omp_par::parallel_for(N, [&A,&a,&C,dof,N](const sctl::Long i) {
    for (sctl::Long k = 0; k < dof; k++) {
      const Real At = A[(k*2+0)*N+i], Ap = A[(k*2+1)*N+i];
      a[(k*2+0)*N+i] = C[0*N+i] * At + C[1*N+i] * Ap;
      a[(k*2+1)*N+i] = C[1*N+i] * At + C[2*N+i] * Ap;
    }
  });
  Div2D(DivA, a); // the two negations cancel
  sctl::omp_par::parallel_for(N, [&DivA,&C,dof,N](const sctl::Long i) {
    for (sctl::Long k = 0; k < dof; k++) DivA[k*N+i] /= C[3*N+i];
  });
}

template <class Real> void SurfaceOp<Real>::ResampleTangents(sctl::Vector<Real>& dX1, sctl::Long Nt1, sctl::Long Np1, const sctl::Vector<Real>& dX) const {
  if (NFP_ == 1) { // on the full torus, the tangent vectors are periodic
    Upsample(dX, Nt_, Np_, dX1, Nt1, Np1);
    return;
  }
  // On one of NFP field periods, the tangent vectors at t + 1/NFP are those at t rotated by 2*pi*tor_orient/NFP about
  // the z-axis. In the frame turned by -2*pi*tor_orient*t, they repeat from one period to the next, and are resampled
  // as periodic functions; their z-components are unchanged.
  const sctl::Long N = Nt_ * Np_;
  SCTL_ASSERT(dX.Dim() == COORD_DIM * 2 * N);
  const auto turn = [this](sctl::Vector<Real>& Y, const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const Real sign) { // the x and y components of X turned by sign*2*pi*tor_orient*t, into Y (which may be X)
    const sctl::Long N = Nt * Np;
    sctl::omp_par::parallel_for(Nt, [this,sign,Nt,Np,&X,N,&Y](const sctl::Long t) {
      const Real angle = sign * 2 * sctl::const_pi<Real>() * tor_orient_ * t / (Real)(NFP_ * Nt);
      const Real c = sctl::cos<Real>(angle), s = sctl::sin<Real>(angle);
      for (sctl::Integer d = 0; d < 2; d++) {
        for (sctl::Long p = 0; p < Np; p++) {
          const sctl::Long i = t * Np + p;
          const Real x = X[(0*2+d)*N+i], y = X[(1*2+d)*N+i];
          Y[(0*2+d)*N+i] = c * x - s * y;
          Y[(1*2+d)*N+i] = s * x + c * y;
        }
      }
    });
  };
  sctl::ScratchBuf<Real> W_buf(COORD_DIM * 2 * N); // in scratch memory: see SurfGrad
  sctl::Vector<Real> W(W_buf);
  turn(W, dX, Nt_, Np_, -1);
  for (sctl::Long i = 0; i < 2 * N; i++) W[2*2*N+i] = dX[2*2*N+i];
  Upsample(W, Nt_, Np_, dX1, Nt1, Np1);
  turn(dX1, dX1, Nt1, Np1, 1);
}

template <class Real> void SurfaceOp<Real>::SurfInteg(sctl::Vector<Real>& I, const sctl::Vector<Real>& area_elem, const sctl::Vector<Real>& F) const {
  sctl::Long dof = F.Dim() / (Nt_ * Np_);
  SCTL_ASSERT(F.Dim() == dof * Nt_ * Np_);
  SCTL_ASSERT(area_elem.Dim() == Nt_ * Np_);

  if (I.Dim() != dof) I.ReInit(dof);
  for (sctl::Long k = 0; k < dof; k++) {
    I[k] = sctl::omp_par::parallel_reduce(Nt_ * Np_, (Real)0, [this,&area_elem,&F,k](const sctl::Long i) { return area_elem[i] * F[k * Nt_ * Np_ + i]; }, std::plus<Real>());
  }
}

template <class Real> void SurfaceOp<Real>::ProjZeroMean(sctl::Vector<Real>& Fproj, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const {
  sctl::Long N = this->Nt_ * this->Np_;
  sctl::ScratchBuf<Real> area_elem_buf(N); // in scratch memory: see SurfGrad
  sctl::Vector<Real> area_elem(area_elem_buf);
  SurfNormalAreaElem(nullptr, &area_elem, dX, nullptr);

  const Real SurfArea = sctl::omp_par::parallel_reduce(N, (Real)0, [&area_elem](const sctl::Long i) { return area_elem[i]; }, std::plus<Real>());

  sctl::ScratchBuf<Real> Favg_buf(F.Dim() / N); // the integrals, of the right size for SurfInteg, also in scratch memory
  sctl::Vector<Real> Favg(Favg_buf);
  SurfInteg(Favg, area_elem, F);
  Favg /= SurfArea;

  sctl::Long dof = Favg.Dim();
  if (Fproj.Dim() != dof * N) Fproj.ReInit(dof * N);
  for (sctl::Long k = 0; k < dof; k++) {
    sctl::omp_par::parallel_for(N, [&Fproj,k,N,&F,&Favg](const sctl::Long i) {
      Fproj[k * N + i] = F[k * N + i] - Favg[k];
    });
  }
}

template <class Real> typename SurfaceOp<Real>::SpectralInvLapOp SurfaceOp<Real>::SpectralInvLap(const sctl::Vector<Real>& dX, const sctl::Long Nt_, const sctl::Long Np_, const sctl::Integer NFP) {
  struct Lengths { // the sums of |dX/dt| and of |dX/dp|
    Real t, p;
  };
  sctl::Long N = Nt_ * Np_;
  const Lengths L = sctl::omp_par::parallel_reduce(N, Lengths{0, 0}, [&dX,N](const sctl::Long i) {
    Real Xt = dX[(2*0+0) * N + i];
    Real Yt = dX[(2*1+0) * N + i];
    Real Zt = dX[(2*2+0) * N + i];

    Real Xp = dX[(2*0+1) * N + i];
    Real Yp = dX[(2*1+1) * N + i];
    Real Zp = dX[(2*2+1) * N + i];

    return Lengths{sqrt(Xt * Xt + Yt * Yt + Zt * Zt), sqrt(Xp * Xp + Yp * Yp + Zp * Zp)};
  }, [](const Lengths& a, const Lengths& b) { return Lengths{a.t + b.t, a.p + b.p}; });
  const Real Lt = L.t, Lp = L.p;
  const Real invLt2 = N * N / (Lt * Lt) * (NFP * NFP); // the toroidal frequencies of one of NFP field periods are those of the torus over NFP
  const Real invLp2 = N * N / (Lp * Lp);
  return SpectralInvLapOp{Nt_, Np_, invLt2, invLp2};
}

template <class Real> void SurfaceOp<Real>::SpectralInvLapOp::operator()(sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) const {
  const sctl::Long dof = x.Dim() / (Nt * Np);
  SCTL_ASSERT(x.Dim() == dof * Nt * Np);
  if (Sx.Dim() != x.Dim()) Sx.ReInit(x.Dim());
  const Real invLt2_ = invLt2, invLp2_ = invLp2;
  detail_surfop::FourierMultiply<Real>(Sx.begin(), x.begin(), dof, Nt, Np, [invLt2_,invLp2_](Real* f, const sctl::Long tt, const sctl::Long pp) { // coeff(t,p) <-- coeff(t,p) / (t*t + p*p)
    f[0] = (tt || pp ? 1 / (Real)(tt*tt*invLt2_ + pp*pp*invLp2_) : 0);
    f[1] = 0;
  });
}

template <class Real> void SurfaceOp<Real>::InvSurfLap(sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter, Real upsample) const {
  auto spectral_InvSurfLap = [this](sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, sctl::Long Nt_, sctl::Long Np_, Real tol, sctl::Integer max_iter) {
    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> IdentityOp =[](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { Sx = x; };
    // As a lambda of one reference to the preconditioner, which a std::function holds itself, where it would
    // allocate a copy of SpectralInvLapOp
    const SpectralInvLapOp spectral_inv_lap = SpectralInvLap(dX, Nt_, Np_, NFP_);
    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> FPrecond = [&spectral_inv_lap](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { spectral_inv_lap(Sx, x); };
    SurfaceOp<Real> Sop(comm_, Nt_, Np_, NFP_, tor_orient_);
    Sop.InvSurfLapPrecond(InvLapF, FPrecond, IdentityOp, dX, F, tol, max_iter);
  };
  sctl::Long Nt_up = (sctl::Long)(Nt_ * upsample);
  sctl::Long Np_up = (sctl::Long)(Np_ * upsample);
  const sctl::Long N = Nt_ * Np_, N_up = Nt_up * Np_up;
  sctl::ScratchBuf<Real> InvLapF_up_buf(F.Dim() / N * N_up), dX_up_buf(dX.Dim() / N * N_up), F_up_buf(F.Dim() / N * N_up); // see SurfGrad
  sctl::Vector<Real> InvLapF_up(InvLapF_up_buf), dX_up(dX_up_buf), F_up(F_up_buf);
  Upsample(F, Nt_, Np_, F_up, Nt_up, Np_up);
  ResampleTangents(dX_up, Nt_up, Np_up, dX); // the metric on the upsampled grid from the tangent vectors there
  spectral_InvSurfLap(InvLapF_up, dX_up, F_up, Nt_up, Np_up, tol, max_iter);
  Upsample(InvLapF_up, Nt_up, Np_up, InvLapF, Nt_, Np_);
}

template <class Real> void SurfaceOp<Real>::GradInvSurfLap(sctl::Vector<Real>& GradInvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter, Real upsample) const {
  // The gradient on the grid of F, of the solution downsampled to it, from the tangent vectors dX at its points: more
  // accurate than the gradient on the upsampled grid, downsampled
  sctl::ScratchBuf<Real> InvLapF_buf(F.Dim()); // see SurfGrad
  sctl::Vector<Real> InvLapF(InvLapF_buf);
  InvSurfLap(InvLapF, dX, F, tol, max_iter, upsample);
  SurfGrad(GradInvLapF, dX, InvLapF);
}

template <class Real> void SurfaceOp<Real>::InvSurfLapPrecond(sctl::Vector<Real>& InvLapF, const std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>& LeftPrecond, const std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>& RightPrecond, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter) const {
  // The coefficients of the operator, set once for all the iterations (in scratch memory: see SurfGrad), and the area
  // in the grid parameters, the sum of sqrt(g), for the means of Sx
  const sctl::Long N = Nt_ * Np_;
  sctl::ScratchBuf<Real> C_buf(4 * N);
  sctl::Vector<Real> C(C_buf);
  LapCoeff(C, dX);
  const Real area = sctl::omp_par::parallel_reduce(N, (Real)0, [&C,N](const sctl::Long i) { return C[3*N+i]; }, std::plus<Real>());

  // What the operator uses besides this, behind one reference: a std::function holds a lambda of two pointers itself,
  // where it would allocate one of four
  const struct {
    const std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>& LeftPrecond;
    const std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>& RightPrecond;
    const sctl::Vector<Real>& C;
    Real area;
  } args{LeftPrecond, RightPrecond, C, area};
  typename sctl::GMRES<Real>::ParallelOp fn = [this,&args](sctl::Vector<Real>* fx, const sctl::Vector<Real>& x) {
    const auto& C = args.C;
    const sctl::Long N = Nt_ * Np_;
    (*fx) = 0;
    SCTL_ASSERT(fx);
    // The temporaries in scratch memory (see SurfGrad); Sx may be resized by RightPrecond, which then allocates it, and
    // is zero for a RightPrecond that adds to its output
    const sctl::Long Nx = x.Dim();
    sctl::ScratchBuf<Real> Sx_buf(Nx), dSx_buf(2 * Nx), LSx_buf(Nx);
    sctl::Vector<Real> Sx(Sx_buf, false), dSx(dSx_buf), LSx(LSx_buf);
    sctl::omp_par::parallel_for(Nx, [&Sx](const sctl::Long i) { Sx[i] = 0; });
    args.RightPrecond(Sx, x);
    SCTL_ASSERT(Sx.Dim() == Nx);

    Grad2D(dSx, Sx); // LSx <-- SurfLap(Sx) + mean(Sx)
    CovDiv(LSx, C, dSx);
    for (sctl::Long k = 0; k < Nx / N; k++) {
      const Real mean = sctl::omp_par::parallel_reduce(N, (Real)0, [&Sx,k,&C,N](const sctl::Long i) { return C[3*N+i] * Sx[k*N+i]; }, std::plus<Real>()) / args.area;
      sctl::omp_par::parallel_for(N, [&LSx,k,mean,N](const sctl::Long i) { LSx[k*N+i] += mean; });
    }

    args.LeftPrecond(*fx, LSx);
  };
  // In scratch memory (see SurfGrad), except the solution of GMRES; SFproj and InvLapF_ may be resized by the
  // preconditioners, which then allocate them, and are zero for preconditioners that add to their outputs
  sctl::ScratchBuf<Real> Fproj_buf(F.Dim()), SFproj_buf(F.Dim()), InvLapF__buf(F.Dim());
  sctl::Vector<Real> Fproj(Fproj_buf), SFproj(SFproj_buf, false), InvLapF_(InvLapF__buf, false);
  sctl::omp_par::parallel_for(F.Dim(), [&SFproj,&InvLapF_](const sctl::Long i) {
    SFproj[i] = 0;
    InvLapF_[i] = 0;
  });
  // The solution of GMRES, given to it empty (a zero initial guess), which it then sizes: kept from one call to the
  // next on each thread, with its capacity, so that it is sized without an allocation also for a new SurfaceOp; a
  // vector of the call itself in a call from inside the operator of another
  static thread_local sctl::Vector<Real> InvSInvLapF_kept;
  static thread_local bool InvSInvLapF_kept_used = false;
  const bool use_kept = !InvSInvLapF_kept_used;
  sctl::Vector<Real> InvSInvLapF_own;
  sctl::Vector<Real>& InvSInvLapF = (use_kept ? InvSInvLapF_kept : InvSInvLapF_own);
  InvSInvLapF.ReInit(0);
  if (use_kept) InvSInvLapF_kept_used = true;
  ProjZeroMean(Fproj, dX, F);
  LeftPrecond(SFproj, Fproj);
  solver(&InvSInvLapF, fn, SFproj, tol, max_iter);
  RightPrecond(InvLapF_, InvSInvLapF);
  if (use_kept) InvSInvLapF_kept_used = false;
  ProjZeroMean(InvLapF, dX, InvLapF_);
}

template <class Real> template <sctl::Integer KDIM0, sctl::Integer KDIM1> void SurfaceOp<Real>::EvalSurfInteg(sctl::Vector<Real>& Utrg, const sctl::Vector<Real>& Xtrg, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& Xn_src, const sctl::Vector<Real>& Xa_src, const sctl::Vector<Real>& Fsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker) const {
  sctl::Long Nsrc = Nt_* Np_;
  sctl::Long dof = Fsrc.Dim() / (KDIM0 * Nsrc);
  SCTL_ASSERT(  Xsrc.Dim() == COORD_DIM * Nsrc);
  SCTL_ASSERT(Xn_src.Dim() == COORD_DIM * Nsrc);
  SCTL_ASSERT(Xa_src.Dim() ==             Nsrc);
  SCTL_ASSERT(  Fsrc.Dim() == dof*KDIM0 * Nsrc);

  // The temporaries in scratch memory (see SurfGrad)
  sctl::ScratchBuf<Real> Fa_buf(dof * KDIM0 * Nsrc);
  sctl::Vector<Real> Fa(Fa_buf);
  for (sctl::Long i = 0; i < Nsrc; i++) {
    for (sctl::Integer j = 0; j < dof * KDIM0; j++) {
      Fa[j * Nsrc + i] = Fsrc[j * Nsrc + i] * Xa_src[i];
    }
  }

  sctl::Long np = comm_.Size();
  sctl::Long rank = comm_.Rank();
  sctl::Long Ntrg = Xtrg.Dim() / COORD_DIM;
  SCTL_ASSERT(Xtrg.Dim() == COORD_DIM * Ntrg);
  sctl::Long a = (rank + 0) * Ntrg / np;
  sctl::Long b = (rank + 1) * Ntrg / np;

  sctl::ScratchBuf<Real> TrgX_buf(COORD_DIM * (b - a)), TrgU_buf(dof * KDIM1 * (b - a));
  sctl::Vector<Real> TrgX(TrgX_buf), TrgU(TrgU_buf);
  { // Init TrgX, TrgU
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = a; i < b; i++) {
        TrgX[k * (b - a) + i - a] = Xtrg[k * Ntrg + i];
      }
    }
    TrgU = 0;
  }
  ker(Xsrc, Xn_src, Fa, TrgX, TrgU);
  { // Set Utrg <-- Reduce(TrgU)
    sctl::ScratchBuf<Real> Uloc_buf(dof * KDIM1 * Ntrg), Uglb_buf(dof * KDIM1 * Ntrg);
    sctl::Vector<Real> Uloc(Uloc_buf), Uglb(Uglb_buf);
    Uloc = 0;
    for (sctl::Integer k = 0; k < dof * ker.Dim(1); k++) {
      for (sctl::Long i = a; i < b; i++) {
        Uloc[k * Ntrg + i] = TrgU[k * (b - a) + i - a];
      }
    }
    comm_.Allreduce(Uloc.begin(), Uglb.begin(), Uglb.Dim(), sctl::CommOp::SUM);
    if (Utrg.Dim() != dof * KDIM1 * Ntrg) {
      Utrg.ReInit(dof * KDIM1 * Ntrg);
      Utrg = 0;
    }
    Utrg += Uglb;
  }
}

template <class Real> void SurfaceOp<Real>::HodgeDecomp(sctl::Vector<Real>& Vn, sctl::Vector<Real>& Vd, sctl::Vector<Real>& Vc, sctl::Vector<Real>& Vh, const sctl::Vector<Real>& V, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& normal, Real tol, sctl::Long max_iter) const {
  sctl::Long N = Nt_ * Np_;
  SCTL_ASSERT(V.Dim() == COORD_DIM * N);
  if (Vn.Dim() != COORD_DIM * N) Vn.ReInit(COORD_DIM * N);
  if (Vd.Dim() != COORD_DIM * N) Vd.ReInit(COORD_DIM * N);
  if (Vc.Dim() != COORD_DIM * N) Vc.ReInit(COORD_DIM * N);
  if (Vh.Dim() != COORD_DIM * N) Vh.ReInit(COORD_DIM * N);
  // The temporaries in scratch memory: see SurfGrad
  { // Set Vn
    for (sctl::Long i = 0; i < N; i++) {
      Real VdotN = 0;
      for (sctl::Long k = 0; k < COORD_DIM; k++) VdotN += V[k * N + i] * normal[k * N + i];
      for (sctl::Long k = 0; k < COORD_DIM; k++) Vn[k * N + i] = VdotN * normal[k * N + i];
    }
  }
  { // Set Vd = Grad(InvLap(Div(V)))
    sctl::ScratchBuf<Real> DivV_buf(N);
    sctl::Vector<Real> DivV(DivV_buf);
    SurfDiv(DivV, dX, V);
    GradInvSurfLap(Vd, dX, DivV, tol * max_norm(V) / max_norm(DivV), max_iter, 1.5);
  }
  { // Set Vc = n x Grad(InvLap(Div(V x n)))
    auto cross_prod = [](sctl::Vector<Real>& axb, const sctl::Vector<Real>& a, const sctl::Vector<Real>& b) {
      sctl::Long N = a.Dim() / COORD_DIM;
      SCTL_ASSERT(a.Dim() == COORD_DIM * N);
      SCTL_ASSERT(b.Dim() == COORD_DIM * N);
      if (axb.Dim() != COORD_DIM * N) axb.ReInit(COORD_DIM * N);
      for (sctl::Long i = 0; i < N; i++) {
        axb[0 * N + i] = a[1 * N + i] * b[2 * N + i] - a[2 * N + i] * b[1 * N + i];
        axb[1 * N + i] = a[2 * N + i] * b[0 * N + i] - a[0 * N + i] * b[2 * N + i];
        axb[2 * N + i] = a[0 * N + i] * b[1 * N + i] - a[1 * N + i] * b[0 * N + i];
      }
    };
    sctl::ScratchBuf<Real> Vxn_buf(COORD_DIM * N), DivVxn_buf(N), GradInvLapDivVxn_buf(COORD_DIM * N);
    sctl::Vector<Real> Vxn(Vxn_buf), DivVxn(DivVxn_buf), GradInvLapDivVxn(GradInvLapDivVxn_buf);
    cross_prod(Vxn, V, normal);
    SurfDiv(DivVxn, dX, Vxn);
    GradInvSurfLap(GradInvLapDivVxn, dX, DivVxn, tol * max_norm(V) / max_norm(DivVxn), max_iter, 1.5);
    cross_prod(Vc, normal, GradInvLapDivVxn);
  }
  for (sctl::Long i = 0; i < COORD_DIM * N; i++) Vh[i] = V[i] - Vn[i] - Vd[i] - Vc[i];
}



template <class Real> void SurfaceOp<Real>::test_SurfGrad(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm) {
  Surface<Real> S(Nt, Np, surf_type);
  const sctl::Vector<Real> f(Nt * Np, (sctl::Iterator<Real>)S.Coord().begin() + 1 * Nt * Np);

  sctl::Vector<Real> Lf, dX, Sn;
  SurfaceOp Op(comm, Nt, Np);
  Op.Grad2D(dX, S.Coord());
  Op.SurfGrad(Lf, dX, f);

  Op.SurfNormalAreaElem(&Sn, nullptr, dX, &S.Coord());
  for (sctl::Long i = 0; i < Nt; i++) { // Lf <-- Lf - (e1 - e1 . n)
    for (sctl::Long j = 0; j < Np; j++) {
      Real n[3];
      n[0] = Sn[(0 * Nt + i) * Np + j];
      n[1] = Sn[(1 * Nt + i) * Np + j];
      n[2] = Sn[(2 * Nt + i) * Np + j];

      Real x[3] = {0,1,0};
      Real ndotx = x[0]*n[0] + x[1]*n[1] + x[2]*n[2];
      Lf[(0 * Nt + i) * Np + j] -= x[0] - ndotx * n[0];
      Lf[(1 * Nt + i) * Np + j] -= x[1] - ndotx * n[1];
      Lf[(2 * Nt + i) * Np + j] -= x[2] - ndotx * n[2];
    }
  }
  std::cout<<"|e|_inf = "<<max_norm(Lf)<<'\n';
}

template <class Real> void SurfaceOp<Real>::test_SurfLap(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm) {
  Surface<Real> S(Nt, Np, surf_type);
  sctl::Vector<Real> dX, d2X;
  SurfaceOp Op(comm, Nt, Np);
  Op.Grad2D(dX, S.Coord());
  Op.Grad2D(d2X, dX);

  sctl::Vector<Real> u0, f0, f1;
  Op.LaplaceBeltramiReference(f0, u0, S.Coord(), dX, d2X);
  Op.SurfLap(f1, dX, u0);

  //WriteVTK("u0", S, u0);
  //WriteVTK("f0", S, f0);
  //WriteVTK("f1", S, f1);
  //WriteVTK("err", S, f1-f0);
  { // print error
    std::cout<<"|u|_inf = "<<max_norm(u0)<<'\n';
    std::cout<<"|f|_inf = "<<max_norm(f0)<<'\n';
    std::cout<<"|e|_inf = "<<max_norm(f1-f0)<<'\n';
  }
}

template <class Real> void SurfaceOp<Real>::test_InvSurfLap(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>* LeftPrecond, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>* RightPrecond, const sctl::Comm& comm) {
  Surface<Real> S(Nt, Np, surf_type);
  sctl::Vector<Real> dX, d2X;
  SurfaceOp Op(comm, Nt, Np);
  Op.Grad2D(dX, S.Coord());
  Op.Grad2D(d2X, dX);

  sctl::Vector<Real> f0, u0, u1;
  Op.LaplaceBeltramiReference(f0, u0, S.Coord(), dX, d2X);

  sctl::Profile::Tic("Solve", &comm);
  if (LeftPrecond == nullptr && RightPrecond == nullptr) {
    Op.InvSurfLap(u1, dX, f0, gmres_tol, gmres_iter, 1.0);
  } else {
    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> IdentityOp = [](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { Sx = x; };
    auto& LPrecond = (LeftPrecond ? *LeftPrecond : IdentityOp);
    auto& RPrecond = (RightPrecond ? *RightPrecond : IdentityOp);
    Op.InvSurfLapPrecond(u1, LPrecond, RPrecond, dX, f0, gmres_tol, gmres_iter);
  }
  sctl::Profile::Toc();

  //WriteVTK("u0", S, u0);
  //WriteVTK("u1", S, u1);
  //WriteVTK("err", S, u1-u0);
  //WriteVTK("f", S, f0);
  { // print error
    auto inf_norm = [](const sctl::Vector<Real>& v){
      Real max_err = 0;
      for (const auto& x: v) max_err = std::max(max_err, fabs(x));
      return max_err;
    };
    std::cout<<"|f|_inf = "<<inf_norm(f0)<<'\n';
    std::cout<<"|u|_inf = "<<inf_norm(u0)<<'\n';
    std::cout<<"|e|_inf = "<<inf_norm(u1-u0)<<'\n';
  }
}

template <class Real> void SurfaceOp<Real>::test_HodgeDecomp(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm) {
  sctl::Long N = Nt * Np;
  Surface<Real> S(Nt, Np, surf_type);
  sctl::Vector<Real> dX, normal, area_elem;
  SurfaceOp Op(comm, Nt, Np);
  Op.Grad2D(dX, S.Coord());
  Op.SurfNormalAreaElem(&normal, &area_elem, dX, &S.Coord());

  Real tol = 1e-9;
  sctl::Long max_iter = 100;
  sctl::Vector<Real> V(COORD_DIM * N), Vn, Vd, Vc, Vh;
  for (sctl::Long i = 0; i < N; i++) { // Set V = dX/du
    for (sctl::Long k = 0; k < COORD_DIM; k++) {
      V[k * N + i] = dX[(2*k+0) * N + i];
    }
  }

  Op.HodgeDecomp(Vn, Vd, Vc, Vh, V, dX, normal, tol, max_iter);

  auto l2norm = [](const sctl::Vector<Real>& v) {
    Real sum = 0;
    for (const auto x : v) sum += x*x;
    return sqrt(sum);
  };

  std::cout<<"|Vn|_2: "<<l2norm(Vn)<<'\n';
  std::cout<<"|Vd|_2: "<<l2norm(Vd)<<'\n';
  std::cout<<"|Vc|_2: "<<l2norm(Vc)<<'\n';
  std::cout<<"|Vh|_2: "<<l2norm(Vh)<<'\n';
  std::cout<<"|V |_2: "<<l2norm(V )<<'\n';

  //WriteVTK("V" , S, V);
  //WriteVTK("Vn", S, Vn);
  //WriteVTK("Vd", S, Vd);
  //WriteVTK("Vc", S, Vc);
  //WriteVTK("Vh", S, Vh);
}

template <class Real> void SurfaceOp<Real>::Init(const sctl::Comm& comm, sctl::Long Nt, sctl::Long Np, sctl::Integer NFP, sctl::Integer tor_orient) {
  SCTL_ASSERT(NFP >= 1);
  SCTL_ASSERT_MSG(NFP == 1 || tor_orient == 1 || tor_orient == -1, "SurfaceOp on one of NFP > 1 field periods needs the toroidal orientation, +1 or -1.");
  Nt_ = Nt;
  Np_ = Np;
  NFP_ = NFP;
  tor_orient_ = (NFP == 1 ? 1 : tor_orient);
  comm_ = comm;
  solver = sctl::GMRES<Real>(comm_,false);

  if (!Nt_ || !Np_) return;
}

template <class Real> Real SurfaceOp<Real>::max_norm(const sctl::Vector<Real>& x) {
  Real err = 0;
  for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
  return err;
}

template <class Real> Real SurfaceOp<Real>::compute_area_elem(sctl::StaticArray<Real, COORD_DIM>& xn, const sctl::StaticArray<Real, COORD_DIM>& xt, const sctl::StaticArray<Real, COORD_DIM>& xp) {
    xn[0] = xt[1] * xp[2] - xp[1] * xt[2];
    xn[1] = xt[2] * xp[0] - xp[2] * xt[0];
    xn[2] = xt[0] * xp[1] - xp[0] * xt[1];
    Real xa = sctl::sqrt<Real>(xn[0] * xn[0] + xn[1] * xn[1] + xn[2] * xn[2]);
    Real xa_inv = 1 / xa;
    xn[0] *= xa_inv;
    xn[1] *= xa_inv;
    xn[2] *= xa_inv;
    return xa;
};

template <class Real> void SurfaceOp<Real>::LaplaceBeltramiReference(sctl::Vector<Real>& f0, sctl::Vector<Real>& u0, const sctl::Vector<Real>& X, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X) const {
  sctl::Vector<Real> normal, area_elem;
  SurfNormalAreaElem(&normal, &area_elem, dX, &X);
  sctl::StaticArray<Real, COORD_DIM> Xs{2, 2, 2};
  { // Set u0
    sctl::Vector<Real> SrcCoord, SrcNormal, SrcValue;
    SrcCoord.ReInit(COORD_DIM, Xs, false);;
    SrcNormal = SrcCoord;
    SrcValue.PushBack(1);
    const auto& ker = Laplace3D<Real>::FxU();
    ker(SrcCoord, SrcNormal, SrcValue, X, u0);
    { // Project u0
      sctl::Vector<Real> one, surf_area, u0_avg;
      one = u0 * 0 + 1;
      SurfInteg(surf_area, area_elem, one);
      SurfInteg(u0_avg, area_elem, u0);
      u0_avg[0] /= surf_area[0];
      u0 -= u0_avg[0];
    }
  }
  { // Set f0
    sctl::Long N = Nt_ * Np_;
    f0.ReInit(N);
    for (sctl::Long i = 0; i < N; i++) {
      sctl::StaticArray<Real, COORD_DIM> Xt{X[0*N+i],X[1*N+i],X[2*N+i]};
      sctl::StaticArray<Real, COORD_DIM> Xn{normal[0*N+i],normal[1*N+i],normal[2*N+i]};
      sctl::StaticArray<Real, COORD_DIM> dR{Xt[0]-Xs[0], Xt[1]-Xs[1], Xt[2]-Xs[2]};
      Real R2 = dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2];
      Real invR = 1.0/sqrt(R2);
      Real invR3 = invR*invR*invR;
      Real invR5 = invR3*invR*invR;
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      sctl::StaticArray<Real, COORD_DIM> dG{
        -scal*dR[0]*invR3, -scal*dR[1]*invR3, -scal*dR[2]*invR3
      };
      sctl::StaticArray<Real, COORD_DIM*COORD_DIM> d2G{
        scal*(3*dR[0]*dR[0]*invR5-invR3), scal*(3*dR[0]*dR[1]*invR5      ), scal*(3*dR[0]*dR[2]*invR5      ),
        scal*(3*dR[1]*dR[0]*invR5      ), scal*(3*dR[1]*dR[1]*invR5-invR3), scal*(3*dR[1]*dR[2]*invR5      ),
        scal*(3*dR[2]*dR[0]*invR5      ), scal*(3*dR[2]*dR[1]*invR5      ), scal*(3*dR[2]*dR[2]*invR5-invR3)
      };

      Real Un = dG[0]*Xn[0] + dG[1]*Xn[1] + dG[2]*Xn[2];

      Real Unn = 0;
      Unn += d2G[0*COORD_DIM+0]*Xn[0]*Xn[0];
      Unn += d2G[0*COORD_DIM+1]*Xn[0]*Xn[1];
      Unn += d2G[0*COORD_DIM+2]*Xn[0]*Xn[2];
      Unn += d2G[1*COORD_DIM+0]*Xn[1]*Xn[0];
      Unn += d2G[1*COORD_DIM+1]*Xn[1]*Xn[1];
      Unn += d2G[1*COORD_DIM+2]*Xn[1]*Xn[2];
      Unn += d2G[2*COORD_DIM+0]*Xn[2]*Xn[0];
      Unn += d2G[2*COORD_DIM+1]*Xn[2]*Xn[1];
      Unn += d2G[2*COORD_DIM+2]*Xn[2]*Xn[2];

      Real H=0;
      { // TODO: this is wrong
        Real ds0 = (dX[0*N+i]*dX[0*N+i] + dX[2*N+i]*dX[2*N+i] + dX[4*N+i]*dX[4*N+i]);
        Real ds1 = (dX[1*N+i]*dX[1*N+i] + dX[3*N+i]*dX[3*N+i] + dX[5*N+i]*dX[5*N+i]);
        H -= d2X[ 0*N+i]*Xn[0]*0.5/ds0;
        H -= d2X[ 4*N+i]*Xn[1]*0.5/ds0;
        H -= d2X[ 8*N+i]*Xn[2]*0.5/ds0;
        H -= d2X[ 3*N+i]*Xn[0]*0.5/ds1;
        H -= d2X[ 7*N+i]*Xn[1]*0.5/ds1;
        H -= d2X[11*N+i]*Xn[2]*0.5/ds1;
      }

      f0[i] = -2*H*Un - Unn;
    }
  }
  { // Compute f0 numerically
    sctl::Vector<Real> f1, dX1, u1;
    sctl::Long upsample = 3;
    sctl::Long Nt1 = Nt_ * upsample, Np1 = Np_ * upsample;
    Upsample(dX, Nt_, Np_, dX1, Nt1, Np1);
    Upsample(u0, Nt_, Np_,  u1, Nt1, Np1);
    SurfaceOp<Real> SOp(comm_, Nt1, Np1);
    SOp.SurfLap(f1, dX1, u1);
    Upsample(f1, Nt1, Np1,  f0, Nt_, Np_);
  }
}




template <class Real> void SurfaceOp<Real>::RotateToroidal(sctl::Vector<Real>& X_, const sctl::Vector<Real>& X, const sctl::Long Nt_, const sctl::Long Np_, const Real dtheta) {
  const sctl::Long dof = X.Dim() / (Nt_*Np_);
  SCTL_ASSERT(X.Dim() == dof*Nt_*Np_);
  if (X_.Dim() != dof*Nt_*Np_) X_.ReInit(dof*Nt_*Np_);

  // coeff(k,t,p) <-- coeff(k,t,p) rotated by t*dtheta
  sctl::ScratchBuf<Real> rot(Nt_ * 2); // the factors of the toroidal frequencies, at t in [0, Nt_)
  for (sctl::Long t = 0; t < Nt_; t++) {
    rot[t * 2 + 0] = sctl::cos<Real>((t - (t > Nt_ / 2 ? Nt_ : 0))*dtheta);
    rot[t * 2 + 1] = sctl::sin<Real>((t - (t > Nt_ / 2 ? Nt_ : 0))*dtheta);
  }
  detail_surfop::FourierMultiply<Real>(X_.begin(), X.begin(), dof, Nt_, Np_, [&rot,Nt_](Real* f, const sctl::Long tt, const sctl::Long) {
    const sctl::Long t = (tt < 0 ? tt + Nt_ : tt);
    f[0] = rot[t * 2 + 0];
    f[1] = rot[t * 2 + 1];
  });
}
template <class Real> sctl::Integer SurfaceOp<Real>::ToroidalOrientation(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np) {
  const sctl::Long N = Nt * Np;
  SCTL_ASSERT(X.Dim() == COORD_DIM * N);
  Real cross_z = 0; // sum of z-components of X(t,p) x X(t+1,p)
  for (sctl::Long t = 0; t + 1 < Nt; t++) {
    for (sctl::Long p = 0; p < Np; p++) {
      const sctl::Long i0 = t * Np + p, i1 = (t + 1) * Np + p;
      cross_z += X[0 * N + i0] * X[1 * N + i1] - X[1 * N + i0] * X[0 * N + i1];
    }
  }
  return (cross_z < 0 ? -1 : 1);
}
template <class Real> void SurfaceOp<Real>::CompleteVecField(sctl::Vector<Real>& X, const bool is_surf, const bool half_period, const sctl::Integer NFP, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Y, const Real dtheta, const sctl::Integer tor_orient) {
  static constexpr sctl::Integer COORD_DIM = 3;
  const sctl::Long dof = Y.Dim() / (Nt*Np);
  SCTL_ASSERT(Y.Dim() == dof*Nt*Np);
  const sctl::Integer orient = (tor_orient ? tor_orient : (is_surf && dof == COORD_DIM ? ToroidalOrientation(Y, Nt, Np) : 1));

  if (half_period) {
    sctl::Vector<Real> Y_(dof*(Nt*2)*Np);
    if (dof == COORD_DIM) {
      const Real cos_theta = sctl::cos<Real>(2*sctl::const_pi<Real>()/NFP);
      const Real sin_theta = sctl::sin<Real>(2*sctl::const_pi<Real>()/NFP) * orient;

      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          Y_[(0*2*Nt+t)*Np+p] = Y[(0*Nt+t)*Np+p];
          Y_[(1*2*Nt+t)*Np+p] = Y[(1*Nt+t)*Np+p];
          Y_[(2*2*Nt+t)*Np+p] = Y[(2*Nt+t)*Np+p];

          const Real x =  Y[(0*Nt+Nt-t-1)*Np+((Np-p)%Np)] * (is_surf?1:-1);
          const Real y = -Y[(1*Nt+Nt-t-1)*Np+((Np-p)%Np)] * (is_surf?1:-1);
          const Real z = -Y[(2*Nt+Nt-t-1)*Np+((Np-p)%Np)] * (is_surf?1:-1);
          Y_[(0*2*Nt+Nt+t)*Np+p] = x*cos_theta - y*sin_theta;
          Y_[(1*2*Nt+Nt+t)*Np+p] = x*sin_theta + y*cos_theta;
          Y_[(2*2*Nt+Nt+t)*Np+p] = z;
        }
      }
    } else {
      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          for (sctl::Long k = 0; k < dof; k++) {
            Y_[(k*2*Nt+t)*Np+p] = Y[(k*Nt+t)*Np+p];
            Y_[(k*2*Nt+Nt+t)*Np+p] = Y[(k*Nt+Nt-t-1)*Np+((Np-p)%Np)];
          }
        }
      }
    }
    return CompleteVecField(X, is_surf, false, NFP, 2*Nt, Np, Y_, dtheta, orient);
  }

  if (X.Dim() != dof*NFP*Nt*Np) X.ReInit(dof*NFP*Nt*Np);
  if (dof == COORD_DIM) {
    for (sctl::Long j = 0; j < NFP; j++) {
      const Real cost = sctl::cos<Real>(2*sctl::const_pi<Real>()*j/NFP);
      const Real sint = sctl::sin<Real>(2*sctl::const_pi<Real>()*j/NFP) * orient;
      for (sctl::Long i = 0; i < Nt*Np; i++) {
        const Real x0 = Y[0*Nt*Np+i];
        const Real y0 = Y[1*Nt*Np+i];
        const Real z0 = Y[2*Nt*Np+i];

        const Real x = x0*cost - y0*sint;
        const Real y = x0*sint + y0*cost;
        const Real z = z0;

        X[(0*NFP+j)*Nt*Np+i] = x;
        X[(1*NFP+j)*Nt*Np+i] = y;
        X[(2*NFP+j)*Nt*Np+i] = z;
      }
    }
  } else {
    for (sctl::Long j = 0; j < NFP; j++) {
      for (sctl::Long i = 0; i < Nt*Np; i++) {
        for (sctl::Long k = 0; k < dof; k++) {
          X[(k*NFP+j)*Nt*Np+i] = Y[k*Nt*Np+i];
        }
      }
    }
  }
  if (dtheta != (Real)0) { // rotate by dtheta
    sctl::Vector<Real> X_;
    RotateToroidal(X_, X, NFP*Nt, Np, dtheta);
    X = X_;
  }
}
template <class Real> void SurfaceOp<Real>::Resample(sctl::Vector<Real>& X1, const sctl::Long Nt1, const sctl::Long Np1, const sctl::Vector<Real>& X0, const sctl::Long Nt0, const sctl::Long Np0) {
  const sctl::Long skip_tor = (sctl::Long)sctl::ceil(Nt0/(Real)Nt1);
  const sctl::Long skip_pol = (sctl::Long)sctl::ceil(Np0/(Real)Np1);
  const sctl::Long dof = X0.Dim() / (Nt0 * Np0);
  SCTL_ASSERT(X0.Dim() == dof*Nt0*Np0);

  if (skip_tor == 1 && skip_pol == 1) { // no downsampling
    biest::SurfaceOp<Real>::Upsample(X0, Nt0, Np0, X1, Nt1, Np1);
    return;
  }
  sctl::ScratchBuf<Real> XX_buf(dof * Nt1*skip_tor * Np1*skip_pol);
  sctl::Vector<Real> XX(XX_buf);
  biest::SurfaceOp<Real>::Upsample(X0, Nt0, Np0, XX, Nt1*skip_tor, Np1*skip_pol);

  if (X1.Dim() != dof * Nt1*Np1) X1.ReInit(dof * Nt1*Np1);
  sctl::omp_par::parallel_for(dof * Nt1, [Np1,&X1,&XX,skip_tor,skip_pol](const sctl::Long ki) { // rows i of field k, as in Upsample
    for (sctl::Long j = 0; j < Np1; j++) {
      X1[ki*Np1+j] = XX[(ki*skip_tor*Np1+j)*skip_pol];
    }
  });
}

}

