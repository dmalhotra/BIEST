#include <sctl.hpp>
#include <omp.h>

namespace biest {

template <class Real> SurfaceOp<Real>::SurfaceOp(const sctl::Comm& comm, sctl::Long Nt, sctl::Long Np) {
  Init(comm, Nt, Np);
}

template <class Real> SurfaceOp<Real>::SurfaceOp(const SurfaceOp& Op) {
  Init(Op.comm_, Op.Nt_, Op.Np_);
}

template <class Real> SurfaceOp<Real>& SurfaceOp<Real>::operator=(const SurfaceOp& Op) {
  Init(Op.comm_, Op.Nt_, Op.Np_);
  return *this;
}

template <class Real> void SurfaceOp<Real>::Upsample(const sctl::Vector<Real>& X0_, sctl::Long Nt0, sctl::Long Np0, sctl::Vector<Real>& X1_, sctl::Long Nt1, sctl::Long Np1) {
  auto FFT_Helper = [](const sctl::FFT<Real>& fft, const sctl::Vector<Real>& in, sctl::Vector<Real>& out) {
    sctl::Long dof = in.Dim() / fft.Dim(0);
    if (out.Dim() != dof * fft.Dim(1)) {
      out.ReInit(dof * fft.Dim(1));
    }
    for (sctl::Long i = 0; i < dof; i++) {
      const sctl::Vector<Real> in_(fft.Dim(0), (sctl::Iterator<Real>)in.begin() + i * fft.Dim(0), false);
      sctl::Vector<Real> out_(fft.Dim(1), out.begin() + i * fft.Dim(1), false);
      fft.Execute(in_, out_);
    }
  };
  assert(X0_.Dim() % (Nt0 * Np0) == 0);

  sctl::Long Nt0_ = Nt0;
  sctl::Long Np0_ = Np0 / 2 + 1;

  sctl::Long Nt1_ = Nt1;
  sctl::Long Np1_ = Np1 / 2 + 1;

  sctl::FFT<Real> fft_r2c0, fft_c2r_;
  { // Initialize fft_r2c0, fft_c2r_
    sctl::StaticArray<sctl::Long, 2> fft_dim0{Nt0, Np0};
    sctl::StaticArray<sctl::Long, 2> fft_dim_{Nt1, Np1};
    fft_r2c0.Setup(sctl::FFT_Type::R2C, 1, sctl::Vector<sctl::Long>(2, fft_dim0, false), omp_get_max_threads());
    fft_c2r_.Setup(sctl::FFT_Type::C2R, 1, sctl::Vector<sctl::Long>(2, fft_dim_, false), omp_get_max_threads());
  }

  sctl::Vector<Real> tmp0, tmp_;
  auto X0__ = X0_; // TODO: make unaligned plans and remove this workaround
  FFT_Helper(fft_r2c0, X0__, tmp0);

  sctl::Long dof = tmp0.Dim() / (Nt0_*Np0_*2);
  SCTL_ASSERT(tmp0.Dim() == dof * Nt0_ * Np0_ * 2);
  if (tmp_.Dim() != dof * Nt1_ * Np1_ * 2) tmp_.ReInit(dof * Nt1_ * Np1_ * 2);
  tmp_.SetZero();

  Real scale = sctl::sqrt<Real>(Nt1 * Np1) / sctl::sqrt<Real>(Nt0 * Np0);
  sctl::Long Ntt = std::min(Nt0_, Nt1_);
  sctl::Long Npp = std::min(Np0_, Np1_);
  for (sctl::Long k = 0; k < dof; k++) {
    for (sctl::Long t = 0; t <= Ntt / 2; t++) {
      Real scale_ = 1;
      if (Nt0%2==0 && Nt0_ < Nt1_ && t == Ntt/2) scale_ = 0.5;
      if (Nt1%2==0 && Nt1_ < Nt0_ && t == Ntt/2) scale_ = 2.0;
      for (sctl::Long p = 0; p < Npp; p++) {
        Real scale__ = 1;
        if (Np0%2==0 && Np0_ < Np1_ && p == Npp-1) scale__ = 0.5;
        if (Np1%2==0 && Np1_ < Np0_ && p == Npp-1) scale__ = 2.0;
        tmp_[((k * Nt1_ + t) * Np1_ + p) * 2 + 0] = scale * scale_ * scale__ * tmp0[((k * Nt0_ + t) * Np0_ + p) * 2 + 0];
        tmp_[((k * Nt1_ + t) * Np1_ + p) * 2 + 1] = scale * scale_ * scale__ * tmp0[((k * Nt0_ + t) * Np0_ + p) * 2 + 1];
      }
    }
    for (sctl::Long t = 0; t < Ntt / 2; t++) {
      Real scale_ = 1;
      if (Nt0%2==0 && Nt0_ < Nt1_ && t == Ntt/2-1) scale_ = 0.5;
      if (Nt1%2==0 && Nt1_ < Nt0_ && t == Ntt/2-1) scale_ = 2.0;
      for (sctl::Long p = 0; p < Npp; p++) {
        Real scale__ = 1;
        if (Np0%2==0 && Np0_ < Np1_ && p == Npp-1) scale__ = 0.5;
        if (Np1%2==0 && Np1_ < Np0_ && p == Npp-1) scale__ = 2.0;
        tmp_[((k * Nt1_ + (Nt1_ - t - 1)) * Np1_ + p) * 2 + 0] = scale * scale_ * scale__ * tmp0[((k * Nt0_ + (Nt0_ - t - 1)) * Np0_ + p) * 2 + 0];
        tmp_[((k * Nt1_ + (Nt1_ - t - 1)) * Np1_ + p) * 2 + 1] = scale * scale_ * scale__ * tmp0[((k * Nt0_ + (Nt0_ - t - 1)) * Np0_ + p) * 2 + 1];
      }
    }
  }

  if (X1_.Dim() != dof  * Nt1 * Np1) X1_.ReInit(dof * Nt1 * Np1);
  FFT_Helper(fft_c2r_, tmp_, X1_);

  { // Floating-point correction
    sctl::Long Ut = Nt1 / Nt0;
    sctl::Long Up = Np1 / Np0;
    if (Nt1 == Nt0 * Ut && Np1 == Np0 * Up) {
      for (sctl::Long k = 0; k < dof; k++) {
        for (sctl::Long t = 0; t < Nt0; t++) {
          for (sctl::Long p = 0; p < Np0; p++) {
            X1_[(k * Nt1 + t * Ut) * Np1 + p * Up] = X0_[(k * Nt0 + t) * Np0 + p];
          }
        }
      }
    }
  }
}

template <class Real> void SurfaceOp<Real>::Grad2D(sctl::Vector<Real>& dX, const sctl::Vector<Real>& X) const {
  sctl::Long dof = X.Dim() / (Nt_ * Np_);
  assert(X.Dim() == dof * Nt_ * Np_);
  if (dX.Dim() != dof * 2 * Nt_ * Np_) {
    dX.ReInit(dof * 2 * Nt_ * Np_);
  }

  sctl::Long Nt = Nt_;
  sctl::Long Np = fft_r2c.Dim(1) / (Nt * 2);
  SCTL_ASSERT(fft_r2c.Dim(1) == Nt * Np * 2);
  sctl::Vector<Real> coeff(fft_r2c.Dim(1));
  sctl::Vector<Real> grad_coeff(fft_r2c.Dim(1));
  for (sctl::Long k = 0; k < dof; k++) {
    fft_r2c.Execute(sctl::Vector<Real>(Nt_*Np_, (sctl::Iterator<Real>)X.begin() + k*Nt_*Np_, false), coeff);

    Real scal = (2 * sctl::const_pi<Real>());
    #pragma omp parallel for schedule(static)
    for (sctl::Long t = 0; t < Nt; t++) { // grad_coeff(t,p) <-- imag * t * coeff(t,p)
      for (sctl::Long p = 0; p < Np; p++) {
        Real real = coeff[(t * Np + p) * 2 + 0] * scal;
        Real imag = coeff[(t * Np + p) * 2 + 1] * scal;
        grad_coeff[(t * Np + p) * 2 + 0] =  imag * (t - (t > Nt / 2 ? Nt : 0));
        grad_coeff[(t * Np + p) * 2 + 1] = -real * (t - (t > Nt / 2 ? Nt : 0));
      }
    }
    { // dX <-- IFFT(grad_coeff)
      sctl::Vector<Real> fft_out(Nt_*Np_, dX.begin() + (k*2+0)*Nt_*Np_, false);
      fft_c2r.Execute(grad_coeff, fft_out);
    }

    scal = (2 * sctl::const_pi<Real>());
    #pragma omp parallel for schedule(static)
    for (sctl::Long t = 0; t < Nt; t++) { // grad_coeff(t,p) <-- imag * p * coeff(t,p)
      for (sctl::Long p = 0; p < Np; p++) {
        Real real = coeff[(t * Np + p) * 2 + 0] * scal;
        Real imag = coeff[(t * Np + p) * 2 + 1] * scal;
        grad_coeff[(t * Np + p) * 2 + 0] =  imag * p;
        grad_coeff[(t * Np + p) * 2 + 1] = -real * p;
      }
    }
    { // dX <-- IFFT(grad_coeff)
      sctl::Vector<Real> fft_out(Nt_*Np_, dX.begin() + (k*2+1)*Nt_*Np_, false);
      fft_c2r.Execute(grad_coeff, fft_out);
    }
  }
}

template <class Real> Real SurfaceOp<Real>::SurfNormalAreaElem(sctl::Vector<Real>* normal, sctl::Vector<Real>* area_elem, const sctl::Vector<Real>& dX, const sctl::Vector<Real>* X) const {
  if (normal != nullptr && normal->Dim() != COORD_DIM * Nt_ * Np_) {
    normal->ReInit(COORD_DIM * Nt_ * Np_);
  }
  if (area_elem != nullptr && area_elem->Dim() != Nt_ * Np_) {
    area_elem->ReInit(Nt_ * Np_);
  }

  sctl::Long N = Nt_ * Np_;
  Real scal = 1 / (Real)N;
  if (normal != nullptr && area_elem != nullptr) {
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < N; i++) {
      sctl::StaticArray<Real, COORD_DIM> xt{dX[0*N+i], dX[2*N+i], dX[4*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xp{dX[1*N+i], dX[3*N+i], dX[5*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xn;
      (*area_elem)[i] = compute_area_elem(xn, xt, xp) * scal;
      (*normal)[0*N+i] = xn[0];
      (*normal)[1*N+i] = xn[1];
      (*normal)[2*N+i] = xn[2];
    }
  } else if (normal != nullptr) {
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < N; i++) {
      sctl::StaticArray<Real, COORD_DIM> xt{dX[0*N+i], dX[2*N+i], dX[4*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xp{dX[1*N+i], dX[3*N+i], dX[5*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xn;
      compute_area_elem(xn, xt, xp);
      (*normal)[0*N+i] = xn[0];
      (*normal)[1*N+i] = xn[1];
      (*normal)[2*N+i] = xn[2];
    }
  } else if (area_elem != nullptr) {
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < N; i++) {
      sctl::StaticArray<Real, COORD_DIM> xt{dX[0*N+i], dX[2*N+i], dX[4*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xp{dX[1*N+i], dX[3*N+i], dX[5*N+i]};
      sctl::StaticArray<Real, COORD_DIM> xn;
      (*area_elem)[i] = compute_area_elem(xn, xt, xp) * scal;
    }
  }

  Real normal_scal = 0;
  if (X != nullptr && normal != nullptr) { // Set Orientation
    sctl::Long idx = 0;
    for (sctl::Long j = 0; j < Nt_*Np_; j++) {
      if ((*X)[idx] < (*X)[j]) idx = j;
    }
    if ((*normal)[idx] < 0) {
      normal_scal = -1;
      (*normal) = (*normal) * normal_scal;
    } else {
      normal_scal = 1;
    }
  }
  return normal_scal;
}

template <class Real> void SurfaceOp<Real>::SurfCurl(sctl::Vector<Real>& CurlF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& normal, const sctl::Vector<Real>& F) const {
  sctl::Vector<Real> GradF;
  SurfGrad(GradF, dX, F);
  { // Compute CurlF <--curl(GradF)
    sctl::Long N = Nt_ * Np_;
    sctl::Long dof = F.Dim() / (COORD_DIM * Nt_ * Np_);
    assert(F.Dim() == dof * COORD_DIM * Nt_ * Np_);
    assert(normal.Dim() == COORD_DIM * Nt_ * Np_);
    if (CurlF.Dim() != dof * Nt_ * Np_) {
      CurlF.ReInit(dof * Nt_ * Np_);
    }
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
      for (sctl::Long k = 0; k < dof; k++) {
        Real curl = 0;
        sctl::Long idx = k*COORD_DIM*COORD_DIM + i;
        curl += normal[0*N+i] * (GradF[(1*COORD_DIM+2) * N + idx] - GradF[(2*COORD_DIM+1) * N + idx]);
        curl += normal[1*N+i] * (GradF[(2*COORD_DIM+0) * N + idx] - GradF[(0*COORD_DIM+2) * N + idx]);
        curl += normal[2*N+i] * (GradF[(0*COORD_DIM+1) * N + idx] - GradF[(1*COORD_DIM+0) * N + idx]);
        CurlF[k * N + i] = curl;
      }
    }
  }
}

template <class Real> void SurfaceOp<Real>::SurfGrad(sctl::Vector<Real>& GradFvec, const sctl::Vector<Real>& dXvec_, const sctl::Vector<Real>& Fvec) const {
  auto transpose0 = [this](sctl::Vector<Real>& V) {
    sctl::Long dof = V.Dim() / (Nt_ * Np_);
    assert(V.Dim() == dof * Nt_ * Np_);
    sctl::Matrix<Real>(Nt_ * Np_, dof, V.begin(), false) = sctl::Matrix<Real>(dof, Nt_ * Np_, V.begin(), false).Transpose();
  };
  auto transpose1 = [this](sctl::Vector<Real>& V) {
    sctl::Long dof = V.Dim() / (Nt_ * Np_);
    assert(V.Dim() == Nt_ * Np_ * dof);
    sctl::Matrix<Real>(dof, Nt_ * Np_, V.begin(), false) = sctl::Matrix<Real>(Nt_ * Np_, dof, V.begin(), false).Transpose();
  };
  auto inv2x2 = [](Mat<Real, 2, 2> M) {
    Mat<Real, 2, 2> Mout;
    Real oodet = 1 / (M[0][0] * M[1][1] - M[0][1] * M[1][0]);
    Mout[0][0] = M[1][1] * oodet;
    Mout[0][1] = -M[0][1] * oodet;
    Mout[1][0] = -M[1][0] * oodet;
    Mout[1][1] = M[0][0] * oodet;
    return Mout;
  };
  SCTL_ASSERT(dXvec_.Dim() == 2 * COORD_DIM * Nt_ * Np_);

  auto dXvec = dXvec_;
  sctl::Vector<Real> dFvec;
  Grad2D(dFvec, Fvec);
  transpose0(dXvec);
  transpose0(dFvec);

  { // Set GradFvec
    sctl::Long dof = Fvec.Dim() / (Nt_ * Np_);
    assert(Fvec.Dim() == Nt_ * Np_ * dof);
    if (GradFvec.Dim() != dof * COORD_DIM * Nt_ * Np_) {
      GradFvec.ReInit(dof * COORD_DIM * Nt_ * Np_);
    }
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
      const Mat<Real, 3, 2, false> dX_du(dXvec.begin() + i*3*2);
      const auto du_dX = inv2x2(dX_du.Transpose() * dX_du) * dX_du.Transpose();
      for (sctl::Long k = 0; k < dof; k++) {
        const Mat<Real, 1, 2, false> dF_du(dFvec.begin() + (i*dof+k)*1*2);
        Mat<Real, 1, 3, false> GradF(GradFvec.begin() + (i*dof+k)*1*3);
        GradF = dF_du * du_dX;
      }
    }
    transpose1(GradFvec);
  }
}

template <class Real> void SurfaceOp<Real>::SurfDiv(sctl::Vector<Real>& DivF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const {
  sctl::Vector<Real> GradF;
  SurfGrad(GradF, dX, F);
  { // Compute DivF <-- trace(GradF)
    sctl::Long N = Nt_ * Np_;
    sctl::Long dof = F.Dim() / (COORD_DIM * Nt_ * Np_);
    assert(F.Dim() == dof * COORD_DIM * Nt_ * Np_);
    if (DivF.Dim() != dof * Nt_ * Np_) {
      DivF.ReInit(dof * Nt_ * Np_);
    }
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
      for (sctl::Long k = 0; k < dof; k++) {
        Real trace = 0;
        for (sctl::Long l = 0; l < COORD_DIM; l++) {
          trace += GradF[(k * COORD_DIM * COORD_DIM + (COORD_DIM + 1) * l) * N + i];
        }
        DivF[k * N + i] = trace;
      }
    }
  }
}

template <class Real> void SurfaceOp<Real>::SurfLap(sctl::Vector<Real>& LapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const {
  sctl::Vector<Real> GradF;
  SurfGrad(GradF, dX, F);
  SurfDiv(LapF, dX, GradF);
}

template <class Real> void SurfaceOp<Real>::SurfInteg(sctl::Vector<Real>& I, const sctl::Vector<Real>& area_elem, const sctl::Vector<Real>& F) const {
  sctl::Long dof = F.Dim() / (Nt_ * Np_);
  SCTL_ASSERT(F.Dim() == dof * Nt_ * Np_);
  SCTL_ASSERT(area_elem.Dim() == Nt_ * Np_);

  if (I.Dim() != dof) I.ReInit(dof);
  for (sctl::Long k = 0; k < dof; k++) {
    Real sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
      sum += area_elem[i] * F[k * Nt_ * Np_ + i];
    }
    I[k] = sum;
  }
}

template <class Real> void SurfaceOp<Real>::ProjZeroMean(sctl::Vector<Real>& Fproj, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const {
  sctl::Long N = this->Nt_ * this->Np_;
  sctl::Vector<Real> area_elem;
  SurfNormalAreaElem(nullptr, &area_elem, dX, nullptr);

  sctl::Vector<Real> Favg, SurfArea, one(N);
  one = 1;
  SurfInteg(SurfArea, area_elem, one);

  SurfInteg(Favg, area_elem, F);
  Favg /= SurfArea[0];

  sctl::Long dof = Favg.Dim();
  Fproj.ReInit(dof * N);
  for (sctl::Long k = 0; k < dof; k++) {
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < N; i++) {
      Fproj[k * N + i] = F[k * N + i] - Favg[k];
    }
  }
}

template <class Real> void SurfaceOp<Real>::InvSurfLap(sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter, Real upsample) const {
  auto spectral_InvSurfLap = [this](sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, sctl::Long Nt_, sctl::Long Np_, Real tol, sctl::Integer max_iter) {
    sctl::FFT<Real> fft_r2c, fft_c2r;
    sctl::StaticArray<sctl::Long, 2> fft_dim = {Nt_, Np_};
    fft_r2c.Setup(sctl::FFT_Type::R2C, 1, sctl::Vector<sctl::Long>(2, fft_dim, false), omp_get_max_threads());
    fft_c2r.Setup(sctl::FFT_Type::C2R, 1, sctl::Vector<sctl::Long>(2, fft_dim, false), omp_get_max_threads());

    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> IdentityOp =[](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { Sx = x; };
    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> FPrecond = [Nt_, Np_, &fft_r2c, &fft_c2r, &dX](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) {
      sctl::Vector<Real> coeff;
      fft_r2c.Execute(x, coeff);

      Real Lt = 0, Lp = 0;
      sctl::Long N = Nt_ * Np_;
      for (sctl::Long i = 0; i < N; i++) {
        Real Xt = dX[(2*0+0) * N + i];
        Real Yt = dX[(2*1+0) * N + i];
        Real Zt = dX[(2*2+0) * N + i];

        Real Xp = dX[(2*0+1) * N + i];
        Real Yp = dX[(2*1+1) * N + i];
        Real Zp = dX[(2*2+1) * N + i];

        Lt += sqrt(Xt * Xt + Yt * Yt + Zt * Zt);
        Lp += sqrt(Xp * Xp + Yp * Yp + Zp * Zp);
      }
      Real invLt2 = N * N / (Lt * Lt);
      Real invLp2 = N * N / (Lp * Lp);

      sctl::Long Nt0_ = Nt_;
      sctl::Long Np0_ = fft_r2c.Dim(1) / (Nt0_ * 2);
      for (sctl::Long t = 0; t < Nt0_; t++) { // coeff(t,p) <-- coeff(t,p) / (t*t + p*p)
        for (sctl::Long p = 0; p < Np0_; p++) {
          sctl::Long tt = (t - (t > Nt0_ / 2 ? Nt0_ : 0));
          sctl::Long pp = p;
          Real scal = (tt || pp ? 1 / (Real)(tt*tt*invLt2 + pp*pp*invLp2) : 0);
          coeff[(t * Np0_ + p) * 2 + 0] *= scal;
          coeff[(t * Np0_ + p) * 2 + 1] *= scal;
        }
      }
      fft_c2r.Execute(coeff, Sx);
    };
    SurfaceOp<Real> Sop(comm_, Nt_, Np_);
    Sop.InvSurfLapPrecond(InvLapF, FPrecond, IdentityOp, dX, F, tol, max_iter);
  };
  sctl::Long Nt_up = (sctl::Long)(Nt_ * upsample);
  sctl::Long Np_up = (sctl::Long)(Np_ * upsample);
  sctl::Vector<Real> InvLapF_up, dX_up, F_up;
  Upsample(F, Nt_, Np_, F_up, Nt_up, Np_up);
  Upsample(dX, Nt_, Np_, dX_up, Nt_up, Np_up);
  spectral_InvSurfLap(InvLapF_up, dX_up, F_up, Nt_up, Np_up, tol, max_iter);
  Upsample(InvLapF_up, Nt_up, Np_up, InvLapF, Nt_, Np_);
}

template <class Real> void SurfaceOp<Real>::GradInvSurfLap(sctl::Vector<Real>& GradInvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter, Real upsample) const {
  auto spectral_GradInvSurfLap = [this](sctl::Vector<Real>& GradInvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, sctl::Long Nt_, sctl::Long Np_, Real tol, sctl::Integer max_iter) {
    sctl::FFT<Real> fft_r2c, fft_c2r;
    sctl::StaticArray<sctl::Long, 2> fft_dim{Nt_, Np_};
    fft_r2c.Setup(sctl::FFT_Type::R2C, 1, sctl::Vector<sctl::Long>(2, fft_dim, false), omp_get_max_threads());
    fft_c2r.Setup(sctl::FFT_Type::C2R, 1, sctl::Vector<sctl::Long>(2, fft_dim, false), omp_get_max_threads());

    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> IdentityOp = [](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { Sx = x; };
    std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> FPrecond = [Nt_, Np_, &fft_r2c, &fft_c2r, &dX](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) {
      sctl::Vector<Real> coeff;
      fft_r2c.Execute(x, coeff);

      Real Lt = 0, Lp = 0;
      sctl::Long N = Nt_ * Np_;
      #pragma omp parallel for schedule(static) reduction(+:Lt,Lp)
      for (sctl::Long i = 0; i < N; i++) {
        Real Xt = dX[(2*0+0) * N + i];
        Real Yt = dX[(2*1+0) * N + i];
        Real Zt = dX[(2*2+0) * N + i];

        Real Xp = dX[(2*0+1) * N + i];
        Real Yp = dX[(2*1+1) * N + i];
        Real Zp = dX[(2*2+1) * N + i];

        Lt += sqrt(Xt * Xt + Yt * Yt + Zt * Zt);
        Lp += sqrt(Xp * Xp + Yp * Yp + Zp * Zp);
      }
      Real invLt2 = N * N / (Lt * Lt);
      Real invLp2 = N * N / (Lp * Lp);

      sctl::Long Nt0_ = Nt_;
      sctl::Long Np0_ = fft_r2c.Dim(1) / (Nt0_ * 2);
      #pragma omp parallel for schedule(static)
      for (sctl::Long t = 0; t < Nt0_; t++) { // coeff(t,p) <-- coeff(t,p) / (t*t + p*p)
        for (sctl::Long p = 0; p < Np0_; p++) {
          sctl::Long tt = (t - (t > Nt0_ / 2 ? Nt0_ : 0));
          sctl::Long pp = p;
          Real scal = (tt || pp ? 1 / (Real)(tt*tt*invLt2 + pp*pp*invLp2) : 0);
          coeff[(t * Np0_ + p) * 2 + 0] *= scal;
          coeff[(t * Np0_ + p) * 2 + 1] *= scal;
        }
      }
      fft_c2r.Execute(coeff, Sx);
    };
    sctl::Vector<Real> InvLapF;
    SurfaceOp<Real> Sop(comm_, Nt_, Np_);
    Sop.InvSurfLapPrecond(InvLapF, FPrecond, IdentityOp, dX, F, tol, max_iter);
    Sop.SurfGrad(GradInvLapF, dX, InvLapF);
  };
  sctl::Long Nt_up = (sctl::Long)(Nt_ * upsample);
  sctl::Long Np_up = (sctl::Long)(Np_ * upsample);
  sctl::Vector<Real> GradInvLapF_up, dX_up, F_up;
  Upsample(F, Nt_, Np_, F_up, Nt_up, Np_up);
  Upsample(dX, Nt_, Np_, dX_up, Nt_up, Np_up);
  spectral_GradInvSurfLap(GradInvLapF_up, dX_up, F_up, Nt_up, Np_up, tol, max_iter);
  Upsample(GradInvLapF_up, Nt_up, Np_up, GradInvLapF, Nt_, Np_);
}

template <class Real> void SurfaceOp<Real>::InvSurfLapPrecond(sctl::Vector<Real>& InvLapF, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> LeftPrecond, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> RightPrecond, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter) const {
  typename sctl::ParallelSolver<Real>::ParallelOp fn = [this,&LeftPrecond,&RightPrecond,&dX](sctl::Vector<Real>* fx, const sctl::Vector<Real>& x) {
    (*fx) = 0;
    SCTL_ASSERT(fx);
    sctl::Vector<Real> Sx, LSx;
    RightPrecond(Sx, x);

    SurfLap(LSx, dX, Sx);
    sctl::Vector<Real> Proj_Sx;
    ProjZeroMean(Proj_Sx, dX, Sx);
    LSx += (Sx - Proj_Sx)*1;

    LeftPrecond(*fx, LSx);
  };
  sctl::Vector<Real> Fproj, SFproj, InvSInvLapF, InvLapF_;
  ProjZeroMean(Fproj, dX, F);
  LeftPrecond(SFproj, Fproj);
  solver(&InvSInvLapF, fn, SFproj, tol, max_iter);
  RightPrecond(InvLapF_, InvSInvLapF);
  ProjZeroMean(InvLapF, dX, InvLapF_);
}

template <class Real> template <sctl::Integer KDIM0, sctl::Integer KDIM1> void SurfaceOp<Real>::EvalSurfInteg(sctl::Vector<Real>& Utrg, const sctl::Vector<Real>& Xtrg, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& Xn_src, const sctl::Vector<Real>& Xa_src, const sctl::Vector<Real>& Fsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker) const {
  sctl::Long Nsrc = Nt_* Np_;
  sctl::Long dof = Fsrc.Dim() / (KDIM0 * Nsrc);
  SCTL_ASSERT(  Xsrc.Dim() == COORD_DIM * Nsrc);
  SCTL_ASSERT(Xn_src.Dim() == COORD_DIM * Nsrc);
  SCTL_ASSERT(Xa_src.Dim() ==             Nsrc);
  SCTL_ASSERT(  Fsrc.Dim() == dof*KDIM0 * Nsrc);

  sctl::Vector<Real> Fa(dof * KDIM0 * Nsrc);
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

  sctl::Vector<Real> TrgX, TrgU;
  { // Init TrgX, TrgU
    TrgX.ReInit(COORD_DIM * (b - a));
    TrgU.ReInit(dof * KDIM1 * (b - a));
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = a; i < b; i++) {
        TrgX[k * (b - a) + i - a] = Xtrg[k * Ntrg + i];
      }
    }
    TrgU = 0;
  }
  ker(Xsrc, Xn_src, Fa, TrgX, TrgU);
  { // Set Utrg <-- Reduce(TrgU)
    sctl::Vector<Real> Uloc(dof * KDIM1 * Ntrg), Uglb(dof * KDIM1 * Ntrg); Uloc = 0;
    for (sctl::Integer k = 0; k < dof * ker.Dim(1); k++) {
      for (sctl::Long i = a; i < b; i++) {
        Uloc[k * Ntrg + i] = TrgU[k * (b - a) + i - a];
      }
    }
    comm_.template Allreduce<Real>(Uloc.begin(), Uglb.begin(), Uglb.Dim(), sctl::Comm::CommOp::SUM);
    if (Utrg.Dim() != dof * KDIM1 * Ntrg) {
      Utrg.ReInit(dof * KDIM1 * Ntrg);
      Utrg = 0;
    }
    Utrg += Uglb;
  }
}

template <class Real> template <class SingularCorrection, class Kernel> void SurfaceOp<Real>::SetupSingularCorrection(sctl::Vector<SingularCorrection>& singular_correction, sctl::Integer TRG_SKIP, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& dXsrc, const Kernel& ker, const Real normal_scal, const sctl::Vector<sctl::Long>& trg_idx) const {
  const sctl::Long np = comm_.Size();
  const sctl::Long rank = comm_.Rank();
  const sctl::Long Ntrg = trg_idx.Dim();
  //const sctl::Long Nt0 = Nt_ / TRG_SKIP;
  const sctl::Long Np0 = Np_ / TRG_SKIP;
  const sctl::Long a = (rank + 0) * Ntrg / np;
  const sctl::Long b = (rank + 1) * Ntrg / np;
  const sctl::Long Nloc = b-a;
  singular_correction.ReInit(Nloc);

  const sctl::Integer Nthread = omp_get_max_threads();
  sctl::Vector<sctl::Vector<Real>> work_buff(Nthread);
  #pragma omp parallel for schedule(static)
  for (sctl::Integer tid = 0; tid < Nthread; tid++) {
    const sctl::Long a_ = a + Nloc*(tid+0)/Nthread;
    const sctl::Long b_ = a + Nloc*(tid+1)/Nthread;
    for (sctl::Long i = a_; i < b_; i++) {
      const sctl::Long t = (trg_idx[i] / Np0) * TRG_SKIP;
      const sctl::Long p = (trg_idx[i] % Np0) * TRG_SKIP;
      singular_correction[i - a].Setup(TRG_SKIP, Nt_, Np_, Xsrc, dXsrc, t, p, i, trg_idx.Dim(), ker, normal_scal, work_buff[tid]);
    }
  }
}

template <class Real> template <class SingularCorrection> void SurfaceOp<Real>::EvalSingularCorrection(sctl::Vector<Real>& U, const sctl::Vector<SingularCorrection>& singular_correction, sctl::Integer kdim0, sctl::Integer kdim1, const sctl::Vector<Real>& F) const {
  sctl::Long Ntrg;
  { // Set N
    sctl::Long Nloc = singular_correction.Dim();
    comm_.Allreduce(sctl::Ptr2ConstItr<sctl::Long>(&Nloc,1), sctl::Ptr2Itr<sctl::Long>(&Ntrg,1), 1, sctl::Comm::CommOp::SUM);
  }
  sctl::Long dof = F.Dim() / (kdim0 * Nt_ * Np_);
  SCTL_ASSERT(F.Dim() == dof * kdim0 * Nt_ * Np_);
  sctl::Vector<Real> Uloc(dof * kdim1 * Ntrg), Uglb(dof * kdim1 * Ntrg);
  for (sctl::Long k = 0; k < dof; k++) { // Set Uloc
    const sctl::Vector<Real> F_(kdim0 * Nt_ * Np_, (sctl::Iterator<Real>)F.begin() + k * kdim0 * Nt_ * Np_, false);
    sctl::Vector<Real> U_(kdim1 * Ntrg, Uloc.begin() + k * kdim1 * Ntrg, false);
    U_ = 0;
    #pragma omp parallel for schedule(static)
    for (sctl::Long i = 0; i < singular_correction.Dim(); i++) {
      singular_correction[i](F_, U_);
    }
  }
  comm_.Allreduce(Uloc.begin(), Uglb.begin(), Uglb.Dim(), sctl::Comm::CommOp::SUM);
  if (U.Dim() != dof * kdim1 * Ntrg) { // Init U
    U.ReInit(dof * kdim1 * Ntrg);
    U = 0;
  }
  U += Uglb;
}

template <class Real> void SurfaceOp<Real>::HodgeDecomp(sctl::Vector<Real>& Vn, sctl::Vector<Real>& Vd, sctl::Vector<Real>& Vc, sctl::Vector<Real>& Vh, const sctl::Vector<Real>& V, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& normal, Real tol, sctl::Long max_iter) const {
  sctl::Long N = Nt_ * Np_;
  SCTL_ASSERT(V.Dim() == COORD_DIM * N);
  if (Vn.Dim() != COORD_DIM * N) Vn.ReInit(COORD_DIM * N);
  if (Vd.Dim() != COORD_DIM * N) Vd.ReInit(COORD_DIM * N);
  if (Vc.Dim() != COORD_DIM * N) Vc.ReInit(COORD_DIM * N);
  if (Vh.Dim() != COORD_DIM * N) Vh.ReInit(COORD_DIM * N);
  { // Set Vn
    Vn.ReInit(V.Dim());
    for (sctl::Long i = 0; i < N; i++) {
      Real VdotN = 0;
      for (sctl::Long k = 0; k < COORD_DIM; k++) VdotN += V[k * N + i] * normal[k * N + i];
      for (sctl::Long k = 0; k < COORD_DIM; k++) Vn[k * N + i] = VdotN * normal[k * N + i];
    }
  }
  { // Set Vd = Grad(InvLap(Div(V)))
    sctl::Vector<Real> DivV, GradInvLapDivV;
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
    sctl::Vector<Real> Vxn, DivVxn, GradInvLapDivVxn;
    cross_prod(Vxn, V, normal);
    SurfDiv(DivVxn, dX, Vxn);
    GradInvSurfLap(GradInvLapDivVxn, dX, DivVxn, tol * max_norm(V) / max_norm(DivVxn), max_iter, 1.5);
    cross_prod(Vc, normal, GradInvLapDivVxn);
  }
  Vh = V - Vn - Vd - Vc;
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

template <class Real> void SurfaceOp<Real>::Init(const sctl::Comm& comm, sctl::Long Nt, sctl::Long Np) {
  Nt_ = Nt;
  Np_ = Np;
  comm_ = comm;
  solver = sctl::ParallelSolver<Real>(comm_,false);

  if (!Nt_ || !Np_) return;
  sctl::StaticArray<sctl::Long, 2> fft_dim{Nt_, Np_};
  fft_r2c.Setup(sctl::FFT_Type::R2C, 1, sctl::Vector<sctl::Long>(2, fft_dim, false));
  fft_c2r.Setup(sctl::FFT_Type::C2R, 1, sctl::Vector<sctl::Long>(2, fft_dim, false));
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
    Real xa = sqrt(xn[0] * xn[0] + xn[1] * xn[1] + xn[2] * xn[2]);
    Real xa_inv = 1 / xa;
    xn[0] *= xa_inv;
    xn[1] *= xa_inv;
    xn[2] *= xa_inv;
    return xa;
};

template <class Real> void SurfaceOp<Real>::LaplaceBeltramiReference(sctl::Vector<Real>& f0, sctl::Vector<Real>& u0, const sctl::Vector<Real>& X, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X) const {
  sctl::Vector<Real> normal, area_elem;
  SurfNormalAreaElem(&normal, &area_elem, dX, &X);
  sctl::StaticArray<Real, COORD_DIM> Xs = {2, 2, 2};
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
      sctl::StaticArray<Real, COORD_DIM> Xt = {X[0*N+i],X[1*N+i],X[2*N+i]};
      sctl::StaticArray<Real, COORD_DIM> Xn = {normal[0*N+i],normal[1*N+i],normal[2*N+i]};
      sctl::StaticArray<Real, COORD_DIM> dR = {Xt[0]-Xs[0], Xt[1]-Xs[1], Xt[2]-Xs[2]};
      Real R2 = dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2];
      Real invR = 1.0/sqrt(R2);
      Real invR3 = invR*invR*invR;
      Real invR5 = invR3*invR*invR;
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      sctl::StaticArray<Real, COORD_DIM> dG = {
        -scal*dR[0]*invR3, -scal*dR[1]*invR3, -scal*dR[2]*invR3
      };
      sctl::StaticArray<Real, COORD_DIM*COORD_DIM> d2G = {
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

  sctl::FFT<Real> fft_r2c, fft_c2r;
  sctl::StaticArray<sctl::Long, 2> fft_dim{Nt_, Np_};
  fft_r2c.Setup(sctl::FFT_Type::R2C, 1, sctl::Vector<sctl::Long>(2, fft_dim, false));
  fft_c2r.Setup(sctl::FFT_Type::C2R, 1, sctl::Vector<sctl::Long>(2, fft_dim, false));

  sctl::Long Nt = Nt_;
  sctl::Long Np = fft_r2c.Dim(1) / (Nt * 2);
  SCTL_ASSERT(fft_r2c.Dim(1) == Nt * Np * 2);
  sctl::Vector<Real> coeff(fft_r2c.Dim(1));
  sctl::Vector<Real> coeff_(fft_r2c.Dim(1));
  for (sctl::Long k = 0; k < dof; k++) {
    fft_r2c.Execute(sctl::Vector<Real>(Nt_*Np_, (sctl::Iterator<Real>)X.begin() + k*Nt_*Np_, false), coeff);

    #pragma omp parallel for schedule(static)
    for (sctl::Long t = 0; t < Nt; t++) { // coeff_(t,p) <-- imag * t * coeff(t,p)
      const Real cos_tdt = sctl::cos<Real>((t - (t > Nt / 2 ? Nt : 0))*dtheta);
      const Real sin_tdt = sctl::sin<Real>((t - (t > Nt / 2 ? Nt : 0))*dtheta);
      for (sctl::Long p = 0; p < Np; p++) {
        Real real = coeff[(t * Np + p) * 2 + 0];
        Real imag = coeff[(t * Np + p) * 2 + 1];
        coeff_[(t * Np + p) * 2 + 0] = real*cos_tdt - imag*sin_tdt;
        coeff_[(t * Np + p) * 2 + 1] = real*sin_tdt + imag*cos_tdt;
      }
    }
    { // X_ <-- IFFT(coeff_)
      sctl::Vector<Real> fft_out(Nt_*Np_, X_.begin() + k*Nt_*Np_, false);
      fft_c2r.Execute(coeff_, fft_out);
    }
  }
}
template <class Real> void SurfaceOp<Real>::CompleteVecField(sctl::Vector<Real>& X, const bool is_surf, const bool half_period, const sctl::Integer NFP, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Y, const Real dtheta) {
  static constexpr sctl::Integer COORD_DIM = 3;
  const sctl::Long dof = Y.Dim() / (Nt*Np);
  SCTL_ASSERT(Y.Dim() == dof*Nt*Np);

  if (half_period) {
    sctl::Vector<Real> Y_(dof*(Nt*2)*Np);
    if (dof == COORD_DIM) {
      const Real cos_theta = sctl::cos<Real>(2*sctl::const_pi<Real>()/NFP);
      const Real sin_theta = sctl::sin<Real>(2*sctl::const_pi<Real>()/NFP);

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
    return CompleteVecField(X, is_surf, false, NFP, 2*Nt, Np, Y_, dtheta);
  }

  if (X.Dim() != dof*NFP*Nt*Np) X.ReInit(dof*NFP*Nt*Np);
  if (dof == COORD_DIM) {
    for (sctl::Long j = 0; j < NFP; j++) {
      const Real cost = sctl::cos<Real>(2*sctl::const_pi<Real>()*j/NFP);
      const Real sint = sctl::sin<Real>(2*sctl::const_pi<Real>()*j/NFP);
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
  const sctl::Long skip_tor = (sctl::Long)std::ceil(Nt0/(Real)Nt1);
  const sctl::Long skip_pol = (sctl::Long)std::ceil(Np0/(Real)Np1);
  const sctl::Long dof = X0.Dim() / (Nt0 * Np0);
  SCTL_ASSERT(X0.Dim() == dof*Nt0*Np0);

  sctl::Vector<Real> XX;
  biest::SurfaceOp<Real>::Upsample(X0, Nt0, Np0, XX, Nt1*skip_tor, Np1*skip_pol);

  if (X1.Dim() != dof * Nt1*Np1) X1.ReInit(dof * Nt1*Np1);
  for (sctl::Long k = 0; k < dof; k++) {
    for (sctl::Long i = 0; i < Nt1; i++) {
      for (sctl::Long j = 0; j < Np1; j++) {
        X1[(k*Nt1+i)*Np1+j] = XX[((k*Nt1+i)*skip_tor*Np1+j)*skip_pol];
      }
    }
  }
}

}

