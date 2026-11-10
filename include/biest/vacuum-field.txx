// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <biest/surface_op.hpp>

namespace biest {

  template <class Real, bool exterior> class VacuumFieldBase {
    static constexpr sctl::Integer COORD_DIM = 3;

    public:

    explicit VacuumFieldBase(bool verbose = false);

    protected:

    void Setup_(const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np);

    std::vector<Real> ComputeBdotN(const std::vector<Real>& B) const;

    /**
     * Construct a magnetic field B0 with with prescribed B0.n + B1_dot_N = 0 and circulation equal to I0.
     */
    std::tuple<std::vector<Real>,std::vector<Real>,std::vector<Real>> ComputeB_(const std::vector<Real>& B1_dot_N, const Real I0) const;

    std::vector<Real> ComputeU_(const std::vector<Real>& sigma) const;

    /**
     * Evaluate the magnetic field at off-surface target points.
     *
     * \param[in] Xt target coordinates.
     *
     * \param[in] sigma charge density on the surface.
     *
     * \param[in] J current density on the surface.
     *
     * \param[in] I0 current through the axis.
     */
    std::vector<Real> EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const std::vector<Real>& J, const Real I0) const;

    private:

    static void DotProd(sctl::Vector<Real>& AdotB, const sctl::Vector<Real>& A, const sctl::Vector<Real>& B);

    using OffSurfLaplaceFxdU = decltype(OffSurfaceOp(Laplace3D<Real>::FxdU()));
    using OffSurfBiotSavartFxU = decltype(OffSurfaceOp(BiotSavart3D<Real>::FxU()));

    mutable FieldPeriodBIOp<Real> LaplaceFxdU;
    mutable FieldPeriodBIOp<Real> LaplaceFxU;
    sctl::Vector<Surface<Real>> Svec;
    sctl::Integer NFP_, digits_;
    sctl::Long Nt_, Np_;
    sctl::Integer tor_orient_; // ToroidalOrientation() of the surface
    bool verbose_;
    mutable sctl::Vector<Real> XX, normal_, dX_, Xt_, Xp_, J0_; // NFP_ * Nt_ * Np_
    mutable sctl::Vector<Real> normal, B0J0_; // Nt_ * Np_: B0J0_ the field of the current J0_ (I0 = 1), once computed
    mutable bool dosetup, dosetup_LaplaceFxU;
    // The operators of EvalOffSurface(), with their panels fit to the surface and the densities of the first call after
    // Setup_()
    mutable std::unique_ptr<OffSurfLaplaceFxdU> off_lap_;
    mutable std::unique_ptr<OffSurfBiotSavartFxU> off_bs_;
  };


  template <class Real, bool exterior> VacuumFieldBase<Real,exterior>::VacuumFieldBase(bool verbose) : LaplaceFxdU(sctl::Comm::Self()), LaplaceFxU(sctl::Comm::Self()), Svec(1), NFP_(0), digits_(10), Nt_(0), Np_(0), tor_orient_(1), verbose_(verbose), dosetup(true), dosetup_LaplaceFxU(true) {
  }

  template <class Real, bool exterior> void VacuumFieldBase<Real,exterior>::Setup_(const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np) {
    bool half_period = false;
    digits_ = digits;
    NFP_ = NFP;
    Nt_ = Nt;
    Np_ = Np;

    dosetup = true;
    dosetup_LaplaceFxU = true;
    SCTL_ASSERT(surf_Nt*surf_Np*COORD_DIM == (sctl::Long)X.size());
    const sctl::Vector<Real> X_view((sctl::Long)X.size(), sctl::Ptr2Itr<Real>((Real*)X.data(), (sctl::Long)X.size()), false);
    tor_orient_ = SurfaceOp<Real>::ToroidalOrientation(X_view, surf_Nt, surf_Np);
    if (half_period) { // upsample surf_Nt by 1
      sctl::Vector<Real> X0, X1;
      Svec[0] = Surface<Real>(NFP*2*(surf_Nt+1), surf_Np, SurfType::None);
      SurfaceOp<Real>::CompleteVecField(X0, half_period, NFP, surf_Nt, surf_Np, X_view, Parity::Even, tor_orient_, -sctl::const_pi<Real>()/(NFP*surf_Nt*2));
      SurfaceOp<Real>::Resample(X1, NFP*2*(surf_Nt+1), surf_Np, X0, NFP*2*surf_Nt, surf_Np);
      SurfaceOp<Real>::RotateToroidal(Svec[0].Coord(), X1, NFP*2*(surf_Nt+1), surf_Np, sctl::const_pi<Real>()/(NFP*Nt*2));
    } else {
      Svec[0] = Surface<Real>(NFP*surf_Nt, surf_Np, SurfType::None);
      SurfaceOp<Real>::CompleteVecField(Svec[0].Coord(), half_period, NFP, surf_Nt, surf_Np, X_view, Parity::Even, tor_orient_, (Real)0);
    }
    { // Set normal_, dX_, Xt_, normal, Xp
      SurfaceOp<Real>::Resample(XX, NFP_*Nt_, Np_, Svec[0].Coord(), Svec[0].NTor(), Svec[0].NPol());

      SurfaceOp<Real> SurfOp(sctl::Comm::Self(), NFP_*Nt_, Np_);
      SurfOp.Grad2D(dX_, XX);
      SurfOp.SurfNormalAreaElem(&normal_, nullptr, dX_, &XX);

      Xt_.ReInit(COORD_DIM*NFP_*Nt_*Np_);
      Xp_.ReInit(COORD_DIM*NFP_*Nt_*Np_);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) { // Set Xt_, Xp_: -dX/dt and -dX/dp (see SurfaceOp::Grad2D())
        for (sctl::Long i = 0; i < NFP_*Nt_*Np_; i++) {
          Xt_[k*NFP_*Nt_*Np_+i] = dX_[(k*2+0)*NFP_*Nt_*Np_+i];
          Xp_[k*NFP_*Nt_*Np_+i] = dX_[(k*2+1)*NFP_*Nt_*Np_+i];
        }
      }

      normal.ReInit(COORD_DIM*Nt_*Np_);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) { // Set normal
        for (sctl::Long i = 0; i < Nt_*Np_; i++) {
          normal[k*Nt_*Np_+i] = normal_[k*NFP_*Nt_*Np_+i];
        }
      }

      J0_.ReInit(0);
      B0J0_.ReInit(0);
      off_lap_.reset();
      off_bs_.reset();
    }
  }

  template <class Real, bool exterior> std::vector<Real> VacuumFieldBase<Real,exterior>::ComputeBdotN(const std::vector<Real>& B) const {
    SCTL_ASSERT((sctl::Long)B.size() == COORD_DIM * Nt_ * Np_);

    std::vector<Real> BdotN_(Nt_ * Np_);
    sctl::Vector<Real> BdotN((sctl::Long)BdotN_.size(), sctl::Ptr2Itr<Real>(BdotN_.data(), (sctl::Long)BdotN_.size()), false);
    const sctl::Vector<Real> B_view((sctl::Long)B.size(), sctl::Ptr2Itr<Real>((Real*)B.data(), (sctl::Long)B.size()), false);
    DotProd(BdotN, B_view, normal);
    return BdotN_;
  }

  template <class Real, bool exterior> std::tuple<std::vector<Real>,std::vector<Real>,std::vector<Real>> VacuumFieldBase<Real,exterior>::ComputeB_(const std::vector<Real>& B1_dot_N, const Real I0) const {
    SCTL_ASSERT((sctl::Long)B1_dot_N.size() == Nt_*Np_);
    constexpr bool half_period = false;
    const sctl::Integer max_iter = 200;

    if (dosetup) { // Quadrature setup, for a grid that resolves the right-hand side B1_dot_N too
      sctl::ScratchBuf<Real> R_buf(NFP_ * Nt_ * Np_);
      sctl::Vector<Real> R(R_buf);
      SurfaceOp<Real>::CompleteVecField(R, half_period, NFP_, Nt_, Np_, sctl::Vector<Real>((sctl::Long)B1_dot_N.size(), sctl::Ptr2Itr<Real>((Real*)B1_dot_N.data(), (sctl::Long)B1_dot_N.size()), false), Parity::Odd, tor_orient_);
      LaplaceFxdU.SetupSingular(Svec[0], Laplace3D<Real>::FxdU(), digits_, NFP_*(half_period?2:1), Nt_, Np_, -1, -1, R);
      dosetup = false;
    }

    // The temporaries in scratch memory (see SurfaceOp::SurfGrad); LaplaceFxdU.Eval() completes a density on one field
    // period to the full torus, and resamples it to the quadrature grid, in scratch memory as well
    const sctl::Long N = Nt_ * Np_;
    auto LinOp = [this,N](sctl::Vector<Real>* Ax, const sctl::Vector<Real>& x) {
      sctl::ScratchBuf<Real> grad_phi_buf(COORD_DIM * N);
      sctl::Vector<Real> grad_phi(grad_phi_buf);
      grad_phi.SetZero();
      LaplaceFxdU.Eval(grad_phi, x, NFP_*(half_period?2:1), Nt_, Np_);

      DotProd(*Ax, grad_phi, normal);
      const Real s = (exterior ? (Real)-0.5 : (Real)0.5); // potential in the exterior (interior) of the torus
      for (sctl::Long i = 0; i < N; i++) (*Ax)[i] += s * x[i];
    };

    sctl::ScratchBuf<Real> B0_buf(COORD_DIM * N), rhs_buf(N), sigma_buf(N), grad_phi_buf(COORD_DIM * N);
    sctl::Vector<Real> B0(B0_buf), rhs(rhs_buf), sigma(sigma_buf), grad_phi(grad_phi_buf); // B0: field from current I0
    B0.SetZero();
    if (I0 != 0) { // Set B0
      if (exterior) {
        if (B0J0_.Dim() == 0) { // Set B0J0_, the field of the current J0_, and J0_ first if needed
          if (J0_.Dim() == 0) { // Compute J0 (toroidal surface current of unit magnitude)
            sctl::ScratchBuf<Real> Vn_buf(COORD_DIM*NFP_*Nt_*Np_), Vd_buf(COORD_DIM*NFP_*Nt_*Np_), Vc_buf(COORD_DIM*NFP_*Nt_*Np_);
            sctl::Vector<Real> Vn(Vn_buf), Vd(Vd_buf), Vc(Vc_buf); // the other parts of the decomposition, in scratch memory
            SurfaceOp<Real> surf_op(sctl::Comm::Self(), NFP_*Nt_, Np_);
            surf_op.HodgeDecomp(Vn, Vd, Vc, J0_, Xt_, dX_, normal_, sctl::pow<Real>(0.1,digits_), max_iter);

            { // normalize J0_: the current flows in the direction of Xt_ = -dX/dt, of decreasing toroidal index
              const sctl::Long N = NFP_*Nt_*Np_;

              Real orientation = 0;
              { // Set orientation
                Real n[3];
                n[0] = Xt_[1*N] * Xp_[2*N] - Xt_[2*N] * Xp_[1*N];
                n[1] = Xt_[2*N] * Xp_[0*N] - Xt_[0*N] * Xp_[2*N];
                n[2] = Xt_[0*N] * Xp_[1*N] - Xt_[1*N] * Xp_[0*N];
                orientation = (n[0]*normal_[0*N] + n[1]*normal_[1*N] + n[2]*normal_[2*N] > 0 ? 1.0 : -1.0);
              }

              Real Jtor_flux = 0;
              for (sctl::Long i = 0; i < N; i++) {
                for (sctl::Integer k0 = 0; k0 < COORD_DIM; k0++) {
                  sctl::Integer k1 = (k0+1)%COORD_DIM;
                  sctl::Integer k2 = (k0+2)%COORD_DIM;
                  Jtor_flux += J0_[k0*N+i] * Xp_[k1*N+i] * normal_[k2*N+i];
                  Jtor_flux -= J0_[k0*N+i] * Xp_[k2*N+i] * normal_[k1*N+i];
                }
              }
              J0_ *= (NFP_*Nt_*Np_)/Jtor_flux*orientation;
            }
          }

          // B0J0_ = curl S[J0_] - n x J0_ / 2, by components: the gradients of S[J0_ k], the components J0_ k on the full
          // torus as COORD_DIM densities of one Eval()
          B0J0_.ReInit(COORD_DIM * N);
          B0J0_.SetZero();
          sctl::ScratchBuf<Real> gradG_J_buf(COORD_DIM * COORD_DIM * N);
          sctl::Vector<Real> gradG_J(gradG_J_buf); // the gradient of S[J0_ k] at offset k * COORD_DIM * N
          gradG_J.SetZero();
          LaplaceFxdU.Eval(gradG_J, J0_, 1, NFP_*Nt_, Np_);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            sctl::Integer k1 = (k+1)%COORD_DIM;
            sctl::Integer k2 = (k+2)%COORD_DIM;
            for (sctl::Long i = 0; i < N; i++) {
              B0J0_[k2 * N + i] -= gradG_J[(k * COORD_DIM + k1) * N + i] - 0.5 * J0_[k * NFP_*Nt_*Np_ + i] * normal[k1 * N + i];
              B0J0_[k1 * N + i] += gradG_J[(k * COORD_DIM + k2) * N + i] - 0.5 * J0_[k * NFP_*Nt_*Np_ + i] * normal[k2 * N + i];
            }
          }
        }
        for (sctl::Long i = 0; i < COORD_DIM * N; i++) B0[i] = I0 * B0J0_[i];
      } else {
        const Real scal = I0/(2*sctl::const_pi<Real>());
        for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
          const Real R2inv = 1/(XX[0 * NFP_*Nt_*Np_ + i]*XX[0 * NFP_*Nt_*Np_ + i] + XX[1 * NFP_*Nt_*Np_ + i]*XX[1 * NFP_*Nt_*Np_ + i]);
          B0[0 * Nt_*Np_ + i] = -XX[1 * NFP_*Nt_*Np_ + i] * scal * R2inv; // Assuming axis is through the origin along Z
          B0[1 * Nt_*Np_ + i] =  XX[0 * NFP_*Nt_*Np_ + i] * scal * R2inv;
          B0[2 * Nt_*Np_ + i] = 0;
        }
      }
    }
    DotProd(rhs, B0, normal); // rhs <-- B1.n + B0.n
    for (sctl::Long i = 0; i < N; i++) rhs[i] += B1_dot_N[i];

    { // Solve for sigma
      sigma.SetZero();
      sctl::GMRES<Real> solver(sctl::Comm::Self(), verbose_);
      solver(&sigma, LinOp, rhs, sctl::pow<Real>(0.1,digits_), max_iter);
    }
    { // Compute B0 <-- B0 - (LaplaceFxdU[sigma] - 0.5*sigma*normal)
      grad_phi.SetZero();
      LaplaceFxdU.Eval(grad_phi, sigma, NFP_*(half_period?2:1), Nt_, Np_);

      for (sctl::Long i = 0; i < Nt_*Np_; i++) { // B0 <-- B0 - (grad_phi - 0.5*sigma*normal)
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          if (exterior) B0[k*Nt_*Np_+i] -= grad_phi[k*Nt_*Np_+i] - 0.5*sigma[i] * normal[k*Nt_*Np_+i];
          else          B0[k*Nt_*Np_+i] -= grad_phi[k*Nt_*Np_+i] + 0.5*sigma[i] * normal[k*Nt_*Np_+i];
        }
      }
    }

    std::vector<Real> B0_(COORD_DIM * Nt_*Np_), sigma_(Nt_*Np_), J0;
    B0_.assign(B0.begin(), B0.end());
    sigma_.assign(sigma.begin(), sigma.end());
    if (exterior && I0 != 0) {
      J0.resize(COORD_DIM*Nt_*Np_);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        for (sctl::Long i = 0; i < Nt_*Np_; i++) {
          J0[k*Nt_*Np_+i] = J0_[k*NFP_*Nt_*Np_+i] * I0;
        }
      }
    }
    return std::make_tuple(std::move(B0_), std::move(sigma_), std::move(J0));
  }

  template <class Real, bool exterior> std::vector<Real> VacuumFieldBase<Real,exterior>::ComputeU_(const std::vector<Real>& sigma) const {
    constexpr bool half_period = false;

    if (dosetup_LaplaceFxU) { // Quadrature setup, for a grid that resolves sigma too
      sctl::ScratchBuf<Real> R_buf(NFP_ * Nt_ * Np_);
      sctl::Vector<Real> R(R_buf);
      SurfaceOp<Real>::CompleteVecField(R, half_period, NFP_, Nt_, Np_, sctl::Vector<Real>((sctl::Long)sigma.size(), sctl::Ptr2Itr<Real>((Real*)sigma.data(), (sctl::Long)sigma.size()), false), Parity::Odd, tor_orient_);
      LaplaceFxU.SetupSingular(Svec[0], Laplace3D<Real>::FxU(), digits_, NFP_*(half_period?2:1), Nt_, Np_, -1, -1, R);
      dosetup_LaplaceFxU = false;
    }

    std::vector<Real> U0_(Nt_*Np_, (Real)0); // U0 <-- LaplaceFxU[sigma], sigma completed to the full torus and resampled by Eval()
    sctl::Vector<Real> U0((sctl::Long)U0_.size(), sctl::Ptr2Itr<Real>(U0_.data(), (sctl::Long)U0_.size()), false);
    const sctl::Vector<Real> sigma_view((sctl::Long)sigma.size(), sctl::Ptr2Itr<Real>((Real*)sigma.data(), (sctl::Long)sigma.size()), false);
    LaplaceFxU.Eval(U0, sigma_view, NFP_*(half_period?2:1), Nt_, Np_);
    return U0_;
  }

  template <class Real, bool exterior> std::vector<Real> VacuumFieldBase<Real,exterior>::EvalOffSurface(const std::vector<Real>& Xtrg, const std::vector<Real>& sigma, const std::vector<Real>& J, const Real I0) const {
    constexpr bool half_period = false;
    const sctl::Long Ntrg = Xtrg.size() / COORD_DIM;
    SCTL_ASSERT((sctl::Long)Xtrg.size() == COORD_DIM * Ntrg);
    SCTL_ASSERT((sctl::Long)sigma.size() == 0 || (sctl::Long)sigma.size() == Nt_ * Np_);
    SCTL_ASSERT(J.size() == 0 || (sctl::Long)J.size() == COORD_DIM * Nt_ * Np_);

    // The densities on the full surface, in scratch memory (see SurfaceOp::SurfGrad), and the operators of the object,
    // with their panels fit by the first call (see OffSurfaceOp::SetTargets)
    const sctl::Long N = NFP_*Nt_*Np_;
    const Real tol = sctl::pow<Real>((Real)0.1, digits_);
    std::vector<Real> B(COORD_DIM * Ntrg, (Real)0);
    sctl::Vector<Real> Bplasma((sctl::Long)B.size(), sctl::Ptr2Itr<Real>(B.data(), (sctl::Long)B.size()), false);
    const sctl::Vector<Real> Xtrg_view((sctl::Long)Xtrg.size(), sctl::Ptr2Itr<Real>((Real*)Xtrg.data(), (sctl::Long)Xtrg.size()), false);
    if (sigma.size()) {
      sctl::ScratchBuf<Real> F_buf(N);
      sctl::Vector<Real> F(F_buf);
      const sctl::Vector<Real> sigma_view((sctl::Long)sigma.size(), sctl::Ptr2Itr<Real>((Real*)sigma.data(), (sctl::Long)sigma.size()), false);
      SurfaceOp<Real>::CompleteVecField(F, half_period, NFP_, Nt_, Np_, sigma_view, Parity::Odd, tor_orient_);
      for (auto& f : F) f = -f;
      if (!off_lap_) {
        off_lap_.reset(new OffSurfLaplaceFxdU(Laplace3D<Real>::FxdU()));
        off_lap_->SetupSurface(XX, NFP_*Nt_, Np_, tol, F);
      }
      off_lap_->SetTargets(Xtrg_view);
      off_lap_->Eval(Bplasma, F);
    }
    if (J.size()) {
      sctl::ScratchBuf<Real> J_buf(COORD_DIM * N);
      sctl::Vector<Real> J_(J_buf);
      const sctl::Vector<Real> J_view((sctl::Long)J.size(), sctl::Ptr2Itr<Real>((Real*)J.data(), (sctl::Long)J.size()), false);
      SurfaceOp<Real>::CompleteVecField(J_, half_period, NFP_, Nt_, Np_, J_view, Parity::Odd, tor_orient_);
      if (!off_bs_) {
        off_bs_.reset(new OffSurfBiotSavartFxU(BiotSavart3D<Real>::FxU()));
        off_bs_->SetupSurface(XX, NFP_*Nt_, Np_, tol, J_);
      }
      off_bs_->SetTargets(Xtrg_view);
      off_bs_->Eval(Bplasma, J_);
    }

    if (I0 != 0) {
      const Real scal = I0/(2*sctl::const_pi<Real>());
      for (sctl::Long i = 0; i < Ntrg; i++) {
        const Real R2inv = 1/(Xtrg[0 * Ntrg + i]*Xtrg[0 * Ntrg + i] + Xtrg[1 * Ntrg + i]*Xtrg[1 * Ntrg + i]);
        Bplasma[0 * Ntrg + i] += -Xtrg[1 * Ntrg + i] * scal * R2inv; // Assuming axis is through the origin along Z
        Bplasma[1 * Ntrg + i] +=  Xtrg[0 * Ntrg + i] * scal * R2inv;
        //Bplasma[2 * Ntrg + i] += 0;
      }
    }
    return B;
  }

  template <class Real, bool exterior> void VacuumFieldBase<Real,exterior>::DotProd(sctl::Vector<Real>& AdotB, const sctl::Vector<Real>& A, const sctl::Vector<Real>& B) {
    sctl::Long N = A.Dim() / COORD_DIM;
    SCTL_ASSERT(A.Dim() == COORD_DIM * N);
    SCTL_ASSERT(B.Dim() == COORD_DIM * N);
    if (AdotB.Dim() != N) AdotB.ReInit(N);
    for (sctl::Long i = 0; i < N; i++) {
      Real AdotB_ = 0;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        AdotB_ += A[k*N+i] * B[k*N+i];
      }
      AdotB[i] = AdotB_;
    }
  }



  template <class Real> void ExtVacuumField<Real>::Setup(const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np) {
    VacuumFieldBase<Real,true>::Setup_(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);
  }

  template <class Real> std::vector<Real> ExtVacuumField<Real>::ComputeBdotN(const std::vector<Real>& B) const {
    return VacuumFieldBase<Real,true>::ComputeBdotN(B);
  }

  template <class Real> std::tuple<std::vector<Real>,std::vector<Real>,std::vector<Real>> ExtVacuumField<Real>::ComputeBplasma(const std::vector<Real>& Bcoil_dot_N, const Real Jplasma) const {
    return VacuumFieldBase<Real,true>::ComputeB_(Bcoil_dot_N, Jplasma);
  }

  template <class Real> std::vector<Real> ExtVacuumField<Real>::ComputeU(const std::vector<Real>& sigma) const {
    return VacuumFieldBase<Real,true>::ComputeU_(sigma);
  }

  template <class Real> std::vector<Real> ExtVacuumField<Real>::EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const std::vector<Real>& J) const {
    return VacuumFieldBase<Real,true>::EvalOffSurface(Xt, sigma, J, 0);
  }



  template <class Real> void IntVacuumField<Real>::Setup(const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np) {
    VacuumFieldBase<Real,false>::Setup_(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);
  }

  template <class Real> std::vector<Real> IntVacuumField<Real>::ComputeBdotN(const std::vector<Real>& B) const {
    return VacuumFieldBase<Real,false>::ComputeBdotN(B);
  }

  template <class Real> std::tuple<std::vector<Real>,std::vector<Real>> IntVacuumField<Real>::ComputeB(const std::vector<Real>& B1_dot_N, const Real I0) const {
    std::vector<Real> B, sigma;
    std::tie(B, sigma, std::ignore) = VacuumFieldBase<Real,false>::ComputeB_(B1_dot_N, I0);
    return std::make_tuple(std::move(B), std::move(sigma));
  }

  template <class Real> std::vector<Real> IntVacuumField<Real>::ComputeU(const std::vector<Real>& sigma) const {
    return VacuumFieldBase<Real,false>::ComputeU_(sigma);
  }

  template <class Real> std::vector<Real> IntVacuumField<Real>::EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const Real I0) const {
    return VacuumFieldBase<Real,false>::EvalOffSurface(Xt, sigma, std::vector<Real>(), I0);
  }



  template <class Real> std::vector<Real> ExtVacuumFieldTest<Real>::SurfaceCoordinates(const sctl::Integer NFP, const sctl::Long Nt, const sctl::Long Np, const SurfType surf_type) {
    const bool half_period = false;

    sctl::Vector<Real> X_;
    const sctl::Long Nt_ = (half_period?2:1)*Nt;
    Surface<Real> S(NFP*Nt_, Np, surf_type);
    SurfaceOp<Real>::RotateToroidal(X_, S.Coord(), NFP*Nt_, Np, (half_period?sctl::const_pi<Real>()/(NFP*Nt*2):0));

    std::vector<Real> X(COORD_DIM*Nt*Np);
    for (sctl::Long k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < Nt*Np; i++) {
        X[k*Nt*Np+i] = X_[k*NFP*Nt_*Np+i];
      }
    }
    return X;
  }

  template <class Real> std::tuple<std::vector<Real>,std::vector<Real>> ExtVacuumFieldTest<Real>::BFieldData(const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long trg_Nt, const sctl::Long trg_Np, const std::vector<Real>& Xt_) {
    auto eval_LaplaceGrad = [](const sctl::Vector<Real>& Xt, const sctl::Vector<sctl::Vector<Real>>& source, const sctl::Vector<sctl::Vector<Real>>& density) {
      const auto& kernel = Laplace3D<Real>::FxdU();
      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
      SCTL_ASSERT(source.Dim() == density.Dim());

      sctl::Vector<Real> B(COORD_DIM*Nt); B = 0;
      for (sctl::Long i = 0; i < source.Dim(); i++) {
        const auto& Xs = source[i];
        const auto& Fs = density[i];
        sctl::Long Ns = Xs.Dim() / COORD_DIM;
        SCTL_ASSERT(Xs.Dim() == COORD_DIM * Ns);
        SCTL_ASSERT(Fs.Dim() == Ns);
        sctl::Vector<Real> SrcNormal(COORD_DIM*Ns);
        kernel(Xs,SrcNormal,Fs, Xt,B);
      }
      return B;
    };
    auto eval_BiotSavart = [](const sctl::Vector<Real>& Xt, const sctl::Vector<sctl::Vector<Real>>& source, const sctl::Vector<sctl::Vector<Real>>& density) {
      const auto& kernel = BiotSavart3D<Real>::FxU();
      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
      SCTL_ASSERT(source.Dim() == density.Dim());

      sctl::Vector<Real> B(COORD_DIM*Nt); B = 0;
      for (sctl::Long i = 0; i < source.Dim(); i++) {
        const auto& Xs = source[i];
        const auto& Fs = density[i];
        sctl::Long Ns = Xs.Dim() / COORD_DIM;
        SCTL_ASSERT(Xs.Dim() == COORD_DIM * Ns);
        SCTL_ASSERT(Fs.Dim() == COORD_DIM * Ns);
        sctl::Vector<Real> SrcNormal(COORD_DIM*Ns);
        kernel(Xs,SrcNormal,Fs, Xt,B);
      }
      return B;
    };
    const sctl::Comm comm = sctl::Comm::Self();
    constexpr bool half_period = false;

    sctl::Vector<Real> X_surf, X_trg, X_trg_offsurf(Xt_);
    { // Set X_surf, X_trg
      sctl::Vector<Real> XX;
      const sctl::Vector<Real> X_view((sctl::Long)X.size(), sctl::Ptr2Itr<Real>((Real*)X.data(), (sctl::Long)X.size()), false);
      SurfaceOp<Real>::CompleteVecField(XX, half_period, NFP, surf_Nt, surf_Np, X_view, Parity::Even, SurfaceOp<Real>::ToroidalOrientation(X_view, surf_Nt, surf_Np), (half_period?-sctl::const_pi<Real>()/(NFP*surf_Nt*2):0));
      SurfaceOp<Real>::Resample(X_surf, NFP*(half_period?2:1)*(surf_Nt+1), surf_Np, XX, NFP*(half_period?2:1)*surf_Nt, surf_Np);

      sctl::Vector<Real> X_surf_shifted, X_trg_;
      const sctl::Long trg_Nt_ = (half_period?2:1)*trg_Nt;
      SurfaceOp<Real>::RotateToroidal(X_surf_shifted, X_surf, NFP*(half_period?2:1)*(surf_Nt+1), surf_Np, (half_period?sctl::const_pi<Real>()/(NFP*trg_Nt*2):0));
      SurfaceOp<Real>::Resample(X_trg_, NFP*trg_Nt_, trg_Np, X_surf_shifted, NFP*(half_period?2:1)*(surf_Nt+1), surf_Np);
      X_trg.ReInit(COORD_DIM*trg_Nt_*trg_Np);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        for (sctl::Long i = 0; i < trg_Nt_*trg_Np; i++) {
          X_trg[k*trg_Nt_*trg_Np+i] = X_trg_[k*NFP*trg_Nt_*trg_Np+i];
        }
      }
    }

    sctl::Vector<sctl::Vector<Real>> source0, density0, J0;
    { // Set inside sources (source0, density0)
      sctl::Long N = 20000;
      sctl::Vector<Real> X(COORD_DIM*N), F(N), J(COORD_DIM*N);
      { // Set X, F, J
        sctl::Long Nt = 100, Np = 100;
        sctl::Vector<Real> coord(COORD_DIM*Nt);
        { // Set coord
          auto S = Surface<Real>(Nt,Np, SurfType::None);
          SurfaceOp<Real>::Upsample(X_surf, NFP*(half_period?2:1)*(surf_Nt+1), surf_Np, S.Coord(), Nt, Np);

          sctl::Vector<Real> normal, dX;
          SurfaceOp<Real> SurfOp(comm, Nt, Np);
          SurfOp.Grad2D(dX, S.Coord());
          SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &S.Coord());
          S.Coord() += -2.17*normal*0.5; // TODO: automatically choose best scaling

          coord = 0;
          for (sctl::Long t = 0; t < Nt; t++) {
            for (sctl::Long p = 0; p < Np; p++) {
              coord[0*Nt+t] += S.Coord()[(0*Nt+t)*Np+p]/Np;
              coord[1*Nt+t] += S.Coord()[(1*Nt+t)*Np+p]/Np;
              coord[2*Nt+t] += S.Coord()[(2*Nt+t)*Np+p]/Np;
            }
          }
        }

        Real N_inv = 1/(Real)N;
        sctl::Vector<Real> dX_;
        SurfaceOp<Real>::Upsample(coord,Nt,1, X,N,1);
        SurfaceOp<Real> SurfOp(comm,N,1);
        SurfOp.Grad2D(dX_, X);
        SCTL_ASSERT(dX_.Dim() == COORD_DIM*2*N);
        for (sctl::Long i = 0; i < N; i++) {
          F[i] = sctl::sqrt<Real>( dX_[(0*2+0)*N+i]*dX_[(0*2+0)*N+i] + dX_[(1*2+0)*N+i]*dX_[(1*2+0)*N+i] + dX_[(2*2+0)*N+i]*dX_[(2*2+0)*N+i] ) * N_inv;
          J[0*N+i] = dX_[(0*2+0)*N+i] * N_inv;
          J[1*N+i] = dX_[(1*2+0)*N+i] * N_inv;
          J[2*N+i] = dX_[(2*2+0)*N+i] * N_inv;
        }
      }
      source0.PushBack(X);
      density0.PushBack(F);
      J0.PushBack(J);
    }
    const auto B_ = eval_LaplaceGrad(X_trg, source0, density0) + eval_BiotSavart(X_trg, source0, J0);
    const auto B_pts_ = eval_LaplaceGrad(X_trg_offsurf, source0, density0) + eval_BiotSavart(X_trg_offsurf, source0, J0);

    std::vector<Real> B(COORD_DIM*trg_Nt*trg_Np);
    for (sctl::Integer k = 0; k < COORD_DIM; k++) { // Set B
      const sctl::Long trg_Nt_ = (half_period?2:1)*trg_Nt;
      for (sctl::Long i = 0; i < trg_Nt*trg_Np; i++) {
        B[k*trg_Nt*trg_Np+i] = B_[k*trg_Nt_*trg_Np+i];
      }
    }

    std::vector<Real> B_pts(B_pts_.Dim());
    for (sctl::Long i = 0; i < B_pts_.Dim(); i++) {
      B_pts[i] = B_pts_[i];
    }

    if (0) { // Visualization
      auto WriteVTK_ = [](const std::string& fname, const sctl::Vector<sctl::Vector<Real>>& coords, const sctl::Vector<sctl::Vector<Real>>& values) {
        VTKData data;
        typedef VTKData::VTKReal VTKReal;
        auto& point_coord =data.point_coord ;
        auto& point_value =data.point_value ;
        auto& line_connect=data.line_connect;
        auto& line_offset =data.line_offset ;
        constexpr sctl::Integer COORD_DIM = VTKData::COORD_DIM;

        SCTL_ASSERT(coords.Dim() == values.Dim());
        for (sctl::Long j = 0; j < coords.Dim(); j++) { // set point_coord, line_connect
          const auto& coord = coords[j];
          const auto& value = values[j];
          sctl::Long N = coord.Dim() / COORD_DIM;
          sctl::Long dof = value.Dim() / N;
          SCTL_ASSERT(value.Dim() == dof * N);
          for (sctl::Long i = 0; i < N; i++) {
            line_connect.push_back(point_coord.size()/COORD_DIM);
            point_coord.push_back((VTKReal)coord[0*N+i]);
            point_coord.push_back((VTKReal)coord[1*N+i]);
            point_coord.push_back((VTKReal)coord[2*N+i]);
            for (sctl::Long k = 0; k < dof; k++) {
              point_value.push_back((VTKReal)value[k*N+i]);
            }
          }
          line_offset.push_back(line_connect.size());
        }
        data.WriteVTK(fname.c_str(), sctl::Comm::Self());
      };
      WriteVTK("B", NFP, half_period, surf_Nt, surf_Np, sctl::Vector<Real>(X), trg_Nt, trg_Np, sctl::Vector<Real>(B), Parity::Odd);
      WriteVTK_("loop0", source0, density0);
    }

    return std::make_tuple(std::move(B), std::move(B_pts));;
  }

}
