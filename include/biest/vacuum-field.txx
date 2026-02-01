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
     *
     * \param[in] max_Nt restrict max order in toroidal direction to NFP * max_Nt.
     *
     * \param[in] max_Np restrict max order in poloidal direction to max_Np.
     */
    std::vector<Real> EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const std::vector<Real>& J, const Real I0, const sctl::Long max_Nt, const sctl::Long max_Np) const;

    private:

    static void DotProd(sctl::Vector<Real>& AdotB, const sctl::Vector<Real>& A, const sctl::Vector<Real>& B);

    mutable FieldPeriodBIOp<Real,COORD_DIM,1,3> LaplaceFxdU;
    mutable FieldPeriodBIOp<Real,COORD_DIM,1,1> LaplaceFxU;
    sctl::Vector<Surface<Real>> Svec;
    sctl::Integer NFP_, digits_;
    sctl::Long Nt_, Np_;
    bool verbose_;
    mutable sctl::Long quad_Nt_, quad_Np_;
    mutable sctl::Vector<Real> XX, normal_, dX_, Xt_, Xp_, J0_; // NFP_ * Nt_ * Np_
    mutable sctl::Vector<Real> normal; // Nt_ * Np_
    mutable bool dosetup, dosetup_LaplaceFxU;
  };


  template <class Real, bool exterior> VacuumFieldBase<Real,exterior>::VacuumFieldBase(bool verbose) : LaplaceFxdU(sctl::Comm::Self()), LaplaceFxU(sctl::Comm::Self()), Svec(1), NFP_(0), digits_(10), Nt_(0), Np_(0), verbose_(verbose), dosetup(true), dosetup_LaplaceFxU(true) {
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
    if (half_period) { // upsample surf_Nt by 1
      sctl::Vector<Real> X0, X1;
      Svec[0] = Surface<Real>(NFP*2*(surf_Nt+1), surf_Np, SurfType::None);
      SurfaceOp<Real>::CompleteVecField(X0, true, half_period, NFP, surf_Nt, surf_Np, sctl::Vector<Real>(X), -sctl::const_pi<Real>()/(NFP*surf_Nt*2));
      SurfaceOp<Real>::Resample(X1, NFP*2*(surf_Nt+1), surf_Np, X0, NFP*2*surf_Nt, surf_Np);
      SurfaceOp<Real>::RotateToroidal(Svec[0].Coord(), X1, NFP*2*(surf_Nt+1), surf_Np, sctl::const_pi<Real>()/(NFP*Nt*2));
    } else {
      Svec[0] = Surface<Real>(NFP*surf_Nt, surf_Np, SurfType::None);
      SurfaceOp<Real>::CompleteVecField(Svec[0].Coord(), true, half_period, NFP, surf_Nt, surf_Np, sctl::Vector<Real>(X), (Real)0);
    }
    { // Set normal_, dX_, Xt_, normal, Xp
      SurfaceOp<Real>::Resample(XX, NFP_*Nt_, Np_, Svec[0].Coord(), Svec[0].NTor(), Svec[0].NPol());

      SurfaceOp<Real> SurfOp(sctl::Comm::Self(), NFP_*Nt_, Np_);
      SurfOp.Grad2D(dX_, XX);
      SurfOp.SurfNormalAreaElem(&normal_, nullptr, dX_, &XX);

      Xt_.ReInit(COORD_DIM*NFP_*Nt_*Np_);
      Xp_.ReInit(COORD_DIM*NFP_*Nt_*Np_);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) { // Set Xt_, Xp_
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
    }
  }

  template <class Real, bool exterior> std::vector<Real> VacuumFieldBase<Real,exterior>::ComputeBdotN(const std::vector<Real>& B) const {
    SCTL_ASSERT((sctl::Long)B.size() == COORD_DIM * Nt_ * Np_);

    sctl::Vector<Real> BdotN;
    DotProd(BdotN, sctl::Vector<Real>(B), normal);

    std::vector<Real> BdotN_;
    BdotN_.assign(BdotN.begin(), BdotN.end());
    return BdotN_;
  }

  template <class Real, bool exterior> std::tuple<std::vector<Real>,std::vector<Real>,std::vector<Real>> VacuumFieldBase<Real,exterior>::ComputeB_(const std::vector<Real>& B1_dot_N, const Real I0) const {
    SCTL_ASSERT((sctl::Long)B1_dot_N.size() == Nt_*Np_);
    constexpr bool half_period = false;
    const sctl::Integer max_iter = 200;

    if (dosetup) { // Quadrature setup
      LaplaceFxdU.SetupSingular(Svec, Laplace3D<Real>::FxdU(), digits_, NFP_*(half_period?2:1), NFP_*(half_period?2:1)*Nt_, Np_, Nt_, Np_);
      quad_Nt_ = LaplaceFxdU.QuadNt();
      quad_Np_ = LaplaceFxdU.QuadNp();
      dosetup = false;
    }

    auto LinOp = [this](sctl::Vector<Real>* Ax, const sctl::Vector<Real>& x) {
      sctl::Vector<Real> sigma_, sigma__, grad_phi;
      SurfaceOp<Real>::CompleteVecField(sigma_, false, half_period, NFP_, Nt_, Np_, x);
      SurfaceOp<Real>::Resample(sigma__, quad_Nt_, quad_Np_, sigma_, NFP_*Nt_, Np_);
      LaplaceFxdU.Eval(grad_phi, sigma__);

      DotProd(*Ax, grad_phi, normal);
      if (exterior) (*Ax) -= x*0.5; // potential in exterior of torus
      else (*Ax) += x*0.5; // potential in interior of torus
    };

    sctl::Vector<Real> B0, B0_dot_N; // field from current I0
    B0_dot_N.ReInit(Nt_ * Np_); B0_dot_N = 0;
    B0.ReInit(COORD_DIM * Nt_ * Np_); B0 = 0;
    if (I0 != 0) { // Set B0, B0_dot_N
      if (exterior) {
        if (J0_.Dim() == 0) { // Compute J0 (toroidal surface current of unit magnitude)
          sctl::Vector<Real> Vn, Vd, Vc;
          SurfaceOp<Real> surf_op(sctl::Comm::Self(), NFP_*Nt_, Np_);
          surf_op.HodgeDecomp(Vn, Vd, Vc, J0_, Xt_, dX_, normal_, sctl::pow<Real>(0.1,digits_), max_iter);

          { // normalize J0_ (the current is oriented in the direction of Xt_)
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

        sctl::Vector<Real> J, gradG_Jk(COORD_DIM * Nt_ * Np_);
        SurfaceOp<Real>::Resample(J, quad_Nt_, quad_Np_, J0_*I0, NFP_*Nt_, Np_);
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          gradG_Jk = 0;
          LaplaceFxdU.Eval(gradG_Jk, sctl::Vector<Real>(quad_Nt_*quad_Np_, J.begin() + k*quad_Nt_*quad_Np_, false));

          sctl::Integer k1 = (k+1)%COORD_DIM;
          sctl::Integer k2 = (k+2)%COORD_DIM;
          for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
            B0[k2 * Nt_*Np_ + i] -= gradG_Jk[k1 * Nt_*Np_ + i] - 0.5*I0 * J0_[k * NFP_*Nt_*Np_ + i] * normal[k1 * Nt_*Np_ + i];
            B0[k1 * Nt_*Np_ + i] += gradG_Jk[k2 * Nt_*Np_ + i] - 0.5*I0 * J0_[k * NFP_*Nt_*Np_ + i] * normal[k2 * Nt_*Np_ + i];
          }
        }
      } else {
        const Real scal = I0/(2*sctl::const_pi<Real>());
        for (sctl::Long i = 0; i < Nt_ * Np_; i++) {
          const Real R2inv = 1/(XX[0 * Nt_*Np_ + i]*XX[0 * Nt_*Np_ + i] + XX[1 * Nt_*Np_ + i]*XX[1 * Nt_*Np_ + i]);
          B0[0 * Nt_*Np_ + i] = -XX[1 * Nt_*Np_ + i] * scal * R2inv; // Assuming axis is through the origin along Z
          B0[1 * Nt_*Np_ + i] =  XX[0 * Nt_*Np_ + i] * scal * R2inv;
          B0[2 * Nt_*Np_ + i] = 0;
        }
      }
      DotProd(B0_dot_N, B0, normal);
    }

    sctl::Vector<Real> sigma, grad_phi;
    { // Solve for sigma
      sctl::GMRES<Real> solver(sctl::Comm::Self(), verbose_);
      solver(&sigma, LinOp, sctl::Vector<Real>(B1_dot_N) + B0_dot_N, sctl::pow<Real>(0.1,digits_), max_iter);
    }
    { // Compute B0 <-- B0 - (LaplaceFxdU[sigma] - 0.5*sigma*normal)
      sctl::Vector<Real> sigma_, sigma__;
      SurfaceOp<Real>::CompleteVecField(sigma_, false, half_period, NFP_, Nt_, Np_, sigma);
      SurfaceOp<Real>::Resample(sigma__, quad_Nt_, quad_Np_, sigma_, NFP_*Nt_, Np_);
      LaplaceFxdU.Eval(grad_phi, sigma__);

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

    if (dosetup_LaplaceFxU) { // Quadrature setup
      LaplaceFxU.SetupSingular(Svec, Laplace3D<Real>::FxU(), digits_, NFP_*(half_period?2:1), NFP_*(half_period?2:1)*Nt_, Np_, Nt_, Np_);
      quad_Nt_ = LaplaceFxU.QuadNt();
      quad_Np_ = LaplaceFxU.QuadNp();
      dosetup_LaplaceFxU = false;
    }

    sctl::Vector<Real> U0;
    { // Compute U0 <-- LaplaceFxU[sigma]
      sctl::Vector<Real> sigma_, sigma__;
      SurfaceOp<Real>::CompleteVecField(sigma_, false, half_period, NFP_, Nt_, Np_, sctl::Vector<Real>(sigma));
      SurfaceOp<Real>::Resample(sigma__, quad_Nt_, quad_Np_, sigma_, NFP_*Nt_, Np_);
      LaplaceFxU.Eval(U0, sigma__);
    }

    std::vector<Real> U0_(Nt_*Np_);
    U0_.assign(U0.begin(), U0.end());
    return U0_;
  }

  template <class Real, bool exterior> std::vector<Real> VacuumFieldBase<Real,exterior>::EvalOffSurface(const std::vector<Real>& Xtrg, const std::vector<Real>& sigma, const std::vector<Real>& J, const Real I0, const sctl::Long max_Nt, const sctl::Long max_Np) const {
    constexpr bool half_period = false;
    const sctl::Long Ntrg = Xtrg.size() / COORD_DIM;
    SCTL_ASSERT((sctl::Long)Xtrg.size() == COORD_DIM * Ntrg);
    SCTL_ASSERT((sctl::Long)sigma.size() == 0 || (sctl::Long)sigma.size() == Nt_ * Np_);
    SCTL_ASSERT(J.size() == 0 || (sctl::Long)J.size() == COORD_DIM * Nt_ * Np_);

    sctl::Vector<Real> Xtrg_(Xtrg);
    sctl::Long Nt = Nt_, Np = Np_;
    sctl::Vector<Surface<Real>> Svec(1);
    while (1) {
      Svec[0] = Surface<Real>(NFP_*Nt, Np);
      SurfaceOp<Real>::Resample(Svec[0].Coord(), NFP_*Nt, Np, XX, NFP_*Nt_, Np_);

      sctl::Vector<Real> F(NFP_*Nt*Np), U; F = 1;
      BoundaryIntegralOp<Real,1,1,1> LaplaceDxU_(sctl::Comm::Self());
      LaplaceDxU_.SetupSingular(Svec, Laplace3D<Real>::DxU(), sctl::Vector<sctl::Vector<sctl::Long>>(Svec.Dim()));
      LaplaceDxU_.EvalOffSurface(U, Xtrg_, F);

      Real err = 0;
      for (const auto& x : U) err = std::max<Real>(err, std::min(fabs(1.0-x), fabs(x)));
      if (err > sctl::pow((Real)0.1, digits_)) {
        Nt *= 2; // TODO: Upsample Nt and Np independently
        Np *= 2;
      } else break;

      if ((max_Nt > 0 && Nt > max_Nt) || (max_Np > 0 && Np > max_Np)) {
        if (max_Nt > 0) Nt = std::min(Nt, max_Nt);
        if (max_Np > 0) Np = std::min(Np, max_Np);

        Svec[0] = Surface<Real>(NFP_*Nt, Np);
        SurfaceOp<Real>::Resample(Svec[0].Coord(), NFP_*Nt, Np, XX, NFP_*Nt_, Np_);
        break;
      }
    }

    BoundaryIntegralOp<Real,1,3,1> LaplaceFxdU_(sctl::Comm::Self());
    BoundaryIntegralOp<Real,3,3,1> BiotSavartFxU_(sctl::Comm::Self());
    LaplaceFxdU_.SetupSingular(Svec, Laplace3D<Real>::FxdU(), sctl::Vector<sctl::Vector<sctl::Long>>(Svec.Dim()));
    BiotSavartFxU_.SetupSingular(Svec, BiotSavart3D<Real>::FxU(), sctl::Vector<sctl::Vector<sctl::Long>>(Svec.Dim()));

    sctl::Vector<Real> sigma_, J_;
    { // Set sigma_, J_;
      sctl::Vector<Real> tmp;
      SurfaceOp<Real>::CompleteVecField(tmp, false, half_period, NFP_, Nt_, Np_, sctl::Vector<Real>(sigma));
      SurfaceOp<Real>::Resample(sigma_, NFP_*Nt, Np, tmp, NFP_*Nt_, Np_);

      SurfaceOp<Real>::CompleteVecField(tmp, true, half_period, NFP_, Nt_, Np_, sctl::Vector<Real>(J));
      SurfaceOp<Real>::Resample(J_, NFP_*Nt, Np, tmp, NFP_*Nt_, Np_);
    }

    sctl::Vector<Real> Bplasma(COORD_DIM * Ntrg); Bplasma.SetZero();
    if (sigma_.Dim()) LaplaceFxdU_.EvalOffSurface(Bplasma, Xtrg_, sigma_*(Real)-1);
    if (J_.Dim()) BiotSavartFxU_.EvalOffSurface(Bplasma, Xtrg_, J_);

    if (I0 != 0) {
      const Real scal = I0/(2*sctl::const_pi<Real>());
      for (sctl::Long i = 0; i < Ntrg; i++) {
        const Real R2inv = 1/(Xtrg[0 * Ntrg + i]*Xtrg[0 * Ntrg + i] + Xtrg[1 * Ntrg + i]*Xtrg[1 * Ntrg + i]);
        Bplasma[0 * Ntrg + i] += -Xtrg[1 * Ntrg + i] * scal * R2inv; // Assuming axis is through the origin along Z
        Bplasma[1 * Ntrg + i] +=  Xtrg[0 * Ntrg + i] * scal * R2inv;
        //Bplasma[2 * Ntrg + i] += 0;
      }
    }

    std::vector<Real> B;
    B.assign(Bplasma.begin(), Bplasma.end());
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

  template <class Real> std::vector<Real> ExtVacuumField<Real>::EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const std::vector<Real>& J, const sctl::Long max_Nt, const sctl::Long max_Np) const {
    return VacuumFieldBase<Real,true>::EvalOffSurface(Xt, sigma, J, 0, max_Nt, max_Np);
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

  template <class Real> std::vector<Real> IntVacuumField<Real>::EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const Real I0, const sctl::Long max_Nt, const sctl::Long max_Np) const {
    return VacuumFieldBase<Real,false>::EvalOffSurface(Xt, sigma, std::vector<Real>(), I0, max_Nt, max_Np);
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
      SurfaceOp<Real>::CompleteVecField(XX, true, half_period, NFP, surf_Nt, surf_Np, sctl::Vector<Real>(X), (half_period?-sctl::const_pi<Real>()/(NFP*surf_Nt*2):0));
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
      WriteVTK("B", NFP, half_period, surf_Nt, surf_Np, sctl::Vector<Real>(X), trg_Nt, trg_Np, sctl::Vector<Real>(B));
      WriteVTK_("loop0", source0, density0);
    }

    return std::make_tuple(std::move(B), std::move(B_pts));;
  }

}
