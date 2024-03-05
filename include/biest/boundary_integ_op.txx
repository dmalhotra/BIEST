#include <biest/singular_correction.hpp>

namespace biest {

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::BoundaryIntegralOp(const sctl::Comm& comm) : comm_(comm) {
      dim[0] = 0;
      dim[1] = 0;
      ker_ptr = nullptr;
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> sctl::Long BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::Dim(sctl::Integer i) const { return dim[i]; }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::SetupHelper(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_) {
      dim[0] = 0;
      dim[1] = 0;
      if (!Svec.Dim()) return;

      ker_ptr = &ker;
      sctl::Long Nsurf = Svec.Dim();
      Nt0   .ReInit(Nsurf);
      Np0   .ReInit(Nsurf);
      Op    .ReInit(Nsurf);
      Xtrg  .ReInit(Nsurf);
      Xsrc  .ReInit(Nsurf);
      dXsrc .ReInit(Nsurf);
      Xn_src.ReInit(Nsurf);
      Xa_src.ReInit(Nsurf);
      normal_orient.ReInit(Nsurf);

      trg_idx = trg_idx_;
      if (!trg_idx.Dim()) {
        trg_idx.ReInit(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) {
          const sctl::Long Ntrg = Svec[i].NTor() * Svec[i].NPol();
          trg_idx[i].ReInit(Ntrg);
          for (sctl::Long j = 0; j < Ntrg; j++) trg_idx[i][j] = j;
        }
      }
      SCTL_ASSERT(trg_idx.Dim()==Nsurf);

      for (sctl::Long i = 0; i < Nsurf; i++) {
        const auto& S = Svec[i];
        const sctl::Long Nsrc0 = S.NTor() * S.NPol();
        const sctl::Vector<Real> Xsrc0 = S.Coord();

        { // Set Xtrg
          const sctl::Long Ntrg = trg_idx[i].Dim();
          if (Xtrg[i].Dim() != Ntrg*COORD_DIM) {
            Xtrg[i].ReInit(COORD_DIM*Ntrg);
          }
          for (sctl::Long j = 0; j < Ntrg; j++) {
            const sctl::Long trg_idx_ = trg_idx[i][j];
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
              Xtrg[i][k*Ntrg+j] = Xsrc0[k*Nsrc0+trg_idx_];
            }
          }
        }

        Nt0[i] = S.NTor();
        Np0[i] = S.NPol();
        Op[i] = SurfaceOp<Real>(comm_, Nt0[i]*UPSAMPLE, Np0[i]*UPSAMPLE);
        Upsample(S.Coord(), Xsrc[i], Nt0[i], Np0[i]);
        Op[i].Grad2D(dXsrc[i], Xsrc[i]);
        normal_orient = Op[i].SurfNormalAreaElem(&Xn_src[i], &Xa_src[i], dXsrc[i], &Xsrc[i]);
        dim[0] += ker.Dim(0) * (Xsrc0.Dim() / COORD_DIM);
        dim[1] += ker.Dim(1) * (Xtrg[i].Dim() / COORD_DIM);
      }
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_) {
      SetupHelper(Svec, ker, trg_idx_);
      sctl::Long Nsurf = Svec.Dim();
      singular_correction.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        Op[i].SetupSingularCorrection(singular_correction[i], UPSAMPLE, Xsrc[i], dXsrc[i], ker, normal_orient[i], trg_idx[i]);
      }
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(ker_ptr);
      sctl::Long dof = F.Dim() / Dim(0);
      SCTL_ASSERT(F.Dim() == dof * Dim(0));
      if (U.Dim() != dof * Dim(1)) {
        U.ReInit(dof * Dim(1));
        U = 0;
      }

      sctl::Vector<Real> Fsrc;
      sctl::Long SrcOffset = 0, SingularTrgOffset = 0;
      for (sctl::Long s = 0; s < Xsrc.Dim(); s++) {
        const sctl::Vector<Real> F_(dof * ker_ptr->Dim(0) * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + dof * ker_ptr->Dim(0) * SrcOffset, false);
        Upsample(F_, Fsrc, Nt0[s], Np0[s]);

        sctl::Long TrgOffset = 0;
        for (sctl::Long t = 0; t < Xtrg.Dim(); t++) {
          sctl::Vector<Real> U_(dof * ker_ptr->Dim(1) * trg_idx[t].Dim(), U.begin() + dof * ker_ptr->Dim(1) * TrgOffset, false);
          Op[s].EvalSurfInteg(U_, Xtrg[t], Xsrc[s], Xn_src[s], Xa_src[s], Fsrc, *ker_ptr);
          SCTL_ASSERT(Xtrg[t].Dim()/COORD_DIM == trg_idx[t].Dim());
          TrgOffset += trg_idx[t].Dim();
        }

        sctl::Vector<Real> U_(dof * ker_ptr->Dim(1) * trg_idx[s].Dim(), U.begin() + dof * ker_ptr->Dim(1) * SingularTrgOffset, false);
        Op[s].EvalSingularCorrection(U_, singular_correction[s], ker_ptr->Dim(0), ker_ptr->Dim(1), Fsrc);
        SingularTrgOffset += trg_idx[s].Dim();
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real> Xt, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(ker_ptr);
      SCTL_ASSERT(F.Dim() == Dim(0));
      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
      if (U.Dim() != ker_ptr->Dim(1) * Nt) {
        U.ReInit(ker_ptr->Dim(1) * Nt);
        U = 0;
      }

      sctl::Vector<Real> Fsrc;
      sctl::Long SrcOffset = 0;
      for (sctl::Long s = 0; s < Xsrc.Dim(); s++) {
        const sctl::Vector<Real> F_(ker_ptr->Dim(0) * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + ker_ptr->Dim(0) * SrcOffset, false);
        Upsample(F_, Fsrc, Nt0[s], Np0[s]);
        Op[s].EvalSurfInteg(U, Xt, Xsrc[s], Xn_src[s], Xa_src[s], Fsrc, *ker_ptr);
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> template <class Kernel> sctl::Vector<Real> BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::test(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& ker, const sctl::Comm& comm) {
      // Parameters
      // UPSAMPLE     Upsample factor
      // PATCH_DIM    Patch dimension (has optimal value, too small/large will increase error)
      // RADIAL_DIM   Radial quadrature order
      // INTERP_ORDER Interpolation order
      // POU-function Partition-of-unity function

      sctl::Vector<Surface<Real>> Svec;
      Svec.PushBack(Surface<Real>(Nt, Np, surf_type));
      BoundaryIntegralOp integ_op(comm);

      sctl::Profile::Tic("Setup", &comm);
      integ_op.SetupSingular(Svec, ker);
      sctl::Profile::Toc();

      sctl::Vector<Real> U, Fsrc(ker.Dim(0) * Nt * Np);
      Fsrc = 1;

      sctl::Profile::Tic("Eval", &comm);
      integ_op(U, Fsrc);
      sctl::Profile::Toc();

      if (!comm.Rank()) {
        Real max_err = 0;
        for (sctl::Long i = 0; i < U.Dim(); i++) max_err = std::max<Real>(max_err, fabs(U[i]-(Real)0.5));
        std::cout<<(double)max_err<<'\n';
      }
      return U;
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> template <class Kernel, class GradKernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::test_GreensIdentity(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& sl_ker, const Kernel& dl_ker, const GradKernel& sl_grad_ker, const sctl::Comm& comm) {
      sctl::Vector<Surface<Real>> Svec;
      Svec.PushBack(Surface<Real>(Nt, Np, surf_type));
      BoundaryIntegralOp<Real, Kernel::Dim(0), Kernel::Dim(1), UPSAMPLE, PATCH_DIM0, RAD_DIM, HedgehogOrder> SL(comm);
      BoundaryIntegralOp<Real, Kernel::Dim(0), Kernel::Dim(1), UPSAMPLE, PATCH_DIM0, RAD_DIM, HedgehogOrder> DL(comm);

      sctl::Vector<Real> Fs, Fd;
      { // Set Fs, Fd
        const auto& S = Svec[0];
        const auto& Xt = S.Coord();
        sctl::Long Nt = S.NTor(), Np = S.NPol();

        sctl::Vector<Real> dX, Xn, Xa;
        SurfaceOp<Real> Op(comm, Nt, Np);
        Op.Grad2D(dX, Xt);
        Op.SurfNormalAreaElem(&Xn, &Xa, dX, &Xt);

        sctl::Vector<Real> Xsrc;
        sctl::Vector<Real> Fsrc;
        Xsrc.PushBack(1);
        Xsrc.PushBack(2);
        Xsrc.PushBack(3);
        for (sctl::Long i = 0; i < sl_ker.Dim(0); i++) Fsrc.PushBack(drand48());
        Fsrc /= max_norm(Fsrc);

        sctl::Vector<Real> U, dU;
        sl_ker(Xsrc, Xsrc, Fsrc, Xt, U);
        sl_grad_ker(Xsrc, Xsrc, Fsrc, Xt, dU);

        Fd = U;
        Fs = U * 0;
        for (sctl::Long j = 0; j < COORD_DIM; j++) {
          for (sctl::Long k = 0; k < sl_ker.Dim(0); k++) {
            for (sctl::Long i = 0; i < Nt*Np; i++) {
              Fs[k * Nt*Np + i] += Xn[j * Nt*Np + i] * dU[(j*sl_ker.Dim(0)+k) * Nt*Np + i];
            }
          }
        }
      }

      sctl::Profile::Tic("Setup", &comm);
      SL.SetupSingular(Svec, sl_ker);
      DL.SetupSingular(Svec, dl_ker);
      sctl::Profile::Toc();

      sctl::Profile::Tic("Eval", &comm);
      sctl::Vector<Real> Us, Ud;
      SL(Us, Fs);
      DL(Ud, Fd);
      sctl::Profile::Toc();

      std::cout<<"Error: "<<max_norm(Us+Ud-0.5*Fd)/max_norm(Fd)<<'\n';
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::test_Precond(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm) {
      sctl::Profile::Tic("SLayerPrecond", &comm);
      sctl::Profile::Tic("Setup", &comm);
      Surface<Real> S(Nt, Np, surf_type);
      BoundaryIntegralOp integ_op(comm);
      const auto& ker = Laplace3D<Real>::FxU();
      sctl::Vector<Surface<Real>> Svec; Svec.PushBack(S);
      integ_op.SetupSingular(Svec, ker);
      std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> SPrecond = [&integ_op](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { integ_op(Sx, x); };
      sctl::Profile::Toc();
      SurfaceOp<Real>::test_InvSurfLap(Nt, Np, surf_type, gmres_tol, gmres_iter, &SPrecond, &SPrecond, comm);
      sctl::Profile::Toc();
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::Upsample(const sctl::Vector<Real>& X0_, sctl::Vector<Real>& X_, sctl::Long Nt0, sctl::Long Np0) {
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
      sctl::Long Nt1 = UPSAMPLE * Nt0;
      sctl::Long Np1 = UPSAMPLE * Np0;

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
        for (sctl::Long t = 0; t < (Ntt + 1) / 2; t++) {
          for (sctl::Long p = 0; p < Npp; p++) {
            tmp_[((k * Nt1_ + t) * Np1_ + p) * 2 + 0] = scale * tmp0[((k * Nt0_ + t) * Np0_ + p) * 2 + 0];
            tmp_[((k * Nt1_ + t) * Np1_ + p) * 2 + 1] = scale * tmp0[((k * Nt0_ + t) * Np0_ + p) * 2 + 1];
          }
        }
        for (sctl::Long t = 0; t < Ntt / 2; t++) {
          for (sctl::Long p = 0; p < Npp; p++) {
            tmp_[((k * Nt1_ + (Nt1_ - t - 1)) * Np1_ + p) * 2 + 0] = scale * tmp0[((k * Nt0_ + (Nt0_ - t - 1)) * Np0_ + p) * 2 + 0];
            tmp_[((k * Nt1_ + (Nt1_ - t - 1)) * Np1_ + p) * 2 + 1] = scale * tmp0[((k * Nt0_ + (Nt0_ - t - 1)) * Np0_ + p) * 2 + 1];
          }
        }
      }

      if (X_.Dim() != dof  * Nt1 * Np1) X_.ReInit(dof * Nt1 * Np1);
      FFT_Helper(fft_c2r_, tmp_, X_);

      { // Floating-point correction
        sctl::Long Ut = Nt1 / Nt0;
        sctl::Long Up = Np1 / Np0;
        if (Nt1 == Nt0 * Ut && Np1 == Np0 * Up) {
          for (sctl::Long k = 0; k < dof; k++) {
            for (sctl::Long t = 0; t < Nt0; t++) {
              for (sctl::Long p = 0; p < Np0; p++) {
                X_[(k * Nt1 + t * Ut) * Np1 + p * Up] = X0_[(k * Nt0 + t) * Np0 + p];
              }
            }
          }
        }
      }
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer HedgehogOrder> Real BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,PATCH_DIM0,RAD_DIM,HedgehogOrder>::max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }



    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::FieldPeriodBIOp(const sctl::Comm& comm) : biop(nullptr), comm_(comm) {
      NFP_ = 0;
      trg_Nt_ = 0;
      trg_Np_ = 0;
      quad_Nt_ = 0;
      quad_Np_ = 0;
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::~FieldPeriodBIOp() {
      if (biop) biop_delete(&biop);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long qNt, const sctl::Long qNp) {
      SCTL_ASSERT(Svec[0].NTor() % NFP == 0);
      trg_Nt_ = trg_Nt;
      trg_Np_ = trg_Np;
      NFP_ = NFP;

      Real optim_aspect_ratio = 0, cond = 1;
      sctl::StaticArray<Real,2> aspect_ratio{100,0};
      { // Set cond
        sctl::Vector<Real> dX;
        SCTL_ASSERT(Svec.Dim() == 1);
        sctl::StaticArray<sctl::Long,2> SurfDim{Svec[0].NTor(),Svec[0].NPol()};
        biest::SurfaceOp<Real> SurfOp(comm_, SurfDim[0], SurfDim[1]);
        SurfOp.Grad2D(dX, Svec[0].Coord());

        sctl::Matrix<Real> M(2,2), U, S, Vt;
        const sctl::Long N = SurfDim[0] * SurfDim[1];
        for (sctl::Long i = 0; i < N; i++) {
          for (sctl::Integer k0 = 0; k0 < 2; k0++) {
            for (sctl::Integer k1 = 0; k1 < 2; k1++) {
              Real dot_prod = 0;
              for (sctl::Integer j = 0; j < COORD_DIM; j++) {
                dot_prod += dX[(j*2+k0)*N+i] * dX[(j*2+k1)*N+i] / SurfDim[k0] / SurfDim[k1];
              }
              M[k0][k1] = dot_prod;
            }
          }
          aspect_ratio[0] = std::min<Real>(aspect_ratio[0], sctl::sqrt<Real>(M[0][0]/M[1][1]));
          aspect_ratio[1] = std::max<Real>(aspect_ratio[1], sctl::sqrt<Real>(M[0][0]/M[1][1]));

          //M.SVD(U,S,Vt);
          //Real cond2 = std::max<Real>(S[0][0],S[1][1]) / std::min<Real>(S[0][0],S[1][1]);
          //cond = std::max<Real>(cond, sctl::sqrt<Real>(cond2));
        }
      }
      { // Set tor_upsample, cond
        optim_aspect_ratio = sctl::sqrt<Real>(aspect_ratio[0]*aspect_ratio[1]) * Svec[0].NTor() / Svec[0].NPol();
        cond = sctl::sqrt<Real>(aspect_ratio[1]/aspect_ratio[0]);
      }
      if (cond > 4) {
        SCTL_WARN("The surface mesh is highly anisotropic! Quadrature generation will be very slow. Consider using a better surface discretization.");
        std::cout<<"Mesh anisotropy = "<<cond<<'\n';
      }

      const sctl::Integer PDIM = (sctl::Integer)(digits*cond*1.6);
      if (qNt > 0 && qNp > 0) { // Set quad_Nt_, quad_Np_
        quad_Nt_ = qNt;
        quad_Np_ = qNp;
      } else {
        const sctl::Long surf_Nt = Svec[0].NTor();
        const sctl::Long surf_Np = Svec[0].NPol();

        quad_Np_ =       trg_Np_  * (sctl::Long)sctl::ceil(std::max<sctl::Long>(std::max<sctl::Long>(surf_Np, src_Np),                    2*PDIM+1) / (Real)      trg_Np_ );
        quad_Nt_ = (NFP_*trg_Nt_) * (sctl::Long)sctl::ceil(std::max<Real>((Real)std::max<sctl::Long>(surf_Nt, src_Nt), optim_aspect_ratio*quad_Np_) / (Real)(NFP_*trg_Nt_));

        for (sctl::Integer i = 0; i < 3; i++) { // adaptive refinement using double-layer test
          sctl::Long quad_Nt = (sctl::Long)sctl::ceil(quad_Nt_ / (Real)surf_Nt) * surf_Nt;
          sctl::Long quad_Np = (sctl::Long)sctl::ceil(quad_Np_ / (Real)surf_Np) * surf_Np;

          FieldPeriodBIOp<Real, COORD_DIM, 1, 1> dbl_op(comm_);
          dbl_op.SetupSingular(Svec, biest::Laplace3D<Real>::DxU(), digits, NFP_, 0, 0, surf_Nt/NFP_, surf_Np, quad_Nt, quad_Np);
          sctl::Vector<Real> U, F(quad_Nt * quad_Np); F = 1;
          dbl_op.Eval(U, F);

          Real err = 0;
          for (const auto& a : U) err = std::max<Real>(err, sctl::fabs<Real>(a-(Real)0.5));
          Real scal = std::max<Real>(1, (digits+1)/(sctl::log(err)/sctl::log((Real)0.1))); // assuming exponential/geometric convergence
          quad_Nt_ = (sctl::Long)(scal * quad_Nt);
          quad_Np_ = (sctl::Long)(scal * quad_Np);
          if (err < sctl::pow<Real>((Real)0.1,digits) || scal < 1.5) break;
        }

        // quad_Nt_/quad_Np_ ~ optim_aspect_ratio
        quad_Np_ =       trg_Np_  * (sctl::Long)sctl::round((quad_Nt_/optim_aspect_ratio) / (Real)      trg_Np_ );
        quad_Nt_ = (NFP_*trg_Nt_) * (sctl::Long)sctl::round((optim_aspect_ratio*quad_Np_) / (Real)(NFP_*trg_Nt_));
      }
      if (!(quad_Np_ > 2*PDIM  ) ||
          !(quad_Nt_ >= Svec[0].NTor()) ||
          !(quad_Np_ >= Svec[0].NPol()) ||
          !(quad_Nt_ >= src_Nt ) ||
          !(quad_Np_ >= src_Np ) ||
          !((quad_Nt_/(NFP_*trg_Nt_))*(NFP_*trg_Nt_) == quad_Nt_) ||
          !((quad_Np_/      trg_Np_ )*      trg_Np_  == quad_Np_)) {
        std::cout<<"digits             = "<<digits            <<'\n';
        std::cout<<"cond               = "<<cond              <<'\n';
        std::cout<<"PDIM               = "<<PDIM              <<'\n';
        std::cout<<"NFP_               = "<<NFP_              <<'\n';
        std::cout<<"surf_Nt            = "<<Svec[0].NTor()    <<'\n';
        std::cout<<"surf_Np            = "<<Svec[0].NPol()    <<'\n';
        std::cout<<"src_Nt             = "<<src_Nt            <<'\n';
        std::cout<<"src_Np             = "<<src_Np            <<'\n';
        std::cout<<"trg_Nt_            = "<<trg_Nt_           <<'\n';
        std::cout<<"trg_Np_            = "<<trg_Np_           <<'\n';
        std::cout<<"quad_Nt_           = "<<quad_Nt_          <<'\n';
        std::cout<<"quad_Np_           = "<<quad_Np_          <<'\n';
        std::cout<<"optim_aspect_ratio = "<<optim_aspect_ratio<<'\n';
        SCTL_ASSERT(false);
      }
      { // Setup singular
        sctl::Vector<Real> XX;
        SurfaceOp<Real>::Resample(XX, quad_Nt_, quad_Np_, Svec[0].Coord(), Svec[0].NTor(), Svec[0].NPol());

        Svec_.ReInit(1);
        Svec_[0] = biest::Surface<Real>(quad_Nt_, quad_Np_, biest::SurfType::None);
        Svec_[0].Coord() = XX;

        sctl::Vector<sctl::Vector<sctl::Long>> trg_idx(1);
        trg_idx[0].ReInit(trg_Nt_*trg_Np_);
        const sctl::Long skip_Nt = quad_Nt_ / (NFP_ * trg_Nt_);
        const sctl::Long skip_Np = quad_Np_ / trg_Np_;
        for (sctl::Long i = 0; i < trg_Nt_; i++) {
          for (sctl::Long j = 0; j < trg_Np_; j++) {
            trg_idx[0][i*trg_Np_+j] = (i*skip_Nt*trg_Np_+j)*skip_Np;
            SCTL_ASSERT(trg_idx[0][i*trg_Np_+j] < quad_Nt_*quad_Np_);
          }
        }
        SetupSingular_(Svec_, ker, PDIM, trg_idx);
      }
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(F.Dim() == KDIM0 * quad_Nt_ * quad_Np_);
      biop_eval(U, F, biop);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <sctl::Integer PDIM, sctl::Integer RDIM> void* FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::BIOpBuild(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Comm& comm, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx) {
      using BIOp = biest::BoundaryIntegralOp<Real,KDIM0,KDIM1,1,PDIM,RDIM,HedgehogOrder>;
      BIOp* biop = new BIOp(comm);
      biop[0].SetupSingular(Svec, ker, trg_idx);
      return biop;
    }
    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <sctl::Integer PDIM, sctl::Integer RDIM> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::BIOpDelete(void** self) {
      using BIOp = biest::BoundaryIntegralOp<Real,KDIM0,KDIM1,1,PDIM,RDIM,HedgehogOrder>;
      delete (BIOp*)self[0];
      self[0] = nullptr;
    }
    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <sctl::Integer PDIM, sctl::Integer RDIM> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::BIOpEval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, void* self) {
      using BIOp = biest::BoundaryIntegralOp<Real,KDIM0,KDIM1,1,PDIM,RDIM,HedgehogOrder>;
      ((BIOp*)self)[0](U, F);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <sctl::Integer PDIM, sctl::Integer RDIM> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular0(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx) {
      if (biop) biop_delete(&biop);

      biop_build = BIOpBuild<PDIM,RDIM>;
      biop_delete = BIOpDelete<PDIM,RDIM>;
      biop_eval = BIOpEval<PDIM,RDIM>;

      biop = biop_build(Svec, ker, comm_, trg_idx);
    }
    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular_(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer PDIM_, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx) {
      SCTL_ASSERT(Svec.Dim() == 1);
      if (PDIM_ >= 64) {
        static constexpr sctl::Integer PDIM = 64, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 60) {
        static constexpr sctl::Integer PDIM = 60, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 56) {
        static constexpr sctl::Integer PDIM = 56, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 52) {
        static constexpr sctl::Integer PDIM = 52, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 48) {
        static constexpr sctl::Integer PDIM = 48, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 44) {
        static constexpr sctl::Integer PDIM = 44, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 40) {
        static constexpr sctl::Integer PDIM = 40, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 36) {
        static constexpr sctl::Integer PDIM = 36, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 32) {
        static constexpr sctl::Integer PDIM = 32, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 28) {
        static constexpr sctl::Integer PDIM = 28, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 24) {
        static constexpr sctl::Integer PDIM = 24, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 20) {
        static constexpr sctl::Integer PDIM = 20, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 16) {
        static constexpr sctl::Integer PDIM = 16, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >= 12) {
        static constexpr sctl::Integer PDIM = 12, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else if (PDIM_ >=  8) {
        static constexpr sctl::Integer PDIM =  8, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      } else {
        static constexpr sctl::Integer PDIM =  6, RDIM = (sctl::Integer)(PDIM*1.6);
        SetupSingular0<PDIM,RDIM>(Svec, ker, trg_idx);
      }
    }

}
