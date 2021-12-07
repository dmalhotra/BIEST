#ifndef _BOUNDARY_INTEG_OP_HPP_
#define _BOUNDARY_INTEG_OP_HPP_

#include <biest/surface.hpp>
#include <biest/singular_correction.hpp>
#include <sctl.hpp>

namespace biest {

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE = 1, sctl::Integer PATCH_DIM0 = 25, sctl::Integer RAD_DIM = 18> class BoundaryIntegralOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    BoundaryIntegralOp(const sctl::Comm& comm = sctl::Comm::Self()) : comm_(comm) {
      dim[0] = 0;
      dim[1] = 0;
      ker_ptr = nullptr;
    }

    sctl::Long Dim(sctl::Integer i) const { return dim[i]; }

    template <class Surface, class Kernel> void Setup(const sctl::Vector<Surface>& Svec, const Kernel& ker) {
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
      normal_scal.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        const auto& S = Svec[i];
        Nt0[i] = S.NTor();
        Np0[i] = S.NPol();
        Op[i] = SurfaceOp<Real>(comm_, Nt0[i]*UPSAMPLE, Np0[i]*UPSAMPLE);

        Xtrg[i] = S.Coord();
        Upsample(Xtrg[i], Xsrc[i], Nt0[i], Np0[i]);
        Op[i].Grad2D(dXsrc[i], Xsrc[i]);
        normal_scal = Op[i].SurfNormalAreaElem(&Xn_src[i], &Xa_src[i], dXsrc[i], &Xsrc[i]);
        dim[0] += ker.Dim(0) * (Xtrg[i].Dim() / COORD_DIM);
        dim[1] += ker.Dim(1) * (Xtrg[i].Dim() / COORD_DIM);
      }
    }

    template <class Surface, class Kernel> void SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker) {
      Setup(Svec, ker);
      sctl::Long Nsurf = Svec.Dim();
      singular_correction.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        Op[i].SetupSingularCorrection(singular_correction[i], UPSAMPLE, Xsrc[i], dXsrc[i], ker, normal_scal[i]);
      }
    }

    void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(ker_ptr);
      sctl::Long dof = F.Dim() / Dim(0);
      SCTL_ASSERT(F.Dim() == dof * Dim(0));
      if (U.Dim() != dof * Dim(1)) {
        U.ReInit(dof * Dim(1));
        U = 0;
      }

      sctl::Vector<Real> Fsrc;
      sctl::Long SrcOffset = 0;
      for (sctl::Long s = 0; s < Xsrc.Dim(); s++) {
        const sctl::Vector<Real> F_(dof * ker_ptr->Dim(0) * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + dof * ker_ptr->Dim(0) * SrcOffset, false);
        Upsample(F_, Fsrc, Nt0[s], Np0[s]);

        sctl::Long TrgOffset = 0;
        for (sctl::Long t = 0; t < Xtrg.Dim(); t++) {
          sctl::Vector<Real> U_(dof * ker_ptr->Dim(1) * Nt0[t] * Np0[t], U.begin() + dof * ker_ptr->Dim(1) * TrgOffset, false);
          Op[s].EvalSurfInteg(U_, Xtrg[t], Xsrc[s], Xn_src[s], Xa_src[s], Fsrc, *ker_ptr);
          TrgOffset += Nt0[t] * Np0[t];
        }

        sctl::Vector<Real> U_(dof * ker_ptr->Dim(1) * Nt0[s] * Np0[s], U.begin() + dof * ker_ptr->Dim(1) * SrcOffset, false);
        Op[s].EvalSingularCorrection(U_, singular_correction[s], ker_ptr->Dim(0), ker_ptr->Dim(1), Fsrc);
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    void EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real> Xt, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(ker_ptr);
      SCTL_ASSERT(F.Dim() == Dim(0));
      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
      if (U.Dim() != Nt * ker_ptr->Dim(1)) {
        U.ReInit(Nt * ker_ptr->Dim(1));
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

    template <class Kernel> static sctl::Vector<Real> test(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& ker, const sctl::Comm& comm) {
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

    template <class Kernel, class GradKernel> static void test_GreensIdentity(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& sl_ker, const Kernel& dl_ker, const GradKernel& sl_grad_ker, const sctl::Comm& comm) {
      sctl::Vector<Surface<Real>> Svec;
      Svec.PushBack(Surface<Real>(Nt, Np, surf_type));
      BoundaryIntegralOp<Real, Kernel::Dim(0), Kernel::Dim(1), UPSAMPLE, PATCH_DIM0, RAD_DIM> SL(comm);
      BoundaryIntegralOp<Real, Kernel::Dim(0), Kernel::Dim(1), UPSAMPLE, PATCH_DIM0, RAD_DIM> DL(comm);

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

    static void test_Precond(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm) {
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

  private:

    static void Upsample(const sctl::Vector<Real>& X0_, sctl::Vector<Real>& X_, sctl::Long Nt0, sctl::Long Np0) { // TODO: Replace by upsample in SurfaceOp
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

    static Real max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }

    sctl::Comm comm_;
    sctl::Vector<sctl::Long> Nt0, Np0;
    sctl::Vector<SurfaceOp<Real>> Op;
    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>* ker_ptr;

    sctl::StaticArray<sctl::Long,2> dim;
    sctl::Vector<Real> normal_scal;
    sctl::Vector<sctl::Vector<Real>> Xtrg, Xsrc, dXsrc, Xn_src, Xa_src;
    sctl::Vector<sctl::Vector<SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>>> singular_correction;
};

}

#endif  //_BOUNDARY_INTEG_OP_HPP_
