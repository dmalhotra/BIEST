// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <biest/singular_correction.hpp>

namespace biest {

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::BoundaryIntegralOp(const sctl::Comm& comm) : comm_(comm) {
      dim[0] = 0;
      dim[1] = 0;
      ker_ptr = nullptr;
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> sctl::Long BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::Dim(sctl::Integer i) const { return dim[i]; }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::SetupHelper(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_) {
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
        const sctl::Vector<Real>& Xsrc0 = S.Coord();

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
        normal_orient[i] = Op[i].SurfNormalAreaElem(&Xn_src[i], &Xa_src[i], dXsrc[i], &Xsrc[i]);
        dim[0] += ker.Dim(0) * (Xsrc0.Dim() / COORD_DIM);
        dim[1] += ker.Dim(1) * (Xtrg[i].Dim() / COORD_DIM);
      }
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_, const Real tol, const SingularQuadParams& quad) {
      using SingCorr = SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>;
      SetupHelper(Svec, ker, trg_idx_);
      const sctl::Long Nsurf = Svec.Dim();
      off_op_.clear(); // the panels of EvalOffSurface() for the earlier surfaces
      while ((sctl::Long)singular_correction.size() < Nsurf) singular_correction.emplace_back(comm_); // those of the earlier setup kept
      singular_correction.resize(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        const sctl::Long Nt = Nt0[i] * UPSAMPLE, Np = Np0[i] * UPSAMPLE;
        SingularQuadParams q = quad; // and the values not given for tol and the grid, the patch within it
        if (!q.patch_rad || !q.rad_nodes || !q.ang_nodes) {
          const SingularQuadParams q0 = SingCorr::QuadParams(tol, SingCorr::GridAnisotropy(dXsrc[i], Nt, Np));
          if (!q.patch_rad) q.patch_rad = std::min<sctl::Long>(q0.patch_rad, (std::min(Nt, Np) - 1) / 2);
          if (!q.rad_nodes) q.rad_nodes = q0.rad_nodes;
          if (!q.ang_nodes) q.ang_nodes = q0.ang_nodes;
        }
        singular_correction[i].Setup(q, UPSAMPLE, Nt, Np, Xsrc[i], dXsrc[i], ker, normal_orient[i], trg_idx[i]);
      }
      SetupCross(Svec, ker, tol);
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::SetupCross(const sctl::Vector<Surface>& Svec, const Kernel& ker, const Real tol) {
      const sctl::Long Nsurf = Svec.Dim();
      cross.clear();
      cross.resize(Nsurf * Nsurf);
      if (Nsurf < 2 || !NewNearOp(ker, comm_)) return; // one surface, or a kernel without OffSurfaceOp

      // A target is near s if it is within R_j = c h_j of a point j of the grid of s upsampled UPSAMPLE times, with h_j
      // the larger grid spacing at j (the derivatives are with respect to the parameters in [0,1)): one more than
      // log(1/tol) / (2 pi) grid spacings, for a target that is that far from the surface but not from a grid point.
      // The grid points are in tiles of B x B, each in a ball (center, radius): a target is near a point of the tile
      // only if it is within the radius plus the largest R_j of the center.
      const Real c = sctl::log<Real>(1 / tol) / (2 * sctl::const_pi<Real>()) + 1;
      constexpr sctl::Long B = 8;
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const sctl::Long Nt = Nt0[s] * UPSAMPLE, Np = Np0[s] * UPSAMPLE, N = Nt * Np;
        const sctl::Long Bt = (Nt + B - 1) / B, Bp = (Np + B - 1) / B;
        const sctl::Vector<Real>& X = Xsrc[s];
        const sctl::Vector<Real>& dX = dXsrc[s];
        sctl::Vector<Real> R2(N), tile(Bt * Bp * (COORD_DIM + 1)); // R_j^2; and for each tile, the center and the reach
        #pragma omp parallel for schedule(static)
        for (sctl::Long j = 0; j < N; j++) {
          Real ht2 = 0, hp2 = 0;
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            ht2 += dX[(k * 2 + 0) * N + j] * dX[(k * 2 + 0) * N + j];
            hp2 += dX[(k * 2 + 1) * N + j] * dX[(k * 2 + 1) * N + j];
          }
          const Real h = std::max<Real>(sctl::sqrt<Real>(ht2) / Nt, sctl::sqrt<Real>(hp2) / Np);
          R2[j] = c * c * h * h;
        }
        #pragma omp parallel for schedule(static)
        for (sctl::Long b = 0; b < Bt * Bp; b++) {
          const sctl::Long t0 = (b / Bp) * B, t1 = std::min(Nt, t0 + B), p0 = (b % Bp) * B, p1 = std::min(Np, p0 + B);
          Real x0[COORD_DIM] = {0, 0, 0}, rad2 = 0, R2max = 0;
          for (sctl::Long t = t0; t < t1; t++) {
            for (sctl::Long p = p0; p < p1; p++) {
              for (sctl::Integer k = 0; k < COORD_DIM; k++) x0[k] += X[k * N + t * Np + p] / ((t1 - t0) * (p1 - p0));
            }
          }
          for (sctl::Long t = t0; t < t1; t++) {
            for (sctl::Long p = p0; p < p1; p++) {
              Real r2 = 0;
              for (sctl::Integer k = 0; k < COORD_DIM; k++) r2 += (X[k * N + t * Np + p] - x0[k]) * (X[k * N + t * Np + p] - x0[k]);
              rad2 = std::max<Real>(rad2, r2);
              R2max = std::max<Real>(R2max, R2[t * Np + p]);
            }
          }
          for (sctl::Integer k = 0; k < COORD_DIM; k++) tile[b * (COORD_DIM + 1) + k] = x0[k];
          tile[b * (COORD_DIM + 1) + COORD_DIM] = sctl::sqrt<Real>(rad2) + sctl::sqrt<Real>(R2max);
        }

        for (sctl::Long t = 0; t < Nsurf; t++) {
          if (t == s) continue;
          const sctl::Long Ntrg = trg_idx[t].Dim();
          const sctl::Vector<Real>& Xt = Xtrg[t];
          sctl::Vector<char> is_near(Ntrg);
          #pragma omp parallel for schedule(dynamic, 64)
          for (sctl::Long i = 0; i < Ntrg; i++) {
            const Real x[COORD_DIM] = {Xt[0 * Ntrg + i], Xt[1 * Ntrg + i], Xt[2 * Ntrg + i]};
            const auto dist2 = [&x](const Real* y) {
              Real r2 = 0;
              for (sctl::Integer k = 0; k < COORD_DIM; k++) r2 += (x[k] - y[k]) * (x[k] - y[k]);
              return r2;
            };
            bool near = false;
            for (sctl::Long b = 0; b < Bt * Bp && !near; b++) {
              const Real reach = tile[b * (COORD_DIM + 1) + COORD_DIM];
              if (dist2(&tile[b * (COORD_DIM + 1)]) >= reach * reach) continue;
              const sctl::Long t0 = (b / Bp) * B, t1 = std::min(Nt, t0 + B), p0 = (b % Bp) * B, p1 = std::min(Np, p0 + B);
              for (sctl::Long tt = t0; tt < t1 && !near; tt++) {
                for (sctl::Long pp = p0; pp < p1 && !near; pp++) {
                  const sctl::Long j = tt * Np + pp;
                  const Real y[COORD_DIM] = {X[0 * N + j], X[1 * N + j], X[2 * N + j]};
                  near = (dist2(y) < R2[j]);
                }
              }
            }
            is_near[i] = near;
          }

          CrossInterac& ci = cross[s * Nsurf + t];
          for (sctl::Long i = 0; i < Ntrg; i++) (is_near[i] ? ci.near_idx : ci.far_idx).PushBack(i);
          if (!ci.near_idx.Dim()) { // all of them by the trapezoidal rule, on Xtrg[t]
            ci.far_idx.ReInit(0);
            continue;
          }
          const sctl::Long Nfar = ci.far_idx.Dim(), Nnear = ci.near_idx.Dim();
          sctl::Vector<Real> Xnear(COORD_DIM * Nnear);
          ci.Xfar.ReInit(COORD_DIM * Nfar);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            for (sctl::Long i = 0; i < Nfar; i++) ci.Xfar[k * Nfar + i] = Xt[k * Ntrg + ci.far_idx[i]];
            for (sctl::Long i = 0; i < Nnear; i++) Xnear[k * Nnear + i] = Xt[k * Ntrg + ci.near_idx[i]];
          }
          ci.near_op = NewNearOp(ker, comm_);
          ci.near_op->Setup(Svec[s].Coord(), Nt0[s], Np0[s], Xnear, tol);
        }
      }
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(ker_ptr);
      sctl::Long dof = F.Dim() / Dim(0);
      SCTL_ASSERT(F.Dim() == dof * Dim(0));
      if (U.Dim() != dof * Dim(1)) {
        U.ReInit(dof * Dim(1));
        U = 0;
      }

      sctl::Long SrcOffset = 0, SingularTrgOffset = 0;
      for (sctl::Long s = 0; s < Xsrc.Dim(); s++) {
        const sctl::Vector<Real> F_(dof * ker_ptr->Dim(0) * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + dof * ker_ptr->Dim(0) * SrcOffset, false);
        // Fsrc: on the same grid F_ itself, else F_ upsampled, in scratch memory as the solvers call this at every
        // iteration (see SurfaceOp::SurfGrad)
        sctl::ScratchBuf<Real> Fsrc_buf(UPSAMPLE == 1 ? 0 : F_.Dim() * UPSAMPLE * UPSAMPLE);
        sctl::Vector<Real> Fsrc;
        if (UPSAMPLE == 1) Fsrc.ReInit(F_.Dim(), (sctl::Iterator<Real>)F_.begin(), false);
        else {
          Fsrc.ReInit(Fsrc_buf.size(), Fsrc_buf.begin(), false);
          Upsample(F_, Fsrc, Nt0[s], Np0[s]);
        }

        sctl::Long TrgOffset = 0;
        for (sctl::Long t = 0; t < Xtrg.Dim(); t++) {
          const sctl::Long Ntrg = trg_idx[t].Dim();
          sctl::Vector<Real> U_(dof * ker_ptr->Dim(1) * Ntrg, U.begin() + dof * ker_ptr->Dim(1) * TrgOffset, false);
          SCTL_ASSERT(Xtrg[t].Dim()/COORD_DIM == Ntrg);
          const CrossInterac* ci = (t != s ? &cross[s * Xtrg.Dim() + t] : nullptr);
          if (!ci || !ci->near_op) {
            Op[s].EvalSurfInteg(U_, Xtrg[t], Xsrc[s], Xn_src[s], Xa_src[s], Fsrc, *ker_ptr);
          } else { // the trapezoidal rule at the far targets, and near_op at the near ones (see SetupSingular)
            const sctl::Long Nfar = ci->far_idx.Dim(), Nnear = ci->near_idx.Dim(), Nc = dof * ker_ptr->Dim(1);
            sctl::ScratchBuf<Real> Ufar_buf(Nc * Nfar), Unear_buf(Nc * Nnear);
            sctl::Vector<Real> Ufar(Ufar_buf), Unear(Unear_buf);
            Ufar = 0;
            Unear = 0;
            if (Nfar) Op[s].EvalSurfInteg(Ufar, ci->Xfar, Xsrc[s], Xn_src[s], Xa_src[s], Fsrc, *ker_ptr);
            ci->near_op->Eval(Unear, F_);
            for (sctl::Long k = 0; k < Nc; k++) {
              for (sctl::Long i = 0; i < Nfar; i++) U_[k * Ntrg + ci->far_idx[i]] += Ufar[k * Nfar + i];
              for (sctl::Long i = 0; i < Nnear; i++) U_[k * Ntrg + ci->near_idx[i]] += Unear[k * Nnear + i];
            }
          }
          TrgOffset += Ntrg;
        }

        sctl::Vector<Real> U_(dof * ker_ptr->Dim(1) * trg_idx[s].Dim(), U.begin() + dof * ker_ptr->Dim(1) * SingularTrgOffset, false);
        singular_correction[s](U_, Fsrc);
        SingularTrgOffset += trg_idx[s].Dim();
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol) const {
      SCTL_ASSERT(ker_ptr);
      SCTL_ASSERT_MSG(&ker == ker_ptr, "EvalOffSurface: the kernel function must be the one given to SetupSingular.");
      SCTL_ASSERT(F.Dim() == Dim(0));
      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
      if (U.Dim() != ker_ptr->Dim(1) * Nt) {
        U.ReInit(ker_ptr->Dim(1) * Nt);
        U = 0;
      }

      const sctl::Long Nsurf = Xsrc.Dim();
      if ((sctl::Long)off_op_.size() != Nsurf || off_tol_ != tol) { // the panels fit for another tolerance
        off_op_.clear();
        off_op_.resize(Nsurf);
        off_tol_ = tol;
      }
      sctl::Long SrcOffset = 0;
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const sctl::Vector<Real> F_(ker_ptr->Dim(0) * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + ker_ptr->Dim(0) * SrcOffset, false);
        if (!off_op_[s]) {
          off_op_[s] = NewNearOp(ker, comm_);
          off_op_[s]->SetupSurface(Xsrc[s], Nt0[s] * UPSAMPLE, Np0[s] * UPSAMPLE, tol);
        }
        off_op_[s]->SetTargets(Xt);
        // Fsrc: F_ upsampled, in scratch memory (see SurfaceOp::SurfGrad), or F_ itself
        sctl::ScratchBuf<Real> Fsrc_buf(UPSAMPLE == 1 ? 0 : F_.Dim() * UPSAMPLE * UPSAMPLE);
        sctl::Vector<Real> Fsrc;
        if (UPSAMPLE == 1) Fsrc.ReInit(F_.Dim(), (sctl::Iterator<Real>)F_.begin(), false);
        else {
          Fsrc.ReInit(Fsrc_buf.size(), Fsrc_buf.begin(), false);
          Upsample(F_, Fsrc, Nt0[s], Np0[s]);
        }
        off_op_[s]->Eval(U, Fsrc);
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> template <class Kernel> sctl::Vector<Real> BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::test(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& ker, const sctl::Comm& comm, const SingularQuadParams& quad) {
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
      integ_op.SetupSingular(Svec, ker, sctl::Vector<sctl::Vector<sctl::Long>>(), (Real)1e-10, quad);
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

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> template <class Kernel, class DLKernel, class GradKernel> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::test_GreensIdentity(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& sl_ker, const DLKernel& dl_ker, const GradKernel& sl_grad_ker, const sctl::Comm& comm, const SingularQuadParams& quad) {
      sctl::Vector<Surface<Real>> Svec;
      Svec.PushBack(Surface<Real>(Nt, Np, surf_type));
      BoundaryIntegralOp<Real, Kernel::Dim(0), Kernel::Dim(1), UPSAMPLE, HedgehogOrder> SL(comm);
      BoundaryIntegralOp<Real, Kernel::Dim(0), Kernel::Dim(1), UPSAMPLE, HedgehogOrder> DL(comm);

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
      SL.SetupSingular(Svec, sl_ker, sctl::Vector<sctl::Vector<sctl::Long>>(), (Real)1e-10, quad);
      DL.SetupSingular(Svec, dl_ker, sctl::Vector<sctl::Vector<sctl::Long>>(), (Real)1e-10, quad);
      sctl::Profile::Toc();

      sctl::Profile::Tic("Eval", &comm);
      sctl::Vector<Real> Us, Ud;
      SL(Us, Fs);
      DL(Ud, Fd);
      sctl::Profile::Toc();

      std::cout<<"Error: "<<max_norm(Us-Ud-0.5*Fd)/max_norm(Fd)<<'\n';
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::test_Precond(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm, const SingularQuadParams& quad) {
      sctl::Profile::Tic("SLayerPrecond", &comm);
      sctl::Profile::Tic("Setup", &comm);
      Surface<Real> S(Nt, Np, surf_type);
      BoundaryIntegralOp integ_op(comm);
      const auto& ker = Laplace3D<Real>::FxU();
      sctl::Vector<Surface<Real>> Svec; Svec.PushBack(S);
      integ_op.SetupSingular(Svec, ker, sctl::Vector<sctl::Vector<sctl::Long>>(), (Real)1e-10, quad);
      std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> SPrecond = [&integ_op](sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) { integ_op(Sx, x); };
      sctl::Profile::Toc();
      SurfaceOp<Real>::test_InvSurfLap(Nt, Np, surf_type, gmres_tol, gmres_iter, &SPrecond, &SPrecond, comm);
      sctl::Profile::Toc();
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::Upsample(const sctl::Vector<Real>& X0_, sctl::Vector<Real>& X_, sctl::Long Nt0, sctl::Long Np0) {
      SurfaceOp<Real>::Upsample(X0_, Nt0, Np0, X_, UPSAMPLE * Nt0, UPSAMPLE * Np0);
    }

    template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE, sctl::Integer HedgehogOrder> Real BoundaryIntegralOp<Real,KDIM0,KDIM1,UPSAMPLE,HedgehogOrder>::max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }



    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> biest::Surface<Real> FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::BuildSurface(const sctl::Vector<Real>& X, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np) {
      SCTL_ASSERT(surf_Nt*surf_Np*COORD_DIM == X.Dim());
      return biest::Surface<Real>(NFP, surf_Nt, surf_Np, X);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::FieldPeriodBIOp(const sctl::Comm& comm) : biop(nullptr), dbl_op_(nullptr), NFP_(0), comm_(comm) {
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::~FieldPeriodBIOp() {
      delete biop;
      delete dbl_op_;
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long qNt, const sctl::Long qNp) {
      SCTL_ASSERT(Svec.Dim() == 1);
      SCTL_ASSERT(Svec[0].NTor() % NFP == 0);
      sctl::StaticArray<sctl::Long,6> d{src_Nt, src_Np, trg_Nt, trg_Np, qNt, qNp};
      const auto one = [&d](const sctl::Integer k) { return sctl::Vector<sctl::Long>(1, d + k, false); }; // a view of d[k]
      const bool q = (qNt > 0 && qNp > 0);
      SetupDigits(Svec, ker, digits, NFP, one(0), one(1), one(2), one(3), (q ? one(4) : sctl::Vector<sctl::Long>()), (q ? one(5) : sctl::Vector<sctl::Long>()));
    }

    namespace detail_fieldperiod {
      /**
       * Returns the field periods common to the surfaces: the greatest common divisor of their Surface::NFP().
       */
      template <class Real> sctl::Integer CommonNFP(const sctl::Vector<biest::Surface<Real>>& Svec) {
        sctl::Integer NFP = 0;
        for (const auto& S : Svec) {
          sctl::Integer a = NFP, b = S.NFP();
          while (b) { const sctl::Integer r = a % b; a = b; b = r; }
          NFP = a;
        }
        return std::max<sctl::Integer>(NFP, 1);
      }
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <class Kernel> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const sctl::Integer digits, const sctl::Vector<sctl::Long>& trg_Nt, const sctl::Vector<sctl::Long>& trg_Np, const sctl::Vector<sctl::Long>& qNt, const sctl::Vector<sctl::Long>& qNp) {
      const sctl::Long Nsurf = Svec.Dim();
      sctl::ScratchBuf<sctl::Long> src_buf(2 * Nsurf); // the densities on the surface grids, at most; in scratch memory
      sctl::Vector<sctl::Long> src_Nt(Nsurf, src_buf.begin(), false), src_Np(Nsurf, src_buf.begin() + Nsurf, false);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        src_Nt[i] = Svec[i].NTor();
        src_Np[i] = Svec[i].NPol();
      }
      SetupDigits(Svec, ker, digits, detail_fieldperiod::CommonNFP(Svec), src_Nt, src_Np, trg_Nt, trg_Np, qNt, qNp);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <sctl::Integer UPSAMPLE, class Kernel> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const Real tol, const SingularQuadParams& quad) {
      const sctl::Long Nsurf = Svec.Dim();
      const sctl::Integer NFP = detail_fieldperiod::CommonNFP(Svec);
      trg_Nt_.ReInit(Nsurf);
      trg_Np_.ReInit(Nsurf);
      quad_Nt_.ReInit(Nsurf);
      quad_Np_.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // the targets: the grid points of one field period
        trg_Nt_[i] = Svec[i].NTor() / NFP;
        trg_Np_[i] = Svec[i].NPol();
        quad_Nt_[i] = UPSAMPLE * Svec[i].NTor();
        quad_Np_[i] = UPSAMPLE * Svec[i].NPol();
      }
      SetGrids(Svec, NFP);
      SetupSingular0(ker, tol, quad);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <class Kernel> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupDigits(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const sctl::Vector<sctl::Long>& trg_Nt, const sctl::Vector<sctl::Long>& trg_Np, const sctl::Vector<sctl::Long>& qNt, const sctl::Vector<sctl::Long>& qNp) {
      using SingCorr = SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>;
      const sctl::Long Nsurf = Svec.Dim();
      SCTL_ASSERT(src_Nt.Dim() == Nsurf && src_Np.Dim() == Nsurf && trg_Nt.Dim() == Nsurf && trg_Np.Dim() == Nsurf);
      SCTL_ASSERT((!qNt.Dim() && !qNp.Dim()) || (qNt.Dim() == Nsurf && qNp.Dim() == Nsurf));

      // For each surface, the anisotropy of its grid (cond), the aspect ratio of the quadrature grid that makes its cells
      // square on average, and the patch radius of the singular quadrature on a grid of that aspect ratio
      // The temporaries in scratch memory (see SurfaceOp::SurfGrad)
      const Real tol = sctl::pow<Real>((Real)0.1,digits);
      sctl::ScratchBuf<Real> optim_aspect_ratio_buf(Nsurf);
      sctl::ScratchBuf<sctl::Long> PDIM_buf(Nsurf);
      sctl::Vector<Real> optim_aspect_ratio(optim_aspect_ratio_buf);
      sctl::Vector<sctl::Long> PDIM(PDIM_buf);
      Real cond = 1;
      for (sctl::Long s = 0; s < Nsurf; s++) {
        sctl::StaticArray<Real,2> aspect_ratio{100,0};
        sctl::ScratchBuf<Real> dX_buf(COORD_DIM * 2 * Svec[s].NTor() * Svec[s].NPol());
        sctl::Vector<Real> dX(dX_buf);
        { // Set aspect_ratio
          sctl::StaticArray<sctl::Long,2> SurfDim{Svec[s].NTor(),Svec[s].NPol()};
          biest::SurfaceOp<Real> SurfOp(comm_, SurfDim[0], SurfDim[1]);
          SurfOp.Grad2D(dX, Svec[s].Coord());

          Real M[2][2];
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
          }
        }
        optim_aspect_ratio[s] = sctl::sqrt<Real>(aspect_ratio[0]*aspect_ratio[1]) * Svec[s].NTor() / Svec[s].NPol();
        cond = (s ? std::max<Real>(cond, sctl::sqrt<Real>(aspect_ratio[1]/aspect_ratio[0])) : sctl::sqrt<Real>(aspect_ratio[1]/aspect_ratio[0]));
        PDIM[s] = SingCorr::QuadParams(tol, SingCorr::GridAnisotropy(dX, Svec[s].NTor(), Svec[s].NPol(), optim_aspect_ratio[s], 1)).patch_rad;
      }
      if (cond > 4) {
        SCTL_WARN("The surface mesh is highly anisotropic! Quadrature generation will be very slow. Consider using a better surface discretization.");
        std::cout<<"Mesh anisotropy = "<<cond<<'\n';
      }

      trg_Nt_ = trg_Nt;
      trg_Np_ = trg_Np;
      quad_Nt_.ReInit(Nsurf);
      quad_Np_.ReInit(Nsurf);
      for (sctl::Long s = 0; s < Nsurf; s++) { // Set quad_Nt_, quad_Np_
        if (qNt.Dim()) {
          quad_Nt_[s] = qNt[s];
          quad_Np_[s] = qNp[s];
          continue;
        }
        const sctl::Long surf_Nt = Svec[s].NTor(), surf_Np = Svec[s].NPol();
        sctl::Long& quad_Nt = quad_Nt_[s];
        sctl::Long& quad_Np = quad_Np_[s];
        quad_Np =       trg_Np[s]  * (sctl::Long)sctl::ceil(std::max<sctl::Long>(std::max<sctl::Long>(surf_Np, src_Np[s]),                    2*PDIM[s]+1) / (Real)      trg_Np[s] );
        quad_Nt = (NFP*trg_Nt[s]) * (sctl::Long)sctl::ceil(std::max<Real>((Real)std::max<sctl::Long>(surf_Nt, src_Nt[s]), optim_aspect_ratio[s]*quad_Np) / (Real)(NFP*trg_Nt[s]));

        // Adaptive refinement using the double-layer test at the targets of the surface alone, at most three tests:
        // while the error is above 10^-digits, a grid finer by the factor that the error predicts for digits+1 digits,
        // unless that factor is below 1.5. The grids are multiples of the target grid, and the last one tested is the
        // quadrature grid.
        // The test operator: that of the object, set up in the memory of the earlier tests; S: surface s alone (a view)
        const sctl::Vector<biest::Surface<Real>> S(1, (sctl::Iterator<biest::Surface<Real>>)Svec.begin() + s, false);
        if (!dbl_op_) dbl_op_ = new FieldPeriodBIOp<Real, COORD_DIM, 1, 1>(comm_);
        auto& dbl_op = *dbl_op_;
        for (sctl::Integer i = 0; i < 3; i++) {
          dbl_op.trg_Nt_.ReInit(1);
          dbl_op.trg_Np_.ReInit(1);
          dbl_op.quad_Nt_.ReInit(1);
          dbl_op.quad_Np_.ReInit(1);
          dbl_op.trg_Nt_[0] = trg_Nt[s];
          dbl_op.trg_Np_[0] = trg_Np[s];
          dbl_op.quad_Nt_[0] = quad_Nt;
          dbl_op.quad_Np_[0] = quad_Np;
          dbl_op.SetGrids(S, NFP);
          dbl_op.SetupSingular0(biest::Laplace3D<Real>::DxU(), tol, SingularQuadParams());
          sctl::ScratchBuf<Real> U_buf(dbl_op.Dim(1)), F_buf(dbl_op.Dim(0));
          sctl::Vector<Real> U(U_buf), F(F_buf);
          U = 0;
          F = 1;
          dbl_op(U, F);

          Real err = 0;
          for (const auto& a : U) err = std::max<Real>(err, sctl::fabs<Real>(a+(Real)0.5));
          if (err < tol) break;
          const Real got = std::max<Real>(1, sctl::log(err)/sctl::log((Real)0.1)); // the digits of the test, at least 1
          const Real scal = (digits+1)/got; // assuming exponential/geometric convergence
          if (scal < 1.5 || i == 2) break;
          quad_Np =       trg_Np[s]  * (sctl::Long)sctl::ceil(scal * quad_Np / (Real)      trg_Np[s] );
          quad_Nt = (NFP*trg_Nt[s]) * (sctl::Long)sctl::ceil(scal * quad_Nt / (Real)(NFP*trg_Nt[s]));
        }
      }
      for (sctl::Long s = 0; s < Nsurf; s++) { // the grids hold the patch, the surface and the densities, and the targets are on them
        const sctl::Long qNt_ = quad_Nt_[s], qNp_ = quad_Np_[s];
        if (!(qNp_ > 2*PDIM[s]) ||
            !(qNt_ >= Svec[s].NTor()) ||
            !(qNp_ >= Svec[s].NPol()) ||
            !(qNt_ >= src_Nt[s] ) ||
            !(qNp_ >= src_Np[s] ) ||
            !((qNt_/(NFP*trg_Nt[s]))*(NFP*trg_Nt[s]) == qNt_) ||
            !((qNp_/     trg_Np[s] )*     trg_Np[s]  == qNp_)) {
          std::cout<<"surface            = "<<s                    <<'\n';
          std::cout<<"digits             = "<<digits               <<'\n';
          std::cout<<"cond               = "<<cond                 <<'\n';
          std::cout<<"PDIM               = "<<PDIM[s]              <<'\n';
          std::cout<<"NFP                = "<<NFP                  <<'\n';
          std::cout<<"surf_Nt            = "<<Svec[s].NTor()       <<'\n';
          std::cout<<"surf_Np            = "<<Svec[s].NPol()       <<'\n';
          std::cout<<"src_Nt             = "<<src_Nt[s]            <<'\n';
          std::cout<<"src_Np             = "<<src_Np[s]            <<'\n';
          std::cout<<"trg_Nt             = "<<trg_Nt[s]            <<'\n';
          std::cout<<"trg_Np             = "<<trg_Np[s]            <<'\n';
          std::cout<<"quad_Nt            = "<<qNt_                 <<'\n';
          std::cout<<"quad_Np            = "<<qNp_                 <<'\n';
          std::cout<<"optim_aspect_ratio = "<<optim_aspect_ratio[s]<<'\n';
          SCTL_ASSERT(false);
        }
      }

      SetGrids(Svec, NFP);
      SetupSingular0(ker, tol, SingularQuadParams());
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetGrids(const sctl::Vector<biest::Surface<Real>>& Svec, const sctl::Integer NFP) {
      const sctl::Long Nsurf = Svec.Dim();
      SCTL_ASSERT(trg_Nt_.Dim() == Nsurf && trg_Np_.Dim() == Nsurf && quad_Nt_.Dim() == Nsurf && quad_Np_.Dim() == Nsurf);
      NFP_ = NFP;
      tor_orient_.ReInit(Nsurf);
      Svec_.ReInit(Nsurf);
      trg_idx_.ReInit(Nsurf);
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const auto& S = Svec[s];
        const sctl::Long qNt = quad_Nt_[s], qNp = quad_Np_[s], tNt = trg_Nt_[s], tNp = trg_Np_[s];
        SCTL_ASSERT(S.NTor() % NFP == 0 && qNt % (NFP * tNt) == 0 && qNp % tNp == 0);
        tor_orient_[s] = biest::SurfaceOp<Real>::ToroidalOrientation(S.Coord(), S.NTor(), S.NPol());

        // The surface on its quadrature grid, in the storage of the earlier setup when it has the size
        if (qNt == S.NTor() && qNp == S.NPol()) {
          Svec_[s] = S;
        } else {
          if (Svec_[s].NTor() != qNt || Svec_[s].NPol() != qNp) Svec_[s] = biest::Surface<Real>(qNt, qNp, biest::SurfType::None);
          SurfaceOp<Real>::Resample(Svec_[s].Coord(), qNt, qNp, S.Coord(), S.NTor(), S.NPol());
        }

        // The targets: the points of the tNt x tNp grid of one field period, which are quadrature grid points
        trg_idx_[s].ReInit(tNt * tNp);
        const sctl::Long skip_Nt = qNt / (NFP * tNt);
        const sctl::Long skip_Np = qNp / tNp;
        for (sctl::Long i = 0; i < tNt; i++) {
          for (sctl::Long j = 0; j < tNp; j++) {
            trg_idx_[s][i*tNp+j] = (i*skip_Nt*tNp+j)*skip_Np;
            SCTL_ASSERT(trg_idx_[s][i*tNp+j] < qNt*qNp);
          }
        }
      }
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP, const sctl::Integer src_Nt, const sctl::Integer src_Np) const {
      SCTL_ASSERT(Svec_.Dim() == 1);
      const sctl::Long src_Nt_ = (src_Nt==-1 ? QuadNt() : src_Nt);
      const sctl::Long src_Np_ = (src_Np==-1 ? QuadNp() : src_Np);
      SCTL_ASSERT(F.Dim() == KDIM0 * src_Nt_ * src_Np_);
      sctl::ScratchBuf<sctl::Long> src_Nt_buf(1), src_Np_buf(1);
      sctl::Vector<sctl::Long> src_Nt__(src_Nt_buf), src_Np__(src_Np_buf);
      src_Nt__[0] = src_Nt_;
      src_Np__[0] = src_Np_;
      Eval(U, F, NFP, src_Nt__, src_Np__);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np) const {
      const sctl::Long Nsurf = Svec_.Dim();
      SCTL_ASSERT(src_Nt.Dim() == Nsurf && src_Np.Dim() == Nsurf);
      sctl::Long Nsrc = 0, Nquad = 0;
      bool on_quad = (NFP == 1); // the densities on the quadrature grids
      for (sctl::Long s = 0; s < Nsurf; s++) {
        Nsrc += src_Nt[s] * src_Np[s];
        Nquad += quad_Nt_[s] * quad_Np_[s];
        on_quad = on_quad && src_Nt[s] == quad_Nt_[s] && src_Np[s] == quad_Np_[s];
      }
      const sctl::Long dof = F.Dim() / (KDIM0 * Nsrc);
      SCTL_ASSERT(F.Dim() == dof * KDIM0 * Nsrc);
      if (on_quad) {
        (*biop)(U, F);
        return;
      }

      // The densities of each surface completed to the full torus by symmetry, then resampled to the quadrature grid; in
      // scratch memory, as the solvers call this at every iteration (see SurfaceOp::SurfGrad)
      sctl::ScratchBuf<Real> Fq_buf(dof * KDIM0 * Nquad);
      sctl::Long src_dsp = 0, quad_dsp = 0;
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const sctl::Long Ns = src_Nt[s] * src_Np[s], Nq = quad_Nt_[s] * quad_Np_[s];
        const sctl::Vector<Real> F_(dof * KDIM0 * Ns, (sctl::Iterator<Real>)F.begin() + dof * KDIM0 * src_dsp, false);
        sctl::Vector<Real> Fq_(dof * KDIM0 * Nq, Fq_buf.begin() + dof * KDIM0 * quad_dsp, false);
        sctl::ScratchBuf<Real> Ff_buf(dof * KDIM0 * NFP * Ns);
        sctl::Vector<Real> Ff(Ff_buf);
        biest::SurfaceOp<Real>::CompleteVecField(Ff, false, false, NFP, src_Nt[s], src_Np[s], F_, 0, tor_orient_[s]);
        biest::SurfaceOp<Real>::Resample(Fq_, quad_Nt_[s], quad_Np_[s], Ff, NFP * src_Nt[s], src_Np[s]);
        src_dsp += Ns;
        quad_dsp += Nq;
      }
      const sctl::Vector<Real> Fq(dof * KDIM0 * Nquad, Fq_buf.begin(), false);
      (*biop)(U, Fq);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(biop && F.Dim() % Dim(0) == 0);
      (*biop)(U, F);
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> sctl::Long FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::Dim(sctl::Integer i) const {
      sctl::Long N = 0;
      for (sctl::Long s = 0; s < Svec_.Dim(); s++) N += (i ? KDIM1 * trg_Nt_[s] * trg_Np_[s] : KDIM0 * quad_Nt_[s] * quad_Np_[s]);
      return N;
    }

    template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <class Kernel> void FieldPeriodBIOp<Real,COORD_DIM,KDIM0,KDIM1,HedgehogOrder>::SetupSingular0(const Kernel& ker, const Real tol, const SingularQuadParams& quad) {
      if (!biop) biop = new BIOp(comm_);
      biop->SetupSingular(Svec_, ker, trg_idx_, tol, quad); // in the memory of the earlier setup
    }

}
