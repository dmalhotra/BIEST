#include "biest/bie_solvers.hpp"

#include <biest/boundary_integ_op.hpp>
#include <biest/kernel.hpp>
#include <biest/surface_op.hpp>
#include <biest/surface.hpp>
#include <sctl.hpp>

namespace biest {

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void VacuumField<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::Compute(sctl::Vector<Real>& B, sctl::Vector<Real>& J, Real tor_flux, Real pol_flux, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter) {
      sctl::Vector<sctl::Vector<Real>> B_, J_;
      sctl::Vector<Real> tor_flux_, pol_flux_;
      ComputeHelper(B_, J_, tor_flux_, pol_flux_, Svec, comm, gmres_tol, gmres_iter);
      if (Svec.Dim() == 1) {
        B = B_[0] * tor_flux / tor_flux_[0];
        J = J_[0] * tor_flux / tor_flux_[0];
      } else if (Svec.Dim() == 2) {
        Real alpha = (tor_flux * pol_flux_[1] - pol_flux * tor_flux_[1]) / (tor_flux_[0] * pol_flux_[1] - tor_flux_[1] * pol_flux_[0]);
        Real beta  = (pol_flux * tor_flux_[0] - tor_flux * pol_flux_[0]) / (tor_flux_[0] * pol_flux_[1] - tor_flux_[1] * pol_flux_[0]);
        B = B_[0] * alpha + B_[1] * beta;
        J = J_[0] * alpha + J_[1] * beta;
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void VacuumField<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::EvalOffSurface(sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& Xtrg, sctl::Vector<Surface<Real>> Svec, sctl::Vector<Real> J0, const sctl::Comm& comm) {
      sctl::Vector<sctl::Vector<Real>> Xsrc, dXsrc, Xn_src, Xa_src, J;
      { // Upsample
        sctl::Long Nsurf = Svec.Dim();
        Xsrc .ReInit(Nsurf);
        dXsrc .ReInit(Nsurf);
        Xn_src.ReInit(Nsurf);
        Xa_src.ReInit(Nsurf);
        J.ReInit(Nsurf);
        sctl::Long dsp = 0;
        for (sctl::Long i = 0; i < Nsurf; i++) {
          const auto& S = Svec[i];
          sctl::Long Nt0 = S.NTor();
          sctl::Long Np0 = S.NPol();
          sctl::Long Nup = Nt0*Np0*UPSAMPLE*UPSAMPLE;
          SurfaceOp<Real> Sop(comm, Nt0*UPSAMPLE, Np0*UPSAMPLE);

          Sop.Upsample(S.Coord(), Nt0, Np0, Xsrc[i], Nt0*UPSAMPLE, Np0*UPSAMPLE);
          Sop.Grad2D(dXsrc[i], Xsrc[i]);
          Sop.SurfNormalAreaElem(&Xn_src[i], &Xa_src[i], dXsrc[i], &Xsrc[i]);

          sctl::Vector<Real> J0_(COORD_DIM*Nt0*Np0, J0.begin()+COORD_DIM*dsp, false);
          Sop.Upsample(J0_, Nt0, Np0, J[i], Nt0*UPSAMPLE, Np0*UPSAMPLE);

          for (sctl::Long j = 0; j < Nup; j++) { // J <-- J * Xa
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              J[i][k * Nup + j] *= Xa_src[i][j];
            }
          }

          dsp += Nt0 * Np0;
        }
      }
      { // Compute Btrg at Xtrg
        sctl::Vector<Real> X = Xtrg, B;
        sctl::Matrix<Real>(COORD_DIM, X.Dim()/COORD_DIM, X.begin(), false) = sctl::Matrix<Real>(X.Dim()/COORD_DIM, COORD_DIM, X.begin(), false).Transpose();
        ComputeFarField(B, X, Xsrc, Xn_src, J);
        sctl::Matrix<Real>(B.Dim()/COORD_DIM, COORD_DIM, B.begin(), false) = sctl::Matrix<Real>(COORD_DIM, B.Dim()/COORD_DIM, B.begin(), false).Transpose();
        Btrg = B;
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void VacuumField<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::test(sctl::Long upsample, const sctl::Comm& comm) {
      sctl::Vector<Real> B, J;
      sctl::Vector<Surface<Real>> Svec;
      Svec.PushBack(Surface<Real>( 50*upsample+1, 20*upsample+1, SurfType::AxisymCircleWide));
      Svec.PushBack(Surface<Real>(100*upsample+1, 20*upsample+1, SurfType::Quas3));
      Compute(B, J, 1, 1, Svec, comm, 1e-8, 30);
      WriteVTK("B", Svec, B);
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> Real VacuumField<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void VacuumField<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::ComputeFarField(sctl::Vector<Real>& B, const sctl::Vector<Real>& Xt, const sctl::Vector<sctl::Vector<Real>>& Xsrc, const sctl::Vector<sctl::Vector<Real>>& Xn_src, const sctl::Vector<sctl::Vector<Real>>& J) {
      sctl::Long Nsurf = Xsrc.Dim();
      sctl::Vector<sctl::Long> SurfDim(Nsurf), SurfDsp(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set SurfDim, SurfDsp
        SurfDim[i] = Xsrc[i].Dim() / COORD_DIM;
        if (i) SurfDsp[i] = SurfDsp[i-1] + SurfDim[i-1];
        else SurfDsp[i] = 0;
      }

      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == Nt * COORD_DIM);
      if (B.Dim() != COORD_DIM * Nt) B.ReInit(COORD_DIM * Nt);
      B = 0;

      const auto& ker_biot_savart = BiotSavart3D<Real>::FxU();
      for (sctl::Long i = 0; i < Nsurf; i++) {
        ker_biot_savart(Xsrc[i], Xn_src[i], J[i], Xt, B);
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void VacuumField<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::ComputeHelper(sctl::Vector<sctl::Vector<Real>>& B, sctl::Vector<sctl::Vector<Real>>& J, sctl::Vector<Real>& tor_flux, sctl::Vector<Real>& pol_flux, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter) {
      sctl::Long Nsurf = Svec.Dim();
      sctl::Vector<sctl::Long> SurfDim(Nsurf), SurfDsp(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set SurfDim, SurfDsp
        SurfDim[i] = Svec[i].NTor() * Svec[i].NPol();
        if (i) SurfDsp[i] = SurfDsp[i-1] + SurfDim[i-1];
        else SurfDsp[i] = 0;
      }

      sctl::Vector<Real> SurfArea(Svec.Dim());
      sctl::Vector<sctl::Vector<Real>> dX(Svec.Dim()), Xn(Svec.Dim()), Xa(Svec.Dim());
      for (sctl::Long i = 0; i < Svec.Dim(); i++) { // Set dX, Xn, Xa
        const auto& S = Svec[i];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();

        SurfaceOp<Real> surf_op(comm, Nt, Np);
        surf_op.Grad2D(dX[i], S.Coord());
        surf_op.SurfNormalAreaElem(&Xn[i], &Xa[i], dX[i], &S.Coord());

        SurfArea[i] = 0;
        for (const auto& da : Xa[i]) SurfArea[i] += da;
      }
      { // Set normal to outer-normal
        sctl::Long OuterSurfIdx = 0, InnerSurfIdx = 0;
        for (sctl::Long i = 0; i < Svec.Dim(); i++) { // TODO: this is not reliable, use the normal direction to do this
          if (SurfArea[i] < SurfArea[InnerSurfIdx]) InnerSurfIdx = i;
          if (SurfArea[i] > SurfArea[OuterSurfIdx]) OuterSurfIdx = i;
        }
        for (sctl::Long i = 0; i < Svec.Dim(); i++) { // Set normal to outer-normal
          Real scal = 0;
          if (i == InnerSurfIdx) scal = -1;
          if (i == OuterSurfIdx) scal = 1;
          Xn[i] *= scal;
        }
      }

      sctl::Profile::Tic("Setup", &comm);
      const auto& ker_laplace = Laplace3D<Real>::FxU();
      const auto& ker_lap_grad = Laplace3D<Real>::FxdU();
      const auto& ker_biot_savart = BiotSavart3D<Real>::FxU();
      BoundaryIntegralOp<Real,3,3,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_biot_savart(comm);
      BoundaryIntegralOp<Real,1,3,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_lap_grad(comm);
      BoundaryIntegralOp<Real,1,1,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_laplace(comm);
      BI_biot_savart.SetupSingular(Svec, ker_biot_savart);
      BI_lap_grad.SetupSingular(Svec, ker_lap_grad);
      BI_laplace.SetupSingular(Svec, ker_laplace);
      sctl::Profile::Toc();

      sctl::Profile::Tic("ComputeJ", &comm);
      sctl::Vector<sctl::Vector<Real>> J0(Nsurf);
      auto ComputeJ = [&Svec,&SurfDim,&SurfDsp,&dX,&Xn,&comm](sctl::Vector<Real>& J, sctl::Integer Sidx, Real gmres_tol, sctl::Long max_iter) { // Set Jp
        const auto& S = Svec[Sidx];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();
        sctl::Long N = Nt * Np;

        std::cout << "\033[1;31m";
        sctl::Vector<Real> V(COORD_DIM * N), DivV, InvLapDivV, GradInvLapDivV;
        for (sctl::Long i = 0; i < N; i++) { // Set V
          for (sctl::Long k =0; k < COORD_DIM; k++) {
            V[k * N + i] = dX[Sidx][(k*2+0) * N + i] + dX[Sidx][(k*2+1) * N + i];
          }
        }
        SurfaceOp<Real> surf_op(comm, Nt, Np);
        surf_op.SurfDiv(DivV, dX[Sidx], V);
        surf_op.GradInvSurfLap(GradInvLapDivV, dX[Sidx], DivV, gmres_tol * max_norm(V) / max_norm(DivV), max_iter, 1.5);
        V = V - GradInvLapDivV;
        if (0) { // Print err
          surf_op.SurfDiv(DivV, dX[Sidx], V);
          Real err = 0;
          for (const auto x : DivV) err = std::max(err, fabs(x));
          std::cout<<err<<'\n';
        }
        std::cout<<"\033[0m";

        { // Set J
          sctl::Long Nsurf = SurfDim.Dim();
          J.ReInit(COORD_DIM * (SurfDim[Nsurf-1] + SurfDsp[Nsurf-1]));
          J = 0;
          sctl::Vector<Real> J_(COORD_DIM * SurfDim[Sidx], J.begin() + COORD_DIM * SurfDsp[Sidx], false);
          J_ = V;
        }
      };
      for (sctl::Long i = 0; i < Nsurf; i++) ComputeJ(J0[i], i, gmres_tol * 0.01, gmres_iter); //-1);
      sctl::Profile::Toc();

      sctl::Profile::Tic("ComputeB0", &comm);
      sctl::Vector<sctl::Vector<Real>> B0(Nsurf);
      auto eval_B0 = [Nsurf,&SurfDim,&SurfDsp,&Xn,&BI_biot_savart](sctl::Vector<Real>& B, const sctl::Vector<Real>& J) { // B <-- BiotSavart[J] + 0.5 * Xn x J
        sctl::Long N = J.Dim();
        sctl::Vector<Real> NcrossJ(N);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Set NcrossJ
          sctl::Long N = SurfDim[i];
          const sctl::Vector<Real> J_(COORD_DIM * N, (sctl::Iterator<Real>)J.begin() + COORD_DIM  * SurfDsp[i], false);
          sctl::Vector<Real> NcrossJ_(COORD_DIM * N, NcrossJ.begin() + COORD_DIM  * SurfDsp[i], false);
          for (sctl::Long j = 0; j < N; j++) {
            sctl::StaticArray<Real, COORD_DIM> Xn_, V0_, nxV0_;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Xn_[k] = Xn[i][k * N + j];
              V0_[k] = J_[k * N + j];
            }

            nxV0_[0] = Xn_[1] * V0_[2] - Xn_[2] * V0_[1];
            nxV0_[1] = Xn_[2] * V0_[0] - Xn_[0] * V0_[2];
            nxV0_[2] = Xn_[0] * V0_[1] - Xn_[1] * V0_[0];

            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              NcrossJ_[k * N + j] = nxV0_[k];
            }
          }
        }
        BI_biot_savart(B, J);
        B += NcrossJ * 0.5;
      };
      for (sctl::Long i = 0; i < Nsurf; i++) eval_B0(B0[i], J0[i]);
      sctl::Profile::Toc();

      auto eval_B = [Nsurf,&SurfDim,SurfDsp,&Xn,&BI_lap_grad](sctl::Vector<Real>& B, const sctl::Vector<Real>& F) { // B <-- dG[F] + 0.5 * Xn * F
        if (B.Dim() != BI_lap_grad.Dim(1)) {
          B.ReInit(BI_lap_grad.Dim(1));
          B = 0;
        }
        BI_lap_grad(B, F);
        for (sctl::Long j = 0; j < Nsurf; j++) {
          sctl::Long N = SurfDim[j];
          sctl::Long offset = SurfDsp[j];
          for (sctl::Long i = 0; i < N; i++) { // B <-- B + 0.5 * Xn * F
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              B[COORD_DIM*offset + k*N+i] += 0.5 * Xn[j][k*N+i] * F[offset + i];
            }
          }
        }
      };
      auto eval_BdotXn = [Nsurf,&SurfDim,SurfDsp,&Xn](sctl::Vector<Real>& BdotXn, const sctl::Vector<Real>& B) { // BdotXn <-- B . Xn
        sctl::Long N = SurfDsp[Nsurf-1] + SurfDim[Nsurf-1];
        if (BdotXn.Dim() != N) BdotXn.ReInit(N);
        for (sctl::Long j = 0; j < Nsurf; j++) {
          sctl::Long N = SurfDim[j];
          sctl::Long offset = SurfDsp[j];
          for (sctl::Long i = 0; i < N; i++) { // BdotXn = B . Xn
            BdotXn[offset + i] = 0;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              BdotXn[offset + i] += B[COORD_DIM*offset + k*N+i] * Xn[j][k*N+i];
            }
          }
        }
      };
      typename sctl::ParallelSolver<Real>::ParallelOp fn = [&eval_B,&eval_BdotXn](sctl::Vector<Real>* BdotXn, const sctl::Vector<Real>& F) {
        SCTL_ASSERT(BdotXn);
        sctl::Vector<Real> B;
        eval_B(B, F);
        eval_BdotXn(*BdotXn, B);
      };
      auto compute_J = [&Nsurf,&SurfDim,&SurfDsp,&Xn](sctl::Vector<Real>& J, const sctl::Vector<Real>& B) { // J <-- B x Xn
        J.ReInit(B.Dim());
        for (sctl::Long i = 0; i < Nsurf; i++) {
          sctl::Long N = SurfDim[i];
          sctl::Long offset = SurfDsp[i];
          for (sctl::Long j = 0; j < N; j++) {
            sctl::StaticArray<Real, COORD_DIM> J_, B_, Xn_;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              B_[k] = B[COORD_DIM*offset + k*N + j];
              Xn_[k] = Xn[i][k*N + j];
            }
            J_[0] = B_[1]*Xn_[2] - B_[2]*Xn_[1];
            J_[1] = B_[2]*Xn_[0] - B_[0]*Xn_[2];
            J_[2] = B_[0]*Xn_[1] - B_[1]*Xn_[0];
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              J[COORD_DIM*offset + k*N + j] = J_[k];
            }
          }
        }
      };

      sctl::Profile::Tic("Solve", &comm);
      { // Compute B, J
        B.ReInit(Nsurf);
        J.ReInit(Nsurf);
        sctl::Vector<Real> u0dotXn, F;
        sctl::ParallelSolver<Real> solver(comm, true);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Compute B[i], J[i]
          B[i] = B0[i];
          eval_BdotXn(u0dotXn, B[i]);
          solver(&F, fn, u0dotXn*(-1), gmres_tol, gmres_iter);
          eval_B(B[i], F);
          compute_J(J[i], B[i]);
        }
      }
      sctl::Profile::Toc();

      auto compute_tor_circ = [&Svec,&dX](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) {
        sctl::Long offset = 0;
        SCTL_ASSERT(surf_id < Svec.Dim());
        for (sctl::Long i = 0; i < surf_id; i++) {
          const auto& S = Svec[i];
          sctl::Long Nt = S.NTor();
          sctl::Long Np = S.NPol();
          offset += Nt * Np;
        }
        const auto& S = Svec[surf_id];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();

        Real circ_ = 0;
        if (circ && circ->Dim() != Np) circ->ReInit(Np);
        for (sctl::Long p = 0; p < Np; p++) {
          Real integ0 = 0;
          for (sctl::Long t = 0; t < Nt; t++) {
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Real A0 = A[COORD_DIM*offset + k*Nt*Np + t*Np + p];
              Real dX_ = dX[surf_id][(2*k+0)*Nt*Np + t*Np + p];
              integ0 += A0 * dX_;
            }
          }
          if (circ) circ[0][p] = integ0 / Nt;
          circ_ += integ0;
        }
        return circ_ / (Nt*Np);
      };
      auto compute_pol_circ = [&Svec,&dX](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) {
        sctl::Long offset = 0;
        SCTL_ASSERT(surf_id < Svec.Dim());
        for (sctl::Long i = 0; i < surf_id; i++) {
          const auto& S = Svec[i];
          sctl::Long Nt = S.NTor();
          sctl::Long Np = S.NPol();
          offset += Nt * Np;
        }
        const auto& S = Svec[surf_id];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();

        Real circ_ = 0;
        if (circ && circ->Dim() != Nt) circ->ReInit(Nt);
        for (sctl::Long t = 0; t < Nt; t++) {
          Real integ0 = 0;
          for (sctl::Long p = 0; p < Np; p++) {
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Real A0 = A[COORD_DIM*offset + k*Nt*Np + t*Np + p];
              Real dX_ = dX[surf_id][(2*k+1)*Nt*Np + t*Np + p];
              integ0 += A0 * dX_;
            }
          }
          if (circ) circ[0][t] = integ0 / Np;
          circ_ += integ0;
        }
        return circ_ / (Nt*Np);
      };

#ifdef BIEST_VERBOSE
      if (1) { // Check circulation error
        Real max_err = 0;
        for (sctl::Long i = 0; i < Svec.Dim(); i++) {
          sctl::Vector<Real> circ_t, circ_p;
          for (sctl::Long j = 0; j < Nsurf; j++) {
            Real circ_t_avg = compute_tor_circ(&circ_t, B[j], i);
            Real circ_p_avg = compute_pol_circ(&circ_p, B[j], i);
            for (const auto& x : circ_t) max_err = std::max(max_err, fabs(x-circ_t_avg));
            for (const auto& x : circ_p) max_err = std::max(max_err, fabs(x-circ_p_avg));
          }
        }
        std::cout<<"Circulation error: "<<max_err<<'\n';
      }
      if (0) { // Print error || B - BiotSavart(J) ||
        auto print_err = [&BI_biot_savart](const sctl::Vector<Real>& J, const sctl::Vector<Real>& B) {
          sctl::Vector<Real> B0;
          BI_biot_savart(B0, J*2);
          { // Print error
            auto Berr = B0 - B;
            Real max_err = 0, B_norm = 0;
            for (const auto& x : Berr) max_err = std::max(max_err, fabs(x));
            for (const auto& x : B) B_norm = std::max(B_norm, fabs(x));
            std::cout<<"Relative error : "<<max_err / B_norm<<'\n';
          }
        };
        for (sctl::Long i = 0; i < Nsurf; i++) print_err(J[i], B[i]);
      }
#endif

      sctl::Profile::Tic("CompFlux", &comm);
      { // Compute flux
        sctl::Vector<Real> A;
        tor_flux.ReInit(Nsurf);
        pol_flux.ReInit(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) {
          BI_laplace(A, J[i]);
          tor_flux[i] = -compute_pol_circ(nullptr, A, i);
          pol_flux[i] = -compute_tor_circ(nullptr, A, i);
        }
      }
      sctl::Profile::Toc();
    }



    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::Compute(sctl::Vector<Real>& B_out, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol, sctl::Vector<sctl::Vector<Real>>* m, sctl::Vector<sctl::Vector<Real>>* sigma) {
      sctl::Vector<sctl::Vector<Real>> B, m_, sigma_;
      ComputeHelper(B, m_, sigma_, lambda, Svec, comm, gmres_tol, gmres_iter, LB_tol);
      if (B.Dim() > 0) B_out = B[0] * tor_flux;
      if (B.Dim() > 1) B_out += B[1] * pol_flux;
      if (m && sigma) {
        *m = m_;
        *sigma = sigma_;
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::EvalOffSurface(sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& Xtrg, sctl::Vector<Surface<Real>> Svec, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<sctl::Vector<Real>> m_, const sctl::Vector<sctl::Vector<Real>>& sigma_, const sctl::Comm& comm) {
      sctl::Vector<Real> m0, sigma0;
      if (Svec.Dim() > 0) m0  = m_[0] * tor_flux;
      if (Svec.Dim() > 1) m0 += m_[1] * pol_flux;
      if (Svec.Dim() > 0) sigma0  = sigma_[0] * tor_flux;
      if (Svec.Dim() > 1) sigma0 += sigma_[1] * pol_flux;

      sctl::Vector<sctl::Vector<Real>> Xsrc, dXsrc, Xn_src, Xa_src, m, sigma;
      { // Upsample
        sctl::Long Nsurf = Svec.Dim();
        Xsrc .ReInit(Nsurf);
        dXsrc .ReInit(Nsurf);
        Xn_src.ReInit(Nsurf);
        Xa_src.ReInit(Nsurf);
        sigma.ReInit(Nsurf);
        m.ReInit(Nsurf);
        sctl::Long dsp = 0;
        for (sctl::Long i = 0; i < Nsurf; i++) {
          const auto& S = Svec[i];
          sctl::Long Nt0 = S.NTor();
          sctl::Long Np0 = S.NPol();
          sctl::Long Nup = Nt0*Np0*UPSAMPLE*UPSAMPLE;
          SurfaceOp<Real> Sop(comm, Nt0*UPSAMPLE, Np0*UPSAMPLE);

          Sop.Upsample(S.Coord(), Nt0, Np0, Xsrc[i], Nt0*UPSAMPLE, Np0*UPSAMPLE);
          Sop.Grad2D(dXsrc[i], Xsrc[i]);
          Sop.SurfNormalAreaElem(&Xn_src[i], &Xa_src[i], dXsrc[i], &Xsrc[i]);

          sctl::Vector<Real> m0_(COORD_DIM*2*Nt0*Np0, m0.begin()+COORD_DIM*2*dsp, false);
          Sop.Upsample(m0_, Nt0, Np0, m[i], Nt0*UPSAMPLE, Np0*UPSAMPLE);
          for (sctl::Long j = 0; j < Nup; j++) { // m <-- m * Xa
            for (sctl::Long k = 0; k < COORD_DIM*2; k++) {
              m[i][k * Nup + j] *= Xa_src[i][j];
            }
          }

          sctl::Vector<Real> sigma0_(2*Nt0*Np0, sigma0.begin()+2*dsp, false);
          Sop.Upsample(sigma0_, Nt0, Np0, sigma[i], Nt0*UPSAMPLE, Np0*UPSAMPLE);
          for (sctl::Long j = 0; j < Nup; j++) { // sigma <-- sigma * Xa
            for (sctl::Long k = 0; k < 2; k++) {
              sigma[i][k * Nup + j] *= Xa_src[i][j];
            }
          }

          dsp += Nt0 * Np0;
        }
      }
      { // Compute Btrg at Xtrg
        sctl::Vector<Real> X = Xtrg, B;
        sctl::Matrix<Real>(COORD_DIM, X.Dim()/COORD_DIM, X.begin(), false) = sctl::Matrix<Real>(X.Dim()/COORD_DIM, COORD_DIM, X.begin(), false).Transpose();
        ComputeFarField(B, X, Xsrc, Xn_src, sigma, m, lambda);
        sctl::Matrix<Real>(B.Dim()/COORD_DIM, COORD_DIM, B.begin(), false) = sctl::Matrix<Real>(COORD_DIM, B.Dim()/COORD_DIM, B.begin(), false).Transpose();
        Btrg = B;
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::test_conv(Real lambda, const sctl::Vector<Surface<Real>>& Svec_, sctl::Long upsample, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm, Real LB_tol) {
      auto append_vec = [](const sctl::Vector<Real>& a, const sctl::Vector<Real>& b) {
        sctl::Vector<Real> c(a.Dim()+b.Dim());
        for (sctl::Long i = 0; i < a.Dim(); i++) c[i] = a[i];
        for (sctl::Long i = 0; i < b.Dim(); i++) c[a.Dim() + i] = b[i];
        return c;
      };

      sctl::Vector<Surface<Real>> Svec;
      for (const auto& S0 : Svec_) {
        Surface<Real> S(S0.NTor()*upsample, S0.NPol()*upsample);
        SurfaceOp<Real>::Upsample(S0.Coord(), S0.NTor(), S0.NPol(), S.Coord(), S.NTor(), S.NPol());
        Svec.PushBack(S);
      }

      sctl::Vector<Real> B0;
      auto ref_sol = [&lambda](sctl::Vector<Real>& B, const sctl::Vector<Real>& Xt) {
        Helmholtz3D<Real> helm(lambda);
        const auto& ker_potn = helm.FxU();
        const auto& ker_grad = helm.FxdU();
        const auto pi = sctl::const_pi<Real>();

        sctl::Long N = 1000;
        sctl::Vector<Real> X(COORD_DIM * N);
        sctl::Vector<Real> dX(COORD_DIM * N);
        sctl::Vector<Real> Xn(COORD_DIM * N);
        for (sctl::Long i = 0; i < N; i++) {
          Real t = 2*pi*i/N;

          X[0*N+i] = 0; //5+sin(t);
          X[1*N+i] = 2*(cos(t) + 1); //5+cos(t);
          X[2*N+i] = 2*(sin(t)); //sin(t);
          dX[0*N+i] = 0; // cos(t);
          dX[1*N+i] =-sin(t)*2; //-sin(t);
          dX[2*N+i] = cos(t)*2; // cos(t);

          //X[0*N+i] = 0;
          //X[1*N+i] = 0;
          //X[2*N+i] = tan(t/2);
          //dX[0*N+i] = 0;
          //dX[1*N+i] = 0;
          //dX[2*N+i] = 1/(cos(t/2)*cos(t/2)+1e-3)/2;
        }

        { // B <-- lambda Q + Curl(Q)
          sctl::Long Nt = Xt.Dim() / COORD_DIM;
          B.ReInit(COORD_DIM * ker_potn.Dim(1) * Nt);
          B = 0;

          sctl::Vector<Real>     U(ker_potn.Dim(1) * Nt);
          sctl::Vector<Real> gradU(ker_grad.Dim(1) * Nt);
          sctl::Vector<Real> J(ker_potn.Dim(0) * N);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            J = 0;
            U = 0;
            gradU = 0;
            for (sctl::Long i = 0; i < N; i++) J[i] = dX[k * N + i] / N;
            ker_potn(X, Xn, J, Xt,     U);
            ker_grad(X, Xn, J, Xt, gradU);

            sctl::Integer k1 = (k + 1) % COORD_DIM;
            sctl::Integer k2 = (k + 2) % COORD_DIM;
            for (sctl::Long i = 0; i < 2 * Nt; i++) {
              B[k  * 2 * Nt + i] += lambda * U[i];
              B[k1 * 2 * Nt + i] += gradU[k2 * 2 * Nt + i];
              B[k2 * 2 * Nt + i] -= gradU[k1 * 2 * Nt + i];
            }
          }
        }
      };
      for (sctl::Long k = 0; k < Svec.Dim(); k++) {
        sctl::Vector<Real> B_;
        ref_sol(B_, Svec[k].Coord());
        B0 = append_vec(B0, B_);
      }
      // TODO: Check normal-flux = 0

      sctl::Profile::Tic("TaylorConv", &comm);
      { // Compute BI solution
        sctl::Long Nsurf = Svec.Dim();
        if (Nsurf != 1 && Nsurf != 2) return;
        sctl::Vector<sctl::Long> SurfDim(Nsurf), SurfDsp(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Set SurfDim, SurfDsp
          SurfDim[i] = Svec[i].NTor() * Svec[i].NPol();
          if (i) SurfDsp[i] = SurfDsp[i-1] + SurfDim[i-1];
          else SurfDsp[i] = 0;
        }

        sctl::Vector<sctl::Vector<Real>> B_taylor;
        { // Set B_taylor
          sctl::Vector<sctl::Vector<Real>> B, m, sigma;
          ComputeHelper(B, m, sigma, lambda, Svec, comm, gmres_tol, gmres_iter);
          auto append_imag_part = [&SurfDim,&SurfDsp] (sctl::Vector<Real>& w, const sctl::Vector<Real>& v) {
            sctl::Long Nsurf = SurfDim.Dim();
            SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
            sctl::Long N = SurfDsp[Nsurf-1]+SurfDim[Nsurf-1];
            sctl::Long dof = v.Dim() / N;
            SCTL_ASSERT(v.Dim() == N * dof);
            if (w.Dim() != N * dof) w.ReInit(dof * 2 * N);
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < dof; k++) {
                sctl::Long offset = dof*SurfDsp[i];
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  w[2*offset + (k*2+0)*SurfDim[i] + j] = v[offset + k*SurfDim[i] + j];
                  w[2*offset + (k*2+1)*SurfDim[i] + j] = 0;
                }
              }
            }
          };
          B_taylor.ReInit(B.Dim());
          for (sctl::Long i = 0; i < B.Dim(); i++) append_imag_part(B_taylor[i], B[i]);
        }

        sctl::Vector<Real> SurfArea(Nsurf); // TODO: remove
        sctl::Long OuterSurfIdx = 0, InnerSurfIdx = 0;
        sctl::Vector<sctl::Vector<Real>> dX(Nsurf), Xn(Nsurf), Xa(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Set dX, Xn, Xa
          const auto& S = Svec[i];
          sctl::Long Nt = S.NTor();
          sctl::Long Np = S.NPol();

          SurfaceOp<Real> surf_op(comm, Nt, Np);
          surf_op.Grad2D(dX[i], S.Coord());
          surf_op.SurfNormalAreaElem(&Xn[i], &Xa[i], dX[i], &S.Coord());

          SurfArea[i] = 0;
          for (const auto& da : Xa[i]) SurfArea[i] += da;
        }
        for (sctl::Long i = 0; i < Nsurf; i++) { // TODO: this is not reliable, use the normal direction to do this
          if (SurfArea[i] < SurfArea[InnerSurfIdx]) InnerSurfIdx = i;
          if (SurfArea[i] > SurfArea[OuterSurfIdx]) OuterSurfIdx = i;
        }
        for (sctl::Long i = 0; i < Nsurf; i++) { // Set normal to outer-normal
          Real scal = 0;
          if (i == InnerSurfIdx) scal = -1;
          if (i == OuterSurfIdx) scal = 1;
          Xn[i] *= scal;
        }

        sctl::Profile::Tic("Setup", &comm);
        Helmholtz3D<Real> helm(lambda);
        HelmholtzDiff3D<Real> helm_diff(lambda);
        const auto& ker_potn = helm.FxU();
        const auto& ker_grad = helm.FxdU();
        const auto& ker_grad_diff = helm_diff.FxdU();
        BoundaryIntegralOp<Real,2,6,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_grad_diff(comm);
        BoundaryIntegralOp<Real,2,6,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_grad(comm);
        BoundaryIntegralOp<Real,2,2,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_potn(comm);
        BI_grad_diff.SetupSingular(Svec, ker_grad_diff);
        BI_grad.SetupSingular(Svec, ker_grad);
        BI_potn.SetupSingular(Svec, ker_potn);
        sctl::Profile::Toc();

        auto Compute_m = [lambda,&Svec,&SurfDim,&SurfDsp,&dX,&Xn,&comm](sctl::Vector<Real>& m, const sctl::Vector<Real>& sigma, Real gmres_tol, sctl::Long gmres_iter) { // m <-- lambda ( i Grad(invLap(sigma)) - Grad(invLap(sigma)) x n )
          sctl::Long Nsurf = SurfDim.Dim();
          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
          if (m.Dim() != COORD_DIM * 2 * N) m.ReInit(COORD_DIM * 2 * N);
          if (!sigma.Dim() || lambda == 0) { // m = 0
            m = 0;
          } else {
            for (sctl::Long i = 0; i < Nsurf; i++) {
              const sctl::Vector<Real> sigma_(2 * SurfDim[i], (sctl::Iterator<Real>)sigma.begin() + 2 * SurfDsp[i], false);
              sctl::Vector<Real> m_(COORD_DIM * 2 * SurfDim[i], m.begin() + COORD_DIM * 2 * SurfDsp[i], false);

              const auto& S = Svec[i];
              sctl::Long Nt = S.NTor();
              sctl::Long Np = S.NPol();
              sctl::Long N= Nt * Np;

              SurfaceOp<Real> surf_op(comm, Nt, Np);
              sctl::StaticArray<sctl::Vector<Real>, 2> GradInvLapSigma;
              for (sctl::Long k = 0; k < 2; k++) { // Set GradInvLapSigma <-- Grad(invLap(sigma))
                const sctl::Vector<Real> sigma__(N, (sctl::Iterator<Real>)sigma_.begin() + k * N, false);
                if (max_norm(sigma__) > 0) {
                  surf_op.GradInvSurfLap(GradInvLapSigma[k], dX[i], sigma__, gmres_tol, gmres_iter, 1.5);
                } else {
                  GradInvLapSigma[k].ReInit(COORD_DIM * N);
                  GradInvLapSigma[k].SetZero();
                }
              }
              for (sctl::Long j = 0; j < N; j++) { // m <-- lambda ( i GradInvLapSigma - n x GradInvLapSigma )
                sctl::StaticArray<Real, COORD_DIM> Xn_, V0_, V1_, nxV0_, nxV1_;
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  Xn_[k] = Xn[i][k * N + j];
                  V0_[k] = GradInvLapSigma[0][k * N + j];
                  V1_[k] = GradInvLapSigma[1][k * N + j];
                }

                nxV0_[0] = Xn_[1] * V0_[2] - Xn_[2] * V0_[1];
                nxV0_[1] = Xn_[2] * V0_[0] - Xn_[0] * V0_[2];
                nxV0_[2] = Xn_[0] * V0_[1] - Xn_[1] * V0_[0];

                nxV1_[0] = Xn_[1] * V1_[2] - Xn_[2] * V1_[1];
                nxV1_[1] = Xn_[2] * V1_[0] - Xn_[0] * V1_[2];
                nxV1_[2] = Xn_[0] * V1_[1] - Xn_[1] * V1_[0];

                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  m_[(k*2+0) * N + j] = lambda * (-V1_[k] - nxV0_[k]);
                  m_[(k*2+1) * N + j] = lambda * ( V0_[k] - nxV1_[k]);
                }
              }
            }
          }
          if (1) {
            auto extract_comp = [&SurfDim,&SurfDsp,Nsurf,N](const sctl::Vector<Real>& in, sctl::Long surf_id, sctl::Long comp_id) {
              sctl::Long dof = in.Dim() / N;
              SCTL_ASSERT(in.Dim() == N * dof);
              sctl::Vector<Real> out(SurfDim[surf_id]);
              for (sctl::Long i = 0; i < SurfDim[surf_id]; i++) {
                out[i] = in[dof * SurfDsp[surf_id] + comp_id * SurfDim[surf_id] + i];
              }
              return out;
            };
            auto concat_vec = [](const sctl::Vector<Real>& x, const sctl::Vector<Real>& y) {
              sctl::Vector<Real> out;
              for (const auto& a : x) out.PushBack(a);
              for (const auto& a : y) out.PushBack(a);
              return out;
            };
            { // Print error <-- | SurfDiv(m) - i lambda sigma |_inf
              Real err = 0;
              for (sctl::Long i = 0; i < Nsurf; i++) {
                const auto& S = Svec[i];
                sctl::Vector<Real> m_real, m_imag;
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  m_real = concat_vec(m_real, extract_comp(m, i, 2*k+0));
                  m_imag = concat_vec(m_imag, extract_comp(m, i, 2*k+1));
                }

                sctl::Vector<Real> Divm_real, Divm_imag;
                SurfaceOp<Real> surf_op(comm, S.NTor(), S.NPol());
                surf_op.SurfDiv(Divm_real, dX[i], m_real);
                surf_op.SurfDiv(Divm_imag, dX[i], m_imag);

                sctl::Vector<Real> sigma_real, sigma_imag;
                if (sigma.Dim()) {
                  sigma_real = extract_comp(sigma, i, 0);
                  sigma_imag = extract_comp(sigma, i, 1);
                } else {
                  sigma_real = Divm_real * 0;
                  sigma_imag = Divm_imag * 0;
                }

                err = std::max(err, max_norm(Divm_real + sigma_imag * lambda));
                err = std::max(err, max_norm(Divm_imag - sigma_real * lambda));
              }
              std::cout<<"Error : | SurfDiv(m) - i lambda sigma |_inf = "<<err<<";   | m | = "<<max_norm(m)<<";   | lambda sigma | = "<<lambda * max_norm(sigma)<<'\n';
            }
            { // Print error <-- n x m + i m
              Real err = 0;
              for (sctl::Long i = 0; i < Nsurf; i++) {
                sctl::Vector<Real> m_real, m_imag;
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  m_real = concat_vec(m_real, extract_comp(m, i, 2*k+0));
                  m_imag = concat_vec(m_imag, extract_comp(m, i, 2*k+1));
                }

                auto cross_prod = [](const sctl::Vector<Real>& x, const sctl::Vector<Real>& y) {
                  sctl::Long N = x.Dim() / COORD_DIM;
                  sctl::Vector<Real> z(N * COORD_DIM);
                  SCTL_ASSERT(x.Dim() == N * COORD_DIM);
                  SCTL_ASSERT(y.Dim() == N * COORD_DIM);
                  for (sctl::Long i = 0; i < N; i++) {
                    sctl::StaticArray<Real, COORD_DIM> x_, y_, z_;
                    for (sctl::Long k = 0; k < COORD_DIM; k++) {
                      x_[k] = x[k * N + i];
                      y_[k] = y[k * N + i];
                    }
                    z_[0] = x_[1] * y_[2] - x_[2] * y_[1];
                    z_[1] = x_[2] * y_[0] - x_[0] * y_[2];
                    z_[2] = x_[0] * y_[1] - x_[1] * y_[0];
                    for (sctl::Long k = 0; k < COORD_DIM; k++) {
                      z[k * N + i] = z_[k];
                    }
                  }
                  return z;
                };
                sctl::Vector<Real> nxm_real = cross_prod(Xn[i], m_real);
                sctl::Vector<Real> nxm_imag = cross_prod(Xn[i], m_imag);

                err = std::max(err, max_norm(nxm_real - m_imag));
                err = std::max(err, max_norm(nxm_imag + m_real));
              }
              std::cout<<"Error : | n x m + i m |_inf = "<<err<<'\n';
            }
          }
        };
        auto Compute_B = [lambda,&SurfDim,&SurfDsp,&Xn,&BI_grad_diff,&BI_grad,&BI_potn,&ker_grad_diff,&ker_grad,&ker_potn,&comm] (sctl::Vector<Real>* B, sctl::Vector<Real>* B_flux, const sctl::Vector<Real>& sigma, const sctl::Vector<Real>& m0, const sctl::Vector<Real>& mH) {
          sctl::Long Nsurf = SurfDim.Dim();
          SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];

          sctl::Vector<Real> m;
          if (m0.Dim() && mH.Dim()) {
            SCTL_ASSERT(m0.Dim() == mH.Dim());
            m = m0 + mH;
          } else if (m0.Dim()) {
            m = m0;
          } else if (mH.Dim()) {
            m = mH;
          }

          sctl::Vector<Real> Grad_v, iQ, iCurlQ, iCurlQ_m0, iCurlQ_mH;
          if (B && sigma.Dim()) { // Compute Grad_v
            sctl::Profile::Tic("Compute_Grad_v", &comm);
            Compute_Grad_v(Grad_v, sigma, BI_grad, ker_grad);
            sctl::Profile::Toc();
          }
          if (m.Dim()) { // Compute iQ
            sctl::Profile::Tic("Compute_iQ", &comm);
            Compute_iQ(iQ, m, SurfDim, SurfDsp, BI_potn, ker_potn, lambda);
            sctl::Profile::Toc();
          }
          if (B && m.Dim()) { // Compute iCurlQ
            sctl::Profile::Tic("Compute_iCurlQ", &comm);
            Compute_iCurlQ(iCurlQ, m, SurfDim, SurfDsp, BI_grad, ker_grad);
            sctl::Profile::Toc();
          }
          if (B_flux) { // Compute iCurlQ_m0, iCurlQ_mH
            sctl::Profile::Tic("Compute_iCurlQ_", &comm);
            if (m0.Dim()) Compute_iCurlQ(iCurlQ_m0, m0, SurfDim, SurfDsp, BI_grad     , ker_grad     );
            if (mH.Dim()) Compute_iCurlQ(iCurlQ_mH, mH, SurfDim, SurfDsp, BI_grad_diff, ker_grad_diff);
            sctl::Profile::Toc();
          }

          if (B_flux) { // B_flux <-- iQ*lambda + iCurlQ_mH + iCurlQ_m0 + 0.5*m0
            (*B_flux).ReInit(COORD_DIM * 2 * N);
            (*B_flux) = 0;
            if (m .Dim()) (*B_flux) += iQ*lambda;
            if (mH.Dim()) (*B_flux) += iCurlQ_mH;
            if (m0.Dim()) (*B_flux) += iCurlQ_m0 + m0*(0.5);
          }
          if (B) { // B <-- -0.5*sigma_Xn - Grad_v + iQ*lambda + iCurlQ + 0.5*m
            (*B).ReInit(COORD_DIM * 2 * N);
            (*B) = 0;
            if (m.Dim()) (*B) += iQ*lambda + iCurlQ + m*(0.5);
            if (sigma.Dim()) { // B <-- B - 0.5*sigma_Xn - Grad_v
              sctl::Vector<Real> sigma_Xn(COORD_DIM * 2 * N);
              for (sctl::Long i = 0; i < Nsurf; i++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                    sigma_Xn[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j] = Xn[i][k * SurfDim[i] + j] * sigma[2 * SurfDsp[i] + 0 * SurfDim[i] + j];
                    sigma_Xn[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j] = Xn[i][k * SurfDim[i] + j] * sigma[2 * SurfDsp[i] + 1 * SurfDim[i] + j];
                  }
                }
              }
              (*B) += sigma_Xn*(-0.5) - Grad_v;
            }
          }
        };
        auto Compute_BdotXn = [&SurfDim,&SurfDsp,&Xn](sctl::Vector<Real>& BdotXn, const sctl::Vector<Real>& B) {
          sctl::Long Nsurf = SurfDim.Dim();
          SCTL_ASSERT(SurfDsp.Dim() == Nsurf);

          sctl::Long N = B.Dim() / (COORD_DIM * 2);
          SCTL_ASSERT(B.Dim() == COORD_DIM * 2 * N);
          if (BdotXn.Dim() != 2 * N) BdotXn.ReInit(2 * N);
          for (sctl::Long i = 0; i < Nsurf; i++) {
            for (sctl::Long j = 0; j < SurfDim[i]; j++) {
              Real B0dotXn = 0, B1dotXn = 0;
              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                Real Xn_ = Xn[i][k * SurfDim[i] + j];
                Real B0 = B[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j];
                Real B1 = B[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j];
                B0dotXn += B0 * Xn_;
                B1dotXn += B1 * Xn_;
              }
              BdotXn[2 * SurfDsp[i] + 0 * SurfDim[i] + j] = B0dotXn;
              BdotXn[2 * SurfDsp[i] + 1 * SurfDim[i] + j] = B1dotXn;
            }
          }
        };
        auto solve_taylor = [lambda,LB_tol,&comm,&Compute_m,&Compute_B,&Compute_BdotXn] (sctl::Vector<Real>& sigma, const sctl::Vector<Real>& rhs, Real gmres_tol, sctl::Long gmres_iter) {
          sctl::Profile::Tic("Solve", &comm);
          sctl::ParallelSolver<Real> solver(comm, true);
          typename sctl::ParallelSolver<Real>::ParallelOp fn = [lambda,LB_tol,&comm,&Compute_m,&Compute_B,&Compute_BdotXn,&gmres_tol,&gmres_iter](sctl::Vector<Real>* BdotXn, const sctl::Vector<Real>& sigma) {
            SCTL_ASSERT(BdotXn);
            sctl::Vector<Real> m, B;
            sctl::Profile::Tic("Compute_B", &comm);
            Compute_m(m, sigma, gmres_tol / lambda * LB_tol, gmres_iter);
            Compute_B(&B, nullptr, sigma, m, sctl::Vector<Real>());
            Compute_BdotXn(*BdotXn, B);
            { // Project to space of mean-zero functions : BdotXn <-- P BdotXn
              // TODO
            }
            sctl::Profile::Toc();
          };
          solver(&sigma, fn, rhs, gmres_tol, gmres_iter);
          sctl::Profile::Toc();
        };

        sctl::Vector<Real> B;
        { // Set B <-- B0 - B_sigma
          sctl::Vector<Real> rhs, sigma, m;
          Compute_BdotXn(rhs, B0);
          solve_taylor(sigma, rhs*(-1), gmres_tol, gmres_iter);
          Compute_m(m, sigma, gmres_tol / lambda * LB_tol, gmres_iter);
          Compute_B(&B, nullptr, sigma, m, sctl::Vector<Real>());
          B += B0;
        }

        sctl::Vector<Real> B_res;
        sctl::Profile::Tic("CompResid", &comm);
        { // Set B_res <-- B - B_taylor * x
          auto complex_vec_scal_prod = [&SurfDim,&SurfDsp] (sctl::Vector<Real>& v, sctl::Complex<Real> c) {
            sctl::Long Nsurf = SurfDim.Dim();
            SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
            sctl::Long N = SurfDsp[Nsurf-1]+SurfDim[Nsurf-1];
            sctl::Long dof = v.Dim() / (N * 2);
            SCTL_ASSERT(v.Dim() == N * dof * 2);
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < dof; k++) {
                sctl::Long offset = SurfDsp[i]*dof*2 + k*2*SurfDim[i];
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  Real v0 = v[offset + 0*SurfDim[i] + j];
                  Real v1 = v[offset + 1*SurfDim[i] + j];
                  Real cv0 = v0 * c.real - v1 * c.imag;
                  Real cv1 = v0 * c.imag + v1 * c.real;
                  v[offset + 0*SurfDim[i] + j] = cv0;
                  v[offset + 1*SurfDim[i] + j] = cv1;
                }
              }
            }
          };
          auto complex_vec_dot_prod = [&SurfDim,&SurfDsp] (const sctl::Vector<Real>& v0, const sctl::Vector<Real>& v1) {
            sctl::Long Nsurf = SurfDim.Dim();
            SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
            sctl::Long N = SurfDsp[Nsurf-1]+SurfDim[Nsurf-1];
            sctl::Long dof = v0.Dim() / (N * 2);
            SCTL_ASSERT(v0.Dim() == N * dof * 2);
            SCTL_ASSERT(v1.Dim() == N * dof * 2);
            sctl::Complex<Real> v1dotv2(0,0);
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < dof; k++) {
                sctl::Long offset = SurfDsp[i]*dof*2 + k*2*SurfDim[i];
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  sctl::Complex<Real> c0(v0[offset + 0*SurfDim[i] + j], v0[offset + 1*SurfDim[i] + j]);
                  sctl::Complex<Real> c1(v1[offset + 0*SurfDim[i] + j], v1[offset + 1*SurfDim[i] + j]);
                  v1dotv2 += c0 * c1;
                }
              }
            }
            return v1dotv2;
          };

          sctl::Matrix<sctl::Complex<Real>> b(B_taylor.Dim(), 1);
          sctl::Matrix<sctl::Complex<Real>> M(B_taylor.Dim(), B_taylor.Dim());
          for (sctl::Long i = 0; i < B_taylor.Dim(); i++) { // Set b, M
            b[i][0] = complex_vec_dot_prod(B,B_taylor[i]);
            for (sctl::Long j = 0; j < B_taylor.Dim(); j++) {
              M[i][j] = complex_vec_dot_prod(B_taylor[i],B_taylor[j]);
            }
          }

          sctl::Matrix<sctl::Complex<Real>> Minv(B_taylor.Dim(), B_taylor.Dim());
          if (M.Dim(0) == 1 && M.Dim(1) == 1) { // Set Minv <-- inv(M)
            Minv[0][0] = 1 / M[0][0];
          } else if (M.Dim(0) == 2 && M.Dim(1) == 2) {
            sctl::Complex<Real> oodet = 1 / (M[0][0] * M[1][1] - M[0][1] * M[1][0]);
            Minv[0][0] = M[1][1] * oodet;
            Minv[0][1] = -M[0][1] * oodet;
            Minv[1][0] = -M[1][0] * oodet;
            Minv[1][1] = M[0][0] * oodet;
          } else {
            SCTL_ASSERT(false);
          }

          B_res = B;
          auto x = Minv * b;
          for (sctl::Long i = 0; i < B_taylor.Dim(); i++) {
            complex_vec_scal_prod(B_taylor[i], x[i][0]);
            B_res -= B_taylor[i];
          }
        }
        sctl::Profile::Toc();

        std::cout<<"Error: "<<max_norm(B_res)<<" / "<<max_norm(B0)<<'\n';
        WriteVTK("B0", Svec, B0);
        WriteVTK("B1", Svec, B);
        WriteVTK("B2", Svec, B_res);
      }
      sctl::Profile::Toc();
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> Real TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::ComputeFarField(sctl::Vector<Real>& Breal, const sctl::Vector<Real>& Xt, const sctl::Vector<sctl::Vector<Real>>& Xsrc, const sctl::Vector<sctl::Vector<Real>>& Xn_src, const sctl::Vector<sctl::Vector<Real>>& sigma, const sctl::Vector<sctl::Vector<Real>>& m, Real lambda) {
      sctl::Long Nsurf = Xsrc.Dim();
      Helmholtz3D<Real> helm(lambda);
      const auto& ker_potn = helm.FxU();
      const auto& ker_grad = helm.FxdU();

      auto Compute_Grad_v = [&Xt,&Xsrc,&Xn_src,&sigma,&ker_grad,Nsurf](sctl::Vector<Real>& Grad_v) { // grad_v <-- Grad(g[sigma])
        sctl::Long Nt = Xt.Dim() / COORD_DIM;
        SCTL_ASSERT(Xt.Dim() == Nt * COORD_DIM);
        if (Grad_v.Dim() != COORD_DIM * 2 * Nt) Grad_v.ReInit(COORD_DIM * 2 * Nt);
        Grad_v = 0;

        for (sctl::Long i = 0; i < Nsurf; i++) {
          ker_grad(Xsrc[i], Xn_src[i], sigma[i], Xt, Grad_v);
        }
      };
      auto Compute_iQ = [&Xt,&Xsrc,&Xn_src,&m,&ker_potn,Nsurf](sctl::Vector<Real>& iQ) { // iQ <-- g[im]
        sctl::Long Nt = Xt.Dim() / COORD_DIM;
        SCTL_ASSERT(Xt.Dim() == Nt * COORD_DIM);
        if (iQ.Dim() != COORD_DIM * 2 * Nt) iQ.ReInit(COORD_DIM * 2 * Nt);
        iQ = 0;

        for (sctl::Long i = 0; i < Nsurf; i++) {
          sctl::Long N = Xsrc[i].Dim() / COORD_DIM;
          sctl::Vector<Real> iQ_(2 * Nt), im_(2 * N);
          for (sctl::Long k = 0; k < COORD_DIM; k++) {
            for (sctl::Long j = 0; j < N; j++) {
              im_[0 * N + j] = -m[i][(k*2+1) * N + j];
              im_[1 * N + j] =  m[i][(k*2+0) * N + j];
            }

            iQ_ = 0;
            ker_potn(Xsrc[i], Xn_src[i], im_, Xt, iQ_);

            for (sctl::Long j = 0; j < Nt; j++) {
              iQ[(k*2+0) * Nt + j] += iQ_[0 * Nt + j];
              iQ[(k*2+1) * Nt + j] += iQ_[1 * Nt + j];
            }
          }
        }
      };
      auto Compute_iCurlQ = [&Xt,&Xsrc,&Xn_src,&m,&ker_grad,Nsurf](sctl::Vector<Real>& iCurlQ) { // iCurlQ <-- Curl(g[im])
        sctl::Long Nt = Xt.Dim() / COORD_DIM;
        SCTL_ASSERT(Xt.Dim() == Nt * COORD_DIM);
        if (iCurlQ.Dim() != COORD_DIM * 2 * Nt) iCurlQ.ReInit(COORD_DIM * 2 * Nt);
        iCurlQ = 0;

        for (sctl::Long i = 0; i < Nsurf; i++) {
          sctl::Long N = Xsrc[i].Dim() / COORD_DIM;
          sctl::Vector<Real> iGradQ_(2 * COORD_DIM * Nt), im_(2 * N);
          for (sctl::Long k = 0; k < COORD_DIM; k++) {
            for (sctl::Long j = 0; j < N; j++) {
              im_[0 * N + j] = -m[i][(k*2+1) * N + j];
              im_[1 * N + j] =  m[i][(k*2+0) * N + j];
            }

            iGradQ_ = 0;
            ker_grad(Xsrc[i], Xn_src[i], im_, Xt, iGradQ_);

            sctl::Long k0 = (k + 2) % COORD_DIM;
            sctl::Long k1 = (k + 1) % COORD_DIM;
            for (sctl::Long j = 0; j < Nt; j++) {
              iCurlQ[(k1*2+0) * Nt + j] += iGradQ_[(k0*2+0) * Nt + j];
              iCurlQ[(k1*2+1) * Nt + j] += iGradQ_[(k0*2+1) * Nt + j];
              iCurlQ[(k0*2+0) * Nt + j] -= iGradQ_[(k1*2+0) * Nt + j];
              iCurlQ[(k0*2+1) * Nt + j] -= iGradQ_[(k1*2+1) * Nt + j];
            }
          }
        }
      };

      sctl::Long N = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * N);
      sctl::Vector<Real> Grad_v, iQ, iCurlQ;
      if (sigma.Dim()) { // Compute Grad_v
        Compute_Grad_v(Grad_v);
      }
      if (m.Dim()) { // Compute iQ
        Compute_iQ(iQ);
      }
      if (m.Dim()) { // Compute iCurlQ
        Compute_iCurlQ(iCurlQ);
      }

      sctl::Vector<Real> B(COORD_DIM * 2 * N);
      { // B <-- - Grad_v + iQ*lambda + iCurlQ
        B = 0;
        if (m.Dim()) B += iQ*lambda + iCurlQ;
        if (sigma.Dim()) B -= Grad_v;
      }

      { // Breal <-- real(B)
        sctl::Long N = B.Dim() / (2 * COORD_DIM);
        if (Breal.Dim() != COORD_DIM * N) Breal.ReInit(COORD_DIM * N);
        for (sctl::Long i = 0; i < N; i++) {
          for (sctl::Long k = 0; k < COORD_DIM; k++) {
            Breal[k * N + i] = B[k * 2 * N + i];
          }
        }
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::Compute_mH(sctl::Vector<Real>& mH, const Surface<Real>& S, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& Xn, const sctl::Comm& comm, Real tol, sctl::Long max_iter) {
      sctl::Long Nt = S.NTor();
      sctl::Long Np = S.NPol();
      sctl::Long N= Nt * Np;

      sctl::Vector<Real> V(COORD_DIM * N);
      #pragma omp parallel for schedule(static)
      for (sctl::Long i = 0; i < N; i++) { // Set V
        for (sctl::Long k =0; k < COORD_DIM; k++) {
          V[k * N + i] = dX[(k*2+0) * N + i];
        }
      }

      sctl::Vector<Real> Vn, Vd, Vc, Vh;
      SurfaceOp<Real> surf_op(comm, Nt, Np);
      surf_op.HodgeDecomp(Vn, Vd, Vc, Vh, V, dX, Xn, tol, max_iter);

      if (mH.Dim() != COORD_DIM * 2 * N) mH.ReInit(COORD_DIM * 2 * N);
      #pragma omp parallel for schedule(static)
      for (sctl::Long i = 0; i < N; i++) { // Set mH
        sctl::StaticArray<Real, COORD_DIM> Xn_, Vh_, nxVh_;
        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          Xn_[k] = Xn[k * N + i];
          Vh_[k] = Vh[k * N + i];
        }

        nxVh_[0] = Xn_[1] * Vh_[2] - Xn_[2] * Vh_[1];
        nxVh_[1] = Xn_[2] * Vh_[0] - Xn_[0] * Vh_[2];
        nxVh_[2] = Xn_[0] * Vh_[1] - Xn_[1] * Vh_[0];

        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          mH[(k*2+0) * N + i] =   Vh_[k];
          mH[(k*2+1) * N + i] = nxVh_[k];
        }
      }
    }
    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::Compute_Grad_v(sctl::Vector<Real>& Grad_v, const sctl::Vector<Real>& sigma, const BoundaryIntegralOp<Real,2,6,UPSAMPLE,PATCH_DIM0,RAD_DIM>& BI_grad, const KernelFunction<Real,COORD_DIM,2,COORD_DIM*2>& ker_grad) { // grad_v <-- Grad(g[sigma])
      Grad_v = 0;
      BI_grad(Grad_v, sigma);
    }
    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::Compute_iQ(sctl::Vector<Real>& iQ, const sctl::Vector<Real>& m, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const BoundaryIntegralOp<Real,2,2,UPSAMPLE,PATCH_DIM0,RAD_DIM>& BI_potn, const KernelFunction<Real,COORD_DIM,2,2>& ker_potn, Real lambda) { // iQ <-- i g[m]
      sctl::Long Nsurf = SurfDim.Dim();
      sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
      if (iQ.Dim() != COORD_DIM * 2 * N) iQ.ReInit(COORD_DIM * 2 * N);
      SCTL_ASSERT(m.Dim() == COORD_DIM * 2 * N);

      iQ = 0;
      #ifdef USE_QBX
      BI_potn.template HelmholtzQBX<2>(iQ, m, ker_potn, lambda);
      #else
      BI_potn(iQ, m);
      #endif

      for (sctl::Long k = 0; k < COORD_DIM; k++) {
        for (sctl::Long i = 0; i < Nsurf; i++) {
          #pragma omp parallel for schedule(static)
          for (sctl::Long j = 0; j < SurfDim[i]; j++) {
            Real real_Q = iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j];
            Real imag_Q = iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j];
            iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j] = -imag_Q;
            iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j] =  real_Q;
          }
        }
      }
    }
    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::Compute_iCurlQ(sctl::Vector<Real>& iCurlQ, const sctl::Vector<Real>& m, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const BoundaryIntegralOp<Real,2,6,UPSAMPLE,PATCH_DIM0,RAD_DIM>& BI_grad, const KernelFunction<Real,COORD_DIM,2,COORD_DIM*2>& ker_grad) { // iCurlQ <-- i Curl(g[m])
      sctl::Long Nsurf = SurfDim.Dim();
      sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
      if (iCurlQ.Dim() != COORD_DIM * 2 * N) iCurlQ.ReInit(COORD_DIM * 2 * N);
      SCTL_ASSERT(m.Dim() == COORD_DIM * 2 * N);

      sctl::Vector<Real> GradQ_(COORD_DIM * COORD_DIM * 2 * N);
      GradQ_ = 0;
      BI_grad(GradQ_, m);

      iCurlQ = 0;
      for (sctl::Long k = 0; k < COORD_DIM; k++) {
        sctl::Long k0 = (k + 2) % COORD_DIM;
        sctl::Long k1 = (k + 1) % COORD_DIM;
        for (sctl::Long i = 0; i < Nsurf; i++) {
          #pragma omp parallel for schedule(static)
          for (sctl::Long j = 0; j < SurfDim[i]; j++) {
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k1*2+0) * SurfDim[i] + j] += -GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k0)*2+1) * SurfDim[i] + j];
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k1*2+1) * SurfDim[i] + j] +=  GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k0)*2+0) * SurfDim[i] + j];
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k0*2+0) * SurfDim[i] + j] -= -GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k1)*2+1) * SurfDim[i] + j];
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k0*2+1) * SurfDim[i] + j] -=  GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k1)*2+0) * SurfDim[i] + j];
          }
        }
      }
    }

    template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM> void TaylorState<Real,UPSAMPLE,PATCH_DIM0,RAD_DIM>::ComputeHelper(sctl::Vector<sctl::Vector<Real>>& B_out, sctl::Vector<sctl::Vector<Real>>& m_out, sctl::Vector<sctl::Vector<Real>>& sigma_out, Real lambda, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol) {
      sctl::Long Nsurf = Svec.Dim();
      if (Nsurf != 1 && Nsurf != 2) return;
      sctl::Vector<sctl::Long> SurfDim(Nsurf), SurfDsp(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set SurfDim, SurfDsp
        SurfDim[i] = Svec[i].NTor() * Svec[i].NPol();
        if (i) SurfDsp[i] = SurfDsp[i-1] + SurfDim[i-1];
        else SurfDsp[i] = 0;
      }

      sctl::Vector<Real> SurfArea(Nsurf); // TODO: remove
      sctl::Long OuterSurfIdx = 0, InnerSurfIdx = 0;
      sctl::Vector<sctl::Vector<Real>> dX(Nsurf), Xn(Nsurf), Xa(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set dX, Xn, Xa
        const auto& S = Svec[i];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();

        SurfaceOp<Real> surf_op(comm, Nt, Np);
        surf_op.Grad2D(dX[i], S.Coord());
        surf_op.SurfNormalAreaElem(&Xn[i], &Xa[i], dX[i], &S.Coord());

        SurfArea[i] = 0;
        for (const auto& da : Xa[i]) SurfArea[i] += da;
      }
      for (sctl::Long i = 0; i < Nsurf; i++) { // TODO: this is not reliable, use the normal direction to do this
        if (SurfArea[i] < SurfArea[InnerSurfIdx]) InnerSurfIdx = i;
        if (SurfArea[i] > SurfArea[OuterSurfIdx]) OuterSurfIdx = i;
      }
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set normal to outer-normal
        Real scal = 0;
        if (i == InnerSurfIdx) scal = -1;
        if (i == OuterSurfIdx) scal = 1;
        Xn[i] *= scal;
      }

      sctl::Profile::Tic("Setup", &comm);
      Helmholtz3D<Real> helm(lambda);
      HelmholtzDiff3D<Real> helm_diff(lambda);
      const auto& ker_potn = helm.FxU();
      const auto& ker_grad = helm.FxdU();
      const auto& ker_grad_diff = helm_diff.FxdU();
      BoundaryIntegralOp<Real,2,6,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_grad_diff(comm);
      BoundaryIntegralOp<Real,2,6,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_grad(comm);
      BoundaryIntegralOp<Real,2,2,UPSAMPLE,PATCH_DIM0,RAD_DIM> BI_potn(comm);
      BI_grad_diff.SetupSingular(Svec, ker_grad_diff);
      BI_grad.SetupSingular(Svec, ker_grad);
      BI_potn.SetupSingular(Svec, ker_potn);
      sctl::Profile::Toc();

      auto Compute_m = [lambda,&Svec,&SurfDim,&SurfDsp,&dX,&Xn,&comm](sctl::Vector<Real>& m, const sctl::Vector<Real>& sigma, Real gmres_tol, sctl::Long gmres_iter) { // m <-- lambda ( i Grad(invLap(sigma)) - Grad(invLap(sigma)) x n )
        sctl::Long Nsurf = SurfDim.Dim();
        sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
        if (m.Dim() != COORD_DIM * 2 * N) m.ReInit(COORD_DIM * 2 * N);
        if (!sigma.Dim() || lambda == 0) { // m = 0
          m = 0;
        } else {
          for (sctl::Long i = 0; i < Nsurf; i++) {
            const sctl::Vector<Real> sigma_(2 * SurfDim[i], (sctl::Iterator<Real>)sigma.begin() + 2 * SurfDsp[i], false);
            sctl::Vector<Real> m_(COORD_DIM * 2 * SurfDim[i], m.begin() + COORD_DIM * 2 * SurfDsp[i], false);

            const auto& S = Svec[i];
            sctl::Long Nt = S.NTor();
            sctl::Long Np = S.NPol();
            sctl::Long N= Nt * Np;

            SurfaceOp<Real> surf_op(comm, Nt, Np);
            sctl::StaticArray<sctl::Vector<Real>, 2> GradInvLapSigma;
            for (sctl::Long k = 0; k < 2; k++) { // Set GradInvLapSigma <-- Grad(invLap(sigma))
              const sctl::Vector<Real> sigma__(N, (sctl::Iterator<Real>)sigma_.begin() + k * N, false);
              if (max_norm(sigma__) > 0) {
                surf_op.GradInvSurfLap(GradInvLapSigma[k], dX[i], sigma__, gmres_tol, gmres_iter, 1.5);
              } else {
                GradInvLapSigma[k].ReInit(COORD_DIM * N);
                GradInvLapSigma[k].SetZero();
              }
            }
            for (sctl::Long j = 0; j < N; j++) { // m <-- lambda ( i GradInvLapSigma - n x GradInvLapSigma )
              sctl::StaticArray<Real, COORD_DIM> Xn_, V0_, V1_, nxV0_, nxV1_;
              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                Xn_[k] = Xn[i][k * N + j];
                V0_[k] = GradInvLapSigma[0][k * N + j];
                V1_[k] = GradInvLapSigma[1][k * N + j];
              }

              nxV0_[0] = Xn_[1] * V0_[2] - Xn_[2] * V0_[1];
              nxV0_[1] = Xn_[2] * V0_[0] - Xn_[0] * V0_[2];
              nxV0_[2] = Xn_[0] * V0_[1] - Xn_[1] * V0_[0];

              nxV1_[0] = Xn_[1] * V1_[2] - Xn_[2] * V1_[1];
              nxV1_[1] = Xn_[2] * V1_[0] - Xn_[0] * V1_[2];
              nxV1_[2] = Xn_[0] * V1_[1] - Xn_[1] * V1_[0];

              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                m_[(k*2+0) * N + j] = lambda * (-V1_[k] - nxV0_[k]);
                m_[(k*2+1) * N + j] = lambda * ( V0_[k] - nxV1_[k]);
              }
            }
          }
        }
        if (1) {
#ifdef BIEST_VERBOSE
          auto extract_comp = [&SurfDim,&SurfDsp,Nsurf,N](const sctl::Vector<Real>& in, sctl::Long surf_id, sctl::Long comp_id) {
            sctl::Long dof = in.Dim() / N;
            SCTL_ASSERT(in.Dim() == N * dof);
            sctl::Vector<Real> out(SurfDim[surf_id]);
            for (sctl::Long i = 0; i < SurfDim[surf_id]; i++) {
              out[i] = in[dof * SurfDsp[surf_id] + comp_id * SurfDim[surf_id] + i];
            }
            return out;
          };
          auto concat_vec = [](const sctl::Vector<Real>& x, const sctl::Vector<Real>& y) {
            sctl::Vector<Real> out;
            for (const auto& a : x) out.PushBack(a);
            for (const auto& a : y) out.PushBack(a);
            return out;
          };
          { // Print error <-- | SurfDiv(m) - i lambda sigma |_inf
            Real err = 0;
            for (sctl::Long i = 0; i < Nsurf; i++) {
              const auto& S = Svec[i];
              sctl::Vector<Real> m_real, m_imag;
              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                m_real = concat_vec(m_real, extract_comp(m, i, 2*k+0));
                m_imag = concat_vec(m_imag, extract_comp(m, i, 2*k+1));
              }

              sctl::Vector<Real> Divm_real, Divm_imag;
              SurfaceOp<Real> surf_op(comm, S.NTor(), S.NPol());
              surf_op.SurfDiv(Divm_real, dX[i], m_real);
              surf_op.SurfDiv(Divm_imag, dX[i], m_imag);

              sctl::Vector<Real> sigma_real, sigma_imag;
              if (sigma.Dim()) {
                sigma_real = extract_comp(sigma, i, 0);
                sigma_imag = extract_comp(sigma, i, 1);
              } else {
                sigma_real = Divm_real * 0;
                sigma_imag = Divm_imag * 0;
              }

              err = std::max(err, max_norm(Divm_real + sigma_imag * lambda));
              err = std::max(err, max_norm(Divm_imag - sigma_real * lambda));
            }
            std::cout<<"Error : | SurfDiv(m) - i lambda sigma |_inf = "<<err<<";   | m | = "<<max_norm(m)<<";   | lambda sigma | = "<<lambda * max_norm(sigma)<<'\n';
          }
          { // Print error <-- n x m + i m
            Real err = 0;
            for (sctl::Long i = 0; i < Nsurf; i++) {
              sctl::Vector<Real> m_real, m_imag;
              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                m_real = concat_vec(m_real, extract_comp(m, i, 2*k+0));
                m_imag = concat_vec(m_imag, extract_comp(m, i, 2*k+1));
              }

              auto cross_prod = [](const sctl::Vector<Real>& x, const sctl::Vector<Real>& y) {
                sctl::Long N = x.Dim() / COORD_DIM;
                sctl::Vector<Real> z(N * COORD_DIM);
                SCTL_ASSERT(x.Dim() == N * COORD_DIM);
                SCTL_ASSERT(y.Dim() == N * COORD_DIM);
                for (sctl::Long i = 0; i < N; i++) {
                  sctl::StaticArray<Real, COORD_DIM> x_, y_, z_;
                  for (sctl::Long k = 0; k < COORD_DIM; k++) {
                    x_[k] = x[k * N + i];
                    y_[k] = y[k * N + i];
                  }
                  z_[0] = x_[1] * y_[2] - x_[2] * y_[1];
                  z_[1] = x_[2] * y_[0] - x_[0] * y_[2];
                  z_[2] = x_[0] * y_[1] - x_[1] * y_[0];
                  for (sctl::Long k = 0; k < COORD_DIM; k++) {
                    z[k * N + i] = z_[k];
                  }
                }
                return z;
              };
              sctl::Vector<Real> nxm_real = cross_prod(Xn[i], m_real);
              sctl::Vector<Real> nxm_imag = cross_prod(Xn[i], m_imag);

              err = std::max(err, max_norm(nxm_real - m_imag));
              err = std::max(err, max_norm(nxm_imag + m_real));
            }
            std::cout<<"Error : | n x m + i m |_inf = "<<err<<'\n';
          }
#endif
        }
      };
      auto Compute_B = [lambda,&SurfDim,&SurfDsp,&Xn,&BI_grad_diff,&BI_grad,&BI_potn,&ker_grad_diff,&ker_grad,&ker_potn,&comm] (sctl::Vector<Real>* B, sctl::Vector<Real>* B_flux, const sctl::Vector<Real>& sigma, const sctl::Vector<Real>& m0, const sctl::Vector<Real>& mH) {
        sctl::Long Nsurf = SurfDim.Dim();
        SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
        sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];

        sctl::Vector<Real> m;
        if (m0.Dim() && mH.Dim()) {
          SCTL_ASSERT(m0.Dim() == mH.Dim());
          m = m0 + mH;
        } else if (m0.Dim()) {
          m = m0;
        } else if (mH.Dim()) {
          m = mH;
        }

        sctl::Vector<Real> Grad_v, iQ, iCurlQ, iCurlQ_m0, iCurlQ_mH;
        if (B && sigma.Dim()) { // Compute Grad_v
          sctl::Profile::Tic("Compute_Grad_v", &comm);
          Compute_Grad_v(Grad_v, sigma, BI_grad, ker_grad);
          sctl::Profile::Toc();
        }
        if (m.Dim()) { // Compute iQ
          sctl::Profile::Tic("Compute_iQ", &comm);
          Compute_iQ(iQ, m, SurfDim, SurfDsp, BI_potn, ker_potn, lambda);
          sctl::Profile::Toc();
        }
        if (B && m.Dim()) { // Compute iCurlQ
          sctl::Profile::Tic("Compute_iCurlQ", &comm);
          Compute_iCurlQ(iCurlQ, m, SurfDim, SurfDsp, BI_grad, ker_grad);
          sctl::Profile::Toc();
        }
        if (B_flux) { // Compute iCurlQ_m0, iCurlQ_mH
          sctl::Profile::Tic("Compute_iCurlQ_", &comm);
          if (m0.Dim()) Compute_iCurlQ(iCurlQ_m0, m0, SurfDim, SurfDsp, BI_grad     , ker_grad     );
          if (mH.Dim()) Compute_iCurlQ(iCurlQ_mH, mH, SurfDim, SurfDsp, BI_grad_diff, ker_grad_diff);
          sctl::Profile::Toc();
        }

        if (B_flux) { // B_flux <-- iQ*lambda + iCurlQ_mH + iCurlQ_m0 + 0.5*m0
          (*B_flux).ReInit(COORD_DIM * 2 * N);
          (*B_flux) = 0;
          if (m .Dim()) (*B_flux) += iQ*lambda;
          if (mH.Dim()) (*B_flux) += iCurlQ_mH;
          if (m0.Dim()) (*B_flux) += iCurlQ_m0 + m0*(0.5);
        }
        if (B) { // B <-- -0.5*sigma_Xn - Grad_v + iQ*lambda + iCurlQ + 0.5*m
          (*B).ReInit(COORD_DIM * 2 * N);
          (*B) = 0;
          if (m.Dim()) (*B) += iQ*lambda + iCurlQ + m*(0.5);
          if (sigma.Dim()) { // B <-- B - 0.5*sigma_Xn - Grad_v
            sctl::Vector<Real> sigma_Xn(COORD_DIM * 2 * N);
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  sigma_Xn[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j] = Xn[i][k * SurfDim[i] + j] * sigma[2 * SurfDsp[i] + 0 * SurfDim[i] + j];
                  sigma_Xn[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j] = Xn[i][k * SurfDim[i] + j] * sigma[2 * SurfDsp[i] + 1 * SurfDim[i] + j];
                }
              }
            }
            (*B) += sigma_Xn*(-0.5) - Grad_v;
          }
        }
      };
      auto Compute_BdotXn = [&SurfDim,&SurfDsp,&Xn](sctl::Vector<Real>& BdotXn, const sctl::Vector<Real>& B) {
        sctl::Long Nsurf = SurfDim.Dim();
        SCTL_ASSERT(SurfDsp.Dim() == Nsurf);

        sctl::Long N = B.Dim() / (COORD_DIM * 2);
        SCTL_ASSERT(B.Dim() == COORD_DIM * 2 * N);
        if (BdotXn.Dim() != 2 * N) BdotXn.ReInit(2 * N);
        for (sctl::Long i = 0; i < Nsurf; i++) {
          for (sctl::Long j = 0; j < SurfDim[i]; j++) {
            Real B0dotXn = 0, B1dotXn = 0;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Real Xn_ = Xn[i][k * SurfDim[i] + j];
              Real B0 = B[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j];
              Real B1 = B[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j];
              B0dotXn += B0 * Xn_;
              B1dotXn += B1 * Xn_;
            }
            BdotXn[2 * SurfDsp[i] + 0 * SurfDim[i] + j] = B0dotXn;
            BdotXn[2 * SurfDsp[i] + 1 * SurfDim[i] + j] = B1dotXn;
          }
        }
      };
      auto solve_taylor = [lambda,LB_tol,&comm,&Compute_m,&Compute_B,&Compute_BdotXn] (sctl::Vector<Real>& sigma, const sctl::Vector<Real>& rhs, Real gmres_tol, sctl::Long gmres_iter) {
        sctl::Profile::Tic("Solve", &comm);
        sctl::ParallelSolver<Real> solver(comm, true);
        typename sctl::ParallelSolver<Real>::ParallelOp fn = [lambda,LB_tol,&comm,&Compute_m,&Compute_B,&Compute_BdotXn,&gmres_tol,&gmres_iter](sctl::Vector<Real>* BdotXn, const sctl::Vector<Real>& sigma) {
          SCTL_ASSERT(BdotXn);
          sctl::Vector<Real> m, B;
          sctl::Profile::Tic("Compute_B", &comm);
          Compute_m(m, sigma, gmres_tol / lambda * LB_tol, gmres_iter);
          Compute_B(&B, nullptr, sigma, m, sctl::Vector<Real>());
          Compute_BdotXn(*BdotXn, B);
          { // Project to space of mean-zero functions : BdotXn <-- P BdotXn
            // TODO
          }
          sctl::Profile::Toc();
        };
        solver(&sigma, fn, rhs, gmres_tol, gmres_iter);
        sctl::Profile::Toc();
      };

      sctl::Vector<sctl::Vector<Real>> mH(Nsurf), B0(Nsurf);
      sctl::Profile::Tic("Compute_B0", &comm);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Compute mH, B0
        mH[i].ReInit(COORD_DIM * 2 * (SurfDim[Nsurf-1] + SurfDsp[Nsurf-1])); mH[i] = 0;
        sctl::Vector<Real> mH_(COORD_DIM * 2 * SurfDim[i], mH[i].begin() + COORD_DIM * 2 * SurfDsp[i], false);
        Compute_mH(mH_, Svec[i], dX[i], Xn[i], comm, gmres_tol * LB_tol, gmres_iter);
        Compute_B(&B0[i], nullptr, sctl::Vector<Real>(), sctl::Vector<Real>(), mH[i]*(-1));
      }
      sctl::Profile::Toc();

      sctl::Vector<sctl::Vector<Real>> sigma(Nsurf), m(Nsurf), B(Nsurf), B_flux(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Solve
        sctl::Vector<Real> rhs;
        Compute_BdotXn(rhs, B0[i]);
        solve_taylor(sigma[i], rhs, gmres_tol, gmres_iter);
        Compute_m(m[i], sigma[i], gmres_tol * LB_tol, gmres_iter);
        Compute_B(&B[i], &B_flux[i], sigma[i], m[i], mH[i]);
        m[i] += mH[i];
      }

      sctl::Profile::Tic("CompFlux", &comm);
      { // Compute flux
        auto compute_tor_circ = [&Svec,&SurfDim,&SurfDsp,dX](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) {
          sctl::Long Nsurf = Svec.Dim();
          SCTL_ASSERT(surf_id < Nsurf);
          const auto& S = Svec[surf_id];
          sctl::Long Nt = S.NTor();
          sctl::Long Np = S.NPol();

          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
          sctl::Long dof = A.Dim() / (N * COORD_DIM);
          SCTL_ASSERT(A.Dim() == N * COORD_DIM * dof);
          sctl::Long offset = SurfDsp[surf_id] * COORD_DIM * dof;

          sctl::Complex<Real> circ_ = 0;
          sctl::Vector<Real> circ_vec(dof*Np);
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long p = 0; p < Np; p++) {
              Real integ0 = 0;
              for (sctl::Long t = 0; t < Nt; t++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  Real A0 = A[offset + (k*dof+d)*Nt*Np + t*Np + p];
                  Real dX_ = dX[surf_id][(2*k+0)*Nt*Np + t*Np + p];
                  integ0 += A0 * dX_;
                }
              }
              circ_vec[d * Np + p] = integ0 / Nt;
              if (!d) circ_.real += integ0 / (Nt*Np);
              else    circ_.imag += integ0 / (Nt*Np);
            }
          }
          if (circ) (*circ) = circ_vec;
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long p = 0; p < Np; p++) {
              if (!d) circ_vec[d * Np + p] -= circ_.real;
              else    circ_vec[d * Np + p] -= circ_.imag;
            }
          }
          return circ_;
        };
        auto compute_pol_circ = [&Svec,&SurfDim,&SurfDsp,dX](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) {
          sctl::Long Nsurf = Svec.Dim();
          SCTL_ASSERT(surf_id < Nsurf);
          const auto& S = Svec[surf_id];
          sctl::Long Nt = S.NTor();
          sctl::Long Np = S.NPol();

          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
          sctl::Long dof = A.Dim() / (N * COORD_DIM);
          SCTL_ASSERT(A.Dim() == N * COORD_DIM * dof);
          sctl::Long offset = SurfDsp[surf_id] * COORD_DIM * dof;

          sctl::Complex<Real> circ_ = 0;
          sctl::Vector<Real> circ_vec(dof*Nt);
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long t = 0; t < Nt; t++) {
              Real integ0 = 0;
              for (sctl::Long p = 0; p < Np; p++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  Real A0 = A[offset + (k*dof+d)*Nt*Np + t*Np + p];
                  Real dX_ = dX[surf_id][(2*k+1)*Nt*Np + t*Np + p];
                  integ0 += A0 * dX_;
                }
              }
              circ_vec[d * Nt + t] = integ0 / Np;
              if (!d) circ_.real += integ0 / (Nt*Np);
              else    circ_.imag += integ0 / (Nt*Np);
            }
          }
          if (circ) (*circ) = circ_vec;
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long t = 0; t < Nt; t++) {
              if (!d) circ_vec[d * Nt + t] -= circ_.real;
              else    circ_vec[d * Nt + t] -= circ_.imag;
            }
          }
          return circ_;
        };
        auto complex_vec_prod = [&SurfDim,&SurfDsp] (sctl::Vector<Real>& v, Real c0, Real c1) {
          sctl::Long Nsurf = SurfDim.Dim();
          SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
          sctl::Long N = SurfDsp[Nsurf-1]+SurfDim[Nsurf-1];
          sctl::Long dof = v.Dim() / (N * 2);
          SCTL_ASSERT(v.Dim() == N * dof * 2);

          for (sctl::Long i = 0; i < Nsurf; i++) {
            for (sctl::Long k = 0; k < dof; k++) {
              sctl::Long offset = SurfDsp[i]*dof*2 + k*2*SurfDim[i];
              for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                Real v0 = v[offset + 0*SurfDim[i] + j];
                Real v1 = v[offset + 1*SurfDim[i] + j];
                Real cv0 = v0 * c0 - v1 * c1;
                Real cv1 = v0 * c1 + v1 * c0;
                v[offset + 0*SurfDim[i] + j] = cv0;
                v[offset + 1*SurfDim[i] + j] = cv1;
              }
            }
          }
        };
        auto real_imag_part = [&SurfDim,&SurfDsp] (sctl::Vector<Real>& w, const sctl::Vector<Real>& v, bool real_part) {
          sctl::Long Nsurf = SurfDim.Dim();
          SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
          sctl::Long N = SurfDsp[Nsurf-1]+SurfDim[Nsurf-1];
          sctl::Long dof = v.Dim() / (N * 2);
          SCTL_ASSERT(v.Dim() == N * dof * 2);
          if (w.Dim() != N * dof) w.ReInit(N * dof);

          for (sctl::Long i = 0; i < Nsurf; i++) {
            for (sctl::Long k = 0; k < dof; k++) {
              sctl::Long offset = SurfDsp[i]*dof;
              for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                w[offset + k*SurfDim[i] + j] = v[offset*2 + (k*2+(real_part?0:1))*SurfDim[i] + j];
              }
            }
          }
        };
        auto print_imag_norm = [&real_imag_part] (const sctl::Vector<Real>& B) { // Print | imag(B) |_inf
#ifdef BIEST_VERBOSE
          sctl::Vector<Real> B_imag;
          real_imag_part(B_imag, B, false);
          std::cout<<"Error : | imag(B) |_inf / | real(B) |_inf = "<<max_norm(B_imag) / max_norm(B)<<'\n';
#endif
        };

        if (Nsurf == 1) {
          if (B_out.Dim() < 1) B_out.ReInit(1);
          if (m_out.Dim() < 1) m_out.ReInit(1);
          if (sigma_out.Dim() < 1) sigma_out.ReInit(1);

          // TODO: change flux evaluation only on curve
          sctl::Complex<Real> tor_flux = compute_pol_circ(nullptr, B_flux[0], 0) / lambda;
          sctl::Complex<Real> scal = sctl::Complex<Real>(1,0) / tor_flux;
          complex_vec_prod(B[0], scal.real, scal.imag);
          complex_vec_prod(m[0], scal.real, scal.imag);
          complex_vec_prod(sigma[0], scal.real, scal.imag);
          real_imag_part(B_out[0], B[0], true);
          sigma_out[0] = sigma[0];
          m_out[0] = m[0];

          print_imag_norm(B[0]);
        } else if (Nsurf == 2) {
          if (B_out.Dim() < 2) B_out.ReInit(2);
          if (m_out.Dim() < 2) m_out.ReInit(2);
          if (sigma_out.Dim() < 2) sigma_out.ReInit(2);

          sctl::Complex<Real> tor_flux0 = (compute_pol_circ(nullptr, B_flux[0], OuterSurfIdx) - compute_pol_circ(nullptr, B_flux[0], InnerSurfIdx)) / lambda;
          sctl::Complex<Real> tor_flux1 = (compute_pol_circ(nullptr, B_flux[1], OuterSurfIdx) - compute_pol_circ(nullptr, B_flux[1], InnerSurfIdx)) / lambda;
          sctl::Complex<Real> pol_flux0 = (compute_tor_circ(nullptr, B_flux[0], OuterSurfIdx) - compute_tor_circ(nullptr, B_flux[0], InnerSurfIdx)) / lambda;
          sctl::Complex<Real> pol_flux1 = (compute_tor_circ(nullptr, B_flux[1], OuterSurfIdx) - compute_tor_circ(nullptr, B_flux[1], InnerSurfIdx)) / lambda;

          sctl::Matrix<sctl::Complex<Real>> M(2,2);
          M[0][0] = tor_flux0; M[0][1] = tor_flux1;
          M[1][0] = pol_flux0; M[1][1] = pol_flux1;

          sctl::Matrix<sctl::Complex<Real>> M_inv(2,2);
          sctl::Complex<Real> det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
          M_inv[0][0] = M[1][1]/det; M_inv[0][1] =-M[0][1]/det;
          M_inv[1][0] =-M[1][0]/det; M_inv[1][1] = M[0][0]/det;

          auto ComplexMatVec2x2 = [&complex_vec_prod](sctl::Vector<sctl::Vector<Real>>& Vout, const sctl::Vector<sctl::Vector<Real>>& Vin, const sctl::Matrix<sctl::Complex<Real>>& M) {
            SCTL_ASSERT(Vin.Dim() == 2 && Vout.Dim() == 2);
            SCTL_ASSERT(M.Dim(0) == 2 && M.Dim(1) == 2);
            sctl::Vector<Real> V0, V1;

            V0 = Vin[0]; V1 = Vin[1];
            complex_vec_prod(V0, M[0][0].real, M[0][0].imag);
            complex_vec_prod(V1, M[1][0].real, M[1][0].imag);
            Vout[0] = V0 + V1;

            V0 = Vin[0]; V1 = Vin[1];
            complex_vec_prod(V0, M[0][1].real, M[0][1].imag);
            complex_vec_prod(V1, M[1][1].real, M[1][1].imag);
            Vout[1] = V0 + V1;
          };

          sctl::Vector<sctl::Vector<Real>> B_(2);
          ComplexMatVec2x2(B_, B, M_inv);
          ComplexMatVec2x2(m_out, m, M_inv);
          ComplexMatVec2x2(sigma_out, sigma, M_inv);
          real_imag_part(B_out[0], B_[0], true);
          real_imag_part(B_out[1], B_[1], true);
          print_imag_norm(B_[0]);
          print_imag_norm(B_[1]);
        } else {
          SCTL_ASSERT(false);
        }
      }
      sctl::Profile::Toc();
    }

}
