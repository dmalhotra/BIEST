// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>
#include <biest/surface_op.hpp>
#include <biest/kernel.hpp>

namespace biest {

namespace detail_bie_solvers {

  /**
   * Returns the indices of the outer and the inner surface of nested toroidal
   * surfaces: the surfaces enclosing the largest and the smallest volume,
   * V = (1/3) sum(X.n w), with normals Xn and quadrature weights Xa from
   * SurfaceOp::SurfNormalAreaElem().
   */
  template <class Real> std::pair<sctl::Long, sctl::Long> OuterInnerSurfIdx(const sctl::Vector<Surface<Real>>& Svec, const sctl::Vector<sctl::Vector<Real>>& Xn, const sctl::Vector<sctl::Vector<Real>>& Xa) {
    constexpr sctl::Integer COORD_DIM = 3;
    const sctl::Long Nsurf = Svec.Dim();
    sctl::ScratchBuf<Real> Volume(Nsurf);
    for (sctl::Long i = 0; i < Nsurf; i++) {
      const sctl::Long N = Xa[i].Dim();
      Real V = 0;
      for (sctl::Long j = 0; j < N; j++) {
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          V += Svec[i].Coord()[k * N + j] * Xn[i][k * N + j] * Xa[i][j];
        }
      }
      Volume[i] = sctl::fabs<Real>(V) / 3;
    }

    sctl::Long OuterSurfIdx = 0, InnerSurfIdx = 0;
    for (sctl::Long i = 0; i < Nsurf; i++) {
      if (Volume[i] < Volume[InnerSurfIdx]) InnerSurfIdx = i;
      if (Volume[i] > Volume[OuterSurfIdx]) OuterSurfIdx = i;
    }
    return {OuterSurfIdx, InnerSurfIdx};
  }

  /**
   * Set, for each surface, its grid points on one of the NFP field periods (SurfDim) and their offset in the vectors of
   * all the surfaces (SurfDsp), the same for the full torus (SurfDimF, SurfDspF: NFP times those), the dimensions of
   * its grid (SurfNt x SurfNp) and the direction of its toroidal index (SurfOrient).
   */
  template <class Real> void SurfDims(sctl::Vector<sctl::Long>& SurfDim, sctl::Vector<sctl::Long>& SurfDsp, sctl::Vector<sctl::Long>& SurfDimF, sctl::Vector<sctl::Long>& SurfDspF, sctl::Vector<sctl::Long>& SurfNt, sctl::Vector<sctl::Long>& SurfNp, sctl::Vector<sctl::Integer>& SurfOrient, const sctl::Vector<Surface<Real>>& Svec, const sctl::Integer NFP) {
    const sctl::Long Nsurf = Svec.Dim();
    for (auto* v : {&SurfDim, &SurfDsp, &SurfDimF, &SurfDspF, &SurfNt, &SurfNp}) v->ReInit(Nsurf);
    SurfOrient.ReInit(Nsurf);
    for (sctl::Long i = 0; i < Nsurf; i++) {
      SurfNt[i] = Svec[i].NTor();
      SurfNp[i] = Svec[i].NPol();
      SurfDimF[i] = SurfNt[i] * SurfNp[i];
      SurfDim[i] = SurfDimF[i] / NFP;
      SurfDsp[i] = (i ? SurfDsp[i-1] + SurfDim[i-1] : 0);
      SurfDspF[i] = NFP * SurfDsp[i];
      SurfOrient[i] = SurfaceOp<Real>::ToroidalOrientation(Svec[i].Coord(), SurfNt[i], SurfNp[i]);
    }
  }

  /**
   * Set Y[i] to the restriction of X[i], a field on the full torus of surface i (SurfDimF[i] grid points), to its first
   * field period (the first SurfDim[i] points of each component): the geometry of the Laplace-Beltrami solves on one
   * field period (see SurfaceOp).
   */
  template <class Real> void FirstPeriod(sctl::Vector<sctl::Vector<Real>>& Y, const sctl::Vector<sctl::Vector<Real>>& X, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDimF) {
    if (Y.Dim() != X.Dim()) Y.ReInit(X.Dim());
    for (sctl::Long i = 0; i < X.Dim(); i++) {
      const sctl::Long M = SurfDim[i], MF = SurfDimF[i], dof = X[i].Dim() / MF;
      if (Y[i].Dim() != dof * M) Y[i].ReInit(dof * M);
      for (sctl::Long k = 0; k < dof; k++) {
        for (sctl::Long n = 0; n < M; n++) Y[i][k * M + n] = X[i][k * MF + n];
      }
    }
  }

  /**
   * Complete fields given on one of the NFP field periods of each surface (dof components at each of SurfDim[i] grid
   * points, at SurfDsp[i]) to the full tori, by the symmetry: scalars (vec = 0) are copied, and a Cartesian vector
   * (vec = 1, dof = 3) or the real and imaginary parts of a complex one (vec = 2, dof = 6, component 2k+c part c of
   * coordinate k) rotated by 2 pi j / NFP to field period j, in the direction of the toroidal index (SurfOrient).
   */
  template <class Real> void CompletePeriods(sctl::Vector<Real>& Y, const sctl::Vector<Real>& X, const sctl::Integer dof, const sctl::Integer vec, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const sctl::Vector<sctl::Integer>& SurfOrient, const sctl::Integer NFP) {
    const sctl::Long Nsurf = SurfDim.Dim(), N = SurfDsp[Nsurf-1] + SurfDim[Nsurf-1];
    const sctl::Integer cdim = (vec == 2 ? 2 : 1);
    SCTL_ASSERT(X.Dim() == dof * N && (vec == 0 || dof == 3 * cdim));
    if (Y.Dim() != dof * NFP * N) Y.ReInit(dof * NFP * N);
    for (sctl::Long i = 0; i < Nsurf; i++) {
      const sctl::Long M = SurfDim[i], MF = NFP * M;
      const sctl::ConstIterator<Real> X_ = X.begin() + dof * SurfDsp[i];
      const sctl::Iterator<Real> Y_ = Y.begin() + dof * NFP * SurfDsp[i];
      for (sctl::Integer j = 0; j < NFP; j++) {
        if (!vec) {
          for (sctl::Integer q = 0; q < dof; q++) {
            sctl::omp_par::parallel_for(M, [Y_,q,MF,j,M,X_](const sctl::Long n) { Y_[q * MF + j * M + n] = X_[q * M + n]; });
          }
          continue;
        }
        const Real c = sctl::cos<Real>(2 * sctl::const_pi<Real>() * j / NFP), s = sctl::sin<Real>(2 * sctl::const_pi<Real>() * j / NFP) * SurfOrient[i];
        for (sctl::Integer cc = 0; cc < cdim; cc++) {
          sctl::omp_par::parallel_for(M, [X_,cdim,cc,M,Y_,MF,j,c,s](const sctl::Long n) {
            const Real x = X_[(0 * cdim + cc) * M + n], y = X_[(1 * cdim + cc) * M + n], z = X_[(2 * cdim + cc) * M + n];
            Y_[(0 * cdim + cc) * MF + j * M + n] = x * c - y * s;
            Y_[(1 * cdim + cc) * MF + j * M + n] = x * s + y * c;
            Y_[(2 * cdim + cc) * MF + j * M + n] = z;
          });
        }
      }
    }
  }

  /**
   * A FieldPeriodBIOp applied to densities on the full tori, on the surface grids (SurfNt[i] x SurfNp[i]), which it
   * resamples to its quadrature grids when those are finer.
   */
  template <class BIOp> struct SurfGridOp {
    const BIOp& op;
    const sctl::Vector<sctl::Long>& SurfNt;
    const sctl::Vector<sctl::Long>& SurfNp;
    template <class Real> void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      sctl::ScratchBuf<sctl::Integer> NFP_buf(SurfNt.Dim()); // the densities on the full tori
      sctl::Vector<sctl::Integer> NFP(NFP_buf);
      NFP = 1;
      op.Eval(U, F, NFP, SurfNt, SurfNp);
    }
  };

}

    template <class Real> VacuumField<Real>::VacuumField(const sctl::Comm& comm, const SingularQuadParams& quad) : comm_(comm), quad_(quad), BI_biot_savart_(comm), BI_lap_grad_(comm), BI_laplace_(comm) {}

    template <class Real> void VacuumField<Real>::Compute(sctl::Vector<Real>& B, Real tor_flux, Real pol_flux, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, sctl::Vector<sctl::Vector<Real>>* J) {
      VacuumField(comm).Compute(B, tor_flux, pol_flux, Svec, gmres_tol, gmres_iter, J);
    }

    template <class Real> void VacuumField<Real>::Compute(sctl::Vector<Real>& B, Real tor_flux, Real pol_flux, const sctl::Vector<Surface<Real>>& Svec, Real gmres_tol, sctl::Long gmres_iter, sctl::Vector<sctl::Vector<Real>>* J) {
      ComputeHelper(B_, J_, tor_flux_, pol_flux_, Svec, comm_, gmres_tol, gmres_iter); // in the vectors of the object
      // The coefficients of the solutions (one for one surface) in the field of the fluxes (tf, pf)
      const auto coefs = [this,&Svec](const Real tf, const Real pf) {
        if (Svec.Dim() == 1) return std::make_pair(tf / tor_flux_[0], (Real)0);
        const Real det = tor_flux_[0] * pol_flux_[1] - tor_flux_[1] * pol_flux_[0];
        return std::make_pair((tf * pol_flux_[1] - pf * tor_flux_[1]) / det, (pf * tor_flux_[0] - tf * pol_flux_[0]) / det);
      };
      // V: the solutions V_ combined with the coefficients c, on one field period, then completed to the full tori by
      // the symmetry, in the storage of V when it has the size
      const sctl::Integer NFP = detail_fieldperiod::CommonNFP(Svec);
      const auto combine = [this,&Svec,NFP](sctl::Vector<Real>& V, const sctl::Vector<sctl::Vector<Real>>& V_, const std::pair<Real,Real> c) {
        const sctl::Long N = V_[0].Dim();
        sctl::ScratchBuf<Real> V1_buf(N);
        sctl::Vector<Real> V1(V1_buf);
        if (Svec.Dim() == 1) for (sctl::Long i = 0; i < N; i++) V1[i] = V_[0][i] * c.first;
        else for (sctl::Long i = 0; i < N; i++) V1[i] = V_[0][i] * c.first + V_[1][i] * c.second;
        detail_bie_solvers::CompletePeriods(V, V1, COORD_DIM, 1, SurfDim_, SurfDsp_, SurfOrient_, NFP);
      };
      combine(B, B_, coefs(tor_flux, pol_flux));
      if (J) {
        if (J->Dim() != Svec.Dim()) J->ReInit(Svec.Dim());
        for (sctl::Long l = 0; l < Svec.Dim(); l++) combine((*J)[l], J_, coefs((Real)(l == 0), (Real)(l == 1)));
      }
    }

    template <class Real> void VacuumField<Real>::EvalOffSurface(sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& Xtrg, sctl::Vector<Surface<Real>> Svec, Real tor_flux, Real pol_flux, const sctl::Vector<sctl::Vector<Real>>& J, const sctl::Comm& comm, const Real tol) {
      SCTL_ASSERT(J.Dim() == Svec.Dim());
      sctl::Vector<Real> J0 = J[0] * tor_flux; // the current of the fluxes
      if (Svec.Dim() > 1) J0 += J[1] * pol_flux;
      const sctl::Long Ntrg = Xtrg.Dim() / COORD_DIM;
      sctl::Vector<Real> X = Xtrg, B(COORD_DIM * Ntrg);
      sctl::Matrix<Real>(COORD_DIM, Ntrg, X.begin(), false) = sctl::Matrix<Real>(Ntrg, COORD_DIM, X.begin(), false).Transpose();
      B = 0;
      sctl::Long dsp = 0;
      for (sctl::Long i = 0; i < Svec.Dim(); i++) { // B += BiotSavart[J0] from each surface
        const auto& S = Svec[i];
        const sctl::Long N = S.NTor() * S.NPol();
        const sctl::Vector<Real> J0_(COORD_DIM * N, J0.begin() + COORD_DIM * dsp, false);
        OffSurfaceOp op(BiotSavart3D<Real>::FxU(), comm);
        op.Setup(S.Coord(), S.NTor(), S.NPol(), X, tol, J0_);
        op.Eval(B, J0_);
        dsp += N;
      }
      sctl::Matrix<Real>(Ntrg, COORD_DIM, B.begin(), false) = sctl::Matrix<Real>(COORD_DIM, Ntrg, B.begin(), false).Transpose();
      Btrg = B;
    }

    template <class Real> void VacuumField<Real>::test(sctl::Long upsample, const sctl::Comm& comm) {
      sctl::Vector<Real> B;
      sctl::Vector<Surface<Real>> Svec;
      Svec.PushBack(Surface<Real>( 50*upsample+1, 20*upsample+1, SurfType::AxisymCircleWide));
      Svec.PushBack(Surface<Real>(100*upsample+1, 20*upsample+1, SurfType::Quas3));
      Compute(B, 1, 1, Svec, comm, 1e-8, 30);
      WriteVTK("B", Svec, B);
    }

    template <class Real> Real VacuumField<Real>::max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }

    template <class Real> void VacuumField<Real>::ComputeHelper(sctl::Vector<sctl::Vector<Real>>& B, sctl::Vector<sctl::Vector<Real>>& J, sctl::Vector<Real>& tor_flux, sctl::Vector<Real>& pol_flux, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter) {
      // The vectors of the solve are those of the object, which keeps them with their storage from one call to the next
      // (see VacuumField()); the temporaries are in scratch memory (see SurfaceOp::SurfGrad). The solve is on one of the
      // NFP field periods common to the surfaces: the unknown and the fields B and J, the targets of the operators, and
      // the Laplace-Beltrami solve of J0 (SurfaceOp on one field period). The geometry is on the full tori, and so are
      // the densities of the operators, completed by the symmetry from one field period.
      sctl::Long Nsurf = Svec.Dim();
      const sctl::Integer NFP = detail_fieldperiod::CommonNFP(Svec);
      auto& SurfDim = SurfDim_; // one field period
      auto& SurfDsp = SurfDsp_;
      auto& SurfDimF = SurfDimF_; // the full torus
      auto& SurfDspF = SurfDspF_;
      auto& SurfOrient = SurfOrient_;
      detail_bie_solvers::SurfDims(SurfDim, SurfDsp, SurfDimF, SurfDspF, SurfNt_, SurfNp_, SurfOrient, Svec, NFP);
      const sctl::Long NtotF = SurfDspF[Nsurf-1] + SurfDimF[Nsurf-1];

      auto& dX = dX_;
      auto& Xn = Xn_;
      auto& Xa = Xa_;
      dX.ReInit(Nsurf);
      Xn.ReInit(Nsurf);
      Xa.ReInit(Nsurf);
      auto& SurfPolOrient = SurfPolOrient_;
      SurfPolOrient.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Svec.Dim(); i++) { // Set dX, Xn, Xa, SurfPolOrient
        const auto& S = Svec[i];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();

        SurfaceOp<Real> surf_op(comm, Nt, Np);
        surf_op.Grad2D(dX[i], S.Coord());
        const Real normal_orient = surf_op.SurfNormalAreaElem(&Xn[i], &Xa[i], dX[i], &S.Coord()); // +1 if dX/dt x dX/dp points out of the torus
        SurfPolOrient[i] = SurfOrient[i] * (sctl::Integer)normal_orient;
      }
      sctl::Long OuterSurfIdx, InnerSurfIdx;
      std::tie(OuterSurfIdx, InnerSurfIdx) = detail_bie_solvers::OuterInnerSurfIdx(Svec, Xn, Xa);
      for (sctl::Long i = 0; i < Svec.Dim(); i++) { // Set normal to outer-normal
        Real scal = 0;
        if (i == InnerSurfIdx) scal = -1;
        if (i == OuterSurfIdx) scal = 1;
        Xn[i] *= scal;
      }
      auto& dXp = dXp_; // on one field period, for the Laplace-Beltrami solve of ComputeJ
      detail_bie_solvers::FirstPeriod(dXp, dX, SurfDim, SurfDimF);

      sctl::Profile::Tic("Setup", &comm);
      const auto& ker_laplace = Laplace3D<Real>::FxU();
      const auto& ker_lap_grad = Laplace3D<Real>::FxdU();
      const auto& ker_biot_savart = BiotSavart3D<Real>::FxU();
      auto& BI_biot_savart = BI_biot_savart_; // the operators of the object: see VacuumField()
      auto& BI_lap_grad = BI_lap_grad_;
      auto& BI_laplace = BI_laplace_;
      BI_biot_savart.SetupSingular(Svec, ker_biot_savart, gmres_tol * (Real)0.1, quad_);
      BI_lap_grad.SetupSingular(Svec, ker_lap_grad, gmres_tol * (Real)0.1, quad_);
      BI_laplace.SetupSingular(Svec, ker_laplace, gmres_tol * (Real)0.1, quad_);
      // the operators of densities on the surface grids
      using detail_bie_solvers::SurfGridOp;
      const SurfGridOp<std::remove_reference_t<decltype(BI_biot_savart)>> BS_op{BI_biot_savart, SurfNt_, SurfNp_};
      const SurfGridOp<std::remove_reference_t<decltype(BI_lap_grad)>> LG_op{BI_lap_grad, SurfNt_, SurfNp_};
      const SurfGridOp<std::remove_reference_t<decltype(BI_laplace)>> L_op{BI_laplace, SurfNt_, SurfNp_};
      sctl::Profile::Toc();

      sctl::Profile::Tic("ComputeJ", &comm);
      auto& J0 = J0_;
      J0.ReInit(Nsurf);
      const auto complete_J = [&SurfDim,&SurfDsp,&SurfOrient,NFP](sctl::Vector<Real>& J_full, const sctl::Vector<Real>& J) { // a Cartesian vector
        detail_bie_solvers::CompletePeriods(J_full, J, COORD_DIM, 1, SurfDim, SurfDsp, SurfOrient, NFP);
      };
      auto ComputeJ = [NFP,&Svec,&SurfDim,&SurfDsp,&SurfOrient,&complete_vec = complete_J,&dX = dXp,&comm](sctl::Vector<Real>& J, sctl::Integer Sidx, Real gmres_tol, sctl::Long max_iter) { // Set Jp, on the full tori from one field period
        const auto& S = Svec[Sidx];
        sctl::Long Nt = S.NTor() / NFP;
        sctl::Long Np = S.NPol();
        sctl::Long N = Nt * Np;

        sctl::ScratchBuf<Real> V_buf(COORD_DIM * N), DivV_buf(N), GradInvLapDivV_buf(COORD_DIM * N);
        sctl::Vector<Real> V(V_buf), DivV(DivV_buf), GradInvLapDivV(GradInvLapDivV_buf);
        for (sctl::Long i = 0; i < N; i++) { // Set V
          for (sctl::Long k =0; k < COORD_DIM; k++) {
            V[k * N + i] = dX[Sidx][(k*2+0) * N + i] + dX[Sidx][(k*2+1) * N + i];
          }
        }
        SurfaceOp<Real> surf_op(comm, Nt, Np, NFP, SurfOrient[Sidx]);
        surf_op.SurfDiv(DivV, dX[Sidx], V);
        surf_op.GradInvSurfLap(GradInvLapDivV, dX[Sidx], DivV, gmres_tol * max_norm(V) / max_norm(DivV), max_iter, 1.5);
        for (sctl::Long i = 0; i < COORD_DIM * N; i++) V[i] -= GradInvLapDivV[i];
        if (0) { // Print err
          surf_op.SurfDiv(DivV, dX[Sidx], V);
          Real err = 0;
          for (const auto x : DivV) err = std::max(err, fabs(x));
          std::cout<<err<<'\n';
        }

        { // Set J
          sctl::Long Nsurf = SurfDim.Dim();
          sctl::ScratchBuf<Real> Jp_buf(COORD_DIM * (SurfDim[Nsurf-1] + SurfDsp[Nsurf-1]));
          sctl::Vector<Real> Jp(Jp_buf), J_(COORD_DIM * SurfDim[Sidx], Jp.begin() + COORD_DIM * SurfDsp[Sidx], false);
          Jp = 0;
          J_ = V;
          complete_vec(J, Jp);
        }
      };
      for (sctl::Long i = 0; i < Nsurf; i++) ComputeJ(J0[i], i, gmres_tol * 0.01, gmres_iter); //-1);
      sctl::Profile::Toc();

      sctl::Profile::Tic("ComputeB0", &comm);
      auto& B0 = B0_;
      B0.ReInit(Nsurf);
      auto eval_B0 = [Nsurf,&SurfDim,&SurfDsp,&SurfDimF,&SurfDspF,&Xn,&BS_op,&BI_biot_savart](sctl::Vector<Real>& B, const sctl::Vector<Real>& J) { // B <-- BiotSavart[J] + 0.5 * Xn x J, of J on the full tori
        sctl::Long N = BI_biot_savart.Dim(1);
        sctl::ScratchBuf<Real> NcrossJ_buf(N);
        sctl::Vector<Real> NcrossJ(NcrossJ_buf);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Set NcrossJ
          sctl::Long N = SurfDim[i], NF = SurfDimF[i];
          const sctl::Vector<Real> J_(COORD_DIM * NF, (sctl::Iterator<Real>)J.begin() + COORD_DIM  * SurfDspF[i], false);
          sctl::Vector<Real> NcrossJ_(COORD_DIM * N, NcrossJ.begin() + COORD_DIM  * SurfDsp[i], false);
          for (sctl::Long j = 0; j < N; j++) {
            sctl::StaticArray<Real, COORD_DIM> Xn_, V0_, nxV0_;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Xn_[k] = Xn[i][k * NF + j];
              V0_[k] = J_[k * NF + j];
            }

            nxV0_[0] = Xn_[1] * V0_[2] - Xn_[2] * V0_[1];
            nxV0_[1] = Xn_[2] * V0_[0] - Xn_[0] * V0_[2];
            nxV0_[2] = Xn_[0] * V0_[1] - Xn_[1] * V0_[0];

            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              NcrossJ_[k * N + j] = nxV0_[k];
            }
          }
        }
        if (B.Dim() != BI_biot_savart.Dim(1)) B.ReInit(BI_biot_savart.Dim(1));
        B = 0; // BI_biot_savart adds to a B of its size
        BS_op(B, J);
        for (sctl::Long i = 0; i < N; i++) B[i] += NcrossJ[i] * 0.5;
      };
      for (sctl::Long i = 0; i < Nsurf; i++) eval_B0(B0[i], J0[i]);
      sctl::Profile::Toc();

      auto eval_B = [Nsurf,NFP,NtotF,&SurfDim,&SurfDsp,&SurfDimF,&SurfOrient,&Xn,&LG_op,&BI_lap_grad](sctl::Vector<Real>& B, const sctl::Vector<Real>& F) { // B <-- dG[F] + 0.5 * Xn * F, of F on one field period
        if (B.Dim() != BI_lap_grad.Dim(1)) {
          B.ReInit(BI_lap_grad.Dim(1));
          B = 0;
        }
        sctl::ScratchBuf<Real> F_full_buf(NtotF);
        sctl::Vector<Real> F_full(F_full_buf);
        detail_bie_solvers::CompletePeriods(F_full, F, 1, 0, SurfDim, SurfDsp, SurfOrient, NFP);
        LG_op(B, F_full);
        for (sctl::Long j = 0; j < Nsurf; j++) {
          sctl::Long N = SurfDim[j], NF = SurfDimF[j];
          sctl::Long offset = SurfDsp[j];
          for (sctl::Long i = 0; i < N; i++) { // B <-- B + 0.5 * Xn * F
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              B[COORD_DIM*offset + k*N+i] += 0.5 * Xn[j][k*NF+i] * F[offset + i];
            }
          }
        }
      };
      auto eval_BdotXn = [Nsurf,&SurfDim,&SurfDsp,&SurfDimF,&Xn](sctl::Vector<Real>& BdotXn, const sctl::Vector<Real>& B) { // BdotXn <-- B . Xn
        sctl::Long N = SurfDsp[Nsurf-1] + SurfDim[Nsurf-1];
        if (BdotXn.Dim() != N) BdotXn.ReInit(N);
        for (sctl::Long j = 0; j < Nsurf; j++) {
          sctl::Long N = SurfDim[j], NF = SurfDimF[j];
          sctl::Long offset = SurfDsp[j];
          for (sctl::Long i = 0; i < N; i++) { // BdotXn = B . Xn
            BdotXn[offset + i] = 0;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              BdotXn[offset + i] += B[COORD_DIM*offset + k*N+i] * Xn[j][k*NF+i];
            }
          }
        }
      };
      // What the operator uses, behind one reference: a std::function holds a lambda of one pointer itself, where it
      // would allocate one of three
      using EvalB = decltype(eval_B);
      using EvalBdotXn = decltype(eval_BdotXn);
      using BILapGrad = std::remove_reference_t<decltype(BI_lap_grad)>;
      const struct {
        EvalB& eval_B;
        EvalBdotXn& eval_BdotXn;
        BILapGrad& BI_lap_grad;
      } fn_args{eval_B, eval_BdotXn, BI_lap_grad};
      typename sctl::GMRES<Real>::ParallelOp fn = [&fn_args](sctl::Vector<Real>* BdotXn, const sctl::Vector<Real>& F) {
        SCTL_ASSERT(BdotXn);
        sctl::ScratchBuf<Real> B_buf(fn_args.BI_lap_grad.Dim(1)); // in scratch memory: see SurfaceOp::SurfGrad
        sctl::Vector<Real> B(B_buf);
        B = 0; // eval_B adds to a B of this size
        fn_args.eval_B(B, F);
        fn_args.eval_BdotXn(*BdotXn, B);
      };
      auto compute_J = [&Nsurf,&SurfDim,&SurfDsp,&SurfDimF,&Xn](sctl::Vector<Real>& J, const sctl::Vector<Real>& B) { // J <-- B x Xn
        J.ReInit(B.Dim());
        for (sctl::Long i = 0; i < Nsurf; i++) {
          sctl::Long N = SurfDim[i], NF = SurfDimF[i];
          sctl::Long offset = SurfDsp[i];
          for (sctl::Long j = 0; j < N; j++) {
            sctl::StaticArray<Real, COORD_DIM> J_, B_, Xn_;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              B_[k] = B[COORD_DIM*offset + k*N + j];
              Xn_[k] = Xn[i][k*NF + j];
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
        // u0dotXn, negated for the right-hand side; and F, the solution of GMRES, given to it empty for the first solve
        // (a zero initial guess), which GMRES then sizes, and the initial guess of the next
        const sctl::Long Ntot = SurfDsp[Nsurf-1] + SurfDim[Nsurf-1];
        sctl::ScratchBuf<Real> u0dotXn_buf(Ntot);
        sctl::Vector<Real> u0dotXn(u0dotXn_buf);
        auto& F = F_;
        F.ReInit(0);
        sctl::GMRES<Real> solver(comm, true);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Compute B[i], J[i]
          B[i] = B0[i];
          eval_BdotXn(u0dotXn, B[i]);
          for (sctl::Long j = 0; j < Ntot; j++) u0dotXn[j] = -u0dotXn[j];
          solver(&F, fn, u0dotXn, gmres_tol, gmres_iter);
          eval_B(B[i], F);
          compute_J(J[i], B[i]);
        }
      }
      sctl::Profile::Toc();

      auto compute_tor_circ = [&Svec,&dX,&SurfDsp,&SurfOrient,NFP](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) {
        // A on one field period: its toroidal grid lines are a fraction 1/NFP of the full lines, and by the symmetry the
        // integral along them is that fraction of the integral along the full lines; counterclockwise about the z-axis
        SCTL_ASSERT(surf_id < Svec.Dim());
        const Real orient = SurfOrient[surf_id];
        const sctl::Long offset = SurfDsp[surf_id];
        const auto& S = Svec[surf_id];
        const sctl::Long NtF = S.NTor();
        const sctl::Long Nt = NtF / NFP;
        const sctl::Long Np = S.NPol();

        Real circ_ = 0;
        if (circ && circ->Dim() != Np) circ->ReInit(Np);
        for (sctl::Long p = 0; p < Np; p++) {
          Real integ0 = 0;
          for (sctl::Long t = 0; t < Nt; t++) {
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Real A0 = A[COORD_DIM*offset + k*Nt*Np + t*Np + p];
              Real dX_ = dX[surf_id][(2*k+0)*NtF*Np + t*Np + p];
              integ0 += A0 * dX_ * orient;
            }
          }
          if (circ) circ[0][p] = integ0 / Nt;
          circ_ += integ0;
        }
        return circ_ / (Nt*Np);
      };
      auto compute_pol_circ = [&Svec,&dX,&SurfDsp,&SurfPolOrient,NFP](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) {
        // A on one field period: its toroidal grid lines are a fraction 1/NFP of the full lines, and by the symmetry the
        // integral along them is that fraction of the integral along the full lines; counterclockwise in the (R, z)
        // half-plane
        SCTL_ASSERT(surf_id < Svec.Dim());
        const Real orient = SurfPolOrient[surf_id];
        const sctl::Long offset = SurfDsp[surf_id];
        const auto& S = Svec[surf_id];
        const sctl::Long NtF = S.NTor();
        const sctl::Long Nt = NtF / NFP;
        const sctl::Long Np = S.NPol();

        Real circ_ = 0;
        if (circ && circ->Dim() != Nt) circ->ReInit(Nt);
        for (sctl::Long t = 0; t < Nt; t++) {
          Real integ0 = 0;
          for (sctl::Long p = 0; p < Np; p++) {
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Real A0 = A[COORD_DIM*offset + k*Nt*Np + t*Np + p];
              Real dX_ = dX[surf_id][(2*k+1)*NtF*Np + t*Np + p];
              integ0 += A0 * dX_ * orient;
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
#endif

      sctl::Profile::Tic("CompFlux", &comm);
      { // Compute flux
        sctl::ScratchBuf<Real> A_buf(COORD_DIM * BI_laplace.Dim(1)), J_full_buf(COORD_DIM * NtotF); // A: the potential of each component of J
        sctl::Vector<Real> A(A_buf), J_full(J_full_buf);
        tor_flux.ReInit(Nsurf);
        pol_flux.ReInit(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) {
          A.SetZero();
          detail_bie_solvers::CompletePeriods(J_full, J[i], COORD_DIM, 1, SurfDim, SurfDsp, SurfOrient, NFP);
          L_op(A, J_full);
          if (Nsurf == 1) {
            tor_flux[i] = compute_pol_circ(nullptr, A, 0);
            pol_flux[i] = compute_tor_circ(nullptr, A, 0);
          } else if (Nsurf == 2) {
            tor_flux[i] = compute_pol_circ(nullptr, A, OuterSurfIdx) - compute_pol_circ(nullptr, A, InnerSurfIdx);
            pol_flux[i] = compute_tor_circ(nullptr, A, OuterSurfIdx) - compute_tor_circ(nullptr, A, InnerSurfIdx);
          } else {
            SCTL_ASSERT(false);
          }
        }
      }
      sctl::Profile::Toc();
    }



    template <class Real> TaylorState<Real>::TaylorState(const sctl::Comm& comm, const SingularQuadParams& quad) : comm_(comm), quad_(quad), BI_grad_diff_(comm), BI_grad_(comm), BI_potn_(comm) {}

    template <class Real> void TaylorState<Real>::Compute(sctl::Vector<Real>& B_out, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol, sctl::Vector<sctl::Vector<Real>>* m, sctl::Vector<sctl::Vector<Real>>* sigma) {
      TaylorState(comm).Compute(B_out, tor_flux, pol_flux, lambda, Svec, gmres_tol, gmres_iter, LB_tol, m, sigma);
    }

    template <class Real> void TaylorState<Real>::Compute(sctl::Vector<Real>& B_out, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<Surface<Real>>& Svec, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol, sctl::Vector<sctl::Vector<Real>>* m, sctl::Vector<sctl::Vector<Real>>* sigma) {
      auto& B = B_out_; // in the vectors of the object
      auto& m_ = m_out_;
      auto& sigma_ = sigma_out_;
      ComputeHelper(B, m_, sigma_, lambda, Svec, comm_, gmres_tol, gmres_iter, LB_tol);
      if (B.Dim() > 0) { // B_out <-- B[0] * tor_flux + B[1] * pol_flux, in the storage of B_out when it has the size
        const sctl::Long N = B[0].Dim();
        if (B_out.Dim() != N) B_out.ReInit(N);
        for (sctl::Long i = 0; i < N; i++) B_out[i] = B[0][i] * tor_flux;
        if (B.Dim() > 1) for (sctl::Long i = 0; i < N; i++) B_out[i] += B[1][i] * pol_flux;
      }
      if (m && sigma) {
        *m = m_;
        *sigma = sigma_;
      }
    }

    template <class Real> void TaylorState<Real>::EvalOffSurface(sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& Xtrg, sctl::Vector<Surface<Real>> Svec, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<sctl::Vector<Real>> m_, const sctl::Vector<sctl::Vector<Real>>& sigma_, const sctl::Comm& comm, const Real tol) {
      sctl::Vector<Real> m0, sigma0;
      if (Svec.Dim() > 0) m0  = m_[0] * tor_flux;
      if (Svec.Dim() > 1) m0 += m_[1] * pol_flux;
      if (Svec.Dim() > 0) sigma0  = sigma_[0] * tor_flux;
      if (Svec.Dim() > 1) sigma0 += sigma_[1] * pol_flux;

      sctl::Vector<Real> X = Xtrg, B;
      sctl::Matrix<Real>(COORD_DIM, X.Dim()/COORD_DIM, X.begin(), false) = sctl::Matrix<Real>(X.Dim()/COORD_DIM, COORD_DIM, X.begin(), false).Transpose();
      ComputeFarField(B, X, Svec, sigma0, m0, lambda, comm, tol);
      sctl::Matrix<Real>(B.Dim()/COORD_DIM, COORD_DIM, B.begin(), false) = sctl::Matrix<Real>(COORD_DIM, B.Dim()/COORD_DIM, B.begin(), false).Transpose();
      Btrg = B;
    }

    template <class Real> void TaylorState<Real>::test_conv(Real lambda, const sctl::Vector<Surface<Real>>& Svec_, sctl::Long upsample, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm, Real LB_tol, const SingularQuadParams& quad) {
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
          TaylorState(comm).ComputeHelper(B, m, sigma, lambda, Svec, comm, gmres_tol, gmres_iter);
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

        sctl::Vector<sctl::Vector<Real>> dX(Nsurf), Xn(Nsurf), Xa(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) { // Set dX, Xn, Xa
          const auto& S = Svec[i];
          sctl::Long Nt = S.NTor();
          sctl::Long Np = S.NPol();

          SurfaceOp<Real> surf_op(comm, Nt, Np);
          surf_op.Grad2D(dX[i], S.Coord());
          surf_op.SurfNormalAreaElem(&Xn[i], &Xa[i], dX[i], &S.Coord());
        }
        sctl::Long OuterSurfIdx, InnerSurfIdx;
        std::tie(OuterSurfIdx, InnerSurfIdx) = detail_bie_solvers::OuterInnerSurfIdx(Svec, Xn, Xa);
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
        FieldPeriodBIOp<Real> BI_grad_diff_op(comm);
        FieldPeriodBIOp<Real> BI_grad_op(comm);
        FieldPeriodBIOp<Real> BI_potn_op(comm);
        BI_grad_diff_op.SetupSingular(Svec, ker_grad_diff, gmres_tol * (Real)0.1, quad);
        BI_grad_op.SetupSingular(Svec, ker_grad, gmres_tol * (Real)0.1, quad);
        BI_potn_op.SetupSingular(Svec, ker_potn, gmres_tol * (Real)0.1, quad);
        sctl::Vector<sctl::Long> SurfNt(Nsurf), SurfNp(Nsurf); // the operators of densities on the surface grids
        for (sctl::Long i = 0; i < Nsurf; i++) {
          SurfNt[i] = Svec[i].NTor();
          SurfNp[i] = Svec[i].NPol();
        }
        using detail_bie_solvers::SurfGridOp;
        const SurfGridOp<FieldPeriodBIOp<Real>> BI_grad_diff{BI_grad_diff_op, SurfNt, SurfNp};
        const SurfGridOp<FieldPeriodBIOp<Real>> BI_grad{BI_grad_op, SurfNt, SurfNp};
        const SurfGridOp<FieldPeriodBIOp<Real>> BI_potn{BI_potn_op, SurfNt, SurfNp};
        sctl::Profile::Toc();

        auto Compute_m = [lambda,&Svec,&SurfDim,&SurfDsp,&dX,&Xn,&comm](sctl::Vector<Real>& m, const sctl::Vector<Real>& sigma, Real gmres_tol, sctl::Long gmres_iter) { // m <-- lambda ( i Grad(invLap(sigma)) - n x Grad(invLap(sigma)) )
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
              sctl::ScratchBuf<Real> GradInvLapSigma_buf(2 * COORD_DIM * N); // in scratch memory: see SurfaceOp::SurfGrad
              sctl::StaticArray<sctl::Vector<Real>, 2> GradInvLapSigma;
              for (sctl::Long k = 0; k < 2; k++) { // Set GradInvLapSigma <-- Grad(invLap(sigma))
                GradInvLapSigma[k].ReInit(COORD_DIM * N, GradInvLapSigma_buf.begin() + k * COORD_DIM * N, false);
                const sctl::Vector<Real> sigma__(N, (sctl::Iterator<Real>)sigma_.begin() + k * N, false);
                if (max_norm(sigma__) > 0) {
                  surf_op.GradInvSurfLap(GradInvLapSigma[k], dX[i], sigma__, gmres_tol, gmres_iter, 1.5);
                } else {
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
        auto Compute_B = [lambda,&SurfDim,&SurfDsp,&Xn,&BI_grad_diff,&BI_grad,&BI_potn,&ker_grad_diff,&ker_grad,&comm] (sctl::Vector<Real>* B, sctl::Vector<Real>* B_flux, const sctl::Vector<Real>& sigma, const sctl::Vector<Real>& m0, const sctl::Vector<Real>& mH) {
          sctl::Long Nsurf = SurfDim.Dim();
          SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
          const sctl::Long N6 = COORD_DIM * 2 * N;

          // The temporaries in scratch memory (see SurfaceOp::SurfGrad). m: m0 + mH, else the one of them given
          sctl::ScratchBuf<Real> m_buf(m0.Dim() && mH.Dim() ? m0.Dim() : 0);
          sctl::Vector<Real> m;
          if (m0.Dim() && mH.Dim()) {
            SCTL_ASSERT(m0.Dim() == mH.Dim());
            m.ReInit(m_buf.size(), m_buf.begin(), false);
            for (sctl::Long i = 0; i < m.Dim(); i++) m[i] = m0[i] + mH[i];
          } else if (m0.Dim()) {
            m.ReInit(m0.Dim(), (sctl::Iterator<Real>)m0.begin(), false);
          } else if (mH.Dim()) {
            m.ReInit(mH.Dim(), (sctl::Iterator<Real>)mH.begin(), false);
          }

          sctl::ScratchBuf<Real> Grad_v_buf(B && sigma.Dim() ? N6 : 0), iQ_buf(m.Dim() ? N6 : 0), iCurlQ_buf(B && m.Dim() ? N6 : 0), iCurlQ_m0_buf(B_flux && m0.Dim() ? N6 : 0), iCurlQ_mH_buf(B_flux && mH.Dim() ? N6 : 0);
          sctl::Vector<Real> Grad_v(Grad_v_buf), iQ(iQ_buf), iCurlQ(iCurlQ_buf), iCurlQ_m0(iCurlQ_m0_buf), iCurlQ_mH(iCurlQ_mH_buf);
          if (B && sigma.Dim()) { // Compute Grad_v
            sctl::Profile::Tic("Compute_Grad_v", &comm);
            Compute_Grad_v(Grad_v, sigma, BI_grad, ker_grad);
            sctl::Profile::Toc();
          }
          if (m.Dim()) { // Compute iQ
            sctl::Profile::Tic("Compute_iQ", &comm);
            Compute_iQ(iQ, m, SurfDim, SurfDsp, BI_potn);
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
            if (B_flux->Dim() != N6) B_flux->ReInit(N6);
            (*B_flux) = 0;
            if (m .Dim()) for (sctl::Long i = 0; i < N6; i++) (*B_flux)[i] += iQ[i]*lambda;
            if (mH.Dim()) for (sctl::Long i = 0; i < N6; i++) (*B_flux)[i] += iCurlQ_mH[i];
            if (m0.Dim()) for (sctl::Long i = 0; i < N6; i++) (*B_flux)[i] += iCurlQ_m0[i] + m0[i]*(Real)0.5;
          }
          if (B) { // B <-- -0.5*sigma_Xn - Grad_v + iQ*lambda + iCurlQ + 0.5*m
            if (B->Dim() != N6) B->ReInit(N6);
            (*B) = 0;
            if (m.Dim()) for (sctl::Long i = 0; i < N6; i++) (*B)[i] += iQ[i]*lambda + iCurlQ[i] + m[i]*(Real)0.5;
            if (sigma.Dim()) { // B <-- B - 0.5*sigma_Xn - Grad_v
              for (sctl::Long i = 0; i < Nsurf; i++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                    const sctl::Long idx0 = COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j, idx1 = idx0 + SurfDim[i];
                    (*B)[idx0] += (Xn[i][k * SurfDim[i] + j] * sigma[2 * SurfDsp[i] + 0 * SurfDim[i] + j])*(Real)(-0.5) - Grad_v[idx0];
                    (*B)[idx1] += (Xn[i][k * SurfDim[i] + j] * sigma[2 * SurfDsp[i] + 1 * SurfDim[i] + j])*(Real)(-0.5) - Grad_v[idx1];
                  }
                }
              }
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
          sctl::GMRES<Real> solver(comm, true);
          typename sctl::GMRES<Real>::ParallelOp fn = [lambda,LB_tol,&comm,&Compute_m,&Compute_B,&Compute_BdotXn,&gmres_tol,&gmres_iter](sctl::Vector<Real>* BdotXn, const sctl::Vector<Real>& sigma) {
            SCTL_ASSERT(BdotXn);
            sctl::ScratchBuf<Real> m_buf(COORD_DIM * sigma.Dim()), B_buf(COORD_DIM * sigma.Dim()); // see SurfaceOp::SurfGrad
            sctl::Vector<Real> m(m_buf), B(B_buf);
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
          auto complex_vec_scal_prod = [&SurfDim,&SurfDsp] (sctl::Vector<Real>& v, std::complex<Real> c) {
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
                  Real cv0 = v0 * c.real() - v1 * c.imag();
                  Real cv1 = v0 * c.imag() + v1 * c.real();
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
            std::complex<Real> v1dotv2(0,0);
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < dof; k++) {
                sctl::Long offset = SurfDsp[i]*dof*2 + k*2*SurfDim[i];
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  std::complex<Real> c0(v0[offset + 0*SurfDim[i] + j], v0[offset + 1*SurfDim[i] + j]);
                  std::complex<Real> c1(v1[offset + 0*SurfDim[i] + j], v1[offset + 1*SurfDim[i] + j]);
                  v1dotv2 += c0 * c1;
                }
              }
            }
            return v1dotv2;
          };

          sctl::Matrix<std::complex<Real>> b(B_taylor.Dim(), 1);
          sctl::Matrix<std::complex<Real>> M(B_taylor.Dim(), B_taylor.Dim());
          for (sctl::Long i = 0; i < B_taylor.Dim(); i++) { // Set b, M
            b[i][0] = complex_vec_dot_prod(B,B_taylor[i]);
            for (sctl::Long j = 0; j < B_taylor.Dim(); j++) {
              M[i][j] = complex_vec_dot_prod(B_taylor[i],B_taylor[j]);
            }
          }

          sctl::Matrix<std::complex<Real>> Minv(B_taylor.Dim(), B_taylor.Dim());
          if (M.Dim(0) == 1 && M.Dim(1) == 1) { // Set Minv <-- inv(M)
            Minv[0][0] = Real(1) / M[0][0];
          } else if (M.Dim(0) == 2 && M.Dim(1) == 2) {
            std::complex<Real> oodet = Real(1) / (M[0][0] * M[1][1] - M[0][1] * M[1][0]);
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

    template <class Real> Real TaylorState<Real>::max_norm(const sctl::Vector<Real>& x) {
      Real err = 0;
      for (const auto& a : x) err = std::max(err, sctl::fabs<Real>(a));
      return err;
    }

    template <class Real> void TaylorState<Real>::ComputeFarField(sctl::Vector<Real>& Breal, const sctl::Vector<Real>& Xt, const sctl::Vector<Surface<Real>>& Svec, const sctl::Vector<Real>& sigma, const sctl::Vector<Real>& m, Real lambda, const sctl::Comm& comm, const Real tol) {
      Helmholtz3D<Real> helm(lambda);
      const sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == Nt * COORD_DIM);

      sctl::Vector<Real> Ugrad(4 * COORD_DIM * 2 * Nt), Upotn(COORD_DIM * 2 * Nt); // Grad(g[sigma]), Grad(g[i m_k]) for each k; and g[i m_k] for each k
      Ugrad.SetZero();
      Upotn.SetZero();
      sctl::Long dsp = 0;
      for (sctl::Long i = 0; i < Svec.Dim(); i++) {
        const auto& S = Svec[i];
        const sctl::Long N = S.NTor() * S.NPol();
        sctl::Vector<Real> Fgrad(4 * 2 * N); // sigma, then i m_k for each k (complex)
        for (sctl::Long j = 0; j < 2 * N; j++) Fgrad[j] = sigma[2 * dsp + j];
        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          for (sctl::Long j = 0; j < N; j++) {
            Fgrad[((k + 1) * 2 + 0) * N + j] = -m[COORD_DIM * 2 * dsp + (k * 2 + 1) * N + j];
            Fgrad[((k + 1) * 2 + 1) * N + j] =  m[COORD_DIM * 2 * dsp + (k * 2 + 0) * N + j];
          }
        }
        const sctl::Vector<Real> Fpotn(COORD_DIM * 2 * N, Fgrad.begin() + 2 * N, false);

        OffSurfaceOp op_grad(helm.FxdU(), comm), op_potn(helm.FxU(), comm);
        op_grad.Setup(S.Coord(), S.NTor(), S.NPol(), Xt, tol, Fgrad);
        op_grad.Eval(Ugrad, Fgrad);
        op_potn.Setup(S.Coord(), S.NTor(), S.NPol(), Xt, tol, Fpotn);
        op_potn.Eval(Upotn, Fpotn);
        dsp += N;
      }

      sctl::Vector<Real> B(COORD_DIM * 2 * Nt); // - Grad(g[sigma]) + lambda g[i m] + Curl(g[i m])
      for (sctl::Long j = 0; j < COORD_DIM * 2 * Nt; j++) B[j] = Upotn[j] * lambda - Ugrad[j];
      for (sctl::Long k = 0; k < COORD_DIM; k++) { // Curl(g[i m_k e_k]) = Grad(g[i m_k]) x e_k
        const sctl::Vector<Real> iGradQ_(COORD_DIM * 2 * Nt, Ugrad.begin() + (k + 1) * COORD_DIM * 2 * Nt, false);
        const sctl::Long k0 = (k + 2) % COORD_DIM;
        const sctl::Long k1 = (k + 1) % COORD_DIM;
        for (sctl::Long j = 0; j < Nt; j++) {
          B[(k1*2+0) * Nt + j] += iGradQ_[(k0*2+0) * Nt + j];
          B[(k1*2+1) * Nt + j] += iGradQ_[(k0*2+1) * Nt + j];
          B[(k0*2+0) * Nt + j] -= iGradQ_[(k1*2+0) * Nt + j];
          B[(k0*2+1) * Nt + j] -= iGradQ_[(k1*2+1) * Nt + j];
        }
      }

      if (Breal.Dim() != COORD_DIM * Nt) Breal.ReInit(COORD_DIM * Nt);
      for (sctl::Long j = 0; j < Nt; j++) { // Breal <-- real(B)
        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          Breal[k * Nt + j] = B[k * 2 * Nt + j];
        }
      }
    }

    template <class Real> void TaylorState<Real>::Compute_mH(sctl::Vector<Real>& mH, sctl::Long Nt, sctl::Long Np, sctl::Integer NFP, sctl::Integer tor_orient, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& Xn, const sctl::Comm& comm, Real tol, sctl::Long max_iter) {
      sctl::Long N= Nt * Np;

      // In scratch memory: see SurfaceOp::SurfGrad
      sctl::ScratchBuf<Real> V_buf(COORD_DIM * N), Vn_buf(COORD_DIM * N), Vd_buf(COORD_DIM * N), Vc_buf(COORD_DIM * N), Vh_buf(COORD_DIM * N);
      sctl::Vector<Real> V(V_buf);
      sctl::omp_par::parallel_for(N, [&V,N,&dX](const sctl::Long i) { // Set V
        for (sctl::Long k =0; k < COORD_DIM; k++) {
          V[k * N + i] = dX[(k*2+0) * N + i];
        }
      });

      sctl::Vector<Real> Vn(Vn_buf), Vd(Vd_buf), Vc(Vc_buf), Vh(Vh_buf);
      SurfaceOp<Real> surf_op(comm, Nt, Np, NFP, tor_orient);
      surf_op.HodgeDecomp(Vn, Vd, Vc, Vh, V, dX, Xn, tol, max_iter);

      if (mH.Dim() != COORD_DIM * 2 * N) mH.ReInit(COORD_DIM * 2 * N);
      sctl::omp_par::parallel_for(N, [&Xn,N,&Vh,&mH](const sctl::Long i) { // Set mH
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
      });
    }
    template <class Real> template <class Op> void TaylorState<Real>::Compute_Grad_v(sctl::Vector<Real>& Grad_v, const sctl::Vector<Real>& sigma, const Op& BI_grad, const KernelFunction<Real,COORD_DIM,2,COORD_DIM*2>& ker_grad) { // grad_v <-- Grad(g[sigma])
      Grad_v = 0;
      BI_grad(Grad_v, sigma);
    }
    template <class Real> template <class Op> void TaylorState<Real>::Compute_iQ(sctl::Vector<Real>& iQ, const sctl::Vector<Real>& m, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const Op& BI_potn) { // iQ <-- i g[m]
      sctl::Long Nsurf = SurfDim.Dim();
      sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
      if (iQ.Dim() != COORD_DIM * 2 * N) iQ.ReInit(COORD_DIM * 2 * N);
      SCTL_ASSERT(m.Dim() % (COORD_DIM * 2) == 0); // on the grids of BI, the full tori

      iQ = 0;
      BI_potn(iQ, m);

      for (sctl::Long k = 0; k < COORD_DIM; k++) {
        for (sctl::Long i = 0; i < Nsurf; i++) {
          sctl::omp_par::parallel_for(SurfDim[i], [&iQ,&SurfDsp,i,k,&SurfDim](const sctl::Long j) {
            Real real_Q = iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j];
            Real imag_Q = iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j];
            iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j] = -imag_Q;
            iQ[COORD_DIM * 2 * SurfDsp[i] + (k*2+1) * SurfDim[i] + j] =  real_Q;
          });
        }
      }
    }
    template <class Real> template <class Op> void TaylorState<Real>::Compute_iCurlQ(sctl::Vector<Real>& iCurlQ, const sctl::Vector<Real>& m, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const Op& BI_grad, const KernelFunction<Real,COORD_DIM,2,COORD_DIM*2>& ker_grad) { // iCurlQ <-- i Curl(g[m])
      sctl::Long Nsurf = SurfDim.Dim();
      sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
      if (iCurlQ.Dim() != COORD_DIM * 2 * N) iCurlQ.ReInit(COORD_DIM * 2 * N);
      SCTL_ASSERT(m.Dim() % (COORD_DIM * 2) == 0); // on the grids of BI, the full tori

      sctl::ScratchBuf<Real> GradQ_buf(COORD_DIM * COORD_DIM * 2 * N); // in scratch memory: see SurfaceOp::SurfGrad
      sctl::Vector<Real> GradQ_(GradQ_buf);
      GradQ_ = 0;
      BI_grad(GradQ_, m);

      iCurlQ = 0;
      for (sctl::Long k = 0; k < COORD_DIM; k++) {
        sctl::Long k0 = (k + 2) % COORD_DIM;
        sctl::Long k1 = (k + 1) % COORD_DIM;
        for (sctl::Long i = 0; i < Nsurf; i++) {
          sctl::omp_par::parallel_for(SurfDim[i], [&iCurlQ,&SurfDsp,i,k1,&SurfDim,&GradQ_,k,k0](const sctl::Long j) {
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k1*2+0) * SurfDim[i] + j] += -GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k0)*2+1) * SurfDim[i] + j];
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k1*2+1) * SurfDim[i] + j] +=  GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k0)*2+0) * SurfDim[i] + j];
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k0*2+0) * SurfDim[i] + j] -= -GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k1)*2+1) * SurfDim[i] + j];
            iCurlQ[COORD_DIM * 2 * SurfDsp[i] + (k0*2+1) * SurfDim[i] + j] -=  GradQ_[COORD_DIM * COORD_DIM * 2 * SurfDsp[i] + ((k*COORD_DIM+k1)*2+0) * SurfDim[i] + j];
          });
        }
      }
    }

    template <class Real> void TaylorState<Real>::ComputeHelper(sctl::Vector<sctl::Vector<Real>>& B_out, sctl::Vector<sctl::Vector<Real>>& m_out, sctl::Vector<sctl::Vector<Real>>& sigma_out, Real lambda, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol) {
      sctl::Long Nsurf = Svec.Dim();
      if (Nsurf != 1 && Nsurf != 2) return;
      // The vectors of the solve are those of the object, which keeps them with their storage from one call to the next
      // (see TaylorState()); the temporaries are in scratch memory (see SurfaceOp::SurfGrad). The solve is on one of the
      // NFP field periods common to the surfaces: sigma, m and the Laplace-Beltrami solves that give m from sigma
      // (SurfaceOp on one field period), and the fields B, the targets of the operators. The densities of the operators,
      // sigma and m, are completed by the symmetry to the full tori, on which the geometry is computed.
      const sctl::Integer NFP = detail_fieldperiod::CommonNFP(Svec);
      auto& SurfDim = SurfDim_; // one field period
      auto& SurfDsp = SurfDsp_;
      auto& SurfDimF = SurfDimF_; // the full torus
      auto& SurfDspF = SurfDspF_;
      auto& SurfOrient = SurfOrient_;
      detail_bie_solvers::SurfDims(SurfDim, SurfDsp, SurfDimF, SurfDspF, SurfNt_, SurfNp_, SurfOrient, Svec, NFP);
      const sctl::Long Ntot = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1], NtotF = NFP * Ntot;
      const auto complete_sigma = [&SurfDim,&SurfDsp,&SurfOrient,NFP](sctl::Vector<Real>& sigma_full, const sctl::Vector<Real>& sigma) { // a complex scalar
        detail_bie_solvers::CompletePeriods(sigma_full, sigma, 2, 0, SurfDim, SurfDsp, SurfOrient, NFP);
      };
      const auto complete_m = [&SurfDim,&SurfDsp,&SurfOrient,NFP](sctl::Vector<Real>& m_full, const sctl::Vector<Real>& m) { // a complex vector
        detail_bie_solvers::CompletePeriods(m_full, m, COORD_DIM * 2, 2, SurfDim, SurfDsp, SurfOrient, NFP);
      };

      auto& dX = dX_;
      auto& Xn = Xn_;
      auto& Xa = Xa_;
      dX.ReInit(Nsurf);
      Xn.ReInit(Nsurf);
      Xa.ReInit(Nsurf);
      auto& SurfPolOrient = SurfPolOrient_;
      SurfPolOrient.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set dX, Xn, Xa, SurfPolOrient
        const auto& S = Svec[i];
        sctl::Long Nt = S.NTor();
        sctl::Long Np = S.NPol();

        SurfaceOp<Real> surf_op(comm, Nt, Np);
        surf_op.Grad2D(dX[i], S.Coord());
        const Real normal_orient = surf_op.SurfNormalAreaElem(&Xn[i], &Xa[i], dX[i], &S.Coord()); // +1 if dX/dt x dX/dp points out of the torus
        SurfPolOrient[i] = SurfOrient[i] * (sctl::Integer)normal_orient;
      }
      sctl::Long OuterSurfIdx, InnerSurfIdx;
      std::tie(OuterSurfIdx, InnerSurfIdx) = detail_bie_solvers::OuterInnerSurfIdx(Svec, Xn, Xa);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Set normal to outer-normal
        Real scal = 0;
        if (i == InnerSurfIdx) scal = -1;
        if (i == OuterSurfIdx) scal = 1;
        Xn[i] *= scal;
      }
      auto& dXp = dXp_; // on one field period, for the Laplace-Beltrami solves
      auto& Xnp = Xnp_;
      detail_bie_solvers::FirstPeriod(dXp, dX, SurfDim, SurfDimF);
      detail_bie_solvers::FirstPeriod(Xnp, Xn, SurfDim, SurfDimF);

      sctl::Profile::Tic("Setup", &comm);
      Helmholtz3D<Real> helm(lambda);
      HelmholtzDiff3D<Real> helm_diff(lambda);
      const auto& ker_potn = helm.FxU();
      const auto& ker_grad = helm.FxdU();
      const auto& ker_grad_diff = helm_diff.FxdU();
      auto& BI_grad_diff = BI_grad_diff_; // the operators of the object: see TaylorState()
      auto& BI_grad = BI_grad_;
      auto& BI_potn = BI_potn_;
      BI_grad_diff.SetupSingular(Svec, ker_grad_diff, gmres_tol * (Real)0.1, quad_);
      BI_grad.SetupSingular(Svec, ker_grad, gmres_tol * (Real)0.1, quad_);
      BI_potn.SetupSingular(Svec, ker_potn, gmres_tol * (Real)0.1, quad_);
      // the operators of densities on the surface grids
      using detail_bie_solvers::SurfGridOp;
      const SurfGridOp<std::remove_reference_t<decltype(BI_grad_diff)>> GD_op{BI_grad_diff, SurfNt_, SurfNp_};
      const SurfGridOp<std::remove_reference_t<decltype(BI_grad)>> G_op{BI_grad, SurfNt_, SurfNp_};
      const SurfGridOp<std::remove_reference_t<decltype(BI_potn)>> P_op{BI_potn, SurfNt_, SurfNp_};
      sctl::Profile::Toc();

      auto Compute_m = [lambda,NFP,&Svec,&SurfDim,&SurfDsp,&SurfOrient,&dX = dXp,&Xn = Xnp,&comm](sctl::Vector<Real>& m, const sctl::Vector<Real>& sigma, Real gmres_tol, sctl::Long gmres_iter) { // m <-- lambda ( i Grad(invLap(sigma)) - n x Grad(invLap(sigma)) ), on one field period
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
            sctl::Long Nt = S.NTor() / NFP;
            sctl::Long Np = S.NPol();
            sctl::Long N= Nt * Np;

            SurfaceOp<Real> surf_op(comm, Nt, Np, NFP, SurfOrient[i]);
            sctl::ScratchBuf<Real> GradInvLapSigma_buf(2 * COORD_DIM * N); // in scratch memory: see SurfaceOp::SurfGrad
            sctl::StaticArray<sctl::Vector<Real>, 2> GradInvLapSigma;
            for (sctl::Long k = 0; k < 2; k++) { // Set GradInvLapSigma <-- Grad(invLap(sigma))
              GradInvLapSigma[k].ReInit(COORD_DIM * N, GradInvLapSigma_buf.begin() + k * COORD_DIM * N, false);
              const sctl::Vector<Real> sigma__(N, (sctl::Iterator<Real>)sigma_.begin() + k * N, false);
              if (max_norm(sigma__) > 0) {
                surf_op.GradInvSurfLap(GradInvLapSigma[k], dX[i], sigma__, gmres_tol, gmres_iter, 1.5);
              } else {
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
              SurfaceOp<Real> surf_op(comm, S.NTor() / NFP, S.NPol(), NFP, SurfOrient[i]);
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
      auto Compute_B = [lambda,&SurfDim,&SurfDsp,&SurfDimF,&SurfDspF,&Xn,&GD_op,&G_op,&P_op,&ker_grad_diff,&ker_grad,&comm] (sctl::Vector<Real>* B, sctl::Vector<Real>* B_flux, const sctl::Vector<Real>& sigma, const sctl::Vector<Real>& m0, const sctl::Vector<Real>& mH) { // B and B_flux on one field period, of sigma, m0 and mH on the full tori
        sctl::Long Nsurf = SurfDim.Dim();
        // f(i, iF) for the index i of each point of a field of 6 components on one field period, and its index iF in the
        // field on the full tori
        const auto for_each_period = [&SurfDim,&SurfDsp,&SurfDimF,&SurfDspF,Nsurf](const auto& f) {
          for (sctl::Long s = 0; s < Nsurf; s++) {
            for (sctl::Integer q = 0; q < COORD_DIM * 2; q++) {
              for (sctl::Long j = 0; j < SurfDim[s]; j++) f(COORD_DIM * 2 * SurfDsp[s] + q * SurfDim[s] + j, COORD_DIM * 2 * SurfDspF[s] + q * SurfDimF[s] + j);
            }
          }
        };
        SCTL_ASSERT(SurfDsp.Dim() == Nsurf);
        sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
        const sctl::Long N6 = COORD_DIM * 2 * N;

        // The temporaries in scratch memory (see SurfaceOp::SurfGrad). m: m0 + mH, else the one of them given
        sctl::ScratchBuf<Real> m_buf(m0.Dim() && mH.Dim() ? m0.Dim() : 0);
        sctl::Vector<Real> m;
        if (m0.Dim() && mH.Dim()) {
          SCTL_ASSERT(m0.Dim() == mH.Dim());
          m.ReInit(m_buf.size(), m_buf.begin(), false);
          for (sctl::Long i = 0; i < m.Dim(); i++) m[i] = m0[i] + mH[i];
        } else if (m0.Dim()) {
          m.ReInit(m0.Dim(), (sctl::Iterator<Real>)m0.begin(), false);
        } else if (mH.Dim()) {
          m.ReInit(mH.Dim(), (sctl::Iterator<Real>)mH.begin(), false);
        }

        sctl::ScratchBuf<Real> Grad_v_buf(B && sigma.Dim() ? N6 : 0), iQ_buf(m.Dim() ? N6 : 0), iCurlQ_buf(B && m.Dim() ? N6 : 0), iCurlQ_m0_buf(B_flux && m0.Dim() ? N6 : 0), iCurlQ_mH_buf(B_flux && mH.Dim() ? N6 : 0);
        sctl::Vector<Real> Grad_v(Grad_v_buf), iQ(iQ_buf), iCurlQ(iCurlQ_buf), iCurlQ_m0(iCurlQ_m0_buf), iCurlQ_mH(iCurlQ_mH_buf);
        if (B && sigma.Dim()) { // Compute Grad_v
          sctl::Profile::Tic("Compute_Grad_v", &comm);
          Compute_Grad_v(Grad_v, sigma, G_op, ker_grad);
          sctl::Profile::Toc();
        }
        if (m.Dim()) { // Compute iQ
          sctl::Profile::Tic("Compute_iQ", &comm);
          Compute_iQ(iQ, m, SurfDim, SurfDsp, P_op);
          sctl::Profile::Toc();
        }
        if (B && m.Dim()) { // Compute iCurlQ
          sctl::Profile::Tic("Compute_iCurlQ", &comm);
          Compute_iCurlQ(iCurlQ, m, SurfDim, SurfDsp, G_op, ker_grad);
          sctl::Profile::Toc();
        }
        if (B_flux) { // Compute iCurlQ_m0, iCurlQ_mH
          sctl::Profile::Tic("Compute_iCurlQ_", &comm);
          if (m0.Dim()) Compute_iCurlQ(iCurlQ_m0, m0, SurfDim, SurfDsp, G_op , ker_grad     );
          if (mH.Dim()) Compute_iCurlQ(iCurlQ_mH, mH, SurfDim, SurfDsp, GD_op, ker_grad_diff);
          sctl::Profile::Toc();
        }

        if (B_flux) { // B_flux <-- iQ*lambda + iCurlQ_mH + iCurlQ_m0 + 0.5*m0
          if (B_flux->Dim() != N6) B_flux->ReInit(N6);
          (*B_flux) = 0;
          if (m .Dim()) for (sctl::Long i = 0; i < N6; i++) (*B_flux)[i] += iQ[i]*lambda;
          if (mH.Dim()) for (sctl::Long i = 0; i < N6; i++) (*B_flux)[i] += iCurlQ_mH[i];
          if (m0.Dim()) for_each_period([B_flux,&iCurlQ_m0,&m0](const sctl::Long i, const sctl::Long iF) { (*B_flux)[i] += iCurlQ_m0[i] + m0[iF]*(Real)0.5; });
        }
        if (B) { // B <-- -0.5*sigma_Xn - Grad_v + iQ*lambda + iCurlQ + 0.5*m
          if (B->Dim() != N6) B->ReInit(N6);
          (*B) = 0;
          if (m.Dim()) for_each_period([B,&iQ,&lambda,&iCurlQ,&m](const sctl::Long i, const sctl::Long iF) { (*B)[i] += iQ[i]*lambda + iCurlQ[i] + m[iF]*(Real)0.5; });
          if (sigma.Dim()) { // B <-- B - 0.5*sigma_Xn - Grad_v
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < COORD_DIM; k++) {
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  const sctl::Long idx0 = COORD_DIM * 2 * SurfDsp[i] + (k*2+0) * SurfDim[i] + j, idx1 = idx0 + SurfDim[i];
                  (*B)[idx0] += (Xn[i][k * SurfDimF[i] + j] * sigma[2 * SurfDspF[i] + 0 * SurfDimF[i] + j])*(Real)(-0.5) - Grad_v[idx0];
                  (*B)[idx1] += (Xn[i][k * SurfDimF[i] + j] * sigma[2 * SurfDspF[i] + 1 * SurfDimF[i] + j])*(Real)(-0.5) - Grad_v[idx1];
                }
              }
            }
          }
        }
      };
      auto Compute_BdotXn = [&SurfDim,&SurfDsp,&SurfDimF,&Xn](sctl::Vector<Real>& BdotXn, const sctl::Vector<Real>& B) { // on one field period
        sctl::Long Nsurf = SurfDim.Dim();
        SCTL_ASSERT(SurfDsp.Dim() == Nsurf);

        sctl::Long N = B.Dim() / (COORD_DIM * 2);
        SCTL_ASSERT(B.Dim() == COORD_DIM * 2 * N);
        if (BdotXn.Dim() != 2 * N) BdotXn.ReInit(2 * N);
        for (sctl::Long i = 0; i < Nsurf; i++) {
          for (sctl::Long j = 0; j < SurfDim[i]; j++) {
            Real B0dotXn = 0, B1dotXn = 0;
            for (sctl::Long k = 0; k < COORD_DIM; k++) {
              Real Xn_ = Xn[i][k * SurfDimF[i] + j];
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
      auto solve_taylor = [lambda,LB_tol,NFP,&comm,&complete_sigma,&complete_m,&Compute_m,&Compute_B,&Compute_BdotXn] (sctl::Vector<Real>& sigma, const sctl::Vector<Real>& rhs, Real gmres_tol, sctl::Long gmres_iter) { // sigma on one field period
        sctl::Profile::Tic("Solve", &comm);
        sctl::GMRES<Real> solver(comm, true);
        // What the operator uses, behind one reference: a std::function holds a lambda of one pointer itself, where it
        // would allocate one of eight
        using CompleteSigma = std::remove_reference_t<decltype(complete_sigma)>;
        using CompleteM = std::remove_reference_t<decltype(complete_m)>;
        using ComputeM = decltype(Compute_m);
        using ComputeB = decltype(Compute_B);
        using ComputeBdotXn = decltype(Compute_BdotXn);
        const struct {
          Real lambda, LB_tol;
          sctl::Integer NFP;
          const sctl::Comm& comm;
          CompleteSigma& complete_sigma;
          CompleteM& complete_m;
          ComputeM& Compute_m;
          ComputeB& Compute_B;
          ComputeBdotXn& Compute_BdotXn;
          Real gmres_tol;
          sctl::Long gmres_iter;
        } args{lambda, LB_tol, NFP, comm, complete_sigma, complete_m, Compute_m, Compute_B, Compute_BdotXn, gmres_tol, gmres_iter};
        typename sctl::GMRES<Real>::ParallelOp fn = [&args](sctl::Vector<Real>* BdotXn, const sctl::Vector<Real>& sigma) {
          SCTL_ASSERT(BdotXn);
          // m and B on one field period, sigma and m completed to the full tori, in scratch memory (see
          // SurfaceOp::SurfGrad)
          sctl::ScratchBuf<Real> m_buf(COORD_DIM * sigma.Dim()), sigma_full_buf(args.NFP * sigma.Dim()), m_full_buf(COORD_DIM * args.NFP * sigma.Dim()), B_buf(COORD_DIM * sigma.Dim());
          sctl::Vector<Real> m(m_buf), sigma_full(sigma_full_buf), m_full(m_full_buf), B(B_buf);
          sctl::Profile::Tic("Compute_B", &args.comm);
          args.Compute_m(m, sigma, args.gmres_tol / args.lambda * args.LB_tol, args.gmres_iter);
          args.complete_sigma(sigma_full, sigma);
          args.complete_m(m_full, m);
          args.Compute_B(&B, nullptr, sigma_full, m_full, sctl::Vector<Real>());
          args.Compute_BdotXn(*BdotXn, B);
          { // Project to space of mean-zero functions : BdotXn <-- P BdotXn
            // TODO
          }
          sctl::Profile::Toc();
        };
        sigma.ReInit(0); // a zero initial guess, in the storage that the object keeps: GMRES sizes it
        solver(&sigma, fn, rhs, gmres_tol, gmres_iter);
        sctl::Profile::Toc();
      };

      auto& mH = mH_; // on the full tori, from one field period
      auto& B0 = B0_;
      mH.ReInit(Nsurf);
      B0.ReInit(Nsurf);
      sctl::Profile::Tic("Compute_B0", &comm);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Compute mH, B0
        {
          sctl::ScratchBuf<Real> mH_p_buf(COORD_DIM * 2 * Ntot);
          sctl::Vector<Real> mH_p(mH_p_buf), mH_i(COORD_DIM * 2 * SurfDim[i], mH_p.begin() + COORD_DIM * 2 * SurfDsp[i], false);
          mH_p = 0;
          Compute_mH(mH_i, SurfNt_[i] / NFP, SurfNp_[i], NFP, SurfOrient[i], dXp[i], Xnp[i], comm, gmres_tol * LB_tol, gmres_iter);
          complete_m(mH[i], mH_p);
        }
        sctl::ScratchBuf<Real> neg_mH_buf(mH[i].Dim());
        sctl::Vector<Real> neg_mH(neg_mH_buf);
        for (sctl::Long j = 0; j < neg_mH.Dim(); j++) neg_mH[j] = -mH[i][j];
        Compute_B(&B0[i], nullptr, sctl::Vector<Real>(), sctl::Vector<Real>(), neg_mH);
      }
      sctl::Profile::Toc();

      auto& sigma = sigma_; // on one field period
      auto& m = m_; // on the full tori, from one field period
      auto& B = B_; // on one field period
      auto& B_flux = B_flux_;
      sigma.ReInit(Nsurf);
      m.ReInit(Nsurf);
      B.ReInit(Nsurf);
      B_flux.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // Solve
        sctl::ScratchBuf<Real> rhs_buf(2 * Ntot);
        sctl::Vector<Real> rhs(rhs_buf);
        Compute_BdotXn(rhs, B0[i]);
        solve_taylor(sigma[i], rhs, gmres_tol, gmres_iter);
        sctl::ScratchBuf<Real> sigma_full_buf(2 * NtotF), m_p_buf(COORD_DIM * 2 * Ntot);
        sctl::Vector<Real> sigma_full(sigma_full_buf), m_p(m_p_buf);
        complete_sigma(sigma_full, sigma[i]);
        Compute_m(m_p, sigma[i], gmres_tol * LB_tol, gmres_iter);
        complete_m(m[i], m_p);
        Compute_B(&B[i], &B_flux[i], sigma_full, m[i], mH[i]);
        m[i] += mH[i];
      }

      sctl::Profile::Tic("CompFlux", &comm);
      { // Compute flux
        auto compute_tor_circ = [&Svec,&SurfDim,&SurfDsp,&dX,&SurfOrient,NFP](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) { // of A on one field period (see VacuumField)
          sctl::Long Nsurf = Svec.Dim();
          SCTL_ASSERT(surf_id < Nsurf);
          const Real orient = SurfOrient[surf_id];
          const auto& S = Svec[surf_id];
          sctl::Long NtF = S.NTor();
          sctl::Long Nt = NtF / NFP;
          sctl::Long Np = S.NPol();

          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
          sctl::Long dof = A.Dim() / (N * COORD_DIM);
          SCTL_ASSERT(A.Dim() == N * COORD_DIM * dof);
          sctl::Long offset = SurfDsp[surf_id] * COORD_DIM * dof;

          std::complex<Real> circ_ = 0;
          sctl::ScratchBuf<Real> circ_vec_buf(dof*Np);
          sctl::Vector<Real> circ_vec(circ_vec_buf);
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long p = 0; p < Np; p++) {
              Real integ0 = 0;
              for (sctl::Long t = 0; t < Nt; t++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  Real A0 = A[offset + (k*dof+d)*Nt*Np + t*Np + p];
                  Real dX_ = dX[surf_id][(2*k+0)*NtF*Np + t*Np + p];
                  integ0 += A0 * dX_ * orient;
                }
              }
              circ_vec[d * Np + p] = integ0 / Nt;
              if (!d) circ_.real(circ_.real() + integ0 / (Nt*Np));
              else    circ_.imag(circ_.imag() + integ0 / (Nt*Np));
            }
          }
          if (circ) (*circ) = circ_vec;
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long p = 0; p < Np; p++) {
              if (!d) circ_vec[d * Np + p] -= circ_.real();
              else    circ_vec[d * Np + p] -= circ_.imag();
            }
          }
          return circ_;
        };
        auto compute_pol_circ = [&Svec,&SurfDim,&SurfDsp,&dX,&SurfPolOrient,NFP](sctl::Vector<Real>* circ, const sctl::Vector<Real>& A, sctl::Long surf_id) { // of A on one field period (see VacuumField)
          sctl::Long Nsurf = Svec.Dim();
          SCTL_ASSERT(surf_id < Nsurf);
          const Real orient = SurfPolOrient[surf_id];
          const auto& S = Svec[surf_id];
          sctl::Long NtF = S.NTor();
          sctl::Long Nt = NtF / NFP;
          sctl::Long Np = S.NPol();

          sctl::Long N = SurfDim[Nsurf-1] + SurfDsp[Nsurf-1];
          sctl::Long dof = A.Dim() / (N * COORD_DIM);
          SCTL_ASSERT(A.Dim() == N * COORD_DIM * dof);
          sctl::Long offset = SurfDsp[surf_id] * COORD_DIM * dof;

          std::complex<Real> circ_ = 0;
          sctl::ScratchBuf<Real> circ_vec_buf(dof*Nt);
          sctl::Vector<Real> circ_vec(circ_vec_buf);
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long t = 0; t < Nt; t++) {
              Real integ0 = 0;
              for (sctl::Long p = 0; p < Np; p++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) {
                  Real A0 = A[offset + (k*dof+d)*Nt*Np + t*Np + p];
                  Real dX_ = dX[surf_id][(2*k+1)*NtF*Np + t*Np + p];
                  integ0 += A0 * dX_ * orient;
                }
              }
              circ_vec[d * Nt + t] = integ0 / Np;
              if (!d) circ_.real(circ_.real() + integ0 / (Nt*Np));
              else    circ_.imag(circ_.imag() + integ0 / (Nt*Np));
            }
          }
          if (circ) (*circ) = circ_vec;
          for (sctl::Long d = 0; d < dof; d++) {
            for (sctl::Long t = 0; t < Nt; t++) {
              if (!d) circ_vec[d * Nt + t] -= circ_.real();
              else    circ_vec[d * Nt + t] -= circ_.imag();
            }
          }
          return circ_;
        };
        auto complex_vec_prod = [] (sctl::Vector<Real>& v, Real c0, Real c1, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp) { // of v on the grids SurfDim, SurfDsp
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
        // B_out: the real part of B on one field period, completed to the full tori
        auto real_and_complete = [&real_imag_part,&SurfDim,&SurfDsp,&SurfOrient,NFP,Ntot] (sctl::Vector<Real>& B_out, const sctl::Vector<Real>& B) {
          sctl::ScratchBuf<Real> B_real_buf(COORD_DIM * Ntot);
          sctl::Vector<Real> B_real(B_real_buf);
          real_imag_part(B_real, B, true);
          detail_bie_solvers::CompletePeriods(B_out, B_real, COORD_DIM, 1, SurfDim, SurfDsp, SurfOrient, NFP);
        };
        auto print_imag_norm = [&real_imag_part] (const sctl::Vector<Real>& B) { // Print | imag(B) |_inf
#ifdef BIEST_VERBOSE
          sctl::Vector<Real> B_imag;
          real_imag_part(B_imag, B, false);
          std::cout<<"Error : | imag(B) |_inf / | real(B) |_inf = "<<max_norm(B_imag) / max_norm(B)<<'\n';
#endif
        };

        if (Nsurf == 1) {
          if (B_out.Dim() != 1) B_out.ReInit(1);
          if (m_out.Dim() != 1) m_out.ReInit(1);
          if (sigma_out.Dim() != 1) sigma_out.ReInit(1);

          // TODO: change flux evaluation only on curve
          std::complex<Real> tor_flux = compute_pol_circ(nullptr, B_flux[0], 0) / lambda;
          std::complex<Real> scal = std::complex<Real>(1,0) / tor_flux;
          complex_vec_prod(B[0], scal.real(), scal.imag(), SurfDim, SurfDsp);
          complex_vec_prod(m[0], scal.real(), scal.imag(), SurfDimF, SurfDspF);
          complex_vec_prod(sigma[0], scal.real(), scal.imag(), SurfDim, SurfDsp);
          real_and_complete(B_out[0], B[0]);
          complete_sigma(sigma_out[0], sigma[0]);
          m_out[0] = m[0];

          print_imag_norm(B[0]);
        } else if (Nsurf == 2) {
          if (B_out.Dim() != 2) B_out.ReInit(2);
          if (m_out.Dim() != 2) m_out.ReInit(2);
          if (sigma_out.Dim() != 2) sigma_out.ReInit(2);

          std::complex<Real> tor_flux0 = (compute_pol_circ(nullptr, B_flux[0], OuterSurfIdx) - compute_pol_circ(nullptr, B_flux[0], InnerSurfIdx)) / lambda;
          std::complex<Real> tor_flux1 = (compute_pol_circ(nullptr, B_flux[1], OuterSurfIdx) - compute_pol_circ(nullptr, B_flux[1], InnerSurfIdx)) / lambda;
          std::complex<Real> pol_flux0 = (compute_tor_circ(nullptr, B_flux[0], OuterSurfIdx) - compute_tor_circ(nullptr, B_flux[0], InnerSurfIdx)) / lambda;
          std::complex<Real> pol_flux1 = (compute_tor_circ(nullptr, B_flux[1], OuterSurfIdx) - compute_tor_circ(nullptr, B_flux[1], InnerSurfIdx)) / lambda;

          std::complex<Real> M[2][2];
          M[0][0] = tor_flux0; M[0][1] = tor_flux1;
          M[1][0] = pol_flux0; M[1][1] = pol_flux1;

          std::complex<Real> M_inv[2][2];
          std::complex<Real> det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
          M_inv[0][0] = M[1][1]/det; M_inv[0][1] =-M[0][1]/det;
          M_inv[1][0] =-M[1][0]/det; M_inv[1][1] = M[0][0]/det;

          // Vout <-- Vin[0] M[0][l] + Vin[1] M[1][l], products of complex numbers as complex_vec_prod, on the grids SurfDim,
          // SurfDsp, in the storage of Vout when it has the size
          auto ComplexMatVec2x2 = [](sctl::Vector<Real>& Vout, const sctl::Vector<sctl::Vector<Real>>& Vin, const std::complex<Real> (&M)[2][2], const sctl::Integer l, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp) {
            SCTL_ASSERT(Vin.Dim() == 2);
            const sctl::Long Nsurf = SurfDim.Dim();
            const sctl::Long N = SurfDsp[Nsurf-1]+SurfDim[Nsurf-1];
            const sctl::Long dof = Vin[0].Dim() / (N * 2);
            SCTL_ASSERT(Vin[0].Dim() == N * dof * 2 && Vin[1].Dim() == N * dof * 2);
            const Real a0 = M[0][l].real(), a1 = M[0][l].imag(), b0 = M[1][l].real(), b1 = M[1][l].imag();
            if (Vout.Dim() != Vin[0].Dim()) Vout.ReInit(Vin[0].Dim());
            for (sctl::Long i = 0; i < Nsurf; i++) {
              for (sctl::Long k = 0; k < dof; k++) {
                sctl::Long offset = SurfDsp[i]*dof*2 + k*2*SurfDim[i];
                for (sctl::Long j = 0; j < SurfDim[i]; j++) {
                  const Real x0 = Vin[0][offset + 0*SurfDim[i] + j], x1 = Vin[0][offset + 1*SurfDim[i] + j];
                  const Real y0 = Vin[1][offset + 0*SurfDim[i] + j], y1 = Vin[1][offset + 1*SurfDim[i] + j];
                  Vout[offset + 0*SurfDim[i] + j] = (x0 * a0 - x1 * a1) + (y0 * b0 - y1 * b1);
                  Vout[offset + 1*SurfDim[i] + j] = (x0 * a1 + x1 * a0) + (y0 * b1 + y1 * b0);
                }
              }
            }
          };

          auto& B2 = B2_; // on one field period
          B2.ReInit(2);
          for (sctl::Integer l = 0; l < 2; l++) {
            ComplexMatVec2x2(B2[l], B, M_inv, l, SurfDim, SurfDsp);
            ComplexMatVec2x2(m_out[l], m, M_inv, l, SurfDimF, SurfDspF);
            sctl::ScratchBuf<Real> sigma_l_buf(sigma[0].Dim());
            sctl::Vector<Real> sigma_l(sigma_l_buf);
            ComplexMatVec2x2(sigma_l, sigma, M_inv, l, SurfDim, SurfDsp);
            complete_sigma(sigma_out[l], sigma_l);
            real_and_complete(B_out[l], B2[l]);
          }
          print_imag_norm(B2[0]);
          print_imag_norm(B2[1]);
        } else {
          SCTL_ASSERT(false);
        }
      }
      sctl::Profile::Toc();
    }

}
