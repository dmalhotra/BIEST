// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Tests of BoundaryIntegralOp and FieldPeriodBIOp with Gauss's identity: the Laplace double-layer potential of
// the density 1 is -1 inside the surface, -1/2 on it (principal value) and 0 outside.
// Also a test of OffSurfaceOp with the single-layer potential, on a surface that is hard for its panels.

#include "test-utils.hpp"

using namespace biest_test;

int main(int argc, char** argv) {
  const sctl::Comm comm = sctl::Comm::Self();
  const Real pi = sctl::const_pi<Real>();
  const auto& ker = biest::Laplace3D<Real>::DxU();
  Results res;

  { // BoundaryIntegralOp on one surface
    const sctl::Long Nt = 60, Np = 30, N = Nt * Np;
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::RotatingEllipseWide);
    biest::BoundaryIntegralOp<Real,1,1,1,6,10> op(comm);
    op.SetupSingular(Svec, ker);

    sctl::Vector<Real> F(N), U;
    F = 1;
    op(U, F);
    res.Check("BoundaryIntegralOp: D[1] = -1/2 on the surface", MaxAbs(U + 0.5), 1e-4);

    sctl::Vector<Real> U2 = U;
    op(U2, F);
    res.Check("BoundaryIntegralOp: operator() adds into U of the right size", MaxRelDiff(U2, U * 2), 1e-15);

    { // EvalOffSurface at points close to the surface
      sctl::Vector<Real> dX, normal;
      biest::SurfaceOp<Real> surf_op(comm, Nt, Np);
      surf_op.Grad2D(dX, Svec[0].Coord());
      surf_op.SurfNormalAreaElem(&normal, nullptr, dX, &Svec[0].Coord());
      const Real h = 0.02; // about a tenth of the grid spacing
      sctl::Vector<Real> Xin(COORD_DIM * N), Xout(COORD_DIM * N), Uin, Uout;
      for (sctl::Long i = 0; i < COORD_DIM * N; i++) {
        Xin[i] = Svec[0].Coord()[i] - h * normal[i];
        Xout[i] = Svec[0].Coord()[i] + h * normal[i];
      }
      op.EvalOffSurface(Uin, Xin, F, ker, 1e-10);
      op.EvalOffSurface(Uout, Xout, F, ker, 1e-10);
      res.Check("BoundaryIntegralOp::EvalOffSurface: D[1] = -1 inside, near the surface", MaxAbs(Uin + 1), 1e-8);
      res.Check("BoundaryIntegralOp::EvalOffSurface: D[1] = 0 outside, near the surface", MaxAbs(Uout), 1e-8);
    }
  }

  { // BoundaryIntegralOp on two surfaces whose grids have opposite orientations, in both orders
    const sctl::Long Nt = 60, Np = 30, N = Nt * Np;
    const biest::Surface<Real> A(Nt, Np, biest::SurfType::AxisymCircleWide);
    biest::Surface<Real> B(Nt, Np, biest::SurfType::None);
    for (sctl::Long t = 0; t < Nt; t++) { // B: A shifted by 10 in x, with the poloidal direction reversed
      for (sctl::Long p = 0; p < Np; p++) {
        B.Coord(t, p, 0) = A.Coord(t, (Np - p) % Np, 0) + 10;
        B.Coord(t, p, 1) = A.Coord(t, (Np - p) % Np, 1);
        B.Coord(t, p, 2) = A.Coord(t, (Np - p) % Np, 2);
      }
    }
    Real err = 0;
    for (const bool B_first : {false, true}) {
      sctl::Vector<biest::Surface<Real>> Svec(2);
      Svec[0] = (B_first ? B : A);
      Svec[1] = (B_first ? A : B);
      biest::BoundaryIntegralOp<Real,1,1,1,6,10> op(comm);
      op.SetupSingular(Svec, ker);
      sctl::Vector<Real> F(2 * N), U;
      F = 1;
      op(U, F);
      err = std::max<Real>(err, MaxAbs(U + 0.5));
    }
    res.Check("BoundaryIntegralOp: two surfaces with opposite orientations", err, 1e-4);
  }

  { // BoundaryIntegralOp on two nested tori 0.1 apart, closer than the trapezoidal rule is accurate for (about 0.75
    // toroidal grid spacings): the potential of each torus at the grid points of the other, against EvalOffSurface
    const auto torus = [pi](const sctl::Long Nt, const sctl::Long Np, const Real r) {
      biest::Surface<Real> S(Nt, Np, biest::SurfType::None);
      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          const Real phi = 2 * pi * t / Nt, theta = 2 * pi * p / Np;
          const Real R = 1 + r * sctl::cos<Real>(theta) + 0.02 * sctl::cos<Real>(theta - 3 * phi);
          S.Coord(t, p, 0) = R * sctl::cos<Real>(phi);
          S.Coord(t, p, 1) = R * sctl::sin<Real>(phi);
          S.Coord(t, p, 2) = r * sctl::sin<Real>(theta) - 0.02 * sctl::sin<Real>(theta - 3 * phi);
        }
      }
      return S;
    };
    const auto density = [](const biest::Surface<Real>& S) { // 1 + x y + z
      const sctl::Long N = S.NTor() * S.NPol();
      sctl::Vector<Real> F(N);
      for (sctl::Long i = 0; i < N; i++) F[i] = 1 + S.Coord()[0 * N + i] * S.Coord()[1 * N + i] + S.Coord()[2 * N + i];
      return F;
    };
    sctl::Vector<biest::Surface<Real>> Svec(2);
    Svec[0] = torus(60, 30, 0.3);
    Svec[1] = torus(48, 24, 0.2);
    const auto check = [&](const auto& ker, const char* name) {
      constexpr sctl::Integer KDIM1 = std::decay_t<decltype(ker)>::Dim(1);
      biest::BoundaryIntegralOp<Real,1,KDIM1,1,6,10> op(comm);
      op.SetupSingular(Svec, ker);
      sctl::Vector<Real> F, U;
      for (const auto& S : Svec) for (const auto& f : density(S)) F.PushBack(f);
      op(U, F);

      sctl::Vector<Real> Uref; // for each surface t, its own potential and that of the other one, s
      for (sctl::Long t = 0; t < 2; t++) {
        const sctl::Long s = 1 - t;
        sctl::Vector<biest::Surface<Real>> St(1), Ss(1);
        St[0] = Svec[t];
        Ss[0] = Svec[s];
        biest::BoundaryIntegralOp<Real,1,KDIM1,1,6,10> op_t(comm), op_s(comm);
        op_t.SetupSingular(St, ker);
        op_s.SetupSingular(Ss, ker);
        sctl::Vector<Real> U_t;
        op_t(U_t, density(Svec[t]));
        op_s.EvalOffSurface(U_t, Svec[t].Coord(), density(Svec[s]), ker, 1e-12);
        for (const auto& u : U_t) Uref.PushBack(u);
      }
      res.Check((std::string("BoundaryIntegralOp: two tori 0.1 apart, ") + name + " = own + EvalOffSurface").c_str(), MaxRelDiff(U, Uref), 1e-9);

      sctl::Vector<Real> F2, U2, U2ref; // two densities, F and 2 F, stacked surface by surface
      sctl::Long offset = 0;
      for (sctl::Long s = 0; s < 2; s++) {
        const sctl::Long N = Svec[s].NTor() * Svec[s].NPol();
        for (const Real a : {1, 2}) for (sctl::Long i = 0; i < N; i++) F2.PushBack(a * F[offset + i]);
        offset += N;
      }
      op(U2, F2);
      offset = 0;
      for (sctl::Long t = 0; t < 2; t++) {
        const sctl::Long M = KDIM1 * Svec[t].NTor() * Svec[t].NPol();
        for (const Real a : {1, 2}) for (sctl::Long i = 0; i < M; i++) U2ref.PushBack(a * U[offset + i]);
        offset += M;
      }
      res.Check((std::string("BoundaryIntegralOp: two tori 0.1 apart, ") + name + " of two stacked densities").c_str(), MaxRelDiff(U2, U2ref), 1e-10);
    };
    check(biest::Laplace3D<Real>::FxU(), "S[F]");
    check(biest::Laplace3D<Real>::FxdU(), "grad S[F]");
  }

  { // HedgehogOrder > 1 gives the one-sided limit from the interior
    const sctl::Long Nt = 120, Np = 60, N = Nt * Np;
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::AxisymCircleWide);
    biest::BoundaryIntegralOp<Real,1,1,1,12,20,4> op(comm);
    op.SetupSingular(Svec, ker);
    sctl::Vector<Real> F(N), U;
    F = 1;
    op(U, F);
    res.Check("BoundaryIntegralOp<HedgehogOrder=4>: D[1] = -1 (interior limit)", MaxAbs(U + 1), 1e-3);
  }

  { // OffSurfaceOp: the single-layer potential S[1] far from an axisymmetric torus whose cross-section
    // r = 0.25 (1 + 0.6 cos 6 phi) has six sharp concave folds, where the panels must be smaller for the area element
    // than for the surface coordinates; against the trapezoidal rule on the grid upsampled 32 times poloidally
    const sctl::Long Nt = 96, Np = 64, N = Nt * Np, M = 64;
    biest::Surface<Real> S(Nt, Np, biest::SurfType::None);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        const Real theta = 2 * pi * t / Nt, phi = 2 * pi * p / Np;
        const Real r = 0.25 * (1 + 0.6 * sctl::cos<Real>(6 * phi)), R = 1 + r * sctl::cos<Real>(phi);
        S.Coord(t, p, 0) = R * sctl::cos<Real>(theta);
        S.Coord(t, p, 1) = R * sctl::sin<Real>(theta);
        S.Coord(t, p, 2) = r * sctl::sin<Real>(phi);
      }
    }
    sctl::Vector<Real> Xt(COORD_DIM * M); // points on the sphere of radius 5
    for (sctl::Long i = 0; i < M; i++) {
      const Real z = -1 + (2 * i + 1) / (Real)M, a = 2.399963 * i, s = sctl::sqrt<Real>(1 - z * z);
      Xt[0 * M + i] = 5 * s * sctl::cos<Real>(a);
      Xt[1 * M + i] = 5 * s * sctl::sin<Real>(a);
      Xt[2 * M + i] = 5 * z;
    }
    const auto& sl = biest::Laplace3D<Real>::FxU();
    sctl::Vector<Real> F(N), U, Uref;
    F = 1;
    biest::OffSurfaceOp op(sl, comm);
    op.Setup(S.Coord(), Nt, Np, Xt, 1e-10, F);
    op.Eval(U, F);

    const sctl::Long Mp = 32 * Np;
    sctl::Vector<Real> Xu, Fu(Nt * Mp), dXu, normal, area;
    biest::SurfaceOp<Real>::Upsample(S.Coord(), Nt, Np, Xu, Nt, Mp);
    Fu = 1;
    biest::SurfaceOp<Real> surf_op(comm, Nt, Mp);
    surf_op.Grad2D(dXu, Xu);
    surf_op.SurfNormalAreaElem(&normal, &area, dXu, &Xu);
    surf_op.EvalSurfInteg(Uref, Xt, Xu, normal, area, Fu, sl);
    res.Check("OffSurfaceOp: S[1] off a surface with sharp concave folds", MaxRelDiff(U, Uref), 1e-9);
  }

  for (const sctl::Integer orient : {1, -1}) { // FieldPeriodBIOp on a surface with 5 field periods, with the toroidal index increasing counterclockwise (orient = 1) or clockwise
    const int digits = 8, NFP = 5;
    const sctl::Long Nt = 30, Np = 30; // grid in one field period
    sctl::Vector<Real> X(COORD_DIM * Nt * Np);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        const Real theta = orient * 2 * pi * t / (NFP * Nt);
        const Real phi = 2 * pi * p / Np;
        const Real R = 1.0 + 0.3 * sctl::cos<Real>(phi) + 0.05 * sctl::cos<Real>(phi - NFP * theta);
        const Real Z = 0.3 * sctl::sin<Real>(phi) - 0.05 * sctl::sin<Real>(phi - NFP * theta);
        X[(0 * Nt + t) * Np + p] = R * sctl::cos<Real>(theta);
        X[(1 * Nt + t) * Np + p] = R * sctl::sin<Real>(theta);
        X[(2 * Nt + t) * Np + p] = Z;
      }
    }
    const std::string name = (orient == 1 ? "FieldPeriodBIOp (counterclockwise grid): " : "FieldPeriodBIOp (clockwise grid): ");
    res.Check((name + "ToroidalOrientation").c_str(), (biest::SurfaceOp<Real>::ToroidalOrientation(X, Nt, Np) == orient ? 0 : 1), 0);

    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = biest::FieldPeriodBIOp<Real,COORD_DIM,1,1>::BuildSurface(X, NFP, Nt, Np);
    { // scalar density
      biest::FieldPeriodBIOp<Real,COORD_DIM,1,1> biop(comm);
      biop.SetupSingular(Svec, ker, digits, NFP, NFP * Nt, Np, Nt, Np);

      sctl::Vector<Real> F(Nt * Np), U;
      F = 1;
      biop.Eval(U, F, NFP, Nt, Np);
      res.Check((name + "D[1] = -1/2 on one field period").c_str(), MaxAbs(U + 0.5), 1e-7);

      sctl::Vector<Real> F_full(NFP * Nt * Np), U_full;
      F_full = 1;
      biop.Eval(U_full, F_full, 1, NFP * Nt, Np);
      res.Check((name + "scalar density given on the full torus").c_str(), MaxRelDiff(U_full, U), 1e-12);
    }
    { // vector density F(x) = (-y, x, z) given on one field period, and on the full torus
      const auto density = [](const sctl::Vector<Real>& X) {
        const sctl::Long N = X.Dim() / COORD_DIM;
        sctl::Vector<Real> F(COORD_DIM * N);
        for (sctl::Long i = 0; i < N; i++) {
          F[0 * N + i] = -X[1 * N + i];
          F[1 * N + i] = X[0 * N + i];
          F[2 * N + i] = X[2 * N + i];
        }
        return F;
      };
      biest::FieldPeriodBIOp<Real,COORD_DIM,3,3> biop(comm);
      biop.SetupSingular(Svec, biest::BiotSavart3D<Real>::FxU(), digits, NFP, NFP * Nt, Np, Nt, Np);
      sctl::Vector<Real> U, U_full;
      biop.Eval(U, density(X), NFP, Nt, Np);
      biop.Eval(U_full, density(Svec[0].Coord()), 1, NFP * Nt, Np);
      res.Check((name + "vector density given on one period and on the full torus").c_str(), MaxRelDiff(U, U_full), 1e-12);
    }
  }

  { // FieldPeriodBIOp on two nested tori with 3 field periods 0.1 apart: on one field period, the potential of
    // BoundaryIntegralOp on the full tori, with the density given on the full tori or on one field period; and the
    // double-layer identity with the quadrature from digits
    const sctl::Integer NFP = 3;
    const auto torus = [pi,NFP](const sctl::Long Nt, const sctl::Long Np, const Real r) {
      biest::Surface<Real> S(Nt, Np, biest::SurfType::None);
      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          const Real phi = 2 * pi * t / Nt, theta = 2 * pi * p / Np;
          const Real R = 1 + r * sctl::cos<Real>(theta) + 0.02 * sctl::cos<Real>(theta - NFP * phi);
          S.Coord(t, p, 0) = R * sctl::cos<Real>(phi);
          S.Coord(t, p, 1) = R * sctl::sin<Real>(phi);
          S.Coord(t, p, 2) = r * sctl::sin<Real>(theta) - 0.02 * sctl::sin<Real>(theta - NFP * phi);
        }
      }
      S.SetNFP(NFP);
      return S;
    };
    sctl::Vector<biest::Surface<Real>> Svec(2);
    Svec[0] = torus(60, 30, 0.3);
    Svec[1] = torus(48, 24, 0.2);

    { // the quadrature given, against BoundaryIntegralOp
      const auto& ker = biest::Laplace3D<Real>::FxdU();
      sctl::Vector<Real> F, F1; // 1 + z + x^2 + y^2 (invariant under the rotations) on the full tori, and on one period
      for (const auto& S : Svec) {
        const sctl::Long N = S.NTor() * S.NPol();
        for (sctl::Long i = 0; i < N; i++) {
          const Real x = S.Coord()[0 * N + i], y = S.Coord()[1 * N + i], z = S.Coord()[2 * N + i];
          F.PushBack(1 + z + x * x + y * y);
          if (i < N / NFP) F1.PushBack(1 + z + x * x + y * y);
        }
      }
      biest::BoundaryIntegralOp<Real,1,3,1,6,10> op(comm);
      op.SetupSingular(Svec, ker);
      sctl::Vector<Real> U, Uref; // Uref: U on one field period
      op(U, F);
      sctl::Long offset = 0;
      for (const auto& S : Svec) {
        const sctl::Long N = S.NTor() * S.NPol();
        for (sctl::Integer k = 0; k < 3; k++) {
          for (sctl::Long i = 0; i < N / NFP; i++) Uref.PushBack(U[offset + k * N + i]);
        }
        offset += 3 * N;
      }

      biest::FieldPeriodBIOp<Real,COORD_DIM,1,3> fp_op(comm);
      fp_op.SetupSingular<1,6,10>(Svec, ker, 1e-10);
      sctl::Vector<Real> Ufp, Ufp1;
      fp_op(Ufp, F);
      sctl::Vector<sctl::Long> Nt1(2), Np1(2);
      for (sctl::Long s = 0; s < 2; s++) {
        Nt1[s] = Svec[s].NTor() / NFP;
        Np1[s] = Svec[s].NPol();
      }
      fp_op.Eval(Ufp1, F1, NFP, Nt1, Np1);
      res.Check("FieldPeriodBIOp, two tori with 3 field periods: = BoundaryIntegralOp on one period", MaxRelDiff(Ufp, Uref), 1e-13);
      res.Check("FieldPeriodBIOp, two tori with 3 field periods: the density on one period", MaxRelDiff(Ufp1, Uref), 1e-13);
    }
    { // the quadrature from digits: D[1] of both = -1/2 + 0 on the outer torus and -1/2 - 1 on the inner one
      const sctl::Integer digits = 8;
      sctl::Vector<sctl::Long> trg_Nt(2), trg_Np(2);
      for (sctl::Long s = 0; s < 2; s++) {
        trg_Nt[s] = Svec[s].NTor() / NFP;
        trg_Np[s] = Svec[s].NPol();
      }
      biest::FieldPeriodBIOp<Real,COORD_DIM,1,1> fp_op(comm);
      fp_op.SetupSingular(Svec, ker, digits, trg_Nt, trg_Np);
      sctl::Vector<Real> F(fp_op.Dim(0)), U;
      F = 1;
      fp_op(U, F);
      Real err = 0;
      const sctl::Long M0 = trg_Nt[0] * trg_Np[0];
      for (sctl::Long i = 0; i < U.Dim(); i++) err = std::max<Real>(err, sctl::fabs<Real>(U[i] - (i < M0 ? -0.5 : -1.5)));
      res.Check("FieldPeriodBIOp from digits (8), two tori with 3 field periods: D[1] = -1/2, -3/2", err, 1e-7);
    }
  }

  return res.ExitCode();
}
