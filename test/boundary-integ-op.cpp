// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Tests of FieldPeriodBIOp with Gauss's identity: the Laplace double-layer potential of
// the density 1 is -1 inside the surface, -1/2 on it (principal value) and 0 outside.
// Also a test of OffSurfaceOp with the single-layer potential, on a surface that is hard for its panels.

#include "test-utils.hpp"

using namespace biest_test;

int main(int argc, char** argv) {
  const sctl::Comm comm = sctl::Comm::Self();
  const Real pi = sctl::const_pi<Real>();
  const auto& ker = biest::Laplace3D<Real>::DxU();
  const biest::SingularQuadParams quad{6, 10, 20}; // the patch radius, and the radial and angular nodes
  Results res;

  // The surfaces below declare no field periods, unless stated: the densities on the full tori, on the surface grids,
  // and the potential at all the grid points

  { // FieldPeriodBIOp on one surface
    const sctl::Long Nt = 60, Np = 30, N = Nt * Np;
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::RotatingEllipseWide);
    biest::FieldPeriodBIOp<Real> op(comm);
    op.SetupSingular(Svec, ker, (Real)1e-10, quad);

    sctl::Vector<Real> F(N), U;
    F = 1;
    op.Eval(U, F, 1, Nt, Np);
    res.Check("FieldPeriodBIOp: D[1] = -1/2 on the surface", MaxAbs(U + 0.5), 1e-4);

    sctl::Vector<Real> U2 = U;
    op.Eval(U2, F, 1, Nt, Np);
    res.Check("FieldPeriodBIOp: Eval() adds into U of the right size", MaxRelDiff(U2, U * 2), 1e-15);

    sctl::Vector<Real> F3, U3, U3ref; // two densities, F and 2 F, one after the other
    for (const Real a : {1, 2}) for (const auto& f : F) F3.PushBack(a * f);
    for (const Real a : {1, 2}) for (const auto& u : U) U3ref.PushBack(a * u);
    op.Eval(U3, F3, 1, Nt, Np);
    res.Check("FieldPeriodBIOp: Eval() of two densities one after the other", MaxRelDiff(U3, U3ref), 1e-10);

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
      op.EvalOffSurface(Uin, Xin, F, 1, Nt, Np, ker, 1e-10);
      op.EvalOffSurface(Uout, Xout, F, 1, Nt, Np, ker, 1e-10);
      res.Check("FieldPeriodBIOp::EvalOffSurface: D[1] = -1 inside, near the surface", MaxAbs(Uin + 1), 1e-8);
      res.Check("FieldPeriodBIOp::EvalOffSurface: D[1] = 0 outside, near the surface", MaxAbs(Uout), 1e-8);
    }
  }

  { // FieldPeriodBIOp on two surfaces whose grids have opposite orientations, in both orders
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
      biest::FieldPeriodBIOp<Real> op(comm);
      op.SetupSingular(Svec, ker, (Real)1e-10, quad);
      sctl::Vector<sctl::Integer> NFP2(2);
      sctl::Vector<sctl::Long> Nt2(2), Np2(2);
      NFP2 = 1;
      Nt2 = Nt;
      Np2 = Np;
      sctl::Vector<Real> F(2 * N), U;
      F = 1;
      op.Eval(U, F, NFP2, Nt2, Np2);
      err = std::max<Real>(err, MaxAbs(U + 0.5));
    }
    res.Check("FieldPeriodBIOp: two surfaces with opposite orientations", err, 1e-4);
  }

  { // FieldPeriodBIOp on two nested tori 0.1 apart, closer than the trapezoidal rule is accurate for (about 0.75
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
    sctl::Vector<sctl::Integer> SurfNFP(2); // the densities on the full tori, on the surface grids
    sctl::Vector<sctl::Long> SurfNt(2), SurfNp(2);
    SurfNFP = 1;
    for (sctl::Long s = 0; s < 2; s++) {
      SurfNt[s] = Svec[s].NTor();
      SurfNp[s] = Svec[s].NPol();
    }
    const auto check = [&comm,&Svec,&SurfNFP,&SurfNt,&SurfNp,&quad,&density,&res](const auto& ker, const char* name) {
      constexpr sctl::Integer KDIM1 = std::decay_t<decltype(ker)>::Dim(1);
      biest::FieldPeriodBIOp<Real> op(comm);
      op.SetupSingular(Svec, ker, (Real)1e-10, quad);
      sctl::Vector<Real> F, U;
      for (const auto& S : Svec) for (const auto& f : density(S)) F.PushBack(f);
      op.Eval(U, F, SurfNFP, SurfNt, SurfNp);

      sctl::Vector<Real> Uref; // for each surface t, its own potential and that of the other one, s
      for (sctl::Long t = 0; t < 2; t++) {
        const sctl::Long s = 1 - t;
        sctl::Vector<biest::Surface<Real>> St(1), Ss(1);
        St[0] = Svec[t];
        Ss[0] = Svec[s];
        biest::FieldPeriodBIOp<Real> op_t(comm), op_s(comm);
        op_t.SetupSingular(St, ker, (Real)1e-10, quad);
        op_s.SetupSingular(Ss, ker, (Real)1e-10, quad);
        sctl::Vector<Real> U_t;
        op_t.Eval(U_t, density(Svec[t]), 1, SurfNt[t], SurfNp[t]);
        op_s.EvalOffSurface(U_t, Svec[t].Coord(), density(Svec[s]), 1, SurfNt[s], SurfNp[s], ker, 1e-12);
        for (const auto& u : U_t) Uref.PushBack(u);
      }
      res.Check((std::string("FieldPeriodBIOp: two tori 0.1 apart, ") + name + " = own + EvalOffSurface").c_str(), MaxRelDiff(U, Uref), 1e-9);

      sctl::Vector<Real> F2, U2, U2ref; // two densities, F and 2 F, stacked surface by surface
      sctl::Long offset = 0;
      for (sctl::Long s = 0; s < 2; s++) {
        const sctl::Long N = Svec[s].NTor() * Svec[s].NPol();
        for (const Real a : {1, 2}) for (sctl::Long i = 0; i < N; i++) F2.PushBack(a * F[offset + i]);
        offset += N;
      }
      op.Eval(U2, F2, SurfNFP, SurfNt, SurfNp);
      offset = 0;
      for (sctl::Long t = 0; t < 2; t++) {
        const sctl::Long M = KDIM1 * Svec[t].NTor() * Svec[t].NPol();
        for (const Real a : {1, 2}) for (sctl::Long i = 0; i < M; i++) U2ref.PushBack(a * U[offset + i]);
        offset += M;
      }
      res.Check((std::string("FieldPeriodBIOp: two tori 0.1 apart, ") + name + " of two stacked densities").c_str(), MaxRelDiff(U2, U2ref), 1e-10);
    };
    check(biest::Laplace3D<Real>::FxU(), "S[F]");
    check(biest::Laplace3D<Real>::FxdU(), "grad S[F]");
  }

  { // HedgehogOrder > 1 gives the one-sided limit from the interior
    const sctl::Long Nt = 120, Np = 60, N = Nt * Np;
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::AxisymCircleWide);
    biest::FieldPeriodBIOp<Real,4> op(comm);
    op.SetupSingular(Svec, ker, (Real)1e-10, {12, 60, 40});
    sctl::Vector<Real> F(N), U;
    F = 1;
    op.Eval(U, F, 1, Nt, Np);
    res.Check("FieldPeriodBIOp<HedgehogOrder=4>: D[1] = -1 (interior limit)", MaxAbs(U + 1), 1e-3);
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
    Svec[0] = biest::Surface<Real>(NFP, Nt, Np, X);
    { // scalar density
      biest::FieldPeriodBIOp<Real> biop(comm);
      biop.SetupSingular(Svec[0], ker, digits, NFP, Nt, Np);

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
      biest::FieldPeriodBIOp<Real> biop(comm);
      biop.SetupSingular(Svec[0], biest::BiotSavart3D<Real>::FxU(), digits, NFP, Nt, Np);
      sctl::Vector<Real> U, U_full;
      biop.Eval(U, density(X), NFP, Nt, Np);
      biop.Eval(U_full, density(Svec[0].Coord()), 1, NFP * Nt, Np);
      res.Check((name + "vector density given on one period and on the full torus").c_str(), MaxRelDiff(U, U_full), 1e-12);
    }
    { // densities given on half a field period, the other half by stellarator symmetry, and on one field period: on a
      // surface whose grid points are half a grid spacing from the planes of symmetry
      const sctl::Long Nh = Nt / 2, N = Nt * Np; // grid in half a field period, grid points in one field period
      sctl::Vector<Real> Xh(COORD_DIM * N), f_even(N), f_odd(N), v_odd(COORD_DIM * N);
      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          const sctl::Long i = t * Np + p;
          const Real theta = orient * (t + 0.5) * pi / (NFP * Nh);
          const Real phi = 2 * pi * p / Np;
          const Real R = 1.0 + 0.3 * sctl::cos<Real>(phi) + 0.05 * sctl::cos<Real>(phi - NFP * theta);
          const Real Z = 0.3 * sctl::sin<Real>(phi) - 0.05 * sctl::sin<Real>(phi - NFP * theta);
          Xh[0 * N + i] = R * sctl::cos<Real>(theta);
          Xh[1 * N + i] = R * sctl::sin<Real>(theta);
          Xh[2 * N + i] = Z;
          f_even[i] = sctl::cos<Real>(phi - NFP * theta) + 0.3 * sctl::cos<Real>(phi);
          f_odd[i] = sctl::sin<Real>(phi - NFP * theta) + 0.3 * sctl::sin<Real>(phi);
          v_odd[0 * N + i] = -Xh[1 * N + i];
          v_odd[1 * N + i] = Xh[0 * N + i];
          v_odd[2 * N + i] = 1;
        }
      }
      const biest::Surface<Real> Sh(NFP, Nt, Np, Xh);
      biest::FieldPeriodBIOp<Real> dl(comm), bs(comm);
      dl.SetupSingular(Sh, ker, digits, NFP, Nt, Np);
      bs.SetupSingular(Sh, biest::BiotSavart3D<Real>::FxU(), digits, NFP, Nt, Np);
      const auto check = [&res,&name,NFP,Nt,Nh,Np](const biest::FieldPeriodBIOp<Real>& biop, const sctl::Vector<Real>& F, const biest::Parity parity, const std::string& density) {
        const sctl::Long dof = F.Dim() / (Nt * Np);
        sctl::Vector<Real> Fh(dof * Nh * Np), U, Uh; // Fh: F on the first half of the field period
        for (sctl::Long k = 0; k < dof; k++) {
          for (sctl::Long i = 0; i < Nh * Np; i++) Fh[k * Nh * Np + i] = F[k * Nt * Np + i];
        }
        biop.Eval(U, F, NFP, Nt, Np);
        biop.Eval(Uh, Fh, NFP, Nh, Np, true, parity);
        res.Check((name + density + " given on half a field period and on one").c_str(), MaxRelDiff(Uh, U), 1e-12);
      };
      check(dl, f_even, biest::Parity::Even, "even scalar density");
      check(dl, f_odd, biest::Parity::Odd, "odd scalar density");
      check(bs, Xh, biest::Parity::Even, "even vector density (x,y,z)");
      check(bs, v_odd, biest::Parity::Odd, "odd vector density (-y,x,1)");
    }
  }

  { // FieldPeriodBIOp on two nested tori with 3 field periods 0.1 apart: on one field period, the potential on the full
    // tori of the same tori without their field periods, with the density given on the full tori or on one field
    // period; and the double-layer identity with the quadrature for a tolerance
    const sctl::Integer NFP = 3;
    const auto torus = [pi](const sctl::Long Nt, const sctl::Long Np, const Real r) {
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

    { // the quadrature given, without upsampling (the densities on the surface grids), against the full tori
      const auto& ker = biest::Laplace3D<Real>::FxdU();
      const biest::SingularQuadParams quad1{quad.patch_rad, quad.rad_nodes, quad.ang_nodes, 1, 1};
      sctl::Vector<Real> F, F1; // 1 + z + x^2 + y^2 (invariant under the rotations) on the full tori, and on one period
      for (const auto& S : Svec) {
        const sctl::Long N = S.NTor() * S.NPol();
        for (sctl::Long i = 0; i < N; i++) {
          const Real x = S.Coord()[0 * N + i], y = S.Coord()[1 * N + i], z = S.Coord()[2 * N + i];
          F.PushBack(1 + z + x * x + y * y);
          if (i < N / NFP) F1.PushBack(1 + z + x * x + y * y);
        }
      }
      sctl::Vector<biest::Surface<Real>> Svec1 = Svec; // the tori without their field periods
      sctl::Vector<sctl::Integer> SurfNFP(2); // the densities on the full tori, on the surface grids
      sctl::Vector<sctl::Long> SurfNt(2), SurfNp(2);
      SurfNFP = 1;
      for (sctl::Long s = 0; s < 2; s++) {
        Svec1[s].SetNFP(1);
        SurfNt[s] = Svec[s].NTor();
        SurfNp[s] = Svec[s].NPol();
      }
      biest::FieldPeriodBIOp<Real> op(comm);
      op.SetupSingular(Svec1, ker, (Real)1e-10, quad1);
      sctl::Vector<Real> U, Uref; // Uref: U on one field period
      op.Eval(U, F, SurfNFP, SurfNt, SurfNp);
      sctl::Long offset = 0;
      for (const auto& S : Svec) {
        const sctl::Long N = S.NTor() * S.NPol();
        for (sctl::Integer k = 0; k < 3; k++) {
          for (sctl::Long i = 0; i < N / NFP; i++) Uref.PushBack(U[offset + k * N + i]);
        }
        offset += 3 * N;
      }

      biest::FieldPeriodBIOp<Real> fp_op(comm);
      fp_op.SetupSingular(Svec, ker, 1e-10, quad1);
      sctl::Vector<Real> Ufp, Ufp1;
      fp_op.Eval(Ufp, F, SurfNFP, SurfNt, SurfNp);
      sctl::Vector<sctl::Integer> NFP1(2);
      sctl::Vector<sctl::Long> Nt1(2), Np1(2);
      NFP1 = NFP;
      for (sctl::Long s = 0; s < 2; s++) {
        Nt1[s] = Svec[s].NTor() / NFP;
        Np1[s] = Svec[s].NPol();
      }
      fp_op.Eval(Ufp1, F1, NFP1, Nt1, Np1);
      res.Check("FieldPeriodBIOp, two tori with 3 field periods: = on the full tori, on one period", MaxRelDiff(Ufp, Uref), 1e-13);
      res.Check("FieldPeriodBIOp, two tori with 3 field periods: the density on one period", MaxRelDiff(Ufp1, Uref), 1e-13);

      sctl::Vector<Real> Fmix, Ufp_mix; // the density on one period of the first torus and on the full second torus
      const sctl::Long N0 = Svec[0].NTor() * Svec[0].NPol();
      for (sctl::Long i = 0; i < N0 / NFP; i++) Fmix.PushBack(F1[i]);
      for (sctl::Long i = N0; i < F.Dim(); i++) Fmix.PushBack(F[i]);
      sctl::Vector<sctl::Integer> NFPmix(2);
      sctl::Vector<sctl::Long> Ntmix(2);
      NFPmix[0] = NFP;
      NFPmix[1] = 1;
      Ntmix[0] = Nt1[0];
      Ntmix[1] = SurfNt[1];
      fp_op.Eval(Ufp_mix, Fmix, NFPmix, Ntmix, SurfNp);
      res.Check("FieldPeriodBIOp, two tori with 3 field periods: the density on one period of one torus, the full other", MaxRelDiff(Ufp_mix, Uref), 1e-13);
    }
    { // the quadrature for the tolerance 1e-8: D[1] of both = -1/2 + 0 on the outer torus and -1/2 - 1 on the inner one
      sctl::Vector<sctl::Integer> trg_NFP(2);
      sctl::Vector<sctl::Long> trg_Nt(2), trg_Np(2); // one field period of the grids
      trg_NFP = NFP;
      for (sctl::Long s = 0; s < 2; s++) {
        trg_Nt[s] = Svec[s].NTor() / NFP;
        trg_Np[s] = Svec[s].NPol();
      }
      biest::FieldPeriodBIOp<Real> fp_op(comm);
      fp_op.SetupSingular(Svec, ker, (Real)1e-8);
      sctl::Vector<Real> F(trg_Nt[0] * trg_Np[0] + trg_Nt[1] * trg_Np[1]), U;
      F = 1;
      fp_op.Eval(U, F, trg_NFP, trg_Nt, trg_Np);
      Real err = 0;
      const sctl::Long M0 = trg_Nt[0] * trg_Np[0];
      for (sctl::Long i = 0; i < U.Dim(); i++) err = std::max<Real>(err, sctl::fabs<Real>(U[i] - (i < M0 ? -0.5 : -1.5)));
      res.Check("FieldPeriodBIOp for the tolerance 1e-8, two tori with 3 field periods: D[1] = -1/2, -3/2", err, 1e-7);
    }
  }

  return res.ExitCode();
}
