// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Tests of SurfaceOp: derivatives, normals, quadrature weights, surface operators and resampling.

#include "test-utils.hpp"

using namespace biest_test;

int main(int argc, char** argv) {
  const sctl::Comm comm = sctl::Comm::Self();
  const Real pi = sctl::const_pi<Real>();
  Results res;

  { // Grad2D returns the negated derivatives with respect to the grid parameters t, p in [0,1)
    const sctl::Long Nt = 32, Np = 16, N = Nt * Np;
    sctl::Vector<Real> X(N), dX, dX_ref(2 * N);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        const Real s = t / (Real)Nt, q = p / (Real)Np;
        X[t * Np + p] = sctl::sin<Real>(2 * pi * s) + sctl::sin<Real>(4 * pi * q);
        dX_ref[0 * N + t * Np + p] = -2 * pi * sctl::cos<Real>(2 * pi * s);
        dX_ref[1 * N + t * Np + p] = -4 * pi * sctl::cos<Real>(4 * pi * q);
      }
    }
    biest::SurfaceOp<Real>(comm, Nt, Np).Grad2D(dX, X);
    res.Check("Grad2D: negated parameter derivatives", MaxRelDiff(dX, dX_ref), 1e-12);
  }

  { // Area and enclosed volume of a circular torus (R0 = 2, a = 1); the volume is positive for outward normals
    const sctl::Long Nt = 64, Np = 32, N = Nt * Np;
    const biest::Surface<Real> S(Nt, Np, biest::SurfType::AxisymCircleWide);
    biest::SurfaceOp<Real> op(comm, Nt, Np);
    sctl::Vector<Real> dX, normal, area_elem;
    op.Grad2D(dX, S.Coord());
    const Real orient = op.SurfNormalAreaElem(&normal, &area_elem, dX, &S.Coord());
    Real area = 0, volume = 0;
    for (sctl::Long i = 0; i < N; i++) {
      area += area_elem[i];
      for (sctl::Integer k = 0; k < COORD_DIM; k++) volume += S.Coord()[k * N + i] * normal[k * N + i] * area_elem[i] / 3;
    }
    res.Check("SurfNormalAreaElem: area of a torus", sctl::fabs<Real>(area / (8 * pi * pi) - 1), 1e-12);
    res.Check("SurfNormalAreaElem: volume of a torus (outward normals)", sctl::fabs<Real>(volume / (4 * pi * pi) - 1), 1e-12);
    res.Check("SurfNormalAreaElem: returned orientation is +1 or -1", sctl::fabs<Real>(sctl::fabs<Real>(orient) - 1), 0);
  }

  { // Operators on a non-axisymmetric surface
    const sctl::Long Nt = 240, Np = 80, N = Nt * Np;
    const biest::Surface<Real> S(Nt, Np, biest::SurfType::RotatingEllipseWide);
    biest::SurfaceOp<Real> op(comm, Nt, Np);
    sctl::Vector<Real> dX, normal;
    op.Grad2D(dX, S.Coord());
    op.SurfNormalAreaElem(&normal, nullptr, dX, &S.Coord());

    sctl::Vector<Real> U(N), V(COORD_DIM * N);
    for (sctl::Long i = 0; i < N; i++) {
      const Real x = S.Coord()[0 * N + i], y = S.Coord()[1 * N + i], z = S.Coord()[2 * N + i];
      U[i] = x * z + sctl::sin<Real>(y);
      V[0 * N + i] = z;
      V[1 * N + i] = x * y;
      V[2 * N + i] = x;
    }

    { // InvSurfLap(SurfLap(U)) is the mean-zero part of U
      sctl::Vector<Real> LapU, InvLapLapU, U0;
      op.SurfLap(LapU, dX, U);
      op.InvSurfLap(InvLapLapU, dX, LapU, 1e-12, 200);
      op.ProjZeroMean(U0, dX, U);
      res.Check("InvSurfLap(SurfLap(U)) = U - mean(U)", MaxRelDiff(InvLapLapU, U0), 1e-6);
    }

    { // SurfCurl of two stacked vector fields equals SurfCurl of each field
      sctl::Vector<Real> V2(2 * COORD_DIM * N), CurlV, CurlV2;
      for (sctl::Long i = 0; i < COORD_DIM * N; i++) {
        V2[i] = V[i];
        V2[COORD_DIM * N + i] = 2 * V[i];
      }
      op.SurfCurl(CurlV, dX, normal, V);
      op.SurfCurl(CurlV2, dX, normal, V2);
      sctl::Vector<Real> CurlV_ref(2 * N);
      for (sctl::Long i = 0; i < N; i++) {
        CurlV_ref[i] = CurlV[i];
        CurlV_ref[N + i] = 2 * CurlV[i];
      }
      res.Check("SurfCurl: two stacked fields", MaxRelDiff(CurlV2, CurlV_ref), 1e-14);
    }

    { // HodgeDecomp: the parts add up to V, SurfCurl(Vd) = 0 and SurfDiv(Vc) = 0
      sctl::Vector<Real> Vn, Vd, Vc, Vh;
      op.HodgeDecomp(Vn, Vd, Vc, Vh, V, dX, normal, 1e-12, 200);
      sctl::Vector<Real> CurlV, CurlVd, DivVt, DivVc;
      op.SurfCurl(CurlV, dX, normal, V);
      op.SurfCurl(CurlVd, dX, normal, Vd);
      op.SurfDiv(DivVt, dX, V - Vn);
      op.SurfDiv(DivVc, dX, Vc);
      res.Check("HodgeDecomp: Vn + Vd + Vc + Vh = V", MaxRelDiff(Vn + Vd + Vc + Vh, V), 1e-12);
      res.Check("HodgeDecomp: SurfCurl(Vd) = 0", MaxAbs(CurlVd) / MaxAbs(CurlV), 1e-6);
      res.Check("HodgeDecomp: SurfDiv(Vc) = 0", MaxAbs(DivVc) / MaxAbs(DivVt), 1e-6);
    }

    { // Upsample to a finer grid and Resample back
      sctl::Vector<Real> X1, X0;
      biest::SurfaceOp<Real>::Upsample(S.Coord(), Nt, Np, X1, 2 * Nt, 2 * Np);
      biest::SurfaceOp<Real>::Resample(X0, Nt, Np, X1, 2 * Nt, 2 * Np);
      res.Check("Upsample, then Resample back", MaxRelDiff(X0, S.Coord()), 1e-13);
    }

    { // RotateToroidal by one grid spacing shifts the data by one grid point
      sctl::Vector<Real> X_, X_ref(COORD_DIM * N);
      biest::SurfaceOp<Real>::RotateToroidal(X_, S.Coord(), Nt, Np, 2 * pi / Nt);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        for (sctl::Long t = 0; t < Nt; t++) {
          for (sctl::Long p = 0; p < Np; p++) X_ref[(k * Nt + t) * Np + p] = S.Coord()[(k * Nt + (t + 1) % Nt) * Np + p];
        }
      }
      res.Check("RotateToroidal by one grid spacing", MaxRelDiff(X_, X_ref), 1e-13);
    }
  }

  { // Field periods: W7-X has 5, found with the toroidal index in either direction; a surface built from one period
    const sctl::Long Nt = 70, Np = 14, N = Nt * Np, M = N / 5;
    const biest::Surface<Real> S(Nt, Np, biest::SurfType::W7X_);
    biest::Surface<Real> S_rev(Nt, Np, biest::SurfType::None); // the toroidal index reversed
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        for (sctl::Integer k = 0; k < COORD_DIM; k++) S_rev.Coord(t, p, k) = S.Coord((Nt - t) % Nt, p, k);
      }
    }
    const bool found = S.HasSymmetry(5) && S_rev.HasSymmetry(5) && S.HasSymmetry(1);
    const bool not_found = S.HasSymmetry(7) || S.HasSymmetry(3) || S.HasSymmetry(10);
    res.Check("Surface::HasSymmetry: W7-X has 5 field periods (either direction), not 7, 3 or 10", (found && !not_found ? 0 : 1), 0);

    sctl::Vector<Real> X(COORD_DIM * M); // the first field period
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < M; i++) X[k * M + i] = S.Coord()[k * N + i];
    }
    const biest::Surface<Real> S5(5, Nt / 5, Np, X);
    res.Check("Surface from one field period: the coordinates of the torus", MaxRelDiff(S5.Coord(), S.Coord()), 1e-14);
    biest::Surface<Real> S_ = S;
    S_.SetNFP(5);
    res.Check("Surface::NFP: from the constructor and SetNFP", (S5.NFP() == 5 && S_.NFP() == 5 && S.NFP() == 1 ? 0 : 1), 0);
  }

  { // SurfaceOp on one of 5 field periods of W7-X, with the toroidal index in either direction: each result is the
    // restriction to the period of the result on the full torus
    const sctl::Integer NFP = 5;
    const sctl::Long Nt = 210, Np = 42, N = Nt * Np, Ntp = Nt / NFP, M = Ntp * Np;
    const biest::Surface<Real> S_fwd(Nt, Np, biest::SurfType::W7X_);
    biest::Surface<Real> S_rev(Nt, Np, biest::SurfType::None);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        for (sctl::Integer k = 0; k < COORD_DIM; k++) S_rev.Coord(t, p, k) = S_fwd.Coord((Nt - t) % Nt, p, k);
      }
    }
    const auto period = [N,M](const sctl::Vector<Real>& X) { // the first field period of each component of X
      const sctl::Long dof = X.Dim() / N;
      sctl::Vector<Real> Y(dof * M);
      for (sctl::Long k = 0; k < dof; k++) {
        for (sctl::Long i = 0; i < M; i++) Y[k * M + i] = X[k * N + i];
      }
      return Y;
    };
    for (const biest::Surface<Real>* S_ : {&S_fwd, (const biest::Surface<Real>*)&S_rev}) {
      const auto& S = *S_;
      const sctl::Integer orient = biest::SurfaceOp<Real>::ToroidalOrientation(S.Coord(), Nt, Np);
      const std::string dir = (S_ == &S_fwd ? "" : ", toroidal index reversed");
      biest::SurfaceOp<Real> op(comm, Nt, Np), op5(comm, Ntp, Np, NFP, orient);
      sctl::Vector<Real> dX, normal, area_elem;
      op.Grad2D(dX, S.Coord());
      const Real orient_n = op.SurfNormalAreaElem(&normal, &area_elem, dX, &S.Coord());
      const sctl::Vector<Real> X5 = period(S.Coord()), dX5 = period(dX), normal5_ref = period(normal);

      // f, a function that repeats from one period to the next, and V = a e_R + b e_phi + c e_z, a vector field that
      // rotates with the periods (a, b, c of R, z and cos(NFP phi))
      sctl::Vector<Real> f(N), V(COORD_DIM * N), Vt(COORD_DIM * N);
      for (sctl::Long i = 0; i < N; i++) {
        const Real x = S.Coord()[0 * N + i], y = S.Coord()[1 * N + i], z = S.Coord()[2 * N + i];
        const Real R = sctl::sqrt<Real>(x * x + y * y), c5 = sctl::cos<Real>(NFP * std::atan2(y, x));
        f[i] = R * z + R * c5 + z * z;
        const Real a = R * z, b = c5 + z, c = R;
        V[0 * N + i] = (a * x - b * y) / R;
        V[1 * N + i] = (a * y + b * x) / R;
        V[2 * N + i] = c;
        for (sctl::Integer k = 0; k < COORD_DIM; k++) Vt[k * N + i] = dX[(k * 2 + 0) * N + i]; // a tangential field
      }
      const sctl::Vector<Real> f5 = period(f), V5 = period(V), Vt5 = period(Vt);

      sctl::Vector<Real> df, df5, normal5, area_elem5;
      op.Grad2D(df, f);
      op5.Grad2D(df5, f5);
      const Real orient5 = op5.SurfNormalAreaElem(&normal5, &area_elem5, dX5, &X5);
      res.Check(("SurfaceOp on one field period: Grad2D" + dir).c_str(), MaxRelDiff(df5, period(df)), 1e-12);
      res.Check(("SurfaceOp on one field period: normals and quadrature weights" + dir).c_str(), std::max(MaxRelDiff(normal5, normal5_ref), MaxRelDiff(area_elem5, period(area_elem))) + sctl::fabs<Real>(orient5 - orient_n), 1e-14);

      sctl::Vector<Real> Lf, Lf5, DivV, DivV5, CurlV, CurlV5, Gf, Gf5;
      op.SurfLap(Lf, dX, f);
      op5.SurfLap(Lf5, dX5, f5);
      op.SurfDiv(DivV, dX, V);
      op5.SurfDiv(DivV5, dX5, V5);
      op.SurfCurl(CurlV, dX, normal, V);
      op5.SurfCurl(CurlV5, dX5, normal5, V5);
      op.SurfGrad(Gf, dX, f);
      op5.SurfGrad(Gf5, dX5, f5);
      const Real err_ops = std::max(std::max(MaxRelDiff(Lf5, period(Lf)), MaxRelDiff(DivV5, period(DivV))), std::max(MaxRelDiff(CurlV5, period(CurlV)), MaxRelDiff(Gf5, period(Gf))));
      res.Check(("SurfaceOp on one field period: SurfGrad, SurfLap, SurfDiv, SurfCurl" + dir).c_str(), err_ops, 1e-11);

      sctl::Vector<Real> u, u5, Gu, Gu5;
      op.InvSurfLap(u, dX, f, 1e-12, 200, 1.5);
      op5.InvSurfLap(u5, dX5, f5, 1e-12, 200, 1.5);
      op.GradInvSurfLap(Gu, dX, f, 1e-12, 200, 1.5);
      op5.GradInvSurfLap(Gu5, dX5, f5, 1e-12, 200, 1.5);
      res.Check(("SurfaceOp on one field period: InvSurfLap, GradInvSurfLap" + dir).c_str(), std::max(MaxRelDiff(u5, period(u)), MaxRelDiff(Gu5, period(Gu))), 1e-11);

      sctl::Vector<Real> Vn, Vd, Vc, Vh, Vn5, Vd5, Vc5, Vh5;
      op.HodgeDecomp(Vn, Vd, Vc, Vh, Vt, dX, normal, 1e-12, 200);
      op5.HodgeDecomp(Vn5, Vd5, Vc5, Vh5, Vt5, dX5, normal5, 1e-12, 200);
      res.Check(("SurfaceOp on one field period: HodgeDecomp" + dir).c_str(), std::max(MaxRelDiff(Vh5, period(Vh)), std::max(MaxRelDiff(Vd5, period(Vd)), MaxRelDiff(Vc5, period(Vc)))), 1e-9);
    }
  }

  return res.ExitCode();
}
