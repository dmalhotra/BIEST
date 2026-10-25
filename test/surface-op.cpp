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

  return res.ExitCode();
}
