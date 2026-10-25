// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Vacuum fields with a given normal component.

#include <biest.hpp>

using Real = double;
constexpr int COORD_DIM = 3;

// Coordinates (SoA order) on an Nt x Np grid in one field period, of a surface with minor radius about a
std::vector<Real> SurfaceCoord(const int NFP, const sctl::Long Nt, const sctl::Long Np, const Real a) {
  std::vector<Real> X(COORD_DIM * Nt * Np);
  for (sctl::Long t = 0; t < Nt; t++) {
    for (sctl::Long p = 0; p < Np; p++) {
      const Real theta = 2 * sctl::const_pi<Real>() * t / (NFP * Nt);
      const Real phi = 2 * sctl::const_pi<Real>() * p / Np;
      const Real R = 1.0 + a * sctl::cos<Real>(phi) + 0.05 * sctl::cos<Real>(phi - NFP * theta);
      const Real Z = a * sctl::sin<Real>(phi) - 0.05 * sctl::sin<Real>(phi - NFP * theta);
      X[(0 * Nt + t) * Np + p] = R * sctl::cos<Real>(theta);
      X[(1 * Nt + t) * Np + p] = R * sctl::sin<Real>(theta);
      X[(2 * Nt + t) * Np + p] = Z;
    }
  }
  return X;
}

// Field at the points X (SoA order) of a circular loop of radius r at height z, carrying the current I
// counterclockwise about the z-axis
std::vector<Real> LoopField(const std::vector<Real>& X, const Real r, const Real z, const Real I) {
  const sctl::Long N = 1000;
  sctl::Vector<Real> Xs(COORD_DIM * N), Js(COORD_DIM * N), Xt(X), B;
  for (sctl::Long i = 0; i < N; i++) {
    const Real s = 2 * sctl::const_pi<Real>() * i / N;
    Xs[0 * N + i] = r * sctl::cos<Real>(s);
    Xs[1 * N + i] = r * sctl::sin<Real>(s);
    Xs[2 * N + i] = z;
    Js[0 * N + i] = -r * sctl::sin<Real>(s) * I * 2 * sctl::const_pi<Real>() / N; // current times length element
    Js[1 * N + i] = r * sctl::cos<Real>(s) * I * 2 * sctl::const_pi<Real>() / N;
    Js[2 * N + i] = 0;
  }
  biest::BiotSavart3D<Real>::FxU()(Xs, Xs, Js, Xt, B);
  return std::vector<Real>(B.begin(), B.end());
}

Real MaxRelDiff(const std::vector<Real>& A, const std::vector<Real>& B) {
  Real err = 0, max_val = 0;
  for (size_t i = 0; i < A.size(); i++) err = std::max<Real>(err, sctl::fabs(A[i] - B[i]));
  for (size_t i = 0; i < A.size(); i++) max_val = std::max<Real>(max_val, sctl::fabs(B[i]));
  return err / max_val;
}

int main(int argc, char** argv) {
  const int digits = 10, NFP = 5;
  const sctl::Long surf_Nt = 30, surf_Np = 30; // surface grid in one field period
  const sctl::Long Nt = 40, Np = 40; // field grid in one field period

  const std::vector<Real> X = SurfaceCoord(NFP, surf_Nt, surf_Np, 0.3); // the surface
  const std::vector<Real> XB = SurfaceCoord(NFP, Nt, Np, 0.3); // the points of the field grid

  { // Exterior: the field of a current loop inside the torus
    biest::ExtVacuumField<Real> ext_field;
    ext_field.Setup(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);

    const Real I = 1.0;
    const std::vector<Real> Bcoil = LoopField(XB, 1.0, 0.0, I);

    std::vector<Real> Bplasma, sigma, J;
    std::tie(Bplasma, sigma, J) = ext_field.ComputeBplasma(ext_field.ComputeBdotN(Bcoil), I);

    const std::vector<Real> Xout = SurfaceCoord(NFP, 10, 10, 0.5); // points outside the surface
    const std::vector<Real> Bplasma_out = ext_field.EvalOffSurface(Xout, sigma, J);

    // Bplasma = -Bcoil outside the surface
    std::cout << "exterior, on the surface:  " << MaxRelDiff(Bplasma, LoopField(XB, 1.0, 0.0, -I)) << '\n';
    std::cout << "exterior, off the surface: " << MaxRelDiff(Bplasma_out, LoopField(Xout, 1.0, 0.0, -I)) << '\n';
  }

  { // Interior: the field of a current loop outside the torus
    const std::vector<Real> B1 = LoopField(XB, 3.0, 1.0, 1.0);

    biest::IntVacuumField<Real> int_field;
    int_field.Setup(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);

    std::vector<Real> B, sigma;
    std::tie(B, sigma) = int_field.ComputeB(int_field.ComputeBdotN(B1), 0);

    const std::vector<Real> Xin = SurfaceCoord(NFP, 10, 10, 0.15); // points inside the surface
    const std::vector<Real> B_in = int_field.EvalOffSurface(Xin, sigma, 0);

    // B = -B1 inside the surface
    std::cout << "interior, on the surface:  " << MaxRelDiff(B, LoopField(XB, 3.0, 1.0, -1.0)) << '\n';
    std::cout << "interior, off the surface: " << MaxRelDiff(B_in, LoopField(Xin, 3.0, 1.0, -1.0)) << '\n';
  }
  return 0;
}
