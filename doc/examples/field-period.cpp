// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Layer potentials with field-period symmetry.

#include <biest.hpp>

int main(int argc, char** argv) {
  using Real = double;
  constexpr int COORD_DIM = 3;
  const int NFP = 5; // number of field periods
  const sctl::Long Nt = 30, Np = 30; // grid dimensions in one field period
  const int digits = 10; // requested number of accurate digits

  // Surface coordinates in one field period, toroidal angle in [0, 2*pi/NFP)
  sctl::Vector<Real> X(COORD_DIM * Nt * Np);
  for (sctl::Long t = 0; t < Nt; t++) {
    for (sctl::Long p = 0; p < Np; p++) {
      const Real theta = 2 * sctl::const_pi<Real>() * t / (NFP * Nt);
      const Real phi = 2 * sctl::const_pi<Real>() * p / Np;
      const Real R = 1.0 + 0.3 * sctl::cos<Real>(phi) + 0.05 * sctl::cos<Real>(phi - NFP * theta);
      const Real Z = 0.3 * sctl::sin<Real>(phi) - 0.05 * sctl::sin<Real>(phi - NFP * theta);
      X[(0 * Nt + t) * Np + p] = R * sctl::cos<Real>(theta);
      X[(1 * Nt + t) * Np + p] = R * sctl::sin<Real>(theta);
      X[(2 * Nt + t) * Np + p] = Z;
    }
  }

  // Laplace double-layer operator: 1 density and 1 potential component
  biest::FieldPeriodBIOp<Real, COORD_DIM, 1, 1> biop;
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biop.BuildSurface(X, NFP, Nt, Np); // full torus
  biop.SetupSingular(Svec, biest::Laplace3D<Real>::DxU(), digits, NFP, NFP * Nt, Np, Nt, Np);

  // Double-layer potential of a constant density; it equals -1/2 on the surface
  sctl::Vector<Real> F(Nt * Np), U;
  F = 1;
  biop.Eval(U, F, NFP, Nt, Np);

  Real err = 0;
  for (const auto u : U) err = std::max<Real>(err, sctl::fabs(u + 0.5));
  std::cout << "quadrature grid: " << biop.QuadNt() << " x " << biop.QuadNp() << '\n';
  std::cout << "max |U + 1/2| = " << err << '\n';

  biest::WriteVTK("U", NFP, false, Nt, Np, X, Nt, Np, U); // U on all field periods
  return 0;
}
