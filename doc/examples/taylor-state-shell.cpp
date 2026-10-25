// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// A Taylor state in a toroidal shell.

#include <biest.hpp>

int main(int argc, char** argv) {
  using Real = double;
  const sctl::Comm comm = sctl::Comm::Self();

  sctl::Vector<biest::Surface<Real>> Svec(2);
  Svec[0] = biest::Surface<Real>(104, 52, biest::SurfType::RotatingEllipseWide); // outer surface
  Svec[1] = biest::Surface<Real>(104, 26, biest::SurfType::RotatingEllipseNarrow); // inner surface

  const Real lambda = 1.0, tor_flux = 1.0, pol_flux = 0.5;
  sctl::Vector<Real> B;
  sctl::Vector<sctl::Vector<Real>> m, sigma;
  biest::TaylorState<Real,1,12,24>::Compute(B, tor_flux, pol_flux, lambda, Svec, comm, 1e-6, 200, 0.1, &m, &sigma);

  // B holds the field on the grid of each surface, one after the other
  std::cout << "B.Dim() = " << B.Dim() << " = 3 * (104 * 52 + 104 * 26)\n";

  sctl::Vector<Real> Xt(3), Bt; // a point inside the shell, {x, y, z}
  Xt[0] = 2.5;
  Xt[1] = 0.0;
  Xt[2] = 0.0;
  biest::TaylorState<Real,1,12,24>::EvalOffSurface(Bt, Xt, Svec, tor_flux, pol_flux, lambda, m, sigma, comm, 1e-6);
  std::cout << "B(2.5, 0, 0) = " << Bt[0] << ' ' << Bt[1] << ' ' << Bt[2] << '\n';
  biest::WriteVTK("B", Svec, B);
  return 0;
}
