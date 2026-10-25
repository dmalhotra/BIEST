// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Defining surfaces.

#include <biest.hpp>

int main(int argc, char** argv) {
  using Real = double;
  const sctl::Long Nt = 100, Np = 40; // grid dimensions in the toroidal and poloidal directions

  // A predefined shape
  biest::Surface<Real> S0(Nt, Np, biest::SurfType::W7X_);

  // A five-period stellarator from Fourier coefficients
  const int NFP = 5;
  biest::Surface<Real> S1(Nt, Np, biest::SurfType::None);
  for (sctl::Long t = 0; t < Nt; t++) {
    for (sctl::Long p = 0; p < Np; p++) {
      const Real theta = 2 * sctl::const_pi<Real>() * t / Nt; // toroidal angle
      const Real phi = 2 * sctl::const_pi<Real>() * p / Np; // poloidal angle
      const Real R = 1.0 + 0.3 * sctl::cos<Real>(phi) + 0.05 * sctl::cos<Real>(phi - NFP * theta);
      const Real Z = 0.3 * sctl::sin<Real>(phi) - 0.05 * sctl::sin<Real>(phi - NFP * theta);
      S1.Coord(t, p, 0) = R * sctl::cos<Real>(theta);
      S1.Coord(t, p, 1) = R * sctl::sin<Real>(theta);
      S1.Coord(t, p, 2) = Z;
    }
  }

  // Normals, area elements and the surface area
  biest::SurfaceOp<Real> surf_op(sctl::Comm::Self(), Nt, Np);
  sctl::Vector<Real> dX, normal, area_elem, area;
  surf_op.Grad2D(dX, S1.Coord());
  surf_op.SurfNormalAreaElem(&normal, &area_elem, dX, &S1.Coord());
  sctl::Vector<Real> one(Nt * Np);
  one = 1;
  surf_op.SurfInteg(area, area_elem, one); // area[0] is the surface area
  std::cout << "area = " << area[0] << '\n';

  // Visualization
  biest::WriteVTK("S0", S0); // writes S0.pvtp and S0_000000.vtp
  biest::WriteVTK("S1", S1, normal); // the surface with its normal vectors
  return 0;
}
