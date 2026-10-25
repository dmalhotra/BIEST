// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Surface operators.

#include <biest.hpp>

int main(int argc, char** argv) {
  using Real = double;
  constexpr int COORD_DIM = 3;
  const sctl::Long Nt = 140, Np = 70;
  const sctl::Comm comm = sctl::Comm::Self();

  const biest::Surface<Real> S(Nt, Np, biest::SurfType::Stell);
  const sctl::Vector<Real>& X = S.Coord();

  biest::SurfaceOp<Real> surf_op(comm, Nt, Np);
  sctl::Vector<Real> dX, normal, area_elem;
  surf_op.Grad2D(dX, X);
  surf_op.SurfNormalAreaElem(&normal, &area_elem, dX, &X);

  // A smooth function on the surface
  sctl::Vector<Real> f(Nt * Np);
  for (sctl::Long i = 0; i < Nt * Np; i++) f[i] = sctl::sin<Real>(X[0 * Nt * Np + i]) * X[2 * Nt * Np + i];

  { // Differential operators; the integral of lap(f) over the closed surface is zero
    sctl::Vector<Real> GradF, LapF, absLapF(Nt * Np), I, I_abs;
    surf_op.SurfGrad(GradF, dX, f); // surface gradient (3 components per component of f)
    surf_op.SurfDiv(LapF, dX, GradF); // surface divergence of a tangential vector field
    for (sctl::Long i = 0; i < Nt * Np; i++) absLapF[i] = sctl::fabs(LapF[i]);
    surf_op.SurfInteg(I, area_elem, LapF);
    surf_op.SurfInteg(I_abs, area_elem, absLapF);
    std::cout << "integral of lap(f) / integral of |lap(f)| = " << I[0] / I_abs[0] << '\n';
  }

  { // Solve the Laplace-Beltrami equation and check the residual
    sctl::Vector<Real> u, Lu, f0;
    surf_op.InvSurfLap(u, dX, f, 1e-10, 100);
    surf_op.SurfLap(Lu, dX, u);
    surf_op.ProjZeroMean(f0, dX, f); // the equation is solved for the mean-zero part of f
    Real err = 0, max_val = 0;
    for (sctl::Long i = 0; i < Nt * Np; i++) err = std::max<Real>(err, sctl::fabs(Lu[i] - f0[i]));
    for (sctl::Long i = 0; i < Nt * Np; i++) max_val = std::max<Real>(max_val, sctl::fabs(f0[i]));
    std::cout << "Laplace-Beltrami residual: " << err / max_val << '\n';
  }

  { // Hodge decomposition of the toroidal tangent vector
    sctl::Vector<Real> V(COORD_DIM * Nt * Np);
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < Nt * Np; i++) V[k * Nt * Np + i] = dX[(2 * k + 0) * Nt * Np + i];
    }
    sctl::Vector<Real> Vn, Vd, Vc, Vh, divV, divVh;
    surf_op.HodgeDecomp(Vn, Vd, Vc, Vh, V, dX, normal, 1e-10, 100);
    surf_op.SurfDiv(divV, dX, V);
    surf_op.SurfDiv(divVh, dX, Vh);
    Real max_divV = 0, max_divVh = 0;
    for (const auto x : divV) max_divV = std::max<Real>(max_divV, sctl::fabs(x));
    for (const auto x : divVh) max_divVh = std::max<Real>(max_divVh, sctl::fabs(x));
    std::cout << "max |div(Vh)| / max |div(V)| = " << max_divVh / max_divV << '\n';

    // Upsample the surface and the harmonic vector field to a finer grid for visualization
    biest::Surface<Real> S_fine(2 * Nt, 2 * Np, biest::SurfType::None);
    sctl::Vector<Real> Vh_fine;
    biest::SurfaceOp<Real>::Upsample(X, Nt, Np, S_fine.Coord(), 2 * Nt, 2 * Np);
    biest::SurfaceOp<Real>::Upsample(Vh, Nt, Np, Vh_fine, 2 * Nt, 2 * Np);
    biest::WriteVTK("Vh", S_fine, Vh_fine);
  }
  return 0;
}
