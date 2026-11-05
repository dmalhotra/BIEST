// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Vacuum fields and Taylor states for a family of surfaces, as in an optimization of the shape: solver objects keep
// their operators and vectors from one call to the next.

#include <biest.hpp>

int main(int argc, char** argv) {
  using Real = double;
  const sctl::Comm comm = sctl::Comm::Self();
  const Real pi = sctl::const_pi<Real>();
  const sctl::Long Nt = 70, Np = 28;

  biest::VacuumField<Real> vacuum_field(comm);
  biest::TaylorState<Real> taylor_state(comm);
  for (const Real a : {0.25, 0.30, 0.35}) { // the minor radius of a torus with 5 field periods
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::None);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        const Real phi = 2 * pi * t / Nt, theta = 2 * pi * p / Np;
        const Real R = 1 + a * sctl::cos<Real>(theta) + 0.05 * sctl::cos<Real>(theta - 5 * phi);
        Svec[0].Coord(t, p, 0) = R * sctl::cos<Real>(phi);
        Svec[0].Coord(t, p, 1) = R * sctl::sin<Real>(phi);
        Svec[0].Coord(t, p, 2) = a * sctl::sin<Real>(theta) - 0.05 * sctl::sin<Real>(theta - 5 * phi);
      }
    }

    sctl::Vector<Real> B0, B1;
    vacuum_field.Compute(B0, 1.0, 0.0, Svec, 1e-8, 100);
    taylor_state.Compute(B1, 1.0, 0.0, 1.0, Svec, 1e-8, 100);

    Real B0max = 0, B1max = 0;
    for (const auto& b : B0) B0max = std::max<Real>(B0max, sctl::fabs<Real>(b));
    for (const auto& b : B1) B1max = std::max<Real>(B1max, sctl::fabs<Real>(b));
    std::cout << "a = " << a << ": max |B| = " << B0max << " (vacuum field), " << B1max << " (Taylor state)\n";
  }
  return 0;
}
