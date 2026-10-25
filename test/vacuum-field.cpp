// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Tests of ExtVacuumField and IntVacuumField on a surface with 5 field periods, against the fields of current
// loops and of a straight wire on the z-axis.

#include "test-utils.hpp"

using namespace biest_test;

int main(int argc, char** argv) {
  const Real pi = sctl::const_pi<Real>();
  const int digits = 10, NFP = 5;
  const sctl::Long surf_Nt = 30, surf_Np = 30; // surface grid in one field period
  const sctl::Long Nt = 40, Np = 40; // field grid in one field period
  Results res;

  // Coordinates (SoA order) on an Nt x Np grid in one field period, of a surface with minor radius about a; the
  // toroidal index increases counterclockwise (orient = 1) or clockwise (orient = -1) about the z-axis
  const auto surface_coord = [pi,NFP](const sctl::Long Nt, const sctl::Long Np, const Real a, const sctl::Integer orient) {
    std::vector<Real> X(COORD_DIM * Nt * Np);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        const Real theta = orient * 2 * pi * t / (NFP * Nt);
        const Real phi = 2 * pi * p / Np;
        const Real R = 1.0 + a * sctl::cos<Real>(phi) + 0.05 * sctl::cos<Real>(phi - NFP * theta);
        const Real Z = a * sctl::sin<Real>(phi) - 0.05 * sctl::sin<Real>(phi - NFP * theta);
        X[(0 * Nt + t) * Np + p] = R * sctl::cos<Real>(theta);
        X[(1 * Nt + t) * Np + p] = R * sctl::sin<Real>(theta);
        X[(2 * Nt + t) * Np + p] = Z;
      }
    }
    return X;
  };

  // Field at the points X (SoA order) of a circular loop of radius r at height z, carrying the current I
  // counterclockwise about the z-axis, plus the field I0 phi_hat / (2 pi R) of a straight wire on the z-axis
  const auto loop_wire_field = [pi](const std::vector<Real>& X, const Real r, const Real z, const Real I, const Real I0) {
    const sctl::Long Ns = 1000, Nx = X.size() / COORD_DIM;
    sctl::Vector<Real> Xs(COORD_DIM * Ns), Js(COORD_DIM * Ns), Xt(X), B;
    for (sctl::Long i = 0; i < Ns; i++) {
      const Real s = 2 * pi * i / Ns;
      Xs[0 * Ns + i] = r * sctl::cos<Real>(s);
      Xs[1 * Ns + i] = r * sctl::sin<Real>(s);
      Xs[2 * Ns + i] = z;
      Js[0 * Ns + i] = -r * sctl::sin<Real>(s) * I * 2 * pi / Ns; // current times length element
      Js[1 * Ns + i] = r * sctl::cos<Real>(s) * I * 2 * pi / Ns;
      Js[2 * Ns + i] = 0;
    }
    biest::BiotSavart3D<Real>::FxU()(Xs, Xs, Js, Xt, B);
    for (sctl::Long i = 0; i < Nx; i++) {
      const Real R2 = X[0 * Nx + i] * X[0 * Nx + i] + X[1 * Nx + i] * X[1 * Nx + i];
      B[0 * Nx + i] += -X[1 * Nx + i] * I0 / (2 * pi * R2);
      B[1 * Nx + i] += X[0 * Nx + i] * I0 / (2 * pi * R2);
    }
    return std::vector<Real>(B.begin(), B.end());
  };

  const auto rel_diff = [](const std::vector<Real>& A, const std::vector<Real>& B) {
    return MaxRelDiff(sctl::Vector<Real>(A), sctl::Vector<Real>(B));
  };

  for (const sctl::Integer orient : {1, -1}) {
    const std::string grid = (orient == 1 ? " (counterclockwise grid)" : " (clockwise grid)");
    const std::vector<Real> X = surface_coord(surf_Nt, surf_Np, 0.3, orient); // the surface
    const std::vector<Real> XB = surface_coord(Nt, Np, 0.3, orient); // the points of the field grid

    { // Exterior: Bcoil is the field of a loop inside the torus, and Bplasma = -Bcoil outside
      const Real I = 1.0;
      biest::ExtVacuumField<Real> ext_field;
      ext_field.Setup(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);

      // Jplasma is the current in the direction of decreasing toroidal index; Bplasma is the field of the current -I
      // counterclockwise about the z-axis
      std::vector<Real> Bplasma, sigma, J;
      std::tie(Bplasma, sigma, J) = ext_field.ComputeBplasma(ext_field.ComputeBdotN(loop_wire_field(XB, 1.0, 0.0, I, 0)), orient * I);

      const std::vector<Real> Xout = surface_coord(10, 10, 0.5, orient);
      const std::vector<Real> Bplasma_out = ext_field.EvalOffSurface(Xout, sigma, J);
      res.Check(("ExtVacuumField: Bplasma on the surface" + grid).c_str(), rel_diff(Bplasma, loop_wire_field(XB, 1.0, 0.0, -I, 0)), 1e-5);
      res.Check(("ExtVacuumField::EvalOffSurface: Bplasma outside" + grid).c_str(), rel_diff(Bplasma_out, loop_wire_field(Xout, 1.0, 0.0, -I, 0)), 1e-9);
    }

    { // Interior: B1 is minus the field of a loop outside the torus and of a wire carrying I0; then B = -B1 inside
      const Real I0 = 1.0;
      biest::IntVacuumField<Real> int_field;
      int_field.Setup(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);

      std::vector<Real> B, sigma;
      std::tie(B, sigma) = int_field.ComputeB(int_field.ComputeBdotN(loop_wire_field(XB, 3.0, 1.0, -1.0, -I0)), I0);

      const std::vector<Real> Xin = surface_coord(10, 10, 0.15, orient);
      const std::vector<Real> B_in = int_field.EvalOffSurface(Xin, sigma, I0);
      res.Check(("IntVacuumField: B on the surface" + grid).c_str(), rel_diff(B, loop_wire_field(XB, 3.0, 1.0, 1.0, I0)), 1e-5);
      res.Check(("IntVacuumField::EvalOffSurface: B inside" + grid).c_str(), rel_diff(B_in, loop_wire_field(Xin, 3.0, 1.0, 1.0, I0)), 1e-9);
    }
  }

  return res.ExitCode();
}
