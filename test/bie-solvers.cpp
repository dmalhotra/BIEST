// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Tests of VacuumField and TaylorState. The toroidal flux of the computed field is measured independently, by
// integrating B . phi_hat over the cross-section of the domain in the plane phi = 0.

#include "test-utils.hpp"

using namespace biest_test;

int main(int argc, char** argv) {
  // Toroidal surface R = 1 + r cos(theta) + delta cos(theta - 5 phi), Z = r sin(theta) - delta sin(theta - 5 phi),
  // with r = a (1 + eps cos(m theta)); grid index t is at phi = 2 pi t / Nt and p at theta = 2 pi p / Np.
  struct TorusShape {
    Real a, eps, delta;
    int m;
  };

  const sctl::Comm comm = sctl::Comm::Self();
  const Real pi = sctl::const_pi<Real>();
  const Real gmres_tol = 1e-10;
  const sctl::Long gmres_iter = 200;
  using VacuumField = biest::VacuumField<Real,1,6,10>;
  using TaylorState = biest::TaylorState<Real,1,6,10>;
  Results res;

  const auto torus = [pi](const sctl::Long Nt, const sctl::Long Np, const TorusShape& T) {
    biest::Surface<Real> S(Nt, Np, biest::SurfType::None);
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        const Real phi = 2 * pi * t / Nt, theta = 2 * pi * p / Np;
        const Real r = T.a * (1 + T.eps * sctl::cos<Real>(T.m * theta));
        const Real R = 1 + r * sctl::cos<Real>(theta) + T.delta * sctl::cos<Real>(theta - 5 * phi);
        S.Coord(t, p, 0) = R * sctl::cos<Real>(phi);
        S.Coord(t, p, 1) = R * sctl::sin<Real>(phi);
        S.Coord(t, p, 2) = r * sctl::sin<Real>(theta) - T.delta * sctl::sin<Real>(theta - 5 * phi);
      }
    }
    return S;
  };

  // Quadrature nodes (AoS) and weights on the cross-section at phi = 0 between the curves of the surfaces inner
  // (or the point R = 1, Z = 0 if null) and outer.
  const auto cross_section = [pi](sctl::Vector<Real>& Xtrg, sctl::Vector<Real>& W, const TorusShape* inner, const TorusShape& outer) {
    const auto curve = [](const TorusShape& T, const Real theta, Real* P, Real* dP) { // (R - 1, Z) and its derivative
      const Real r = T.a * (1 + T.eps * sctl::cos<Real>(T.m * theta));
      const Real dr = -T.a * T.eps * T.m * sctl::sin<Real>(T.m * theta);
      P[0] = (r + T.delta) * sctl::cos<Real>(theta);
      P[1] = (r - T.delta) * sctl::sin<Real>(theta);
      dP[0] = dr * sctl::cos<Real>(theta) - (r + T.delta) * sctl::sin<Real>(theta);
      dP[1] = dr * sctl::sin<Real>(theta) + (r - T.delta) * sctl::cos<Real>(theta);
    };
    const sctl::Long Ntheta = 128, Ns = 16;
    sctl::Vector<Real> s, ws;
    sctl::ChebBasis<Real>::quad_rule(Ns, s, ws); // Gauss-Legendre rule on [0,1]
    Xtrg.ReInit(COORD_DIM * Ntheta * Ns);
    W.ReInit(Ntheta * Ns);
    for (sctl::Long i = 0; i < Ntheta; i++) {
      const Real theta = 2 * pi * i / Ntheta;
      Real A[2] = {0, 0}, dA[2] = {0, 0}, B[2], dB[2];
      if (inner) curve(*inner, theta, A, dA);
      curve(outer, theta, B, dB);
      for (sctl::Long j = 0; j < Ns; j++) { // P = (1 - s) A + s B
        const sctl::Long n = i * Ns + j;
        const Real dP_dtheta[2] = {(1 - s[j]) * dA[0] + s[j] * dB[0], (1 - s[j]) * dA[1] + s[j] * dB[1]};
        const Real dP_ds[2] = {B[0] - A[0], B[1] - A[1]};
        Xtrg[n * COORD_DIM + 0] = 1 + (1 - s[j]) * A[0] + s[j] * B[0];
        Xtrg[n * COORD_DIM + 1] = 0;
        Xtrg[n * COORD_DIM + 2] = (1 - s[j]) * A[1] + s[j] * B[1];
        W[n] = sctl::fabs<Real>(dP_dtheta[0] * dP_ds[1] - dP_dtheta[1] * dP_ds[0]) * ws[j] * 2 * pi / Ntheta;
      }
    }
  };

  const auto toroidal_flux = [](const sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& W) { // integral of B . phi_hat; phi_hat = y_hat at phi = 0
    Real flux = 0;
    for (sctl::Long n = 0; n < W.Dim(); n++) flux += Btrg[n * COORD_DIM + 1] * W[n];
    return flux;
  };

  const TorusShape outer{0.45, 0, 0.05, 0}, inner{0.2, 0, 0.05, 0};
  const sctl::Long Nt = 72, Np = 32;

  { // One surface: B.n = 0, and the toroidal flux; VacuumField with tor_flux > 0 gives B in the direction of decreasing t
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = torus(Nt, Np, outer);
    sctl::Vector<Real> Xtrg, W;
    cross_section(Xtrg, W, nullptr, outer);

    sctl::Vector<Real> B, J, Btrg;
    VacuumField::Compute(B, J, 1.0, 0.0, Svec, comm, gmres_tol, gmres_iter);
    VacuumField::EvalOffSurface(Btrg, Xtrg, Svec, J, comm);
    res.Check("VacuumField, one surface: toroidal flux = -tor_flux", sctl::fabs<Real>(toroidal_flux(Btrg, W) + 1), 1e-3);

    sctl::Vector<Real> dX, normal, BdotN(Nt * Np);
    biest::SurfaceOp<Real> surf_op(comm, Nt, Np);
    surf_op.Grad2D(dX, Svec[0].Coord());
    surf_op.SurfNormalAreaElem(&normal, nullptr, dX, &Svec[0].Coord());
    BdotN = 0;
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < Nt * Np; i++) BdotN[i] += B[k * Nt * Np + i] * normal[k * Nt * Np + i];
    }
    res.Check("VacuumField, one surface: B.n = 0", MaxAbs(BdotN) / MaxAbs(B), 1e-8);

    sctl::Vector<sctl::Vector<Real>> m, sigma;
    TaylorState::Compute(B, 1.0, 0.0, 1.0, Svec, comm, gmres_tol, gmres_iter, 0.1, &m, &sigma);
    TaylorState::EvalOffSurface(Btrg, Xtrg, Svec, 1.0, 0.0, 1.0, m, sigma, comm);
    res.Check("TaylorState (lambda = 1), one surface: toroidal flux = tor_flux", sctl::fabs<Real>(toroidal_flux(Btrg, W) - 1), 1e-3);
  }

  { // Shell: VacuumField(tor_flux, pol_flux) is the limit lambda -> 0 of TaylorState(-tor_flux, -pol_flux)
    sctl::Vector<biest::Surface<Real>> Svec(2);
    Svec[0] = torus(Nt, Np, outer);
    Svec[1] = torus(Nt, Np, inner);
    sctl::Vector<Real> Bv, J, Bt;
    VacuumField::Compute(Bv, J, 1.0, 0.5, Svec, comm, gmres_tol, gmres_iter);
    TaylorState::Compute(Bt, -1.0, -0.5, 1e-10, Svec, comm, gmres_tol, gmres_iter);
    res.Check("Shell: VacuumField(1, 0.5) = TaylorState(lambda = 1e-10; -1, -0.5)", MaxRelDiff(Bt, Bv), 1e-2);
  }

  { // Shell whose inner surface has the larger area, in both orders: the outer surface encloses the larger volume
    const TorusShape outer2{0.55, 0, 0.05, 0}, inner2{0.25, 0.6, 0, 6};
    const sctl::Long Nt2 = 96, Np2 = 64;
    const biest::Surface<Real> S_outer = torus(Nt2, Np2, outer2), S_inner = torus(Nt2, Np2, inner2);
    { // test setup: the inner surface has the larger area
      const auto area = [&comm,Nt2,Np2](const biest::Surface<Real>& S) {
        sctl::Vector<Real> dX, area_elem;
        biest::SurfaceOp<Real> surf_op(comm, Nt2, Np2);
        surf_op.Grad2D(dX, S.Coord());
        surf_op.SurfNormalAreaElem(nullptr, &area_elem, dX, nullptr);
        Real A = 0;
        for (const auto& a : area_elem) A += a;
        return A;
      };
      res.Check("Shell: the inner surface has the larger area (test setup)", (area(S_inner) > area(S_outer) ? 0 : 1), 0);
    }

    sctl::Vector<Real> Xtrg, W;
    cross_section(Xtrg, W, &inner2, outer2);
    for (const bool inner_first : {false, true}) {
      sctl::Vector<biest::Surface<Real>> Svec(2);
      Svec[0] = (inner_first ? S_inner : S_outer);
      Svec[1] = (inner_first ? S_outer : S_inner);

      sctl::Vector<Real> B, J, Btrg0, Btrg1;
      VacuumField::Compute(B, J, 1.0, 0.0, Svec, comm, gmres_tol, gmres_iter);
      VacuumField::EvalOffSurface(Btrg0, Xtrg, Svec, J, comm);
      VacuumField::Compute(B, J, 0.0, 1.0, Svec, comm, gmres_tol, gmres_iter);
      VacuumField::EvalOffSurface(Btrg1, Xtrg, Svec, J, comm);
      const std::string name = (inner_first ? "Shell (inner surface first): " : "Shell (outer surface first): ");
      res.Check((name + "toroidal flux for (tor_flux, pol_flux) = (1, 0)").c_str(), sctl::fabs<Real>(toroidal_flux(Btrg0, W) + 1), 1e-2);
      res.Check((name + "toroidal flux for (tor_flux, pol_flux) = (0, 1)").c_str(), sctl::fabs<Real>(toroidal_flux(Btrg1, W)), 1e-2);
    }
  }

  return res.ExitCode();
}
