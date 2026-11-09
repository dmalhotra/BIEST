// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// Tests of VacuumField and TaylorState. The toroidal flux of the computed field is measured independently, by
// integrating B . phi_hat over the cross-section of the domain in the plane phi = 0, and the poloidal flux of a shell
// by integrating B . (-z_hat) over a band of the midplane between the surfaces.

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
  using VacuumField = biest::VacuumField<Real>;
  using TaylorState = biest::TaylorState<Real>;
  const biest::SingularQuadParams quad{6, 10, 20}; // a cheap singular quadrature: the patch radius, the radial and angular nodes
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

  { // One surface: B.n = 0, and the toroidal flux, counterclockwise about the z-axis
    sctl::Vector<biest::Surface<Real>> Svec(1);
    Svec[0] = torus(Nt, Np, outer);
    sctl::Vector<Real> Xtrg, W;
    cross_section(Xtrg, W, nullptr, outer);

    sctl::Vector<Real> B, Btrg;
    sctl::Vector<sctl::Vector<Real>> J;
    VacuumField(comm, quad).Compute(B, 1.0, 0.0, Svec, gmres_tol, gmres_iter, &J);
    VacuumField::EvalOffSurface(Btrg, Xtrg, Svec, 1.0, 0.0, J, comm);
    res.Check("VacuumField, one surface: toroidal flux = tor_flux", sctl::fabs<Real>(toroidal_flux(Btrg, W) - 1), 1e-3);

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
    TaylorState(comm, quad).Compute(B, 1.0, 0.0, 1.0, Svec, gmres_tol, gmres_iter, 0.1, &m, &sigma);
    TaylorState::EvalOffSurface(Btrg, Xtrg, Svec, 1.0, 0.0, 1.0, m, sigma, comm);
    res.Check("TaylorState (lambda = 1), one surface: toroidal flux = tor_flux", sctl::fabs<Real>(toroidal_flux(Btrg, W) - 1), 1e-3);
  }

  { // An object keeps its operators from one call to the next, also when the surface changes: the fields of a new
    // object, for a surface, then a smaller one (set up in the memory of the first), then the first again
    VacuumField vacuum_field(comm, quad);
    TaylorState taylor_state(comm, quad);
    const biest::Surface<Real> S[3] = {torus(Nt, Np, outer), torus(48, 24, inner), torus(Nt, Np, outer)};
    Real err_vacuum = 0, err_taylor = 0;
    for (const auto& S_ : S) {
      sctl::Vector<biest::Surface<Real>> Svec(1);
      Svec[0] = S_;
      sctl::Vector<Real> B0, B1;
      VacuumField(comm, quad).Compute(B0, 1.0, 0.0, Svec, gmres_tol, gmres_iter);
      vacuum_field.Compute(B1, 1.0, 0.0, Svec, gmres_tol, gmres_iter);
      err_vacuum = std::max(err_vacuum, MaxRelDiff(B1, B0));
      TaylorState(comm, quad).Compute(B0, 1.0, 0.0, 1.0, Svec, gmres_tol, gmres_iter);
      taylor_state.Compute(B1, 1.0, 0.0, 1.0, Svec, gmres_tol, gmres_iter);
      err_taylor = std::max(err_taylor, MaxRelDiff(B1, B0));
    }
    res.Check("VacuumField object over 3 surfaces: B = that of a new object", err_vacuum, 1e-8);
    res.Check("TaylorState object over 3 surfaces: B = that of a new object", err_taylor, 1e-8);
  }

  { // Field periods: the tori have 5 (delta cos(theta - 5 phi)); declared, the solve is on one of them, and the fields
    // are those of the solve on the full tori, for one surface and for a shell
    const sctl::Long Nt5 = 80; // a multiple of 5
    for (const sctl::Long Nsurf : {1, 2}) {
      sctl::Vector<biest::Surface<Real>> Svec(Nsurf), Svec5(Nsurf);
      Svec[0] = torus(Nt5, Np, outer);
      if (Nsurf == 2) Svec[1] = torus(Nt5, Np, inner);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        Svec5[i] = Svec[i];
        Svec5[i].SetNFP(5);
      }
      sctl::Vector<Real> B, B5;
      sctl::Vector<sctl::Vector<Real>> J, J5;
      VacuumField(comm, quad).Compute(B, 1.0, 0.5, Svec, gmres_tol, gmres_iter, &J);
      VacuumField(comm, quad).Compute(B5, 1.0, 0.5, Svec5, gmres_tol, gmres_iter, &J5);
      Real err_vacuum = MaxRelDiff(B5, B);
      for (sctl::Long i = 0; i < J.Dim(); i++) err_vacuum = std::max(err_vacuum, MaxRelDiff(J5[i], J[i]));
      sctl::Vector<sctl::Vector<Real>> m, sigma, m5, sigma5;
      TaylorState(comm, quad).Compute(B, 1.0, 0.5, 1.0, Svec, gmres_tol, gmres_iter, 0.1, &m, &sigma);
      TaylorState(comm, quad).Compute(B5, 1.0, 0.5, 1.0, Svec5, gmres_tol, gmres_iter, 0.1, &m5, &sigma5);
      Real err_taylor = MaxRelDiff(B5, B);
      for (sctl::Long i = 0; i < m.Dim(); i++) err_taylor = std::max(err_taylor, std::max(MaxRelDiff(m5[i], m[i]), MaxRelDiff(sigma5[i], sigma[i])));
      const std::string name = (Nsurf == 1 ? "one surface" : "shell");
      res.Check(("VacuumField on one of 5 field periods, " + name + ": B, J = those on the full torus").c_str(), err_vacuum, 1e-8);
      res.Check(("TaylorState on one of 5 field periods, " + name + ": B, m, sigma = those on the full torus").c_str(), err_taylor, 1e-8);
    }
  }

  { // Shell: VacuumField(tor_flux, pol_flux) is the limit lambda -> 0 of TaylorState(tor_flux, pol_flux)
    sctl::Vector<biest::Surface<Real>> Svec(2);
    Svec[0] = torus(Nt, Np, outer);
    Svec[1] = torus(Nt, Np, inner);
    sctl::Vector<Real> Bv, Bt;
    VacuumField(comm, quad).Compute(Bv, 1.0, 0.5, Svec, gmres_tol, gmres_iter);
    TaylorState(comm, quad).Compute(Bt, 1.0, 0.5, 1e-10, Svec, gmres_tol, gmres_iter);
    res.Check("Shell: VacuumField(1, 0.5) = TaylorState(lambda = 1e-10; 1, 0.5)", MaxRelDiff(Bt, Bv), 1e-2);
  }

  { // Shell whose inner surface has the larger area, in both orders: the outer surface encloses the larger volume
    const TorusShape outer2{0.55, 0, 0.05, 0}, inner2{0.25, 0.6, 0, 6};
    const sctl::Long Nt2 = 96, Np2 = 96; // 64 poloidal points do not resolve the six lobes of the inner surface
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

      sctl::Vector<Real> B, Btrg0, Btrg1;
      sctl::Vector<sctl::Vector<Real>> J; // one solve, the field inside for both fluxes
      VacuumField(comm, quad).Compute(B, 1.0, 0.0, Svec, gmres_tol, gmres_iter, &J);
      VacuumField::EvalOffSurface(Btrg0, Xtrg, Svec, 1.0, 0.0, J, comm);
      VacuumField::EvalOffSurface(Btrg1, Xtrg, Svec, 0.0, 1.0, J, comm);
      const std::string name = (inner_first ? "Shell (inner surface first): " : "Shell (outer surface first): ");
      res.Check((name + "toroidal flux for (tor_flux, pol_flux) = (1, 0)").c_str(), sctl::fabs<Real>(toroidal_flux(Btrg0, W) - 1), 1e-2);
      res.Check((name + "toroidal flux for (tor_flux, pol_flux) = (0, 1)").c_str(), sctl::fabs<Real>(toroidal_flux(Btrg1, W)), 1e-2);
    }
  }

  { // The signs of the fluxes for each direction of the grid indices: the toroidal flux counterclockwise about the
    // z-axis, and the poloidal flux of a shell, through the band Z = 0 between the surfaces on the outboard side, toward
    // -z, the direction of the field of a current counterclockwise about the z-axis inside the inner surface; circular
    // cross-sections, so that the fields are axisymmetric and the flux through the band is 2 pi times that at phi = 0.
    // The singular quadrature chosen for the tolerance: the cheap one gives the fluxes of a shell to 2 digits.
    const TorusShape outer0{0.45, 0, 0, 0}, inner0{0.2, 0, 0, 0};
    const sctl::Long Nt0 = 48, Np0 = 32;
    const Real gmres_tol0 = 1e-6;
    const auto reversed = [](const biest::Surface<Real>& S, const bool rev_t, const bool rev_p) { // S with the grid indices reversed
      const sctl::Long Nt = S.NTor(), Np = S.NPol();
      biest::Surface<Real> S_(Nt, Np, biest::SurfType::None);
      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          for (sctl::Integer k = 0; k < COORD_DIM; k++) S_.Coord(t, p, k) = S.Coord(rev_t ? (Nt - t) % Nt : t, rev_p ? (Np - p) % Np : p, k);
        }
      }
      return S_;
    };
    sctl::Vector<Real> Xb, Wb; // the band at phi = 0, R from 1 + a of the inner surface to that of the outer
    {
      const sctl::Long Ns = 16;
      sctl::Vector<Real> s, ws;
      sctl::ChebBasis<Real>::quad_rule(Ns, s, ws); // Gauss-Legendre rule on [0,1]
      Xb.ReInit(COORD_DIM * Ns);
      Wb.ReInit(Ns);
      for (sctl::Long j = 0; j < Ns; j++) {
        const Real R = 1 + inner0.a + (outer0.a - inner0.a) * s[j];
        Xb[j * COORD_DIM + 0] = R;
        Xb[j * COORD_DIM + 1] = 0;
        Xb[j * COORD_DIM + 2] = 0;
        Wb[j] = 2 * pi * R * (outer0.a - inner0.a) * ws[j];
      }
    }
    const auto poloidal_flux = [&Xb,&Wb](const sctl::Vector<Real>& Btrg) { // B . (-z_hat) over the band, at the targets after those of the cross-section
      const sctl::Long n0 = Btrg.Dim() / COORD_DIM - Wb.Dim();
      Real flux = 0;
      for (sctl::Long j = 0; j < Wb.Dim(); j++) flux -= Btrg[(n0 + j) * COORD_DIM + 2] * Wb[j];
      return flux;
    };
    for (const sctl::Integer Nsurf : {1, 2}) {
      sctl::Vector<Real> Xtrg, W;
      cross_section(Xtrg, W, (Nsurf == 2 ? &inner0 : nullptr), outer0);
      if (Nsurf == 2) { // the band after the cross-section
        const sctl::Long N0 = Xtrg.Dim();
        sctl::Vector<Real> X(N0 + Xb.Dim());
        for (sctl::Long i = 0; i < N0; i++) X[i] = Xtrg[i];
        for (sctl::Long i = 0; i < Xb.Dim(); i++) X[N0 + i] = Xb[i];
        Xtrg.Swap(X);
      }
      for (const sctl::Integer rev : {0, 1, 2, 3}) {
        const bool rev_t = rev & 1, rev_p = rev & 2;
        sctl::Vector<biest::Surface<Real>> Svec(Nsurf);
        Svec[0] = reversed(torus(Nt0, Np0, outer0), rev_t, rev_p);
        if (Nsurf == 2) Svec[1] = reversed(torus(Nt0, Np0, inner0), rev_t, rev_p);
        sctl::Vector<Real> B, Btrg;
        sctl::Vector<sctl::Vector<Real>> J, m, sigma; // of the fields of unit flux: one solve each for both fluxes
        VacuumField(comm).Compute(B, 1.0, 0.0, Svec, gmres_tol0, gmres_iter, &J);
        TaylorState(comm).Compute(B, 1.0, 0.0, 1.0, Svec, gmres_tol0, gmres_iter, 0.1, &m, &sigma);
        Real err = 0;
        for (const auto& flux : std::vector<std::pair<Real, Real>>{{1, 0}, {0, 1}}) {
          if (Nsurf == 1 && flux.second) continue;
          for (const bool taylor : {false, true}) {
            if (taylor) TaylorState::EvalOffSurface(Btrg, Xtrg, Svec, flux.first, flux.second, 1.0, m, sigma, comm);
            else VacuumField::EvalOffSurface(Btrg, Xtrg, Svec, flux.first, flux.second, J, comm);
            err = std::max(err, sctl::fabs<Real>(toroidal_flux(Btrg, W) - flux.first));
            if (Nsurf == 2) err = std::max(err, sctl::fabs<Real>(poloidal_flux(Btrg) - flux.second));
          }
        }
        const std::string name = std::string(Nsurf == 1 ? "One surface" : "Shell") + ", t " + (rev_t ? "clockwise" : "counterclockwise") + ", p " + (rev_p ? "clockwise" : "counterclockwise") + ": VacuumField and TaylorState (lambda = 1) fluxes = tor_flux" + (Nsurf == 2 ? ", pol_flux" : "");
        res.Check(name.c_str(), err, 1e-4);
      }
    }
  }

  return res.ExitCode();
}
