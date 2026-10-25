// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

// The virtual casing principle with BoundaryIntegralOp.

#include <biest.hpp>

using Real = double;
constexpr int COORD_DIM = 3;

// Field at the points X (SoA order) of a circular loop of radius r at height z, carrying the current I
// counterclockwise about the z-axis
sctl::Vector<Real> LoopField(const sctl::Vector<Real>& X, const Real r, const Real z, const Real I) {
  const sctl::Long N = 2000;
  sctl::Vector<Real> Xs(COORD_DIM * N), Js(COORD_DIM * N), B;
  for (sctl::Long i = 0; i < N; i++) {
    const Real s = 2 * sctl::const_pi<Real>() * i / N;
    Xs[0 * N + i] = r * sctl::cos<Real>(s);
    Xs[1 * N + i] = r * sctl::sin<Real>(s);
    Xs[2 * N + i] = z;
    Js[0 * N + i] = -r * sctl::sin<Real>(s) * I * 2 * sctl::const_pi<Real>() / N;
    Js[1 * N + i] = r * sctl::cos<Real>(s) * I * 2 * sctl::const_pi<Real>() / N;
    Js[2 * N + i] = 0;
  }
  biest::BiotSavart3D<Real>::FxU()(Xs, Xs, Js, X, B);
  return B;
}

int main(int argc, char** argv) {
  const sctl::Long Nt = 140, Np = 70;
  const sctl::Long N = Nt * Np;
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::AxisymCircleWide);
  const sctl::Vector<Real>& X = Svec[0].Coord();

  // Total field on the surface: a "plasma" loop inside the torus and a "coil" loop outside
  const sctl::Vector<Real> Bplasma = LoopField(X, 2.0, 0.1, 1.0);
  const sctl::Vector<Real> B = Bplasma + LoopField(X, 4.0, 1.5, 0.7);

  // The densities n.B and n x B
  sctl::Vector<Real> dX, normal;
  biest::SurfaceOp<Real> surf_op(sctl::Comm::Self(), Nt, Np);
  surf_op.Grad2D(dX, X);
  surf_op.SurfNormalAreaElem(&normal, nullptr, dX, &X);
  sctl::Vector<Real> BdotN(N), NxB(COORD_DIM * N);
  for (sctl::Long i = 0; i < N; i++) {
    const Real b[3] = {B[0 * N + i], B[1 * N + i], B[2 * N + i]};
    const Real n[3] = {normal[0 * N + i], normal[1 * N + i], normal[2 * N + i]};
    BdotN[i] = b[0] * n[0] + b[1] * n[1] + b[2] * n[2];
    NxB[0 * N + i] = n[1] * b[2] - n[2] * b[1];
    NxB[1 * N + i] = n[2] * b[0] - n[0] * b[2];
    NxB[2 * N + i] = n[0] * b[1] - n[1] * b[0];
  }

  // Layer-potential operators: Biot-Savart (3 -> 3 components) and Laplace gradient (1 -> 3 components)
  constexpr int UPSAMPLE = 1; // upsampling factor for the quadrature
  constexpr int PDIM = 20; // radius of the patch for the singular quadrature (PATCH_DIM0)
  constexpr int RDIM = 32; // order of the polar quadrature (RAD_DIM)
  biest::BoundaryIntegralOp<Real, 3, 3, UPSAMPLE, PDIM, RDIM> BiotSavart;
  biest::BoundaryIntegralOp<Real, 1, 3, UPSAMPLE, PDIM, RDIM> LaplaceGrad;
  BiotSavart.SetupSingular(Svec, biest::BiotSavart3D<Real>::FxU());
  LaplaceGrad.SetupSingular(Svec, biest::Laplace3D<Real>::FxdU());

  { // Plasma field on the surface: B/2 + curl S[n x B] - grad S[n.B]
    sctl::Vector<Real> curlS, gradS;
    BiotSavart(curlS, NxB);
    LaplaceGrad(gradS, BdotN);
    const sctl::Vector<Real> Bvc = B * 0.5 + curlS - gradS;

    Real err = 0, max_val = 0;
    for (sctl::Long i = 0; i < COORD_DIM * N; i++) err = std::max<Real>(err, sctl::fabs(Bvc[i] - Bplasma[i]));
    for (sctl::Long i = 0; i < COORD_DIM * N; i++) max_val = std::max<Real>(max_val, sctl::fabs(Bplasma[i]));
    std::cout << "on the surface:  relative error = " << err / max_val << '\n';
  }

  { // Plasma field at points outside: curl S[n x B] - grad S[n.B]
    const sctl::Long Ntrg = 50;
    sctl::Vector<Real> Xt(COORD_DIM * Ntrg); // a circle 0.05 outside the surface
    for (sctl::Long i = 0; i < Ntrg; i++) {
      const Real s = 2 * sctl::const_pi<Real>() * i / Ntrg;
      Xt[0 * Ntrg + i] = 3.05 * sctl::cos<Real>(s);
      Xt[1 * Ntrg + i] = 3.05 * sctl::sin<Real>(s);
      Xt[2 * Ntrg + i] = 0.2;
    }
    sctl::Vector<Real> curlS, gradS;
    BiotSavart.EvalOffSurface(curlS, Xt, NxB, biest::BiotSavart3D<Real>::FxU(), 1e-10);
    LaplaceGrad.EvalOffSurface(gradS, Xt, BdotN, biest::Laplace3D<Real>::FxdU(), 1e-10);
    const sctl::Vector<Real> Bvc = curlS - gradS;
    const sctl::Vector<Real> Bplasma_t = LoopField(Xt, 2.0, 0.1, 1.0);

    Real err = 0, max_val = 0;
    for (sctl::Long i = 0; i < COORD_DIM * Ntrg; i++) err = std::max<Real>(err, sctl::fabs(Bvc[i] - Bplasma_t[i]));
    for (sctl::Long i = 0; i < COORD_DIM * Ntrg; i++) max_val = std::max<Real>(max_val, sctl::fabs(Bplasma_t[i]));
    std::cout << "off the surface: relative error = " << err / max_val << '\n';
  }
  return 0;
}
