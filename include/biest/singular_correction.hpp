// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _SINGULAR_CORRECTION_HPP_
#define _SINGULAR_CORRECTION_HPP_

#include <algorithm>
#include <memory>
#include <vector>

#include <biest/kernel.hpp>
#include <biest/mat.hpp>
#include <sctl.hpp>

namespace biest {

/**
 * The singular quadrature at a target (see Singular quadrature in the documentation): the partition of unity on the
 * patch of the grid within patch_rad grid spacings of the target, and the polar grid on it, of rad_nodes radii and
 * ang_nodes angles. A value of zero is chosen from the tolerance and the anisotropy of the grid (see
 * SingularCorrection::QuadParams()).
 */
struct SingularQuadParams {
  sctl::Integer patch_rad = 0; ///< radius of the patch, in grid spacings: (2 patch_rad + 1)^2 grid points
  sctl::Integer rad_nodes = 0; ///< radial nodes of the polar grid
  sctl::Integer ang_nodes = 0; ///< angular nodes of the polar grid, an even number for principal values
};

/**
 * The singular-quadrature corrections at the targets on one surface: for each target, the weights on the patch of the
 * grid around it that, applied to a density, give the difference between the potential and its trapezoidal rule.
 *
 * @tparam HedgehogOrder 0 or 1 gives the principal value on the surface; a value in [2, 16] the one-sided limit from
 * the side opposite to the normal, by extrapolation from points off the surface (see BoundaryIntegralOp).
 */
template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder = 1> class SingularCorrection {
    static constexpr sctl::Integer COORD_DIM = 3;
    static constexpr sctl::Integer HedgehogOrder_ = (HedgehogOrder > 1 ? HedgehogOrder : 1);
    static constexpr sctl::Integer MAX_INTERP_ORDER = 16; // the order of the Lagrange interpolation, or the patch width when narrower
    // The coarse nodes on the circles of the polar grid (see Precomp): enough for the angular frequencies to pi times the
    // radius in grid spacings, those of data that the grid resolves, and ANG_EXTRA more
    static constexpr sctl::Integer ANG_EXTRA = 6;

    using Vec = sctl::Vec<Real>;
    static constexpr sctl::Integer VecLen = Vec::Size();

    // The vectors of the interpolation between the patch and the polar grid, of order IORD (0: an order known only at
    // run time): the widest of at most VecLen values that divides IORD, so that the rows of a window are whole vectors
    // (4 values with AVX2 or 8 with AVX-512 for order 16), and 1 for IORD = 0.
    static constexpr sctl::Integer InterpVecLen(const sctl::Integer IORD) {
      sctl::Integer n = 1;
      while (IORD && 2 * n <= VecLen && IORD % (2 * n) == 0) n *= 2;
      return n;
    }

    // The precomputed tables of a quadrature, in sctl::Vector for its 64-byte alignment.
    //
    // The patch is the PATCH_DIM x PATCH_DIM grid points around the target, PATCH_DIM = 2 PATCH_RAD + 1, and the polar
    // grid has RAD_DIM radii and ANG_DIM angles. The data on the patch grid (the surface and the density) is
    // interpolated to Mc[i] equispaced angles on circle i of the polar grid, the coarse nodes, enough for data that the
    // grid resolves and an even number, so that they are symmetric under the rotation by pi as the polar nodes are (a
    // surface with stellarator symmetry then gives symmetric potentials); and then to the ANG_DIM angles of the polar
    // grid, which the kernel needs, by trigonometric interpolation: U holds, at U_dsp[i], the ANG_DIM x Mc[i] matrix of circle i, whose coarse nodes are Mc_dsp[i]
    // onwards of the Npc. Mc[i] = ANG_DIM when it would not be fewer: then the coarse nodes are those of the polar grid.
    //
    // The Lagrange weights from the patch grid to coarse node j are the products of 1D ones along the two directions of
    // the patch: W_G2P holds, for each node, the INTERP_ORDER weights a along the first direction and then the
    // INTERP_ORDER weights b along the second, of the window of the patch at I_G2P[j].
    //
    // The corrections of a target are KDIM1 rows of Kp values, the KDIM0 * Ngrid of the patch padded to whole vectors.
    struct Precomp {
      SingularQuadParams params;
      sctl::Integer PATCH_RAD, PATCH_DIM, INTERP_ORDER, RAD_DIM, ANG_DIM, ANGp;
      sctl::Long Ngrid, Npolar, Npc, Kp, WorkBuffSize;
      sctl::Vector<Real> qx, qw;
      sctl::Vector<Real> Gpou_, Ppou_;
      sctl::Vector<sctl::Integer> Mc;
      sctl::Vector<sctl::Long> Mc_dsp, U_dsp;
      sctl::Vector<Real> U;
      sctl::Vector<sctl::Long> I_G2P;
      sctl::Vector<Real> W_G2P;
      sctl::Vector<Real> HedgehogWts_;
      explicit Precomp(const SingularQuadParams& params);
    };

  public:

    /**
     * Constructor.
     *
     * @param comm the communicator: the targets are divided among the processes, and every process holds all the data.
     */
    explicit SingularCorrection(const sctl::Comm& comm = sctl::Comm::Self());

    /**
     * Returns the quadrature for the tolerance tol on a grid of anisotropy cond (see GridAnisotropy()).
     */
    static SingularQuadParams QuadParams(const Real tol, const Real cond);

    /**
     * Returns the anisotropy of a surface grid: the largest, over the grid points, ratio of the singular values of the
     * Jacobian of the surface with respect to the grid indices, for the derivatives dX of the surface on an Nt x Np
     * grid (in the order of SurfaceOp::Grad2D()). With gNt and gNp, that of a gNt x gNp grid of the surface, from dX:
     * it depends only on the ratio gNt / gNp.
     */
    static Real GridAnisotropy(const sctl::Vector<Real>& dX, const sctl::Long Nt, const sctl::Long Np, const Real gNt = 0, const Real gNp = 0);

    /**
     * Set up the corrections at the targets, in the memory of the earlier setup when it is no larger.
     *
     * @param[in] params the quadrature, every value given.
     *
     * @param[in] TRG_SKIP the targets are on the grid of every TRG_SKIP-th point of the surface grid.
     *
     * @param[in] Nt, Np the dimensions of the surface grid.
     *
     * @param[in] Xsrc the surface on the Nt x Np grid.
     *
     * @param[in] dXsrc its derivatives (see SurfaceOp::Grad2D()).
     *
     * @param[in] ker the kernel function.
     *
     * @param[in] normal_orient the orientation of the normal, 1 or -1 (see SurfaceOp::SurfNormalAreaElem()).
     *
     * @param[in] trg_idx the targets: indices of points of the (Nt/TRG_SKIP) x (Np/TRG_SKIP) grid.
     */
    void Setup(const SingularQuadParams& params, sctl::Integer TRG_SKIP, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& dXsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, Real normal_orient, const sctl::Vector<sctl::Long>& trg_idx);

    /**
     * Add the corrections for the density F on the Nt x Np grid to the potential U at the targets, in SoA order.
     * Several densities can be stacked in F; their potentials are stacked in U. If U does not have the right size, it
     * is resized and set to zero first.
     */
    void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

  private:

    // The corrections of local target i, for the interpolation order IORD (0 for the order of the tables below
    // MAX_INTERP_ORDER)
    template <sctl::Integer IORD> void SetupTarget(sctl::Long i, const sctl::Vector<Real>& SrcCoord, const sctl::Vector<Real>& SrcGrad, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, Real normal_orient, sctl::Vector<Real>& work_buffer);

    static void SetPatch(sctl::Vector<Real>& out, sctl::Long t0, sctl::Long p0, const sctl::Vector<Real>& in, sctl::Long Nt, sctl::Long Np, sctl::Integer PATCH_DIM);

    sctl::Comm comm_;
    // The tables of the last TABLES_KEPT quadratures, the most recent first: those of the setup, and those of the
    // earlier ones, which a later setup with the same quadrature uses again instead of building them
    static constexpr sctl::Integer TABLES_KEPT = 4;
    std::vector<std::shared_ptr<const Precomp>> tables_;
    sctl::Long Nt_, Np_, Ntrg_, trg_dsp_; // the grid, and the targets of all the processes, of which the local ones are trg_dsp_ onwards
    sctl::Vector<sctl::Long> trg_t_, trg_p_; // the grid point of each local target
    sctl::Vector<Real> MGrid_; // the corrections of each local target (see Precomp)
};

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::Precomp::Precomp(const SingularQuadParams& params_) : params(params_) {
  SCTL_ASSERT(params.patch_rad > 0 && params.rad_nodes > 0 && params.ang_nodes > 0);
  PATCH_RAD = params.patch_rad;
  PATCH_DIM = 2 * PATCH_RAD + 1;
  INTERP_ORDER = std::min<sctl::Integer>(PATCH_DIM, MAX_INTERP_ORDER);
  RAD_DIM = params.rad_nodes;
  ANG_DIM = params.ang_nodes;
  ANGp = (ANG_DIM + VecLen - 1) / VecLen * VecLen; // the columns of U, padded to whole vectors
  Ngrid = PATCH_DIM * PATCH_DIM;
  Npolar = RAD_DIM * ANG_DIM;
  Kp = (KDIM0 * Ngrid + VecLen - 1) / VecLen * VecLen;
  WorkBuffSize = (Ngrid+VecLen+Npolar)*(COORD_DIM*4+1) + KDIM0*Npolar*KDIM1*HedgehogOrder_ + KDIM0*Ngrid*KDIM1;

  { // Set qx, qw
    sctl::Integer order = RAD_DIM;
    sctl::Vector<Real> qx_(order);
    sctl::Vector<Real> qw_(order);
    if (1) { // Set qx_, qw_
      sctl::ChebBasis<Real>::quad_rule(order, qx_, qw_);

      if (HedgehogOrder > 1) {
        for (long i = 0; i < qx_.Dim(); i++) {
          qw_[i] *= (Real)2 * qx_[i];
          qx_[i] = qx_[i] * qx_[i];
        }
      }
    } else { // Trapezoidal rule (does not work for Helmholtz imaginary part)
      for (sctl::Long i = 0; i < order; i++) {
        qx_[i] = (2 * i + 1) / (Real)(2 * order);
        qw_[i] = 1 / (Real)order;
      }
    }
    qx.Swap(qx_);
    qw.Swap(qw_);
  }
  { // Set Gpou, Ppou
    auto pou = [&](Real r) {
      Real a = 0;
      Real b = 1;
      //return r < a ? 1 : (r >= b ? 0 : sctl::exp<Real>(1 - 1 / (1 - ((r-a)/(b-a)) * ((r-a)/(b-a)))) );
      //return r >= b ? 0 : (r < a ? 1 : sctl::exp<Real>(2*sctl::exp<Real>(-(b-a)/(r-a)) / ((r-a)/(b-a)-1))); // BRUNO AND KUNYANSKY
      if (PATCH_DIM > 45) {
        return r < a ? 1 : sctl::exp<Real>(-(Real)36 * sctl::pow<10,Real>((r-a)/(b-a)));
      } else if (PATCH_DIM > 20) {
        return r < a ? 1 : sctl::exp<Real>(-(Real)36 * sctl::pow< 8,Real>((r-a)/(b-a)));
      } else {
        return r < a ? 1 : sctl::exp<Real>(-(Real)36 * sctl::pow< 6,Real>((r-a)/(b-a)));
      }
    };

    sctl::Vector<Real> Gpou(Ngrid);
    Real h = 1 / (Real)PATCH_RAD;
    for (sctl::Integer i = 0; i < PATCH_DIM; i++){
      for (sctl::Integer j = 0; j < PATCH_DIM; j++){
        Real dr[2] = {(i - PATCH_RAD) * h, (j - PATCH_RAD) * h};
        Real r = sctl::sqrt<Real>(dr[0] * dr[0] + dr[1] * dr[1]);
        Gpou[i * PATCH_DIM + j] = -pou(r);
      }
    }

    sctl::Vector<Real> Ppou(Npolar);
    Real dt = 2 * sctl::const_pi<Real>() / ANG_DIM;
    for (sctl::Integer i = 0; i < RAD_DIM; i++){
      for (sctl::Integer j = 0; j < ANG_DIM; j++){
        Real dr = qw[i] * PATCH_RAD;
        Real rdt = qx[i] * PATCH_RAD * dt;
        Ppou[i * ANG_DIM + j] = pou(qx[i]) * dr * rdt;
      }
    }
    Gpou_.Swap(Gpou);
    Ppou_.Swap(Ppou);
    if (0) {
      // this does not give accurate error estimate for trapezoidal rule
      // since |r| is not smooth.
      Real sum0=0, sum1=0;
      for (auto a : Ppou_) sum0 += a;
      for (auto a : Gpou_) sum1 += a;
      std::cout<<"POU error: "<<(double)fabs((sum0+sum1)/(fabs(sum0)+fabs(sum1)))<<'\n';
    }
  }
  { // Set Mc, Mc_dsp, Npc, U, U_dsp: on a circle of radius rho grid spacings, data that the grid resolves has angular
    // frequencies to about pi * rho
    Mc.ReInit(RAD_DIM);
    Mc_dsp.ReInit(RAD_DIM);
    U_dsp.ReInit(RAD_DIM);
    Npc = 0;
    sctl::Long U_len = 0;
    for (sctl::Integer i = 0; i < RAD_DIM; i++) {
      const sctl::Integer m = (sctl::Integer)sctl::ceil<Real>(sctl::const_pi<Real>() * qx[i] * PATCH_RAD) + ANG_EXTRA;
      Mc[i] = (2 * m + 2 < ANG_DIM ? 2 * m + 2 : ANG_DIM);
      Mc_dsp[i] = Npc;
      U_dsp[i] = U_len;
      Npc += Mc[i];
      if (Mc[i] < ANG_DIM) U_len += ANGp * Mc[i];
    }
    U.ReInit(U_len);
    U.SetZero();
    for (sctl::Integer i = 0; i < RAD_DIM; i++) { // U[l][j] = D(2 pi l / ANG_DIM - 2 pi j / M), with D(x) = sin(M x / 2) / (M tan(x / 2)) the interpolant of the M (even) angles; by columns, ANGp apart
      const sctl::Integer M = Mc[i];
      if (M == ANG_DIM) continue;
      for (sctl::Integer j = 0; j < M; j++) {
        for (sctl::Integer l = 0; l < ANG_DIM; l++) {
          const Real x = 2 * sctl::const_pi<Real>() * (l / (Real)ANG_DIM - j / (Real)M);
          const Real s = sctl::sin<Real>(x / 2);
          U[U_dsp[i] + j * ANGp + l] = (sctl::fabs<Real>(s) < 1e-3 * sctl::machine_eps<Real>() ? (Real)1 : sctl::sin<Real>(M * x / 2) * sctl::cos<Real>(x / 2) / (M * s));
        }
      }
    }
  }
  { // Set I_G2P, W_G2P
    sctl::Vector<sctl::Long> I_G2P_(Npc);
    sctl::Vector<Real> W_G2P_(Npc * 2 * INTERP_ORDER);

    Real h = 1 / (Real)(INTERP_ORDER - 1);
    auto lagrange_interp = [&](Real x, sctl::Integer i) { // the weight of node i at x \in [0,1]
      Real p=1;
      Real z = i * h;
      for (sctl::Integer j = 0; j < INTERP_ORDER; j++) {
        if (j != i) {
          Real y = j * h;
          p *= (x - y) / (z - y);
        }
      }
      return p;
    };
    for (sctl::Integer i0 = 0; i0 < RAD_DIM; i0++) {
      const Real h_ang = 2 * sctl::const_pi<Real>() / Mc[i0];
      for (sctl::Integer i1 = 0; i1 < Mc[i0]; i1++) {
        Real x0 = (Real)0.5 + (Real)0.5 * qx[i0] * sctl::cos<Real>(h_ang * i1);
        Real x1 = (Real)0.5 + (Real)0.5 * qx[i0] * sctl::sin<Real>(h_ang * i1);

        sctl::Integer y0 = std::max<sctl::Integer>(0, std::min<sctl::Integer>((sctl::Integer)(x0 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2), PATCH_DIM - INTERP_ORDER));
        sctl::Integer y1 = std::max<sctl::Integer>(0, std::min<sctl::Integer>((sctl::Integer)(x1 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2), PATCH_DIM - INTERP_ORDER));

        Real z0 = (x0 * (PATCH_DIM - 1) - y0) * h;
        Real z1 = (x1 * (PATCH_DIM - 1) - y1) * h;

        const sctl::Long j0 = Mc_dsp[i0] + i1;
        I_G2P_[j0] = y0 * PATCH_DIM + y1;
        for (sctl::Integer j = 0; j < INTERP_ORDER; j++) {
          W_G2P_[(j0 * 2 + 0) * INTERP_ORDER + j] = lagrange_interp(z0, j);
          W_G2P_[(j0 * 2 + 1) * INTERP_ORDER + j] = lagrange_interp(z1, j);
        }
      }
    }
    I_G2P.Swap(I_G2P_);
    W_G2P.Swap(W_G2P_);
  }
  { // Set HedgehogWts_
    sctl::Vector<Real> HedgehogWts(HedgehogOrder_);
    if (HedgehogOrder_ > 1) {
      constexpr Real interp_nds[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
      for (sctl::Integer k = 0; k < HedgehogOrder_; k++) {
        Real Pn = 1, Pd = 1;
        for (sctl::Integer i = 0; i < HedgehogOrder_; i++) {
          if (i != k) {
            Pn *= (interp_nds[i]);
            Pd *= (interp_nds[i]-interp_nds[k]);
          }
        }
        HedgehogWts[k] = Pn/Pd;
      }
    } else {
      HedgehogWts[0] = (Real)1;
    }
    HedgehogWts_.Swap(HedgehogWts);
  }
}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::SingularCorrection(const sctl::Comm& comm) : comm_(comm), Nt_(0), Np_(0), Ntrg_(0), trg_dsp_(0) {}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> SingularQuadParams SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::QuadParams(const Real tol, const Real cond) {
  Real digits = std::max<Real>(1, -sctl::log<Real>(tol) / sctl::log<Real>(10));
  if (sctl::fabs<Real>(digits - sctl::round<Real>(digits)) < (Real)1e-6) digits = sctl::round<Real>(digits);
  // Fit to the error of Green's identity for the Laplace kernels on tori and rotating ellipses of anisotropy 1.3 to 4.4
  // (see the Singular quadrature section of the documentation): the patch radius grows with the anisotropy, for the
  // trapezoidal rule outside the patch; the radial nodes need only the digits; and the angles are those of the kernel,
  // whose angular profile on a grid of anisotropy c is analytic in a strip of half-width ln((c+1)/(c-1)) / 2, an even
  // number of them so that the rule is symmetric under the rotation by pi (for principal values). The hedgehog, fit
  // to the Hessian of the single layer in the test of virtual-casing, needs a larger patch and more angles, and its
  // radial nodes (3 times more, squared) set the distance of its points from the surface.
  const Real c = std::max<Real>(cond, (Real)1.2);
  SingularQuadParams params;
  params.patch_rad = (sctl::Integer)sctl::ceil<Real>(digits * ((Real)1.1 + (Real)0.5 * c));
  params.rad_nodes = (sctl::Integer)sctl::ceil<Real>((Real)2.6 * digits) + 2;
  params.ang_nodes = 2 * ((sctl::Integer)sctl::ceil<Real>(digits / sctl::log<Real>((c + 1) / (c - 1)) * sctl::log<Real>(10)) + 4);
  if (HedgehogOrder > 1) {
    params.patch_rad = (sctl::Integer)sctl::ceil<Real>((Real)1.3 * params.patch_rad);
    params.rad_nodes = 3 * (sctl::Integer)sctl::ceil<Real>((Real)1.6 * params.patch_rad);
    params.ang_nodes = 2 * (sctl::Integer)sctl::ceil<Real>((Real)0.7 * params.ang_nodes);
  }
  return params;
}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> Real SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::GridAnisotropy(const sctl::Vector<Real>& dX, const sctl::Long Nt, const sctl::Long Np, const Real gNt_, const Real gNp_) {
  const sctl::Long N = Nt * Np;
  const Real gNt = (gNt_ > 0 ? gNt_ : (Real)Nt), gNp = (gNp_ > 0 ? gNp_ : (Real)Np);
  SCTL_ASSERT(dX.Dim() == COORD_DIM * 2 * N);
  return sctl::omp_par::parallel_reduce(N, (Real)1, [&](const sctl::Long i) { // the metric per grid spacing, [a c; c b], and the square root of the ratio of its eigenvalues (at least 1)
    Real a = 0, b = 0, c = 0;
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      const Real xt = dX[(k * 2 + 0) * N + i] / gNt, xp = dX[(k * 2 + 1) * N + i] / gNp;
      a += xt * xt;
      b += xp * xp;
      c += xt * xp;
    }
    const Real m = (a + b) / 2, d = sctl::sqrt<Real>((a - b) * (a - b) / 4 + c * c);
    return sctl::sqrt<Real>((m + d) / (m - d));
  }, [](const Real x, const Real y) { return std::max<Real>(x, y); });
}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::Setup(const SingularQuadParams& params, sctl::Integer TRG_SKIP, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& dXsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, Real normal_orient, const sctl::Vector<sctl::Long>& trg_idx) {
  { // The tables of the quadrature, first in tables_: those kept if any, else new ones in place of the oldest
    const auto same = [&params](const std::shared_ptr<const Precomp>& t) { return t->params.patch_rad == params.patch_rad && t->params.rad_nodes == params.rad_nodes && t->params.ang_nodes == params.ang_nodes; };
    auto it = std::find_if(tables_.begin(), tables_.end(), same);
    if (it == tables_.end()) {
      if (tables_.capacity() < (size_t)TABLES_KEPT) tables_.reserve(TABLES_KEPT);
      if (tables_.size() < (size_t)TABLES_KEPT) tables_.emplace_back();
      it = tables_.end() - 1;
      *it = nullptr; // the oldest freed before the new ones are built
      *it = std::make_shared<const Precomp>(params);
    }
    std::rotate(tables_.begin(), it, it + 1);
  }
  const Precomp& T = *tables_[0];
  SCTL_ASSERT(Nt >= T.PATCH_DIM && Np >= T.PATCH_DIM);
  assert(KDIM0 == ker.Dim(0));
  assert(KDIM1 == ker.Dim(1));

  // The targets of this process: trg_dsp_ onwards of the Ntrg_
  Nt_ = Nt;
  Np_ = Np;
  Ntrg_ = trg_idx.Dim();
  const sctl::Long np = comm_.Size(), rank = comm_.Rank();
  const sctl::Long a = rank * Ntrg_ / np, b = (rank + 1) * Ntrg_ / np, Nloc = b - a;
  trg_dsp_ = a;
  trg_t_.ReInit(Nloc);
  trg_p_.ReInit(Nloc);
  const sctl::Long Np0 = Np / TRG_SKIP;
  for (sctl::Long i = 0; i < Nloc; i++) {
    trg_t_[i] = (trg_idx[a + i] / Np0) * TRG_SKIP;
    trg_p_[i] = (trg_idx[a + i] % Np0) * TRG_SKIP;
  }
  MGrid_.ReInit(Nloc * KDIM1 * T.Kp);

  const sctl::Integer Nthread = omp_get_max_threads();
  sctl::omp_par::parallel_for(Nthread, [&](const sctl::Long tid) {
    // The work buffer of the thread, in its scratch memory, which the next setup uses again (see SurfaceOp::SurfGrad)
    sctl::ScratchBuf<Real> work_buff_buf(T.WorkBuffSize);
    sctl::Vector<Real> work_buff(work_buff_buf, false);
    const sctl::Long i0 = Nloc * (tid + 0) / Nthread;
    const sctl::Long i1 = Nloc * (tid + 1) / Nthread;
    for (sctl::Long i = i0; i < i1; i++) {
      if (T.INTERP_ORDER == MAX_INTERP_ORDER) SetupTarget<MAX_INTERP_ORDER>(i, Xsrc, dXsrc, ker, normal_orient, work_buff);
      else SetupTarget<0>(i, Xsrc, dXsrc, ker, normal_orient, work_buff);
    }
  });
}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> template <sctl::Integer IORD> void SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::SetupTarget(sctl::Long trg, const sctl::Vector<Real>& SrcCoord, const sctl::Vector<Real>& SrcGrad, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, Real normal_orient, sctl::Vector<Real>& work_buff) {
  const sctl::Long Nt = Nt_, Np = Np_, t = trg_t_[trg], p = trg_p_[trg];

  const Precomp& tables = *tables_[0];
  const sctl::Integer PATCH_DIM = tables.PATCH_DIM;
  const sctl::Integer RAD_DIM = tables.RAD_DIM;
  const sctl::Integer ANG_DIM = tables.ANG_DIM;
  const sctl::Integer ANGp = tables.ANGp;
  const sctl::Long Ngrid = tables.Ngrid;
  const sctl::Long Npolar = tables.Npolar;
  const sctl::Vector<Real>& Gpou_ = tables.Gpou_;
  const sctl::Vector<Real>& Ppou_ = tables.Ppou_;
  const sctl::Vector<sctl::Long>& I_G2P = tables.I_G2P;
  const sctl::Vector<Real>& W_G2P = tables.W_G2P;
  const sctl::Vector<Real>& HedgehogWts_ = tables.HedgehogWts_;
  const sctl::Vector<sctl::Integer>& Mc = tables.Mc;
  const sctl::Vector<sctl::Long>& Mc_dsp = tables.Mc_dsp;
  const sctl::Vector<sctl::Long>& U_dsp = tables.U_dsp;
  const sctl::Vector<Real>& U = tables.U;
  const sctl::Long Npc = tables.Npc;
  const bool coarse = (Npc < Npolar); // whether there are fewer coarse nodes than polar ones

  // The order of the interpolation (a constant for IORD > 0) and its vectors (see InterpVecLen)
  const sctl::Integer INTERP_ORDER = (IORD ? IORD : tables.INTERP_ORDER);
  constexpr sctl::Integer IVecLen = InterpVecLen(IORD);
  constexpr sctl::Integer INTERP_ORDER_MAX = (IORD ? IORD : MAX_INTERP_ORDER); // for the sizes of arrays
  using IVec = sctl::Vec<Real, IVecLen>;

  Real invNt = 1 / (Real)Nt;
  Real invNp = 1 / (Real)Np;

  SCTL_ASSERT(work_buff.Dim() >= tables.WorkBuffSize);

  sctl::Vector<Real> G (    COORD_DIM * Ngrid, work_buff.begin() + (Ngrid+VecLen)*COORD_DIM*0, false);
  sctl::Vector<Real> Gg(2 * COORD_DIM * Ngrid, work_buff.begin() + (Ngrid+VecLen)*COORD_DIM*1, false);
  sctl::Vector<Real> Gn(    COORD_DIM * Ngrid, work_buff.begin() + (Ngrid+VecLen)*COORD_DIM*3, false);
  sctl::Vector<Real> Ga(                Ngrid, work_buff.begin() + (Ngrid+VecLen)*COORD_DIM*4, false);

  sctl::Vector<Real> P (    COORD_DIM * Npolar, work_buff.begin() + (Ngrid+VecLen)*(COORD_DIM*4+1) + Npolar*COORD_DIM*0, false);
  sctl::Vector<Real> Pg(2 * COORD_DIM * Npolar, work_buff.begin() + (Ngrid+VecLen)*(COORD_DIM*4+1) + Npolar*COORD_DIM*1, false);
  sctl::Vector<Real> Pn(    COORD_DIM * Npolar, work_buff.begin() + (Ngrid+VecLen)*(COORD_DIM*4+1) + Npolar*COORD_DIM*3, false);
  sctl::Vector<Real> Pa(                Npolar, work_buff.begin() + (Ngrid+VecLen)*(COORD_DIM*4+1) + Npolar*COORD_DIM*4, false);

  // The corrections of the target, KDIM0 * Ngrid rows of KDIM1, after the polar values (MPolar)
  sctl::Matrix<Real> MGrid(KDIM0 * Ngrid, KDIM1, work_buff.begin() + (Ngrid+VecLen+Npolar)*(COORD_DIM*4+1) + KDIM0*Npolar*KDIM1*HedgehogOrder_, false);

  sctl::StaticArray<Real, COORD_DIM> TrgCoord_;
  sctl::Vector<Real> TrgCoord(COORD_DIM, TrgCoord_, false);
  for (sctl::Integer k = 0; k < COORD_DIM; k++) { // Set TrgCoord
    TrgCoord[k] = SrcCoord[(k * Nt + t) * Np + p];
  }

  sctl::StaticArray<Real, COORD_DIM*HedgehogOrder_> TrgCoordPolar_;
  sctl::Vector<Real> TrgCoordPolar;
  if (HedgehogOrder_ > 1) {
    constexpr Real interp_nds[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    Real n[COORD_DIM];
    const sctl::Long N = Nt * Np;
    const sctl::Long offset = t * Np + p;
    n[0] = SrcGrad[2 * N + offset] * SrcGrad[5 * N + offset] - SrcGrad[3 * N + offset] * SrcGrad[4 * N + offset];
    n[1] = SrcGrad[4 * N + offset] * SrcGrad[1 * N + offset] - SrcGrad[5 * N + offset] * SrcGrad[0 * N + offset];
    n[2] = SrcGrad[0 * N + offset] * SrcGrad[3 * N + offset] - SrcGrad[1 * N + offset] * SrcGrad[2 * N + offset];
    Real r = sctl::sqrt<Real>(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

    Real scal = sctl::sqrt<Real>(r * invNt * invNp) * normal_orient / r * (-20.0/RAD_DIM);
    for (sctl::Integer k = 0; k < COORD_DIM; k++) n[k] *= scal;

    TrgCoordPolar.ReInit(COORD_DIM*HedgehogOrder_, TrgCoordPolar_, false);
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Integer i = 0; i < HedgehogOrder_; i++) {
        TrgCoordPolar[k*HedgehogOrder_+i] = TrgCoord[k] + interp_nds[i]*n[k];
      }
    }
  } else {
    TrgCoordPolar.ReInit(COORD_DIM, TrgCoord.begin(), false);
  }

  SetPatch(G , t, p, SrcCoord     , Nt, Np, PATCH_DIM);
  SetPatch(Gg, t, p, SrcGrad      , Nt, Np, PATCH_DIM);
  for (sctl::Integer k = 0; k < COORD_DIM; k++) { // the coordinates relative to the target, so that the polar points near it lose no digits
    const Real x0 = TrgCoord[k];
    for (sctl::Long i = 0; i < Ngrid; i++) G[k * Ngrid + i] -= x0;
    if (HedgehogOrder_ > 1) {
      for (sctl::Integer i = 0; i < HedgehogOrder_; i++) TrgCoordPolar[k * HedgehogOrder_ + i] -= x0;
    }
    TrgCoord[k] = 0; // and TrgCoordPolar without the hedgehog, which is a view of it
  }

  for (sctl::Long i = 0; i < Ngrid; i++) {
    Real n0 = Gg[2 * Ngrid + i] * Gg[5 * Ngrid + i] - Gg[3 * Ngrid + i] * Gg[4 * Ngrid + i];
    Real n1 = Gg[4 * Ngrid + i] * Gg[1 * Ngrid + i] - Gg[5 * Ngrid + i] * Gg[0 * Ngrid + i];
    Real n2 = Gg[0 * Ngrid + i] * Gg[3 * Ngrid + i] - Gg[1 * Ngrid + i] * Gg[2 * Ngrid + i];
    Real r = sctl::sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
    Real inv_r = 1 / r;
    Gn[0 * Ngrid + i] = n0 * inv_r * normal_orient;
    Gn[1 * Ngrid + i] = n1 * inv_r * normal_orient;
    Gn[2 * Ngrid + i] = n2 * inv_r * normal_orient;
    Ga[i] = r * invNt * invNp;
    Gg[0 * Ngrid + i] *= invNt;
    Gg[2 * Ngrid + i] *= invNt;
    Gg[4 * Ngrid + i] *= invNt;
    Gg[1 * Ngrid + i] *= invNp;
    Gg[3 * Ngrid + i] *= invNp;
    Gg[5 * Ngrid + i] *= invNp;
  }

  { // Lagrange interpolation to the coarse nodes, then trigonometric interpolation to the polar grid
    // To coarse node j, the value of a component is a^T W b, for the INTERP_ORDER x INTERP_ORDER window W of its patch
    // values at I_G2P[j] and the 1D weights a, b of W_G2P: the rows of W scaled by a and summed, in vectors, then the
    // dot with b. COORD_DIM components at a time, Ngrid apart in Vin and Npc apart in Vout.
    constexpr sctl::Integer NV_MAX = INTERP_ORDER_MAX / IVecLen;
    const sctl::Integer NV = INTERP_ORDER / IVecLen; // vectors in a row of a window
    auto interp = [&](const sctl::Iterator<Real> Vout, const sctl::ConstIterator<Real> Vin) {
      for (sctl::Long j = 0; j < Npc; j++) {
        const sctl::ConstIterator<Real> a = W_G2P.begin() + j * 2 * INTERP_ORDER, b = a + INTERP_ORDER;
        const sctl::ConstIterator<Real> W = Vin + I_G2P[j];
        IVec V[COORD_DIM][NV_MAX]; // all of them set, also those past NV for a run-time order, for -Wmaybe-uninitialized
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          for (sctl::Integer v = 0; v < NV_MAX; v++) V[k][v] = IVec::Zero();
        }
        for (sctl::Integer i0 = 0; i0 < INTERP_ORDER; i0++) {
          const IVec ai = IVec::Load1(&a[i0]);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            for (sctl::Integer v = 0; v < NV; v++) V[k][v] = FMA(ai, IVec::Load(&W[k * Ngrid + i0 * PATCH_DIM + v * IVecLen]), V[k][v]);
          }
        }
        IVec bv[NV_MAX];
        for (sctl::Integer v = 0; v < NV_MAX; v++) bv[v] = (v < NV ? IVec::Load(&b[v * IVecLen]) : IVec::Zero());
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          IVec acc = V[k][0] * bv[0];
          for (sctl::Integer v = 1; v < NV; v++) acc = FMA(V[k][v], bv[v], acc);
          Vout[k * Npc + j] = sctl::reduce(acc);
        }
      }
    };
    if (!coarse) {
      interp(P.begin(), G.begin());
      interp(Pg.begin(), Gg.begin());
      interp(Pg.begin() + COORD_DIM * Npolar, Gg.begin() + COORD_DIM * Ngrid);
    } else {
      // The NC = 3*COORD_DIM components on the coarse nodes, then on the polar grid: on circle i, U of the circle times
      // its Mc[i] values (the same values when Mc[i] = ANG_DIM), VecLen rows of U at a time, its columns scaled by the
      // values and summed
      constexpr sctl::Integer NC = 3 * COORD_DIM;
      sctl::ScratchBuf<Real> Vc_buf(NC * Npc);
      interp(Vc_buf.begin() + COORD_DIM * Npc * 0, G.begin());
      interp(Vc_buf.begin() + COORD_DIM * Npc * 1, Gg.begin());
      interp(Vc_buf.begin() + COORD_DIM * Npc * 2, Gg.begin() + COORD_DIM * Ngrid);
      const auto Vf = [&P,&Pg,Npolar](const sctl::Integer k) { return (k < COORD_DIM ? P.begin() + k * Npolar : Pg.begin() + (k - COORD_DIM) * Npolar); };
      for (sctl::Integer i = 0; i < RAD_DIM; i++) {
        const sctl::Integer M = Mc[i];
        const sctl::ConstIterator<Real> vc = Vc_buf.begin() + Mc_dsp[i];
        if (M == ANG_DIM) {
          for (sctl::Integer k = 0; k < NC; k++) {
            for (sctl::Integer l = 0; l < ANG_DIM; l++) Vf(k)[i * ANG_DIM + l] = vc[k * Npc + l];
          }
          continue;
        }
        const sctl::ConstIterator<Real> Ui = U.begin() + U_dsp[i];
        for (sctl::Integer l0 = 0; l0 < ANG_DIM; l0 += VecLen) {
          Vec acc[NC];
          for (sctl::Integer k = 0; k < NC; k++) acc[k] = Vec::Zero();
          for (sctl::Integer j = 0; j < M; j++) {
            const Vec u = Vec::LoadAligned(&Ui[j * ANGp + l0]);
            for (sctl::Integer k = 0; k < NC; k++) acc[k] = FMA(u, Vec::Load1(&vc[k * Npc + j]), acc[k]);
          }
          for (sctl::Integer k = 0; k < NC; k++) {
            const sctl::Iterator<Real> vf = Vf(k) + i * ANG_DIM + l0;
            if (l0 + VecLen <= ANG_DIM) {
              acc[k].Store(&vf[0]);
            } else {
              alignas(sizeof(Vec)) Real tmp[VecLen];
              acc[k].StoreAligned(tmp);
              for (sctl::Integer l = 0; l < ANG_DIM - l0; l++) vf[l] = tmp[l];
            }
          }
        }
      }
    }
    for (sctl::Long i = 0; i < Npolar; i++) { // Compute Pn, Pa
       Real n0 = Pg[2 * Npolar + i] * Pg[5 * Npolar + i] - Pg[3 * Npolar + i] * Pg[4 * Npolar + i];
       Real n1 = Pg[4 * Npolar + i] * Pg[1 * Npolar + i] - Pg[5 * Npolar + i] * Pg[0 * Npolar + i];
       Real n2 = Pg[0 * Npolar + i] * Pg[3 * Npolar + i] - Pg[1 * Npolar + i] * Pg[2 * Npolar + i];
       Real r = sctl::sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
       n0 /= r;
       n1 /= r;
       n2 /= r;
       Pn[0 * Npolar + i] = n0 * normal_orient;
       Pn[1 * Npolar + i] = n1 * normal_orient;
       Pn[2 * Npolar + i] = n2 * normal_orient;
       Pa[i] = r;
    }
  }

  { // Subtract singular part from U
    { // Subtract singular part from U
      #ifndef DISABLE_FAR_FIELD
      ker.BuildMatrix(G, Gn, TrgCoord, MGrid);
      for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) { // apply weights (Ga * Gpou)
        for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
          for (sctl::Long i = 0; i < Ngrid; i++) {
            MGrid[k0 * Ngrid + i][k1] *= Ga[i] * Gpou_[i];
          }
        }
      }
      #else
      MGrid.SetZero();
      #endif
    }
    { // Add singular part to U
      sctl::Matrix<Real> MPolar(KDIM0 * Npolar, KDIM1 * HedgehogOrder_, work_buff.begin() + (Ngrid+VecLen+Npolar)*(COORD_DIM*4+1), false);

      ker.BuildMatrix(P, Pn, TrgCoordPolar, MPolar); // MPolar <-- ker(P, Pn, TrgCoordPolar)
      for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) { // MPolar <-- Mpolar * Pa * Ppou
        for (sctl::Integer j = 0; j < KDIM1*HedgehogOrder_; j++) {
          for (sctl::Long i = 0; i < Npolar; i++) {
            MPolar[k0 * Npolar + i][j] *= Pa[i] * Ppou_[i];
          }
        }
      }
      // The KDIM0*KDIM1 values of each polar point (its hedgehog values combined), Vf, and then those of the coarse
      // nodes, Vc: the transpose of the trigonometric interpolation, on circle i U^T times its ANG_DIM values
      // (Vf padded with zeros, which the vectors of the last circle read past its end)
      sctl::ScratchBuf<Real> V_buf(KDIM0 * KDIM1 * Npolar + (coarse ? VecLen + KDIM0 * KDIM1 * Npc : 0));
      const sctl::Iterator<Real> Vf = V_buf.begin(), Vc = (coarse ? Vf + KDIM0 * KDIM1 * Npolar + VecLen : Vf);
      if (coarse) {
        for (sctl::Integer l = 0; l < VecLen; l++) Vf[KDIM0 * KDIM1 * Npolar + l] = 0;
      }
      for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
        for (sctl::Long j = 0; j < Npolar; j++) {
          const sctl::ConstIterator<Real> MPolar_ = MPolar[k0 * Npolar + j];
          for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
            Real v = 0;
            for (sctl::Integer l = 0; l < HedgehogOrder_; l++) v += MPolar_[k1 * HedgehogOrder_ + l] * HedgehogWts_[l];
            Vf[(k0 * KDIM1 + k1) * Npolar + j] = v;
          }
        }
      }
      if (coarse) { // value j of circle i: the dot of column j of U with the ANG_DIM values of the circle, in vectors
        constexpr sctl::Integer NC = KDIM0 * KDIM1;
        for (sctl::Integer i = 0; i < RAD_DIM; i++) {
          const sctl::Integer M = Mc[i];
          const sctl::ConstIterator<Real> vf = Vf + i * ANG_DIM;
          const sctl::Iterator<Real> vc = Vc + Mc_dsp[i];
          if (M == ANG_DIM) {
            for (sctl::Integer k = 0; k < NC; k++) {
              for (sctl::Integer l = 0; l < ANG_DIM; l++) vc[k * Npc + l] = vf[k * Npolar + l];
            }
            continue;
          }
          const sctl::ConstIterator<Real> Ui = U.begin() + U_dsp[i];
          for (sctl::Integer j = 0; j < M; j++) {
            Vec acc[NC];
            for (sctl::Integer k = 0; k < NC; k++) acc[k] = Vec::Zero();
            for (sctl::Integer l0 = 0; l0 < ANG_DIM; l0 += VecLen) {
              const Vec u = Vec::LoadAligned(&Ui[j * ANGp + l0]);
              for (sctl::Integer k = 0; k < NC; k++) acc[k] = FMA(u, Vec::Load(&vf[k * Npolar + l0]), acc[k]);
            }
            for (sctl::Integer k = 0; k < NC; k++) vc[k * Npc + j] = sctl::reduce(acc[k]);
          }
        }
      }

      // MGrid <-- the transpose of the Lagrange interpolation times Vc: for a source component, with v the KDIM1 values
      // of coarse node j, the window of MGrid at I_G2P[j] gets a b^T v; row i0 of the window, its INTERP_ORDER * KDIM1
      // values contiguous, gets a[i0] times u = b (x) v.
      constexpr sctl::Integer NU_MAX = INTERP_ORDER_MAX * KDIM1 / IVecLen;
      const sctl::Integer NU = INTERP_ORDER * KDIM1 / IVecLen; // vectors in a row of a window
      for (sctl::Long j = 0; j < Npc; j++) {
        const sctl::ConstIterator<Real> a = W_G2P.begin() + j * 2 * INTERP_ORDER, b = a + INTERP_ORDER;
        for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
          alignas(sizeof(IVec)) Real u[INTERP_ORDER_MAX * KDIM1];
          for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
            const Real v = Vc[(k0 * KDIM1 + k1) * Npc + j];
            for (sctl::Integer i1 = 0; i1 < INTERP_ORDER; i1++) u[i1 * KDIM1 + k1] = b[i1] * v;
          }
          IVec uv[NU_MAX];
          for (sctl::Integer m = 0; m < NU; m++) uv[m] = IVec::LoadAligned(u + m * IVecLen);
          const sctl::Iterator<Real> R0 = MGrid[k0 * Ngrid + I_G2P[j]];
          for (sctl::Integer i0 = 0; i0 < INTERP_ORDER; i0++) {
            const IVec ai = IVec::Load1(&a[i0]);
            const sctl::Iterator<Real> R = R0 + i0 * PATCH_DIM * KDIM1;
            for (sctl::Integer m = 0; m < NU; m++) FMA(ai, uv[m], IVec::Load(&R[m * IVecLen])).Store(&R[m * IVecLen]);
          }
        }
      }
    }
  }

  { // The corrections of the target, by rows of KDIM1 (see Precomp), padded with zeros
    const sctl::Long N = KDIM0 * Ngrid, Kp = tables.Kp;
    const sctl::Iterator<Real> M = MGrid_.begin() + trg * KDIM1 * Kp;
    for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
      for (sctl::Long i = 0; i < N; i++) M[k1 * Kp + i] = MGrid[i][k1];
      for (sctl::Long i = N; i < Kp; i++) M[k1 * Kp + i] = 0;
    }
  }
}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
  const sctl::Long Nloc = trg_t_.Dim();
  const sctl::Long dof = F.Dim() / (KDIM0 * Nt_ * Np_);
  SCTL_ASSERT(F.Dim() == dof * KDIM0 * Nt_ * Np_);
  sctl::ScratchBuf<Real> Uloc_buf(dof * KDIM1 * Ntrg_), Uglb_buf(dof * KDIM1 * Ntrg_); // see SurfaceOp::EvalSurfInteg
  sctl::Vector<Real> Uloc(Uloc_buf), Uglb(Uglb_buf);
  Uloc = 0;
  if (Nloc) {
    const Precomp& T = *tables_[0];
    const sctl::Long N = KDIM0 * T.Ngrid, Kp = T.Kp;
    const sctl::Integer Nthread = omp_get_max_threads();
    sctl::omp_par::parallel_for(Nthread, [&](const sctl::Long tid) { // the targets i0 to i1
      // The density on the patch of a target, padded with zeros to Kp values; its dot with each row of the corrections
      sctl::ScratchBuf<Real> GF_buf(Kp);
      sctl::Vector<Real> GF(N, GF_buf.begin(), false);
      for (sctl::Long l = N; l < Kp; l++) GF_buf.begin()[l] = 0;
      const sctl::Long i0 = Nloc * (tid + 0) / Nthread, i1 = Nloc * (tid + 1) / Nthread;
      for (sctl::Long k = 0; k < dof; k++) {
        const sctl::Vector<Real> F_(KDIM0 * Nt_ * Np_, (sctl::Iterator<Real>)F.begin() + k * KDIM0 * Nt_ * Np_, false);
        for (sctl::Long i = i0; i < i1; i++) {
          SetPatch(GF, trg_t_[i], trg_p_[i], F_, Nt_, Np_, T.PATCH_DIM);
          const sctl::ConstIterator<Real> M = MGrid_.begin() + i * KDIM1 * Kp;
          Vec acc[KDIM1];
          for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) acc[k1] = Vec::Zero();
          for (sctl::Long l = 0; l < Kp; l += VecLen) {
            const Vec g = Vec::LoadAligned(&GF_buf.begin()[l]);
            for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) acc[k1] = FMA(g, Vec::LoadAligned(&M[k1 * Kp + l]), acc[k1]);
          }
          for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) Uloc[(k * KDIM1 + k1) * Ntrg_ + trg_dsp_ + i] = sctl::reduce(acc[k1]);
        }
      }
    });
  }
  comm_.Allreduce(Uloc.begin(), Uglb.begin(), Uglb.Dim(), sctl::CommOp::SUM);
  if (U.Dim() != dof * KDIM1 * Ntrg_) { // Init U
    U.ReInit(dof * KDIM1 * Ntrg_);
    U = 0;
  }
  U += Uglb;
}

template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer HedgehogOrder> void SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>::SetPatch(sctl::Vector<Real>& out, sctl::Long t0, sctl::Long p0, const sctl::Vector<Real>& in, sctl::Long Nt, sctl::Long Np, sctl::Integer PATCH_DIM) {
  SCTL_ASSERT(Nt >= PATCH_DIM);
  SCTL_ASSERT(Np >= PATCH_DIM);

  const sctl::Long Ngrid = PATCH_DIM * PATCH_DIM;
  sctl::Long dof = in.Dim() / (Nt * Np);
  SCTL_ASSERT(in.Dim() == dof * Nt * Np);
  SCTL_ASSERT(out.Dim() == dof * Ngrid);
  sctl::Integer tt0 = t0 - (PATCH_DIM - 1) / 2;
  sctl::Integer pp0 = p0 - (PATCH_DIM - 1) / 2;
  for (sctl::Long k = 0; k < dof; k++) {
    for (sctl::Long i = 0; i < PATCH_DIM; i++) {
      sctl::Long t = tt0 + i;
      if (t >= Nt) t = t - Nt;
      if (t < 0) t = t + Nt;
      sctl::ConstIterator<Real> in_ = in.begin() + (k * Nt + t) * Np;
      sctl::Iterator<Real> out_ = out.begin() + (k * PATCH_DIM + i) * PATCH_DIM;
      for (sctl::Long j = 0; j < PATCH_DIM; j++) {
        sctl::Long p = pp0 + j;
        if (p >= Np) p = p - Np;
        if (p < 0) p = p + Np;
        out_[j] = in_[p];
      }
    }
  }
}

}

#endif  //_SINGULAR_CORRECTION_HPP_
