// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _BOUNDARY_INTEG_OP_HPP_
#define _BOUNDARY_INTEG_OP_HPP_

#include <memory>
#include <vector>

#include <biest/offsurface_op.hpp>
#include <biest/singular_correction.hpp>
#include <biest/surface.hpp>
#include <sctl.hpp>

namespace biest {

namespace detail_biop {
  template <class Real, sctl::Integer HedgehogOrder> class BoundaryIntegralOp; // the layer potentials on the full tori, in boundary_integ_op.txx
}

/**
 * Layer potentials on toroidal surfaces: the integrals over the full tori, at
 * targets on the first 1/NFP of each torus. A density may be given on one
 * field period and is then completed by rotation. The components of the
 * density and of the potential are those of the kernel given to
 * SetupSingular(). A later SetupSingular() reuses the memory of the earlier
 * one where the sizes allow. An object must not be copied after
 * SetupSingular().
 *
 * @tparam Real float, double or sctl::QuadReal (with SCTL_QUAD_T defined).
 *
 * @tparam HedgehogOrder 0 or 1 gives the principal value on the surface; a
 * value in [2, 16] the limit from inside the surface, by extrapolation from
 * points off it.
 */
template <class Real, sctl::Integer HedgehogOrder = 1> class FieldPeriodBIOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructor.
     *
     * @param comm the communicator.
     */
    explicit FieldPeriodBIOp(const sctl::Comm& comm = sctl::Comm::Self());

    ~FieldPeriodBIOp();

    /**
     * Set up the operator for one surface, to the accuracy 10^-digits; no
     * symmetry is assumed. The quadrature grid is that of
     * SingularCorrection::QuadGrid(): of dimensions that are multiples of
     * trg_NFP*trg_Nt and trg_Np, at least twice the larger of the surface grid
     * and src_Nt x src_Np, and with refine, at least the upsampling of
     * src_Nt x src_Np on which the singular quadrature interpolates refine to
     * the accuracy.
     *
     * @param[in] S the surface (for several, see the SetupSingular() from a
     * tolerance).
     *
     * @param[in] ker the kernel; it must outlive the operator.
     *
     * @param[in] digits the number of decimal digits of accuracy.
     *
     * @param[in] trg_NFP, trg_Nt, trg_Np the targets: the trg_Nt x trg_Np grid
     * on the first 1/trg_NFP of the torus.
     *
     * @param[in] src_Nt, src_Np (optional) the finest density grid, counted on
     * the full torus, that the quadrature must resolve; -1 for the target
     * grid, trg_NFP*trg_Nt x trg_Np. Eval() takes densities on any grid.
     *
     * @param[in] refine (optional) fields on the src_Nt x src_Np grid of the
     * full torus that the quadrature grid must resolve, of any number of
     * components in SoA order: e.g. the density, or the right-hand side of a
     * linear system whose solution will be the density; used only to choose
     * the quadrature grid, and not stored.
     */
    template <class Kernel> void SetupSingular(const biest::Surface<Real>& S, const Kernel& ker, const sctl::Integer digits, const sctl::Integer trg_NFP, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long src_Nt = -1, const sctl::Long src_Np = -1, const sctl::Vector<Real>& refine = sctl::Vector<Real>());

    /**
     * Set up the operator for one or several surfaces, to the tolerance tol,
     * at the grid points of the first 1/NFP of each surface, NFP the greatest
     * common divisor of their Surface::NFP(). The quadrature grid of each
     * surface is that of SingularCorrection::QuadGrid(): of dimensions that
     * are multiples of the surface grid, at least twice it, and with refine,
     * at least the upsampling of it on which the singular quadrature
     * interpolates refine to the tolerance, unless quad gives the upsampling
     * factors.
     *
     * The potential of a surface at the targets of another is the trapezoidal
     * rule on its quadrature grid, except at the targets within
     * (log(1/tol) / (2 pi) + 1) h of a point of that grid, h the larger grid
     * spacing there: for a kernel that is an SctlKernelFunction, it is computed
     * there by OffSurfaceOp, on panels that resolve the surface and refine.
     *
     * @param[in] Svec the surfaces.
     *
     * @param[in] ker the kernel; it must outlive the operator.
     *
     * @param[in] tol the tolerance.
     *
     * @param[in] quad (optional) the singular quadrature and the upsampling
     * factors (see SingularQuadParams); those that are zero are chosen for tol.
     *
     * @param[in] refine (optional) fields that the quadrature must resolve,
     * e.g. the right-hand side of a linear system whose solution will be the
     * density: for each surface, on its targets (the grid of its first 1/NFP),
     * one surface after the other, of any number of components in SoA order,
     * the same for each surface; completed to the full tori as by
     * SurfaceOp::CompleteVecField() (3 components as a Cartesian vector). Used
     * only to choose the quadrature grids and the panels, and not stored.
     */
    template <class Kernel> void SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const Real tol, const SingularQuadParams& quad = SingularQuadParams(), const sctl::Vector<Real>& refine = sctl::Vector<Real>());

    /**
     * Evaluate the potential of the density F at the targets, for an operator
     * set up with one surface.
     *
     * @param[in,out] U the potential: each of its components on the target
     * grid, in SoA order; added to U if it has the right size.
     *
     * @param[in] F the density on one field period: each of its components on
     * the src_Nt x src_Np grid, in SoA order. It is copied NFP times in the
     * toroidal direction to cover the whole torus (a density of 3 components
     * rotated as a Cartesian vector to each field period), and resampled to
     * the quadrature grid; with NFP = 1, src_Nt = QuadNt(), src_Np = QuadNp()
     * and no half_period (the defaults), it is used as given. Several
     * densities are stored one after the other, and U holds their
     * potentials in the same order.
     *
     * @param[in] NFP the number of field periods of F, independent of the NFP
     * of the setup.
     *
     * @param[in] src_Nt, src_Np the grid of F on one field period, or on half
     * of one with half_period (-1 for QuadNt(), QuadNp()).
     *
     * @param[in] half_period F is given on half a field period, and the other
     * half is built by stellarator symmetry (see
     * SurfaceOp::CompleteVecField()): the grid point t of F is at the toroidal
     * angle (t+1/2)*pi/(NFP*src_Nt) from a plane of symmetry, the first at the
     * first grid point of the surface, and the full torus has NFP*2*src_Nt x
     * src_Np points.
     *
     * @param[in] parity the parity of F under stellarator symmetry. Used only
     * with half_period.
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP=1, const sctl::Long src_Nt=-1, const sctl::Long src_Np=-1, const bool half_period=false, const Parity parity=Parity::Even) const;

    /**
     * As Eval() above, for any number of surfaces and densities: F holds, for
     * each surface i, its densities one after the other, each component on
     * the src_Nt[i] x src_Np[i] grid of one of the NFP[i] field periods of
     * surface i; U holds, for each surface, their potentials in the same
     * order. half_period and parity apply to all the surfaces.
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Vector<sctl::Integer>& NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const bool half_period=false, const Parity parity=Parity::Even) const;

    /**
     * Evaluate the potential at points at any distance from the surfaces, by
     * OffSurfaceOp: its panels are fit to the surfaces and to the densities of
     * the first call after SetupSingular(), and again for another tol, and it
     * is set up for the targets again only for other targets.
     *
     * @param[in,out] U the potential at Xt in SoA order, added to U if it has
     * the right size.
     *
     * @param[in] Xt the coordinates of the targets in SoA order.
     *
     * @param[in] F one density on each surface, as for Eval().
     *
     * @param[in] NFP, src_Nt, src_Np the field periods and the grids of F, as
     * for Eval().
     *
     * @param[in] ker the kernel given to SetupSingular().
     *
     * @param[in] tol the tolerance.
     */
    template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const sctl::Vector<sctl::Integer>& NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol = (Real)1e-10) const;

    /**
     * As EvalOffSurface() above, for one surface, with F on the src_Nt x
     * src_Np grid of one of NFP field periods.
     */
    template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol = (Real)1e-10) const;

    /**
     * Returns the size of one density on the quadrature grids (i = 0) and of
     * its potential (i = 1).
     */
    sctl::Long Dim(sctl::Integer i) const;

    /**
     * Returns the toroidal dimension of the quadrature grid of surface i.
     */
    sctl::Long QuadNt(sctl::Long i = 0) const { return quad_Nt_[i]; }

    /**
     * Returns the poloidal dimension of the quadrature grid of surface i.
     */
    sctl::Long QuadNp(sctl::Long i = 0) const { return quad_Np_[i]; }

  private:

    /**
     * Resample the surfaces to the quadrature grids quad_Nt_ x quad_Np_ and set the targets, the points of the trg_Nt_ x
     * trg_Np_ grids on the first 1/NFP of each torus.
     */
    void SetGrids(const sctl::Vector<biest::Surface<Real>>& Svec, const sctl::Integer NFP);

    /**
     * Calls op(Fq), with Fq the densities F of Eval() on the quadrature grids: F itself when it is on them, and else F
     * completed to the full tori by the symmetry and resampled, in scratch memory (see SurfaceOp::SurfGrad).
     */
    template <class Op> void OnQuadGrids(const sctl::Vector<Real>& F, const sctl::Vector<sctl::Integer>& NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const bool half_period, const Parity parity, const Op& op) const;

    using BIOp = detail_biop::BoundaryIntegralOp<Real,HedgehogOrder>;

    /**
     * Set up the operator on the surfaces and the targets of SetGrids(), in the memory of the earlier setup, with the
     * panels between surfaces fit to the fields refine on the quadrature grids too (see BIOp::SetupSingular()).
     */
    template <class Kernel> void SetupSingular0(const Kernel& ker, const Real tol, const SingularQuadParams& quad, const sctl::Vector<Real>& refine);

    BIOp* biop;
    sctl::StaticArray<sctl::Integer,2> kdim_; // the density and potential components of the kernel of the setup
    sctl::Vector<sctl::Integer> tor_orient_; // for each surface
    sctl::Vector<sctl::Long> trg_Nt_, trg_Np_, quad_Nt_, quad_Np_; // for each surface
    sctl::Vector<sctl::Vector<sctl::Long>> trg_idx_; // for each surface, the targets: indices of quadrature grid points
    sctl::Vector<biest::Surface<Real>> Svec_; // the surfaces on the quadrature grids
    sctl::Comm comm_;
};

}

#include <biest/boundary_integ_op.txx>

#endif  //_BOUNDARY_INTEG_OP_HPP_
