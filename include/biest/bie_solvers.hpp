// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _BIE_SOLVERS_HPP_
#define _BIE_SOLVERS_HPP_

#include <biest/boundary_integ_op.hpp>
#include <biest/surface_op.hpp>
#include <biest/surface.hpp>
#include <sctl.hpp>

namespace biest {

/**
 * Compute vacuum magnetic field enclosed within one toroidal surface or
 * between two toroidal surfaces.
 *
 * @tparam Real datatype for reals can be float, double or sctl::QuadReal.
 *
 * The quadrature grids and the singular quadrature are chosen for the accuracy
 * 0.1 gmres_tol (see FieldPeriodBIOp::SetupSingular()), unless given.
 */
template <class Real> class VacuumField {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * An object of this class keeps the boundary integral operators of Compute() from one call to the next: a later
     * call sets them up again, for the surfaces it is given, in the memory of the earlier one when the surfaces have
     * no more grid points, rather than allocating and page faulting that memory again. That memory, the singular
     * corrections of the operators, is 13 (2 P + 1)^2 values for each grid point, with P the radius of the patch of
     * the singular quadrature, held until the object is destroyed. The vectors of the solve are kept in the same way,
     * a few tens of values for each grid point and each surface.
     *
     * @param[in] comm the communicator when running in parallel.
     *
     * @param[in] quad (optional) the singular quadrature of the operators, its zero values chosen for the accuracy
     * 0.1 gmres_tol (see FieldPeriodBIOp::SetupSingular()); those of the static Compute().
     */
    explicit VacuumField(const sctl::Comm& comm = sctl::Comm::Self(), const SingularQuadParams& quad = SingularQuadParams());

    VacuumField(const VacuumField&) = delete;
    VacuumField& operator=(const VacuumField&) = delete;

    /**
     * As the static Compute() below, with the communicator and the operators of the object.
     */
    void Compute(sctl::Vector<Real>& B, Real tor_flux, Real pol_flux, const sctl::Vector<Surface<Real>>& Svec, Real gmres_tol, sctl::Long gmres_iter, sctl::Vector<sctl::Vector<Real>>* J = nullptr);

    /**
     * Compute the vacuum magnetic field (curl B = 0, div B = 0, B.n = 0) in the
     * region enclosed by the surfaces, on the surface grid points, and
     * optionally the surface currents which produce the fields of unit flux.
     *
     * @param[out] B the magnetic field on the surface.  B = {Bx11, Bx12, ...,
     * Bx1Np, Bx21, Bx22, ... , BxNtNp, By11, ... , Bz11, ...}, where Nt and Np
     * are the number of discretizations in toroidal and poloidal directions;
     * the surfaces are concatenated.
     *
     * @param[in] tor_flux the toroidal flux, through a cross-section at
     * constant toroidal angle: positive in the direction of the field of a
     * current along +z on the z-axis, counterclockwise about the z-axis,
     * whichever way the grid indices of the surfaces increase. The field is
     * the limit lambda -> 0 of the TaylorState field for the same fluxes.
     *
     * @param[in] pol_flux the poloidal flux (for toroidal shells; ignored for
     * one surface), through a ribbon that connects the two surfaces at
     * constant poloidal angle: positive in the direction of the field of a
     * current along the toroidal direction counterclockwise about the z-axis
     * inside the inner surface (at the outboard midplane, the direction -z).
     *
     * @param[in] Svec one toroidal surface, or two nested toroidal surfaces in
     * either order (the outer one is the one enclosing the larger volume).
     * When the surfaces declare field periods (Surface::NFP()), the solve is
     * on one of those common to all of them, with the targets of the
     * operators and the unknowns on that period: the setup and the memory of
     * the singular quadrature, and its cost at each iteration, are divided by
     * their number. The results are on the full tori.
     * @see biest::Surface
     * @see sctl::Vector
     *
     * @param[in] comm the communicator when running in parallel.
     *
     * @param[in] gmres_tol the relative tolerance for the GMRES linear solve;
     * with several surfaces, the potential of a surface at the grid points
     * of another is accurate to 0.1 gmres_tol (see
     * FieldPeriodBIOp::SetupSingular()).
     *
     * @param[in] gmres_iter the maximum number of GMRES iterations.
     *
     * @param[out] J (optional) the surface currents whose Biot-Savart fields
     * inside the region are the fields of unit flux, one per surface, in the
     * layout of B: for a shell, J[0] for (tor_flux, pol_flux) = (1, 0) and
     * J[1] for (0, 1). The current of B is tor_flux J[0] + pol_flux J[1];
     * pass them unchanged to EvalOffSurface().
     */
    static void Compute(sctl::Vector<Real>& B, Real tor_flux, Real pol_flux, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, sctl::Vector<sctl::Vector<Real>>* J = nullptr);

    /**
     * Compute the vacuum magnetic field of the fluxes tor_flux and pol_flux at
     * points inside the region, from the surface currents of Compute(). Unlike
     * the rest of the library, Xtrg and Btrg are in AoS order.
     *
     * @param[out] Btrg the magnetic field at target points.  Btrg = {Bx1, By1,
     * Bz1, Bx2, By2, ..., BxN, ByN, BzN}. Where N is the number of target
     * points.
     *
     * @param[in] Xtrg the location of the target points.  Xtrg = {x1, y1, z1,
     * x2, y2, ..., xN, yN, zN}. Where N is the number of target points.
     *
     * @param[in] Svec the surfaces given to Compute().
     *
     * @param[in] tor_flux, pol_flux the fluxes of the field (see Compute()).
     *
     * @param[in] J the surface currents returned by Compute().
     *
     * @param[in] comm the communicator when running in parallel.
     *
     * @param[in] tol the accuracy tolerance.
     */
    static void EvalOffSurface(sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& Xtrg, sctl::Vector<Surface<Real>> Svec, Real tor_flux, Real pol_flux, const sctl::Vector<sctl::Vector<Real>>& J, const sctl::Comm& comm, const Real tol = (Real)1e-10);

    /**
     * A simple subroutine to test this class.
     */
    static void test(sctl::Long upsample, const sctl::Comm& comm);

  private:

    static Real max_norm(const sctl::Vector<Real>& x);

    void ComputeHelper(sctl::Vector<sctl::Vector<Real>>& B, sctl::Vector<sctl::Vector<Real>>& J, sctl::Vector<Real>& tor_flux, sctl::Vector<Real>& pol_flux, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter);

    sctl::Comm comm_;
    SingularQuadParams quad_;
    FieldPeriodBIOp<Real> BI_biot_savart_;
    FieldPeriodBIOp<Real> BI_lap_grad_;
    FieldPeriodBIOp<Real> BI_laplace_;

    // The vectors of Compute(), kept from one call to the next with their storage, as the operators: for each surface,
    // its grid points on one field period (SurfDim_) and their offset in the vectors of all the surfaces (SurfDsp_), and
    // the same for the full torus (SurfDimF_, SurfDspF_), the dimensions of its grid, and the directions of its toroidal
    // index (+1: counterclockwise about the z-axis) and of its poloidal index (+1: counterclockwise in the (R, z)
    // half-plane, R to the right and z up); the derivatives of its coordinates, its normal and area element, and the
    // derivatives on one field period; for each solution, J0 and B0, and the fields B and J with their fluxes (the
    // results of ComputeHelper); the solution of GMRES
    sctl::Vector<sctl::Long> SurfDim_, SurfDsp_, SurfDimF_, SurfDspF_, SurfNt_, SurfNp_;
    sctl::Vector<sctl::Integer> SurfOrient_, SurfPolOrient_;
    sctl::Vector<sctl::Vector<Real>> dX_, Xn_, Xa_, dXp_, J0_, B0_, B_, J_;
    sctl::Vector<Real> tor_flux_, pol_flux_, F_;
};

/**
 * Compute Taylor states magnetic field enclosed within one toroidal surface or
 * between two toroidal surfaces.
 *
 * @tparam Real datatype for reals can be float, double or sctl::QuadReal.
 *
 * The quadrature grids and the singular quadrature are chosen for the accuracy
 * 0.1 gmres_tol (see FieldPeriodBIOp::SetupSingular()), unless given.
 */
template <class Real> class TaylorState {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * An object of this class keeps the boundary integral operators of Compute() from one call to the next: a later
     * call sets them up again, for the surfaces it is given, in the memory of the earlier one when the surfaces have
     * no more grid points, rather than allocating and page faulting that memory again. That memory, the singular
     * corrections of the operators, is 28 (2 P + 1)^2 values for each grid point, with P the radius of the patch of
     * the singular quadrature, held until the object is destroyed. The vectors of the solve are kept in the same way,
     * a few tens of values for each grid point and each surface.
     *
     * @param[in] comm the communicator when running in parallel.
     *
     * @param[in] quad (optional) the singular quadrature of the operators, its zero values chosen for the accuracy
     * 0.1 gmres_tol (see FieldPeriodBIOp::SetupSingular()); those of the static Compute().
     */
    explicit TaylorState(const sctl::Comm& comm = sctl::Comm::Self(), const SingularQuadParams& quad = SingularQuadParams());

    TaylorState(const TaylorState&) = delete;
    TaylorState& operator=(const TaylorState&) = delete;

    /**
     * As the static Compute() below, with the communicator and the operators of the object.
     */
    void Compute(sctl::Vector<Real>& B, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<Surface<Real>>& Svec, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol = 0.1, sctl::Vector<sctl::Vector<Real>>* m = nullptr, sctl::Vector<sctl::Vector<Real>>* sigma = nullptr);

    /**
     * Compute the Taylor state (curl B = lambda B, B.n = 0) in the region
     * enclosed by the surfaces, on the surface grid points, and optionally
     * the generalized Debye source densities which produce this field.
     *
     * @param[out] B the magnetic field on the surface.  B = {Bx11, Bx12, ...,
     * Bx1Np, Bx21, Bx22, ... , BxNtNp, By11, ... , Bz11, ...}, where Nt and Np
     * are the number of discretizations in toroidal and poloidal directions;
     * the surfaces are concatenated.
     *
     * @param[in] tor_flux the toroidal flux, positive counterclockwise about
     * the z-axis (see VacuumField::Compute()).
     *
     * @param[in] pol_flux the poloidal flux (for toroidal shells), positive in
     * the direction of the field of a toroidal current counterclockwise about
     * the z-axis (see VacuumField::Compute()).
     *
     * @param[in] lambda the Taylor state parameter lambda; must be nonzero (use
     * VacuumField for lambda = 0).
     *
     * @param[in] Svec one toroidal surface, or two nested toroidal surfaces in
     * either order (the outer one is the one enclosing the larger volume).
     * When the surfaces declare field periods (Surface::NFP()), the solve is
     * on one of those common to all of them, with the targets of the
     * operators and the unknowns on that period: the setup and the memory of
     * the singular quadrature, and its cost at each iteration, are divided by
     * their number, and so is the cost of the Laplace-Beltrami solves, which
     * are on that period too (see SurfaceOp). The results are on the full
     * tori.
     * @see biest::Surface
     * @see sctl::Vector
     *
     * @param[in] comm the communicator when running in parallel.
     *
     * @param[in] gmres_tol the relative tolerance for the GMRES linear solve;
     * with several surfaces, the potential of a surface at the grid points
     * of another is accurate to 0.1 gmres_tol (see
     * FieldPeriodBIOp::SetupSingular()).
     *
     * @param[in] gmres_iter the maximum number of GMRES iterations.
     *
     * @param[in] LB_tol the GMRES tolerance for the Laplace-Beltrami solves is
     * LB_tol*gmres_tol/lambda inside the GMRES iterations and LB_tol*gmres_tol
     * for the final densities.
     *
     * @param[out] m the complex generalized Debye source densities m of the
     * basis fields with unit flux, one per surface. Returned only if both m
     * and sigma are not null; pass them unchanged to EvalOffSurface().
     *
     * @param[out] sigma the complex generalized Debye source densities sigma,
     * stored as m.
     */
    static void Compute(sctl::Vector<Real>& B, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol = 0.1, sctl::Vector<sctl::Vector<Real>>* m = nullptr, sctl::Vector<sctl::Vector<Real>>* sigma = nullptr);

    /**
     * Evaluate the Taylor states field (generated by given generalized Debye
     * source densities) at points inside the region. Xtrg and Btrg are in AoS
     * order.
     *
     * @param[out] Btrg the magnetic field at target points.  Btrg = {Bx1, By1,
     * Bz1, Bx2, By2, ..., BxN, ByN, BzN}. Where N is the number of target
     * points.
     *
     * @param[in] Xtrg the location of the target points.  Xtrg = {x1, y1, z1,
     * x2, y2, ..., xN, yN, zN}. Where N is the number of target points.
     *
     * @param[in] Svec the surfaces given to Compute().
     *
     * @param[in] tor_flux, pol_flux, lambda the values given to Compute().
     *
     * @param[in] m the densities m returned by Compute().
     *
     * @param[in] sigma the densities sigma returned by Compute().
     *
     * @param[in] comm the communicator when running in parallel.
     *
     * @param[in] tol the accuracy tolerance.
     */
    static void EvalOffSurface(sctl::Vector<Real>& Btrg, const sctl::Vector<Real>& Xtrg, sctl::Vector<Surface<Real>> Svec, Real tor_flux, Real pol_flux, Real lambda, const sctl::Vector<sctl::Vector<Real>> m, const sctl::Vector<sctl::Vector<Real>>& sigma, const sctl::Comm& comm, const Real tol = (Real)1e-10);

    /**
     * Test this class.
     */
    static void test_conv(Real lambda, const sctl::Vector<Surface<Real>>& Svec_, sctl::Long upsample, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm, Real LB_tol = 0.1, const SingularQuadParams& quad = SingularQuadParams());

  private:

    static Real max_norm(const sctl::Vector<Real>& x);

    /**
     * Compute the real part of the magnetic field at the off-surface points Xt from the Debye source densities sigma and
     * m on the surfaces.
     */
    static void ComputeFarField(sctl::Vector<Real>& Breal, const sctl::Vector<Real>& Xt, const sctl::Vector<Surface<Real>>& Svec, const sctl::Vector<Real>& sigma, const sctl::Vector<Real>& m, Real lambda, const sctl::Comm& comm, const Real tol);

    // mH, the harmonic surface currents (the harmonic part of dX/dt and n times it), on the Nt x Np grid of one of NFP
    // field periods (see SurfaceOp)
    static void Compute_mH(sctl::Vector<Real>& mH, sctl::Long Nt, sctl::Long Np, sctl::Integer NFP, sctl::Integer tor_orient, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& Xn, const sctl::Comm& comm, Real tol, sctl::Long max_iter);
    // The potentials at the targets, of SurfDim[i] points of surface i at SurfDsp[i], of densities sigma and m on the
    // grids of the operators (BI_grad(U, F), a FieldPeriodBIOp applied to densities on the surface grids)
    template <class Op> static void Compute_Grad_v(sctl::Vector<Real>& Grad_v, const sctl::Vector<Real>& sigma, const Op& BI_grad, const KernelFunction<Real,COORD_DIM,2,COORD_DIM*2>& ker_grad); // grad_v <-- Grad(g[sigma])
    template <class Op> static void Compute_iQ(sctl::Vector<Real>& iQ, const sctl::Vector<Real>& m, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const Op& BI_potn); // iQ <-- i g[m]
    template <class Op> static void Compute_iCurlQ(sctl::Vector<Real>& iCurlQ, const sctl::Vector<Real>& m, const sctl::Vector<sctl::Long>& SurfDim, const sctl::Vector<sctl::Long>& SurfDsp, const Op& BI_grad, const KernelFunction<Real,COORD_DIM,2,COORD_DIM*2>& ker_grad); // iCurlQ <-- i Curl(g[m])

    void ComputeHelper(sctl::Vector<sctl::Vector<Real>>& B_out, sctl::Vector<sctl::Vector<Real>>& m_out, sctl::Vector<sctl::Vector<Real>>& sigma_out, Real lambda, const sctl::Vector<Surface<Real>>& Svec, const sctl::Comm& comm, Real gmres_tol, sctl::Long gmres_iter, Real LB_tol = 0.1);

    sctl::Comm comm_;
    SingularQuadParams quad_;
    FieldPeriodBIOp<Real> BI_grad_diff_;
    FieldPeriodBIOp<Real> BI_grad_;
    FieldPeriodBIOp<Real> BI_potn_;

    // The vectors of Compute(), kept from one call to the next with their storage, as the operators: for each surface,
    // its grid points on one field period (SurfDim_) and their offset in the vectors of all the surfaces (SurfDsp_), and
    // the same for the full torus (SurfDimF_, SurfDspF_), the dimensions of its grid, and the directions of its toroidal
    // and poloidal indices (as in VacuumField); the derivatives of its coordinates, its normal and area element, and the
    // derivatives and normal on one field period; for each solution,
    // mH, B0, sigma, m, B and B_flux, and B in the basis of the fluxes (with two surfaces); and the results of
    // ComputeHelper
    sctl::Vector<sctl::Long> SurfDim_, SurfDsp_, SurfDimF_, SurfDspF_, SurfNt_, SurfNp_;
    sctl::Vector<sctl::Integer> SurfOrient_, SurfPolOrient_;
    sctl::Vector<sctl::Vector<Real>> dX_, Xn_, Xa_, dXp_, Xnp_, mH_, B0_, sigma_, m_, B_, B_flux_, B2_;
    sctl::Vector<sctl::Vector<Real>> B_out_, m_out_, sigma_out_;
};

}

#include <biest/bie_solvers.txx>

#endif  //_BIE_SOLVERS_HPP_
