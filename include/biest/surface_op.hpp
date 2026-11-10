// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _SURFACE_OP_HPP_
#define _SURFACE_OP_HPP_

#include <map>
#include <utility>

#include <biest/kernel.hpp>
#include <biest/surface.hpp>
#include <sctl.hpp>

namespace biest {

/**
 * Spectral (FFT-based) differential operators, quadrature and resampling for
 * data on a toroidal surface given on a uniform periodic Nt x Np grid in SoA
 * order, with any number of components. The surface parameters t and p run
 * over [0,1) in the toroidal and poloidal directions.
 *
 * The grid can also be one of the NFP field periods of a torus with that
 * symmetry (see Surface::NFP()), the first Nt rows of its NFP*Nt x Np grid, on
 * which t runs over [0,1/NFP). Each input and output is then the restriction
 * to the field period of the same quantity on the full torus: dX is the
 * restriction of Grad2D() of the coordinates on the full torus. This holds for
 * Grad2D() and SurfGrad() of functions that repeat from one field period to
 * the next, for SurfDiv() and SurfCurl() of vector fields that rotate with the
 * field periods, and for SurfNormalAreaElem(), SurfLap(), SurfInteg(),
 * ProjZeroMean(), InvSurfLap(), GradInvSurfLap(), InvSurfLapPrecond() and
 * HodgeDecomp(). The differential operators are in the intrinsic form, which
 * differentiates only the metric and the components of vector fields along
 * the tangent vectors: quantities that repeat from one field period to the
 * next, unlike the Cartesian components of vectors.
 */
template <class Real> class SurfaceOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructor.
     *
     * @param[in] comm the communicator, passed to the GMRES solver; the FFTs
     * are not distributed.
     *
     * @param[in] Nt number of surface discretization points in the toroidal
     * direction.
     *
     * @param[in] Np number of surface discretization points in the poloidal
     * direction.
     *
     * @param[in] NFP the grid is one of NFP field periods of a torus with that
     * symmetry (see above), and the full torus if 1.
     *
     * @param[in] tor_orient for NFP > 1, +1 or -1 if the toroidal index of the
     * torus increases counterclockwise or clockwise about the z-axis (see
     * ToroidalOrientation()).
     */
    SurfaceOp(const sctl::Comm& comm = sctl::Comm::Self(), sctl::Long Nt = 0, sctl::Long Np = 0, sctl::Integer NFP = 1, sctl::Integer tor_orient = 0);

    /**
     * Copy constructor.
     */
    SurfaceOp(const SurfaceOp& Op);

    /**
     * Assignment operator.
     */
    SurfaceOp& operator=(const SurfaceOp& Op);

    /**
     * Fourier upsample (or downsample) data defined on 2D periodic grid.
     *
     * @param[in] X0_ the vector of input data in SoA order on a Nt0xNp0 grid in
     * row-major order.
     *
     * @param[in] Nt0 the number of discretization points in toroidal direction
     * for the input data.
     *
     * @param[in] Np0 the number of discretization points in poloidal direction
     * for the input data.
     *
     * @param[out] X1_ the vector of output data in SoA order on a Nt1xNp1 grid in
     * row-major order.
     *
     * @param[in] Nt1 the number of discretization points in toroidal direction
     * for the output data.
     *
     * @param[in] Np1 the number of discretization points in poloidal direction
     * for the output data.
     */
    static void Upsample(const sctl::Vector<Real>& X0_, sctl::Long Nt0, sctl::Long Np0, sctl::Vector<Real>& X1_, sctl::Long Nt1, sctl::Long Np1);

    /**
     * Computes the gradient of data X(t,p) defined on a periodic 2D grid.
     *
     * @param[out] dX the output vector containing the negated derivatives
     * {-dX/dt, -dX/dp} for each component of X, stored in SoA order. The sign
     * does not affect the normals and surface operators computed from dX.
     * The derivatives are with respect to the grid parameters t, p in [0, 1)
     * (t in [0, 1/NFP) on one of NFP field periods).
     *
     * @param[in] X the vector of input data X(t,p) sampled on a grid of
     * dimensions NtxNp stored in row-major order and as a structure-of-arrays.
     */
    void Grad2D(sctl::Vector<Real>& dX, const sctl::Vector<Real>& X) const;

    /**
     * Computes the surface normal and the quadrature weight at each grid
     * point.
     *
     * @param[out] normal pointer to the vector where the per grid point unit
     * surface normals will be written in SoA order (may be null).
     *
     * @param[out] area_elem pointer to the vector where the per grid point
     * trapezoidal quadrature weights |dX/dt x dX/dp|/(NFP*Nt*Np) will be
     * written (may be null).
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] X (optional) the vector of position/coordinate values of each
     * grid point in SoA order.  This is used only to determine the correct
     * orientation of the normal vectors (i.e. outward from the surface), from
     * the normal at the grid point farthest from the z-axis.
     *
     * @return +1 or -1, the factor applied to dX/dt x dX/dp to orient the
     * normals, or 0 if X or normal is null.
     */
    Real SurfNormalAreaElem(sctl::Vector<Real>* normal, sctl::Vector<Real>* area_elem, const sctl::Vector<Real>& dX, const sctl::Vector<Real>* X) const;

    /**
     * Computes the normal component of the surface curl, -n . curl(F), for
     * each 3-component vector field in F, as the surface divergence of n x F
     * (see SurfDiv()).
     *
     * @param[out] CurlF the output vector containing one scalar per grid point
     * for each vector field in F, stored in SoA order.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] normal the vector of surface normals at each grid point in SoA
     * order.
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfCurl(sctl::Vector<Real>& CurlF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& normal, const sctl::Vector<Real>& F) const;

    /**
     * Computes the on-surface gradient operation.
     *
     * @param[out] GradFvec the output vector containing the gradient of F (3
     * components for each component of F) at each grid point stored in SoA
     * order.
     *
     * @param[in] dXvec_ the derivatives of the surface coordinates, from
     * Grad2D().
     *
     * @param[in] Fvec the vector of data values at each grid point in SoA
     * order.
     */
    void SurfGrad(sctl::Vector<Real>& GradFvec, const sctl::Vector<Real>& dXvec_, const sctl::Vector<Real>& Fvec) const;

    /**
     * Computes the on-surface divergence operation, in the intrinsic form
     * (1/sqrt(g)) d_i(sqrt(g) g^ij F.dX/du_j) of the components of F along the
     * tangent vectors: the divergence of the tangential part of F.
     *
     * @param[out] DivF the output vector containing the divergence of F at each
     * grid point stored in SoA order.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] F the tangential vector field at each grid point in SoA
     * order.
     */
    void SurfDiv(sctl::Vector<Real>& DivF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    /**
     * Computes the on-surface Laplacian operation, in the intrinsic form
     * (1/sqrt(g)) d_i(sqrt(g) g^ij d_j F) of each component of F, with g_ij the
     * metric of the grid parameters (the dot products of the tangent vectors).
     *
     * @param[out] LapF the output vector containing the laplacian of F at each grid
     * point stored in SoA order.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfLap(sctl::Vector<Real>& LapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    /**
     * Compute the surface integral of a function (on periodic NtxNp grid) using
     * trapezoidal rule.
     *
     * @param[out] I the vector containing the surface integral of each
     * component of F.
     *
     * @param[in] area_elem the quadrature weights at each grid point, from
     * SurfNormalAreaElem().
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfInteg(sctl::Vector<Real>& I, const sctl::Vector<Real>& area_elem, const sctl::Vector<Real>& F) const;

    /**
     * Subtracts the area-weighted average value of a given function on a
     * periodic NtxNp grid.
     *
     * @param[out] Fproj the output mean-zero function at each grid point.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void ProjZeroMean(sctl::Vector<Real>& Fproj, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    /**
     * Compute the inverse of the surface-Laplacian: solve the Laplace-Beltrami
     * problem for the mean-zero solution, with the right-hand side projected
     * to mean zero. The solve uses GMRES, preconditioned with the exact
     * inverse (implemented using FFT) for a flat torus.
     *
     * @param[out] InvLapF the solution to the Laplace-Beltrami problem
     * evaluated at each grid point.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] F the RHS in the Laplace-Beltrami problem evaluated at each
     * grid point.
     *
     * @param[in] tol the tolerance for the GMRES solve.
     *
     * @param[in] max_iter the maximum number of GMRES iterations.
     *
     * @param[in] upsample the upsample factor for the solve. The final result
     * is downsampled back to the original resolution. The metric on the
     * upsampled grid is computed from the tangent vectors upsampled (on one
     * field period, in the frame that turns with the toroidal parameter, in
     * which they repeat from one period to the next).
     */
    void InvSurfLap(sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1, Real upsample = 1.0) const;

    /**
     * Compute the surface gradient of the solution of InvSurfLap(), on the
     * grid of F from the solution downsampled.
     *
     * @param[out] GradInvLapF the gradient of the solution to the
     * Laplace-Beltrami problem evaluated at each grid point.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] F the RHS in the Laplace-Beltrami problem evaluated at each
     * grid point.
     *
     * @param[in] tol the tolerance for the GMRES solve.
     *
     * @param[in] max_iter the maximum number of GMRES iterations.
     *
     * @param[in] upsample the upsample factor for the solve. The final result
     * is downsampled back to the original resolution.
     */
    void GradInvSurfLap(sctl::Vector<Real>& GradInvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1, Real upsample = 1.0) const;

    /**
     * Compute the inverse of the surface-Laplacian (solve Laplace-Beltrami
     * problem) using given preconditioner.
     *
     * @param[out] InvLapF the solution to the Laplace-Beltrami problem
     * evaluated at each grid point.
     *
     * @param[in] LeftPrecond the left preconditioner.
     *
     * @param[in] RightPrecond the right preconditioner. Each preconditioner
     * may add to its output, which it is given empty or zero.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] F the RHS in the Laplace-Beltrami problem evaluated at each
     * grid point.
     *
     * @param[in] tol the tolerance for the GMRES solve.
     *
     * @param[in] max_iter the maximum number of GMRES iterations.
     */
    void InvSurfLapPrecond(sctl::Vector<Real>& InvLapF, const std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>& LeftPrecond, const std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>& RightPrecond, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1) const;

    /**
     * Evaluate the potential from a given surface density at target point by
     * computing the integral using periodic trapezoidal rule, without
     * singular correction. Can be used to evaluate the potential at
     * off-surface points (far from the surface compared to the grid spacing).
     *
     * @tparam KDIM0 degrees-of-freedom of the density per source point.
     *
     * @tparam KDIM1 degrees-of-freedom of the potential per target point.
     *
     * @param[in,out] Utrg the potential at each target point in SoA order,
     * added to Utrg if it already has the right size.
     *
     * @param[in] Xtrg the location of the target points in SoA order.
     *
     * @param[in] Xsrc the location of the source points in SoA order.
     *
     * @param[in] Xn_src the surface normals at the source points in SoA order.
     *
     * @param[in] Xa_src the quadrature weights at the source points, from
     * SurfNormalAreaElem().
     *
     * @param[in] Fsrc the density function at the source points in SoA order.
     *
     * @param[in] ker the kernel function.
     */
    template <sctl::Integer KDIM0, sctl::Integer KDIM1> void EvalSurfInteg(sctl::Vector<Real>& Utrg, const sctl::Vector<Real>& Xtrg, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& Xn_src, const sctl::Vector<Real>& Xa_src, const sctl::Vector<Real>& Fsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker) const;

    /**
     * Computes the Hodge decomposition of the given vector field on a surface,
     * V = Vn + Vd + Vc + Vh, with Vd = grad(alpha) and Vc = n x grad(beta).
     *
     * @param[out] Vn the normal part of the field at each grid point in SoA
     * order.
     *
     * @param[out] Vd the gradient part of the field at each grid point in SoA
     * order.
     *
     * @param[out] Vc the solenoidal part of the field at each grid point in SoA
     * order.
     *
     * @param[out] Vh the harmonic part of the field at each grid point in SoA
     * order.
     *
     * @param[in] V the input vector field at each grid point in SoA order.
     *
     * @param[in] dX the derivatives of the surface coordinates, from Grad2D().
     *
     * @param[in] normal the surface normals at each grid point in SoA order.
     *
     * @param[in] tol the tolerance for the Laplace-Beltrami solves.
     *
     * @param[in] max_iter maximum number of GMRES iterations.
     */
    void HodgeDecomp(sctl::Vector<Real>& Vn, sctl::Vector<Real>& Vd, sctl::Vector<Real>& Vc, sctl::Vector<Real>& Vh, const sctl::Vector<Real>& V, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& normal, Real tol, sctl::Long max_iter = -1) const;



    static void test_SurfGrad(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm);

    static void test_SurfLap(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm);

    static void test_InvSurfLap(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>* LeftPrecond, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>* RightPrecond, const sctl::Comm& comm);

    static void test_HodgeDecomp(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm);



    /**
     * Shift data on a periodic Nt_ x Np_ grid in the toroidal direction by
     * Fourier interpolation: X_ at toroidal angle theta is X at theta+dtheta,
     * where grid index t is at angle 2*pi*t/Nt_.
     */
    static void RotateToroidal(sctl::Vector<Real>& X_, const sctl::Vector<Real>& X, const sctl::Long Nt_, const sctl::Long Np_, const Real dtheta);

    /**
     * Returns +1 if the toroidal index of the surface coordinates X (Nt x Np
     * grid, SoA order) increases counterclockwise about the z-axis, and -1 if
     * it increases clockwise.
     */
    static sctl::Integer ToroidalOrientation(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np);

    /**
     * Build data X on the full torus (NFP*Nt x Np grid, SoA order) from data Y
     * on one field period (Nt x Np grid). The other field periods are placed
     * in the direction in which the toroidal index of Y increases. Data with 3
     * components is rotated about the z-axis as a Cartesian vector; other data
     * is copied.
     *
     * @param[out] X the data on the full torus.
     *
     * @param[in] half_period Y is given on half a field period, at toroidal
     * angles (t+1/2)*pi/(NFP*Nt) in the direction of the toroidal index, and
     * the other half is built by stellarator symmetry (see Parity). X then
     * has NFP*2*Nt x Np points, at toroidal angles (t+1/2)*pi/(NFP*Nt).
     *
     * @param[in] NFP number of toroidal field periods.
     *
     * @param[in] Nt, Np the grid dimensions of Y.
     *
     * @param[in] Y the data on one field period, in SoA order.
     *
     * @param[in] parity the parity of Y under stellarator symmetry. Used only
     * with half_period.
     *
     * @param[in] tor_orient +1 or -1 if the toroidal index of Y increases
     * counterclockwise or clockwise about the z-axis: ToroidalOrientation() of
     * the surface coordinates. Used only for data with 3 components.
     *
     * @param[in] dtheta the result is shifted toroidally with RotateToroidal().
     */
    static void CompleteVecField(sctl::Vector<Real>& X, const bool half_period, const sctl::Integer NFP, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Y, const Parity parity, const sctl::Integer tor_orient, const Real dtheta = 0);

    /**
     * Resample X0 from an Nt0 x Np0 periodic grid to an Nt1 x Np1 grid by
     * evaluating its Fourier interpolant at the new grid points. Note the
     * argument order (output first).
     */
    static void Resample(sctl::Vector<Real>& X1, const sctl::Long Nt1, const sctl::Long Np1, const sctl::Vector<Real>& X0, const sctl::Long Nt0, const sctl::Long Np0);

  private:

    void Init(const sctl::Comm& comm, sctl::Long Nt, sctl::Long Np, sctl::Integer NFP, sctl::Integer tor_orient);

    /**
     * The negated divergence in the grid parameters, -(dA_t/dt + dA_p/dp), of the dof fields A (A_t, A_p of field k
     * the components 2*k and 2*k+1): the counterpart of Grad2D(), with the transforms of the columns of A_t only.
     */
    void Div2D(sctl::Vector<Real>& DivA, const sctl::Vector<Real>& A) const;

    /**
     * The coefficients of the intrinsic Laplace-Beltrami operator at each grid point, from the derivatives dX: C holds
     * sqrt(g) g^tt, sqrt(g) g^tp, sqrt(g) g^pp and sqrt(g), with g_ij the metric of the grid parameters.
     */
    void LapCoeff(sctl::Vector<Real>& C, const sctl::Vector<Real>& dX) const;

    /**
     * The intrinsic divergence (1/sqrt(g)) d_i(sqrt(g) g^ij A_j) of the dof fields of covariant components A (A_t, A_p
     * of field k the components 2*k and 2*k+1, both negated, as Grad2D() returns them), with C from LapCoeff().
     */
    void CovDiv(sctl::Vector<Real>& DivA, const sctl::Vector<Real>& C, const sctl::Vector<Real>& A) const;

    /**
     * The derivatives dX (from Grad2D() of the coordinates) resampled to an Nt1 x Np1 grid: on one of NFP field
     * periods, in the frame that turns by 2*pi*tor_orient*t about the z-axis, in which they repeat.
     */
    void ResampleTangents(sctl::Vector<Real>& dX1, sctl::Long Nt1, sctl::Long Np1, const sctl::Vector<Real>& dX) const;

    static Real compute_area_elem(sctl::StaticArray<Real, COORD_DIM>& xn, const sctl::StaticArray<Real, COORD_DIM>& xt, const sctl::StaticArray<Real, COORD_DIM>& xp);

    static Real max_norm(const sctl::Vector<Real>& x);

    /**
     * A preconditioner of the Laplace-Beltrami operator on an Nt x Np grid: the inverse of the Laplacian in the grid
     * parameters, scaled by the mean lengths of the grid lines (1 / invLt2 and 1 / invLp2 their squares, in grid
     * spacings), in Fourier space.
     */
    struct SpectralInvLapOp {
      sctl::Long Nt, Np;
      Real invLt2, invLp2;
      void operator()(sctl::Vector<Real>& Sx, const sctl::Vector<Real>& x) const;
    };

    /**
     * Returns SpectralInvLapOp on an Nt_ x Np_ grid with derivatives dX, one of NFP field periods.
     */
    static SpectralInvLapOp SpectralInvLap(const sctl::Vector<Real>& dX, const sctl::Long Nt_, const sctl::Long Np_, const sctl::Integer NFP);

    void LaplaceBeltramiReference(sctl::Vector<Real>& f0, sctl::Vector<Real>& u0, const sctl::Vector<Real>& X, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X) const;

    sctl::Comm comm_;
    sctl::Long Nt_, Np_;
    sctl::Integer NFP_, tor_orient_;
    mutable sctl::GMRES<Real> solver;
};

}

#include <biest/surface_op.txx>

#endif  //_SURFACE_OP_HPP_
