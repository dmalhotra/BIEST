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

/**
 * Compute the layer-potentials from a vector of toroidal surfaces.
 *
 * @tparam Real datatype for reals can be float, double or sctl::QuadReal.
 *
 * @tparam KDIM0 degrees-of-freedom of the density per source point.
 *
 * @tparam KDIM1 degrees-of-freedom of the potential per target point.
 *
 * @tparam UPSAMPLE source mesh upsample factor for boundary quadratures.
 *
 * @tparam HedgehogOrder 0 or 1 gives the principal value on the surface. A
 * value in [2, 16] gives the one-sided limit from the side opposite to the
 * normal (the interior for outward normals), by extrapolation from points
 * off the surface.
 */
template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE = 1, sctl::Integer HedgehogOrder = 1> class BoundaryIntegralOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructor.
     *
     * @param comm the communicator. With several processes, the target points
     * are divided among the processes, and every process holds all the data.
     */
    BoundaryIntegralOp(const sctl::Comm& comm = sctl::Comm::Self());

    /**
     * Returns the dimensions of the boundary integral operator: the size of
     * the density vector (KDIM0 times the number of grid points) for i==0, and
     * of the potential vector (KDIM1 times the number of target points) for
     * i==1.
     */
    sctl::Long Dim(sctl::Integer i) const;

    /**
     * Setup the layer-potential operator and precompute the singular
     * corrections. The target points are the grid points of the surfaces.
     *
     * The singular quadrature of each surface (see SingularQuadParams) is that
     * given, with the values not given chosen for the tolerance tol and the
     * anisotropy of the grid of the surface, upsampled UPSAMPLE times.
     *
     * With several surfaces, the potential of a surface at the targets of
     * another one is the trapezoidal rule on the grid upsampled UPSAMPLE times,
     * except at the targets too close to the surface for it to be accurate to
     * tol: within (log(1/tol) / (2 pi) + 1) times the local grid spacing, as
     * its error is about exp(-2 pi d / h) at a distance d for a spacing h.
     * There, with a kernel that is an SctlKernelFunction, the potential is
     * that of OffSurfaceOp, by quadrature on panels fit to the surface; with
     * another kernel, the trapezoidal rule still.
     *
     * @param[in] Svec vector containing the surfaces.
     *
     * @param[in] ker the kernel function; it must outlive the operator.
     *
     * @param[in] trg_idx_ (optional) for each surface, the indices of the grid
     * points which will be the target points.
     *
     * @param[in] tol the accuracy of the singular quadrature, and of the
     * potential of a surface at the targets of another one.
     *
     * @param[in] quad (optional) the singular quadrature, its zero values
     * chosen from tol.
     */
    template <class Surface, class Kernel> void SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_ = sctl::Vector<sctl::Vector<sctl::Long>>(), const Real tol = (Real)1e-10, const SingularQuadParams& quad = SingularQuadParams());

    /**
     * Evaluate the layer-potential operator. The potential of a surface at
     * the targets of another one is that described in SetupSingular().
     *
     * @param[in,out] U the vector of computed potentials at the target points
     * in SoA order. If U already has the right size the potential is added to
     * it; otherwise it is resized and set to zero first.
     *
     * @param[in] F the vector of surface density values at surface grid points
     * in SoA order. Several densities can be stacked; their potentials are
     * stacked in U in the same order.
     */
    void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

    /**
     * Evaluate the potential at off-surface points, at any distance from the
     * surfaces. The panels for near-surface quadrature are rebuilt on every
     * call.
     *
     * @param[in,out] U the computed potential in SoA order, added to U if it
     * already has the right size.
     *
     * @param[in] Xt the position of the target points in SoA order.
     *
     * @param[in] F one density function at grid points in SoA order for each
     * surface concatenated together.
     *
     * @param[in] ker the kernel function given to SetupSingular().
     *
     * @param[in] tol the accuracy tolerance.
     */
    template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real> Xt, const sctl::Vector<Real>& F, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol = (Real)1e-10) const;

    template <class Kernel> static sctl::Vector<Real> test(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& ker, const sctl::Comm& comm, const SingularQuadParams& quad = SingularQuadParams());

    template <class Kernel, class DLKernel, class GradKernel> static void test_GreensIdentity(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& sl_ker, const DLKernel& dl_ker, const GradKernel& sl_grad_ker, const sctl::Comm& comm, const SingularQuadParams& quad = SingularQuadParams());

    static void test_Precond(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm, const SingularQuadParams& quad = SingularQuadParams());

  private:

    template <class Surface, class Kernel> void SetupHelper(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_ = sctl::Vector<sctl::Vector<sctl::Long>>());

    /**
     * The potential of a surface at targets near it, by OffSurfaceOp: for a kernel whose SCTL kernel is SKer.
     */
    struct NearOpBase {
      virtual ~NearOpBase() = default;
      virtual void Setup(const sctl::Vector<Real>& X, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& Xt, Real tol) = 0;
      virtual void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) = 0;
    };
    template <class SKer> struct NearOp : NearOpBase {
      template <sctl::Integer DIGITS, sctl::Integer Nv> NearOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) : op(ker, comm) {}
      void Setup(const sctl::Vector<Real>& X, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& Xt, Real tol) override { op.Setup(X, Nt, Np, Xt, tol); }
      void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) override { op.Eval(U, F); }
      OffSurfaceOp<Real,SKer> op;
    };

    /**
     * Returns a NearOp for a kernel that is an SctlKernelFunction, and else null.
     */
    template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> static std::unique_ptr<NearOpBase> NewNearOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) { return std::unique_ptr<NearOpBase>(new NearOp<SKer>(ker, comm)); }
    template <class Kernel> static std::unique_ptr<NearOpBase> NewNearOp(const Kernel&, const sctl::Comm&) { return nullptr; }

    /**
     * Set up the potential of each surface at the targets of the others: find the targets near the surface, and set
     * up a NearOp for them (see SetupSingular()).
     */
    template <class Surface, class Kernel> void SetupCross(const sctl::Vector<Surface>& Svec, const Kernel& ker, const Real tol);

    static void Upsample(const sctl::Vector<Real>& X0_, sctl::Vector<Real>& X_, sctl::Long Nt0, sctl::Long Np0); // SurfaceOp::Upsample by UPSAMPLE in each direction

    static Real max_norm(const sctl::Vector<Real>& x);

    sctl::Comm comm_;
    sctl::Vector<sctl::Long> Nt0, Np0;
    sctl::Vector<SurfaceOp<Real>> Op;
    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>* ker_ptr;

    sctl::StaticArray<sctl::Long,2> dim;
    sctl::Vector<Real> normal_orient;
    sctl::Vector<sctl::Vector<Real>> Xtrg, Xsrc, dXsrc, Xn_src, Xa_src;
    std::vector<SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>> singular_correction; // of each surface
    sctl::Vector<sctl::Vector<sctl::Long>> trg_idx;

    // The targets of surface t for the potential of a surface s != t, at cross[s * Nsurf + t]: those far from s, for
    // the trapezoidal rule, and those near it, for near_op. Without near_op, all of them by the trapezoidal rule.
    struct CrossInterac {
      sctl::Vector<sctl::Long> far_idx, near_idx; // indices into the targets of t
      sctl::Vector<Real> Xfar; // the coordinates of the far targets, SoA
      std::unique_ptr<NearOpBase> near_op;
    };
    std::vector<CrossInterac> cross;
};

/**
 * Compute layer-potentials on toroidal surfaces with field-period symmetry:
 * the densities on the full tori, given on one field period or on the full
 * tori, and the potential on one field period of each surface. The
 * quadrature parameters are chosen at run time from the number of requested
 * digits, or given. An object must not be copied after SetupSingular().
 *
 * @tparam Real datatype for reals can be float, double or sctl::QuadReal.
 *
 * @tparam KDIM0 degrees-of-freedom of the density per source point.
 *
 * @tparam KDIM1 degrees-of-freedom of the potential per target point.
 *
 * @tparam HedgehogOrder as in BoundaryIntegralOp: a value > 1 gives the
 * one-sided limit from the side opposite to the normal instead of the
 * principal value.
 */
template <class Real, sctl::Integer COORD_DIM=3, sctl::Integer KDIM0=1, sctl::Integer KDIM1=1, sctl::Integer HedgehogOrder = 1> class FieldPeriodBIOp {
  public:

    /**
     * Constructor.
     *
     * @param comm the communicator.
     */
    explicit FieldPeriodBIOp(const sctl::Comm& comm = sctl::Comm::Self());

    ~FieldPeriodBIOp();

    /**
     * Build the full-torus surface object (NFP*surf_Nt x surf_Np grid) from
     * the surface coordinates on one field period: the surface of
     * Surface(NFP, surf_Nt, surf_Np, X), which has NFP field periods.
     *
     * @param[in] X the surface coordinates (in one field period) in the order
     * {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}. The
     * toroidal index may increase in either direction about the z-axis.
     *
     * @param[in] NFP number of toroidal field periods. The surface as well as
     * the magnetic field must have this toroidal periodic symmetry.
     *
     * @param[in] surf_Nt surface discretization order in toroidal direction (in one field period).
     *
     * @param[in] surf_Np surface discretization order in poloidal direction.
     */
    static biest::Surface<Real> BuildSurface(const sctl::Vector<Real>& X, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np);

    /**
     * Setup layer potential operator.
     *
     * @param[in] Svec a vector of one surface (for several, see below).
     *
     * @param[in] ker kernel function object.
     *
     * @param[in] digits number of decimal digits of accuracy.
     *
     * @param[in] NFP number of toroidal field periods. The output potential will only be computed for one field-period.
     *
     * @param[in] src_Nt maximum discretization order for input density in toroidal direction (on the full torus;
     * NFP*Nt for a density on an Nt x Np grid per field period).
     *
     * @param[in] src_Np maximum discretization order for input density in poloidal direction.
     *
     * @param[in] trg_Nt output potential discretization order in toroidal direction (in one field period).
     *
     * @param[in] trg_Np output potential discretization order in poloidal direction.
     *
     * @param[in] qNt, qNp dimensions of the quadrature grid on the full torus.
     * When either is zero, they are chosen from digits and refined, with at
     * most three tests, until the error of the Laplace double-layer identity
     * at the targets is below 10^-digits or the next grid would be less than
     * 1.5 times finer; the quadrature grid is the last grid tested.
     */
    void SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long qNt = 0, const sctl::Long qNp = 0);

    /**
     * Setup the layer potential operator of several surfaces, with the field
     * periods common to all (the greatest common divisor of their
     * Surface::NFP()): the potential is computed on one field period of each surface, at the points
     * of a grid of trg_Nt[i] x trg_Np[i] points on that period. Each surface
     * is resampled to its quadrature grid, chosen from digits as by the
     * single-surface SetupSingular() for the surface alone, with the patch
     * size from the largest anisotropy of the surfaces, or given.
     *
     * The potential of a surface at the targets of another is that of
     * BoundaryIntegralOp::SetupSingular() to the tolerance 10^-digits: by
     * quadrature on panels at the targets near the surface, with a kernel that
     * is an SctlKernelFunction.
     *
     * @param[in] Svec the surfaces.
     *
     * @param[in] ker the kernel function; it must outlive the operator.
     *
     * @param[in] digits number of decimal digits of accuracy.
     *
     * @param[in] trg_Nt, trg_Np for each surface, the dimensions of the grid of
     * the targets on one field period.
     *
     * @param[in] qNt, qNp (optional) for each surface, the dimensions of the
     * quadrature grid on the full torus: multiples of NFP*trg_Nt[i] and of
     * trg_Np[i], at least the dimensions of the surface grid.
     */
    template <class Kernel> void SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const sctl::Integer digits, const sctl::Vector<sctl::Long>& trg_Nt, const sctl::Vector<sctl::Long>& trg_Np, const sctl::Vector<sctl::Long>& qNt = sctl::Vector<sctl::Long>(), const sctl::Vector<sctl::Long>& qNp = sctl::Vector<sctl::Long>());

    /**
     * As the SetupSingular() of several surfaces above, with the quadrature
     * grids given: that of BoundaryIntegralOp::SetupSingular() for tol and
     * quad on the grid of each surface upsampled UPSAMPLE times in each
     * direction, with the potential computed at the grid points of one field
     * period of each surface. A later call sets up the operator in the memory
     * of the earlier one when the surfaces have no more grid points.
     *
     * @param[in] Svec the surfaces.
     *
     * @param[in] ker the kernel function; it must outlive the operator.
     *
     * @param[in] tol the accuracy of the singular quadrature, and of the
     * potential of a surface at the targets of another.
     *
     * @param[in] quad (optional) the singular quadrature, its zero values
     * chosen from tol.
     */
    template <sctl::Integer UPSAMPLE = 1, class Kernel> void SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const Real tol, const SingularQuadParams& quad = SingularQuadParams());

    /**
     * Evaluate the potential from given density.
     *
     * @param[in,out] U output potential computed on surface nodes in one
     * field-period, added to U if it already has the right size.
     *
     * @param[in] F density function given at discretization nodes in one field-period.
     *
     * @param[in] NFP number of field-periods for discretizing F.
     *
     * @param[in] src_Nt discretization order for F in toroidal direction (in one field period).
     *
     * @param[in] src_Np discretization order for F in poloidal direction.
     *
     * The parameters NFP, src_Nt, and src_Np for F do not have to be the same as those in SetupSingular. The density F
     * is completed to the full torus by symmetry (a 3-component density is rotated as a Cartesian vector) and
     * re-sampled to the resolution QuadNt() x QuadNp() for evaluating the quadratures. The user can avoid this
     * extra work by providing F already sampled at this resolution (i.e. NFP=1, src_Nt=QuadNt(), src_Np=QuadNp()).
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP=1, const sctl::Integer src_Nt=-1, const sctl::Integer src_Np=-1) const;

    /**
     * As Eval() above, for several surfaces: F holds the densities of the
     * surfaces one after the other, on a grid of src_Nt[i] x src_Np[i] points
     * of each, on one field period if NFP > 1, completed by symmetry (a
     * density of 3 components is rotated as a Cartesian vector), and on the
     * full torus if NFP = 1. Several densities can be stored one after the
     * other on each surface, as for BoundaryIntegralOp; the potential holds,
     * for each surface, the potentials of the densities one after the other.
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np) const;

    /**
     * Evaluate the potential of densities F on the quadrature grids of the
     * surfaces (the full tori), without resampling them: as the Eval() above
     * with NFP = 1 and src_Nt[i] = QuadNt(i), src_Np[i] = QuadNp(i).
     */
    void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

    /**
     * Returns the size of a density on the quadrature grids (i = 0) and of the
     * potential (i = 1).
     */
    sctl::Long Dim(sctl::Integer i) const;

    /**
     * Returns the toroidal resolution of the source data for quadrature
     * evaluation, of surface i.
     */
    sctl::Long QuadNt(sctl::Long i = 0) const { return quad_Nt_[i]; }

    /**
     * Returns the poloidal resolution of the source data for quadrature
     * evaluation, of surface i.
     */
    sctl::Long QuadNp(sctl::Long i = 0) const { return quad_Np_[i]; }

    static void test() {
      constexpr int DIM = 3; // dimensions of coordinate space
      const int digits = 10; // number of digits of accuracy requested

      const int NFP = 1, Nt = 70, Np = 20;
      sctl::Vector<Real> X(DIM*Nt*Np), F(Nt*Np), U;
      for (int i = 0; i < Nt; i++) { // initialize data X, F
        for (int j = 0; j < Np; j++) {
          const Real phi = 2*sctl::const_pi<Real>()*i/Nt;
          const Real theta = 2*sctl::const_pi<Real>()*j/Np;

          const Real R = 1 + 0.25*sctl::cos<Real>(theta);
          const Real x = R * sctl::cos<Real>(phi);
          const Real y = R * sctl::sin<Real>(phi);
          const Real z = 0.25*sctl::sin<Real>(theta);

          X[(0*Nt+i)*Np+j] = x;
          X[(1*Nt+i)*Np+j] = y;
          X[(2*Nt+i)*Np+j] = z;
          F[i*Np+j] = x+y+z;
        }
      }

      //const auto kernel = biest::Laplace3D<Real>::FxU(); // Laplace single-layer kernel function
      const auto kernel = biest::Laplace3D<Real>::DxU(); // Laplace double-layer kernel function
      constexpr int KER_DIM0 = 1; // input degrees-of-freedom of kernel
      constexpr int KER_DIM1 = 1; // output degrees-of-freedom of kernel

      biest::FieldPeriodBIOp<Real,DIM,KER_DIM0,KER_DIM1,0> biop; // boundary integral operator

      sctl::Vector<biest::Surface<Real>> Svec(1);
      Svec[0] = biop.BuildSurface(X, NFP, Nt, Np); // build surface object

      biop.SetupSingular(Svec, kernel, digits, NFP, Nt, Np, Nt, Np); // initialize biop
      biop.Eval(U, F, NFP, Nt, Np); // evaluate potential

      WriteVTK("F", Svec, F); // visualize F
      WriteVTK("U", Svec, U); // visualize U
    }

  private:

    /**
     * SetupSingular() from digits: the quadrature grids and the patch size,
     * then Setup(). The grids are at least src_Nt[i] x src_Np[i] (on the full
     * torus) unless given.
     */
    template <class Kernel> void SetupDigits(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const sctl::Vector<sctl::Long>& trg_Nt, const sctl::Vector<sctl::Long>& trg_Np, const sctl::Vector<sctl::Long>& qNt, const sctl::Vector<sctl::Long>& qNp);

    /**
     * Resample the surfaces to the quadrature grids quad_Nt_ x quad_Np_ and set the targets, the points of the trg_Nt_ x
     * trg_Np_ grids on one field period.
     */
    void SetGrids(const sctl::Vector<biest::Surface<Real>>& Svec, const sctl::Integer NFP);

    using BIOp = BoundaryIntegralOp<Real,KDIM0,KDIM1,1,HedgehogOrder>;

    /**
     * Set up the operator on the surfaces and the targets of SetGrids(), in the memory of the earlier setup.
     */
    template <class Kernel> void SetupSingular0(const Kernel& ker, const Real tol, const SingularQuadParams& quad);

    template <class, sctl::Integer, sctl::Integer, sctl::Integer, sctl::Integer> friend class FieldPeriodBIOp;

    BIOp* biop;
    sctl::Integer NFP_;
    sctl::Vector<sctl::Integer> tor_orient_; // for each surface
    sctl::Vector<sctl::Long> trg_Nt_, trg_Np_, quad_Nt_, quad_Np_; // for each surface
    sctl::Vector<sctl::Vector<sctl::Long>> trg_idx_; // for each surface, the targets: indices of quadrature grid points
    sctl::Vector<biest::Surface<Real>> Svec_; // the surfaces on the quadrature grids
    sctl::Comm comm_;
};

}

#include <biest/boundary_integ_op.txx>

#endif  //_BOUNDARY_INTEG_OP_HPP_
