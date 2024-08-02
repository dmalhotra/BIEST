#ifndef _BOUNDARY_INTEG_OP_HPP_
#define _BOUNDARY_INTEG_OP_HPP_

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
 * @tparam PATCH_DIM0 the partition of unity function is defined on a grid of
 * dimensions PATCH_DIM0 x PATCH_DIM0.
 *
 * @tparam RAD_DIM the order of the polar quadrature rule. Radial dimension is
 * RAD_DIM and angular dimension is 2*RAD_DIM
 */
template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE = 1, sctl::Integer PATCH_DIM0 = 25, sctl::Integer RAD_DIM = 18, sctl::Integer HedgehogOrder = 1> class BoundaryIntegralOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructor.
     *
     * @param comm the communicator.
     */
    BoundaryIntegralOp(const sctl::Comm& comm = sctl::Comm::Self());

    /**
     * Returns the dimensions of the boundary integral operator.
     *
     * @param[in] i indicates source dimensions when i==0 and target dimensions
     * when i==1.
     */
    sctl::Long Dim(sctl::Integer i) const;

    /**
     * Setup the layer-potential operator.
     *
     * @param[in] Svec vector containing the surfaces.
     *
     * @param[in] ker the kernel function.
     *
     * @param[in] trg_idx_ vector of indices identifying a subset of the surface
     * grid points which will be the target points.
     */
    template <class Surface, class Kernel> void SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_ = sctl::Vector<sctl::Vector<sctl::Long>>());

    /**
     * Evaluate the layer-potential operator.
     *
     * @param[out] U the vector of computed potentials at the target points in
     * SoA order.
     *
     * @param[in] F the vector of surface density values at surface grid points
     * in SoA order.
     */
    void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

    /**
     * Evaluate the potential at off-surface points
     *
     * @param[out] U the computed potential in SoA order.
     *
     * @param[in] Xt the position of the target points in SoA order.
     *
     * @param[in] F the density function at grid points in SoA order for each
     * surface concatenated together.
     */
    void EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real> Xt, const sctl::Vector<Real>& F) const;

    template <class Kernel> static sctl::Vector<Real> test(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& ker, const sctl::Comm& comm);

    template <class Kernel, class GradKernel> static void test_GreensIdentity(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const Kernel& sl_ker, const Kernel& dl_ker, const GradKernel& sl_grad_ker, const sctl::Comm& comm);

    static void test_Precond(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, const sctl::Comm& comm);

  private:

    template <class Surface, class Kernel> void SetupHelper(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_ = sctl::Vector<sctl::Vector<sctl::Long>>());

    static void Upsample(const sctl::Vector<Real>& X0_, sctl::Vector<Real>& X_, sctl::Long Nt0, sctl::Long Np0); // TODO: Replace by upsample in SurfaceOp

    static Real max_norm(const sctl::Vector<Real>& x);

    sctl::Comm comm_;
    sctl::Vector<sctl::Long> Nt0, Np0;
    sctl::Vector<SurfaceOp<Real>> Op;
    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>* ker_ptr;

    sctl::StaticArray<sctl::Long,2> dim;
    sctl::Vector<Real> normal_orient;
    sctl::Vector<sctl::Vector<Real>> Xtrg, Xsrc, dXsrc, Xn_src, Xa_src;
    sctl::Vector<sctl::Vector<SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1,HedgehogOrder>>> singular_correction;
    sctl::Vector<sctl::Vector<sctl::Long>> trg_idx;
};

/**
 * Compute layer-potentials on toroidal surfaces with field-period symmetry.
 *
 * @tparam Real datatype for reals can be float, double or sctl::QuadReal.
 *
 * @tparam KDIM0 degrees-of-freedom of the density per source point.
 *
 * @tparam KDIM1 degrees-of-freedom of the potential per target point.
 *
 * @tparam HedgehogOrder Use Hedgehog quadrature (required for hyper-singular kernels) when HedgehogOrder > 1
 */
template <class Real, sctl::Integer COORD_DIM=3, sctl::Integer KDIM0=1, sctl::Integer KDIM1=1, sctl::Integer HedgehogOrder = 1> class FieldPeriodBIOp {
  public:

    explicit FieldPeriodBIOp(const sctl::Comm& comm = sctl::Comm::Self());

    ~FieldPeriodBIOp();

    /**
     * Build the surface object from data surface nodal data on one field period.
     *
     * @param[in] X the surface coordinates (in one field period) in the order
     * {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}.
     *
     * @param[in] NFP number of toroidal field periods. The surface as well as
     * the magnetic field must have this toroidal periodic symmetry.
     *
     * @param[in] Nt surface discretization order in toroidal direction (in one field period).
     *
     * @param[in] Np surface discretization order in poloidal direction.
     */
    static biest::Surface<Real> BuildSurface(const sctl::Vector<Real>& X, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np);

    /**
     * Setup layer potential operator.
     *
     * @param[in] Svec vector of surface objects (currently, only one surface is supported)
     *
     * @param[in] ker kernel function object.
     *
     * @param[in] digits number of decimal digits of accuracy.
     *
     * @param[in] NFP number of toroidal field periods. The output potential will only be computed for one field-period.
     *
     * @param[in] src_Nt maximum discretization order for input density in toroidal direction (in one field period).
     *
     * @param[in] src_Np maximum discretization order for input density in poloidal direction.
     *
     * @param[in] trg_Nt output potential discretization order in toroidal direction (in one field period).
     *
     * @param[in] trg_Np output potential discretization order in poloidal direction.
     */
    void SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long qNt = 0, const sctl::Long qNp = 0);

    /**
     * Evaluate the potential from given density.
     *
     * @param[in] U output potential computed on surface nodes in one field-period.
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
     * will be re-sampled to the resolution QuadNt() x QuadNp() for evaluating the quadratures. The user can avoid this
     * extra work by providing F already sampled at this resolution (i.e. NFP=1, src_Nt=QuadNt(), src_Np=QuadNp()).
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP=1, const sctl::Integer src_Nt=-1, const sctl::Integer src_Np=-1) const;

    /**
     * Returns the toroidal resolution of the source data for quadrature evaluation.
     */
    sctl::Long QuadNt() const { return quad_Nt_; }

    /**
     * Returns the poloidal resolution of the source data for quadrature evaluation.
     */
    sctl::Long QuadNp() const { return quad_Np_; }

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

    template <sctl::Integer PDIM, sctl::Integer RDIM> static void* BIOpBuild(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Comm& comm, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx);
    template <sctl::Integer PDIM, sctl::Integer RDIM> static void BIOpDelete(void** self);
    template <sctl::Integer PDIM, sctl::Integer RDIM> static void BIOpEval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, void* self);

    template <sctl::Integer PDIM, sctl::Integer RDIM> void SetupSingular0(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx);
    void SetupSingular_(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer PDIM_, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx);

    void* biop;
    void* (*biop_build)(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Comm& comm, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx);
    void (*biop_delete)(void** self);
    void (*biop_eval)(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, void* self);
    sctl::Integer NFP_;
    sctl::Long trg_Nt_, trg_Np_;
    sctl::Long quad_Nt_, quad_Np_;
    sctl::Vector<biest::Surface<Real>> Svec_;
    sctl::Comm comm_;
};

}

#include <biest/boundary_integ_op.txx>

#endif  //_BOUNDARY_INTEG_OP_HPP_
