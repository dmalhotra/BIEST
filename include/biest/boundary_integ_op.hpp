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
template <class Real, sctl::Integer KDIM0, sctl::Integer KDIM1, sctl::Integer UPSAMPLE = 1, sctl::Integer PATCH_DIM0 = 25, sctl::Integer RAD_DIM = 18> class BoundaryIntegralOp {
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
    sctl::Vector<Real> normal_scal;
    sctl::Vector<sctl::Vector<Real>> Xtrg, Xsrc, dXsrc, Xn_src, Xa_src;
    sctl::Vector<sctl::Vector<SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>>> singular_correction;
    sctl::Vector<sctl::Vector<sctl::Long>> trg_idx;
};

template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> class BIOpWrapper {
  public:

    BIOpWrapper(const sctl::Comm& comm);

    ~BIOpWrapper();

    void SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const biest::KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long qNt = 0, const sctl::Long qNp = 0);

    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

    sctl::Long QuadNt() const { return quad_Nt_; }
    sctl::Long QuadNp() const { return quad_Np_; }

  private:

    static void Resample(sctl::Vector<Real>& X1, const sctl::Long Nt1, const sctl::Long Np1, const sctl::Vector<Real>& X0, const sctl::Long Nt0, const sctl::Long Np0);

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
