#ifndef _SURFACE_OP_HPP_
#define _SURFACE_OP_HPP_

#include <biest/surface.hpp>
#include <biest/singular_correction.hpp>
#include <sctl.hpp>

namespace biest {

/**
 * Implements various operations on surfaces.
 */
template <class Real> class SurfaceOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructor.
     *
     * @param[in] comm the MPI communicator (optional and not fully supported
     * yet).
     *
     * @param[in] Nt number of surface discretization points in the toroidal
     * direction.
     *
     * @param[in] Np number of surface discretization points in the poloidal
     * direction.
     */
    SurfaceOp(const sctl::Comm& comm = sctl::Comm::Self(), sctl::Long Nt = 0, sctl::Long Np = 0);

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
     * @param[in] Np0 the number of discretization points in polodial direction
     * for the input data.
     *
     * @param[out] X1_ the vector of output data in SoA order on a Nt1xNp1 grid in
     * row-major order.
     *
     * @param[in] Nt1 the number of discretization points in toroidal direction
     * for the output data.
     *
     * @param[in] Np1 the number of discretization points in polodial direction
     * for the output data.
     */
    static void Upsample(const sctl::Vector<Real>& X0_, sctl::Long Nt0, sctl::Long Np0, sctl::Vector<Real>& X1_, sctl::Long Nt1, sctl::Long Np1);

    /**
     * Computes the gradient of data X(t,p) defined on a periodic 2D grid.
     *
     * @param[in] X the vector of input data X(t,p) sampled on a grid of
     * dimensions NtxNp stored in row-major order and as a structure-of-arrays.
     *
     * @param[out] dX the output vector containing {dX/dt, dX/dp} for each grid
     * point stored in SoA order.
     */
    void Grad2D(sctl::Vector<Real>& dX, const sctl::Vector<Real>& X) const;

    /**
     * Computes the surface normal and differential area element at each grid
     * point.
     *
     * @param[out] normal pointer to the vector where the per grid point surface
     * normals will be written in SoA order.
     *
     * @param[out] area_elem pointer to the vector where the per grid point
     * differential area-elements (|dX/ds x dX/dp|) will be written in SoA
     * order.
     *
     * @param[in] dX the vector of the surface gradient of the position vector
     * at each grid point in SoA order.
     *
     * @param[in] X (optional) the vector of position/coordinate values of each
     * grid point in SoA order.  This is used only to determine the correct
     * orientation of the normal vectors (i.e. outward from the surface).
     */
    Real SurfNormalAreaElem(sctl::Vector<Real>* normal, sctl::Vector<Real>* area_elem, const sctl::Vector<Real>& dX, const sctl::Vector<Real>* X) const;

    /**
     * Computes the on-surface curl operation.
     *
     * @param[out] CurlF the output vector containing the curl of F at each grid
     * point stored in SoA order.
     *
     * @param[in] dX the vector of the surface gradient of the position vector
     * at each grid point in SoA order.
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
     * @param[out] GradFvec the output vector containing the gradient of F at
     * each grid point stored in SoA order.
     *
     * @param[in] dX the vector of the surface gradient of the position vector
     * at each grid point in SoA order.
     *
     * @param[in] normal the vector of surface normals at each grid point in SoA
     * order.
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfGrad(sctl::Vector<Real>& GradFvec, const sctl::Vector<Real>& dXvec_, const sctl::Vector<Real>& Fvec) const;

    /**
     * Computes the on-surface divergence operation.
     *
     * @param[out] DivF the output vector containing the divergence of F at each
     * grid point stored in SoA order.
     *
     * @param[in] dX the vector of the surface gradient of the position vector
     * at each grid point in SoA order.
     *
     * @param[in] normal the vector of surface normals at each grid point in SoA
     * order.
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfDiv(sctl::Vector<Real>& DivF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    /**
     * Computes the on-surface Laplacian operation.
     *
     * @param[out] LapF the output vector containing the laplacian of F at each grid
     * point stored in SoA order.
     *
     * @param[in] dX the vector of the surface gradient of the position vector
     * at each grid point in SoA order.
     *
     * @param[in] normal the vector of surface normals at each grid point in SoA
     * order.
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfLap(sctl::Vector<Real>& LapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    /**
     * Compute the surface integral of a function (on periodic NtxNp grid) using
     * trapezoidal rule.
     *
     * @param[out] the vector containing the output surface integral.
     *
     * @param[out] area_elem the vector of differential area-element values at
     * each grid point.
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void SurfInteg(sctl::Vector<Real>& I, const sctl::Vector<Real>& area_elem, const sctl::Vector<Real>& F) const;

    /**
     * Subtracts the average value of a given function on a period NtxNp grid.
     *
     * @param[out] Fproj the output mean-zero function at each grid point.
     *
     * @param[in] dX the vector of the surface gradient of the position vector
     * at each grid point in SoA order.
     *
     * @param[in] F the vector of data values at each grid point in SoA order.
     */
    void ProjZeroMean(sctl::Vector<Real>& Fproj, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    void InvSurfLap(sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1, Real upsample = 1.0) const;

    void GradInvSurfLap(sctl::Vector<Real>& GradInvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1, Real upsample = 1.0) const;

    void InvSurfLapPrecond(sctl::Vector<Real>& InvLapF, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> LeftPrecond, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> RightPrecond, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1) const;

    template <sctl::Integer KDIM0, sctl::Integer KDIM1> void EvalSurfInteg(sctl::Vector<Real>& Utrg, const sctl::Vector<Real>& Xtrg, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& Xn_src, const sctl::Vector<Real>& Xa_src, const sctl::Vector<Real>& Fsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker) const;

    template <class SingularCorrection, class Kernel> void SetupSingularCorrection(sctl::Vector<SingularCorrection>& singular_correction, sctl::Integer TRG_SKIP, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& dXsrc, const Kernel& ker, const Real normal_scal, const sctl::Vector<sctl::Long>& trg_idx) const;

    template <class SingularCorrection> void EvalSingularCorrection(sctl::Vector<Real>& U, const sctl::Vector<SingularCorrection>& singular_correction, sctl::Integer kdim0, sctl::Integer kdim1, const sctl::Vector<Real>& F) const;

    void HodgeDecomp(sctl::Vector<Real>& Vn, sctl::Vector<Real>& Vd, sctl::Vector<Real>& Vc, sctl::Vector<Real>& Vh, const sctl::Vector<Real>& V, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X, const sctl::Vector<Real>& normal, Real tol, sctl::Long max_iter = -1) const;



    static void test_SurfGrad(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm);

    static void test_SurfLap(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm);

    static void test_InvSurfLap(sctl::Long Nt, sctl::Long Np, SurfType surf_type, Real gmres_tol, sctl::Long gmres_iter, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>* LeftPrecond, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)>* RightPrecond, const sctl::Comm& comm);

    static void test_HodgeDecomp(sctl::Long Nt, sctl::Long Np, SurfType surf_type, const sctl::Comm& comm);

  private:

    void Init(const sctl::Comm& comm, sctl::Long Nt, sctl::Long Np);

    static Real compute_area_elem(sctl::StaticArray<Real, COORD_DIM>& xn, const sctl::StaticArray<Real, COORD_DIM>& xt, const sctl::StaticArray<Real, COORD_DIM>& xp);

    static Real max_norm(const sctl::Vector<Real>& x);

    void LaplaceBeltramiReference(sctl::Vector<Real>& f0, sctl::Vector<Real>& u0, const sctl::Vector<Real>& X, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X) const;

    sctl::Comm comm_;
    sctl::Long Nt_, Np_;
    mutable sctl::FFT<Real> fft_r2c, fft_c2r;
    mutable sctl::ParallelSolver<Real> solver;
};

}

#include <biest/surface_op.txx>

#endif  //_SURFACE_OP_HPP_
