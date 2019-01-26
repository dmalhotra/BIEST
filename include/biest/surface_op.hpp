#ifndef _SURFACE_OP_HPP_
#define _SURFACE_OP_HPP_

#include <biest/surface.hpp>
#include <biest/singular_correction.hpp>
#include <sctl.hpp>

namespace biest {

template <class Real> class SurfaceOp {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    SurfaceOp(const sctl::Comm& comm = sctl::Comm::Self(), sctl::Long Nt = 0, sctl::Long Np = 0);

    SurfaceOp(const SurfaceOp& Op);

    SurfaceOp& operator=(const SurfaceOp& Op);

    static void Upsample(const sctl::Vector<Real>& X0_, sctl::Long Nt0, sctl::Long Np0, sctl::Vector<Real>& X1_, sctl::Long Nt1, sctl::Long Np1);

    void Grad2D(sctl::Vector<Real>& dX, const sctl::Vector<Real>& X) const;

    Real SurfNormalAreaElem(sctl::Vector<Real>* normal, sctl::Vector<Real>* area_elem, const sctl::Vector<Real>& dX, const sctl::Vector<Real>* X) const;

    void SurfGrad(sctl::Vector<Real>& GradFvec, const sctl::Vector<Real>& dXvec_, const sctl::Vector<Real>& Fvec) const;

    void SurfDiv(sctl::Vector<Real>& DivF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    void SurfLap(sctl::Vector<Real>& LapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    void SurfInteg(sctl::Vector<Real>& I, const sctl::Vector<Real>& area_elem, const sctl::Vector<Real>& F) const;

    void ProjZeroMean(sctl::Vector<Real>& Fproj, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F) const;

    void InvSurfLap(sctl::Vector<Real>& InvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1, Real upsample = 1.0) const;

    void GradInvSurfLap(sctl::Vector<Real>& GradInvLapF, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& d2X, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1, Real upsample = 1.0) const;

    void InvSurfLapPrecond(sctl::Vector<Real>& InvLapF, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> LeftPrecond, std::function<void(sctl::Vector<Real>&, const sctl::Vector<Real>&)> RightPrecond, const sctl::Vector<Real>& dX, const sctl::Vector<Real>& F, Real tol, sctl::Integer max_iter = -1) const;

    template <sctl::Integer KDIM0, sctl::Integer KDIM1> void EvalSurfInteg(sctl::Vector<Real>& Utrg, const sctl::Vector<Real>& Xtrg, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& Xn_src, const sctl::Vector<Real>& Xa_src, const sctl::Vector<Real>& Fsrc, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker) const;

    template <class SingularCorrection, class Kernel> void SetupSingularCorrection(sctl::Vector<SingularCorrection>& singular_correction, sctl::Integer TRG_SKIP, const sctl::Vector<Real>& Xsrc, const sctl::Vector<Real>& dXsrc, const Kernel& ker, Real normal_scal = 1.0) const;

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
