#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <algorithm>
#include <functional>
#include <type_traits>

#include <sctl.hpp>

namespace biest {

template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> class KernelFunction {
    typedef void (KerFn)(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx);

    typedef void (MatFn)(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& r_trg, sctl::Matrix<Real>& M, const void* ctx);

  public:

    /**
     * Constructor.
     *
     * @param[in] kerfn_ the kernel evaluation function.
     *
     * @param[in] cost_ the cost in FLOPs per kernel evaluation (used for profiling).
     *
     * @param[in] ctx_ a context pointer passed to the kernel evaluation
     * function.
     */
    KernelFunction(std::function<KerFn> kerfn_, sctl::Long cost_, const void* ctx_) : kerfn(kerfn_), ctx(ctx_), cost(cost_) {}

    /**
     * Constructor, with a specialized kernel-matrix builder.
     *
     * @param[in] matfn_ builds the kernel matrix for all sources at once. When
     * omitted, the matrix is assembled by evaluating kerfn_ once per source and
     * per source degree-of-freedom, which costs one call per (source, KDIM0)
     * pair.
     */
    KernelFunction(std::function<KerFn> kerfn_, std::function<MatFn> matfn_, sctl::Long cost_, const void* ctx_) : kerfn(kerfn_), matfn(matfn_), ctx(ctx_), cost(cost_) {}

    /**
     * Gives the number of degrees-of-freedom per-source or target.
     *
     * @param[in] i integrer value in {0, 1} which indicates sources (for 0) and
     * targets (for 1).
     *
     * @return the number of degrees-of-freedom per-source (when i==0) and
     * per-target (when i==1).
     */
    static constexpr sctl::Integer Dim(sctl::Integer i) { return i == 0 ? KDIM0 : KDIM1; }

    /**
     * Evaluates the kernel function.
     *
     * @param[in] r_src the coordinates of the source points in
     * structure-of-array (SoA) order i.e.  {x1, ..., xn, y1, ..., z1, ...}.
     *
     * @param[in] n_src the normals at each source point in SoA order.
     *
     * @param[in] v_src the density values at each source point in SoA order.
     *
     * @param[in] r_trg the coordinates of the target points in SoA order.
     *
     * @param[out] v_trg the computed potential values at each target point in
     * SoA order.
     */
    void operator()(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg) const {
      sctl::Long Ns = r_src.Dim() / COORD_DIM;
      sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      sctl::Long dof = v_src.Dim() / (Ns ? Ns : 1) / Dim(0);
      assert(v_src.Dim() == dof * Dim(0) * Ns);
      assert(n_src.Dim() == COORD_DIM * Ns);
      assert(r_src.Dim() == COORD_DIM * Ns);
      assert(r_trg.Dim() == COORD_DIM * Nt);
      if(v_trg.Dim() != dof * Dim(1) * Nt) {
        v_trg.ReInit(dof * Dim(1) * Nt);
        v_trg = 0;
      }
      kerfn(r_src, n_src, v_src, r_trg, v_trg, 0, ctx);
      sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, Ns * Nt * cost);
    }

    /**
     * Build the kernel matrix.
     *
     * @param[in] r_src the coordinates of the source points in
     * structure-of-array (SoA) order i.e.  {x1, ..., xn, y1, ..., z1, ...}.
     *
     * @param[in] n_src the normals at each source point in SoA order.
     *
     * @param[in] r_trg the coordinates of the target points in SoA order.
     *
     * @param[out] M the kernel matrix of dimensions (KDIM0*Ns)x(KDIM1*Nt),
     * where Ns is the number of sources, Nt is the number of targets, KDIM0 is
     * the number of degrees-of-freedom (DOF) per source and KDIM1 is the number
     * of DOF per target.
     *
     */
    void BuildMatrix(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& r_trg, sctl::Matrix<Real>& M) const {
      sctl::Long Ns = r_src.Dim() / COORD_DIM;
      sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      if (M.Dim(0) != Ns * Dim(0) || M.Dim(1) != Nt * Dim(1)) {
        M.ReInit(Ns * Dim(0), Nt * Dim(1));
      }

      if (matfn) {
        matfn(r_src, n_src, r_trg, M, ctx);
      } else { // one kerfn call per (source, source-DOF) pair
        sctl::Integer omp_p = omp_get_max_threads();
        #pragma omp parallel for schedule(static)
        for (sctl::Integer tid = 0; tid < omp_p; tid++) {
          sctl::Long s0 = (tid + 0) * Ns / omp_p;
          sctl::Long s1 = (tid + 1) * Ns / omp_p;

          sctl::StaticArray<Real,COORD_DIM> r_src0_;
          sctl::StaticArray<Real,COORD_DIM> n_src0_;
          sctl::StaticArray<Real,   Dim(0)> f_src0_;
          sctl::Vector<Real> r_src0(COORD_DIM, r_src0_, false);
          sctl::Vector<Real> n_src0(COORD_DIM, n_src0_, false);
          sctl::Vector<Real> f_src0(   Dim(0), f_src0_, false);
          f_src0 = 0;

          for (sctl::Long s = s0; s < s1; s++) {
            for (sctl::Integer i = 0; i < COORD_DIM; i++) {
              r_src0[i] = r_src[i * Ns + s];
              n_src0[i] = n_src[i * Ns + s];
            }
            for (sctl::Integer k = 0; k < Dim(0); k++) {
              f_src0[k] = 1;
              sctl::Vector<Real> v_trg(M.Dim(1), M[k * Ns + s], false);
              v_trg = 0;
              kerfn(r_src0, n_src0, f_src0, r_trg, v_trg, 1, ctx);
              f_src0[k] = 0;
            }
          }
        }
      }
      sctl::Profile::IncrementCounter(sctl::ProfileCounter::FLOP, Ns * Nt * cost);
    }

  private:
    std::function<KerFn> kerfn;
    std::function<MatFn> matfn;
    const void* ctx;
    sctl::Long cost;
};

namespace kernel_impl {

  // Detects the optional fused apply a micro-kernel may provide (see
  // sctl/kernel_functions.hpp). Local to BIEST: SCTL keeps its own copy of this
  // as an implementation detail of GenericKernel::Eval.
  template <class uKer, class = void> struct FusedApply : std::false_type {};
  template <class uKer> struct FusedApply<uKer, std::void_t<decltype(uKer::FUSED_APPLY)>> : std::bool_constant<uKer::FUSED_APPLY> {};

}  // namespace kernel_impl

/**
 * Evaluates a kernel in BIEST's structure-of-array layout with dof density
 * blocks, built from an SCTL kernel (one of the GenericKernel aliases published
 * by sctl/kernel_functions.hpp, e.g. sctl::Laplace3D_FxU).
 *
 * The underlying micro-kernel may additionally provide
 *
 *   static constexpr bool FUSED_APPLY = true;
 *   template <Integer digits, Integer DOF, class VecType>
 *   static void uKerApply(VecType* v, const VecType (&r)[COORD_DIM], const VecType* n, const VecType* f, const void* ctx);
 *
 * which applies the density directly instead of going through the KDIM0 x KDIM1
 * matrix. This is worth doing when that matrix has structural zeros or repeated
 * entries (antisymmetric or complex-valued kernels), where the generic apply
 * would spend multiply-adds on entries it cannot see are redundant. uKerApply
 * must accumulate the same unscaled values as uKerMatrix.
 *
 * @tparam DIGITS digits of accuracy for the reciprocal square root; -1 selects
 * machine precision.
 */
template <class Real, class SKer, sctl::Integer DIGITS, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> struct KerWrapper {
    using RealVec = sctl::Vec<Real,Nv>;

    static constexpr sctl::Integer COORD_DIM = SKer::CoordDim();
    static constexpr sctl::Integer NDIM  = SKer::NormalDim();
    static constexpr sctl::Integer NDIM_ = (NDIM ? NDIM : 1); // non-zero
    static constexpr sctl::Integer KDIM0 = SKer::SrcDim();
    static constexpr sctl::Integer KDIM1 = SKer::TrgDim();
    static constexpr sctl::Integer DIGITS_ = (DIGITS < 0 ? (sctl::Integer)(sctl::TypeTraits<Real>::SigBits*0.3010299957) : DIGITS);
    static constexpr bool FUSED = kernel_impl::FusedApply<SKer>::value;

    static void Eval(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx) {
      const sctl::Long Ns = r_src.Dim() / COORD_DIM;
      const sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      const sctl::Long dof = v_src.Dim() / (Ns ? Ns : 1) / KDIM0;
      const sctl::Long NNt = ((Nt + Nv - 1) / Nv) * Nv;
      if (!Ns || !Nt || !dof) return;

      sctl::ScratchBuf<Real> buff(COORD_DIM * NNt);
      sctl::Matrix<Real> Xt(COORD_DIM, NNt, buff.begin(), false);
      for (sctl::Long k = 0; k < COORD_DIM; k++) { // Set Xt, zero-padded to a multiple of Nv
        for (sctl::Long i = 0; i < Nt; i++) Xt[k][i] = r_trg[k * Nt + i];
        for (sctl::Long i = Nt; i < NNt; i++) Xt[k][i] = 0;
      }

      if (dof == 1) eval<1>(v_trg, Xt, r_src, n_src, v_src, Ns, Nt, Nthread, ctx);
      else if (dof == 2) eval<2>(v_trg, Xt, r_src, n_src, v_src, Ns, Nt, Nthread, ctx);
      else if (dof == 3) eval<3>(v_trg, Xt, r_src, n_src, v_src, Ns, Nt, Nthread, ctx);
      else SCTL_ASSERT(false);
    }

    /**
     * One uKerMatrix call per (source-block, target) gives the whole
     * KDIM0 x KDIM1 block, with the source index vectorized.
     */
    static void BuildMatrix(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& r_trg, sctl::Matrix<Real>& M, const void* ctx) {
      const sctl::Long Ns = r_src.Dim() / COORD_DIM;
      const sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      const Real scal = SKer::template uKerScaleFactor<Real>();
      const sctl::Long NNs = ((Ns + Nv - 1) / Nv) * Nv;
      SCTL_ASSERT(M.Dim(0) == Ns * KDIM0 && M.Dim(1) == Nt * KDIM1);
      if (!Ns || !Nt) return;

      sctl::ScratchBuf<Real> buff((COORD_DIM + NDIM_) * NNs);
      sctl::Matrix<Real> Xs(COORD_DIM, NNs, buff.begin(), false);
      sctl::Matrix<Real> Xn(   NDIM_, NNs, buff.begin() + COORD_DIM * NNs, false);
      for (sctl::Long k = 0; k < COORD_DIM; k++) { // Set Xs, zero-padded to a multiple of Nv
        for (sctl::Long i = 0; i < Ns; i++) Xs[k][i] = r_src[k * Ns + i];
        for (sctl::Long i = Ns; i < NNs; i++) Xs[k][i] = 0;
      }
      for (sctl::Long k = 0; k < NDIM; k++) { // Set Xn
        for (sctl::Long i = 0; i < Ns; i++) Xn[k][i] = n_src[k * Ns + i];
        for (sctl::Long i = Ns; i < NNs; i++) Xn[k][i] = 0;
      }

      #pragma omp parallel for schedule(static)
      for (sctl::Long s = 0; s < NNs; s += Nv) {
        RealVec xs[COORD_DIM], ns[NDIM_], dX[COORD_DIM], U[KDIM0][KDIM1];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::LoadAligned(&Xs[k][s]);
        for (sctl::Integer k = 0; k < NDIM; k++) ns[k] = RealVec::LoadAligned(&Xn[k][s]);

        alignas(sizeof(RealVec)) sctl::StaticArray<Real,Nv> out;
        const sctl::Long s1 = std::min(s + Nv, Ns);
        for (sctl::Long t = 0; t < Nt; t++) {
          for (sctl::Integer k = 0; k < COORD_DIM; k++) dX[k] = RealVec::Load1(&r_trg[k * Nt + t]) - xs[k];
          SKer::template uKerMatrix<DIGITS_>(U, dX, ns, ctx);
          for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
            for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
              U[k0][k1].StoreAligned(&out[0]);
              for (sctl::Long i = s; i < s1; i++) M[k0 * Ns + i][k1 * Nt + t] = out[i - s] * scal;
            }
          }
        }
      }
    }

  private:

    template <sctl::Integer DOF> static void eval(sctl::Vector<Real>& v_trg, const sctl::Matrix<Real>& Xt, const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, sctl::Long Ns, sctl::Long Nt, sctl::Integer Nthread, const void* ctx) {
      const Real scal = SKer::template uKerScaleFactor<Real>();
      const sctl::Long NNt = Xt.Dim(1);

      auto trg_blk = [&](sctl::Long t) {
        RealVec xt[COORD_DIM], dX[COORD_DIM], ns[NDIM_], vs[DOF*KDIM0], vt[DOF*KDIM1];
        for (sctl::Integer k = 0; k < DOF*KDIM1; k++) vt[k] = RealVec::Zero();
        for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt[k][t]);
        for (sctl::Long s = 0; s < Ns; s++) {
          for (sctl::Integer k = 0; k < COORD_DIM; k++) dX[k] = xt[k] - RealVec::Load1(&r_src[k * Ns + s]);
          for (sctl::Integer k = 0; k < NDIM; k++) ns[k] = RealVec::Load1(&n_src[k * Ns + s]);
          for (sctl::Integer k = 0; k < DOF*KDIM0; k++) vs[k] = RealVec::Load1(&v_src[k * Ns + s]);
          if constexpr (FUSED) {
            SKer::template uKerApply<DIGITS_,DOF>(vt, dX, ns, vs, ctx);
          } else {
            RealVec U[KDIM0][KDIM1];
            SKer::template uKerMatrix<DIGITS_>(U, dX, ns, ctx);
            for (sctl::Integer d = 0; d < DOF; d++) {
              for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
                for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) vt[d*KDIM1+k1] = sctl::FMA(U[k0][k1], vs[d*KDIM0+k0], vt[d*KDIM1+k1]);
              }
            }
          }
        }
        alignas(sizeof(RealVec)) sctl::StaticArray<Real,Nv> out;
        const sctl::Long t1 = std::min(t + Nv, Nt);
        for (sctl::Integer k = 0; k < DOF*KDIM1; k++) {
          vt[k].StoreAligned(&out[0]);
          for (sctl::Long i = t; i < t1; i++) v_trg[k * Nt + i] += out[i - t] * scal;
        }
      };

      if (Nthread == 1) {
        for (sctl::Long t = 0; t < NNt; t += Nv) trg_blk(t);
      } else {
        #pragma omp parallel for schedule(static)
        for (sctl::Long t = 0; t < NNt; t += Nv) trg_blk(t);
      }
    }
};

template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Stokes3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 3;
  static constexpr sctl::Integer KDIM1 = 3;

  public:
    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& DxU() {
      using Ker = KerWrapper<Real, sctl::Stokes3D_DxU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(Ker::Eval, Ker::BuildMatrix, 31, nullptr);
      return ker;
    }
};

template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Laplace3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 1;
  static constexpr sctl::Integer KDIM1 = 1;

  public:
    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() {
      using Ker = KerWrapper<Real, sctl::Laplace3D_FxU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(Ker::Eval, Ker::BuildMatrix, 12, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() {
      using Ker = KerWrapper<Real, sctl::Laplace3D_FxdU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker(Ker::Eval, Ker::BuildMatrix, 19, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM*COORD_DIM>& Fxd2U() {
      using Ker = KerWrapper<Real, sctl::Laplace3D_Fxd2U, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM*COORD_DIM> ker(Ker::Eval, Ker::BuildMatrix, 61, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& DxU() {
      using Ker = KerWrapper<Real, sctl::Laplace3D_DxU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(Ker::Eval, Ker::BuildMatrix, 20, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& DxdU() {
      using Ker = KerWrapper<Real, sctl::Laplace3D_DxdU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker(Ker::Eval, Ker::BuildMatrix, 39, nullptr);
      return ker;
    }
};

template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class BiotSavart3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 3;
  static constexpr sctl::Integer KDIM1 = 3;

  public:
    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() {
      using Ker = KerWrapper<Real, sctl::BiotSavart3D_FxU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(Ker::Eval, Ker::BuildMatrix, 27, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() {
      using Ker = KerWrapper<Real, sctl::BiotSavart3D_FxdU, ORDER, Nv>;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker(Ker::Eval, Ker::BuildMatrix, 127, nullptr);
      return ker;
    }
};

template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Helmholtz3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  using KerFxU = KerWrapper<Real, sctl::Helmholtz3D_FxU, ORDER, Nv>;
  using KerDxU = KerWrapper<Real, sctl::Helmholtz3D_DxU, ORDER, Nv>;
  using KerFxdU = KerWrapper<Real, sctl::Helmholtz3D_FxdU, ORDER, Nv>;

  public:

    Helmholtz3D(Real k) :
      k_(k),
      ker_FxU(KerFxU::Eval, KerFxU::BuildMatrix, 24, &k_),
      ker_DxU(KerDxU::Eval, KerDxU::BuildMatrix, 38, &k_),
      ker_FxdU(KerFxdU::Eval, KerFxdU::BuildMatrix, 41, &k_)
      {}

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() const { return ker_FxU; }

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() const { return ker_FxdU; }

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& DxU() const { return ker_DxU; }

  private:

    Real k_;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker_FxU, ker_DxU;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker_FxdU;
};

template <class Real> class HelmholtzDiff3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  using KerFxdU = KerWrapper<Real, sctl::HelmholtzDiff3D_FxdU, -1>;

  public:

    HelmholtzDiff3D(Real k) :
      k_(k),
      ker_FxdU(KerFxdU::Eval, KerFxdU::BuildMatrix, 47, &k_) {}

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() const { return ker_FxdU; }

  private:

    Real k_;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker_FxdU;
};

}

#endif  //_KERNEL_HPP_
