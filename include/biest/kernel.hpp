// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <algorithm>
#include <functional>
#include <type_traits>

#include <sctl.hpp>

namespace biest {

/**
 * The kernel function interface, with KDIM0 density components per source
 * point and KDIM1 potential components per target point.
 */
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
     *
     * @param[in] kerfn_, cost_, ctx_ as in the constructor above.
     */
    KernelFunction(std::function<KerFn> kerfn_, std::function<MatFn> matfn_, sctl::Long cost_, const void* ctx_) : kerfn(kerfn_), matfn(matfn_), ctx(ctx_), cost(cost_) {}

    /**
     * Gives the number of degrees-of-freedom per-source or target.
     *
     * @param[in] i integer value in {0, 1} which indicates sources (for 0) and
     * targets (for 1).
     *
     * @return the number of degrees-of-freedom per-source (when i==0) and
     * per-target (when i==1).
     */
    static constexpr sctl::Integer Dim(sctl::Integer i) { return i == 0 ? KDIM0 : KDIM1; }

    /**
     * Evaluates the kernel function by direct summation, and adds the
     * potential to v_trg.
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
     * @param[in,out] v_trg the computed potential values at each target point
     * in SoA order; it is resized and set to zero first if its size is wrong.
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

/**
 * A kernel function built from an SCTL kernel. It also keeps the SCTL kernel, for use with sctl::BoundaryIntegralOp.
 *
 * @tparam SKer the SCTL kernel, one of the GenericKernel aliases in sctl/kernel_functions.hpp.
 *
 * @tparam DIGITS digits of accuracy for the reciprocal square root; -1 selects machine precision.
 */
template <class Real, class SKer, sctl::Integer DIGITS, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class SctlKernelFunction : public KernelFunction<Real, SKer::CoordDim(), SKer::SrcDim(), SKer::TrgDim()> {
    using Ker = KerWrapper<Real, SKer, DIGITS, Nv>;

  public:

    /**
     * Constructor.
     *
     * @param[in] cost_ the cost in FLOPs per kernel evaluation (used for profiling).
     *
     * @param[in] ctx_ a context pointer passed to the kernel evaluation function.
     */
    SctlKernelFunction(sctl::Long cost_, const void* ctx_) : KernelFunction<Real, SKer::CoordDim(), SKer::SrcDim(), SKer::TrgDim()>(Ker::Eval, Ker::BuildMatrix, cost_, ctx_) {
      sctl_ker.SetCtxPtr(const_cast<void*>(ctx_));
    }

    /**
     * Returns the SCTL kernel, with the context pointer of this kernel function.
     */
    const SKer& GetSctlKernel() const { return sctl_ker; }

  private:

    SKer sctl_ker;
};

/**
 * Stokes kernels. The kernel objects are static.
 *
 * @tparam ORDER digits of accuracy of the reciprocal square root in the
 * kernel evaluation.
 *
 * @tparam Nv the SIMD vector length.
 */
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Stokes3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 3;
  static constexpr sctl::Integer KDIM1 = 3;

  public:
    /**
     * Stokes double layer; 3 density and 3 potential components.
     */
    static SctlKernelFunction<Real,sctl::Stokes3D_DxU,ORDER,Nv>& DxU() {
      static SctlKernelFunction<Real,sctl::Stokes3D_DxU,ORDER,Nv> ker(31, nullptr);
      return ker;
    }
};

/**
 * Laplace kernels, with r = x - y for a target point x and a source point y
 * with normal n. The kernel objects are static.
 *
 * @tparam ORDER digits of accuracy of the reciprocal square root in the
 * kernel evaluation.
 *
 * @tparam Nv the SIMD vector length.
 */
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Laplace3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 1;
  static constexpr sctl::Integer KDIM1 = 1;

  public:
    /**
     * Single layer 1/(4 pi |r|); 1 density and 1 potential component.
     */
    static SctlKernelFunction<Real,sctl::Laplace3D_FxU,ORDER,Nv>& FxU() {
      static SctlKernelFunction<Real,sctl::Laplace3D_FxU,ORDER,Nv> ker(12, nullptr);
      return ker;
    }

    /**
     * Gradient of the single layer with respect to x, -r/(4 pi |r|^3); 1
     * density and 3 potential components.
     */
    static SctlKernelFunction<Real,sctl::Laplace3D_FxdU,ORDER,Nv>& FxdU() {
      static SctlKernelFunction<Real,sctl::Laplace3D_FxdU,ORDER,Nv> ker(19, nullptr);
      return ker;
    }

    /**
     * Hessian of the single layer with respect to x; 1 density and 9
     * potential components.
     */
    static SctlKernelFunction<Real,sctl::Laplace3D_Fxd2U,ORDER,Nv>& Fxd2U() {
      static SctlKernelFunction<Real,sctl::Laplace3D_Fxd2U,ORDER,Nv> ker(61, nullptr);
      return ker;
    }

    /**
     * Double layer n.r/(4 pi |r|^3); 1 density and 1 potential component.
     */
    static SctlKernelFunction<Real,sctl::Laplace3D_DxU,ORDER,Nv>& DxU() {
      static SctlKernelFunction<Real,sctl::Laplace3D_DxU,ORDER,Nv> ker(20, nullptr);
      return ker;
    }

    /**
     * Gradient of the double layer with respect to x; 1 density and 3
     * potential components.
     */
    static SctlKernelFunction<Real,sctl::Laplace3D_DxdU,ORDER,Nv>& DxdU() {
      static SctlKernelFunction<Real,sctl::Laplace3D_DxdU,ORDER,Nv> ker(39, nullptr);
      return ker;
    }
};

/**
 * Biot-Savart kernels, with r = x - y for a target point x and a source point
 * y. The kernel objects are static.
 *
 * @tparam ORDER digits of accuracy of the reciprocal square root in the
 * kernel evaluation.
 *
 * @tparam Nv the SIMD vector length.
 */
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class BiotSavart3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 3;
  static constexpr sctl::Integer KDIM1 = 3;

  public:
    /**
     * Biot-Savart law f x r/(4 pi |r|^3) for the density f; 3 density and 3
     * potential components.
     */
    static SctlKernelFunction<Real,sctl::BiotSavart3D_FxU,ORDER,Nv>& FxU() {
      static SctlKernelFunction<Real,sctl::BiotSavart3D_FxU,ORDER,Nv> ker(27, nullptr);
      return ker;
    }

    /**
     * Gradient of the Biot-Savart field with respect to x; 3 density and 9
     * potential components.
     */
    static SctlKernelFunction<Real,sctl::BiotSavart3D_FxdU,ORDER,Nv>& FxdU() {
      static SctlKernelFunction<Real,sctl::BiotSavart3D_FxdU,ORDER,Nv> ker(127, nullptr);
      return ker;
    }
};

/**
 * Helmholtz kernels with a real wavenumber k, with r = x - y for a target
 * point x and a source point y with normal n. Complex values are stored as
 * real and imaginary parts. The object holds k and must outlive any operator
 * that uses its kernels.
 *
 * @tparam ORDER digits of accuracy of the reciprocal square root in the
 * kernel evaluation.
 *
 * @tparam Nv the SIMD vector length.
 */
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Helmholtz3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  using KerFxU = SctlKernelFunction<Real, sctl::Helmholtz3D_FxU, ORDER, Nv>;
  using KerDxU = SctlKernelFunction<Real, sctl::Helmholtz3D_DxU, ORDER, Nv>;
  using KerFxdU = SctlKernelFunction<Real, sctl::Helmholtz3D_FxdU, ORDER, Nv>;

  public:

    /**
     * Constructor.
     *
     * @param[in] k the wavenumber.
     */
    Helmholtz3D(Real k) :
      k_(k),
      ker_FxU(24, &k_),
      ker_DxU(38, &k_),
      ker_FxdU(41, &k_)
      {}

    /**
     * Single layer exp(ik|r|)/(4 pi |r|); 2 density and 2 potential
     * components.
     */
    const KerFxU& FxU() const { return ker_FxU; }

    /**
     * Gradient of the single layer with respect to x; 2 density and 6
     * potential components.
     */
    const KerFxdU& FxdU() const { return ker_FxdU; }

    /**
     * Double layer; 2 density and 2 potential components.
     */
    const KerDxU& DxU() const { return ker_DxU; }

  private:

    Real k_;
    KerFxU ker_FxU;
    KerDxU ker_DxU;
    KerFxdU ker_FxdU;
};

/**
 * Difference of the Helmholtz and Laplace single layers, (exp(ik|r|) - 1)/(4 pi
 * |r|), with a real wavenumber k. Complex values are stored as real and
 * imaginary parts. The object holds k and must outlive any operator that uses
 * its kernels.
 */
template <class Real> class HelmholtzDiff3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  using KerFxdU = SctlKernelFunction<Real, sctl::HelmholtzDiff3D_FxdU, -1>;

  public:

    /**
     * Constructor.
     *
     * @param[in] k the wavenumber.
     */
    HelmholtzDiff3D(Real k) :
      k_(k),
      ker_FxdU(47, &k_) {}

    /**
     * Gradient with respect to x; 2 density and 6 potential components.
     */
    const KerFxdU& FxdU() const { return ker_FxdU; }

  private:

    Real k_;
    KerFxdU ker_FxdU;
};

}

#endif  //_KERNEL_HPP_
