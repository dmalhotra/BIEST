#ifndef _KERNEL_HPP_
#define _KERNEL_HPP_

#include <sctl.hpp>

namespace biest {

template <class Real, sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> class KernelFunction {
    void ker(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f) const {
      Real eps = (Real)1e-30;
      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);

      Real fdotr = 0;
      Real ndotr = 0;
      Real invr2 = invr*invr;
      Real invr3 = invr2*invr;
      Real invr5 = invr2*invr3;
      for(sctl::Integer k=0;k<3;k++) fdotr += f[k] * x[k];
      for(sctl::Integer k=0;k<3;k++) ndotr += n[k] * x[k];
      static const Real scal = 3 / (4 * sctl::const_pi<Real>());
      if (0) {
        v[0] += f[0];
        v[1] += f[1] * ndotr * invr3 / 3;
        v[2] += f[2] * invr;
      } else {
        v[0] += x[0] * fdotr * ndotr * invr5 * scal;
        v[1] += x[1] * fdotr * ndotr * invr5 * scal;
        v[2] += x[2] * fdotr * ndotr * invr5 * scal;
      }
    };

    typedef void (KerFn)(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx);

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
      sctl::Profile::Add_FLOP(Ns * Nt * cost);
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
      sctl::Integer kdim[2] = {this->Dim(0), this->Dim(1)};
      sctl::Long Ns = r_src.Dim() / COORD_DIM;
      sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      if (M.Dim(0) != Ns * kdim[0] || M.Dim(1) != Nt * kdim[1]) {
        M.ReInit(Ns * kdim[0], Nt * kdim[1]);
      }
      sctl::Integer omp_p = omp_get_max_threads();

      if (1) {
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
            for (sctl::Integer k = 0; k < kdim[0]; k++) {
              f_src0[k] = 1;
              sctl::Vector<Real> v_trg(M.Dim(1), M[k * Ns + s], false);
              v_trg = 0;
              kerfn(r_src0, n_src0, f_src0, r_trg, v_trg, 1, ctx);
              f_src0[k] = 0;
            }
          }
        }
      } else {
        // Does not work if the source has an associated normal direction.
        // Only works for single-layer, not for double-layer
        constexpr sctl::Long BLOCK_SIZE = 10000;
        #pragma omp parallel for schedule(static)
        for (sctl::Integer tid = 0; tid < omp_p; tid++) {
          sctl::Long i0 = (tid + 0) * Ns*Nt / omp_p;
          sctl::Long i1 = (tid + 1) * Ns*Nt / omp_p;

          sctl::StaticArray<Real,COORD_DIM*2+KDIM0> sbuff;
          sctl::Vector<Real> r_src0(COORD_DIM, sbuff + COORD_DIM * 0, false);
          sctl::Vector<Real> n_src0(COORD_DIM, sbuff + COORD_DIM * 1, false);
          sctl::Vector<Real> f_src0(   Dim(0), sbuff + COORD_DIM * 2, false);
          f_src0 = 0;

          sctl::StaticArray<Real,COORD_DIM*BLOCK_SIZE> Xt_;
          sctl::StaticArray<Real,    KDIM0*BLOCK_SIZE> Vt_;
          for (sctl::Long i = i0; i < i1; i += BLOCK_SIZE) {
            sctl::Long s = i / Nt;
            sctl::Long t = i - s * Nt;

            sctl::Long j0 = i0;
            sctl::Long j1 = std::min(i0 + BLOCK_SIZE, i1);
            sctl::Matrix<Real> Mr_trg0(COORD_DIM, j1-j0, Xt_, false);
            sctl::Matrix<Real> Mv_trg0(    KDIM0, j1-j0, Vt_, false);
            sctl::Vector<Real> r_trg0(COORD_DIM * (j1-j0), Xt_, false);
            sctl::Vector<Real> v_trg0(    KDIM0 * (j1-j0), Vt_, false);

            { // Set r_trg0
              sctl::Long s0 = s, t0 = t;
              for (sctl::Long j = j0; j < j1; j++) {
                for (sctl::Long k = 0; k < COORD_DIM; k++) Mr_trg0[k][j-j0] = r_trg[k*Nt+t0] - r_src[k*Ns+s0];
                t0++;
                if (t0 == Nt) {
                  s0++;
                  t0 = 0;
                }
              }
            }

            f_src0 = 0;
            for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
              { // Set v_trg0
                f_src0[k0] = 1;
                v_trg0 = 0;
                kerfn(r_src0, n_src0, f_src0, r_trg0, v_trg0, 1, ctx);
                f_src0[k0] = 0;
              }

              { // Set M
                sctl::Long s0 = s, t0 = t;
                for (sctl::Long j = j0; j < j1; j++) {
                  for (sctl::Long k1 = 0; k1 < KDIM1; k1++) {
                    M[k0*Ns+s0][k1*Nt+t0] = Mv_trg0[k1][j-j0];
                  }
                  t0++;
                  if (t0 == Nt) {
                    s0++;
                    t0 = 0;
                  }
                }
              }
            }
          }
        }
      }
      sctl::Profile::Add_FLOP(Ns * Nt * cost);
    }

  private:
    std::function<KerFn> kerfn;
    const void* ctx;
    sctl::Long cost;
};

template <class Real, sctl::Integer COORD_DIM, sctl::Integer KER_DIM0, sctl::Integer KER_DIM1, void (UKER)(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx)> static void GenericKerWrapper(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx) {
  sctl::Long Ns = r_src.Dim() / COORD_DIM;
  sctl::Long Nt = r_trg.Dim() / COORD_DIM;
  sctl::Long dof = v_src.Dim() / (Ns ? Ns : 1) / KER_DIM0;

  auto ker = [&](sctl::Long t, sctl::Integer k) {
    sctl::StaticArray<Real, KER_DIM1> v;
    for (sctl::Integer i = 0; i < KER_DIM1; i++) v[i] = 0;
    for (sctl::Long s = 0; s < Ns; s++) {
      sctl::StaticArray<Real, COORD_DIM> r{r_trg[0 * Nt + t] - r_src[0 * Ns + s],
                                        r_trg[1 * Nt + t] - r_src[1 * Ns + s],
                                        r_trg[2 * Nt + t] - r_src[2 * Ns + s]};
      sctl::StaticArray<Real, COORD_DIM> n{n_src[0 * Ns + s], n_src[1 * Ns + s], n_src[2 * Ns + s]};
      sctl::StaticArray<Real, KER_DIM0> f;
      for (sctl::Integer i = 0; i < KER_DIM0; i++)  f[i] = v_src[(k*KER_DIM0+i) * Ns + s];
      UKER(v, r, n, f, ctx);
    }
    for (sctl::Integer i = 0; i < KER_DIM1; i++)  v_trg[(k*KER_DIM1+i) * Nt + t] += v[i];
  };

  for (sctl::Integer k = 0; k < dof; k++) {
    if (Nthread == 1) {
      for (sctl::Long t = 0; t < Nt; t++) ker(t, k);
    } else {
      #pragma omp parallel for schedule(static) num_threads(Nthread)
      for (sctl::Long t = 0; t < Nt; t++) ker(t, k);
    }
  }
}

template <sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, class Real, class RealVec, void (UKER)(RealVec vt[KDIM1], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec vs[KDIM1]), sctl::Long scale = 1> static void GenericKerWrapperVec(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx) {
  static constexpr sctl::Integer Nv = RealVec::Size();
  static const Real scal = scale / (4 * sctl::const_pi<Real>());
  auto ker = [](sctl::Matrix<Real>& Vt_, const sctl::Matrix<Real>& Xt_, const sctl::Matrix<Real>& Xs_, const sctl::Matrix<Real>& Ns_, const sctl::Matrix<Real>& Vs_, sctl::Integer Nthread) {
    sctl::Long Nt = Xt_.Dim(1);
    sctl::Long Ns = Xs_.Dim(1);
    sctl::Long dof = Vs_.Dim(0) / KDIM0;
    assert((Nt / Nv) * Nv == Nt);
    assert(dof == 1);
    SCTL_UNUSED(dof);

    assert(Xs_.Dim(0) == COORD_DIM);
    assert(Vs_.Dim(0) == dof * KDIM0);
    assert(Vs_.Dim(1) == Ns);
    assert(Xt_.Dim(0) == COORD_DIM);
    assert(Vt_.Dim(0) == dof * KDIM1);
    assert(Vt_.Dim(1) == Nt);

    if (Nthread == 1) {
      for (sctl::Long t = 0; t < Nt; t += Nv) {
        RealVec xt[COORD_DIM], vt[KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[KDIM0];
        for (sctl::Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
        for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
        for (sctl::Long s = 0; s < Ns; s++) {
          for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&Xs_[k][s]);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&Ns_[k][s]);
          for (sctl::Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[k][s]);
          UKER(vt, xt, xs, ns, vs);
        }
        for (sctl::Integer k = 0; k < KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (sctl::Long t = 0; t < Nt; t += Nv) {
        RealVec xt[COORD_DIM], vt[KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[KDIM0];
        for (sctl::Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
        for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
        for (sctl::Long s = 0; s < Ns; s++) {
          for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&Xs_[k][s]);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&Ns_[k][s]);
          for (sctl::Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[k][s]);
          UKER(vt, xt, xs, ns, vs);
        }
        for (sctl::Integer k = 0; k < KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
      }
    }
  };

  sctl::Long Ns = r_src.Dim() / COORD_DIM;
  sctl::Long Nt = r_trg.Dim() / COORD_DIM;
  sctl::Long NNt = ((Nt + Nv - 1) / Nv) * Nv;
  if (NNt == Nv) {
    sctl::StaticArray<Real,Nv> tmp_xt;
    RealVec xt[COORD_DIM], vt[KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[KDIM0];
    for (sctl::Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Integer i = 0; i < Nt; i++) tmp_xt[i] = r_trg[k*Nt+i];
      xt[k] = RealVec::Load(&tmp_xt[0]);
    }
    for (sctl::Long s = 0; s < Ns; s++) {
      for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&r_src[k*Ns+s]);
      for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&n_src[k*Ns+s]);
      for (sctl::Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&v_src[k*Ns+s]);
      UKER(vt, xt, xs, ns, vs);
    }
    for (sctl::Integer k = 0; k < KDIM1; k++) {
      alignas(sizeof(RealVec)) Real out[Nv];
      vt[k].StoreAligned(&out[0]);
      for (sctl::Long t = 0; t < Nt; t++) {
        v_trg[k*Nt+t] += out[t] * scal;
      }
    }
  } else {
    const sctl::Matrix<Real> Xs_(COORD_DIM, Ns, (sctl::Iterator<Real>)r_src.begin(), false);
    const sctl::Matrix<Real> Ns_(COORD_DIM, Ns, (sctl::Iterator<Real>)n_src.begin(), false);
    const sctl::Matrix<Real> Vs_(KDIM0    , Ns, (sctl::Iterator<Real>)v_src.begin(), false);
    sctl::Matrix<Real> Xt_(COORD_DIM, NNt), Vt_(KDIM1, NNt);
    for (sctl::Long k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < Nt; i++) {
        Xt_[k][i] = r_trg[k * Nt + i];
      }
      for (sctl::Long i = Nt; i < NNt; i++) {
        Xt_[k][i] = 0;
      }
    }
    ker(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread);
    for (sctl::Long k = 0; k < KDIM1; k++) {
      for (sctl::Long i = 0; i < Nt; i++) {
        v_trg[k * Nt + i] += Vt_[k][i] * scal;
      }
    }
  }
}

template <class Real> class Stokes3D_ {
  static constexpr sctl::Integer COORD_DIM = 3;

  public:
    static const KernelFunction<Real,COORD_DIM,3,3>& DxU() {
      static constexpr sctl::Integer KDIM0 = COORD_DIM;
      static constexpr sctl::Integer KDIM1 = COORD_DIM;
      // TODO: Clean-up, avoid having to specify COORD_DIM, KDIM0, KDIM1 twice
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1, uker_DxU>, 31, nullptr);
      return ker;
    }

  private:
    static void uker_DxU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      static const Real scal = 3 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);

      Real fdotr = 0;
      Real ndotr = 0;
      Real invr2 = invr*invr;
      Real invr3 = invr2*invr;
      Real invr5 = invr2*invr3;
      for(sctl::Integer k = 0; k < COORD_DIM; k++) fdotr += f[k] * x[k];
      for(sctl::Integer k = 0; k < COORD_DIM; k++) ndotr += n[k] * x[k];
      Real ker_term0 = fdotr * ndotr * invr5 * scal;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) v[k] -= x[k] * ker_term0;
    }
};
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Stokes3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 3;
  static constexpr sctl::Integer KDIM1 = 3;

  using RealVec = sctl::Vec<Real,Nv>;

  public:
    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& DxU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1, Real, RealVec, uker_DxU, 3>, 31, nullptr);
      return ker;
    }

  private:

    static void uker_DxU(RealVec v[KDIM1], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec rinv2 = rinv*rinv;
      RealVec rinv3 = rinv2*rinv;
      RealVec rinv5 = rinv2*rinv3;

      RealVec ndotr = ns[0] * dx[0] + ns[1] * dx[1] + ns[2] * dx[2];
      RealVec fdotr =  f[0] * dx[0] +  f[1] * dx[1] +  f[2] * dx[2];
      RealVec ker_term0 = fdotr * ndotr * rinv5;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) v[k] -= dx[k] * ker_term0;
    }
};

template <class Real> class Laplace3D_ {
  static constexpr sctl::Integer COORD_DIM = 3;

  public:
    static const KernelFunction<Real,COORD_DIM,1,1>& FxU() {
      static constexpr sctl::Integer KDIM0 = 1;
      static constexpr sctl::Integer KDIM1 = 1;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1, uker_FxU>, 9, nullptr);
      return ker;
    }

    static const KernelFunction<Real,COORD_DIM,1,3>& FxdU() {
      static constexpr sctl::Integer KDIM0 = 1;
      static constexpr sctl::Integer KDIM1 = COORD_DIM;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1, uker_FxdU>, 16, nullptr);
      return ker;
    }

    static const KernelFunction<Real,COORD_DIM,1,1>& DxU() {
      static constexpr sctl::Integer KDIM0 = 1;
      static constexpr sctl::Integer KDIM1 = 1;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1, uker_DxU>, 19, nullptr);
      return ker;
    }

  private:
    static void uker_FxU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);
      v[0] += f[0] * invr * scal;
    }

    static void uker_FxdU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);
      Real invr3 = invr * invr * invr;
      Real ker_term0 = f[0] * invr3 * scal;
      for (sctl::Integer k = 0; k < COORD_DIM; k++) v[k] -= x[k] * ker_term0;
    }

    static void uker_DxU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);

      Real ndotr = 0;
      Real invr2 = invr*invr;
      Real invr3 = invr2*invr;
      for(sctl::Integer k = 0; k < COORD_DIM; k++) ndotr -= n[k] * x[k];
      v[0] += f[0] * ndotr * invr3 * scal;
    }
};
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Laplace3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 1;
  static constexpr sctl::Integer KDIM1 = 1;

  using RealVec = sctl::Vec<Real,Nv>;

  public:
    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1, uker_FxU<1>, uker_FxU<2>, uker_FxU<3>>, 12, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1*COORD_DIM, uker_FxdU<1>, uker_FxdU<2>, uker_FxdU<3>>, 19, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM*COORD_DIM>& Fxd2U() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM*COORD_DIM> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1*COORD_DIM*COORD_DIM, uker_Fxd2U<1>, uker_Fxd2U<2>, uker_Fxd2U<3>>, 61, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& DxU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1, uker_DxU<1>, uker_DxU<2>, uker_DxU<3>>, 20, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& DxdU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1*COORD_DIM, uker_DxdU<1>, uker_DxdU<2>, uker_DxdU<3>>, 39, nullptr);
      return ker;
    }

  private:

    typedef void (UKerFn)(RealVec* vt, const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec* vs);
    template <sctl::Integer COORD_DIM, sctl::Integer DOF, sctl::Integer KDIM0, sctl::Integer KDIM1, UKerFn UKER> static void ker(sctl::Matrix<Real>& Vt_, const sctl::Matrix<Real>& Xt_, const sctl::Matrix<Real>& Xs_, const sctl::Matrix<Real>& Ns_, const sctl::Matrix<Real>& Vs_, sctl::Integer Nthread) {
      sctl::Long Nt = Xt_.Dim(1);
      sctl::Long Ns = Xs_.Dim(1);
      assert((Nt / Nv) * Nv == Nt);

      assert(Xs_.Dim(0) == COORD_DIM);
      assert(Vs_.Dim(0) == DOF * KDIM0);
      assert(Vs_.Dim(1) == Ns);
      assert(Xt_.Dim(0) == COORD_DIM);
      assert(Vt_.Dim(0) == DOF * KDIM1);
      assert(Vt_.Dim(1) == Nt);

      if (Nthread == 1) {
        for (sctl::Long t = 0; t < Nt; t += Nv) {
          RealVec xt[COORD_DIM], vt[DOF * KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[DOF * KDIM0];
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k] = RealVec::Zero();
          for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
          for (sctl::Long s = 0; s < Ns; s++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&Xs_[k][s]);
            for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&Ns_[k][s]);
            for (sctl::Integer k = 0; k < DOF * KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[k][s]);
            UKER(vt, xt, xs, ns, vs);
          }
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
        }
      } else {
        #pragma omp parallel for schedule(static)
        for (sctl::Long t = 0; t < Nt; t += Nv) {
          RealVec xt[COORD_DIM], vt[DOF * KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[DOF * KDIM0];
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k] = RealVec::Zero();
          for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
          for (sctl::Long s = 0; s < Ns; s++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&Xs_[k][s]);
            for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&Ns_[k][s]);
            for (sctl::Integer k = 0; k < DOF * KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[k][s]);
            UKER(vt, xt, xs, ns, vs);
          }
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
        }
      }
    }
    template <sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, UKerFn UKER1, UKerFn UKER2, UKerFn UKER3, sctl::Long scale = 1> static void GenericKerWrapperVec(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx) {
      static const Real scal = scale / (4 * sctl::const_pi<Real>());

      sctl::Long Ns = r_src.Dim() / COORD_DIM;
      sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      sctl::Long NNt = ((Nt + Nv - 1) / Nv) * Nv;
      sctl::Long NNs = ((Ns + Nv - 1) / Nv) * Nv;
      sctl::Long dof = v_src.Dim() / (Ns ? Ns : 1) / KDIM0;
      if (NNt == Nv) {
        sctl::StaticArray<Real,COORD_DIM*Nv> Xt_;
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          for (sctl::Integer i = 0; i < Nt; i++) {
            Xt_[k*Nv+i] = r_trg[k*Nt+i];
          }
        }

        RealVec xt[COORD_DIM], vt[KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[KDIM0];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k*Nv]);
        for (sctl::Integer i = 0; i < dof; i++) {
          for (sctl::Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
          for (sctl::Long s = 0; s < Ns; s++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&r_src[k*Ns+s]);
            for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&n_src[k*Ns+s]);
            for (sctl::Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&v_src[(i*KDIM0+k)*Ns+s]);
            UKER1(vt, xt, xs, ns, vs);
          }
          for (sctl::Integer k = 0; k < KDIM1; k++) {
            alignas(sizeof(RealVec)) Real out[Nv];
            vt[k].StoreAligned(&out[0]);
            for (sctl::Long t = 0; t < Nt; t++) {
              v_trg[(i*KDIM1+k)*Nt+t] += out[t] * scal;
            }
          }
        }
      } else {
        sctl::Matrix<Real> Xs_(COORD_DIM, NNs);
        sctl::Matrix<Real> Ns_(COORD_DIM, NNs);
        sctl::Matrix<Real> Vs_(dof*KDIM0, NNs);
        sctl::Matrix<Real> Xt_(COORD_DIM, NNt), Vt_(dof*KDIM1, NNt);
        auto fill_mat = [](sctl::Matrix<Real>& M, const sctl::Vector<Real>& v, sctl::Long N) {
          SCTL_ASSERT(N <= M.Dim(1));
          for (sctl::Long i = 0; i < M.Dim(0); i++) {
            for (sctl::Long j = 0; j < N; j++) M[i][j] = v[i*N+j];
            for (sctl::Long j = N; j < M.Dim(1); j++) M[i][j] = 0;
          }
        };
        fill_mat(Xs_, r_src, Ns);
        fill_mat(Ns_, n_src, Ns);
        fill_mat(Vs_, v_src, Ns);
        fill_mat(Xt_, r_trg, Nt);

        if (dof == 1) ker<COORD_DIM, 1, KDIM0, KDIM1, UKER1>(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread);
        if (dof == 2) ker<COORD_DIM, 2, KDIM0, KDIM1, UKER2>(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread);
        if (dof == 3) ker<COORD_DIM, 3, KDIM0, KDIM1, UKER3>(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread);
        if (dof > 3) SCTL_ASSERT(false);
        for (sctl::Long k = 0; k < dof * KDIM1; k++) {
          for (sctl::Long i = 0; i < Nt; i++) {
            v_trg[k * Nt + i] += Vt_[k][i] * scal;
          }
        }
      }
    }

    template <sctl::Integer DOF> static void uker_FxU(RealVec v[DOF*KDIM1], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[DOF*KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      for (sctl::Integer k = 0; k < DOF; k++) v[k] += f[k] * rinv;
    }

    template <sctl::Integer DOF> static void uker_FxdU(RealVec v[DOF*KDIM1*COORD_DIM], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[DOF*KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      for (sctl::Integer k = 0; k < DOF; k++) {
        RealVec ker_term0 = f[k] * rinv * rinv * rinv;
        for (sctl::Integer i = 0; i < COORD_DIM; i++) v[k*COORD_DIM+i] -= dx[i] * ker_term0;
      }
    }

    template <sctl::Integer DOF> static void uker_Fxd2U(RealVec v[DOF*KDIM1*COORD_DIM*COORD_DIM], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[DOF*KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec r[COORD_DIM];
      r[0] = xt[0] - xs[0];
      r[1] = xt[1] - xs[1];
      r[2] = xt[2] - xs[2];
      const RealVec r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      const RealVec rinv2 = rinv * rinv;
      const RealVec rinv3 = rinv * rinv2;
      const RealVec rinv5 = rinv3 * rinv2;

      RealVec u[9];
      u[0+3*0] = -rinv3 + (Real)3 * r[0] * r[0] * rinv5;
      u[1+3*0] =          (Real)3 * r[0] * r[1] * rinv5;
      u[2+3*0] =          (Real)3 * r[0] * r[2] * rinv5;

      u[0+3*1] =          (Real)3 * r[1] * r[0] * rinv5;
      u[1+3*1] = -rinv3 + (Real)3 * r[1] * r[1] * rinv5;
      u[2+3*1] =          (Real)3 * r[1] * r[2] * rinv5;

      u[0+3*2] =          (Real)3 * r[2] * r[0] * rinv5;
      u[1+3*2] =          (Real)3 * r[2] * r[1] * rinv5;
      u[2+3*2] = -rinv3 + (Real)3 * r[2] * r[2] * rinv5;

      for (sctl::Integer k = 0; k < DOF; k++) {
        for (sctl::Integer i = 0; i < COORD_DIM*COORD_DIM; i++) v[k*COORD_DIM*COORD_DIM+i] += u[i] * f[k];
      }
    }

    template <sctl::Integer DOF> static void uker_DxU(RealVec v[DOF*KDIM1], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[DOF*KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec ndotr = ns[0] * dx[0] + ns[1] * dx[1] + ns[2] * dx[2];
      for (sctl::Integer k = 0; k < DOF; k++) v[k] -= f[k] * ndotr * rinv * rinv * rinv;
    }

    template <sctl::Integer DOF> static void uker_DxdU(RealVec v[DOF*KDIM1*COORD_DIM], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[DOF*KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec r[COORD_DIM];
      r[0] = xt[0] - xs[0];
      r[1] = xt[1] - xs[1];
      r[2] = xt[2] - xs[2];
      const RealVec r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
      const RealVec rdotn = r[0]*ns[0] + r[1]*ns[1] + r[2]*ns[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      const RealVec rinv2 = rinv * rinv;
      const RealVec rinv3 = rinv * rinv2;
      const RealVec rinv5 = rinv3 * rinv2;

      RealVec u[3];
      u[0] = ns[0] * rinv3 - (Real)3*rdotn * r[0] * rinv5;
      u[1] = ns[1] * rinv3 - (Real)3*rdotn * r[1] * rinv5;
      u[2] = ns[2] * rinv3 - (Real)3*rdotn * r[2] * rinv5;

      for (sctl::Integer k = 0; k < DOF; k++) {
        v[k*COORD_DIM+0] -= f[k] * u[0];
        v[k*COORD_DIM+1] -= f[k] * u[1];
        v[k*COORD_DIM+2] -= f[k] * u[2];
      }
    }
};

template <class Real> class BiotSavart3D_ {
  static constexpr sctl::Integer COORD_DIM = 3;

  public:
    static const KernelFunction<Real,COORD_DIM,3,3>& FxU() {
      static constexpr sctl::Integer KDIM0 = 3;
      static constexpr sctl::Integer KDIM1 = 3;
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1, uker_FxU>, 27, nullptr);
      return ker;
    }

  private:
    static void uker_FxU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);
      Real ker_term0 = invr * invr * invr * scal;
      v[0] -= (f[1]*x[2] - x[1]*f[2]) * ker_term0;
      v[1] -= (f[2]*x[0] - x[2]*f[0]) * ker_term0;
      v[2] -= (f[0]*x[1] - x[0]*f[1]) * ker_term0;
    }
};
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class BiotSavart3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 3;
  static constexpr sctl::Integer KDIM1 = 3;

  using RealVec = sctl::Vec<Real,Nv>;

  public:
    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1, Real, RealVec, uker_FxU>, 27, nullptr);
      return ker;
    }

    static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() {
      static KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1*COORD_DIM, Real, RealVec, uker_FxdU>, 127, nullptr);
      return ker;
    }

  private:

    static void uker_FxU(RealVec v[KDIM1], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec rinv3 = rinv * rinv * rinv;
      v[0] -= (f[1]*dx[2] - dx[1]*f[2]) * rinv3;
      v[1] -= (f[2]*dx[0] - dx[2]*f[0]) * rinv3;
      v[2] -= (f[0]*dx[1] - dx[0]*f[1]) * rinv3;
    }

    static void uker_FxdU(RealVec v[KDIM1*COORD_DIM], const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec f[KDIM1]) {
      constexpr Real eps = (Real)1e-30;
      RealVec r[COORD_DIM];
      r[0] = xt[0] - xs[0];
      r[1] = xt[1] - xs[1];
      r[2] = xt[2] - xs[2];
      RealVec r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec rinv2 = rinv * rinv;
      RealVec rinv3 = rinv2 * rinv;
      RealVec rinv5 = rinv2 * rinv3;

      RealVec u[3][9];
      u[0][0] =                                      0; u[1][0] =        + (Real)3 * r[2] * r[0] * rinv5; u[2][0] =        - (Real)3 * r[1] * r[0] * rinv5;
      u[0][1] =                                      0; u[1][1] =        + (Real)3 * r[2] * r[1] * rinv5; u[2][1] =  rinv3 - (Real)3 * r[1] * r[1] * rinv5;
      u[0][2] =                                      0; u[1][2] = -rinv3 + (Real)3 * r[2] * r[2] * rinv5; u[2][2] =        - (Real)3 * r[1] * r[2] * rinv5;

      u[0][3] =        - (Real)3 * r[2] * r[0] * rinv5; u[1][3] =                                      0; u[2][3] = -rinv3 + (Real)3 * r[0] * r[0] * rinv5;
      u[0][4] =        - (Real)3 * r[2] * r[1] * rinv5; u[1][4] =                                      0; u[2][4] =        + (Real)3 * r[0] * r[1] * rinv5;
      u[0][5] =  rinv3 - (Real)3 * r[2] * r[2] * rinv5; u[1][5] =                                      0; u[2][5] =        + (Real)3 * r[0] * r[2] * rinv5;

      u[0][6] =        + (Real)3 * r[1] * r[0] * rinv5; u[1][6] =  rinv3 - (Real)3 * r[0] * r[0] * rinv5; u[2][6] =                                      0;
      u[0][7] = -rinv3 + (Real)3 * r[1] * r[1] * rinv5; u[1][7] =        - (Real)3 * r[0] * r[1] * rinv5; u[2][7] =                                      0;
      u[0][8] =        + (Real)3 * r[1] * r[2] * rinv5; u[1][8] =        - (Real)3 * r[0] * r[2] * rinv5; u[2][8] =                                      0;

      v[0] += u[0][0] * f[0];
      v[1] += u[0][1] * f[0];
      v[2] += u[0][2] * f[0];
      v[3] += u[0][3] * f[0];
      v[4] += u[0][4] * f[0];
      v[5] += u[0][5] * f[0];
      v[6] += u[0][6] * f[0];
      v[7] += u[0][7] * f[0];
      v[8] += u[0][8] * f[0];

      v[0] += u[1][0] * f[1];
      v[1] += u[1][1] * f[1];
      v[2] += u[1][2] * f[1];
      v[3] += u[1][3] * f[1];
      v[4] += u[1][4] * f[1];
      v[5] += u[1][5] * f[1];
      v[6] += u[1][6] * f[1];
      v[7] += u[1][7] * f[1];
      v[8] += u[1][8] * f[1];

      v[0] += u[2][0] * f[2];
      v[1] += u[2][1] * f[2];
      v[2] += u[2][2] * f[2];
      v[3] += u[2][3] * f[2];
      v[4] += u[2][4] * f[2];
      v[5] += u[2][5] * f[2];
      v[6] += u[2][6] * f[2];
      v[7] += u[2][7] * f[2];
      v[8] += u[2][8] * f[2];
    }
};

template <class Real> class Helmholtz3D_ {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  public:

    Helmholtz3D_(Real k) :
      k_(k),
      ker_FxU(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1, uker_FxU>, 24, this),
      ker_FxdU(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1*COORD_DIM, uker_FxdU>, 42, this) {}

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() const { return ker_FxU; }

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() const { return ker_FxdU; }

  private:

    static void uker_FxU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      const Real k = static_cast<const Helmholtz3D_<Real>*>(ctx)->k_;
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);

      Real cos_kr = sctl::cos<Real>(k*r);
      Real sin_kr = sctl::sin<Real>(k*r);

      Real G[2];
      G[0] = cos_kr*invr*scal;
      G[1] = sin_kr*invr*scal;

      v[0] += f[0]*G[0] - f[1]*G[1];
      v[1] += f[0]*G[1] + f[1]*G[0];
    }

    static void uker_FxdU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      const Real k = static_cast<const Helmholtz3D_<Real>*>(ctx)->k_;
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);
      Real invr2 = invr * invr;

      Real cos_kr = sctl::cos<Real>(k*r);
      Real sin_kr = sctl::sin<Real>(k*r);

      Real G[2], fG[2];
      G[0] = (-k*sin_kr - cos_kr * invr) * invr2 * scal;
      G[1] = ( k*cos_kr - sin_kr * invr) * invr2 * scal;

      fG[0] = f[0]*G[0] - f[1]*G[1];
      fG[1] = f[0]*G[1] + f[1]*G[0];

      for (sctl::Integer i = 0; i < COORD_DIM; i++) {
        v[i*2+0] += fG[0] * x[i];
        v[i*2+1] += fG[1] * x[i];
      }
    }

    Real k_;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker_FxU;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker_FxdU;
};
template <class Real, sctl::Integer ORDER = 13, sctl::Integer Nv = sctl::DefaultVecLen<Real>()> class Helmholtz3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  using RealVec = sctl::Vec<Real,Nv>;

  public:

    Helmholtz3D(Real k) :
      k_(k),
      ker_FxU(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1, uker_FxU<1>, uker_FxU<2>, uker_FxU<3>>, 24, this),
      ker_DxU(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1, uker_DxU<1>, uker_DxU<2>, uker_DxU<3>>, 38, this),
      ker_FxdU(GenericKerWrapperVec<COORD_DIM, KDIM0, KDIM1*COORD_DIM, uker_FxdU<1>, uker_FxdU<2>, uker_FxdU<3>>, 41, this)
      {}

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& FxU() const { return ker_FxU; }

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() const { return ker_FxdU; }

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& DxU() const { return ker_DxU; }

  private:

    typedef void (UKerFn)(RealVec* vt, const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec* vs, const RealVec mu);
    template <sctl::Integer COORD_DIM, sctl::Integer DOF, sctl::Integer KDIM0, sctl::Integer KDIM1, UKerFn UKER> static void ker(sctl::Matrix<Real>& Vt_, const sctl::Matrix<Real>& Xt_, const sctl::Matrix<Real>& Xs_, const sctl::Matrix<Real>& Ns_, const sctl::Matrix<Real>& Vs_, sctl::Integer Nthread, const RealVec mu) {
      sctl::Long Nt = Xt_.Dim(1);
      sctl::Long Ns = Xs_.Dim(1);
      assert((Nt / Nv) * Nv == Nt);

      assert(Xs_.Dim(0) == COORD_DIM);
      assert(Vs_.Dim(0) == DOF * KDIM0);
      assert(Vs_.Dim(1) == Ns);
      assert(Xt_.Dim(0) == COORD_DIM);
      assert(Vt_.Dim(0) == DOF * KDIM1);
      assert(Vt_.Dim(1) == Nt);

      if (Nthread == 1) {
        for (sctl::Long t = 0; t < Nt; t += Nv) {
          RealVec xt[COORD_DIM], vt[DOF * KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[DOF * KDIM0];
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k] = RealVec::Zero();
          for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
          for (sctl::Long s = 0; s < Ns; s++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&Xs_[k][s]);
            for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&Ns_[k][s]);
            for (sctl::Integer k = 0; k < DOF * KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[k][s]);
            UKER(vt, xt, xs, ns, vs, mu);
          }
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
        }
      } else {
        #pragma omp parallel for schedule(static)
        for (sctl::Long t = 0; t < Nt; t += Nv) {
          RealVec xt[COORD_DIM], vt[DOF * KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[DOF * KDIM0];
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k] = RealVec::Zero();
          for (sctl::Integer k = 0; k < COORD_DIM; k++) xt[k] = RealVec::LoadAligned(&Xt_[k][t]);
          for (sctl::Long s = 0; s < Ns; s++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&Xs_[k][s]);
            for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&Ns_[k][s]);
            for (sctl::Integer k = 0; k < DOF * KDIM0; k++) vs[k] = RealVec::Load1(&Vs_[k][s]);
            UKER(vt, xt, xs, ns, vs, mu);
          }
          for (sctl::Integer k = 0; k < DOF * KDIM1; k++) vt[k].StoreAligned(&Vt_[k][t]);
        }
      }
    }
    template <sctl::Integer COORD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1, UKerFn UKER1, UKerFn UKER2, UKerFn UKER3> static void GenericKerWrapperVec(const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, sctl::Integer Nthread, const void* ctx) {
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      const RealVec mu(static_cast<const Helmholtz3D*>(ctx)->k_);

      sctl::Long Ns = r_src.Dim() / COORD_DIM;
      sctl::Long Nt = r_trg.Dim() / COORD_DIM;
      sctl::Long NNt = ((Nt + Nv - 1) / Nv) * Nv;
      sctl::Long NNs = ((Ns + Nv - 1) / Nv) * Nv;
      sctl::Long dof = v_src.Dim() / (Ns ? Ns : 1) / KDIM0;
      if (NNt == Nv) {
        sctl::StaticArray<Real,Nv> tmp_xt;
        RealVec xt[COORD_DIM], vt[KDIM1], xs[COORD_DIM], ns[COORD_DIM], vs[KDIM0];
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          for (sctl::Integer i = 0; i < Nt; i++) tmp_xt[i] = r_trg[k*Nt+i];
          xt[k] = RealVec::Load(&tmp_xt[0]);
        }
        for (sctl::Integer i = 0; i < dof; i++) {
          for (sctl::Integer k = 0; k < KDIM1; k++) vt[k] = RealVec::Zero();
          for (sctl::Long s = 0; s < Ns; s++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) xs[k] = RealVec::Load1(&r_src[k*Ns+s]);
            for (sctl::Integer k = 0; k < COORD_DIM; k++) ns[k] = RealVec::Load1(&n_src[k*Ns+s]);
            for (sctl::Integer k = 0; k < KDIM0; k++) vs[k] = RealVec::Load1(&v_src[(i*KDIM0+k)*Ns+s]);
            UKER1(vt, xt, xs, ns, vs, mu);
          }
          for (sctl::Integer k = 0; k < KDIM1; k++) {
            alignas(sizeof(RealVec)) Real out[Nv];
            vt[k].StoreAligned(&out[0]);
            for (sctl::Long t = 0; t < Nt; t++) {
              v_trg[(i*KDIM1+k)*Nt+t] += out[t] * scal;
            }
          }
        }
      } else {
        sctl::Matrix<Real> Xs_(COORD_DIM, NNs);
        sctl::Matrix<Real> Ns_(COORD_DIM, NNs);
        sctl::Matrix<Real> Vs_(dof*KDIM0, NNs);
        sctl::Matrix<Real> Xt_(COORD_DIM, NNt), Vt_(dof*KDIM1, NNt);
        auto fill_mat = [](sctl::Matrix<Real>& M, const sctl::Vector<Real>& v, sctl::Long N) {
          SCTL_ASSERT(N <= M.Dim(1));
          for (sctl::Long i = 0; i < M.Dim(0); i++) {
            for (sctl::Long j = 0; j < N; j++) M[i][j] = v[i*N+j];
            for (sctl::Long j = N; j < M.Dim(1); j++) M[i][j] = 0;
          }
        };
        fill_mat(Xs_, r_src, Ns);
        fill_mat(Ns_, n_src, Ns);
        fill_mat(Vs_, v_src, Ns);
        fill_mat(Xt_, r_trg, Nt);

        if (dof == 1) ker<COORD_DIM, 1, KDIM0, KDIM1, UKER1>(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread, mu);
        if (dof == 2) ker<COORD_DIM, 2, KDIM0, KDIM1, UKER2>(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread, mu);
        if (dof == 3) ker<COORD_DIM, 3, KDIM0, KDIM1, UKER3>(Vt_, Xt_, Xs_, Ns_, Vs_, Nthread, mu);
        if (dof > 3) SCTL_ASSERT(false);
        for (sctl::Long k = 0; k < dof * KDIM1; k++) {
          for (sctl::Long i = 0; i < Nt; i++) {
            v_trg[k * Nt + i] += Vt_[k][i] * scal;
          }
        }
      }
    }

    template <sctl::Integer DOF> static void uker_FxU(RealVec* v, const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec* f, const RealVec mu) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec r = r2 * rinv;

      RealVec rmu = r * mu;
      RealVec cos_rmu, sin_rmu;
      sincos(sin_rmu, cos_rmu, rmu);

      RealVec G[2];
      G[0] = cos_rmu * rinv;
      G[1] = sin_rmu * rinv;
      for (sctl::Integer k = 0; k < DOF; k++) {
        v[k*2+0] += f[k*2+0]*G[0] - f[k*2+1]*G[1];
        v[k*2+1] += f[k*2+0]*G[1] + f[k*2+1]*G[0];
      }
    }

    template <sctl::Integer DOF> static void uker_FxdU(RealVec* v, const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec* f, const RealVec mu) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec rinv2 = rinv * rinv;
      RealVec r = r2 * rinv;

      RealVec rmu = r * mu;
      RealVec cos_rmu, sin_rmu;
      sincos(sin_rmu, cos_rmu, rmu);

      RealVec G[2], fG[2];
      G[0] = (-mu*sin_rmu - cos_rmu * rinv) * rinv2;
      G[1] = ( mu*cos_rmu - sin_rmu * rinv) * rinv2;

      for (sctl::Integer k = 0; k < DOF; k++) {
        fG[0] = f[k*2+0]*G[0] - f[k*2+1]*G[1];
        fG[1] = f[k*2+0]*G[1] + f[k*2+1]*G[0];

        for (sctl::Integer i = 0; i < COORD_DIM; i++) {
          v[k*COORD_DIM*2+i*2+0] += fG[0] * dx[i];
          v[k*COORD_DIM*2+i*2+1] += fG[1] * dx[i];
        }
      }
    }

    template <sctl::Integer DOF> static void uker_DxU(RealVec* v, const RealVec xt[COORD_DIM], const RealVec xs[COORD_DIM], const RealVec ns[COORD_DIM], const RealVec* f, const RealVec mu) {
      constexpr Real eps = (Real)1e-30;
      RealVec dx[COORD_DIM];
      dx[0] = xt[0] - xs[0];
      dx[1] = xt[1] - xs[1];
      dx[2] = xt[2] - xs[2];
      RealVec ndotr = ns[0] * dx[0] + ns[1] * dx[1] + ns[2] * dx[2];
      RealVec r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

      const RealVec rinv = sctl::approx_rsqrt<ORDER>(r2, r2 > eps);
      RealVec rinv2 = rinv * rinv;
      RealVec r = r2 * rinv;

      RealVec rmu = r * mu;
      RealVec cos_rmu, sin_rmu;
      sincos(sin_rmu, cos_rmu, rmu);

      RealVec G[2];
      G[0] = (-mu*sin_rmu - cos_rmu * rinv) * rinv2 * ndotr;
      G[1] = ( mu*cos_rmu - sin_rmu * rinv) * rinv2 * ndotr;

      for (sctl::Integer k = 0; k < DOF; k++) {
        v[k*2+0] += f[k*2+0]*G[0] - f[k*2+1]*G[1];
        v[k*2+1] += f[k*2+0]*G[1] + f[k*2+1]*G[0];
      }
    }

    Real k_;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1> ker_FxU, ker_DxU;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker_FxdU;
};

template <class Real> class HelmholtzDiff3D {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr sctl::Integer KDIM0 = 2;
  static constexpr sctl::Integer KDIM1 = 2;

  public:

    HelmholtzDiff3D(Real k) :
      k_(k),
      ker_FxdU(GenericKerWrapper<Real, COORD_DIM, KDIM0, KDIM1*COORD_DIM, uker_FxdU>, 47, this) {}

    const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM>& FxdU() const { return ker_FxdU; }

  private:

    static void uker_FxdU(sctl::Iterator<Real> v, sctl::ConstIterator<Real> x, sctl::ConstIterator<Real> n, sctl::ConstIterator<Real> f, const void* ctx) {
      const Real k = static_cast<const HelmholtzDiff3D<Real>*>(ctx)->k_;
      static const Real scal = 1 / (4 * sctl::const_pi<Real>());
      constexpr Real eps = (Real)1e-30;

      Real r = sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
      Real invr = (r > eps ? 1 / r : 0);
      Real invr2 = invr * invr;

      Real cos_kr = sctl::cos<Real>(k*r);
      Real sin_kr = sctl::sin<Real>(k*r);

      Real sin_kr2 = sctl::sin<Real>(k*r*0.5);
      Real cos_kr_ = -2 * sin_kr2 * sin_kr2;

      Real G[2], fG[2];
      G[0] = (-k*sin_kr - cos_kr_ * invr) * invr2 * scal;
      G[1] = ( k*cos_kr - sin_kr * invr) * invr2 * scal;

      fG[0] = f[0]*G[0] - f[1]*G[1];
      fG[1] = f[0]*G[1] + f[1]*G[0];

      for (sctl::Integer i = 0; i < COORD_DIM; i++) {
        v[i*2+0] += fG[0] * x[i];
        v[i*2+1] += fG[1] * x[i];
      }
    }

    Real k_;
    KernelFunction<Real,COORD_DIM,KDIM0,KDIM1*COORD_DIM> ker_FxdU;
};

}

#endif  //_KERNEL_HPP_
