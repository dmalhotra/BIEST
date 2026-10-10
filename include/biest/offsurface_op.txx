#include <algorithm>
#include <utility>

namespace biest {

namespace detail_offsurf {

  /**
   * Returns the parameter values of the points w, given in [0,1], on each of M equal panels of [0,1], panel by panel.
   * If flip is true, the parameter is reversed.
   */
  template <class Real> sctl::Vector<Real> PanelPoints(const sctl::Long M, const sctl::Vector<Real>& w, const bool flip) {
    sctl::Vector<Real> s(M * w.Dim());
    for (sctl::Long a = 0; a < M; a++) {
      for (sctl::Long i = 0; i < w.Dim(); i++) {
        const Real s_ = (a + w[i]) / M;
        s[a * w.Dim() + i] = (flip ? 1 - s_ : s_);
      }
    }
    return s;
  }

  /**
   * The Gauss-Legendre nodes of a panel, and the weights that interpolate and differentiate from them to the check
   * points, at which the interpolation error on the panel is measured.
   */
  template <class Real> struct PanelRule {
    sctl::Vector<Real> nds; // Gauss-Legendre nodes in [0,1]
    sctl::Vector<Real> w; // check points in [0,1], uniform and including the end points
    sctl::Vector<Real> L, Ld; // interpolation and differentiation weights, order x number of check points
    Real L_max, Ld_max; // largest sum over the nodes of the absolute weights at a check point

    explicit PanelRule(const sctl::Integer order) : nds(sctl::QuadElemList<Real>::ParamNodes(order)), w(2 * order) {
      const sctl::Long Nchk = w.Dim();
      for (sctl::Long c = 0; c < Nchk; c++) w[c] = c / (Real)(Nchk - 1);
      sctl::LagrangeInterp<Real>::Interpolate(L, nds, w);

      sctl::Vector<Real> I(order * order), dI; // dI[j*order+i]: derivative of the j-th basis polynomial at node i
      I.SetZero();
      for (sctl::Integer i = 0; i < order; i++) I[i * order + i] = 1;
      sctl::LagrangeInterp<Real>::Derivative(dI, I, nds);
      Ld.ReInit(order * Nchk);
      for (sctl::Integer j = 0; j < order; j++) {
        for (sctl::Long c = 0; c < Nchk; c++) {
          Real v = 0;
          for (sctl::Integer i = 0; i < order; i++) v += dI[j * order + i] * L[i * Nchk + c];
          Ld[j * Nchk + c] = v;
        }
      }

      L_max = 0;
      Ld_max = 0;
      for (sctl::Long c = 0; c < Nchk; c++) {
        Real l = 0, ld = 0;
        for (sctl::Integer i = 0; i < order; i++) {
          l += sctl::fabs<Real>(L[i * Nchk + c]);
          ld += sctl::fabs<Real>(Ld[i * Nchk + c]);
        }
        L_max = std::max<Real>(L_max, l);
        Ld_max = std::max<Real>(Ld_max, ld);
      }
    }

    /**
     * Set E and D to the largest error, at the check points, of the interpolation of exp(i theta u) from the nodes, and
     * of the derivative of the interpolation.
     */
    void ModeError(Real& E, Real& D, const Real theta) const {
      const sctl::Long order = nds.Dim(), Nchk = w.Dim();
      sctl::StaticArray<Real,64> cs; // cos and sin of theta times each node
      SCTL_ASSERT(2 * order <= 64);
      for (sctl::Long i = 0; i < order; i++) {
        cs[2 * i + 0] = sctl::cos<Real>(theta * nds[i]);
        cs[2 * i + 1] = sctl::sin<Real>(theta * nds[i]);
      }
      E = 0;
      D = 0;
      for (sctl::Long c = 0; c < Nchk; c++) {
        Real re = 0, im = 0, dre = 0, dim = 0;
        for (sctl::Long i = 0; i < order; i++) {
          re += L[i * Nchk + c] * cs[2 * i + 0];
          im += L[i * Nchk + c] * cs[2 * i + 1];
          dre += Ld[i * Nchk + c] * cs[2 * i + 0];
          dim += Ld[i * Nchk + c] * cs[2 * i + 1];
        }
        const Real cw = sctl::cos<Real>(theta * w[c]), sw = sctl::sin<Real>(theta * w[c]);
        E = std::max<Real>(E, sctl::sqrt<Real>((re - cw) * (re - cw) + (im - sw) * (im - sw)));
        D = std::max<Real>(D, sctl::sqrt<Real>((dre + theta * sw) * (dre + theta * sw) + (dim - theta * cw) * (dim - theta * cw)));
      }
    }
  };

  /**
   * Set wts to the weights of the Lagrange interpolation from the p integer points -p/2+1, ..., p/2 to the point x.
   */
  template <class Real> void LocalWeights(sctl::Iterator<Real> wts, const sctl::Integer p, const Real x) {
    for (sctl::Integer j = 0; j < p; j++) { // x at a point
      if (x == j - p / 2 + 1) {
        for (sctl::Integer m = 0; m < p; m++) wts[m] = (m == j ? 1 : 0);
        return;
      }
    }
    Real binom = 1, sum = 0; // barycentric weights of equispaced points: (-1)^j binomial(p-1, j)
    for (sctl::Integer j = 0; j < p; j++) {
      wts[j] = (j % 2 ? -binom : binom) / (x - (j - p / 2 + 1));
      sum += wts[j];
      binom = binom * (p - 1 - j) / (j + 1);
    }
    for (sctl::Integer j = 0; j < p; j++) wts[j] /= sum;
  }

  /**
   * Returns the largest error of the Lagrange interpolation of exp(i phi x) from the p integer points -p/2+1, ..., p/2 to
   * x in [0,1].
   */
  template <class Real> Real LocalModeError(const sctl::Integer p, const Real phi) {
    sctl::StaticArray<Real,64> wts;
    SCTL_ASSERT(p <= 64);
    Real err = 0;
    for (sctl::Integer q = 0; q <= 8; q++) { // x in [0,1/2]; the error is symmetric about 1/2
      const Real x = q / (Real)16;
      LocalWeights<Real>(wts, p, x);
      Real re = -sctl::cos<Real>(phi * x), im = -sctl::sin<Real>(phi * x);
      for (sctl::Integer j = 0; j < p; j++) {
        re += wts[j] * sctl::cos<Real>(phi * (j - p / 2 + 1));
        im += wts[j] * sctl::sin<Real>(phi * (j - p / 2 + 1));
      }
      err = std::max<Real>(err, sctl::sqrt<Real>(re * re + im * im));
    }
    return err;
  }

  /**
   * Set the index of the first stencil point and the weights of the p-point local interpolation, on a periodic grid of N
   * points uniform on [0,1), to each point s.
   */
  template <class Real> void LocalInterp(sctl::Vector<sctl::Long>& i0, sctl::Matrix<Real>& W, const sctl::Long N, const sctl::Vector<Real>& s, const sctl::Integer p) {
    if (i0.Dim() != s.Dim()) i0.ReInit(s.Dim());
    if (W.Dim(0) != s.Dim() || W.Dim(1) != p) W.ReInit(s.Dim(), p);
    for (sctl::Long r = 0; r < s.Dim(); r++) {
      const Real x = (s[r] - sctl::floor<Real>(s[r])) * N;
      const sctl::Long i = (sctl::Long)sctl::floor<Real>(x);
      i0[r] = i - p / 2 + 1;
      LocalWeights<Real>(W[r], p, x - i);
    }
  }

  /**
   * Sizes of the Fourier modes of fields on an Nt x Np periodic grid. Row c of each matrix is for field component c. Its
   * column k is the sum of the absolute values of the Fourier coefficients of all modes with frequency k or -k along
   * one direction.
   */
  template <class Real> struct ModeSizes {
    sctl::Matrix<Real> t, p; // frequency along the toroidal and the poloidal direction
    sctl::Matrix<Real> t_dp, p_dt; // the same, for the derivative along the other direction (parameter in [0,1))

    ModeSizes(const sctl::Vector<Real>& Y, const sctl::Long Nt, const sctl::Long Np) {
      const sctl::Long N = Nt * Np, Nc = Y.Dim() / N, Np_ = Np / 2 + 1;
      SCTL_ASSERT(Y.Dim() == Nc * N);
      t.ReInit(Nc, Nt / 2 + 1);
      p.ReInit(Nc, Np / 2 + 1);
      t_dp.ReInit(Nc, Nt / 2 + 1);
      p_dt.ReInit(Nc, Np / 2 + 1);
      t.SetZero();
      p.SetZero();
      t_dp.SetZero();
      p_dt.SetZero();
      if (!Nc) return;

      sctl::FFT<Real> fft;
      sctl::StaticArray<sctl::Long,2> fft_dim{Nt, Np};
      fft.Setup(sctl::FFT_Type::R2C, Nc, sctl::Vector<sctl::Long>(2, fft_dim, false), omp_get_max_threads());
      sctl::Vector<Real> coeff;
      fft.Execute(Y, coeff);
      const Real two_pi = 2 * sctl::const_pi<Real>();
      for (sctl::Long c = 0; c < Nc; c++) {
        for (sctl::Long i = 0; i < Nt; i++) {
          const sctl::Long k = std::min(i, Nt - i);
          for (sctl::Long l = 0; l < Np_; l++) {
            const Real re = coeff[((c * Nt + i) * Np_ + l) * 2 + 0], im = coeff[((c * Nt + i) * Np_ + l) * 2 + 1];
            const Real m = sctl::sqrt<Real>(re * re + im * im) / N * (l == 0 || 2 * l == Np ? 1 : 2); // with the mode (-i,-l)
            t[c][k] += m;
            p[c][l] += m;
            t_dp[c][k] += m * two_pi * l;
            p_dt[c][l] += m * two_pi * k;
          }
        }
      }
    }
  };

  /**
   * Returns the largest over the rows of S of the bound on an interpolation error, from the mode sizes in the row and the
   * error err[k] for a mode of frequency k, less the rounding error, and multiplied by scal.
   *
   * @param[in] S the mode sizes, one row per field component.
   *
   * @param[in] err the error for a mode of size 1, for each frequency.
   *
   * @param[in] round the rounding error of the interpolation of a field of size 1.
   *
   * @param[in] scal the scale of the error.
   */
  template <class Real> Real ModeBound(const sctl::Matrix<Real>& S, const sctl::Vector<Real>& err, const Real round, const Real scal) {
    Real bound = 0;
    for (sctl::Long c = 0; c < S.Dim(0); c++) {
      Real b = 0, size = 0;
      for (sctl::Long k = 0; k < S.Dim(1); k++) {
        b += S[c][k] * err[k];
        size += S[c][k];
      }
      bound = std::max<Real>(bound, (b - round * size) * scal);
    }
    return bound;
  }

  /**
   * Returns the largest over the rows of S, each multiplied by scal, of the bound on the error at the check points of the
   * interpolation on M panels (or of its derivative), from the mode sizes in the row, less the rounding error.
   */
  template <class Real> Real PanelBound(const PanelRule<Real>& rule, const sctl::Long M, const sctl::Matrix<Real>& S, const Real scal, const bool derivative) {
    sctl::Vector<Real> err(S.Dim(1));
    for (sctl::Long k = 0; k < S.Dim(1); k++) {
      Real E, D;
      rule.ModeError(E, D, 2 * sctl::const_pi<Real>() * k / M);
      err[k] = (derivative ? M * D : E);
    }
    const Real round = 16 * sctl::machine_eps<Real>() * (derivative ? M * rule.Ld_max : rule.L_max);
    return ModeBound<Real>(S, err, round, scal);
  }

  /**
   * Returns the largest over the rows of S, each multiplied by scal, of the bound on the error of the p-point local
   * interpolation on the grid of N points upsampled by U, from the mode sizes in the row, less the rounding error.
   */
  template <class Real> Real LocalBound(const sctl::Integer p, const sctl::Long U, const sctl::Long N, const sctl::Matrix<Real>& S, const Real scal) {
    sctl::Vector<Real> err(S.Dim(1));
    for (sctl::Long k = 0; k < S.Dim(1); k++) err[k] = LocalModeError<Real>(p, 2 * sctl::const_pi<Real>() * k / (U * N));
    Real wts_sum = 0; // largest sum of the absolute values of the stencil weights
    {
      sctl::StaticArray<Real,64> wts;
      for (sctl::Integer q = 0; q <= 8; q++) {
        LocalWeights<Real>(wts, p, q / (Real)16);
        Real sum = 0;
        for (sctl::Integer j = 0; j < p; j++) sum += sctl::fabs<Real>(wts[j]);
        wts_sum = std::max<Real>(wts_sum, sum);
      }
    }
    return ModeBound<Real>(S, err, 16 * sctl::machine_eps<Real>() * wts_sum, scal);
  }

  /**
   * Returns the smallest M in [M0, Mmax] for which resolved(M) holds, by doubling and then bisection, assuming that it
   * holds for every larger M. Returns Mmax if resolved(Mmax) does not hold.
   */
  template <class Pred> sctl::Long SmallestCount(const sctl::Long M0, const sctl::Long Mmax, const Pred& resolved) {
    if (resolved(M0)) return M0;
    sctl::Long lo = M0, hi = M0; // resolved(lo) does not hold
    while (true) {
      hi = std::min(Mmax, 2 * lo);
      if (hi == lo) return Mmax;
      if (resolved(hi)) break;
      if (hi == Mmax) return Mmax;
      lo = hi;
    }
    while (hi - lo > 1) {
      const sctl::Long mid = (lo + hi) / 2;
      if (resolved(mid)) hi = mid;
      else lo = mid;
    }
    return hi;
  }

  /**
   * Returns the reciprocal of the largest absolute value in V, or 1 if V is zero or empty.
   */
  template <class Real> Real InvMaxAbs(const sctl::Vector<Real>& V) {
    Real max = 0;
    for (const auto& v : V) max = std::max<Real>(max, sctl::fabs<Real>(v));
    return (max > 0 ? 1 / max : (Real)1);
  }

}

template <class Real, class SKer> template <sctl::Integer DIGITS, sctl::Integer Nv> OffSurfaceOp<Real,SKer>::OffSurfaceOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) : ker_(ker.GetSctlKernel()), comm_(comm), tol_(0), Nt_(0), Np_(0), Ntrg(0), trg0(0), trg1(0), Mt(0), Mp(0), Ut(1), Up(1), flip(false) {
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::Setup(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Xt, const Real tol, const sctl::Vector<Real>& F) {
  SCTL_ASSERT(X.Dim() == COORD_DIM * Nt * Np);
  Nt_ = Nt;
  Np_ = Np;
  X_ = X;
  tol_ = tol;

  Ntrg = Xt.Dim() / COORD_DIM;
  SCTL_ASSERT(Xt.Dim() == COORD_DIM * Ntrg);
  trg0 = (comm_.Rank() + 0) * Ntrg / comm_.Size();
  trg1 = (comm_.Rank() + 1) * Ntrg / comm_.Size();
  Xt_.ReInit(COORD_DIM * (trg1 - trg0));
  for (sctl::Long i = trg0; i < trg1; i++) {
    for (sctl::Integer k = 0; k < COORD_DIM; k++) Xt_[(i - trg0) * COORD_DIM + k] = Xt[k * Ntrg + i];
  }

  Mt = 4;
  Mp = 4;
  Ut = 1;
  Up = 1;
  FitPanels(F);
  BuildPanels();
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) {
  const sctl::Long N = Nt_ * Np_;
  const sctl::Long dof = F.Dim() / (KDIM0 * N);
  SCTL_ASSERT(F.Dim() == dof * KDIM0 * N);
  if (!Resolves(F)) {
    FitPanels(F);
    BuildPanels();
  }

  const sctl::Long Nsrc = KDIM0 * Mt * Mp * ORDER * ORDER; // values of a density at the panel nodes
  const sctl::Long Nloc = trg1 - trg0;
  sctl::ScratchBuf<Real> Fnds_buf(dof * Nsrc), Utrg_buf(KDIM1 * Nloc), Uloc_buf(dof * KDIM1 * Ntrg), Uglb_buf(dof * KDIM1 * Ntrg);
  sctl::Vector<Real> Fnds(Fnds_buf), Utrg(Utrg_buf), Uloc(Uloc_buf), Uglb(Uglb_buf); // Utrg: potential at the targets of this process, AoS
  InterpToNodes(Fnds, F, KDIM0);
  Uloc.SetZero();
  for (sctl::Long d = 0; d < dof; d++) {
    const sctl::Vector<Real> F_(Nsrc, Fnds.begin() + d * Nsrc, false);
    op->ComputePotential(Utrg, F_);
    for (sctl::Long i = trg0; i < trg1; i++) {
      for (sctl::Integer k = 0; k < KDIM1; k++) Uloc[(d * KDIM1 + k) * Ntrg + i] = Utrg[(i - trg0) * KDIM1 + k];
    }
  }
  comm_.Allreduce(Uloc.begin(), Uglb.begin(), Uglb.Dim(), sctl::CommOp::SUM);
  if (U.Dim() != dof * KDIM1 * Ntrg) {
    U.ReInit(dof * KDIM1 * Ntrg);
    U.SetZero();
  }
  U += Uglb;
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::FitPanels(const sctl::Vector<Real>& F) {
  const sctl::Long N = Nt_ * Np_;
  SCTL_ASSERT(F.Dim() % N == 0);
  const detail_offsurf::PanelRule<Real> rule(ORDER);
  const detail_offsurf::ModeSizes<Real> Sx(X_, Nt_, Np_), Sf(F, Nt_, Np_);

  Real st, sp; // reciprocals of the largest derivative of X along each direction
  {
    sctl::Vector<Real> dX; // dX/dt and dX/dp, for each component of X
    SurfaceOp<Real>(sctl::Comm::Self(), Nt_, Np_).Grad2D(dX, X_);
    Real max_t = 0, max_p = 0;
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < N; i++) {
        max_t = std::max<Real>(max_t, sctl::fabs<Real>(dX[(k * 2 + 0) * N + i]));
        max_p = std::max<Real>(max_p, sctl::fabs<Real>(dX[(k * 2 + 1) * N + i]));
      }
    }
    st = (max_t > 0 ? 1 / max_t : 1);
    sp = (max_p > 0 ? 1 / max_p : 1);
  }
  const Real sX = detail_offsurf::InvMaxAbs<Real>(X_), sF = detail_offsurf::InvMaxAbs<Real>(F);

  { // Set Mt, Mp: in each direction the values of X, F and the derivative of X along the other direction, and the
    // derivative of X along this direction, which give the normal and the area element
    const auto resolved_t = [&](const sctl::Long M) {
      return detail_offsurf::PanelBound<Real>(rule, M, Sx.t, sX, false) <= tol_ && detail_offsurf::PanelBound<Real>(rule, M, Sf.t, sF, false) <= tol_ &&
             detail_offsurf::PanelBound<Real>(rule, M, Sx.t_dp, sp, false) <= tol_ && detail_offsurf::PanelBound<Real>(rule, M, Sx.t, st, true) <= tol_;
    };
    const auto resolved_p = [&](const sctl::Long M) {
      return detail_offsurf::PanelBound<Real>(rule, M, Sx.p, sX, false) <= tol_ && detail_offsurf::PanelBound<Real>(rule, M, Sf.p, sF, false) <= tol_ &&
             detail_offsurf::PanelBound<Real>(rule, M, Sx.p_dt, st, false) <= tol_ && detail_offsurf::PanelBound<Real>(rule, M, Sx.p, sp, true) <= tol_;
    };
    const sctl::Long Mt_max = std::max<sctl::Long>(Mt, (8 * Nt_ + ORDER - 1) / ORDER); // at most 8 panel nodes per grid point
    const sctl::Long Mp_max = std::max<sctl::Long>(Mp, (8 * Np_ + ORDER - 1) / ORDER);
    Mt = detail_offsurf::SmallestCount(Mt, Mt_max, resolved_t);
    Mp = detail_offsurf::SmallestCount(Mp, Mp_max, resolved_p);
    if (!resolved_t(Mt) || !resolved_p(Mp)) SCTL_WARN("The surface or the density is not resolved on its grid; the panels may not reach the requested accuracy.");
  }

  { // Set Ut, Up so that the error of the local interpolation, which differentiation on a panel amplifies, is within tol/2
    const Real cX = std::max<Real>(rule.L_max * sX, rule.Ld_max * std::max<Real>(Mt * st, Mp * sp)), cF = rule.L_max * sF;
    const auto resolved = [this,cX,cF](const sctl::Long U, const sctl::Long Ng, const sctl::Matrix<Real>& SX, const sctl::Matrix<Real>& SF) {
      return detail_offsurf::LocalBound<Real>(STENCIL, U, Ng, SX, cX) <= tol_ / 2 && detail_offsurf::LocalBound<Real>(STENCIL, U, Ng, SF, cF) <= tol_ / 2;
    };
    const auto upsample_factor = [&resolved](const sctl::Long U0, const sctl::Long Ng, const sctl::Matrix<Real>& SX, const sctl::Matrix<Real>& SF) {
      for (const sctl::Long U : {1, 2, 3, 4, 6, 8}) {
        if (U >= U0 && resolved(U, Ng, SX, SF)) return U;
      }
      SCTL_WARN("The local interpolation of the surface or the density may not reach the requested accuracy.");
      return std::max<sctl::Long>(U0, 8);
    };
    Ut = upsample_factor(Ut, Nt_, Sx.t, Sf.t);
    Up = upsample_factor(Up, Np_, Sx.p, Sf.p);
  }
}

template <class Real, class SKer> bool OffSurfaceOp<Real,SKer>::Resolves(const sctl::Vector<Real>& F) const {
  const detail_offsurf::PanelRule<Real> rule(ORDER);
  const detail_offsurf::ModeSizes<Real> Sf(F, Nt_, Np_);
  const Real sF = detail_offsurf::InvMaxAbs<Real>(F), cF = rule.L_max * sF;
  return detail_offsurf::PanelBound<Real>(rule, Mt, Sf.t, sF, false) <= tol_ && detail_offsurf::PanelBound<Real>(rule, Mp, Sf.p, sF, false) <= tol_ &&
         detail_offsurf::LocalBound<Real>(STENCIL, Ut, Nt_, Sf.t, cF) <= tol_ / 2 && detail_offsurf::LocalBound<Real>(STENCIL, Up, Np_, Sf.p, cF) <= tol_ / 2;
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::BuildPanels() {
  const sctl::Vector<Real>& nds = sctl::QuadElemList<Real>::ParamNodes(ORDER);
  detail_offsurf::LocalInterp<Real>(it0, Wt, Ut * Nt_, detail_offsurf::PanelPoints<Real>(Mt, nds, false), STENCIL);

  ElemList elem_lst;
  const auto build = [this,&nds,&elem_lst](const bool flip_) { // Set ip0, Wp, elem_lst
    flip = flip_;
    detail_offsurf::LocalInterp<Real>(ip0, Wp, Up * Np_, detail_offsurf::PanelPoints<Real>(Mp, nds, flip), STENCIL);
    sctl::Vector<Real> coord;
    InterpToNodes(coord, X_, COORD_DIM);
    elem_lst.Init(ORDER, coord);
  };

  build(false);
  { // Rebuild with the poloidal parameter reversed if the normal points inward at the node of largest x
    sctl::Vector<Real> Xnds, Xn;
    elem_lst.GetNodeCoord(&Xnds, &Xn, nullptr);
    sctl::Long idx = 0;
    for (sctl::Long i = 0; i < Xnds.Dim() / COORD_DIM; i++) {
      if (Xnds[i * COORD_DIM] > Xnds[idx * COORD_DIM]) idx = i;
    }
    if (Xn[idx * COORD_DIM] < 0) build(true);
  }

  op.reset(new sctl::BoundaryIntegralOp<Real,SKer>(ker_, false, sctl::Comm::Self()));
  op->SetAccuracy(tol_);
  op->AddElemList(elem_lst);
  op->SetTargetCoord(Xt_);
  op->Setup();
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::InterpToNodes(sctl::Vector<Real>& Ynds, const sctl::Vector<Real>& Y, const sctl::Integer kdim) const {
  const sctl::Long N = Nt_ * Np_, Nelem = Mt * Mp, Nnds = ORDER * ORDER;
  const sctl::Long Nc = Y.Dim() / N; // number of components over all fields
  SCTL_ASSERT(Y.Dim() == Nc * N);
  SCTL_ASSERT(Nc % kdim == 0);
  if (Ynds.Dim() != Nc * Nelem * Nnds) Ynds.ReInit(Nc * Nelem * Nnds);

  const sctl::Long Ntf = Ut * Nt_, Npf = Up * Np_; // upsampled grid
  sctl::Vector<Real> Yf;
  if (Ut == 1 && Up == 1) Yf.ReInit(Y.Dim(), (sctl::Iterator<Real>)Y.begin(), false);
  else SurfaceOp<Real>::Resample(Yf, Ntf, Npf, Y, Nt_, Np_);

  const auto wrap = [](const sctl::Long i, const sctl::Long n) { return ((i % n) + n) % n; };
  // T_buf: a component interpolated along the toroidal direction, each row extended periodically by STENCIL values on
  // each side so that the poloidal stencils need no wrap-around
  const sctl::Long Rt = Mt * ORDER, Tdim = Npf + 2 * STENCIL;
  sctl::ScratchBuf<Real> T_buf(Rt * Tdim);
  for (sctl::Long c = 0; c < Nc; c++) {
    const sctl::Long d = c / kdim, k = c % kdim; // field and component
    #pragma omp parallel for schedule(static)
    for (sctl::Long r = 0; r < Rt; r++) {
      const sctl::Long n = Npf;
      Real* __restrict__ Tr = T_buf.begin() + r * Tdim + STENCIL;
      for (sctl::Long j = 0; j < n; j++) Tr[j] = 0;
      for (sctl::Integer m = 0; m < STENCIL; m++) {
        const Real w = Wt[r][m];
        const Real* __restrict__ Yrow = &Yf[(c * Ntf + wrap(it0[r] + m, Ntf)) * n];
        for (sctl::Long j = 0; j < n; j++) Tr[j] += w * Yrow[j];
      }
      for (sctl::Integer j = 0; j < STENCIL; j++) {
        Tr[j - STENCIL] = Tr[wrap(j - STENCIL, n)];
        Tr[n + j] = Tr[wrap(n + j, n)];
      }
    }
    #pragma omp parallel for schedule(static)
    for (sctl::Long e = 0; e < Nelem; e++) {
      const sctl::Long a = e / Mp, b = e % Mp;
      for (sctl::Integer i = 0; i < ORDER; i++) {
        const Real* Tr = T_buf.begin() + (a * ORDER + i) * Tdim + STENCIL;
        for (sctl::Integer j = 0; j < ORDER; j++) {
          const sctl::Long q = b * ORDER + j;
          const Real* x = Tr + ip0[q];
          const Real* w = &Wp[q][0];
          Real v = 0;
          for (sctl::Integer m = 0; m < STENCIL; m++) v += w[m] * x[m];
          Ynds[((d * Nelem + e) * Nnds + i * ORDER + j) * kdim + k] = v;
        }
      }
    }
  }
}

}
