#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

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
   * Returns the largest error of the Lagrange interpolation of exp(i phi x) from the p integer points -p/2+1, ..., p/2
   * to x in [0,1].
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
   * Set the p-point local interpolation from a periodic grid of N points uniform on [0,1) to the nodes s of M equal
   * panels, as one block of weights per panel.
   *
   * @param[out] w0 the index of the first of the K grid points of each panel; it may lie outside [0,N).
   *
   * @param[out] W the weights of each node (row) on the K grid points of its panel, zero outside its stencil.
   *
   * @param[out] K the number of consecutive grid points of each block, spanning the stencils of the nodes of any panel.
   */
  template <class Real> void PanelLocalInterp(sctl::Vector<sctl::Long>& w0, sctl::Matrix<Real>& W, sctl::Long& K, const sctl::Long N, const sctl::Vector<Real>& s, const sctl::Long M, const sctl::Integer p) {
    const sctl::Long order = s.Dim() / M;
    sctl::Vector<sctl::Long> i0(s.Dim()); // first stencil point of each node
    sctl::Matrix<Real> Ws(s.Dim(), p); // stencil weights of each node
    for (sctl::Long r = 0; r < s.Dim(); r++) {
      const Real x = (s[r] - sctl::floor<Real>(s[r])) * N;
      const sctl::Long i = (sctl::Long)sctl::floor<Real>(x);
      i0[r] = i - p / 2 + 1;
      LocalWeights<Real>(Ws[r], p, x - i);
    }

    if (w0.Dim() != M) w0.ReInit(M);
    K = 0;
    for (sctl::Long a = 0; a < M; a++) {
      const auto [lo, hi] = std::minmax_element(i0.begin() + a * order, i0.begin() + (a + 1) * order);
      w0[a] = *lo;
      K = std::max<sctl::Long>(K, *hi + p - *lo);
    }
    W.ReInit(s.Dim(), K);
    W.SetZero();
    for (sctl::Long r = 0; r < s.Dim(); r++) {
      for (sctl::Integer m = 0; m < p; m++) W[r][i0[r] - w0[r / order] + m] = Ws[r][m];
    }
  }

  /**
   * Fourier mode sizes of fields on an Nt x Np periodic grid: for each component and frequency k along a direction, the
   * sum of the absolute values of its coefficients.
   */
  template <class Real> struct ModeSizes {
    sctl::Matrix<Real> t, p; // frequency k or -k along the toroidal and the poloidal direction, for each component
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
            // with the mode (-i,-l), the conjugate
            const Real m = sctl::sqrt<Real>(re * re + im * im) / N * (l == 0 || 2 * l == Np ? 1 : 2);
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
   * Returns the largest over the rows of S of the bound on an interpolation error, less the rounding error, times scal.
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
   * Returns ModeBound() for the interpolation on M panels, or its derivative, from the mode sizes S, measured at the
   * check points.
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
   * Returns the lines of the components of Y along the toroidal (dir = 0) or poloidal (dir = 1) direction, as the rows
   * of a matrix.
   *
   * @param[in] Y fields on an Nt x Np grid in SoA order.
   */
  template <class Real> sctl::Matrix<Real> GridLines(const sctl::Vector<Real>& Y, const sctl::Long Nt, const sctl::Long Np, const sctl::Integer dir) {
    const sctl::Long N = Nt * Np, Nc = Y.Dim() / N;
    sctl::Matrix<Real> R(dir ? Nc * Nt : Nc * Np, dir ? Np : Nt);
    for (sctl::Long c = 0; c < Nc; c++) {
      for (sctl::Long i = 0; i < Nt; i++) {
        for (sctl::Long j = 0; j < Np; j++) {
          if (dir) R[c * Nt + i][j] = Y[(c * Nt + i) * Np + j];
          else R[c * Np + j][i] = Y[(c * Nt + i) * Np + j];
        }
      }
    }
    return R;
  }

  /**
   * Returns the smallest upsampling factor U, at least U0, from 1, 2, 3, 4, 6 and 8, at which the local interpolation
   * along the rows of R is within tol.
   *
   * @param[in] R rows of values uniform on a period.
   *
   * @param[in] scal the scale of the error.
   *
   * @return the factor, or -1 if there is none.
   */
  template <class Real> sctl::Long LocalUpsample(const sctl::Matrix<Real>& R, const Real scal, const Real tol, const sctl::Integer p, const sctl::Long U0) {
    constexpr sctl::Long Nsample = 16; // number of rows on which the error is measured
    const sctl::Long Nr = R.Dim(0), N = R.Dim(1), Nk = N / 2 + 1;
    if (!Nr) return U0;

    sctl::Vector<Real> coeff; // Fourier coefficients of the rows, divided by N
    {
      sctl::FFT<Real> fft;
      sctl::StaticArray<sctl::Long,1> dim{N};
      fft.Setup(sctl::FFT_Type::R2C, Nr, sctl::Vector<sctl::Long>(1, dim, false), omp_get_max_threads());
      fft.Execute(sctl::Vector<Real>(Nr * N, (sctl::Iterator<Real>)R.begin(), false), coeff);
      for (auto& c : coeff) c /= N;
    }
    sctl::StaticArray<Real,64> wmid; // stencil weights at the midpoint of a grid cell
    SCTL_ASSERT(p <= 64);
    Real round = 0; // rounding error of the local interpolation of values of size 1
    for (sctl::Integer q = 0; q <= 8; q++) {
      LocalWeights<Real>(wmid, p, q / (Real)16);
      Real sum = 0;
      for (sctl::Integer j = 0; j < p; j++) sum += sctl::fabs<Real>(wmid[j]);
      round = std::max<Real>(round, 16 * sctl::machine_eps<Real>() * sum);
    }
    LocalWeights<Real>(wmid, p, (Real)0.5);

    // the error, less the rounding error, is measured at the midpoints of the upsampled grid, where it is largest, on
    // the rows with the largest error estimated from their Fourier coefficients
    for (const sctl::Long U : {1, 2, 3, 4, 6, 8}) {
      if (U < U0) continue;
      sctl::Vector<sctl::Long> rows; // the rows with the largest estimated error, from their mode sizes
      {
        sctl::Vector<Real> err_k(Nk), est(Nr);
        for (sctl::Long k = 0; k < Nk; k++) err_k[k] = LocalModeError<Real>(p, 2 * sctl::const_pi<Real>() * k / (U * N));
        #pragma omp parallel for schedule(static)
        for (sctl::Long r = 0; r < Nr; r++) {
          Real e = 0, size = 0;
          for (sctl::Long k = 0; k < Nk; k++) {
            const Real m = sctl::sqrt<Real>(coeff[(r * Nk + k) * 2] * coeff[(r * Nk + k) * 2] + coeff[(r * Nk + k) * 2 + 1] * coeff[(r * Nk + k) * 2 + 1]) * (k == 0 || 2 * k == N ? 1 : 2);
            e += m * err_k[k];
            size += m;
          }
          est[r] = (e - round * size) * scal;
        }
        std::vector<sctl::Long> order_(Nr);
        std::iota(order_.begin(), order_.end(), 0);
        const sctl::Long Ns = std::min(Nsample, Nr);
        std::partial_sort(order_.begin(), order_.begin() + Ns, order_.end(), [&est](sctl::Long a, sctl::Long b) { return est[a] > est[b]; });
        if (est[order_[0]] <= tol) return U; // the estimate already meets tol
        rows.ReInit(Ns);
        for (sctl::Long i = 0; i < Ns; i++) rows[i] = order_[i];
      }

      // the rows upsampled 2U times: the even points are the rows upsampled U times, the odd points the midpoints
      const sctl::Long Nf = 2 * U * N, Nfk = Nf / 2 + 1, Ns = rows.Dim();
      sctl::Vector<Real> fine_coeff(Ns * Nfk * 2), fine;
      fine_coeff.SetZero();
      for (sctl::Long i = 0; i < Ns; i++) {
        for (sctl::Long k = 0; k < Nk; k++) {
          const Real scal_k = (2 * k == N ? (Real)0.5 : (Real)1); // the Nyquist mode of even N is a cosine
          fine_coeff[(i * Nfk + k) * 2 + 0] = coeff[(rows[i] * Nk + k) * 2 + 0] * scal_k;
          fine_coeff[(i * Nfk + k) * 2 + 1] = coeff[(rows[i] * Nk + k) * 2 + 1] * scal_k;
        }
      }
      {
        sctl::FFT<Real> fft;
        sctl::StaticArray<sctl::Long,1> dim{Nf};
        fft.Setup(sctl::FFT_Type::C2R, Ns, sctl::Vector<sctl::Long>(1, dim, false), omp_get_max_threads());
        fft.Execute(fine_coeff, fine, false);
      }
      Real err = 0;
      for (sctl::Long i = 0; i < Ns; i++) {
        const sctl::Long n = U * N;
        const sctl::ConstIterator<Real> g = fine.begin() + i * Nf;
        Real e = 0, size = 0;
        for (sctl::Long j = 0; j < n; j++) {
          Real v = 0;
          for (sctl::Integer m = 0; m < p; m++) v += wmid[m] * g[2 * (((j - p / 2 + 1 + m) % n + n) % n)];
          e = std::max<Real>(e, sctl::fabs<Real>(v - g[2 * j + 1]));
          size = std::max<Real>(size, sctl::fabs<Real>(g[2 * j]));
        }
        err = std::max<Real>(err, (e - round * size) * scal);
      }
      if (err <= tol) return U;
    }
    return -1;
  }

  /**
   * Returns the smallest M in [M0, Mmax] for which resolved(M) holds, found by doubling and bisection, or Mmax if
   * resolved(Mmax) does not hold.
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

template <class Real, class SKer> template <sctl::Integer DIGITS, sctl::Integer Nv> OffSurfaceOp<Real,SKer>::OffSurfaceOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) : ker_(ker.GetSctlKernel()), comm_(comm), tol_(0), Nt_(0), Np_(0), Ntrg(0), trg0(0), trg1(0), Mt(0), Mp(0), Ut(1), Up(1), flip(false), Kt(0), Kp(0) {
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
  // Utrg: the potential at the targets of this process, AoS
  sctl::Vector<Real> Fnds(Fnds_buf), Utrg(Utrg_buf), Uloc(Uloc_buf), Uglb(Uglb_buf);
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
    // at most 8 panel nodes per grid point
    const sctl::Long Mt_max = std::max<sctl::Long>(Mt, (8 * Nt_ + ORDER - 1) / ORDER);
    const sctl::Long Mp_max = std::max<sctl::Long>(Mp, (8 * Np_ + ORDER - 1) / ORDER);
    Mt = detail_offsurf::SmallestCount(Mt, Mt_max, resolved_t);
    Mp = detail_offsurf::SmallestCount(Mp, Mp_max, resolved_p);
    if (!resolved_t(Mt) || !resolved_p(Mp)) SCTL_WARN("The surface or the density is not resolved on its grid; the panels may not reach the requested accuracy.");
  }

  { // Set Ut, Up: the error of the local interpolation, which differentiation on a panel amplifies, is within tol/2
    const Real cX = std::max<Real>(rule.L_max * sX, rule.Ld_max * std::max<Real>(Mt * st, Mp * sp)), cF = rule.L_max * sF;
    const auto upsample_factor = [this,&F,cX,cF](const sctl::Long U0, const sctl::Integer dir) {
      const sctl::Long UX = detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(X_, Nt_, Np_, dir), cX, tol_ / 2, STENCIL, U0);
      const sctl::Long U = (UX < 0 ? UX : detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, dir), cF, tol_ / 2, STENCIL, UX));
      if (U < 0) {
        SCTL_WARN("The local interpolation of the surface or the density may not reach the requested accuracy.");
        return std::max<sctl::Long>(U0, 8);
      }
      return U;
    };
    Ut = upsample_factor(Ut, 0);
    Up = upsample_factor(Up, 1);
  }
}

template <class Real, class SKer> bool OffSurfaceOp<Real,SKer>::Resolves(const sctl::Vector<Real>& F) const {
  const detail_offsurf::PanelRule<Real> rule(ORDER);
  const detail_offsurf::ModeSizes<Real> Sf(F, Nt_, Np_);
  const Real sF = detail_offsurf::InvMaxAbs<Real>(F), cF = rule.L_max * sF;
  return detail_offsurf::PanelBound<Real>(rule, Mt, Sf.t, sF, false) <= tol_ && detail_offsurf::PanelBound<Real>(rule, Mp, Sf.p, sF, false) <= tol_ &&
         detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, 0), cF, tol_ / 2, STENCIL, Ut) == Ut &&
         detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, 1), cF, tol_ / 2, STENCIL, Up) == Up;
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::BuildPanels() {
  const sctl::Vector<Real>& nds = sctl::QuadElemList<Real>::ParamNodes(ORDER);
  detail_offsurf::PanelLocalInterp<Real>(wt0, Wt, Kt, Ut * Nt_, detail_offsurf::PanelPoints<Real>(Mt, nds, false), Mt, STENCIL);

  ElemList elem_lst;
  const auto build = [this,&nds,&elem_lst](const bool flip_) { // Set wp0, Kp, WpT, elem_lst
    flip = flip_;
    sctl::Matrix<Real> Wp;
    detail_offsurf::PanelLocalInterp<Real>(wp0, Wp, Kp, Up * Np_, detail_offsurf::PanelPoints<Real>(Mp, nds, flip), Mp, STENCIL);
    WpT.ReInit(Mp * Kp, ORDER);
    for (sctl::Long b = 0; b < Mp; b++) {
      for (sctl::Long k = 0; k < Kp; k++) {
        for (sctl::Integer j = 0; j < ORDER; j++) WpT[b * Kp + k][j] = Wp[b * ORDER + j][k];
      }
    }
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
  // T_buf: a component interpolated along the toroidal direction, each row extended periodically by PL values on the
  // left and Kp + STENCIL on the right, so that the poloidal blocks need no wrap-around
  const sctl::Long Rt = Mt * ORDER, PL = STENCIL, Tdim = PL + Npf + Kp + STENCIL;
  sctl::ScratchBuf<Real> T_buf(Rt * Tdim);
  const sctl::SmallGEMM<Real,ORDER> gemm_t(false, ORDER, Npf, Kt, Kt, Npf, Tdim); // rows of a toroidal panel
  const sctl::SmallGEMM<Real,ORDER,ORDER> gemm_p(false, ORDER, ORDER, Kp, Tdim, ORDER, ORDER); // nodes of an element
  for (sctl::Long c = 0; c < Nc; c++) {
    const sctl::Long d = c / kdim, k = c % kdim; // field and component
    #pragma omp parallel for schedule(static)
    for (sctl::Long a = 0; a < Mt; a++) {
      const sctl::Long w0 = wt0[a];
      const bool contiguous = (w0 >= 0 && w0 + Kt <= Ntf);
      sctl::ScratchBuf<Real> rows_buf(contiguous ? 0 : Kt * Npf); // the Kt rows of the panel, wrapped around
      if (!contiguous) {
        for (sctl::Long i = 0; i < Kt; i++) {
          for (sctl::Long j = 0; j < Npf; j++) rows_buf[i * Npf + j] = Yf[(c * Ntf + wrap(w0 + i, Ntf)) * Npf + j];
        }
      }
      const sctl::ConstIterator<Real> B = (contiguous ? Yf.begin() + (c * Ntf + w0) * Npf : (sctl::ConstIterator<Real>)rows_buf.begin());
      gemm_t(T_buf.begin() + a * ORDER * Tdim + PL, Wt[a * ORDER], B);
      for (sctl::Integer i = 0; i < ORDER; i++) {
        const sctl::Iterator<Real> Tr = T_buf.begin() + (a * ORDER + i) * Tdim + PL;
        for (sctl::Long j = -PL; j < 0; j++) Tr[j] = Tr[wrap(j, Npf)];
        for (sctl::Long j = Npf; j < Npf + Kp + STENCIL; j++) Tr[j] = Tr[wrap(j, Npf)];
      }
    }
    #pragma omp parallel for schedule(static)
    for (sctl::Long e = 0; e < Nelem; e++) {
      const sctl::Long a = e / Mp, b = e % Mp;
      const sctl::ConstIterator<Real> A = T_buf.begin() + a * ORDER * Tdim + PL + wp0[b];
      if (kdim == 1) {
        gemm_p(Ynds.begin() + (d * Nelem + e) * Nnds, A, WpT[b * Kp]);
      } else {
        sctl::StaticArray<Real,ORDER*ORDER> C;
        gemm_p(C, A, WpT[b * Kp]);
        for (sctl::Long i = 0; i < Nnds; i++) Ynds[((d * Nelem + e) * Nnds + i) * kdim + k] = C[i];
      }
    }
  }
}

}
