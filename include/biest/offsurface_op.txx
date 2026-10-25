// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace biest {

namespace detail_offsurf {

  /**
   * The Gauss-Legendre nodes of a panel, and the weights that interpolate and differentiate from them to the check
   * points; L_max and Ld_max bound how much the interpolation and the differentiation on a panel amplify errors in
   * the values at the nodes.
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
   * Set i0 and w to the first grid point and the weights of the p-point local interpolation from a periodic grid of N
   * points uniform on [0,1) to the parameter value s. The index i0 may lie outside [0,N).
   */
  template <class Real> void LocalStencil(sctl::Long& i0, sctl::Iterator<Real> w, const sctl::Long N, const Real s, const sctl::Integer p) {
    const Real x = (s - sctl::floor<Real>(s)) * N;
    const sctl::Long i = (sctl::Long)sctl::floor<Real>(x);
    i0 = i - p / 2 + 1;
    LocalWeights<Real>(w, p, x - i);
  }

  /**
   * Interpolate fields on a periodic Nt x Np grid to the order x order nodes of each panel, by p-point local
   * interpolation along each direction.
   *
   * @param[out] Ynds the values for each component c, panel e and node (i,j), with i along the toroidal and j along the
   * poloidal direction: Ynds[((c * P + e) * order + i) * order + j] for P panels.
   *
   * @param[in] Y the fields on the grid in SoA order.
   *
   * @param[in] Mt0, Mp0 the numbers of panels of the coarse mesh in each direction.
   *
   * @param[in] nds the nodes of a panel in [0,1].
   *
   * @param[in] flip whether the poloidal nodes of each panel are in decreasing order.
   */
  template <class Real> void PanelNodeValues(sctl::Vector<Real>& Ynds, const sctl::Vector<Real>& Y, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Panel>& panels, const sctl::Long Mt0, const sctl::Long Mp0, const sctl::Vector<Real>& nds, const bool flip, const sctl::Integer p) {
    const sctl::Long N = Nt * Np, Nc = Y.Dim() / N, P = panels.Dim(), order = nds.Dim();
    SCTL_ASSERT(Y.Dim() == Nc * N);
    SCTL_ASSERT(p <= 64);
    if (Ynds.Dim() != Nc * P * order * order) Ynds.ReInit(Nc * P * order * order);

    // the panels in the order of their toroidal intervals: the panels of an interval share the interpolation along the
    // toroidal direction; grp holds the position in idx of the first panel of each interval, and P
    std::vector<sctl::Long> idx(P), grp;
    std::iota(idx.begin(), idx.end(), 0);
    const auto interval = [&panels](const sctl::Long e) { return std::make_pair(panels[e].lt, panels[e].it); };
    std::sort(idx.begin(), idx.end(), [&interval](const sctl::Long a, const sctl::Long b) { return interval(a) < interval(b); });
    for (sctl::Long n = 0; n < P; n++) {
      if (!n || interval(idx[n]) != interval(idx[n - 1])) grp.push_back(n);
    }
    grp.push_back(P);

    const auto wrap = [](const sctl::Long i, const sctl::Long n) { return ((i % n) + n) % n; };
    #pragma omp parallel for schedule(dynamic)
    for (sctl::Long g = 0; g < (sctl::Long)grp.size() - 1; g++) {
      const Panel& q0 = panels[idx[grp[g]]];
      sctl::Vector<Real> T(Nc * order * Np); // the fields at the toroidal nodes of the interval, on the poloidal grid
      T.SetZero();
      sctl::StaticArray<Real,64> w;
      for (sctl::Long i = 0; i < order; i++) {
        sctl::Long i0;
        LocalStencil<Real>(i0, w, Nt, (q0.it + nds[i]) / (Real)(Mt0 << q0.lt), p);
        for (sctl::Long c = 0; c < Nc; c++) {
          const sctl::Iterator<Real> Tr = T.begin() + (c * order + i) * Np;
          for (sctl::Integer m = 0; m < p; m++) {
            const sctl::ConstIterator<Real> Yr = Y.begin() + (c * Nt + wrap(i0 + m, Nt)) * Np;
            for (sctl::Long j = 0; j < Np; j++) Tr[j] += w[m] * Yr[j];
          }
        }
      }
      for (sctl::Long n = grp[g]; n < grp[g + 1]; n++) {
        const sctl::Long e = idx[n];
        const Panel& q = panels[e];
        for (sctl::Long j = 0; j < order; j++) {
          sctl::Long j0;
          LocalStencil<Real>(j0, w, Np, (q.ip + (flip ? 1 - nds[j] : nds[j])) / (Real)(Mp0 << q.lp), p);
          for (sctl::Long c = 0; c < Nc; c++) {
            for (sctl::Long i = 0; i < order; i++) {
              const sctl::ConstIterator<Real> Tr = T.begin() + (c * order + i) * Np;
              Real v = 0;
              for (sctl::Integer m = 0; m < p; m++) v += w[m] * Tr[wrap(j0 + m, Np)];
              Ynds[((c * P + e) * order + i) * order + j] = v;
            }
          }
        }
      }
    }
  }

  /**
   * Returns the matrix A that maps values at the Gauss-Legendre nodes of [0,1] to the Legendre coefficients of the
   * polynomial that interpolates them: the coefficient of P_m(2x-1) is the sum over i of A[m][i] v_i.
   */
  template <class Real> sctl::Matrix<Real> LegendreCoeffMatrix(const sctl::Integer order) {
    const sctl::Vector<Real>& nds = sctl::LegQuadRule<Real>::nds(order);
    const sctl::Vector<Real>& wts = sctl::LegQuadRule<Real>::wts(order);
    sctl::Matrix<Real> A(order, order);
    for (sctl::Integer i = 0; i < order; i++) { // the quadrature of (2m+1) v(x) P_m(2x-1) on [0,1], exact for the interpolant
      const Real x = 2 * nds[i] - 1;
      Real P0 = 1, P1 = x; // P_m(x) and P_{m+1}(x)
      for (sctl::Integer m = 0; m < order; m++) {
        A[m][i] = (2 * m + 1) * wts[i] * P0;
        const Real P2 = ((2 * m + 3) * x * P1 - (m + 1) * P0) / (m + 2);
        P0 = P1;
        P1 = P2;
      }
    }
    return A;
  }

  /**
   * Set tt and tp to the size of the last two Legendre coefficients, along the toroidal (first) and along the poloidal
   * direction, of the polynomial that interpolates the values V at the ORDER x ORDER nodes of a panel: the sum of their
   * absolute values, which bounds their part of the polynomial on the panel, times scal and less their rounding error.
   */
  template <class Real, sctl::Integer ORDER> void LegendreTails(Real& tt, Real& tp, const sctl::ConstIterator<Real> V, const sctl::Matrix<Real>& A, const Real scal) {
    sctl::StaticArray<Real,ORDER*ORDER> B, C; // B = A V and C = A V A^T
    for (sctl::Integer m = 0; m < ORDER; m++) {
      for (sctl::Integer j = 0; j < ORDER; j++) {
        Real v = 0;
        for (sctl::Integer i = 0; i < ORDER; i++) v += A[m][i] * V[i * ORDER + j];
        B[m * ORDER + j] = v;
      }
    }
    for (sctl::Integer m = 0; m < ORDER; m++) {
      for (sctl::Integer n = 0; n < ORDER; n++) {
        Real v = 0;
        for (sctl::Integer j = 0; j < ORDER; j++) v += B[m * ORDER + j] * A[n][j];
        C[m * ORDER + n] = v;
      }
    }
    tt = 0;
    tp = 0;
    for (sctl::Integer k = 0; k < ORDER; k++) {
      tt += sctl::fabs<Real>(C[(ORDER - 2) * ORDER + k]) + sctl::fabs<Real>(C[(ORDER - 1) * ORDER + k]);
      tp += sctl::fabs<Real>(C[k * ORDER + ORDER - 2]) + sctl::fabs<Real>(C[k * ORDER + ORDER - 1]);
    }
    const Real round = 64 * ORDER * sctl::machine_eps<Real>(); // for values of size 1
    tt = tt * scal - round;
    tp = tp * scal - round;
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
   * Returns the smallest upsampling factor U, at least U0, from 1, 2, 3, 4, 6, 8, 12 and 16, at which the local
   * interpolation along the rows of R is within tol.
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
    for (const sctl::Long U : {1, 2, 3, 4, 6, 8, 12, 16}) {
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
   * Returns the reciprocal of the largest absolute value in V, or 1 if V is zero or empty.
   */
  template <class Real> Real InvMaxAbs(const sctl::Vector<Real>& V) {
    Real max = 0;
    for (const auto& v : V) max = std::max<Real>(max, sctl::fabs<Real>(v));
    return (max > 0 ? 1 / max : (Real)1);
  }

}

template <class Real, class SKer> template <sctl::Integer DIGITS, sctl::Integer Nv> OffSurfaceOp<Real,SKer>::OffSurfaceOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) : ker_(ker.GetSctlKernel()), comm_(comm), tol_(0), Nt_(0), Np_(0), Ntrg(0), trg0(0), trg1(0), Ut(1), Up(1), warned_panels(false), warned_interp(false), flip(false) {
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

  panels.ReInit(Mt0 * Mp0); // the coarse mesh
  for (sctl::Long a = 0; a < Mt0; a++) {
    for (sctl::Long b = 0; b < Mp0; b++) panels[a * Mp0 + b] = detail_offsurf::Panel{0, 0, a, b};
  }
  Ut = 1;
  Up = 1;
  warned_panels = false;
  warned_interp = false;
  FitPanels(F);
  BuildPanels();
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) {
  const sctl::Long N = Nt_ * Np_;
  const sctl::Long dof = F.Dim() / (KDIM0 * N);
  SCTL_ASSERT(F.Dim() == dof * KDIM0 * N);
  if (!Resolves(F)) { // the panels are only split, so the fit is unchanged if no panel could be split
    const sctl::Long P0 = panels.Dim(), Ut0 = Ut, Up0 = Up;
    FitPanels(F);
    if (panels.Dim() != P0 || Ut != Ut0 || Up != Up0) BuildPanels();
  }

  const sctl::Long Nsrc = KDIM0 * panels.Dim() * ORDER * ORDER; // values of a density at the panel nodes
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
  const sctl::Long N = Nt_ * Np_, Nf = F.Dim() / N, Nnds = ORDER * ORDER;
  SCTL_ASSERT(F.Dim() == Nf * N);
  const detail_offsurf::PanelRule<Real> rule(ORDER);

  // Y: the fields that the panels must resolve, X, its derivatives along each direction, which give the normal and the
  // area element, and F; scal: for each component, the reciprocal of the largest value of its field
  const sctl::Long Nc = 3 * COORD_DIM + Nf;
  sctl::Vector<Real> Y(Nc * N), scal(Nc);
  Real sX, st, sp, sF;
  {
    sctl::Vector<Real> dX; // dX/dt and dX/dp, for each component of X
    SurfaceOp<Real>(sctl::Comm::Self(), Nt_, Np_).Grad2D(dX, X_);
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < N; i++) {
        Y[(0 * COORD_DIM + k) * N + i] = X_[k * N + i];
        Y[(1 * COORD_DIM + k) * N + i] = dX[(k * 2 + 0) * N + i];
        Y[(2 * COORD_DIM + k) * N + i] = dX[(k * 2 + 1) * N + i];
      }
    }
    for (sctl::Long i = 0; i < Nf * N; i++) Y[3 * COORD_DIM * N + i] = F[i];
    const auto inv_max = [&Y,N](const sctl::Long c0, const sctl::Long c1) { return detail_offsurf::InvMaxAbs<Real>(sctl::Vector<Real>((c1 - c0) * N, Y.begin() + c0 * N, false)); };
    sX = inv_max(0, COORD_DIM);
    st = inv_max(COORD_DIM, 2 * COORD_DIM);
    sp = inv_max(2 * COORD_DIM, 3 * COORD_DIM);
    sF = inv_max(3 * COORD_DIM, Nc);
    for (sctl::Long c = 0; c < Nc; c++) scal[c] = (c < COORD_DIM ? sX : (c < 2 * COORD_DIM ? st : (c < 3 * COORD_DIM ? sp : sF)));
  }

  const auto max_level = [](const sctl::Long M0, const sctl::Long N_) { // at most 16 panel nodes per grid point
    sctl::Integer l = 0;
    while ((M0 << (l + 1)) * ORDER <= 16 * N_) l++;
    return l;
  };
  const sctl::Integer lt_max = max_level(Mt0, Nt_), lp_max = max_level(Mp0, Np_);
  const sctl::Vector<Real>& nds = sctl::QuadElemList<Real>::ParamNodes(ORDER);
  const sctl::Matrix<Real> A = detail_offsurf::LegendreCoeffMatrix<Real>(ORDER);
  bool resolved = true;
  for (sctl::Long Ut0 = 0, Up0 = 0; ; ) {
    { // Set Ut, Up: the error of the local interpolation, which differentiation on the finest panels amplifies, is
      // within tol/2
      sctl::Integer lt = 0, lp = 0;
      for (const auto& q : panels) {
        lt = std::max(lt, q.lt);
        lp = std::max(lp, q.lp);
      }
      const Real cX = std::max<Real>(rule.L_max * sX, rule.Ld_max * std::max<Real>((Mt0 << lt) * st, (Mp0 << lp) * sp)), cF = rule.L_max * sF;
      const auto upsample_factor = [this,&F,cX,cF](const sctl::Long U0, const sctl::Integer dir) {
        const sctl::Long UX = detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(X_, Nt_, Np_, dir), cX, tol_ / 2, STENCIL, U0);
        const sctl::Long U = (UX < 0 ? UX : detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, dir), cF, tol_ / 2, STENCIL, UX));
        if (U < 0) {
          if (!warned_interp) SCTL_WARN("The local interpolation of the surface or the density may not reach the requested accuracy.");
          warned_interp = true;
          return std::max<sctl::Long>(U0, 16);
        }
        return U;
      };
      Ut = upsample_factor(Ut, 0);
      Up = upsample_factor(Up, 1);
    }
    if (Ut == Ut0 && Up == Up0) break; // the panels were fit with this upsampling
    Ut0 = Ut;
    Up0 = Up;

    // Split the panels, along one or both directions, until the last two Legendre coefficients of each field along
    // each direction are within tol, or the panels are the finest allowed
    sctl::Vector<Real> Yf;
    UpsampleGrid(Yf, Y);
    sctl::Vector<detail_offsurf::Panel> todo = panels, done;
    resolved = true;
    while (todo.Dim()) {
      const sctl::Long P = todo.Dim();
      sctl::Vector<Real> V;
      detail_offsurf::PanelNodeValues<Real>(V, Yf, Ut * Nt_, Up * Np_, todo, Mt0, Mp0, nds, false, STENCIL);
      sctl::Vector<sctl::Integer> split(P); // bit 0: split along the toroidal direction, bit 1: along the poloidal
      bool resolved_ = true;
      #pragma omp parallel for schedule(static) reduction(&&:resolved_)
      for (sctl::Long e = 0; e < P; e++) {
        Real tt = 0, tp = 0;
        for (sctl::Long c = 0; c < Nc; c++) {
          Real tt_, tp_;
          detail_offsurf::LegendreTails<Real,ORDER>(tt_, tp_, V.begin() + (c * P + e) * Nnds, A, scal[c]);
          tt = std::max<Real>(tt, tt_);
          tp = std::max<Real>(tp, tp_);
        }
        const detail_offsurf::Panel& q = todo[e];
        split[e] = (tt > tol_ && q.lt < lt_max ? 1 : 0) | (tp > tol_ && q.lp < lp_max ? 2 : 0);
        if ((tt > tol_ && q.lt >= lt_max) || (tp > tol_ && q.lp >= lp_max)) resolved_ = false;
      }
      resolved = resolved && resolved_;

      sctl::Vector<detail_offsurf::Panel> next;
      for (sctl::Long e = 0; e < P; e++) {
        const detail_offsurf::Panel& q = todo[e];
        if (!split[e]) {
          done.PushBack(q);
          continue;
        }
        const sctl::Integer nt = (split[e] & 1 ? 2 : 1), np = (split[e] & 2 ? 2 : 1);
        for (sctl::Integer a = 0; a < nt; a++) {
          for (sctl::Integer b = 0; b < np; b++) next.PushBack(detail_offsurf::Panel{q.lt + nt - 1, q.lp + np - 1, q.it * nt + a, q.ip * np + b});
        }
      }
      todo = next;
    }
    panels = done;
  }
  if (!resolved) {
    if (!warned_panels) SCTL_WARN("The surface or the density is not resolved on its grid; the panels may not reach the requested accuracy.");
    warned_panels = true;
  }
}

template <class Real, class SKer> bool OffSurfaceOp<Real,SKer>::Resolves(const sctl::Vector<Real>& F) const {
  const sctl::Long N = Nt_ * Np_, Nf = F.Dim() / N, P = panels.Dim(), Nnds = ORDER * ORDER;
  const detail_offsurf::PanelRule<Real> rule(ORDER);
  const Real sF = detail_offsurf::InvMaxAbs<Real>(F), cF = rule.L_max * sF;
  if (detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, 0), cF, tol_ / 2, STENCIL, Ut) != Ut ||
      detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, 1), cF, tol_ / 2, STENCIL, Up) != Up) return false;

  sctl::Vector<Real> Ff, V;
  UpsampleGrid(Ff, F);
  detail_offsurf::PanelNodeValues<Real>(V, Ff, Ut * Nt_, Up * Np_, panels, Mt0, Mp0, sctl::QuadElemList<Real>::ParamNodes(ORDER), false, STENCIL);
  const sctl::Matrix<Real> A = detail_offsurf::LegendreCoeffMatrix<Real>(ORDER);
  bool resolved = true;
  #pragma omp parallel for schedule(static) reduction(&&:resolved)
  for (sctl::Long e = 0; e < P; e++) {
    for (sctl::Long c = 0; c < Nf; c++) {
      Real tt, tp;
      detail_offsurf::LegendreTails<Real,ORDER>(tt, tp, V.begin() + (c * P + e) * Nnds, A, sF);
      if (tt > tol_ || tp > tol_) resolved = false;
    }
  }
  return resolved;
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::BuildPanels() {
  ElemList elem_lst;
  const auto build = [this,&elem_lst](const bool flip_) {
    flip = flip_;
    sctl::Vector<Real> coord;
    InterpToNodes(coord, X_, COORD_DIM);
    elem_lst.Init(ORDER, coord);
  };

  build(false);
  { // Rebuild with the poloidal nodes of each panel reversed if the normal points inward at the node of largest x
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

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::UpsampleGrid(sctl::Vector<Real>& Yf, const sctl::Vector<Real>& Y) const {
  if (Ut == 1 && Up == 1) Yf.ReInit(Y.Dim(), (sctl::Iterator<Real>)Y.begin(), false);
  else SurfaceOp<Real>::Resample(Yf, Ut * Nt_, Up * Np_, Y, Nt_, Np_);
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::InterpToNodes(sctl::Vector<Real>& Ynds, const sctl::Vector<Real>& Y, const sctl::Integer kdim) const {
  const sctl::Long N = Nt_ * Np_, P = panels.Dim(), Nnds = ORDER * ORDER;
  const sctl::Long Nc = Y.Dim() / N; // number of components over all fields
  SCTL_ASSERT(Y.Dim() == Nc * N);
  SCTL_ASSERT(Nc % kdim == 0);
  if (Ynds.Dim() != Nc * P * Nnds) Ynds.ReInit(Nc * P * Nnds);

  sctl::Vector<Real> Yf, V; // V: the values at the nodes, component by component
  UpsampleGrid(Yf, Y);
  detail_offsurf::PanelNodeValues<Real>(V, Yf, Ut * Nt_, Up * Np_, panels, Mt0, Mp0, sctl::QuadElemList<Real>::ParamNodes(ORDER), flip, STENCIL);
  for (sctl::Long c = 0; c < Nc; c++) {
    const sctl::Long d = c / kdim, k = c % kdim; // field and component
    #pragma omp parallel for schedule(static)
    for (sctl::Long e = 0; e < P; e++) {
      for (sctl::Long n = 0; n < Nnds; n++) Ynds[((d * P + e) * Nnds + n) * kdim + k] = V[(c * P + e) * Nnds + n];
    }
  }
}

}
