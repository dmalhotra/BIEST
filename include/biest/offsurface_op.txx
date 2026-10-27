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
   * Order the panels by their toroidal interval, whose panels share the interpolation along the toroidal direction:
   * the panels of interval g are idx[grp[g]], ..., idx[grp[g+1]-1].
   */
  inline void GroupByInterval(std::vector<sctl::Long>& idx, std::vector<sctl::Long>& grp, const sctl::Vector<Panel>& panels) {
    const sctl::Long P = panels.Dim();
    idx.resize(P);
    std::iota(idx.begin(), idx.end(), 0);
    const auto interval = [&panels](const sctl::Long e) { return std::make_pair(panels[e].lt, panels[e].it); };
    std::sort(idx.begin(), idx.end(), [&interval](const sctl::Long a, const sctl::Long b) { return interval(a) < interval(b); });
    grp.clear();
    for (sctl::Long n = 0; n < P; n++) {
      if (!n || interval(idx[n]) != interval(idx[n - 1])) grp.push_back(n);
    }
    grp.push_back(P);
  }

  /**
   * Interpolate fields on a periodic Nt x Np grid along the toroidal direction to the order toroidal nodes of the
   * interval it at level lt, by p-point local interpolation, on every poloidal grid line.
   *
   * @param[out] T the values of component c at toroidal node i and poloidal grid point j, at
   * T[(c * order + i) * (Np + 2 * p) + p + j]; each row is extended periodically by p values on each side, so that the
   * poloidal stencils need no wrap-around.
   *
   * @param[in] Y the fields on the grid in SoA order.
   */
  template <class Real> void ToroidalRows(const sctl::Iterator<Real> T, const sctl::Vector<Real>& Y, const sctl::Long Nt, const sctl::Long Np, const sctl::Integer lt, const sctl::Long it, const sctl::Long Mt0, const sctl::Vector<Real>& nds, const sctl::Integer p) {
    const sctl::Long N = Nt * Np, Nc = Y.Dim() / N, order = nds.Dim(), Ld = Np + 2 * p;
    SCTL_ASSERT(Y.Dim() == Nc * N);
    SCTL_ASSERT(p <= 64);
    const auto wrap = [](const sctl::Long i, const sctl::Long n) { return ((i % n) + n) % n; };
    sctl::StaticArray<Real,64> w;
    for (sctl::Long i = 0; i < order; i++) {
      sctl::Long i0;
      LocalStencil<Real>(i0, w, Nt, (it + nds[i]) / (Real)(Mt0 << lt), p);
      for (sctl::Long c = 0; c < Nc; c++) {
        const sctl::Iterator<Real> Tr = T + (c * order + i) * Ld + p;
        for (sctl::Long j = 0; j < Np; j++) Tr[j] = 0;
        for (sctl::Integer m = 0; m < p; m++) {
          const sctl::ConstIterator<Real> Yr = Y.begin() + (c * Nt + wrap(i0 + m, Nt)) * Np;
          const Real wm = w[m];
          for (sctl::Long j = 0; j < Np; j++) Tr[j] += wm * Yr[j];
        }
        for (sctl::Long j = -p; j < 0; j++) Tr[j] = Tr[wrap(j, Np)];
        for (sctl::Long j = Np; j < Np + p; j++) Tr[j] = Tr[wrap(j, Np)];
      }
    }
  }

  /**
   * Interpolate the rows T from ToroidalRows, of Nc components on a poloidal grid of Np points, along the poloidal
   * direction to the order x order nodes of the panel whose poloidal interval is ip at level lp, by p-point local
   * interpolation.
   *
   * @param[out] V the value of component c at node (i,j), with i along the toroidal and j along the poloidal direction,
   * at V[c * ldv + i * order + j].
   *
   * @param[in] flip whether the poloidal nodes are in decreasing order.
   */
  template <class Real> void PoloidalNodes(const sctl::Iterator<Real> V, const sctl::Long ldv, const sctl::ConstIterator<Real> T, const sctl::Long Nc, const sctl::Long Np, const sctl::Integer lp, const sctl::Long ip, const sctl::Long Mp0, const sctl::Vector<Real>& nds, const bool flip, const sctl::Integer p) {
    const sctl::Long order = nds.Dim(), Ld = Np + 2 * p;
    sctl::StaticArray<Real,64> w;
    for (sctl::Long j = 0; j < order; j++) {
      sctl::Long j0; // in [-p/2+1, Np-p/2], so that the stencil lies in the extended rows
      LocalStencil<Real>(j0, w, Np, (ip + nds[flip ? order - 1 - j : j]) / (Real)(Mp0 << lp), p);
      for (sctl::Long c = 0; c < Nc; c++) {
        for (sctl::Long i = 0; i < order; i++) {
          const sctl::ConstIterator<Real> Tr = T + (c * order + i) * Ld + p + j0;
          Real v = 0;
          for (sctl::Integer m = 0; m < p; m++) v += w[m] * Tr[m];
          V[c * ldv + i * order + j] = v;
        }
      }
    }
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
    const sctl::Long N = Nt * Np, Nc = Y.Dim() / N, P = panels.Dim(), order = nds.Dim(), Nnds = order * order;
    SCTL_ASSERT(Y.Dim() == Nc * N);
    if (Ynds.Dim() != Nc * P * Nnds) Ynds.ReInit(Nc * P * Nnds);

    std::vector<sctl::Long> idx, grp;
    GroupByInterval(idx, grp, panels);
    #pragma omp parallel for schedule(dynamic)
    for (sctl::Long g = 0; g < (sctl::Long)grp.size() - 1; g++) {
      const Panel& q0 = panels[idx[grp[g]]];
      sctl::ScratchBuf<Real> T(Nc * order * (Np + 2 * p)); // the fields at the toroidal nodes of the interval
      ToroidalRows<Real>(T.begin(), Y, Nt, Np, q0.lt, q0.it, Mt0, nds, p);
      for (sctl::Long n = grp[g]; n < grp[g + 1]; n++) {
        const sctl::Long e = idx[n];
        PoloidalNodes<Real>(Ynds.begin() + e * Nnds, P * Nnds, T.begin(), Nc, Np, panels[e].lp, panels[e].ip, Mp0, nds, flip, p);
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
   * Returns the area of panel e of P, with the grid parameters in [0,1), by the Gauss-Legendre rule with weights wts on
   * its ORDER x ORDER nodes.
   *
   * @param[in] dX the derivatives of the surface coordinates at the nodes of the P panels: the 3 components of dX/dt,
   * then those of dX/dp, each for all panels.
   */
  template <class Real, sctl::Integer ORDER> Real PanelArea(const sctl::ConstIterator<Real> dX, const sctl::Long P, const sctl::Long e, const Panel& q, const sctl::Long Mt0, const sctl::Long Mp0, const sctl::Vector<Real>& wts) {
    Real area = 0;
    for (sctl::Integer i = 0; i < ORDER; i++) {
      for (sctl::Integer j = 0; j < ORDER; j++) {
        const sctl::Long n = i * ORDER + j;
        Real xt[3], xp[3];
        for (sctl::Integer k = 0; k < 3; k++) {
          xt[k] = dX[(k * P + e) * ORDER * ORDER + n];
          xp[k] = dX[((3 + k) * P + e) * ORDER * ORDER + n];
        }
        const Real n0 = xt[1] * xp[2] - xt[2] * xp[1], n1 = xt[2] * xp[0] - xt[0] * xp[2], n2 = xt[0] * xp[1] - xt[1] * xp[0];
        area += wts[i] * wts[j] * sctl::sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
      }
    }
    return area / ((Real)(Mt0 << q.lt) * (Real)(Mp0 << q.lp));
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

    // The transforms are small: on one thread, as many threads would cost more to set up and synchronize than they save
    sctl::Vector<Real> coeff; // Fourier coefficients of the rows, divided by N
    {
      sctl::FFT<Real> fft;
      sctl::StaticArray<sctl::Long,1> dim{N};
      fft.Setup(sctl::FFT_Type::R2C, Nr, sctl::Vector<sctl::Long>(1, dim, false), 1);
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
        fft.Setup(sctl::FFT_Type::C2R, Ns, sctl::Vector<sctl::Long>(1, dim, false), 1);
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
  (void)F;
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
  sctl::Vector<Real> coord;
  FitPanels(coord);
  BuildPanels(coord);
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) {
  const sctl::Long N = Nt_ * Np_;
  const sctl::Long dof = F.Dim() / (KDIM0 * N);
  SCTL_ASSERT(F.Dim() == dof * KDIM0 * N);

  // The FFT upsampling of F before its local interpolation to the nodes: at least that of the surface, and enough for
  // the error of the local interpolation to be within tol/2
  const Real cF = detail_offsurf::PanelRule<Real>(ORDER).L_max * detail_offsurf::InvMaxAbs<Real>(F);
  const auto upsample_factor = [this,&F,cF](const sctl::Long U0, const sctl::Integer dir) {
    const sctl::Long U = detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(F, Nt_, Np_, dir), cF, tol_ / 2, STENCIL, U0);
    if (U < 0) {
      if (!warned_interp) SCTL_WARN("The local interpolation of the density may not reach the requested accuracy.");
      warned_interp = true;
      return std::max<sctl::Long>(U0, 16);
    }
    return U;
  };
  const sctl::Long UFt = upsample_factor(Ut, 0), UFp = upsample_factor(Up, 1);

  const sctl::Long Nsrc = KDIM0 * panels.Dim() * ORDER * ORDER; // values of a density at the panel nodes
  const sctl::Long Nloc = trg1 - trg0;
  sctl::ScratchBuf<Real> Fnds_buf(dof * Nsrc), Utrg_buf(KDIM1 * Nloc), Uloc_buf(dof * KDIM1 * Ntrg), Uglb_buf(dof * KDIM1 * Ntrg);
  // Utrg: the potential at the targets of this process, AoS
  sctl::Vector<Real> Fnds(Fnds_buf), Utrg(Utrg_buf), Uloc(Uloc_buf), Uglb(Uglb_buf);
  InterpToNodes(Fnds, F, KDIM0, UFt, UFp);
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

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::FitPanels(sctl::Vector<Real>& coord) {
  const sctl::Long N = Nt_ * Np_, Nnds = ORDER * ORDER;
  const detail_offsurf::PanelRule<Real> rule(ORDER);

  // Y: the fields that the panels must resolve, X and its derivatives along each direction, which give the normal and
  // the area element; scal: for each component, the reciprocal of the largest value of its field; sA: the reciprocal
  // of the surface area, by the trapezoidal rule on the grid with the parameters in [0,1)
  const sctl::Long Nc = 3 * COORD_DIM;
  sctl::ScratchBuf<Real> Y_buf(Nc * N);
  sctl::Vector<Real> Y(Y_buf), scal(Nc);
  Real sX, st, sp, sA;
  {
    sctl::ScratchBuf<Real> dX_buf(2 * COORD_DIM * N);
    sctl::Vector<Real> dX(dX_buf); // dX/dt and dX/dp, for each component of X
    SurfaceOp<Real>(sctl::Comm::Self(), Nt_, Np_).Grad2D(dX, X_);
    for (sctl::Integer k = 0; k < COORD_DIM; k++) {
      for (sctl::Long i = 0; i < N; i++) {
        Y[(0 * COORD_DIM + k) * N + i] = X_[k * N + i];
        Y[(1 * COORD_DIM + k) * N + i] = dX[(k * 2 + 0) * N + i];
        Y[(2 * COORD_DIM + k) * N + i] = dX[(k * 2 + 1) * N + i];
      }
    }
    const auto inv_max = [&Y,N](const sctl::Long c0, const sctl::Long c1) { return detail_offsurf::InvMaxAbs<Real>(sctl::Vector<Real>((c1 - c0) * N, Y.begin() + c0 * N, false)); };
    sX = inv_max(0, COORD_DIM);
    st = inv_max(COORD_DIM, 2 * COORD_DIM);
    sp = inv_max(2 * COORD_DIM, 3 * COORD_DIM);
    for (sctl::Long c = 0; c < Nc; c++) scal[c] = (c < COORD_DIM ? sX : (c < 2 * COORD_DIM ? st : sp));
    Real area = 0;
    for (sctl::Long i = 0; i < N; i++) {
      Real xt[COORD_DIM], xp[COORD_DIM];
      for (sctl::Integer k = 0; k < COORD_DIM; k++) {
        xt[k] = dX[(k * 2 + 0) * N + i];
        xp[k] = dX[(k * 2 + 1) * N + i];
      }
      const Real n0 = xt[1] * xp[2] - xt[2] * xp[1], n1 = xt[2] * xp[0] - xt[0] * xp[2], n2 = xt[0] * xp[1] - xt[1] * xp[0];
      area += sctl::sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
    }
    sA = N / area;
  }

  const auto max_level = [](const sctl::Long M0, const sctl::Long N_) { // at most 32 panel nodes per grid point
    sctl::Integer l = 0;
    while ((M0 << (l + 1)) * ORDER <= 32 * N_) l++;
    return l;
  };
  const sctl::Integer lt_max = max_level(Mt0, Nt_), lp_max = max_level(Mp0, Np_);
  const sctl::Vector<Real>& nds = sctl::QuadElemList<Real>::ParamNodes(ORDER);
  const sctl::Vector<Real>& wts = sctl::LegQuadRule<Real>::wts(ORDER);
  const sctl::Matrix<Real> A = detail_offsurf::LegendreCoeffMatrix<Real>(ORDER);
  constexpr sctl::Long Nd = 2 * COORD_DIM; // dX/dt and dX/dp
  std::vector<sctl::Vector<Real>> Xlev; // in the last fit, X at the nodes of the panels of each level, node by node
  std::vector<std::vector<sctl::Long>> kept; // in the last fit, the panels of each level that are not split
  Real xmax = 0, nmax = 0; // the largest x at the nodes of the kept panels, and there the x component of X_t x X_p
  bool resolved = true;
  for (sctl::Long Ut0 = 0, Up0 = 0; ; ) {
    { // Set Ut, Up: the error of the local interpolation, which differentiation on the finest panels amplifies, is
      // within tol/2
      sctl::Integer lt = 0, lp = 0;
      for (const auto& q : panels) {
        lt = std::max(lt, q.lt);
        lp = std::max(lp, q.lp);
      }
      const Real cX = std::max<Real>(rule.L_max * sX, rule.Ld_max * std::max<Real>((Mt0 << lt) * st, (Mp0 << lp) * sp));
      const auto upsample_factor = [this,cX](const sctl::Long U0, const sctl::Integer dir) {
        const sctl::Long U = detail_offsurf::LocalUpsample<Real>(detail_offsurf::GridLines<Real>(X_, Nt_, Np_, dir), cX, tol_ / 2, STENCIL, U0);
        if (U < 0) {
          if (!warned_interp) SCTL_WARN("The local interpolation of the surface may not reach the requested accuracy.");
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

    // Split the panels, along one or both directions, until along each direction the last two Legendre coefficients of
    // each field are within tol, and the area of the panel by its quadrature differs from the sum of the areas of its
    // two halves by at most tol times the surface area, or the panels are the finest allowed. The test of the area
    // catches an area element that the quadrature does not resolve although X and its derivatives are resolved: the
    // area element is the square root of a polynomial, and near a sharp concave fold of the surface its branch points
    // are close to the panel. The values at the nodes of the panels and of their halves are computed and tested
    // interval by interval of the toroidal direction, in buffers of each thread; X at the nodes is kept for coord.
    const sctl::Long Ntf = Ut * Nt_, Npf = Up * Np_, Nfine = Ntf * Npf, Ld = Npf + 2 * STENCIL;
    sctl::ScratchBuf<Real> Yf_buf(Ut * Up > 1 ? Nc * Nfine : 0);
    sctl::Vector<Real> Yf(Yf_buf, false);
    UpsampleGrid(Yf, Y, Ut, Up);
    const sctl::Vector<Real> dXf(Nd * Nfine, Yf.begin() + COORD_DIM * Nfine, false); // dX/dt and dX/dp in Yf
    sctl::Vector<detail_offsurf::Panel> todo = panels, done;
    Xlev.clear();
    kept.clear();
    resolved = true;
    while (todo.Dim()) {
      const sctl::Long P = todo.Dim();
      std::vector<sctl::Long> idx, grp;
      detail_offsurf::GroupByInterval(idx, grp, todo);
      sctl::Vector<sctl::Integer> split(P); // bit 0: split along the toroidal direction, bit 1: along the poloidal
      Xlev.emplace_back(COORD_DIM * Nnds * P);
      sctl::Vector<Real>& Xl = Xlev.back();
      sctl::Vector<Real> xm(P), nm(P); // of each panel, the largest x at its nodes, and there the x component of X_t x X_p
      bool resolved_ = true;
      #pragma omp parallel for schedule(dynamic) reduction(&&:resolved_)
      for (sctl::Long g = 0; g < (sctl::Long)grp.size() - 1; g++) {
        // T: Yf at the toroidal nodes of the interval, and Th: dXf at those of its two halves, on every poloidal grid
        // line; V: Yf at the nodes of a panel, and Vh: dXf at the nodes of its halves
        const detail_offsurf::Panel& q0 = todo[idx[grp[g]]];
        sctl::ScratchBuf<Real> T(Nc * ORDER * Ld), Th(2 * Nd * ORDER * Ld), V(Nc * Nnds), Vh(4 * Nd * Nnds);
        detail_offsurf::ToroidalRows<Real>(T.begin(), Yf, Ntf, Npf, q0.lt, q0.it, Mt0, nds, STENCIL);
        for (sctl::Integer h = 0; h < 2; h++) detail_offsurf::ToroidalRows<Real>(Th.begin() + h * Nd * ORDER * Ld, dXf, Ntf, Npf, q0.lt + 1, 2 * q0.it + h, Mt0, nds, STENCIL);
        for (sctl::Long n = grp[g]; n < grp[g + 1]; n++) {
          const sctl::Long e = idx[n];
          const detail_offsurf::Panel& q = todo[e];
          // the halves: 0 and 1 along the toroidal direction, 2 and 3 along the poloidal
          const detail_offsurf::Panel halves[4] = {{q.lt + 1, q.lp, 2 * q.it + 0, q.ip}, {q.lt + 1, q.lp, 2 * q.it + 1, q.ip}, {q.lt, q.lp + 1, q.it, 2 * q.ip + 0}, {q.lt, q.lp + 1, q.it, 2 * q.ip + 1}};
          detail_offsurf::PoloidalNodes<Real>(V.begin(), Nnds, T.begin(), Nc, Npf, q.lp, q.ip, Mp0, nds, false, STENCIL);
          for (sctl::Integer h = 0; h < 4; h++) {
            const sctl::ConstIterator<Real> Tq = (h < 2 ? Th.begin() + h * Nd * ORDER * Ld : T.begin() + COORD_DIM * ORDER * Ld);
            detail_offsurf::PoloidalNodes<Real>(Vh.begin() + h * Nd * Nnds, Nnds, Tq, Nd, Npf, halves[h].lp, halves[h].ip, Mp0, nds, false, STENCIL);
          }

          Real tt = 0, tp = 0;
          for (sctl::Long c = 0; c < Nc; c++) {
            Real tt_, tp_;
            detail_offsurf::LegendreTails<Real,ORDER>(tt_, tp_, V.begin() + c * Nnds, A, scal[c]);
            tt = std::max<Real>(tt, tt_);
            tp = std::max<Real>(tp, tp_);
          }
          { // the area of the panel against the sum of the areas of its halves along each direction
            const auto area = [&](const sctl::ConstIterator<Real> dX, const detail_offsurf::Panel& r) { return detail_offsurf::PanelArea<Real,ORDER>(dX, 1, 0, r, Mt0, Mp0, wts); };
            const auto half = [&](const sctl::Integer h) { return area(Vh.begin() + h * Nd * Nnds, halves[h]); };
            const Real A0 = area(V.begin() + COORD_DIM * Nnds, q);
            const Real round = 64 * ORDER * sctl::machine_eps<Real>(); // for areas of size 1
            tt = std::max<Real>(tt, sA * sctl::fabs<Real>(A0 - half(0) - half(1)) - round);
            tp = std::max<Real>(tp, sA * sctl::fabs<Real>(A0 - half(2) - half(3)) - round);
          }
          split[e] = (tt > tol_ && q.lt < lt_max ? 1 : 0) | (tp > tol_ && q.lp < lp_max ? 2 : 0);
          if ((tt > tol_ && q.lt >= lt_max) || (tp > tol_ && q.lp >= lp_max)) resolved_ = false;

          sctl::Long m = 0; // the first node of largest x
          for (sctl::Long i = 0; i < Nnds; i++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) Xl[(e * Nnds + i) * COORD_DIM + k] = V[k * Nnds + i];
            if (V[i] > V[m]) m = i;
          }
          const auto Vm = [&V,Nnds,m](const sctl::Long c) { return V[c * Nnds + m]; };
          xm[e] = Vm(0);
          nm[e] = Vm(COORD_DIM + 1) * Vm(2 * COORD_DIM + 2) - Vm(COORD_DIM + 2) * Vm(2 * COORD_DIM + 1);
        }
      }
      resolved = resolved && resolved_;

      kept.emplace_back();
      sctl::Vector<detail_offsurf::Panel> next;
      for (sctl::Long e = 0; e < P; e++) {
        const detail_offsurf::Panel& q = todo[e];
        if (!split[e]) {
          done.PushBack(q);
          kept.back().push_back(e);
          if (done.Dim() == 1 || xm[e] > xmax) {
            xmax = xm[e];
            nmax = nm[e];
          }
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
    if (!warned_panels) SCTL_WARN("The surface is not resolved on its grid; the panels may not reach the requested accuracy.");
    warned_panels = true;
  }

  // coord: X at the nodes of the kept panels, in the order of panels, with the poloidal nodes of each panel reversed if
  // the normal X_t x X_p points inward at the node of largest x
  flip = (nmax < 0);
  coord.ReInit(COORD_DIM * Nnds * panels.Dim());
  sctl::Long offset = 0;
  for (sctl::Long l = 0; l < (sctl::Long)kept.size(); l++) {
    const std::vector<sctl::Long>& K = kept[l];
    const sctl::Vector<Real>& Xl = Xlev[l];
    #pragma omp parallel for schedule(static)
    for (sctl::Long a = 0; a < (sctl::Long)K.size(); a++) {
      const sctl::ConstIterator<Real> src = Xl.begin() + K[a] * COORD_DIM * Nnds;
      const sctl::Iterator<Real> dst = coord.begin() + (offset + a) * COORD_DIM * Nnds;
      for (sctl::Long i = 0; i < ORDER; i++) {
        for (sctl::Long j = 0; j < ORDER; j++) {
          const sctl::Long j_ = (flip ? ORDER - 1 - j : j);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) dst[(i * ORDER + j) * COORD_DIM + k] = src[(i * ORDER + j_) * COORD_DIM + k];
        }
      }
    }
    offset += (sctl::Long)K.size();
  }
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::BuildPanels(const sctl::Vector<Real>& coord) {
  ElemList elem_lst;
  elem_lst.Init(ORDER, coord);

  op.reset(new sctl::BoundaryIntegralOp<Real,SKer>(ker_, false, sctl::Comm::Self()));
  op->SetAccuracy(tol_);
  op->AddElemList(elem_lst);
  op->SetTargetCoord(Xt_);
  op->Setup();
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::UpsampleGrid(sctl::Vector<Real>& Yf, const sctl::Vector<Real>& Y, const sctl::Long ut, const sctl::Long up) const {
  if (ut == 1 && up == 1) Yf.ReInit(Y.Dim(), (sctl::Iterator<Real>)Y.begin(), false);
  else SurfaceOp<Real>::Resample(Yf, ut * Nt_, up * Np_, Y, Nt_, Np_);
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::InterpToNodes(sctl::Vector<Real>& Ynds, const sctl::Vector<Real>& Y, const sctl::Integer kdim, const sctl::Long ut, const sctl::Long up) const {
  const sctl::Long N = Nt_ * Np_, P = panels.Dim(), Nnds = ORDER * ORDER;
  const sctl::Long Nc = Y.Dim() / N; // number of components over all fields
  SCTL_ASSERT(Y.Dim() == Nc * N);
  SCTL_ASSERT(Nc % kdim == 0);
  if (Ynds.Dim() != Nc * P * Nnds) Ynds.ReInit(Nc * P * Nnds);

  sctl::ScratchBuf<Real> Yf_buf(ut * up > 1 ? Nc * N * ut * up : 0), V_buf(Nc * P * Nnds);
  sctl::Vector<Real> Yf(Yf_buf, false), V(V_buf); // V: the values at the nodes, component by component
  UpsampleGrid(Yf, Y, ut, up);
  detail_offsurf::PanelNodeValues<Real>(V, Yf, ut * Nt_, up * Np_, panels, Mt0, Mp0, sctl::QuadElemList<Real>::ParamNodes(ORDER), flip, STENCIL);
  for (sctl::Long c = 0; c < Nc; c++) {
    const sctl::Long d = c / kdim, k = c % kdim; // field and component
    #pragma omp parallel for schedule(static)
    for (sctl::Long e = 0; e < P; e++) {
      for (sctl::Long n = 0; n < Nnds; n++) Ynds[((d * P + e) * Nnds + n) * kdim + k] = V[(c * P + e) * Nnds + n];
    }
  }
}

}
