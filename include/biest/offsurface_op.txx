// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

namespace biest {

namespace detail_offsurf {

  // The loops of the evaluation run through sctl::omp_par::parallel_for, without a parallel region when it would have
  // one thread: libgomp allocates and frees the team of such a region each time.

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
   * the panels of interval g are idx[grp[g]], ..., idx[grp[g+1]-1]. Returns the number of intervals G; idx gets the P
   * indices of the panels and grp the G + 1 offsets, at most P + 1.
   */
  template <class Iter> sctl::Long GroupByInterval(const Iter idx, const Iter grp, const sctl::Vector<Panel>& panels) {
    const sctl::Long P = panels.Dim();
    std::iota(idx, idx + P, 0);
    const auto interval = [&panels](const sctl::Long e) { return std::make_pair(panels[e].lt, panels[e].it); };
    std::sort(idx, idx + P, [&interval](const sctl::Long a, const sctl::Long b) { return interval(a) < interval(b); });
    sctl::Long G = 0;
    for (sctl::Long n = 0; n < P; n++) {
      if (!n || interval(idx[n]) != interval(idx[n - 1])) grp[G++] = n;
    }
    grp[G] = P;
    return G;
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
   * Returns the smallest upsampling factor U, at least U0, from 1, 2, 3, 4, 6, 8, 12 and 16, at which the local
   * interpolation along the lines of Y in the direction dir, toroidal (dir = 0) or poloidal (dir = 1), is within tol;
   * or -1 if there is none.
   *
   * @param[in] Y fields on an Nt x Np grid in SoA order.
   *
   * @param[in] scal the scale of the error.
   */
  template <class Real> sctl::Long LocalUpsampleLines(const sctl::Vector<Real>& Y, const sctl::Long Nt, const sctl::Long Np, const sctl::Integer dir, const Real scal, const Real tol, const sctl::Integer p, const sctl::Long U0) {
    constexpr sctl::Long Nsample = 16; // number of rows on which the error is measured
    const sctl::Long Nc = Y.Dim() / (Nt * Np), Nr = (dir ? Nc * Nt : Nc * Np), N = (dir ? Np : Nt), Nk = N / 2 + 1;
    if (!Nr) return U0;

    // The errors of the p-point local interpolation of exp(i phi x), the largest over x in [0,1], for phi = 2 pi k / n
    // and k = 0, ..., Nk-1: computed on the first use and kept until the program ends. The memory manager of SCTL is
    // constructed before the cache, so that it is destroyed after it.
    const auto mode_errors = [p,Nk](const sctl::Long n) -> const sctl::Vector<Real>& {
      const auto mode_error = [p](const Real phi) {
        sctl::StaticArray<Real,64> wts, c, s; // c, s: exp(i phi x) at the points
        SCTL_ASSERT(p <= 64);
        for (sctl::Integer j = 0; j < p; j++) {
          c[j] = sctl::cos<Real>(phi * (j - p / 2 + 1));
          s[j] = sctl::sin<Real>(phi * (j - p / 2 + 1));
        }
        Real err = 0;
        for (sctl::Integer q = 0; q <= 8; q++) { // x in [0,1/2]; the error is symmetric about 1/2
          const Real x = q / (Real)16;
          LocalWeights<Real>(wts, p, x);
          Real re = -sctl::cos<Real>(phi * x), im = -sctl::sin<Real>(phi * x);
          for (sctl::Integer j = 0; j < p; j++) {
            re += wts[j] * c[j];
            im += wts[j] * s[j];
          }
          err = std::max<Real>(err, sctl::sqrt<Real>(re * re + im * im));
        }
        return err;
      };
      using Key = std::tuple<sctl::Integer, sctl::Long, sctl::Long>;
      (void)sctl::MemoryManager::glbMemMgr();
      static std::map<Key, std::unique_ptr<sctl::Vector<Real>>> cache;
      const Key key(p, n, Nk);
      const sctl::Vector<Real>* err = nullptr;
      #pragma omp critical(BIEST_ModeErrors)
      {
        const auto it = cache.find(key);
        if (it != cache.end()) err = it->second.get();
      }
      if (err) return *err;
      std::unique_ptr<sctl::Vector<Real>> err_(new sctl::Vector<Real>(Nk));
      sctl::Vector<Real>& e = *err_;
      sctl::omp_par::parallel_for(Nk, [&e,&mode_error,n](const sctl::Long k) { e[k] = mode_error(2 * sctl::const_pi<Real>() * k / n); });
      #pragma omp critical(BIEST_ModeErrors)
      {
        auto& entry = cache[key];
        if (!entry) entry = std::move(err_);
        err = entry.get();
      }
      return *err;
    };

    // R: the lines, one row each, of component c and poloidal index j (dir = 0) or toroidal index i (dir = 1)
    sctl::ScratchBuf<Real> R(Nr * N);
    if (dir) {
      sctl::omp_par::memcpy(R.begin(), Y.begin(), Y.Dim());
    } else {
      sctl::omp_par::parallel_for(Nr, [&R,&Y,Nt,Np](const sctl::Long r) {
        const sctl::Long c = r / Np, j = r % Np;
        for (sctl::Long i = 0; i < Nt; i++) R[r * Nt + i] = Y[(c * Nt + i) * Np + j];
      });
    }

    // the Fourier coefficients of the rows, divided by N; their sizes, those of the modes of the real rows; and the sum
    // of these over each row
    sctl::ScratchBuf<Real> coeff(Nr * Nk * 2), mag(Nr * Nk), mag_sum(Nr), est(Nr);
    sctl::ScratchBuf<sctl::Long> order(Nr), rows(std::min(Nsample, Nr));
    {
      const sctl::Iterator<Real> C = coeff.begin(), M = mag.begin(), Ms = mag_sum.begin();
      const sctl::ConstIterator<Real> R_ = R.begin();
      detail_surfop::RowsR2C<Real>(Nr, N, Nk, [R_,N](const sctl::Long r) { return R_ + r * N; }, [C,M,Ms,Nk,N](const sctl::Long r, const sctl::ConstIterator<Real> c) {
        Real sum = 0;
        for (sctl::Long k = 0; k < Nk; k++) {
          const Real re = c[k * 2 + 0] / N, im = c[k * 2 + 1] / N;
          C[(r * Nk + k) * 2 + 0] = re;
          C[(r * Nk + k) * 2 + 1] = im;
          M[r * Nk + k] = sctl::sqrt<Real>(re * re + im * im) * (k == 0 || 2 * k == N ? 1 : 2);
          sum += M[r * Nk + k];
        }
        Ms[r] = sum;
      });
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
    const sctl::Long Ns = std::min(Nsample, Nr);
    for (const sctl::Long U : {1, 2, 3, 4, 6, 8, 12, 16}) {
      if (U < U0) continue;
      { // Set rows: the rows with the largest estimated error, from their mode sizes
        const sctl::Vector<Real>& err_k = mode_errors(U * N);
        sctl::omp_par::parallel_for(Nr, [&est,&mag,&mag_sum,&err_k,Nk,round,scal](const sctl::Long r) {
          Real e = 0;
          for (sctl::Long k = 0; k < Nk; k++) e += mag[r * Nk + k] * err_k[k];
          est[r] = (e - round * mag_sum[r]) * scal;
        });
        std::iota(order.begin(), order.begin() + Nr, 0);
        std::partial_sort(order.begin(), order.begin() + Ns, order.begin() + Nr, [&est](sctl::Long a, sctl::Long b) { return est[a] > est[b]; });
        if (est[order[0]] <= tol) return U; // the estimate already meets tol
        for (sctl::Long i = 0; i < Ns; i++) rows[i] = order[i];
      }

      // the rows upsampled 2U times: the even points are the rows upsampled U times, the odd points the midpoints;
      // ext: the even points of each row, extended periodically by p on each side, and mid: the odd points
      const sctl::Long n = U * N, Ld = n + 2 * p;
      sctl::ScratchBuf<Real> ext(Ns * Ld), mid(Ns * n);
      {
        const sctl::Iterator<Real> E = ext.begin(), Mi = mid.begin();
        const sctl::ConstIterator<Real> C = coeff.begin();
        const sctl::ConstIterator<sctl::Long> rows_ = rows.begin();
        detail_surfop::RowsC2R<Real>(Ns, 2 * n, Nk, [C,rows_,Nk,N](const sctl::Long i, const sctl::Iterator<Real> c) {
          for (sctl::Long k = 0; k < Nk; k++) {
            const Real scal_k = (2 * k == N ? (Real)0.5 : (Real)1); // the Nyquist mode of even N is a cosine
            c[k * 2 + 0] = C[(rows_[i] * Nk + k) * 2 + 0] * scal_k;
            c[k * 2 + 1] = C[(rows_[i] * Nk + k) * 2 + 1] * scal_k;
          }
        }, [E,Mi,n,p,Ld](const sctl::Long i, const sctl::ConstIterator<Real> g) {
          for (sctl::Long j = -p; j < n + p; j++) E[i * Ld + p + j] = g[2 * (((j % n) + n) % n)];
          for (sctl::Long j = 0; j < n; j++) Mi[i * n + j] = g[2 * j + 1];
        });
      }
      constexpr sctl::Long CHUNK = 256; // points of a row in a task
      const sctl::Long Nch = (n + CHUNK - 1) / CHUNK;
      sctl::ScratchBuf<Real> err_ch(Ns * Nch), size_ch(Ns * Nch); // the largest error and value in each chunk of each row
      sctl::omp_par::parallel_for(Ns * Nch, [&ext,&mid,&err_ch,&size_ch,&wmid,Nch,n,Ld,p](const sctl::Long task) {
        const sctl::Long i = task / Nch, j0 = (task % Nch) * CHUNK, j1 = std::min(n, j0 + CHUNK);
        const sctl::ConstIterator<Real> e_ = ext.begin() + i * Ld + p, m_ = mid.begin() + i * n;
        Real e = 0, size = 0;
        for (sctl::Long j = j0; j < j1; j++) {
          Real v = 0;
          for (sctl::Integer m = 0; m < p; m++) v += wmid[m] * e_[j - p / 2 + 1 + m];
          e = std::max<Real>(e, sctl::fabs<Real>(v - m_[j]));
          size = std::max<Real>(size, sctl::fabs<Real>(e_[j]));
        }
        err_ch[task] = e;
        size_ch[task] = size;
      });
      Real err_max = 0;
      for (sctl::Long i = 0; i < Ns; i++) {
        Real e = 0, size = 0;
        for (sctl::Long c = 0; c < Nch; c++) {
          e = std::max<Real>(e, err_ch[i * Nch + c]);
          size = std::max<Real>(size, size_ch[i * Nch + c]);
        }
        err_max = std::max<Real>(err_max, (e - round * size) * scal);
      }
      if (err_max <= tol) return U;
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

template <class Real, class SKer> template <sctl::Integer DIGITS, sctl::Integer Nv> OffSurfaceOp<Real,SKer>::OffSurfaceOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) : ker_(ker.GetSctlKernel()), comm_(comm), tol_(0), Nt_(0), Np_(0), Ntrg(0), trg0(0), trg1(0), L_max_(0), Ut(1), Up(1), warned_panels(false), warned_interp(false), flip(false) {
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::Setup(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Xt, const Real tol, const sctl::Vector<Real>& F) {
  (void)F;
  SetupSurface(X, Nt, Np, tol);
  SetTargets(Xt);
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::SetupSurface(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const Real tol) {
  SCTL_ASSERT(X.Dim() == COORD_DIM * Nt * Np);
  Nt_ = Nt;
  Np_ = Np;
  X_ = X;
  tol_ = tol;

  panels.ReInit(Mt0 * Mp0); // the coarse mesh
  for (sctl::Long a = 0; a < Mt0; a++) {
    for (sctl::Long b = 0; b < Mp0; b++) panels[a * Mp0 + b] = detail_offsurf::Panel{0, 0, a, b};
  }
  Ut = 1;
  Up = 1;
  warned_panels = false;
  warned_interp = false;
  FitPanels();
  grp_idx_.resize(panels.Dim());
  grp_.resize(panels.Dim() + 1);
  grp_.resize(detail_offsurf::GroupByInterval(grp_idx_.begin(), grp_.begin(), panels) + 1);
  InterpToNodes(coord_, X_, COORD_DIM, Ut, Up); // the surface coordinates at the nodes of the panels
  L_max_ = detail_offsurf::PanelRule<Real>(ORDER).L_max;
  op.reset(); // for the targets of the next SetTargets()
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::SetTargets(const sctl::Vector<Real>& Xt) {
  const sctl::Long Ntrg_ = Xt.Dim() / COORD_DIM;
  SCTL_ASSERT(Xt.Dim() == COORD_DIM * Ntrg_);
  if (op && Ntrg_ == Ntrg) { // the targets of the operator already: nothing to do
    bool same = true;
    for (sctl::Long i = trg0; i < trg1 && same; i++) {
      for (sctl::Integer k = 0; k < COORD_DIM; k++) same = same && (Xt_[(i - trg0) * COORD_DIM + k] == Xt[k * Ntrg + i]);
    }
    if (same) return;
  }
  Ntrg = Ntrg_;
  trg0 = (comm_.Rank() + 0) * Ntrg / comm_.Size();
  trg1 = (comm_.Rank() + 1) * Ntrg / comm_.Size();
  Xt_.ReInit(COORD_DIM * (trg1 - trg0));
  for (sctl::Long i = trg0; i < trg1; i++) {
    for (sctl::Integer k = 0; k < COORD_DIM; k++) Xt_[(i - trg0) * COORD_DIM + k] = Xt[k * Ntrg + i];
  }
  BuildPanels(coord_);
}

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) {
  const sctl::Long N = Nt_ * Np_;
  const sctl::Long dof = F.Dim() / (KDIM0 * N);
  SCTL_ASSERT(F.Dim() == dof * KDIM0 * N);

  // The FFT upsampling of F before its local interpolation to the nodes: at least that of the surface, and enough for
  // the error of the local interpolation to be within tol/2
  const Real cF = L_max_ * detail_offsurf::InvMaxAbs<Real>(F);
  const auto upsample_factor = [this,&F,cF](const sctl::Long U0, const sctl::Integer dir) {
    const sctl::Long U = detail_offsurf::LocalUpsampleLines<Real>(F, Nt_, Np_, dir, cF, tol_ / 2, STENCIL, U0);
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

template <class Real, class SKer> void OffSurfaceOp<Real,SKer>::FitPanels() {
  const sctl::Long N = Nt_ * Np_, Nnds = ORDER * ORDER;
  const detail_offsurf::PanelRule<Real> rule(ORDER);

  // Y: the fields that the panels must resolve, X and its derivatives along each direction, which give the normal and
  // the area element; scal: for each component, the reciprocal of the largest value of its field; sA: the reciprocal
  // of the surface area, by the trapezoidal rule on the grid with the parameters in [0,1)
  const sctl::Long Nc = 3 * COORD_DIM;
  sctl::ScratchBuf<Real> Y_buf(Nc * N), scal(Nc);
  sctl::Vector<Real> Y(Y_buf);
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
  constexpr sctl::Long Nd = 2 * COORD_DIM; // dX/dt and dX/dp

  // A: the matrix that maps values at the Gauss-Legendre nodes of [0,1] to the Legendre coefficients of the polynomial
  // that interpolates them, the coefficient of P_m(2x-1) being the sum over i of A[m * ORDER + i] v_i; by the
  // quadrature of (2m+1) v(x) P_m(2x-1) on [0,1], exact for the interpolant
  sctl::ScratchBuf<Real> A(ORDER * ORDER);
  {
    const sctl::Vector<Real>& leg_nds = sctl::LegQuadRule<Real>::nds(ORDER);
    for (sctl::Integer i = 0; i < ORDER; i++) {
      const Real x = 2 * leg_nds[i] - 1;
      Real P0 = 1, P1 = x; // P_m(x) and P_{m+1}(x)
      for (sctl::Integer m = 0; m < ORDER; m++) {
        A[m * ORDER + i] = (2 * m + 1) * wts[i] * P0;
        const Real P2 = ((2 * m + 3) * x * P1 - (m + 1) * P0) / (m + 2);
        P0 = P1;
        P1 = P2;
      }
    }
  }
  // Set tt and tp to the size of the last two Legendre coefficients, along the toroidal (first) and along the poloidal
  // direction, of the polynomial that interpolates the values V at the ORDER x ORDER nodes of a panel: the sum of their
  // absolute values, which bounds their part of the polynomial on the panel, times sc and less their rounding error
  const auto legendre_tails = [&A](Real& tt, Real& tp, const sctl::ConstIterator<Real> V, const Real sc) {
    sctl::StaticArray<Real,ORDER*ORDER> B, C; // B = A V and C = A V A^T
    for (sctl::Integer m = 0; m < ORDER; m++) {
      for (sctl::Integer j = 0; j < ORDER; j++) {
        Real v = 0;
        for (sctl::Integer i = 0; i < ORDER; i++) v += A[m * ORDER + i] * V[i * ORDER + j];
        B[m * ORDER + j] = v;
      }
    }
    for (sctl::Integer m = 0; m < ORDER; m++) {
      for (sctl::Integer n = 0; n < ORDER; n++) {
        Real v = 0;
        for (sctl::Integer j = 0; j < ORDER; j++) v += B[m * ORDER + j] * A[n * ORDER + j];
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
    tt = tt * sc - round;
    tp = tp * sc - round;
  };
  // The area of panel q, with the grid parameters in [0,1), by the Gauss-Legendre rule on its ORDER x ORDER nodes, from
  // the derivatives dX of the surface coordinates at the nodes: the 3 components of dX/dt, then those of dX/dp
  const auto panel_area = [&wts](const sctl::ConstIterator<Real> dX, const detail_offsurf::Panel& q) {
    Real area = 0;
    for (sctl::Integer i = 0; i < ORDER; i++) {
      for (sctl::Integer j = 0; j < ORDER; j++) {
        const sctl::Long n = i * ORDER + j;
        Real xt[3], xp[3];
        for (sctl::Integer k = 0; k < 3; k++) {
          xt[k] = dX[k * ORDER * ORDER + n];
          xp[k] = dX[(3 + k) * ORDER * ORDER + n];
        }
        const Real n0 = xt[1] * xp[2] - xt[2] * xp[1], n1 = xt[2] * xp[0] - xt[0] * xp[2], n2 = xt[0] * xp[1] - xt[1] * xp[0];
        area += wts[i] * wts[j] * sctl::sqrt<Real>(n0 * n0 + n1 * n1 + n2 * n2);
      }
    }
    return area / ((Real)(Mt0 << q.lt) * (Real)(Mp0 << q.lp));
  };
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
        const sctl::Long U = detail_offsurf::LocalUpsampleLines<Real>(X_, Nt_, Np_, dir, cX, tol_ / 2, STENCIL, U0);
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
    // interval by interval of the toroidal direction, in buffers of each thread.
    const sctl::Long Ntf = Ut * Nt_, Npf = Up * Np_, Nfine = Ntf * Npf, Ld = Npf + 2 * STENCIL;
    sctl::ScratchBuf<Real> Yf_buf(Ut * Up > 1 ? Nc * Nfine : 0);
    sctl::Vector<Real> Yf(Yf_buf, false);
    UpsampleGrid(Yf, Y, Ut, Up);
    const sctl::Vector<Real> dXf(Nd * Nfine, Yf.begin() + COORD_DIM * Nfine, false); // dX/dt and dX/dp in Yf
    todo_ = panels;
    done_.ReInit(0);
    resolved = true;
    while (todo_.Dim()) {
      const sctl::Long P = todo_.Dim();
      sctl::ScratchBuf<sctl::Long> idx(P), grp(P + 1);
      const sctl::Long Ng = detail_offsurf::GroupByInterval(idx.begin(), grp.begin(), todo_);
      sctl::ScratchBuf<sctl::Integer> split(P); // bit 0: split along the toroidal direction, bit 1: along the poloidal
      sctl::ScratchBuf<Real> xm(P), nm(P); // of each panel, the largest x at its nodes, and there the x component of X_t x X_p
      sctl::ScratchBuf<char> group_resolved(Ng); // whether the panels of each group are resolved or split
      sctl::omp_par::parallel_for(Ng, [this,&idx,&grp,&Yf,&dXf,&legendre_tails,&scal,&panel_area,sA,&split,&xm,&nm,&group_resolved,Ld,Ntf,Npf,&nds,lt_max,lp_max](const sctl::Long g) {
        bool resolved_ = true;
        // T: Yf at the toroidal nodes of the interval, and Th: dXf at those of its two halves, on every poloidal grid
        // line; V: Yf at the nodes of a panel, and Vh: dXf at the nodes of its halves
        const detail_offsurf::Panel& q0 = todo_[idx[grp[g]]];
        sctl::ScratchBuf<Real> T(Nc * ORDER * Ld), Th(2 * Nd * ORDER * Ld), V(Nc * Nnds), Vh(4 * Nd * Nnds);
        detail_offsurf::ToroidalRows<Real>(T.begin(), Yf, Ntf, Npf, q0.lt, q0.it, Mt0, nds, STENCIL);
        for (sctl::Integer h = 0; h < 2; h++) detail_offsurf::ToroidalRows<Real>(Th.begin() + h * Nd * ORDER * Ld, dXf, Ntf, Npf, q0.lt + 1, 2 * q0.it + h, Mt0, nds, STENCIL);
        for (sctl::Long n = grp[g]; n < grp[g + 1]; n++) {
          const sctl::Long e = idx[n];
          const detail_offsurf::Panel& q = todo_[e];
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
            legendre_tails(tt_, tp_, V.begin() + c * Nnds, scal[c]);
            tt = std::max<Real>(tt, tt_);
            tp = std::max<Real>(tp, tp_);
          }
          { // the area of the panel against the sum of the areas of its halves along each direction
            const auto half = [&panel_area,&Vh,&halves](const sctl::Integer h) { return panel_area(Vh.begin() + h * Nd * Nnds, halves[h]); };
            const Real A0 = panel_area(V.begin() + COORD_DIM * Nnds, q);
            const Real round = 64 * ORDER * sctl::machine_eps<Real>(); // for areas of size 1
            tt = std::max<Real>(tt, sA * sctl::fabs<Real>(A0 - half(0) - half(1)) - round);
            tp = std::max<Real>(tp, sA * sctl::fabs<Real>(A0 - half(2) - half(3)) - round);
          }
          split[e] = (tt > tol_ && q.lt < lt_max ? 1 : 0) | (tp > tol_ && q.lp < lp_max ? 2 : 0);
          if ((tt > tol_ && q.lt >= lt_max) || (tp > tol_ && q.lp >= lp_max)) resolved_ = false;

          sctl::Long m = 0; // the first node of largest x
          for (sctl::Long i = 0; i < Nnds; i++) {
            if (V[i] > V[m]) m = i;
          }
          const auto Vm = [&V,m](const sctl::Long c) { return V[c * Nnds + m]; };
          xm[e] = Vm(0);
          nm[e] = Vm(COORD_DIM + 1) * Vm(2 * COORD_DIM + 2) - Vm(COORD_DIM + 2) * Vm(2 * COORD_DIM + 1);
        }
        group_resolved[g] = resolved_;
      }, true);
      for (sctl::Long g = 0; g < Ng; g++) resolved = resolved && group_resolved[g];

      next_.ReInit(0);
      for (sctl::Long e = 0; e < P; e++) {
        const detail_offsurf::Panel& q = todo_[e];
        if (!split[e]) {
          done_.PushBack(q);
          if (done_.Dim() == 1 || xm[e] > xmax) {
            xmax = xm[e];
            nmax = nm[e];
          }
          continue;
        }
        const sctl::Integer nt = (split[e] & 1 ? 2 : 1), np = (split[e] & 2 ? 2 : 1);
        for (sctl::Integer a = 0; a < nt; a++) {
          for (sctl::Integer b = 0; b < np; b++) next_.PushBack(detail_offsurf::Panel{q.lt + nt - 1, q.lp + np - 1, q.it * nt + a, q.ip * np + b});
        }
      }
      todo_.Swap(next_);
    }
    panels.Swap(done_);
  }
  if (!resolved) {
    if (!warned_panels) SCTL_WARN("The surface is not resolved on its grid; the panels may not reach the requested accuracy.");
    warned_panels = true;
  }

  flip = (nmax < 0); // the normal X_t x X_p points inward at the node of largest x
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

  sctl::ScratchBuf<Real> Yf_buf(ut * up > 1 ? Nc * N * ut * up : 0), V(Nc * P * Nnds);
  sctl::Vector<Real> Yf(Yf_buf, false);
  UpsampleGrid(Yf, Y, ut, up);
  { // V: the values at the nodes of panel e, of component c at node n at V[(c * P + e) * Nnds + n], by p-point local
    // interpolation along the toroidal and then the poloidal direction, interval by interval of the toroidal direction
    const sctl::Long Ntf = ut * Nt_, Npf = up * Np_, Ld = Npf + 2 * STENCIL;
    const sctl::Vector<Real>& nds = sctl::QuadElemList<Real>::ParamNodes(ORDER);
    sctl::omp_par::parallel_for((sctl::Long)grp_.size() - 1, [this,&V,&Yf,&nds,Nc,Ntf,Npf,Ld,P](const sctl::Long g) {
      const detail_offsurf::Panel& q0 = panels[grp_idx_[grp_[g]]];
      sctl::ScratchBuf<Real> T(Nc * ORDER * Ld); // the fields at the toroidal nodes of the interval
      detail_offsurf::ToroidalRows<Real>(T.begin(), Yf, Ntf, Npf, q0.lt, q0.it, Mt0, nds, STENCIL);
      for (sctl::Long n = grp_[g]; n < grp_[g + 1]; n++) {
        const sctl::Long e = grp_idx_[n];
        detail_offsurf::PoloidalNodes<Real>(V.begin() + e * Nnds, P * Nnds, T.begin(), Nc, Npf, panels[e].lp, panels[e].ip, Mp0, nds, flip, STENCIL);
      }
    }, true);
  }
  sctl::omp_par::parallel_for(P, [kdim,&Ynds,P,&V,Nc](const sctl::Long e) { // panel e, for each component c of field d
    for (sctl::Long c = 0; c < Nc; c++) {
      const sctl::Long d = c / kdim, k = c % kdim;
      for (sctl::Long n = 0; n < Nnds; n++) Ynds[((d * P + e) * Nnds + n) * kdim + k] = V[(c * P + e) * Nnds + n];
    }
  });
}

}
