// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#include <numeric>

#include <biest/singular_correction.hpp>

namespace biest {

    namespace detail_quadgrid {
      /**
       * Sets qNt x qNp to the quadrature grid of SingCorr::QuadGrid() for the surface X on its Nt x Np grid, from its
       * derivatives in scratch memory (see SurfaceOp::SurfGrad) unless quad gives the upsampling factors.
       */
      template <class SingCorr, class Real> void QuadGrid(sctl::Long& qNt, sctl::Long& qNp, const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const Real tol, const SingularQuadParams& quad, const sctl::Long dNt, const sctl::Long dNp, const sctl::Long mt, const sctl::Long mp, const sctl::Comm& comm) {
        sctl::ScratchBuf<Real> dX_buf(quad.upsample_t ? 0 : 2 * X.Dim());
        sctl::Vector<Real> dX(dX_buf);
        if (!quad.upsample_t) SurfaceOp<Real>(comm, Nt, Np).Grad2D(dX, X);
        SingCorr::QuadGrid(qNt, qNp, dX, Nt, Np, tol, quad, dNt, dNp, mt, mp);
      }
    }

    namespace detail_biop {

    /**
     * The layer potentials of a vector of toroidal surfaces, on their full tori: the densities on the surface grids, and
     * the potential at given grid points of each surface. The components of the density and of the potential are those of
     * the kernel given to SetupSingular().
     *
     * @tparam Real datatype for reals can be float, double or sctl::QuadReal.
     *
     * @tparam HedgehogOrder 0 or 1 gives the principal value on the surface. A
     * value in [2, 16] gives the one-sided limit from the side opposite to the
     * normal (the interior for outward normals), by extrapolation from points
     * off the surface.
     */
    template <class Real, sctl::Integer HedgehogOrder> class BoundaryIntegralOp {
        static constexpr sctl::Integer COORD_DIM = 3;

      public:

        /**
         * Constructor.
         *
         * @param comm the communicator. With several processes, the target points
         * are divided among the processes, and every process holds all the data.
         */
        BoundaryIntegralOp(const sctl::Comm& comm = sctl::Comm::Self());

        /**
         * Returns the dimensions of the boundary integral operator: the size of
         * the density vector (the density components of the kernel times the
         * number of grid points) for i==0, and of the potential vector (its
         * potential components times the number of target points) for i==1.
         */
        sctl::Long Dim(sctl::Integer i) const;

        /**
         * Setup the layer-potential operator and precompute the singular
         * corrections. The target points are the grid points of the surfaces.
         *
         * The quadrature grid of each surface and the singular quadrature on it
         * (see SingularQuadParams) are those given, with the values not given
         * chosen for the tolerance tol: the grid of SingularCorrection::QuadGrid(),
         * the surface grid upsampled at least twice in each direction, and the
         * singular quadrature for tol and the anisotropy of that grid.
         *
         * With several surfaces, the potential of a surface at the targets of
         * another one is the trapezoidal rule on its quadrature grid,
         * except at the targets too close to the surface for it to be accurate to
         * tol: within (log(1/tol) / (2 pi) + 1) times the local grid spacing, as
         * its error is about exp(-2 pi d / h) at a distance d for a spacing h.
         * There, with a kernel that is an SctlKernelFunction, the potential is
         * that of OffSurfaceOp, by quadrature on panels fit to the surface; with
         * another kernel, the trapezoidal rule still.
         *
         * @param[in] Svec vector containing the surfaces.
         *
         * @param[in] ker the kernel function; it must outlive the operator.
         *
         * @param[in] trg_idx_ (optional) for each surface, the indices of the grid
         * points which will be the target points.
         *
         * @param[in] tol the accuracy of the singular quadrature, and of the
         * potential of a surface at the targets of another one.
         *
         * @param[in] quad (optional) the singular quadrature, its zero values
         * chosen from tol.
         */
        template <class Surface, class Kernel> void SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_ = sctl::Vector<sctl::Vector<sctl::Long>>(), const Real tol = (Real)1e-10, const SingularQuadParams& quad = SingularQuadParams());

        /**
         * Evaluate the layer-potential operator. The potential of a surface at
         * the targets of another one is that described in SetupSingular().
         *
         * @param[in,out] U the vector of computed potentials at the target points
         * in SoA order. If U already has the right size the potential is added to
         * it; otherwise it is resized and set to zero first.
         *
         * @param[in] F the vector of surface density values at surface grid points
         * in SoA order. Several densities can be stacked; their potentials are
         * stacked in U in the same order.
         */
        void operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const;

        /**
         * Evaluate the potential at off-surface points, at any distance from the
         * surfaces. The panels for near-surface quadrature are fit to the surfaces
         * by the first call after SetupSingular() (and again for another tol), and
         * their operator is set up for the targets, again only for other targets.
         *
         * @param[in,out] U the computed potential in SoA order, added to U if it
         * already has the right size.
         *
         * @param[in] Xt the position of the target points in SoA order.
         *
         * @param[in] F one density function at grid points in SoA order for each
         * surface concatenated together.
         *
         * @param[in] ker the kernel function given to SetupSingular().
         *
         * @param[in] tol the accuracy tolerance.
         */
        template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol = (Real)1e-10) const;

      private:

        template <class Surface, class Kernel> void SetupHelper(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_, const Real tol, const SingularQuadParams& quad);

        /**
         * The potential of a surface at targets near it, by OffSurfaceOp: for a kernel whose SCTL kernel is SKer.
         */
        struct NearOpBase {
          virtual ~NearOpBase() = default;
          virtual void Setup(const sctl::Vector<Real>& X, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& Xt, Real tol) = 0;
          virtual void SetupSurface(const sctl::Vector<Real>& X, sctl::Long Nt, sctl::Long Np, Real tol) = 0;
          virtual void SetTargets(const sctl::Vector<Real>& Xt) = 0;
          virtual void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) = 0;
        };
        template <class SKer> struct NearOp : NearOpBase {
          template <sctl::Integer DIGITS, sctl::Integer Nv> NearOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) : op(ker, comm) {}
          void Setup(const sctl::Vector<Real>& X, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& Xt, Real tol) override { op.Setup(X, Nt, Np, Xt, tol); }
          void SetupSurface(const sctl::Vector<Real>& X, sctl::Long Nt, sctl::Long Np, Real tol) override { op.SetupSurface(X, Nt, Np, tol); }
          void SetTargets(const sctl::Vector<Real>& Xt) override { op.SetTargets(Xt); }
          void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) override { op.Eval(U, F); }
          OffSurfaceOp<Real,SKer> op;
        };

        /**
         * Returns a NearOp for a kernel that is an SctlKernelFunction, and else null.
         */
        template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> static std::unique_ptr<NearOpBase> NewNearOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm) { return std::unique_ptr<NearOpBase>(new NearOp<SKer>(ker, comm)); }
        template <class Kernel> static std::unique_ptr<NearOpBase> NewNearOp(const Kernel&, const sctl::Comm&) { return nullptr; }

        /**
         * Set up the potential of each surface at the targets of the others: find the targets near the surface, and set
         * up a NearOp for them (see SetupSingular()).
         */
        template <class Surface, class Kernel> void SetupCross(const sctl::Vector<Surface>& Svec, const Kernel& ker, const Real tol);

        /**
         * The parts of the operator that depend on the components of the kernel: the kernel, the trapezoidal rule with it,
         * and the singular corrections of each surface.
         */
        struct KernelOpBase {
          virtual ~KernelOpBase() = default;
          virtual void EvalSurfInteg(const SurfaceOp<Real>& op, sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& X, const sctl::Vector<Real>& Xn, const sctl::Vector<Real>& Xa, const sctl::Vector<Real>& F) const = 0;
          virtual void Correct(sctl::Long s, sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const = 0; // the singular corrections of surface s
        };
        template <sctl::Integer KDIM0, sctl::Integer KDIM1> struct KernelOp : KernelOpBase {
          void EvalSurfInteg(const SurfaceOp<Real>& op, sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& X, const sctl::Vector<Real>& Xn, const sctl::Vector<Real>& Xa, const sctl::Vector<Real>& F) const override { op.EvalSurfInteg(U, Xt, X, Xn, Xa, F, *ker); }
          void Correct(sctl::Long s, sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const override { corr[s](U, F); }
          const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>* ker = nullptr;
          std::vector<SingularCorrection<Real,KDIM0,KDIM1,HedgehogOrder>> corr; // of each surface
        };

        sctl::Comm comm_;
        sctl::Vector<sctl::Long> Nt0, Np0, quad_Nt, quad_Np; // for each surface, its grid and its quadrature grid
        sctl::Vector<SurfaceOp<Real>> Op;
        const void* ker_ptr; // the kernel of the setup
        sctl::StaticArray<sctl::Integer,2> kdim; // its density and potential components
        std::unique_ptr<KernelOpBase> ker_op_; // kept, with its singular corrections, by a later setup of a kernel of the same components

        sctl::StaticArray<sctl::Long,2> dim;
        sctl::Vector<Real> normal_orient;
        sctl::Vector<sctl::Vector<Real>> Xtrg, Xsrc, dXsrc, Xn_src, Xa_src;
        sctl::Vector<sctl::Vector<sctl::Long>> trg_idx;

        // The targets of surface t for the potential of a surface s != t, at cross[s * Nsurf + t]: those far from s, for
        // the trapezoidal rule, and those near it, for near_op. Without near_op, all of them by the trapezoidal rule.
        struct CrossInterac {
          sctl::Vector<sctl::Long> far_idx, near_idx; // indices into the targets of t
          sctl::Vector<Real> Xfar; // the coordinates of the far targets, SoA
          std::unique_ptr<NearOpBase> near_op;
        };
        std::vector<CrossInterac> cross;

        // The operators of EvalOffSurface(), one for each surface, with their panels fit for the tolerance off_tol_, kept
        // until the next SetupSingular()
        mutable std::vector<std::unique_ptr<NearOpBase>> off_op_;
        mutable Real off_tol_ = 0;
    };

    template <class Real, sctl::Integer HedgehogOrder> BoundaryIntegralOp<Real,HedgehogOrder>::BoundaryIntegralOp(const sctl::Comm& comm) : comm_(comm) {
      dim[0] = 0;
      dim[1] = 0;
      ker_ptr = nullptr;
      kdim[0] = 0;
      kdim[1] = 0;
    }

    template <class Real, sctl::Integer HedgehogOrder> sctl::Long BoundaryIntegralOp<Real,HedgehogOrder>::Dim(sctl::Integer i) const { return dim[i]; }

    template <class Real, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,HedgehogOrder>::SetupHelper(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_, const Real tol, const SingularQuadParams& quad) {
      dim[0] = 0;
      dim[1] = 0;
      if (!Svec.Dim()) return;

      ker_ptr = &ker;
      kdim[0] = ker.Dim(0);
      kdim[1] = ker.Dim(1);
      sctl::Long Nsurf = Svec.Dim();
      Nt0   .ReInit(Nsurf);
      Np0   .ReInit(Nsurf);
      quad_Nt.ReInit(Nsurf);
      quad_Np.ReInit(Nsurf);
      Op    .ReInit(Nsurf);
      Xtrg  .ReInit(Nsurf);
      Xsrc  .ReInit(Nsurf);
      dXsrc .ReInit(Nsurf);
      Xn_src.ReInit(Nsurf);
      Xa_src.ReInit(Nsurf);
      normal_orient.ReInit(Nsurf);

      trg_idx = trg_idx_;
      if (!trg_idx.Dim()) {
        trg_idx.ReInit(Nsurf);
        for (sctl::Long i = 0; i < Nsurf; i++) {
          const sctl::Long Ntrg = Svec[i].NTor() * Svec[i].NPol();
          trg_idx[i].ReInit(Ntrg);
          for (sctl::Long j = 0; j < Ntrg; j++) trg_idx[i][j] = j;
        }
      }
      SCTL_ASSERT(trg_idx.Dim()==Nsurf);

      for (sctl::Long i = 0; i < Nsurf; i++) {
        const auto& S = Svec[i];
        const sctl::Long Nsrc0 = S.NTor() * S.NPol();
        const sctl::Vector<Real>& Xsrc0 = S.Coord();

        { // Set Xtrg
          const sctl::Long Ntrg = trg_idx[i].Dim();
          if (Xtrg[i].Dim() != Ntrg*COORD_DIM) {
            Xtrg[i].ReInit(COORD_DIM*Ntrg);
          }
          for (sctl::Long j = 0; j < Ntrg; j++) {
            const sctl::Long trg_idx_ = trg_idx[i][j];
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
              Xtrg[i][k*Ntrg+j] = Xsrc0[k*Nsrc0+trg_idx_];
            }
          }
        }

        Nt0[i] = S.NTor();
        Np0[i] = S.NPol();
        detail_quadgrid::QuadGrid<SingularCorrection<Real,Kernel::Dim(0),Kernel::Dim(1),HedgehogOrder>>(quad_Nt[i], quad_Np[i], Xsrc0, Nt0[i], Np0[i], tol, quad, Nt0[i], Np0[i], Nt0[i], Np0[i], comm_);
        Op[i] = SurfaceOp<Real>(comm_, quad_Nt[i], quad_Np[i]);
        SurfaceOp<Real>::Upsample(Xsrc0, Nt0[i], Np0[i], Xsrc[i], quad_Nt[i], quad_Np[i]);
        Op[i].Grad2D(dXsrc[i], Xsrc[i]);
        normal_orient[i] = Op[i].SurfNormalAreaElem(&Xn_src[i], &Xa_src[i], dXsrc[i], &Xsrc[i]);
        dim[0] += ker.Dim(0) * (Xsrc0.Dim() / COORD_DIM);
        dim[1] += ker.Dim(1) * (Xtrg[i].Dim() / COORD_DIM);
      }
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,HedgehogOrder>::SetupSingular(const sctl::Vector<Surface>& Svec, const Kernel& ker, const sctl::Vector<sctl::Vector<sctl::Long>>& trg_idx_, const Real tol, const SingularQuadParams& quad) {
      using SingCorr = SingularCorrection<Real,Kernel::Dim(0),Kernel::Dim(1),HedgehogOrder>;
      using KerOp = KernelOp<Kernel::Dim(0),Kernel::Dim(1)>;
      SetupHelper(Svec, ker, trg_idx_, tol, quad);
      const sctl::Long Nsurf = Svec.Dim();
      off_op_.clear(); // the panels of EvalOffSurface() for the earlier surfaces
      if (!dynamic_cast<KerOp*>(ker_op_.get())) ker_op_ = std::make_unique<KerOp>(); // that of the earlier setup kept for a kernel of the same components
      KerOp& ker_op = static_cast<KerOp&>(*ker_op_);
      ker_op.ker = &ker;
      auto& singular_correction = ker_op.corr;
      while ((sctl::Long)singular_correction.size() < Nsurf) singular_correction.emplace_back(comm_); // those of the earlier setup kept
      singular_correction.resize(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) {
        const sctl::Long Nt = quad_Nt[i], Np = quad_Np[i];
        SingularQuadParams q = quad; // and the values not given for tol and the grid, the patch within it
        if (!q.patch_rad || !q.rad_nodes || !q.ang_nodes) {
          const SingularQuadParams q0 = SingCorr::QuadParams(tol, SingCorr::GridAnisotropy(dXsrc[i], Nt, Np));
          if (!q.patch_rad) q.patch_rad = std::min<sctl::Long>(q0.patch_rad, (std::min(Nt, Np) - 1) / 2);
          if (!q.rad_nodes) q.rad_nodes = q0.rad_nodes;
          if (!q.ang_nodes) q.ang_nodes = q0.ang_nodes;
        }
        singular_correction[i].Setup(q, Nt / Nt0[i], Np / Np0[i], Nt, Np, Xsrc[i], dXsrc[i], ker, normal_orient[i], trg_idx[i]);
      }
      SetupCross(Svec, ker, tol);
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class Surface, class Kernel> void BoundaryIntegralOp<Real,HedgehogOrder>::SetupCross(const sctl::Vector<Surface>& Svec, const Kernel& ker, const Real tol) {
      const sctl::Long Nsurf = Svec.Dim();
      cross.clear();
      cross.resize(Nsurf * Nsurf);
      if (Nsurf < 2 || !NewNearOp(ker, comm_)) return; // one surface, or a kernel without OffSurfaceOp

      // A target is near s if it is within R_j = c h_j of a point j of the quadrature grid of s, with h_j
      // the larger grid spacing at j (the derivatives are with respect to the parameters in [0,1)): one more than
      // log(1/tol) / (2 pi) grid spacings, for a target that is that far from the surface but not from a grid point.
      // The grid points are in tiles of B x B, each in a ball (center, radius): a target is near a point of the tile
      // only if it is within the radius plus the largest R_j of the center.
      const Real c = sctl::log<Real>(1 / tol) / (2 * sctl::const_pi<Real>()) + 1;
      constexpr sctl::Long B = 8;
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const sctl::Long Nt = quad_Nt[s], Np = quad_Np[s], N = Nt * Np;
        const sctl::Long Bt = (Nt + B - 1) / B, Bp = (Np + B - 1) / B;
        const sctl::Vector<Real>& X = Xsrc[s];
        const sctl::Vector<Real>& dX = dXsrc[s];
        sctl::ScratchBuf<Real> R2(N), tile(Bt * Bp * (COORD_DIM + 1)); // R_j^2; and for each tile, the center and the reach
        sctl::omp_par::parallel_for(N, [N,Nt,Np,&R2,c,&dX](const sctl::Long j) {
          Real ht2 = 0, hp2 = 0;
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            ht2 += dX[(k * 2 + 0) * N + j] * dX[(k * 2 + 0) * N + j];
            hp2 += dX[(k * 2 + 1) * N + j] * dX[(k * 2 + 1) * N + j];
          }
          const Real h = std::max<Real>(sctl::sqrt<Real>(ht2) / Nt, sctl::sqrt<Real>(hp2) / Np);
          R2[j] = c * c * h * h;
        });
        sctl::omp_par::parallel_for(Bt * Bp, [Bp,N,Np,&R2,&tile,Nt,&X](const sctl::Long b) {
          const sctl::Long t0 = (b / Bp) * B, t1 = std::min(Nt, t0 + B), p0 = (b % Bp) * B, p1 = std::min(Np, p0 + B);
          Real x0[COORD_DIM] = {0, 0, 0}, rad2 = 0, R2max = 0;
          for (sctl::Long t = t0; t < t1; t++) {
            for (sctl::Long p = p0; p < p1; p++) {
              for (sctl::Integer k = 0; k < COORD_DIM; k++) x0[k] += X[k * N + t * Np + p] / ((t1 - t0) * (p1 - p0));
            }
          }
          for (sctl::Long t = t0; t < t1; t++) {
            for (sctl::Long p = p0; p < p1; p++) {
              Real r2 = 0;
              for (sctl::Integer k = 0; k < COORD_DIM; k++) r2 += (X[k * N + t * Np + p] - x0[k]) * (X[k * N + t * Np + p] - x0[k]);
              rad2 = std::max<Real>(rad2, r2);
              R2max = std::max<Real>(R2max, R2[t * Np + p]);
            }
          }
          for (sctl::Integer k = 0; k < COORD_DIM; k++) tile[b * (COORD_DIM + 1) + k] = x0[k];
          tile[b * (COORD_DIM + 1) + COORD_DIM] = sctl::sqrt<Real>(rad2) + sctl::sqrt<Real>(R2max);
        });

        for (sctl::Long t = 0; t < Nsurf; t++) {
          if (t == s) continue;
          const sctl::Long Ntrg = trg_idx[t].Dim();
          const sctl::Vector<Real>& Xt = Xtrg[t];
          sctl::ScratchBuf<char> is_near(Ntrg);
          sctl::omp_par::parallel_for(Ntrg, [Ntrg,Bt,Bp,&tile,Np,N,&R2,&is_near,Nt,&Xt,&X](const sctl::Long i) {
            const Real x[COORD_DIM] = {Xt[0 * Ntrg + i], Xt[1 * Ntrg + i], Xt[2 * Ntrg + i]};
            const auto dist2 = [&x](const auto y) { // y: a pointer or an iterator to the coordinates
              Real r2 = 0;
              for (sctl::Integer k = 0; k < COORD_DIM; k++) r2 += (x[k] - y[k]) * (x[k] - y[k]);
              return r2;
            };
            bool near = false;
            for (sctl::Long b = 0; b < Bt * Bp && !near; b++) {
              const Real reach = tile[b * (COORD_DIM + 1) + COORD_DIM];
              if (dist2(tile.begin() + b * (COORD_DIM + 1)) >= reach * reach) continue;
              const sctl::Long t0 = (b / Bp) * B, t1 = std::min(Nt, t0 + B), p0 = (b % Bp) * B, p1 = std::min(Np, p0 + B);
              for (sctl::Long tt = t0; tt < t1 && !near; tt++) {
                for (sctl::Long pp = p0; pp < p1 && !near; pp++) {
                  const sctl::Long j = tt * Np + pp;
                  const Real y[COORD_DIM] = {X[0 * N + j], X[1 * N + j], X[2 * N + j]};
                  near = (dist2(y) < R2[j]);
                }
              }
            }
            is_near[i] = near;
          }, true);

          CrossInterac& ci = cross[s * Nsurf + t];
          for (sctl::Long i = 0; i < Ntrg; i++) (is_near[i] ? ci.near_idx : ci.far_idx).PushBack(i);
          if (!ci.near_idx.Dim()) { // all of them by the trapezoidal rule, on Xtrg[t]
            ci.far_idx.ReInit(0);
            continue;
          }
          const sctl::Long Nfar = ci.far_idx.Dim(), Nnear = ci.near_idx.Dim();
          sctl::ScratchBuf<Real> Xnear_buf(COORD_DIM * Nnear);
          sctl::Vector<Real> Xnear(Xnear_buf);
          ci.Xfar.ReInit(COORD_DIM * Nfar);
          for (sctl::Integer k = 0; k < COORD_DIM; k++) {
            for (sctl::Long i = 0; i < Nfar; i++) ci.Xfar[k * Nfar + i] = Xt[k * Ntrg + ci.far_idx[i]];
            for (sctl::Long i = 0; i < Nnear; i++) Xnear[k * Nnear + i] = Xt[k * Ntrg + ci.near_idx[i]];
          }
          ci.near_op = NewNearOp(ker, comm_);
          ci.near_op->Setup(Svec[s].Coord(), Nt0[s], Np0[s], Xnear, tol);
        }
      }
    }

    template <class Real, sctl::Integer HedgehogOrder> void BoundaryIntegralOp<Real,HedgehogOrder>::operator()(sctl::Vector<Real>& U, const sctl::Vector<Real>& F) const {
      SCTL_ASSERT(ker_ptr);
      sctl::Long dof = F.Dim() / Dim(0);
      SCTL_ASSERT(F.Dim() == dof * Dim(0));
      if (U.Dim() != dof * Dim(1)) {
        U.ReInit(dof * Dim(1));
        U = 0;
      }

      sctl::Long SrcOffset = 0, SingularTrgOffset = 0;
      for (sctl::Long s = 0; s < Xsrc.Dim(); s++) {
        const sctl::Vector<Real> F_(dof * kdim[0] * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + dof * kdim[0] * SrcOffset, false);
        // Fsrc: on the same grid F_ itself, else F_ upsampled to the quadrature grid, in scratch memory (see
        // SurfaceOp::SurfGrad)
        const bool same_grid = (quad_Nt[s] == Nt0[s] && quad_Np[s] == Np0[s]);
        sctl::ScratchBuf<Real> Fsrc_buf(same_grid ? 0 : dof * kdim[0] * quad_Nt[s] * quad_Np[s]);
        sctl::Vector<Real> Fsrc;
        if (same_grid) Fsrc.ReInit(F_.Dim(), (sctl::Iterator<Real>)F_.begin(), false);
        else {
          Fsrc.ReInit(Fsrc_buf.size(), Fsrc_buf.begin(), false);
          SurfaceOp<Real>::Upsample(F_, Nt0[s], Np0[s], Fsrc, quad_Nt[s], quad_Np[s]);
        }

        sctl::Long TrgOffset = 0;
        for (sctl::Long t = 0; t < Xtrg.Dim(); t++) {
          const sctl::Long Ntrg = trg_idx[t].Dim();
          sctl::Vector<Real> U_(dof * kdim[1] * Ntrg, U.begin() + dof * kdim[1] * TrgOffset, false);
          SCTL_ASSERT(Xtrg[t].Dim()/COORD_DIM == Ntrg);
          const CrossInterac* ci = (t != s ? &cross[s * Xtrg.Dim() + t] : nullptr);
          if (!ci || !ci->near_op) {
            ker_op_->EvalSurfInteg(Op[s], U_, Xtrg[t], Xsrc[s], Xn_src[s], Xa_src[s], Fsrc);
          } else { // the trapezoidal rule at the far targets, and near_op at the near ones (see SetupSingular)
            const sctl::Long Nfar = ci->far_idx.Dim(), Nnear = ci->near_idx.Dim(), Nc = dof * kdim[1];
            sctl::ScratchBuf<Real> Ufar_buf(Nc * Nfar), Unear_buf(Nc * Nnear);
            sctl::Vector<Real> Ufar(Ufar_buf), Unear(Unear_buf);
            Ufar = 0;
            Unear = 0;
            if (Nfar) ker_op_->EvalSurfInteg(Op[s], Ufar, ci->Xfar, Xsrc[s], Xn_src[s], Xa_src[s], Fsrc);
            ci->near_op->Eval(Unear, F_);
            for (sctl::Long k = 0; k < Nc; k++) {
              for (sctl::Long i = 0; i < Nfar; i++) U_[k * Ntrg + ci->far_idx[i]] += Ufar[k * Nfar + i];
              for (sctl::Long i = 0; i < Nnear; i++) U_[k * Ntrg + ci->near_idx[i]] += Unear[k * Nnear + i];
            }
          }
          TrgOffset += Ntrg;
        }

        sctl::Vector<Real> U_(dof * kdim[1] * trg_idx[s].Dim(), U.begin() + dof * kdim[1] * SingularTrgOffset, false);
        ker_op_->Correct(s, U_, Fsrc);
        SingularTrgOffset += trg_idx[s].Dim();
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    template <class Real, sctl::Integer HedgehogOrder> template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void BoundaryIntegralOp<Real,HedgehogOrder>::EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol) const {
      SCTL_ASSERT(ker_ptr);
      SCTL_ASSERT_MSG(&ker == ker_ptr, "EvalOffSurface: the kernel function must be the one given to SetupSingular.");
      SCTL_ASSERT(F.Dim() == Dim(0));
      sctl::Long Nt = Xt.Dim() / COORD_DIM;
      SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
      if (U.Dim() != kdim[1] * Nt) {
        U.ReInit(kdim[1] * Nt);
        U = 0;
      }

      const sctl::Long Nsurf = Xsrc.Dim();
      if ((sctl::Long)off_op_.size() != Nsurf || off_tol_ != tol) { // the panels fit for another tolerance
        off_op_.clear();
        off_op_.resize(Nsurf);
        off_tol_ = tol;
      }
      sctl::Long SrcOffset = 0;
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const sctl::Vector<Real> F_(kdim[0] * Nt0[s] * Np0[s], (sctl::Iterator<Real>)F.begin() + kdim[0] * SrcOffset, false);
        if (!off_op_[s]) {
          off_op_[s] = NewNearOp(ker, comm_);
          off_op_[s]->SetupSurface(Xsrc[s], quad_Nt[s], quad_Np[s], tol);
        }
        off_op_[s]->SetTargets(Xt);
        // Fsrc: F_ upsampled to the quadrature grid, in scratch memory (see SurfaceOp::SurfGrad), or F_ itself
        const bool same_grid = (quad_Nt[s] == Nt0[s] && quad_Np[s] == Np0[s]);
        sctl::ScratchBuf<Real> Fsrc_buf(same_grid ? 0 : kdim[0] * quad_Nt[s] * quad_Np[s]);
        sctl::Vector<Real> Fsrc;
        if (same_grid) Fsrc.ReInit(F_.Dim(), (sctl::Iterator<Real>)F_.begin(), false);
        else {
          Fsrc.ReInit(Fsrc_buf.size(), Fsrc_buf.begin(), false);
          SurfaceOp<Real>::Upsample(F_, Nt0[s], Np0[s], Fsrc, quad_Nt[s], quad_Np[s]);
        }
        off_op_[s]->Eval(U, Fsrc);
        SrcOffset += Nt0[s] * Np0[s];
      }
    };

    }

    template <class Real, sctl::Integer HedgehogOrder> FieldPeriodBIOp<Real,HedgehogOrder>::FieldPeriodBIOp(const sctl::Comm& comm) : biop(nullptr), comm_(comm) {
      kdim_[0] = 0;
      kdim_[1] = 0;
    }

    template <class Real, sctl::Integer HedgehogOrder> FieldPeriodBIOp<Real,HedgehogOrder>::~FieldPeriodBIOp() {
      delete biop;
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class Kernel> void FieldPeriodBIOp<Real,HedgehogOrder>::SetupSingular(const biest::Surface<Real>& S, const Kernel& ker, const sctl::Integer digits, const sctl::Integer trg_NFP, const sctl::Long trg_Nt, const sctl::Long trg_Np, const sctl::Long src_Nt, const sctl::Long src_Np) {
      const Real tol = sctl::pow<Real>((Real)0.1, digits);
      trg_Nt_.ReInit(1);
      trg_Np_.ReInit(1);
      quad_Nt_.ReInit(1);
      quad_Np_.ReInit(1);
      trg_Nt_[0] = trg_Nt;
      trg_Np_[0] = trg_Np;
      // The quadrature grid: for the grids of the surface and of the density, a multiple of the target grid
      detail_quadgrid::QuadGrid<SingularCorrection<Real,Kernel::Dim(0),Kernel::Dim(1),HedgehogOrder>>(quad_Nt_[0], quad_Np_[0], S.Coord(), S.NTor(), S.NPol(), tol, SingularQuadParams(), std::max<sctl::Long>(S.NTor(), src_Nt == -1 ? trg_NFP * trg_Nt : src_Nt), std::max<sctl::Long>(S.NPol(), src_Np == -1 ? trg_Np : src_Np), trg_NFP * trg_Nt, trg_Np, comm_);
      const sctl::Vector<biest::Surface<Real>> Svec(1, sctl::Ptr2Itr<biest::Surface<Real>>((biest::Surface<Real>*)&S, 1), false); // a view of S
      SetGrids(Svec, trg_NFP);
      SetupSingular0(ker, tol, SingularQuadParams());
    }

    namespace detail_fieldperiod {
      /**
       * Returns the field periods common to the surfaces: the greatest common divisor of their Surface::NFP().
       */
      template <class Real> sctl::Integer CommonNFP(const sctl::Vector<biest::Surface<Real>>& Svec) {
        sctl::Integer NFP = 0;
        for (const auto& S : Svec) NFP = std::gcd(NFP, S.NFP());
        return std::max<sctl::Integer>(NFP, 1);
      }
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class Kernel> void FieldPeriodBIOp<Real,HedgehogOrder>::SetupSingular(const sctl::Vector<biest::Surface<Real>>& Svec, const Kernel& ker, const Real tol, const SingularQuadParams& quad) {
      const sctl::Long Nsurf = Svec.Dim();
      const sctl::Integer NFP = detail_fieldperiod::CommonNFP(Svec);
      trg_Nt_.ReInit(Nsurf);
      trg_Np_.ReInit(Nsurf);
      quad_Nt_.ReInit(Nsurf);
      quad_Np_.ReInit(Nsurf);
      for (sctl::Long i = 0; i < Nsurf; i++) { // the targets: the grid points of one field period; the quadrature grid: a multiple of the surface grid
        const sctl::Long Nt = Svec[i].NTor(), Np = Svec[i].NPol();
        trg_Nt_[i] = Nt / NFP;
        trg_Np_[i] = Np;
        detail_quadgrid::QuadGrid<SingularCorrection<Real,Kernel::Dim(0),Kernel::Dim(1),HedgehogOrder>>(quad_Nt_[i], quad_Np_[i], Svec[i].Coord(), Nt, Np, tol, quad, Nt, Np, Nt, Np, comm_);
      }
      SetGrids(Svec, NFP);
      SetupSingular0(ker, tol, quad);
    }

    template <class Real, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,HedgehogOrder>::SetGrids(const sctl::Vector<biest::Surface<Real>>& Svec, const sctl::Integer NFP) {
      const sctl::Long Nsurf = Svec.Dim();
      SCTL_ASSERT(trg_Nt_.Dim() == Nsurf && trg_Np_.Dim() == Nsurf && quad_Nt_.Dim() == Nsurf && quad_Np_.Dim() == Nsurf);
      tor_orient_.ReInit(Nsurf);
      Svec_.ReInit(Nsurf);
      trg_idx_.ReInit(Nsurf);
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const auto& S = Svec[s];
        const sctl::Long qNt = quad_Nt_[s], qNp = quad_Np_[s], tNt = trg_Nt_[s], tNp = trg_Np_[s];
        SCTL_ASSERT(qNt % (NFP * tNt) == 0 && qNp % tNp == 0);
        tor_orient_[s] = biest::SurfaceOp<Real>::ToroidalOrientation(S.Coord(), S.NTor(), S.NPol());

        // The surface on its quadrature grid, in the storage of the earlier setup when it has the size
        if (qNt == S.NTor() && qNp == S.NPol()) {
          Svec_[s] = S;
        } else {
          if (Svec_[s].NTor() != qNt || Svec_[s].NPol() != qNp) Svec_[s] = biest::Surface<Real>(qNt, qNp, biest::SurfType::None);
          SurfaceOp<Real>::Resample(Svec_[s].Coord(), qNt, qNp, S.Coord(), S.NTor(), S.NPol());
        }

        // The targets: the points of the tNt x tNp grid of one field period, which are quadrature grid points
        trg_idx_[s].ReInit(tNt * tNp);
        const sctl::Long skip_Nt = qNt / (NFP * tNt);
        const sctl::Long skip_Np = qNp / tNp;
        for (sctl::Long i = 0; i < tNt; i++) {
          for (sctl::Long j = 0; j < tNp; j++) {
            trg_idx_[s][i*tNp+j] = (i*skip_Nt*tNp+j)*skip_Np;
            SCTL_ASSERT(trg_idx_[s][i*tNp+j] < qNt*qNp);
          }
        }
      }
    }

    template <class Real, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,HedgehogOrder>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np) const {
      SCTL_ASSERT(Svec_.Dim() == 1);
      const sctl::Long src_Nt_ = (src_Nt==-1 ? QuadNt() : src_Nt);
      const sctl::Long src_Np_ = (src_Np==-1 ? QuadNp() : src_Np);
      sctl::ScratchBuf<sctl::Integer> NFP_buf(1);
      sctl::ScratchBuf<sctl::Long> src_Nt_buf(1), src_Np_buf(1);
      sctl::Vector<sctl::Integer> NFP_(NFP_buf);
      sctl::Vector<sctl::Long> src_Nt__(src_Nt_buf), src_Np__(src_Np_buf);
      NFP_[0] = NFP;
      src_Nt__[0] = src_Nt_;
      src_Np__[0] = src_Np_;
      Eval(U, F, NFP_, src_Nt__, src_Np__);
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class Op> void FieldPeriodBIOp<Real,HedgehogOrder>::OnQuadGrids(const sctl::Vector<Real>& F, const sctl::Vector<sctl::Integer>& NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const Op& op) const {
      const sctl::Long Nsurf = Svec_.Dim();
      SCTL_ASSERT(NFP.Dim() == Nsurf && src_Nt.Dim() == Nsurf && src_Np.Dim() == Nsurf);
      // The grid points of the density and of the quadrature grid of each surface, and their offsets
      sctl::ScratchBuf<sctl::Long> src_cnt(Nsurf), quad_cnt(Nsurf), src_dsp(Nsurf), quad_dsp(Nsurf);
      bool on_quad = true; // the densities on the quadrature grids
      for (sctl::Long s = 0; s < Nsurf; s++) {
        src_cnt[s] = src_Nt[s] * src_Np[s];
        quad_cnt[s] = quad_Nt_[s] * quad_Np_[s];
        on_quad = on_quad && NFP[s] == 1 && src_Nt[s] == quad_Nt_[s] && src_Np[s] == quad_Np_[s];
      }
      sctl::omp_par::scan(src_cnt.begin(), src_dsp.begin(), Nsurf, (sctl::Long)0);
      sctl::omp_par::scan(quad_cnt.begin(), quad_dsp.begin(), Nsurf, (sctl::Long)0);
      const sctl::Long Nsrc = (Nsurf ? src_dsp[Nsurf - 1] + src_cnt[Nsurf - 1] : 0), Nquad = (Nsurf ? quad_dsp[Nsurf - 1] + quad_cnt[Nsurf - 1] : 0);
      const sctl::Long dof = F.Dim() / (kdim_[0] * Nsrc);
      SCTL_ASSERT(F.Dim() == dof * kdim_[0] * Nsrc);
      if (on_quad) {
        op(F);
        return;
      }

      // The densities of each surface completed to the full torus by symmetry, then resampled to the quadrature grid
      sctl::ScratchBuf<Real> Fq_buf(dof * kdim_[0] * Nquad);
      for (sctl::Long s = 0; s < Nsurf; s++) {
        const sctl::Long Ns = src_cnt[s], Nq = quad_cnt[s];
        const sctl::Vector<Real> F_(dof * kdim_[0] * Ns, (sctl::Iterator<Real>)F.begin() + dof * kdim_[0] * src_dsp[s], false);
        sctl::Vector<Real> Fq_(dof * kdim_[0] * Nq, Fq_buf.begin() + dof * kdim_[0] * quad_dsp[s], false);
        sctl::ScratchBuf<Real> Ff_buf(dof * kdim_[0] * NFP[s] * Ns);
        sctl::Vector<Real> Ff(Ff_buf);
        for (sctl::Long d = 0; d < dof; d++) { // one density at a time: CompleteVecField rotates 3 components as a vector
          const sctl::Vector<Real> Fd(kdim_[0] * Ns, (sctl::Iterator<Real>)F_.begin() + d * kdim_[0] * Ns, false);
          sctl::Vector<Real> Ffd(kdim_[0] * NFP[s] * Ns, Ff.begin() + d * kdim_[0] * NFP[s] * Ns, false);
          biest::SurfaceOp<Real>::CompleteVecField(Ffd, false, false, NFP[s], src_Nt[s], src_Np[s], Fd, 0, tor_orient_[s]);
        }
        biest::SurfaceOp<Real>::Resample(Fq_, quad_Nt_[s], quad_Np_[s], Ff, NFP[s] * src_Nt[s], src_Np[s]);
      }
      const sctl::Vector<Real> Fq(dof * kdim_[0] * Nquad, Fq_buf.begin(), false);
      op(Fq);
    }

    template <class Real, sctl::Integer HedgehogOrder> void FieldPeriodBIOp<Real,HedgehogOrder>::Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F, const sctl::Vector<sctl::Integer>& NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np) const {
      OnQuadGrids(F, NFP, src_Nt, src_Np, [this,&U](const sctl::Vector<Real>& Fq) { (*biop)(U, Fq); });
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void FieldPeriodBIOp<Real,HedgehogOrder>::EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const sctl::Vector<sctl::Integer>& NFP, const sctl::Vector<sctl::Long>& src_Nt, const sctl::Vector<sctl::Long>& src_Np, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol) const {
      SCTL_ASSERT(biop);
      OnQuadGrids(F, NFP, src_Nt, src_Np, [this,&U,&Xt,&ker,tol](const sctl::Vector<Real>& Fq) { biop->EvalOffSurface(U, Xt, Fq, ker, tol); });
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class SKer, sctl::Integer DIGITS, sctl::Integer Nv> void FieldPeriodBIOp<Real,HedgehogOrder>::EvalOffSurface(sctl::Vector<Real>& U, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& F, const sctl::Integer NFP, const sctl::Long src_Nt, const sctl::Long src_Np, const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const Real tol) const {
      SCTL_ASSERT(Svec_.Dim() == 1);
      sctl::ScratchBuf<sctl::Integer> NFP_buf(1);
      sctl::ScratchBuf<sctl::Long> src_Nt_buf(1), src_Np_buf(1);
      sctl::Vector<sctl::Integer> NFP_(NFP_buf);
      sctl::Vector<sctl::Long> src_Nt_(src_Nt_buf), src_Np_(src_Np_buf);
      NFP_[0] = NFP;
      src_Nt_[0] = src_Nt;
      src_Np_[0] = src_Np;
      EvalOffSurface(U, Xt, F, NFP_, src_Nt_, src_Np_, ker, tol);
    }

    template <class Real, sctl::Integer HedgehogOrder> sctl::Long FieldPeriodBIOp<Real,HedgehogOrder>::Dim(sctl::Integer i) const {
      sctl::Long N = 0;
      for (sctl::Long s = 0; s < Svec_.Dim(); s++) N += (i ? kdim_[1] * trg_Nt_[s] * trg_Np_[s] : kdim_[0] * quad_Nt_[s] * quad_Np_[s]);
      return N;
    }

    template <class Real, sctl::Integer HedgehogOrder> template <class Kernel> void FieldPeriodBIOp<Real,HedgehogOrder>::SetupSingular0(const Kernel& ker, const Real tol, const SingularQuadParams& quad) {
      if (!biop) biop = new BIOp(comm_);
      kdim_[0] = ker.Dim(0);
      kdim_[1] = ker.Dim(1);
      SingularQuadParams q = quad; // on the surfaces resampled to their quadrature grids
      q.upsample_t = 1;
      q.upsample_p = 1;
      biop->SetupSingular(Svec_, ker, trg_idx_, tol, q); // in the memory of the earlier setup
    }

}
