// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _OFFSURFACE_OP_HPP_
#define _OFFSURFACE_OP_HPP_

#include <memory>

#include <biest/kernel.hpp>
#include <biest/surface.hpp>
#include <biest/surface_op.hpp>
#include <sctl.hpp>
#include <sctl/experimental/quad_element.hpp>
#include <sctl/experimental/quad_element.cpp>

namespace biest {

namespace detail_offsurf {

  /**
   * A panel: the product of the toroidal interval [it, it+1) / (Mt0 2^lt) and the poloidal interval [ip, ip+1) /
   * (Mp0 2^lp) of the grid parameters in [0,1), where Mt0 x Mp0 are the panels of the coarse mesh.
   */
  struct Panel {
    sctl::Integer lt, lp; // the refinement level along each direction
    sctl::Long it, ip; // the index of the interval at that level
  };

  /**
   * An sctl::QuadElemList whose elements all have order ORDER. With it, sctl::BoundaryIntegralOp compiles the near
   * quadrature for this order only, and no self-interaction quadrature.
   */
  template <class Real, sctl::Integer ORDER> class QuadElemListOrder : public sctl::QuadElemList<Real> {
    public:

      using sctl::QuadElemList<Real>::QuadElemList;

      /**
       * Reports an error: a target lies at a panel node, which an off-surface evaluation does not allow.
       */
      template <class Kernel> static void SelfInterac(sctl::Vector<sctl::Matrix<Real>>& M_lst, const Kernel& ker, const Real tol, const bool trg_dot_prod, const sctl::ElementListBase<Real>* self) {
        SCTL_ASSERT_MSG(false, "A target point lies at a node of the panels; off-surface evaluation needs targets off the surface.");
      }

      /**
       * Compute the near-interaction operator of an element for each target, as sctl::QuadElemList::NearInterac() does.
       */
      template <class Kernel> static void NearInterac(sctl::Matrix<Real>& M, const sctl::Vector<Real>& Xt, const sctl::Vector<Real>& normal_trg, const Kernel& ker, const Real tol, const sctl::Long elem_idx, const sctl::ElementListBase<Real>* self) {
        const auto& qel = *static_cast<const sctl::QuadElemList<Real>*>(self);
        SCTL_ASSERT(qel.Order() == ORDER);
        sctl::detail_dyadic_near::NearInteracDyadic<ORDER,Real>(M, Xt, normal_trg, ker, elem_idx, qel, sctl::detail_quadelem::DigitsFromTol<Real>(tol));
      }
  };

}

/**
 * Layer potentials of a toroidal surface at off-surface targets, by quadrature on panels that resolve the surface. The
 * panels are refined adaptively from a coarse mesh, along one or both directions, where the surface or the density
 * needs them. The panels and the operator are set up once per set of targets. Target points must not lie on the
 * surface.
 *
 * @tparam SKer the SCTL kernel, e.g. sctl::Laplace3D_FxdU.
 */
template <class Real, class SKer> class OffSurfaceOp {
    static constexpr sctl::Integer COORD_DIM = 3;
    static constexpr sctl::Integer ORDER = 12; // element order of the panels
    static constexpr sctl::Integer STENCIL = 12; // number of points of the local interpolation along each direction
    static constexpr sctl::Long Mt0 = 4, Mp0 = 4; // number of panels of the coarse mesh along each direction
    static constexpr sctl::Integer KDIM0 = SKer::SrcDim();
    static constexpr sctl::Integer KDIM1 = SKer::TrgDim();
    using ElemList = detail_offsurf::QuadElemListOrder<Real,ORDER>;

  public:

    /**
     * Constructor.
     *
     * @param[in] ker the kernel function.
     *
     * @param[in] comm the communicator, whose processes share the target points.
     */
    template <sctl::Integer DIGITS, sctl::Integer Nv> explicit OffSurfaceOp(const SctlKernelFunction<Real,SKer,DIGITS,Nv>& ker, const sctl::Comm& comm = sctl::Comm::Self());

    /**
     * Fit panels to the surface and set up the operator for the target points.
     *
     * @param[in] X the surface coordinates on an Nt x Np grid in SoA order.
     *
     * @param[in] Nt the grid dimension in the toroidal direction.
     *
     * @param[in] Np the grid dimension in the poloidal direction.
     *
     * @param[in] Xt the coordinates of the target points in SoA order.
     *
     * @param[in] tol the accuracy tolerance.
     *
     * @param[in] F unused: the panels depend only on the surface. It is kept for compatibility.
     */
    void Setup(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Xt, const Real tol, const sctl::Vector<Real>& F = sctl::Vector<Real>());

    /**
     * Add the potential of the densities F at the target points to U. F is interpolated to the panel nodes by FFT
     * upsampling, by a factor chosen for F, and local interpolation; the panels are not changed.
     *
     * @param[in,out] U the potential at the target points in SoA order; it is resized and set to zero first if its
     * size is wrong.
     *
     * @param[in] F the densities on the grid in SoA order.
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F);

    /**
     * Returns the number of panels.
     */
    sctl::Long PanelCount() const { return panels.Dim(); }

  private:

    /**
     * Split the panels until they resolve the surface: on each panel, along each direction, the last two Legendre
     * coefficients of the surface coordinates and of their derivatives are within the tolerance, and the area of the
     * panel by its quadrature differs from the sum over its two halves by at most the tolerance times the surface
     * area, unless the panel is the finest allowed (32 nodes per grid point). Set Ut and Up to the smallest values,
     * at least their current values, for which the local interpolation of the surface is accurate on the finest
     * panels. Set flip, from the normal at the node of largest x.
     *
     * @param[out] coord the surface coordinates at the nodes of the panels, as sctl::QuadElemList takes them.
     */
    void FitPanels(sctl::Vector<Real>& coord);

    /**
     * Build the elements from the surface coordinates at their nodes, and set up the operator.
     */
    void BuildPanels(const sctl::Vector<Real>& coord);

    /**
     * Set Yf to the fields Y on the grid upsampled ut x up times, in its storage if it has their size; Yf refers to Y
     * if ut = up = 1.
     */
    void UpsampleGrid(sctl::Vector<Real>& Yf, const sctl::Vector<Real>& Y, const sctl::Long ut, const sctl::Long up) const;

    /**
     * Interpolate fields on the grid to the panel nodes, by FFT upsampling ut x up times and local interpolation.
     *
     * @param[out] Ynds the fields at the panel nodes in AoS order, one block of kdim values per node for each field.
     *
     * @param[in] Y the fields on the grid in SoA order, kdim components each.
     *
     * @param[in] kdim the number of components of a field.
     *
     * @param[in] ut, up the upsampling factors along the toroidal and the poloidal direction.
     */
    void InterpToNodes(sctl::Vector<Real>& Ynds, const sctl::Vector<Real>& Y, const sctl::Integer kdim, const sctl::Long ut, const sctl::Long up) const;

    SKer ker_;
    sctl::Comm comm_;
    Real tol_;
    sctl::Long Nt_, Np_; // grid dimensions
    sctl::Vector<Real> X_; // surface coordinates on the grid, SoA
    sctl::Long Ntrg; // number of target points, over all processes
    sctl::Long trg0, trg1; // indices of the first and one past the last target point of this process
    sctl::Vector<Real> Xt_; // coordinates of the target points of this process, AoS

    sctl::Vector<detail_offsurf::Panel> panels;
    sctl::Long Ut, Up; // upsampling factor of the grid before the local interpolation of the surface, in each direction
    bool warned_panels, warned_interp; // whether each warning has been printed since Setup
    bool flip; // whether the poloidal nodes of each panel are reversed, so that the panel normals point outward
    std::unique_ptr<sctl::BoundaryIntegralOp<Real,SKer>> op;
};

}

#include <biest/offsurface_op.txx>

#endif  //_OFFSURFACE_OP_HPP_
