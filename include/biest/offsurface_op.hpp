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
 * panels and the operator are set up once per set of targets.
 *
 * @tparam SKer the SCTL kernel.
 */
template <class Real, class SKer> class OffSurfaceOp {
    static constexpr sctl::Integer COORD_DIM = 3;
    static constexpr sctl::Integer ORDER = 12; // element order of the panels
    static constexpr sctl::Integer STENCIL = 12; // number of points of the local interpolation along each direction
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
     * @param[in] F (optional) densities on the grid in SoA order, which the panels must also resolve.
     */
    void Setup(const sctl::Vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const sctl::Vector<Real>& Xt, const Real tol, const sctl::Vector<Real>& F = sctl::Vector<Real>());

    /**
     * Add the potential of the densities F at the target points to U. If the panels do not resolve F, more panels are
     * fit and the operator is set up again.
     *
     * @param[in,out] U the potential at the target points in SoA order.
     *
     * @param[in] F the densities on the grid in SoA order.
     */
    void Eval(sctl::Vector<Real>& U, const sctl::Vector<Real>& F);

    /**
     * Returns the number of panels.
     */
    sctl::Long PanelCount() const { return Mt * Mp; }

  private:

    /**
     * Set Mt, Mp, Ut and Up to the smallest values, at least their current values, that resolve the surface and the
     * densities F.
     */
    void FitPanels(const sctl::Vector<Real>& F);

    /**
     * Returns whether the panels and the upsampling resolve the densities F.
     */
    bool Resolves(const sctl::Vector<Real>& F) const;

    /**
     * Build the panels from Mt, Mp, Ut and Up, and set up the operator.
     */
    void BuildPanels();

    /**
     * Interpolate fields on the grid to the panel nodes, by FFT upsampling and local interpolation.
     *
     * @param[out] Ynds the fields at the panel nodes in AoS order, one block of kdim values per node for each field.
     *
     * @param[in] Y the fields on the grid in SoA order, kdim components each.
     *
     * @param[in] kdim the number of components of a field.
     */
    void InterpToNodes(sctl::Vector<Real>& Ynds, const sctl::Vector<Real>& Y, const sctl::Integer kdim) const;

    SKer ker_;
    sctl::Comm comm_;
    Real tol_;
    sctl::Long Nt_, Np_; // grid dimensions
    sctl::Vector<Real> X_; // surface coordinates on the grid, SoA
    sctl::Long Ntrg; // number of target points, over all processes
    sctl::Long trg0, trg1; // indices of the first and one past the last target point of this process
    sctl::Vector<Real> Xt_; // coordinates of the target points of this process, AoS

    sctl::Long Mt, Mp; // number of panels in the toroidal and the poloidal direction
    sctl::Long Ut, Up; // upsampling factor of the grid before the local interpolation, in each direction
    bool flip; // whether the poloidal parameter of the panels is reversed, so that the panel normals point outward
    sctl::Vector<sctl::Long> it0, ip0; // index on the upsampled grid of the first stencil point of each panel node
    sctl::Matrix<Real> Wt, Wp; // weights of the stencil points of each panel node, along each direction
    std::unique_ptr<sctl::BoundaryIntegralOp<Real,SKer>> op;
};

}

#include <biest/offsurface_op.txx>

#endif  //_OFFSURFACE_OP_HPP_
