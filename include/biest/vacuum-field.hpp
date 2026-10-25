// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _VACUUM_FIELD_HPP_
#define _VACUUM_FIELD_HPP_

#include <biest/boundary_integ_op.hpp>
#include <biest/surface.hpp>
#include <sctl.hpp>

namespace biest {

  template <class Real, bool exterior> class VacuumFieldBase;

  /**
   * Constructs a vacuum field in the exterior of a surface given B.n on the
   * surface and magnitude of the toroidal current. The surface and the fields
   * have toroidal field-period symmetry, and are given on one field period.
   */
  template <class Real> class ExtVacuumField : public VacuumFieldBase<Real,true> {
    public:

    /**
     * Constructor.
     *
     * @param[in] verbose print the GMRES residuals.
     */
    explicit ExtVacuumField(bool verbose = false) : VacuumFieldBase<Real,true>(verbose) {}

    /**
     * Setup the ExtVacuumField object.
     *
     * @param[in] digits number of decimal digits of accuracy. It sets the
     * quadrature parameters and the GMRES tolerance 10^-digits; the quadratures
     * are set up at the first solve.
     *
     * @param[in] NFP number of toroidal field periods. The surface as well as
     * the magnetic field must have this toroidal periodic symmetry.
     *
     * @param[in] surf_Nt surface discretization order in toroidal direction (in
     * one field period).
     *
     * @param[in] surf_Np surface discretization order in poloidal direction.
     *
     * @param[in] X the surface coordinates in the order {x11, x12, ..., x1Np,
     * x21, x22, ... , x(surf_Nt,surf_Np), y11, ... , z11, ...}. The toroidal
     * index may increase in either direction about the z-axis; the magnetic
     * field grid follows the same direction.
     *
     * @param[in] Nt B-field discretization order in toroidal direction (in one
     * field period).
     *
     * @param[in] Np B-field discretization order in poloidal direction.
     *
     * The resolution parameters for the surface shape (surf_Nt, surf_Np), and
     * the magnetic field (Nt, Np) do not need to be related to each other in
     * any particular way.
     */
    void Setup(const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np);

    /**
     * Compute the dot product of a given vector field B on the surface with the
     * surface normal vector.
     *
     * @param[in] B the surface vector field B = {Bx11, Bx12, ..., Bx1Np, Bx21,
     * Bx22, ... , BxNtNp, By11, ... , Bz11, ...} on the Nt x Np grid of one
     * field period.
     *
     * @return the dot product of the surface field B with the unit normal
     * pointing out of the torus.
     */
    std::vector<Real> ComputeBdotN(const std::vector<Real>& B) const;

    /**
     * Computes Bplasma on the exterior of a toroidal surface such that
     * Bplasma.n + Bcoil_dot_N = 0 on the surface and poloidal circulation of
     * Bplasma equals Jplasma. Bplasma is represented using layer potentials as
     * -grad(S[sigma]) + curl(S[J]). The surface current J is a harmonic surface
     * vector field with net toroidal current Jplasma, flowing in the direction
     * of decreasing toroidal grid index, and sigma is computed by solving a
     * boundary integral equation (BIE) formulation using GMRES. Bplasma, sigma
     * and J are returned.
     *
     * @param[in] Bcoil_dot_N the normal component of the external field on the
     * Nt x Np grid, e.g. from ComputeBdotN().
     *
     * @param[in] Jplasma the net toroidal current of J (with mu0 = 1).
     *
     * @return Bplasma, sigma and J on the Nt x Np grid of one field period (in
     * SoA order). J is empty when Jplasma == 0.
     */
    std::tuple<std::vector<Real>,std::vector<Real>,std::vector<Real>> ComputeBplasma(const std::vector<Real>& Bcoil_dot_N, const Real Jplasma = 0) const;

    /**
     * Compute the single-layer potential U = S[sigma] on the toroidal surface.
     *
     * @param[in] sigma the single-layer density on the Nt x Np grid of one
     * field period.
     *
     * @return the potential U on the same grid.
     */
    std::vector<Real> ComputeU(const std::vector<Real>& sigma) const;

    /**
     * Evaluate Bplasma = -grad(S[sigma]) + curl(S[J]) at points outside the
     * surface, at any distance from it.
     *
     * @param[in] Xt the target points in SoA order {x1, x2, ..., y1, ..., z1, ...}.
     *
     * @param[in] sigma, J the densities returned by ComputeBplasma().
     *
     * @return Bplasma at the target points in SoA order.
     */
    std::vector<Real> EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const std::vector<Real>& J) const;
  };

  /**
   * Constructs a vacuum field in the interior of a surface given B.n on the
   * surface and the magnitude of the current through the center of the torus.
   * The surface and the fields have toroidal field-period symmetry, and are
   * given on one field period. The z-axis must pass through the hole of the
   * torus.
   */
  template <class Real> class IntVacuumField : public VacuumFieldBase<Real,false> {
    public:

    /**
     * Constructor.
     *
     * @param[in] verbose print the GMRES residuals.
     */
    explicit IntVacuumField(bool verbose = false) : VacuumFieldBase<Real,false>(verbose) {}

    /**
     * Setup the IntVacuumField object.
     *
     * @param[in] digits number of decimal digits of accuracy. It sets the
     * quadrature parameters and the GMRES tolerance 10^-digits; the quadratures
     * are set up at the first solve.
     *
     * @param[in] NFP number of toroidal field periods. The surface as well as
     * the magnetic field must have this toroidal periodic symmetry.
     *
     * @param[in] surf_Nt surface discretization order in toroidal direction (in
     * one field period).
     *
     * @param[in] surf_Np surface discretization order in poloidal direction.
     *
     * @param[in] X the surface coordinates in the order {x11, x12, ..., x1Np,
     * x21, x22, ... , x(surf_Nt,surf_Np), y11, ... , z11, ...}. The toroidal
     * index may increase in either direction about the z-axis; the magnetic
     * field grid follows the same direction.
     *
     * @param[in] Nt B-field discretization order in toroidal direction (in one
     * field period).
     *
     * @param[in] Np B-field discretization order in poloidal direction.
     *
     * The resolution parameters for the surface shape (surf_Nt, surf_Np), and
     * the magnetic field (Nt, Np) do not need to be related to each other in
     * any particular way.
     */
    void Setup(const sctl::Integer digits, const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np);

    /**
     * Compute the dot product of a given vector field B on the surface with the
     * surface normal vector.
     *
     * @param[in] B the surface vector field B = {Bx11, Bx12, ..., Bx1Np, Bx21,
     * Bx22, ... , BxNtNp, By11, ... , Bz11, ...} on the Nt x Np grid of one
     * field period.
     *
     * @return the dot product of the surface field B with the unit normal
     * pointing out of the torus.
     */
    std::vector<Real> ComputeBdotN(const std::vector<Real>& B) const;

    /**
     * Computes B on the interior of a toroidal surface such that B.n + B1_dot_N = 0 on
     * the surface and toroidal circulation of B equals I0. B is represented as the sum of
     * a layer potential and the field due to a straight wire carrying a current I0
     * through the axis of the torus: -grad(S[sigma]) + I0/(2 pi R) phi_hat. The unknown
     * density sigma is computed by solving a boundary integral equation (BIE) formulation
     * using GMRES. B, and sigma are returned.
     *
     * @param[in] B1_dot_N the normal component of the field B1 on the Nt x Np
     * grid, e.g. from ComputeBdotN().
     *
     * @param[in] I0 the toroidal circulation of B, along a loop that goes
     * counterclockwise around the z-axis.
     *
     * @return B, sigma on the Nt x Np grid of one field period (in SoA order).
     */
    std::tuple<std::vector<Real>,std::vector<Real>> ComputeB(const std::vector<Real>& B1_dot_N, const Real I0 = 0) const;

    /**
     * Compute the single-layer potential U = S[sigma] on the toroidal surface.
     *
     * @param[in] sigma the single-layer density on the Nt x Np grid of one
     * field period.
     *
     * @return the potential U on the same grid.
     */
    std::vector<Real> ComputeU(const std::vector<Real>& sigma) const;

    /**
     * Evaluate B = -grad(S[sigma]) + I0/(2 pi R) phi_hat at points inside
     * the surface, at any distance from it.
     *
     * @param[in] Xt the target points in SoA order {x1, x2, ..., y1, ..., z1, ...}.
     *
     * @param[in] sigma the density returned by ComputeB().
     *
     * @param[in] I0 the current given to ComputeB().
     *
     * @return B at the target points in SoA order.
     */
    std::vector<Real> EvalOffSurface(const std::vector<Real>& Xt, const std::vector<Real>& sigma, const Real I0) const;
  };

  /**
   * Generate data for testing class ExtVacuumField.
   */
  template <class Real> class ExtVacuumFieldTest {
    static constexpr int COORD_DIM = 3;

    public:

    /**
     * Generate nodal coordinates for toroidal surfaces.
     *
     * @param[in] NFP number of toroidal field periods.
     *
     * @param[in] Nt surface discretization order in toroidal direction (in one
     * field period).
     *
     * @param[in] Np surface discretization order in poloidal direction.
     *
     * @param[in] surf_type prebuilt surface geometries. Possible values
     * SurfType::{AxisymCircleWide, AxisymCircleNarrow, AxisymWide,
     * AxisymNarrow, RotatingEllipseWide, RotatingEllipseNarrow, Quas3, LHD,
     * W7X, Stell}
     *
     * @return the surface coordinates in the order {x11, x12, ..., x1Np,
     * x21, x22, ... , xNtNp, y11, ... , z11, ...}. The coordinates correspond
     * to the surface in the toroidal angle interval [0, 2*pi/NFP).
     */
    static std::vector<Real> SurfaceCoordinates(const sctl::Integer NFP, const sctl::Long Nt, const sctl::Long Np, const SurfType surf_type = SurfType::AxisymNarrow);

    /**
     * Generate a vector field grad(S[sigma]) + BiotSavart(J), where S is the
     * single-layer Laplace kernel 1/(4 pi |r|), sigma is a line source inside
     * the surface and J is a current loop in the torus.
     *
     * @param[in] NFP number of toroidal field periods.
     *
     * @param[in] surf_Nt surface discretization order in toroidal direction (in
     * one field period).
     *
     * @param[in] surf_Np surface discretization order in poloidal direction.
     *
     * @param[in] X the surface coordinates in the order {x11, x12, ..., x1Np,
     * x21, x22, ... , x(surf_Nt,surf_Np), y11, ... , z11, ...}.
     *
     * @param[in] Nt the output field discretization order in toroidal direction
     * (in one field period).
     *
     * @param[in] Np the output field discretization order in poloidal
     * direction.
     *
     * @param[in] Xt exterior off-surface target points in the order {x1, x2,
     * ..., xNtrg, y1, ..., z1, ...}.
     *
     * @return the vector field B = grad(S[sigma]) + BiotSavart(J) at on-surface points in one
     * field period on the Nt x Np grid in the order B = {Bx11, Bx12, ...,
     * Bx1Np, Bx21, Bx22, ... , BxNtNp, By11, ... , Bz11, ...}, Also returns
     * the B field at the exterior off-surface target points Xt in the order
     * {Bx1, Bx2, ..., BxNtrg, By1, ..., Bz1, ...}.
     */
    static std::tuple<std::vector<Real>,std::vector<Real>> BFieldData(const sctl::Integer NFP, const sctl::Long surf_Nt, const sctl::Long surf_Np, const std::vector<Real>& X, const sctl::Long Nt, const sctl::Long Np, const std::vector<Real>& Xt = std::vector<Real>());

    /**
     * Test the ExtVacuumField class.
     */
    static void test(int digits, int NFP, long surf_Nt, long surf_Np, SurfType surf_type, long Nt, long Np) {
      // Construct the surface
      std::vector<Real> X(3*surf_Nt*surf_Np), X_offsurf;
      { // Set X_offsurf
        Surface<Real> S(NFP*Nt, Np, SurfType::AxisymCircleWide);
        X_offsurf.assign(S.Coord().begin(), S.Coord().end());
      }
      X = ExtVacuumFieldTest<Real>::SurfaceCoordinates(NFP, surf_Nt, surf_Np, surf_type);
      //for (long t = 0; t < surf_Nt; t++) { // toroidal direction
      //  for (long p = 0; p < surf_Np; p++) { // poloidal direction
      //    Real x = (2 + 0.5*cos(2*M_PI*p/surf_Np)) * cos(2*M_PI*t/(NFP*surf_Nt));
      //    Real y = (2 + 0.5*cos(2*M_PI*p/surf_Np)) * sin(2*M_PI*t/(NFP*surf_Nt));
      //    Real z = 0.5*sin(2*M_PI*p/surf_Np);
      //    X[(0*surf_Nt+t)*surf_Np+p] = x;
      //    X[(1*surf_Nt+t)*surf_Np+p] = y;
      //    X[(2*surf_Nt+t)*surf_Np+p] = z;
      //  }
      //}

      // Generate B field for testing exterior vacuum fields
      std::vector<Real> B, B_pts;
      std::tie(B, B_pts) = ExtVacuumFieldTest<Real>::BFieldData(NFP, surf_Nt, surf_Np, X, Nt, Np, X_offsurf);

      // Setup
      ExtVacuumField<Real> vacuum_field;
      vacuum_field.Setup(digits, NFP, surf_Nt, surf_Np, X, Nt, Np);

      // Compute Bplassma field such that Bplasma.n = -BdotN
      std::vector<Real> Bplasma, sigma, J;
      const auto BdotN = vacuum_field.ComputeBdotN(B);
      std::tie(Bplasma, sigma, J) = vacuum_field.ComputeBplasma(BdotN, (Real)-1);
      std::vector<Real> Bplasma_pts = vacuum_field.EvalOffSurface(X_offsurf, sigma, J);

      // print error
      Real max_err = 0, max_val = 0;
      std::vector<Real> Berr = B;
      for (long i = 0; i < (long)Berr.size(); i++) Berr[i] += Bplasma[i];
      for (const auto& x:B   ) max_val = std::max<Real>(max_val,fabs(x));
      for (const auto& x:Berr) max_err = std::max<Real>(max_err,fabs(x));
      std::cout<<"Maximum relative error: "<<max_err/max_val<<'\n';

      Berr = B_pts;
      max_err = 0, max_val = 0;
      for (long i = 0; i < (long)Berr.size(); i++) Berr[i] += Bplasma_pts[i];
      for (const auto& x:B   ) max_val = std::max<Real>(max_val,fabs(x));
      for (const auto& x:Berr) max_err = std::max<Real>(max_err,fabs(x));
      std::cout<<"Maximum relative error (off-surface): "<<max_err/max_val<<'\n';
    }

  };

}

#include <biest/vacuum-field.txx>

#endif  //_VACUUM_FIELD_HPP_
