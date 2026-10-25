// Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
// Author: Dhairya Malhotra
// SPDX-License-Identifier: Apache-2.0

#ifndef _SURFACE_HPP_
#define _SURFACE_HPP_

#include <sctl.hpp>

namespace biest {

/**
 * Predefined surface shapes, used for testing and in the examples. Quas3, LHD
 * and W7X read their coordinates from files in geom/, relative to the current
 * working directory.
 */
enum class SurfType {
  /** Axisymmetric torus with major radius 2 and a circular cross-section of radius 1. */
  AxisymCircleWide,        // 125 x 50
  /** Axisymmetric torus with major radius 2 and a circular cross-section of radius 0.5. */
  AxisymCircleNarrow,      // 250 x 50
  /** Axisymmetric torus with major radius 2 and a non-circular cross-section. */
  AxisymWide,              // 125 x 50
  /** As AxisymWide, with a cross-section half the size. */
  AxisymNarrow,            // 250 x 50
  /** Rotating ellipse with major radius 2 and semi-axes 0.7 and 1.0: the outer surface of the toroidal shell in the Taylor states paper. */
  RotatingEllipseWide,     // 125 x 50
  /** Rotating ellipse with major radius 2 and semi-axes 0.3 and 0.55: the inner surface of the toroidal shell in the Taylor states paper. */
  RotatingEllipseNarrow,   // 250 x 50
  /** QAS3 stellarator boundary, read from geom/Quas3-*.mat. */
  Quas3,                   // 250 x 50
  /** LHD boundary, read from geom/LHD-*.mat. */
  LHD,                     // 250 x 50
  /** W7-X boundary, read from geom/W7X-*.mat. */
  W7X,                     // 250 x 45
  /** Five-period stellarator with major radius 10, from a few Fourier coefficients. */
  Stell,                   // 250 x 50
  /** W7-X boundary computed from its Fourier coefficients (no file needed). */
  W7X_,                    // 250 x 50
  /** No shape; the coordinates are not initialized. */
  None
};

/**
 * A toroidal surface given by the coordinates of the points of a uniform Nt x
 * Np grid in the toroidal and poloidal angles, in SoA order.
 */
template <class Real> class Surface {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructs an empty surface.
     */
    Surface() : Nt0(0), Np0(0) {}

    /**
     * Constructs a surface of given dimensions.
     *
     * @param[in] Nt number of discretization nodes in toroidal direction.
     *
     * @param[in] Np number of discretization nodes in poloidal direction.
     *
     * @param[in] type one of the predefined surface shapes; SurfType::None
     * leaves the coordinates uninitialized, to be set with Coord().
     */
    Surface(sctl::Long Nt, sctl::Long Np, SurfType type = SurfType::AxisymCircleWide);

    /**
     * Returns the number of surface discretization nodes in toroidal
     * direction.
     */
    sctl::Long NTor() const {return Nt0;}

    /**
     * Returns the number of surface discretization nodes in poloidal
     * direction.
     */
    sctl::Long NPol() const {return Np0;}

    /**
     * Get the surface coordinates vector.  The coordinates are in the order:
     * {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}, where
     * Nt and Np are the number of discretizations in toroidal and poloidal
     * directions.
     */
    sctl::Vector<Real>& Coord() {return X0_;}
    const sctl::Vector<Real>& Coord() const {return X0_;}

    /**
     * Access surface coordinates in-place.
     *
     * @param[in] i the toroidal index.
     * @param[in] j the poloidal index.
     * @param[in] k the coordinate index, 0, 1 or 2 for x, y or z.
     */
    Real& Coord(sctl::Long i, sctl::Long j, sctl::Long k) {return X0_[(k*Nt0+i)*Np0+j];}
    const Real& Coord(sctl::Long i, sctl::Long j, sctl::Long k) const {return X0_[(k*Nt0+i)*Np0+j];}

  private:

    sctl::Long Nt0, Np0;
    sctl::Vector<Real> X0_;
};

/**
 * Write surfaces and a field on them to the VTK files fname.pvtp and
 * fname_000000.vtp.
 *
 * @param[in] fname the file name prefix.
 *
 * @param[in] Svec the surfaces.
 *
 * @param[in] F optional field on the grids of the surfaces, concatenated, with
 * any number of components per point in SoA order.
 *
 * @param[in] comm the communicator.
 */
template <class Real> void WriteVTK(const char* fname, const sctl::Vector<Surface<Real>>& Svec, const sctl::Vector<Real> F = sctl::Vector<Real>(), const sctl::Comm& comm = sctl::Comm::Self());

/**
 * Write a surface and a field on it to the VTK files fname.pvtp and
 * fname_000000.vtp.
 *
 * @param[in] fname the file name prefix.
 *
 * @param[in] S the surface.
 *
 * @param[in] F optional field on the grid of S, with any number of components
 * per point in SoA order.
 *
 * @param[in] comm the communicator.
 */
template <class Real> void WriteVTK(const char* fname, const Surface<Real>& S, const sctl::Vector<Real> F = sctl::Vector<Real>(), const sctl::Comm& comm = sctl::Comm::Self());

/**
 * Write a surface and a field given on one field period, completed to all NFP
 * field periods by symmetry (SurfaceOp::CompleteVecField()), to the VTK files
 * fname.pvtp and fname_000000.vtp.
 *
 * @param[in] fname the file name prefix.
 *
 * @param[in] NFP number of toroidal field periods.
 *
 * @param[in] half_period X and F are given on half a field period, and the
 * other half is built by stellarator symmetry.
 *
 * @param[in] surf_Nt, surf_Np the grid dimensions of X in one field period.
 *
 * @param[in] X the surface coordinates on one field period, in SoA order.
 *
 * @param[in] src_Nt, src_Np the grid dimensions of F in one field period.
 *
 * @param[in] F the field on one field period, in SoA order.
 */
template <class Real> void WriteVTK(const std::string& fname, const sctl::Integer NFP, bool half_period, const sctl::Long surf_Nt, const sctl::Long surf_Np, const sctl::Vector<Real>& X, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Vector<Real>& F);

}

#include <biest/surface.txx>

#endif  //_SURFACE_HPP_
