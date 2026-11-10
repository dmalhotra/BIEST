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
 * The parity of data on a surface under stellarator symmetry, which maps the
 * point (x,y,z) to (x,-y,-z) and reverses the poloidal angle. The image of
 * data with 3 components V is (Vx,-Vy,-Vz), and that of other data is the data
 * itself.
 */
enum class Parity {
  /** The data at the mapped point is the image of the data: surface coordinates. */
  Even,
  /** The data at the mapped point is minus the image of the data: a stellarator symmetric magnetic field B, and B.n. */
  Odd
};

/**
 * A toroidal surface given by the coordinates of the points of a uniform Nt x
 * Np grid in the toroidal and poloidal angles, in SoA order. A surface may
 * have NFP toroidal field periods: its coordinates are then invariant under
 * the rotation by 2*pi/NFP about the z-axis, which maps the grid points of one
 * field period to those of the next.
 */
template <class Real> class Surface {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructs an empty surface.
     */
    Surface() : Nt0(0), Np0(0), NFP_(1) {}

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
     * Constructs a surface with NFP toroidal field periods from its
     * coordinates on one field period, completed to the full torus by
     * symmetry: the surface has NFP*Nt x Np grid points.
     *
     * @param[in] NFP the number of toroidal field periods.
     *
     * @param[in] Nt, Np the grid dimensions of one field period.
     *
     * @param[in] X the coordinates on one field period, in the order of
     * Coord(). The toroidal index may increase in either direction about the
     * z-axis.
     */
    Surface(sctl::Integer NFP, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& X);

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
     * Returns the number of toroidal field periods: 1 unless given to the
     * constructor or to SetNFP(). The grid points of field period j are those
     * with toroidal index in [j*NTor()/NFP(), (j+1)*NTor()/NFP()).
     */
    sctl::Integer NFP() const {return NFP_;}

    /**
     * Declares that the surface has NFP toroidal field periods, which
     * HasSymmetry() must confirm. The coordinates must not change afterwards
     * unless the symmetry is kept or NFP is set back to 1.
     *
     * @param[in] NFP the number of field periods; 1 declares none.
     *
     * @param[in] tol the tolerance of HasSymmetry().
     */
    void SetNFP(sctl::Integer NFP, Real tol = (Real)1e-10);

    /**
     * Returns whether the surface has NFP field periods: NTor() is a multiple
     * of NFP, and the coordinates of each grid point, rotated by 2*pi/NFP about
     * the z-axis (in the direction of the toroidal index), are those of the
     * point NTor()/NFP() grid points further, to within tol relative to the
     * largest coordinate.
     */
    bool HasSymmetry(sctl::Integer NFP, Real tol = (Real)1e-10) const;

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
    sctl::Integer NFP_;
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
