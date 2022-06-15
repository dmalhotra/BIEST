#ifndef _SURFACE_HPP_
#define _SURFACE_HPP_

#include <sctl.hpp>

namespace biest {

/**
 * Predefined surface shapes.
 */
enum class SurfType {
  AxisymCircleWide,        // 125 x 50
  AxisymCircleNarrow,      // 250 x 50
  AxisymWide,              // 125 x 50
  AxisymNarrow,            // 250 x 50
  RotatingEllipseWide,     // 125 x 50
  RotatingEllipseNarrow,   // 250 x 50
  Quas3,                   // 250 x 50
  LHD,                     // 250 x 50
  W7X,                     // 250 x 45
  Stell,                   // 250 x 50
  W7X_,                    // 250 x 50
  None
};

template <class Real> class Surface {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    /**
     * Constructor.
     */
    Surface() : Nt0(0), Np0(0) {}

    /**
     * Constructor surface of given dimensions.
     *
     * @param[in] Nt number of discretization nodes in toroidal direction.
     *
     * @param[in] Np number of discretization nodes in poloidal direction.
     *
     * @param[in] type one of the predefined surface shapes
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
     * @param[in] k the poloidal index, can be 1, 2 or 3.
     */
    Real& Coord(sctl::Long i, sctl::Long j, sctl::Long k) {return X0_[(k*Nt0+i)*Np0+j];}
    const Real& Coord(sctl::Long i, sctl::Long j, sctl::Long k) const {return X0_[(k*Nt0+i)*Np0+j];}

  private:

    sctl::Long Nt0, Np0;
    sctl::Vector<Real> X0_;
};

// Forward declare
struct VTKData;
struct VTUData;

template <class Real> void WriteVTK(const char* fname, const sctl::Vector<Surface<Real>>& Svec, const sctl::Vector<Real> F = sctl::Vector<Real>(), const sctl::Comm& comm = sctl::Comm::Self());

template <class Real> void WriteVTK(const char* fname, const Surface<Real>& S, const sctl::Vector<Real> F = sctl::Vector<Real>(), const sctl::Comm& comm = sctl::Comm::Self());

}

#include <biest/surface.txx>

#endif  //_SURFACE_HPP_
