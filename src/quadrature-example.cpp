#include <biest.hpp>
#include <sctl.hpp>

int main(int argc, char** argv) {
  { // Compute layer potential on a toroidal surface
    sctl::Profile::Enable(true);
    typedef double Real;

    long Nt = 200, Np = 50;
    sctl::Vector<biest::Surface<Real>> Svec(1); // vector of surfaces
    sctl::Vector<Real> f(Nt * Np); // surface charge density
    { // Initialize Svec, f
      auto density_fn = [](Real X, Real Y, Real Z) {
        Real R2 = (X-4)*(X-4) + Y*Y + Z*Z;
        return exp(-2*R2);
      };
      Svec[0] = biest::Surface<Real>(Nt, Np);
      for (long t = 0; t < Nt; t++) { // toroidal direction
        for (long p = 0; p < Np; p++) { // poloidal direction
          Real X = (3 + cos(2*M_PI*p/Np)) * cos(2*M_PI*t/Nt);
          Real Y = (3 + cos(2*M_PI*p/Np)) * sin(2*M_PI*t/Nt);
          Real Z = sin(2*M_PI*p/Np);
          Svec[0].Coord(t,p,0) = X;
          Svec[0].Coord(t,p,1) = Y;
          Svec[0].Coord(t,p,2) = Z;
          f[t*Np+p] = density_fn(X,Y,Z);
        }
      }
    }
    WriteVTK("f", Svec, f); // visualize f

    // Setup the boundary integral operator
    constexpr int DENSITY_DOF = 1; // degree-of-freedom of density function at each grid point
    constexpr int POTENTIAL_DOF = 1; // degree-of-freedom of the potential at each grid point
    constexpr int UPSAMPLE = 1; // upsample factor for the quadrature
    constexpr int PDIM = 24; // dimension of the square patch for singular integration
    constexpr int QDIM = 24; // order of the polar-coordinate quadrature
    biest::BoundaryIntegralOp<Real, DENSITY_DOF, POTENTIAL_DOF, UPSAMPLE, PDIM, QDIM> LaplaceSingleLayer;
    LaplaceSingleLayer.SetupSingular(Svec, biest::Laplace3D<Real>::FxU());

    // Evaluate boundary integral operator
    sctl::Vector<Real> u;
    LaplaceSingleLayer(u, f);
    WriteVTK("u", Svec, u); // visualize f
  }
  return 0;
}

