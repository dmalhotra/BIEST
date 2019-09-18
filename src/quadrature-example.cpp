#include <biest.hpp>
#include <sctl.hpp>

int main(int argc, char** argv) {
#ifdef SCTL_HAVE_PETSC
  PetscInitialize(&argc, &argv, NULL, NULL);
#elif defined(SCTL_HAVE_MPI)
  MPI_Init(&argc, &argv);
#endif

  { // Compute vacuum field and Taylor state
    typedef double Real;
    sctl::Comm comm = sctl::Comm::Self();
    sctl::Profile::Enable(true);

    sctl::Vector<biest::Surface<Real>> Svec(1);
    { // Initialize Svec
      long Nt = 200, Np = 50;
      Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::Quas3);
      // To modify the surface,
      // for (long t = 0; t < Nt; t++) {
      //   for (long p = 0; p < Np; p++) {
      //     Svec[0].Coord()[0*Nt*Np + t*Np + p] = X(theta(t),phi(p));
      //     Svec[0].Coord()[1*Nt*Np + t*Np + p] = Y(theta(t),phi(p));
      //     Svec[0].Coord()[2*Nt*Np + t*Np + p] = Z(theta(t),phi(p));
      //   }
      // }
    }

    // TODO
  }

#ifdef SCTL_HAVE_PETSC
  PetscFinalize();
#elif defined(SCTL_HAVE_MPI)
  MPI_Finalize();
#endif
  return 0;
}

