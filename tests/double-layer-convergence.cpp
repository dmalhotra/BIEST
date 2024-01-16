#include <biest.hpp>
#include <sctl.hpp>

template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void test(sctl::Long Nt, sctl::Long Np) {
  sctl::Comm comm = sctl::Comm::Self();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt,Np);
  { // Set surface
    auto& coord = Svec[0].Coord();
    for (sctl::Long t = 0; t < Nt; t++) {
      for (sctl::Long p = 0; p < Np; p++) {
        Real theta = t*2*sctl::const_pi<Real>()/Nt;
        Real phi = p*2*sctl::const_pi<Real>()/Np;
        Real R = 1 + 0.1 * sctl::cos<Real>(phi);
        Real Z = 0.1 * sctl::sin<Real>(phi);
        coord[(0*Nt+t)*Np+p] = R*sctl::cos<Real>(theta);
        coord[(1*Nt+t)*Np+p] = R*sctl::sin<Real>(theta);
        coord[(2*Nt+t)*Np+p] = Z;
      }
    }
  }

  biest::BoundaryIntegralOp<Real, 1, 1, UPSAMPLE, PDIM, RDIM> BIOp(comm);
  BIOp.SetupSingular(Svec, biest::Laplace3D<Real>::DxU());

  sctl::Vector<Real> U, F(Svec[0].NTor()*Svec[0].NPol());
  F = 1;
  BIOp(U, F);

  Real err = 0;
  for (auto x:U) err = std::max(err, fabs(x-0.5));
  std::cout<<err<<'\n';

  WriteVTK("S", Svec, U-0.5);
}

template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void test_W7X(sctl::Long Nt, sctl::Long Np) {
  sctl::Comm comm = sctl::Comm::Self();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt,Np, biest::SurfType::W7X_);

  biest::BoundaryIntegralOp<Real, 1, 1, UPSAMPLE, PDIM, RDIM> BIOp(comm);
  BIOp.SetupSingular(Svec, biest::Laplace3D<Real>::DxU());

  sctl::Vector<Real> U, F(Svec[0].NTor()*Svec[0].NPol());
  F = 1;
  BIOp(U, F);

  Real err = 0;
  for (auto x:U) err = std::max(err, fabs(x-0.5));
  std::cout<<err<<'\n';

  WriteVTK("S", Svec, U-0.5);
}

int main(int argc, char** argv) {
#ifdef SCTL_HAVE_PETSC
  PetscInitialize(&argc, &argv, NULL, NULL);
#elif defined(SCTL_HAVE_MPI)
  MPI_Init(&argc, &argv);
#endif

  {
    typedef double Real;
    sctl::Comm comm = sctl::Comm::Self();
    sctl::Profile::Enable(true);

    if (1) { // Double-layer Laplace convergence (torus)
      test<Real, 1,  6, 10>( 15, 13);
      test<Real, 1,  6, 10>( 30, 13);
      test<Real, 1,  6, 12>( 45, 13);
      test<Real, 1,  6, 12>( 60, 13);
      test<Real, 1,  6, 12>( 75, 13);
      test<Real, 1,  9, 16>( 90, 19);
      test<Real, 1,  9, 16>(105, 19);
      test<Real, 1,  9, 18>(120, 19);
      test<Real, 1,  9, 18>(135, 19);
      test<Real, 1, 12, 20>(150, 25);
      test<Real, 1, 15, 22>(210, 31);
      test<Real, 1, 18, 24>(240, 37);
      test<Real, 1, 21, 28>(300, 43);
    }

    if (1) { // Double-layer Laplace convergence (W7X)
      test_W7X<Real, 1,  6, 10>( 1*70,  1*14);
      test_W7X<Real, 1, 13, 15>( 2*70,  2*14);
      test_W7X<Real, 1, 15, 20>( 3*70,  3*14);
      test_W7X<Real, 1, 15, 30>( 4*70,  4*14);
      test_W7X<Real, 1, 20, 35>( 5*70,  5*14);
      test_W7X<Real, 1, 25, 40>( 6*70,  6*14);
      test_W7X<Real, 1, 25, 45>( 7*70,  7*14);
      test_W7X<Real, 1, 30, 50>( 8*70,  8*14);
      test_W7X<Real, 1, 30, 55>( 9*70,  9*14);
      test_W7X<Real, 1, 35, 55>(10*70, 10*14);
      test_W7X<Real, 1, 40, 60>(11*70, 11*14);
      test_W7X<Real, 1, 45, 60>(12*70, 12*14);
      test_W7X<Real, 1, 45, 65>(13*70, 13*14);
      test_W7X<Real, 1, 45, 70>(14*70, 14*14);
    }

    sctl::Profile::print(&comm);
  }

#ifdef SCTL_HAVE_PETSC
  PetscFinalize();
#elif defined(SCTL_HAVE_MPI)
  MPI_Finalize();
#endif
  return 0;
}

