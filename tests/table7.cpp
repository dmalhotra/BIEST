#include <biest.hpp>
#include <sctl.hpp>
typedef double Real;

auto print_B = [](const sctl::Vector<Real>& B0, long Nt0, long Np0) { // print B0
  std::cout<<"B=[\n";
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt0, Np0, biest::SurfType::W7X_);
  for (long t = 0; t < Nt0; t+=70) {
    for (long p = 0; p < Np0; p+=14) {
      Real theta = t * 2 * sctl::const_pi<Real>() / Nt0;
      //Real phi   = p * 2 * sctl::const_pi<Real>() / Np0;

      //Real X = Svec[0].Coord()[0*Nt*Np + t*Np + p];
      //Real Y = Svec[0].Coord()[1*Nt*Np + t*Np + p];
      //Real Z = Svec[0].Coord()[2*Nt*Np + t*Np + p];
      //Real R = sqrt(X*X + Y*Y + Z*Z);

      Real BX = B0[0*Nt0*Np0 + t*Np0 + p];
      Real BY = B0[1*Nt0*Np0 + t*Np0 + p];
      Real BZ = B0[2*Nt0*Np0 + t*Np0 + p];
      Real BR   =  cos(theta) * BX + sin(theta) * BY;
      Real Bphi = -sin(theta) * BX + cos(theta) * BY;
      printf("%.12e  %.12e  %.12e\n", BR, Bphi, BZ);
    }
  }
  std::cout<<"];\n";
};
template <class Real, long M, long q> sctl::Vector<Real> test_W7X(long up, Real gmres_tol, long Nt0, long Np0, Real tor_flux, Real pol_flux, Real lambda, sctl::Vector<Real> B0, sctl::Comm& comm) {
  auto Resample = [](const sctl::Vector<Real>& B, long Nt, long Np, long Nt0, long Np0) {
    sctl::Vector<Real> B0;
    biest::SurfaceOp<Real>::Upsample(B,Nt,Np, B0,Nt0,Np0);
    return B0;
  };
  auto print_error = [](Real gmres_tol, const sctl::Vector<Real>& B0, const sctl::Vector<Real>& B){
    if (B0.Dim() != B.Dim()) return;
    auto err = B - B0;
    Real max_err = 0, max_val = 0;
    for (auto x : err) max_err = std::max(max_err, fabs(x));
    for (auto x : B0) max_val = std::max(max_val, fabs(x));
    std::cout<<"Result: "<<gmres_tol<<"   "<<max_err<<" / "<<max_val<<'\n';
  };
  long Nt = 70*up, Np = 14*up;
  sctl::Vector<Real> B, J;
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt, Np, biest::SurfType::W7X_);
  sctl::Profile:: Tic("TaylorSolve", &comm);
  if (lambda > 1e-12) biest::TaylorState<Real,1,M,q>::Compute(B, -tor_flux, -pol_flux, lambda, Svec, comm, gmres_tol, 100);
  else biest::VacuumField<Real,1,M,q>::Compute(B, J, tor_flux, pol_flux, Svec, comm, gmres_tol, 100);
  sctl::Profile:: Toc();
  B = Resample(B, Nt, Np, Nt0, Np0);
  print_error(gmres_tol, B0, B);
  return B;
}

int main(int argc, char** argv) {
#ifdef SCTL_HAVE_PETSC
  PetscInitialize(&argc, &argv, NULL, NULL);
#elif defined(SCTL_HAVE_MPI)
  MPI_Init(&argc, &argv);
#endif

  {
    sctl::Comm comm = sctl::Comm::Self();
    sctl::Profile::Enable(true);

    { // Compute Taylor state
      { // lambda = 0.0
        Real tor_flux = 1, pol_flux = 0, lambda = 0.0;
        long Nt0 = 70*32, Np0 = 14*32; // reference solution resolution
        sctl::Vector<Real> B0;
        if (1) {
          B0 = test_W7X<Real, 55, 75>(30, 1e-12, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
          B0.Write("tmp-B0.data");
        } else {
          B0.Read("tmp-B0.data");
        }
        print_B(B0, Nt0, Np0);

        test_W7X<Real,  6, 12>( 1, 1e-00, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real,  6, 12>( 2, 1e-01, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real,  6, 15>( 3, 3e-02, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real,  8, 20>( 4, 1e-02, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 12, 20>( 5, 3e-03, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 12, 25>( 6, 1e-03, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 12, 25>( 7, 3e-04, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 15, 30>( 8, 3e-04, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 15, 35>( 9, 1e-04, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 35>(10, 3e-05, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 35>(11, 1e-05, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 40>(12, 3e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 40>(13, 3e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(14, 1e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(15, 1e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(16, 1e-07, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 50>(17, 1e-07, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 50>(18, 1e-07, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 55>(19, 3e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 30, 55>(20, 3e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 30, 55>(21, 1e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 30, 55>(22, 1e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 30, 60>(23, 1e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 35, 60>(24, 3e-09, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 35, 60>(25, 3e-09, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 35, 65>(26, 1e-09, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 40, 65>(27, 1e-09, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 40, 65>(28, 1e-09, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 45, 70>(29, 1e-10, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);

        // SPEC
        //  Np  Nt  Lr      error  SolveTime
        //  11  11   5     0.3302      14.28
        //  13  13   5     0.0600      38.21
        //  15  15   5     0.0079     115.80
        //  17  17   5     0.0195     163.14
        //  19  19   5     0.0032     336.33
        //  21  21   5     0.0016     527.69*
        //  23  23   5     0.0015     884.88
        //  25  25   5     0.0041    1514.13*
        // * ill conditioned ; construction of Beltrami field failed ;
      }
      { // lambda = 1.0
        Real tor_flux = 1, pol_flux = 0, lambda = 1.0;
        long Nt0 = 70*32, Np0 = 14*32; // reference solution resolution
        sctl::Vector<Real> B0;
        if (1) {
          B0 = test_W7X<Real, 40, 75>(29, 1e-12, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
          B0.Write("tmp-B0.data");
        } else {
          B0.Read("tmp-B0.data");
        }
        print_B(B0, Nt0, Np0);

        test_W7X<Real,  6, 12>( 1, 1e-00, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real,  6, 12>( 2, 1e-01, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real,  6, 15>( 3, 3e-02, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real,  8, 20>( 4, 1e-02, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 12, 20>( 5, 3e-03, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 12, 25>( 6, 1e-03, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 12, 25>( 7, 3e-04, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 15, 30>( 8, 3e-04, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 15, 35>( 9, 1e-04, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 35>(10, 3e-05, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 35>(11, 1e-05, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 40>(12, 3e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 40>(13, 3e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 20, 40>(13, 3e-07, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(14, 1e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(15, 1e-06, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(15, 1e-07, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 45>(16, 1e-07, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 25, 50>(18, 1e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 30, 55>(21, 1e-08, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 30, 60>(23, 1e-09, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 35, 60>(25, 3e-10, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);
        test_W7X<Real, 40, 65>(28, 1e-10, Nt0, Np0, tor_flux, pol_flux, lambda, B0, comm);

        // SPEC
        //  Np  Nt  Lr      error  SolveTime
        //  11  11   5     0.3276      13.72
        //  13  13   5     0.0614      38.20
        //  15  15   5     0.0091     118.38
        //  17  17   5     0.0192     194.66
        //  19  19   5     0.0033     389.74
        //  21  21   5     0.0020     541.61*
        //  23  23   5 8.5127e-04     879.61
        //  25  25   5     0.0024    1530.59*
        //
        //  12  12   5     0.4729      22.87
        //  14  14   5     0.1630      59.36
        //  16  16   5     0.0052     114.94
        //  18  18   5   202.6400     218.54
        //  20  20   5  1540.7000     409.18
        //  22  22   5   855.6692     692.87*
        // * ill conditioned ; construction of Beltrami field failed ;
      }
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

