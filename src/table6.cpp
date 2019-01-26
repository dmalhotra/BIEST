#include <biest.hpp>
#include <sctl.hpp>

int main(int argc, char** argv) {
#ifdef SCTL_HAVE_PETSC
  PetscInitialize(&argc, &argv, NULL, NULL);
#elif defined(SCTL_HAVE_MPI)
  MPI_Init(&argc, &argv);
#endif

  {
    typedef double Real;
    sctl::Comm comm = sctl::Comm::World();
    sctl::Profile::Enable(true);

    { // TaylorState::test_conv
      sctl::Vector<biest::Surface<Real>> Svec;
      Svec.PushBack(biest::Surface<Real>(52, 26, biest::SurfType::RotatingEllipseWide));
      Svec.PushBack(biest::Surface<Real>(52, 13, biest::SurfType::RotatingEllipseNarrow));

      biest::TaylorState<Real, 1,  6, 15>::test_conv(1.0, Svec, 1, 2.0e-3, 100, comm, 1e-1);
      biest::TaylorState<Real, 1, 12, 24>::test_conv(1.0, Svec, 2, 1.3e-5, 100, comm, 2e-2);
      biest::TaylorState<Real, 1, 18, 36>::test_conv(1.0, Svec, 3, 3.0e-7, 100, comm, 2e-2);
      biest::TaylorState<Real, 1, 24, 40>::test_conv(1.0, Svec, 4, 3.2e-8, 100, comm, 2e-2);
      biest::TaylorState<Real, 1, 25, 45>::test_conv(1.0, Svec, 5, 2.6e-9, 100, comm, 2e-2);

      // +----------------------------------------------------------------------------------------------------------------------------------+
      // | lambda  up  gmres_tol         N  | gmres_iter  rel_inf_error   |    Setup         LB           Quadrature                  Solve |
      // |----------------------------------|-----------------------------|-----------------------------------------------------------------|
      // |    1.0   1     2.0e-3  2.028e+3  |         78     7.2882e-03   |   0.1337     6.8292      0.9149 (28.152)       7.8778 ( 5.1273) |
      // |    1.0   2     1.3e-5  8.112e+3  |        104     5.1287e-05   |   1.2230    33.7480     10.6380 (50.045)      45.6095 (18.3637) |
      // |    1.0   3     3.0e-7  1.825e+4  |        119     1.3888e-06   |   5.9178   111.8300     53.6900 (56.780)     171.4405 (28.0270) |
      // |    1.0   4     3.2e-8  3.245e+4  |        129     1.1232e-07   |  13.7256   239.7400    173.1800 (59.943)     426.6453 (38.3646) |
      // |    1.0   5     2.6e-9  5.070e+4  |        139     1.3980e-08   |  26.4357   552.6200    419.7100 (62.839)     998.7673 (41.9187) |
      // +----------------------------------------------------------------------------------------------------------------------------------+
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

