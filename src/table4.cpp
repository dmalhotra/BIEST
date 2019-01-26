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

    { // SurfaceOp::test_InvSurfLap
      biest::SurfaceOp<Real>::test_InvSurfLap( 70,  14, biest::SurfType::Quas3, 2.4e-01, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(140,  28, biest::SurfType::Quas3, 3.5e-03, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(210,  42, biest::SurfType::Quas3, 5.2e-04, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(280,  56, biest::SurfType::Quas3, 6.0e-05, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(350,  70, biest::SurfType::Quas3, 7.6e-06, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(420,  84, biest::SurfType::Quas3, 2.1e-06, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(490,  98, biest::SurfType::Quas3, 4.3e-07, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(560, 112, biest::SurfType::Quas3, 8.5e-08, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(630, 126, biest::SurfType::Quas3, 3.0e-08, 100, nullptr, nullptr, comm);
      biest::SurfaceOp<Real>::test_InvSurfLap(700, 140, biest::SurfType::Quas3, 9.8e-09, 100, nullptr, nullptr, comm);

      biest::BoundaryIntegralOp<Real, 1, 1, 1,  6, 15>::test_Precond( 70,  14, biest::SurfType::Quas3, 1.0e+00, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 12, 24>::test_Precond(140,  28, biest::SurfType::Quas3, 8.8e-02, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 15, 27>::test_Precond(210,  42, biest::SurfType::Quas3, 1.3e-02, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 15, 30>::test_Precond(280,  56, biest::SurfType::Quas3, 1.5e-03, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 20, 33>::test_Precond(350,  70, biest::SurfType::Quas3, 1.9e-04, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 20, 36>::test_Precond(420,  84, biest::SurfType::Quas3, 5.1e-05, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 20, 36>::test_Precond(490,  98, biest::SurfType::Quas3, 1.1e-05, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 25, 39>::test_Precond(560, 112, biest::SurfType::Quas3, 2.1e-06, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 25, 39>::test_Precond(630, 126, biest::SurfType::Quas3, 7.5e-07, 100, comm);
      biest::BoundaryIntegralOp<Real, 1, 1, 1, 25, 42>::test_Precond(700, 140, biest::SurfType::Quas3, 2.4e-07, 100, comm);

      //  Nt      Np       gmres_tol   gmres_iter   Rel-InfError    Solve_1   Solve_60
      //  70      14         2.4e-01            1        3.7e-01     0.0014     0.0052
      // 140      28         3.5e-03            7        1.9e-02     0.0099     0.0225
      // 210      42         5.2e-04           11        2.8e-03     0.0290     0.0501
      // 280      56         6.0e-05           16        4.6e-04     0.0729     0.1004
      // 350      70         7.6e-06           21        8.3e-05     0.1605     0.1944
      // 420      84         2.1e-06           24        2.0e-05     0.2648     0.3657
      // 490      98         4.3e-07           28        4.8e-06     0.4232     0.5167
      // 560     112         8.5e-08           32        1.1e-06     0.6257     0.7783
      // 630     126         3.0e-08           35        2.7e-07     0.9847     1.1784
      // 700     140         9.8e-09           37        9.1e-08     1.3520     1.4163

      // SL-Precond, Convergence, Scaling (QUAS3)
      // ========================================
      //  Nt      Np      PATCH_DIM    RAD_DIM       gmres_tol   gmres_iter   Rel-InfError    Setup_60    Solve_1    Setup_60  Solve_60
      //  70      14              6         15         1.0e+00            0        1.0e+00      0.3461     0.0058      0.2248    0.0023
      // 140      28             12         24         8.8e-02            3        1.4e-01      3.1599     0.3077      0.1127    0.0180
      // 210      42             15         27         1.3e-02            6        7.2e-03      9.1648     2.5929      0.2064    0.0869
      // 280      56             15         30         1.5e-03            7        1.3e-03     19.6581     8.8980      0.4235    0.2865
      // 350      70             20         33         1.9e-04            9        7.2e-05     38.7581    26.8777      0.8141    0.6994
      // 420      84             20         36         5.1e-05           10        1.6e-05     65.6292    60.3666      1.3598    1.3961
      // 490      98             20         36         1.1e-05           11        4.0e-06     89.2224   118.3804      1.7408    2.6418
      // 560     112             25         39         2.1e-06           12        8.0e-07    139.9248   222.8234      2.7547    4.7613
      // 630     126             25         39         7.5e-07           12        3.7e-07    175.4283   349.0480      3.4530    7.3486
      // 700     140             25         42         2.4e-07           13        6.3e-08    249.4009   576.0112      5.0012   11.9320
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

