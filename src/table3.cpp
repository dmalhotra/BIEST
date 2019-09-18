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
    sctl::Comm comm = sctl::Comm::Self();
    sctl::Profile::Enable(true);

    { // BoundaryIntegralOp Convergence Laplace
      const auto& sl_ker = biest::Laplace3D<Real>::FxU();
      const auto& dl_ker = biest::Laplace3D<Real>::DxU();
      const auto& sl_grad_ker = biest::Laplace3D<Real>::FxdU();

      biest::BoundaryIntegralOp<Real, 0, 0, 1,  6, 15>::test_GreensIdentity( 70, 14, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 12, 24>::test_GreensIdentity(140, 28, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 15, 27>::test_GreensIdentity(210, 42, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 15, 30>::test_GreensIdentity(280, 56, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 20, 33>::test_GreensIdentity(350, 70, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 20, 36>::test_GreensIdentity(420, 84, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 20, 36>::test_GreensIdentity(490, 98, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 25, 39>::test_GreensIdentity(560,112, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 25, 39>::test_GreensIdentity(630,126, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 25, 42>::test_GreensIdentity(700,140, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);

      // Laplace: Convergence, Scaling (QUAS3)
      // =====================================
      //  Nt      Np      PATCH_DIM    RAD_DIM     Rel-InfError                Setup_1  SingularCorrection_1             Eval_1              Setup_60 SingularCorrection_60             Eval_60
      //  70      14              6         15      2.01309e-03        0.6712 (0.0296)       0.0014 (0.4611)    0.0074 (4.2220)       0.0294 (0.6768)      0.0012 ( 0.5510)   0.0045 (  6.9344)
      // 140      28             12         24      3.94924e-05        6.8881 (0.0328)       0.0156 (0.6267)    0.1103 (4.5487)       0.1595 (1.4161)      0.0016 ( 6.1070)   0.0153 ( 32.8623)
      // 210      42             15         27      5.10610e-06       19.9200 (0.0347)       0.0554 (0.6127)    0.4940 (5.1076)       0.4116 (1.6774)      0.0081 ( 4.2070)   0.0347 ( 72.7426)
      // 280      56             15         30      1.23117e-06       42.8899 (0.0327)       0.0913 (0.6607)    1.4317 (5.5373)       0.8515 (1.6447)      0.0081 ( 7.4681)   0.1013 ( 78.2566)
      // 350      70             20         33      3.18388e-07       83.9184 (0.0364)       0.2321 (0.7100)    3.4784 (5.5695)       1.6283 (1.8739)      0.0122 (13.4872)   0.1313 (147.5204)
      // 420      84             20         36      1.20546e-07      142.9880 (0.0340)       0.3342 (0.7100)    7.0220 (5.7059)       2.7147 (1.7920)      0.0156 (15.1643)   0.2246 (178.3850)
      // 490      98             20         36      4.25015e-08      190.8103 (0.0347)       0.4554 (0.7093)   12.6750 (5.8471)       3.6029 (1.8378)      0.0214 (15.1124)   0.3596 (206.1143)
      // 560     112             25         39      1.58003e-08      309.8085 (0.0368)       0.9047 (0.7214)   21.8694 (5.7859)       5.7617 (1.9793)      0.0310 (21.0360)   0.5861 (215.8979)
      // 630     126             25         39      6.75882e-09      382.7198 (0.0377)       1.1449 (0.7215)   34.2077 (5.9187)       7.1946 (2.0061)      0.0390 (21.1568)   0.8436 (239.9884)
      // 700     140             25         42      3.26562e-09      543.5166 (0.0356)       1.4185 (0.7189)   51.8979 (5.9414)      10.3366 (1.8722)      0.0682 (14.9507)   1.2054 (255.8115)
    }
    { // BoundaryIntegralOp Convergence Helmholtz
      biest::Helmholtz3D<Real> helm(1);
      const auto& sl_ker = helm.FxU();
      const auto& dl_ker = helm.DxU();
      const auto& sl_grad_ker = helm.FxdU();

      biest::BoundaryIntegralOp<Real, 0, 0, 1,  6, 15>::test_GreensIdentity( 70, 14, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 12, 24>::test_GreensIdentity(140, 28, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 15, 27>::test_GreensIdentity(210, 42, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 15, 30>::test_GreensIdentity(280, 56, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 20, 33>::test_GreensIdentity(350, 70, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 20, 36>::test_GreensIdentity(420, 84, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 20, 36>::test_GreensIdentity(490, 98, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 25, 39>::test_GreensIdentity(560,112, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 25, 39>::test_GreensIdentity(630,126, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);
      biest::BoundaryIntegralOp<Real, 0, 0, 1, 25, 42>::test_GreensIdentity(700,140, biest::SurfType::Quas3, sl_ker, dl_ker, sl_grad_ker, comm);

      // Helmholtz: Convergence, Scaling (QUAS3)
      // =======================================
      //  Nt      Np      PATCH_DIM    RAD_DIM     Rel-InfError                Setup_1  SingularCorrection_1             Eval_1              Setup_60 SingularCorrection_60            Eval_60
      //  70      14              6         15      2.58954e-03        1.1405 (0.0338)       0.0038 (0.6962)    0.0372 (1.6711)       0.0416 (0.9258)      0.0016 ( 2.0000)    0.0050 (12.5612)
      // 140      28             12         24      1.48998e-04       11.9208 (0.0367)       0.0433 (0.9061)    0.6096 (1.6273)       0.2606 (1.6797)      0.0047 ( 8.0000)    0.0223 (44.4003)
      // 210      42             15         27      1.43666e-05       34.8412 (0.0384)       0.1444 (0.9393)    2.9359 (1.6890)       0.7496 (1.7844)      0.0112 (12.0000)    0.0944 (52.5438)
      // 280      56             15         30      3.99907e-06       74.9218 (0.0362)       0.2502 (0.9638)    8.9246 (1.7351)       1.4222 (1.9078)      0.0154 (17.0000)    0.2049 (75.5864)
      // 350      70             20         33      7.68115e-07      147.8727 (0.0400)       0.6391 (1.0312)   21.8899 (1.7302)       2.7717 (2.1330)      0.0289 (22.0000)    0.4509 (83.9916)
      // 420      84             20         36      4.04507e-07      248.1457 (0.0380)       0.9438 (1.0055)   44.6486 (1.7496)       4.5682 (2.0633)      0.0400 (24.0000)    0.8855 (88.2162)
      // 490      98             20         36      1.11807e-07      337.7842 (0.0380)       1.2844 (1.0057)   81.9467 (1.7604)       6.1405 (2.0892)      0.0528 (24.0000)    1.5868 (90.9145)
      // 560     112             25         39      4.93482e-08      537.5907 (0.0411)       2.5404 (1.0275)  140.1854 (1.7584)       9.7783 (2.2596)      0.0959 (27.0000)    2.6649 (92.4997)
      // 630     126             25         39      2.14927e-08      678.2406 (0.0412)       3.2260 (1.0241)  223.4211 (1.7634)      12.3020 (2.2731)      0.1292 (25.6000)    4.1977 (93.8549)
      // 700     140             25         42      7.21371e-09      946.6914 (0.0396)       3.9925 (1.0216)  338.8612 (1.7692)      17.5156 (2.1407)      0.1701 (24.0000)    6.3097 (95.0169)
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

