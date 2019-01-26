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

    auto laplace_ker = [](const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg) {
      constexpr Real scal = 1/(4*M_PI);
      constexpr int COORD_DIM = 3;
      long Ns = r_src.Dim() / COORD_DIM;
      long Nt = r_trg.Dim() / COORD_DIM;
      for (long t = 0; t < Nt; t++) {
        Real u = 0;
        for (long s = 0; s < Ns; s++) {
          Real dx = r_src[0*Ns + s] - r_trg[0*Nt + t];
          Real dy = r_src[1*Ns + s] - r_trg[1*Nt + t];
          Real dz = r_src[2*Ns + s] - r_trg[2*Nt + t];
          Real r2 = (dx*dx+dy*dy+dz*dz);
          Real rinv = 0;
          if (r2 > 0) rinv = 1/sqrt(r2);
          u += v_src[s]*rinv;
        }
        v_trg[t] += u*scal;
      }
    };
    auto helmholtz_ker = [](const sctl::Vector<Real>& r_src, const sctl::Vector<Real>& n_src, const sctl::Vector<Real>& v_src, const sctl::Vector<Real>& r_trg, sctl::Vector<Real>& v_trg, Real mu) {
      constexpr Real scal = 1/(4*M_PI);
      constexpr int COORD_DIM = 3;
      long Ns = r_src.Dim() / COORD_DIM;
      long Nt = r_trg.Dim() / COORD_DIM;
      for (long t = 0; t < Nt; t++) {
        Real u[2] = {0,0};
        for (long s = 0; s < Ns; s++) {
          Real dx = r_src[0*Ns + s] - r_trg[0*Nt + t];
          Real dy = r_src[1*Ns + s] - r_trg[1*Nt + t];
          Real dz = r_src[2*Ns + s] - r_trg[2*Nt + t];
          Real r2 = (dx*dx+dy*dy+dz*dz);
          Real r = sqrt(r2);
          Real rinv = 0;
          if (r > 0) rinv = 1 / r;

          Real rmu = r * mu;
          Real cos_rmu, sin_rmu;
          sin_rmu = sin(rmu);
          cos_rmu = cos(rmu);

          Real G[2];
          G[0] = cos_rmu * rinv;
          G[1] = sin_rmu * rinv;
          u[0] += v_src[0*Ns+s]*G[0] - v_src[1*Ns+s]*G[1];
          u[1] += v_src[0*Ns+s]*G[1] + v_src[1*Ns+s]*G[0];
        }
        v_trg[0*Nt+t] += u[0]*scal;
        v_trg[1*Nt+t] += u[1]*scal;
      }
    };
    auto err = [](const sctl::Vector<Real>& v0, const sctl::Vector<Real>& v1) {
      Real error = 0;
      assert(v0.Dim() == v1.Dim());
      for (long i = 0; i < v0.Dim(); i++) error = std::max(error, fabs(v0[i]-v1[i]));
      return error;
    };

    long Ns = 1000;
    long Nt = 1000;
    { // Laplace Kernel
      const auto& ker0 = laplace_ker;
      const auto& ker1 = biest::Laplace3D<Real>::FxU();

      auto rand_vec = [](long N) {
        sctl::Vector<Real> v(N);
        for (long i = 0; i < N; i++) v[i] = 10*M_PI*drand48();
        return v;
      };
      sctl::Vector<Real> r_src = rand_vec(Ns*3);
      sctl::Vector<Real> n_src = rand_vec(Ns*3);
      sctl::Vector<Real> v_src = rand_vec(Ns*1);
      sctl::Vector<Real> r_trg = rand_vec(Nt*3);
      sctl::Vector<Real> v_trg0 = rand_vec(Nt*1)*0;
      sctl::Vector<Real> v_trg1 = rand_vec(Nt*1)*0;

      sctl::Profile::Tic("LaplaceUnVec");
      for (long i = 0; i < 1000; i++) ker0(r_src, n_src, v_src, r_trg, v_trg0);
      sctl::Profile::Toc();
      sctl::Profile::Tic("LaplaceVec");
      for (long i = 0; i < 1000; i++) ker1(r_src, n_src, v_src, r_trg, v_trg1);
      sctl::Profile::Toc();
      std::cout<<err(v_trg0, v_trg1)<<'\n';
    }
    { // Helmholtz Kernel
      biest::Helmholtz3D<Real> helm(1);
      const auto& ker0 = helmholtz_ker;
      const auto& ker1 = helm.FxU();

      auto rand_vec = [](long N) {
        sctl::Vector<Real> v(N);
        for (long i = 0; i < N; i++) v[i] = 10*M_PI*drand48();
        return v;
      };
      sctl::Vector<Real> r_src = rand_vec(Ns*3);
      sctl::Vector<Real> n_src = rand_vec(Ns*3);
      sctl::Vector<Real> v_src = rand_vec(Ns*2);
      sctl::Vector<Real> r_trg = rand_vec(Nt*3);
      sctl::Vector<Real> v_trg0 = rand_vec(Nt*3)*0;
      sctl::Vector<Real> v_trg1 = rand_vec(Nt*3)*0;

      sctl::Profile::Tic("HelmholtzUnVec");
      for (long i = 0; i < 1000; i++) ker0(r_src, n_src, v_src, r_trg, v_trg0, 1.0);
      sctl::Profile::Toc();
      sctl::Profile::Tic("HelmholtzVec");
      for (long i = 0; i < 1000; i++) ker1(r_src, n_src, v_src, r_trg, v_trg1);
      sctl::Profile::Toc();
      std::cout<<err(v_trg0, v_trg1)<<'\n';
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

