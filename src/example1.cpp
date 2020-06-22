// This example shows how to build a surface from RBC and ZBS coefficients. It
// then computes the force-free field on the boundary and also evaluates at
// points in the interior (away from the surface).
// Note: Evaluation points very close to the surface will have large errors
// because the code is not designed to handle near-singular integration.
//
// # Compiling and running:
// make bin/example1
// ./bin/example1 geom/NCSX

#include <biest.hpp>
#include <sctl.hpp>

template <class Real> class VMECSurface {
  public:

    VMECSurface() {
      order = 10;
      Rcoeff = sctl::Matrix<Real>(2*order+1,2*order+1);
      Zcoeff = sctl::Matrix<Real>(2*order+1,2*order+1);
      Rcoeff = 0;
      Zcoeff = 0;
    }

    void ReadFile(std::string fname) {
      std::ifstream file(fname);
      SCTL_ASSERT_MSG(!file.fail(), "Could not open geometry file!");

      std::string str;
      while(file >> str) {
        if (str.compare("NFP")==0) {
          file >> str;
          SCTL_ASSERT(str.compare("=")==0);

          int nfp = 0;
          file >> nfp;
          SCTL_ASSERT(nfp);
          SetNFP(nfp);
        }
        if (str.compare(0,3,"RBC")==0) {
          int i, j;
          sscanf(str.c_str(), "RBC(%d,%d)", &i, &j);

          file >> str;
          SCTL_ASSERT(str.compare("=")==0);

          Real value = 0;
          file >> value;
          SetRBC(i,j,value);
        }
        if (str.compare(0,3,"ZBS")==0) {
          int i, j;
          sscanf(str.c_str(), "ZBS(%d,%d)", &i, &j);

          file >> str;
          SCTL_ASSERT(str.compare("=")==0);

          Real value = 0;
          file >> value;
          SetZBS(i,j,value);
        }
      }
      file.close();
    }

    void SetNFP(sctl::Long value) {
      NFP = value;
    }

    void SetRBC(sctl::Long i, sctl::Long j, Real value) {
      if (std::max<sctl::Long>(abs(i),abs(j)) > order) {
        resize_coeff(std::max<sctl::Long>(abs(i),abs(j)));
      }
      Rcoeff[i+order][j+order] = value;
    }

    void SetZBS(sctl::Long i, sctl::Long j, Real value) {
      if (std::max<sctl::Long>(abs(i),abs(j)) > order) {
        resize_coeff(std::max<sctl::Long>(abs(i),abs(j)));
      }
      Zcoeff[i+order][j+order] = value;
    }

    biest::Surface<Real> GetSurface(sctl::Long Nt, sctl::Long Np) {
      biest::Surface<Real> S(Nt, Np);
      sctl::Vector<Real> X0(3*Nt*Np);
      for (sctl::Long t = 0; t < Nt; t++) {
        for (sctl::Long p = 0; p < Np; p++) {
          Real theta = t * 2 * sctl::const_pi<Real>() / Nt;
          Real phi   = p * 2 * sctl::const_pi<Real>() / Np;
          Real R = 0, Z = 0;
          for (sctl::Long i = -order; i <= order; i++) {
            for (sctl::Long j = -order; j <= order; j++) {
              R += Rcoeff[i+order][j+order] * sctl::cos(j*phi - NFP*i*theta);
              Z += Zcoeff[i+order][j+order] * sctl::sin(j*phi - NFP*i*theta);
            }
          }
          S.Coord()[0*Nt*Np + t*Np + p] = R * sctl::cos(theta);
          S.Coord()[1*Nt*Np + t*Np + p] = R * sctl::sin(theta);
          S.Coord()[2*Nt*Np + t*Np + p] = Z;
        }
      }
      return S;
    }

  private:

    void resize_coeff(sctl::Long order_new) {
      if (order_new > order) {
        sctl::Matrix<Real> Rcoeff_(2*order+1,2*order+1);
        sctl::Matrix<Real> Zcoeff_(2*order+1,2*order+1);
        Rcoeff_ = 0;
        Zcoeff_ = 0;
        for (sctl::Long i = -order; i < order; i++) {
          for (sctl::Long j = -order; j < order; j++) {
            Rcoeff_[i+order_new][j+order_new] = Rcoeff[i+order][j+order];
            Zcoeff_[i+order_new][j+order_new] = Zcoeff[i+order][j+order];
          }
        }
        order = order_new;
        Rcoeff.Swap(Rcoeff_);
        Zcoeff.Swap(Zcoeff_);
      }
    }

    sctl::Long NFP;
    sctl::Long order;
    sctl::Matrix<Real> Rcoeff;
    sctl::Matrix<Real> Zcoeff;
};

int main(int argc, char** argv) {
  sctl::Comm::MPI_Init(&argc, &argv);
  typedef double Real;

  { // Compute vacuum field and Taylor state
    sctl::Comm comm = sctl::Comm::Self();
    sctl::Profile::Enable(true);

    // Set parameters
    Real beltrami_param = 0.0, tor_flux = 1.0, pol_flux = 0.0;
    long Nt = 200, Np = 50; // grid resolution
    long gmres_iter = 100; // maximum number of GMRES iterations
    Real gmres_tol = 1e-4; // tolerance for GMRES solve

    sctl::Vector<biest::Surface<Real>> Svec(1);
    { // Read RBC and ZBS coefficients from file
      VMECSurface<Real> Svmec;
      SCTL_ASSERT_MSG(argc>1, "Input geometry file must be provided.");
      Svmec.ReadFile(argv[1]);
      Svec[0] = Svmec.GetSurface(Nt,Np);
    }

    long Ntrg = 2; // Number of target points
    sctl::Vector<Real> Xtrg(Ntrg*3);
    { // Set target coordinates
      Xtrg[0] =-1.4; // X-coordinate
      Xtrg[1] = 0.0; // Y-coordinate
      Xtrg[2] = 0.0; // Z-coordinate

      Xtrg[3] = 1.6;
      Xtrg[4] = 0.0;
      Xtrg[5] = 0.0;
    }

    sctl::Vector<Real> B_bdry, B_off_surf;
    sctl::Profile::Tic("ComputeField",&comm);
    if (beltrami_param == 0.0){ // Compute vacuum field
      sctl::Vector<Real> J;
      biest::VacuumField<Real,2,24,35>::Compute(B_bdry, J, tor_flux, pol_flux, Svec, comm, gmres_tol, gmres_iter);
      biest::VacuumField<Real,2,24,35>::EvalOffSurface(B_off_surf, Xtrg, Svec, J, comm);
    } else { // Compute Taylor state
      sctl::Vector<sctl::Vector<Real>> m, sigma;
      biest::TaylorState<Real,2,24,35>::Compute(B_bdry, tor_flux, pol_flux, beltrami_param, Svec, comm, gmres_tol, gmres_iter, 0.1, &m, &sigma);
      biest::TaylorState<Real,2,24,35>::EvalOffSurface(B_off_surf, Xtrg, Svec, tor_flux, pol_flux, beltrami_param, m, sigma, comm);
    }
    sctl::Profile::Toc();

    // Generate VTK visualization of the surface
    WriteVTK("B", Svec, B_bdry, comm);

    { // Print B
      printf("\n\nResult:\n");
      printf("          Bx            By            Bz\n");
      for (long i = 0; i < Ntrg; i++) {
        printf("% 12.8f  % 12.8f  % 12.8f\n", B_off_surf[i*3+0], B_off_surf[i*3+1], B_off_surf[i*3+2]);
      }
    }

    // Print profiling output
    sctl::Profile::print(&comm);
  }

  sctl::Comm::MPI_Finalize();
  return 0;
}

