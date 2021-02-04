#include <biest.hpp>
#include <sctl.hpp>

template <class Real> class CoilsField {
  static constexpr sctl::Integer COORD_DIM = 3;
  static constexpr Real vacuum_permeability = 1.25663706212e-6;

  public:

    void Init(std::string fname_coils, Real CoilUpsampleFactor = 2) {
      auto read_coils = [](sctl::Vector<sctl::Vector<Real>>& Xcoil, sctl::Vector<Real>& current, std::string fname) {
        std::ifstream file(fname);
        if (!file.good()) {
          std::cout << "Unable to open file for reading:" << fname << '\n';
          return;
        }

        std::string line;
        { // Read file header
          std::getline(file, line);
          std::getline(file, line);
          std::getline(file, line);
        }

        sctl::Vector<Real> current_;
        sctl::Vector<sctl::Vector<Real>> Xcoil_;
        { // Set current_, Xcoil_
          sctl::Vector<Real> VecX, VecY, VecZ, VecI;
          while (std::getline(file, line)) { // Read coils
            Real x,y,z,I;
            std::istringstream iss(line);
            if (iss >> x >> y >> z >> I) {
              if (!iss.eof()) {
                SCTL_ASSERT(I==0);
                SCTL_ASSERT(VecI.Dim());
                current_.PushBack(VecI[0]);

                sctl::Vector<Real> XYZ;
                for (auto a : VecX) XYZ.PushBack(a);
                for (auto a : VecY) XYZ.PushBack(a);
                for (auto a : VecZ) XYZ.PushBack(a);
                Xcoil_.PushBack(XYZ);

                VecX.ReInit(0);
                VecY.ReInit(0);
                VecZ.ReInit(0);
                VecI.ReInit(0);
                continue;
              }
              VecX.PushBack(x);
              VecY.PushBack(y);
              VecZ.PushBack(z);
              VecI.PushBack(I);
            }
          }
        }
        current = current_;
        Xcoil = Xcoil_;
      };
      read_coils(Xcoil, current, fname_coils);
      { // Set dIcoil
        sctl::Comm comm = sctl::Comm::Self();
        for (sctl::Long i = 0; i < current.Dim(); i++) {
          sctl::Vector<Real>& X = Xcoil[i];
          const sctl::Long N0 = X.Dim() / COORD_DIM;
          const sctl::Long N = N0 * CoilUpsampleFactor;
          { // Upsample X
            sctl::Vector<Real> X_;
            biest::SurfaceOp<Real>::Upsample(X,N0,1, X_,N,1);
            X = X_;
          }

          sctl::Vector<Real> dX(COORD_DIM*N);
          { // Set dX
            biest::SurfaceOp<Real> SurfOp(comm,N,1);
            sctl::Vector<Real> dX_;
            SurfOp.Grad2D(dX_, X);
            for (sctl::Long j = 0; j < N; j++) {
              for (sctl::Integer k = 0; k < COORD_DIM; k++) {
                dX[k*N+j] = dX_[(2*k+0)*N+j];
              }
            }
          }
          dIcoil.PushBack(dX*current[i]/N);
        }
      }
    }

    sctl::Vector<Real> EvalBcoil(const sctl::Vector<Real>& Xtrg) const {
      sctl::Vector<Real> Bcoil(Xtrg.Dim()); Bcoil = 0;
      for (sctl::Long i = 0; i < Xcoil.Dim(); i++) {
        sctl::Vector<Real> Bcoil_;
        biest::BiotSavart3D<Real>::FxU()(Xcoil[i], Xcoil[i], dIcoil[i], Xtrg, Bcoil_);
        Bcoil += Bcoil_;
      }
      Bcoil *= vacuum_permeability;
      return Bcoil;
    }

    void WriteVTK(std::string fname) const {
      biest::VTUData vtu_data;
      sctl::Long Ncoil = Xcoil.Dim();
      for (sctl::Long j = 0; j < Ncoil; j++) {
        sctl::Long CoilLength = Xcoil[j].Dim() / COORD_DIM;
        for (sctl::Long i = 0; i < CoilLength; i++) {
          vtu_data.connect.PushBack(vtu_data.coord.Dim()/COORD_DIM);
          for (sctl::Long k = 0; k < COORD_DIM; k++) {
            vtu_data.coord.PushBack(Xcoil[j][k*CoilLength+i]);
          }
        }
        vtu_data.offset.PushBack(vtu_data.connect.Dim());
        vtu_data.types.PushBack(4);
      }
      vtu_data.WriteVTK(fname);
    }

  private:

    sctl::Vector<Real> current;
    sctl::Vector<sctl::Vector<Real>> Xcoil, dIcoil;
};

template <class Real> class VirtualCasing {
  static constexpr sctl::Integer COORD_DIM = 3;

  public:

    void Init(std::string fname_Bfield, Real NtUpsampleFactor = 8, Real NpUpsampleFactor = 4) {
      auto read_surface_field = [](biest::Surface<Real>& S, sctl::Vector<Real>& B, std::string fname) {
        std::ifstream file(fname);
        if (!file.good()) {
          std::cout << "Unable to open file for reading:" << fname << '\n';
          return;
        }

        std::string line;
        { // Read file header
          std::getline(file, line);
        }
        sctl::Vector<Real> theta_vec, phi_vec;
        sctl::Vector<Real> X_vec, Y_vec, Z_vec;
        sctl::Vector<Real> Bx_vec, By_vec, Bz_vec;
        while (std::getline(file, line)) {
          Real theta,phi, x,y,z, Bx,By,Bz;
          std::istringstream iss(line);
          if (iss >> theta >> phi >> x >> y >> z >> Bx >> By >> Bz) {
            theta_vec.PushBack(theta);
            phi_vec.PushBack(phi);
            X_vec.PushBack(x);
            Y_vec.PushBack(y);
            Z_vec.PushBack(z);
            Bx_vec.PushBack(Bx);
            By_vec.PushBack(By);
            Bz_vec.PushBack(Bz);
          }
        }

        sctl::Long theta_run = 0, phi_run = 0;
        for (sctl::Long i = 1; (i<theta_vec.Dim()) && (theta_vec[i]==theta_vec[0]); i++) {
          theta_run = i;
        }
        for (sctl::Long i = 1; (i<phi_vec.Dim()) && (phi_vec[i]==phi_vec[0]); i++) {
          phi_run = i;
        }

        sctl::Long N = theta_vec.Dim();
        if (theta_run) { // Set S, B
          SCTL_ASSERT(!phi_run);
          sctl::Long Nt = theta_run;  // toroidal dimension
          sctl::Long Np = N/(Nt+1)-1; // poloidal dimension
          SCTL_ASSERT(N == (Nt+1)*(Np+1));
          S = biest::Surface<Real>(Nt, Np);
          if (B.Dim() != 3*Nt*Np) B.ReInit(3*Nt*Np);
          for (sctl::Long p = 0; p < Np; p++) {
            for (sctl::Long t = 0; t < Nt; t++) {
              S.Coord(t, p, 0) = X_vec[p*(Nt+1)+t];
              S.Coord(t, p, 1) = Y_vec[p*(Nt+1)+t];
              S.Coord(t, p, 2) = Z_vec[p*(Nt+1)+t];
              B[(0*Nt+t)*Np+p] = Bx_vec[p*(Nt+1)+t];
              B[(1*Nt+t)*Np+p] = By_vec[p*(Nt+1)+t];
              B[(2*Nt+t)*Np+p] = Bz_vec[p*(Nt+1)+t];
            }
          }
        } else {
          SCTL_ASSERT(phi_run);
          sctl::Long Np = phi_run;    // poloidal dimension
          sctl::Long Nt = N/(Np+1)-1; // toroidal dimension
          SCTL_ASSERT(N == (Nt+1)*(Np+1));
          S = biest::Surface<Real>(Nt, Np);
          if (B.Dim() != 3*Nt*Np) B.ReInit(3*Nt*Np);
          for (sctl::Long t = 0; t < Nt; t++) {
            for (sctl::Long p = 0; p < Np; p++) {
              S.Coord(t, p, 0) = X_vec[t*(Np+1)+p];
              S.Coord(t, p, 1) = Y_vec[t*(Np+1)+p];
              S.Coord(t, p, 2) = Z_vec[t*(Np+1)+p];
              B[(0*Nt+t)*Np+p] = Bx_vec[t*(Np+1)+p];
              B[(1*Nt+t)*Np+p] = By_vec[t*(Np+1)+p];
              B[(2*Nt+t)*Np+p] = Bz_vec[t*(Np+1)+p];
            }
          }
        }
      };
      read_surface_field(S, B, fname_Bfield);

      { // Upsample S, B
        sctl::Vector<Real> B_;
        biest::Surface<Real> S_(S.NTor()*NtUpsampleFactor, S.NPol()*NpUpsampleFactor);
        biest::SurfaceOp<Real>::Upsample(S.Coord(),S.NTor(),S.NPol(), S_.Coord(),S_.NTor(),S_.NPol());
        biest::SurfaceOp<Real>::Upsample(B,S.NTor(),S.NPol(), B_,S_.NTor(),S_.NPol());
        S = S_;
        B = B_;
      }
      if (0) { // generate artifical Bplasma from internal current loop (for testing)
        sctl::Long N = 10000;
        sctl::Vector<Real> X(COORD_DIM*N), dX(COORD_DIM*N);
        { // Set X, dX
          sctl::Long Nt = S.NTor(), Np = S.NPol();
          sctl::Vector<Real> coord(COORD_DIM*Nt); coord = 0;
          for (sctl::Long i = 0; i < Nt; i++) { // Set coord
            for (sctl::Long j = 0; j < Np; j++) {
              coord[0*Nt+i] += S.Coord(i,j,0)/Np;
              coord[1*Nt+i] += S.Coord(i,j,1)/Np;
              coord[2*Nt+i] += S.Coord(i,j,2)/Np;
            }
          }

          sctl::Vector<Real> dX_;
          biest::SurfaceOp<Real>::Upsample(coord,Nt,1, X,N,1);
          biest::SurfaceOp<Real> SurfOp(sctl::Comm::Self(),N,1);
          SurfOp.Grad2D(dX_, X);
          for (sctl::Long i = 0; i < N; i++) {
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
              dX[k*N+i] = dX_[(2*k+0)*N+i];
            }
          }
        }

        sctl::Vector<Real> Bplasma;
        biest::BiotSavart3D<Real>::FxU()(X, X, dX, S.Coord(), Bplasma);
        B = Bplasma;
      }
      { // Set sources (BdotN, J,  BdotN_dA, J_dA) for virtual-casing
        sctl::Vector<Real> area_elem;
        { // Set normal, area_elem
          sctl::Vector<Real> dX;
          biest::SurfaceOp<Real> SurfOp(sctl::Comm::Self(), S.NTor(), S.NPol());
          SurfOp.Grad2D(dX, S.Coord());
          SurfOp.SurfNormalAreaElem(&normal, &area_elem, dX, &S.Coord());
        }
        auto DotProd = [](sctl::Vector<Real>& AdotB, const sctl::Vector<Real>& A, const sctl::Vector<Real>& B) {
          sctl::Long N = A.Dim() / COORD_DIM;
          SCTL_ASSERT(A.Dim() == COORD_DIM * N);
          SCTL_ASSERT(B.Dim() == COORD_DIM * N);
          if (AdotB.Dim() != N) AdotB.ReInit(N);
          for (sctl::Long i = 0; i < N; i++) {
            Real AdotB_ = 0;
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
              AdotB_ += A[k*N+i] * B[k*N+i];
            }
            AdotB[i] = AdotB_;
          }
        };
        auto CrossProd = [](sctl::Vector<Real>& AcrossB, const sctl::Vector<Real>& A, const sctl::Vector<Real>& B) {
          sctl::Long N = A.Dim() / COORD_DIM;
          SCTL_ASSERT(A.Dim() == COORD_DIM * N);
          SCTL_ASSERT(B.Dim() == COORD_DIM * N);
          if (AcrossB.Dim() != COORD_DIM * N) AcrossB.ReInit(COORD_DIM * N);
          for (sctl::Long i = 0; i < N; i++) {
            sctl::StaticArray<Real,COORD_DIM> A_, B_, AcrossB_;
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
              A_[k] = A[k*N+i];
              B_[k] = B[k*N+i];
            }
            AcrossB_[0] = A_[1] * B_[2] - B_[1] * A_[2];
            AcrossB_[1] = A_[2] * B_[0] - B_[2] * A_[0];
            AcrossB_[2] = A_[0] * B_[1] - B_[0] * A_[1];
            for (sctl::Integer k = 0; k < COORD_DIM; k++) {
              AcrossB[k*N+i] = AcrossB_[k];
            }
          }
        };
        DotProd(BdotN, B, normal);
        CrossProd(J, normal, B);

        BdotN_dA.ReInit(area_elem.Dim());
        J_dA.ReInit(COORD_DIM*area_elem.Dim());
        for (sctl::Long i = 0; i < area_elem.Dim(); i++) {
          BdotN_dA[i] = BdotN[i] * area_elem[i];
          J_dA[0*area_elem.Dim()+i] = J[0*area_elem.Dim()+i] * area_elem[i];
          J_dA[1*area_elem.Dim()+i] = J[1*area_elem.Dim()+i] * area_elem[i];
          J_dA[2*area_elem.Dim()+i] = J[2*area_elem.Dim()+i] * area_elem[i];
        }
      }
    }

    const biest::Surface<Real>& GetSurface() const {
      return S;
    }

    const sctl::Vector<Real>& GetBOnSurface() const {
      return B;
    }

    template <sctl::Integer PDIM=30, sctl::Integer RDIM=60, sctl::Integer UPSAMPLE=2> sctl::Vector<Real> EvalBplasmaOnSurface() const {
      biest::BoundaryIntegralOp<Real, 3, 3, UPSAMPLE, PDIM, RDIM> BiotSavartFxU(sctl::Comm::Self());
      biest::BoundaryIntegralOp<Real, 1, 3, UPSAMPLE, PDIM, RDIM> LaplaceFxdU(sctl::Comm::Self());
      { // Setup BiotSavartFxU, LaplaceFxdU
        sctl::Vector<biest::Surface<Real>> Svec(1); Svec[0] = S;
        BiotSavartFxU.SetupSingular(Svec, biest::BiotSavart3D<Real>::FxU());
        LaplaceFxdU.SetupSingular(Svec, biest::Laplace3D<Real>::FxdU());
      }

      sctl::Vector<Real> B0, B1;
      LaplaceFxdU(B0, BdotN);
      BiotSavartFxU(B1, J);
      return 0.5 * B - B0 - B1;
    }

    sctl::Vector<Real> EvalBplasma(const sctl::Vector<Real>& Xtrg) const {
      sctl::Vector<Real> B0, B1;
      biest::Laplace3D<Real>::FxdU()(S.Coord(), normal, BdotN_dA, Xtrg, B0);
      biest::BiotSavart3D<Real>::FxU()(S.Coord(), normal, J_dA, Xtrg, B1);
      return (B0 + B1)*(-1);
    }

    static void test() {
      CoilsField<Real> coils;
      coils.Init("ellipse.coils");

      VirtualCasing virtual_casing;
      virtual_casing.Init("BetaScan_run_20.Bxyz.txt");

      auto B = virtual_casing.GetBOnSurface();
      auto Bplasma = virtual_casing.EvalBplasmaOnSurface();
      auto Bcoil = coils.EvalBcoil(virtual_casing.GetSurface().Coord());

      auto inf_norm = [](const sctl::Vector<Real>& X) {
        Real norm = 0;
        for (const auto& a : X) norm = std::max<Real>(norm, fabs(a));
        return norm;
      };
      std::cout<<"err = "<<inf_norm(B-Bplasma-Bcoil)<<'\n';

      if (0) { // Eval Bplasma_offsurf
        biest::Surface<Real> Strg;
        { // Set Strg
          Strg = virtual_casing.GetSurface();
          sctl::Vector<Real> normal, dX;
          biest::SurfaceOp<Real> SurfOp(sctl::Comm::Self(), Strg.NTor(), Strg.NPol());
          SurfOp.Grad2D(dX, Strg.Coord());
          SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &Strg.Coord());
          Strg.Coord() += normal*0.05;
        }
        auto Bplasma_offsurf = virtual_casing.EvalBplasma(Strg.Coord());
        biest::WriteVTK("Bplasma-offsurf", Strg, Bplasma_offsurf);
      }
      biest::WriteVTK("B", virtual_casing.GetSurface(), B);
      biest::WriteVTK("Bcoil", virtual_casing.GetSurface(), Bcoil);
      biest::WriteVTK("Bplasma", virtual_casing.GetSurface(), Bplasma);
      coils.WriteVTK("coils");
    }

  private:

    sctl::Vector<Real> B;
    biest::Surface<Real> S;

    sctl::Vector<Real> normal;
    sctl::Vector<Real> BdotN, J;
    sctl::Vector<Real> BdotN_dA, J_dA;
};

int main(int argc, char** argv) {
  static constexpr sctl::Integer COORD_DIM = 3;
  using Real = double;

  // Read coil data
  CoilsField<Real> coils;
  coils.Init("ellipse.coils", 2); // upsample my factor 2
  coils.WriteVTK("coils"); // Write VTK visualization of coils

  // Read plasma surface and B-field data
  VirtualCasing<Real> virtual_casing;
  virtual_casing.Init("BetaScan_run_20.Bxyz.txt", 4, 2); // upsample grid by factor {4,2}
  biest::WriteVTK("B", virtual_casing.GetSurface(), virtual_casing.GetBOnSurface()); // Write VTK visualization of B
  biest::WriteVTK("Bplasma", virtual_casing.GetSurface(), virtual_casing.EvalBplasma(virtual_casing.GetSurface().Coord())); // Write VTK visualization of Bplasma

  // Funcation to evaluate B field in region outside the plasma
  // Xtrg (input): location to evaluation points in the format {x1,x2,...,xn, y1,y2,...,yn, z1,z2,...,zn}
  // B (output): B-field in the format {Bx1,Bx2,...,Bxn, By1,By2,...,Byn, Bz1,Bz2,...,Bzn}
  std::function<void(sctl::Vector<Real>*, const sctl::Vector<Real>&)> compute_B =[&virtual_casing,&coils](sctl::Vector<Real>* B, const sctl::Vector<Real>& Xtrg) {
    auto Bcoil = coils.EvalBcoil(Xtrg);
    auto Bplasma = virtual_casing.EvalBplasma(Xtrg);
    (*B) = Bcoil + Bplasma;
  };

  // Initial points for field tracing
  sctl::Vector<Real> Xpath, X(COORD_DIM*2); // {x1,x2, y1,y2, z1,z2}
  X[0] =  8.6; X[2] = 0; X[4] = 0;
  X[1] = 11.4; X[3] = 0; X[5] = 0;

  // Trace field lines and print paths
  Real t = 0.0, dt = 1.0e-1, tol = 1e-8;
  constexpr sctl::Integer TimeStepOrder = 6;
  sctl::SDC<Real, TimeStepOrder> ode_solver;
  std::cout<<"  x1          x2        y1         y2         z1          z2\n";
  while (t < 100.0) { // Adaptive time-stepping loop
    sctl::Vector<Real> X_;
    Real error_interp, error_picard;
    ode_solver(&X_, dt, X, compute_B, TimeStepOrder, tol*0.1, &error_interp, &error_picard);
    Real error = std::max(error_interp, error_picard);
    if (error < tol) { // Accept solution
      std::cout<<X; // print position
      for (auto a : X) {
        Xpath.PushBack(a);
      }
      X = X_;
      t = t + dt;
      if (error < tol*0.1) { // increase step-size
        dt = dt * 1.2;
      }
    } else { // reduce step-size
      dt = dt * 0.8;
    }
  }

  { // Write VTK output of field-lines
    biest::VTUData vtu_data;
    sctl::Long Npt = X.Dim() / COORD_DIM;
    sctl::Long Nsteps = Xpath.Dim() / X.Dim();
    for (sctl::Long j = 0; j < Npt; j++) {
      for (sctl::Long i = 0; i < Nsteps; i++) {
        for (sctl::Long k = 0; k < COORD_DIM; k++) {
          vtu_data.coord.PushBack(Xpath[(i*COORD_DIM+k)*Npt+j]);
        }
      }
    }
    for (sctl::Long i = 0; i < Npt*Nsteps; i++) {
      vtu_data.connect.PushBack(i);
    }
    for (sctl::Long i = 0; i < Npt; i++) {
      vtu_data.offset.PushBack((i+1)*Nsteps);
      vtu_data.types.PushBack(4);
    }
    vtu_data.WriteVTK("field_lines");
  }

  return 0;
}

