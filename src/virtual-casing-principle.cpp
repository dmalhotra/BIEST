#include <biest.hpp>
#include <sctl.hpp>
typedef double Real;

// Generalized virtual-casing principle (without solving a Laplace Neumann problem)
template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void GeneralVirtualCasing(sctl::Vector<Real>& Bext, const biest::Surface<Real>& S, const sctl::Vector<Real>& B) {
  constexpr sctl::Integer COORD_DIM = 3;
  sctl::Comm comm = sctl::Comm::Self();
  bool prof_state = sctl::Profile::Enable(true);
  sctl::Profile::Tic("Total", &comm);

  sctl::Long Nt = S.NTor();
  sctl::Long Np = S.NPol();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = S;

  sctl::Profile::Tic("Setup", &comm);
  biest::BoundaryIntegralOp<Real, 3, 3, UPSAMPLE, PDIM, RDIM> BiotSavartFxU(comm);
  BiotSavartFxU.SetupSingular(Svec, biest::BiotSavart3D<Real>::FxU());

  biest::BoundaryIntegralOp<Real, 1, 3, UPSAMPLE, PDIM, RDIM> LaplaceFxdU(comm);
  LaplaceFxdU.SetupSingular(Svec, biest::Laplace3D<Real>::FxdU());

  sctl::Vector<Real> dX, normal;
  biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
  { // set dX, normal
    SurfOp.Grad2D(dX, S.Coord());
    SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &S.Coord());
  }
  sctl::Profile::Toc();

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

  sctl::Profile::Tic("B-ext", &comm);
  sctl::Vector<Real> BdotN, J, Bext_;
  DotProd(BdotN, B, normal);
  CrossProd(J, normal, B);
  LaplaceFxdU(Bext_, BdotN);
  BiotSavartFxU(Bext, J);
  Bext += Bext_ + 0.5 * B;
  sctl::Profile::Toc();

  sctl::Profile::Toc();
  sctl::Profile::print(&comm);
  sctl::Profile::Enable(prof_state);
}

// Virtual-casing principle for the case when B.n=0 (and constructing (B+Bvac).n = 0 when B.n!=0)
template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void VirtualCasing(sctl::Vector<Real>& Bext, const biest::Surface<Real>& S, const sctl::Vector<Real>& B, Real gmres_tol = 1e-12) {
  constexpr sctl::Integer COORD_DIM = 3;
  sctl::Comm comm = sctl::Comm::Self();
  bool prof_state = sctl::Profile::Enable(true);
  sctl::Profile::Tic("Total", &comm);

  sctl::Long Nt = S.NTor();
  sctl::Long Np = S.NPol();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = S;

  sctl::Profile::Tic("Setup", &comm);
  biest::BoundaryIntegralOp<Real, 3, 3, UPSAMPLE, PDIM, RDIM> BiotSavartFxU(comm);
  BiotSavartFxU.SetupSingular(Svec, biest::BiotSavart3D<Real>::FxU());

  biest::BoundaryIntegralOp<Real, 1, 1, UPSAMPLE, PDIM, RDIM> LaplaceFxU(comm);
  LaplaceFxU.SetupSingular(Svec, biest::Laplace3D<Real>::FxU());

  biest::BoundaryIntegralOp<Real, 1, 1, UPSAMPLE, PDIM, RDIM> LaplaceDxU(comm);
  LaplaceDxU.SetupSingular(Svec, biest::Laplace3D<Real>::DxU());

  sctl::Vector<Real> dX, normal;
  biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
  { // set dX, normal
    SurfOp.Grad2D(dX, S.Coord());
    SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &S.Coord());
  }
  sctl::Profile::Toc();

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

  sctl::Profile::Tic("B-aux", &comm);
  sctl::Vector<Real> Baux(COORD_DIM*Nt*Np);
  if (gmres_tol < 1.0) { // Compute Baux
    sctl::Vector<Real> BdotN, rhs(Nt*Np);
    rhs = 0;
    DotProd(BdotN, B, normal);
    LaplaceFxU(rhs, BdotN*(-1));

    sctl::Vector<Real> phi(Nt*Np);
    auto BIEOp = [&LaplaceDxU](sctl::Vector<Real>* u, const sctl::Vector<Real>& phi) { // u = (1 + D) \phi
      (*u) = 0; //phi * 0.5;
      LaplaceDxU(*u, phi*(1));
      (*u) += phi * 0.5;
    };
    sctl::GMRES<Real> solve(comm,1);
    solve(&phi, BIEOp, rhs, gmres_tol, 100);

    { // Set Baux
      sctl::Long N = phi.Dim();
      SurfOp.SurfGrad(Baux,dX,phi);
      SCTL_ASSERT(normal.Dim() == COORD_DIM*N);
      SCTL_ASSERT(Baux.Dim() == COORD_DIM*N);
      for (sctl::Long i = 0; i < N; i++) {
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          Baux[k*N+i] += BdotN[i] * normal[k*N+i];
        }
      }
    }
  } else {
    Baux = 0;
  }
  sctl::Profile::Toc();

  sctl::Profile::Tic("B-ext", &comm);
  sctl::Vector<Real> J, JxN;
  CrossProd(J, normal, B-Baux);
  CrossProd(JxN, J, normal);
  BiotSavartFxU(Bext, J);
  Bext += 0.5 * JxN;
  sctl::Profile::Toc();

  sctl::Profile::Toc();
  sctl::Profile::print(&comm);
  sctl::Profile::Enable(prof_state);
}

// Virtual-casing principle (using vector potential)
template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void VirtualCasingNormalComponent(sctl::Vector<Real>& Bext, const biest::Surface<Real>& S, const sctl::Vector<Real>& B, Real gmres_tol = 1e-12) {
  constexpr sctl::Integer COORD_DIM = 3;
  sctl::Comm comm = sctl::Comm::Self();
  bool prof_state = sctl::Profile::Enable(true);
  sctl::Profile::Tic("Total", &comm);

  sctl::Long Nt = S.NTor();
  sctl::Long Np = S.NPol();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = S;

  sctl::Profile::Tic("Setup", &comm);
  biest::BoundaryIntegralOp<Real, 1, 1, UPSAMPLE, PDIM, RDIM> LaplaceFxU(comm);
  LaplaceFxU.SetupSingular(Svec, biest::Laplace3D<Real>::FxU());

  biest::BoundaryIntegralOp<Real, 1, 1, UPSAMPLE, PDIM, RDIM> LaplaceDxU(comm);
  LaplaceDxU.SetupSingular(Svec, biest::Laplace3D<Real>::DxU());

  sctl::Vector<Real> dX, normal;
  biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
  { // Set dX, normal
    SurfOp.Grad2D(dX, S.Coord());
    SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &S.Coord());
  }
  sctl::Profile::Toc();

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

  sctl::Profile::Tic("B-aux", &comm);
  sctl::Vector<Real> Baux(COORD_DIM*Nt*Np);
  if (gmres_tol < 1.0) { // Compute Baux
    sctl::Vector<Real> BdotN, rhs(Nt*Np);
    rhs = 0;
    DotProd(BdotN, B, normal);
    LaplaceFxU(rhs, BdotN*(-1));

    sctl::Vector<Real> phi(Nt*Np);
    auto BIEOp = [&LaplaceDxU](sctl::Vector<Real>* u, const sctl::Vector<Real>& phi) { // u = (1 + D) \phi
      (*u) = 0; //phi * 0.5;
      LaplaceDxU(*u, phi*(1));
      (*u) += phi * 0.5;
    };
    sctl::GMRES<Real> solve(comm,1);
    solve(&phi, BIEOp, rhs, gmres_tol, 100);

    { // Set Baux
      sctl::Long N = phi.Dim();
      SurfOp.SurfGrad(Baux,dX,phi);
      SCTL_ASSERT(normal.Dim() == COORD_DIM*N);
      SCTL_ASSERT(Baux.Dim() == COORD_DIM*N);
      for (sctl::Long i = 0; i < N; i++) {
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          Baux[k*N+i] += BdotN[i] * normal[k*N+i];
        }
      }
    }
  } else {
    Baux = 0;
  }
  sctl::Profile::Toc();

  sctl::Profile::Tic("B-ext", &comm);
  sctl::Vector<Real> J, Aext;
  CrossProd(J, normal, B-Baux);
  LaplaceFxU(Aext, J);
  SurfOp.SurfCurl(Bext,dX,normal,Aext);
  sctl::Profile::Toc();

  sctl::Profile::Toc();
  sctl::Profile::print(&comm);
  sctl::Profile::Enable(prof_state);
}

template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void test_TaylorState(sctl::Long Nt, sctl::Long Np, sctl::Long Nt0, sctl::Long Np0, const sctl::Vector<Real>& B0, const sctl::Vector<Real>& B0ext) {
  constexpr sctl::Integer COORD_DIM = 3;
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

  sctl::Comm comm = sctl::Comm::Self();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt,Np, biest::SurfType::W7X_);

  sctl::Vector<Real> Bext, Bint, B;
  { // Set Bext, Bint, B
    biest::SurfaceOp<Real>::Upsample(B0,Nt0,Np0, B,Nt,Np);
    biest::SurfaceOp<Real>::Upsample(B0ext,Nt0,Np0, Bext,Nt,Np);
    Bint = B - Bext;
  }

  sctl::Vector<Real> normal;
  { // set normal
    sctl::Vector<Real> dX;
    biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
    const auto& X = Svec[0].Coord();
    SurfOp.Grad2D(dX, X);
    SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &X);
  }

  sctl::Vector<Real> Berr;
  if (0) { // Set Berr
    sctl::Vector<Real> Bext_;
    VirtualCasing<Real,UPSAMPLE,PDIM,RDIM>(Bext_, Svec[0], B, 1);
    auto Berr_ = Bext - Bext_;
    DotProd(Berr, Berr_, normal);
  } else {
    sctl::Vector<Real> NdotBext_, NdotBext;
    VirtualCasingNormalComponent<Real,UPSAMPLE,PDIM,RDIM>(NdotBext_, Svec[0], B, 1);
    DotProd(NdotBext, Bext, normal);
    Berr = NdotBext - NdotBext_;
  }

  Real max_err = 0, max_val = 0;
  for (const auto& x:B   ) max_val = std::max<Real>(max_val,fabs(x));
  for (const auto& x:Berr) max_err = std::max<Real>(max_err,fabs(x));
  std::cout<<"Maximum relative error: "<<max_err/max_val<<'\n';
}

template <class Real, sctl::Integer UPSAMPLE, sctl::Integer PDIM, sctl::Integer RDIM> void test_CurrentLoops(sctl::Long Nt, sctl::Long Np, Real gmres_tol = 1e-12) {
  constexpr sctl::Integer COORD_DIM = 3;
  auto WriteVTK_ = [](std::string fname, const sctl::Vector<sctl::Vector<Real>>& coords, const sctl::Vector<sctl::Vector<Real>>& values) {
    biest::VTKData data;
    typedef biest::VTKData::VTKReal VTKReal;
    auto& point_coord =data.point_coord ;
    auto& point_value =data.point_value ;
    auto& line_connect=data.line_connect;
    auto& line_offset =data.line_offset ;
    constexpr sctl::Integer COORD_DIM = biest::VTKData::COORD_DIM;

    SCTL_ASSERT(coords.Dim() == values.Dim());
    for (sctl::Long j = 0; j < coords.Dim(); j++) { // set point_coord, line_connect
      const auto& coord = coords[j];
      const auto& value = values[j];
      sctl::Long N = coord.Dim() / COORD_DIM;
      sctl::Long dof = value.Dim() / N;
      SCTL_ASSERT(value.Dim() == dof * N);
      for (sctl::Long i = 0; i < N; i++) {
        line_connect.push_back(point_coord.size()/COORD_DIM);
        point_coord.push_back((VTKReal)coord[0*N+i]);
        point_coord.push_back((VTKReal)coord[1*N+i]);
        point_coord.push_back((VTKReal)coord[2*N+i]);
        for (sctl::Long k = 0; k < dof; k++) {
          point_value.push_back((VTKReal)value[k*N+i]);
        }
      }
      line_offset.push_back(line_connect.size());
    }
    data.WriteVTK(fname.c_str(), sctl::Comm::Self());
  };
  auto eval_BiotSavart = [](const sctl::Vector<Real>& Xt, const sctl::Vector<sctl::Vector<Real>>& source, const sctl::Vector<sctl::Vector<Real>>& density) {
    const auto& kernel = biest::BiotSavart3D<Real>::FxU();
    sctl::Long Nt = Xt.Dim() / COORD_DIM;
    SCTL_ASSERT(Xt.Dim() == COORD_DIM * Nt);
    SCTL_ASSERT(source.Dim() == density.Dim());

    sctl::Vector<Real> B(COORD_DIM*Nt);
    B = 0;
    for (sctl::Long i = 0; i < source.Dim(); i++) {
      const auto& Xs = source[i];
      const auto& Fs = density[i];
      sctl::Long Ns = Xs.Dim() / COORD_DIM;
      SCTL_ASSERT(Xs.Dim() == COORD_DIM * Ns);
      SCTL_ASSERT(Fs.Dim() == COORD_DIM * Ns);
      sctl::Vector<Real> SrcNormal(COORD_DIM*Ns);
      kernel(Xs,SrcNormal,Fs, Xt,B);
    }
    return B;
  };
  auto add_source_loop = [](sctl::Vector<sctl::Vector<Real>>& source, sctl::Vector<sctl::Vector<Real>>& density, std::initializer_list<Real> coord, std::initializer_list<Real> normal, Real radius) {
    auto cross_norm = [](const sctl::Vector<Real>& A, const sctl::Vector<Real>& B) {
      sctl::Vector<Real> C(COORD_DIM);
      C[0] = A[1]*B[2] - B[1]*A[2];
      C[1] = A[2]*B[0] - B[2]*A[0];
      C[2] = A[0]*B[1] - B[0]*A[1];
      Real r = sctl::sqrt<Real>(C[0]*C[0]+C[1]*C[1]+C[2]*C[2]);
      return C*(1/r);
    };
    sctl::Vector<Real> coord_(COORD_DIM), normal_(COORD_DIM), e0(COORD_DIM), e1(COORD_DIM);
    normal_[0] = normal.begin()[0];
    normal_[1] = normal.begin()[1];
    normal_[2] = normal.begin()[2];
    coord_[0] = coord.begin()[0];
    coord_[1] = coord.begin()[1];
    coord_[2] = coord.begin()[2];
    e0[0] = drand48();
    e0[1] = drand48();
    e0[2] = drand48();
    e0 = cross_norm(e0,normal_)*radius;
    e1 = cross_norm(e0,normal_)*radius;

    sctl::Long N = 10000;
    sctl::Vector<Real> X(COORD_DIM * N);
    sctl::Vector<Real> dX(COORD_DIM * N);
    for (sctl::Long i = 0; i < N; i++) {
      Real t = 2*sctl::const_pi<Real>()*i/N;
      sctl::Vector<Real> r = coord_ + e0*sctl::sin<Real>(t) + e1*sctl::cos<Real>(t);
      sctl::Vector<Real> dr = e0*sctl::cos<Real>(t) - e1*sctl::sin<Real>(t);
      X[0*N+i] = r[0];
      X[1*N+i] = r[1];
      X[2*N+i] = r[2];
      dX[0*N+i] = dr[0];
      dX[1*N+i] = dr[1];
      dX[2*N+i] = dr[2];
    }
    source.PushBack(X);
    density.PushBack(dX);
  };
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
  SCTL_UNUSED(WriteVTK_);

  sctl::Comm comm = sctl::Comm::Self();
  sctl::Vector<biest::Surface<Real>> Svec(1);
  Svec[0] = biest::Surface<Real>(Nt,Np, biest::SurfType::W7X_);

  sctl::Vector<sctl::Vector<Real>> source0, density0;
  sctl::Vector<sctl::Vector<Real>> source1, density1;
  { // Set inside sources (source0, density0)
    sctl::Long N = 10000;
    sctl::Vector<Real> X(COORD_DIM*N), dX(COORD_DIM*N);
    { // Set X, dX
      sctl::Long Nt = 100, Np = 100;
      sctl::Vector<Real> coord(COORD_DIM*Nt);
      { // Set coord
        auto S = biest::Surface<Real>(Nt,Np, biest::SurfType::W7X_);
        sctl::Vector<Real> normal, dX;
        biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
        SurfOp.Grad2D(dX, S.Coord());
        SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &S.Coord());
        S.Coord() += -2.17*normal;

        coord = 0;
        for (sctl::Long t = 0; t < Nt; t++) {
          for (sctl::Long p = 0; p < Np; p++) {
            coord[0*Nt+t] += S.Coord()[(0*Nt+t)*Np+p]/Np;
            coord[1*Nt+t] += S.Coord()[(1*Nt+t)*Np+p]/Np;
            coord[2*Nt+t] += S.Coord()[(2*Nt+t)*Np+p]/Np;
          }
        }
      }

      sctl::Vector<Real> dX_;
      biest::SurfaceOp<Real>::Upsample(coord,Nt,1, X,N,1);
      biest::SurfaceOp<Real> SurfOp(comm,N,1);
      SurfOp.Grad2D(dX_, X);
      for (sctl::Long i = 0; i < N; i++) {
        for (sctl::Integer k = 0; k < COORD_DIM; k++) {
          dX[k*N+i] = dX_[(2*k+0)*N+i];
        }
      }
    }
    source0.PushBack(X);
    density0.PushBack(dX*0.05);
  }
  { // Set outside sources (source1, density1)
    add_source_loop(source1, density1, {6,0,0}, {0,1,0}, 2);
  }

  sctl::Vector<Real> Bext, Bint, B;
  { // Set Bext, Bint, B
    Bint = eval_BiotSavart(Svec[0].Coord(), source0, density0);
    Bext = eval_BiotSavart(Svec[0].Coord(), source1, density1);
    B = Bint + Bext;
  }

  sctl::Vector<Real> normal;
  { // set normal
    sctl::Vector<Real> dX;
    biest::SurfaceOp<Real> SurfOp(comm, Nt, Np);
    const auto& X = Svec[0].Coord();
    SurfOp.Grad2D(dX, X);
    SurfOp.SurfNormalAreaElem(&normal, nullptr, dX, &X);
  }

  sctl::Vector<Real> Berr;
  if (0) { // Set Berr
    sctl::Vector<Real> Bext_;
    VirtualCasing<Real,UPSAMPLE,PDIM,RDIM>(Bext_, Svec[0], B, gmres_tol);
    auto Berr_ = Bext - Bext_;
    DotProd(Berr, Berr_, normal);
  } else {
    sctl::Vector<Real> NdotBext_, NdotBext;
    VirtualCasingNormalComponent<Real,UPSAMPLE,PDIM,RDIM>(NdotBext_, Svec[0], B, gmres_tol);
    DotProd(NdotBext, Bext, normal);
    Berr = NdotBext - NdotBext_;
  }

  Real max_err = 0, max_val = 0;
  for (const auto& x:B   ) max_val = std::max<Real>(max_val,fabs(x));
  for (const auto& x:Berr) max_err = std::max<Real>(max_err,fabs(x));
  std::cout<<"Maximum relative error: "<<max_err/max_val<<'\n';
}

int main(int argc, char** argv) {
#ifdef SCTL_HAVE_PETSC
  PetscInitialize(&argc, &argv, NULL, NULL);
#elif defined(SCTL_HAVE_MPI)
  MPI_Init(&argc, &argv);
#endif

  if (1) {
    long Nt0 = 70*32, Np0 = 14*32; // reference solution resolution
    sctl::Vector<Real> B, Bext;
    if (0) {
      sctl::Comm comm = sctl::Comm::Self();

      Real gmres_tol = 1e-12;
      sctl::Long Nt = 70*29, Np = 14*29;
      sctl::Vector<biest::Surface<Real>> Svec(1);
      Svec[0] = biest::Surface<Real>(Nt,Np, biest::SurfType::W7X_);

      sctl::Vector<Real> B_, Bext_;
      { // set B_, Bext_
        Real tor_flux=1, pol_flux=0, lambda = 1.0;
        bool prof_state = sctl::Profile::Enable(true);
        biest::TaylorState<Real,1,40,75>::Compute(B_, tor_flux, pol_flux, lambda, Svec, comm, gmres_tol, 100);
        VirtualCasing<Real,1,40,75>(Bext_, Svec[0], B_, 10);
        sctl::Profile::Enable(prof_state);
      }

      biest::SurfaceOp<Real>::Upsample(B_,Nt,Np, B,Nt0,Np0);
      biest::SurfaceOp<Real>::Upsample(Bext_,Nt,Np, Bext,Nt0,Np0);

      B.Write("tmp-B.data");
      Bext.Write("tmp-Bext.data");
    } else {
      B.Read("tmp-B.data");
      Bext.Read("tmp-Bext.data");
    }

    test_TaylorState<Real,1, 6,12>(70* 1, 14* 1, Nt0, Np0, B, Bext); // 0.0249058    0.0155974
    test_TaylorState<Real,1, 6,12>(70* 2, 14* 2, Nt0, Np0, B, Bext); // 0.0011495    0.0161189
    test_TaylorState<Real,1, 6,15>(70* 3, 14* 3, Nt0, Np0, B, Bext); // 0.000537568  0.00415999
    test_TaylorState<Real,1, 8,20>(70* 4, 14* 4, Nt0, Np0, B, Bext); // 0.000146963  0.0018603
    test_TaylorState<Real,1,12,20>(70* 5, 14* 5, Nt0, Np0, B, Bext); // 2.73052e-05  0.00120471
    test_TaylorState<Real,1,12,25>(70* 6, 14* 6, Nt0, Np0, B, Bext); // 4.02025e-06  0.000225547
    test_TaylorState<Real,1,12,25>(70* 7, 14* 7, Nt0, Np0, B, Bext); // 3.05336e-06  0.000222321
    test_TaylorState<Real,1,15,30>(70* 8, 14* 8, Nt0, Np0, B, Bext); // 1.10189e-06  4.49451e-05
    test_TaylorState<Real,1,15,35>(70* 9, 14* 9, Nt0, Np0, B, Bext); // 6.13794e-07  6.62496e-06
    test_TaylorState<Real,1,20,35>(70*10, 14*10, Nt0, Np0, B, Bext); // 9.90141e-08  7.18611e-06
    test_TaylorState<Real,1,20,35>(70*11, 14*11, Nt0, Np0, B, Bext); // 6.71431e-08  7.12210e-06
    test_TaylorState<Real,1,20,40>(70*12, 14*12, Nt0, Np0, B, Bext); // 5.86476e-08  2.07163e-06
    test_TaylorState<Real,1,20,40>(70*13, 14*13, Nt0, Np0, B, Bext); // 5.46263e-08  2.18712e-06
    test_TaylorState<Real,1,25,45>(70*14, 14*14, Nt0, Np0, B, Bext); // 3.33689e-09  2.35615e-07
    test_TaylorState<Real,1,25,45>(70*15, 14*15, Nt0, Np0, B, Bext); // 2.05500e-09  2.27660e-07
    test_TaylorState<Real,1,25,45>(70*16, 14*16, Nt0, Np0, B, Bext); // 1.55405e-09  2.23890e-07
    test_TaylorState<Real,1,25,50>(70*17, 14*17, Nt0, Np0, B, Bext); // 1.02093e-09  5.12808e-08
    test_TaylorState<Real,1,25,50>(70*18, 14*18, Nt0, Np0, B, Bext); // 8.99951e-10  5.26030e-08
    test_TaylorState<Real,1,25,55>(70*19, 14*19, Nt0, Np0, B, Bext); // 8.75942e-10  1.64075e-08
    test_TaylorState<Real,1,30,55>(70*20, 14*20, Nt0, Np0, B, Bext); // 2.76843e-10  4.70224e-09
    test_TaylorState<Real,1,30,55>(70*21, 14*21, Nt0, Np0, B, Bext); // 2.61857e-10  4.50111e-09
    test_TaylorState<Real,1,30,55>(70*22, 14*22, Nt0, Np0, B, Bext); // 2.68988e-10  4.41811e-09
    test_TaylorState<Real,1,30,60>(70*23, 14*23, Nt0, Np0, B, Bext); // 2.75871e-10  4.04524e-09
    test_TaylorState<Real,1,35,60>(70*24, 14*24, Nt0, Np0, B, Bext); // 2.32377e-10  1.60351e-09
    test_TaylorState<Real,1,35,60>(70*25, 14*25, Nt0, Np0, B, Bext); // 2.37569e-10  1.63277e-09
    test_TaylorState<Real,1,35,65>(70*26, 14*26, Nt0, Np0, B, Bext); // 2.37988e-10  5.31487e-10
    test_TaylorState<Real,1,40,65>(70*27, 14*27, Nt0, Np0, B, Bext); // 2.29965e-10  2.07195e-10
    test_TaylorState<Real,1,40,65>(70*28, 14*28, Nt0, Np0, B, Bext); // 2.34845e-10  2.01065e-10
    test_TaylorState<Real,1,45,70>(70*29, 14*29, Nt0, Np0, B, Bext); // 2.33638e-10  1.41012e-10
  }
  if (1) {
    test_CurrentLoops<Real,1, 6,12>(70* 1, 14* 1, 1e-00*1e-1); // 0.101738     0.124209
    test_CurrentLoops<Real,1, 6,12>(70* 2, 14* 2, 1e-01*1e-1); // 0.0250247    0.033822
    test_CurrentLoops<Real,1, 6,15>(70* 3, 14* 3, 3e-02*1e-1); // 0.0147037    0.0223968
    test_CurrentLoops<Real,1, 8,20>(70* 4, 14* 4, 1e-02*1e-1); // 0.00519251   0.00744083
    test_CurrentLoops<Real,1,12,20>(70* 5, 14* 5, 3e-03*1e-1); // 0.000949045  0.00220551
    test_CurrentLoops<Real,1,12,25>(70* 6, 14* 6, 1e-03*1e-1); // 0.000283139  0.000924378
    test_CurrentLoops<Real,1,12,25>(70* 7, 14* 7, 3e-04*1e-1); // 0.000193408  0.000617451
    test_CurrentLoops<Real,1,15,30>(70* 8, 14* 8, 3e-04*1e-1); // 5.17719e-05  0.000104611
    test_CurrentLoops<Real,1,15,35>(70* 9, 14* 9, 1e-04*1e-1); // 3.07832e-05  5.38333e-05
    test_CurrentLoops<Real,1,20,35>(70*10, 14*10, 3e-05*1e-1); // 9.74924e-06  2.90806e-05
    test_CurrentLoops<Real,1,20,35>(70*11, 14*11, 1e-05*1e-1); // 5.61443e-06  1.95736e-05
    test_CurrentLoops<Real,1,20,40>(70*12, 14*12, 3e-06*1e-1); // 3.82595e-06  5.22990e-06
    test_CurrentLoops<Real,1,20,40>(70*13, 14*13, 3e-06*1e-1); // 3.98806e-06  5.20004e-06
    test_CurrentLoops<Real,1,25,45>(70*14, 14*14, 1e-06*1e-1); // 5.25503e-07  1.14772e-06
    test_CurrentLoops<Real,1,25,45>(70*15, 14*15, 1e-06*1e-1); // 2.89939e-07  7.45186e-07
    test_CurrentLoops<Real,1,25,45>(70*16, 14*16, 1e-07*1e-1); // 1.80578e-07  5.98980e-07
    test_CurrentLoops<Real,1,25,50>(70*17, 14*17, 1e-07*1e-1); // 1.06620e-07  1.33825e-07
    test_CurrentLoops<Real,1,25,50>(70*18, 14*18, 1e-07*1e-1); // 7.30727e-08  9.60245e-08
    test_CurrentLoops<Real,1,25,55>(70*19, 14*19, 3e-08*1e-1); // 6.39203e-08  8.78734e-08
    test_CurrentLoops<Real,1,30,55>(70*20, 14*20, 3e-08*1e-1); // 2.06118e-08  3.73103e-08
    test_CurrentLoops<Real,1,30,55>(70*21, 14*21, 1e-08*1e-1); // 1.36523e-08  2.54951e-08
    test_CurrentLoops<Real,1,30,55>(70*22, 14*22, 1e-08*1e-1); // 9.91856e-09  1.92745e-08
    test_CurrentLoops<Real,1,30,60>(70*23, 14*23, 1e-08*1e-1); // 1.05747e-08  1.56373e-08
    test_CurrentLoops<Real,1,35,60>(70*24, 14*24, 3e-09*1e-1); // 3.12361e-09  4.35193e-09
    test_CurrentLoops<Real,1,35,60>(70*25, 14*25, 3e-09*1e-1); // 2.60186e-09  3.85617e-09
    test_CurrentLoops<Real,1,35,65>(70*26, 14*26, 1e-09*1e-1); // 2.29237e-09  3.69802e-09
    test_CurrentLoops<Real,1,40,65>(70*27, 14*27, 1e-09*1e-1); // 9.83529e-10  1.34430e-09
    test_CurrentLoops<Real,1,40,65>(70*28, 14*28, 1e-09*1e-1); // 8.28677e-10  1.21468e-09
    test_CurrentLoops<Real,1,45,70>(70*29, 14*29, 1e-10*1e-1); // 4.50924e-10  5.51574e-10
  }

#ifdef SCTL_HAVE_PETSC
  PetscFinalize();
#elif defined(SCTL_HAVE_MPI)
  MPI_Finalize();
#endif
  return 0;
}
