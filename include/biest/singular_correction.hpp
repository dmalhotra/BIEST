#ifndef _SINGULAR_CORRECTION_HPP_
#define _SINGULAR_CORRECTION_HPP_

#include <biest/kernel.hpp>
#include <biest/mat.hpp>
#include <sctl.hpp>

namespace biest {

template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> class SingularCorrection {
    static constexpr sctl::Integer COORD_DIM = 3;
    static constexpr sctl::Integer PATCH_DIM = 2 * PATCH_DIM0 + 1; // <= 55 (odd)
    //static constexpr sctl::Integer RAD_DIM = 18; // <= 18
    static constexpr sctl::Integer ANG_DIM = 2 * RAD_DIM; // <= 39
    static constexpr sctl::Integer INTERP_ORDER = 12;
    static constexpr sctl::Integer Ngrid = PATCH_DIM * PATCH_DIM;
    static constexpr sctl::Integer Npolar = RAD_DIM * ANG_DIM;
    static_assert(INTERP_ORDER <= PATCH_DIM, "Must have INTERP_ORDER <= PATCH_DIM");

    static std::vector<Real> qx, qw;
    static std::vector<Real> Gpou_, Ppou_;
    static std::vector<sctl::Integer> I_G2P;
    static std::vector<Real> M_G2P;

  public:

    SingularCorrection() { InitPrecomp(); }

    void Setup(sctl::Integer TRG_SKIP, sctl::Long Nt, sctl::Long Np, const sctl::Vector<Real>& SrcCoord, const sctl::Vector<Real>& SrcGrad, sctl::Long t_, sctl::Long p_, sctl::Long trg_idx_, sctl::Long Ntrg_, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, Real normal_scal, sctl::Vector<Real>& work_buffer);

    void operator()(const sctl::Vector<Real>& SrcDensity, sctl::Vector<Real>& Potential) const;

  private:

    void SetPatch(sctl::Vector<Real>& out, sctl::Long t0, sctl::Long p0, const sctl::Vector<Real>& in, sctl::Long Nt, sctl::Long Np) const;

    static void InitPrecomp();

    sctl::StaticArray<Real, KDIM0 * Ngrid * KDIM1> MGrid_;
    sctl::Long TRG_SKIP, Nt, Np, t, p, trg_idx, Ntrg;
};

template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> std::vector<Real> SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::qx;
template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> std::vector<Real> SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::qw;
template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> std::vector<Real> SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::Gpou_;
template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> std::vector<Real> SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::Ppou_;
template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> std::vector<Real> SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::M_G2P;
template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> std::vector<sctl::Integer> SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::I_G2P;

template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> void SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::InitPrecomp() {
  if (!qx.size()) { // Set qx, qw
    sctl::Integer order = RAD_DIM;
    std::vector<Real> qx_(order);
    std::vector<Real> qw_(order);
    if (1) { // Set qx_, qw_
      sctl::Vector<Real> qx__(qx_.size(), sctl::Ptr2Itr<Real>(qx_.data(), qx_.size()), false);
      sctl::Vector<Real> qw__(qw_.size(), sctl::Ptr2Itr<Real>(qw_.data(), qw_.size()), false);
      sctl::ChebBasis<Real>::quad_rule(order, qx__, qw__);
    } else { // Trapezoidal rule (does not work for Helmholtz imaginary part)
      for (sctl::Long i = 0; i < order; i++) {
        qx_[i] = (2 * i + 1) / (Real)(2 * order);
        qw_[i] = 1 / (Real)order;
      }
    }

    #pragma omp critical (SingularCorrection_InitPrecomp)
    if (!qx.size()) {
      qx.swap(qx_);
      qw.swap(qw_);
    }
  }
  if (!Gpou_.size()) { // Set Gpou, Ppou
    auto pou = [&](Real r) {
      Real a = 0;
      Real b = 1;
      //return r < a ? 1 : (r >= b ? 0 : sctl::exp<Real>(1 - 1 / (1 - ((r-a)/(b-a)) * ((r-a)/(b-a)))) );
      //return r >= b ? 0 : (r < a ? 1 : sctl::exp<Real>(2*sctl::exp<Real>(-(b-a)/(r-a)) / ((r-a)/(b-a)-1))); // BRUNO AND KUNYANSKY
      if (PATCH_DIM > 45) {
        return r < a ? 1 : sctl::exp<Real>(-(Real)36 * sctl::pow<10,Real>((r-a)/(b-a)));
      } else if (PATCH_DIM > 20) {
        return r < a ? 1 : sctl::exp<Real>(-(Real)36 * sctl::pow< 8,Real>((r-a)/(b-a)));
      } else {
        return r < a ? 1 : sctl::exp<Real>(-(Real)36 * sctl::pow< 6,Real>((r-a)/(b-a)));
      }
    };

    std::vector<Real> Gpou__(Ngrid);
    sctl::Vector<Real> Gpou(Gpou__.size(), sctl::Ptr2Itr<Real>(Gpou__.data(), Gpou__.size()), false);
    sctl::Integer PATCH_RAD = (PATCH_DIM - 1) / 2;
    SCTL_ASSERT(PATCH_DIM == 2 * PATCH_RAD + 1);
    Real h = 1 / (Real)PATCH_RAD;
    for (sctl::Integer i = 0; i < PATCH_DIM; i++){
      for (sctl::Integer j = 0; j < PATCH_DIM; j++){
        Real dr[2] = {(i - PATCH_RAD) * h, (j - PATCH_RAD) * h};
        Real r = sqrt(dr[0] * dr[0] + dr[1] * dr[1]);
        Gpou[i * PATCH_DIM + j] = -pou(r);
      }
    }

    std::vector<Real> Ppou__(Npolar);
    sctl::Vector<Real> Ppou(Ppou__.size(), sctl::Ptr2Itr<Real>(Ppou__.data(), Ppou__.size()), false);
    Real dt = 2 * sctl::const_pi<Real>() / ANG_DIM;
    for (sctl::Integer i = 0; i < RAD_DIM; i++){
      for (sctl::Integer j = 0; j < ANG_DIM; j++){
        Real dr = qw[i] * PATCH_RAD;
        Real rdt = qx[i] * PATCH_RAD * dt;
        Ppou[i * ANG_DIM + j] = pou(qx[i]) * dr * rdt;
      }
    }

    #pragma omp critical (SingularCorrection_InitPrecomp)
    if (!Gpou_.size()) {
      Gpou_.swap(Gpou__);
      Ppou_.swap(Ppou__);
      if (0) {
        // this does not give accurate error estimate for trapezoidal rule
        // since |r| is not smooth.
        Real sum0=0, sum1=0;
        for (auto a : Ppou_) sum0 += a;
        for (auto a : Gpou_) sum1 += a;
        std::cout<<"POU error: "<<(double)fabs((sum0+sum1)/(fabs(sum0)+fabs(sum1)))<<'\n';
      }
    }
  }
  if (!I_G2P.size()) { // Set I_G2P, M_G2P
    std::vector<sctl::Integer> I_G2P__(Npolar);
    std::vector<Real> M_G2P__(Npolar * INTERP_ORDER * INTERP_ORDER);
    sctl::Vector<sctl::Integer> I_G2P_(I_G2P__.size(), sctl::Ptr2Itr<sctl::Integer>(I_G2P__.data(), I_G2P__.size()), false);
    sctl::Vector<Real> M_G2P_(M_G2P__.size(), sctl::Ptr2Itr<Real>(M_G2P__.data(), M_G2P__.size()), false);

    Real h = 1 / (Real)(INTERP_ORDER - 1);
    auto lagrange_interp = [&](Real x0, Real x1, sctl::Integer i0, sctl::Integer i1) { // (x0,x1) \in [0,1]x[0,1]
      Real p=1;
      Real z0 = i0 * h;
      Real z1 = i1 * h;
      for (sctl::Integer j0 = 0; j0 < INTERP_ORDER; j0++) {
        if (j0 != i0) {
          Real y0 = j0 * h;
          p *= (x0 - y0) / (z0 - y0);
        }
      }
      for (sctl::Integer j1 = 0; j1 < INTERP_ORDER; j1++) {
        if (j1 != i1) {
          Real y1 = j1 * h;
          p *= (x1 - y1) / (z1 - y1);
        }
      }
      return p;
    };
    Real h_ang = 2 * sctl::const_pi<Real>() / ANG_DIM;
    for (sctl::Integer i0 = 0; i0 < RAD_DIM; i0++) {
      for (sctl::Integer i1 = 0; i1 < ANG_DIM; i1++) {
        Real x0 = (Real)0.5 + (Real)0.5 * qx[i0] * cos(h_ang * i1);
        Real x1 = (Real)0.5 + (Real)0.5 * qx[i0] * sin(h_ang * i1);

        sctl::Integer y0 = std::max<sctl::Integer>(0, std::min<sctl::Integer>((sctl::Integer)(x0 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2), PATCH_DIM - INTERP_ORDER));
        sctl::Integer y1 = std::max<sctl::Integer>(0, std::min<sctl::Integer>((sctl::Integer)(x1 * (PATCH_DIM - 1) - (INTERP_ORDER - 1) / 2), PATCH_DIM - INTERP_ORDER));

        Real z0 = (x0 * (PATCH_DIM - 1) - y0) * h;
        Real z1 = (x1 * (PATCH_DIM - 1) - y1) * h;

        I_G2P_[i0 * ANG_DIM + i1] = y0 * PATCH_DIM + y1;
        for (sctl::Integer j0 = 0; j0 < INTERP_ORDER; j0++) {
          for (sctl::Integer j1 = 0; j1 < INTERP_ORDER; j1++) {
            M_G2P_[(((i0 * ANG_DIM + i1) * INTERP_ORDER) + j0) * INTERP_ORDER + j1] = lagrange_interp(z0, z1, j0, j1);
          }
        }
      }
    }

    #pragma omp critical (SingularCorrection_InitPrecomp)
    if (!I_G2P.size()) {
      I_G2P.swap(I_G2P__);
      M_G2P.swap(M_G2P__);
    }
  }
}

template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> void SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::Setup(sctl::Integer TRG_SKIP_, sctl::Long SrcNTor, sctl::Long SrcNPol, const sctl::Vector<Real>& SrcCoord, const sctl::Vector<Real>& SrcGrad, sctl::Long t_, sctl::Long p_, sctl::Long trg_idx_, sctl::Long Ntrg_, const KernelFunction<Real,COORD_DIM,KDIM0,KDIM1>& ker, Real normal_scal, sctl::Vector<Real>& work_buff) {
  TRG_SKIP = TRG_SKIP_;
  Nt = SrcNTor;
  Np = SrcNPol;
  t = t_;
  p = p_;

  trg_idx = trg_idx_;
  Ntrg = Ntrg_;

  assert(KDIM0 == ker.Dim(0));
  assert(KDIM1 == ker.Dim(1));
  sctl::Matrix<Real> MGrid(KDIM0 * Ngrid, KDIM1, MGrid_, false);

  sctl::Long Nbuff = (Ngrid+Npolar)*(COORD_DIM*4+1);
  if (work_buff.Dim() < Nbuff) work_buff.ReInit(Nbuff);
  //sctl::StaticArray<Real,    COORD_DIM * Ngrid> G_ ;
  //sctl::StaticArray<Real,2 * COORD_DIM * Ngrid> Gg_;
  //sctl::StaticArray<Real,    COORD_DIM * Ngrid> Gn_;
  //sctl::StaticArray<Real,                Ngrid> Ga_;

  //sctl::StaticArray<Real,    COORD_DIM * Npolar> P_ ;
  //sctl::StaticArray<Real,2 * COORD_DIM * Npolar> Pg_;
  //sctl::StaticArray<Real,    COORD_DIM * Npolar> Pn_;
  //sctl::StaticArray<Real,                Npolar> Pa_;

  sctl::Vector<Real> G (    COORD_DIM * Ngrid, work_buff.begin() + Ngrid*COORD_DIM*0, false);
  sctl::Vector<Real> Gg(2 * COORD_DIM * Ngrid, work_buff.begin() + Ngrid*COORD_DIM*1, false);
  sctl::Vector<Real> Gn(    COORD_DIM * Ngrid, work_buff.begin() + Ngrid*COORD_DIM*3, false);
  sctl::Vector<Real> Ga(                Ngrid, work_buff.begin() + Ngrid*COORD_DIM*4, false);

  sctl::Vector<Real> P (    COORD_DIM * Npolar, work_buff.begin() + Ngrid*(COORD_DIM*4+1) + Npolar*COORD_DIM*0, false);
  sctl::Vector<Real> Pg(2 * COORD_DIM * Npolar, work_buff.begin() + Ngrid*(COORD_DIM*4+1) + Npolar*COORD_DIM*1, false);
  sctl::Vector<Real> Pn(    COORD_DIM * Npolar, work_buff.begin() + Ngrid*(COORD_DIM*4+1) + Npolar*COORD_DIM*3, false);
  sctl::Vector<Real> Pa(                Npolar, work_buff.begin() + Ngrid*(COORD_DIM*4+1) + Npolar*COORD_DIM*4, false);

  sctl::StaticArray<Real, COORD_DIM> TrgCoord_;
  sctl::Vector<Real> TrgCoord(COORD_DIM, TrgCoord_, false);
  for (sctl::Integer k = 0; k < COORD_DIM; k++) { // Set TrgCoord
    TrgCoord[k] = SrcCoord[(k * Nt + t) * Np + p];
  }

  SetPatch(G , t, p, SrcCoord     , Nt, Np);
  SetPatch(Gg, t, p, SrcGrad      , Nt, Np);

  Real invNt = 1 / (Real)Nt;
  Real invNp = 1 / (Real)Np;
  for (sctl::Integer i = 0; i < Ngrid; i++) {
     Real n0 = Gg[2 * Ngrid + i] * Gg[5 * Ngrid + i] - Gg[3 * Ngrid + i] * Gg[4 * Ngrid + i];
     Real n1 = Gg[4 * Ngrid + i] * Gg[1 * Ngrid + i] - Gg[5 * Ngrid + i] * Gg[0 * Ngrid + i];
     Real n2 = Gg[0 * Ngrid + i] * Gg[3 * Ngrid + i] - Gg[1 * Ngrid + i] * Gg[2 * Ngrid + i];
     Real r = sqrt(n0 * n0 + n1 * n1 + n2 * n2);
     Real inv_r = 1 / r;
     Gn[0 * Ngrid + i] = n0 * inv_r * normal_scal;
     Gn[1 * Ngrid + i] = n1 * inv_r * normal_scal;
     Gn[2 * Ngrid + i] = n2 * inv_r * normal_scal;
     Ga[i] = r * invNt * invNp;
     Gg[0 * Ngrid + i] *= invNt;
     Gg[2 * Ngrid + i] *= invNt;
     Gg[4 * Ngrid + i] *= invNt;
     Gg[1 * Ngrid + i] *= invNp;
     Gg[3 * Ngrid + i] *= invNp;
     Gg[5 * Ngrid + i] *= invNp;
  }

  { // Lagrange interpolation
    auto matvec = [&](sctl::Vector<Real>& Vout, const sctl::Vector<Real>& Vin, const std::vector<Real>& M) {
      sctl::Integer dof = Vin.Dim() / Ngrid;
      SCTL_ASSERT(Vin.Dim() == dof * Ngrid);
      SCTL_ASSERT(Vout.Dim() == dof * Npolar);
      Vout.SetZero();
      for (sctl::Integer k = 0; k < dof; k++) {
        for (sctl::Integer j = 0; j < Npolar; j++) {
          Real tmp = 0;
          sctl::ConstIterator<Real> M_ = sctl::Ptr2ConstItr<Real>(M.data(), M.size()) + j * INTERP_ORDER * INTERP_ORDER;
          sctl::ConstIterator<Real> Vin_ = Vin.begin() + k * Ngrid + I_G2P[j];
          if (1) {
            for (sctl::Integer i0 = 0; i0 < INTERP_ORDER; i0++) {
              for (sctl::Integer i1 = 0; i1 < INTERP_ORDER; i1++) {
                tmp += M_[i0 * INTERP_ORDER + i1] * Vin_[i0 * PATCH_DIM + i1];
              }
            }
          }
          if (0 && PATCH_DIM == 4) {
            tmp += M_[ 0] * Vin_[0*PATCH_DIM+0];
            tmp += M_[ 1] * Vin_[0*PATCH_DIM+1];
            tmp += M_[ 2] * Vin_[0*PATCH_DIM+2];
            tmp += M_[ 3] * Vin_[0*PATCH_DIM+3];

            tmp += M_[ 4] * Vin_[1*PATCH_DIM+0];
            tmp += M_[ 5] * Vin_[1*PATCH_DIM+1];
            tmp += M_[ 6] * Vin_[1*PATCH_DIM+2];
            tmp += M_[ 7] * Vin_[1*PATCH_DIM+3];

            tmp += M_[ 8] * Vin_[2*PATCH_DIM+0];
            tmp += M_[ 9] * Vin_[2*PATCH_DIM+1];
            tmp += M_[10] * Vin_[2*PATCH_DIM+2];
            tmp += M_[11] * Vin_[2*PATCH_DIM+3];

            tmp += M_[12] * Vin_[3*PATCH_DIM+0];
            tmp += M_[13] * Vin_[3*PATCH_DIM+1];
            tmp += M_[14] * Vin_[3*PATCH_DIM+2];
            tmp += M_[15] * Vin_[3*PATCH_DIM+3];
          }
          Vout[k * Npolar + j] = tmp;
        }
      }
      //sctl::Profile::Add_FLOP(dof*Npolar*INTERP_ORDER*INTERP_ORDER);
    };
    matvec(P , G , M_G2P);
    matvec(Pg, Gg , M_G2P);
    for (sctl::Integer i = 0; i < Npolar; i++) { // Compute Pn, Pa
       Real n0 = Pg[2 * Npolar + i] * Pg[5 * Npolar + i] - Pg[3 * Npolar + i] * Pg[4 * Npolar + i];
       Real n1 = Pg[4 * Npolar + i] * Pg[1 * Npolar + i] - Pg[5 * Npolar + i] * Pg[0 * Npolar + i];
       Real n2 = Pg[0 * Npolar + i] * Pg[3 * Npolar + i] - Pg[1 * Npolar + i] * Pg[2 * Npolar + i];
       Real r = sqrt(n0 * n0 + n1 * n1 + n2 * n2);
       n0 /= r;
       n1 /= r;
       n2 /= r;
       Pn[0 * Npolar + i] = n0 * normal_scal;
       Pn[1 * Npolar + i] = n1 * normal_scal;
       Pn[2 * Npolar + i] = n2 * normal_scal;
       Pa[i] = r;
    }
  }

  { // Subtract singular part from U
    { // Subtract singular part from U
      #ifndef DISABLE_FAR_FIELD
      ker.BuildMatrix(G, Gn, TrgCoord, MGrid);
      for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) { // apply weights (Ga * Gpou)
        for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
          for (sctl::Integer i = 0; i < Ngrid; i++) {
            MGrid[k0 * Ngrid + i][k1] *= Ga[i] * Gpou_[i];
          }
        }
      }
      #else
      MGrid.SetZero();
      #endif
    }
    { // Add singular part to U
      sctl::Matrix<Real> MPolar(KDIM0 * Npolar, KDIM1);

      ker.BuildMatrix(P, Pn, TrgCoord, MPolar); // MPolar <-- ker(P, Pn, TrgCoord)
      for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) { // MPolar <-- Mpolar * Pa * Ppou
        for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
          for (sctl::Integer i = 0; i < Npolar; i++) {
            MPolar[k0 * Npolar + i][k1] *= Pa[i] * Ppou_[i];
          }
        }
      }
      for (sctl::Integer j = 0; j < Npolar; j++) { // MGrid <-- MPolar * Transpose(M_G2P)
        sctl::Iterator<Real> MGrid_ = MGrid[I_G2P[j]];
        sctl::ConstIterator<Real> MPolar_ = MPolar[j];
        sctl::ConstIterator<Real> M_G2P_ = sctl::Ptr2ConstItr<Real>(&M_G2P[0], M_G2P.size()) + j * INTERP_ORDER * INTERP_ORDER;
        for (sctl::Integer i0 = 0; i0 < INTERP_ORDER; i0++) {
          for (sctl::Integer i1 = 0; i1 < INTERP_ORDER; i1++) {
            Real M_G2P__ = M_G2P_[i0 * INTERP_ORDER + i1];
            sctl::Iterator<Real> MGrid__ = MGrid_ + (i0 * PATCH_DIM + i1) * KDIM1;
            for (sctl::Integer k0 = 0; k0 < KDIM0; k0++) {
              for (sctl::Integer k1 = 0; k1 < KDIM1; k1++) {
                MGrid__[k0 * Ngrid * KDIM1 + k1] += M_G2P__ * MPolar_[k0 * Npolar * KDIM1 + k1];
              }
            }
          }
        }
      }
    }
  }
}

template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> void SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::operator()(const sctl::Vector<Real>& SrcDensity, sctl::Vector<Real>& Potential) const {
  sctl::StaticArray<Real, KDIM0*Ngrid> GF_;
  sctl::Vector<Real> GF(KDIM0 * Ngrid, GF_, false);
  SetPatch(GF, t, p, SrcDensity, Nt, Np);

  sctl::StaticArray<Real,KDIM1> U_;
  sctl::Matrix<Real> U(1,KDIM1, U_, false);
  { // Compute correction U
    const sctl::Matrix<Real> Vin(1, GF.Dim(), GF.begin(), false);
    const sctl::Matrix<Real> MGrid(KDIM0 * Ngrid, KDIM1, (sctl::Iterator<Real>)(sctl::ConstIterator<Real>)MGrid_, false);
    sctl::Matrix<Real>::GEMM(U, Vin, MGrid);
  }
  for (sctl::Integer k = 0; k < KDIM1; k++) { // Add correction to Potential
    Potential[k * Ntrg + trg_idx] += U[0][k];
  }
}

template <class Real, sctl::Integer PATCH_DIM0, sctl::Integer RAD_DIM, sctl::Integer KDIM0, sctl::Integer KDIM1> void SingularCorrection<Real,PATCH_DIM0,RAD_DIM,KDIM0,KDIM1>::SetPatch(sctl::Vector<Real>& out, sctl::Long t0, sctl::Long p0, const sctl::Vector<Real>& in, sctl::Long Nt, sctl::Long Np) const {
  SCTL_ASSERT(Nt >= PATCH_DIM);
  SCTL_ASSERT(Np >= PATCH_DIM);

  sctl::Long dof = in.Dim() / (Nt * Np);
  SCTL_ASSERT(in.Dim() == dof * Nt * Np);
  SCTL_ASSERT(out.Dim() == dof * Ngrid);
  sctl::Integer tt0 = t0 - (PATCH_DIM - 1) / 2;
  sctl::Integer pp0 = p0 - (PATCH_DIM - 1) / 2;
  for (sctl::Long k = 0; k < dof; k++) {
    for (sctl::Long i = 0; i < PATCH_DIM; i++) {
      sctl::Long t = tt0 + i;
      if (t >= Nt) t = t - Nt;
      if (t < 0) t = t + Nt;
      sctl::ConstIterator<Real> in_ = in.begin() + (k * Nt + t) * Np;
      sctl::Iterator<Real> out_ = out.begin() + (k * PATCH_DIM + i) * PATCH_DIM;
      for (sctl::Long j = 0; j < PATCH_DIM; j++) {
        sctl::Long p = pp0 + j;
        if (p >= Np) p = p - Np;
        if (p < 0) p = p + Np;
        out_[j] = in_[p];
      }
    }
  }
}

}

#endif  //_SINGULAR_CORRECTION_HPP_
