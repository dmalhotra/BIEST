#ifndef _SURFACE_HPP_
#define _SURFACE_HPP_

#include <sctl.hpp>

namespace biest {

enum class SurfType {
  AxisymCircleWide,        // 125 x 50
  AxisymCircleNarrow,      // 250 x 50
  AxisymWide,              // 125 x 50
  AxisymNarrow,            // 250 x 50
  RotatingEllipseWide,     // 125 x 50
  RotatingEllipseNarrow,   // 250 x 50
  Quas3,                   // 250 x 50
  LHD,                     // 250 x 50
  W7X,                     // 250 x 45
  Stell,                   // 250 x 50
  W7X_                     // 250 x 50
};

template <class Real> class Surface {
    static constexpr sctl::Integer COORD_DIM = 3;

  public:

    Surface() : Nt0(0), Np0(0) {}

    Surface(sctl::Long Nt, sctl::Long Np, SurfType type = SurfType::AxisymCircleWide);

    sctl::Long NTor() const {return Nt0;}
    sctl::Long NPol() const {return Np0;}
    sctl::Vector<Real>& Coord() {return X0_;}
    const sctl::Vector<Real>& Coord() const {return X0_;}

  private:

    sctl::Long Nt0, Np0;
    sctl::Vector<Real> X0_;
};

struct VTKData {
  typedef double VTKReal;
  static constexpr sctl::Integer COORD_DIM = 3;

  std::vector<VTKReal> point_coord;
  std::vector<VTKReal> point_value;
  std::vector<int32_t> line_connect;
  std::vector<int32_t> line_offset;
  std::vector<int32_t> poly_connect;
  std::vector<int32_t> poly_offset;

  void WriteVTK(const char* fname, const sctl::Comm& comm = sctl::Comm::Self()) {
    sctl::Integer np = comm.Size();
    sctl::Integer myrank = comm.Rank();

    std::vector<VTKReal>& coord=this->point_coord;
    std::vector<VTKReal>& value=this->point_value;
    std::vector<int32_t>& line_connect=this->line_connect;
    std::vector<int32_t>& line_offset=this->line_offset;
    std::vector<int32_t>& poly_connect=this->poly_connect;
    std::vector<int32_t>& poly_offset=this->poly_offset;

    sctl::Long pt_cnt=coord.size()/COORD_DIM;
    sctl::Long line_cnt=line_offset.size();
    sctl::Long poly_cnt=poly_offset.size();

    // Open file for writing.
    std::stringstream vtufname;
    vtufname<<fname<<"_"<<std::setfill('0')<<std::setw(6)<<myrank<<".vtp";
    std::ofstream vtufile;
    vtufile.open(vtufname.str().c_str());
    if(vtufile.fail()) return;

    bool isLittleEndian;
    { // Set isLittleEndian
      uint16_t number = 0x1;
      uint8_t *numPtr = (uint8_t*)&number;
      isLittleEndian=(numPtr[0] == 1);
    }

    // Proceed to write to file.
    sctl::Long data_size=0;
    vtufile<<"<?xml version=\"1.0\"?>\n";
    if(isLittleEndian) vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    else               vtufile<<"<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    //===========================================================================
    vtufile<<"  <PolyData>\n";
    vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfVerts=\"0\" NumberOfLines=\""<<line_cnt<<"\" NumberOfStrips=\"0\" NumberOfPolys=\""<<poly_cnt<<"\">\n";

    //---------------------------------------------------------------------------
    vtufile<<"      <Points>\n";
    vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKReal);
    vtufile<<"      </Points>\n";
    //---------------------------------------------------------------------------
    if(value.size()){ // value
      vtufile<<"      <PointData>\n";
      vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
      data_size+=sizeof(uint32_t)+value.size()*sizeof(VTKReal);
      vtufile<<"      </PointData>\n";
    }
    //---------------------------------------------------------------------------
    vtufile<<"      <Lines>\n";
    vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+line_connect.size()*sizeof(int32_t);
    vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+line_offset.size() *sizeof(int32_t);
    vtufile<<"      </Lines>\n";
    //---------------------------------------------------------------------------
    vtufile<<"      <Polys>\n";
    vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+poly_connect.size()*sizeof(int32_t);
    vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+poly_offset.size() *sizeof(int32_t);
    vtufile<<"      </Polys>\n";
    //---------------------------------------------------------------------------

    vtufile<<"    </Piece>\n";
    vtufile<<"  </PolyData>\n";
    //===========================================================================
    vtufile<<"  <AppendedData encoding=\"raw\">\n";
    vtufile<<"    _";

    int32_t block_size;
    block_size=coord.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord  [0], coord.size()*sizeof(VTKReal));
    if(value.size()){ // value
      block_size=value.size()*sizeof(VTKReal); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value  [0], value.size()*sizeof(VTKReal));
    }
    block_size=line_connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&line_connect[0], line_connect.size()*sizeof(int32_t));
    block_size=line_offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&line_offset [0], line_offset .size()*sizeof(int32_t));
    block_size=poly_connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&poly_connect[0], poly_connect.size()*sizeof(int32_t));
    block_size=poly_offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&poly_offset [0], poly_offset .size()*sizeof(int32_t));

    vtufile<<"\n";
    vtufile<<"  </AppendedData>\n";
    //===========================================================================
    vtufile<<"</VTKFile>\n";
    vtufile.close();


    if(myrank) return;
    std::stringstream pvtufname;
    pvtufname<<fname<<".pvtp";
    std::ofstream pvtufile;
    pvtufile.open(pvtufname.str().c_str());
    if(pvtufile.fail()) return;
    pvtufile<<"<?xml version=\"1.0\"?>\n";
    pvtufile<<"<VTKFile type=\"PPolyData\">\n";
    pvtufile<<"  <PPolyData GhostLevel=\"0\">\n";
    pvtufile<<"      <PPoints>\n";
    pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\"/>\n";
    pvtufile<<"      </PPoints>\n";
    if(value.size()){ // value
      pvtufile<<"      <PPointData>\n";
      pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal)*8<<"\" NumberOfComponents=\""<<value.size()/pt_cnt<<"\" Name=\""<<"value"<<"\"/>\n";
      pvtufile<<"      </PPointData>\n";
    }
    {
      // Extract filename from path.
      std::stringstream vtupath;
      vtupath<<'/'<<fname;
      std::string pathname = vtupath.str();
      auto found = pathname.find_last_of("/\\");
      std::string fname_ = pathname.substr(found+1);
      for(sctl::Integer i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<"_"<<std::setfill('0')<<std::setw(6)<<i<<".vtp\"/>\n";
    }
    pvtufile<<"  </PPolyData>\n";
    pvtufile<<"</VTKFile>\n";
    pvtufile.close();
  }
};
struct VTUData {
  typedef float VTKReal;

  // Point data
  sctl::Vector<VTKReal> coord;  // always 3D
  sctl::Vector<VTKReal> value;

  // Cell data
  sctl::Vector<int32_t> connect;
  sctl::Vector<int32_t> offset;
  sctl::Vector<uint8_t> types;

  void WriteVTK(const std::string& fname, const sctl::Comm& comm = sctl::Comm::Self()) const {
    typedef typename VTUData::VTKReal VTKReal;
    sctl::Long value_dof = 0;
    {  // Write vtu file.
      std::ofstream vtufile;
      {  // Open file for writing.
        std::stringstream vtufname;
        vtufname << fname << std::setfill('0') << std::setw(6) << comm.Rank() << ".vtu";
        vtufile.open(vtufname.str().c_str());
        if (vtufile.fail()) return;
      }
      {  // Write to file.
        sctl::Long pt_cnt = coord.Dim() / 3;
        sctl::Long cell_cnt = types.Dim();
        value_dof = (pt_cnt ? value.Dim() / pt_cnt : 0);

        sctl::Vector<int32_t> mpi_rank;
        {  // Set  mpi_rank
          sctl::Integer new_myrank = comm.Rank();
          mpi_rank.ReInit(pt_cnt);
          for (sctl::Long i = 0; i < mpi_rank.Dim(); i++) mpi_rank[i] = new_myrank;
        }

        bool isLittleEndian;
        {  // Set isLittleEndian
          uint16_t number = 0x1;
          uint8_t *numPtr = (uint8_t *)&number;
          isLittleEndian = (numPtr[0] == 1);
        }

        sctl::Long data_size = 0;
        vtufile << "<?xml version=\"1.0\"?>\n";
        vtufile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"" << (isLittleEndian ? "LittleEndian" : "BigEndian") << "\">\n";
        // ===========================================================================
        vtufile << "  <UnstructuredGrid>\n";
        vtufile << "    <Piece NumberOfPoints=\"" << pt_cnt << "\" NumberOfCells=\"" << cell_cnt << "\">\n";
        //---------------------------------------------------------------------------
        vtufile << "      <Points>\n";
        vtufile << "        <DataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"3\" Name=\"Position\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + coord.Dim() * sizeof(VTKReal);
        vtufile << "      </Points>\n";
        //---------------------------------------------------------------------------
        vtufile << "      <PointData>\n";
        if (value_dof) {  // value
          vtufile << "        <DataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"" << value_dof << "\" Name=\"value\" format=\"appended\" offset=\"" << data_size << "\" />\n";
          data_size += sizeof(uint32_t) + value.Dim() * sizeof(VTKReal);
        }
        {  // mpi_rank
          vtufile << "        <DataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\" format=\"appended\" offset=\"" << data_size << "\" />\n";
          data_size += sizeof(uint32_t) + pt_cnt * sizeof(int32_t);
        }
        vtufile << "      </PointData>\n";
        //---------------------------------------------------------------------------
        //---------------------------------------------------------------------------
        vtufile << "      <Cells>\n";
        vtufile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + connect.Dim() * sizeof(int32_t);
        vtufile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + offset.Dim() * sizeof(int32_t);
        vtufile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\"" << data_size << "\" />\n";
        data_size += sizeof(uint32_t) + types.Dim() * sizeof(uint8_t);
        vtufile << "      </Cells>\n";
        //---------------------------------------------------------------------------
        vtufile << "    </Piece>\n";
        vtufile << "  </UnstructuredGrid>\n";
        // ===========================================================================
        vtufile << "  <AppendedData encoding=\"raw\">\n";
        vtufile << "    _";

        int32_t block_size;
        {  // coord
          block_size = coord.Dim() * sizeof(VTKReal);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (coord.Dim()) vtufile.write((char *)&coord[0], coord.Dim() * sizeof(VTKReal));
        }
        if (value_dof) {  // value
          block_size = value.Dim() * sizeof(VTKReal);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (value.Dim()) vtufile.write((char *)&value[0], value.Dim() * sizeof(VTKReal));
        }
        {  // mpi_rank
          block_size = mpi_rank.Dim() * sizeof(int32_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (mpi_rank.Dim()) vtufile.write((char *)&mpi_rank[0], mpi_rank.Dim() * sizeof(int32_t));
        }
        {  // block_size
          block_size = connect.Dim() * sizeof(int32_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (connect.Dim()) vtufile.write((char *)&connect[0], connect.Dim() * sizeof(int32_t));
        }
        {  // offset
          block_size = offset.Dim() * sizeof(int32_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (offset.Dim()) vtufile.write((char *)&offset[0], offset.Dim() * sizeof(int32_t));
        }
        {  // types
          block_size = types.Dim() * sizeof(uint8_t);
          vtufile.write((char *)&block_size, sizeof(int32_t));
          if (types.Dim()) vtufile.write((char *)&types[0], types.Dim() * sizeof(uint8_t));
        }

        vtufile << "\n";
        vtufile << "  </AppendedData>\n";
        // ===========================================================================
        vtufile << "</VTKFile>\n";
      }
      vtufile.close();  // close file
    }
    if (!comm.Rank()) {  // Write pvtu file
      std::ofstream pvtufile;
      {  // Open file for writing
        std::stringstream pvtufname;
        pvtufname << fname << ".pvtu";
        pvtufile.open(pvtufname.str().c_str());
        if (pvtufile.fail()) return;
      }
      {  // Write to file.
        pvtufile << "<?xml version=\"1.0\"?>\n";
        pvtufile << "<VTKFile type=\"PUnstructuredGrid\">\n";
        pvtufile << "  <PUnstructuredGrid GhostLevel=\"0\">\n";
        pvtufile << "      <PPoints>\n";
        pvtufile << "        <PDataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"3\" Name=\"Position\"/>\n";
        pvtufile << "      </PPoints>\n";
        pvtufile << "      <PPointData>\n";
        if (value_dof) {  // value
          pvtufile << "        <PDataArray type=\"Float" << sizeof(VTKReal) * 8 << "\" NumberOfComponents=\"" << value_dof << "\" Name=\"value\"/>\n";
        }
        {  // mpi_rank
          pvtufile << "        <PDataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\"/>\n";
        }
        pvtufile << "      </PPointData>\n";
        {
          // Extract filename from path.
          std::stringstream vtupath;
          vtupath << '/' << fname;
          std::string pathname = vtupath.str();
          std::string fname_ = pathname.substr(pathname.find_last_of("/\\") + 1);
          // char *fname_ = (char*)strrchr(vtupath.str().c_str(), '/') + 1;
          // std::string fname_ =
          // boost::filesystem::path(fname).filename().string().
          for (sctl::Integer i = 0; i < comm.Size(); i++) pvtufile << "      <Piece Source=\"" << fname_ << std::setfill('0') << std::setw(6) << i << ".vtu\"/>\n";
        }
        pvtufile << "  </PUnstructuredGrid>\n";
        pvtufile << "</VTKFile>\n";
      }
      pvtufile.close();  // close file
    }
  };
};

template <class Real> void WriteVTK(const char* fname, const sctl::Vector<Surface<Real>>& Svec, const sctl::Vector<Real> F = sctl::Vector<Real>(), const sctl::Comm& comm = sctl::Comm::Self()) {
  VTKData data;
  typedef VTKData::VTKReal VTKReal;
  auto& point_coord =data.point_coord ;
  auto& point_value =data.point_value ;
  auto& poly_connect=data.poly_connect;
  auto& poly_offset =data.poly_offset ;
  constexpr sctl::Integer COORD_DIM = VTKData::COORD_DIM;

  sctl::Long dof, offset = 0;
  { // Set dof
    sctl::Long Npt = 0;
    for (const auto& S : Svec) Npt += S.NTor() * S.NPol();
    dof = F.Dim() / Npt;
    SCTL_ASSERT(F.Dim() == dof * Npt);
  }
  for (auto& S: Svec) { // Set point_coord, point_value, poly_connect
    sctl::Long Nt = S.NTor();
    sctl::Long Np = S.NPol();
    sctl::Long N = Nt * Np;

    for (sctl::Long i = 0; i < Nt; i++) {
      for (sctl::Long j = 0; j < Np; j++) {
        sctl::Long i0 = (i + 0) % Nt;
        sctl::Long i1 = (i + 1) % Nt;
        sctl::Long j0 = (j + 0) % Np;
        sctl::Long j1 = (j + 1) % Np;

        poly_connect.push_back(point_coord.size() / COORD_DIM + Np*i0+j0);
        poly_connect.push_back(point_coord.size() / COORD_DIM + Np*i1+j0);
        poly_connect.push_back(point_coord.size() / COORD_DIM + Np*i1+j1);
        poly_connect.push_back(point_coord.size() / COORD_DIM + Np*i0+j1);
        poly_offset.push_back(poly_connect.size());
      }
    }

    const auto X = S.Coord();
    SCTL_ASSERT(X.Dim() == COORD_DIM * N);
    for (sctl::Long i = 0; i < N; i++){ // Set point_coord
      for (sctl::Integer k = 0; k < COORD_DIM; k++) point_coord.push_back((VTKReal)X[k * N + i]);
    }

    for (sctl::Long i = 0; i < N; i++){ // Set point_value
      for (sctl::Long k = 0; k < dof; k++) point_value.push_back((VTKReal)F[dof * offset + k * N + i]);
    }
    offset += N;
  }
  data.WriteVTK(fname, comm);
}

template <class Real> void WriteVTK(const char* fname, const Surface<Real>& S, const sctl::Vector<Real> F = sctl::Vector<Real>(), const sctl::Comm& comm = sctl::Comm::Self()) {
  WriteVTK(fname, sctl::Vector<Surface<Real>>(1,sctl::Ptr2Itr<Surface<Real>>((Surface<Real>*)&S,1),false), F, comm);
}

}

#include <biest/surface.txx>

#endif  //_SURFACE_HPP_
