CXX = icpc
CXXFLAGS = -std=c++11 -fopenmp -mavx -Wall # need C++11 and OpenMP

#Optional flags
#CXXFLAGS += -O0 # debug build
CXXFLAGS += -O3 -DNDEBUG # release build

ifeq ($(shell uname -s),Darwin)
	CXXFLAGS += -g -rdynamic -Wl,-no_pie # for stack trace (on Mac)
else
	CXXFLAGS += -g -rdynamic # for stack trace
endif

#CXXFLAGS += -DSCTL_MEMDEBUG # Enable memory checks
CXXFLAGS += -DSCTL_PROFILE=5 -DSCTL_VERBOSE # Enable profiling

#CXXFLAGS += -lblas -DSCTL_HAVE_BLAS # use BLAS
#CXXFLAGS += -llapack -DSCTL_HAVE_LAPACK # use LAPACK
CXXFLAGS += -mkl -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK # use MKL BLAS and LAPACK
#CXXFLAGS += -DSCTL_HAVE_SVML

CXXFLAGS += -lfftw3_omp -DSCTL_FFTW_THREADS
CXXFLAGS += -lfftw3 -DSCTL_HAVE_FFTW
CXXFLAGS += -lfftw3f -DSCTL_HAVE_FFTWF
CXXFLAGS += -lfftw3l -DSCTL_HAVE_FFTWL

#PSC_INC = -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include
#PSC_LIB = -L$(PETSC_DIR)/lib -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
#CXXFLAGS += $(PSC_INC) $(PSC_LIB) -DSCTL_HAVE_PETSC

RM = rm -f
MKDIRS = mkdir -p

BINDIR = ./bin
SRCDIR = ./src
OBJDIR = ./obj
INCDIR = ./include

TARGET_BIN = \
       $(BINDIR)/table1 \
       $(BINDIR)/table3 \
       $(BINDIR)/table4 \
       $(BINDIR)/table5 \
       $(BINDIR)/table6 \
       $(BINDIR)/table7 \
       $(BINDIR)/quadrature-example \
       $(BINDIR)/force-free-fields-example \
       $(BINDIR)/double-layer-convergence \
       $(BINDIR)/virtual-casing-principle

all : $(TARGET_BIN)

$(BINDIR)/%: $(OBJDIR)/%.o
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $^ -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJDIR)/*

