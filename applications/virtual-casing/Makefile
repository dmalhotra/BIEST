BIEST_INCDIR = ./../../include

CXX = g++
CXXFLAGS = -std=c++11 -fopenmp -march=native -Wall # need C++11 and OpenMP

#Optional flags
#CXXFLAGS += -O0 # debug build
CXXFLAGS += -O3 -DNDEBUG # release build

ifeq ($(shell uname -s),Darwin)
	CXXFLAGS += -g -rdynamic -Wl,-no_pie # for stack trace (on Mac)
else
	CXXFLAGS += -g -rdynamic # for stack trace
endif

#CXXFLAGS += -DSCTL_MEMDEBUG # Enable memory checks
#CXXFLAGS += -DSCTL_PROFILE=5 -DSCTL_VERBOSE # Enable profiling
CXXFLAGS += -DSCTL_QUAD_T=__float128 # Enable quadruple precision

#CXXFLAGS += -lblas -DSCTL_HAVE_BLAS # use BLAS
#CXXFLAGS += -llapack -DSCTL_HAVE_LAPACK # use LAPACK
#CXXFLAGS += -mkl -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK # use MKL BLAS and LAPACK
CXXFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK # use MKL BLAS and LAPACK (non-Intel compiler)
#CXXFLAGS += -DSCTL_HAVE_SVML

#CXXFLAGS += -lfftw3_omp -DSCTL_FFTW_THREADS
#CXXFLAGS += -lfftw3 -DSCTL_HAVE_FFTW
#CXXFLAGS += -lfftw3f -DSCTL_HAVE_FFTWF
#CXXFLAGS += -lfftw3l -DSCTL_HAVE_FFTWL

TARGET_BIN = virtual-casing

all : $(TARGET_BIN)

%: %.cpp
	$(CXX) $(CXXFLAGS) -I$(BIEST_INCDIR) $^ -o $@

clean:
	rm -f *.o *.vtp *.pvtp *.vtu *.pvtu $(TARGET_BIN)

