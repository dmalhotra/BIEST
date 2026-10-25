# Builds the tutorial programs. Run from this directory: make && make run

BIEST_DIR = ../..

CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -O3 -march=native
CXXFLAGS += -I$(BIEST_DIR)/include -I$(BIEST_DIR)/extern/SCTL/include

# Optional libraries; SCTL uses its own (slower) routines without them
CXXFLAGS += -DSCTL_HAVE_FFTW
LDLIBS += -lfftw3
#CXXFLAGS += -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK
#LDLIBS += -lblas -llapack

EXAMPLES = surface surface-op layer-potentials field-period ext-vacuum taylor-state-shell
TARGETS = $(EXAMPLES:%=bin/%)

.PHONY: all run clean

all: $(TARGETS)

bin/%: %.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

# Runs every program in bin/, where they write their VTK files, and stops at the first one that fails
run: $(TARGETS)
	@for p in $(EXAMPLES); do echo "== $$p"; (cd bin && ./$$p) || exit 1; done

clean:
	rm -rf bin
