// Scientific Computing Template Library

#ifndef _SCTL_HPP_
#define _SCTL_HPP_

#include "sctl/common.hpp"

// Import PVFMM preprocessor macro definitions
#ifdef SCTL_HAVE_PVFMM
#ifndef SCTL_HAVE_MPI
#define SCTL_HAVE_MPI
#endif
#include "pvfmm_config.h"
#if defined(PVFMM_QUAD_T) && !defined(SCTL_QUAD_T)
#define SCTL_QUAD_T PVFMM_QUAD_T
#endif
#endif

// Math utilities
#include "sctl/math_utils.hpp"

// FMM wrapper
#include "sctl/fmm-wrapper.hpp"

// Boundary Integrals
#include "sctl/boundary_integral.hpp"
#include "sctl/slender_element.hpp"
#include "sctl/quadrule.hpp"
#include "sctl/lagrange-interp.hpp"

// ODE solver
#include "sctl/ode-solver.hpp"

// Tensor
#include "sctl/tensor.hpp"

// Tree
#include "sctl/tree.hpp"
#include "sctl/vtudata.hpp"

// MPI Wrapper
#include "sctl/comm.hpp"

// Memory Manager, Iterators
#include "sctl/mem_mgr.hpp"

// Vector
#include "sctl/vector.hpp"

// Matrix, Permutation operators
#include "sctl/matrix.hpp"

// Template vector intrinsics (new)
#include "sctl/vec.hpp"
#include "sctl/vec-test.hpp"

// OpenMP merge-sort and scan
#include "sctl/ompUtils.hpp"

// Parallel solver
#include "sctl/parallel_solver.hpp"

// Chebyshev basis
#include "sctl/cheb_utils.hpp"

// Morton
#include "sctl/morton.hpp"

// Spherical Harmonics
#include "sctl/sph_harm.hpp"

#include "sctl/fft_wrapper.hpp"

#include "sctl/legendre_rule.hpp"

// Profiler
#include "sctl/profile.hpp"

// Print stack trace
#include "sctl/stacktrace.h"

// Set signal handler
const int sgh = sctl::SetSigHandler();

// Boundary quadrature, Kernel functions
#include "sctl/kernel_functions.hpp"
#include "sctl/boundary_quadrature.hpp"

#endif //_SCTL_HPP_
