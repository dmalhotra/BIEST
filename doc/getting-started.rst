..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _biest_getting-started:

Getting Started
===============

.. note::

    BIEST requires a C++17 compliant compiler with OpenMP 4.0 support: GCC 9 or newer, the Intel compiler (with GCC 9 compatibility or newer), or Clang with LLVM 10 or newer.


Downloading BIEST
-----------------

Download the latest version of BIEST from the `BIEST GitHub repository <https://github.com/dmalhotra/BIEST>`_.
The `SCTL library <https://sctl.readthedocs.io/>`_ is included as a git submodule in ``extern/SCTL``, so clone with ``--recurse-submodules``:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/dmalhotra/BIEST.git

If the repository was cloned without this flag, fetch the submodule afterwards:

.. code-block:: bash

   git submodule update --init


Using BIEST in Your Projects
----------------------------

BIEST is a header-only library; it does not require compilation or installation.
Everything is in the ``biest::`` namespace.
Include the ``biest.hpp`` header file in your C++ code:

.. code-block:: cpp

   #include <biest.hpp>

BIEST is built on `SCTL <https://sctl.readthedocs.io/>`_, and its interface uses SCTL types: containers
(`sctl::Vector <https://sctl.readthedocs.io/en/latest/tutorial/vector.html>`_),
the MPI communicator
(`sctl::Comm <https://sctl.readthedocs.io/en/latest/tutorial/comm.html>`_) and
the `profiler <https://sctl.readthedocs.io/en/latest/tutorial/profile.html>`_.
The compiler must find both header directories:

.. code-block:: bash

   -I ${BIEST_ROOT}/include -I ${BIEST_ROOT}/extern/SCTL/include

The following compiler flags are **required**:

- ``-std=c++17``: Enables the C++17 standard.
- ``-fopenmp``: Enables OpenMP support for parallelization.


Optional Compiler Flags
-----------------------

- ``-O3 -march=native``: Maximum optimization, with SIMD vectorization for the local machine.
  On Intel Sapphire Rapids and newer CPUs, GCC with ``-march=native`` emits AVX-512-FP16 instructions that binutils older than 2.38 cannot assemble; add ``-mno-avx512fp16`` in that case (the included ``Makefile`` does this on Linux).
- ``-DSCTL_PROFILE=5 -DSCTL_VERBOSE``: Enables `profiling <https://sctl.readthedocs.io/en/latest/tutorial/profile.html>`_ output (setup, solve and quadrature times, GMRES residuals).
- ``-DSCTL_MEMDEBUG``: Enables memory checks. This incurs a significant performance penalty; use it only for debugging.
- ``-DSCTL_SIG_HANDLER``: Prints a stack trace on a crash. On Linux with glibc older than 2.34, also link with ``-ldl``.
- ``-DSCTL_GLOBAL_MEM_BUFF=0``: Size in MB of SCTL's global memory buffer (``0`` disables it).


Optional Libraries
------------------

BIEST runs without any external library, but the following libraries make it faster.
To enable one, define the corresponding flag during compilation and link the library.

- `FFTW <https://www.fftw.org>`_ (**recommended**): BIEST uses FFTs for spectral differentiation, upsampling and on-surface solves. Enable double precision by defining ``SCTL_HAVE_FFTW``, single precision by defining ``SCTL_HAVE_FFTWF``, long double precision by defining ``SCTL_HAVE_FFTWL``, and threaded FFTW by defining ``SCTL_FFTW_THREADS``.
- **BLAS**: Enable by defining ``SCTL_HAVE_BLAS``.
- **LAPACK**: Enable by defining ``SCTL_HAVE_LAPACK``.
- **libmvec**: Enable by defining ``SCTL_HAVE_LIBMVEC``.
- **Intel SVML**: Enable by defining ``SCTL_HAVE_SVML``.
- `LIBXSMM <https://github.com/libxsmm/libxsmm>`_: Enable by defining ``SCTL_HAVE_LIBXSMM``, for small matrix products.
- **MPI**: Enable by defining ``SCTL_HAVE_MPI`` (see `Comm <https://sctl.readthedocs.io/en/latest/tutorial/comm.html>`_). BIEST divides the target points of the quadratures among the MPI processes of the communicator passed to it; all other data and work are replicated on every process.

For example, to use FFTW and OpenBLAS:

.. code-block:: bash

   g++ -std=c++17 -fopenmp -O3 -march=native \
       -I ${BIEST_ROOT}/include -I ${BIEST_ROOT}/extern/SCTL/include \
       -DSCTL_HAVE_FFTW -DSCTL_HAVE_BLAS -DSCTL_HAVE_LAPACK \
       my_code.cpp -o my_code -lfftw3 -lopenblas


Compiling and Running the Example Codes
---------------------------------------

The ``src/`` directory contains example codes and the codes used to generate the results in the papers.
In the BIEST directory, modify the ``Makefile`` as necessary for your environment, in particular the C++ compiler (``CXX``) and the BLAS/LAPACK and FFTW lines.
The ``Makefile`` has commented-out alternatives for OpenBLAS and Intel MKL.
It can also be used as a template for new projects.
Then, compile and run an example:

.. code-block:: bash

   make bin/force-free-fields-example && ./bin/force-free-fields-example

Run the programs from the BIEST root directory.
The predefined surfaces ``SurfType::Quas3``, ``SurfType::LHD`` and ``SurfType::W7X`` read their shape from ``geom/*.mat``, and this path is relative to the current working directory.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Program
     - Description
   * - ``force-free-fields-example``
     - Computes a vacuum field and a Taylor state on a predefined surface (:ref:`tutorial <tutorial-taylor-state>`).
   * - ``example1``
     - Builds a surface from VMEC-style ``RBC``/``ZBS`` Fourier coefficients (``./bin/example1 geom/NCSX``), computes the field and evaluates it at points inside the domain (:ref:`tutorial <tutorial-taylor-state>`).
   * - ``quadrature-example``
     - Evaluates the Laplace single-layer potential on a torus (:ref:`tutorial <tutorial-layer-potentials>`).
   * - ``double-layer-convergence``
     - Convergence of the Laplace double-layer potential (Gauss's identity) on a torus and on W7-X.
   * - ``virtual-casing-principle``
     - Convergence study of virtual casing computations with ``BoundaryIntegralOp`` on W7-X. It first computes a reference solution on a fine grid, which takes several hours.
   * - ``table1``, ``table3`` ... ``table7``
     - Codes used to generate the tables in the :ref:`Taylor states paper <biest_citing>`.

For production virtual casing computations (with field-period and stellarator symmetry, and Python bindings), use the `virtual-casing <https://github.com/hiddenSymmetries/virtual-casing>`_ library, which is built on BIEST.


Running the Unit Tests
----------------------

The ``test/`` directory has unit tests of the surface operators, the layer-potential operators and the solvers, which check the results against exact solutions and identities.
Build and run them with

.. code-block:: bash

   make -j test

Each test program prints one line per check (``PASS`` or ``FAIL``, the error and the tolerance) and returns a nonzero exit code if a check fails.
The tests take about half a minute on 4 cores.

The programs of the :ref:`tutorials <tutorial_index>` are in ``doc/examples``, with their own ``Makefile``:

.. code-block:: bash

   make -C doc/examples && make -C doc/examples run

GitHub Actions (``.github/workflows/test.yml``) runs the unit tests and the tutorial programs, and builds this documentation, on every push.


Visualization
-------------

``biest::WriteVTK`` writes a surface and a field on it to VTK files: ``WriteVTK("B", Svec, B)`` creates ``B.pvtp`` and ``B_000000.vtp``.
Open the ``.pvtp`` file in `ParaView <https://www.paraview.org>`_:

.. code-block:: bash

   ./bin/force-free-fields-example && paraview B1.pvtp


Features and Capabilities
-------------------------

The underlying `SCTL library <https://sctl.readthedocs.io/>`_ provides
containers
(`Vector <https://sctl.readthedocs.io/en/latest/tutorial/vector.html>`_,
`Matrix <https://sctl.readthedocs.io/en/latest/tutorial/matrix.html>`_),
`GMRES <https://sctl.readthedocs.io/en/latest/tutorial/gmres.html>`_ for solving linear systems,
`profiling <https://sctl.readthedocs.io/en/latest/tutorial/profile.html>`_,
`SIMD vectorization <https://sctl.readthedocs.io/en/latest/tutorial/vec.html>`_,
`FFT <https://sctl.readthedocs.io/en/latest/tutorial/fft.html>`_ and
`SDC (Spectral Deferred Correction) <https://sctl.readthedocs.io/en/latest/tutorial/sdc.html>`_ for high order time-stepping.

The BIEST library extends these features with:

- :ref:`Surface class <tutorial-surface>` for representing toroidal surfaces on a uniform grid in the toroidal and poloidal angles.

- :ref:`VacuumField and TaylorState classes <tutorial-taylor-state>` for computing vacuum fields and Taylor states (force-free fields) inside a toroidal surface or in a toroidal shell between two surfaces.

- :ref:`ExtVacuumField and IntVacuumField classes <tutorial-ext-vacuum>` for computing vacuum fields outside or inside a surface with a given normal component, with toroidal field-period symmetry.

- :ref:`BoundaryIntegralOp and FieldPeriodBIOp classes <tutorial-layer-potentials>` for evaluating layer potentials (Laplace, Biot--Savart, Helmholtz and Stokes kernels) on toroidal surfaces to high order accuracy, at on-surface and off-surface points.

- :ref:`SurfaceOp class <tutorial-surface-op>` for spectral differential operators on surfaces (gradient, divergence, curl, Laplace--Beltrami operator and its inverse, Hodge decomposition) and for Fourier resampling.
