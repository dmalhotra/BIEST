..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial_index:

Tutorials
=========

The tutorials cover the public interface of BIEST. Each one ends with a complete program; the programs that are
also in the repository can be built with the ``Makefile`` (see :ref:`Getting Started <biest_getting-started>`).
The others can be compiled in the same way, for example:

.. code-block:: bash

   g++ -std=c++17 -fopenmp -O3 -march=native -I include -I extern/SCTL/include \
       -DSCTL_HAVE_FFTW tutorial.cpp -o tutorial -lfftw3

Geometry
--------

.. toctree::
   :maxdepth: 1

   Defining surfaces <surface>
   Surface operators <surface-op>

Field Solvers
-------------

.. toctree::
   :maxdepth: 1

   Vacuum fields and Taylor states <taylor-state>
   Vacuum fields with given normal component <ext-vacuum>

Layer Potentials
----------------

.. toctree::
   :maxdepth: 1

   Evaluating layer potentials <layer-potentials>
   Layer potentials with field-period symmetry <field-period>
