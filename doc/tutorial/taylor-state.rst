..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-taylor-state:

Vacuum Fields and Taylor States
===============================

The classes ``VacuumField`` and ``TaylorState`` compute magnetic fields inside a toroidal surface, or in a toroidal shell between two nested toroidal surfaces, that are tangent to the boundary (:math:`\mathbf{B}\cdot\mathbf{n} = 0`) and have prescribed fluxes:

- ``VacuumField``: :math:`\nabla\times\mathbf{B} = 0`.
- ``TaylorState``: :math:`\nabla\times\mathbf{B} = \lambda\mathbf{B}` for a given nonzero constant :math:`\lambda` (a force-free field).

This tutorial follows the example code ``src/force-free-fields-example.cpp``.
For the methods see :ref:`Taylor states <algorithms-taylor-states>`, and for the full API :ref:`bie_solvers.hpp <bie_solvers_hpp>`.

1. Define the surface
---------------------

The domain is given by a vector of surfaces: one surface for the region inside a torus, or two nested surfaces for a toroidal shell (see :ref:`Defining surfaces <tutorial-surface>`).

.. literalinclude:: ../../src/force-free-fields-example.cpp
   :language: c++
   :start-at: sctl::Vector<biest::Surface<Real>> Svec(1);
   :end-before: { // Compute vacuum field
   :dedent: 4

2. Compute the vacuum field
---------------------------

``VacuumField::Compute`` returns the magnetic field ``B`` on the surface grid points, and a surface current ``J`` whose Biot--Savart field is ``B`` inside the domain:

.. literalinclude:: ../../src/force-free-fields-example.cpp
   :language: c++
   :start-at: sctl::Vector<Real> B, J;
   :end-at: WriteVTK("B0"
   :dedent: 6

The arguments are the toroidal flux, the poloidal flux (used only for a shell; here 0), the surfaces, the communicator, the GMRES tolerance (here ``1e-4``) and the maximum number of GMRES iterations.
``B`` has the same layout as the surface coordinates: ``B[(k*Nt+t)*Np+p]`` is component ``k`` at grid point ``(t, p)``.
The GMRES iterations stop when the relative residual is below the GMRES tolerance, which, together with the grid resolution and the quadrature parameters below, sets the accuracy.

3. Compute the Taylor state
---------------------------

``TaylorState::Compute`` takes, in addition, the parameter :math:`\lambda`:

.. literalinclude:: ../../src/force-free-fields-example.cpp
   :language: c++
   :start-at: sctl::Vector<Real> B;
   :end-at: WriteVTK("B1"
   :dedent: 6

Inside each GMRES iteration, ``TaylorState`` solves Laplace--Beltrami problems on the surfaces to tolerance ``LB_tol*gmres_tol/lambda`` (the optional argument ``LB_tol`` defaults to 0.1).
:math:`\lambda` must be nonzero, and must not be an eigenvalue of the curl operator for the domain, where the problem has no unique solution.

.. note::

   The two classes use opposite signs for the toroidal flux.
   With a positive ``tor_flux``, ``VacuumField`` gives a field that points toward decreasing toroidal grid index :math:`t`, and ``TaylorState`` a field that points toward increasing :math:`t`.
   To compute fields in the same direction, pass ``-tor_flux`` and ``-pol_flux`` to one of them, as ``src/table7.cpp`` does.

4. Evaluate the field inside the domain
---------------------------------------

To evaluate the field at points inside the domain, keep the surface current ``J`` (vacuum field) or the densities ``m`` and ``sigma`` (Taylor state), and pass them to ``EvalOffSurface``.
The target points and the result are arrays of structures, ``{x1, y1, z1, x2, y2, z2, ...}``.
This is from ``src/example1.cpp``:

.. literalinclude:: ../../src/example1.cpp
   :language: c++
   :start-at: long Ntrg = 2;
   :end-before: sctl::Profile::Toc();
   :dedent: 4

The evaluation is accurate at any distance from the surface (see :ref:`Off-surface evaluation <algorithms-off-surface>`).
Its optional last argument is the accuracy tolerance, ``1e-10`` by default. If the grid does not resolve the surface or the densities to this accuracy, warnings are printed; a tolerance close to the accuracy of the solve avoids them.

5. Toroidal shells
------------------

For the region between two nested surfaces, pass both surfaces, and both the toroidal and the poloidal flux.
The two surfaces may have different grid resolutions; ``B`` holds the field on both grids, one after the other.
This is from ``doc/examples/taylor-state-shell.cpp``:

.. literalinclude:: ../examples/taylor-state-shell.cpp
   :language: c++
   :start-at: sctl::Vector<biest::Surface<Real>> Svec(2);
   :end-at: biest::TaylorState<Real,1,12,24>::Compute(
   :dedent: 2

The program also evaluates the field at a point inside the shell, and prints (the last digits may differ between runs; :math:`B_x` vanishes by symmetry and is at the level of rounding errors)

.. code-block:: text

   B.Dim() = 24336 = 3 * (104 * 52 + 104 * 26)
   B(2.5, 0, 0) = -3.13866e-13 0.474928 -0.110743

.. _tutorial-quadrature-params:

6. Choosing the quadrature parameters
-------------------------------------

The template parameters ``<Real, UPSAMPLE, PATCH_DIM0, RAD_DIM>`` set the singular quadrature (see :ref:`Singular quadrature <algorithms-singular-quadrature>`).
``PATCH_DIM0`` and ``RAD_DIM`` must increase with the grid resolution for the quadrature error to decrease, and each direction of the (upsampled) grid needs at least ``2*PATCH_DIM0+1`` points.
The following values were used in the Taylor states paper for a toroidal domain (QAS3, ``src/table5.cpp``), with the GMRES tolerance and the resulting error of the Taylor state for :math:`\lambda = 1`:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Grid
     - ``PATCH_DIM0``
     - ``RAD_DIM``
     - ``gmres_tol``
     - GMRES iterations
     - Error
   * - :math:`210\times 42`
     - 15
     - 27
     - 9.6e-3
     - 34
     - 3.3e-2
   * - :math:`350\times 70`
     - 20
     - 33
     - 6.5e-4
     - 45
     - 2.6e-3
   * - :math:`700\times 140`
     - 25
     - 42
     - 3.6e-6
     - 63
     - 2.3e-5
   * - :math:`1050\times 210`
     - 30
     - 51
     - 6.8e-8
     - 78
     - 5.5e-7
   * - :math:`1750\times 350`
     - 40
     - 66
     - 1.0e-9
     - 88
     - 6.5e-9

The programs ``src/table6.cpp`` (toroidal shell) and ``src/table7.cpp`` (W7-X) list the values for other cases.
The setup time and memory grow with ``PATCH_DIM0`` and ``RAD_DIM``, and the time of each GMRES iteration grows with the number of grid points squared.

Compiling and Running the Code
------------------------------

From the BIEST root directory (``SurfType::Quas3`` reads ``geom/Quas3-*.mat``):

.. code-block:: bash

   make bin/force-free-fields-example && ./bin/force-free-fields-example

This writes ``B0.pvtp`` (vacuum field) and ``B1.pvtp`` (Taylor state), which can be opened in ParaView.
Similarly, ``make bin/example1 && ./bin/example1 geom/NCSX`` computes the vacuum field for the NCSX boundary and prints it at two points inside.

Complete Example Code
---------------------

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../src/force-free-fields-example.cpp
   :language: c++
