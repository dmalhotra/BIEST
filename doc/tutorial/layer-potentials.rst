..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-layer-potentials:

Evaluating Layer Potentials
===========================

The ``FieldPeriodBIOp`` class evaluates layer potentials

.. math::

   U(\mathbf{x}) = \int_\Gamma K(\mathbf{x}-\mathbf{y})\,F(\mathbf{y})\,\mathrm{d}a(\mathbf{y})

on toroidal surfaces, with high-order accuracy at the grid points of the surface (where the kernel is singular) and at points off the surface.
It is the building block of the solvers, and can be used directly, for example for the virtual casing principle.
This page uses a surface that declares no field periods: the densities are given on the full torus and the potential is computed at all the grid points (for surfaces with field periods, see :ref:`Layer potentials with field-period symmetry <tutorial-field-period>`).
For the method see :ref:`Singular quadrature <algorithms-singular-quadrature>`, and for the full API :ref:`boundary_integ_op.hpp <boundary_integ_op_hpp>` and :ref:`kernel.hpp <kernel_hpp>`.

1. Set up the operator
----------------------

The number of components of the density and of the potential are those of the kernel given to ``SetupSingular``.
``SetupSingular`` takes the vector of surfaces, the kernel and a tolerance, and precomputes the singular corrections at the targets, the grid points of one field period of each surface: here all the grid points.
The tolerance sets the quadrature grid, the surface grid upsampled at least twice in each direction, and the singular quadrature on it, with the anisotropy of the grid (see :ref:`Upsampling <algorithms-upsampling>` and :ref:`Choosing the parameters <algorithms-quadrature-params>`); a ``SingularQuadParams`` after it gives the quadrature instead.
With several surfaces, the potential of each surface at the grid points of the others is the trapezoidal rule, except at the grid points too close to the surface for it, where it is computed as by ``EvalOffSurface``, to the same tolerance (see :ref:`Interactions between surfaces <algorithms-between-surfaces>`).
The code on this page is from the :ref:`virtual casing example <tutorial-virtual-casing>` below, which uses the layer potentials of the Biot--Savart kernel (3 density components, 3 potential components) and of the gradient of the Laplace single layer (1 density component, 3 potential components):

.. literalinclude:: ../examples/layer-potentials.cpp
   :language: c++
   :start-at: const Real tol = 1e-12;
   :end-at: LaplaceGrad.SetupSingular(
   :dedent: 2

The kernel objects of ``Laplace3D``, ``BiotSavart3D`` and ``Stokes3D`` are static.
A ``Helmholtz3D`` object holds its wavenumber and must outlive the operators that use its kernels: for example, ``biest::Helmholtz3D<Real> helmholtz(1.0)`` and an operator with 2 density and 2 potential components (complex values) set up with ``helmholtz.FxU()``.

2. Evaluate on the surface
--------------------------

``Eval`` takes the densities (here ``NxB`` and ``BdotN``), the number of field periods they are given on (here 1, the full torus) and their grid (here the surface grid), and returns the potentials at the targets; both are in the same layout as the surface coordinates:

.. literalinclude:: ../examples/layer-potentials.cpp
   :language: c++
   :start-at: sctl::Vector<Real> curlS, gradS;
   :end-at: LaplaceGrad.Eval(gradS, BdotN, 1, Nt, Np);
   :dedent: 4

If the potential vector already has the right size, the potential is added to it. Several densities can be stored one after the other; the potentials are then returned in the same order.
For kernels of principal-value type (the gradient of the single layer, the double layer, Biot--Savart), the result is the principal value.

3. Evaluate off the surface
---------------------------

``EvalOffSurface`` evaluates the potential at points ``Xt`` (in SoA order, ``{x1, x2, ..., y1, ..., z1, ...}``) at any distance from the surface, to the requested accuracy. It must be given the same kernel object as ``SetupSingular``, and the densities as ``Eval``:

.. literalinclude:: ../examples/layer-potentials.cpp
   :language: c++
   :start-at: BiotSavart.EvalOffSurface(
   :end-at: LaplaceGrad.EvalOffSurface(
   :dedent: 4

.. _tutorial-virtual-casing:

Example: the virtual casing principle
-------------------------------------

The magnetic field :math:`\mathbf{B}` on a closed surface :math:`\Gamma` is produced by currents inside (the plasma) and outside (the coils).
The virtual casing principle gives the part of the field produced by the currents inside, from :math:`\mathbf{B}` on :math:`\Gamma` alone (see :ref:`Virtual casing principle <algorithms-virtual-casing>`):

.. math::

   \mathbf{B}_{plasma} = \tfrac{1}{2}\mathbf{B} + \nabla\times\mathcal{S}_0[\mathbf{n}\times\mathbf{B}] - \nabla\mathcal{S}_0[\mathbf{n}\cdot\mathbf{B}]
   \ \text{ on }\Gamma,\qquad
   \mathbf{B}_{plasma} = \nabla\times\mathcal{S}_0[\mathbf{n}\times\mathbf{B}] - \nabla\mathcal{S}_0[\mathbf{n}\cdot\mathbf{B}]
   \ \text{ outside }\Gamma .

The densities are :math:`\mathbf{n}\cdot\mathbf{B}` and :math:`\mathbf{n}\times\mathbf{B}`:

.. literalinclude:: ../examples/layer-potentials.cpp
   :language: c++
   :start-at: // The densities n.B and n x B
   :end-before: // Layer-potential operators
   :dedent: 2

With the potentials ``curlS`` and ``gradS`` of step 2, the plasma field on the surface is

.. literalinclude:: ../examples/layer-potentials.cpp
   :language: c++
   :start-at: const sctl::Vector<Real> Bvc = B * 0.5
   :end-at: const sctl::Vector<Real> Bvc = B * 0.5
   :dedent: 4

In the complete program ``doc/examples/layer-potentials.cpp`` below, :math:`\mathbf{B}` is the sum of the fields of a current loop inside a torus and a current loop outside it. The program computes the plasma part on the surface and at points 0.05 outside it, and compares it with the field of the inner loop:

.. code-block:: text

   on the surface:  relative error = 1.62961e-13
   off the surface: relative error = 4.7769e-10

The `virtual-casing <https://github.com/hiddenSymmetries/virtual-casing>`_ library, which is built on BIEST, implements the same formulas with field-period and stellarator symmetry, and has Python bindings.

Compiling and Running the Code
------------------------------

The example code ``src/quadrature-example.cpp`` evaluates the Laplace single-layer potential of a density on a torus and writes the density and the potential to VTK files:

.. code-block:: bash

   make bin/quadrature-example && ./bin/quadrature-example

Complete Example Code
---------------------

``src/quadrature-example.cpp``:

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../src/quadrature-example.cpp
   :language: c++

The virtual casing example:

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../examples/layer-potentials.cpp
   :language: c++
