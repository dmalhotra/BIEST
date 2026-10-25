..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-field-period:

Layer Potentials with Field-Period Symmetry
===========================================

Stellarators are invariant under rotation by :math:`2\pi/N_{fp}` about the :math:`z`-axis, where :math:`N_{fp}` is the number of field periods.
The ``FieldPeriodBIOp`` class evaluates layer potentials for such surfaces, with densities that have the same symmetry: the density is given on one field period and the potential is computed on one field period, which reduces the cost of the singular quadrature by the factor :math:`N_{fp}`.
Unlike ``BoundaryIntegralOp``, it chooses the quadrature parameters and the quadrature grid at run time from the requested number of digits (see :ref:`Choosing the parameters <algorithms-quadrature-params>`).
For the full API see :ref:`boundary_integ_op.hpp <boundary_integ_op_hpp>`.

1. Build the surface
--------------------

The surface coordinates are given on one field period, :math:`\theta\in[0,2\pi/N_{fp})`, on an ``Nt x Np`` grid in SoA order.
``BuildSurface`` rotates them to build the full torus:

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
   :start-at: // Laplace double-layer operator
   :end-at: Svec[0] = biop.BuildSurface(
   :dedent: 2

2. Set up the operator
----------------------

``SetupSingular`` takes the surface, the kernel, the number of digits, the number of field periods, the resolution of the densities on the full torus (``NFP*Nt x Np``), and the resolution of the potential on one field period (``Nt x Np``):

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
   :start-at: biop.SetupSingular(
   :end-at: biop.SetupSingular(
   :dedent: 2

The densities are resampled to a quadrature grid of ``QuadNt() x QuadNp()`` points on the full torus, chosen so that the quadrature reaches the requested accuracy.
The grid can also be set explicitly with the optional last two arguments.

3. Evaluate
-----------

``Eval`` takes the density on one field period, with its number of field periods and resolution, and returns the potential on the ``Nt x Np`` grid of the first field period:

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
   :start-at: sctl::Vector<Real> F(Nt * Np), U;
   :end-at: biop.Eval(U, F, NFP, Nt, Np);
   :dedent: 2

The double-layer potential of a constant density equals :math:`-1/2` on the surface, which the complete program checks.
A 3-component density, such as a surface current, is rotated as a Cartesian vector from one field period to the next.

4. Visualize
------------

This version of ``WriteVTK`` takes the surface and the data on one field period, and writes all field periods:

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
   :start-at: biest::WriteVTK("U"
   :end-at: biest::WriteVTK("U"
   :dedent: 2

Complete Example Code
---------------------

The program ``doc/examples/field-period.cpp`` prints

.. code-block:: text

   quadrature grid: 300 x 90
   max |U + 1/2| = 3.01326e-12

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
