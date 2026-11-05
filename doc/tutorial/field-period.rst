..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-field-period:

Layer Potentials with Field-Period Symmetry
===========================================

Stellarators are invariant under rotation by :math:`2\pi/N_{fp}` about the :math:`z`-axis, where :math:`N_{fp}` is the number of field periods.
The ``FieldPeriodBIOp`` class evaluates layer potentials for such surfaces, with densities that have the same symmetry: the density is given on one field period and the potential is computed on one field period, which reduces the cost of the singular quadrature by the factor :math:`N_{fp}`.
It chooses the quadrature grid and the quadrature parameters from the requested number of digits, or from a tolerance (see :ref:`Upsampling <algorithms-upsampling>` and :ref:`Choosing the parameters <algorithms-quadrature-params>`).
For the full API see :ref:`boundary_integ_op.hpp <boundary_integ_op_hpp>`.

1. Build the surface
--------------------

The surface coordinates are given on one field period, :math:`\theta\in[0,2\pi/N_{fp})`, on an ``Nt x Np`` grid in SoA order.
The ``Surface`` constructor with the number of field periods rotates them to build the full torus:

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
   :start-at: // Laplace double-layer operator
   :end-at: const biest::Surface<Real> S(NFP, Nt, Np, X);
   :dedent: 2

2. Set up the operator
----------------------

``SetupSingular`` takes the surface, the kernel, the number of digits, and the targets: the ``Nt x Np`` grid on the first ``1/NFP`` of the torus. Two optional arguments give the finest density grid that the quadrature must resolve, counted on the full torus; by default it is the target grid on the full torus (``NFP*Nt x Np``):

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
   :start-at: biop.SetupSingular(
   :end-at: biop.SetupSingular(
   :dedent: 2

The densities are resampled to a quadrature grid of ``QuadNt() x QuadNp()`` points on the full torus, chosen so that the quadrature reaches the requested accuracy.

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

5. Several surfaces
-------------------

``FieldPeriodBIOp`` also takes several surfaces, such as the two surfaces of a toroidal shell, that declare their field periods (``Surface::SetNFP``, see :ref:`Field periods <tutorial-surface-nfp>`); it uses those common to all.
``SetupSingular`` then takes a tolerance instead of the digits: the quadrature grids and the singular quadrature are chosen for ``tol`` (or given in ``quad``, a ``SingularQuadParams``), and the targets are the grid points of one field period of each surface.
The densities of the surfaces are stored one after the other, and ``Eval`` takes them with the field periods and the grid of each:

.. code-block:: c++

   sctl::Vector<sctl::Integer> nfp(2); // for each surface, the field periods of F: here NFP, those common to the surfaces
   sctl::Vector<sctl::Long> Nt(2), Np(2); // and the grid of F on one field period
   for (int s = 0; s < 2; s++) {
     nfp[s] = NFP;
     Nt[s] = Svec[s].NTor() / NFP;
     Np[s] = Svec[s].NPol();
   }
   biest::FieldPeriodBIOp<Real> biop;
   biop.SetupSingular(Svec, biest::Laplace3D<Real>::DxU(), tol);
   biop.Eval(U, F, nfp, Nt, Np);

Each surface gets its own quadrature grid (``QuadNt(s) x QuadNp(s)``).
The potential of a surface at the targets of another is computed as described in :ref:`Interactions between surfaces <algorithms-between-surfaces>`, to the tolerance; ``VacuumField`` and ``TaylorState`` use this setup.

Complete Example Code
---------------------

The program ``doc/examples/field-period.cpp`` prints

.. code-block:: text

   quadrature grid: 300 x 60
   max |U + 1/2| = 4.06425e-11

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../examples/field-period.cpp
   :language: c++
