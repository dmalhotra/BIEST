..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-ext-vacuum:

Vacuum Fields with Given Normal Component
=========================================

The classes ``ExtVacuumField`` and ``IntVacuumField`` compute vacuum fields (:math:`\nabla\times\mathbf{B} = 0`, :math:`\nabla\cdot\mathbf{B} = 0`) outside or inside a toroidal surface, with a given normal component on the surface:

- ``ExtVacuumField``: given :math:`\mathbf{B}_{coil}\cdot\mathbf{n}` on the surface and a net toroidal current :math:`I_p`, it computes the field :math:`\mathbf{B}_{plasma}` outside the surface with :math:`\mathbf{B}_{plasma}\cdot\mathbf{n} = -\mathbf{B}_{coil}\cdot\mathbf{n}` that decays at infinity. Then :math:`\mathbf{B}_{coil}+\mathbf{B}_{plasma}` is tangent to the surface, as in free-boundary equilibrium calculations.
- ``IntVacuumField``: given :math:`\mathbf{B}_1\cdot\mathbf{n}` on the surface and the current :math:`I_0` through the hole of the torus, it computes the field :math:`\mathbf{B}` inside the surface with :math:`\mathbf{B}\cdot\mathbf{n} = -\mathbf{B}_1\cdot\mathbf{n}`.

Both classes use the toroidal field-period symmetry of stellarators: the surface and the fields are given and computed on one field period.
For the methods see :ref:`Vacuum fields <algorithms-vacuum-fields>`, and for the full API :ref:`vacuum-field.hpp <vacuum-field_hpp>`.
The interface uses ``std::vector<Real>``.

1. Data on one field period
---------------------------

For a surface with ``NFP`` field periods, the coordinates are given on the toroidal interval :math:`[0, 2\pi/N_{fp})`, on a ``surf_Nt x surf_Np`` grid, in the usual SoA order.
The magnetic field is given on a separate ``Nt x Np`` grid on the same interval:
grid point :math:`(t,p)` is the point of the surface at toroidal angle :math:`2\pi t/(N_{fp}N_t)` (in the direction of the toroidal index of the surface) and poloidal angle :math:`2\pi p/N_p`.
The two grids are independent; the surface is resampled with the FFT.

.. literalinclude:: ../examples/ext-vacuum.cpp
   :language: c++
   :start-at: const int digits = 10, NFP = 5;
   :end-at: const std::vector<Real> XB =
   :dedent: 2

Here ``SurfaceCoord`` evaluates the five-period surface of the :ref:`surface tutorial <tutorial-surface>` (see the complete code below).

2. Set up the solver
--------------------

``Setup`` takes the number of accurate digits requested. It sets the quadrature parameters, the quadrature grid and the GMRES tolerance; the quadratures are set up at the first solve.

.. literalinclude:: ../examples/ext-vacuum.cpp
   :language: c++
   :start-at: biest::ExtVacuumField<Real> ext_field;
   :end-at: ext_field.Setup(
   :dedent: 4

3. Solve
--------

``ComputeBdotN`` computes :math:`\mathbf{B}\cdot\mathbf{n}` on the field grid, with the normal pointing out of the torus.
In this example, :math:`\mathbf{B}_{coil}` is the field of a circular current loop inside the torus, carrying the current :math:`I` counterclockwise about the :math:`z`-axis.
``ComputeBplasma`` returns the field on the surface, the single-layer density :math:`\sigma` and the surface current :math:`\mathbf{J}`:

.. literalinclude:: ../examples/ext-vacuum.cpp
   :language: c++
   :start-at: const Real I = 1.0;
   :end-at: ext_field.ComputeBplasma(
   :dedent: 4

Outside the surface the solution is unique, so :math:`\mathbf{B}_{plasma} = -\mathbf{B}_{coil}`.
The second argument :math:`I_p` of ``ComputeBplasma`` is the net toroidal current of the surface current :math:`\mathbf{J}`, which flows in the direction of decreasing toroidal grid index :math:`t` (here, clockwise about the :math:`z`-axis).
So :math:`I_p = I` gives :math:`\mathbf{J}` the same total current as the loop of :math:`-\mathbf{B}_{coil}`.

4. Evaluate outside the surface
-------------------------------

``EvalOffSurface`` evaluates :math:`\mathbf{B}_{plasma}` at any points outside the surface, also close to it. The points and the result are in SoA order, ``{x1, x2, ..., y1, y2, ..., z1, z2, ...}``.

.. literalinclude:: ../examples/ext-vacuum.cpp
   :language: c++
   :start-at: const std::vector<Real> Xout =
   :end-at: ext_field.EvalOffSurface(
   :dedent: 4

5. Interior fields
------------------

``IntVacuumField`` has the same interface. Its ``ComputeB(B1_dot_N, I0)`` returns the field and the density, and ``EvalOffSurface(Xt, sigma, I0)`` evaluates the field inside the surface.
The field is represented as the gradient of a single-layer potential plus the field :math:`I_0\hat{\boldsymbol\phi}/(2\pi R)` of a straight wire on the :math:`z`-axis, so :math:`I_0` is the circulation of :math:`\mathbf{B}` counterclockwise around the :math:`z`-axis, and the :math:`z`-axis must pass through the hole of the torus.
In the example below, :math:`\mathbf{B}_1` is the field of a current loop outside the torus, and :math:`I_0 = 0`.

.. literalinclude:: ../examples/ext-vacuum.cpp
   :language: c++
   :start-at: biest::IntVacuumField<Real> int_field;
   :end-at: int_field.EvalOffSurface(
   :dedent: 4

Complete Example Code
---------------------

The program ``doc/examples/ext-vacuum.cpp`` checks the results against the exact solutions and prints the relative errors (the last digits may differ between runs):

.. code-block:: text

   exterior, on the surface:  2.58571e-07
   exterior, off the surface: 1.61674e-12
   interior, on the surface:  9.90124e-07
   interior, off the surface: 3.31073e-11

The errors on the surface are limited by the resolution of the field grid (``Nt x Np``); with ``Nt = Np = 80`` they decrease to about :math:`10^{-10}`.

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../examples/ext-vacuum.cpp
   :language: c++
