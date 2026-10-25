..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _vacuum-field_hpp:

vacuum-field.hpp
================

This header file provides solvers for vacuum fields outside or inside a toroidal surface with a prescribed normal component on the surface.
The surface and the fields must have toroidal field-period symmetry, and all data is given on one field period, :math:`\theta \in [0, 2\pi/N_{fp})`.

The resolution of the surface (``surf_Nt x surf_Np``) and of the magnetic field (``Nt x Np``) are independent.
The field grid point :math:`(t,p)` is the point of the surface at toroidal angle :math:`2\pi t/(N_{fp} N_t)` (in the direction of the toroidal index of the surface) and poloidal angle :math:`2\pi p/N_p`.

Classes and Types
-----------------

.. doxygenclass:: biest::ExtVacuumField
   :members: ExtVacuumField, Setup, ComputeBdotN, ComputeBplasma, ComputeU, EvalOffSurface

``Jplasma`` is the net toroidal current of :math:`\mathbf{J}`, which flows in the direction of **decreasing** toroidal grid index :math:`t`.
For example, if :math:`t` increases counterclockwise about the :math:`z`-axis and :math:`\mathbf{B}_{coil}` is the field of a current loop inside the torus carrying a current :math:`I` counterclockwise about the :math:`z`-axis, then ``ComputeBplasma(ComputeBdotN(Bcoil), I)`` returns :math:`\mathbf{B}_{plasma} = -\mathbf{B}_{coil}` outside the surface.

**Usage guide**: :ref:`Vacuum fields with given normal component <tutorial-ext-vacuum>`

|

.. doxygenclass:: biest::IntVacuumField
   :members: IntVacuumField, Setup, ComputeBdotN, ComputeB, ComputeU, EvalOffSurface

**Usage guide**: :ref:`Vacuum fields with given normal component <tutorial-ext-vacuum>`

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/vacuum-field.hpp
   :language: c++
