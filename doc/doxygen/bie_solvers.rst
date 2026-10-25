..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _bie_solvers_hpp:

bie_solvers.hpp
===============

This header file provides solvers for vacuum fields and Taylor states (force-free fields) inside a toroidal surface, or in a toroidal shell between two nested toroidal surfaces.

Both classes have only static member functions. The template parameters ``UPSAMPLE``, ``PATCH_DIM0`` and ``RAD_DIM`` select the singular quadrature (see :ref:`Singular quadrature <algorithms-singular-quadrature>`); ``PATCH_DIM0`` is half of the patch width :math:`M` and ``RAD_DIM`` the order :math:`q` in the papers.
Larger ``PATCH_DIM0`` and ``RAD_DIM`` give more accurate quadratures at a higher cost, and should grow with the grid resolution.
Each grid direction of each surface needs at least ``2*PATCH_DIM0+1`` points after upsampling (49 with the defaults).
The example codes use the defaults with a :math:`200 \times 50` grid; the :ref:`tutorial <tutorial-quadrature-params>` lists the values used in the papers for other resolutions.

Classes and Types
-----------------

.. doxygenclass:: biest::VacuumField
   :members: Compute, EvalOffSurface

**Usage guide**: :ref:`Computing vacuum fields and Taylor states <tutorial-taylor-state>`

|

.. doxygenclass:: biest::TaylorState
   :members: Compute, EvalOffSurface

**Usage guide**: :ref:`Computing vacuum fields and Taylor states <tutorial-taylor-state>`

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/bie_solvers.hpp
   :language: c++
