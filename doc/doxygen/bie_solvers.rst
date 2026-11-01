..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _bie_solvers_hpp:

bie_solvers.hpp
===============

This header file provides solvers for vacuum fields and Taylor states (force-free fields) inside a toroidal surface, or in a toroidal shell between two nested toroidal surfaces.

The template parameter ``UPSAMPLE`` upsamples the grids for the quadrature, and the singular quadrature (see :ref:`Singular quadrature <algorithms-singular-quadrature>`) is chosen for the accuracy ``0.1*gmres_tol`` and the anisotropy of the grid of each surface (see :ref:`Choosing the parameters <algorithms-quadrature-params>`).
An object can be given the quadrature instead (``SingularQuadParams``), and keeps its operators from one call of ``Compute`` to the next; the static ``Compute`` uses a new object.

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
