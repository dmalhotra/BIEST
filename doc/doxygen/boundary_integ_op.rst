..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _boundary_integ_op_hpp:

boundary_integ_op.hpp
=====================

This header file provides operators that evaluate layer potentials

.. math::

   U(\mathbf{x}) = \int_\Gamma K(\mathbf{x} - \mathbf{y}, \mathbf{n}(\mathbf{y}))\, F(\mathbf{y})\, \mathrm{d}a(\mathbf{y})

on toroidal surfaces, for the kernels in :ref:`kernel.hpp <kernel_hpp>`, with the :ref:`singular quadrature <algorithms-singular-quadrature>`.
At a target point on the surface, the integral is the principal value where the kernel requires it.

Classes and Types
-----------------

.. doxygenclass:: biest::BoundaryIntegralOp
   :members: BoundaryIntegralOp, SetupSingular, Dim, operator(), EvalOffSurface

``SetupSingular`` chooses the singular quadrature for its tolerance and the anisotropy of the grid of each surface (see :ref:`Choosing the parameters <algorithms-quadrature-params>`), or takes it from a ``SingularQuadParams``:

.. doxygenstruct:: biest::SingularQuadParams
   :members:

**Usage guide**: :ref:`Evaluating layer potentials <tutorial-layer-potentials>`

|

.. doxygenclass:: biest::FieldPeriodBIOp
   :members: FieldPeriodBIOp, BuildSurface, SetupSingular, Eval, QuadNt, QuadNp

**Usage guide**: :ref:`Layer potentials with field-period symmetry <tutorial-field-period>`

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/boundary_integ_op.hpp
   :language: c++
