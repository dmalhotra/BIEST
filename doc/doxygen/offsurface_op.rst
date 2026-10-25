..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _offsurface_op_hpp:

offsurface_op.hpp
=================

This header file provides an operator that evaluates layer potentials of a toroidal surface at target points off the surface, at any distance from it, including points close to the surface.

Classes and Types
-----------------

.. doxygenclass:: biest::OffSurfaceOp
   :members: OffSurfaceOp, Setup, Eval, PanelCount

The constructor takes one of the BIEST kernel functions in :ref:`kernel.hpp <kernel_hpp>`, e.g. ``OffSurfaceOp op(biest::Laplace3D<double>::FxdU())``.
The surface and the densities, given on the periodic grid, are interpolated to high-order quadrilateral panels, and the layer potential is computed with the adaptive near-singular quadrature of SCTL's ``BoundaryIntegralOp`` (see :ref:`Off-surface evaluation <algorithms-off-surface>`).

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/offsurface_op.hpp
   :language: c++
