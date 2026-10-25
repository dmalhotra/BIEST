..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _api_index:


API Reference
=============

All classes and functions are in the ``biest::`` namespace, and ``#include <biest.hpp>`` makes all of them available.
This reference covers the public interface; :ref:`Data layout <api-data-layout>` below describes the array ordering used by every function.

Field Solvers
-------------

.. toctree::
   :maxdepth: 1

   bie_solvers
   vacuum-field

Geometry and Surface Operators
------------------------------

.. toctree::
   :maxdepth: 1

   surface
   surface_op

Layer Potentials
----------------

.. toctree::
   :maxdepth: 1

   boundary_integ_op
   offsurface_op
   kernel


.. _api-data-layout:

Data layout
-----------

- **Surface grids.** A function on a toroidal surface is sampled on a uniform :math:`N_t \times N_p` grid, where
  index :math:`t` is the toroidal direction and :math:`p` the poloidal direction. Grid point :math:`(t,p)` is at
  toroidal angle :math:`2\pi t/N_t` and poloidal angle :math:`2\pi p/N_p`. The values are stored in row-major order
  (toroidal index outer, poloidal index inner).
- **Vector fields on a grid** are stored as a structure of arrays (SoA): all :math:`x`-components, then all
  :math:`y`-components, then all :math:`z`-components,

  .. code-block:: text

     X = {x11, x12, ..., x1Np, x21, ..., xNtNp,  y11, ..., yNtNp,  z11, ..., zNtNp}

  so that component :math:`k` of grid point :math:`(t,p)` is ``X[(k*Nt+t)*Np+p]``. Surface coordinates, magnetic
  fields and surface currents all use this layout.
- **Several surfaces** (``sctl::Vector<Surface<Real>>``): the arrays of the surfaces are concatenated, each in the
  layout above.
- **Field periods.** ``ExtVacuumField``, ``IntVacuumField`` and ``FieldPeriodBIOp`` take data on one toroidal field
  period, :math:`\theta \in [0, 2\pi/N_{fp})`, with :math:`N_t` the grid dimension in one period. The other
  classes take data on the full torus. The toroidal index may increase in either direction about the :math:`z`-axis;
  vector fields on one period follow the direction of their surface grid.
- **Off-surface target points.** ``VacuumField::EvalOffSurface`` and ``TaylorState::EvalOffSurface`` take and return
  arrays of structures (AoS), ``{x1, y1, z1, x2, y2, z2, ...}``. ``ExtVacuumField::EvalOffSurface``,
  ``IntVacuumField::EvalOffSurface``, ``BoundaryIntegralOp::EvalOffSurface`` and ``OffSurfaceOp`` use SoA,
  ``{x1, x2, ..., y1, y2, ..., z1, z2, ...}``.
- **Units.** BIEST uses units in which :math:`\mu_0 = 1`.
