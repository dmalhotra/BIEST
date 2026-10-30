..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-surface:

Defining Surfaces
=================

All BIEST solvers work on toroidal surfaces represented by the ``Surface`` class.
This tutorial shows how to construct a surface, access its coordinates, compute normals and areas, and write VTK files for visualization.
For the full API documentation please refer to :ref:`surface.hpp <surface_hpp>`.

The surface grid
----------------

A ``Surface<Real>`` holds the coordinates of the points of a uniform grid in the toroidal angle :math:`\theta` and the poloidal angle :math:`\phi`:
``Nt`` points in the toroidal direction and ``Np`` points in the poloidal direction, with point :math:`(t,p)` at
:math:`\theta = 2\pi t/N_t` and :math:`\phi = 2\pi p/N_p`.
The grid covers the full torus, with no repeated end points.
The coordinates are stored as a structure of arrays, all :math:`x`, then all :math:`y`, then all :math:`z`, each in row-major order (toroidal index outer):

.. code-block:: text

   S.Coord() = {x11, x12, ..., x1Np, x21, ..., xNtNp,  y11, ..., yNtNp,  z11, ..., zNtNp}

So coordinate ``k`` of point ``(t, p)`` is ``S.Coord()[(k*Nt+t)*Np+p]``, which can also be accessed as ``S.Coord(t, p, k)``.
Fields on the surface, such as the magnetic field computed by the solvers, use the same layout.

Predefined shapes
-----------------

The constructor ``Surface(Nt, Np, type)`` builds one of the predefined shapes of ``SurfType`` at the given resolution (here ``Nt = 100`` and ``Np = 40``):

.. literalinclude:: ../examples/surface.cpp
   :language: c++
   :start-at: biest::Surface<Real> S0(
   :end-at: biest::Surface<Real> S0(
   :dedent: 2

``W7X_`` is a Wendelstein 7-X boundary given by its Fourier coefficients; the :ref:`API reference <surface_hpp>` lists the other shapes.
The shapes ``Quas3``, ``LHD`` and ``W7X`` read their coordinates from the ``geom/`` directory, relative to the current working directory, so programs that use them must run from the BIEST root directory.

Surfaces from Fourier coefficients
----------------------------------

To define any other surface, construct it with ``SurfType::None`` and set the coordinates.
In stellarator codes such as VMEC, the boundary is given by Fourier coefficients :math:`R_{mn}` and :math:`Z_{mn}` in cylindrical coordinates,

.. math::

   R(\theta,\phi) = \sum_{m,n} R_{mn}\cos(m\phi - N_{fp}\,n\,\theta),\qquad
   Z(\theta,\phi) = \sum_{m,n} Z_{mn}\sin(m\phi - N_{fp}\,n\,\theta),

where :math:`N_{fp}` is the number of field periods. The following builds a five-period stellarator with three nonzero coefficients of each:

.. literalinclude:: ../examples/surface.cpp
   :language: c++
   :start-at: const int NFP = 5;
   :end-before: // Normals, area elements
   :dedent: 2

The example code ``src/example1.cpp`` reads the coefficients ``RBC(n,m)`` and ``ZBS(n,m)`` from a VMEC-style input file (such as ``geom/NCSX``) and builds the surface in the same way.

The grid must resolve the surface: the number of grid points in each direction must be larger than twice the highest Fourier mode in that direction.
The solvers also need a minimum number of points for their quadrature (see :ref:`Vacuum fields and Taylor states <tutorial-taylor-state>`).
Either orientation of the grid is allowed: the solvers orient the normals to point out of each surface.
For data given on one field period (``FieldPeriodBIOp``, ``ExtVacuumField``, ``IntVacuumField`` and the field-period ``WriteVTK``), the other periods are placed in the direction in which the toroidal index increases, which is detected from the surface coordinates.

.. _tutorial-surface-nfp:

Field periods
-------------

A stellarator with :math:`N_{fp}` field periods is invariant under the rotation by :math:`2\pi/N_{fp}` about the :math:`z`-axis.
A surface declares this symmetry with ``SetNFP(NFP)``, as the example above does; ``NFP()`` returns it, 1 by default.
``SetNFP`` checks the symmetry with ``HasSymmetry(NFP, tol)``: the number of toroidal grid points ``NTor()`` must be a multiple of ``NFP``, and the coordinates of each grid point, rotated by :math:`2\pi/N_{fp}`, must be those of the point ``NTor()/NFP()`` grid points further, to within ``tol`` (``1e-10`` by default) relative to the largest coordinate.
The grid points of field period ``j`` are then those with toroidal index in ``[j*NTor()/NFP(), (j+1)*NTor()/NFP())``.
The constructor ``Surface(NFP, Nt, Np, X)`` builds the full torus, ``NFP*Nt`` by ``Np`` points, from the coordinates ``X`` of one field period, and declares the symmetry.

The predefined shapes do not declare their field periods; ``W7X``, ``W7X_`` and ``Stell`` have 5, ``Quas3`` and the rotating ellipses 3, and ``LHD`` (from its file in ``geom/``) 5, when ``NTor()`` is a multiple of that number.

Normals and areas
-----------------

The ``SurfaceOp`` class (see :ref:`Surface operators <tutorial-surface-op>`) computes the unit normals and the area elements, here used to compute the surface area:

.. literalinclude:: ../examples/surface.cpp
   :language: c++
   :start-at: biest::SurfaceOp<Real> surf_op
   :end-at: surf_op.SurfInteg(
   :dedent: 2

Visualization
-------------

``WriteVTK`` writes a surface, optionally with a field on its grid (with any number of components, in the same layout as the coordinates):

.. literalinclude:: ../examples/surface.cpp
   :language: c++
   :start-at: biest::WriteVTK("S0"
   :end-at: biest::WriteVTK("S1"
   :dedent: 2

Open the ``.pvtp`` files in `ParaView <https://www.paraview.org>`_.

Complete Example Code
---------------------

The program ``doc/examples/surface.cpp`` (built with ``make`` in ``doc/examples``) prints ``area = 12.1169``.

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../examples/surface.cpp
   :language: c++
