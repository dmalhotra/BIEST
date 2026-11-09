..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _tutorial-surface-op:

Surface Operators
=================

The ``SurfaceOp`` class provides spectral operations on functions given on the grid of a toroidal surface: differentiation, normals and quadrature weights, surface differential operators, a Laplace--Beltrami solver, the Hodge decomposition, and Fourier resampling.
The solvers use them internally; they are also useful for pre- and post-processing.
For the methods see :ref:`Surface discretization <algorithms-discretization>`, and for the full API :ref:`surface_op.hpp <surface_op_hpp>`.

1. Construct the operator
-------------------------

A ``SurfaceOp`` is constructed for a grid size. All functions take data in the SoA layout of the surface coordinates, with any number of components:

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: biest::SurfaceOp<Real> surf_op(comm
   :end-at: biest::SurfaceOp<Real> surf_op(comm
   :dedent: 2

2. Derivatives, normals and quadrature weights
----------------------------------------------

``Grad2D`` computes the derivatives with respect to the surface parameters :math:`t,p\in[0,1)` (with a negative sign; see the API reference).
The other operators take this ``dX`` of the surface coordinates as input.

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: sctl::Vector<Real> dX, normal, area_elem;
   :end-at: surf_op.SurfNormalAreaElem(
   :dedent: 2

``normal`` holds the unit normals, oriented out of the surface. ``area_elem`` holds the quadrature weights of the trapezoidal rule, so that ``SurfInteg(I, area_elem, F)`` computes the integral of ``F`` over the surface.

3. Differential operators
-------------------------

``SurfGrad`` computes the surface gradient and ``SurfDiv`` the surface divergence of a tangential vector field, so together they give the Laplace--Beltrami operator, which ``SurfLap(LapF, dX, F)`` also computes in one call:

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: surf_op.SurfGrad(GradF
   :end-at: surf_op.SurfDiv(LapF
   :dedent: 4

4. The Laplace--Beltrami equation
---------------------------------

``InvSurfLap`` solves :math:`\Delta_\Gamma u = f - \bar f` for the mean-zero solution :math:`u`, where :math:`\bar f` is the mean of :math:`f`, with GMRES to the given tolerance:

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: sctl::Vector<Real> u, Lu, f0;
   :end-at: surf_op.InvSurfLap(
   :dedent: 4

``GradInvSurfLap`` returns :math:`\nabla_\Gamma u` instead.

5. The Hodge decomposition
--------------------------

``HodgeDecomp`` splits a vector field into its normal part, a surface gradient, a divergence-free part :math:`\mathbf{n}\times\nabla_\Gamma\beta`, and a harmonic part :math:`\mathbf{V}_h`, which has zero surface divergence and zero surface curl:

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: sctl::Vector<Real> Vn, Vd, Vc, Vh
   :end-at: surf_op.HodgeDecomp(
   :dedent: 4

The solvers compute the harmonic part of the toroidal tangent vector in this way, to build surface currents that carry a net current around the torus.

6. Resampling
-------------

``Upsample`` moves data between grids of different sizes by Fourier interpolation; here the surface and the harmonic vector field are moved to a finer grid for visualization:

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: biest::SurfaceOp<Real>::Upsample(X,
   :end-at: biest::SurfaceOp<Real>::Upsample(Vh,
   :dedent: 4

.. _tutorial-surface-op-nfp:

7. One field period
-------------------

A ``SurfaceOp`` can act on one of the ``NFP`` field periods of a surface with that symmetry (see :ref:`Field periods <tutorial-surface-nfp>`): it is constructed with the grid of the period, ``Nt/NFP`` by ``Np`` (the first ``Nt/NFP`` rows of the grid of the torus), the number of field periods and the direction of the toroidal index, and takes the restrictions to the period of the data on the full torus.
The Laplace--Beltrami solve of a function that repeats from one field period to the next gives the restriction of the solution on the full torus, at a fifth of the cost for ``NFP = 5``:

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
   :start-at: sctl::Vector<Real> u_full, u_period;
   :end-at: op_period.InvSurfLap(
   :dedent: 4

``Grad2D`` and ``SurfGrad`` of functions that repeat, ``SurfDiv`` and ``SurfCurl`` of vector fields that rotate with the field periods (such as the tangent vectors, the normals and the surface gradients), ``SurfNormalAreaElem``, ``SurfLap``, ``SurfInteg``, ``ProjZeroMean``, ``InvSurfLap``, ``GradInvSurfLap`` and ``HodgeDecomp`` act on one field period in this way.
The differential operators are in the intrinsic form: they differentiate only the metric of the surface and the components of vector fields along the tangent vectors, which repeat from one field period to the next, and never the Cartesian components of a vector, which rotate (see :ref:`Surface differential operators <algorithms-surface-diff-ops>`).
The derivatives of the coordinates are computed on the full torus, where the coordinates are periodic, and restricted to the period.

Complete Example Code
---------------------

The program ``doc/examples/surface-op.cpp`` checks that the integral of the Laplace--Beltrami operator of a function over the closed surface vanishes, solves a Laplace--Beltrami equation on a stellarator surface and checks the residual, computes the harmonic part of the toroidal tangent vector, and solves a Laplace--Beltrami equation on one field period of a W7-X surface. It prints (the last digits may differ between runs, and the first and last values are at the level of rounding errors)

.. code-block:: text

   integral of lap(f) / integral of |lap(f)| = 1.06103e-17
   Laplace-Beltrami residual: 2.89596e-07
   max |div(Vh)| / max |div(V)| = 9.01078e-08
   one field period vs full torus: 7.97279e-15

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../examples/surface-op.cpp
   :language: c++
