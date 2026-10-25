..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _algorithms-discretization:

Surface Discretization and Operators
====================================

This page describes how BIEST represents surfaces and functions on them, and the surface operators that the
solvers need ([MCIO2019]_, Sec. 3.1, 3.3, 3.4). The implementation is in the classes ``Surface`` and ``SurfaceOp``.

Uniform grid
------------

Each toroidal surface :math:`\Gamma` is given by a doubly periodic map
:math:`\mathbf{x}(\theta,\phi)`, where :math:`\theta` is the toroidal angle and :math:`\phi` the poloidal angle.
The surface, and every function on it, is sampled on a uniform :math:`N_\theta\times N_\phi` grid,

.. math::

   \mathsf{f}_{ij} = f(\theta_i,\phi_j), \qquad
   \theta_i = \frac{2\pi i}{N_\theta},\quad \phi_j = \frac{2\pi j}{N_\phi},

and is represented by its trigonometric interpolant,

.. math::

   f(\theta,\phi) \approx \sum_{n=-N_\theta/2}^{N_\theta/2}\ \sum_{m=-N_\phi/2}^{N_\phi/2}
   \widehat{\mathsf{f}}_{nm}\, e^{i(n\theta+m\phi)},

where the coefficients :math:`\widehat{\mathsf{f}}_{nm}` are computed with the FFT. The same grid is used for the
surface coordinates, for the unknown densities and for the computed field on the boundary. The unknowns are the
grid values, and the boundary conditions are imposed at the grid points (a Nyström discretization), while
operations such as differentiation are carried out in Fourier space.
For smooth surfaces and smooth data, all these approximations converge spectrally (faster than any power of
the grid spacing).

The grid is the reason BIEST suits stellarator geometries: the shapes are smooth and are often given as Fourier
series in the two angles (e.g. by VMEC). It is not suited to surfaces with sharp features, such as divertor
geometries ([MCIO2019]_, Sec. 5).

Spectral differentiation and resampling
---------------------------------------

Derivatives are computed by multiplying the Fourier coefficients by :math:`in` and :math:`im` and transforming back,
in :math:`\mathcal{O}(N\log N)` operations (``SurfaceOp::Grad2D``, with FFTW when available).
From the tangent vectors :math:`\mathbf{x}_\theta,\mathbf{x}_\phi`, the unit normal and the metric tensor are
computed at each grid point,

.. math::

   \mathbf{n} = \frac{\mathbf{x}_\theta\times\mathbf{x}_\phi}{\|\mathbf{x}_\theta\times\mathbf{x}_\phi\|},\qquad
   \mathbf{G} = \begin{bmatrix}
     \mathbf{x}_\theta\cdot\mathbf{x}_\theta & \mathbf{x}_\theta\cdot\mathbf{x}_\phi\\
     \mathbf{x}_\theta\cdot\mathbf{x}_\phi & \mathbf{x}_\phi\cdot\mathbf{x}_\phi
   \end{bmatrix},

and the area element is :math:`\sqrt{\det\mathbf{G}} = \|\mathbf{x}_\theta\times\mathbf{x}_\phi\|`
(``SurfaceOp::SurfNormalAreaElem``). The normals are flipped if needed so that they point out of the surface;
BIEST checks the normal at the grid point with the largest :math:`x`-coordinate. So the grid may have either
orientation. Data given on one field period is extended to the full torus by rotating it about the :math:`z`-axis
in the direction in which its toroidal index increases (``SurfaceOp::ToroidalOrientation``).

Resampling to a finer or coarser grid (``SurfaceOp::Upsample``, ``SurfaceOp::Resample``) pads or truncates the
Fourier coefficients. The quadratures use it to upsample the sources, and the solvers use it to move data
between the grids of the surface, of the field and of the quadrature.

Surface integrals and the trapezoidal rule
------------------------------------------

The integral of a smooth function over the surface is computed with the trapezoidal rule on the grid,

.. math::

   \int_\Gamma f\,\mathrm{d}a \approx \sum_{i,j} \mathsf{f}_{ij}\,\mathsf{w}_{ij},\qquad
   \mathsf{w}_{ij} = \frac{4\pi^2}{N_\theta N_\phi}\sqrt{\det\mathbf{G}_{ij}},

which converges spectrally for smooth periodic integrands (``SurfaceOp::SurfInteg``). This rule is also the
far-field part of the :ref:`singular quadrature <algorithms-singular-quadrature>`. Line integrals along
grid lines (for example, the circulations that impose the flux conditions) also use the trapezoidal rule.

Surface differential operators
------------------------------

With :math:`\mathbf{J} = [\mathbf{x}_\theta\ \mathbf{x}_\phi]`, the surface gradient, divergence and the
Laplace--Beltrami operator are ([MCIO2019]_, Eq. 3.13--3.15)

.. math::

   \nabla_\Gamma f = \mathbf{J}\,\mathbf{G}^{-1}
   \begin{bmatrix}\partial_\theta f\\ \partial_\phi f\end{bmatrix},\qquad
   \nabla_\Gamma\cdot\mathbf{v} = \frac{1}{\sqrt{\det\mathbf{G}}}
   \begin{bmatrix}\partial_\theta & \partial_\phi\end{bmatrix}
   \sqrt{\det\mathbf{G}}\,\begin{bmatrix}v_1\\ v_2\end{bmatrix},\qquad
   \Delta_\Gamma f = \nabla_\Gamma\cdot\nabla_\Gamma f,

where :math:`\mathbf{v} = v_1\mathbf{x}_\theta + v_2\mathbf{x}_\phi` is a tangential vector field. They are
applied pointwise on the grid with spectral derivatives (``SurfGrad``, ``SurfDiv``, ``SurfLap``).

Laplace--Beltrami solver
------------------------

The Taylor state formulation and the Hodge decomposition need the inverse of :math:`\Delta_\Gamma` on mean-zero
functions. The null space of :math:`\Delta_\Gamma` is the constant functions, so BIEST solves

.. math::

   \Delta_\Gamma u + \int_\Gamma u\,\mathrm{d}a\,/\,|\Gamma| = f - \bar f ,

whose solution :math:`u` has zero mean (:math:`\bar f` is the area-weighted mean of :math:`f`). The added rank-one
term makes the operator invertible. The system is solved with GMRES. It is preconditioned with the exact inverse
of the Laplacian on a flat torus ([MCIO2019]_, Sec. 3.3.1): in Fourier space,

.. math::

   \widehat{u}_{nm} = \frac{\widehat{f}_{nm}}{n^2/L_\theta^2 + m^2/L_\phi^2}\ \ (n,m)\neq(0,0),\qquad
   \widehat{u}_{00} = 0,

where :math:`L_\theta` and :math:`L_\phi` are the mean lengths of :math:`\mathbf{x}_\theta` and
:math:`\mathbf{x}_\phi` over the surface, which accounts for the different grid spacing in the two directions.
(The overall sign and scaling of the preconditioner do not change the GMRES iterations.)
The preconditioner costs two FFTs. In the tests of [MCIO2019]_ (Table 4), the number of GMRES iterations grows
slowly with the resolution, from 7 to 37 as the error decreases from :math:`10^{-2}` to :math:`10^{-7}`; without a
preconditioner, GMRES did not converge in 300 iterations.

This solver is ``SurfaceOp::InvSurfLap``; ``SurfaceOp::GradInvSurfLap`` also applies :math:`\nabla_\Gamma` to the
solution. Both can solve on an upsampled grid and downsample the result.
``InvSurfLapPrecond`` accepts other preconditioners; [MCIO2019]_ (Sec. 3.3.2) also tests preconditioning with the
Laplace single-layer operator, which needs fewer iterations but costs much more per iteration.

.. _algorithms-hodge:

Hodge decomposition and harmonic vector fields
----------------------------------------------

A tangential vector field :math:`\mathbf{v}` on the surface has the Hodge decomposition

.. math::

   \mathbf{v} = \nabla_\Gamma\alpha + \mathbf{n}\times\nabla_\Gamma\beta + \mathbf{v}_H,\qquad
   \Delta_\Gamma\alpha = \nabla_\Gamma\cdot\mathbf{v},\qquad
   \Delta_\Gamma\beta = -\nabla_\Gamma\cdot(\mathbf{n}\times\mathbf{v}),

where :math:`\mathbf{v}_H` is a harmonic vector field: :math:`\nabla_\Gamma\cdot\mathbf{v}_H = 0` and
:math:`\nabla_\Gamma\cdot(\mathbf{n}\times\mathbf{v}_H) = 0`. On a torus (genus one) the harmonic vector
fields form a two-dimensional space. ``SurfaceOp::HodgeDecomp`` computes the decomposition with two
Laplace--Beltrami solves (it also returns the normal part of a field that is not tangential).

The solvers compute a harmonic vector field from the toroidal tangent vector, :math:`\mathbf{v} = \mathbf{x}_\theta`
([MCIO2019]_, Sec. 3.4). Its harmonic part :math:`\mathbf{v}_H` and :math:`\mathbf{n}\times\mathbf{v}_H`, which is
also harmonic and is orthogonal to :math:`\mathbf{v}_H` at every point, span the harmonic fields.
They are used as divergence-free surface currents that carry a net current around the torus:

- ``TaylorState`` uses :math:`\mathbf{m}_H = \mathbf{v}_H + i\,\mathbf{n}\times\mathbf{v}_H`
  (see :ref:`Taylor states <algorithms-taylor-states>`).
- ``ExtVacuumField`` uses :math:`\mathbf{v}_H`, scaled to carry a given net toroidal current
  (see :ref:`Vacuum fields <algorithms-vacuum-fields>`).
