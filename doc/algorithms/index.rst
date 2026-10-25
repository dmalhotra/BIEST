..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _algorithms-overview:

Overview
========

BIEST solves boundary value problems for magnetic fields in toroidal domains by representing the field with
layer potentials on the boundary. The pages in this section describe the methods, following the two papers
listed in :ref:`Citing BIEST <biest_citing>` and the code.
Equation and table numbers refer to the arXiv versions of the papers:
`arXiv:1902.01205 <https://arxiv.org/abs/1902.01205>`_ (Taylor states, [MCIO2019]_) and
`arXiv:1909.07417 <https://arxiv.org/abs/1909.07417>`_ (singular quadrature, [MCOT2020]_).

Problems
--------

Let :math:`\Omega` be the region inside a toroidal surface :math:`\Gamma`, or the toroidal shell between two
nested toroidal surfaces :math:`\Gamma = \Gamma_1 \cup \Gamma_2`, and let :math:`\mathbf{n}` be the unit normal
pointing out of :math:`\Omega`. BIEST computes:

- **Taylor states** (:ref:`TaylorState <algorithms-taylor-states>`): the force-free field

  .. math::

     \nabla\times\mathbf{B} = \lambda\mathbf{B} \ \text{ in } \Omega, \qquad
     \mathbf{B}\cdot\mathbf{n} = 0 \ \text{ on } \Gamma, \qquad
     \int_{S_i} \mathbf{B}\cdot \mathrm{d}\mathbf{a} = \Phi_i,

  where :math:`\lambda` is a given real constant (the Beltrami parameter) and :math:`\Phi_i` are the fluxes
  through cross-sections :math:`S_i` of :math:`\Omega`: the toroidal flux for a torus, and the toroidal and
  poloidal fluxes for a shell. The number of flux conditions equals the number of boundary surfaces.

- **Vacuum fields in** :math:`\Omega` (:ref:`VacuumField <algorithms-taylor-states-vacuum>`): the case
  :math:`\lambda = 0` of the problem above, which has its own formulation with Laplace kernels.

- **Vacuum fields with given normal component**
  (:ref:`ExtVacuumField, IntVacuumField <algorithms-vacuum-fields>`): a curl-free and divergence-free field
  outside or inside a single toroidal surface, with prescribed :math:`\mathbf{B}\cdot\mathbf{n}` on the surface
  and a prescribed circulation. These solvers use the toroidal field-period symmetry of stellarators.

- **Layer potentials** (:ref:`BoundaryIntegralOp, FieldPeriodBIOp <algorithms-singular-quadrature>`):
  surface integrals :math:`\int_\Gamma K(\mathbf{x}-\mathbf{y})\,f(\mathbf{y})\,\mathrm{d}a(\mathbf{y})`
  with a singular kernel :math:`K` (Laplace, Biot--Savart, Helmholtz, Stokes), evaluated on the surface or
  off the surface. These are the building blocks of the solvers above, and of applications such as the
  :ref:`virtual casing principle <algorithms-virtual-casing>`.

Components
----------

All the solvers combine three components:

1. :ref:`Discretization <algorithms-discretization>`: each surface and each function on it is sampled on a
   uniform grid in the toroidal and poloidal angles. Derivatives, resampling and surface differential operators
   use the FFT. The Laplace--Beltrami equation and the Hodge decomposition on the surface are solved with
   GMRES and an FFT-based preconditioner.

2. :ref:`Singular quadrature <algorithms-singular-quadrature>`: a layer potential at a point on the surface is
   computed with the trapezoidal rule, plus a precomputed sparse correction on a small patch around the point.
   The correction replaces the trapezoidal rule on the patch with a polar-coordinate quadrature, which
   integrates the singular kernel to high order.

3. **Integral equation solve**: the boundary condition gives a second-kind integral equation for an unknown
   surface density. It is solved with GMRES, where each iteration applies the layer potentials once.

The following table lists the public classes and the method each one implements.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Method
   * - ``Surface``, ``SurfaceOp``
     - :ref:`Spectral discretization and surface operators <algorithms-discretization>`
   * - ``BoundaryIntegralOp``, ``FieldPeriodBIOp``
     - :ref:`Singular quadrature on the surface <algorithms-singular-quadrature>`
   * - ``OffSurfaceOp``, ``EvalOffSurface`` methods
     - :ref:`Quadrature at off-surface points <algorithms-off-surface>`
   * - ``TaylorState``
     - :ref:`Generalized Debye source formulation <algorithms-taylor-states>`
   * - ``VacuumField``
     - :ref:`Vacuum field formulation <algorithms-taylor-states-vacuum>`
   * - ``ExtVacuumField``, ``IntVacuumField``
     - :ref:`Exterior and interior Neumann problems <algorithms-vacuum-fields>`

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Symbol
     - Meaning
   * - :math:`\theta,\ \phi`
     - toroidal and poloidal angle, each in :math:`[0,2\pi)` (the reverse of the names used by VMEC and by virtual-casing)
   * - :math:`N_\theta \times N_\phi`
     - grid dimensions in the toroidal and poloidal direction (``Nt`` and ``Np`` in the code)
   * - :math:`N`
     - total number of grid points on all surfaces
   * - :math:`\mathbf{x}(\theta,\phi)`
     - surface position; :math:`\mathbf{x}_\theta,\mathbf{x}_\phi` are its derivatives
   * - :math:`\mathbf{n}`, :math:`\mathbf{G}`
     - unit normal (pointing out of :math:`\Omega`) and metric tensor; :math:`\sqrt{\det\mathbf{G}}` is the area element
   * - :math:`\nabla_\Gamma,\ \nabla_\Gamma\cdot,\ \Delta_\Gamma`
     - surface gradient, surface divergence, Laplace--Beltrami operator
   * - :math:`g_k(\mathbf{r})`
     - Helmholtz Green's function :math:`e^{ik|\mathbf{r}|}/(4\pi|\mathbf{r}|)`; :math:`g_0` is the Laplace Green's function
   * - :math:`\mathcal{S}_k[f]`
     - single-layer potential :math:`\int_\Gamma g_k(\mathbf{x}-\mathbf{y}) f(\mathbf{y})\,\mathrm{d}a(\mathbf{y})`
   * - :math:`\lambda`
     - Beltrami parameter
   * - :math:`\sigma,\ \mathbf{m}`
     - scalar and vector surface densities (generalized Debye sources)
   * - :math:`M`, :math:`q`
     - width (in grid points) of the partition-of-unity patch, and order of the polar quadrature

References
----------

.. [MCIO2019] D. Malhotra, A. Cerfon, L.-M. Imbert-Gérard, M. O'Neil, "Taylor states in stellarators: A fast high-order boundary integral solver," *Journal of Computational Physics* 397 (2019) 108791, `doi:10.1016/j.jcp.2019.06.067 <https://doi.org/10.1016/j.jcp.2019.06.067>`__, `arXiv:1902.01205 <https://arxiv.org/abs/1902.01205>`__.

.. [MCOT2020] D. Malhotra, A. J. Cerfon, M. O'Neil, E. Toler, "Efficient high-order singular quadrature schemes in magnetic fusion," *Plasma Physics and Controlled Fusion* 62 (2020) 024004, `doi:10.1088/1361-6587/ab57f4 <https://doi.org/10.1088/1361-6587/ab57f4>`__, `arXiv:1909.07417 <https://arxiv.org/abs/1909.07417>`__.
