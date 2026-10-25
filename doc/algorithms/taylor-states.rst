..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _algorithms-taylor-states:

Taylor States and Vacuum Fields in a Toroidal Domain
====================================================

This page describes the formulations solved by ``TaylorState`` and ``VacuumField`` ([MCIO2019]_, Sec. 2 and 3).

The problem
-----------

Taylor states are the magnetic fields of minimal energy for a given magnetic helicity. They are force-free
fields, :math:`\nabla\times\mathbf{B} = \lambda\mathbf{B}`, and are used in the stepped-pressure equilibrium model
of stellarators (as in the SPEC code), where each region between two interfaces holds a Taylor state.
Let :math:`\Omega` be the region inside a toroidal surface, or between two nested toroidal surfaces, with boundary
:math:`\Gamma = \Gamma_1\cup\dots\cup\Gamma_{N_s}` (:math:`N_s = 1` or 2) and outward unit normal :math:`\mathbf{n}`.
For a given real :math:`\lambda`, BIEST solves ([MCIO2019]_, Eq. 2.11)

.. math::

   \nabla\times\mathbf{B} = \lambda\mathbf{B} \ \text{ in }\Omega,\qquad
   \mathbf{B}\cdot\mathbf{n} = 0 \ \text{ on }\Gamma,\qquad
   \int_{S_i}\mathbf{B}\cdot\mathrm{d}\mathbf{a} = \Phi_i,\quad i = 1,\dots,N_s .

The boundary conditions alone do not determine the field: one flux condition is needed for each boundary
surface. For a torus, :math:`\Phi_1` is the toroidal flux, through a cross-section at constant toroidal angle.
For a shell, the second condition is the poloidal flux, through a ribbon that connects the two surfaces at constant
poloidal angle. The problem has no unique solution when :math:`\lambda` is an eigenvalue of the curl operator on
fields tangent to :math:`\Gamma`; away from these values the formulation below is well conditioned.
The case :math:`\lambda = 0` is a vacuum field, which has a separate formulation
(:ref:`below <algorithms-taylor-states-vacuum>`).

Generalized Debye source representation
---------------------------------------

The field is represented by two surface densities, a scalar :math:`\sigma` and a tangential vector field
:math:`\mathbf{m}` (generalized Debye sources, from the work of Epstein and Greengard on Maxwell's equations)
([MCIO2019]_, Eq. 2.13):

.. math::

   \mathbf{B} = i\lambda\,\mathcal{S}_\lambda[\mathbf{m}] - \nabla\mathcal{S}_\lambda[\sigma] + i\,\nabla\times\mathcal{S}_\lambda[\mathbf{m}],
   \qquad
   \mathcal{S}_\lambda[f](\mathbf{x}) = \int_\Gamma \frac{e^{i\lambda|\mathbf{x}-\mathbf{y}|}}{4\pi|\mathbf{x}-\mathbf{y}|} f(\mathbf{y})\,\mathrm{d}a(\mathbf{y}),

where :math:`\mathcal{S}_\lambda` is the single-layer potential of the Helmholtz equation with wavenumber
:math:`\lambda`. This field satisfies :math:`\nabla\times\mathbf{B} = \lambda\mathbf{B}` and
:math:`\nabla\cdot\mathbf{B} = 0` if :math:`\nabla_\Gamma\cdot\mathbf{m} = i\lambda\sigma`. With the additional condition
:math:`\mathbf{n}\times\mathbf{m} = -i\,\mathbf{m}`, the vector density is determined by :math:`\sigma` up to a
harmonic vector field ([MCIO2019]_, Eq. 2.14--2.16):

.. math::

   \mathbf{m} = \mathbf{m}_0(\sigma) + \sum_{k=1}^{N_s}\alpha_k\,\mathbf{m}_H^k,\qquad
   \mathbf{m}_0(\sigma) = i\lambda\left(\nabla_\Gamma\Delta_\Gamma^{-1}\sigma + i\,\mathbf{n}\times\nabla_\Gamma\Delta_\Gamma^{-1}\sigma\right),

where :math:`\mathbf{m}_H^k = \mathbf{v}_H + i\,\mathbf{n}\times\mathbf{v}_H` on surface :math:`k` (and zero on the other
surface) is built from the harmonic part :math:`\mathbf{v}_H` of the toroidal tangent vector of surface :math:`k`
(see :ref:`Hodge decomposition <algorithms-hodge>`). The unknowns are the complex density :math:`\sigma` and
the :math:`N_s` coefficients :math:`\alpha_k`. Computing :math:`\mathbf{m}_0(\sigma)` requires a Laplace--Beltrami solve
for the real and for the imaginary part of :math:`\sigma`.

Boundary integral equation
--------------------------

The limit of the field on :math:`\Gamma` from inside :math:`\Omega` is ([MCIO2019]_, Eq. 2.17)

.. math::

   \mathbf{B} = -\frac{\sigma}{2}\mathbf{n} + \frac{\mathbf{m}}{2} + i\lambda\,\mathcal{S}_\lambda[\mathbf{m}]
   - \nabla\mathcal{S}_\lambda[\sigma] + i\,\nabla\times\mathcal{S}_\lambda[\mathbf{m}]
   \qquad\text{on }\Gamma,

where the integrals are principal values (and :math:`i\,\mathbf{n}\times\mathbf{m}/2 = \mathbf{m}/2`). The
condition :math:`\mathbf{B}\cdot\mathbf{n} = 0` gives a second-kind integral equation ([MCIO2019]_, Eq. 2.18--2.19),

.. math::

   -\frac{\sigma}{2} + i\lambda\,\mathbf{n}\cdot\mathcal{S}_\lambda[\mathbf{m}] - \mathbf{n}\cdot\nabla\mathcal{S}_\lambda[\sigma]
   + i\,\mathbf{n}\cdot\nabla\times\mathcal{S}_\lambda[\mathbf{m}] = 0,
   \qquad \mathbf{m} = \mathbf{m}_0(\sigma) + \textstyle\sum_k\alpha_k\mathbf{m}_H^k .

The layer potentials are computed with the :ref:`singular quadrature <algorithms-singular-quadrature>` for the
kernels :math:`g_\lambda` and :math:`\nabla g_\lambda`; :math:`\nabla\times\mathcal{S}_\lambda` is assembled from
:math:`\nabla\mathcal{S}_\lambda` applied to each component of :math:`\mathbf{m}`.

Flux conditions
---------------

By Stokes' theorem and :math:`\mathbf{B} = \lambda^{-1}\nabla\times\mathbf{B}`, the flux through a cross-section
:math:`S` equals :math:`\lambda^{-1}` times the circulation of :math:`\mathbf{B}` along its boundary
:math:`\partial S\subset\Gamma`, so only loops on the boundary are needed. The terms
:math:`-\sigma\mathbf{n}/2 - \nabla\mathcal{S}_\lambda[\sigma]` have zero circulation and drop out. A direct
evaluation of the rest would divide by :math:`\lambda` and lose accuracy for small :math:`\lambda`. Since the
Laplace part :math:`\mathbf{m}_H/2 + i\nabla\times\mathcal{S}_0[\mathbf{m}_H]` has zero circulation (a lemma in
[MCIO2019]_), it can be subtracted, which gives the flux conditions ([MCIO2019]_, Eq. 2.23)

.. math::

   \oint_{\partial S_i} i\,\mathcal{S}_\lambda[\mathbf{m}]\cdot\mathrm{d}\mathbf{l}
   + \frac{1}{\lambda}\oint_{\partial S_i}\left(\frac{\mathbf{m}_0}{2} + i\,\nabla\times\mathcal{S}_\lambda[\mathbf{m}_0]\right)\cdot\mathrm{d}\mathbf{l}
   + \oint_{\partial S_i} i\,\nabla\times\left(\frac{\mathcal{S}_\lambda-\mathcal{S}_0}{\lambda}\right)[\mathbf{m}_H]\cdot\mathrm{d}\mathbf{l}
   = \Phi_i .

All terms remain bounded as :math:`\lambda\to 0`, because :math:`\mathbf{m}_0 = \mathcal{O}(\lambda)` and the kernel
:math:`(g_\lambda - g_0)/\lambda` is bounded. The last term uses the ``HelmholtzDiff3D`` kernel. The circulations
are computed with the trapezoidal rule along the grid lines and averaged over all grid lines of each family.

Linear solve
------------

The discrete system has the block form ([MCIO2019]_, Eq. 3.2--3.5)

.. math::

   \begin{bmatrix}\mathbf{A}_{11} & \mathbf{A}_{12}\\ \mathbf{A}_{21} & \mathbf{A}_{22}\end{bmatrix}
   \begin{bmatrix}\boldsymbol{\sigma}\\ \boldsymbol{\alpha}\end{bmatrix} =
   \begin{bmatrix}\mathbf{0}\\ \boldsymbol{\Phi}\end{bmatrix},

where the first row is the integral equation at the :math:`N` grid points and the second row the :math:`N_s` flux
conditions. BIEST solves it as follows:

1. For each surface :math:`k`, solve :math:`\mathbf{A}_{11}\boldsymbol{\sigma}_k = -\mathbf{A}_{12}\mathbf{e}_k` with GMRES
   (relative tolerance ``gmres_tol``). This gives the field :math:`\mathbf{B}_k` with :math:`\alpha_k = 1`, the other
   :math:`\alpha` zero, and :math:`\mathbf{B}_k\cdot\mathbf{n} = 0`. Each GMRES iteration applies
   :math:`\mathbf{A}_{11}`: two Laplace--Beltrami solves (tolerance ``LB_tol*gmres_tol/lambda``) for
   :math:`\mathbf{m}_0(\sigma)`, and the layer potentials.
2. Compute the fluxes of each :math:`\mathbf{B}_k`, and solve the small :math:`N_s\times N_s` system for the
   combination with the requested fluxes. The real part of this combination is the Taylor state.

The densities :math:`\sigma` and :math:`\mathbf{m}` of the basis fields can be returned by ``TaylorState::Compute``,
and ``TaylorState::EvalOffSurface`` evaluates the representation above at points inside :math:`\Omega`
(see :ref:`Off-surface evaluation <algorithms-off-surface>`).

The cost of each stage, for :math:`N_{gmres}` iterations and :math:`N_{LB}` Laplace--Beltrami iterations, is
([MCIO2019]_, Table 2): setup :math:`\mathcal{O}(N q^2)`; trapezoidal sums :math:`\mathcal{O}(N_s N_{gmres} N^2)`;
singular corrections :math:`\mathcal{O}(N_s N_{gmres} N R^2)`; Laplace--Beltrami solves
:math:`\mathcal{O}(N_s N_{gmres} N_{LB} N\log N)`. For large :math:`N` the trapezoidal sums dominate.

.. _algorithms-taylor-states-vacuum:

Vacuum fields (:math:`\lambda = 0`)
-----------------------------------

For :math:`\lambda = 0`, ``VacuumField`` uses a formulation with Laplace kernels and no Laplace--Beltrami solve
inside the iteration ([MCIO2019]_, Sec. 2.2.4). For each surface :math:`k`:

1. Build a divergence-free surface current :math:`\mathbf{J}_k` on surface :math:`k` from its tangent vectors,
   by removing the surface-gradient part with one Laplace--Beltrami solve. This current carries a net current
   around the torus, so its Biot--Savart field :math:`\mathbf{B}_0 = \nabla\times\mathcal{S}_0[\mathbf{J}_k]` has
   nonzero circulation.
2. Correct its normal component with the gradient of a single-layer potential,
   :math:`\mathbf{B}_k = \mathbf{B}_0 + \nabla\mathcal{S}_0[\sigma]`, where :math:`\sigma` solves the second-kind
   integral equation

   .. math::

      \frac{\sigma}{2} + \mathbf{n}\cdot\nabla\mathcal{S}_0[\sigma] = -\,\mathbf{n}\cdot\mathbf{B}_0 \qquad\text{on }\Gamma

   with GMRES (principal value on the left, and the limit from inside :math:`\Omega` for :math:`\mathbf{B}_0`).

3. Compute the fluxes of :math:`\mathbf{B}_k`. With the surface current :math:`\mathbf{j} = \mathbf{B}_k\times\mathbf{n}`,
   the field inside :math:`\Omega` is the Biot--Savart field :math:`\mathbf{B}_k = \nabla\times\mathcal{S}_0[\mathbf{j}]`
   (and zero outside). So by Stokes' theorem the flux through a cross-section is the circulation of the vector potential
   :math:`\mathcal{S}_0[\mathbf{j}]` along its boundary ([MCIO2019]_, Eq. 2.27).

The field with the requested fluxes is the combination of the :math:`\mathbf{B}_k` that matches them.
``VacuumField::Compute`` returns the field and the surface current :math:`\mathbf{j}`, and
``VacuumField::EvalOffSurface`` evaluates the Biot--Savart field of :math:`\mathbf{j}` at points inside :math:`\Omega`.

Results
-------

[MCIO2019]_ (Sec. 4) reports spectral convergence of the full solver:

- Taylor state inside the QAS3 stellarator (Table 5): for :math:`\lambda = 0.5` the error decreases from
  :math:`2.6\times 10^{-2}` (:math:`N = 8.8\times 10^3`) to :math:`4.4\times 10^{-9}` (:math:`N = 6.1\times 10^5`)
  and the number of GMRES iterations grows slowly, from 24 to 66; for :math:`\lambda = 1` the error decreases from
  :math:`3.3\times 10^{-2}` to :math:`6.5\times 10^{-9}`, with 34 to 88 iterations.
- Taylor state in a toroidal shell between two rotating ellipses, :math:`\lambda = 1` (Table 6): the error decreases
  from :math:`7.3\times 10^{-3}` to :math:`1.4\times 10^{-8}`.
- W7-X, compared with SPEC (Table 7): at equal accuracy, BIEST was faster than SPEC, by more than an order of
  magnitude for the vacuum field (for example, error :math:`1.6\times 10^{-3}` in 11 s with BIEST and 527 s with
  SPEC), and reached errors near :math:`10^{-9}`, where SPEC became ill-conditioned.

The programs ``src/table5.cpp``, ``src/table6.cpp`` and ``src/table7.cpp`` reproduce these tests.
