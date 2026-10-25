..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _algorithms-vacuum-fields:

Vacuum Fields and the Virtual Casing Principle
==============================================

This page describes the formulations solved by ``ExtVacuumField`` and ``IntVacuumField``, and the virtual casing
principle, an application of the layer-potential operators ([MCOT2020]_, Sec. 2).

Exterior vacuum field
---------------------

In free-boundary equilibrium calculations and in coil design, one needs the vacuum field outside the plasma
boundary :math:`\Gamma` that makes :math:`\Gamma` a flux surface of the total field. Given the normal component
:math:`\mathbf{B}_{coil}\cdot\mathbf{n}` of an external field on :math:`\Gamma` and the net toroidal plasma current
:math:`I_p`, ``ExtVacuumField`` finds the field :math:`\mathbf{B}_{plasma}` outside :math:`\Gamma` with

.. math::

   \nabla\times\mathbf{B}_{plasma} = 0,\quad \nabla\cdot\mathbf{B}_{plasma} = 0 \ \text{ outside }\Gamma,\qquad
   \mathbf{B}_{plasma}\cdot\mathbf{n} = -\mathbf{B}_{coil}\cdot\mathbf{n} \ \text{ on }\Gamma,\qquad
   \oint_{C}\mathbf{B}_{plasma}\cdot\mathrm{d}\mathbf{l} = I_p,

where :math:`C` is a loop around the cross-section of the torus (with :math:`\mu_0 = 1`) and
:math:`\mathbf{B}_{plasma}\to 0` at infinity. The field is represented as

.. math::

   \mathbf{B}_{plasma} = -\nabla\mathcal{S}_0[\sigma] + \nabla\times\mathcal{S}_0[\mathbf{J}],
   \qquad \mathbf{J} = I_p\,\mathbf{j}_H ,

where :math:`\mathbf{j}_H` is the harmonic part of the toroidal tangent vector
(see :ref:`Hodge decomposition <algorithms-hodge>`), scaled to carry a unit net current around the
torus. The second term is the Biot--Savart field of the surface current :math:`\mathbf{J}`, which provides the
circulation; the first term is curl-free and single-valued, and has zero circulation. The limit on :math:`\Gamma`
from outside gives the second-kind integral equation for the scalar density :math:`\sigma`,

.. math::

   -\frac{\sigma}{2} + \mathbf{n}\cdot\nabla\mathcal{S}_0[\sigma] =
   \mathbf{B}_{coil}\cdot\mathbf{n} + \mathbf{n}\cdot\nabla\times\mathcal{S}_0[\mathbf{J}]
   \qquad\text{on }\Gamma,

with principal-value integrals on the left and the exterior limit on the right. It is solved with GMRES.

Interior vacuum field
---------------------

``IntVacuumField`` finds a vacuum field :math:`\mathbf{B}` inside :math:`\Gamma` with
:math:`\mathbf{B}\cdot\mathbf{n} = -\mathbf{B}_1\cdot\mathbf{n}` on :math:`\Gamma` and toroidal circulation :math:`I_0`
(the current through the hole of the torus). The circulation is provided by the field of a straight wire on the
:math:`z`-axis:

.. math::

   \mathbf{B} = -\nabla\mathcal{S}_0[\sigma] + \frac{I_0}{2\pi R}\,\hat{\boldsymbol{\phi}},
   \qquad
   \frac{\sigma}{2} + \mathbf{n}\cdot\nabla\mathcal{S}_0[\sigma] =
   \mathbf{B}_1\cdot\mathbf{n} + \frac{I_0}{2\pi R}\,\hat{\boldsymbol{\phi}}\cdot\mathbf{n}
   \qquad\text{on }\Gamma,

where :math:`R` is the distance from the :math:`z`-axis.

These are indirect formulations: the unknown is a density :math:`\sigma`, not the scalar potential. [MCOT2020]_
(Sec. 2.2) describes the direct formulation based on Green's identity used in other codes (such as NESTOR); both
lead to second-kind integral equations with the same singular kernels.

Field-period symmetry
---------------------

A stellarator with :math:`N_{fp}` field periods is invariant under rotation by :math:`2\pi/N_{fp}` about the
:math:`z`-axis, and so are the fields considered here. The vacuum field classes take the surface and the data on
one field period, and compute the solution on one field period, with ``FieldPeriodBIOp``:

- the densities are extended to the full torus by rotation;
- the layer potentials are computed only at the target points of one field period, which reduces the number of
  targets, and the cost and memory of the singular corrections, by a factor :math:`N_{fp}`;
- the quadrature parameters and the quadrature grid are chosen from the requested number of digits
  (see :ref:`Choosing the parameters <algorithms-quadrature-params>`), and the GMRES tolerance is
  :math:`10^{-\text{digits}}`.

The resolutions of the surface and of the field may differ: the surface is resampled to the field grid and to the
quadrature grid with the FFT.

.. _algorithms-virtual-casing:

Virtual casing principle
------------------------

The field :math:`\mathbf{B}` of a plasma equilibrium is produced by the plasma currents inside :math:`\Gamma` and by
the coils outside. The virtual casing principle computes the plasma part :math:`\mathbf{B}_{plasma}` from the total
field on :math:`\Gamma` alone, without the plasma current density: it replaces the plasma by surface currents and
charges on :math:`\Gamma` that produce the same field outside. In terms of layer potentials,

.. math::

   \mathbf{B}_{plasma}(\mathbf{x}) =
   \nabla\times\mathcal{S}_0[\mathbf{n}\times\mathbf{B}](\mathbf{x}) - \nabla\mathcal{S}_0[\mathbf{n}\cdot\mathbf{B}](\mathbf{x})
   \qquad \mathbf{x}\ \text{outside}\ \Gamma,

and on :math:`\Gamma`, with principal-value integrals,

.. math::

   \mathbf{B}_{plasma}(\mathbf{x}) = \frac{1}{2}\mathbf{B}(\mathbf{x}) +
   \nabla\times\mathcal{S}_0[\mathbf{n}\times\mathbf{B}](\mathbf{x}) - \nabla\mathcal{S}_0[\mathbf{n}\cdot\mathbf{B}](\mathbf{x})
   \qquad \mathbf{x}\in\Gamma .

On a flux surface (:math:`\mathbf{B}\cdot\mathbf{n} = 0`), the normal component reduces to the formula used in fusion
codes ([MCOT2020]_, Eq. 2.2),

.. math::

   \mathbf{B}_{plasma}\cdot\mathbf{n}(\mathbf{x}) = \frac{1}{4\pi}\,\mathbf{n}(\mathbf{x})\cdot\int_\Gamma
   \frac{(\mathbf{n}(\mathbf{y})\times\mathbf{B}(\mathbf{y}))\times(\mathbf{x}-\mathbf{y})}{|\mathbf{x}-\mathbf{y}|^3}\,\mathrm{d}a(\mathbf{y}),

and the external field has :math:`\mathbf{B}_{ext}\cdot\mathbf{n} = -\mathbf{B}_{plasma}\cdot\mathbf{n}`, which is the
input to coil design.
The integrals on :math:`\Gamma` are the Biot--Savart and the Laplace gradient layer potentials, computed with
``BoundaryIntegralOp`` and the kernels ``BiotSavart3D::FxU`` and ``Laplace3D::FxdU``; off the surface,
``BoundaryIntegralOp::EvalOffSurface`` computes them at any distance. ``src/virtual-casing-principle.cpp`` contains
several variants, and the `virtual-casing <https://github.com/hiddenSymmetries/virtual-casing>`_ library implements these formulas with
field-period symmetry.

In [MCOT2020]_ (Sec. 4.2), the normal component computed with this method for a Taylor state in W7-X converges
spectrally to about 10 digits (Table 1). The layer-potential formulas above hold on any closed surface; the classical
formula for the normal component drops the :math:`\mathbf{n}\cdot\mathbf{B}` term and needs a flux surface, which
[MCOT2020]_ (Sec. 4.3, Table 2) obtains by first adding an exterior vacuum field as above.
