..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _kernel_hpp:

kernel.hpp
==========

This header file provides the kernel functions for the layer-potential operators.
The kernels are SIMD-vectorized wrappers of the `SCTL kernel functions <https://sctl.readthedocs.io/en/latest/doxygen/kernel_functions.html>`_.

With :math:`\mathbf{r} = \mathbf{x} - \mathbf{y}` the difference between a target point :math:`\mathbf{x}` and a source point :math:`\mathbf{y}`, and :math:`\mathbf{n}` the normal at the source point:

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Kernel
     - Dimensions
     - Definition
   * - ``Laplace3D<Real>::FxU()``
     - 1 → 1
     - single layer :math:`\frac{1}{4\pi|\mathbf{r}|}`
   * - ``Laplace3D<Real>::FxdU()``
     - 1 → 3
     - gradient of the single layer with respect to :math:`\mathbf{x}`, :math:`-\frac{\mathbf{r}}{4\pi|\mathbf{r}|^3}`
   * - ``Laplace3D<Real>::Fxd2U()``
     - 1 → 9
     - Hessian of the single layer
   * - ``Laplace3D<Real>::DxU()``
     - 1 → 1
     - double layer :math:`\frac{\mathbf{n}\cdot\mathbf{r}}{4\pi|\mathbf{r}|^3}`
   * - ``Laplace3D<Real>::DxdU()``
     - 1 → 3
     - gradient of the double layer
   * - ``BiotSavart3D<Real>::FxU()``
     - 3 → 3
     - Biot--Savart law :math:`\mathbf{f}\mapsto\frac{\mathbf{f}\times\mathbf{r}}{4\pi|\mathbf{r}|^3}`
   * - ``BiotSavart3D<Real>::FxdU()``
     - 3 → 9
     - gradient of the Biot--Savart field
   * - ``Helmholtz3D<Real>(k).FxU()``
     - 2 → 2
     - Helmholtz single layer :math:`\frac{e^{ik|\mathbf{r}|}}{4\pi|\mathbf{r}|}` (complex values stored as real and imaginary parts)
   * - ``Helmholtz3D<Real>(k).FxdU()``
     - 2 → 6
     - gradient of the Helmholtz single layer
   * - ``Helmholtz3D<Real>(k).DxU()``
     - 2 → 2
     - Helmholtz double layer
   * - ``HelmholtzDiff3D<Real>(k).FxdU()``
     - 2 → 6
     - gradient of :math:`\frac{e^{ik|\mathbf{r}|}-1}{4\pi|\mathbf{r}|}`
   * - ``Stokes3D<Real>::DxU()``
     - 3 → 3
     - Stokes double layer

The kernels evaluate to zero at :math:`\mathbf{r} = 0`. They do not include :math:`\mu_0`.
The ``Laplace3D``, ``BiotSavart3D`` and ``Stokes3D`` kernels are static objects; a ``Helmholtz3D`` or ``HelmholtzDiff3D`` object holds the wavenumber ``k`` and must outlive any operator that uses its kernels.

Classes and Types
-----------------

.. doxygenclass:: biest::KernelFunction
   :members: KernelFunction, Dim, operator(), BuildMatrix

|

.. doxygenclass:: biest::SctlKernelFunction
   :members: SctlKernelFunction, GetSctlKernel

|

.. doxygenclass:: biest::Laplace3D
   :members: FxU, FxdU, Fxd2U, DxU, DxdU

|

.. doxygenclass:: biest::BiotSavart3D
   :members: FxU, FxdU

|

.. doxygenclass:: biest::Helmholtz3D
   :members: Helmholtz3D, FxU, FxdU, DxU

|

.. doxygenclass:: biest::HelmholtzDiff3D
   :members: HelmholtzDiff3D, FxdU

|

.. doxygenclass:: biest::Stokes3D
   :members: DxU

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/kernel.hpp
   :language: c++
