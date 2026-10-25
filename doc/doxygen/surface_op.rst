..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _surface_op_hpp:

surface_op.hpp
==============

This header file provides the ``SurfaceOp`` class: spectral (FFT-based) differential operators, quadrature and resampling for functions on a toroidal surface given on a uniform periodic grid.

Classes and Types
-----------------

.. doxygenclass:: biest::SurfaceOp
   :members: SurfaceOp, Upsample, Resample, RotateToroidal, CompleteVecField, ToroidalOrientation, Grad2D, SurfNormalAreaElem, SurfGrad, SurfDiv, SurfLap, SurfCurl, SurfInteg, ProjZeroMean, InvSurfLap, GradInvSurfLap, InvSurfLapPrecond, HodgeDecomp, EvalSurfInteg

**Usage guide**: :ref:`Surface operators <tutorial-surface-op>`

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/surface_op.hpp
   :language: c++
