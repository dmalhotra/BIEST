..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _surface_hpp:

surface.hpp
===========

This header file provides the ``Surface`` class for toroidal surfaces, the predefined surface shapes, and functions for writing VTK visualization files.

Classes and Types
-----------------

.. doxygenclass:: biest::Surface
   :members: Surface, NTor, NPol, Coord

**Usage guide**: :ref:`Defining surfaces <tutorial-surface>`

|

.. doxygenenum:: biest::SurfType

|

Functions
---------

.. doxygenfunction:: biest::WriteVTK(const char *fname, const Surface<Real> &S, const sctl::Vector<Real> F, const sctl::Comm &comm)

.. doxygenfunction:: biest::WriteVTK(const char *fname, const sctl::Vector<Surface<Real>> &Svec, const sctl::Vector<Real> F, const sctl::Comm &comm)

.. doxygenfunction:: biest::WriteVTK(const std::string &fname, const sctl::Integer NFP, bool half_period, const sctl::Long surf_Nt, const sctl::Long surf_Np, const sctl::Vector<Real> &X, const sctl::Long src_Nt, const sctl::Long src_Np, const sctl::Vector<Real> &F)

|

.. raw:: html

   <div style="border-top: 1px solid"></div>
   <br>

.. literalinclude:: ../../include/biest/surface.hpp
   :language: c++
