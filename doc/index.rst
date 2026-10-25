..
   Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
   Author: Dhairya Malhotra
   SPDX-License-Identifier: Apache-2.0

.. _biest_documentation:

BIEST: Boundary Integral Equation Solver for Taylor states
==========================================================

.. only:: html

   .. image:: https://github.com/dmalhotra/BIEST/actions/workflows/test.yml/badge.svg
      :target: https://github.com/dmalhotra/BIEST/actions/workflows/test.yml
      :alt: Test status

   .. image:: https://badgen.net/github/tag/dmalhotra/BIEST
      :target: https://github.com/dmalhotra/BIEST/tags
      :alt: Stable version

   .. image:: https://img.shields.io/github/v/release/dmalhotra/BIEST?color=%233D9970
      :target: https://github.com/dmalhotra/BIEST/releases
      :alt: Latest release

   .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.23511944.svg
      :target: https://doi.org/10.5281/zenodo.23511944
      :alt: DOI

Author: **Dhairya Malhotra** (lead author).

`BIEST <https://github.com/dmalhotra/BIEST>`_ is a header-only C++ library of high-order boundary integral equation solvers on toroidal surfaces,
for Taylor states (force-free fields, :math:`\nabla \times \mathbf{B} = \lambda \mathbf{B}`) and vacuum magnetic fields in stellarators.
It computes the magnetic field inside a toroidal surface, or in a toroidal shell between two nested toroidal surfaces,
given the toroidal flux (and the poloidal flux for a shell) and the condition :math:`\mathbf{B} \cdot \mathbf{n} = 0` on the boundary.
The field is represented by layer potentials on the boundary, so only the boundary is discretized,
and the unknowns are found by solving a boundary integral equation with GMRES.

The main features are:

- **Spectral discretization** of the surfaces and of the unknowns on a uniform grid in the toroidal and poloidal angles,
  with FFT-based differentiation, resampling, and surface operators (Laplace--Beltrami solver, Hodge decomposition).
- **High-order singular quadrature** for weakly singular layer potentials, with a partition of unity around each target point
  and a polar-coordinate quadrature for the singular part, applied as a sparse correction to the trapezoidal rule.
- **Generalized Debye source** formulation for Taylor states, which gives a second-kind integral equation, well conditioned for values of :math:`\lambda` away from the eigenvalues of the curl operator in the domain.
- **Vacuum field solvers** for the regions inside and outside a surface with a given :math:`\mathbf{B} \cdot \mathbf{n}`, which exploit toroidal field-period symmetry.
- **Layer potential operators** for the Laplace, Biot--Savart, Helmholtz and Stokes kernels on toroidal surfaces, at on-surface and off-surface points,
  for applications such as the virtual casing principle.
- **SIMD vectorization and OpenMP parallelism**, through the `SCTL library <https://sctl.readthedocs.io/>`_.

The methods are described in the :ref:`Algorithms <algorithms-overview>` section and in the papers below.
It is research software; use at your own risk.

How these pages are organized
-----------------------------

- **User Guide**: :ref:`Getting Started <biest_getting-started>` (download, compiler flags, optional libraries,
  example codes and unit tests) and a series of :ref:`tutorials <tutorial_index>` that use only the public interface.
- **Algorithms** describes the discretization, the singular quadrature and the integral equation formulations, with
  references to the papers.
- **API Reference** lists the public classes and functions, and the array layout they use.


.. _biest_citing:

Citing BIEST
------------

If you use BIEST, or code derived from it, please cite the software and the papers below:

- D. Malhotra, *BIEST: Boundary Integral Equation Solver for Taylor states*, https://github.com/dmalhotra/BIEST, `doi:10.5281/zenodo.23511944 <https://doi.org/10.5281/zenodo.23511944>`__.
- D. Malhotra, T. Li, R. Blackwell, A. Rahimian, W. Yan, X. Gao, *SCTL: Scientific Computing Template Library*, https://github.com/dmalhotra/SCTL, `doi:10.5281/zenodo.23511629 <https://doi.org/10.5281/zenodo.23511629>`__.
- D. Malhotra, A. Cerfon, L.-M. Imbert-Gérard, M. O'Neil, "Taylor states in stellarators: A fast high-order boundary integral solver," *Journal of Computational Physics* 397 (2019) 108791, `doi:10.1016/j.jcp.2019.06.067 <https://doi.org/10.1016/j.jcp.2019.06.067>`__, `arXiv:1902.01205 <https://arxiv.org/abs/1902.01205>`__.
- D. Malhotra, A. J. Cerfon, M. O'Neil, E. Toler, "Efficient high-order singular quadrature schemes in magnetic fusion," *Plasma Physics and Controlled Fusion* 62 (2020) 024004, `doi:10.1088/1361-6587/ab57f4 <https://doi.org/10.1088/1361-6587/ab57f4>`__, `arXiv:1909.07417 <https://arxiv.org/abs/1909.07417>`__.

.. raw:: html

   <details>
   <summary>BibTeX</summary>

.. code-block:: bibtex

   @software{biest,
     author    = {Malhotra, Dhairya},
     title     = {{BIEST}: Boundary Integral Equation Solver for {Taylor} states},
     url       = {https://github.com/dmalhotra/BIEST},
     doi       = {10.5281/zenodo.23511944},
     version   = {1.0.0},
     year      = {2026}
   }

   @software{sctl,
     author    = {Malhotra, Dhairya and Li, Tianyue and Blackwell, Robert and Rahimian, Abtin and Yan, Wen and Gao, Xuanzhao},
     title     = {{SCTL}: Scientific Computing Template Library},
     url       = {https://github.com/dmalhotra/SCTL},
     doi       = {10.5281/zenodo.23511629},
     version   = {2.0.0},
     year      = {2026}
   }

   @article{malhotra2019taylor,
     author  = {Malhotra, Dhairya and Cerfon, Antoine and Imbert-G{\'e}rard, Lise-Marie and O'Neil, Michael},
     title   = {Taylor states in stellarators: A fast high-order boundary integral solver},
     journal = {Journal of Computational Physics},
     volume  = {397},
     pages   = {108791},
     year    = {2019},
     doi     = {10.1016/j.jcp.2019.06.067}
   }

   @article{malhotra2020quadrature,
     author  = {Malhotra, Dhairya and Cerfon, Antoine J. and O'Neil, Michael and Toler, Evan},
     title   = {Efficient high-order singular quadrature schemes in magnetic fusion},
     journal = {Plasma Physics and Controlled Fusion},
     volume  = {62},
     number  = {2},
     pages   = {024004},
     year    = {2020},
     doi     = {10.1088/1361-6587/ab57f4}
   }

.. raw:: html

   </details>


License
-------

BIEST is licensed under the `Apache License 2.0 <https://github.com/dmalhotra/BIEST/blob/master/LICENSE>`_.
The `NOTICE <https://github.com/dmalhotra/BIEST/blob/master/NOTICE>`_ file lists the authors and must be kept in any redistribution, including ports to other languages.
It also has the authors' requests on crediting them and on AI use.



.. toctree::
   :maxdepth: 1
   :hidden:

   Introduction <self>

.. toctree::
   :caption: User Guide
   :maxdepth: 1
   :hidden:

   getting-started
   tutorial/index

.. toctree::
   :caption: Algorithms
   :maxdepth: 1
   :hidden:

   algorithms/index
   algorithms/discretization
   algorithms/singular-quadrature
   algorithms/taylor-states
   algorithms/vacuum-fields

.. toctree::
   :caption: API Reference
   :maxdepth: 1
   :hidden:

   doxygen/index
