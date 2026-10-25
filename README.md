# BIEST: Boundary Integral Equation Solver for Taylor states

[![Stable Version](https://badgen.net/github/tag/dmalhotra/BIEST)](https://github.com/dmalhotra/BIEST/tags)
[![Latest Release](https://img.shields.io/github/v/release/dmalhotra/BIEST?color=%233D9970)](https://github.com/dmalhotra/BIEST/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23511944.svg)](https://doi.org/10.5281/zenodo.23511944)

### Minimum requirements to compile:
C++ compiler that supports C++17 standard

### Running the tests:
`make -j test` builds and runs the unit tests in `test/`.


### Optional libraries for better performance:
BLAS, LAPACK, FFTW, LIBXSMM (see the Makefile)


### Citing BIEST

If you use BIEST, or code derived from it, please cite the software and the papers below (also listed in [CITATION.cff](CITATION.cff)):

- D. Malhotra, *BIEST: Boundary Integral Equation Solver for Taylor states*, https://github.com/dmalhotra/BIEST, [doi:10.5281/zenodo.23511944](https://doi.org/10.5281/zenodo.23511944).
- D. Malhotra, A. Cerfon, L.-M. Imbert-Gérard, M. O'Neil, "Taylor states in stellarators: A fast high-order boundary integral solver," *Journal of Computational Physics* 397 (2019) 108791, [doi:10.1016/j.jcp.2019.06.067](https://doi.org/10.1016/j.jcp.2019.06.067).
- D. Malhotra, A. J. Cerfon, M. O'Neil, E. Toler, "Efficient high-order singular quadrature schemes in magnetic fusion," *Plasma Physics and Controlled Fusion* 62 (2020) 024004, [doi:10.1088/1361-6587/ab57f4](https://doi.org/10.1088/1361-6587/ab57f4).

<details>
<summary>BibTeX</summary>

```bibtex
@software{biest,
  author    = {Malhotra, Dhairya},
  title     = {{BIEST}: Boundary Integral Equation Solver for {Taylor} states},
  url       = {https://github.com/dmalhotra/BIEST},
  doi       = {10.5281/zenodo.23511944},
  version   = {1.0.0},
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
```

</details>


### License

BIEST is licensed under the [Apache License 2.0](LICENSE). The [NOTICE](NOTICE) file lists the authors and must be kept in any redistribution, including ports to other languages. It also has the authors' requests on crediting them and on AI use.
