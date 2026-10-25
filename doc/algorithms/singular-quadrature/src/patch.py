# Copyright (C) 2018-2026 The Simons Foundation, Inc. and the BIEST authors (see NOTICE)
# Author: Dhairya Malhotra
# SPDX-License-Identifier: Apache-2.0
#
# Draws ../patch.svg: the partition of unity, the polar quadrature nodes and the interpolation stencil.
# Usage: python3 patch.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 6  # patch radius in grid spacings (PATCH_DIM0, M/2 in the paper)
q = 8  # polar quadrature order (RAD_DIM)
INTERP = 12  # interpolation stencil width (INTERP_ORDER)
chi = lambda r: np.exp(-36 * r**6)  # exponent 6 for PATCH_DIM = 2R+1 <= 20 (singular_correction.hpp)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
ext = R + 3.5
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.set_xticks([])
    ax.set_yticks([])

# left: partition of unity on the grid
ax = axs[0]
xx, yy = np.meshgrid(np.linspace(-ext, ext, 120), np.linspace(-ext, ext, 120))
cs = ax.imshow(chi(np.sqrt(xx**2 + yy**2) / R), extent=(-ext, ext, -ext, ext), origin="lower", cmap="Blues", vmin=0, vmax=1)
g = np.arange(-int(ext), int(ext) + 1)
gx, gy = np.meshgrid(g, g)
inside = np.sqrt(gx**2 + gy**2) <= R
ax.plot(gx[~inside], gy[~inside], ".", color="0.55", ms=4)
ax.plot(gx[inside], gy[inside], "o", color="k", ms=3.5, label="grid points on the patch")
ax.plot(0, 0, "*", color="crimson", ms=14, label="target")
ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color="k", lw=1))
ax.set_title(r"partition of unity $\eta = \chi(r/R)$ on the grid")
ax.set_xlabel(r"toroidal index ($\theta$)")
ax.set_ylabel(r"poloidal index ($\phi$)")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02)

# right: polar quadrature nodes and the interpolation stencil of one node
ax = axs[1]
xr, _ = np.polynomial.legendre.leggauss(q)
rad = (xr + 1) / 2 * R
ang = 2 * np.pi * np.arange(2 * q) / (2 * q)
rr, aa = np.meshgrid(rad, ang)
px, py = rr * np.cos(aa), rr * np.sin(aa)
ax.plot(gx.ravel(), gy.ravel(), ".", color="0.7", ms=4)
ax.plot(px.ravel(), py.ravel(), "o", color="tab:blue", ms=3.5, label=r"polar nodes ($q \times 2q$)")
node = (px[1, 3], py[1, 3])
s0x = int(np.floor(node[0])) - INTERP // 2 + 1
s0y = int(np.floor(node[1])) - INTERP // 2 + 1
ax.add_patch(plt.Rectangle((s0x - 0.4, s0y - 0.4), INTERP - 0.2, INTERP - 0.2, fill=False, color="darkorange", lw=1.5, label=f"{INTERP}x{INTERP} interpolation stencil"))
ax.plot(*node, "s", color="darkorange", ms=7)
ax.plot(0, 0, "*", color="crimson", ms=14, label="target")
ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color="k", lw=1))
ax.set_title("polar quadrature centered at the target")
ax.set_xlabel(r"toroidal index ($\theta$)")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patch.svg"))
