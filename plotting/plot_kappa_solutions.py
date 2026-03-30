#!/usr/bin/env python3
"""
plot_kappa_solutions.py
-----------------------
Visualise the κ values required to achieve target c·τ values across the
(m_X, m_piD) parameter grid, with fD = m_piD.

Two figures:
  1. Annotated heatmaps — κ as a colour matrix for each (species, c·τ target).
  2. Line plots — κ vs m_X for each c·τ target, lines per m_piD.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from dark_pion_widths import DarkPionModel

# ---------------------------------------------------------------------------
# Parameter grid  (mirror solve_kappa.py)
# ---------------------------------------------------------------------------
MX_VALS      = [1000, 2000, 3000, 4000, 5000]   # GeV
MPID_VALS    = [5, 10, 20]                       # GeV  (fD = m_piD)
CTAU_TARGETS = [1, 10, 100, 1000]                # mm

# Only T15 and (0,1) — (0,2) is identical to (0,1) for universal κ
SPECIES = [
    ("T15",   lambda m: m.ctau_diag_mm(14)),
    ("(0,1)", lambda m: m.ctau_off_diag_mm(0, 1)),
]

MPID_COLORS = {5: "#1b7837", 10: "#762a83", 20: "#d6604d"}
MPID_LABELS = {v: rf"$m_{{\pi_D}} = {v}$ GeV" for v in MPID_VALS}

# ---------------------------------------------------------------------------
# Compute solutions
# ---------------------------------------------------------------------------

def _solve(ctau_ref, ctau_target):
    if not np.isfinite(ctau_ref) or ctau_ref <= 0:
        return np.nan
    return (ctau_ref / ctau_target) ** 0.25


# kappa[spec_idx, mX_idx, mpiD_idx, ctau_idx]
n_spec  = len(SPECIES)
n_mX    = len(MX_VALS)
n_mpiD  = len(MPID_VALS)
n_ctau  = len(CTAU_TARGETS)

kappa_grid = np.full((n_spec, n_mX, n_mpiD, n_ctau), np.nan)

for si, (spec_label, getter) in enumerate(SPECIES):
    for xi, mX in enumerate(MX_VALS):
        for pi, mpiD in enumerate(MPID_VALS):
            model = DarkPionModel(fD=float(mpiD), m_piD=mpiD, m_X=mX, kappa=1.0)
            ctau_ref = getter(model)
            for ci, ctau_t in enumerate(CTAU_TARGETS):
                kappa_grid[si, xi, pi, ci] = _solve(ctau_ref, ctau_t)

# ---------------------------------------------------------------------------
# Figure 1: Heatmaps  —  rows = species, cols = c·τ target
# ---------------------------------------------------------------------------

KAPPA_VMIN, KAPPA_VMAX = 0.05, 3.0
norm = mcolors.LogNorm(vmin=KAPPA_VMIN, vmax=KAPPA_VMAX)
cmap = plt.get_cmap("viridis")

fig1, axes1 = plt.subplots(n_spec, n_ctau,
                            figsize=(3.8 * n_ctau, 3.5 * n_spec),
                            squeeze=False,
                            constrained_layout=True)
fig1.suptitle(
    r"Required $\kappa$ for target $c\tau$  ($f_D = m_{\pi_D}$)",
    fontsize=14, y=1.01,
)

for si, (spec_label, _) in enumerate(SPECIES):
    for ci, ctau_t in enumerate(CTAU_TARGETS):
        ax = axes1[si, ci]

        # Build 2-D array: rows = m_X (y), cols = m_piD (x)
        Z = kappa_grid[si, :, :, ci]   # shape (n_mX, n_mpiD)

        im = ax.imshow(Z, norm=norm, cmap=cmap,
                       origin="upper", aspect="auto",
                       extent=[-0.5, n_mpiD - 0.5, n_mX - 0.5, -0.5])

        # Annotate each cell with the κ value
        for xi in range(n_mX):
            for pi in range(n_mpiD):
                k = Z[xi, pi]
                txt = f"{k:.3f}" if np.isfinite(k) else "—"
                ax.text(pi, xi, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

        ax.set_xticks(range(n_mpiD))
        ax.set_xticklabels([f"{v}" for v in MPID_VALS], fontsize=9)
        ax.set_yticks(range(n_mX))
        ax.set_yticklabels([f"{v}" for v in MX_VALS], fontsize=9)

        if si == n_spec - 1:
            ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=10)
        if ci == 0:
            ax.set_ylabel(r"$m_X$  [GeV]", fontsize=10)

        ax.set_title(
            rf"$\pi^{{{spec_label}}}$,  $c\tau = {ctau_t}$ mm",
            fontsize=10,
        )

        # draw grid lines
        for x in np.arange(-0.5, n_mpiD, 1):
            ax.axvline(x, color="gray", lw=0.5)
        for y in np.arange(-0.5, n_mX, 1):
            ax.axhline(y, color="gray", lw=0.5)

# shared colorbar
fig1.colorbar(im, ax=axes1, orientation="vertical",
              fraction=0.02, pad=0.02,
              label=r"$\kappa$")

out1 = "kappa_solutions_heatmap.pdf"
fig1.savefig(out1, bbox_inches="tight")
print(f"Saved {out1}")
plt.close(fig1)

# ---------------------------------------------------------------------------
# Figure 2: κ vs m_X line plots  —  cols = c·τ target
#           lines per m_piD; solid = T15, dashed = (0,1)
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(1, n_ctau, figsize=(4.5 * n_ctau, 5), sharey=True)
fig2.suptitle(
    r"Required $\kappa$ vs $m_X$  ($f_D = m_{\pi_D}$)",
    fontsize=13,
)

LS = {0: "-", 1: "--"}   # species index → linestyle

for ci, ctau_t in enumerate(CTAU_TARGETS):
    ax = axes2[ci]

    for si, (spec_label, _) in enumerate(SPECIES):
        for pi, mpiD in enumerate(MPID_VALS):
            y = kappa_grid[si, :, pi, ci]
            mask = np.isfinite(y)
            ax.plot(np.array(MX_VALS)[mask], y[mask],
                    color=MPID_COLORS[mpiD],
                    ls=LS[si], lw=1.8,
                    marker="o", markersize=5)

    ax.set_xlabel(r"$m_X$  [GeV]", fontsize=11)
    ax.set_title(rf"$c\tau = {ctau_t}$ mm", fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 4.0)
    ax.set_xticks(MX_VALS)
    ax.set_xticklabels([str(v) for v in MX_VALS], rotation=30, fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)

axes2[0].set_ylabel(r"$\kappa$", fontsize=12)

# Legend: m_piD by colour, species by linestyle
color_handles = [Line2D([0], [0], color=MPID_COLORS[v], lw=2,
                         marker="o", markersize=5, label=MPID_LABELS[v])
                 for v in MPID_VALS]
style_handles = [Line2D([0], [0], color="k", lw=1.8, ls=LS[si],
                         label=rf"$\pi^{{{sl}}}$")
                 for si, (sl, _) in enumerate(SPECIES)]

axes2[-1].legend(handles=color_handles + style_handles,
                  fontsize=8, loc="upper left", ncol=1)

fig2.tight_layout()
out2 = "kappa_solutions_lines.pdf"
fig2.savefig(out2, bbox_inches="tight")
print(f"Saved {out2}")
plt.close(fig2)
