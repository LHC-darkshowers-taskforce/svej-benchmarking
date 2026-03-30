#!/usr/bin/env python3
"""
plot_ctau_contours.py
---------------------
Iso-lifetime contours (c·τ = 1, 10, 100, 1000 mm) in three 2D parameter
planes spanned by {m_piD, m_X, κ}, with the constraint f_D = m_piD enforced
everywhere:

  1. (m_piD, m_X)  — one panel per κ
  2. (m_piD, κ)    — one panel per m_X
  3. (m_X,   κ)    — one panel per m_piD

Plus an overlay figure showing how the c·τ = 10 mm contour shifts with κ
in the (m_piD, m_X) plane.

Species shown on each panel:
  - π^T15  (diagonal, solid lines)   — only non-zero diagonal for universal κ
  - π^(0,1) (off-diagonal, dashed)
  - π^(0,2) (off-diagonal, dotted)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dark_pion_widths import DarkPionModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CTAU_LEVELS_MM = [1.0, 10.0, 100.0, 1000.0]
CTAU_COLORS    = {1.0: "#e41a1c", 10.0: "#ff7f00", 100.0: "#377eb8", 1000.0: "#4daf4a"}
CTAU_LABELS    = {v: rf"$c\tau = {v:.0f}$ mm" for v in CTAU_LEVELS_MM}

# Pion species: (getter_key, label, linestyle, linewidth)
SPECIES = [
    ("diag_14", r"$\pi^{T15}$",   "-",  1.8),
    ("off_01",  r"$\pi^{(0,1)}$", "--", 1.6),
    ("off_02",  r"$\pi^{(0,2)}$", ":",  1.8),
]

MPID_REF  = 10.0    # GeV — default when m_piD is not the scan axis (fD = MPID_REF too)
MX_REF    = 2000.0  # GeV — default when m_X is not the scan axis
KAPPA_REF = 0.5     # default when κ is not the scan axis

KAPPA_SCAN = [0.1, 0.3, 0.5, 1.0]
MX_SCAN    = [500, 1000, 2000, 5000]    # GeV
MIPID_SCAN = [5.0, 10.0, 50.0, 100.0]  # GeV

N_POINTS = 120  # grid resolution per axis


# ---------------------------------------------------------------------------
# Grid builder  —  enforces fD = m_piD at every point
# ---------------------------------------------------------------------------

def _ctau_grid(x_vals: np.ndarray, y_vals: np.ndarray,
               x_name: str, y_name: str,
               fixed: dict) -> dict:
    """
    Compute c·τ [mm] on an (Nx, Ny) grid for each pion species.
    The constraint fD = m_piD is applied at every grid point.

    Parameters
    ----------
    x_vals, y_vals : 1-D arrays for the two scan axes.
    x_name, y_name : parameter names ('m_piD', 'm_X', 'kappa').
    fixed          : dict of fixed parameters.

    Returns
    -------
    dict {species_key: 2-D array of ctau_mm, shape (Ny, Nx)}
    """
    Nx, Ny = len(x_vals), len(y_vals)
    grids = {s[0]: np.empty((Ny, Nx)) for s in SPECIES}

    for iy, yv in enumerate(y_vals):
        for ix, xv in enumerate(x_vals):
            params = dict(fixed)
            params[x_name] = xv
            params[y_name] = yv
            params["fD"] = params["m_piD"]   # enforce fD = m_piD
            model = DarkPionModel(**params)

            grids["diag_14"][iy, ix] = model.ctau_diag_mm(14)
            grids["off_01"] [iy, ix] = model.ctau_off_diag_mm(0, 1)
            grids["off_02"] [iy, ix] = model.ctau_off_diag_mm(0, 2)

    return grids


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _draw_contours(ax, X, Y, grids, levels_mm=CTAU_LEVELS_MM, label_contours=True):
    for key, label, ls, lw in SPECIES:
        Z = grids[key]
        if not np.any(np.isfinite(Z)):
            continue
        for ctau_val in levels_mm:
            try:
                cs = ax.contour(X, Y, Z,
                                levels=[ctau_val],
                                colors=[CTAU_COLORS[ctau_val]],
                                linestyles=[ls],
                                linewidths=[lw])
                if label_contours and key == "diag_14":
                    ax.clabel(cs, fmt=f"{ctau_val:.0f} mm", fontsize=7, inline=True)
            except Exception:
                pass


def _species_legend_handles():
    return [Line2D([0], [0], color="k", ls=ls, lw=lw, label=label)
            for _, label, ls, lw in SPECIES]


def _ctau_legend_handles():
    return [Line2D([0], [0], color=CTAU_COLORS[v], lw=2, label=CTAU_LABELS[v])
            for v in CTAU_LEVELS_MM]


def _add_legends(axes):
    leg1 = axes[-1].legend(handles=_ctau_legend_handles(),
                            title=r"$c\tau$", fontsize=8, loc="lower right")
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=_species_legend_handles(),
                    title="species", fontsize=8, loc="upper left")


# ---------------------------------------------------------------------------
# Figure 1: (m_piD, m_X) plane — one panel per κ
# ---------------------------------------------------------------------------

def figure_mpiD_mX():
    m_piD_vals = np.geomspace(2.0,   500.0, N_POINTS)
    m_X_vals   = np.geomspace(300.0, 1e4,   N_POINTS)
    X, Y = np.meshgrid(m_piD_vals, m_X_vals)

    ncols = len(KAPPA_SCAN)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
    fig.suptitle(
        r"Iso-$c\tau$ contours in $(m_{\pi_D},\, m_X)$ plane  ($f_D = m_{\pi_D}$)",
        fontsize=13,
    )

    for ax, kappa in zip(axes, KAPPA_SCAN):
        fixed = dict(kappa=kappa, m_piD=MPID_REF, m_X=MX_REF, fD=MPID_REF)
        grids = _ctau_grid(m_piD_vals, m_X_vals, "m_piD", "m_X", fixed)
        _draw_contours(ax, X, Y, grids, label_contours=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=11)
        ax.set_title(rf"$\kappa = {kappa}$", fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.35)

    axes[0].set_ylabel(r"$m_X$  [GeV]", fontsize=11)
    _add_legends(axes)

    fig.tight_layout()
    out = "ctau_contours_mpiD_mX.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: (m_piD, κ) plane — one panel per m_X
# ---------------------------------------------------------------------------

def figure_mpiD_kappa():
    m_piD_vals = np.geomspace(2.0, 500.0, N_POINTS)
    kappa_vals = np.geomspace(0.1, 1.0,   N_POINTS)
    X, Y = np.meshgrid(m_piD_vals, kappa_vals)

    ncols = len(MX_SCAN)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
    fig.suptitle(
        r"Iso-$c\tau$ contours in $(m_{\pi_D},\, \kappa)$ plane  ($f_D = m_{\pi_D}$)",
        fontsize=13,
    )

    for ax, mX in zip(axes, MX_SCAN):
        fixed = dict(m_X=mX, m_piD=MPID_REF, kappa=KAPPA_REF, fD=MPID_REF)
        grids = _ctau_grid(m_piD_vals, kappa_vals, "m_piD", "kappa", fixed)
        _draw_contours(ax, X, Y, grids, label_contours=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=11)
        ax.set_title(rf"$m_X = {mX}$ GeV", fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.35)

    axes[0].set_ylabel(r"$|\kappa|$", fontsize=11)

    leg1 = axes[-1].legend(handles=_ctau_legend_handles(),
                            title=r"$c\tau$", fontsize=8, loc="lower right")
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=_species_legend_handles(),
                    title="species", fontsize=8, loc="upper right")

    fig.tight_layout()
    out = "ctau_contours_mpiD_kappa.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: (m_X, κ) plane — one panel per m_piD
# ---------------------------------------------------------------------------

def figure_mX_kappa():
    m_X_vals   = np.geomspace(300.0, 1e4, N_POINTS)
    kappa_vals = np.geomspace(0.1,   1.0, N_POINTS)
    X, Y = np.meshgrid(m_X_vals, kappa_vals)

    ncols = len(MIPID_SCAN)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)
    fig.suptitle(
        r"Iso-$c\tau$ contours in $(m_X,\, \kappa)$ plane  ($f_D = m_{\pi_D}$)",
        fontsize=13,
    )

    for ax, mpiD in zip(axes, MIPID_SCAN):
        fixed = dict(m_piD=mpiD, m_X=MX_REF, kappa=KAPPA_REF, fD=mpiD)
        grids = _ctau_grid(m_X_vals, kappa_vals, "m_X", "kappa", fixed)
        _draw_contours(ax, X, Y, grids, label_contours=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_X$  [GeV]", fontsize=11)
        ax.set_title(rf"$m_{{\pi_D}} = f_D = {mpiD:.0f}$ GeV", fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.35)

    axes[0].set_ylabel(r"$|\kappa|$", fontsize=11)

    leg1 = axes[-1].legend(handles=_ctau_legend_handles(),
                            title=r"$c\tau$", fontsize=8, loc="upper left")
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=_species_legend_handles(),
                    title="species", fontsize=8, loc="lower right")

    fig.tight_layout()
    out = "ctau_contours_mX_kappa.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: κ-overlay — how the c·τ = 10 mm contour shifts with κ
#           in the (m_piD, m_X) plane  (fD = m_piD at every point)
# ---------------------------------------------------------------------------

def figure_overlay_kappa():
    m_piD_vals = np.geomspace(2.0,  500.0, N_POINTS)
    m_X_vals   = np.geomspace(300.0, 1e4,  N_POINTS)
    X, Y = np.meshgrid(m_piD_vals, m_X_vals)

    kappa_fine = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    kappa_cmap = plt.get_cmap("cool", len(kappa_fine))
    ctau_target = 10.0  # mm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        rf"$c\tau = {ctau_target:.0f}$ mm contour in $(m_{{\pi_D}},\, m_X)$ for varying $\kappa$"
        r"  ($f_D = m_{\pi_D}$)",
        fontsize=12,
    )

    for ax, spec_key, spec_label in [
        (ax1, "diag_14", r"$\pi^{T15}$"),
        (ax2, "off_01",  r"$\pi^{(0,1)}$"),
    ]:
        legend_handles = []
        for ci, kappa in enumerate(kappa_fine):
            fixed = dict(kappa=kappa, m_piD=MPID_REF, m_X=MX_REF, fD=MPID_REF)
            grids = _ctau_grid(m_piD_vals, m_X_vals, "m_piD", "m_X", fixed)
            Z = grids[spec_key]
            if not np.any(np.isfinite(Z)):
                continue
            try:
                ax.contour(X, Y, Z,
                           levels=[ctau_target],
                           colors=[kappa_cmap(ci)],
                           linewidths=[1.8])
                legend_handles.append(
                    Line2D([0], [0], color=kappa_cmap(ci), lw=1.8,
                           label=rf"$\kappa = {kappa}$")
                )
            except Exception:
                pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=11)
        ax.set_title(spec_label, fontsize=12)
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
        ax.grid(True, which="both", ls=":", alpha=0.35)

    ax1.set_ylabel(r"$m_X$  [GeV]", fontsize=11)
    fig.tight_layout()
    out = "ctau_contours_kappa_overlay.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Figure 1: (m_piD, m_X) plane ...")
    figure_mpiD_mX()

    print("Figure 2: (m_piD, κ) plane ...")
    figure_mpiD_kappa()

    print("Figure 3: (m_X, κ) plane ...")
    figure_mX_kappa()

    print("Figure 4: κ-overlay on (m_piD, m_X) ...")
    figure_overlay_kappa()
