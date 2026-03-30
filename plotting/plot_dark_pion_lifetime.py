#!/usr/bin/env python3
"""
plot_dark_pion_lifetime.py
--------------------------
Plots c·τ (mm) of dark pions as a function of:
  1. m_piD  (dark pion mass)
  2. m_X    (mediator mass)
  3. |κ|    (coupling strength)
  4. fD     (dark decay constant)

Each plot overlays all non-zero pion species (off-diagonal and diagonal).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering; change to "TkAgg" for interactive
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dark_pion_widths import DarkPionModel, DIAGONAL_PION_NAMES

# ---------------------------------------------------------------------------
# Reference (default) parameters
# ---------------------------------------------------------------------------
REF = dict(fD=10.0, m_piD=10.0, m_X=2000.0, kappa=1.0)

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
# Off-diagonal pions: solid lines with distinct colors
OFF_DIAG_PAIRS = [(0, 1), (0, 2), (1, 2)]
OFF_DIAG_COLORS = {(0, 1): "tab:blue", (0, 2): "tab:orange", (1, 2): "tab:green"}
OFF_DIAG_LABELS = {(a, b): f"$\\pi^{{({a},{b})}}$" for (a, b) in OFF_DIAG_PAIRS}

# Diagonal pions: dashed lines with distinct colors
DIAG_INDICES  = [2, 7, 14]
DIAG_COLORS   = {2: "tab:red", 7: "tab:purple", 14: "tab:brown"}
DIAG_LABELS   = {b: f"$\\pi^{{\\mathrm{{{DIAGONAL_PION_NAMES[b]}}}}}$" for b in DIAG_INDICES}


def _collect_ctaus(param_name: str, values: np.ndarray, base_params: dict) -> dict:
    """
    Scan one parameter over `values`, keeping all others fixed at `base_params`.

    Returns
    -------
    dict
        off_diagonal : {(α,β): np.ndarray of c·τ [mm]}
        diagonal     : {b:     np.ndarray of c·τ [mm]}
    """
    off_diag = {pair: np.empty(len(values)) for pair in OFF_DIAG_PAIRS}
    diag     = {b:    np.empty(len(values)) for b in DIAG_INDICES}

    for k, v in enumerate(values):
        params = dict(base_params)
        params[param_name] = v
        model = DarkPionModel(**params)

        for pair in OFF_DIAG_PAIRS:
            off_diag[pair][k] = model.ctau_off_diag_mm(*pair)
        for b in DIAG_INDICES:
            diag[b][k] = model.ctau_diag_mm(b)

    return {"off_diagonal": off_diag, "diagonal": diag}


def _make_legend_handles():
    """Return legend handles for all pion species."""
    handles = []
    for pair in OFF_DIAG_PAIRS:
        handles.append(Line2D([0], [0], color=OFF_DIAG_COLORS[pair],
                               ls="-", lw=1.8, label=OFF_DIAG_LABELS[pair]))
    for b in DIAG_INDICES:
        handles.append(Line2D([0], [0], color=DIAG_COLORS[b],
                               ls="--", lw=1.8, label=DIAG_LABELS[b]))
    return handles


def _plot_ctaus(ax, x_values, ctaus, finite_only=True):
    """Draw all pion c·τ curves onto `ax`."""
    for pair in OFF_DIAG_PAIRS:
        y = ctaus["off_diagonal"][pair]
        mask = np.isfinite(y) if finite_only else np.ones(len(y), dtype=bool)
        if mask.any():
            ax.plot(x_values[mask], y[mask],
                    color=OFF_DIAG_COLORS[pair], ls="-", lw=1.8)

    for b in DIAG_INDICES:
        y = ctaus["diagonal"][b]
        mask = np.isfinite(y) if finite_only else np.ones(len(y), dtype=bool)
        if mask.any():
            ax.plot(x_values[mask], y[mask],
                    color=DIAG_COLORS[b], ls="--", lw=1.8)


# ---------------------------------------------------------------------------
# Individual scan functions
# ---------------------------------------------------------------------------

def plot_vs_m_piD(ax, base_params: dict, n_points: int = 200):
    m_piD_min = base_params.get("_m_piD_min", 2.0)
    m_piD_max = base_params.get("_m_piD_max", 500.0)
    values = np.geomspace(m_piD_min, m_piD_max, n_points)
    ctaus  = _collect_ctaus("m_piD", values, base_params)

    _plot_ctaus(ax, values, ctaus)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=12)
    ax.set_ylabel(r"$c\tau$  [mm]", fontsize=12)
    ax.set_title(
        rf"Lifetime vs $m_{{\pi_D}}$   "
        rf"($m_X={base_params['m_X']:.0f}$ GeV, "
        rf"$\kappa={abs(base_params['kappa']):.2g}$, "
        rf"$f_D={base_params['fD']:.1f}$ GeV)",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(handles=_make_legend_handles(), fontsize=9, ncol=2)


def plot_vs_m_X(ax, base_params: dict, n_points: int = 200):
    m_X_min = base_params.get("_m_X_min", 500.0)
    m_X_max = base_params.get("_m_X_max", 1e4)
    values = np.geomspace(m_X_min, m_X_max, n_points)
    ctaus  = _collect_ctaus("m_X", values, base_params)

    _plot_ctaus(ax, values, ctaus)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_X$  [GeV]", fontsize=12)
    ax.set_ylabel(r"$c\tau$  [mm]", fontsize=12)
    ax.set_title(
        rf"Lifetime vs $m_X$   "
        rf"($m_{{\pi_D}}={base_params['m_piD']:.1f}$ GeV, "
        rf"$\kappa={abs(base_params['kappa']):.2g}$, "
        rf"$f_D={base_params['fD']:.1f}$ GeV)",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(handles=_make_legend_handles(), fontsize=9, ncol=2)


def plot_vs_kappa(ax, base_params: dict, n_points: int = 200):
    kappa_min = base_params.get("_kappa_min", 1e-3)
    kappa_max = base_params.get("_kappa_max", 10.0)
    values = np.geomspace(kappa_min, kappa_max, n_points)
    ctaus  = _collect_ctaus("kappa", values, base_params)

    _plot_ctaus(ax, values, ctaus)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\kappa|$", fontsize=12)
    ax.set_ylabel(r"$c\tau$  [mm]", fontsize=12)
    ax.set_title(
        rf"Lifetime vs $\kappa$   "
        rf"($m_{{\pi_D}}={base_params['m_piD']:.1f}$ GeV, "
        rf"$m_X={base_params['m_X']:.0f}$ GeV, "
        rf"$f_D={base_params['fD']:.1f}$ GeV)",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(handles=_make_legend_handles(), fontsize=9, ncol=2)


def plot_vs_fD(ax, base_params: dict, n_points: int = 200):
    fD_min = base_params.get("_fD_min", 0.5)
    fD_max = base_params.get("_fD_max", 200.0)
    values = np.geomspace(fD_min, fD_max, n_points)
    ctaus  = _collect_ctaus("fD", values, base_params)

    _plot_ctaus(ax, values, ctaus)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f_D$  [GeV]", fontsize=12)
    ax.set_ylabel(r"$c\tau$  [mm]", fontsize=12)
    ax.set_title(
        rf"Lifetime vs $f_D$   "
        rf"($m_{{\pi_D}}={base_params['m_piD']:.1f}$ GeV, "
        rf"$m_X={base_params['m_X']:.0f}$ GeV, "
        rf"$\kappa={abs(base_params['kappa']):.2g}$)",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(handles=_make_legend_handles(), fontsize=9, ncol=2)


# ---------------------------------------------------------------------------
# 2-D contour: c·τ in the (m_piD, m_X) plane for a representative pion
# ---------------------------------------------------------------------------

def plot_contour_mpiD_mX(ax, base_params: dict, pion_spec=("diag", 2),
                          n_points: int = 80):
    """
    2-D log-log contour of log10(c·τ/mm) in the (m_piD, m_X) plane.

    pion_spec : ("diag", b) or ("off", alpha, beta)
    """
    m_piD_vals = np.geomspace(2.0,   500.0, n_points)
    m_X_vals   = np.geomspace(500.0, 1e4,   n_points)

    Z = np.empty((n_points, n_points))
    for ki, mX in enumerate(m_X_vals):
        for kj, mpiD in enumerate(m_piD_vals):
            params = dict(base_params)
            params["m_piD"] = mpiD
            params["m_X"]   = mX
            model = DarkPionModel(**params)
            if pion_spec[0] == "diag":
                ctau = model.ctau_diag_mm(pion_spec[1])
            else:
                ctau = model.ctau_off_diag_mm(pion_spec[1], pion_spec[2])
            Z[ki, kj] = np.log10(ctau) if np.isfinite(ctau) and ctau > 0 else np.nan

    M_PION, M_X = np.meshgrid(m_piD_vals, m_X_vals)

    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    if not (np.isfinite(z_min) and np.isfinite(z_max)):
        ax.text(0.5, 0.5, "No finite decay width\n(zero amplitude for this coupling)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_xscale("log")
        ax.set_yscale("log")
        return

    levels = np.arange(z_min - 0.5, z_max + 1.5, 0.5)
    if len(levels) < 2:
        levels = np.linspace(z_min, z_max, 10)

    cf = ax.contourf(M_PION, M_X, Z, levels=levels, cmap="viridis", extend="both")
    plt.colorbar(cf, ax=ax, label=r"$\log_{10}(c\tau / \mathrm{mm})$")
    ax.contour(M_PION, M_X, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=12)
    ax.set_ylabel(r"$m_X$  [GeV]", fontsize=12)
    if pion_spec[0] == "diag":
        label = DIAGONAL_PION_NAMES[pion_spec[1]]
        ax.set_title(rf"$\log_{{10}}(c\tau)$ for $\pi^{{\mathrm{{{label}}}}}$   "
                     rf"($\kappa={abs(base_params['kappa']):.2g}$, $f_D={base_params['fD']:.1f}$ GeV)",
                     fontsize=10)
    else:
        a, b_ = pion_spec[1], pion_spec[2]
        ax.set_title(rf"$\log_{{10}}(c\tau)$ for $\pi^{{({a},{b_})}}$   "
                     rf"($\kappa={abs(base_params['kappa']):.2g}$, $f_D={base_params['fD']:.1f}$ GeV)",
                     fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base = dict(REF)

    # --- Figure 1: 1-D scans (2×2) ---
    fig1, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig1.suptitle("Dark pion lifetime  $c\\tau$  vs model parameters", fontsize=14)

    plot_vs_m_piD(axes[0, 0], base)
    plot_vs_m_X  (axes[0, 1], base)
    plot_vs_kappa(axes[1, 0], base)
    plot_vs_fD   (axes[1, 1], base)

    fig1.tight_layout()
    out1 = "dark_pion_lifetime_1d.pdf"
    fig1.savefig(out1, bbox_inches="tight")
    print(f"Saved {out1}")

    # --- Figure 2: 2-D contours in (m_piD, m_X) plane ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle(r"$\log_{10}(c\tau / \mathrm{mm})$ in the $(m_{\pi_D}, m_X)$ plane  [non-zero species for universal $\kappa$]", fontsize=12)

    plot_contour_mpiD_mX(axes2[0], base, pion_spec=("diag", 14))   # T15 (non-zero for universal κ)
    plot_contour_mpiD_mX(axes2[1], base, pion_spec=("off",  0, 1))
    plot_contour_mpiD_mX(axes2[2], base, pion_spec=("off",  0, 2))

    fig2.tight_layout()
    out2 = "dark_pion_lifetime_2d.pdf"
    fig2.savefig(out2, bbox_inches="tight")
    print(f"Saved {out2}")

    # --- Figure 3: lifetime vs m_piD for several m_X values ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    m_X_samples = [500, 1000, 2000, 5000, 10000]
    cmap = plt.get_cmap("plasma", len(m_X_samples))

    m_piD_vals = np.geomspace(2.0, 500.0, 300)
    for ci, mX in enumerate(m_X_samples):
        params = dict(base)
        params["m_X"] = mX
        ctaus  = _collect_ctaus("m_piD", m_piD_vals, params)
        # Show T15 diagonal pion as the representative (T3/T8 vanish for universal κ)
        y = ctaus["diagonal"][14]
        mask = np.isfinite(y)
        ax3.plot(m_piD_vals[mask], y[mask], color=cmap(ci), lw=1.8,
                 label=rf"$m_X = {mX}$ GeV")

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=12)
    ax3.set_ylabel(r"$c\tau$  [mm]", fontsize=12)
    ax3.set_title(
        rf"$\pi^{{T15}}$ lifetime vs $m_{{\pi_D}}$ for various $m_X$   "
        rf"($\kappa={abs(base['kappa']):.2g}$, $f_D={base['fD']:.1f}$ GeV)",
        fontsize=10,
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, which="both", ls=":", alpha=0.4)

    fig3.tight_layout()
    out3 = "dark_pion_lifetime_mpiD_vs_mX.pdf"
    fig3.savefig(out3, bbox_inches="tight")
    print(f"Saved {out3}")

    # --- Figure 4: lifetime vs kappa for several m_X values ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    kappa_vals = np.geomspace(1e-3, 10.0, 300)
    for ci, mX in enumerate(m_X_samples):
        params = dict(base)
        params["m_X"] = mX
        ctaus  = _collect_ctaus("kappa", kappa_vals, params)
        y = ctaus["diagonal"][14]
        mask = np.isfinite(y)
        ax4.plot(kappa_vals[mask], y[mask], color=cmap(ci), lw=1.8,
                 label=rf"$m_X = {mX}$ GeV")

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel(r"$|\kappa|$", fontsize=12)
    ax4.set_ylabel(r"$c\tau$  [mm]", fontsize=12)
    ax4.set_title(
        rf"$\pi^{{T15}}$ lifetime vs $\kappa$ for various $m_X$   "
        rf"($m_{{\pi_D}}={base['m_piD']:.1f}$ GeV, $f_D={base['fD']:.1f}$ GeV)",
        fontsize=10,
    )
    ax4.legend(fontsize=9)
    ax4.grid(True, which="both", ls=":", alpha=0.4)

    fig4.tight_layout()
    out4 = "dark_pion_lifetime_kappa_vs_mX.pdf"
    fig4.savefig(out4, bbox_inches="tight")
    print(f"Saved {out4}")


if __name__ == "__main__":
    main()
