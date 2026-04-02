#!/usr/bin/env python3
"""
coupling_solutions.py
---------------------
Visualise the coupling values (κ for t-channel, g_qd for s-channel) required
to achieve target c·τ values across the (mediator_mass, m_piD) parameter grid,
with fD = m_piD.

The solver uses the exact c·τ ∝ coupling^{-4} scaling that holds for both
models (t-channel: width ∝ κ⁴; s-channel anomaly: width ∝ g_qd⁴).

Two PDFs per config:
  1. Annotated heatmaps — coupling value as a colour matrix for each
     (pion species, c·τ target) combination.
  2. Line plots — coupling vs mediator mass for each c·τ target,
     lines coloured by m_piD.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from plotting_utils import apply_style, save_fig, ModelPlotConfig
from model_config import ALL_CONFIGS

apply_style()

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MEDIATOR_VALS  = [1000, 2000, 3000, 4000, 5000]   # GeV
MPID_VALS      = [5, 10, 20]                       # GeV  (fD = m_piD)
CTAU_TARGETS   = [1, 10, 100, 1000]                # mm

MPID_COLORS = {5: "#1b7837", 10: "#762a83", 20: "#d6604d"}
MPID_LABELS = {v: rf"$m_{{\pi_D}} = {v}$ GeV" for v in MPID_VALS}

COUPLING_VMIN, COUPLING_VMAX = 0.05, 3.0

# Coupling exponent: c·τ ∝ coupling^{-exponent}
# Both t-channel (κ) and s-channel anomaly (g_qd) have exponent = 4.
_COUPLING_EXP = 4


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _solve_coupling(ctau_ref: float, ctau_target: float) -> float:
    """Return coupling needed for ctau_target given reference at coupling=1."""
    if not np.isfinite(ctau_ref) or ctau_ref <= 0:
        return np.nan
    return (ctau_ref / ctau_target) ** (1.0 / _COUPLING_EXP)


def _compute_coupling_grid(config: ModelPlotConfig) -> np.ndarray:
    """
    Compute the required coupling on a grid.

    Returns array of shape (n_species, n_mediator, n_mpiD, n_ctau).
    species order matches config.pion_specs.
    """
    n_s    = len(config.pion_specs)
    n_med  = len(MEDIATOR_VALS)
    n_mp   = len(MPID_VALS)
    n_ct   = len(CTAU_TARGETS)
    grid   = np.full((n_s, n_med, n_mp, n_ct), np.nan)

    coup_key = config.coupling_param
    med_key  = config.mediator_param

    for xi, mmed in enumerate(MEDIATOR_VALS):
        for pi, mpiD in enumerate(MPID_VALS):
            params = dict(config.base_params)
            params[med_key]  = float(mmed)
            params["fD"]     = float(mpiD)
            params["m_piD"]  = float(mpiD)
            params[coup_key] = 1.0          # reference at coupling = 1
            model = config.model_class(**params)

            for si, spec in enumerate(config.pion_specs):
                ctau_ref = model.ctau_for_pion(spec.pion_id)
                for ci, ctau_t in enumerate(CTAU_TARGETS):
                    grid[si, xi, pi, ci] = _solve_coupling(ctau_ref, ctau_t)

    return grid


# ---------------------------------------------------------------------------
# Figure 1: Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(config: ModelPlotConfig) -> None:
    grid  = _compute_coupling_grid(config)
    n_s   = len(config.pion_specs)
    n_ct  = len(CTAU_TARGETS)

    norm  = mcolors.LogNorm(vmin=COUPLING_VMIN, vmax=COUPLING_VMAX)
    cmap  = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        n_s, n_ct,
        figsize=(3.8 * n_ct, 3.5 * n_s),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        rf"Required {config.coupling_label} for target $c\tau$  ($f_D = m_{{\pi_D}}$)"
        f"\n{config.name}",
        y=1.02,
    )

    for si, spec in enumerate(config.pion_specs):
        for ci, ctau_t in enumerate(CTAU_TARGETS):
            ax = axes[si, ci]
            Z  = grid[si, :, :, ci]   # shape (n_med, n_mpiD)

            im = ax.imshow(Z, norm=norm, cmap=cmap,
                           origin="upper", aspect="auto",
                           extent=[-0.5, len(MPID_VALS) - 0.5,
                                   len(MEDIATOR_VALS) - 0.5, -0.5])

            for xi in range(len(MEDIATOR_VALS)):
                for pi in range(len(MPID_VALS)):
                    k   = Z[xi, pi]
                    txt = f"{k:.3f}" if np.isfinite(k) else "—"
                    ax.text(pi, xi, txt, ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white")

            ax.set_xticks(range(len(MPID_VALS)))
            ax.set_xticklabels([str(v) for v in MPID_VALS])
            ax.set_yticks(range(len(MEDIATOR_VALS)))
            ax.set_yticklabels([str(v) for v in MEDIATOR_VALS])

            if si == n_s - 1:
                ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
            if ci == 0:
                ax.set_ylabel(config.mediator_label)

            ax.set_title(rf"{spec.label},  $c\tau = {ctau_t}$ mm")

            for x in np.arange(-0.5, len(MPID_VALS), 1):
                ax.axvline(x, color="gray", lw=0.5)
            for y in np.arange(-0.5, len(MEDIATOR_VALS), 1):
                ax.axhline(y, color="gray", lw=0.5)

    fig.colorbar(im, ax=axes, orientation="vertical",
                 fraction=0.02, pad=0.02,
                 label=config.coupling_label)

    save_fig(fig, f"coupling_solutions_heatmap_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Line plots — coupling vs mediator mass
# ---------------------------------------------------------------------------

def plot_lines(config: ModelPlotConfig) -> None:
    grid   = _compute_coupling_grid(config)
    n_s    = len(config.pion_specs)
    n_ct   = len(CTAU_TARGETS)

    LS     = {si: ["-", "--", ":", "-."][si % 4] for si in range(n_s)}

    fig, axes = plt.subplots(1, n_ct, figsize=(4.5 * n_ct, 5), sharey=True)
    if n_ct == 1:
        axes = [axes]
    fig.suptitle(
        rf"Required {config.coupling_label} vs mediator mass  ($f_D = m_{{\pi_D}}$)"
        f"\n{config.name}"
    )

    for ci, ctau_t in enumerate(CTAU_TARGETS):
        ax = axes[ci]
        for si, spec in enumerate(config.pion_specs):
            for pi, mpiD in enumerate(MPID_VALS):
                y    = grid[si, :, pi, ci]
                mask = np.isfinite(y)
                ax.plot(np.array(MEDIATOR_VALS)[mask], y[mask],
                        color=MPID_COLORS[mpiD],
                        ls=LS[si], lw=1.8,
                        marker="o", markersize=5)

        ax.set_xlabel(config.mediator_label)
        ax.set_title(rf"$c\tau = {ctau_t}$ mm")
        ax.set_yscale("log")
        ax.set_ylim(0.03, 4.0)
        ax.set_xticks(MEDIATOR_VALS)
        ax.set_xticklabels([str(v) for v in MEDIATOR_VALS], rotation=30)

    axes[0].set_ylabel(config.coupling_label)

    color_handles = [
        Line2D([0], [0], color=MPID_COLORS[v], lw=2,
               marker="o", markersize=5, label=MPID_LABELS[v])
        for v in MPID_VALS
    ]
    style_handles = [
        Line2D([0], [0], color="k", lw=1.8, ls=LS[si],
               label=spec.label)
        for si, spec in enumerate(config.pion_specs)
    ]
    axes[-1].legend(handles=color_handles + style_handles,
                    loc="upper left", ncol=1)

    fig.tight_layout()
    save_fig(fig, f"coupling_solutions_lines_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ModelPlotConfig) -> None:
    print(f"  Heatmaps ...")
    plot_heatmaps(config)

    print(f"  Line plots ...")
    plot_lines(config)


if __name__ == "__main__":
    for cfg in ALL_CONFIGS:
        print(f"\n--- {cfg.name} ---")
        main(cfg)
