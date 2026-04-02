#!/usr/bin/env python3
"""
ctau_contours.py
----------------
Iso-lifetime contours (c·τ = 1, 10, 100, 1000 mm) in three 2D parameter
planes, plus a coupling-overlay figure.  The constraint fD = m_piD is
enforced throughout.

Four separate PDFs per config:
  1. (m_piD, mediator)  — one panel per coupling value
  2. (m_piD, coupling)  — one panel per mediator mass
  3. (mediator, coupling) — one panel per m_piD
  4. coupling-overlay   — c·τ = 10 mm contour shift vs coupling
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting_utils import (
    apply_style, save_fig,
    ModelPlotConfig, PionSpec,
    CTAU_LEVELS_MM, CTAU_COLORS,
    ctau_legend_handles, contour_species_handles,
)
from model_config import ALL_CONFIGS

apply_style()

N_POINTS = 120


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def _ctau_grid(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_name: str,
    y_name: str,
    fixed: dict,
    model_class,
    contour_specs: list[PionSpec],
) -> dict:
    """
    Compute c·τ on a 2-D grid for each pion in contour_specs.

    Returns {pion_id: np.ndarray shape (Ny, Nx)}.
    The constraint fD = m_piD is applied whenever m_piD appears in the grid.
    """
    Nx, Ny = len(x_vals), len(y_vals)
    grids  = {s.pion_id: np.empty((Ny, Nx)) for s in contour_specs}

    for iy, yv in enumerate(y_vals):
        for ix, xv in enumerate(x_vals):
            params = dict(fixed)
            params[x_name] = xv
            params[y_name] = yv
            params["fD"]   = params["m_piD"]   # enforce fD = m_piD
            model = model_class(**params)
            for s in contour_specs:
                grids[s.pion_id][iy, ix] = model.ctau_for_pion(s.pion_id)

    return grids


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_contours(ax, X, Y, grids, contour_specs, levels_mm=CTAU_LEVELS_MM,
                   label_first=False):
    first_drawn = True
    for s in contour_specs:
        Z = grids[s.pion_id]
        if not np.any(np.isfinite(Z)):
            continue
        for ctau_val in levels_mm:
            try:
                cs = ax.contour(X, Y, Z,
                                levels=[ctau_val],
                                colors=[CTAU_COLORS[ctau_val]],
                                linestyles=[s.linestyle],
                                linewidths=[s.linewidth])
                if label_first and first_drawn and s is contour_specs[0]:
                    ax.clabel(cs, fmt=f"{ctau_val:.0f} mm", fontsize=7, inline=True)
            except Exception:
                pass
        first_drawn = False


def _add_legends(axes, contour_specs):
    leg1 = axes[-1].legend(handles=ctau_legend_handles(),
                            title=r"$c\tau$", fontsize=8, loc="lower right")
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=contour_species_handles(contour_specs),
                    title="species", fontsize=8, loc="upper left")


# ---------------------------------------------------------------------------
# Figure 1: (m_piD, mediator) plane — panels per coupling value
# ---------------------------------------------------------------------------

def figure_mpiD_mediator(config: ModelPlotConfig,
                          coupling_scan=None) -> None:
    if coupling_scan is None:
        coupling_scan = [0.1, 0.3, 0.5, 1.0]

    med          = config.mediator_param
    coup         = config.coupling_param
    m_piD_vals   = np.geomspace(2.0,   500.0, N_POINTS)
    m_med_vals   = np.geomspace(300.0, 1e4,   N_POINTS)
    X, Y         = np.meshgrid(m_piD_vals, m_med_vals)
    base         = dict(config.base_params)

    fig, axes = plt.subplots(1, len(coupling_scan),
                             figsize=(5 * len(coupling_scan), 5), sharey=True)
    if len(coupling_scan) == 1:
        axes = [axes]
    fig.suptitle(
        rf"Iso-$c\tau$ contours in $(m_{{\pi_D}},\, {config.mediator_label.split('[')[0].strip()})$ plane"
        rf"  ($f_D = m_{{\pi_D}}$)  —  {config.name}"
    )

    for ax, coup_val in zip(axes, coupling_scan):
        fixed = dict(base)
        fixed[coup] = coup_val
        grids = _ctau_grid(m_piD_vals, m_med_vals, "m_piD", med,
                            fixed, config.model_class, config.contour_pion_specs)
        _draw_contours(ax, X, Y, grids, config.contour_pion_specs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
        ax.set_title(rf"{config.coupling_label} $= {coup_val}$")

    axes[0].set_ylabel(config.mediator_label)
    _add_legends(axes, config.contour_pion_specs)
    fig.tight_layout()
    save_fig(fig, f"ctau_contours_mpiD_mediator_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Figure 2: (m_piD, coupling) plane — panels per mediator mass
# ---------------------------------------------------------------------------

def figure_mpiD_coupling(config: ModelPlotConfig,
                          mediator_scan=None) -> None:
    if mediator_scan is None:
        mediator_scan = [500, 1000, 2000, 5000]

    med          = config.mediator_param
    coup         = config.coupling_param
    m_piD_vals   = np.geomspace(2.0,  500.0, N_POINTS)
    coup_vals    = np.geomspace(0.05, 2.0,   N_POINTS)
    X, Y         = np.meshgrid(m_piD_vals, coup_vals)
    base         = dict(config.base_params)

    fig, axes = plt.subplots(1, len(mediator_scan),
                             figsize=(5 * len(mediator_scan), 5), sharey=True)
    if len(mediator_scan) == 1:
        axes = [axes]
    fig.suptitle(
        rf"Iso-$c\tau$ contours in $(m_{{\pi_D}},\, {config.coupling_label})$ plane"
        rf"  ($f_D = m_{{\pi_D}}$)  —  {config.name}"
    )

    for ax, mmed in zip(axes, mediator_scan):
        fixed = dict(base)
        fixed[med] = mmed
        grids = _ctau_grid(m_piD_vals, coup_vals, "m_piD", coup,
                            fixed, config.model_class, config.contour_pion_specs)
        _draw_contours(ax, X, Y, grids, config.contour_pion_specs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
        med_label = config.mediator_label.split('[')[0].strip()
        ax.set_title(rf"{med_label} $= {mmed}$ GeV")

    axes[0].set_ylabel(config.coupling_label)
    leg1 = axes[-1].legend(handles=ctau_legend_handles(),
                            title=r"$c\tau$", fontsize=8, loc="lower right")
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=contour_species_handles(config.contour_pion_specs),
                    title="species", fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig(fig, f"ctau_contours_mpiD_coupling_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Figure 3: (mediator, coupling) plane — panels per m_piD
# ---------------------------------------------------------------------------

def figure_mediator_coupling(config: ModelPlotConfig,
                              mpiD_scan=None) -> None:
    if mpiD_scan is None:
        mpiD_scan = [5.0, 10.0, 50.0, 100.0]

    med        = config.mediator_param
    coup       = config.coupling_param
    m_med_vals = np.geomspace(300.0, 1e4,  N_POINTS)
    coup_vals  = np.geomspace(0.05,  2.0,  N_POINTS)
    X, Y       = np.meshgrid(m_med_vals, coup_vals)
    base       = dict(config.base_params)

    fig, axes = plt.subplots(1, len(mpiD_scan),
                             figsize=(5 * len(mpiD_scan), 5), sharey=True)
    if len(mpiD_scan) == 1:
        axes = [axes]
    fig.suptitle(
        rf"Iso-$c\tau$ contours in $({config.mediator_label.split('[')[0].strip()},\, "
        rf"{config.coupling_label})$ plane"
        rf"  ($f_D = m_{{\pi_D}}$)  —  {config.name}"
    )

    for ax, mpiD in zip(axes, mpiD_scan):
        fixed = dict(base)
        fixed["m_piD"] = mpiD
        grids = _ctau_grid(m_med_vals, coup_vals, med, coup,
                            fixed, config.model_class, config.contour_pion_specs)
        _draw_contours(ax, X, Y, grids, config.contour_pion_specs)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(config.mediator_label)
        ax.set_title(rf"$m_{{\pi_D}} = f_D = {mpiD:.0f}$ GeV")

    axes[0].set_ylabel(config.coupling_label)
    leg1 = axes[-1].legend(handles=ctau_legend_handles(),
                            title=r"$c\tau$", fontsize=8, loc="upper left")
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=contour_species_handles(config.contour_pion_specs),
                    title="species", fontsize=8, loc="lower right")
    fig.tight_layout()
    save_fig(fig, f"ctau_contours_mediator_coupling_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Figure 4: coupling overlay — c·τ = 10 mm contour in (m_piD, mediator)
# ---------------------------------------------------------------------------

def figure_coupling_overlay(config: ModelPlotConfig,
                              coupling_fine=None,
                              ctau_target: float = 10.0) -> None:
    if coupling_fine is None:
        coupling_fine = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

    med          = config.mediator_param
    coup         = config.coupling_param
    m_piD_vals   = np.geomspace(2.0,   500.0, N_POINTS)
    m_med_vals   = np.geomspace(300.0, 1e4,   N_POINTS)
    X, Y         = np.meshgrid(m_piD_vals, m_med_vals)
    cmap         = plt.get_cmap("cool", len(coupling_fine))
    base         = dict(config.base_params)

    for s in config.contour_pion_specs:
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.suptitle(
            rf"$c\tau = {ctau_target:.0f}$ mm contour in "
            rf"$(m_{{\pi_D}},\, {config.mediator_label.split('[')[0].strip()})$"
            rf" for varying {config.coupling_label}  ($f_D = m_{{\pi_D}}$)"
            f"\n{config.name}  —  {s.label}"
        )
        legend_handles = []

        for ci, coup_val in enumerate(coupling_fine):
            fixed = dict(base)
            fixed[coup] = coup_val
            grids = _ctau_grid(m_piD_vals, m_med_vals, "m_piD", med,
                                fixed, config.model_class, [s])
            Z = grids[s.pion_id]
            if not np.any(np.isfinite(Z)):
                continue
            try:
                ax.contour(X, Y, Z, levels=[ctau_target],
                           colors=[cmap(ci)], linewidths=[1.8])
                legend_handles.append(
                    Line2D([0], [0], color=cmap(ci), lw=1.8,
                           label=rf"{config.coupling_label} $= {coup_val}$")
                )
            except Exception:
                pass

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
        ax.set_ylabel(config.mediator_label)
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
        fig.tight_layout()

        pid_str = f"{s.pion_id[0]}{s.pion_id[1]}" if isinstance(s.pion_id, tuple) \
                  else str(s.pion_id)
        save_fig(fig, f"ctau_contours_coupling_overlay_{config.tag}_pion{pid_str}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ModelPlotConfig) -> None:
    print(f"  Figure 1: (m_piD, mediator) plane ...")
    figure_mpiD_mediator(config)

    print(f"  Figure 2: (m_piD, coupling) plane ...")
    figure_mpiD_coupling(config)

    print(f"  Figure 3: (mediator, coupling) plane ...")
    figure_mediator_coupling(config)

    print(f"  Figure 4: coupling overlay ...")
    figure_coupling_overlay(config)


if __name__ == "__main__":
    for cfg in ALL_CONFIGS:
        print(f"\n--- {cfg.name} ---")
        main(cfg)
