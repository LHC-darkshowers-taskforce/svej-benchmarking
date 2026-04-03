#!/usr/bin/env python3
"""
dark_pion_lifetime.py
---------------------
Plots c·τ (mm) of dark pions as a function of model parameters.

Four separate PDFs are produced per config:
  1. c·τ vs m_piD
  2. c·τ vs mediator mass  (m_X  or m_Zp)
  3. c·τ vs coupling       (κ    or g_qd)
  4. c·τ vs fD

Plus two additional PDFs:
  5. c·τ vs m_piD for several mediator masses (representative pion, coloured)
  6. c·τ vs coupling for several mediator masses (representative pion, coloured)

All output filenames are tagged with config.tag.
"""

import numpy as np
import matplotlib.pyplot as plt

from plotting_utils import (
    apply_style, save_fig,
    ModelPlotConfig, PionSpec,
    species_legend_handles,
)
from model_config import ALL_CONFIGS

apply_style()


# ---------------------------------------------------------------------------
# Core data-collection helper
# ---------------------------------------------------------------------------

def _collect_ctaus(
    param_name: str,
    values: np.ndarray,
    base_params: dict,
    model_class,
    pion_specs: list[PionSpec],
) -> dict:
    """
    Scan one parameter over `values`, keeping all others fixed at `base_params`.

    Returns {pion_id: np.ndarray of c·τ [mm]}.
    """
    result = {s.pion_id: np.empty(len(values)) for s in pion_specs}

    for k, v in enumerate(values):
        params = dict(base_params)
        params[param_name] = v
        model = model_class(**params)
        for s in pion_specs:
            result[s.pion_id][k] = model.ctau_for_pion(s.pion_id)

    return result


def _plot_ctaus(ax, x_values, ctaus, pion_specs, finite_only=True):
    """Draw c·τ curves for all pion species onto `ax`."""
    for s in pion_specs:
        y    = ctaus[s.pion_id]
        mask = np.isfinite(y) if finite_only else np.ones(len(y), dtype=bool)
        if mask.any():
            ax.plot(x_values[mask], y[mask],
                    color=s.color, ls=s.linestyle, lw=s.linewidth)


# ---------------------------------------------------------------------------
# Individual scan figures (one PDF each)
# ---------------------------------------------------------------------------

def plot_vs_m_piD(config: ModelPlotConfig, n_points: int = 200) -> None:
    base   = dict(config.base_params)
    values = np.geomspace(
        base.pop("_m_piD_min", 2.0),
        base.pop("_m_piD_max", 500.0),
        n_points,
    )
    ctaus = _collect_ctaus("m_piD", values, base, config.model_class, config.pion_specs)

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_ctaus(ax, values, ctaus, config.pion_specs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.set_title(
        rf"Lifetime vs $m_{{\pi_D}}$  — {config.name}"
        "\n"
        rf"({config.mediator_label.split('[')[0].strip()} = "
        rf"{base[config.mediator_param]:.0f} GeV, "
        rf"{config.coupling_label} = {base[config.coupling_param]:.2g}, "
        rf"$f_D$ = {base['fD']:.1f} GeV)"
    )
    ax.legend(handles=species_legend_handles(config.pion_specs), ncol=2)
    fig.tight_layout()
    save_fig(fig, f"dark_pion_lifetime_vs_mpiD_{config.tag}.pdf")


def plot_vs_mediator(config: ModelPlotConfig, n_points: int = 200) -> None:
    base     = dict(config.base_params)
    med      = config.mediator_param
    m_min    = base.pop(f"_{med}_min", 300.0)
    m_max    = base.pop(f"_{med}_max", 1e4)
    values   = np.geomspace(m_min, m_max, n_points)
    ctaus    = _collect_ctaus(med, values, base, config.model_class, config.pion_specs)

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_ctaus(ax, values, ctaus, config.pion_specs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(config.mediator_label)
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.set_title(
        rf"Lifetime vs mediator mass  — {config.name}"
        "\n"
        rf"($m_{{\pi_D}}$ = {base['m_piD']:.1f} GeV, "
        rf"{config.coupling_label} = {base[config.coupling_param]:.2g}, "
        rf"$f_D$ = {base['fD']:.1f} GeV)"
    )
    ax.legend(handles=species_legend_handles(config.pion_specs), ncol=2)
    fig.tight_layout()
    save_fig(fig, f"dark_pion_lifetime_vs_mediator_{config.tag}.pdf")


def plot_vs_coupling(config: ModelPlotConfig, n_points: int = 200) -> None:
    base     = dict(config.base_params)
    coup     = config.coupling_param
    c_min    = base.pop(f"_{coup}_min", 1e-3)
    c_max    = base.pop(f"_{coup}_max", 10.0)
    values   = np.geomspace(c_min, c_max, n_points)
    ctaus    = _collect_ctaus(coup, values, base, config.model_class, config.pion_specs)

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_ctaus(ax, values, ctaus, config.pion_specs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(config.coupling_label)
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.set_title(
        rf"Lifetime vs coupling  — {config.name}"
        "\n"
        rf"($m_{{\pi_D}}$ = {base['m_piD']:.1f} GeV, "
        rf"{config.mediator_label.split('[')[0].strip()} = "
        rf"{base[config.mediator_param]:.0f} GeV, "
        rf"$f_D$ = {base['fD']:.1f} GeV)"
    )
    ax.legend(handles=species_legend_handles(config.pion_specs), ncol=2)
    fig.tight_layout()
    save_fig(fig, f"dark_pion_lifetime_vs_coupling_{config.tag}.pdf")


def plot_vs_fD(config: ModelPlotConfig, n_points: int = 200) -> None:
    base   = dict(config.base_params)
    values = np.geomspace(
        base.pop("_fD_min", 0.5),
        base.pop("_fD_max", 200.0),
        n_points,
    )
    ctaus  = _collect_ctaus("fD", values, base, config.model_class, config.pion_specs)

    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_ctaus(ax, values, ctaus, config.pion_specs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f_D$  [GeV]")
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.set_title(
        rf"Lifetime vs $f_D$  — {config.name}"
        "\n"
        rf"($m_{{\pi_D}}$ = {base['m_piD']:.1f} GeV, "
        rf"{config.mediator_label.split('[')[0].strip()} = "
        rf"{base[config.mediator_param]:.0f} GeV, "
        rf"{config.coupling_label} = {base[config.coupling_param]:.2g})"
    )
    ax.legend(handles=species_legend_handles(config.pion_specs), ncol=2)
    fig.tight_layout()
    save_fig(fig, f"dark_pion_lifetime_vs_fD_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Multi-line figures: representative pion, coloured by mediator mass
# ---------------------------------------------------------------------------

def plot_mpiD_vs_mediator_samples(config: ModelPlotConfig,
                                   mediator_samples=None,
                                   n_points: int = 300) -> None:
    """c·τ vs m_piD for several mediator mass values (representative pion)."""
    if mediator_samples is None:
        mediator_samples = [500, 1000, 2000, 5000, 10000]

    rep_spec   = config.pion_specs[-1]   # last species (diagonal pion typically)
    m_piD_vals = np.geomspace(2.0, 500.0, n_points)
    cmap       = plt.get_cmap("plasma", len(mediator_samples))

    fig, ax = plt.subplots(figsize=(8, 6))
    for ci, mmed in enumerate(mediator_samples):
        base = dict(config.base_params)
        base[config.mediator_param] = mmed
        ctaus = _collect_ctaus("m_piD", m_piD_vals, base,
                               config.model_class, [rep_spec])
        y    = ctaus[rep_spec.pion_id]
        mask = np.isfinite(y)
        med_label = config.mediator_label.split('[')[0].strip()
        ax.plot(m_piD_vals[mask], y[mask], color=cmap(ci), lw=1.8,
                label=rf"{med_label} $= {mmed}$ GeV")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.set_title(
        rf"{rep_spec.label} lifetime vs $m_{{\pi_D}}$ for various mediator masses"
        f"\n{config.name}  "
        rf"({config.coupling_label} = {config.base_params[config.coupling_param]:.2g},"
        rf"  $f_D$ = {config.base_params['fD']:.1f} GeV)"
    )
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"dark_pion_lifetime_mpiD_vs_mediator_{config.tag}.pdf")


def plot_coupling_vs_mediator_samples(config: ModelPlotConfig,
                                       mediator_samples=None,
                                       n_points: int = 300) -> None:
    """c·τ vs coupling for several mediator mass values (representative pion)."""
    if mediator_samples is None:
        mediator_samples = [500, 1000, 2000, 5000, 10000]

    rep_spec  = config.pion_specs[-1]
    coup      = config.coupling_param
    coup_vals = np.geomspace(1e-3, 10.0, n_points)
    cmap      = plt.get_cmap("plasma", len(mediator_samples))

    fig, ax = plt.subplots(figsize=(8, 6))
    for ci, mmed in enumerate(mediator_samples):
        base = dict(config.base_params)
        base[config.mediator_param] = mmed
        ctaus = _collect_ctaus(coup, coup_vals, base,
                               config.model_class, [rep_spec])
        y    = ctaus[rep_spec.pion_id]
        mask = np.isfinite(y)
        med_label = config.mediator_label.split('[')[0].strip()
        ax.plot(coup_vals[mask], y[mask], color=cmap(ci), lw=1.8,
                label=rf"{med_label} $= {mmed}$ GeV")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(config.coupling_label)
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.set_title(
        rf"{rep_spec.label} lifetime vs {config.coupling_label} for various mediator masses"
        f"\n{config.name}  "
        rf"($m_{{\pi_D}}$ = {config.base_params['m_piD']:.1f} GeV,"
        rf"  $f_D$ = {config.base_params['fD']:.1f} GeV)"
    )
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"dark_pion_lifetime_coupling_vs_mediator_{config.tag}.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config: ModelPlotConfig) -> None:
    print(f"  vs m_piD ...")
    plot_vs_m_piD(config)

    print(f"  vs mediator mass ...")
    plot_vs_mediator(config)

    print(f"  vs coupling ...")
    plot_vs_coupling(config)

    print(f"  vs fD ...")
    plot_vs_fD(config)

    print(f"  m_piD for mediator samples ...")
    plot_mpiD_vs_mediator_samples(config)

    print(f"  coupling for mediator samples ...")
    plot_coupling_vs_mediator_samples(config)


if __name__ == "__main__":
    for cfg in ALL_CONFIGS:
        print(f"\n--- {cfg.name} ---")
        main(cfg)
