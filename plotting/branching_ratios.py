#!/usr/bin/env python3
"""
branching_ratios.py
-------------------
Branching ratios of dark pions as a function of m_piD, with fD = m_piD.

BRs are independent of the coupling magnitude (κ or g_qd) and mediator mass
since these cancel in the ratio; only the kinematic thresholds matter.

Output: one PDF per model config.

T-channel plots
---------------
Two panels per figure:
  Left  — Representative diagonal pion (T15 for universal/down-only,
           T15 and T8 shown separately for diagonal κ).
  Right — Representative off-diagonal pion (0,1); for universal and
          down-only κ all off-diagonal species share identical BRs.

S-channel plots
---------------
One panel per figure: decaying dark pion (all decaying species share the same
BRs for pure vector coupling; shown for the first decaying pion index).
"""

import numpy as np
import matplotlib.pyplot as plt

from dark_pion_widths import DarkPionTChannelModel, DarkPionSChannelModel
from plotting_utils import apply_style, save_fig, ModelPlotConfig
from model_config import ALL_CONFIGS

apply_style()

# ---------------------------------------------------------------------------
# T-channel branching-ratio logic
# ---------------------------------------------------------------------------

_CHAN_COLORS_TCHAN = {
    (0, 0): "#2166ac",
    (0, 1): "#74add1",
    (0, 2): "#abd9e9",
    (1, 1): "#f4a582",
    (1, 2): "#d6604d",
    (2, 2): "#b2182b",
}
_CHAN_PAIRS_TCHAN = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _tchan_chan_label(quarks, i, j, combined=False):
    if i == j:
        return rf"${quarks[i]}\bar{{{quarks[i]}}}$"
    if combined:
        return rf"${quarks[i]}\bar{{{quarks[j]}}}$ + ${quarks[j]}\bar{{{quarks[i]}}}$"
    return rf"${quarks[i]}\bar{{{quarks[j]}}}$"


def _collect_tchan_brs(config, pion_id, m_piD_vals):
    """Return {chan_pair: np.ndarray of BR} for a single t-channel pion."""
    br_arrays = {p: np.full(len(m_piD_vals), np.nan) for p in _CHAN_PAIRS_TCHAN}
    ref = dict(config.base_params)

    for ki, mpiD in enumerate(m_piD_vals):
        ref["fD"]    = mpiD
        ref["m_piD"] = mpiD
        model = config.model_class(**ref)

        if isinstance(pion_id, tuple):
            res = model.compute_off_diagonal_pion(*pion_id)
        else:
            res = model.compute_diagonal_pion(pion_id)

        if res["total"] <= 0:
            continue

        raw = res["br"]
        if isinstance(pion_id, int):
            # diagonal: combine (i,j) and (j,i) into the unordered pair
            combined: dict = {}
            for (i, j), br in raw.items():
                key = (min(i, j), max(i, j))
                combined[key] = combined.get(key, 0.0) + br
            for p in _CHAN_PAIRS_TCHAN:
                br_arrays[p][ki] = combined.get(p, 0.0)
        else:
            for p in _CHAN_PAIRS_TCHAN:
                br_arrays[p][ki] = raw.get(p, 0.0)

    return br_arrays


def _draw_tchan_panel(ax, br_dict, m_piD_vals, title, quarks, combined_labels):
    ys     = [np.nan_to_num(br_dict[p], nan=0.0) for p in _CHAN_PAIRS_TCHAN]
    colors = [_CHAN_COLORS_TCHAN[p] for p in _CHAN_PAIRS_TCHAN]
    labels = [_tchan_chan_label(quarks, *p, combined=combined_labels)
              for p in _CHAN_PAIRS_TCHAN]

    # Only include channels that ever have nonzero BR
    nonzero = [i for i, y in enumerate(ys) if np.nanmax(y) > 1e-6]
    if not nonzero:
        ax.text(0.5, 0.5, "No open channels", ha="center", va="center",
                transform=ax.transAxes)
        return

    ax.stackplot(m_piD_vals, [ys[i] for i in nonzero],
                 labels=[labels[i] for i in nonzero],
                 colors=[colors[i] for i in nonzero],
                 alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(m_piD_vals[0], m_piD_vals[-1])
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def plot_tchannel_brs(config: ModelPlotConfig) -> None:
    m_piD_vals = np.geomspace(1.0, 50.0, 100)

    # Determine which diagonal and off-diagonal pions to show
    ref   = dict(config.base_params)
    ref["fD"] = ref["m_piD"] = 10.0
    model_ref = config.model_class(**ref)
    quarks    = model_ref._q_labels

    diag_ids  = [s.pion_id for s in config.pion_specs if isinstance(s.pion_id, int)]
    od_ids    = [s.pion_id for s in config.pion_specs if isinstance(s.pion_id, tuple)]

    # One figure per diagonal pion (plus one figure for off-diagonal representative)
    all_panel_ids = diag_ids + (od_ids[:1] if od_ids else [])
    for pion_id in all_panel_ids:
        br_data = _collect_tchan_brs(config, pion_id, m_piD_vals)
        combined = isinstance(pion_id, int)   # combine (i,j)+(j,i) for diagonal

        if isinstance(pion_id, int):
            name  = model_ref._diagonal_names.get(pion_id, f"T{pion_id+1}")
            title = rf"$\pi^{{\mathrm{{{name}}}}}$ (diagonal)"
            fname = f"dark_pion_branching_ratios_{config.tag}_{name.lower()}.pdf"
        else:
            a, b  = pion_id
            title = rf"$\pi^{{({a},{b})}}$ off-diagonal"
            fname = f"dark_pion_branching_ratios_{config.tag}_offdiag_{a}{b}.pdf"

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(
            rf"Dark pion branching ratios vs $m_{{\pi_D}}$  ($f_D = m_{{\pi_D}}$)"
            f"\n{config.name}"
        )
        _draw_tchan_panel(ax, br_data, m_piD_vals, title, quarks, combined_labels=combined)

        # Threshold markers
        masses = dict(zip(quarks, model_ref._q_masses))
        if "b" in masses:
            mb = masses["b"]
            ax.axvline(2 * mb, color="gray", lw=1.2, ls="--", alpha=0.8)
            ax.text(2 * mb * 1.04, 0.98, r"$2m_b$",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8, color="gray", va="top")

        ax.set_ylabel("Branching ratio")
        fig.tight_layout()
        save_fig(fig, fname)


# ---------------------------------------------------------------------------
# S-channel branching-ratio logic
# ---------------------------------------------------------------------------

_SM_QUARK_COLORS = {
    "u": "#2166ac", "d": "#4dac26", "s": "#74add1",
    "c": "#f4a582", "b": "#d6604d",
}
_FOUR_BODY_COLOR = "#762a83"


def _collect_schan_brs(config, pion_index, m_piD_vals):
    """Return {channel_key: np.ndarray of BR} for an s-channel pion."""
    ref_params = dict(config.base_params)
    # Determine quark order from a reference model
    ref_params["fD"] = ref_params["m_piD"] = m_piD_vals[0]
    ref_model  = config.model_class(**ref_params)
    q_labels   = [q for q in ref_model._q_labels if q != "t"]

    channels   = q_labels + ["anom_4body"]
    br_arrays  = {c: np.full(len(m_piD_vals), np.nan) for c in channels}

    for ki, mpiD in enumerate(m_piD_vals):
        params = dict(config.base_params)
        params["fD"]    = mpiD
        params["m_piD"] = mpiD
        model = config.model_class(**params)
        info  = model.compute_diagonal_pion(pion_index)

        if info["total"] <= 0:
            continue

        br = info["br"]
        for q in q_labels:
            br_arrays[q][ki] = br.get(f"anom_2body_{q}", 0.0)
        br_arrays["anom_4body"][ki] = br.get("anom_4body", 0.0)

    return br_arrays, q_labels


def _draw_schan_panel(ax, br_dict, q_labels, title, m_piD_vals, masses):
    ys_quarks = [np.nan_to_num(br_dict[q], nan=0.0) for q in q_labels]
    ys_4body  = np.nan_to_num(br_dict["anom_4body"], nan=0.0)
    colors    = [_SM_QUARK_COLORS.get(q, "gray") for q in q_labels]
    labels    = [rf"${q}\bar{{{q}}}$" for q in q_labels]

    ax.stackplot(m_piD_vals, ys_quarks + [ys_4body],
                 labels=labels + ["4-body (param.)"],
                 colors=colors + [_FOUR_BODY_COLOR],
                 alpha=0.85)

    for q, thresh_label in [("c", r"$2m_c$"), ("b", r"$2m_b$")]:
        if q in masses:
            m = masses[q]
            ax.axvline(2 * m, color="gray", lw=1.2, ls="--", alpha=0.8)
            ax.text(2 * m * 1.04, 0.98, thresh_label,
                    transform=ax.get_xaxis_transform(),
                    fontsize=8, color="gray", va="top")

    ax.set_xscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(m_piD_vals[0], m_piD_vals[-1])
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


def plot_schannel_brs(config: ModelPlotConfig) -> None:
    m_piD_vals = np.geomspace(0.5, 100.0, 200)

    # Reference model for quark masses
    ref        = dict(config.base_params)
    ref["fD"]  = ref["m_piD"] = 1.0
    ref_model  = config.model_class(**ref)
    masses     = dict(zip(ref_model._q_labels, ref_model._q_masses))

    rep_pion = config.pion_specs[0].pion_id if config.pion_specs else 6

    br_pion, q_labels = _collect_schan_brs(config, rep_pion, m_piD_vals)
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig1.suptitle(
        rf"Dark pion branching ratios vs $m_{{\pi_D}}$  ($f_D = m_{{\pi_D}}$)"
        f"\n{config.name}"
    )
    _draw_schan_panel(ax1, br_pion, q_labels,
                      r"Decaying $\pi_D$ (same BRs for all species via anomaly)",
                      m_piD_vals, masses)
    ax1.set_ylabel("Branching ratio")
    fig1.tight_layout()
    save_fig(fig1, f"dark_pion_branching_ratios_{config.tag}_pion.pdf")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def main(config: ModelPlotConfig) -> None:
    if issubclass(config.model_class, DarkPionTChannelModel):
        plot_tchannel_brs(config)
    elif issubclass(config.model_class, DarkPionSChannelModel):
        plot_schannel_brs(config)
    else:
        raise ValueError(f"Unknown model class: {config.model_class}")


if __name__ == "__main__":
    for cfg in ALL_CONFIGS:
        print(f"\n--- {cfg.name} ---")
        main(cfg)
