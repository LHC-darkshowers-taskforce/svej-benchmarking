"""
plotting_utils.py
-----------------
Shared style setup, data structures, and helpers for all dark-QCD plotting
scripts.  Model-specific pion species and parameter names are encapsulated in
ModelPlotConfig objects defined in model_config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import mplhep
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply mplhep ATLAS style (no ATLAS label)."""
    mplhep.style.use("ATLAS")


def save_fig(fig, path: str) -> None:
    """Save figure, print confirmation, and close."""
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-pion plot specification
# ---------------------------------------------------------------------------

@dataclass
class PionSpec:
    """Describes how to display one pion species in a plot."""
    pion_id:   object        # (alpha, beta) tuple or int
    label:     str           # LaTeX label, e.g. r"$\pi^{T15}$"
    color:     str           # matplotlib color string
    linestyle: str  = "-"
    linewidth: float = 1.8


# ---------------------------------------------------------------------------
# Per-model plot configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelPlotConfig:
    """
    All metadata needed to run generic plotting functions for one model variant.

    Attributes
    ----------
    name : str
        Human-readable description, e.g. "T-channel (universal κ)".
    tag : str
        Short identifier used in output filenames, e.g. "tchannel_universal".
    model_class : type
        The model class to instantiate (DarkPionTChannelModel or
        DarkPionSChannelModel).
    base_params : dict
        Default keyword arguments passed to model_class(...).
    mediator_param : str
        Name of the mediator mass parameter, e.g. "m_X" or "m_Zp".
    mediator_label : str
        LaTeX axis label for the mediator mass.
    coupling_param : str
        Name of the main coupling parameter, e.g. "kappa" or "g_qd".
    coupling_label : str
        LaTeX axis label for the coupling.
    pion_specs : list[PionSpec]
        Ordered list of decaying pion species to show in 1-D plots.
    contour_pion_specs : list[PionSpec]
        Subset of pion_specs used in contour/iso-lifetime plots (≤ 3 species).
    """
    name:                str
    tag:                 str
    model_class:         type
    base_params:         dict
    mediator_param:      str
    mediator_label:      str
    coupling_param:      str
    coupling_label:      str
    pion_specs:          list[PionSpec]
    contour_pion_specs:  list[PionSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.contour_pion_specs:
            self.contour_pion_specs = self.pion_specs[:3]


# ---------------------------------------------------------------------------
# c·τ contour levels — shared across all models
# ---------------------------------------------------------------------------

CTAU_LEVELS_MM: list[float] = [1.0, 10.0, 100.0, 1000.0]
CTAU_COLORS:    dict[float, str] = {
    1.0:    "#e41a1c",
    10.0:   "#ff7f00",
    100.0:  "#377eb8",
    1000.0: "#4daf4a",
}
CTAU_LABELS: dict[float, str] = {v: rf"$c\tau = {v:.0f}$ mm" for v in CTAU_LEVELS_MM}


def ctau_legend_handles() -> list:
    """Legend handles for the standard c·τ iso-contour levels."""
    return [
        Line2D([0], [0], color=CTAU_COLORS[v], lw=2, label=CTAU_LABELS[v])
        for v in CTAU_LEVELS_MM
    ]


# ---------------------------------------------------------------------------
# Generic species legend helpers
# ---------------------------------------------------------------------------

def species_legend_handles(pion_specs: list[PionSpec]) -> list:
    """Legend handles for a list of PionSpec objects (1-D plots)."""
    return [
        Line2D([0], [0], color=s.color, ls=s.linestyle, lw=s.linewidth,
               label=s.label)
        for s in pion_specs
    ]


def contour_species_handles(contour_specs: list[PionSpec]) -> list:
    """Legend handles for contour species, drawn in black with species linestyle."""
    return [
        Line2D([0], [0], color="k", ls=s.linestyle, lw=s.linewidth,
               label=s.label)
        for s in contour_specs
    ]


def add_param_box(ax, lines: list[str], loc: str = "lower right") -> None:
    """
    Add a small parameter-summary text box to `ax`.

    Parameters
    ----------
    ax    : matplotlib Axes
    lines : list of strings, each a parameter line, e.g. ["$g_{qd}=0.1$", "$g_q=0.01$"]
    loc   : 'upper right', 'upper left', 'lower right', 'lower left'
    """
    text = "\n".join(lines)
    x_map = {"right": 0.97, "left": 0.03}
    y_map = {"upper": 0.97, "lower": 0.03}
    va_map = {"upper": "top", "lower": "bottom"}
    ha_map = {"right": "right", "left": "left"}

    parts  = loc.split()
    y_key, x_key = parts[0], parts[1]
    ax.text(
        x_map[x_key], y_map[y_key], text,
        transform=ax.transAxes,
        fontsize=9,
        va=va_map[y_key], ha=ha_map[x_key],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )
