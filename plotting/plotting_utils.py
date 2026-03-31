"""
plotting_utils.py
-----------------
Shared constants, style setup, and helpers for all dark-QCD plotting scripts.
"""

import matplotlib.pyplot as plt
import mplhep
from matplotlib.lines import Line2D

from dark_pion_widths import DIAGONAL_PION_NAMES

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def apply_style():
    """Apply mplhep ATLAS style (no ATLAS label)."""
    mplhep.style.use("ATLAS")


def save_fig(fig, path: str) -> None:
    """Save figure, print confirmation, and close."""
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pion species — colors, labels, legend handles
# ---------------------------------------------------------------------------

# Off-diagonal dark pions
OFF_DIAG_PAIRS  = [(0, 1), (0, 2), (1, 2)]
OFF_DIAG_COLORS = {(0, 1): "tab:blue", (0, 2): "tab:orange", (1, 2): "tab:green"}
OFF_DIAG_LABELS = {(a, b): rf"$\pi^{{({a},{b})}}$" for a, b in OFF_DIAG_PAIRS}

# Diagonal dark pions
DIAG_INDICES = [2, 7, 14]
DIAG_COLORS  = {2: "tab:red", 7: "tab:purple", 14: "tab:brown"}
DIAG_LABELS  = {b: rf"$\pi^{{\mathrm{{{DIAGONAL_PION_NAMES[b]}}}}}$"
                for b in DIAG_INDICES}

# Contour species descriptor: (key, label, linestyle, linewidth)
CONTOUR_SPECIES = [
    ("diag_14", r"$\pi^{T15}$",   "-",  1.8),
    ("off_01",  r"$\pi^{(0,1)}$", "--", 1.6),
    ("off_02",  r"$\pi^{(0,2)}$", ":",  1.8),
]


def species_legend_handles():
    """Legend handles for all off-diagonal + diagonal pion species (1-D plots)."""
    handles = [
        Line2D([0], [0], color=OFF_DIAG_COLORS[p], ls="-", lw=1.8,
               label=OFF_DIAG_LABELS[p])
        for p in OFF_DIAG_PAIRS
    ]
    handles += [
        Line2D([0], [0], color=DIAG_COLORS[b], ls="--", lw=1.8,
               label=DIAG_LABELS[b])
        for b in DIAG_INDICES
    ]
    return handles


def contour_species_handles():
    """Legend handles for the three contour species (solid/dashed/dotted)."""
    return [
        Line2D([0], [0], color="k", ls=ls, lw=lw, label=label)
        for _, label, ls, lw in CONTOUR_SPECIES
    ]


# ---------------------------------------------------------------------------
# c·τ contour levels — colors and labels
# ---------------------------------------------------------------------------

CTAU_LEVELS_MM = [1.0, 10.0, 100.0, 1000.0]
CTAU_COLORS    = {1.0: "#e41a1c", 10.0: "#ff7f00", 100.0: "#377eb8", 1000.0: "#4daf4a"}
CTAU_LABELS    = {v: rf"$c\tau = {v:.0f}$ mm" for v in CTAU_LEVELS_MM}


def ctau_legend_handles():
    """Legend handles for the standard c·τ iso-contour levels."""
    return [
        Line2D([0], [0], color=CTAU_COLORS[v], lw=2, label=CTAU_LABELS[v])
        for v in CTAU_LEVELS_MM
    ]
