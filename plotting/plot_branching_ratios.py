#!/usr/bin/env python3
"""
plot_branching_ratios.py
------------------------
Branching ratios of dark pions as a function of m_piD, with fD = m_piD.

BRs are independent of κ, m_X, and fD (all cancel in the ratio); only the
kinematic factor Ω(mi, mj; m_piD) matters, so the only dependence is on the
quark mass thresholds.

Two panels:
  Left  — π^T15 (diagonal, neutral): includes both qi q̄j and q̄i qj orderings,
           shown as combined channels.
  Right — π^(0,1) (off-diagonal, representative): all off-diagonal species
           (0,1), (0,2), (1,2) have identical BRs for universal κ.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from dark_pion_widths import DarkPionModel, DEFAULT_QUARKS

# ---------------------------------------------------------------------------
# Scan range
# ---------------------------------------------------------------------------
MPID_VALS = np.geomspace(1.0, 50.0, 100)
QUARKS    = list(DEFAULT_QUARKS.keys())   # ["d", "s", "b"]
MASSES    = list(DEFAULT_QUARKS.values())

# Reference model — κ, m_X, fD don't affect BRs, so use any convenient values;
# fD=m_piD is set inside the loop.
M_X_REF   = 2000.0
KAPPA_REF = 1.0

# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------
# Unordered pairs (i, j) with i <= j for display
CHAN_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

CHAN_COLORS = {
    (0, 0): "#2166ac",   # dd̄
    (0, 1): "#74add1",   # ds̄
    (0, 2): "#abd9e9",   # db̄
    (1, 1): "#f4a582",   # ss̄
    (1, 2): "#d6604d",   # sb̄
    (2, 2): "#b2182b",   # bb̄
}


def _chan_label(i, j, combined=False):
    q = QUARKS
    if i == j:
        return rf"${q[i]}\bar{{{q[i]}}}$"
    elif combined:
        return rf"${q[i]}\bar{{{q[j]}}}$ + ${q[j]}\bar{{{q[i]}}}$"
    else:
        return rf"${q[i]}\bar{{{q[j]}}}$"


# ---------------------------------------------------------------------------
# Collect BRs vs m_piD
# ---------------------------------------------------------------------------
# br_diag[chan_pair][mpiD_idx]   — diagonal T15, combined (i,j)+(j,i) for i<j
# br_offdiag[chan_pair][mpiD_idx] — off-diagonal π^(0,1), j>=i only

br_diag   = {p: np.full(len(MPID_VALS), np.nan) for p in CHAN_PAIRS}
br_offdiag = {p: np.full(len(MPID_VALS), np.nan) for p in CHAN_PAIRS}

for ki, mpiD in enumerate(MPID_VALS):
    model = DarkPionModel(fD=mpiD, m_piD=mpiD, m_X=M_X_REF, kappa=KAPPA_REF)

    # --- diagonal T15 ---
    res = model.compute_diagonal_pion(14)
    if res["total"] > 0:
        raw = res["br"]   # keys: (i,j) including both orderings for i!=j
        # Combine (i,j) and (j,i) into a single (min,max) pair
        combined = {}
        for (i, j), br in raw.items():
            key = (min(i, j), max(i, j))
            combined[key] = combined.get(key, 0.0) + br
        for p in CHAN_PAIRS:
            br_diag[p][ki] = combined.get(p, 0.0)

    # --- off-diagonal π^(0,1) ---
    res = model.compute_off_diagonal_pion(0, 1)
    if res["total"] > 0:
        for p in CHAN_PAIRS:
            br_offdiag[p][ki] = res["br"].get(p, 0.0)

# ---------------------------------------------------------------------------
# Quark threshold lines visible in the plot range
# ---------------------------------------------------------------------------
THRESHOLDS = {}
masses = dict(zip(QUARKS, MASSES))
pairs_for_thresh = [("b", "b")]   # only bb threshold falls in [1, 500] GeV
for qa, qb in pairs_for_thresh:
    thresh = masses[qa] + masses[qb]
    label  = rf"$2m_b$" if qa == qb else rf"$m_{{{qa}}}+m_{{{qb}}}$"
    THRESHOLDS[thresh] = label

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle(
    r"Dark pion branching ratios vs $m_{\pi_D}$  ($f_D = m_{\pi_D}$)",
    fontsize=13,
)

def _draw_panel(ax, br_dict, title, combined_labels):
    # Build stacked arrays — replace NaN with 0 so stackplot works everywhere
    ys     = [np.nan_to_num(br_dict[p], nan=0.0) for p in CHAN_PAIRS]
    colors = [CHAN_COLORS[p] for p in CHAN_PAIRS]
    labels = [_chan_label(*p, combined=combined_labels) for p in CHAN_PAIRS]

    ax.stackplot(MPID_VALS, ys, labels=labels, colors=colors, alpha=0.85)

    for thresh, tlbl in THRESHOLDS.items():
        ax.axvline(thresh, color="gray", lw=1.2, ls="--", alpha=0.8)
        ax.text(thresh * 1.04, 0.98, tlbl,
                transform=ax.get_xaxis_transform(),
                fontsize=8, color="gray", va="top")

    ax.set_xscale("log")
    ax.set_xlabel(r"$m_{\pi_D}$  [GeV]", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(MPID_VALS[0], MPID_VALS[-1])
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


_draw_panel(ax1, br_diag,
            r"$\pi^{T15}$ (diagonal)",
            combined_labels=True)
_draw_panel(ax2, br_offdiag,
            r"$\pi^{(0,1)}$ off-diagonal  [same for all off-diag. species]",
            combined_labels=False)

ax1.set_ylabel("Branching ratio", fontsize=12)

fig.tight_layout()
out = "dark_pion_branching_ratios.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
plt.close(fig)
