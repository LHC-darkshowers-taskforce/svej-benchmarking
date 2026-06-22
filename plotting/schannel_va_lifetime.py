#!/usr/bin/env python3
"""
schannel_va_lifetime.py
-----------------------
c·τ comparison plots for vector (a_d=0) vs axial (a_d=1) s-channel models.
Nf=3, dark_charges=[1,-2,1], fD=m_piD (where m_piD is scanned or fixed).

One PDF per scanned parameter:
  schannel_va_ctau_vs_mzp.pdf
  schannel_va_ctau_vs_gqd.pdf
  schannel_va_ctau_vs_gq.pdf
  schannel_va_ctau_vs_mpiD.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

from dark_pion_widths import DarkPionSChannelModel
from plotting_utils import apply_style, save_fig, add_param_box

apply_style()

# ---------------------------------------------------------------------------
# Fixed defaults
# ---------------------------------------------------------------------------
DARK_CHARGES = [1.0, -2.0, 1.0]
MZP_DEFAULT  = 2000.0
MPID_DEFAULT = 10.0
FD_DEFAULT   = 10.0
GQD_DEFAULT  = 0.1
GQ_DEFAULT   = 0.01

T3_IDX = 6
T8_IDX = 7

_PIONS = [
    (T3_IDX, r"$\pi^{T_3}$"),
    (T8_IDX, r"$\pi^{T_8}$"),
]

_MODELS = [
    dict(a_d=0.0, label="vector ($a_d=0$)", color_T3="tab:blue",  color_T8="tab:cyan",   ls="-"),
    dict(a_d=1.0, label="axial ($a_d=1$)",  color_T3="tab:red",   color_T8="tab:orange", ls="--"),
]

_COLORS = {
    (0.0, T3_IDX): "tab:blue",
    (0.0, T8_IDX): "tab:cyan",
    (1.0, T3_IDX): "tab:red",
    (1.0, T8_IDX): "tab:orange",
}

# ---------------------------------------------------------------------------
# Generic scan helper
# ---------------------------------------------------------------------------

def _scan(x_vals, scan_param, a_d, pion_idx,
          m_Zp=MZP_DEFAULT, m_piD=MPID_DEFAULT, fD=FD_DEFAULT,
          g_qd=GQD_DEFAULT, g_q=GQ_DEFAULT):
    """
    Scan one parameter over x_vals, return c·τ array.
    scan_param: 'm_Zp' | 'g_qd' | 'g_q' | 'm_piD'
    For 'm_piD', fD is set equal to m_piD at each point.
    """
    ctau = np.full(len(x_vals), np.nan)
    for ki, xv in enumerate(x_vals):
        kw = dict(m_Zp=m_Zp, m_piD=m_piD, fD=fD, g_qd=g_qd, g_q=g_q)
        kw[scan_param] = float(xv)
        if scan_param == "m_piD":
            kw["fD"] = float(xv)
        model = DarkPionSChannelModel(
            Nf=3, Nd=3, dark_charges=DARK_CHARGES,
            a_d=float(a_d), mix_diagonal=False, **kw,
        )
        info = model.compute_diagonal_pion(pion_idx)
        if info["total"] > 0:
            ctau[ki] = info["ctau_mm"]
    return ctau


# ---------------------------------------------------------------------------
# Generic plot builder
# ---------------------------------------------------------------------------

def _make_plot(x_vals, scan_param, xlabel, fname, param_box_lines):
    fig, ax = plt.subplots(figsize=(9, 6))

    for style in _MODELS:
        a_d = style["a_d"]
        for pion_idx, pion_label in _PIONS:
            color = _COLORS[(a_d, pion_idx)]
            ctau  = _scan(x_vals, scan_param, a_d, pion_idx)
            mask  = np.isfinite(ctau)
            ax.plot(x_vals[mask], ctau[mask],
                    color=color, ls=style["ls"], lw=1.8,
                    label=rf"{pion_label} {style['label']}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$c\tau$  [mm]")
    ax.legend(fontsize=10)
    add_param_box(ax, param_box_lines + [r"$q = [1, -2, 1]$"], loc="upper left")
    fig.tight_layout()
    save_fig(fig, fname)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

_make_plot(
    np.geomspace(300.0, 1e4, 300),
    scan_param = "m_Zp",
    xlabel     = r"$m_{Z'}$  [GeV]",
    fname      = "schannel_va_ctau_vs_mzp.pdf",
    param_box_lines = [
        rf"$m_{{\pi_D}} = {MPID_DEFAULT:.0f}$ GeV,  $f_D = {FD_DEFAULT:.0f}$ GeV",
        rf"$g_{{qd}} = {GQD_DEFAULT}$,  $g_q = {GQ_DEFAULT}$",
    ],
)

_make_plot(
    np.geomspace(1e-3, 1.0, 300),
    scan_param = "g_qd",
    xlabel     = r"$g_{qd}$",
    fname      = "schannel_va_ctau_vs_gqd.pdf",
    param_box_lines = [
        rf"$m_{{\pi_D}} = {MPID_DEFAULT:.0f}$ GeV,  $f_D = {FD_DEFAULT:.0f}$ GeV",
        rf"$m_{{Z'}} = {MZP_DEFAULT:.0f}$ GeV,  $g_q = {GQ_DEFAULT}$",
    ],
)

_make_plot(
    np.geomspace(1e-4, 1.0, 300),
    scan_param = "g_q",
    xlabel     = r"$g_q$",
    fname      = "schannel_va_ctau_vs_gq.pdf",
    param_box_lines = [
        rf"$m_{{\pi_D}} = {MPID_DEFAULT:.0f}$ GeV,  $f_D = {FD_DEFAULT:.0f}$ GeV",
        rf"$m_{{Z'}} = {MZP_DEFAULT:.0f}$ GeV,  $g_{{qd}} = {GQD_DEFAULT}$",
    ],
)

_make_plot(
    np.geomspace(1.0, 100.0, 300),
    scan_param = "m_piD",
    xlabel     = r"$m_{\pi_D}$  [GeV]",
    fname      = "schannel_va_ctau_vs_mpiD.pdf",
    param_box_lines = [
        rf"$f_D = m_{{\pi_D}}$",
        rf"$m_{{Z'}} = {MZP_DEFAULT:.0f}$ GeV,  $g_{{qd}} = {GQD_DEFAULT}$,  $g_q = {GQ_DEFAULT}$",
    ],
)
