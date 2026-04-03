"""
model_config.py
---------------
Standard ModelPlotConfig instances for each supported model variant.

Available configs
-----------------
TCHANNEL_UNIVERSAL  — SU(4) t-channel, κ_αi = κ  (original benchmark)
TCHANNEL_DIAGONAL   — SU(4) t-channel, κ_αi = κ δ_αi
TCHANNEL_DOWNONLY   — SU(4) t-channel, κ_α0 = κ (only SM d quark)
SCHANNEL_DEFAULT    — S-channel Z', Nf=3, dark_charges=[1,2,3]

Each config can be iterated:
    from model_config import ALL_CONFIGS
    for cfg in ALL_CONFIGS:
        main(cfg)
"""

from dark_pion_widths import DarkPionTChannelModel, DarkPionSChannelModel
from plotting_utils import ModelPlotConfig, PionSpec

# ---------------------------------------------------------------------------
# T-channel colour palette
# ---------------------------------------------------------------------------
# Off-diagonal pions
_OD_COLORS = {
    (0, 1): "tab:blue",
    (0, 2): "tab:orange",
    (1, 2): "tab:green",
}
_OD_LS = {
    (0, 1): "-",
    (0, 2): ":",
    (1, 2): "--",
}
# Diagonal pions (generator indices 12=T3, 13=T8, 14=T15 for Nf=4)
_DIAG_COLORS = {12: "tab:red", 13: "tab:purple", 14: "tab:brown"}
_DIAG_LS     = {12: "-",       13: "--",          14: "-."}
_DIAG_LABELS = {12: r"$\pi^{\mathrm{T3}}$",
                13: r"$\pi^{\mathrm{T8}}$",
                14: r"$\pi^{\mathrm{T15}}$"}

# S-channel colour palette (Nf=3, dark_charges=[1,2,3]: pions 6 and 7 decay)
_SC_COLORS = {6: "tab:red",  7: "tab:purple"}
_SC_LS     = {6: "-",        7: "--"}


def _od_spec(pair: tuple, label: str | None = None) -> PionSpec:
    a, b = pair
    lbl  = label or rf"$\pi^{{({a},{b})}}$"
    return PionSpec(pair, lbl, _OD_COLORS[pair], _OD_LS[pair])


def _diag_spec(idx: int) -> PionSpec:
    return PionSpec(idx, _DIAG_LABELS[idx], _DIAG_COLORS[idx], _DIAG_LS[idx])


# ---------------------------------------------------------------------------
# T-channel: universal κ  (Nf=4, kappa_mode="universal")
# ---------------------------------------------------------------------------
# Nonzero pions: (0,1), (0,2), (1,2), T15 (index 14)
# T3 and T8 are zero because Σ_α T_{αα} = 0 for active block.

TCHANNEL_UNIVERSAL = ModelPlotConfig(
    name              = "T-channel (universal κ)",
    tag               = "tchannel_universal",
    model_class       = DarkPionTChannelModel,
    base_params       = dict(fD=10.0, m_piD=10.0, m_X=2000.0, kappa=1.0,
                             Nf=4, kappa_mode="universal"),
    mediator_param    = "m_X",
    mediator_label    = r"$m_X$  [GeV]",
    coupling_param    = "kappa",
    coupling_label    = r"$|\kappa|$",
    pion_specs        = [
        _od_spec((0, 1)),
        _od_spec((0, 2)),
        _od_spec((1, 2)),
        _diag_spec(14),
    ],
    contour_pion_specs = [
        _diag_spec(14),
        _od_spec((0, 1)),
        _od_spec((0, 2)),
    ],
)

# ---------------------------------------------------------------------------
# T-channel: diagonal κ  (Nf=4, kappa_mode="diagonal")
# ---------------------------------------------------------------------------
# Nonzero pions: (0,1)→ds̄, (0,2)→db̄, (1,2)→sb̄, T3, T8, T15

TCHANNEL_DIAGONAL = ModelPlotConfig(
    name              = "T-channel (diagonal κ)",
    tag               = "tchannel_diagonal",
    model_class       = DarkPionTChannelModel,
    base_params       = dict(fD=10.0, m_piD=10.0, m_X=2000.0, kappa=1.0,
                             Nf=3, kappa_mode="diagonal"),
    mediator_param    = "m_X",
    mediator_label    = r"$m_X$  [GeV]",
    coupling_param    = "kappa",
    coupling_label    = r"$|\kappa|$",
    pion_specs        = [
        _od_spec((0, 1)),
        _od_spec((0, 2)),
        _od_spec((1, 2)),
        _diag_spec(12),
        _diag_spec(13),
        _diag_spec(14),
    ],
    contour_pion_specs = [
        _diag_spec(14),
        _diag_spec(13),
        _od_spec((0, 1)),
    ],
)

# ---------------------------------------------------------------------------
# T-channel: down-only κ  (Nf=4, kappa_mode="down_only")
# ---------------------------------------------------------------------------
# Nonzero pions: (0,1)→dd̄, (0,2)→dd̄, (1,2)→dd̄, T15 (T3, T8 still zero)

TCHANNEL_DOWNONLY = ModelPlotConfig(
    name              = "T-channel (down-only κ)",
    tag               = "tchannel_downonly",
    model_class       = DarkPionTChannelModel,
    base_params       = dict(fD=10.0, m_piD=10.0, m_X=2000.0, kappa=1.0,
                             Nf=4, kappa_mode="down_only"),
    mediator_param    = "m_X",
    mediator_label    = r"$m_X$  [GeV]",
    coupling_param    = "kappa",
    coupling_label    = r"$|\kappa|$",
    pion_specs        = [
        _od_spec((0, 1)),
        _od_spec((0, 2)),
        _od_spec((1, 2)),
        _diag_spec(14),
    ],
    contour_pion_specs = [
        _diag_spec(14),
        _od_spec((0, 1)),
        _od_spec((0, 2)),
    ],
)

# ---------------------------------------------------------------------------
# S-channel: default  (Nf=3, dark_charges=[1,2,3])
# ---------------------------------------------------------------------------
# Decaying pions: indices 6 (T3-like, A=-3) and 7 (T8-like, A≈-7.51)

SCHANNEL_DEFAULT = ModelPlotConfig(
    name              = "S-channel Z' (Nf=3, q=[1,2,3])",
    tag               = "schannel_default",
    model_class       = DarkPionSChannelModel,
    base_params       = dict(
        Nf=3, Nd=3, dark_charges=[1.0, 2.0, 3.0],
        fD=10.0, m_piD=10.0, m_Zp=2000.0,
        g_qd=0.5, g_q=0.5, a_d=0.0,
    ),
    mediator_param    = "m_Zp",
    mediator_label    = r"$m_{Z'}$  [GeV]",
    coupling_param    = "g_qd",
    coupling_label    = r"$g_{qd}$",
    pion_specs        = [
        PionSpec(6, r"$\pi^{T_3}$",  _SC_COLORS[6], _SC_LS[6]),
        PionSpec(7, r"$\pi^{T_8}$",  _SC_COLORS[7], _SC_LS[7]),
    ],
    contour_pion_specs = [
        PionSpec(6, r"$\pi^{T_3}$",  _SC_COLORS[6], _SC_LS[6]),
        PionSpec(7, r"$\pi^{T_8}$",  _SC_COLORS[7], _SC_LS[7]),
    ],
)

# ---------------------------------------------------------------------------
# Convenience list
# ---------------------------------------------------------------------------

TCHANNEL_CONFIGS: list[ModelPlotConfig] = [
    TCHANNEL_UNIVERSAL,
    TCHANNEL_DIAGONAL,
    TCHANNEL_DOWNONLY,
]

ALL_CONFIGS: list[ModelPlotConfig] = TCHANNEL_CONFIGS + [SCHANNEL_DEFAULT]
