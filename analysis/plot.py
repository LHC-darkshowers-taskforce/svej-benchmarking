#!/usr/bin/env python
"""Overlay normalised distributions for any set of extracted samples.

Select samples by type and parameter values; every combination that has been
extracted (meta.json present) is drawn. Examples:

    # vary ctau at fixed mX, mPiD (on-shell)
    python plot.py --samples onshell --mX 2000 --mPiD 12 --tag vary_ctau
    # on-shell vs merged at one point, only a few variables
    python plot.py --samples onshell merged --mX 2000 --mPiD 12 --ctau 10 \
        --vars n_pi met j1_pt pions:Lxy jets:alpha
    # standard comparison set (vary ctau / mX / mPiD / sample)
    python plot.py --preset standard

Variables are "<column>" for the per-event table or "<table>:<column>" for
pions / jets / mediators. Jets can be restricted with --jet-cut (default:
pt>200 & |eta|<2.5 for per-jet plots). Output: one PDF (all pages) and one PNG
per variable under $EJ_OUT/plots/<tag>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import common as C

# validated categorical order (dataviz reference palette), never cycled;
# linestyles provide a secondary (non-colour) encoding.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (8, 2, 2, 2))]

# ---------------------------------------------------------------------------
# Variable registry: (bins, x label, log-x, log-y)
# bins: ("lin", lo, hi, n) | ("log", lo, hi, n) | ("int", lo, hi)
# ---------------------------------------------------------------------------
V = {
    # dark pions (per event)
    "n_pi": (("int", 0, 80), "dark pions / event", False, False),
    "n_pi_decaying": (("int", 0, 60), "decaying dark pions / event", False, False),
    "n_pi_stable": (("int", 0, 40), "stable (invisible) dark pions / event", False, False),
    "frac_pi_stable": (("lin", 0, 1, 25), "stable fraction of dark pions", False, False),
    "n_rho": (("int", 0, 40), "dark rhos / event", False, False),
    "pi_lxy_mean": (("log", 0.1, 1e5, 40), "mean dark-pion $L_{xy}$ [mm]", True, False),
    "pi_lxy_max": (("log", 0.1, 1e6, 40), "max dark-pion $L_{xy}$ [mm]", True, False),
    "pi_lxy_median": (("log", 0.1, 1e5, 40), "median dark-pion $L_{xy}$ [mm]", True, False),
    "n_pi_lxy_lt1": (("int", 0, 40), "dark pions decaying at $L_{xy}<1$ mm", False, False),
    "n_pi_lxy_1_10": (("int", 0, 40), "dark pions decaying at $1<L_{xy}<10$ mm", False, False),
    "n_pi_lxy_10_300": (("int", 0, 40), "dark pions decaying at $10<L_{xy}<300$ mm", False, False),
    "n_pi_lxy_300_3000": (("int", 0, 40), "dark pions decaying at $0.3<L_{xy}<3$ m", False, False),
    "n_pi_lxy_gt3000": (("int", 0, 40), "dark pions decaying at $L_{xy}>3$ m", False, False),
    "n_pi_outside_calo": (("int", 0, 40), "dark pions decaying outside calorimeter", False, False),
    "pi_pt_mean": (("log", 1, 1000, 40), "mean dark-pion $p_T$ [GeV]", True, False),
    "pi_pt_lead": (("log", 10, 3000, 40), "leading dark-pion $p_T$ [GeV]", True, False),
    "f_vis_pt_darkpi": (("lin", 0, 1, 25), "visible $\\sum p_T$ fraction from dark-pion decays", False, False),
    "n_bhad": (("int", 0, 60), "weakly-decaying b hadrons / event", False, False),
    "n_bhad_from_pi": (("int", 0, 60), "b hadrons from dark-pion decays", False, False),
    # hard process / mediator
    "n_X": (("int", 0, 4), "mediators X (last copy) / event", False, False),
    "n_dq_ME": (("int", 0, 4), "dark quarks from ME (not X decay)", False, False),
    "n_sm_parton_hard": (("int", 0, 4), "hard SM partons (not from X)", False, False),
    "X1_pt": (("lin", 0, 2500, 40), "leading X $p_T$ [GeV]", False, False),
    "X1_eta": (("lin", -5, 5, 40), "leading X $\\eta$", False, False),
    "X1_m": (("lin", 0, 5000, 50), "leading X mass [GeV]", False, False),
    # MET / HT / jets
    "met": (("lin", 0, 2000, 40), "true $E_T^{miss}$ (all invisible) [GeV]", False, True),
    "met_sm": (("lin", 0, 1000, 40), "$E_T^{miss}$ from neutrinos [GeV]", False, True),
    "met_dark": (("lin", 0, 2000, 40), "$E_T^{miss}$ from stable dark hadrons [GeV]", False, True),
    "met_det": (("lin", 0, 2000, 40), "detector-like $E_T^{miss}$ [GeV]", False, True),
    "met_over_ht": (("lin", 0, 2, 40), "$E_T^{miss}/H_T$", False, True),
    "dphi_met_jclosest": (("lin", 0, np.pi, 30), "$\\Delta\\phi(E_T^{miss}$, closest jet)", False, False),
    "dphi_metdet_jclosest": (("lin", 0, np.pi, 30), "$\\Delta\\phi(E_T^{miss,det}$, closest jet)", False, False),
    "ht50": (("lin", 0, 5000, 50), "$H_T$ (jets $p_T>50$ GeV) [GeV]", False, False),
    "ht200": (("lin", 0, 5000, 50), "$H_T$ (jets $p_T>200$ GeV) [GeV]", False, False),
    "n_jet50": (("int", 0, 12), "jets $p_T>50$ GeV", False, False),
    "n_jet200": (("int", 0, 8), "jets $p_T>200$ GeV", False, False),
    "n_jet200_c": (("int", 0, 8), "jets $p_T>200$ GeV, $|\\eta|<2.5$", False, False),
    "n_ej_proxy": (("int", 0, 6), "emerging-jet proxies ($\\alpha<0.25$)", False, False),
    "j1_pt": (("lin", 0, 3000, 40), "leading jet $p_T$ [GeV]", False, False),
    "j2_pt": (("lin", 0, 2500, 40), "subleading jet $p_T$ [GeV]", False, False),
    "j1_eta": (("lin", -4.5, 4.5, 36), "leading jet $\\eta$", False, False),
    "j2_eta": (("lin", -4.5, 4.5, 36), "subleading jet $\\eta$", False, False),
    "j1_alpha": (("lin", -0.04, 1.0, 26), "leading jet $\\alpha$ (prompt track $p_T$ fraction)", False, False),
    "j2_alpha": (("lin", -0.04, 1.0, 26), "subleading jet $\\alpha$", False, False),
    "j1_frac_pt_darkpi": (("lin", 0, 1, 25), "leading jet $p_T$ fraction from dark pions", False, False),
    "dR_j1j2": (("lin", 0, 6, 30), "$\\Delta R(j_1, j_2)$", False, False),
    "dphi_j1j2": (("lin", 0, np.pi, 30), "$\\Delta\\phi(j_1, j_2)$", False, False),
    "mjj": (("lin", 0, 5000, 50), "$m_{j_1 j_2}$ [GeV]", False, False),
    "n_trk": (("lin", 0, 400, 40), "tracks ($p_T>1$ GeV, $|\\eta|<2.5$, $L_{xy}^{orig}<300$ mm)", False, False),
    "n_trk_disp": (("lin", 0, 300, 40), "displaced tracks ($2<L_{xy}^{orig}<300$ mm)", False, False),
    "f_trk_disp": (("lin", 0, 1, 25), "displaced-track fraction", False, False),
    # per-pion table
    "pions:Lxy": (("log", 1e-3, 1e6, 45), "dark-pion $L_{xy}$ [mm]", True, True),
    "pions:L": (("log", 1e-3, 1e6, 45), "dark-pion lab decay length [mm]", True, True),
    "pions:ctau": (("log", 1e-3, 1e4, 45), "dark-pion proper decay length $c\\tau$ [mm]", True, True),
    "pions:betagamma": (("log", 0.1, 1e3, 40), "dark-pion $\\beta\\gamma$", True, False),
    "pions:pt": (("log", 0.1, 3000, 40), "dark-pion $p_T$ [GeV]", True, False),
    "pions:eta": (("lin", -6, 6, 48), "dark-pion $\\eta$", False, False),
    "pions:n_bhad_desc": (("int", 0, 4), "b hadrons per dark-pion decay", False, False),
    # per-jet table (selection via --jet-cut)
    "jets:alpha": (("lin", -0.04, 1.0, 26), "jet $\\alpha$ (prompt track $p_T$ fraction; $-$1: no tracks)", False, False),
    "jets:frac_pt_darkpi": (("lin", 0, 1, 25), "jet $p_T$ fraction from dark-pion decays", False, False),
    "jets:frac_pt_disp1": (("lin", 0, 1, 25), "jet $p_T$ fraction with origin $L_{xy}>1$ mm", False, False),
    "jets:frac_pt_disp10": (("lin", 0, 1, 25), "jet $p_T$ fraction with origin $L_{xy}>10$ mm", False, False),
    "jets:n_trk": (("int", 0, 100), "tracks per jet", False, False),
    "jets:n_trk_disp": (("int", 0, 100), "displaced tracks per jet", False, False),
    "jets:n_charged": (("int", 0, 150), "charged constituents per jet", False, False),
    "jets:n_darkpi_dR": (("int", 0, 30), "decaying dark pions within $\\Delta R<1$", False, False),
    "jets:m": (("lin", 0, 800, 40), "jet mass [GeV]", False, False),
    "jets:pt": (("lin", 0, 3000, 40), "jet $p_T$ [GeV]", False, True),
    # mediator table
    "mediators:pt": (("lin", 0, 2500, 40), "X $p_T$ [GeV]", False, False),
    "mediators:eta": (("lin", -5, 5, 40), "X $\\eta$", False, False),
    "mediators:m": (("lin", 0, 5000, 50), "X mass [GeV]", False, False),
    "mediators:q_pt": (("lin", 0, 3000, 40), "SM quark from X decay $p_T$ [GeV]", False, False),
    "mediators:dq_pt": (("lin", 0, 3000, 40), "dark quark from X decay $p_T$ [GeV]", False, False),
    "mediators:dR_q_dq": (("lin", 0, 6, 30), "$\\Delta R$(q, $Q_D$) from X decay", False, False),
}

DEFAULT_VARS = [
    "n_pi", "n_pi_decaying", "n_pi_stable", "frac_pi_stable", "n_rho", "n_X",
    "pi_lxy_mean", "pi_lxy_max", "n_pi_lxy_lt1", "n_pi_lxy_1_10", "n_pi_lxy_10_300",
    "n_pi_lxy_300_3000", "n_pi_lxy_gt3000", "pi_pt_mean", "f_vis_pt_darkpi", "n_bhad",
    "met", "met_sm", "met_dark", "met_det", "met_over_ht", "dphi_met_jclosest",
    "ht50", "n_jet50", "n_jet200", "n_ej_proxy", "j1_pt", "j2_pt", "j1_eta", "j1_alpha", "j2_alpha",
    "dR_j1j2", "dphi_j1j2", "mjj", "n_trk", "n_trk_disp", "f_trk_disp",
    "pions:Lxy", "pions:ctau", "pions:betagamma", "pions:pt", "pions:eta",
    "jets:alpha", "jets:frac_pt_darkpi", "jets:frac_pt_disp1", "jets:n_trk", "jets:n_trk_disp",
    "jets:n_darkpi_dR", "jets:m",
    "mediators:pt", "mediators:eta", "mediators:m", "mediators:dR_q_dq",
]

# which pions enter per-pion plots: decaying species only for decay-length variables
PION_DECAY_VARS = {"Lxy", "L", "ctau", "betagamma", "n_bhad_desc"}


def make_bins(spec):
    kind = spec[0]
    if kind == "lin":
        return np.linspace(spec[1], spec[2], spec[3] + 1)
    if kind == "log":
        return np.logspace(np.log10(spec[1]), np.log10(spec[2]), spec[3] + 1)
    if kind == "int":
        return np.arange(spec[1], spec[2] + 2) - 0.5
    raise ValueError(spec)


class Sample:
    def __init__(self, outdir: Path):
        self.dir = outdir
        self.meta = C.load_meta(outdir)
        self._t = {}

    def table(self, name):
        if name not in self._t:
            self._t[name] = pd.read_parquet(self.dir / f"{name}.parquet")
        return self._t[name]

    @property
    def label(self):
        m = self.meta
        return (f"{C.SAMPLE_SHORT[m['sample']]}  $m_X$={m['mX']}, $m_{{\\pi_D}}$={m['mPiD']}, "
                f"$c\\tau$={m['ctau']} mm")


def select(samples, mX, mPiD, ctau):
    out = []
    for p in C.discover(samples=samples, mX=mX, mPiD=mPiD, ctau=ctau, require_hepmc=False):
        if (p.outdir / "meta.json").exists():
            out.append(Sample(p.outdir))
    return out


def values(s: Sample, var: str, jet_cut: str):
    if ":" in var:
        tab, col = var.split(":", 1)
    else:
        tab, col = "events", var
    df = s.table(tab)
    if df.empty or col not in df.columns:
        return np.array([])
    if tab == "pions" and col in PION_DECAY_VARS:
        df = df[df.decayed]
    if tab == "jets" and jet_cut:
        df = df.query(jet_cut)
    v = df[col].to_numpy(dtype=float)
    return v[np.isfinite(v)]


def plot_var(ax, smps, var, jet_cut, labels):
    spec, xlabel, logx, logy = V.get(var, (None, var, False, False))
    arrays = [values(s, var, jet_cut) for s in smps]
    if spec is None:
        allv = np.concatenate([a for a in arrays if len(a)]) if any(len(a) for a in arrays) else np.zeros(1)
        lo, hi = np.nanpercentile(allv, [0.5, 99.5])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 41)
    else:
        bins = make_bins(spec)
    for k, (a, lab) in enumerate(zip(arrays, labels)):
        if len(a) == 0:
            continue
        # clip into the visible range so over/underflow stay counted at the edges
        ac = np.clip(a, bins[0], bins[-1] * (1 - 1e-12) if bins[-1] > 0 else bins[-1] - 1e-12)
        h, _ = np.histogram(ac, bins=bins)
        h = h / h.sum()
        ax.stairs(h, bins, color=COLORS[k % len(COLORS)], ls=STYLES[k % len(STYLES)], lw=2,
                  label=f"{lab}  (mean {np.mean(a):.3g})")
    ax.set_xlabel(xlabel + ("  [over/underflow in edge bins]" if spec else ""), fontsize=9)
    ax.set_ylabel("fraction of entries")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, frameon=False)


def make_plots(smps, vars_, outdir: Path, title: str, jet_cut: str, labels=None):
    outdir.mkdir(parents=True, exist_ok=True)
    if len(smps) > len(COLORS):
        raise SystemExit(f"{len(smps)} samples selected; at most {len(COLORS)} per overlay "
                         "(narrow the selection or split into several plots)")
    labels = labels or [s.label for s in smps]
    with PdfPages(outdir / "all.pdf") as pdf:
        for var in vars_:
            fig, ax = plt.subplots(figsize=(7, 4.8))
            plot_var(ax, smps, var, jet_cut, labels)
            sub = f"  [jets: {jet_cut}]" if var.startswith("jets:") and jet_cut else ""
            ax.set_title(title + sub, fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(outdir / f"{var.replace(':', '__')}.png", dpi=110)
            plt.close(fig)
    with open(outdir / "samples.json", "w") as f:
        json.dump([str(s.dir) for s in smps], f, indent=1)
    print(f"wrote {len(vars_)} plots for {len(smps)} samples -> {outdir}")


def common_label(smps):
    """Shorter per-sample labels: drop parameters shared by all selected samples."""
    keys = ["sample", "mX", "mPiD", "ctau"]
    vary = [k for k in keys if len({s.meta[k] for s in smps}) > 1]
    fixed = [k for k in keys if k not in vary]
    fmt = {"sample": lambda v: C.SAMPLE_LABEL[v], "mX": lambda v: f"$m_X$={v} GeV",
           "mPiD": lambda v: f"$m_{{\\pi_D}}$={v} GeV", "ctau": lambda v: f"$c\\tau$={v} mm"}
    labels = [", ".join(fmt[k](s.meta[k]) for k in vary) or s.label for s in smps]
    title = ", ".join(fmt[k](smps[0].meta[k]) for k in fixed)
    return labels, title


PRESETS = {
    # tag: (samples, mX, mPiD, ctau)
    "onshell_vary_ctau_mX2000_mPiD12": (["mediator_onshell"], [2000], [12], None),
    "merged_vary_ctau_mX2000_mPiD12": (["dark_quark_pair_multijet"], [2000], [12], None),
    "onshell_vary_mX_mPiD12_ctau10": (["mediator_onshell"], None, [12], [10]),
    "merged_vary_mX_mPiD12_ctau10": (["dark_quark_pair_multijet"], None, [12], [10]),
    "onshell_vary_mPiD_mX2000_ctau10": (["mediator_onshell"], [2000], None, [10]),
    "merged_vary_mPiD_mX2000_ctau10": (["dark_quark_pair_multijet"], [2000], None, [10]),
    "onshell_vs_merged_mX2000_mPiD12_ctau10": (list(C.SAMPLES), [2000], [12], [10]),
    "onshell_vs_merged_mX2000_mPiD6_ctau10": (list(C.SAMPLES), [2000], [6], [10]),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", default=["onshell", "merged"],
                    help="onshell / merged (or full sample names)")
    ap.add_argument("--mX", nargs="+", type=int)
    ap.add_argument("--mPiD", nargs="+", type=int)
    ap.add_argument("--ctau", nargs="+", type=int)
    ap.add_argument("--vars", nargs="+", default=None, help="variables (default: standard set)")
    ap.add_argument("--jet-cut", default="pt > 200 and abs(eta) < 2.5",
                    help="pandas query applied to jets:* plots ('' for none)")
    ap.add_argument("--tag", default=None, help="output subdirectory name under plots/")
    ap.add_argument("--outdir", type=Path, default=C.PLOTS)
    ap.add_argument("--preset", choices=["standard"] + list(PRESETS),
                    help="'standard' runs every preset comparison with >=2 extracted samples")
    ap.add_argument("--list-vars", action="store_true")
    a = ap.parse_args(argv)

    if a.list_vars:
        for k, v in V.items():
            print(f"{k:28s} {v[1]}")
        return
    vars_ = a.vars or DEFAULT_VARS

    jobs = []
    if a.preset:
        names = list(PRESETS) if a.preset == "standard" else [a.preset]
        for n in names:
            jobs.append((n, *PRESETS[n]))
    else:
        samples = [C.SHORT_TO_SAMPLE.get(s, s) for s in a.samples]
        tag = a.tag or "custom"
        jobs.append((tag, samples, a.mX, a.mPiD, a.ctau))

    for tag, samples, mX, mPiD, ctau in jobs:
        smps = select(samples, mX, mPiD, ctau)
        if len(smps) < (2 if a.preset else 1):
            print(f"[{tag}] only {len(smps)} extracted sample(s) match; skipped")
            continue
        labels, title = common_label(smps)
        make_plots(smps, vars_, a.outdir / tag, title, a.jet_cut, labels)


if __name__ == "__main__":
    main()
