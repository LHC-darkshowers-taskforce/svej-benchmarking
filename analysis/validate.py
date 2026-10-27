#!/usr/bin/env python
"""Sanity checks and grid summary for extracted samples.

For every extracted sample (or a filtered subset) this:
  * compares the mean proper decay length <ctau> of each decaying dark-pion
    species with the card tau0 (exponential => <ctau> = tau0, sigma = tau0/sqrt(N));
    also an MLE fit with the decay-length window correction is not needed since
    Pythia decays every pion (no fiducial cut);
  * checks that dark pions are produced at the primary vertex;
  * summarises the per-event observables (means) incl. cross sections.
Writes $EJ_OUT/validation/ctau_check.csv, summary.csv and ctau plots.

    python validate.py                       # everything extracted so far
    python validate.py --samples onshell --mX 2000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C
from plot import select, COLORS

SUMMARY_COLS = [
    "n_pi", "n_pi_decaying", "n_pi_stable", "frac_pi_stable", "n_rho", "n_X", "n_dq_ME",
    "n_sm_parton_hard", "pi_lxy_mean", "pi_lxy_median", "n_pi_lxy_lt1", "n_pi_lxy_1_10",
    "n_pi_lxy_10_300", "n_pi_lxy_300_3000", "n_pi_lxy_gt3000", "n_pi_outside_calo",
    "f_vis_pt_darkpi", "n_bhad", "met", "met_sm", "met_dark", "met_det", "ht50",
    "n_jet200", "n_ej_proxy", "j1_pt", "j2_pt", "n_trk_disp", "f_trk_disp",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", default=["onshell", "merged"])
    ap.add_argument("--mX", nargs="+", type=int)
    ap.add_argument("--mPiD", nargs="+", type=int)
    ap.add_argument("--ctau", nargs="+", type=int)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    samples = [C.SHORT_TO_SAMPLE.get(s, s) for s in a.samples]
    smps = select(samples, a.mX, a.mPiD, a.ctau)
    outdir = C.OUT / "validation"
    outdir.mkdir(parents=True, exist_ok=True)

    ctau_rows, summ_rows = [], []
    for s in smps:
        m = s.meta
        pi = s.table("pions")
        ev = s.table("events")
        tau0 = {int(k): v for k, v in m["card"]["tau0_mm"].items()}
        dec = pi[pi.decayed]
        # pions should come from the PV (hadronisation / prompt rho decays)
        r_prod = np.sqrt(pi.prod_x**2 + pi.prod_y**2 + pi.prod_z**2)
        for apid, g in dec.groupby("apid"):
            t0 = tau0.get(int(apid), np.nan)
            n = len(g)
            mean = g.ctau.mean()
            err = mean / np.sqrt(n)
            ctau_rows.append(dict(sample=C.SAMPLE_SHORT[m["sample"]], point=m["name"], species=C.PI_NAME[apid],
                                  apid=apid, n=n, tau0_card=t0, ctau_mean=mean, ctau_err=err,
                                  ratio=mean / t0, pull=(mean - t0) / (t0 / np.sqrt(n)),
                                  ctau_t_vs_L_maxreldiff=float(np.nanmax(np.abs(g.ctau_t / g.ctau - 1)))
                                  if n else np.nan))
        # stable species must never decay; decaying species must always decay
        n_bad_stable = int(pi[~pi.decaying_species & pi.decayed].shape[0])
        n_undecayed = int(pi[pi.decaying_species & ~pi.decayed].shape[0])
        row = dict(sample=C.SAMPLE_SHORT[m["sample"]], point=m["name"], mX=m["mX"], mPiD=m["mPiD"],
                   ctau=m["ctau"], n_events=m["n_events"], xsec_pb=m.get("xsec_pb"),
                   xsec_mg5_pb=m.get("xsec_mg5_pb"), sigmaGen_fb=m.get("sigmaGen_fb"),
                   time_s=m.get("processing_time_s"),
                   pi_prod_r_max_mm=float(r_prod.max()) if len(r_prod) else np.nan,
                   n_stable_species_decayed=n_bad_stable, n_decaying_species_undecayed=n_undecayed,
                   frac_events_nX0=float((ev.n_X == 0).mean()), frac_events_nX1=float((ev.n_X == 1).mean()),
                   frac_events_nX2=float((ev.n_X == 2).mean()))
        for c in SUMMARY_COLS:
            row[c] = float(ev[c].mean())
        summ_rows.append(row)

        if not a.no_plots and len(dec):
            fig, ax = plt.subplots(figsize=(7, 4.8))
            for k, (apid, g) in enumerate(dec.groupby("apid")):
                t0 = tau0.get(int(apid), np.nan)
                x = g.ctau / t0
                bins = np.linspace(0, 8, 41)
                h, _ = np.histogram(np.clip(x, 0, 8 - 1e-9), bins=bins)
                ax.stairs(h / h.sum() / np.diff(bins), bins, color=COLORS[k], lw=2,
                          label=f"{C.PI_NAME[apid]} ($\\tau_0$={t0:g} mm): <c$\\tau$>/$\\tau_0$="
                                f"{x.mean():.3f}±{x.mean() / np.sqrt(len(x)):.3f}")
            xx = np.linspace(0, 8, 200)
            ax.plot(xx, np.exp(-xx), color="#52514e", lw=1, ls="--", label="exp($-c\\tau/\\tau_0$)")
            ax.set_yscale("log")
            ax.set_xlabel("proper decay length $c\\tau = L\\,m/|p|$ divided by card $\\tau_0$")
            ax.set_ylabel("probability density")
            ax.set_title(f"{s.label}", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            fig.savefig(outdir / f"ctau_{m['sample']}_{m['name']}.png", dpi=110)
            plt.close(fig)

    cdf = pd.DataFrame(ctau_rows)
    sdf = pd.DataFrame(summ_rows)
    cdf.to_csv(outdir / "ctau_check.csv", index=False)
    sdf.to_csv(outdir / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 2000)
    if len(cdf):
        print("=== dark-pion proper ctau vs card tau0 (per species) ===")
        print(cdf.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"\npull distribution over {len(cdf)} species x samples: mean {cdf.pull.mean():.2f}, "
              f"rms {cdf.pull.std():.2f}, max |pull| {cdf.pull.abs().max():.2f}")
    if len(sdf):
        print("\n=== per-sample summary (event means) ===")
        print(sdf.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nwrote {outdir}/ctau_check.csv, summary.csv")


if __name__ == "__main__":
    main()
