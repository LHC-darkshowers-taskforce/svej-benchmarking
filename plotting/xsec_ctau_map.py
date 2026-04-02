#!/usr/bin/env python3
"""
xsec_ctau_map.py
----------------
2D map of production cross-section in the (m_Xd, c·τ) plane, with an overlaid
contour marking the N-event discovery threshold at a given luminosity.

Data sources (mutually exclusive):
  --dummy        Generate analytic dummy data (σ ∝ 1/m_Xd · 1/(c·τ)²).
  --runs-dir     Slurm grid output (one subdirectory per parameter point).
  --process-dir  Single MadGraph process directory (Events/run_*/).

For real MadGraph runs, c·τ is read from the run directory name if it follows
the submit_grid.py naming convention (…_ctau<value>…); otherwise it is
computed from κ and the model parameters via DarkPionModel.

Usage
-----
    python plotting/xsec_ctau_map.py --dummy
    python plotting/xsec_ctau_map.py --runs-dir runs/
    python plotting/xsec_ctau_map.py --process-dir mg5_output/mediator_pair_down
"""

import argparse
import glob
import gzip
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

from plotting_utils import apply_style, save_fig

apply_style()

# ---------------------------------------------------------------------------
# Discovery threshold
# ---------------------------------------------------------------------------
LUMI_FB    = 300.0      # fb⁻¹
N_SIGNAL   = 3.0        # required signal events
XSEC_THRESHOLD_PB = N_SIGNAL / (LUMI_FB * 1e3)   # fb⁻¹ → pb⁻¹


# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------

def make_dummy_data(
    mxd_vals  = np.geomspace(500,  5000, 15),
    ctau_vals = np.geomspace(1,    3000, 15),
    norm_pb   = 10.0,          # σ at mXd=1000 GeV, c·τ=1 mm
    mxd_ref   = 1000.0,
    ctau_ref  = 1.0,
) -> list[dict]:
    """
    Generate dummy (mXd, ctau, xsec) points with σ ∝ 1/m_Xd · 1/(c·τ)².
    """
    records = []
    C = norm_pb * mxd_ref * ctau_ref**2
    for mxd in mxd_vals:
        for ctau in ctau_vals:
            xsec = C / (mxd * ctau**2)
            records.append({"mXd_GeV": mxd, "ctau_mm": ctau, "xsec_pb": xsec})
    return records


# ---------------------------------------------------------------------------
# Real data: parse MadGraph run directories
# ---------------------------------------------------------------------------
_RE_MASS  = re.compile(r"4900001\s+([0-9.eE+-]+)\s*#\s*[Mm]ass[Xx]")
_RE_KAPPA = re.compile(r"^\s*1\s+([0-9.eE+-]+)\s*#\s*kappa11", re.MULTILINE)
_RE_XSEC  = re.compile(r"Integrated weight\s*\(pb\)\s*:\s*([0-9.eE+-]+)", re.IGNORECASE)
_RE_XSEC2 = re.compile(r"Cross-section\s*:\s*([0-9.eE+-]+)\s*\+-", re.IGNORECASE)
_RE_CTAU_NAME = re.compile(r"[_-]ctau([0-9]+\.?[0-9]*)")


def _read_text(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    with open(path, "r", errors="replace") as f:
        return f.read()


def _parse_banner(text: str) -> dict | None:
    mass_m  = _RE_MASS.search(text)
    kappa_m = _RE_KAPPA.search(text)
    xsec_m  = _RE_XSEC.search(text) or _RE_XSEC2.search(text)
    if not (mass_m and kappa_m and xsec_m):
        return None
    return {
        "mXd_GeV": float(mass_m.group(1)),
        "kappa":   float(kappa_m.group(1)),
        "xsec_pb": float(xsec_m.group(1)),
    }


def _parse_run_dir(run_dir: str, mpiD: float, fD: float | None, species: str) -> dict | None:
    run_name = os.path.basename(run_dir)
    candidates = (
        sorted(glob.glob(os.path.join(run_dir, "*banner*.txt")))
        + sorted(glob.glob(os.path.join(run_dir, "*.lhe")))
        + sorted(glob.glob(os.path.join(run_dir, "*.lhe.gz")))
    )
    parsed = None
    for path in candidates:
        parsed = _parse_banner(_read_text(path))
        if parsed:
            break
    if parsed is None:
        print(f"Warning: could not parse results from {run_dir}")
        return None

    # Resolve c·τ: try run-name first, then compute from model
    ctau_m = _RE_CTAU_NAME.search(run_name)
    if ctau_m:
        parsed["ctau_mm"] = float(ctau_m.group(1))
    else:
        try:
            from dark_pion_widths import DarkPionModel
            fD_val = fD if fD is not None else mpiD
            model  = DarkPionModel(fD=fD_val, m_piD=mpiD,
                                   m_X=parsed["mXd_GeV"], kappa=parsed["kappa"])
            if species == "T15":
                parsed["ctau_mm"] = model.ctau_diag_mm(14)
            elif species == "(0,1)":
                parsed["ctau_mm"] = model.ctau_off_diag_mm(0, 1)
            else:
                parsed["ctau_mm"] = model.ctau_diag_mm(14)
        except Exception as e:
            print(f"Warning: could not compute c·τ for {run_name}: {e}")
            return None

    parsed["run"] = run_name
    return parsed


def _collect_from_dirs(run_dirs, mpiD, fD, species) -> list[dict]:
    return [r for rd in run_dirs
            if (r := _parse_run_dir(rd, mpiD, fD, species)) is not None]


def collect_from_runs_dir(runs_dir, mpiD, fD, species) -> list[dict]:
    point_dirs = sorted(d for d in glob.glob(os.path.join(runs_dir, "*"))
                        if os.path.isdir(d))
    run_dirs = []
    for pd in point_dirs:
        ev = os.path.join(pd, "Events")
        if os.path.isdir(ev):
            run_dirs += sorted(d for d in glob.glob(os.path.join(ev, "*"))
                               if os.path.isdir(d))
    return _collect_from_dirs(run_dirs, mpiD, fD, species)


def collect_from_process_dir(process_dir, mpiD, fD, species) -> list[dict]:
    run_dirs = sorted(glob.glob(os.path.join(process_dir, "Events", "run_*")))
    return _collect_from_dirs(run_dirs, mpiD, fD, species)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_xsec_ctau_map(records: list[dict], output: str) -> None:
    if not records:
        print("No records to plot.", file=sys.stderr)
        return

    mxd  = np.array([r["mXd_GeV"] for r in records])
    ctau = np.array([r["ctau_mm"]  for r in records])
    xsec = np.array([r["xsec_pb"]  for r in records])

    # Interpolate onto a fine regular grid (log-log space)
    log_mxd  = np.log10(mxd)
    log_ctau = np.log10(ctau)

    mxd_grid  = np.linspace(log_mxd.min(),  log_mxd.max(),  300)
    ctau_grid = np.linspace(log_ctau.min(), log_ctau.max(), 300)
    MXD, CTAU = np.meshgrid(mxd_grid, ctau_grid)

    XSEC = griddata(
        np.column_stack([log_mxd, log_ctau]),
        np.log10(np.clip(xsec, 1e-20, None)),
        (MXD, CTAU),
        method="cubic",
    )
    # Convert back to linear for LogNorm
    XSEC_LIN = 10**XSEC

    fig, ax = plt.subplots(figsize=(9, 6))

    # Color map: σ with log scale
    pcm = ax.pcolormesh(
        10**MXD, 10**CTAU, XSEC_LIN,
        cmap="viridis",
        norm=mcolors.LogNorm(vmin=np.nanmin(XSEC_LIN), vmax=np.nanmax(XSEC_LIN)),
        shading="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$\sigma$  [pb]")

    # N=3 signal contour
    from matplotlib.lines import Line2D
    cs = ax.contour(
        10**MXD, 10**CTAU, XSEC_LIN,
        levels=[XSEC_THRESHOLD_PB],
        colors=["red"],
        linewidths=[2.0],
        linestyles=["-"],
    )
    contour_handle = Line2D([0], [0], color="red", lw=2.0,
                            label=rf"$N={N_SIGNAL:.0f}$, $\mathcal{{L}}={LUMI_FB:.0f}\ \mathrm{{fb}}^{{-1}}$")
    ax.legend(handles=[contour_handle], loc="upper right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_{Xd}$  [GeV]")
    ax.set_ylabel(r"$c\tau$  [mm]")

    fig.tight_layout()
    save_fig(fig, output)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot σ vs (m_Xd, c·τ) with a discovery-reach contour.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--dummy",       action="store_true",
                        help="Use analytic dummy data (σ ∝ 1/m_Xd · 1/c·τ²)")
    source.add_argument("--runs-dir",    metavar="DIR",
                        help="Slurm grid output directory")
    source.add_argument("--process-dir", metavar="DIR",
                        help="Single MadGraph process directory (Events/run_*/)")

    p.add_argument("--mpiD",    type=float, default=10.0,
                   help="Dark pion mass [GeV] for c·τ computation (default: 10)")
    p.add_argument("--fD",      type=float, default=None,
                   help="Dark decay constant [GeV] (default: mpiD)")
    p.add_argument("--species", default="T15",
                   choices=["T15", "(0,1)"],
                   help="Pion species for c·τ computation (default: T15)")
    p.add_argument("--lumi",    type=float, default=LUMI_FB,
                   help=f"Luminosity [fb⁻¹] for N-event contour (default: {LUMI_FB})")
    p.add_argument("--n-events", type=float, default=N_SIGNAL,
                   help=f"Target signal events for contour (default: {N_SIGNAL})")
    p.add_argument("--output",  default="xsec_ctau_map.pdf",
                   help="Output PDF filename (default: xsec_ctau_map.pdf)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    global LUMI_FB, N_SIGNAL, XSEC_THRESHOLD_PB
    LUMI_FB            = args.lumi
    N_SIGNAL           = args.n_events
    XSEC_THRESHOLD_PB  = N_SIGNAL / (LUMI_FB * 1e3)

    if args.dummy:
        records = make_dummy_data()
        print(f"Generated {len(records)} dummy data points.")
    elif args.runs_dir:
        if not os.path.isdir(args.runs_dir):
            print(f"ERROR: {args.runs_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)
        records = collect_from_runs_dir(args.runs_dir, args.mpiD, args.fD, args.species)
    else:
        if not os.path.isdir(args.process_dir):
            print(f"ERROR: {args.process_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)
        records = collect_from_process_dir(args.process_dir, args.mpiD, args.fD, args.species)

    if not records:
        print("No records found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'run/point':<45} {'mXd [GeV]':>10} {'ctau [mm]':>10} {'σ [pb]':>14}")
    print("-" * 85)
    for r in sorted(records, key=lambda r: (r["mXd_GeV"], r["ctau_mm"])):
        label = r.get("run", "dummy")
        print(f"{label:<45} {r['mXd_GeV']:>10.1f} {r['ctau_mm']:>10.2f} {r['xsec_pb']:>14.4e}")
    print(f"\nN={N_SIGNAL:.0f} threshold: σ = {XSEC_THRESHOLD_PB:.3e} pb  "
          f"(L = {LUMI_FB:.0f} fb⁻¹)\n")

    plot_xsec_ctau_map(records, args.output)


if __name__ == "__main__":
    main()
