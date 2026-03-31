#!/usr/bin/env python3
"""
cross_section.py
----------------
Extract mediator pair-production cross sections from MadGraph output
and plot them as a function of mediator mass.

Works with two output layouts:

  1. Slurm grid layout (submit_grid.py):
       --runs-dir runs/
     Each point lives in runs/<point>/Events/<point>/ — the script walks
     all subdirectories of <runs_dir> and searches inside their Events/ tree.

  2. Single process-dir layout:
       --process-dir mg5_output/mediator_pair_down
     Scans <process_dir>/Events/run_*/ as before.

For each run the script reads the banner file (or LHE header) and extracts:
  - mediator mass  (4900001 entry in param_card)
  - kappa coupling (kappa11 entry in param_card)
  - cross section  (Integrated weight line in the header)

Results are grouped by kappa value and plotted as σ vs m_Xd.

Usage
-----
    # Slurm grid output
    python plotting/cross_section.py --runs-dir runs/

    # Single process directory
    python plotting/cross_section.py --process-dir mg5_output/mediator_pair_down

    # Disable kappa grouping
    python plotting/cross_section.py --runs-dir runs/ --no-group
"""

import argparse
import glob
import gzip
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from plotting_utils import apply_style, save_fig

apply_style()

# ---------------------------------------------------------------------------
# Regex patterns (work on both banner files and LHE headers)
# ---------------------------------------------------------------------------
_RE_MASS   = re.compile(r"4900001\s+([0-9.eE+-]+)\s*#\s*[Mm]ass[Xx]")
_RE_KAPPA  = re.compile(r"^\s*1\s+([0-9.eE+-]+)\s*#\s*kappa11", re.MULTILINE)
_RE_XSEC   = re.compile(r"Integrated weight\s*\(pb\)\s*:\s*([0-9.eE+-]+)", re.IGNORECASE)
_RE_XSEC2  = re.compile(r"Cross-section\s*:\s*([0-9.eE+-]+)\s*\+-", re.IGNORECASE)


def _read_text(path: str) -> str:
    """Read a plain or gzipped text file."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", errors="replace") as f:
            return f.read()
    with open(path, "r", errors="replace") as f:
        return f.read()


def _parse_banner(text: str) -> dict | None:
    """
    Extract mass, kappa, and cross section from a banner/LHE header.
    Returns None if any required field is missing.
    """
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


def _parse_run_dir(run_dir: str) -> dict | None:
    """
    Try to parse a single MadGraph run directory.
    Looks for banner files first, then LHE files.
    Returns a result dict or None.
    """
    run_name = os.path.basename(run_dir)
    candidates = sorted(glob.glob(os.path.join(run_dir, "*banner*.txt")))
    candidates += sorted(glob.glob(os.path.join(run_dir, "*.lhe")))
    candidates += sorted(glob.glob(os.path.join(run_dir, "*.lhe.gz")))

    for path in candidates:
        parsed = _parse_banner(_read_text(path))
        if parsed:
            parsed["run"] = run_name
            parsed["source"] = os.path.basename(path)
            return parsed
    return None


def collect_from_process_dir(process_dir: str) -> list[dict]:
    """
    Single-process-dir layout: scan <process_dir>/Events/run_*/.
    """
    run_dirs = sorted(glob.glob(os.path.join(process_dir, "Events", "run_*")))
    if not run_dirs:
        print(f"No run_* directories found under {process_dir}/Events/", file=sys.stderr)
        return []
    return _collect_from_dirs(run_dirs)


def collect_from_runs_dir(runs_dir: str) -> list[dict]:
    """
    Slurm grid layout: each point has its own isolated directory under
    <runs_dir>/<point>/Events/<run_name>/.
    Walk all subdirectories and collect every run found.
    """
    point_dirs = sorted(
        d for d in glob.glob(os.path.join(runs_dir, "*"))
        if os.path.isdir(d)
    )
    if not point_dirs:
        print(f"No subdirectories found under {runs_dir}", file=sys.stderr)
        return []

    run_dirs = []
    for point_dir in point_dirs:
        events_dir = os.path.join(point_dir, "Events")
        if os.path.isdir(events_dir):
            run_dirs += sorted(
                d for d in glob.glob(os.path.join(events_dir, "*"))
                if os.path.isdir(d)
            )

    if not run_dirs:
        print(f"No Events/ run directories found under {runs_dir}", file=sys.stderr)
        return []

    return _collect_from_dirs(run_dirs)


def _collect_from_dirs(run_dirs: list[str]) -> list[dict]:
    results = []
    for run_dir in run_dirs:
        parsed = _parse_run_dir(run_dir)
        if parsed:
            results.append(parsed)
        else:
            print(f"Warning: could not parse results from {run_dir}")
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_xsec(results: list[dict], group_by_kappa: bool, output: str) -> None:
    if not results:
        print("No results to plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    if group_by_kappa:
        groups: dict[float, list] = {}
        for r in results:
            key = float(f"{r['kappa']:.4g}")
            groups.setdefault(key, []).append(r)

        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (kappa, group) in enumerate(sorted(groups.items())):
            group.sort(key=lambda r: r["mXd_GeV"])
            mxd  = [r["mXd_GeV"] for r in group]
            xsec = [r["xsec_pb"] for r in group]
            ax.plot(mxd, xsec, marker="o", label=f"κ = {kappa:.4g}",
                    color=colors[i % len(colors)])
        ax.legend(frameon=False)
    else:
        mxd  = [r["mXd_GeV"] for r in sorted(results, key=lambda r: r["mXd_GeV"])]
        xsec = [r["xsec_pb"] for r in sorted(results, key=lambda r: r["mXd_GeV"])]
        ax.plot(mxd, xsec, marker="o")

    ax.set_xlabel(r"$m_{Xd}$ [GeV]")
    ax.set_ylabel(r"$\sigma$ [pb]")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=[1.0], numticks=100))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0,
                                subs=np.arange(2, 10) * 0.1, numticks=100))

    fig.tight_layout()
    save_fig(fig, output)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MadGraph cross section vs mediator mass.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs-dir",
                        help="Slurm grid output directory (contains one subdir per point)")
    source.add_argument("--process-dir",
                        help="Single MadGraph process directory (contains Events/run_*/)")
    p.add_argument("--output", default="xsec.pdf",
                   help="Output PDF filename (default: xsec.pdf)")
    p.add_argument("--no-group", action="store_true",
                   help="Plot all runs as a single series instead of grouping by κ")
    return p.parse_args()


def main():
    args = parse_args()

    if args.runs_dir:
        if not os.path.isdir(args.runs_dir):
            print(f"ERROR: {args.runs_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)
        results = collect_from_runs_dir(args.runs_dir)
    else:
        if not os.path.isdir(args.process_dir):
            print(f"ERROR: {args.process_dir} is not a directory.", file=sys.stderr)
            sys.exit(1)
        results = collect_from_process_dir(args.process_dir)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'run':<40} {'mXd [GeV]':>10} {'kappa':>10} {'xsec [pb]':>14}")
    print("-" * 78)
    for r in sorted(results, key=lambda r: (r["kappa"], r["mXd_GeV"])):
        print(f"{r['run']:<40} {r['mXd_GeV']:>10.1f} {r['kappa']:>10.4g} {r['xsec_pb']:>14.4e}")
    print()

    plot_xsec(results, group_by_kappa=not args.no_group, output=args.output)


if __name__ == "__main__":
    main()
