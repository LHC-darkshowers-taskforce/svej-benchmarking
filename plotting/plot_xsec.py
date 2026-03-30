#!/usr/bin/env python3
"""
plot_xsec.py
------------
Extract mediator pair-production cross sections from MadGraph output
and plot them as a function of mediator mass.

For each run found under <process_dir>/Events/run_*/, the script reads
the banner file (or the LHE file header) and extracts:
  - mediator mass  (4900001 entry in param_card)
  - kappa coupling (kappa11 entry in param_card)
  - cross section  (Integrated weight line in the header)

Results are grouped by kappa value and plotted as σ vs m_Xd.

Usage
-----
    python plotting/plot_xsec.py --process-dir mg5_output/mediator_pair_down
    python plotting/plot_xsec.py --process-dir mg5_output/mediator_pair_down \\
                                  --output xsec.pdf --no-group
"""

import argparse
import glob
import gzip
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Regex patterns (work on both banner files and LHE headers)
# ---------------------------------------------------------------------------
_RE_MASS   = re.compile(r"4900001\s+([0-9.eE+-]+)\s*#\s*[Mm]ass[Xx]")
_RE_KAPPA  = re.compile(r"^\s*1\s+([0-9.eE+-]+)\s*#\s*kappa11", re.MULTILINE)
_RE_XSEC   = re.compile(r"Integrated weight\s*\(pb\)\s*:\s*([0-9.eE+-]+)", re.IGNORECASE)
# Also handle the banner-file format: "Cross-section :  X.XX +- X.XX pb"
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


def collect_results(process_dir: str) -> list[dict]:
    """
    Walk <process_dir>/Events/run_*/ and extract one result per run.
    Tries banner files first, then LHE files (plain or gzipped).
    """
    results = []
    run_dirs = sorted(glob.glob(os.path.join(process_dir, "Events", "run_*")))

    if not run_dirs:
        print(f"No run directories found under {process_dir}/Events/", file=sys.stderr)
        return results

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        # Priority 1: banner file
        candidates = sorted(glob.glob(os.path.join(run_dir, "*banner*.txt")))
        # Priority 2: LHE files
        candidates += sorted(glob.glob(os.path.join(run_dir, "*.lhe")))
        candidates += sorted(glob.glob(os.path.join(run_dir, "*.lhe.gz")))

        parsed = None
        for path in candidates:
            text = _read_text(path)
            parsed = _parse_banner(text)
            if parsed:
                parsed["run"] = run_name
                parsed["source"] = os.path.basename(path)
                break

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
        # Round kappa to 4 sig figs for grouping
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
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot MadGraph cross section vs mediator mass."
    )
    p.add_argument("--process-dir", required=True,
                   help="MadGraph process directory (contains Events/run_*/)")
    p.add_argument("--output", default="xsec.pdf",
                   help="Output PDF filename (default: xsec.pdf)")
    p.add_argument("--no-group", action="store_true",
                   help="Plot all runs as a single series instead of grouping by κ")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.process_dir):
        print(f"ERROR: {args.process_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    results = collect_results(args.process_dir)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # Print table
    print(f"\n{'run':<12} {'mXd [GeV]':>10} {'kappa':>10} {'xsec [pb]':>14}")
    print("-" * 50)
    for r in sorted(results, key=lambda r: (r["kappa"], r["mXd_GeV"])):
        print(f"{r['run']:<12} {r['mXd_GeV']:>10.1f} {r['kappa']:>10.4g} {r['xsec_pb']:>14.4e}")
    print()

    plot_xsec(results, group_by_kappa=not args.no_group, output=args.output)


if __name__ == "__main__":
    main()
