#!/usr/bin/env python3
"""
Overlay arbitrary extracted samples (e.g. benchmark points outside the grid).

    python plot_benchmarks.py --tag my_comparison \
        --sample "point A=$EJ_OUT/features/pointA" \
        --sample "point B=$EJ_OUT/features/pointB" ...

Each --sample is LABEL=FEATURE_DIR (a directory written by extract.py).
Uses the variable definitions and plotting of plot.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import plot as P


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", action="append", required=True, metavar="LABEL=DIR")
    ap.add_argument("--vars", nargs="+", default=None)
    ap.add_argument("--jet-cut", default="pt > 200 and abs(eta) < 2.5")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)

    labels, smps = [], []
    for spec in a.sample:
        lab, d = spec.rsplit("=", 1)
        labels.append(lab)
        smps.append(P.Sample(Path(d)))
    out = Path(a.outdir) if a.outdir else Path(os.environ.get("EJ_OUT", ".")) / "plots" / a.tag
    P.make_plots(smps, a.vars or P.DEFAULT_VARS, out, a.title, a.jet_cut, labels=labels)


if __name__ == "__main__":
    main()
