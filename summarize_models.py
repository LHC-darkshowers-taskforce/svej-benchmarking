#!/usr/bin/env python3
"""
summarize_models.py
-------------------
Print a physics summary for each benchmark model configuration.

Usage:
    python summarize_models.py
    python summarize_models.py --mXd 1500 --mPiD 10 --kappa 0.5
"""

import argparse
from dark_pion_widths import DarkPionTChannelModel, DarkPionSChannelModel
from plotting.model_config import ALL_CONFIGS



def parse_args():
    p = argparse.ArgumentParser(
        description="Print physics summaries for all benchmark model configs."
    )
    p.add_argument("--mXd",   type=float, default=1500.0, help="Mediator mass [GeV]")
    p.add_argument("--mPiD",  type=float, default=10.0,   help="Dark pion mass [GeV]")
    p.add_argument("--kappa", type=float, default=0.4,    help="Coupling κ (t-channel)")
    p.add_argument("--fD",    type=float, default=None,   help="Dark decay constant [GeV] (default: mPiD)")
    p.add_argument("--g_qd",  type=float, default=0.5,    help="Z' coupling g_qd (s-channel)")
    return p.parse_args()


def main():
    args = parse_args()
    fD   = args.fD if args.fD is not None else args.mPiD

    for cfg in ALL_CONFIGS:
        params = dict(cfg.base_params)
        params["fD"]    = fD
        params["m_piD"] = args.mPiD
        params[cfg.mediator_param] = args.mXd

        if cfg.model_class is DarkPionTChannelModel:
            params["kappa"] = args.kappa
        elif cfg.model_class is DarkPionSChannelModel:
            params[cfg.coupling_param] = args.g_qd

        model = cfg.model_class(**params)
        model.print_summary()
        print()


if __name__ == "__main__":
    main()
