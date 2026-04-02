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

# ---------------------------------------------------------------------------
# T-channel summary (reuses the logic from write_cards.py)
# ---------------------------------------------------------------------------

_DIAG_NAMES = {12: "T3", 13: "T8", 14: "T15"}


def _print_tchannel_summary(model: DarkPionTChannelModel, name: str) -> None:
    results = model.compute_all()
    labels  = results["quark_labels"]

    print(f"\n{'='*68}")
    print(f"  {name}")
    print(f"{'='*68}")
    print(f"  mXd    = {model.m_X} GeV")
    print(f"  mPiD   = {model.m_piD} GeV")
    print(f"  fD     = {model.fD} GeV")
    print(f"  κ      = {model.kappa.real:.6f}")
    print(f"  kappa_mode = {model.kappa_mode}")
    print()

    print("  Diagonal dark pions:")
    for b, bname in _DIAG_NAMES.items():
        if b in results["diagonal"]:
            r = results["diagonal"][b]
            print(f"    π^{bname:3s}  ctau = {r['ctau_mm']:.4e} mm  "
                  f"Γ = {r['total']:.3e} GeV")
            for (i, j), br in r["br"].items():
                if br > 1e-4:
                    qi, qj = labels[i], labels[j]
                    ch = f"{qi}{qj}" if i != j else f"{qi}q̄"
                    print(f"             BR({ch}) = {br:.4f}")

    print()
    print("  Off-diagonal dark pions:")
    for ab in [(0, 1), (0, 2), (1, 2)]:
        if ab in results["off_diagonal"]:
            r = results["off_diagonal"][ab]
            a, b = ab
            print(f"    π^({a},{b})  ctau = {r['ctau_mm']:.4e} mm  "
                  f"Γ = {r['total']:.3e} GeV")
            for (i, j), br in r["br"].items():
                if br > 1e-4:
                    print(f"             BR({labels[i]}{labels[j]}) = {br:.4f}")

    print(f"{'='*68}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    args  = parse_args()
    fD    = args.fD if args.fD is not None else args.mPiD

    for cfg in ALL_CONFIGS:
        if cfg.model_class is DarkPionTChannelModel:
            params = dict(cfg.base_params)
            params["fD"]    = fD
            params["m_piD"] = args.mPiD
            params["m_X"]   = args.mXd
            params["kappa"] = args.kappa
            model = DarkPionTChannelModel(**params)
            _print_tchannel_summary(model, cfg.name)

        elif cfg.model_class is DarkPionSChannelModel:
            params = dict(cfg.base_params)
            params["fD"]    = fD
            params["m_piD"] = args.mPiD
            params[cfg.mediator_param]  = args.mXd
            params[cfg.coupling_param]  = args.g_qd
            model = DarkPionSChannelModel(**params)
            model.print_summary()


if __name__ == "__main__":
    main()
