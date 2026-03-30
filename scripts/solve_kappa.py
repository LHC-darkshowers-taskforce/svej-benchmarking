#!/usr/bin/env python3
"""
solve_kappa.py
--------------
For a grid of (m_X, m_piD, c_tau_target) values, compute the coupling κ
required to achieve that lifetime, with the constraint fD = m_piD.

Inversion uses the exact scaling c_tau ∝ κ^{-4}:

    κ_target = ( c_tau(κ=1) / c_tau_target )^{1/4}

Results are printed as a formatted table and saved to kappa_solutions.csv.
"""

import numpy as np
from dark_pion_widths import DarkPionModel

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------
MX_VALS      = [1000, 2000, 3000, 4000, 5000]  # GeV
MPID_VALS    = [5, 10, 20]                      # GeV  (fD = m_piD)
CTAU_TARGETS = [1, 10, 100, 1000]               # mm

# Species: (label, ctau getter)
SPECIES = [
    (r"T15",   lambda m: m.ctau_diag_mm(14)),
    (r"(0,1)", lambda m: m.ctau_off_diag_mm(0, 1)),
    (r"(0,2)", lambda m: m.ctau_off_diag_mm(0, 2)),
]

KAPPA_VALID_RANGE = (0.1, 1.0)

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

def solve_kappa(ctau_at_kappa1: float, ctau_target: float) -> float:
    """
    Invert c_tau ∝ κ^{-4}  →  κ = (ctau_ref / ctau_target)^{1/4}.
    Returns np.nan if the reference width is zero or infinite.
    """
    if not np.isfinite(ctau_at_kappa1) or ctau_at_kappa1 <= 0:
        return np.nan
    return (ctau_at_kappa1 / ctau_target) ** 0.25


rows = []
for mX in MX_VALS:
    for mpiD in MPID_VALS:
        fD = float(mpiD)
        model_ref = DarkPionModel(fD=fD, m_piD=mpiD, m_X=mX, kappa=1.0)

        for spec_label, getter in SPECIES:
            ctau_ref = getter(model_ref)

            for ctau_target in CTAU_TARGETS:
                kappa = solve_kappa(ctau_ref, ctau_target)
                lo, hi = KAPPA_VALID_RANGE
                in_range = (lo <= kappa <= hi) if np.isfinite(kappa) else False
                rows.append({
                    "m_X":      mX,
                    "m_piD":    mpiD,
                    "species":  spec_label,
                    "ctau_mm":  ctau_target,
                    "kappa":    kappa,
                    "in_range": in_range,
                })

# ---------------------------------------------------------------------------
# Pretty-print: one table per species
# ---------------------------------------------------------------------------

def flag(in_range: bool) -> str:
    return "  " if in_range else " *"


for spec_label, _ in SPECIES:
    spec_rows = [r for r in rows if r["species"] == spec_label]

    print(f"\n{'='*72}")
    print(f"  Species: π^{spec_label}   (fD = m_piD,  κ = 1 reference)")
    print(f"{'='*72}")

    # header
    ctau_header = "  ".join(f"c_tau={c:4.0f}mm" for c in CTAU_TARGETS)
    print(f"  {'m_X':>6}  {'m_piD':>6}  |  {ctau_header}")
    print(f"  {'-'*6}  {'-'*6}  |  " + "  ".join(["-"*12]*len(CTAU_TARGETS)))

    # group by (m_X, m_piD)
    for mX in MX_VALS:
        for mpiD in MPID_VALS:
            cell_rows = [r for r in spec_rows
                         if r["m_X"] == mX and r["m_piD"] == mpiD]
            cell_rows.sort(key=lambda r: r["ctau_mm"])
            cells = []
            for r in cell_rows:
                kappa = r["kappa"]
                if np.isfinite(kappa):
                    cells.append(f"{kappa:8.4f}{flag(r['in_range'])}")
                else:
                    cells.append(f"{'—':>8}  ")
            print(f"  {mX:>6}  {mpiD:>6}  |  " + "  ".join(cells))

    print(f"  (* = κ outside [{KAPPA_VALID_RANGE[0]}, {KAPPA_VALID_RANGE[1]}])")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

csv_path = "kappa_solutions.csv"
with open(csv_path, "w") as f:
    f.write("m_X_GeV,m_piD_GeV,fD_GeV,species,ctau_mm,kappa,in_range\n")
    for r in rows:
        kappa_str = f"{r['kappa']:.6f}" if np.isfinite(r['kappa']) else "nan"
        f.write(f"{r['m_X']},{r['m_piD']},{r['m_piD']},"
                f"{r['species']},{r['ctau_mm']},{kappa_str},{r['in_range']}\n")

print(f"\nSaved {csv_path}")
