#!/usr/bin/env python3
"""
write_cards.py
--------------
Generate a MadGraph param_card.dat, run_card.dat, Pythia input card,
and a launch shell script for a given dark QCD parameter point.
Computes the coupling κ required to achieve the target c·τ, then fills
in pion masses, lifetimes, and branching ratios calculated from the
dark_pion_widths module.

Usage
-----
    python write_cards.py --mXd 1500 --ctau 100 --mPiD 10
    python write_cards.py --mXd 2000 --ctau 50 --mPiD 15 --fD 15 \\
                          --LambdaD 4 --process-dir mg5_output/mediator_pair_down

Output
------
  <output>/param_card.dat   — MadGraph parameter card
  <output>/run_card.dat     — MadGraph run card
  <output>/pythia_card.dat  — Pythia 8 settings card
  <output>/launch.sh        — Shell script: copies cards and runs MG5 in batch

All pion masses, lifetimes (tau0 in mm), and branching ratios are
computed from the solved κ.  The stable dark pions (those involving
dark flavor 4, which has κ = 0) have mayDecay = off.
"""

import argparse
import os
import sys
import numpy as np

from dark_pion_widths import DarkPionTChannelModel, DIAGONAL_PION_INDICES

# ---------------------------------------------------------------------------
# PDG conventions for this model (darkQCD_fv_down: down-type mediator)
# ---------------------------------------------------------------------------

# SM quark model index → PDG code  (d, s, b)
_QUARK_PDG = {0: 1, 1: 3, 2: 5}

# Off-diagonal dark pion (alpha, beta) → Pythia PDG  (decaying species)
_OFFDIAG_PDG = {(0, 1): 4900211, (0, 2): 4900311, (1, 2): 4900321}

# Stable off-diagonal dark pions (involve dark flavor 4, which has κ=0)
_STABLE_OFFDIAG_PDG = [4900411, 4900421, 4900431]

# Diagonal dark pion generator index → Pythia PDG
# New ordering: T3=12, T8=13, T15=14  (see dark_pion_widths/generators.py)
_DIAG_PDG = {12: 4900111, 13: 4900221, 14: 4900331}

# The 4th diagonal pion (eta-like, κ=0 direction) — treated as stable
_ETA_PDG = 4900441

# Off-diagonal dark rho PDGs
_OFFDIAG_RHO_PDG = {(0, 1): 4900213, (0, 2): 4900313, (1, 2): 4900323}
_STABLE_OFFDIAG_RHO_PDG = [4900413, 4900423, 4900433]

# Diagonal dark rho PDGs
_DIAG_RHO_PDG = {2: 4900113, 7: 4900223, 14: 4900333}
_ETA_RHO_PDG = 4900443

# Dark quark PDGs (3 active + 1 singlet)
_DARKQ_PDG = [4900101, 4900102, 4900103, 4900104]

# Mediator PDG
_MEDIATOR_PDG = 4900001


# ---------------------------------------------------------------------------
# Kappa solver
# ---------------------------------------------------------------------------

_SPECIES_GETTERS = {
    "T15":   lambda m: m.ctau_diag_mm(14),
    "T8":    lambda m: m.ctau_diag_mm(13),
    "T3":    lambda m: m.ctau_diag_mm(12),
    "(0,1)": lambda m: m.ctau_off_diag_mm(0, 1),
    "(0,2)": lambda m: m.ctau_off_diag_mm(0, 2),
    "(1,2)": lambda m: m.ctau_off_diag_mm(1, 2),
}


def solve_kappa(fD: float, mPiD: float, mXd: float, ctau_target: float,
                species: str = "T15", kappa_mode: str = "universal") -> float:
    """
    Compute κ such that the specified pion species has c·τ = ctau_target (mm).
    Uses the exact scaling c·τ ∝ κ⁻⁴ evaluated at κ = 1.
    """
    if species not in _SPECIES_GETTERS:
        raise ValueError(f"Unknown species '{species}'. "
                         f"Choose from: {list(_SPECIES_GETTERS)}")
    ref_model = DarkPionTChannelModel(
        fD=fD, m_piD=mPiD, m_X=mXd, kappa=1.0,
        kappa_mode=kappa_mode,
    )
    ctau_ref = _SPECIES_GETTERS[species](ref_model)
    if not np.isfinite(ctau_ref) or ctau_ref <= 0:
        raise ValueError(
            f"Reference c·τ is zero or infinite for species {species} "
            f"at mPiD={mPiD}, mXd={mXd}. "
            "Check that the decay is kinematically open."
        )
    return (ctau_ref / ctau_target) ** 0.25


# ---------------------------------------------------------------------------
# Branching-ratio helpers
# ---------------------------------------------------------------------------

def _br_lines(pdg_id: int, br: dict) -> str:
    """
    Format Pythia 'oneChannel / addChannel' lines for a pion.
    Each key in br is a (quark_idx_i, quark_idx_j) pair meaning π → q_i q̄_j.
    meMode 91 = isotropic phase-space decay.
    """
    lines = []
    first = True
    for (i, j), b in br.items():
        qi = _QUARK_PDG[i]
        qj = _QUARK_PDG[j]
        kw = "oneChannel" if first else "addChannel"
        lines.append(f"{pdg_id}:{kw} on {b:.6f} 91 {qi} -{qj}")
        first = False
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Card writers
# ---------------------------------------------------------------------------

def write_param_card(path: str, mXd: float, mDarkQ: float, kappa: float,
                     kappa_mode: str = "universal") -> None:
    """Write MadGraph param_card.dat.

    The kappa_coupling block is filled according to kappa_mode:
      "universal"  — all 9 entries equal κ
      "diagonal"   — κ_{αα} = κ (entries 1, 5, 9); rest = 0
      "down_only"  — κ_{α,d} = κ (entries 1, 4, 7); rest = 0
    Entry numbering: 3*(α-1)+i for dark quark α=1..3, SM quark i=1..3 (d,s,b).
    """
    k = kappa
    z = 0.0

    if kappa_mode == "universal":
        k11, k12, k13 = k, k, k
        k21, k22, k23 = k, k, k
        k31, k32, k33 = k, k, k
    elif kappa_mode == "diagonal":
        k11, k12, k13 = k, z, z
        k21, k22, k23 = z, k, z
        k31, k32, k33 = z, z, k
    elif kappa_mode == "down_only":
        k11, k12, k13 = k, z, z
        k21, k22, k23 = k, z, z
        k31, k32, k33 = k, z, z
    else:
        raise ValueError(f"Unknown kappa_mode '{kappa_mode}'")

    content = f"""\
######################################################################
## PARAM_CARD GENERATED BY write_cards.py
######################################################################

###################################
## INFORMATION FOR CKMBLOCK
###################################
Block ckmblock
    1 2.277360e-01 # cabi

###################################
## INFORMATION FOR KAPPA_COUPLING
## kappa_mode = {kappa_mode}
###################################
Block kappa_coupling
    1 {k11:.6e} # kappa11
    2 {k12:.6e} # kappa12
    3 {k13:.6e} # kappa13
    4 {k21:.6e} # kappa21
    5 {k22:.6e} # kappa22
    6 {k23:.6e} # kappa23
    7 {k31:.6e} # kappa31
    8 {k32:.6e} # kappa32
    9 {k33:.6e} # kappa33

###################################
## INFORMATION FOR MASS
###################################
Block mass
    1 5.040000e-03 # MD
    2 2.550000e-03 # MU
    3 1.010000e-01 # MS
    4 1.270000e+00 # MC
    5 4.700000e+00 # MB
    6 1.720000e+02 # MT
   11 5.110000e-04 # Me
   13 1.056600e-01 # MMU
   15 1.777000e+00 # MTA
   23 9.118760e+01 # MZ
   24 7.982436e+01 # MW
   25 1.250000e+02 # MH
  4900001 {mXd:.6e} # MassX
  4900101 {mDarkQ:.6e} # MDarkq1
  4900102 {mDarkQ:.6e} # MDarkq2
  4900103 {mDarkQ:.6e} # MDarkq3

###################################
## INFORMATION FOR SMINPUTS
###################################
Block sminputs
    1 1.279000e+02 # aEWM1
    2 1.166370e-05 # Gf
    3 1.184000e-01 # aS

###################################
## INFORMATION FOR YUKAWA
###################################
Block yukawa
    1 5.040000e-03 # ymdo
    2 2.550000e-03 # ymup
    3 1.010000e-01 # yms
    4 1.270000e+00 # ymc
    5 4.700000e+00 # ymb
    6 1.720000e+02 # ymt
   11 5.110000e-04 # yme
   13 1.056600e-01 # ymm
   15 1.777000e+00 # ymtau

###################################
## INFORMATION FOR DECAY
###################################
DECAY   6 1.508336e+00 # WT
DECAY  23 2.495200e+00 # WZ
DECAY  24 2.085000e+00 # WW
DECAY  25 4.070000e-03 # WH
DECAY 4900001 Auto # WidthX
"""
    with open(path, "w") as f:
        f.write(content)


def _build_med_channels(model: "DarkPionTChannelModel") -> list:
    """
    Return [(br, sm_pdg, dark_pdg), ...] for the active X → SM_q + dark_q
    channels, with equal BRs, driven by kappa_mode.
    """
    sm_qs   = [1, 3, 5]          # d, s, b PDG codes
    dark_qs = [4900101, 4900102, 4900103]
    k       = model.kappa.real
    if model.kappa_mode == "universal":
        pairs = [(sq, dq) for sq in sm_qs for dq in dark_qs]
    elif model.kappa_mode == "diagonal":
        pairs = list(zip(sm_qs, dark_qs))
    elif model.kappa_mode == "down_only":
        pairs = [(1, dq) for dq in dark_qs]   # d quark only
    else:
        pairs = [(sq, dq) for sq in sm_qs for dq in dark_qs]
    br = 1.0 / len(pairs) if pairs else 0.0
    return [(br, sq, dq) for sq, dq in pairs]


def write_pythia_card(path: str, model: DarkPionTChannelModel, mXd: float,
                      mDarkQ: float, mRhoD: float, LambdaD: float,
                      lhe_file: str) -> None:
    """Write Pythia 8 input card with computed masses, lifetimes, and BRs.

    Uses the generator basis (SU(4) adjoint pions T3/T8/T15).  The 4th
    diagonal pion (4900441, κ=0 direction) is stable.
    """
    results = model.compute_all()
    mPiD    = model.m_piD

    # Generator-basis diagonal pions: T3=12, T8=13, T15=14
    ctau_diag = {b: model.ctau_diag_mm(b) for b in DIAGONAL_PION_INDICES}
    br_diag   = {b: results["diagonal"][b]["br"]
                 for b in DIAGONAL_PION_INDICES if b in results["diagonal"]}

    # Off-diagonal pions: (0,1), (0,2), (1,2)
    ctau_offdiag = {ab: model.ctau_off_diag_mm(*ab)
                    for ab in [(0, 1), (0, 2), (1, 2)]}
    br_offdiag   = {ab: results["off_diagonal"][ab]["br"]
                    for ab in [(0, 1), (0, 2), (1, 2)]
                    if ab in results["off_diagonal"]}

    # Mediator decay channels (kappa_mode-aware, computed from model)
    med_channels = _build_med_channels(model)

    # pTminFSR must be strictly greater than Lambda
    ptmin_fsr = LambdaD * 1.1

    _diag_names = {12: "T3", 13: "T8", 14: "T15"}

    def _diag_lines(b):
        if b not in br_diag or not br_diag[b]:
            return f"! {_DIAG_PDG[b]}: no open decay channels"
        return _br_lines(_DIAG_PDG[b], br_diag[b])

    def _diag_block(b):
        """Return the full decay block for a diagonal generator-basis pion."""
        pdg   = _DIAG_PDG[b]
        name  = _diag_names.get(b, f"T{b}")
        c     = ctau_diag[b]
        if not np.isfinite(c) or b not in br_diag or not br_diag[b]:
            return f"! --- Diagonal dark pion {name} ({pdg}) — stable ---\n{pdg}:mayDecay = off"
        return (f"! --- Diagonal dark pion {name} ({pdg}), ctau = {c:.4f} mm ---\n"
                f"{pdg}:tau0 = {c:.6f}\n"
                f"{_diag_lines(b)}")

    def _offdiag_lines(ab):
        if ab not in br_offdiag or not br_offdiag[ab]:
            return f"! {_OFFDIAG_PDG[ab]}: no open decay channels"
        return _br_lines(_OFFDIAG_PDG[ab], br_offdiag[ab])

    # Build mediator channel lines
    med_ch_lines = []
    for idx, (br, sq, dq) in enumerate(med_channels):
        kw = "oneChannel" if idx == 0 else "addChannel"
        med_ch_lines.append(
            f"{_MEDIATOR_PDG}:{kw} on {br:.6f} 103 {sq}  {dq}"
        )
    med_ch_block = "\n".join(med_ch_lines)

    content = f"""\
! Pythia 8 input card generated by write_cards.py
! Model parameters:
!   mXd        = {mXd} GeV
!   mPiD       = {mPiD} GeV
!   fD         = {model.fD} GeV
!   kappa      = {model.kappa.real:.6f}
!   kappa_mode = {model.kappa_mode}
!   mDarkQ     = {mDarkQ} GeV
!   LambdaD    = {LambdaD} GeV
!   pion_basis = generator  (SU(4) adjoint)

! ================================================================
! 1) Event generation
! ================================================================

Beams:frameType = 4
Beams:LHEF = {lhe_file}
SLHA:allowUserOverride = on

PartonLevel:MPI = off
PartonLevel:ISR = off

HiddenValley:alphaOrder = 1
HiddenValley:FSR = on
HiddenValley:alphaFSR = 0.7
HiddenValley:fragment = on

SpaceShower:QCDshower = on
TimeShower:QCDshower = on
ParticleDecays:FSRinDecays = on
HadronLevel:Hadronize = on

! ================================================================
! 2) Dark sector gauge group
! ================================================================

HiddenValley:Ngauge = 3
HiddenValley:nFlav = {model.Nf}
HiddenValley:probKeepEta1 = 0
HiddenValley:separateFlav = on
HiddenValley:Lambda = {LambdaD:.4f}
HiddenValley:pTminFSR = {ptmin_fsr:.4f}    ! must be > Lambda

! ================================================================
! 3) Mediator ({_MEDIATOR_PDG})
! ================================================================

{_MEDIATOR_PDG}:m0 = {mXd:.4f}
{_MEDIATOR_PDG}:colType = 1
{_MEDIATOR_PDG}:spinType = 2
{_MEDIATOR_PDG}:isResonance = on
{_MEDIATOR_PDG}:mayDecay = on

HiddenValley:spinFv = 0

! Mediator decays: X → SM_q + dark_q  (kappa_mode = {model.kappa_mode})
{med_ch_block}
{_MEDIATOR_PDG}:0:meMode = 103

! ================================================================
! 4) Dark quark masses
! ================================================================

4900101:m0 = {mDarkQ:.4f}
4900102:m0 = {mDarkQ:.4f}
4900103:m0 = {mDarkQ:.4f}
4900104:m0 = {mDarkQ:.4f}

4900101:spinType = 1
4900102:spinType = 1
4900103:spinType = 1
4900104:spinType = 1

4900101:colType = 0
4900102:colType = 0
4900103:colType = 0
4900104:colType = 0

! ================================================================
! 5) Dark pion masses  (mPiD = {mPiD} GeV, mRhoD = {mRhoD:.4f} GeV)
! ================================================================

! Diagonal dark pions (generator basis: T3, T8, T15)
4900111:m0 = {mPiD:.4f}    ! T3
4900221:m0 = {mPiD:.4f}    ! T8
4900331:m0 = {mPiD:.4f}    ! T15
4900441:m0 = {mPiD:.4f}    ! eta-like (stable, kappa=0 direction)

! Diagonal dark rhos
4900113:m0 = {mRhoD:.4f}
4900223:m0 = {mRhoD:.4f}
4900333:m0 = {mRhoD:.4f}
4900443:m0 = {mRhoD:.4f}

! Off-diagonal dark pions (decaying)
4900211:m0 = {mPiD:.4f}    ! pi(0,1)
4900311:m0 = {mPiD:.4f}    ! pi(0,2)
4900321:m0 = {mPiD:.4f}    ! pi(1,2)

! Off-diagonal dark pions (stable, involve dark flavor 4)
4900411:m0 = {mPiD:.4f}
4900421:m0 = {mPiD:.4f}
4900431:m0 = {mPiD:.4f}

! Off-diagonal dark rhos
4900213:m0 = {mRhoD:.4f}
4900313:m0 = {mRhoD:.4f}
4900323:m0 = {mRhoD:.4f}
4900413:m0 = {mRhoD:.4f}
4900423:m0 = {mRhoD:.4f}
4900433:m0 = {mRhoD:.4f}

! ================================================================
! 6) Visibility flags (all dark sector particles are invisible)
! ================================================================

{_MEDIATOR_PDG}:isVisible = off

4900101:isVisible = off
4900102:isVisible = off
4900103:isVisible = off
4900104:isVisible = off

4900111:isVisible = off
4900221:isVisible = off
4900331:isVisible = off
4900441:isVisible = off

4900113:isVisible = off
4900223:isVisible = off
4900333:isVisible = off
4900443:isVisible = off

4900211:isVisible = off
4900213:isVisible = off
4900311:isVisible = off
4900313:isVisible = off
4900321:isVisible = off
4900323:isVisible = off
4900411:isVisible = off
4900413:isVisible = off
4900421:isVisible = off
4900423:isVisible = off
4900431:isVisible = off
4900433:isVisible = off

! ================================================================
! 7) Dark rho decays (into dark pion pairs)
! ================================================================

4900113:mayDecay = on
4900223:mayDecay = on
4900333:mayDecay = on
4900443:mayDecay = on
4900213:mayDecay = on
4900313:mayDecay = on
4900323:mayDecay = on
4900413:mayDecay = on
4900423:mayDecay = on
4900433:mayDecay = on

4900113:oneChannel on 0.001 91 1 -1
4900113:addChannel on 0.333 102 4900211 -4900211
4900113:addChannel on 0.333 102 4900311 -4900311
4900113:addChannel on 0.333 102 4900411 -4900411

4900223:oneChannel on 0.001 91 1 -1
4900223:addChannel on 0.333 102 4900211 -4900211
4900223:addChannel on 0.333 102 4900321 -4900321
4900223:addChannel on 0.333 102 4900421 -4900421

4900333:oneChannel on 0.001 91 1 -1
4900333:addChannel on 0.333 102 4900311 -4900311
4900333:addChannel on 0.333 102 4900321 -4900321
4900333:addChannel on 0.333 102 4900431 -4900431

4900443:oneChannel on 0.001 91 1 -1
4900443:addChannel on 0.333 102 4900411 -4900411
4900443:addChannel on 0.333 102 4900421 -4900421
4900443:addChannel on 0.333 102 4900431 -4900431

4900213:oneChannel on 0.001 91 1 -1
4900213:addChannel on 0.4995 102 4900211 4900111
4900213:addChannel on 0.4995 102 4900211 4900221

4900313:oneChannel on 0.001 91 1 -1
4900313:addChannel on 0.4995 102 4900311 4900111
4900313:addChannel on 0.4995 102 4900311 4900331

4900323:oneChannel on 0.001 91 1 -1
4900323:addChannel on 0.4995 102 4900321 4900221
4900323:addChannel on 0.4995 102 4900321 4900331

4900413:oneChannel on 0.001 91 1 -1
4900413:addChannel on 0.4995 102 4900411 4900111
4900413:addChannel on 0.4995 102 4900411 4900441

4900423:oneChannel on 0.001 91 1 -1
4900423:addChannel on 0.4995 102 4900421 4900221
4900423:addChannel on 0.4995 102 4900421 4900441

4900433:oneChannel on 0.001 91 1 -1
4900433:addChannel on 0.4995 102 4900431 4900331
4900433:addChannel on 0.4995 102 4900431 4900441

! ================================================================
! 8) Dark pion decays — lifetimes and branching ratios
! ================================================================

! --- Stable dark pions (involve dark flavor 4 or kappa=0 direction) ---
4900441:mayDecay = off
4900411:mayDecay = off
4900421:mayDecay = off
4900431:mayDecay = off

{_diag_block(12)}

{_diag_block(13)}

{_diag_block(14)}

! --- Off-diagonal dark pion pi(0,1) (4900211), ctau = {ctau_offdiag[(0,1)]:.4f} mm ---
4900211:tau0 = {ctau_offdiag[(0,1)]:.6f}
{_offdiag_lines((0,1))}

! --- Off-diagonal dark pion pi(0,2) (4900311), ctau = {ctau_offdiag[(0,2)]:.4f} mm ---
4900311:tau0 = {ctau_offdiag[(0,2)]:.6f}
{_offdiag_lines((0,2))}

! --- Off-diagonal dark pion pi(1,2) (4900321), ctau = {ctau_offdiag[(1,2)]:.4f} mm ---
4900321:tau0 = {ctau_offdiag[(1,2)]:.6f}
{_offdiag_lines((1,2))}
"""
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Run card and launch script writers
# ---------------------------------------------------------------------------

def write_run_card(path: str, nevents: int = 10000, sqrts: float = 13600.0) -> None:
    """Write a MadGraph run_card.dat."""
    ebeam = sqrts / 2.0
    content = f"""\
#*********************************************************************
#                       MadGraph5_aMC@NLO                            *
#                     run_card.dat — generated by write_cards.py     *
#*********************************************************************

#*********************************************************************
# Tag name for the run
#*********************************************************************
  tag_1 = run_tag

#*********************************************************************
# Number of events and random seed
#*********************************************************************
  {nevents} = nevents
  0         = iseed

#*********************************************************************
# Collider: LHC proton-proton
#*********************************************************************
  1       = lpp1
  1       = lpp2
  {ebeam:.1f} = ebeam1
  {ebeam:.1f} = ebeam2

#*********************************************************************
# PDF
#*********************************************************************
  nn23lo1 = pdlabel
  230000  = lhaid

#*********************************************************************
# Scale choices
#*********************************************************************
  False = fixed_ren_scale
  False = fixed_fac_scale
  -1    = dynamical_scale_choice
  1.0   = scalefact

#*********************************************************************
# Phase-space optimisation
#*********************************************************************
  0 = nhel
  2 = sde_strategy

#*********************************************************************
# Output format
#*********************************************************************
  False   = gridpack
  -1.0    = time_of_flight
  average = event_norm

#*********************************************************************
# MLM jet matching
#*********************************************************************
  2.0  = lhe_version
  1    = ickkw
  T    = cut_decays
  0.0  = drjj
  20   = xqcut

#*********************************************************************
# Cuts on dark quarks and jets (pT > 20 GeV to regulate t-channel divergence)
#*********************************************************************
  {{4900101: 20, 4900102: 20, 4900103: 20}} = pt_min_pdg

#*********************************************************************
# Misc
#*********************************************************************
  15.0 = bwcutoff
  5    = maxjetflavor
  True = use_syst
  systematics = systematics_program
  ['--mur=0.5,1,2', '--muf=0.5,1,2', '--pdf=errorset'] = systematics_arguments
"""
    with open(path, "w") as f:
        f.write(content)


def write_launch_script(path: str, outdir: str, process_dir: str,
                        mg5_exe: str = "mg5_aMC") -> None:
    """
    Write a shell script that copies the generated cards into the MadGraph
    process directory and launches the run in batch mode (no interactive prompts).

    The process directory must already exist (generated once with
    'import model ... / generate ... / output <process_dir>').
    """
    # Use absolute path for outdir in the script so it works from any cwd
    abs_outdir = os.path.abspath(outdir)
    abs_process_dir = os.path.abspath(process_dir) if process_dir else "<process_dir>"

    content = f"""\
#!/usr/bin/env bash
# launch.sh — generated by write_cards.py
# Copies cards for this parameter point into the MadGraph process directory
# and launches the run in batch mode (no interactive prompts).
#
# Usage:
#   bash launch.sh [process_dir] [mg5_exe]
#
# Arguments override the defaults baked in at generation time.

set -euo pipefail

CARDS_DIR="{abs_outdir}"
PROCESS_DIR="${{1:-{abs_process_dir}}}"
MG5="${{2:-{mg5_exe}}}"

if [[ ! -d "$PROCESS_DIR/Cards" ]]; then
    echo "ERROR: $PROCESS_DIR/Cards not found."
    echo "Generate the process first:"
    echo "  $MG5 proc_card.txt"
    exit 1
fi

echo "Copying cards from $CARDS_DIR → $PROCESS_DIR/Cards/"
cp "$CARDS_DIR/param_card.dat" "$PROCESS_DIR/Cards/param_card.dat"
cp "$CARDS_DIR/run_card.dat"   "$PROCESS_DIR/Cards/run_card.dat"

echo "Launching MadGraph in batch mode..."
echo "launch $PROCESS_DIR -f" | "$MG5"

echo "Done.  Events are in $PROCESS_DIR/Events/"
"""
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(model: DarkPionTChannelModel, mXd: float, species: str,
                  kappa: float, kappa_source: str, ctau_source: str) -> None:
    results      = model.compute_all()
    labels       = results["quark_labels"]
    ctau_species = _SPECIES_GETTERS[species](model)
    _DIAG_NAMES  = {12: "T3", 13: "T8", 14: "T15"}

    print(f"\n{'='*60}")
    print(f"  Parameter point summary  (generator basis)")
    print(f"{'='*60}")
    print(f"  mXd    = {mXd} GeV")
    print(f"  mPiD   = {model.m_piD} GeV")
    print(f"  fD     = {model.fD} GeV")
    print(f"  κ = {kappa:.6f}  ({kappa_source})")
    print(f"  c·τ ({species}) = {ctau_species:.4f} mm  ({ctau_source})")
    print()

    print("  Diagonal dark pions (T3/T8/T15):")
    for b in DIAGONAL_PION_INDICES:
        if b in results["diagonal"]:
            r    = results["diagonal"][b]
            name = _DIAG_NAMES.get(b, f"T{b}")
            print(f"    {name}  ctau = {r['ctau_mm']:.4f} mm  "
                  f"total Γ = {r['total']:.3e} GeV")
            for (i, j), br in r["br"].items():
                if br > 1e-4:
                    ql = labels[i] if i < len(labels) else str(i)
                    qr = labels[j] if j < len(labels) else str(j)
                    print(f"           BR({ql}{qr}) = {br:.4f}")
    print(f"    4900441 (eta-like)  stable (κ=0 direction)")

    print()
    print("  Off-diagonal dark pions:")
    for ab in [(0, 1), (0, 2), (1, 2)]:
        if ab in results["off_diagonal"]:
            r    = results["off_diagonal"][ab]
            a, b = ab
            print(f"    π({a},{b})  ctau = {r['ctau_mm']:.4f} mm  "
                  f"total Γ = {r['total']:.3e} GeV")
            for (i, j), br in r["br"].items():
                if br > 1e-4:
                    ql = labels[i] if i < len(labels) else str(i)
                    qr = labels[j] if j < len(labels) else str(j)
                    print(f"           BR({ql}{qr}) = {br:.4f}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MadGraph and Pythia cards for a dark QCD parameter point."
    )
    p.add_argument("--mXd", type=float, required=True,
                   help="Mediator mass [GeV]")
    p.add_argument("--kappa", type=float, default=None,
                   help="Coupling κ (if given, used directly; ctau is computed from it)")
    p.add_argument("--ctau", type=float, default=None,
                   help="Target dark pion c·τ [mm] for the reference species "
                        "(used to solve for κ if --kappa is not given)")
    p.add_argument("--mPiD", type=float, required=True,
                   help="Dark pion mass [GeV]")
    p.add_argument("--fD", type=float, default=None,
                   help="Dark decay constant [GeV] (default: mPiD)")
    p.add_argument("--LambdaD", type=float, default=None,
                   help="Dark confinement scale [GeV] (default: mPiD)")
    p.add_argument("--mDarkQ", type=float, default=None,
                   help="Dark quark mass [GeV] (default: 2 * mPiD)")
    p.add_argument("--mRhoD", type=float, default=None,
                   help="Dark rho meson mass [GeV] (default: 4 * mPiD)")
    p.add_argument("--species", default="T15",
                   choices=list(_SPECIES_GETTERS),
                   help="Reference pion species used to set the ctau target "
                        "(default: T15 = longest-lived diagonal pion, bb̄-coupled)")
    p.add_argument("--kappa-mode", default="universal",
                   choices=["universal", "diagonal", "down_only"],
                   dest="kappa_mode",
                   help="κ matrix structure (default: universal). "
                        "Affects both the MadGraph param_card kappa_coupling block "
                        "and the Pythia card lifetimes/BRs.")
    p.add_argument("--lhe", default="events.lhe",
                   help="LHE input file name for Pythia card (default: events.lhe)")
    p.add_argument("--nevents", type=int, default=10000,
                   help="Number of events for run_card (default: 10000)")
    p.add_argument("--sqrts", type=float, default=13600.0,
                   help="Centre-of-mass energy [GeV] for run_card (default: 13600)")
    p.add_argument("--process-dir", default=None,
                   help="MadGraph process directory for launch.sh "
                        "(default: placeholder, edit launch.sh before running)")
    p.add_argument("--mg5-exe", default="mg5_aMC",
                   help="MadGraph executable name or path (default: mg5_aMC)")
    p.add_argument("--output", default=None,
                   help="Output directory (default: cards/mXd{mXd}_mPiD{mPiD}_ctau{ctau})")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate: need at least one of --kappa / --ctau
    if args.kappa is None and args.ctau is None:
        print("ERROR: provide at least one of --kappa or --ctau.", file=sys.stderr)
        sys.exit(1)

    # Fill defaults
    fD      = args.fD      if args.fD      is not None else args.mPiD
    LambdaD = args.LambdaD if args.LambdaD is not None else args.mPiD
    mDarkQ  = args.mDarkQ  if args.mDarkQ  is not None else 2.0 * args.mPiD
    mRhoD   = args.mRhoD   if args.mRhoD   is not None else 4.0 * args.mPiD

    # Determine kappa and record how each quantity was obtained
    if args.kappa is not None and args.ctau is not None:
        # Both given: use kappa directly, compute ctau from it, note the target
        kappa = args.kappa
        kappa_source = "provided directly"
        ctau_source  = (f"computed from κ  "
                        f"(target was {args.ctau} mm for {args.species})")
        print(f"\nUsing provided κ = {kappa:.6f}; "
              f"ignoring --ctau={args.ctau} mm for card generation "
              f"(actual c·τ will be shown below).")
    elif args.kappa is not None:
        # kappa only: compute ctau
        kappa = args.kappa
        kappa_source = "provided directly"
        ctau_source  = f"computed from κ"
        print(f"\nUsing provided κ = {kappa:.6f}; computing c·τ.")
    else:
        # ctau only: solve for kappa
        try:
            kappa = solve_kappa(fD, args.mPiD, args.mXd, args.ctau,
                                args.species, args.kappa_mode)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        kappa_source = f"solved from c·τ = {args.ctau} mm ({args.species})"
        ctau_source  = f"target for {args.species}"
        print(f"\nSolved κ = {kappa:.6f} for c·τ = {args.ctau} mm ({args.species}).")

    # Build the model with the final kappa — generator basis (SU(4) adjoint)
    model = DarkPionTChannelModel(fD=fD, m_piD=args.mPiD, m_X=args.mXd,
                                  kappa=kappa, kappa_mode=args.kappa_mode)

    # Default output dir uses whichever of ctau/kappa was given
    if args.output is not None:
        outdir = args.output
    elif args.ctau is not None:
        outdir = f"cards/mXd{args.mXd:.0f}_mPiD{args.mPiD:.0f}_ctau{args.ctau:.0f}"
    else:
        outdir = f"cards/mXd{args.mXd:.0f}_mPiD{args.mPiD:.0f}_kappa{args.kappa:.4f}"

    # Print summary
    print_summary(model, args.mXd, args.species, kappa, kappa_source, ctau_source)

    # Write cards
    os.makedirs(outdir, exist_ok=True)
    param_path   = os.path.join(outdir, "param_card.dat")
    run_path     = os.path.join(outdir, "run_card.dat")
    pythia_path  = os.path.join(outdir, "pythia_card.dat")
    launch_path  = os.path.join(outdir, "launch.sh")

    write_param_card(param_path, args.mXd, mDarkQ, kappa, args.kappa_mode)
    write_run_card(run_path, nevents=args.nevents, sqrts=args.sqrts)
    write_pythia_card(pythia_path, model, args.mXd, mDarkQ, mRhoD, LambdaD,
                      lhe_file=args.lhe)
    write_launch_script(launch_path, outdir,
                        process_dir=args.process_dir or "",
                        mg5_exe=args.mg5_exe)

    print(f"  Wrote {param_path}")
    print(f"  Wrote {run_path}")
    print(f"  Wrote {pythia_path}")
    print(f"  Wrote {launch_path}")
    print()


if __name__ == "__main__":
    main()
