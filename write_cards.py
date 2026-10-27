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

from dark_pion_widths import (DarkPionTChannelModel, DarkPionTChannelFlavourModel,
                              DIAGONAL_PION_INDICES, HBAR_C_GEV_MM)
from dark_pion_widths.generators import get_diagonal_indices, get_diagonal_names

# ---------------------------------------------------------------------------
# PDG conventions for this model (darkQCD_fv_down: down-type mediator)
# ---------------------------------------------------------------------------

# SM quark model index → PDG code  (d, s, b)
_QUARK_PDG = {0: 1, 1: 3, 2: 5}

# Pythia's own quark masses (ParticleData m0), used when it decays a pion with
# meMode 91.  A channel below m_i + m_j here fails in Pythia, which then
# retries and silently reweights the BRs toward the remaining channels.
_PYTHIA_QUARK_M0 = {1: 0.33, 3: 0.50, 5: 4.80}

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

def _diag_index(m, name: str) -> int:
    """Generator index of a named diagonal pion (T3, T8, T15) for model.Nf."""
    for b, n in get_diagonal_names(m.Nf).items():
        if n == name:
            return b
    raise ValueError(f"No diagonal pion {name} for Nf = {m.Nf}")


def _decaying_ctaus(model) -> list:
    res = model.compute_all()
    return [r["ctau_mm"] for r in list(res["diagonal"].values()) + list(res["off_diagonal"].values())
            if np.isfinite(r["ctau_mm"])]


def _eig_ctaus(model) -> list:
    return [e["ctau_mm"] for e in model.decay_eigenstates() if np.isfinite(e["ctau_mm"])]


_SPECIES_GETTERS = {
    "T15":   lambda m: m.ctau_diag_mm(_diag_index(m, "T15")),
    "T8":    lambda m: m.ctau_diag_mm(_diag_index(m, "T8")),
    "T3":    lambda m: m.ctau_diag_mm(_diag_index(m, "T3")),
    "(0,1)": lambda m: m.ctau_off_diag_mm(0, 1),
    "(0,2)": lambda m: m.ctau_off_diag_mm(0, 2),
    "(1,2)": lambda m: m.ctau_off_diag_mm(1, 2),
    # Longest-lived decaying flavour state (stable ones, cτ = ∞, are skipped).
    # All species scale as κ⁻⁴, so the same species is longest at every κ.
    "longest": lambda m: max(_decaying_ctaus(m), default=np.inf),
    # Diagonal pion lifetime in either basis (with mixing: the common value)
    "diag": lambda m: min((r["ctau_mm"] for r in m.compute_all()["diagonal"].values()),
                          default=np.inf),
    # Decay eigenstates (use with --decay-basis eigen)
    "eig_longest":  lambda m: max(_eig_ctaus(m), default=np.inf),
    "eig_shortest": lambda m: min(_eig_ctaus(m), default=np.inf),
}


def _model_class(pion_basis: str):
    return DarkPionTChannelFlavourModel if pion_basis == "flavour" else DarkPionTChannelModel


def solve_kappa(fD: float, mPiD: float, mXd: float, ctau_target: float,
                species: str = "T15", kappa_mode: str = "universal",
                Nf: int = 4, mix_diagonal: bool = True,
                pion_basis: str = "generator") -> float:
    """
    Compute κ such that the specified pion species has c·τ = ctau_target (mm).
    Uses the exact scaling c·τ ∝ κ⁻⁴ evaluated at κ = 1.
    """
    if species not in _SPECIES_GETTERS:
        raise ValueError(f"Unknown species '{species}'. "
                         f"Choose from: {list(_SPECIES_GETTERS)}")
    ref_model = _model_class(pion_basis)(
        fD=fD, m_piD=mPiD, m_X=mXd, kappa=1.0, Nf=Nf,
        kappa_mode=kappa_mode, mix_diagonal=mix_diagonal,
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

    # Mediator width written explicitly rather than "Auto": MG5's auto-width
    # zeroes every partial width below 0.1 GeV for a coloured particle, which
    # at small kappa (e.g. m_X = 1 TeV, long ctau) sets the whole width to 0.
    # Same two-body formula as the UFO's decays.py, with the masses below.
    sm_masses = [5.04e-3, 1.01e-1, 4.7]          # MD, MS, MB in this card
    kmat = [[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]]
    width_x = 0.0
    for row in kmat:
        for kai, mq in zip(row, sm_masses):
            lam = (mXd**4 + mq**4 + mDarkQ**4 - 2 * mXd**2 * mq**2
                   - 2 * mXd**2 * mDarkQ**2 - 2 * mq**2 * mDarkQ**2)
            if lam > 0:
                width_x += (kai**2 * (mXd**2 - mq**2 - mDarkQ**2) * np.sqrt(lam)
                            / (16 * np.pi * mXd**3))

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
DECAY 4900001 {width_x:.6e} # WidthX (analytic; MG5 Auto zeroes partial widths < 0.1 GeV)
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
                      lhe_file: str, merged: bool = True,
                      decay_basis: str = "flavour", xqcut: float = 120.0) -> None:
    """Write Pythia 8 input card with computed masses, lifetimes, and BRs.

    Supports Nf = 3 and 4 (model.Nf).  decay_basis selects how the
    mass-degenerate dark pions decay, see _dark_hadron_section.

    merged=True adds the MLM veto matching the run card (ickkw = 1,
    xqcut, maxjetflavor = 4).  The on-shell sample is unmerged and relies
    on the shower alone for extra radiation.  ISR and MPI are on for both.

    Uses the generator basis (SU(4) adjoint pions T3/T8/T15).  The 4th
    diagonal pion (4900441, κ=0 direction) is stable.
    """
    results = model.compute_all()
    mPiD    = model.m_piD

    # Pythia's HV string fragmentation scales its parameters (bmqv2, sigmamqv)
    # with the dark-quark mass and expects a constituent mass of order Lambda,
    # m_i + m_j = (3 m_rho + m_pi) / 4.  MG5 keeps the Lagrangian mass mDarkQ;
    # Pythia keeps the LHE masses for the hard-process dark quarks.
    mDarkQ_const = (3.0 * mRhoD + mPiD) / 8.0

    if merged:
        matching_block = f"""
! MLM matching for p p > qd qd + 0,1,2 j  (run card: ickkw = 1, xqcut = {xqcut:g})
JetMatching:merge = on
JetMatching:scheme = 1               ! MadGraph-style MLM
JetMatching:setMad = off
JetMatching:jetAlgorithm = 2         ! SlowJet kT clustering (kT-MLM, as in MG5's Pythia8 cards)
JetMatching:slowJetPower = 1
JetMatching:qCut = {1.5 * xqcut:g}              ! 1.5 x xqcut
JetMatching:nQmatch = 4              ! = maxjetflavor (4-flavour scheme)
JetMatching:nJetMax = 2
JetMatching:coneRadius = 1.0
JetMatching:etaJetMax = 10.0"""
    else:
        matching_block = "! Unmerged on-shell sample: no MLM matching (run card: ickkw = 0)"

    # Every channel that can be written below must also be open for Pythia
    open_channels = {(i, j) for r in list(results["diagonal"].values())
                     + list(results["off_diagonal"].values()) for (i, j) in r["br"]}
    open_channels |= {(i, j) for i in range(model._nq) for j in range(model._nq)
                      if model.channel_open(i, j)}
    for (i, j) in sorted(open_channels):
        qi, qj = _QUARK_PDG[i], _QUARK_PDG[j]
        if mPiD <= _PYTHIA_QUARK_M0[qi] + _PYTHIA_QUARK_M0[qj]:
            print(f"WARNING: pi_D -> {qi} -{qj} is open in dark_pion_widths but closed "
                  f"for Pythia at mPiD = {mPiD} GeV (needs > "
                  f"{_PYTHIA_QUARK_M0[qi] + _PYTHIA_QUARK_M0[qj]:.2f} GeV); Pythia "
                  f"will fail and retry these decays.", file=sys.stderr)

    # Mediator decay channels (kappa_mode-aware, computed from model)
    med_channels = _build_med_channels(model)

    # pTminFSR must be strictly greater than Lambda
    ptmin_fsr = np.ceil(LambdaD * 1.1e4) / 1e4   # round up so the written value stays >= 1.1 Lambda

    # Build mediator channel lines
    med_ch_lines = []
    for idx, (br, sq, dq) in enumerate(med_channels):
        kw = "oneChannel" if idx == 0 else "addChannel"
        med_ch_lines.append(
            f"{_MEDIATOR_PDG}:{kw} on {br:.6f} 103 {sq}  {dq}"
        )
    med_ch_block = "\n".join(med_ch_lines)

    hadron_section = _dark_hadron_section(model, mPiD, mRhoD, mDarkQ_const, decay_basis)

    content = f"""\
! Pythia 8 input card generated by write_cards.py
! Model parameters:
!   mXd           = {mXd} GeV
!   mPiD          = {mPiD} GeV
!   fD            = {model.fD} GeV
!   kappa         = {model.kappa.real:.6f}
!   kappa_mode    = {model.kappa_mode}
!   mPiD/LambdaD  = {mPiD/LambdaD:.4f}
!   LambdaD       = {LambdaD:.4f} GeV
!   mDarkQ        = {mDarkQ:.4f} GeV  (Lagrangian, MG5 / LHE)
!   mDarkQ_const  = {mDarkQ_const:.4f} GeV  (constituent, Pythia HV fragmentation)
!   mRhoD         = {mRhoD:.4f} GeV
!   Nf            = {model.Nf}
!   decay_basis   = {decay_basis}

! ================================================================
! 1) Event generation
! ================================================================

Beams:frameType = 4
Beams:LHEF = {lhe_file}
SLHA:allowUserOverride = on

PartonLevel:MPI = on
PartonLevel:ISR = on
{matching_block}

HiddenValley:alphaOrder = 1
HiddenValley:FSR = on
HiddenValley:alphaFSR = 0.7          ! unused when setLambda = on
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
HiddenValley:setLambda = on          ! otherwise Lambda is ignored and derived from alphaFSR
HiddenValley:Lambda = {LambdaD:.4f}
HiddenValley:pTminFSR = {ptmin_fsr:.4f}    ! must be > Lambda

! ================================================================
! 3) Mediator ({_MEDIATOR_PDG})
! ================================================================

{_MEDIATOR_PDG}:m0 = {mXd:.4f}
{_MEDIATOR_PDG}:colType = 1
{_MEDIATOR_PDG}:spinType = 1          ! scalar (2s+1)
{_MEDIATOR_PDG}:isResonance = on
{_MEDIATOR_PDG}:mayDecay = on

HiddenValley:spinFv = 0

! Mediator decays: X → SM_q + dark_q  (kappa_mode = {model.kappa_mode})
{med_ch_block}
{_MEDIATOR_PDG}:0:meMode = 103

{hadron_section}
"""
    with open(path, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Dark hadron section (any Nf, flavour or eigenstate decay basis)
# ---------------------------------------------------------------------------

def _offdiag_pdg(a: int, b: int, spin: int = 1) -> int:
    """Pythia HV meson for dark flavours a < b (0-based): 4900ij{1,3}, i > j."""
    return 4900000 + 100 * (b + 1) + 10 * (a + 1) + spin


def _diag_pdg(a: int, spin: int = 1) -> int:
    return 4900000 + 110 * (a + 1) + spin


def _eig_pdg(k: int) -> int:
    """New Pythia particle for decay eigenstate k (0-based)."""
    return 4910000 + 10 * (k + 1) + 1


def _decay_lines(pdg: int, ctau: float, br: dict, label: str) -> list:
    if not np.isfinite(ctau) or not br:
        return [f"! --- {label} ({pdg}) — stable ---", f"{pdg}:mayDecay = off"]
    out = [f"! --- {label} ({pdg}), ctau = {ctau:.4g} mm ---", f"{pdg}:tau0 = {ctau:.6e}"]
    first = True
    for (i, j), b in sorted(br.items(), key=lambda kv: -kv[1]):
        if b < 1e-7:
            continue
        kw = "oneChannel" if first else "addChannel"
        out.append(f"{pdg}:{kw} on {b:.6f} 91 {_QUARK_PDG[i]} -{_QUARK_PDG[j]}")
        first = False
    return out


def _routing_lines(pdg: int, probs: list, label: str) -> list:
    """Prompt one-body decay of a flavour meson into decay eigenstates."""
    tot = sum(probs)
    out = [f"! --- {label} ({pdg}) -> decay eigenstates (prompt, one-body) ---",
           f"{pdg}:tau0 = 0.0", f"{pdg}:mayDecay = on"]
    first = True
    for k, pk in enumerate(probs):
        if pk / tot < 1e-6:
            continue
        kw = "oneChannel" if first else "addChannel"
        out.append(f"{pdg}:{kw} on {pk / tot:.6f} 0 {_eig_pdg(k)}")
        first = False
    return out + [""]


def _dark_hadron_section(model, mPiD: float, mRhoD: float, mDarkQ_const: float,
                         decay_basis: str) -> str:
    """
    Dark quark masses, dark meson masses/visibility, dark rho decays and dark
    pion decays for any Nf, as Pythia commands.

    Pythia's HV hadronisation (separateFlav = on) produces flavour mesons:
    off-diagonal 4900ij1 (i > j) and diagonal 4900ii1, the last diagonal one
    being the flavour-singlet-like eta_1 (suppressed by probKeepEta1 = 0 and
    kept stable).  The first Nf-1 diagonal mesons are identified with the
    diagonal generators (T3, T8, ...).

    decay_basis = "flavour":  each flavour meson decays with the width of the
        corresponding flavour state (model.compute_all(); diagonal pions follow
        model.mix_diagonal).
    decay_basis = "eigen":  the pions are mass degenerate, so the physical
        decay eigenstates diagonalise the width matrix Γ_ab
        (model.width_matrix()).  Each eigenstate k becomes a new Pythia
        particle 4910{k+1}1 with its own lifetime and BRs; each flavour meson
        decays promptly (one-body, like K0 -> K_S / K_L) into eigenstate k with
        probability |<k|meson>|^2.  Pythia produces the adjoint states with equal
        rates, so every eigenstate ends up equally populated.
    """
    Nf    = model.Nf
    pairs = [(a, b) for a in range(Nf) for b in range(a + 1, Nf)]
    diag_idx = get_diagonal_indices(Nf)
    names = get_diagonal_names(Nf)
    L = []

    L += ["! ================================================================",
          "! 4) Dark quark masses (constituent; the LHE keeps the MG5 mass)",
          "! ================================================================", ""]
    for a in range(Nf):
        q = 4900101 + a
        L += [f"{q}:m0 = {mDarkQ_const:.4f}", f"{q}:spinType = 2", f"{q}:colType = 0",
              f"{q}:isVisible = off"]
    L.append("")

    L += ["! ================================================================",
          f"! 5) Dark mesons (Nf = {Nf}: mPiD = {mPiD} GeV, mRhoD = {mRhoD:.4f} GeV)",
          "! ================================================================", ""]
    mesons = []
    for a in range(Nf):
        mesons += [(_diag_pdg(a, 1), mPiD), (_diag_pdg(a, 3), mRhoD)]
    for a, b in pairs:
        mesons += [(_offdiag_pdg(a, b, 1), mPiD), (_offdiag_pdg(a, b, 3), mRhoD)]
    for pdg, m in mesons:
        L += [f"{pdg}:m0 = {m:.4f}", f"{pdg}:isVisible = off"]
    L.append("")

    # Dark rho decays into dark pion pairs (flavour-conserving).
    L += ["! Dark rho decays (prompt, into dark pions)"]
    for a in range(Nf):
        rho = _diag_pdg(a, 3)
        prods = [_offdiag_pdg(min(a, c), max(a, c), 1) for c in range(Nf) if c != a]
        br = 0.999 / len(prods)
        L += [f"{rho}:mayDecay = on", f"{rho}:oneChannel on 0.001 91 1 -1"]
        L += [f"{rho}:addChannel on {br:.4f} 102 {p} -{p}" for p in prods]
    for a, b in pairs:
        rho = _offdiag_pdg(a, b, 3)
        pi  = _offdiag_pdg(a, b, 1)
        L += [f"{rho}:mayDecay = on", f"{rho}:oneChannel on 0.001 91 1 -1",
              f"{rho}:addChannel on 0.4995 102 {pi} {_diag_pdg(a, 1)}",
              f"{rho}:addChannel on 0.4995 102 {pi} {_diag_pdg(b, 1)}"]
    L.append("")

    L += ["! ================================================================",
          f"! 6) Dark pion decays  (decay_basis = {decay_basis})",
          "! ================================================================", ""]
    flavour_states = isinstance(model, DarkPionTChannelFlavourModel)
    eta = _diag_pdg(Nf - 1, 1)
    if not flavour_states and not (decay_basis == "flavour" and model.mix_diagonal):
        L += [f"! eta_1-like singlet ({eta}): stable", f"{eta}:mayDecay = off", ""]

    if decay_basis == "flavour" and flavour_states:
        # U(Nf) flavour-basis model (arXiv:1803.08080): Pythia's diagonal
        # mesons 4900aa1 are the flavour states pi_aa themselves.
        res = model.compute_all()
        for a in range(Nf):
            r = res["diagonal"].get(a, {"ctau_mm": np.inf, "br": {}})
            L += _decay_lines(_diag_pdg(a, 1), r["ctau_mm"], r["br"],
                              f"diagonal pion pi({a},{a})") + [""]
        for a, b in pairs:
            r = res["off_diagonal"].get((a, b), {"ctau_mm": np.inf, "br": {}})
            L += _decay_lines(_offdiag_pdg(a, b, 1), r["ctau_mm"], r["br"],
                              f"off-diagonal pion pi({a},{b})") + [""]

    elif decay_basis == "flavour":
        res = model.compute_all()
        for k, b in enumerate(diag_idx):
            r = res["diagonal"].get(b, {"ctau_mm": np.inf, "br": {}})
            L += _decay_lines(_diag_pdg(k, 1), r["ctau_mm"], r["br"],
                              f"diagonal pion {names.get(b, b)}") + [""]
        if model.mix_diagonal:
            # Pythia produces flavour states Q_a Q_a~; the highest-coded one is
            # treated as eta_1 only at string breaks (probKeepEta1).  Each diagonal
            # flavour state is an adjoint combination plus a 1/Nf singlet part, and
            # the copies produced in dark-rho decays carry no singlet component, so
            # with mixed diagonal pions it decays like the other diagonal states.
            r = min(res["diagonal"].values(), key=lambda x: x["ctau_mm"],
                    default={"ctau_mm": np.inf, "br": {}})
            L += _decay_lines(eta, r["ctau_mm"], r["br"],
                              f"diagonal pion {eta} (common mixed diagonal decay)") + [""]
        for a, b in pairs:
            r = res["off_diagonal"].get((a, b), {"ctau_mm": np.inf, "br": {}})
            L += _decay_lines(_offdiag_pdg(a, b, 1), r["ctau_mm"], r["br"],
                              f"off-diagonal pion pi({a},{b})") + [""]

    elif decay_basis == "eigen":
        G, _ = model.width_matrix()
        if np.max(np.abs(G.imag)) > 1e-9 * max(np.max(np.abs(G.real)), 1e-300):
            raise ValueError("decay_basis='eigen' needs a real width matrix (real kappa)")
        # Real eigenvectors, so a meson and its antiparticle (mirrored by Pythia)
        # have identical eigenstate content.
        vals, vecs = np.linalg.eigh(G.real)
        order = np.argsort(vals)[::-1]
        gmax  = max(vals.max(), 0.0)
        eig_vecs = []
        L += ["! Decay eigenstates of the degenerate pion multiplet (new particles)"]
        for k, idx in enumerate(order):
            v = vecs[:, idx]
            g = vals[idx] if vals[idx] > 1e-9 * gmax else 0.0
            br = {}
            if g > 0:
                for ij, (w, A) in model._eig_amp.items():
                    gk = w * abs(np.dot(A, v)) ** 2
                    if gk > 0:
                        br[ij] = gk
                tot = sum(br.values())
                br = {ij: x / tot for ij, x in br.items()}
            ctau = HBAR_C_GEV_MM / g if g > 0 else np.inf
            pdg = _eig_pdg(k)
            eig_vecs.append(v)
            L += [f"{pdg}:new = pivE{k + 1} void 1 0 0 {mPiD:.4f} 0.0 0.0 0.0 0.0",
                  f"{pdg}:isVisible = off"]
            L += _decay_lines(pdg, ctau, br, f"eigenstate {k + 1}") + [""]

        n = Nf ** 2 - 1
        for k, b in enumerate(diag_idx):
            L += _routing_lines(_diag_pdg(k, 1), [v[b] ** 2 for v in eig_vecs],
                                f"diagonal pion {names.get(b, b)}")
        for p, (a, b) in enumerate(pairs):
            # |ab> = (T_sym - i T_asym)/sqrt2  ->  |<k|ab>|^2 = (v_s^2 + v_a^2)/2
            s_i, a_i = 2 * p, 2 * p + 1
            L += _routing_lines(_offdiag_pdg(a, b, 1),
                                [0.5 * (v[s_i] ** 2 + v[a_i] ** 2) for v in eig_vecs],
                                f"off-diagonal pion pi({a},{b})")
    else:
        raise ValueError(f"unknown decay_basis '{decay_basis}'")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# Run card and launch script writers
# ---------------------------------------------------------------------------

def write_run_card(path: str, nevents: int = 10000, sqrts: float = 13600.0,
                   merged: bool = True, xqcut: float = 120.0) -> None:
    """Write a MadGraph run_card.dat.

    merged=True   t-channel continuum (p p > qd qd + 0,1,2j): MLM merging on.
    merged=False  on-shell mediator production: merging off.

    The two samples must not share a run card. On-shell mediator events carry
    no light ME partons outside the resonance decay, so MLM matches them
    exclusively and vetoes almost all of them; they also sit at a hard scale of
    2 m_X, where a 20 GeV merging scale is far too low.

    4-flavour scheme in both cases: maxjetflavor = 4, b massive, b not in p/j.
    The mediator still decays to b - that is a decay product, not a merging jet.
    """
    ebeam = sqrts / 2.0
    if merged:
        matching_block = f"""  1    = ickkw
  T    = cut_decays
  0.0  = drjj
  {xqcut:g}   = xqcut"""
    else:
        matching_block = """  0    = ickkw
  T    = cut_decays
  0.0  = drjj
  0.0  = xqcut"""
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
{matching_block}

#*********************************************************************
# Cuts on dark quarks (pT > 20 GeV to regulate the t-channel divergence).
# MG5 applies PDG cuts symmetrically to particle and antiparticle and rejects
# negative codes, so keep these positive even though the UFO now labels the
# dark quarks -4900101..103.  Not needed for the unmerged on-shell sample
# (no t-channel divergence there); left in place so phase space is identical
# between the two runs - drop it deliberately if you want the forward region.
#*********************************************************************
  {{4900101: 20, 4900102: 20, 4900103: 20}} = pt_min_pdg

#*********************************************************************
# Misc
#*********************************************************************
  15.0 = bwcutoff
  4    = maxjetflavor
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
                  kappa: float, kappa_source: str, ctau_source: str,
                  decay_basis: str = "flavour") -> None:
    results = model.compute_all()
    labels  = results["quark_labels"]
    names   = get_diagonal_names(model.Nf)

    def _brs(br):
        return ", ".join(f"{labels[i]}{labels[j]}bar {b:.3f}"
                         for (i, j), b in sorted(br.items(), key=lambda kv: -kv[1]) if b > 1e-3)

    print(f"\n{'='*72}")
    print(f"  Parameter point summary  (Nf = {model.Nf}, kappa_mode = {model.kappa_mode}, "
          f"decay_basis = {decay_basis})")
    print(f"{'='*72}")
    print(f"  mXd = {mXd} GeV,  mPiD = {model.m_piD} GeV,  fD = {model.fD} GeV")
    print(f"  κ = {kappa:.6f}  ({kappa_source})")
    print(f"  c·τ ({species}) = {_SPECIES_GETTERS[species](model):.4g} mm  ({ctau_source})")
    if decay_basis == "eigen":
        eig = model.decay_eigenstates()
        n_dec = sum(e["total"] > 0 for e in eig)
        print(f"  Decay eigenstates: {n_dec}/{len(eig)} decay")
        for k, e in enumerate(eig):
            if e["total"] > 0:
                print(f"    E{k + 1}: ctau = {e['ctau_mm']:.4g} mm  [{_brs(e['br'])}]")
    else:
        flav = isinstance(model, DarkPionTChannelFlavourModel)
        diag_keys = range(model.Nf) if flav else get_diagonal_indices(model.Nf)
        for b in diag_keys:
            r = results["diagonal"].get(b)
            lab = f"π({b},{b})" if flav else names.get(b, b)
            print(f"    {lab:>6}: " + (f"ctau = {r['ctau_mm']:.4g} mm  [{_brs(r['br'])}]"
                                       if r else "stable"))
        for ab, r in results["off_diagonal"].items():
            print(f"    π{ab}: ctau = {r['ctau_mm']:.4g} mm  [{_brs(r['br'])}]")
    print(f"{'='*72}\n")


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
    p.add_argument("--r-piD", type=float, default=0.6, dest="r_piD",
                   help="m_piD / Λ_D ratio (default: 0.6).  "
                        "Sets Λ_D, m_qD, and m_ρD via chiral-scaling relations.")
    p.add_argument("--mDarkQ", type=float, default=None,
                   help="Lagrangian dark quark mass [GeV] used in MadGraph (overrides "
                        "chiral-scaling default derived from --r-piD).  Pythia uses the "
                        "constituent mass (3 m_rhoD + m_piD) / 8 for fragmentation.")
    p.add_argument("--mRhoD", type=float, default=None,
                   help="Dark rho meson mass [GeV] (overrides chiral-scaling default "
                        "derived from --r-piD)")
    p.add_argument("--species", default="T15",
                   choices=list(_SPECIES_GETTERS),
                   help="Reference pion species used to set the ctau target "
                        "(default: T15 = longest-lived diagonal pion, bb̄-coupled)")
    p.add_argument("--pion-basis", default="generator", choices=["generator", "flavour"],
                   dest="pion_basis",
                   help="Dark pion widths in the SU(Nf) generator basis (default; "
                        "arXiv:2411.15073) or the U(Nf) flavour basis "
                        "(DarkPionTChannelFlavourModel; arXiv:1803.08080)")
    p.add_argument("--Nf", type=int, default=4, choices=[3, 4],
                   help="Number of dark flavours (default 4; the 4th has no SM coupling)")
    p.add_argument("--decay-basis", default="flavour", choices=["flavour", "eigen"],
                   dest="decay_basis",
                   help="Dark pion decays: 'flavour' (each flavour meson decays with its "
                        "flavour-state width, as in arXiv:2411.15073 / 1803.08080) or 'eigen' (decay "
                        "eigenstates of the degenerate pion multiplet, see "
                        "_dark_hadron_section)")
    p.add_argument("--no-mix-diagonal", dest="mix_diagonal", action="store_false",
                   help="Do not equalise diagonal-pion lifetimes (flavour basis only)")
    p.add_argument("--xqcut", type=float, default=120.0,
                   help="MLM xqcut [GeV] for --merged (Pythia qCut = 1.5 xqcut; default 120). "
                        "Validate for your process: the matched cross section should be stable "
                        "and the differential jet rates smooth (pythia/mlm_validation.cc)")
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
    sample = p.add_mutually_exclusive_group(required=True)
    sample.add_argument("--merged", dest="merged", action="store_true",
                        help="t-channel continuum sample (proc_cards/"
                             "dark_quark_pair_multijet.txt): MLM merging on "
                             "(ickkw=1, xqcut=20).")
    sample.add_argument("--unmerged", dest="merged", action="store_false",
                        help="On-shell mediator sample (proc_cards/"
                             "mediator_onshell.txt): merging off (ickkw=0). "
                             "Required - these events carry no light ME partons "
                             "outside the resonance decay, so MLM would veto "
                             "almost all of them.")

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
    # Chiral scaling relations (arXiv:XXXX):
    #   m_piD / Λ_D = 5.5 sqrt(m_qD / Λ_D)   →  m_qD = Λ_D (r/5.5)²
    #   m_rhoD / Λ_D = sqrt(5.76 + 1.5 (m_piD/Λ_D)²)
    r       = args.r_piD                                          # m_piD / Λ_D
    fD      = args.fD     if args.fD     is not None else args.mPiD
    LambdaD = args.mPiD / r
    mDarkQ  = args.mDarkQ if args.mDarkQ is not None else LambdaD * (r / 5.5) ** 2
    mRhoD   = args.mRhoD  if args.mRhoD  is not None else LambdaD * np.sqrt(5.76 + 1.5 * r ** 2)

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
                                args.species, args.kappa_mode, Nf=args.Nf,
                                mix_diagonal=args.mix_diagonal,
                                pion_basis=args.pion_basis)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        kappa_source = f"solved from c·τ = {args.ctau} mm ({args.species})"
        ctau_source  = f"target for {args.species}"
        print(f"\nSolved κ = {kappa:.6f} for c·τ = {args.ctau} mm ({args.species}).")

    # Build the model with the final kappa — generator basis (SU(Nf) adjoint)
    model = _model_class(args.pion_basis)(fD=fD, m_piD=args.mPiD, m_X=args.mXd,
                                          kappa=kappa, kappa_mode=args.kappa_mode,
                                          Nf=args.Nf, mix_diagonal=args.mix_diagonal)

    # Default output dir uses whichever of ctau/kappa was given
    if args.output is not None:
        outdir = args.output
    elif args.ctau is not None:
        outdir = f"cards/mXd{args.mXd:.0f}_mPiD{args.mPiD:.0f}_ctau{args.ctau:.0f}"
    else:
        outdir = f"cards/mXd{args.mXd:.0f}_mPiD{args.mPiD:.0f}_kappa{args.kappa:.4f}"

    # Print summary
    print_summary(model, args.mXd, args.species, kappa, kappa_source, ctau_source,
                  decay_basis=args.decay_basis)

    # Write cards
    os.makedirs(outdir, exist_ok=True)
    param_path   = os.path.join(outdir, "param_card.dat")
    run_path     = os.path.join(outdir, "run_card.dat")
    pythia_path  = os.path.join(outdir, "pythia_card.dat")
    launch_path  = os.path.join(outdir, "launch.sh")

    write_param_card(param_path, args.mXd, mDarkQ, kappa, args.kappa_mode)
    write_run_card(run_path, nevents=args.nevents, sqrts=args.sqrts,
                   merged=args.merged, xqcut=args.xqcut)
    write_pythia_card(pythia_path, model, args.mXd, mDarkQ, mRhoD, LambdaD,
                      lhe_file=args.lhe, merged=args.merged,
                      decay_basis=args.decay_basis, xqcut=args.xqcut)
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
