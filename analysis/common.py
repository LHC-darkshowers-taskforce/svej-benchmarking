"""Shared constants and helpers for the dark-QCD / emerging-jets HepMC analysis.

Everything that is not specific to one script lives here: PDG-ID classes,
sample/grid discovery, Pythia-card parsing, output-path conventions and a
few kinematic helpers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locations: $EJ_BASE is the directory holding runs/<sample>/<point>/ (the
# submit_grid.py --runs-dir parent; default: current directory); outputs go
# to $EJ_OUT (default: $EJ_BASE/analysis_outputs).
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("EJ_BASE", "."))
OUT = Path(os.environ.get("EJ_OUT", str(BASE / "analysis_outputs")))
FEATURES = OUT / "features"
PLOTS = OUT / "plots"

SAMPLES = ("mediator_onshell", "dark_quark_pair_multijet")
SAMPLE_LABEL = {
    "mediator_onshell": "on-shell X",
    "dark_quark_pair_multijet": "off-shell t-ch. (merged)",
}
SAMPLE_SHORT = {"mediator_onshell": "onshell", "dark_quark_pair_multijet": "merged"}
SHORT_TO_SAMPLE = {v: k for k, v in SAMPLE_SHORT.items()}

POINT_RE = re.compile(r"^mXd(?P<mX>\d+)_mPiD(?P<mPiD>\d+)_ctau(?P<ctau>\d+)$")

# ---------------------------------------------------------------------------
# PDG IDs (absolute values)
# ---------------------------------------------------------------------------
PID_X = 4900001
PID_DARK_QUARKS = (4900101, 4900102, 4900103, 4900104)
PID_DARK_GLUON = 4900021
PID_PI_DIAG = (4900111, 4900221, 4900331)               # decaying, diagonal
PID_PI_OFFDIAG = (4900211, 4900311, 4900321)            # decaying, off-diagonal
PID_PI_DECAYING = PID_PI_DIAG + PID_PI_OFFDIAG
PID_PI_STABLE = (4900441, 4900411, 4900421, 4900431)    # invisible (4th flavour / eta-like)
PID_PI_ALL = PID_PI_DECAYING + PID_PI_STABLE
PID_RHO = (4900113, 4900223, 4900333, 4900443, 4900213,
           4900313, 4900323, 4900413, 4900423, 4900433)
PID_NU = (12, 14, 16)

PI_NAME = {4900111: "T3", 4900221: "T8", 4900331: "T15",
           4900211: "pi01", 4900311: "pi02", 4900321: "pi12",
           4900441: "eta4", 4900411: "pi03", 4900421: "pi13", 4900431: "pi23"}


def is_eig_pion(apid):
    """Decay-eigenstate pions written by write_cards.py --decay-basis eigen (4910kk1)."""
    return (apid > 4910000) & (apid < 4920000)


def is_dark(apid):
    """|pid| in the hidden-valley range (works on scalars and arrays)."""
    return (apid >= 4900000) & (apid <= 4999999)


def is_b_hadron(apid: np.ndarray) -> np.ndarray:
    """True for SM hadrons containing a b (anti)quark (PDG numbering scheme)."""
    apid = np.asarray(apid)
    ok = (apid > 100) & (apid < 1_000_000)
    n1 = (apid // 1000) % 10
    n2 = (apid // 100) % 10
    n3 = (apid // 10) % 10
    meson = ok & (n1 == 0) & ((n2 == 5) | (n3 == 5))
    baryon = ok & (n1 != 0) & ((n1 == 5) | (n2 == 5) | (n3 == 5))
    return meson | baryon


@lru_cache(maxsize=None)
def _charge_scalar(pid: int) -> float:
    if is_dark(abs(pid)):
        return 0.0
    try:
        from particle.pdgid import charge
        c = charge(pid)
        return float(c) if c is not None else 0.0
    except Exception:  # unknown/odd codes -> treat as neutral
        return 0.0


def charges(pids: np.ndarray) -> np.ndarray:
    """Electric charge for an array of PDG IDs (cached per distinct ID)."""
    u, inv = np.unique(pids, return_inverse=True)
    return np.array([_charge_scalar(int(p)) for p in u])[inv]


# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------
def pt_eta_phi(px, py, pz):
    px, py, pz = (np.asarray(a, dtype=float) for a in (px, py, pz))
    pt = np.hypot(px, py)
    p = np.sqrt(pt**2 + pz**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.arctanh(np.where(p > 0, pz / np.where(p > 0, p, 1), 0.0))
    eta = np.where(pt > 0, eta, np.sign(pz) * 1e3)  # along the beam: huge |eta|
    eta = np.clip(eta, -1e3, 1e3)
    phi = np.arctan2(py, px)
    return pt, eta, phi


def dphi(a, b):
    d = np.asarray(a) - np.asarray(b)
    return (d + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    return np.hypot(np.asarray(eta1) - np.asarray(eta2), dphi(phi1, phi2))


# ---------------------------------------------------------------------------
# Sample discovery
# ---------------------------------------------------------------------------
@dataclass
class Point:
    sample: str
    name: str
    mX: int
    mPiD: int
    ctau: int

    @property
    def hepmc(self) -> Path:
        return BASE / "runs" / self.sample / self.name / "Events" / self.name / f"{self.name}.hepmc.gz"

    @property
    def event_dir(self) -> Path:
        return self.hepmc.parent

    @property
    def card(self) -> Path:
        return BASE / "cards" / self.sample / self.name / "pythia_card.dat"

    @property
    def outdir(self) -> Path:
        return FEATURES / self.sample / self.name

    @property
    def label(self) -> str:
        return f"{SAMPLE_SHORT[self.sample]} mX={self.mX} mπ={self.mPiD} cτ={self.ctau}"

    def asdict(self):
        return asdict(self)


def parse_point_name(name: str) -> dict:
    m = POINT_RE.match(name)
    if not m:
        raise ValueError(f"not a grid point name: {name}")
    return {k: int(v) for k, v in m.groupdict().items()}


def discover(samples=SAMPLES, mX=None, mPiD=None, ctau=None, require_hepmc=True) -> list[Point]:
    """All grid points on disk, optionally filtered by parameter lists."""
    pts = []
    for s in samples:
        d = BASE / "runs" / s
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not POINT_RE.match(sub.name):
                continue
            par = parse_point_name(sub.name)
            if mX and par["mX"] not in mX:
                continue
            if mPiD and par["mPiD"] not in mPiD:
                continue
            if ctau and par["ctau"] not in ctau:
                continue
            p = Point(sample=s, name=sub.name, **par)
            if require_hepmc and not p.hepmc.is_file():
                continue
            pts.append(p)
    pts.sort(key=lambda p: (SAMPLES.index(p.sample), p.mX, p.mPiD, p.ctau))
    return pts


# ---------------------------------------------------------------------------
# Pythia card / run-log parsing (for metadata and validation)
# ---------------------------------------------------------------------------
_HDR_RE = re.compile(r"^!\s+(\w+)\s+=\s+([-+0-9.eE]+)")
_TAU_RE = re.compile(r"^(\d{7}):tau0\s*=\s*([-+0-9.eE]+)")
_M0_RE = re.compile(r"^(\d{7}):m0\s*=\s*([-+0-9.eE]+)")


def parse_pythia_card(path: Path) -> dict:
    """Header model parameters, tau0 [mm] and m0 [GeV] per dark PDG ID."""
    out = {"header": {}, "tau0_mm": {}, "m0": {}, "merging": None}
    if not Path(path).is_file():
        return out
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if m := _HDR_RE.match(s):
            try:
                out["header"][m.group(1)] = float(m.group(2))
            except ValueError:
                pass
        elif m := _TAU_RE.match(s):
            out["tau0_mm"][int(m.group(1))] = float(m.group(2))
        elif m := _M0_RE.match(s):
            out["m0"][int(m.group(1))] = float(m.group(2))
        elif s.startswith("JetMatching:merge") and "=" in s:
            out["merging"] = s.split("=")[1].split("!")[0].strip()
    return out


def parse_run_logs(event_dir: Path) -> dict:
    """MG5 integrated weight (banner) and Pythia sigmaGen (pythia.log), if present."""
    res = {}
    for b in Path(event_dir).glob("*_banner.txt"):
        for line in b.read_text(errors="ignore").splitlines():
            if "Integrated weight (pb)" in line:
                try:
                    res["xsec_mg5_pb"] = float(line.split(":")[-1])
                except ValueError:
                    pass
    log = Path(event_dir) / "pythia.log"
    if log.is_file():
        txt = log.read_text(errors="ignore")
        if m := re.findall(r"sigmaGen\s*=\s*([-+0-9.eE]+)\s*fb", txt):
            res["sigmaGen_fb"] = float(m[-1])
        if m := re.findall(r"accepted\s+(\d+),\s+failed\s+(\d+),\s+LHE events read\s+(\d+)", txt):
            a, f, r = m[-1]
            res.update(pythia_accepted=int(a), pythia_failed=int(f), lhe_read=int(r))
    return res


def load_meta(outdir: Path) -> dict:
    with open(Path(outdir) / "meta.json") as f:
        return json.load(f)
