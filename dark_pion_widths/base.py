"""
base.py
-------
Abstract base class for dark pion decay models.

All concrete models (t-channel, s-channel) inherit from DarkPionModelBase and
implement the abstract interface used by the shared plotting scripts.
"""

from abc import ABC, abstractmethod
import numpy as np

HBAR_C_GEV_MM = 1.973269788e-13   # ℏc  in GeV·mm
HBAR_GEV_S    = 6.582119569e-25   # ℏ   in GeV·s

# Lightest meson for each quark–antiquark flavour pair [GeV] (PDG).  Keys are
# sorted flavour labels; flavour-neutral light pairs use the π⁰ and η.
_LIGHTEST_MESON: dict[tuple[str, str], float] = {
    ("d", "d"): 0.1350, ("u", "u"): 0.1350, ("d", "u"): 0.1396,   # π
    ("s", "s"): 0.5479,                                           # η
    ("d", "s"): 0.4976, ("s", "u"): 0.4937,                       # K
    ("c", "c"): 2.9839,                                           # η_c
    ("c", "d"): 1.8697, ("c", "u"): 1.8648, ("c", "s"): 1.9683,   # D, D_s
    ("b", "b"): 9.3987,                                           # η_b
    ("b", "d"): 5.2797, ("b", "u"): 5.2793, ("b", "s"): 5.3669,   # B, B_s
    ("b", "c"): 6.2745,                                           # B_c
}
_M_PI0 = 0.1350


def _lightest_meson(qi: str, qj: str) -> float | None:
    return _LIGHTEST_MESON.get(tuple(sorted((qi, qj))))


def hadron_threshold(qi: str, qj: str) -> float | None:
    """
    Lightest two-meson final state that a q_i q̄_j pair can hadronise into [GeV].

    A quark pair from the pion decay must end up in at least two hadrons: either
    the (q_i q̄_j) meson plus a pion, or (q_i ℓ̄)(ℓ q̄_j) with a light ℓ = u, d
    popped from the vacuum.  E.g. b s̄ → B_s π (5.50), b d̄ → B π (5.41),
    b b̄ → η_b π (9.53), s s̄ → η π (0.68).  Quantum-number selection rules are
    ignored (the chiral coupling does not conserve parity).

    Returns None for flavours without a hadron table entry (e.g. t), in which
    case only the parton-level threshold applies.
    """
    direct = _lightest_meson(qi, qj)
    if direct is None:
        return None
    via_light = [
        _lightest_meson(qi, l) + _lightest_meson(l, qj) for l in ("u", "d")
    ]
    return min(direct + _M_PI0, *via_light)


class DarkPionModelBase(ABC):
    """
    Abstract base class for dark pion decay models.

    Parameters
    ----------
    fD : float
        Dark pion decay constant [GeV].
    m_piD : float
        Dark pion mass [GeV].
    Nf : int
        Number of dark quark flavors (SU(Nf) dark flavor group).
    Nd : int
        Number of dark colours in the dark gauge group SU(Nd).  Default 3.
    Nc : int
        Number of SM colours (default 3).
    sm_quarks : dict
        SM quark masses {label: mass_GeV}.  Must be provided by the subclass.
    hadron_thresholds : bool
        When True (default), a decay channel π → q_i q̄_j is open only if m_piD
        exceeds the lightest two-meson final state with that flavour content
        (see hadron_threshold), in addition to the parton-level threshold
        m_i + m_j.  Without it, e.g. a 5 GeV pion "decays" to b s̄ at parton
        level although no b-hadron final state is kinematically allowed, and the
        b channel then dominates the helicity-suppressed width.
    mix_diagonal : bool
        When True (default), dark QCD interactions mix all diagonal pions into
        each other faster than they decay.  All diagonal pions therefore share
        the lifetime and branching ratios of the fastest-decaying individual
        diagonal state.  Pions that would individually be stable (e.g. because
        they involve a dark quark with no SM coupling) also pick up the mixed
        lifetime.
    """

    HBAR_C_GEV_MM: float = HBAR_C_GEV_MM
    HBAR_GEV_S:    float = HBAR_GEV_S

    def __init__(
        self,
        fD: float,
        m_piD: float,
        Nf: int,
        Nd: int = 3,
        Nc: int = 3,
        sm_quarks: dict | None = None,
        mix_diagonal: bool = True,
        hadron_thresholds: bool = True,
    ) -> None:
        self.fD    = float(fD)
        self.m_piD = float(m_piD)
        self.Nf    = int(Nf)
        self.Nd    = int(Nd)
        self.Nc    = int(Nc)

        if sm_quarks is None:
            raise ValueError("sm_quarks must be provided by the subclass")
        self.sm_quarks  = dict(sm_quarks)
        self._q_labels  = list(self.sm_quarks.keys())
        self._q_masses  = np.array(list(self.sm_quarks.values()), dtype=float)
        self._nq        = len(self._q_masses)

        self.mix_diagonal      = bool(mix_diagonal)
        self.hadron_thresholds = bool(hadron_thresholds)

    def channel_open(self, i: int, j: int) -> bool:
        """Whether π → q_i q̄_j is kinematically open (see hadron_thresholds)."""
        if self.m_piD <= self._q_masses[i] + self._q_masses[j]:
            return False
        if self.hadron_thresholds:
            thr = hadron_threshold(self._q_labels[i], self._q_labels[j])
            if thr is not None and self.m_piD <= thr:
                return False
        return True

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_all(self) -> dict:
        """
        Compute widths, branching ratios, and lifetimes for every dark pion.

        Returns
        -------
        dict with keys:
            off_diagonal : {(alpha, beta): summary_dict}  — decaying pairs only
            diagonal     : {b: summary_dict}              — decaying diagonal only
            quark_labels : list[str]
        """
        ...

    @abstractmethod
    def get_pion_ids(self) -> list:
        """
        Return the list of pion identifiers that have non-zero total width.

        Both models: tuples (alpha, beta) for off-diagonal pions,
                     ints b for diagonal pions.
        """
        ...

    @abstractmethod
    def ctau_for_pion(self, pion_id) -> float:
        """Return c·τ [mm] for the pion identified by pion_id."""
        ...

    @abstractmethod
    def pion_label(self, pion_id) -> str:
        """Return a LaTeX label string for the given pion."""
        ...

    @abstractmethod
    def print_summary(self) -> None:
        """Print a formatted summary of all dark pion properties."""
        ...

    # ------------------------------------------------------------------
    # Concrete utility
    # ------------------------------------------------------------------

    @staticmethod
    def width_to_ctau_mm(width_GeV: float) -> float:
        """Convert a decay width [GeV] to c·τ [mm]."""
        return HBAR_C_GEV_MM / width_GeV if width_GeV > 0 else np.inf

    @staticmethod
    def _apply_diagonal_mixing(all_diag: dict) -> dict:
        """
        Equalise all diagonal pions to the fastest-decaying one.

        Dark QCD interactions mix diagonal pion states faster than the pions
        decay.  All diagonal pion states therefore share the lifetime and
        branching ratios of the fastest-decaying individual species (the
        "driver").  This includes pions that would individually be stable
        (e.g. those involving dark quarks with no SM coupling), which now
        inherit the driver's channels and BRs.

        If no diagonal pion decays (all total=0), returns input unchanged.

        Parameters
        ----------
        all_diag : dict
            {key: summary_dict} for EVERY diagonal pion (including stable ones).

        Returns
        -------
        dict  Same keys; every entry overwritten with driver's total, ctau_mm,
              channels, and br.
        """
        decaying = {b: r for b, r in all_diag.items() if r["total"] > 0}
        if not decaying:
            return all_diag   # all individually stable — nothing to mix

        driver_b = min(decaying, key=lambda b: decaying[b]["ctau_mm"])
        driver   = decaying[driver_b]

        # Fields that are universal to every summary dict
        shared = {"total": driver["total"], "ctau_mm": driver["ctau_mm"],
                  "br": driver["br"], "_mix_driver": driver_b}
        # "channels" is present in t-channel dicts; copy it if available
        if "channels" in driver:
            shared["channels"] = driver["channels"]

        return {b: {**r, **shared} for b, r in all_diag.items()}
