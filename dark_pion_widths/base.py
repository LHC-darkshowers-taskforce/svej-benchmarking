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

        self.mix_diagonal = bool(mix_diagonal)

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
