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
                     ints b (generator index) for diagonal pions.
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
