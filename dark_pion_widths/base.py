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


class DarkPionModelBase(ABC):
    """
    Abstract base class for dark pion decay models.

    Parameters
    ----------
    fD : float
        Dark pion decay constant [GeV].
    m_piD : float
        Dark pion mass [GeV].
    Nc : int
        Number of dark colours (default 3).
    """

    HBAR_C_GEV_MM: float = HBAR_C_GEV_MM

    def __init__(self, fD: float, m_piD: float, Nc: int = 3) -> None:
        self.fD    = float(fD)
        self.m_piD = float(m_piD)
        self.Nc    = int(Nc)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_all(self) -> dict:
        """
        Compute widths, branching ratios, and lifetimes for every dark pion.

        Returns a model-specific dict; callers should use the generic
        accessors below rather than parsing this dict directly.
        """
        ...

    @abstractmethod
    def get_pion_ids(self) -> list:
        """
        Return the list of pion identifiers that have non-zero total width.

        T-channel: tuples (alpha, beta) for off-diagonal pions and ints b
                   for diagonal pions.
        S-channel: ints 0 .. Nf^2 - 2 (generator index).
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

    # ------------------------------------------------------------------
    # Concrete utility
    # ------------------------------------------------------------------

    @staticmethod
    def width_to_ctau_mm(width_GeV: float) -> float:
        """Convert a decay width [GeV] to c·τ [mm]."""
        return HBAR_C_GEV_MM / width_GeV if width_GeV > 0 else np.inf
