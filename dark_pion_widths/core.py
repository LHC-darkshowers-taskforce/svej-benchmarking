"""
dark_pion_widths.py
-------------------
Module for computing partial widths, total widths, branching ratios,
and lifetimes of dark pions in an SU(4) dark-sector model.

The mediator X couples SM down-type quarks to the dark sector via
κ_{α,i}, where α is the SU(4) dark-flavor index and i labels the SM quark.
Dark pions are the pseudo-Nambu-Goldstone bosons of spontaneous SU(4) breaking.

Two species:
  - Off-diagonal pions π^(α,β)  (α < β < 4, flavored)
  - Diagonal pions π^b           (b ∈ {2, 7, 14} → generators T3, T8, T15)

Usage
-----
    from dark_pion_widths import DarkPionModel

    model = DarkPionModel(fD=10.0, m_piD=10.0, m_X=2000.0, kappa=1.0)
    results = model.compute_all()

    # Lifetime of diagonal pion T3 (b=2)
    print(model.ctau_diag_mm(2))

    # Full results dict
    for (a, b), r in results['off_diagonal'].items():
        print(f"pi^({a},{b}): total={r['total']:.3e} GeV, ctau={r['ctau_mm']:.3e} mm")
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR_C_GEV_MM = 1.973269788e-13   # GeV·mm  (hbar*c in natural units → mm)

# ---------------------------------------------------------------------------
# Default SM down-type quark masses (GeV)
# ---------------------------------------------------------------------------
DEFAULT_QUARKS = {"d": 4.7e-3, "s": 9.6e-2, "b": 4.18}

# Generator indices (0-indexed) for the three diagonal neutral dark pions
DIAGONAL_PION_INDICES = [2, 7, 14]
DIAGONAL_PION_NAMES   = {2: "T3", 7: "T8", 14: "T15"}


# ---------------------------------------------------------------------------
# SU(4) generators  (built once at import time)
# ---------------------------------------------------------------------------
def _build_su4_generators():
    c = complex
    s3 = np.sqrt(3)
    s6 = np.sqrt(6)
    z  = np.zeros((4, 4), dtype=complex)

    def _mat(entries):
        m = z.copy()
        for (r, col), v in entries:
            m[r, col] = v
        return m

    T1  = _mat([((0,1), 0.5),  ((1,0), 0.5)])
    T2  = _mat([((0,1),-0.5j), ((1,0), 0.5j)])
    T3  = _mat([((0,0), 0.5),  ((1,1),-0.5)])
    T4  = _mat([((0,2), 0.5),  ((2,0), 0.5)])
    T5  = _mat([((0,2),-0.5j), ((2,0), 0.5j)])
    T6  = _mat([((1,2), 0.5),  ((2,1), 0.5)])
    T7  = _mat([((1,2),-0.5j), ((2,1), 0.5j)])
    T8  = np.diag([1, 1, -2, 0]).astype(complex) / (2*s3)
    T9  = _mat([((0,3), 0.5),  ((3,0), 0.5)])
    T10 = _mat([((0,3),-0.5j), ((3,0), 0.5j)])
    T11 = _mat([((1,3), 0.5),  ((3,1), 0.5)])
    T12 = _mat([((1,3),-0.5j), ((3,1), 0.5j)])
    T13 = _mat([((2,3), 0.5),  ((3,2), 0.5)])
    T14 = _mat([((2,3),-0.5j), ((3,2), 0.5j)])
    T15 = np.diag([1, 1, 1, -3]).astype(complex) / (2*s6)

    return [T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15]


_T_LIST = _build_su4_generators()


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------
class DarkPionModel:
    """
    SU(4) dark-pion decay model.

    Parameters
    ----------
    fD : float
        Dark pion decay constant (GeV).
    m_piD : float
        Dark pion mass (GeV).
    m_X : float
        Mediator mass (GeV).
    kappa : complex
        Universal coupling κ applied to dark-flavor indices 0, 1, 2.
        Index 3 always has κ = 0 (SM singlet direction).
    Nc : int
        Number of dark colours (default 3).
    quarks : dict, optional
        SM quark masses {label: mass_GeV}. Defaults to down-type {d, s, b}.
    """

    def __init__(
        self,
        fD: float = 10.0,
        m_piD: float = 10.0,
        m_X: float = 2000.0,
        kappa: complex = 1.0 + 0j,
        Nc: int = 3,
        quarks: dict | None = None,
    ):
        self.fD    = float(fD)
        self.m_piD = float(m_piD)
        self.m_X   = float(m_X)
        self.kappa = complex(kappa)
        self.Nc    = int(Nc)
        self.quarks = dict(quarks) if quarks is not None else dict(DEFAULT_QUARKS)

        self._labels = list(self.quarks.keys())
        self._masses = np.array(list(self.quarks.values()), dtype=float)
        self._nq     = len(self._masses)
        self._kmat   = self._build_kappa_matrix()
        self._Tlist  = _T_LIST

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kappa_matrix(self) -> np.ndarray:
        """κ_{α,i} matrix, shape (4, n_quarks).  Row 3 is identically zero."""
        mat = np.zeros((4, self._nq), dtype=complex)
        mat[0:3, :] = self.kappa
        return mat

    def _omega(self, mi: float, mj: float) -> float:
        """
        Kinematic phase-space factor Ω(mi, mj; m_piD).
        Returns 0 if the decay is kinematically forbidden.
        """
        mpiD = self.m_piD
        if mpiD < mi + mj:
            return 0.0
        # Guard against mi == mj == 0
        denom = mi**2 + mj**2
        term1 = 1.0 - ((mi**2 - mj**2)**2) / (denom * mpiD**2) if denom > 0 else 1.0
        arg   = (1 - (mi + mj)**2 / mpiD**2) * (1 - (mi - mj)**2 / mpiD**2)
        return float(term1 * np.sqrt(max(0.0, arg)))

    # ------------------------------------------------------------------
    # Partial widths
    # ------------------------------------------------------------------

    def gamma_off_diag(self, alpha: int, beta: int, i: int, j: int) -> float:
        """
        Partial width Γ(π^(α,β) → q_i q̄_j)  [GeV].

        Parameters
        ----------
        alpha, beta : SU(4) dark-flavor indices (0–3).
        i, j        : SM quark indices (0 .. n_quarks-1).
        """
        if alpha >= 4 or beta >= 4:
            return 0.0
        Omega = self._omega(self._masses[i], self._masses[j])
        if Omega == 0.0:
            return 0.0
        pref  = self.Nc * self.fD**2 * self.m_piD / (128 * np.pi * self.m_X**4)
        kcoef = self._kmat[alpha, i] * np.conj(self._kmat[beta, j])
        return float(pref * abs(kcoef)**2 * (self._masses[i]**2 + self._masses[j]**2) * Omega)

    def gamma_diag(self, b: int, i: int, j: int) -> float:
        """
        Partial width Γ(π^b → q_i q̄_j)  [GeV]  for generator index b (0-indexed).

        Parameters
        ----------
        b    : SU(4) generator index (0–14).
        i, j : SM quark indices.
        """
        T = self._Tlist[b]
        amp = complex(0)
        for alpha in range(4):
            for beta_ in range(4):
                amp += self._kmat[alpha, i] * np.conj(self._kmat[beta_, j]) * T[alpha, beta_]
        Omega = self._omega(self._masses[i], self._masses[j])
        if Omega == 0.0:
            return 0.0
        pref = self.Nc * self.fD**2 * self.m_piD / (64 * np.pi * self.m_X**4)
        return float(pref * abs(amp)**2 * (self._masses[i]**2 + self._masses[j]**2) * Omega)

    # ------------------------------------------------------------------
    # Per-pion summaries
    # ------------------------------------------------------------------

    def _summarise(self, channels: dict) -> dict:
        """
        Given {channel_key: width_GeV}, return a summary dict:
          channels, total, br, ctau_mm.
        """
        total  = sum(channels.values())
        br     = {k: v / total for k, v in channels.items()} if total > 0 else {}
        ctau   = HBAR_C_GEV_MM / total if total > 0 else np.inf
        return {"channels": channels, "total": total, "br": br, "ctau_mm": ctau}

    def compute_off_diagonal_pion(self, alpha: int, beta: int) -> dict:
        """
        Widths, branching ratios, and lifetime for π^(α,β).

        Returns
        -------
        dict
            channels  : {(i, j): width_GeV}
            total     : float  [GeV]
            br        : {(i, j): branching_ratio}
            ctau_mm   : float  [mm]
        """
        channels = {}
        for i in range(self._nq):
            for j in range(i, self._nq):
                g = self.gamma_off_diag(alpha, beta, i, j)
                if g > 0:
                    channels[(i, j)] = g
        return self._summarise(channels)

    def compute_diagonal_pion(self, b: int) -> dict:
        """
        Widths, branching ratios, and lifetime for diagonal pion π^b
        (generator index b, 0-indexed).

        Returns
        -------
        dict
            channels  : {(i, j): width_GeV}
            total     : float  [GeV]
            br        : {(i, j): branching_ratio}
            ctau_mm   : float  [mm]
        """
        channels = {}
        # i == j: self-conjugate final state (count once)
        for i in range(self._nq):
            g = self.gamma_diag(b, i, i)
            if g > 0:
                channels[(i, i)] = g
        # i != j: q_i q̄_j and q̄_i q_j are distinct
        for i in range(self._nq):
            for j in range(self._nq):
                if i != j:
                    g = self.gamma_diag(b, i, j)
                    if g > 0:
                        channels[(i, j)] = g
        return self._summarise(channels)

    # ------------------------------------------------------------------
    # Full model scan
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Compute widths, branching ratios, and lifetimes for every dark pion.

        Returns
        -------
        dict
            off_diagonal : {(alpha, beta): summary_dict}
            diagonal     : {b: summary_dict}
            quark_labels : list[str]
        """
        off_diag = {}
        for a in range(4):
            for b in range(a + 1, 4):
                r = self.compute_off_diagonal_pion(a, b)
                if r["total"] > 0:
                    off_diag[(a, b)] = r

        diag = {}
        for b in DIAGONAL_PION_INDICES:
            r = self.compute_diagonal_pion(b)
            if r["total"] > 0:
                diag[b] = r

        return {
            "off_diagonal": off_diag,
            "diagonal":     diag,
            "quark_labels": self._labels,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def ctau_off_diag_mm(self, alpha: int, beta: int) -> float:
        """c·τ [mm] for off-diagonal pion π^(α,β)."""
        return self.compute_off_diagonal_pion(alpha, beta)["ctau_mm"]

    def ctau_diag_mm(self, b: int) -> float:
        """c·τ [mm] for diagonal pion π^b (generator index b, 0-indexed)."""
        return self.compute_diagonal_pion(b)["ctau_mm"]

    def channel_label(self, i: int, j: int) -> str:
        """Human-readable label for the decay channel (i, j)."""
        if i == j:
            return f"{self._labels[i]}$\\bar{{{self._labels[i]}}}$"
        return f"{self._labels[i]}$\\bar{{{self._labels[j]}}}$"

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def width_to_ctau_mm(width_GeV: float) -> float:
        """Convert a width in GeV to c·τ in mm."""
        return HBAR_C_GEV_MM / width_GeV if width_GeV > 0 else np.inf
