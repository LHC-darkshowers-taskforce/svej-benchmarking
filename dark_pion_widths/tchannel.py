"""
tchannel.py
-----------
Generalised t-channel dark pion decay model.

Mediator X couples SM down-type quarks to dark quarks via the coupling matrix
κ_{α,i}, where α is the dark-quark flavor index and i is the SM quark index.
Dark pions are the pseudo-Nambu-Goldstone bosons of spontaneous SU(Nf) breaking.

κ matrix conventions
---------------------
Shape (Nf, n_quarks).  Only the first n_active = min(Nf, 3) rows can ever be
non-zero; the remaining rows are identically zero (SM-singlet dark flavors).
For Nf ≤ 3 all dark flavors are active.

Three kappa_mode options control the structure of the non-zero block:

  "universal"  — κ_{α,i} = κ  for α < n_active, all i.
                 All active dark flavors couple equally to every SM quark.
                 This reproduces the original SU(4) benchmark exactly.

  "diagonal"   — κ_{α,i} = κ · δ_{α,i}  for α, i < n_active.
                 Dark flavor α couples only to SM quark α.
                 For Nf = 3 and quarks = {d, s, b} this is a proper 3×3
                 diagonal square matrix.

  "down_only"  — κ_{α,0} = κ  for α < n_active, κ_{α,i>0} = 0.
                 All active dark flavors couple only to the SM d quark.

Generator ordering
------------------
Uses build_sun_generators(Nf) from generators.py.  For Nf = 4 the diagonal
pions T3, T8, T15 are at indices 12, 13, 14 respectively.  T15 is still at
index 14, matching the old core.py; T3 and T8 move from 2, 7 to 12, 13.
For universal κ, T3 and T8 have zero width (their generator traces cancel),
so existing benchmark results for T15 are unchanged.
"""

import numpy as np

from .base import DarkPionModelBase, HBAR_C_GEV_MM
from .generators import build_sun_generators, get_diagonal_indices, get_diagonal_names, get_generator_label

# ---------------------------------------------------------------------------
# Default SM down-type quark masses (GeV)
# ---------------------------------------------------------------------------
DEFAULT_QUARKS: dict[str, float] = {"d": 4.7e-3, "s": 9.6e-2, "b": 4.18}

_KAPPA_MODES = ("universal", "diagonal", "down_only")


class DarkPionTChannelModel(DarkPionModelBase):
    """
    Generalised SU(Nf) dark pion decay model with a t-channel mediator X.

    Parameters
    ----------
    fD : float
        Dark pion decay constant [GeV].
    m_piD : float
        Dark pion mass [GeV].
    m_X : float
        Mediator mass [GeV].
    kappa : complex
        Overall coupling magnitude.  The precise structure within the κ matrix
        is controlled by kappa_mode.
    Nf : int
        Total number of dark quark flavors (SU(Nf) dark flavor group).
        Default 4 (reproduces original SU(4) benchmark).
    kappa_mode : str
        One of 'universal', 'diagonal', 'down_only'.  See module docstring.
    Nc : int
        Number of dark colours (default 3).
    sm_quarks : dict, optional
        SM quark masses {label: mass_GeV}.  Defaults to {d, s, b}.
    """

    def __init__(
        self,
        fD: float = 10.0,
        m_piD: float = 10.0,
        m_X: float = 2000.0,
        kappa: complex = 1.0 + 0j,
        Nf: int = 4,
        kappa_mode: str = "universal",
        Nd: int = 3,
        Nc: int = 3,
        sm_quarks: dict | None = None,
    ) -> None:
        resolved_quarks = dict(sm_quarks) if sm_quarks is not None else dict(DEFAULT_QUARKS)
        super().__init__(fD=fD, m_piD=m_piD, Nf=Nf, Nd=Nd, Nc=Nc, sm_quarks=resolved_quarks)

        self.m_X        = float(m_X)
        self.kappa      = complex(kappa)
        self.kappa_mode = str(kappa_mode)

        if self.kappa_mode not in _KAPPA_MODES:
            raise ValueError(
                f"kappa_mode must be one of {_KAPPA_MODES}, got '{kappa_mode}'"
            )

        # Number of active dark flavors that can couple to SM quarks.
        # Rows n_active .. Nf-1 of the κ matrix are identically zero.
        self._n_active = min(self.Nf, 3)

        self._kmat  = self._build_kappa_matrix()
        self._Tlist = build_sun_generators(self.Nf)

        self._diagonal_indices = get_diagonal_indices(self.Nf)
        self._diagonal_names   = get_diagonal_names(self.Nf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kappa_matrix(self) -> np.ndarray:
        """
        Build the κ matrix of shape (Nf, n_quarks).

        Only rows 0 .. n_active-1 are filled; rows n_active .. Nf-1 are zero.
        The non-zero block depends on kappa_mode.
        """
        mat = np.zeros((self.Nf, self._nq), dtype=complex)
        na  = self._n_active

        if self.kappa_mode == "universal":
            mat[:na, :] = self.kappa

        elif self.kappa_mode == "diagonal":
            n_diag = min(na, self._nq)
            for idx in range(n_diag):
                mat[idx, idx] = self.kappa

        elif self.kappa_mode == "down_only":
            mat[:na, 0] = self.kappa   # only SM quark index 0 (d quark)

        return mat

    def _omega(self, mi: float, mj: float) -> float:
        """
        Kinematic phase-space factor Ω(mi, mj; m_piD).
        Returns 0 if the decay is kinematically forbidden.
        """
        mpiD = self.m_piD
        if mpiD < mi + mj:
            return 0.0
        denom = mi ** 2 + mj ** 2
        term1 = (
            1.0 - ((mi ** 2 - mj ** 2) ** 2) / (denom * mpiD ** 2)
            if denom > 0 else 1.0
        )
        arg = (1 - (mi + mj) ** 2 / mpiD ** 2) * (1 - (mi - mj) ** 2 / mpiD ** 2)
        return float(term1 * np.sqrt(max(0.0, arg)))

    # ------------------------------------------------------------------
    # Partial widths
    # ------------------------------------------------------------------

    def gamma_off_diag(self, alpha: int, beta: int, i: int, j: int) -> float:
        """
        Partial width Γ(π^(α,β) → q_i q̄_j)  [GeV].

        Parameters
        ----------
        alpha, beta : dark-flavor indices (0 .. Nf-1).
        i, j        : SM quark indices (0 .. n_quarks-1).
        """
        if alpha >= self.Nf or beta >= self.Nf:
            return 0.0
        Omega = self._omega(self._q_masses[i], self._q_masses[j])
        if Omega == 0.0:
            return 0.0
        pref  = self.Nc * self.fD ** 2 * self.m_piD / (128 * np.pi * self.m_X ** 4)
        kcoef = self._kmat[alpha, i] * np.conj(self._kmat[beta, j])
        return float(pref * abs(kcoef) ** 2 * (self._q_masses[i] ** 2 + self._q_masses[j] ** 2) * Omega)

    def gamma_diag(self, b: int, i: int, j: int) -> float:
        """
        Partial width Γ(π^b → q_i q̄_j)  [GeV]  for generator index b (0-indexed).

        Parameters
        ----------
        b    : generator list index (0 .. Nf^2-2).
        i, j : SM quark indices.
        """
        T   = self._Tlist[b]
        amp = complex(0)
        for alpha in range(self.Nf):
            for beta in range(self.Nf):
                amp += self._kmat[alpha, i] * np.conj(self._kmat[beta, j]) * T[alpha, beta]
        Omega = self._omega(self._q_masses[i], self._q_masses[j])
        if Omega == 0.0:
            return 0.0
        pref = self.Nc * self.fD ** 2 * self.m_piD / (64 * np.pi * self.m_X ** 4)
        return float(pref * abs(amp) ** 2 * (self._q_masses[i] ** 2 + self._q_masses[j] ** 2) * Omega)

    # ------------------------------------------------------------------
    # Per-pion summaries
    # ------------------------------------------------------------------

    def _summarise(self, channels: dict) -> dict:
        total = sum(channels.values())
        br    = {k: v / total for k, v in channels.items()} if total > 0 else {}
        ctau  = HBAR_C_GEV_MM / total if total > 0 else np.inf
        return {"channels": channels, "total": total, "br": br, "ctau_mm": ctau}

    def compute_off_diagonal_pion(self, alpha: int, beta: int) -> dict:
        """
        Widths, branching ratios, and lifetime for π^(α,β).

        Returns
        -------
        dict with keys: channels {(i,j): GeV}, total, br, ctau_mm.
        """
        channels: dict = {}
        for i in range(self._nq):
            for j in range(i, self._nq):
                g = self.gamma_off_diag(alpha, beta, i, j)
                if g > 0:
                    channels[(i, j)] = g
        return self._summarise(channels)

    def compute_diagonal_pion(self, b: int) -> dict:
        """
        Widths, branching ratios, and lifetime for diagonal pion π^b.

        Returns
        -------
        dict with keys: channels {(i,j): GeV}, total, br, ctau_mm.
        """
        channels: dict = {}
        # i == j: self-conjugate final state, counted once
        for i in range(self._nq):
            g = self.gamma_diag(b, i, i)
            if g > 0:
                channels[(i, i)] = g
        # i != j: q_i q̄_j and q̄_i q_j are distinct final states
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
        off_diag: dict = {}
        for a in range(self.Nf):
            for b in range(a + 1, self.Nf):
                r = self.compute_off_diagonal_pion(a, b)
                if r["total"] > 0:
                    off_diag[(a, b)] = r

        diag: dict = {}
        for b in self._diagonal_indices:
            r = self.compute_diagonal_pion(b)
            if r["total"] > 0:
                diag[b] = r

        return {
            "off_diagonal": off_diag,
            "diagonal":     diag,
            "quark_labels": self._q_labels,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def ctau_off_diag_mm(self, alpha: int, beta: int) -> float:
        """c·τ [mm] for off-diagonal pion π^(α,β)."""
        return self.compute_off_diagonal_pion(alpha, beta)["ctau_mm"]

    def ctau_diag_mm(self, b: int) -> float:
        """c·τ [mm] for diagonal pion π^b (generator list index b, 0-indexed)."""
        return self.compute_diagonal_pion(b)["ctau_mm"]

    def channel_label(self, i: int, j: int) -> str:
        """Human-readable LaTeX label for the SM quark decay channel (i, j)."""
        if i == j:
            return rf"${self._q_labels[i]}\bar{{{self._q_labels[i]}}}$"
        return rf"${self._q_labels[i]}\bar{{{self._q_labels[j]}}}$"

    # ------------------------------------------------------------------
    # Base-class interface
    # ------------------------------------------------------------------

    def get_pion_ids(self) -> list:
        """Return pion IDs with non-zero total width (tuples then ints)."""
        results = self.compute_all()
        return list(results["off_diagonal"].keys()) + list(results["diagonal"].keys())

    def ctau_for_pion(self, pion_id) -> float:
        """c·τ [mm] for pion_id — tuple (α,β) for off-diagonal, int b for diagonal."""
        if isinstance(pion_id, tuple):
            return self.ctau_off_diag_mm(*pion_id)
        return self.ctau_diag_mm(pion_id)

    def pion_label(self, pion_id) -> str:
        """LaTeX label for pion_id."""
        if isinstance(pion_id, tuple):
            a, b = pion_id
            return rf"$\pi^{{({a},{b})}}$"
        name = self._diagonal_names.get(pion_id, f"T{pion_id + 1}")
        return rf"$\pi^{{\mathrm{{{name}}}}}$"

    # ------------------------------------------------------------------
    # Summary printer
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted summary of all dark pion properties."""
        total_pions = self.Nf ** 2 - 1
        results     = self.compute_all()
        n_decaying  = len(results["off_diagonal"]) + len(results["diagonal"])

        print("=" * 72)
        print("  Dark Pion T-Channel Model Summary")
        print("=" * 72)
        print(f"  Nf = {self.Nf},  Nd = {self.Nd},  Nc = {self.Nc},  n_active = {self._n_active}")
        print(f"  kappa_mode = {self.kappa_mode},  κ = {self.kappa.real:.6f}")
        print(f"  fD = {self.fD:.2f} GeV,  m_piD = {self.m_piD:.2f} GeV,  "
              f"m_X = {self.m_X:.1f} GeV")
        print()
        print(f"  Total dark pions: {total_pions}  (SU({self.Nf}) adjoint)")
        print(f"    Decaying: {n_decaying}")
        print()

        print("-" * 72)
        print(f"  {'Pion':>8}  {'Type':>12}  {'Gamma [GeV]':>12}  {'ctau [mm]':>12}")
        print("-" * 72)

        pairs = [(a, b) for a in range(self.Nf) for b in range(a + 1, self.Nf)]
        for gen_idx, (a, b) in enumerate(pairs):
            r     = self.compute_off_diagonal_pion(a, b)
            label = get_generator_label(self.Nf, 2 * gen_idx)
            if r["total"] > 0:
                print(f"  {label:>8}  {'off-diagonal':>12}  "
                      f"{r['total']:>12.3e}  {r['ctau_mm']:>12.3e}")
            else:
                print(f"  {label:>8}  {'off-diagonal':>12}  "
                      f"{'stable':>12}  {'inf':>12}")

        for b in self._diagonal_indices:
            r     = self.compute_diagonal_pion(b)
            label = get_generator_label(self.Nf, b)
            if r["total"] > 0:
                print(f"  {label:>8}  {'diagonal':>12}  "
                      f"{r['total']:>12.3e}  {r['ctau_mm']:>12.3e}")
            else:
                print(f"  {label:>8}  {'diagonal':>12}  "
                      f"{'stable':>12}  {'inf':>12}")

        print("=" * 72)

