"""
schannel.py
-----------
S-channel dark pion decay model: general Nf-flavor dark QCD with a massive
Z' vector mediator.

The Z' couples vectorially to dark quarks with flavor-dependent charges q_alpha
and to SM quarks with coupling g_q.  Dark pions are the pseudo-Nambu-Goldstone
bosons of SU(Nf)_L × SU(Nf)_R → SU(Nf)_V chiral symmetry breaking.

Decay mechanisms
----------------
1. Anomaly-mediated two-body:   π_d^a → q q̄
   Via the WZW anomaly vertex (π_d → Z'* Z'*) with both Z' closing into a
   loop on the SM side.  Analogue of π⁰ → e⁺e⁻ in the SM.

2. Anomaly-mediated four-body:  π_d^a → q q̄ q q̄  (parametric estimate)

3. Tree-level (axial Z' only):  π_d → q q̄  via single virtual Z'.
   Only active when a_d ≠ 0.

The key quantity is the anomaly factor A^a = 2 Tr[T^a Q²] where T^a are the
SU(Nf) generators and Q = diag(q_1, …, q_Nf).
"""

from typing import Optional

import numpy as np

from .base import DarkPionModelBase, HBAR_C_GEV_MM
from .generators import build_sun_generators, classify_generator, get_generator_label, get_diagonal_indices

# ---------------------------------------------------------------------------
# Default SM quark masses (GeV) — all six flavors
# ---------------------------------------------------------------------------
DEFAULT_SM_QUARKS: dict[str, float] = {
    "u": 2.16e-3, "d": 4.67e-3, "s": 9.34e-2,
    "c": 1.27,    "b": 4.18,    "t": 172.69,
}


# ---------------------------------------------------------------------------
# Loop function for anomaly-mediated pi -> f fbar
# ---------------------------------------------------------------------------

def loop_function_squared(r: float) -> float:
    """
    |F̃(r)|² where r = m_q² / m_piD².

    Loop function for the anomaly-mediated two-body decay, identical to the
    function appearing in π⁰ → e⁺e⁻ in the SM.

    Returns 0 if r ≥ 1/4 (kinematically forbidden) or r ≤ 0.
    """
    if r >= 0.25 or r <= 0.0:
        return 0.0
    beta     = np.sqrt(1.0 - 4.0 * r)
    log_term = np.log((1.0 + beta) / (1.0 - beta))
    real_part = 0.25 * (log_term ** 2 - np.pi ** 2)
    imag_part = 0.5  * np.pi * log_term
    return real_part ** 2 + imag_part ** 2


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class DarkPionSChannelModel(DarkPionModelBase):
    """
    General Nf-flavor dark QCD model with Z' vector mediator (s-channel).

    Parameters
    ----------
    Nf : int
        Number of dark quark flavors.
    Nd : int
        Number of dark colors (SU(Nd) gauge group).
    dark_charges : list[float]
        U(1)' charges q_alpha for each dark quark flavor.  Length must equal Nf.
    fD : float
        Dark pion decay constant [GeV].
    m_piD : float
        Dark pion mass [GeV].  All Nf²-1 pions are degenerate.
    m_Zp : float
        Z' mass [GeV].
    g_qd : float
        Overall Z' coupling to dark quarks.
    g_q : float
        Overall Z' coupling to SM quarks.
    a_d : float
        Axial coupling component (0 = pure vector, 1 = pure axial).
    Nc : int
        Number of dark colours (default 3).
    sm_quarks : dict or None
        SM quark masses {label: mass_GeV}.  Defaults to all six flavors.
    """

    def __init__(
        self,
        Nf: int = 3,
        Nd: int = 3,
        dark_charges: Optional[list] = None,
        fD: float = 1.0,
        m_piD: float = 10.0,
        m_Zp: float = 1000.0,
        g_qd: float = 0.3,
        g_q: float = 0.3,
        a_d: float = 0.0,
        Nc: int = 3,
        sm_quarks: Optional[dict] = None,
    ) -> None:
        resolved_quarks = dict(sm_quarks) if sm_quarks is not None else dict(DEFAULT_SM_QUARKS)
        super().__init__(fD=fD, m_piD=m_piD, Nf=Nf, Nd=Nd, Nc=Nc, sm_quarks=resolved_quarks)

        self.m_Zp = float(m_Zp)
        self.g_qd = float(g_qd)
        self.g_q  = float(g_q)
        self.a_d  = float(a_d)

        if dark_charges is None:
            dark_charges = [1.0] * self.Nf
        if len(dark_charges) != self.Nf:
            raise ValueError(
                f"dark_charges must have length Nf={self.Nf}, got {len(dark_charges)}"
            )
        self.dark_charges     = np.array(dark_charges, dtype=float)
        self.charge_matrix    = np.diag(self.dark_charges)
        self.charge_matrix_sq = np.diag(self.dark_charges ** 2)

        self._generators   = build_sun_generators(self.Nf)
        self._n_pions      = self.Nf ** 2 - 1
        self._gen_types    = [classify_generator(T) for T in self._generators]
        self._anomaly_factors = self._compute_anomaly_factors()

    # ------------------------------------------------------------------
    # Anomaly factors
    # ------------------------------------------------------------------

    def _compute_anomaly_factors(self) -> np.ndarray:
        """Compute A^a = 2 Tr[T^a Q²] for each generator."""
        A = np.zeros(self._n_pions)
        for a, T in enumerate(self._generators):
            A[a] = 2.0 * np.real(np.trace(T @ self.charge_matrix_sq))
        return A

    def anomaly_factor(self, a: int) -> float:
        """Anomaly factor A^a for pion index a (0-indexed)."""
        return float(self._anomaly_factors[a])

    # ------------------------------------------------------------------
    # Anomaly-mediated two-body: π_d^a → q q̄
    # ------------------------------------------------------------------

    def gamma_anom_2body(self, a: int, q_idx: int) -> float:
        """
        Partial width Γ(π_d^a → q_i q̄_i) [GeV] via anomaly-mediated two-body.

        Parameters
        ----------
        a     : dark pion index (0 .. Nf²-2)
        q_idx : SM quark index
        """
        mq  = self._q_masses[q_idx]
        Aa  = self._anomaly_factors[a]
        if abs(Aa) < 1e-15 or 2 * mq >= self.m_piD:
            return 0.0

        r    = mq ** 2 / self.m_piD ** 2
        Fsq  = loop_function_squared(r)
        beta = np.sqrt(1.0 - 4.0 * r)
        rho  = self.m_piD ** 2 / self.m_Zp ** 2

        numerator = (
            self.Nc * self.Nd ** 2 * self.g_qd ** 4 * Aa ** 2
            * self.g_q ** 4 * mq ** 2 * self.m_piD
        )
        denominator = (
            2.0 * np.pi
            * (4.0 * np.pi ** 2) ** 2
            * (16.0 * np.pi ** 2) ** 2
            * self.fD ** 2
        )
        return float(numerator / denominator * rho ** 2 * Fsq * beta)

    # ------------------------------------------------------------------
    # Anomaly-mediated four-body: π_d^a → q q̄ q q̄  (parametric)
    # ------------------------------------------------------------------

    def gamma_anom_4body(self, a: int) -> float:
        """Parametric estimate of the total four-body width for pion a [GeV]."""
        Aa = self._anomaly_factors[a]
        if abs(Aa) < 1e-15:
            return 0.0
        numerator = (
            self.Nd ** 2 * Aa ** 2
            * self.g_qd ** 4 * self.g_q ** 4
            * self.m_piD ** 13
        )
        denominator = (
            (4.0 * np.pi) ** 9
            * self.fD ** 2
            * self.m_Zp ** 8
        )
        return float(numerator / denominator)

    # ------------------------------------------------------------------
    # Tree-level decay (axial Z' coupling)
    # ------------------------------------------------------------------

    def gamma_tree(self, q_idx: int) -> float:
        """
        Partial width Γ(π_d → q_i q̄_i) [GeV] via tree-level single virtual
        Z' exchange.  Only non-zero when a_d ≠ 0.
        """
        if abs(self.a_d) < 1e-15:
            return 0.0
        mq = self._q_masses[q_idx]
        if 2 * mq >= self.m_piD:
            return 0.0
        beta = np.sqrt(1.0 - 4.0 * mq ** 2 / self.m_piD ** 2)
        numerator = (
            self.Nc * self.g_qd ** 2 * self.a_d ** 2
            * self.g_q ** 2 * self.fD ** 2 * mq ** 2
        )
        denominator = 2.0 * np.pi * self.m_Zp ** 4 * self.m_piD
        return float(numerator / denominator * beta)

    # ------------------------------------------------------------------
    # Per-pion summaries
    # ------------------------------------------------------------------

    def _compute_pion(self, a: int) -> dict:
        """
        Compute all decay widths for dark pion with generator index a.

        Returns a dict with keys:
            generator_index, gen_type, anomaly_factor, anomaly_factor_sq,
            anom_2body, anom_4body, tree_level,
            total_anom_2body, total_tree, total, br, ctau_mm.
        """
        Aa       = self._anomaly_factors[a]
        gen_type = self._gen_types[a]

        anom_2body: dict = {}
        for qi in range(self._nq):
            g = self.gamma_anom_2body(a, qi)
            if g > 0:
                anom_2body[self._q_labels[qi]] = g

        anom_4body = self.gamma_anom_4body(a)

        tree_channels: dict = {}
        if abs(self.a_d) > 1e-15 and gen_type == "diagonal":
            for qi in range(self._nq):
                g = self.gamma_tree(qi)
                if g > 0:
                    tree_channels[self._q_labels[qi]] = g

        total_anom_2body = sum(anom_2body.values())
        total_tree       = sum(tree_channels.values())
        total            = total_anom_2body + anom_4body + total_tree

        all_channels: dict = {}
        for k, v in anom_2body.items():
            all_channels[f"anom_2body_{k}"] = v
        if anom_4body > 0:
            all_channels["anom_4body"] = anom_4body
        for k, v in tree_channels.items():
            all_channels[f"tree_{k}"] = v

        br      = {k: v / total for k, v in all_channels.items()} if total > 0 else {}
        ctau_mm = HBAR_C_GEV_MM / total if total > 0 else np.inf

        return {
            "generator_index":   a,
            "gen_type":          gen_type,
            "anomaly_factor":    Aa,
            "anomaly_factor_sq": Aa ** 2,
            "anom_2body":        anom_2body,
            "anom_4body":        anom_4body,
            "tree_level":        tree_channels,
            "total_anom_2body":  total_anom_2body,
            "total_tree":        total_tree,
            "total":             total,
            "br":                br,
            "ctau_mm":           ctau_mm,
        }

    # ------------------------------------------------------------------
    # Per-pion public interface (mirrors t-channel naming)
    # ------------------------------------------------------------------

    def compute_diagonal_pion(self, b: int) -> dict:
        """Widths, BRs, and lifetime for diagonal pion with generator index b."""
        return self._compute_pion(b)

    def compute_off_diagonal_pion(self, alpha: int, beta: int) -> dict:
        """Widths, BRs, and lifetime for off-diagonal pion π^(α,β)."""
        pairs   = [(j, k) for j in range(self.Nf) for k in range(j + 1, self.Nf)]
        gen_idx = 2 * pairs.index((alpha, beta))
        return self._compute_pion(gen_idx)

    # ------------------------------------------------------------------
    # Full model scan
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Compute widths, BRs, and lifetimes for every dark pion.

        Returns
        -------
        dict with keys:
            off_diagonal : {(alpha, beta): summary_dict}  — decaying pairs only
            diagonal     : {b: summary_dict}              — decaying diagonal only
            quark_labels : list[str]
        """
        off_diag: dict = {}
        pairs = [(j, k) for j in range(self.Nf) for k in range(j + 1, self.Nf)]
        for j, k in pairs:
            r = self.compute_off_diagonal_pion(j, k)
            if r["total"] > 0:
                off_diag[(j, k)] = r

        diag: dict = {}
        for b in get_diagonal_indices(self.Nf):
            r = self.compute_diagonal_pion(b)
            if r["total"] > 0:
                diag[b] = r

        return {
            "off_diagonal": off_diag,
            "diagonal":     diag,
            "quark_labels": self._q_labels,
        }

    # ------------------------------------------------------------------
    # Convenience accessors (mirrors t-channel naming)
    # ------------------------------------------------------------------

    def ctau_diag_mm(self, b: int) -> float:
        """c·τ [mm] for diagonal pion with generator index b."""
        return self.compute_diagonal_pion(b)["ctau_mm"]

    def ctau_off_diag_mm(self, alpha: int, beta: int) -> float:
        """c·τ [mm] for off-diagonal pion π^(α,β)."""
        return self.compute_off_diagonal_pion(alpha, beta)["ctau_mm"]

    # ------------------------------------------------------------------
    # Base-class interface
    # ------------------------------------------------------------------

    def get_pion_ids(self) -> list:
        """Return pion IDs with non-zero total width (tuples then ints)."""
        results = self.compute_all()
        return list(results["off_diagonal"].keys()) + list(results["diagonal"].keys())

    def ctau_for_pion(self, pion_id) -> float:
        """c·τ [mm] — tuple (α,β) for off-diagonal, int b for diagonal."""
        if isinstance(pion_id, tuple):
            return self.ctau_off_diag_mm(*pion_id)
        return self.ctau_diag_mm(int(pion_id))

    def pion_label(self, pion_id) -> str:
        """LaTeX label for pion_id."""
        if isinstance(pion_id, tuple):
            a, b = pion_id
            return rf"$\pi^{{({a},{b})}}$"
        label = get_generator_label(self.Nf, int(pion_id))
        return rf"$\pi^{{\mathrm{{{label}}}}}$"

    # ------------------------------------------------------------------
    # Summary printer
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted summary of all dark pion properties."""
        total_pions = self.Nf ** 2 - 1
        results     = self.compute_all()
        n_decaying  = len(results["off_diagonal"]) + len(results["diagonal"])

        print("=" * 72)
        print("  Dark Pion S-Channel (Z') Model Summary")
        print("=" * 72)
        print(f"  Nf = {self.Nf},  Nd = {self.Nd},  Nc = {self.Nc}")
        print(f"  Dark charges: {self.dark_charges.tolist()}")
        print(f"  fD = {self.fD:.2f} GeV,  m_piD = {self.m_piD:.2f} GeV")
        print(f"  m_Z' = {self.m_Zp:.1f} GeV")
        print(f"  g_qd = {self.g_qd:.3f},  g_q = {self.g_q:.3f}")
        if self.a_d > 0:
            print(f"  Axial coupling a_d = {self.a_d:.3f}")
        print()
        print(f"  Total dark pions: {total_pions}  (SU({self.Nf}) adjoint)")
        print(f"    Decaying: {n_decaying}")
        print()

        print("-" * 72)
        print(f"  {'Pion':>8}  {'Type':>12}  {'A^a':>8}  "
              f"{'Gamma [GeV]':>12}  {'ctau [mm]':>12}")
        print("-" * 72)

        pairs = [(j, k) for j in range(self.Nf) for k in range(j + 1, self.Nf)]
        for gen_idx, (j, k) in enumerate(pairs):
            info  = self.compute_off_diagonal_pion(j, k)
            label = get_generator_label(self.Nf, 2 * gen_idx)
            Aa    = info["anomaly_factor"]
            if info["total"] > 0:
                print(f"  {label:>8}  {'off-diagonal':>12}  {Aa:>8.4f}  "
                      f"{info['total']:>12.3e}  {info['ctau_mm']:>12.3e}")
            else:
                print(f"  {label:>8}  {'off-diagonal':>12}  {Aa:>8.4f}  "
                      f"{'stable':>12}  {'inf':>12}")

        for b in get_diagonal_indices(self.Nf):
            info  = self.compute_diagonal_pion(b)
            label = get_generator_label(self.Nf, b)
            Aa    = info["anomaly_factor"]
            if info["total"] > 0:
                print(f"  {label:>8}  {'diagonal':>12}  {Aa:>8.4f}  "
                      f"{info['total']:>12.3e}  {info['ctau_mm']:>12.3e}")
            else:
                print(f"  {label:>8}  {'diagonal':>12}  {Aa:>8.4f}  "
                      f"{'stable':>12}  {'inf':>12}")

        print("=" * 72)
