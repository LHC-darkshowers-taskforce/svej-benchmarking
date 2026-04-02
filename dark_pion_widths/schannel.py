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
from .generators import build_sun_generators, classify_generator

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
HBAR_GEV_S = 6.582119569e-25      # ℏ in GeV·s

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
    m_etap : float or None
        Dark eta' mass [GeV].  If None, estimated as 4π·fD.
    f_etap : float or None
        Dark eta' decay constant [GeV].  If None, set equal to fD.
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
        m_etap: Optional[float] = None,
        f_etap: Optional[float] = None,
        a_d: float = 0.0,
        Nc: int = 3,
        sm_quarks: Optional[dict] = None,
    ) -> None:
        super().__init__(fD=fD, m_piD=m_piD, Nc=Nc)

        self.Nf   = int(Nf)
        self.Nd   = int(Nd)
        self.m_Zp = float(m_Zp)
        self.g_qd = float(g_qd)
        self.g_q  = float(g_q)
        self.a_d  = float(a_d)

        self.m_etap = float(m_etap) if m_etap is not None else 4.0 * np.pi * self.fD
        self.f_etap = float(f_etap) if f_etap is not None else self.fD

        if dark_charges is None:
            dark_charges = [1.0] * Nf
        if len(dark_charges) != Nf:
            raise ValueError(
                f"dark_charges must have length Nf={Nf}, got {len(dark_charges)}"
            )
        self.dark_charges    = np.array(dark_charges, dtype=float)
        self.charge_matrix   = np.diag(self.dark_charges)
        self.charge_matrix_sq = np.diag(self.dark_charges ** 2)

        self.sm_quarks  = dict(sm_quarks) if sm_quarks is not None else dict(DEFAULT_SM_QUARKS)
        self._q_labels  = list(self.sm_quarks.keys())
        self._q_masses  = np.array(list(self.sm_quarks.values()), dtype=float)
        self._nq        = len(self._q_masses)

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

    def eta_prime_anomaly_factor(self) -> float:
        """Anomaly factor for the dark eta' (flavor singlet)."""
        T0 = np.eye(self.Nf, dtype=complex) / np.sqrt(2.0 * self.Nf)
        return float(2.0 * np.real(np.trace(T0 @ self.charge_matrix_sq)))

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

    def gamma_anom_2body_etap(self, q_idx: int) -> float:
        """Partial width Γ(η'_d → q_i q̄_i) [GeV] via anomaly-mediated two-body."""
        mq = self._q_masses[q_idx]
        Ae = self.eta_prime_anomaly_factor()
        if abs(Ae) < 1e-15 or 2 * mq >= self.m_etap:
            return 0.0

        r    = mq ** 2 / self.m_etap ** 2
        Fsq  = loop_function_squared(r)
        beta = np.sqrt(1.0 - 4.0 * r)
        rho  = self.m_etap ** 2 / self.m_Zp ** 2

        numerator = (
            self.Nc * self.Nd ** 2 * self.g_qd ** 4 * Ae ** 2
            * self.g_q ** 4 * mq ** 2 * self.m_etap
        )
        denominator = (
            2.0 * np.pi
            * (4.0 * np.pi ** 2) ** 2
            * (16.0 * np.pi ** 2) ** 2
            * self.f_etap ** 2
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

    def gamma_anom_4body_etap(self) -> float:
        """Parametric estimate of the total four-body width for the dark eta' [GeV]."""
        Ae = self.eta_prime_anomaly_factor()
        if abs(Ae) < 1e-15:
            return 0.0
        numerator = (
            self.Nd ** 2 * Ae ** 2
            * self.g_qd ** 4 * self.g_q ** 4
            * self.m_etap ** 13
        )
        denominator = (
            (4.0 * np.pi) ** 9
            * self.f_etap ** 2
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

    def _summarise(self, channels: dict, total_override: Optional[float] = None) -> dict:
        total   = total_override if total_override is not None else sum(channels.values())
        br      = {k: v / total for k, v in channels.items()} if total > 0 else {}
        ctau_mm = HBAR_C_GEV_MM / total if total > 0 else np.inf
        tau_s   = HBAR_GEV_S   / total if total > 0 else np.inf
        return {"channels": channels, "total": total, "br": br,
                "ctau_mm": ctau_mm, "tau_s": tau_s}

    def compute_pion(self, a: int) -> dict:
        """
        Compute all decay widths for dark pion π_d^a.

        Returns a dict with keys:
            generator_index, gen_type, anomaly_factor, anomaly_factor_sq,
            anom_2body, anom_4body, tree_level,
            total_anom_2body, total_tree, total, br, ctau_mm, tau_s.
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
        tau_s   = HBAR_GEV_S   / total if total > 0 else np.inf

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
            "tau_s":             tau_s,
        }

    def compute_eta_prime(self) -> dict:
        """Compute decay widths for the dark eta'."""
        Ae = self.eta_prime_anomaly_factor()

        anom_2body: dict = {}
        for qi in range(self._nq):
            g = self.gamma_anom_2body_etap(qi)
            if g > 0:
                anom_2body[self._q_labels[qi]] = g

        anom_4body = self.gamma_anom_4body_etap()
        total      = sum(anom_2body.values()) + anom_4body

        all_channels: dict = {}
        for k, v in anom_2body.items():
            all_channels[f"anom_2body_{k}"] = v
        if anom_4body > 0:
            all_channels["anom_4body"] = anom_4body

        br      = {k: v / total for k, v in all_channels.items()} if total > 0 else {}
        ctau_mm = HBAR_C_GEV_MM / total if total > 0 else np.inf
        tau_s   = HBAR_GEV_S   / total if total > 0 else np.inf

        return {
            "name":            "eta_prime",
            "anomaly_factor":  Ae,
            "anomaly_factor_sq": Ae ** 2,
            "mass":            self.m_etap,
            "f_decay":         self.f_etap,
            "anom_2body":      anom_2body,
            "anom_4body":      anom_4body,
            "total":           total,
            "br":              br,
            "ctau_mm":         ctau_mm,
            "tau_s":           tau_s,
        }

    # ------------------------------------------------------------------
    # Full model scan
    # ------------------------------------------------------------------

    def compute_all(self) -> dict:
        """
        Compute widths, BRs, and lifetimes for every dark pion and the eta'.

        Returns
        -------
        dict with keys: pions, eta_prime, model_params, classification.
        """
        pions: dict = {}
        n_decaying = n_stable_diag = n_stable_offdiag = 0

        for a in range(self._n_pions):
            info  = self.compute_pion(a)
            pions[a] = info
            if info["total"] > 0:
                n_decaying += 1
            elif info["gen_type"] == "diagonal":
                n_stable_diag += 1
            else:
                n_stable_offdiag += 1

        eta = self.compute_eta_prime()

        classification = {
            "total_pions":         self._n_pions,
            "n_diagonal":          sum(1 for t in self._gen_types if t == "diagonal"),
            "n_off_diagonal":      sum(1 for t in self._gen_types if t == "off-diagonal"),
            "n_decaying":          n_decaying,
            "n_stable_diagonal":   n_stable_diag,
            "n_stable_off_diagonal": n_stable_offdiag,
        }

        model_params = {
            "Nf":          self.Nf,
            "Nd":          self.Nd,
            "dark_charges": self.dark_charges.tolist(),
            "fD":          self.fD,
            "m_piD":       self.m_piD,
            "m_Zp":        self.m_Zp,
            "g_qd":        self.g_qd,
            "g_q":         self.g_q,
            "a_d":         self.a_d,
            "m_etap":      self.m_etap,
            "Nc":          self.Nc,
        }

        return {
            "pions":          pions,
            "eta_prime":      eta,
            "model_params":   model_params,
            "classification": classification,
        }

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def ctau_mm(self, a: int) -> float:
        """c·τ [mm] for pion index a."""
        return self.compute_pion(a)["ctau_mm"]

    def tau_s(self, a: int) -> float:
        """Lifetime [s] for pion index a."""
        return self.compute_pion(a)["tau_s"]

    # ------------------------------------------------------------------
    # Base-class interface
    # ------------------------------------------------------------------

    def get_pion_ids(self) -> list:
        """Return pion indices with non-zero total width."""
        results = self.compute_all()
        return [a for a, info in results["pions"].items() if info["total"] > 0]

    def ctau_for_pion(self, pion_id) -> float:
        """c·τ [mm] for pion_id (generator index a)."""
        return self.ctau_mm(int(pion_id))

    def pion_label(self, pion_id) -> str:
        """LaTeX label for pion_id based on its generator type and anomaly factor."""
        a        = int(pion_id)
        gen_type = self._gen_types[a]
        if gen_type == "diagonal":
            return rf"$\pi_D^{{({a})}}$"
        return rf"$\pi_D^{{({a})}}$"

    # ------------------------------------------------------------------
    # Summary printer
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        """Print a formatted summary of all dark pion properties."""
        results = self.compute_all()
        params  = results["model_params"]
        clf     = results["classification"]

        print("=" * 72)
        print("  Dark Pion S-Channel (Z') Model Summary")
        print("=" * 72)
        print(f"  Nf = {params['Nf']},  Nd = {params['Nd']},  Nc = {params['Nc']}")
        print(f"  Dark charges: {params['dark_charges']}")
        print(f"  fD = {params['fD']:.2f} GeV,  m_piD = {params['m_piD']:.2f} GeV")
        print(f"  m_Z' = {params['m_Zp']:.1f} GeV")
        print(f"  g_qd = {params['g_qd']:.3f},  g_q = {params['g_q']:.3f}")
        if params["a_d"] > 0:
            print(f"  Axial coupling a_d = {params['a_d']:.3f}")
        print(f"  m_eta' = {params['m_etap']:.2f} GeV")
        print()
        print(f"  Total dark pions: {clf['total_pions']}")
        print(f"    Decaying: {clf['n_decaying']}")
        print()

        print("-" * 72)
        print(f"  {'Pion':>6}  {'Type':>12}  {'A^a':>8}  "
              f"{'Gamma [GeV]':>12}  {'ctau [mm]':>12}")
        print("-" * 72)

        for a in range(clf["total_pions"]):
            info     = results["pions"][a]
            Aa       = info["anomaly_factor"]
            total    = info["total"]
            ctau     = info["ctau_mm"]
            gen_type = info["gen_type"]
            if total > 0:
                print(f"  {a:>6}  {gen_type:>12}  {Aa:>8.4f}  "
                      f"{total:>12.3e}  {ctau:>12.3e}")
            else:
                print(f"  {a:>6}  {gen_type:>12}  {Aa:>8.4f}  "
                      f"{'stable':>12}  {'inf':>12}")

        eta = results["eta_prime"]
        print()
        print("-" * 72)
        print(f"  Dark eta':  A = {eta['anomaly_factor']:.4f},  "
              f"m = {eta['mass']:.2f} GeV")
        if eta["total"] > 0:
            print(f"    Gamma = {eta['total']:.3e} GeV,  "
                  f"ctau = {eta['ctau_mm']:.3e} mm")
        else:
            print(f"    Stable (zero anomaly coupling)")
        print("=" * 72)
