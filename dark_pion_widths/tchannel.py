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
    mix_diagonal : bool
        When True (default), all diagonal pions share the minimum individual
        ctau.  See base class docstring for details.
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
        mix_diagonal: bool = True,
    ) -> None:
        resolved_quarks = dict(sm_quarks) if sm_quarks is not None else dict(DEFAULT_QUARKS)
        super().__init__(fD=fD, m_piD=m_piD, Nf=Nf, Nd=Nd, Nc=Nc,
                         sm_quarks=resolved_quarks, mix_diagonal=mix_diagonal)

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

    def gamma_flavour(self, alpha: int, beta: int, i: int, j: int) -> float:
        """
        Partial width Γ(π_{αβ} → q_i q̄_j) in the flavour basis  [GeV].

        Applies uniformly to both diagonal (α=β) and off-diagonal (α≠β) pions.
        Formula from arXiv:1803.08080:

            Γ = Nc fD² m_π / (8π m_X⁴)  ×  |κ_{αi} κ*_{βj}|²  ×  (m_i² + m_j²) Ω_{ij}

        Parameters
        ----------
        alpha, beta : dark-flavor indices labelling pion Q̄_α Q_β.
        i, j        : SM quark indices.
        """
        Omega = self._omega(self._q_masses[i], self._q_masses[j])
        if Omega == 0.0:
            return 0.0
        pref  = self.Nc * self.fD ** 2 * self.m_piD / (8 * np.pi * self.m_X ** 4)
        kcoef = self._kmat[alpha, i] * np.conj(self._kmat[beta, j])
        return float(pref * abs(kcoef) ** 2 * (self._q_masses[i] ** 2 + self._q_masses[j] ** 2) * Omega)

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
        Widths, branching ratios, and lifetime for the off-diagonal pion (α,β).

        Partial widths via gamma_off_diag, summed over i ≤ j.

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
        Widths, branching ratios, and lifetime for diagonal pion with
        generator index b (0 … Nf²−2).

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
            off_diagonal : {(alpha, beta): summary_dict}   — same for both bases
            diagonal     : generator basis → {b: summary_dict}
                           flavour basis   → {alpha: summary_dict}
            quark_labels : list[str]

        Diagonal keys are generator-list indices (e.g. 12, 13, 14 for Nf=4).
        """
        off_diag: dict = {}
        for a in range(self.Nf):
            for b in range(a + 1, self.Nf):
                r = self.compute_off_diagonal_pion(a, b)
                if r["total"] > 0:
                    off_diag[(a, b)] = r

        # Compute ALL diagonal pions (including individually-stable ones)
        # so that mixing can propagate the driver's lifetime to every state.
        all_diag = {b: self.compute_diagonal_pion(b) for b in self._diagonal_indices}
        if self.mix_diagonal:
            all_diag = self._apply_diagonal_mixing(all_diag)
        diag = {b: r for b, r in all_diag.items() if r["total"] > 0}

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
        """c·τ [mm] for diagonal pion.

        Generator basis: b is the generator-list index.
        Flavour basis:   b is the dark-quark index α.

        When mix_diagonal=True this returns the common mixed lifetime
        (minimum over all individual diagonal ctau values).
        """
        if self.mix_diagonal:
            results = self.compute_all()
            return results["diagonal"].get(b, {"ctau_mm": np.inf})["ctau_mm"]
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
        """LaTeX label for pion_id (tuple for off-diagonal, int for diagonal)."""
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
        results    = self.compute_all()
        n_decaying = len(results["off_diagonal"]) + len(results["diagonal"])

        print("=" * 72)
        print("  Dark Pion T-Channel Model Summary")
        print("=" * 72)
        print(f"  Nf = {self.Nf},  Nd = {self.Nd},  Nc = {self.Nc},  n_active = {self._n_active}")
        print(f"  kappa_mode = {self.kappa_mode},  κ = {self.kappa.real:.6f}")
        print(f"  fD = {self.fD:.2f} GeV,  m_piD = {self.m_piD:.2f} GeV,  "
              f"m_X = {self.m_X:.1f} GeV")
        print(f"  SU({self.Nf}) adjoint",
              "  (diagonal mixing on)" if self.mix_diagonal else "")
        print()
        print(f"  Total dark pions: {self.Nf ** 2 - 1}  (SU({self.Nf}) adjoint)")
        print(f"    Decaying: {n_decaying}")
        print()

        print("-" * 72)
        print(f"  {'Pion':>8}  {'Type':>12}  {'Gamma [GeV]':>12}  {'ctau [mm]':>12}")
        print("-" * 72)

        pairs = [(a, b) for a in range(self.Nf) for b in range(a + 1, self.Nf)]
        for gen_idx, (a, b) in enumerate(pairs):
            r     = results["off_diagonal"].get((a, b))
            label = get_generator_label(self.Nf, 2 * gen_idx)
            if r is not None and r["total"] > 0:
                print(f"  {label:>8}  {'off-diagonal':>12}  "
                      f"{r['total']:>12.3e}  {r['ctau_mm']:>12.3e}")
            else:
                print(f"  {label:>8}  {'off-diagonal':>12}  "
                      f"{'stable':>12}  {'inf':>12}")

        # Identify the driver pion (pre-mixing minimum ctau) for the footnote
        if self.mix_diagonal and results["diagonal"]:
            unm    = {b: self.compute_diagonal_pion(b) for b in self._diagonal_indices}
            driver = min((b for b in unm if unm[b]["total"] > 0),
                         key=lambda b: unm[b]["ctau_mm"], default=None)
        else:
            driver = None

        for b in self._diagonal_indices:
            r     = results["diagonal"].get(b)
            label = get_generator_label(self.Nf, b)
            if r is not None and r["total"] > 0:
                suffix = " *" if (self.mix_diagonal and b == driver) else ""
                print(f"  {label:>8}  {'diagonal':>12}  "
                      f"{r['total']:>12.3e}  {r['ctau_mm']:>12.3e}{suffix}")
            else:
                print(f"  {label:>8}  {'diagonal':>12}  "
                      f"{'stable':>12}  {'inf':>12}")

        if self.mix_diagonal and driver is not None:
            print(f"\n  * diagonal lifetime set by {get_generator_label(self.Nf, driver)} "
                  f"(fastest-decaying species)")

        print("=" * 72)


# ---------------------------------------------------------------------------
# Flavour-basis subclass
# ---------------------------------------------------------------------------

class DarkPionTChannelFlavourModel(DarkPionTChannelModel):
    """
    T-channel dark pion model in the U(Nf) flavour basis.

    Pions are labelled by their dark quark content Q̄_α Q_β.  The decay width
    formula is (arXiv:1803.08080):

        Γ(π_{αβ} → q_i q̄_j) = Nc fD² m_π / (8π m_X⁴)
                                × |κ_{αi} κ*_{βj}|²
                                × (m_i² + m_j²) Ω_{ij}

    This applies uniformly to both diagonal (α=β) and off-diagonal (α≠β) pions.
    Total states: Nf² (U(Nf), includes the singlet direction).

    Diagonal pions π_{αα} are mixed through dark QCD interactions when
    mix_diagonal=True (default): they all share the lifetime and branching
    ratios of the fastest-decaying individual species.  Pions involving dark
    quarks with no SM coupling (e.g. π_{44} for Nf=4) also inherit the mixed
    lifetime.

    Parameters
    ----------
    mix_diagonal : bool
        When True (default), equalise all diagonal pion lifetimes to the
        minimum.
    """

    def __init__(
        self,
        fD: float = 10.0,
        m_piD: float = 10.0,
        m_X: float = 2000.0,
        kappa: complex = 1.0 + 0j,
        Nf: int = 3,
        kappa_mode: str = "diagonal",
        Nd: int = 3,
        Nc: int = 3,
        sm_quarks: dict | None = None,
        mix_diagonal: bool = True,
    ) -> None:
        super().__init__(
            fD=fD, m_piD=m_piD, m_X=m_X, kappa=kappa,
            Nf=Nf, kappa_mode=kappa_mode, Nd=Nd, Nc=Nc,
            sm_quarks=sm_quarks, mix_diagonal=mix_diagonal,
        )

    # ------------------------------------------------------------------
    # Flavour-basis width overrides
    # ------------------------------------------------------------------

    def compute_off_diagonal_pion(self, alpha: int, beta: int) -> dict:
        """Widths for off-diagonal pion π_{αβ} using the flavour-basis formula."""
        channels: dict = {}
        for i in range(self._nq):
            for j in range(self._nq):
                g = self.gamma_flavour(alpha, beta, i, j)
                if g > 0:
                    channels[(i, j)] = g
        return self._summarise(channels)

    def compute_diagonal_pion(self, alpha: int) -> dict:
        """Widths for diagonal pion π_{αα} using the flavour-basis formula."""
        channels: dict = {}
        for i in range(self._nq):
            for j in range(self._nq):
                g = self.gamma_flavour(alpha, alpha, i, j)
                if g > 0:
                    channels[(i, j)] = g
        return self._summarise(channels)

    def compute_all(self) -> dict:
        """Compute widths for all Nf² flavour-basis pions with optional mixing."""
        off_diag: dict = {}
        for a in range(self.Nf):
            for b in range(a + 1, self.Nf):
                r = self.compute_off_diagonal_pion(a, b)
                if r["total"] > 0:
                    off_diag[(a, b)] = r

        # ALL Nf diagonal pions (α ∈ {0…Nf-1}), including individually-stable
        all_diag = {alpha: self.compute_diagonal_pion(alpha) for alpha in range(self.Nf)}
        if self.mix_diagonal:
            all_diag = self._apply_diagonal_mixing(all_diag)
        diag = {b: r for b, r in all_diag.items() if r["total"] > 0}

        return {"off_diagonal": off_diag, "diagonal": diag, "quark_labels": self._q_labels}

    def ctau_diag_mm(self, alpha: int) -> float:
        """c·τ [mm] for diagonal pion π_{αα} (α is the dark-quark index)."""
        if self.mix_diagonal:
            return self.compute_all()["diagonal"].get(alpha, {"ctau_mm": np.inf})["ctau_mm"]
        return self.compute_diagonal_pion(alpha)["ctau_mm"]

    def pion_label(self, pion_id) -> str:
        """LaTeX label using flavour-basis notation (α,β)."""
        if isinstance(pion_id, tuple):
            a, b = pion_id
            return rf"$\pi^{{({a},{b})}}$"
        return rf"$\pi^{{({pion_id},{pion_id})}}$"

    def print_summary(self) -> None:
        """Print a formatted summary of all dark pion properties."""
        results    = self.compute_all()
        n_decaying = len(results["off_diagonal"]) + len(results["diagonal"])

        print("=" * 72)
        print("  Dark Pion T-Channel Flavour Model Summary")
        print("=" * 72)
        print(f"  Nf = {self.Nf},  Nd = {self.Nd},  Nc = {self.Nc},  n_active = {self._n_active}")
        print(f"  kappa_mode = {self.kappa_mode},  κ = {self.kappa.real:.6f}")
        print(f"  fD = {self.fD:.2f} GeV,  m_piD = {self.m_piD:.2f} GeV,  "
              f"m_X = {self.m_X:.1f} GeV")
        print(f"  U({self.Nf}) flavour basis",
              "  (diagonal mixing on)" if self.mix_diagonal else "")
        print()
        print(f"  Total dark pions: {self.Nf ** 2}  (U({self.Nf}) flavour basis)")
        print(f"    Decaying: {n_decaying}")
        print()

        print("-" * 72)
        print(f"  {'Pion':>8}  {'Type':>12}  {'Gamma [GeV]':>12}  {'ctau [mm]':>12}")
        print("-" * 72)

        for a in range(self.Nf):
            for b in range(a + 1, self.Nf):
                r     = results["off_diagonal"].get((a, b))
                label = f"({a},{b})"
                if r is not None and r["total"] > 0:
                    print(f"  {label:>8}  {'off-diagonal':>12}  "
                          f"{r['total']:>12.3e}  {r['ctau_mm']:>12.3e}")
                else:
                    print(f"  {label:>8}  {'off-diagonal':>12}  "
                          f"{'stable':>12}  {'inf':>12}")

        # Identify driver before mixing for footnote
        if self.mix_diagonal and results["diagonal"]:
            unm    = {alpha: self.compute_diagonal_pion(alpha) for alpha in range(self.Nf)}
            driver = min((a for a in unm if unm[a]["total"] > 0),
                         key=lambda a: unm[a]["ctau_mm"], default=None)
        else:
            driver = None

        for alpha in range(self.Nf):
            r     = results["diagonal"].get(alpha)
            label = f"({alpha},{alpha})"
            if r is not None and r["total"] > 0:
                suffix = " *" if (self.mix_diagonal and alpha == driver) else ""
                print(f"  {label:>8}  {'diagonal':>12}  "
                      f"{r['total']:>12.3e}  {r['ctau_mm']:>12.3e}{suffix}")
            else:
                print(f"  {label:>8}  {'diagonal':>12}  "
                      f"{'stable':>12}  {'inf':>12}")

        if self.mix_diagonal and driver is not None:
            print(f"\n  * diagonal lifetime set by ({driver},{driver}) "
                  f"(fastest-decaying species)")

        print("=" * 72)
