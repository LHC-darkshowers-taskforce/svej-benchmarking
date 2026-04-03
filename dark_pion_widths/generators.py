"""
generators.py
-------------
SU(N) generator construction shared by t-channel and s-channel dark pion models.

Generators are built in the fundamental representation, normalised as
Tr(T^a T^b) = delta^{ab} / 2.

Ordering convention
-------------------
Off-diagonal pairs first (for each j < k: symmetric real, then antisymmetric
imaginary), then diagonal generators in order l = 1 .. n-1.

For SU(4) this gives 15 generators at indices 0-14:
  0-11: off-diagonal pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) × 2
  12  : T3   diag([1,-1, 0, 0]) / 2
  13  : T8   diag([1, 1,-2, 0]) / (2√3)
  14  : T15  diag([1, 1, 1,-3]) / (2√6)

Note: T15 is at index 14 in both the old hardcoded SU(4) list and the new
ordering, so existing code using ctau_diag_mm(14) is unaffected.
T3 and T8 move from indices 2 and 7 to 12 and 13 respectively.
"""

import numpy as np


def build_sun_generators(n: int) -> list[np.ndarray]:
    """
    Build the n^2 - 1 generators of SU(n) in the fundamental representation,
    normalised as Tr(T^a T^b) = delta^{ab} / 2.

    Returns a list of n×n complex matrices in the ordering described in the
    module docstring.
    """
    generators = []

    # Off-diagonal generators: for each pair j < k emit symmetric then antisymmetric
    for j in range(n):
        for k in range(j + 1, n):
            t_sym = np.zeros((n, n), dtype=complex)
            t_sym[j, k] = 0.5
            t_sym[k, j] = 0.5
            generators.append(t_sym)

            t_asym = np.zeros((n, n), dtype=complex)
            t_asym[j, k] = -0.5j
            t_asym[k, j] =  0.5j
            generators.append(t_asym)

    # Diagonal generators: l = 1 .. n-1
    for l in range(1, n):
        t_diag = np.zeros((n, n), dtype=complex)
        norm = 1.0 / np.sqrt(2.0 * l * (l + 1))
        for j in range(l):
            t_diag[j, j] = norm
        t_diag[l, l] = -l * norm
        generators.append(t_diag)

    return generators


def classify_generator(T: np.ndarray) -> str:
    """Return 'diagonal' or 'off-diagonal' for generator matrix T."""
    off_diag_norm = np.sum(np.abs(T - np.diag(np.diag(T))))
    return "diagonal" if off_diag_norm < 1e-12 else "off-diagonal"


def get_diagonal_indices(Nf: int) -> list[int]:
    """
    Return the 0-indexed positions of the diagonal generators for SU(Nf).

    With build_sun_generators ordering, diagonal generators come after all
    off-diagonal pairs.  There are Nf*(Nf-1) off-diagonal generators, so the
    diagonal ones occupy positions Nf*(Nf-1) through Nf^2 - 2 inclusive.

    Examples
    --------
    >>> get_diagonal_indices(3)
    [6, 7]
    >>> get_diagonal_indices(4)
    [12, 13, 14]
    """
    n_off = Nf * (Nf - 1)
    return list(range(n_off, Nf ** 2 - 1))


def get_generator_label(Nf: int, a: int) -> str:
    """
    Return a human-readable label for generator index *a* of SU(Nf).

    Off-diagonal generators are labelled by their (j,k) dark-flavor pair.
    Both generators of a pair (symmetric and antisymmetric) share the same label.

    Diagonal generators are labelled by their conventional name (T3, T8, T15).

    Examples
    --------
    >>> get_generator_label(3, 0)
    '(0,1)'
    >>> get_generator_label(3, 6)
    'T3'
    """
    n_off = Nf * (Nf - 1)
    if a < n_off:
        pairs = [(j, k) for j in range(Nf) for k in range(j + 1, Nf)]
        j, k = pairs[a // 2]
        return f"({j},{k})"
    diag_names = get_diagonal_names(Nf)
    return diag_names.get(a, f"T{a + 1}")


def get_diagonal_names(Nf: int) -> dict[int, str]:
    """
    Map generator list index → conventional physics name for diagonal pions.

    Uses the standard SU(3) and SU(4) names (T3, T8, T15) for Nf = 3 and 4.
    Falls back to 'T{index+1}' (1-indexed) for other values of Nf.

    Examples
    --------
    >>> get_diagonal_names(3)
    {6: 'T3', 7: 'T8'}
    >>> get_diagonal_names(4)
    {12: 'T3', 13: 'T8', 14: 'T15'}
    """
    diag_indices = get_diagonal_indices(Nf)

    _standard: dict[int, list[str]] = {
        2: ["T3"],
        3: ["T3", "T8"],
        4: ["T3", "T8", "T15"],
    }

    names: dict[int, str] = {}
    for l, idx in enumerate(diag_indices):
        if Nf in _standard and l < len(_standard[Nf]):
            names[idx] = _standard[Nf][l]
        else:
            names[idx] = f"T{idx + 1}"
    return names
