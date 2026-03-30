#!/usr/bin/env python3
import numpy as np

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
Nc = 3
fD = 10.0          # dark decay constant (GeV)
m_piD = 10.0       # dark pion mass (GeV)
m_X = 2000.0       # mediator mass (GeV)
kappa = 1.0 + 0j   # universal coupling

# SM down-type quark masses (GeV)
quarks = {
    "d": 4.7e-3,
    "s": 9.6e-2,
    "b": 4.18
}
quark_labels = list(quarks.keys())
masses = np.array(list(quarks.values()))

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------
hbar_GeVs = 6.582119569e-25       # GeV·s
hbar_c_GeVm = 1.973269788e-16     # GeV·m
c = 2.99792458e8                  # m/s


# ------------------------------------------------------------
# SU(4) generators
# ------------------------------------------------------------
def su4_generators():
    T = []
    T1 = np.array([[0,1/2,0,0],[1/2,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=complex)
    T2 = np.array([[0,-1j/2,0,0],[1j/2,0,0,0],[0,0,0,0],[0,0,0,0]],dtype=complex)
    T3 = np.array([[1/2,0,0,0],[0,-1/2,0,0],[0,0,0,0],[0,0,0,0]],dtype=complex)
    T4 = np.array([[0,0,1/2,0],[0,0,0,0],[1/2,0,0,0],[0,0,0,0]],dtype=complex)
    T5 = np.array([[0,0,-1j/2,0],[0,0,0,0],[1j/2,0,0,0],[0,0,0,0]],dtype=complex)
    T6 = np.array([[0,0,0,0],[0,0,1/2,0],[0,1/2,0,0],[0,0,0,0]],dtype=complex)
    T7 = np.array([[0,0,0,0],[0,0,-1j/2,0],[0,1j/2,0,0],[0,0,0,0]],dtype=complex)
    T8 = (1/(2*np.sqrt(3)))*np.array([[1,0,0,0],[0,1,0,0],[0,0,-2,0],[0,0,0,0]],dtype=complex)
    T9 = np.array([[0,0,0,1/2],[0,0,0,0],[0,0,0,0],[1/2,0,0,0]],dtype=complex)
    T10= np.array([[0,0,0,-1j/2],[0,0,0,0],[0,0,0,0],[1j/2,0,0,0]],dtype=complex)
    T11= np.array([[0,0,0,0],[0,0,0,1/2],[0,0,0,0],[0,1/2,0,0]],dtype=complex)
    T12= np.array([[0,0,0,0],[0,0,0,-1j/2],[0,0,0,0],[0,1j/2,0,0]],dtype=complex)
    T13= np.array([[0,0,0,0],[0,0,0,0],[0,0,0,1/2],[0,0,1/2,0]],dtype=complex)
    T14= np.array([[0,0,0,0],[0,0,0,0],[0,0,0,-1j/2],[0,0,1j/2,0]],dtype=complex)
    T15= (1/(2*np.sqrt(6)))*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-3]],dtype=complex)
    return [T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15]

T_list = su4_generators()
# ------------------------------------------------------------
# κ matrix (α,i): 4×3
# α=0..2 → κ ; α=3 → 0
# ------------------------------------------------------------
kappa_mat = np.zeros((4,3),dtype=complex)
kappa_mat[0:3,:] = kappa

# ------------------------------------------------------------
# Kinematic factor Ω_ij
# ------------------------------------------------------------
def omega_ij(mi, mj, mpiD):
    """Kinematic factor Ω_ij."""
    if mpiD < (mi + mj):
        return 0.0
    term1 = 1 - ((mi**2 - mj**2)**2) / ((mi**2 + mj**2) * mpiD**2)
    sqrt_term = np.sqrt(max(0, (1 - (mi + mj)**2 / mpiD**2) * (1 - (mi - mj)**2 / mpiD**2)))
    return term1 * sqrt_term

# ------------------------------------------------------------
# Off-diagonal pion widths: Γ(π^{(α,β)}_tran → q_i q̄_j)
# ------------------------------------------------------------
def gamma_off_diag(alpha, beta, i, j):
    if alpha < 4 and beta < 4:
        Omega = omega_ij(masses[i], masses[j], m_piD)
        if Omega == 0:
            return 0.0
        pref = Nc * fD**2 * m_piD / (128*np.pi*m_X**4)
        kapp = kappa_mat[alpha,i]*np.conj(kappa_mat[beta,j])
        return pref * abs(kapp)**2 * (masses[i]**2 + masses[j]**2) * Omega
    return 0.0

# ------------------------------------------------------------
# Diagonal pion widths: Γ(π^b_tran → q_i q̄_j)
# ------------------------------------------------------------
def gamma_diag(b, i, j):
    T = T_list[b]
    val_sum = 0.0j
    for alpha in range(4):
        for beta in range(4):
            val_sum += kappa_mat[alpha,i]*np.conj(kappa_mat[beta,j])*T[alpha,beta]
    Omega = omega_ij(masses[i], masses[j], m_piD)
    if Omega == 0:
        return 0.0
    pref = Nc * fD**2 * m_piD / (64*np.pi*m_X**4)
    return pref * abs(val_sum)**2 * (masses[i]**2 + masses[j]**2) * Omega

# ------------------------------------------------------------
# Compute and display results
# ------------------------------------------------------------
print("=== Off-diagonal π^(α,β) → q_i q̄_j widths (GeV) ===")
total_off_diag = {}
for a in range(4):
    for b in range(4):
        if a >= b:
            continue
        total = 0.0
        for i in range(3):
            for j in range(i, 3):
                g = gamma_off_diag(a, b, i, j)
                if g > 0:
                    print(f"Γ(π^({a},{b})→{quark_labels[i]}{quark_labels[j]}) = {g:.3e}")
                    total += g
        if total > 0:
            total_off_diag[(a,b)] = total
            print(f"  → Total Γ(π^({a},{b})) = {total:.3e}\n")

        for i in range(3):
            for j in range(i, 3):
                g = gamma_off_diag(a, b, i, j)
                if g > 0:
                    print(f"Br(π^({a},{b})→{quark_labels[i]}{quark_labels[j]}) = {g/total:.6e}")

print("\n=== Diagonal π^b → q_i q̄_j widths (GeV) ===")
total_diag = {}
for b in [2, 7, 14]:
    total = 0.0
    channels = {}

    # i == j: self-conjugate final state, count once
    for i in range(3):
        g = gamma_diag(b, i, i)
        if g > 0:
            label = f"{quark_labels[i]}bar{quark_labels[i]}"
            channels[label] = g
            total += g

    # i != j: count qi qjbar and qibar qj separately
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            g = gamma_diag(b, i, j)
            if g > 0:
                label = f"{quark_labels[i]}{quark_labels[j]}bar"
                channels[label] = g
                total += g

    if total > 0:
        for label, g in channels.items():
            print(f"Γ(π^{b+1}→{label}) = {g:.3e}")
        print(f"  → Total Γ(π^{b+1}) = {total:.3e}")
        total_diag[b+1] = total
        print(f"Branching ratios:")
        for label, g in channels.items():
            print(f"  Br(π^{b+1}→{label}) = {g/total:.6e}")
        print()


def lifetime_from_width(width):
    if width <= 0:
        return np.inf
    c_tau_m = hbar_c_GeVm / width
    return c_tau_m * 1000.


# ------------------------------------------------------------
# Summary tables
# ------------------------------------------------------------
print("\n=== Summary: Total Off-diagonal Widths (GeV) ===")
for (a,b), val in total_off_diag.items():
    print(f"π^({a},{b}): {val:.3e}")
    print(f"lifetime: {lifetime_from_width(val):.3e} mm")

print("\n=== Summary: Total Diagonal Widths (GeV) ===")
for b, val in total_diag.items():
    print(f"π^{b}: {val:.3e}")
    print(f"lifetime: {lifetime_from_width(val):.3e} mm")
