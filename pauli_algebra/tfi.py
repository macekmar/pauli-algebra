import numpy as np

from ._rules import rules
from .pauli_algebra import lst_pauli


def _gen_X_part(N, g):
    """Generates i·(-∑σx).
    Minus sign, so that it is consistent with NetKet definition of Ising."""
    sop = {p: rules["com"]["X", p] for p in lst_pauli(1)}
    weights = -1j * g * np.ones(N, dtype=np.complex128)
    sites = [[i] for i in range(N)]
    return sop, weights, sites


def _gen_ZZ_part(N, J, pbc=True):
    """Generates i·(∑σz·σz)."""
    sop = {p: rules["com"]["ZZ", p] for p in lst_pauli(2)}
    weights = 1j * J * np.ones(N - 1 + pbc, dtype=np.complex128)
    sites = [[i, i + 1] for i in range(N - 1)]
    if pbc:
        sites.append([N - 1, 0])
    return sop, weights, sites


def _gen_dissipation(N):
    """Generates action of L⁺*L - ½{L⁺L,*}, where L=σ-."""
    sop = {p: rules["dis"]["M", p] for p in lst_pauli(1)}
    weights = np.ones(N, dtype=np.complex128)
    sites = [[i] for i in range(N)]
    return sop, weights, sites


def gen_Lindblad_TFI(N, g, J, pbc=True, dissipation=True):
    """Generates superoperator TFI Hamiltonian with dissipation."""
    operations = {}
    operations["X"] = _gen_X_part(N, g)
    operations["ZZ"] = _gen_ZZ_part(N, J, pbc)
    if dissipation:
        operations["dis"] = _gen_dissipation(N)

    return operations
