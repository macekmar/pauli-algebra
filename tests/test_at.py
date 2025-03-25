import pytest

import numpy as np
import pauli_algebra as pa
from itertools import product

N = 6
ops = ["XXIYZY", "XZYIZX"]
weights = [-2.0, 4.5]

@pytest.mark.parametrize("inv_order", [True, False])
def test_at(inv_order):

    if inv_order:
        spins = [-1,1]
    else:
        spins = [1,-1]
    σ = np.array(list(product(*  ((2*N)*[spins]))))

    el = [
        pa.PauliString(w, s, inverted_ordering=inv_order)
        for w, s in zip(weights, ops)
    ]
    op_pa = pa.PauliOperator(strings=el)

    si = σ[:, :N]
    sj = σ[:, N:]
    assert np.linalg.norm(op_pa.at(si, sj) - op_pa.to_dense().reshape(-1)) < 1e-15

    si = si.reshape(2**N, 2**N, N)
    sj = sj.reshape(2**N, 2**N, N)
    assert np.linalg.norm(op_pa.at(si, sj) - op_pa.to_dense()) < 1e-15

@pytest.mark.parametrize("inv_order", [True, False])
def test_at_sampling(inv_order):
    el = [
        pa.PauliString(w, s, inverted_ordering=inv_order)
        for w, s in zip(weights, ops)
    ]
    op_pa = pa.PauliOperator(strings=el)

    si, sj, w = op_pa.sample(100)
    assert np.linalg.norm(op_pa.at(si,sj) - w) < 1e-15
