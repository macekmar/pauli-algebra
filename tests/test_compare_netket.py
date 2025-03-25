import pytest

import numpy as np
import pauli_algebra as pa

try:
    import netket as nk
except ImportError:
    pytest.skip("NetKet not available", allow_module_level=True)

N = 6
ops = ["XXIYZY", "XZYIZX"]
weights = [-2.0, 4.5]


@pytest.mark.parametrize("inv_order", [True, False])
def test_netket_todense(inv_order):
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, inverted_ordering=inv_order)
    op_nk = nk.operator.PauliStrings(hilbert, operators=ops, weights=weights)
    el = [
        pa.PauliString(w, s, inverted_ordering=hilbert._inverted_ordering)
        for w, s in zip(weights, ops)
    ]
    op_pa = pa.PauliOperator(strings=el)

    assert np.linalg.norm(op_pa.to_dense() - op_nk.to_dense()) < 1e-15
    assert np.linalg.norm(op_pa.to_sparse().todense() - op_nk.to_dense()) < 1e-15


@pytest.mark.parametrize("inv_order", [True, False])
def test_netket_at(inv_order):
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, inverted_ordering=inv_order)
    op_nk = nk.operator.PauliStrings(hilbert, operators=ops, weights=weights)
    el = [
        pa.PauliString(w, s, inverted_ordering=hilbert._inverted_ordering)
        for w, s in zip(weights, ops)
    ]
    op_pa = pa.PauliOperator(strings=el)

    dh = nk.hilbert.DoubledHilbert(hilbert)
    σ = dh.all_states()

    si = σ[:, :N]
    sj = σ[:, N:]
    assert np.linalg.norm(op_pa.at(si, sj) - op_nk.to_dense().reshape(-1)) < 1e-15

    si = si.reshape(2**N, 2**N, N)
    sj = sj.reshape(2**N, 2**N, N)
    assert np.linalg.norm(op_pa.at(si, sj) - op_nk.to_dense()) < 1e-15


@pytest.mark.parametrize("inv_order", [True, False])
def test_netket_at_sampling(inv_order):
    hilbert = nk.hilbert.Spin(s=1 / 2, N=N, inverted_ordering=inv_order)
    op_nk = nk.operator.PauliStrings(hilbert, operators=ops, weights=weights)
    el = [
        pa.PauliString(w, s, inverted_ordering=hilbert._inverted_ordering)
        for w, s in zip(weights, ops)
    ]
    op_pa = pa.PauliOperator(strings=el)

    si, sj, w = op_pa.sample(100)
    i = hilbert.states_to_numbers(si)
    j = hilbert.states_to_numbers(sj)
    assert np.linalg.norm(op_nk.to_dense()[i,j] - w) < 1e-15
