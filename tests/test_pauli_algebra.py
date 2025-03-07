from copy import deepcopy

import netket as nk
import numpy as np
from netket.operator.spin import sigmam

import pauli_algebra as pa
from _heis_liovillian import HeisenbergLocalLiouvillian

# Parameters
N = 6
V = 4.7
g = 2.3
dt = 1e-2

# Set up machinery
hilbert = nk.hilbert.Spin(s=1 / 2, N=N)
dh = nk.hilbert.DoubledHilbert(hilbert)
graph = nk.graph.Chain(length=N, pbc=True)

H_final = nk.operator.Ising(hilbert, graph, h=-g, J=V)
jump_ops = [sigmam(hilbert, i) for i in range(N)]
HeisL_final = HeisenbergLocalLiouvillian(H_final, jump_ops)
L_mat = HeisL_final.to_sparse()
L_str = pa.gen_Lindblad_TFI(N, -g, V, dissipation=True)

# Operator
strings = [
    "YIZYIZ",
    "IIXXXI",
    "IZXYIZ",
    "XIZYYZ",
    "XYYXYX",
    "YXZYXX",
]

weights = np.array(
    [
        0.42294081,
        0.66695483,
        0.08733297,
        -0.91745171,
        0.51098743,
        -0.33105704,
    ]
)

ps_nk = nk.operator.PauliStrings(strings, weights)
ps_ext = pa.PauliOperator(N)
for i in range(len(strings)):
    ps_ext[strings[i]] = pa.PauliString(weights[i], strings[i])


def test_time_evolution():
    N_iter = 3

    op_nk_vec = ps_nk.to_dense().reshape(-1)
    for i in range(N_iter):
        op_nk_vec = op_nk_vec + dt * L_mat @ op_nk_vec

    op_str = deepcopy(ps_ext)
    for i in range(N_iter):
        op_str = op_str + dt * pa.apply_superoperator(L_str, op_str)
        op_str.trim_thresh(0)
    op_str_vec = op_str.to_dense().reshape(-1)

    assert np.linalg.norm(op_nk_vec - op_str_vec) < 1e-15 * N_iter * len(strings)


def test_strings_to_numbers_to_strings():
    n_in = np.random.randint(0, 4, 10)
    n_out = pa.pauli_algebra.string_to_numbers(
        pa.pauli_algebra.numbers_to_string(n_in)
    )
    assert np.all(n_in == n_out)
