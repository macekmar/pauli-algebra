import numpy as np

import pauli_algebra as pa


def build_pauli_operator(seed=1234):
    N = 4

    strings = [
        "IXZZ",
        "ZYIZ",  # Same indices as above but imag
        "ZXII",  # Same indices as above but real
        "YZXX",
        "XZYZ",
    ]

    weights = np.array(
        [
            0.08733297,
            0.91745171,
            -0.51098743,
            -0.33105704,
            -1.07963846,
        ]
    )

    ps = pa.PauliOperator(N, seed=seed)
    for i in range(len(strings)):
        ps[strings[i]] = pa.PauliString(weights[i], strings[i])
    return ps


def test_shape():
    ps = build_pauli_operator()
    n_samples = 100000
    σi, σj, w = ps.sample(n_samples)
    assert σi.shape == σj.shape
    assert σi.shape == (n_samples, ps.N)
    assert w.shape == (n_samples,)


def test_cover_identical_ele():
    # This very much depends on seeed
    ps = build_pauli_operator(seed=1234)
    n_samples = 100000
    σi, σj, _ = ps.sample(n_samples)
    σ = np.array([σi.T, σj.T]).T
    σ = (1 - σ) // 2

    uv = (2 ** (np.arange(ps.N)[::-1]) @ σ).real.astype(int)
    u, v = np.unique(uv, axis=0).T

    i, j = np.where(ps.to_dense() != 0)

    assert np.all(i == u)
    assert np.all(j == v)


def test_matrix_is_close():
    # This very much depends on seeed
    ps = build_pauli_operator(seed=1234)
    n_samples = 200000
    σi, σj, w = ps.sample(n_samples)
    σ = np.array([σi.T, σj.T]).T
    σ = (1 - σ) // 2

    uv = (2 ** (np.arange(ps.N)[::-1]) @ σ).real.astype(int)
    # We shouldn't do unique on uv: different Pauli strings act on the same indices, e.g. XZ vs YI
    m = np.zeros((2**ps.N, 2**ps.N), dtype=np.complex128)
    d = np.hstack((uv.astype(np.complex64), w[:, np.newaxis]))
    d_set, counts = np.unique(d, axis=0, return_counts=True)
    uv_set = d_set[:, :2].real.astype(int)
    weights = d_set[:, -1]
    for i in range(d_set.shape[0]):
        sign = np.sign(weights[i].real) + 1j * np.sign(weights[i].imag)
        m[uv_set[i, 0], uv_set[i, 1]] += counts[i] * sign

    # Reweight, obviously 1/num_samples, 2^N because this is how many different elements we have, * sum weight- comes from transforming to pdf
    m = (
        m / σ.shape[0] * 2**ps.N * np.sum(np.abs(ps.weights))
    )  # -- should now be close to ps.to_dense()

    assert np.linalg.norm(m - ps.to_dense()) / np.linalg.norm(ps.to_dense()) < 0.015
