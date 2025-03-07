from __future__ import annotations

from functools import reduce
from numbers import Number
from typing import Iterable, List, Union

import numpy as np
import scipy as sc


class PauliString:
    """Dressed Pauli string with common manipulations."""

    def __init__(
        self, weight: Number, string: Union[List[str], str], N: Union[int, None] = None
    ):
        """Define weight of the Pauli string and store the string as a list."""
        self._w = weight

        if isinstance(string, str):
            string = list(string)
        self._list = np.array(string, dtype=str)
        self._string = "".join(string)

        if N:
            assert len(self._list) == N
        else:
            N = len(self._list)
        self._N = N

    @property
    def string(self):
        """Pauli string"""
        return self._string

    @property
    def list(self):
        return self._list.copy()

    @property
    def weight(self):
        """Weight factor in front of the PauliString."""
        return self._w

    @property
    def acting_on(self):
        """Sites on which the Pauli string is acting on."""
        return tuple(i for i in range(self.N) if self.list[i] != "I")

    @property
    def support(self):
        """Number of non-identity Paulis."""
        return len(self.acting_on)

    @property
    def N(self):
        """Number of sites."""
        return self._N

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.weight:4.3f}·{self.string:s}"

    def __getitem__(self, ind: Union[Iterable, int]):
        if isinstance(ind, int):
            string = [self._string[ind]]
        if isinstance(ind, Iterable):
            string = []
            for i in ind:
                assert isinstance(i, int)
                string.append(self._string[i])

        return PauliString(self.weight, string)

    def __setitem__(self, ind: int, char: str):
        assert len(char) == 1
        assert char in ["I", "X", "Y", "Z"]
        self._list[ind] = char
        self._string = "".join(self._list)

    def __len__(self):
        return self._N

    def __add__(self, pauli_str: PauliString):
        assert self.string == pauli_str.string
        return PauliString(self.weight + pauli_str.weight, self.string, self._N)

    def __imul__(self, w: Number):
        assert np.isscalar(w)
        self._w *= w
        return self

    # def to_netket(self):
    #     """Transforms the PauliString to a NetKet operator."""
    #     return self.weight * nk.operator.PauliStrings(self.string)

    def to_sparse(self):
        """Returns a sparse matrix represenattion in the spin basis."""
        _s_to_sm = {
            "I": sc.sparse.csr_array([[1, 0], [0, 1]], dtype=np.complex128),
            "X": sc.sparse.csr_array([[0, 1], [1, 0]], dtype=np.complex128),
            "Y": sc.sparse.csr_array([[0, -1j], [1j, 0]], dtype=np.complex128),
            "Z": sc.sparse.csr_array([[1, 0], [0, -1]], dtype=np.complex128),
        }

        def _kp(*args):
            return reduce(lambda x, y: sc.sparse.kron(x, y), args, sc.sparse.eye(1))

        mats = [_s_to_sm[s] for s in self.string]
        return self.weight * _kp(*mats)

    def to_dense(self):
        """Returns a dense matrix represenattion in the spin basis."""
        return self.to_sparse().todense()

    def to_Pauli_basis(self):
        """Returns a vector representation in the Pauli basis."""
        _s_to_v = {
            "I": np.array([1, 0, 0, 0], dtype=np.complex128),
            "X": np.array([0, 1, 0, 0], dtype=np.complex128),
            "Y": np.array([0, 0, 1, 0], dtype=np.complex128),
            "Z": np.array([0, 0, 0, 1], dtype=np.complex128),
        }

        def _kp(*args):
            return reduce(lambda x, y: np.kron(x, y), args, 1)

        vecs = [_s_to_v[s] for s in self.string]
        return self.weight * _kp(*vecs)
