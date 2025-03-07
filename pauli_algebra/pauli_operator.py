from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Iterable, Union

import numpy as np

from .pauli_string import PauliString


class PauliOperator(dict):
    """A dressed dictionary class for PauliStrings.

    The keys are strings from the corresponding PauliString.
    Thus `po[ps.string]` is the same  as `po[ps]`.
    It also functions as a defaultdict, whenre po[not_in_po] returns
    `PauliString(0.0, not_in_po)`."""

    def __init__(
        self,
        N: Union[int, None] = None,
        strings: Union[Iterable[PauliString], None] = None,
    ):
        """Creates an empty Pauli operator provided N or from a list of PauliStrings."""
        if N is None and strings is None:
            raise ValueError("Provide at least one argument")
        if strings:
            Ns = np.unique([s.N for s in strings])
            assert len(Ns) == 1
            super().__init__((s.string, s) for s in strings)
            if N:
                assert N == Ns[0]
            N = Ns[0]
        else:
            super().__init__()

        self._N = N

    @property
    def N(self) -> int:
        """Number of sites."""
        return self._N

    @property
    def weights(self):
        return np.array([ps.weight for _, ps in self.items()])

    @property
    def strings(self):
        return [ps.string for _, ps in self.items()]

    @property
    def supports(self):
        return [ps.support for _, ps in self.items()]

    @property
    def lists(self):
        return np.array([ps.list for _, ps in self.items()])

    def __missing__(self, pauli_str: Union[str, PauliString]) -> PauliString:
        assert len(pauli_str) == self.N  # We could take pauli_str.N iff PauliString
        if isinstance(pauli_str, str):
            key = pauli_str
            pauli = PauliString(0.0, pauli_str)
        else:
            key = pauli_str.string
            pauli = PauliString(0.0, key)
        self[key] = pauli
        return self[key]

    def __setitem__(
        self, key: Union[str, PauliString], value: Union[Number, PauliString]
    ) -> None:
        if isinstance(key, PauliString):
            self[key.string] = value
            return

        if not isinstance(value, PauliString):
            assert np.isscalar(value)
            value = PauliString(value, key)

        assert self.N == value.N
        super().__setitem__(key, value)

    def __getitem__(self, key: Union[str, PauliString]) -> PauliString:
        if isinstance(key, PauliString):
            return self[key.string]
        return super().__getitem__(key)

    def __add__(self, dct: PauliOperator) -> PauliOperator:
        assert self.N == dct.N
        new_dct = deepcopy(self)
        for key, pauli_str in dct.items():
            new_dct[key] += pauli_str

        return new_dct

    def __mul__(self, w: Number) -> PauliOperator:
        assert np.isscalar(w)
        new = deepcopy(self)
        for key in new.keys():
            new[key] *= w
        return new

    def __rmul__(self, w: Number) -> PauliOperator:
        return self.__mul__(w)

    def trim_thresh(self, threshold: Number) -> None:
        """Removes PauliStrings with a weight below the threshold."""
        to_del = []
        for key, pauli in self.items():
            if np.abs(pauli.weight) <= threshold:
                to_del.append(key)
        for key in to_del:
            del self[key]

    def trim_num(self, remaining: int):
        """Remove all except `remaining` largest strings by absolute value."""
        weights = self.weights
        inds = np.argsort(np.abs(weights))
        strings = self.strings.copy()
        for ind in inds[:-remaining]:
            key = strings[ind]
            del self[key]

    # def to_netket(self):
    #     """Transforms the PauliOperator to a NetKet operator."""
    #     return nk.operator.PauliStrings(self.strings, self.weights)

    def to_sparse(self):
        """Returns a sparse matrix represenattion in the spin basis."""
        res = 0
        for k, s in self.items():
            res += s.to_sparse()
        return res

    def to_dense(self):
        """Returns a dense matrix represenattion in the spin basis."""
        return self.to_sparse().todense()

    def to_Pauli_basis(self):
        """Returns a vector representation in the Pauli basis."""
        res = 0
        for k, s in self.items():
            res += s.to_Pauli_basis()
        return res
