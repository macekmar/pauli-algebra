from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Iterable, Union, Callable, List
from functools import partial

import jax
import numpy as np
from jax import numpy as jnp

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
        pauli_strings: Union[Iterable[PauliString], None] = None,
        strings: Union[Iterable[str], None] = None,
        weights: Union[Iterable[float], None] = None,
        seed: int = None,
        inverted_ordering: Union[bool, None] = None,
    ):
        """Creates an empty Pauli operator provided N or from a list of PauliStrings."""
        if N is None and strings is None and pauli_strings is None:
            raise TypeError("Provide at least one argument")
        if strings is not None and weights is None:
            raise TypeError("Beside strings you have to provide corresponding weights")

        # Providing strings and weights, prepare PauliStrings
        if strings is not None and weights is not None:
            if pauli_strings is not None:
                raise TypeError("Provide either strings with weights or PauliStrings")
            if len(strings) != len(weights):
                raise ValueError("Lenght of weights and strings should be the same")

            pauli_strings = []
            Ns = []
            for s, w in zip(strings, weights):
                Ns.append(len(s))
                pauli_strings.append(PauliString(w, s))
            Ns = np.unique(Ns)
            inv_ord = [False]

        if pauli_strings is not None:
            for s in pauli_strings:
                if isinstance(s, PauliString) is False:
                    raise TypeError("Provided pauli_strings should be PauliStrings")
            Ns = np.unique([s.N for s in pauli_strings])
            inv_ord = set([s._inverted_ordering for s in pauli_strings])
            assert len(inv_ord) == 1  # They should be all the same

        if pauli_strings:
            if len(Ns) > 1:
                raise ValueError("Provided strings should have the same Hilbert size.")
            if inverted_ordering is not None:
                assert inverted_ordering == inv_ord.pop()
            else:
                inverted_ordering = inv_ord.pop()
            super().__init__((s.string, s) for s in pauli_strings)
            if N:
                assert N == Ns[0]
            N = Ns[0]
        else:
            super().__init__()

        self._N = N

        self._rng = None
        self._seed = seed
        if inverted_ordering is None:
            inverted_ordering = False
        self._inverted_ordering = inverted_ordering

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

    @property
    def rng(self):
        if self._rng is None:
            if self._seed is None:
                self._seed = int(np.random.default_rng().integers(0, 1 << 32))
            self._rng = jax.random.key(self._seed)
        return self._rng

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

    def sample(self, shape: int | List[int], pdf: Callable = None):
        new_key, key = jax.random.split(self.rng)

        if np.isscalar(shape):
            shape = [shape]

        if pdf is None:
            pdf = lambda w: np.abs(w) / np.sum(np.abs(w))  # noqa: E731

        # inds - which Pauli string of all we are taking
        inds = jax.random.choice(
            key, jnp.arange(len(self.weights)), shape=shape, p=pdf(self.weights)
        )
        # x - which element of Pauli matrix we are taking
        shp = list(shape) + [self.N]
        x = jax.random.choice(
            key, jnp.array([0, 1], dtype=jnp.int8), shape=shp, replace=True
        )

        # Map local indices and weights to a global one
        from .pauli_string import _string_to_number

        num_strings = np.array([_string_to_number(st) for st in self.strings])
        σi, σj, w = _local_to_global(
            num_strings[inds], x, self.weights[inds], self._inverted_ordering
        )

        self._rng = new_key

        return σi, σj, w

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

    def at(self, si, sj):
        el = 0
        for k, s in self.items():
            el += s.at(si, sj)
        return el


@partial(jax.jit, static_argnums=3)
def _local_to_global(string, x, ps_weights, inverted_ordering):
    # inds and weigts store for each pauli matrix the indices and weights of two
    # non-zero elements. Say X: X_01 = 1, X_10 = 1
    inds = jnp.array(
        [
            [[0, 0], [1, 1]],
            [[0, 1], [1, 0]],
            [[0, 1], [1, 0]],
            [[0, 0], [1, 1]],
        ],
        dtype=jnp.int8,
    )
    el_weights = jnp.array(
        [
            [1, 1],
            [1, 1],
            [-1j, 1j],
            [1, -1],
        ]
    )

    if inverted_ordering:
        σ = 2 * inds[string, x] - 1
    else:
        σ = 1 - 2 * inds[string, x]
    return σ[..., 0], σ[..., 1], ps_weights * np.prod(el_weights[string, x], -1)
