from typing import Optional

import numpy as np
from netket.operator import DiscreteOperator, LocalLiouvillian
from netket.utils.types import DType

# This is a copy of HeisenbergLocalLiouvillian from my netket_extensions
# It is copied so that we do not have an additional dependency
# It just implements NetKet's LocalLiovillian in the Heisenberg picture

class HeisenbergLocalLiouvillian(LocalLiouvillian):
    def __init__(
        self,
        ham: DiscreteOperator,
        jump_ops: list[DiscreteOperator] = [],
        dtype: Optional[DType] = None,
    ):
        super().__init__(
            ham,
            [op.conjugate(concrete=True).transpose(concrete=True) for op in jump_ops],
            dtype,
        )

    def _compute_hnh(self):
        # There is no i here because it's inserted in the kernel
        Hnh = np.asarray(-1.0, dtype=self.dtype) * self.hamiltonian  # -1!!!
        self._max_dissipator_conn_size = 0
        for L in self._jump_ops:
            Hnh = (
                Hnh
                - np.asarray(0.5j, dtype=self.dtype)
                * L
                @ L.conjugate().transpose()  # !!! swaped conjugation
            )
            self._max_dissipator_conn_size += L.max_conn_size**2

        self._Hnh = Hnh.collect().copy(dtype=self.dtype)

        max_conn_size = self._max_dissipator_conn_size + 2 * Hnh.max_conn_size
        self._max_conn_size = max_conn_size
        self._xprime = np.empty((max_conn_size, self.hilbert.size))
        self._xr_prime = np.empty((max_conn_size, self.hilbert.physical.size))
        self._xc_prime = np.empty((max_conn_size, self.hilbert.physical.size))
        self._xrv = self._xprime[:, 0 : self.hilbert.physical.size]
        self._xcv = self._xprime[
            :, self.hilbert.physical.size : 2 * self.hilbert.physical.size
        ]
        self._mels = np.empty(max_conn_size, dtype=self.dtype)

        self._xprime_f = np.empty((max_conn_size, self.hilbert.size))
        self._mels_f = np.empty(max_conn_size, dtype=self.dtype)
