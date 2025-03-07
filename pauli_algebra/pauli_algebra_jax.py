from collections import defaultdict
from functools import partial
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .pauli_algebra import lst_pauli, numbers_to_string, string_to_numbers
from .pauli_operator import PauliOperator
from .pauli_string import PauliString


def unpack_PauliOperator(operator: PauliOperator) -> defaultdict:
    """Unpacks PauliOperator into a defaultdict: integer-site rep: weight."""
    return defaultdict(
        lambda: 0, ((string_to_numbers(k), ps.weight) for k, ps in operator.items())
    )


def pack_PauliOperator(op_dct: defaultdict) -> PauliOperator:
    """Packs a dictionary of inter-site rep: weight to a PauliOperator."""
    N = len(next(iter(op_dct)))
    po = PauliOperator(N)
    for n, w in op_dct.items():
        string = numbers_to_string(n)
        po[string] = PauliString(weight=complex(w), string=string)
    return po


def _sanitize_operations(operations):
    """Sanitizes dictionaries used in PauliOperator style and _rules.

    [(1, "I"), (-1, "Z")] -> two dictionaries (1, "I") and (-1, "Z")
    [] -> (0, "I")
    Specfically it ensures are dictionaries return exactly one Pauli.
    The default one is (0.0, "IIII...").
    First we pad to equal len of returned Paulis, then we split the list
    into multiple dicitonaries. Finally, we remove the list [(0, "I")]
    to (0, "I").
    """

    def pad_operation(op, max_output_len, len_acting):
        op = op.copy()
        for k, v in op.items():
            out_str = len_acting * "I"
            op[k] = op[k] + (max_output_len - len(v)) * [(0.0, out_str)]
        return op

    def split_dict(op, max_output_len):
        out = [{k: [v[0]] for k, v in op.items()}]

        for i in range(1, max_output_len):
            out.append({k: [v[i]] for k, v in op.items()})
        return out

    operations = operations.copy()
    to_add = []
    to_del = []
    for k, v in operations.items():
        op, weights, sites = v
        max_output_len = max((len(x) for x in op.values()))
        len_acting = len(sites[0])

        # Pad
        op = pad_operation(op, max_output_len, len_acting)
        operations[k] = (op, weights, sites)

        # Split those which return more than one Paulu
        if max_output_len > 1:
            new_dicts = split_dict(op, max_output_len)
            for i in range(len(new_dicts)):
                to_add.append((k + str(i + 1), (new_dicts[i], weights, sites)))
            to_del.append(k)
    for k, v in to_add:
        operations[k] = v
    for k in to_del:
        del operations[k]

    # Remove list - we only have one return Pauli
    for k, v in operations.items():
        op, weights, sites = v
        op = {_k: _v[0] for _k, _v in op.items()}
        operations[k] = (op, weights, sites)
    return operations


def opertations_to_jax(operations):
    """Transforms operations to jax compatible form."""
    operations = _sanitize_operations(operations)
    out_dct = {}
    for k, v in operations.items():
        op, weights, sites = v
        len_acting = len(list(op.keys())[0])
        base = (4 ** jnp.arange(len_acting - 1, -1, -1)).astype(np.int8)

        factors = jnp.array([op[s][0] for s in lst_pauli(len_acting)])
        _rules_str = [op[s][1] for s in lst_pauli(len_acting)]
        _rules_inds = jnp.array([np.array(string_to_numbers(s)) for s in _rules_str])
        rule = to_integer(_rules_inds, base)

        weights = jnp.array(weights, dtype=np.complex128)
        sites = jnp.array(sites, dtype=np.int8)
        assert len_acting == sites.shape[-1]
        out_dct[k] = (factors, rule, base, weights, sites)

    return out_dct


def sum_dicts(d1: defaultdict, d2: defaultdict) -> defaultdict:
    """Sums two dictionaries."""
    if len(d1) < len(d2):  # if we take shorter dict, we will have less searches
        d1, d2 = d2, d1
    for k, v in d2.items():
        d1[k] += v
    return d1


@jax.jit
def to_integer(arr, base):
    """Converts site representation to a single integer.

    Site representations are Paulis as integers:
    XIYZ → 1032 - this fun → 1·64 + 0·16 + 3·4 + 2·1 = 78."""
    return jnp.sum(arr * base, axis=-1, dtype=base.dtype)


@jax.jit
def to_base_4(arr, base):
    """Converts integer to site representation."""
    if jnp.ndim(arr) == 0:
        result = arr
    else:
        result = jnp.repeat(jnp.atleast_2d(arr).T, len(base), axis=1)
    return (result // base) % 4


@partial(
    jax.vmap, in_axes=(None, None, None, 0, 0, None, None), out_axes=(0, 0)
)  # in operation
@partial(jax.vmap, in_axes=(None, None, None, None, None, 0, 0))  # in operator
def _apply_sop(factor, rule, base, weight, acting_on, op_w, op):
    """Apply operation on an operator.

    One operation is one action from, say Lindbladian. This can be commutator
    or dissipation.
    For example ∑_i α_ij [Z_i Z_j,·] is described as a factor and rule how Z_iZ_j acts
    on different Pauli pairs: [Z_i Z_j, IX] = 2j ZY. Thus, `factor[1] = 2j` and
    `rule[1] = 11`. weight stores `α_ij` and `acting_on` stores [i,j].

    The opeator being acted on is described similarly Ô_i = ∑β_i P_i, where
    `op_w` stores β_i and P_i is Pauli string in integer-site representation:
    XIYZ = [1,0,3,2].

    The function is vmaped so that it applys only one operation, α_ij [Z_i Z_j, ·]
    on only one part of the operator, β_i P_i."""
    active_sites = to_integer(op[acting_on], base)
    active_site = to_base_4(rule[active_sites], base)
    new_weights = weight * op_w * factor[active_sites]
    out = op.at[acting_on].set(active_site)
    return new_weights, out


def apply_single_operation_jax(
    factors: jax.Array,
    rule: jax.Array,
    base: jax.Array,
    act_weights: jax.Array,
    sites: jax.Array,
    op_dct: defaultdict,
    chunk_size: int = 1000,
) -> defaultdict:
    """Applies an operation on an operator.

    factors, rule, base store information about the operation, e.g. what [ZZ,·] does.
    It should return only a single Pauli.

    act_weights, sites store information about the acting operator: ∑ α_ij Z_i Z_j

    op_dct is a defaultdict (with the default value 0) storing weights as values and
    where keys are tuples of integer-site representation (XIYZ -> (1,0,3,2)).

    Args:
        factors: Factors for corresponding rules (see below, `factors[1] = 2j`).
        rule: Result of applying it on a Pauli string. Ex.: [ZZ,·] acting on IX gives
              2j·ZY, thus `rule[1] = 11`.
        base: Power of four array: if `len(sites) = 3` it should be
              `jnp.array([16,4,1], dtype=np.int8)`
        act_weights: Weights in front of single operations,
                     for α_ij Z_i Z_j it stores _all_ α_ij
        sites: Sites on which the operations are acting on. Stores all [i,j]
        operator: defaultdict where keys are tuples of integer-site representation
                  and values are corresponding weights.
        chunk_size: calculations are done in chunks so that we avoid recompilation."""
    # Get jnp arrays from dictionary
    # It is important that we transform it into numpy array
    operator = np.array(list(op_dct.keys()), np.int8)
    op_weights = np.array(list(op_dct.values()), dtype=np.complex128)

    # Batching - size of arrays changes and causes recompilation
    # Batch
    L = operator.shape[0]
    K = L // chunk_size + 1
    pad_len = K * chunk_size - L
    operator_ = np.pad(operator, ((0, pad_len), (0, 0)), constant_values=0)
    op_weights_ = np.pad(op_weights, ((0, pad_len)))

    operator_ = operator_.reshape(K, chunk_size, operator_.shape[-1])
    op_weights_ = op_weights_.reshape(K, chunk_size)

    # Apply in chunks
    new_weights = []
    new_op = []
    for i in range(K):
        new_weights_, new_op_ = _apply_sop(
            factors,
            rule,
            base,
            act_weights,
            sites,
            jnp.array(op_weights_[i]),
            jnp.array(operator_[i]),
        )
        new_weights_ = new_weights_.reshape(-1)
        new_op_ = new_op_.reshape(-1, operator.shape[1])

        # Remove zeros
        mask = new_weights_ != 0  # In Jax!
        # Important: we will be changing size, this causes recompilation
        new_weights_ = np.asarray(new_weights_)[mask]
        new_op_ = np.asarray(new_op_)[mask]
        new_weights.append(new_weights_)
        new_op.append(new_op_)

    # Put together
    new_op = np.vstack(new_op)
    new_op = np.array(new_op)
    new_weights = np.concatenate(new_weights)

    out_dct = defaultdict(
        lambda: 0, ((tuple(new_op[i]), new_weights[i]) for i in range(len(new_op)))
    )

    return out_dct


def apply_operation_jax(
    operations: dict,
    operator: PauliOperator,
    N_iter=1,
    callbacks=None,
    chunk_size: int = 1000,
):
    """Applies `operations` on `operator` `N_iter`-many times.

    Args:
    operations: dictionary of operations to perform
    operator: PauliOperator to have operations performed on
    N_iter: how many times we repeat the operations
    callbacks: functions to be called at the end of each iteration
    chunk_size:"""
    op_dct = unpack_PauliOperator(operator)

    if callbacks and not isinstance(callbacks, Iterable):
        callbacks = [callbacks]
    if callbacks is None:
        callbacks = []

    for iter in range(N_iter):
        res_dct_list = []
        for op in operations.keys():
            factors, rule, base, act_weights, sites = operations[op]
            res_dct_list.append(
                apply_single_operation_jax(
                    factors,
                    rule,
                    base,
                    act_weights,
                    sites,
                    op_dct,
                    chunk_size,
                )
            )

        for dct in res_dct_list:
            op_dct = sum_dicts(op_dct, dct)

        for callback in callbacks:
            callback(iter, op_dct)

    return op_dct
