from itertools import product

from .pauli_operator import PauliOperator, PauliString


def lst_pauli(n):
    _lst_pauli = ["I", "X", "Z", "Y"]
    return tuple("".join(s) for s in product(*(n * [_lst_pauli])))


def string_to_numbers(string):
    _s_to_n = {"I": 0, "X": 1, "Z": 2, "Y": 3}
    return tuple([_s_to_n[s] for s in string])


def numbers_to_string(numbers):
    _n_to_s = {0: "I", 1: "X", 2: "Z", 3: "Y"}
    return "".join([_n_to_s[n] for n in numbers])


def _apply_superoperator_operation(sop, weights, sites, operator):
    assert len(weights) == len(sites)
    res = PauliOperator(operator.N)

    for i in range(len(weights)):
        for _, pauli in operator.items():
            string_list = pauli.list  # This returns a copy
            applied_com_list = sop["".join(string_list[sites[i]])]
            for factor, new_local_str in applied_com_list:
                string_list[sites[i]] = list(new_local_str)
                if factor:
                    new_pauli = PauliString(
                        factor * pauli.weight * weights[i], string_list
                    )
                    res[new_pauli] += new_pauli
    return res


def apply_superoperator(superoperator: dict, operator: PauliOperator):
    """Applies a (super)operator to an operator.

    Args:
        superoperator: dictionary of operations (commutation with X, comm. with ZZ,...).
                       Each dictionary is a tuple of rules (dict, e.g.
                       `dct['I'] = [(1, 'X')]` describes [X,I] → X),
                       weights and sites operations acts on: ∑α_i X_i.
        operator: PauliOperator being acted on."""
    new_operator = PauliOperator(operator.N)
    for _, superoperation in superoperator.items():
        sop, weights, sites = superoperation
        new_operator += _apply_superoperator_operation(sop, weights, sites, operator)

    return new_operator
