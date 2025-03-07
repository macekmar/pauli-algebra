# Pauli algebra

`PauliOperator` holds many weighted `PauliStrings`.
Algebra is implemented in `pauli_algebra.py` and `pauli_algebra_jax.py`.
Some common rules are implemented in `_rules.py`.

My testing revealed that libraries like [PauliArray](https://algolab-quantique.github.io/pauliarray/index.html) can be slower.
For example, PauliArray represents Pauli strings using two bitstrings for $X$ and $Z$ (where $X=Z=1$ indicates $Y$).
Manipulating Pauli strings in this way involves vector algebra with integers. I believe this method is more efficient when dealing with general Pauli strings that have long support.

However, for time evolution under typical Hamiltonians, the terms in the Hamiltonian usually have a support of up to two. In such cases, I think using dictionaries can be faster.

## Example

Unitary time evolution $\dot{A} = i [H,A]$ under transverse-field Ising Hamiltonian $\sum_i -2 X_i + 3 Z_i Z_{i+1}$.
First, let us define the operator

```python
N = 3
strings = [
    "YIZ",
    "III",
    "IZZ",
    "XIZ",
    "XYX",
    "YXX",
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

ps_ext = pa.PauliOperator(N)
for i in range(len(strings)):
    ps_ext[strings[i]] = pa.PauliString(weights[i], strings[i])
```

We have to provide rules how $H$ acts on $A$.
We split the operations into distinct rules, in this case how $X$ and $ZZ$ commutes with Pauli strings.
Additionally, we have to provide weights and on which sites it acts on.

```python
# Provide operations of the superoperator
tfi_sop = {}
# Operations are split into distinct rules
tfi_sop["X"] = (
    {'I': [], 'X': [], 'Z': [((-0-2j), 'Y')], 'Y': [(2j, 'Z')]},
    np.array([-2, -2, -2], dtype=np.complex128),
    [[0], [1], [2]]
)
tfi_sop["ZZ"] = (
    {
        'II': [],
        'IX': [(2j, 'ZY')],
        'IZ': [],
        'IY': [((-0-2j), 'ZX')],
        'XI': [(2j, 'YZ')],
        'XX': [],
        'XZ': [(2j, 'YI')],
        'XY': [],
        'ZI': [],
        'ZX': [(2j, 'IY')],
        'ZZ': [],
        'ZY': [((-0-2j), 'IX')],
        'YI': [((-0-2j), 'XZ')],
        'YX': [],
        'YZ': [((-0-2j), 'XI')],
        'YY': []
    },
    np.array([3, 3, 3], dtype=np.complex128),
    [[0, 1], [1, 2], [2, 0]]
)
```

Finally, we can apply the commutator with the Hamiltonian, e.g. $A(t+dt) = i \cdot dt \cdot [H,A(t)]$

```python
A = 1j * dt * pa.apply_superoperator(tfi_sop, A)
```

Note, that in `tfi.py` we have `gen_Lindblad_TFI` which includes factor `1j` inside the weights since dissipation does not have this factor, e.g. doing

```python
A = 1j * dt * pa.apply_superoperator(tfi_sop, A)
```

is wrong.
