import pytest
import pauli_algebra as pa


def test_init():
    pa.PauliOperator(3)

    pauli_strings = [
        pa.PauliString(2.0, "IXZ"),
        pa.PauliString(-1.0, "XZI"),
    ]

    op1 = pa.PauliOperator(strings=["IXZ", "XZI"], weights=[2.0, -1.0])
    assert op1.N == 3

    op2 = pa.PauliOperator(pauli_strings=pauli_strings)
    assert op2.N == 3

    with pytest.raises(TypeError):
        pa.PauliOperator()
    with pytest.raises(ValueError):
        op1 = pa.PauliOperator(strings=["IXZ", "XI"], weights=[2.0, -1.0])
    with pytest.raises(ValueError):
        op1 = pa.PauliOperator(strings=["IXZ", "XIZ"], weights=[2.0])
    with pytest.raises(TypeError):
        pa.PauliOperator(pauli_strings=["IXZ"])
    with pytest.raises(TypeError):
        pa.PauliOperator(
            pauli_strings=pauli_strings,
            strings=["IXZ", "XZI"],
            weights=[2.0, -1.0],
        )

def test_caching():
    pa.PauliOperator(3)

    strings = ["IXZ", "XZI", "XYI", "YZI"]
    weights = [2.0, -1.0, 4.0, -1.5]

    op = pa.PauliOperator(strings=strings, weights=weights)
    assert op._strings is None
    assert op._weights is None

    op_strings = op.strings
    op_weights = op.weights

    for i in range(len(strings)):
        assert strings[i] in op_strings
        assert weights[i] in op_weights

    new_string, new_weight = "YXY", 3.3
    op[new_string] = new_weight
    assert op._strings is None
    assert op._weights is None

    weights.append(new_weight)
    strings.append(new_string)

    op_strings = op.strings
    op_weights = op.weights

    for i in range(len(strings)):
        assert strings[i] in op_strings
        assert weights[i] in op_weights

    del op["IXZ"]
    assert op._strings is None
    assert op._weights is None
    for i in range(1, len(strings)):
        assert strings[i] in op_strings
        assert weights[i] in op_weights
