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
