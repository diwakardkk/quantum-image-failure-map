import importlib.util

import numpy as np
import pytest

from src.quantum_models import initialize_parameters


def test_parameter_initialization_shape():
    params = initialize_parameters(layers=2, qubits=4, seed=11)
    assert params.shape == (2, 4, 2)


@pytest.mark.skipif(importlib.util.find_spec("pennylane") is None, reason="PennyLane not installed")
def test_quantum_model_forward_pass():
    from src.quantum_models import PennyLaneVQC

    model = PennyLaneVQC(4, 4, 1, device_name="default.qubit")
    params = initialize_parameters(1, 4, 11)
    prob = model.predict_proba(np.zeros((2, 4)), params)
    assert prob.shape == (2,)
    assert np.all((prob >= 0) & (prob <= 1))

