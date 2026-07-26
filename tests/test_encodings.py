import numpy as np

from src.encodings import EncodingSpec, amplitude_vectors, encoding_resources, fidelity_matrix, quantum_feature_matrix


def test_amplitude_vectors_are_normalized():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
    z = amplitude_vectors(x, 3)
    assert z.shape == (2, 4)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0)


def test_quantum_feature_matrix_dimensions():
    x = np.random.default_rng(0).normal(size=(5, 6))
    z, meta = quantum_feature_matrix(x, EncodingSpec("angle", 4, 4))
    assert z.shape == (5, 4)
    assert meta["n_qubits"] == 4


def test_fidelity_matrix_shape_and_diagonal():
    x = np.eye(4)
    f = fidelity_matrix(x)
    assert f.shape == (4, 4)
    assert np.allclose(np.diag(f), 1.0)


def test_encoding_resources_contains_gate_counts():
    resources = encoding_resources(EncodingSpec("amplitude", 4, 2))
    assert resources["total_gates"] >= resources["two_qubit_gates"]

