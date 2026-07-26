import numpy as np

from src.datasets import array_hash
from src.sampling import nested_balanced_indices


def test_array_hash_is_deterministic():
    x = np.arange(10)
    assert array_hash(x) == array_hash(x.copy())


def test_nested_balanced_indices_are_nested():
    y = np.array([0] * 10 + [1] * 10)
    subsets = nested_balanced_indices(y, [3, 5], seed=11)
    assert set(subsets[3]).issubset(set(subsets[5]))

