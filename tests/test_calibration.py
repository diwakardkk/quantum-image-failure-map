import numpy as np

from src.calibration import calibration_bins


def test_calibration_bins_counts_all_samples():
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    bins = calibration_bins(y, p, n_bins=4)
    assert bins["count"].sum() == 4

