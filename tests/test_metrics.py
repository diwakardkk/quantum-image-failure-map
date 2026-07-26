import numpy as np

from src.evaluation import binary_metrics, prediction_frame


def test_binary_metrics_expected_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.8, 0.9])
    metrics = binary_metrics(y, p)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


def test_prediction_frame_columns():
    df = prediction_frame(
        np.array([0, 1]),
        np.array([0.2, 0.8]),
        sample_ids=np.array([10, 11]),
        seed=11,
        model="m",
        dataset="d",
        split="test",
        configuration_id="abc",
    )
    assert {"configuration_id", "sample_id", "probability", "predicted_label"}.issubset(df.columns)

