import numpy as np

from src.preprocessing import fit_transform_variant, reconstruction_metrics


def test_pca_fit_train_only_dimensions(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.random((8, 1, 28, 28), dtype=np.float32)
    x_val = rng.random((4, 1, 28, 28), dtype=np.float32)
    x_test = rng.random((4, 1, 28, 28), dtype=np.float32)
    fs = fit_transform_variant("pca_4", x_train, x_val, x_test, tmp_path)
    assert fs.x_train.shape[1] == 4
    assert fs.metadata["fit_on"] == "train"
    assert (tmp_path / "pca_4.joblib").exists()


def test_reconstruction_metrics_are_finite():
    x = np.ones((2, 1, 8, 8), dtype=np.float32) * 0.5
    m = reconstruction_metrics(x, x.copy())
    assert m["mse"] == 0.0
    assert np.isfinite(m["ssim"])

