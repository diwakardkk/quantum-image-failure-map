import pandas as pd

from src.plotting import generate_all_plots


def test_plot_generation_from_saved_metrics(tmp_path):
    (tmp_path / "metrics").mkdir()
    pd.DataFrame(
        {
            "model": ["a", "b"],
            "accuracy": [0.5, 0.75],
            "balanced_accuracy": [0.5, 0.75],
            "macro_f1": [0.4, 0.7],
        }
    ).to_csv(tmp_path / "metrics" / "primary_metrics.csv", index=False)
    figures = generate_all_plots(tmp_path)
    assert "P1_F03_accuracy_vs_feature_count" in figures
    assert (tmp_path / "figures" / "P1_F03_accuracy_vs_feature_count.png").exists()

