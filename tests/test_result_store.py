import pandas as pd

from src.result_store import RUN_SUBDIRS, append_manifest, create_run_dir


def test_run_dir_structure_and_manifest(tmp_path):
    run_dir = create_run_dir(tmp_path, "unit")
    for subdir in RUN_SUBDIRS:
        assert (run_dir / subdir).is_dir()
    append_manifest(run_dir, {"configuration_id": "a", "status": "completed"})
    df = pd.read_parquet(run_dir / "metrics" / "experiment_manifest.parquet")
    assert df.loc[0, "configuration_id"] == "a"

