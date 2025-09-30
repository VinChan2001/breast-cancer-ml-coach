from pathlib import Path

from ml_project.config import TrainingConfig
from ml_project.pipeline import run_pipeline


def test_run_pipeline(tmp_path: Path) -> None:
    config = TrainingConfig(output_dir=tmp_path, test_size=0.25, random_state=0)
    result = run_pipeline(config)

    assert set(result["metrics"]).issuperset({"accuracy", "precision", "recall", "f1"})
    assert result["model_path"].exists()
    assert result["metrics_path"].exists()
    assert result["report_path"].exists()
