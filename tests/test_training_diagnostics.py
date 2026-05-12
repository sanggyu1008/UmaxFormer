"""Lightweight smoke tests for the reorganized project layout."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_experiment_configs_exist() -> None:
    configs = sorted((PROJECT_ROOT / "configs" / "experiments").glob("*/*.yaml"))
    assert [path.name for path in configs] == [
        "umaxformer_v1.yaml",
        "umaxformer_v2.yaml",
    ]


def test_outputs_use_experiment_layout() -> None:
    assert (PROJECT_ROOT / "outputs" / "experiments" / "umaxformer_v1").is_dir()
    assert (PROJECT_ROOT / "outputs" / "experiments" / "umaxformer_v2").is_dir()
