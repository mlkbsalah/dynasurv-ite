"""Run-directory expansion of the ensemble CLIs; no checkpoints are loaded."""

import pytest

from CausalSurv.recommendation.pipeline import expand_runs


def test_expand_runs_globs_dirs_and_dedupes(tmp_path):
    for name in ("a_seed_1", "b_seed_2"):
        (tmp_path / name / "checkpoints").mkdir(parents=True)
    (tmp_path / "not_a_dir").touch()
    runs = expand_runs(
        [str(tmp_path / "*_seed_*"), str(tmp_path / "a_seed_1")], default_glob="unused"
    )
    assert [r.name for r in runs] == ["a_seed_1", "b_seed_2"]


def test_expand_runs_uses_default_glob_and_raises_on_nothing(tmp_path):
    (tmp_path / "run_seed_9").mkdir()
    runs = expand_runs(None, default_glob=str(tmp_path / "*_seed_*"))
    assert [r.name for r in runs] == ["run_seed_9"]
    with pytest.raises(FileNotFoundError):
        expand_runs([str(tmp_path / "nothing_*")], default_glob="unused")
