import numpy as np

from bench_evo import _DummyMatrix, run_compare
from persona.evo import GenerationResult


def _fake_runner(partition, generation, seed):
    n = len(partition.groups)
    values = np.ones(n)
    return GenerationResult(
        partition=partition, generation=generation, seed=seed,
        fractions=values, fraction_mean=values.copy(),
        contribution=np.ones(n), transfer=np.zeros(n),
        replicator_residual=0.0, movement=0.0,
        metrics={"loss_eval": 1.0, "n_steps": 10}, wall_s=0.0)


def test_bench_evo_summary_writes(tmp_path):
    result = run_compare(
        tmp_path, seeds=(0,), players=(4,), inits=("territory",),
        stops=("partition",), steps=10, generations=2, jobs=1,
        prefix="smoke", runner=_fake_runner, matrix=_DummyMatrix(16))
    assert (tmp_path / "smoke_summary.json").exists()
    assert (tmp_path / "smoke_runs.csv").exists()
    assert result["summary"]
    assert result["summary"][0]["players"] == 4
    assert result["summary"][0]["n"] == 1
