def test_initial_vector_varies_by_seed():
    from bench_persona_compare import _initial_for_seed
    from benchmarks.compare.toy_cpg import BOUNDS

    first = _initial_for_seed(0)
    second = _initial_for_seed(1)
    assert first != second
    for name, value in first.items():
        low, high = BOUNDS[name]
        assert low <= value <= high


def test_toy_smoke_writes_files(tmp_path):
    from bench_persona_compare import run_toy

    result = run_toy(
        seeds=(0,), arms=("persona_game", "persona_diag", "cmaes",
                          "surrogate"),
        episodes=2, out_dir=tmp_path)
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "curves.json").exists()
    assert result["summary"]
    assert {row["arm"] for row in result["rows"]} == {
        "persona_game", "persona_diag", "cmaes", "surrogate"}
