def test_synthetic_quick_writes_files(tmp_path):
    from bench_strategy_history import run_synthetic

    result = run_synthetic(out_dir=tmp_path, steps=20, seed=0)
    assert (tmp_path / "synthetic_summary.json").exists()
    assert (tmp_path / "synthetic_runs.csv").exists()
    assert result["trace_rebuild_error"] < 1e-12
    assert result["parameter_trace_error"] > 1e-6
