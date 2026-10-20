import pandas as pd

from bench_recursive_inverse import MODES, run_benchmark


def test_benchmark_writes_tables_and_figures(tmp_path):
    run_benchmark(tmp_path, n_values=(32,), seeds=(0,), steps=8,
                  refresh_every=4, repeats=1, make_figures=True)
    for name in ("performance.csv", "accuracy.csv", "summary.csv"):
        assert (tmp_path / name).exists()
    perf = pd.read_csv(tmp_path / "performance.csv")
    acc = pd.read_csv(tmp_path / "accuracy.csv")
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert {"n", "mode", "seed", "repeat", "step_ms"}.issubset(perf.columns)
    assert set(perf["mode"]) == set(MODES)
    assert {"n", "seed", "step", "k", "rel_error_decaying",
            "rel_error_constant", "cosine_constant",
            "dmu_rel_error_constant"}.issubset(acc.columns)
    assert {"table", "group", "metric", "mean", "std", "median",
            "q25", "q75"}.issubset(summary.columns)
    assert (perf["step_ms"] > 0.0).all()
    assert acc["rel_error_decaying"].max() < 1e-8
    assert acc.loc[acc["k"] == 0, "rel_error_constant"].max() < 1e-8
    assert acc.loc[acc["k"] > 0, "rel_error_constant"].max() > 0.0
    assert (tmp_path / "performance.png").exists()
    assert (tmp_path / "accuracy.png").exists()
    assert (tmp_path / "README.md").exists()
    try:
        import openpyxl  # noqa: F401
        has_writer = True
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401
            has_writer = True
        except ImportError:
            has_writer = False
    if has_writer:
        assert (tmp_path / "recursive_inverse_results.xlsx").exists()
