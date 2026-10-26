import os

import pytest


@pytest.mark.skipif(not os.environ.get("RUN_CPG_SMOKE"),
                    reason="set RUN_CPG_SMOKE=1 to run the NEST CPG smoke")
def test_cpg_smoke_writes_metrics(tmp_path):
    pytest.importorskip("nest")
    from bench_persona_compare import run_cpg

    result = run_cpg(seeds=(0,), arms=("persona_nogame", "persona_diag"),
                     steps=20, out_dir=tmp_path)
    assert (tmp_path / "cpg_runs.csv").exists()
    assert (tmp_path / "cpg_bio.md").exists()
    assert len(result["rows"]) == 2
    assert all(row["violations"] == 0 for row in result["rows"])
