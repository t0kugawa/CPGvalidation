from personalize import run_personalization, DEFAULT_REFERENCE, DEFAULT_STATS


def test_e2e_smoke(tmp_path):
    res = run_personalization(
        reference_path=DEFAULT_REFERENCE,
        stats_path=DEFAULT_STATS,
        output_dir=str(tmp_path),
        n_steps=20,
        small=True,
    )
    assert (tmp_path / "validation_results.png").exists()
    assert (tmp_path / "run_log.npz").exists()
    assert "acceptable" in res
    assert "final_loss" in res