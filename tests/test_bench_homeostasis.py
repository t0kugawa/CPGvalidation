from bench_homeostasis import (
    CURRENT_BEST_MODE,
    _homeostatic_config,
    _paired_summary,
    _phase_agreement,
    _write_share_trajectories,
    inflation_metrics,
    run_synthetic,
    type_sign_violations,
)


def test_inflation_metrics_use_the_eval_window_and_bounds():
    means = {
        "a": [0.0, 1.0, 2.0, 3.0],
        "b": [10.0, 10.5, 11.0, 11.5],
    }
    initial = {"a": 0.0, "b": 10.0}
    bounds = {"a": (-2.0, 2.0), "b": (9.0, 13.0)}
    m = inflation_metrics(means, initial, bounds, eval_start=1)
    assert m["n_params"] == 2
    assert m["r_median"] == 0.5625
    assert m["r_p90"] >= m["r_median"]
    assert m["r_max"] == 0.75
    assert m["r_gt_0_25"] == 2
    assert m["range_fallback"] == 0


def test_type_sign_violations_catch_bounds_and_weight_flips():
    means = {
        "conn_pos_weight": [0.2, -0.1],
        "conn_neg_weight": [-0.2, 0.1],
        "conn_stdp_Wmax": [100.0, 0.0],
        "pop_a": [1.0, 3.5],
    }
    bounds = {
        "conn_pos_weight": (0.0, 1.0),
        "conn_neg_weight": (-1.0, 0.0),
        "conn_stdp_Wmax": (1.0, 1000.0),
        "pop_a": (0.0, 3.0),
    }
    ptypes = {"conn_stdp_Wmax": "stdp"}
    bad = type_sign_violations(means, bounds, ptypes)
    assert set(bad) == {"conn_pos_weight", "conn_neg_weight",
                        "conn_stdp_Wmax", "pop_a"}


def test_phase_agreement_is_fraction_of_matching_labels():
    assert _phase_agreement(["a", "b", "a"], ["a", "a", "a"]) == 2.0 / 3.0
    assert _phase_agreement([], []) == 1.0


def test_share_trajectories_are_written_in_long_form(tmp_path):
    path = tmp_path / "share.csv"
    _write_share_trajectories(path, [
        {"seed": 0, "mode": "off", "share_traj": []},
        {"seed": 0, "mode": "budget", "share_traj": [1.0, 2.0]},
    ])
    text = path.read_text(encoding="utf-8").splitlines()
    assert text[0] == "seed,mode,eval_step,share_norm"
    assert len(text) == 3
    assert "budget,1,1.0" in text[1]
    assert "budget,2,2.0" in text[2]


def test_paired_summary_requires_the_20pct_bar_and_no_q_growth():
    def row(seed, mode, p90, loss, q, t=(1.0, 1.0)):
        return {"seed": seed, "mode": mode, "r_p90": p90,
                "loss_eval": loss, "q_gt_1": q,
                "share_rms": None if mode == "off" else 1.0,
                "T_min": t[0], "T_max": t[1]}
    rows = [
        row(0, "off", 1.0, 1.0, 5),
        row(0, "advantage", 0.7, 1.0, 5),
        row(0, "budget", 0.95, 1.0, 6),
        row(0, "band", 0.7, 1.0, 5, (0.8, 1.2)),
    ]
    summary = {m["mode"]: m for m in _paired_summary(
        rows, (0,), ("off", "advantage", "budget", "band"))}
    assert abs(summary["advantage"]["mean_p90_improvement"] - 0.30) < 1e-12
    assert summary["advantage"]["passes"]
    assert not summary["budget"]["accept_20pct"]
    assert summary["budget"]["q_growth_max"] == 1
    assert summary["band"]["band_ok"]


def test_frozen_synthetic_fixture_clears_the_20pct_threshold():
    result = run_synthetic()
    assert result["passes_20pct"] is True
    assert result["share_reduction"] >= 0.20
    assert result["h_on"]["share_rms"] < result["h_off"]["share_rms"]
    assert result["h_on"]["eval_steps"] == 500
    assert result["h_off"]["eval_steps"] == 500


def test_current_best_mode_maps_to_the_centered_delay_preset():
    cfg = _homeostatic_config(CURRENT_BEST_MODE, 100)
    assert cfg.enabled is True
    assert cfg.global_variance_mode == "fisher"
    assert cfg.lift_mode == "site_centered"
    assert cfg.h_delay_steps == 500
    assert cfg.h_burnin_steps == 100
