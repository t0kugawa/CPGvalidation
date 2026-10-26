def test_bio_table_marks_sole_up_not_measured():
    from bench_persona_compare import bio_table

    table = bio_table({
        "phase_passed": False,
        "biological_acceptable": False,
        "angle_coverage": 0.1,
        "duty_factor_in_distribution": True,
        "violations": 0,
        "at_bound": 3,
        "q_gt_1": 2,
    })
    assert "not measured" in table["sole_up"].lower()
    assert table["violations"] == 0
    assert table["phase_gate"] is False
