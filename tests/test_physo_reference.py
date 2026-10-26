import numpy as np

from persona.partition import PlayerPartition
from persona.physo_reference import (
    global_drive_fate,
    level1_labels,
    level2_labels,
    partition_metrics,
)
from persona.segments import Segment


def _segment(index, name, kind="population", source=None):
    return Segment(index, name, kind, (f"site_{index}",), 1,
                   "exc" if kind == "population" else "static_exc",
                   source=source, target=None)


def _matrix():
    specs = [
        ("pop_BS_E", "population", None),
        ("pop_BS_F", "population", None),
        ("pop_BASE_pg", "population", None),
        ("con_BS_in_e_rg_e_L", "projection", "BS_in_e"),
        ("con_BS_in_e_rg_e_R", "projection", "BS_in_e"),
        ("con_BS_in_f_rg_f_L", "projection", "BS_in_f"),
        ("con_BS_in_f_rg_f_R", "projection", "BS_in_f"),
        ("con_BASE_in_rg_e_L", "projection", "BASE_in"),
        ("con_BASE_in_rg_e_R", "projection", "BASE_in"),
        ("con_BASE_in_rg_f_L", "projection", "BASE_in"),
        ("con_BASE_in_rg_f_R", "projection", "BASE_in"),
    ]
    for side in "LR":
        specs += [
            (f"pop_rg_e_{side}", "population", None),
            (f"pop_rg_f_{side}", "population", None),
            (f"pop_in_e_{side}", "population", None),
            (f"pop_in_f_{side}", "population", None),
            (f"pop_m_e_{side}", "population", None),
            (f"pop_r_e_{side}", "population", None),
            (f"pop_ia_e_{side}", "population", None),
            (f"pop_in_ib_e_{side}", "population", None),
            (f"pop_v2a_{side}", "population", None),
            (f"con_cut_rge_{side}", "projection", f"cut_in_{side}"),
        ]
    specs.append(("con_in_e_L_m_f_L", "projection", "in_e_L"))
    segments = tuple(
        _segment(index, name, kind, source)
        for index, (name, kind, source) in enumerate(specs))
    by_name = {segment.name: segment.id for segment in segments}
    return type("M", (), {
        "segments": segments,
        "segment_of_site": {"pop_BS_E_rate": by_name["pop_BS_E"]},
    })(), by_name, len(segments)


def test_level2_reference_has_nineteen_modules():
    matrix, by_name, n = _matrix()
    labels = level2_labels(matrix)
    assert set(labels) == set(range(n))
    assert len(set(labels.values())) == 19
    assert labels[matrix.segment_of_site["pop_BS_E_rate"]] == "global_drive"
    assert labels[by_name["con_in_e_L_m_f_L"]] == "L_recip_inh"
    assert len(level1_labels(matrix)) == n


def test_partition_metrics_on_perfect_and_merged_partitions():
    matrix, _, n = _matrix()
    labels = level2_labels(matrix)
    groups = {}
    for segment, label in labels.items():
        groups.setdefault(label, []).append(segment)
    perfect = PlayerPartition(tuple(tuple(sorted(group))
                                    for group in groups.values()))
    metrics = partition_metrics(perfect, labels)
    assert np.isclose(metrics["ari"], 1.0)
    assert np.isclose(metrics["nmi"], 1.0)
    assert np.isclose(metrics["purity"], 1.0)
    assert np.isclose(metrics["v_measure"], 1.0)
    assert np.isclose(metrics["coverage"], 1.0)

    merged_groups = list(perfect.groups)
    merged_groups[0] = tuple(sorted(merged_groups[0] + merged_groups[1]))
    del merged_groups[1]
    merged = PlayerPartition(tuple(merged_groups))
    assert partition_metrics(merged, labels)["ari"] < 1.0


def _drive_ids(by_name):
    rate = [by_name[name] for name in ("pop_BS_E", "pop_BS_F",
                                       "pop_BASE_pg")]
    projections = [segment_id for name, segment_id in by_name.items()
                   if name.startswith("con_BS_in_") or
                   name.startswith("con_BASE_in_")]
    return rate, projections


def _partition_with_drive(matrix, by_name, drive_groups):
    assigned = {segment for group in drive_groups for segment in group}
    others = [segment.id for segment in matrix.segments
              if segment.id not in assigned]
    groups = list(drive_groups) + [(segment,) for segment in others]
    return PlayerPartition(tuple(groups))


def test_reference_covers_the_fast_cpg(nest_kernel):
    from persona.registry import Registry
    from persona.segments import SegmentMatrix
    from t2xcpg.bs import BaseDrive, BrainStem
    from t2xcpg.cpg import L2CPG
    from t2xcpg.leg import Leg

    registry = Registry()
    bs = BrainStem(registry=registry, n_bs=4)
    base = BaseDrive(registry=registry, n_bs=4)
    legs = {side: Leg(side, registry, n_rg=4, n_mn=4, n_int=4, n_aff=3)
            for side in "LR"}
    L2CPG(registry=registry, bs=bs, legs=legs, dt_ms=10.0, base=base)
    matrix = SegmentMatrix.from_registry(registry)
    labels = level2_labels(matrix)
    assert set(labels) == set(range(matrix.n_segments))
    assert len(set(labels.values())) == 19
    assert "other" not in labels.values()
    assert labels[matrix.segment_of_site["pop_BS_E_rate"]] == "global_drive"


def test_global_drive_fate_classes():
    matrix, by_name, _ = _matrix()
    rate, projections = _drive_ids(by_name)
    same = _partition_with_drive(matrix, by_name, [tuple(rate + projections)])
    assert global_drive_fate(same, matrix)["classification"] == "same_player"

    left = [segment for segment in projections
            if matrix.segments[segment].name.endswith("_L")]
    right = [segment for segment in projections
             if matrix.segments[segment].name.endswith("_R")]
    split = _partition_with_drive(
        matrix, by_name, [tuple(rate), tuple(left), tuple(right)])
    assert global_drive_fate(split, matrix)["classification"] == \
        "rate_global_projections_limb"

    absorbed = _partition_with_drive(
        matrix, by_name, [tuple(rate), tuple(projections)])
    assert global_drive_fate(absorbed, matrix)["classification"] == "absorbed"
