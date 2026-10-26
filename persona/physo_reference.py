"""Physiological reference hierarchy and partition-comparison metrics.

The reference is evaluation-only. It never trains Evo. Level 1 groups a
population with its efferent projections; Level 2 groups Level 1 units into the
functional modules of the CPG.
"""
from __future__ import annotations

import numpy as np

_GLOBAL_POPULATIONS = {"BS_E", "BS_F", "BASE_pg"}
_DRIVE_PROJECTION_PREFIXES = ("con_BS_in_", "con_BASE_in_")


def _side(name: str) -> str:
    for suffix in ("_L", "_R"):
        if name.endswith(suffix):
            return suffix[-1]
    return "?"


def _population_module(population: str) -> str | None:
    name = str(population)
    if name in _GLOBAL_POPULATIONS:
        return "global_drive"
    side = _side(name)
    if side == "?":
        return None
    if name.startswith("rg_e_"):
        return f"{side}_rg_ext"
    if name.startswith("rg_f_"):
        return f"{side}_rg_flex"
    if name.startswith(("in_e_", "in_f_")):
        return f"{side}_recip_inh"
    if name.startswith(("m_e_", "m_f_")):
        return f"{side}_motor"
    if name.startswith(("r_e_", "r_f_")):
        return f"{side}_renshaw"
    if name.startswith(("ia_e_", "ia_f_", "ia_e_in_", "ia_f_in_")):
        return f"{side}_ia"
    if name.startswith(("in_ib_e_", "in_ib_f_")):
        return f"{side}_ib"
    if name.startswith(("v2a_", "v0v_", "v3f_", "v0d_", "in1_")):
        return f"{side}_commissural"
    return None


def _relay_group(source: str) -> str | None:
    source = str(source)
    if source in ("BS_in_e",):
        return "BS_E"
    if source in ("BS_in_f",):
        return "BS_F"
    if source in ("BASE_in",):
        return "BASE_pg"
    if source.startswith("cut_in_"):
        return f"cutaneous_{_side(source)}"
    if source.startswith(("ib_e_in_", "ib_f_in_")):
        return f"ib_{_side(source)}"
    return None


def level1_labels(matrix) -> dict[int, str]:
    """Population + its efferents; relay-sourced projections join their drive."""
    labels: dict[int, str] = {}
    for segment in matrix.segments:
        if segment.kind == "population":
            labels[segment.id] = f"L1:{segment.name[4:]}"
            continue
        source = str(segment.source or "")
        relay = _relay_group(source)
        if relay is not None:
            labels[segment.id] = f"L1:{relay}"
            continue
        if source:
            labels[segment.id] = f"L1:{source}"
        else:
            labels[segment.id] = f"L1:relay:{segment.name}"
    return labels


def level2_labels(matrix) -> dict[int, str]:
    """Functional modules: global drive + nine per-limb groups."""
    labels: dict[int, str] = {}
    for segment_id, label in level1_labels(matrix).items():
        unit = label[3:]  # strip "L1:"
        if unit.startswith("cutaneous_"):
            module = f"{unit[-1]}_cutaneous"
        elif unit.startswith("ib_"):
            module = f"{unit[-1]}_ib"
        else:
            module = _population_module(unit) or "other"
        labels[segment_id] = module
    return labels


def partition_metrics(partition, labels: dict[int, str]) -> dict:
    """ARI/NMI/V-measure/purity/coverage/pairwise-F1 for a segment partition."""
    segment_ids = sorted(labels)
    if not segment_ids:
        return {key: 0.0 for key in (
            "ari", "nmi", "homogeneity", "completeness", "v_measure",
            "purity", "coverage", "pairwise_f1")}
    player_of = {}
    for index, group in enumerate(partition.groups):
        for segment in group:
            player_of[int(segment)] = index
    true = np.asarray([str(labels[segment]) for segment in segment_ids])
    pred = np.asarray([player_of.get(segment, -1)
                       for segment in segment_ids])
    n = len(segment_ids)
    true_values, true_idx = np.unique(true, return_inverse=True)
    pred_values, pred_idx = np.unique(pred, return_inverse=True)
    table = np.zeros((len(true_values), len(pred_values)), dtype=float)
    np.add.at(table, (true_idx, pred_idx), 1.0)
    a = table.sum(axis=1)
    b = table.sum(axis=0)

    def comb2(values):
        return float(np.sum(values * (values - 1.0) / 2.0))

    sum_comb = comb2(table.ravel())
    a_comb = comb2(a)
    b_comb = comb2(b)
    expected = a_comb * b_comb / max(n * (n - 1.0) / 2.0, 1e-12)
    max_index = 0.5 * (a_comb + b_comb)
    ari = (1.0 if abs(max_index - expected) < 1e-12
           else (sum_comb - expected) / (max_index - expected))

    p = table / n
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(table > 0,
                         np.log((table * n) / np.outer(a, b)), 0.0)
        mi = float(np.sum(p * ratio))
    h_true = _entropy(a / n)
    h_pred = _entropy(b / n)
    denom = 0.5 * (h_true + h_pred)
    nmi = 1.0 if denom <= 1e-12 else float(mi / denom)

    h_true_given_pred = 0.0
    for j, col in enumerate(table.T):
        if b[j] > 0:
            h_true_given_pred -= float(np.sum(
                (col / n) * np.log(np.where(col > 0, col / b[j], 1.0))))
    h_pred_given_true = 0.0
    for i, row in enumerate(table):
        if a[i] > 0:
            h_pred_given_true -= float(np.sum(
                (row / n) * np.log(np.where(row > 0, row / a[i], 1.0))))
    homogeneity = 1.0 if h_true <= 1e-12 else 1.0 - h_true_given_pred / h_true
    completeness = 1.0 if h_pred <= 1e-12 else 1.0 - h_pred_given_true / h_pred
    v_measure = (0.0 if homogeneity + completeness <= 1e-12 else
                 2.0 * homogeneity * completeness
                 / (homogeneity + completeness))
    purity = float(np.sum(np.max(table, axis=0)) / n)
    coverage = float(np.sum(pred >= 0) / n)
    precision = 1.0 if b_comb <= 1e-12 else float(sum_comb / b_comb)
    recall = 1.0 if a_comb <= 1e-12 else float(sum_comb / a_comb)
    pairwise_f1 = (0.0 if precision + recall <= 1e-12 else
                   2.0 * precision * recall / (precision + recall))
    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "purity": float(purity),
        "coverage": float(coverage),
        "pairwise_f1": float(pairwise_f1),
    }


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))


def global_drive_fate(partition, matrix) -> dict:
    """Classify where the BS/BASE rate and projection segments ended up."""
    by_name = {segment.name: segment.id for segment in matrix.segments}
    rate_names = ["pop_BS_E", "pop_BS_F", "pop_BASE_pg"]
    rate_ids = [by_name[name] for name in rate_names if name in by_name]
    projection_ids = [
        segment.id for segment in matrix.segments
        if any(segment.name.startswith(prefix)
               for prefix in _DRIVE_PROJECTION_PREFIXES)
    ]
    player_of = {}
    for index, group in enumerate(partition.groups):
        for segment in group:
            player_of[int(segment)] = index
    rate_players = sorted({player_of[segment] for segment in rate_ids
                           if segment in player_of})
    projection_players = sorted({player_of[segment]
                                 for segment in projection_ids
                                 if segment in player_of})
    left = [segment for segment in rate_ids + projection_ids
            if _side(matrix.segments[segment].name) == "L"]
    right = [segment for segment in rate_ids + projection_ids
             if _side(matrix.segments[segment].name) == "R"]
    left_players = {player_of[segment] for segment in left
                    if segment in player_of}
    right_players = {player_of[segment] for segment in right
                     if segment in player_of}
    all_players = {player_of[segment]
                   for segment in rate_ids + projection_ids
                   if segment in player_of}
    if not rate_ids:
        classification = "missing"
    elif len(all_players) == 1:
        classification = "same_player"
    elif len(rate_players) == 1 and len(projection_players) == 1 \
            and rate_players != projection_players:
        classification = "absorbed"
    elif len(rate_players) == 1 and len(projection_players) == 2:
        classification = "rate_global_projections_limb"
    elif len(left_players) == 1 and len(right_players) == 1 \
            and left_players != right_players:
        classification = "split_by_limb"
    else:
        classification = "other"
    return {
        "classification": classification,
        "rate_segments": [matrix.segments[segment].name
                          for segment in rate_ids],
        "projection_segments": [matrix.segments[segment].name
                                for segment in projection_ids],
        "rate_players": rate_players,
        "projection_players": projection_players,
    }
