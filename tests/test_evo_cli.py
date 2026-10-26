import json

import pytest

from persona.cli import build_parser, validate_args
from persona.evo import EvoConfig, EvoResult, PlayerPartition, write_evo_outputs


class _Matrix:
    n_segments = 3
    n_sites = 3
    segments = (
        type("S", (), {"name": "pop_A"})(),
        type("S", (), {"name": "pop_B"})(),
        type("S", (), {"name": "pop_C"})(),
    )

    def validate_partition(self, partition):
        seen = {segment for group in partition.groups for segment in group}
        if seen != {0, 1, 2}:
            raise ValueError("coverage")


def _result():
    partition = PlayerPartition(((0,), (1, 2)))
    history = [{
        "generation": 0,
        "partition": [[0], [1, 2]],
        "fractions": [1.0, 0.0],
        "fraction_mean": [1.0, 0.0],
        "contribution": [1.0, 0.0],
        "transfer": [0.0, 0.0],
        "replicator_residual": 0.0,
        "movement": 0.0,
        "fraction_stability": None,
        "dead": [False, True],
        "metrics": {"loss_eval": 0.5},
        "wall_s": 1.0,
        "prune_reason": "pruned",
        "redistribution": [],
    }]
    return EvoResult(
        final_partition=partition, history=history, stop_reason="partition",
        equilibrium={"stop_reason": "partition", "generations": 1})


def test_training_defaults_to_manual():
    args = build_parser().parse_args([])
    assert args.training == "manual"
    assert args.evo_players == 24
    assert args.evo_stop == "both"


def test_evo_rejects_game_none_and_player_map():
    args = build_parser().parse_args(["--training", "evo", "--game", "none"])
    with pytest.raises(SystemExit):
        validate_args(args)
    args = build_parser().parse_args(
        ["--training", "evo", "--player-map", "x.json"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_write_evo_outputs_round_trips(tmp_path):
    result = _result()
    matrix = _Matrix()
    config = EvoConfig(n_players=2, seed=0)
    write_evo_outputs(tmp_path, result, matrix, config)
    assert (tmp_path / "evo_config.json").exists()
    assert (tmp_path / "initial_partition.json").exists()
    assert (tmp_path / "gen_000.json").exists()
    assert (tmp_path / "history.csv").exists()
    final = PlayerPartition.from_json(
        json.loads((tmp_path / "final_partition.json").read_text()))
    final.validate(matrix)
    assert final == result.final_partition
    first = (tmp_path / "initial_partition.json").read_text()
    other = tmp_path / "other"
    write_evo_outputs(other, result, matrix, config)
    assert (other / "initial_partition.json").read_text() == first
