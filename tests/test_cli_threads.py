"""The simulator-independent CLI parser exposes --nest-threads."""
import pytest

from persona.cli import build_parser, parse_cli


def test_nest_threads_defaults_to_one():
    assert parse_cli([]).nest_threads == 1


def test_nest_threads_accepts_explicit_count():
    assert parse_cli(["--nest-threads", "4"]).nest_threads == 4


@pytest.mark.parametrize("value", ["0", "-2"])
def test_nest_threads_rejects_non_positive_count(value):
    with pytest.raises(SystemExit):
        parse_cli(["--nest-threads", value])


def test_parser_exposes_nest_threads():
    args = build_parser().parse_args(["--nest-threads", "8"])
    assert args.nest_threads == 8
