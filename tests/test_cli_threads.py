"""The simulator-independent CLI parser exposes --threads."""
import pytest

from persona.cli import build_parser, parse_cli


def test_threads_defaults_to_one():
    assert parse_cli([]).threads == 1


def test_threads_accepts_explicit_count():
    assert parse_cli(["--threads", "4"]).threads == 4


@pytest.mark.parametrize("value", ["0", "-2"])
def test_threads_rejects_non_positive_count(value):
    with pytest.raises(SystemExit):
        parse_cli(["--threads", value])


def test_parser_exposes_threads():
    args = build_parser().parse_args(["--threads", "8"])
    assert args.threads == 8
