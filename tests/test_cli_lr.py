import pytest

from personalize import build_parser, parse_cli


def test_cli_defaults_keep_constant_mode():
    args = parse_cli([])
    assert args.steps == 200
    assert args.lr_mode == "constant"
    assert args.lr_start == 0.03
    assert args.lr_end is None


def test_cli_cosine_accepts_explicit_endpoints():
    args = parse_cli([
        "--lr-mode", "cosine", "--lr-start", "0.03",
        "--lr-end", "0.003", "--steps", "50",
    ])
    assert (args.lr_mode, args.lr_start, args.lr_end, args.steps) == (
        "cosine", 0.03, 0.003, 50)


def test_cli_cosine_without_end_is_an_error():
    with pytest.raises(SystemExit):
        parse_cli(["--lr-mode", "cosine"])


def test_parser_still_exposes_the_flags():
    args = build_parser().parse_args(["--lr-mode", "cosine", "--lr-end", "0.0"])
    assert (args.lr_mode, args.lr_end) == ("cosine", 0.0)
