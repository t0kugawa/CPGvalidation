from persona.cli import build_parser


def test_trace_flags_parse():
    args = build_parser().parse_args(
        ["--trace-strategy", "--trace-float32", "--snapshot-strategy", "500"])
    assert args.trace_strategy is True
    assert args.trace_float32 is True
    assert args.snapshot_strategy == 500


def test_trace_flags_default_off():
    args = build_parser().parse_args([])
    assert args.trace_strategy is False
    assert args.trace_float32 is False
    assert args.snapshot_strategy == 0
