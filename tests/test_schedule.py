import math

import pytest

from persona.schedule import CosineLR, resolve_lr_schedule


def test_endpoints_and_midpoint():
    s = CosineLR(start=0.1, end=0.0, total_steps=11)
    assert s.at(1) == 0.1
    assert s.at(11) == 0.0
    assert s.at(100) == 0.0
    assert math.isclose(s.at(6), 0.05, rel_tol=1e-12)


def test_warmup_plateau_then_decay():
    s = CosineLR(0.1, 0.0, 10, warmup_steps=3)
    assert s.at(1) == 0.1
    assert s.at(4) == 0.1
    assert s.at(5) < 0.1
    assert s.at(10) == 0.0


def test_one_step_and_constant():
    assert CosineLR(0.1, 0.0, 1).at(1) == 0.1
    constant = CosineLR(0.05, 0.05, 10)
    assert all(constant.at(i) == 0.05 for i in range(1, 11))


@pytest.mark.parametrize("kwargs", [
    {"start": -0.1, "end": 0.0, "total_steps": 10},
    {"start": 0.1, "end": 0.0, "total_steps": 0},
    {"start": 0.1, "end": 0.0, "total_steps": 10, "warmup_steps": 10},
    {"start": float("nan"), "end": 0.0, "total_steps": 10},
])
def test_validation(kwargs):
    with pytest.raises(ValueError):
        CosineLR(**kwargs)


def test_resolve_lr_schedule():
    assert resolve_lr_schedule("constant", 0.03, None, 10) is None
    s = resolve_lr_schedule("cosine", 0.03, 0.003, 10)
    assert isinstance(s, CosineLR)
    assert (s.start, s.end, s.total_steps) == (0.03, 0.003, 10)
    with pytest.raises(ValueError):
        resolve_lr_schedule("cosine", 0.03, None, 10)
    with pytest.raises(ValueError):
        resolve_lr_schedule("bogus", 0.03, 0.0, 10)
