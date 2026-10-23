"""Thread count validation and NEST kernel configuration.

These tests use an injected fake kernel so they run without NEST installed.
"""
import pytest

from persona.runtime import configure_kernel, kernel_status, validate_threads


def test_validate_threads_accepts_positive_counts():
    assert validate_threads(1) == 1
    assert validate_threads("8") == 8


@pytest.mark.parametrize("bad", [0, -1, "-3"])
def test_validate_threads_rejects_non_positive_counts(bad):
    with pytest.raises(ValueError):
        validate_threads(bad)


def test_kernel_status_applies_threads_and_keeps_existing_fields():
    status = kernel_status(resolution_ms=0.2, seed=7, threads=4)
    assert status["resolution"] == 0.2
    assert status["print_time"] is False
    assert status["rng_seed"] == 8
    assert status["local_num_threads"] == 4


def test_configure_kernel_resets_before_applying_status():
    calls = []

    class FakeNest:
        @staticmethod
        def ResetKernel():
            calls.append(("reset",))

        @staticmethod
        def SetKernelStatus(status):
            calls.append(("status", dict(status)))

    configure_kernel(FakeNest, resolution_ms=0.2, seed=0, threads=3)

    assert [call[0] for call in calls] == ["reset", "status"]
    assert calls[1][1]["local_num_threads"] == 3


def test_configure_kernel_rejects_bad_threads_before_touching_kernel():
    calls = []

    class FakeNest:
        @staticmethod
        def ResetKernel():
            calls.append("reset")

        @staticmethod
        def SetKernelStatus(status):
            calls.append("status")

    with pytest.raises(ValueError):
        configure_kernel(FakeNest, resolution_ms=0.2, seed=0, threads=0)
    assert calls == []
