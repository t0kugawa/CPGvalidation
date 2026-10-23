"""
Source scan (spec §5.5): every NEST object is created and connected through
`persona/registry.py`; nothing else in `persona/`, `t2xcpg/` or the two entry
points may call `nest.Create`/`nest.Connect` directly.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCANNED = (list((ROOT / "persona").glob("*.py"))
           + list((ROOT / "t2xcpg").glob("*.py"))
           + [ROOT / "personalize.py", ROOT / "run_validation.py"])

SKIP_NAMES = {"registry.py", "nest_adapter.py", "conftest.py"}

# The only NEST surface allowed outside the registry: synapse-model definition
# (`CopyModel` in t2xcpg/utils.py, needed before the registry connects), the
# kernel clock (`Simulate` in t2xcpg/cpg.py) and kernel lifecycle/verbosity at
# the two entry points. Everything that creates or connects nodes goes through
# the registry.
ALLOWED_CALLS = {"CopyModel", "Simulate", "ResetKernel", "SetKernelStatus"}


def _scanned_files():
    return [p for p in SCANNED if p.name not in SKIP_NAMES]


def test_no_raw_nest_create_or_connect_outside_registry():
    bad = []
    for path in _scanned_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"nest\.(Create|Connect)\(", line):
                bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert bad == [], "\n".join(bad)


def test_nest_api_outside_registry_is_allowlisted():
    bad = []
    for path in _scanned_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in re.finditer(r"nest\.([A-Za-z_]\w*)\s*\(", line):
                if m.group(1) not in ALLOWED_CALLS:
                    bad.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
                    break
    assert bad == [], "\n".join(bad)
