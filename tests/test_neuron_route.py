"""Single-route source scan for NEURON.

Raw ``h``/``neuron`` access is allowed only in the adapter file and its helper
package.  Model/learning code must go through the registry.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    Path("persona/neuron_adapter.py"),
    Path("persona/neuron/kernel.py"),
    Path("persona/neuron/cells.py"),
    Path("persona/neuron/groups.py"),
    Path("persona/neuron/recorders.py"),
    Path("persona/neuron/synapses.py"),
}

_PATTERNS = (
    re.compile(r"^\s*import\s+neuron\b"),
    re.compile(r"^\s*from\s+neuron\s+import\b"),
    re.compile(r"\bh\.(?:NetCon|NetStim|Section|ExpSyn|Exp2Syn|IntFire|"
               r"STDP|Random|ParallelContext)\b"),
)


def _scanned():
    for path in (ROOT / "persona").rglob("*.py"):
        rel = path.relative_to(ROOT)
        if rel.parts[:2] == ("persona", "neuron"):
            continue
        if rel in ALLOWED:
            continue
        yield rel, path


def test_no_raw_neuron_outside_adapter_package():
    bad = []
    for rel, path in _scanned():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(p.search(line) for p in _PATTERNS):
                bad.append(f"{rel}:{i}: {line.strip()}")
    assert bad == [], "\n".join(bad)


def test_entry_point_does_not_import_neuron():
    text = (ROOT / "personalize.py").read_text(encoding="utf-8")
    assert "import neuron" not in text
    assert "from neuron" not in text
