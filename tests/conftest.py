import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def nest_kernel():
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})
    return nest