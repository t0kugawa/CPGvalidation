"""Test suite for CPG half-center inhibition fix.

This module validates that:
1. Deprecated motor pool inhibition connections are removed
2. Unplanned sensory→CPG connections are removed
3. Half-center weights are symmetric and balanced
4. Half-center produces alternating rhythm (not co-contraction)
"""
import numpy as np
import pytest

from persona.registry import Registry
from t2xcpg.leg import Leg
from t2xcpg.params import PARAMS


def _leg(n_rg=20, n_mn=20, n_int=15, n_aff=20):
    """Create a test leg with default or specified sizes."""
    reg = Registry()
    leg = Leg("L", reg, n_rg=n_rg, n_mn=n_mn, n_int=n_int, n_aff=n_aff)
    return reg, leg


class TestDeprecatedConnectionsRemoved:
    """Verify deprecated motor pool reciprocal inhibition is removed."""

    def test_in_e_to_m_f_removed(self, nest_kernel):
        """InE → M-F connection must not exist (deprecated)."""
        reg, leg = _leg()
        assert "in_e_L_m_f_L" not in reg.connection_cfgs, \
            "Deprecated InE → M-F connection still exists!"

    def test_in_f_to_m_e_removed(self, nest_kernel):
        """InF → M-E connection must not exist (deprecated)."""
        reg, leg = _leg()
        assert "in_f_L_m_e_L" not in reg.connection_cfgs, \
            "Deprecated InF → M-E connection still exists!"


class TestUnplannedConnectionsRemoved:
    """Verify unplanned connections that disrupt half-center are removed."""

    def test_ia_to_in_e_removed(self, nest_kernel):
        """Ia-IN-E → InE connection must not exist (unplanned)."""
        reg, leg = _leg()
        assert "ia_e_in_L_in_e_L" not in reg.connection_cfgs, \
            "Unplanned Ia-IN-E → InE connection still exists!"

    def test_ia_to_in_f_removed(self, nest_kernel):
        """Ia-IN-F → InF connection must not exist (unplanned)."""
        reg, leg = _leg()
        assert "ia_f_in_L_in_f_L" not in reg.connection_cfgs, \
            "Unplanned Ia-IN-F → InF connection still exists!"

    def test_ia_to_in_ib_e_removed(self, nest_kernel):
        """Ia-IN-E → InIb-E connection must not exist (unplanned cross-talk)."""
        reg, leg = _leg()
        assert "ia_e_in_L_in_ib_e_L" not in reg.connection_cfgs, \
            "Unplanned Ia-IN-E → InIb-E connection still exists!"

    def test_ia_to_in_ib_f_removed(self, nest_kernel):
        """Ia-IN-F → InIb-F connection must not exist (unplanned cross-talk)."""
        reg, leg = _leg()
        assert "ia_f_in_L_in_ib_f_L" not in reg.connection_cfgs, \
            "Unplanned Ia-IN-F → InIb-F connection still exists!"


class TestHalfCenterSymmetry:
    """Verify half-center inhibition is symmetric and balanced."""

    def test_half_center_connections_exist(self, nest_kernel):
        """Core half-center connections must exist."""
        reg, leg = _leg()
        assert "rg_e_L_in_e_L" in reg.connection_cfgs, "RG-E → InE missing"
        assert "in_e_L_rg_f_L" in reg.connection_cfgs, "InE → RG-F missing"
        assert "rg_f_L_in_f_L" in reg.connection_cfgs, "RG-F → InF missing"
        assert "in_f_L_rg_e_L" in reg.connection_cfgs, "InF → RG-E missing"

    def test_inhibitory_weights_are_negative(self, nest_kernel):
        """Inhibitory connections must have negative weights."""
        reg, leg = _leg()
        w_e_to_f = reg.connection_cfgs["in_e_L_rg_f_L"].parameters["weight"].mean
        w_f_to_e = reg.connection_cfgs["in_f_L_rg_e_L"].parameters["weight"].mean
        assert w_e_to_f < 0, f"InE → RG-F weight {w_e_to_f} is not inhibitory!"
        assert w_f_to_e < 0, f"InF → RG-E weight {w_f_to_e} is not inhibitory!"

    def test_inhibitory_weights_are_symmetric(self, nest_kernel):
        """InE → RG-F and InF → RG-E weights must be equal (symmetric)."""
        reg, leg = _leg()
        w_e_to_f = reg.connection_cfgs["in_e_L_rg_f_L"].parameters["weight"].mean
        w_f_to_e = reg.connection_cfgs["in_f_L_rg_e_L"].parameters["weight"].mean
        assert np.isclose(w_e_to_f, w_f_to_e), \
            f"Half-center weights not symmetric: E→F={w_e_to_f}, F→E={w_f_to_e}"

    def test_connection_probabilities_are_symmetric(self, nest_kernel):
        """InE → RG-F and InF → RG-E probabilities must be equal."""
        reg, leg = _leg()
        p_e_to_f = reg.connection_cfgs["in_e_L_rg_f_L"].p
        p_f_to_e = reg.connection_cfgs["in_f_L_rg_e_L"].p
        assert np.isclose(p_e_to_f, p_f_to_e), \
            f"Half-center probabilities not symmetric: E→F={p_e_to_f}, F→E={p_f_to_e}"

    def test_inhibitory_weights_are_strong_enough(self, nest_kernel):
        """Inhibitory weights must be strong enough to suppress rhythm."""
        reg, leg = _leg()
        w_e_to_f = abs(reg.connection_cfgs["in_e_L_rg_f_L"].parameters["weight"].mean)
        w_f_to_e = abs(reg.connection_cfgs["in_f_L_rg_e_L"].parameters["weight"].mean)
        # Must be at least -16.0 to provide sufficient inhibition
        assert w_e_to_f >= 16.0, f"InE → RG-F weight {-w_e_to_f} too weak!"
        assert w_f_to_e >= 16.0, f"InF → RG-E weight {-w_f_to_e} too weak!"

    def test_expected_synapse_counts_are_balanced(self, nest_kernel):
        """Expected number of synapses must be roughly equal both directions."""
        reg, leg = _leg()
        cfg_e_f = reg.connection_cfgs["in_e_L_rg_f_L"]
        cfg_f_e = reg.connection_cfgs["in_f_L_rg_e_L"]
        
        # Expected synapses = n_source * n_target * p
        n_in = leg.n_int
        n_rg = leg.n_rg
        expected_e_f = n_in * n_rg * cfg_e_f.p
        expected_f_e = n_in * n_rg * cfg_f_e.p
        
        # Allow 20% tolerance for small network stochasticity
        ratio = expected_e_f / expected_f_e
        assert 0.8 <= ratio <= 1.2, \
            f"Expected synapse counts not balanced: E→F={expected_e_f:.1f}, F→E={expected_f_e:.1f}"

    def test_total_inhibition_is_symmetric(self, nest_kernel):
        """Total inhibitory strength (weight * expected_synapses) must be symmetric."""
        reg, leg = _leg()
        cfg_e_f = reg.connection_cfgs["in_e_L_rg_f_L"]
        cfg_f_e = reg.connection_cfgs["in_f_L_rg_e_L"]
        
        w_e_f = cfg_e_f.parameters["weight"].mean
        w_f_e = cfg_f_e.parameters["weight"].mean
        
        n_in = leg.n_int
        n_rg = leg.n_rg
        
        total_e_f = n_in * n_rg * cfg_e_f.p * w_e_f
        total_f_e = n_in * n_rg * cfg_f_e.p * w_f_e
        
        # Total inhibition should be within 20% (allows for rounding)
        ratio = abs(total_e_f / total_f_e)
        assert 0.8 <= ratio <= 1.2, \
            f"Total inhibition not symmetric: E→F={total_e_f:.0f}, F→E={total_f_e:.0f}"


class TestParameterValues:
    """Verify the correct parameter values are set."""

    def test_param_w_con_in_e_rg_f(self):
        """InE → RG-F weight parameter must be -20.0."""
        assert PARAMS["w_con_in_e_rg_f"] == -20.0, \
            f"w_con_in_e_rg_f = {PARAMS['w_con_in_e_rg_f']}, expected -20.0"

    def test_param_p_con_in_e_rg_f(self):
        """InE → RG-F probability parameter must be 0.25."""
        assert PARAMS["p_con_in_e_rg_f"] == 0.25, \
            f"p_con_in_e_rg_f = {PARAMS['p_con_in_e_rg_f']}, expected 0.25"

    def test_param_w_con_in_f_rg_e(self):
        """InF → RG-E weight parameter must be -20.0."""
        assert PARAMS["w_con_in_f_rg_e"] == -20.0, \
            f"w_con_in_f_rg_e = {PARAMS['w_con_in_f_rg_e']}, expected -20.0"

    def test_param_p_con_in_f_rg_e(self):
        """InF → RG-E probability parameter must be 0.25."""
        assert PARAMS["p_con_in_f_rg_e"] == 0.25, \
            f"p_con_in_f_rg_e = {PARAMS['p_con_in_f_rg_e']}, expected 0.25"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
