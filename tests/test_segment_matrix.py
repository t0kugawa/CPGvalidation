import numpy as np

from persona.connection import Connection
from persona.parameter import PersonaParam
from persona.population import Population
from persona.segments import SegmentMatrix


class _FakeRegistry:
    def __init__(self):
        self.population_cfgs = {
            "A": Population("izhikevich", 4,
                            {"a": PersonaParam(1.0, 0.1, 0.0, 2.0)}, "A"),
            "B": Population("izhikevich", 4,
                            {"b": PersonaParam(0.2, 0.05, 0.0, 1.0)}, "B"),
            "C": Population("izhikevich", 4,
                            {"c": PersonaParam(-60.0, 1.0, -80.0, 0.0)}, "C"),
        }
        self.connection_cfgs = {
            "A_B": self._conn("A", "B", 1.0),
            "A_C": self._conn("A", "C", 0.5),
            "B_C": self._conn("B", "C", -0.5),
        }
        self._syn_counts = {"A_B": 4, "A_C": 3, "B_C": 2}

    @staticmethod
    def _conn(source, target, weight):
        lo, hi = (-2.0, 0.0) if weight < 0 else (0.0, 2.0)
        return Connection(
            source=source, target=target, model="static_synapse",
            parameters={"weight": PersonaParam(weight, 0.1, lo, hi)},
            rule="pairwise_bernoulli", p=0.5,
            extra_syn_spec={"weight": weight}, key=f"{source}_{target}")

    def get_all_optimizable_params(self):
        out = {}
        for name, pop in self.population_cfgs.items():
            for param in pop.params:
                out[f"pop_{name}_{param}"] = pop.params[param]
        for key, conn in self.connection_cfgs.items():
            for param in conn.parameters:
                out[f"conn_{key}_{param}"] = conn.parameters[param]
        return out

    def relay_links(self):
        return {}


def test_segment_extraction_and_edges():
    matrix = SegmentMatrix.from_registry(_FakeRegistry())
    assert [s.name for s in matrix.segments] == [
        "con_A_B", "con_A_C", "con_B_C", "pop_A", "pop_B", "pop_C"]
    assert matrix.segment_of_site["pop_A_a"] == 3
    assert "incidence" in matrix.edge_types[(0, 3)]
    assert "antagonism" in matrix.edge_types[(1, 2)]
    assert "shared_source" in matrix.edge_types[(0, 1)]
    assert "shared_target" in matrix.edge_types[(1, 2)]
    assert np.allclose(matrix.distance, matrix.distance.T)
    assert np.all(np.diag(matrix.distance) == 0.0)
    assert matrix.n_segments == 6
    assert matrix.n_sites == 6


def test_nearest_segment_and_partition_validation():
    matrix = SegmentMatrix.from_registry(_FakeRegistry())
    assert matrix.nearest_segment(3, [0, 1]) == 0
    assert matrix.nearest_player(3, [[0], [1, 2]]) == 0
    matrix.validate_partition(((0, 1, 2), (3, 4, 5)))
    try:
        matrix.validate_partition(((0, 1), (1, 2)))
    except ValueError as exc:
        assert "two players" in str(exc)
    else:
        raise AssertionError("overlap partition was accepted")


def test_nest_inventory_and_relay_drive(nest_kernel):
    from t2xcpg.bs import BaseDrive, BrainStem
    from t2xcpg.cpg import L2CPG
    from t2xcpg.leg import Leg
    from persona.registry import Registry

    registry = Registry()
    bs = BrainStem(registry=registry, n_bs=4)
    base = BaseDrive(registry=registry, n_bs=4)
    legs = {side: Leg(side, registry, n_rg=4, n_mn=4, n_int=4, n_aff=3)
            for side in "LR"}
    L2CPG(registry=registry, bs=bs, legs=legs, dt_ms=10.0, base=base)
    matrix = SegmentMatrix.from_registry(registry)

    expected_segments = (
        sum(1 for pop in registry.population_cfgs.values() if pop.params)
        + sum(1 for conn in registry.connection_cfgs.values()
              if conn.parameters)
    )
    assert matrix.n_segments == expected_segments
    assert matrix.n_sites == len(registry.get_all_optimizable_params())
    assert set(matrix.segment_of_site) == set(
        registry.get_all_optimizable_params())
    names = {segment.name for segment in matrix.segments}
    assert "con_in_e_L_m_f_L" in names
    assert "con_in_f_L_m_e_L" in names

    by_name = {segment.name: segment.id for segment in matrix.segments}
    for rate_name, efferents in (
        ("pop_BS_E", ("con_BS_in_e_rg_e_L", "con_BS_in_e_rg_e_R")),
        ("pop_BS_F", ("con_BS_in_f_rg_f_L", "con_BS_in_f_rg_f_R")),
    ):
        rate_id = by_name[rate_name]
        for efferent in efferents:
            assert np.isfinite(matrix.distance[rate_id, by_name[efferent]])
            assert "relay_drive" in matrix.edge_types[
                (min(rate_id, by_name[efferent]),
                 max(rate_id, by_name[efferent]))]
    base_id = by_name["pop_BASE_pg"]
    for efferent in ("con_BASE_in_rg_e_L", "con_BASE_in_rg_f_R"):
        assert np.isfinite(matrix.distance[base_id, by_name[efferent]])
