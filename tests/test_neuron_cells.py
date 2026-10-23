"""NEURON cell-factory, path, and connection identity tests."""
import numpy as np
import pytest

pytest.importorskip("neuron")

from persona.connection import Connection
from persona.homeostasis import PhysicalMap
from persona.neuron_adapter import NeuronAdapter
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population


@pytest.fixture()
def reg():
    NeuronAdapter.reset_kernel(resolution_ms=0.05, v_init=-65.0)
    reset_rng_stream(0)
    return NeuronAdapter(seed=7)


def _pop(name, model="interneuron", n=1, **neuron):
    options = {"n_synapses": 4}
    options.update(neuron)
    return Population(model, n, {}, name, neuron=options)


def test_all_builtin_cell_types_create(reg):
    expected = {
        "interneuron": {"soma", "dend", "axon"},
        "bursting_interneuron": {"soma", "dend", "axon"},
        "motoneuron": {"soma", "dend", "axon", "node", "MYSA", "FLUT", "STIN"},
        "muscle": {"muscle_unit", "soma"},
        "afferent": {"soma", "axon", "node", "MYSA", "FLUT", "STIN"},
    }
    for i, (model, sections) in enumerate(expected.items()):
        group = reg.create_population(_pop(f"p{i}", model, n=2))
        assert len(group) == 2
        assert set(group[0].sections) == sections
        assert group[0].syn_pools["exc"]


def test_reference_paths_resolve_on_each_type(reg):
    paths = {
        "interneuron": [
            "soma.gnabar_fastchannels", "soma.gkbar_fastchannels",
            "soma.gl_fastchannels", "soma.el_fastchannels", "soma.cm",
            "soma.Ra", "dend.g_pas", "dend.e_pas", "dend.cm", "dend.Ra",
            "axon.Ra", "axon.gnabar_hh",
        ],
        "bursting_interneuron": ["soma.gbar_nap"],
        "motoneuron": [
            "soma.gnabar_motoneuron", "soma.gcaL_motoneuron",
            "soma.gl_motoneuron", "soma.gkrect_motoneuron",
            "soma.gcak_motoneuron", "soma.cm", "soma.Ra", "dend.g_pas",
            "node.gnabar_hh", "MYSA.g_pas", "MYSA.e_pas", "FLUT.g_pas",
            "STIN.g_pas", "soma.ena", "soma.ek",
        ],
        "muscle": [
            "muscle_unit.cm", "muscle_unit.g_pas", "muscle_unit.e_pas",
            "muscle_unit.gnabar_fastchannels", "soma.gmax_cac1",
            "soma.gbar_na14a", "soma.gkbar_kir", "soma.gnabar_fastchannels",
            "soma.gkmax_K_No", "soma.gcalbar_cal",
        ],
        "afferent": [
            "soma.gnabar_hh", "soma.gkbar_hh", "soma.gl_hh", "soma.el_hh",
            "node.gnabar_hh", "MYSA.g_pas", "FLUT.g_pas", "STIN.g_pas",
        ],
    }
    for i, (model, names) in enumerate(paths.items()):
        group = reg.create_population(_pop(f"p{i}", model))
        for name in names:
            assert group.has_param(name), (model, name)
            group.set({name: 0.001})
            assert abs(float(group.get(name)[0]) - 0.001) < 1e-12


def test_motoneuron_diam_is_per_cell_and_build_time(reg):
    group = reg.create_population(
        _pop("moto", "motoneuron", n=2, diam=[20.0, 60.0])
    )
    assert np.isclose(float(group[0].sections["soma"][0].diam), 20.0)
    assert np.isclose(float(group[1].sections["soma"][0].diam), 60.0)
    # diameter is a build-time structural seed, not a runtime site
    with pytest.raises((ValueError, AttributeError)):
        group.set({"soma.diam": 30.0})


def test_structural_and_unknown_paths_raise(reg):
    group = reg.create_population(_pop("bad"))
    with pytest.raises(ValueError):
        group.set({"soma.nseg": 2})
    with pytest.raises(ValueError):
        group.set({"soma.not_a_range_variable": 1.0})
    with pytest.raises(ValueError):
        reg.create_population(
            Population("interneuron", 1,
                       {"soma.not_a_range_variable": PersonaParam(1.0)},
                       "bad2", neuron={"n_synapses": 2})
        )


def test_bracketed_site_and_freeze_resolve_exactly(reg):
    pop = Population(
        "interneuron", 1,
        {"dend[0].g_pas": PersonaParam(
            mean=0.0002, sigma=1e-7, min=0.0, max=1.0)},
        "br", neuron={"n_synapses": 2, "n_dend": 3},
    )
    reg.create_population(pop)
    key = "pop_br_dend[0].g_pas"
    assert reg.freeze(key) == (key,)
    assert reg.is_frozen(key, "mean")
    assert reg.defaults(key)["mean"] == 0.0002


def test_pool_exhaustion_raises(reg):
    reg.create_population(_pop("src", n=2, n_synapses=1))
    reg.create_population(_pop("tgt", n=1, n_synapses=1))
    conn = Connection("src", "tgt", model="static_exc",
                      parameters={"weight": PersonaParam(1.0)},
                      rule="all_to_all", key="over")
    with pytest.raises(RuntimeError):
        reg.create_connection(conn)


def test_inhibitory_projection_is_classified_by_physical_map(reg):
    reg.create_population(_pop("src"))
    reg.create_population(_pop("tgt"))
    p = PersonaParam(mean=3.0, sigma=0.1, min=-5.0, max=5.0)
    reg.create_connection(Connection(
        "src", "tgt", model="static_inh",
        parameters={"weight": p}, rule="all_to_all", key="inh",
    ))
    assert p.mean < 0.0
    assert np.all(reg.synapses("inh").get("weight") > 0.0)
    assert np.all(reg.synapses("inh").get("signed_weight") < 0.0)
    pm = PhysicalMap.from_registry(reg)
    spec = next(s for s in pm.specs if s.name == "con_inh")
    assert spec.ptype == "static_inh"
