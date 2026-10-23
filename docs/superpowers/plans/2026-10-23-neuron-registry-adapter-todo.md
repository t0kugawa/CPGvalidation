# NEURON registry adapter — implementation TODO

**Owner:** the NEURON implementation team.
**Contract:** `docs/superpowers/specs/2026-10-23-neuron-registry-adapter-design.md`.
**Status:** implemented 2026-10-23. The NEST regression suite and the focused
NEURON suite (architecture, API parity, route scan, cells, groups,
apply/relay, STDP when `stdp.mod` is compiled) are green. The checkboxes below
remain the detailed contract checklist; see the implementation note at the end
of this header for what is deliberately simplified.
**Environment:** WSL venv `/home/t0kgw/.virtualenvs/KinematicsCPG`
(repo `/mnt/d/CPGvalidation`). NEURON 9.0.2 per the existing spec. The
`memcpg/mod_files` mechanisms must be compiled with `nrnivmodl` before the
integration/smoke tests.

This TODO is intentionally written against the NEST registry, not against a
new abstraction. If an item cannot be done without changing
`persona/registry.py`, `Agent`, `Personalizer`, `PhysicalMap`,
`HomeostaticRegulator`, `t2xcpg/`, or `memcpg/`, stop and raise it as an API
defect rather than working around it.

**Implementation note (2026-10-23).** The built-in five cell types are
section/hh/pas equivalences of the memcpg reference, because the delivery does
not modify or require `memcpg`; the canonical paths from §17.1 resolve and the
model may register a factory for the full biophysical mechanisms. `stdp_exc`
and `stdp_inh` require the reference `stdp.mod` to be compiled (`nrnivmodl`);
without it the adapter raises a clear `RuntimeError` and
`tests/test_neuron_stdp.py` skips. v1 is single-process; the stable-ID and
group interfaces are MPI-ready but no `ParallelContext` solve is implemented.
The focused test files are `test_adapter_architecture.py`,
`test_neuron_adapter.py`, `test_neuron_api_surface.py`,
`test_neuron_route.py`, `test_neuron_cells.py`, `test_neuron_groups.py`, and
the optional `test_neuron_stdp.py`.

---

## 0. Definition of done

- [ ] `persona/adapter.py` defines `class Adapter(ABC)` with the full common
      interface; every common method is `@abstractmethod`.
- [ ] `persona/nest_adapter.py` defines `class NestAdapter(Adapter)` and
      implements every abstract method; its NEST behavior is unchanged.
- [ ] `persona/neuron_adapter.py` defines `class NeuronAdapter(Adapter)` and
      implements every abstract method.
- [ ] `persona.registry.Registry` remains a compatibility subclass/alias of
      `NestAdapter`; `persona.neuron.NeuronRegistry` is a compatibility alias
      of `NeuronAdapter`.
- [ ] No `SiteBook` mixin or other shared-state base class exists.
- [ ] `Agent(NeuronRegistry(), ...)` constructs and runs 5 finite steps.
- [ ] `Personalizer` can run a 2-step loop with a small NEURON network without
      importing `neuron` in `personalize.py`.
- [ ] Freeze, spring, site recorders and producer recorders behave identically
      to the NEST registry for the same site table.
- [ ] `static_exc`, `static_inh`, `stdp_exc` and the five memcpg cell types are
      built in and covered by the smoke test.
- [ ] `tests/test_neuron_route.py` passes (single-route source scan).
- [ ] The full existing NEST suite passes unchanged.
- [ ] `t2xcpg/`, `memcpg/`, `tinycpg/` are not modified.
- [ ] The shared-file changes are limited to:
      (a) `persona/adapter.py` (new pure ABC + module helpers);
      (b) `persona/nest_adapter.py` (NEST implementation moved verbatim);
      (c) `persona/registry.py` (thin compatibility shim);
      (d) additive `neuron: dict` fields on `Population`/`Connection`;
      (e) behavior-preserving helper moves, exact-key-first resolution, and the
          hard inhibitory sign normalizer.
      Any other shared change is an API defect.

---

## 1. Do-not-touch list

- [ ] `persona/nest_adapter.py` — the NEST adapter; preserve its public
      behavior. `persona/registry.py` becomes a compatibility shim only.
- [ ] `persona/agent.py`, `persona/strategy.py`, `persona/homeostasis.py`,
      `persona/game.py`, `persona/personalization.py` — no simulator-specific
      code.
- [ ] `persona/parameter.py`, `persona/population.py`,
      `persona/connection.py` — only additive fields, no semantic changes.
- [ ] `t2xcpg/` — the NEST model.
- [ ] `memcpg/` — the reference model; read-only.
- [ ] `tinycpg/` — read-only.
- [ ] `personalize.py` — no NEURON imports; the entry point may grow a
      backend switch later, out of scope here.

---

## 2. Phase 0 — freeze the interface

- [ ] Read the spec completely. The decisions in §20 are resolved: abstract
      `Adapter`, separate NEST/NEURON implementations, all five cell types,
      memcpg step-sampled STDP, single-process v1, NEURON-native names, signed
      weights, registry-owned kernel.
- [ ] Create `persona/adapter.py`:
  - [ ] `class Adapter(ABC)` with the full common interface (creation,
        realization, recorders, site bookkeeping, freeze, springs, noise,
        sampling, kernel).
  - [ ] Every common method/property/classmethod is `@abstractmethod`.
  - [ ] No `import nest`, no `import neuron` in this module.
  - [ ] Shared pure module-level helpers (`clamp_weight_sign`,
        `plastic_delta_from_events`, `resolve_site_keys`,
        `enforce_synaptic_sign`) live here. No `SiteBook` mixin.
- [ ] Move the existing NEST implementation to
      `persona/nest_adapter.py` as `class NestAdapter(Adapter)`; make
      `persona/registry.py` a thin `Registry(NestAdapter)` shim and prove the
      NEST suite still passes. Add kernel delegations (`reset_kernel`,
      `set_resolution`, `initialize`, `simulate`, `time_ms`).
- [ ] Create `persona/neuron_adapter.py` with
      `class NeuronAdapter(Adapter)` and the `persona/neuron/` helper package.
- [ ] Add the API parity test:
      `tests/test_neuron_api_surface.py`.
  - [ ] `Adapter`, `NestAdapter`, `NeuronAdapter`, `Registry`,
        `NeuronRegistry` importable.
  - [ ] `NestAdapter` and `NeuronAdapter` subclass `Adapter` and implement
        every abstract method.
  - [ ] `inspect.signature` parameter names/kinds/defaults match for every
        shared method, except `__init__` (additive keyword-only `seed`/`kernel`
        are allowed).
  - [ ] Public attributes exist: `population_cfgs`, `connection_cfgs`,
        `population_nodes`, `noise_recorders`, `last_apply_errors`.
  - [ ] Test skips cleanly if NEURON or NEST is unavailable.
- [ ] Add the single-route source-scan test skeleton:
      `tests/test_neuron_route.py` (it will fail until the package exists).
- [ ] Add the `neuron: dict` field to `Population` and `Connection` (or the
      agreed equivalent) and prove the NEST suite still passes.
- [ ] Add exact-key-first resolution to the shared `resolve_site_keys` helper
      called by both adapters. Regression test:
      `freeze("pop_x_dend[0].g_pas")` resolves the exact key; a glob still
      resolves the same set as before. NEST suite passes.
- [ ] Add a tiny `tests/test_neuron_groups.py` that exercises `CellGroup`,
      `SynapseGroup`, `RateGroup`, `RelayGroup`, `SpikeRecorder`,
      `WeightRecorder` contracts with a fake cell/synapse factory. No real
      simulation required. Include the `SpikeSource(ref, sec, threshold)`
      contract and the `SynapseContext`/`SynapseHandle` contract.

**Exit:** the interface is frozen and testable without a full NEURON network.

---

## 3. Phase 1 — package and kernel facade

- [ ] Create `persona/neuron/__init__.py` exporting `NeuronRegistry` and the
      protocols.
- [ ] Kernel methods are declared on `Adapter`; implement the NEST delegates in
      `Registry` and the NEURON facade in `persona/neuron/kernel.py`.
  - [ ] NEST: `reset_kernel` → `nest.ResetKernel()` / `SetKernelStatus`,
        `set_resolution` → `SetKernelStatus({"resolution": ms})`,
        `initialize` → no-op, `simulate` → `nest.Simulate(dt_ms)`,
        `time_ms` → `GetKernelStatus("biological_time")`.
  - [ ] NEURON: `reset_kernel(resolution_ms=0.025, v_init=-65.0, cvode=False, maxstep_ms=None)`
        — kernel-only, callable before a registry exists. It MUST NOT reach
        into an instance. The supported pattern is
        `reset_kernel(...)` then a new `NeuronRegistry()`.
  - [ ] `set_resolution(ms)`
  - [ ] `set_maxstep(ms)`
  - [ ] `set_neuron_seed(seed)`
  - [ ] `initialize(v_init=None)`
  - [ ] `simulate(dt_ms)` → new `time_ms`
  - [ ] `time_ms`, `dt_ms`, `rng`, `h`, `parallel_context` properties
  - [ ] `reset_kernel` safe to call repeatedly; a stale registry instance
        raises a clear `RuntimeError` instead of reading dangling handles.
- [ ] `NeuronRegistry.__init__(*, seed=None)` wires the kernel, type tables,
      site tables and RNG.
- [ ] One registry per process; add a clear error on a second live registry
      if the team chooses to enforce it.
- [ ] Tests:
  - [ ] `reset_kernel` + new registry + `initialize` + `simulate(10)` advances
        `time_ms` by 10 ± tolerance.
  - [ ] `simulate` works with a single `h.Section`/`IntFire` and no
        connections.
  - [ ] `set_resolution` before initialize; change after initialize raises.
  - [ ] `NeuronRegistry(seed=1)` and `NeuronRegistry(seed=1)` produce the same
        connectivity and the same built-in cell morphology.
  - [ ] `set_neuron_seed` seeds a `h.Random`/`NetStim.noiseFromRandom` path
        used by the smoke test.

**Exit:** a trivial NEURON network can be stepped through the registry.

---

## 4. Phase 2 — groups, recorders, relays

- [ ] `persona/neuron/groups.py`:
  - [ ] `CellGroup`: `__len__`, `__iter__`, `__getitem__`, `set`, `get`,
        `has_param`, `param_paths`, `spike_source`.
  - [ ] `SynapseGroup`: `__len__`, `__iter__`, `__getitem__`, `set`, `get`,
        `param_paths`.
  - [ ] `RateGroup`: `__len__`, `__iter__`, `__getitem__`, `set`, `get`;
        `rate` ↔ `interval` conversion; `rate=0` disables safely. Implement
        the reference recipe: `start=0`, `number=1_000_000_000`, `noise=1.0`,
        `interval=1000/rate`; remember/restore `number` across disable/enable.
  - [ ] `RelayGroup`: `__len__`, `__iter__`, `__getitem__`, `set`, `get`,
        `spike_source`, `kind` (`"forward"` / `"tap"`). `create_relay` must be
        a forwarding relay; `create_parrot_nodes` is a readout tap.
- [ ] `persona/neuron/recorders.py`:
  - [ ] `SpikeRecorder`: `get("events")` → `times`/`senders`;
        `set({"n_events": 0})`; `len`; strong refs to NetCon/Vector.
  - [ ] `WeightRecorder`: `get("events")` → `times`/`weights`/`senders`/
        `targets`; `set({"n_events": 0})`; `len`.
- [ ] Registry methods:
  - [ ] `register_nodes`
  - [ ] `create_parrot_nodes`
  - [ ] `create_spike_recorder`
  - [ ] `create_weight_recorder`
- [ ] Tests:
  - [ ] A group of `IntFire`/`Section` cells records spikes.
  - [ ] Clearing returns empty arrays and `n_events == 0`.
  - [ ] A `RelayGroup` can be a recorder target and a `Connection.source`.
  - [ ] `RateGroup.set({"rate": 0})` does not divide by zero.
  - [ ] `RateGroup` round-trips `rate` ↔ `interval`.

**Exit:** groups and recorders satisfy the call-site contract from the spec
§8, with fake or trivial NEURON cells.

---

## 5. Phase 3 — cell factories and path resolver

- [ ] `persona/neuron/cells.py`:
  - [ ] `CellFactory` protocol with
        `create(registry, cfg: Population)` (not just `n`), so
        `motoneuron(diam)`, `muscle(n_synapses, nseg)`,
        `afferent(neuron_type, n_synapses, n_axon_nodes)` and
        `interneuron(serotonin_mode, bursting_mode)` can read `cfg.neuron`.
        `Population.neuron` values may be scalars or per-cell sequences.
  - [ ] `SpikeSource(ref, sec, threshold)`; `spike_source(cell)` returns it.
        `memcpg` thresholds: `-20` neurons, `+10` muscle.
  - [ ] `PathSpec` / path resolver:
    - [ ] `section.attr`
    - [ ] `section(x).attr`
    - [ ] `section[i].attr`
    - [ ] `section_list.attr` broadcast
    - [ ] `syn.attr`, `nc.attr`, `stdp.attr`, `stim.attr`
    - [ ] unknown path → `ValueError`
    - [ ] structural path at runtime → `ValueError`
  - [ ] `interneuron` (including bursting and serotonin variants).
  - [ ] `motoneuron` (diameter-dependent seed; `diam` is build-time).
  - [ ] `muscle`.
  - [ ] `afferent` (`bioaffrat`).
  - [ ] Optional `izhikevich` only if a NEURON mechanism is supplied.
- [ ] `NeuronRegistry.register_cell_type`, `cell_type`.
- [ ] `create_population` implements spec §9.1.
- [ ] Tests:
  - [ ] Each built-in type creates `n` cells with the expected sections and
        mechanisms.
  - [ ] `motoneuron` receives a per-cell `diam` sequence from `cfg.neuron`;
        the conductance seed matches `diam > 50`.
  - [ ] `CellGroup.set` applies scalar and per-cell sequences.
  - [ ] Every path in spec §17.1 resolves on the reference type.
  - [ ] An unknown path raises at creation.
  - [ ] A structural path (`nseg`, `insert`) raises.
  - [ ] `create_population` records one site per param and realizes the
        initial `sample_matrix`.
  - [ ] `last_apply_errors` surfaces an injected set failure.

**Exit:** the five memcpg cell types can be created and parameterized through
the registry.

---

## 6. Phase 4 — synapse factories and connectivity

- [ ] `persona/neuron/synapses.py`:
  - [ ] `SynapseFactory` protocol with
        `create(registry, ctx: SynapseContext) -> SynapseHandle`; the registry
        stores the handle. The context carries `conn_key`, both cells,
        `presyn_source`, allocated `postsyn_pool`, signed seed weight, delay,
        params, weight recorder, threshold, stable IDs.
  - [ ] `static_exc` (`ExpSyn`, `synlistex`, signed weights).
  - [ ] `static_inh` (`Exp2Syn`, `synlistinh`, signed weights).
  - [ ] `static_synapse` sign dispatcher: positive → `static_exc`, negative →
        `static_inh`, zero → `ValueError`. This is the NEST-compatible name.
  - [ ] `stdp_exc` (`ExpSyn` + `STDP`, `synlistexstdp`) — creation wiring in
        §11.1.
  - [ ] `stdp_synapse` alias for `stdp_exc`.
  - [ ] `stdp_inh` only if an inhibitory STDP mechanism is supplied.
  - [ ] `weight_sign_mode` default `"signed"`; `"magnitude"` opt-in only.
        Signed site value maps to `NetCon.weight[0] = abs(value)`; the factory
        validates the sign against its `kind`.
- [ ] `NeuronRegistry.register_synapse_type`, `synapse_type`.
- [ ] `create_connection` implements spec §9.2, §10.
  - [ ] `all_to_all`, `one_to_one`, `pairwise_bernoulli`.
  - [ ] `fixed_indegree`, `fixed_outdegree` with jitter.
  - [ ] `allow_autapses`, `allow_multapses`.
  - [ ] `delay`, `delay_cv`, `weight`, `weight_cv`, `threshold`, `seed`.
        Threshold defaults to the presynaptic `SpikeSource.threshold`; the
        STDP teaching NetCons keep NEURON's default.
  - [ ] synapse pool allocation and exhaustion error.
  - [ ] `synapses`, `synapse_count`, `finalize_connections`.
  - [ ] producer recorder for `weight_recorder` with the resolved
        `Wmax`/`wmax`/`stdp.wmax` scale (first present).
- [ ] Tests:
  - [ ] Each rule produces the expected count on a 3×4 network.
  - [ ] `pairwise_bernoulli` seed is reproducible.
  - [ ] `fixed_indegree` with jitter matches the requested range.
  - [ ] `allow_autapses`/`allow_multapses` honored.
  - [ ] delay/weight CV produce per-synapse dispersion.
  - [ ] pool exhaustion raises, never drops.
  - [ ] `synapse_count` equals the number of NetCons.
  - [ ] `static_inh` with a **negative** site value is inhibitory and its
        `NetCon.weight[0]` is the positive magnitude.
  - [ ] `static_synapse` with a negative seed dispatches to `static_inh`; with
        a positive seed dispatches to `static_exc`; zero raises.
  - [ ] `HomeostaticRegulator`/`PhysicalMap` classify the inhibitory
        projection correctly from the negative `PersonaParam.mean`.
  - [ ] `signed` mode clamps the sign; `magnitude` mode is opt-in and tested
        only for a model that does not use H/game.

**Exit:** static connectivity and the STDP construction are complete.

---

## 7. Phase 5 — site bookkeeping, apply, freeze, spring

- [ ] Implement the common pure-Python site behavior in both concrete
      adapters, using the module-level helpers from `persona/adapter.py`:
  - [ ] `_record_defaults`, `recapture_defaults`
  - [ ] `_key_patterns`, `_attrs_tuple`, `_resolve_site_keys` (exact-first)
  - [ ] `freeze`, `unfreeze`, `frozen_attrs`, `is_frozen`, `defaults`
  - [ ] `create_spring`, `remove_spring`, `springs`, `spring_state`
  - [ ] `site_recorder`, `producer_recorder`, `producer_recorders`
  - [ ] `get_all_optimizable_params`, `sample_all_actions`
- [ ] `NestAdapter` keeps the NEST behavior exactly. Prove the NEST suite
      still passes.
- [ ] `apply_explored_parameters` implements spec §9.6 exactly.
  - [ ] population path first, then connection path.
  - [ ] static per-synapse sampling.
  - [ ] plastic scalar hyper-parameters.
  - [ ] zero-synapse error shape.
  - [ ] `last_apply_errors` clear/append.
  - [ ] `recorded = applied`.
- [ ] Tests:
  - [ ] Copy `tests/test_registry_freeze.py` to a NEURON variant; same
        assertions.
  - [ ] Copy `tests/test_registry_spring.py` to a NEURON variant.
  - [ ] `tests/test_apply_realization.py` parity:
        population/dispersal, static synapse array, relay rate, errors.
  - [ ] `Agent(NeuronRegistry())` 5-step run.
  - [ ] `PhysicalMap.from_registry(neuron_registry)` builds the expected
        players and skipped endpoints.
  - [ ] `HomeostaticRegulator` and `StackGame` construct with the NEURON
        registry.

**Exit:** PERSONA's learning layer works unchanged on NEURON.

---

## 8. Phase 6 — STDP recording and plasticity

- [ ] Use the memcpg recording that already exists: one
      `h.Vector().record(stdpmech._ref_synweight, 10.0)` per plastic synapse.
      Do **not** change `stdp.mod`; event-level recording is an optional
      future extension, not this delivery.
- [ ] `WeightRecorder`:
  - [ ] Wrap one vector per synapse and keep `(pre_id, post_id)`.
  - [ ] On read, diff each vector against the previous sample and synthesize
        one event per changed synapse: `times`, `weights`, `senders`,
        `targets`.
  - [ ] `set({"n_events": 0})` truncates the vectors and resets the baseline.
  - [ ] Sampling interval defaults to `10.0 ms` (memcpg); configurable.
  - [ ] A step shorter than the sampling interval yields no new event and a
        `δ` of 0, never a stale read.
- [ ] `read_plastic_events`, `sample_plastic_weight_deltas`,
      `last_plastic_counts`, `sample_plastic_weight_stats`,
      `final_plastic_weights` implement spec §9.9.
- [ ] Canonical STDP site keys are bare (`tauhebb`, `wmax`, ...);
      `stdp.*` aliases normalize to the same key. The personalizable set is
      spec §11.2, not state variables (`tlastpre`, `interval`, `deltaw`, ...).
- [ ] `Wmax`/`wmax`/`stdp.wmax` resolves the producer scale; the canonical
      stored key is `wmax`.
- [ ] Tests:
  - [ ] Event schema (`times`, `weights`, `senders`, `targets`).
  - [ ] Clear semantics.
  - [ ] `plastic_delta_from_events` parity on a synthetic batch.
  - [ ] Weight stays in `[0, wmax]`.
  - [ ] `sample_plastic_weight_stats` and `final_plastic_weights` match live
        `NetCon.weight[0]`.
  - [ ] `producer:stdp:<key>` receives the per-step delta.
  - [ ] A model using `wmax` gets a producer scale; a model using `Wmax` gets
        the same scale; a model with neither gets `None`.

**Exit:** the noise channel for STDP is available.

---

## 9. Phase 7 — memcpg smoke and integration

- [ ] Build a minimal network through `NeuronRegistry` only:
  - [ ] `interneuron` → `interneuron` (`static_exc`/`static_inh`).
  - [ ] `afferent` → `interneuron` (`stdp_exc`).
  - [ ] `motoneuron` → `muscle`.
  - [ ] A `RateGroup` drive.
  - [ ] One `SpikeRecorder` per population and one `WeightRecorder` per STDP
        projection.
- [ ] `simulate(10)`; assert finite spike counts and weight changes.
- [ ] Build a `Connection` with `rule="fixed_indegree"`,
      `indegree=5`, `delay=3`, `delay_cv=0.2`, `weight_cv=0.1`; assert
      per-synapse dispersion.
- [ ] `freeze("pop_<name>_dend[0].g_pas")` resolves the exact bracketed key
      (the shared exact-first fix).
- [ ] An inhibitory projection has a negative `PersonaParam.mean`,
      `NetCon.weight[0]` is positive, and `PhysicalMap` classifies it as
      `static_inh`.
- [ ] Run `Agent` 5 steps with a dummy 2-component output/target.
- [ ] Run `Personalizer` 2 steps with the same fake bench used by the NEST
      tests, or a minimal adapter; no `neuron` import in `personalize.py`.
- [ ] `tests/test_neuron_route.py` passes.
- [ ] Existing NEST suite passes unchanged.

**Exit:** the adapter is integrated at the same level as the NEST registry.

---

## 10. Phase 8 — MPI (only if in scope)

- [ ] Decide with the human partner (spec §20, decision 4).
- [ ] If in scope:
  - [ ] Global stable IDs across ranks.
  - [ ] `CellGroup`/`SynapseGroup` local views; `get`/`population_nodes`
        collective or replicated.
  - [ ] `synapse_count` consistent across ranks.
  - [ ] `apply_explored_parameters` uses the same scalar sample on all ranks.
  - [ ] Recorders aggregate or expose a documented local view.
  - [ ] `simulate` uses `pc.psolve`.
- [ ] If not in scope:
  - [ ] Document single-process only.
  - [ ] Keep `stable_id` and the group interfaces MPI-ready.

---

## 11. API parity checklist (every `Adapter` member)

Use this as the line-by-line checklist for `tests/test_neuron_api_surface.py`.
Every item must be implemented by both `Registry` and `NeuronRegistry`.

- [ ] `__init__(self)` (additive keyword-only `seed`/`kernel` allowed)
- [ ] attributes: `population_cfgs`, `connection_cfgs`, `population_nodes`,
      `noise_recorders`, `last_apply_errors`
- [ ] `reset_kernel`
- [ ] `set_resolution`
- [ ] `initialize`
- [ ] `simulate`
- [ ] `time_ms`, `dt_ms`
- [ ] `register_noise_recorder`
- [ ] `get_noise_recorder`
- [ ] `site_recorder`
- [ ] `producer_recorder`
- [ ] `producer_recorders`
- [ ] `recapture_defaults`
- [ ] `freeze`
- [ ] `unfreeze`
- [ ] `frozen_attrs`
- [ ] `is_frozen`
- [ ] `defaults`
- [ ] `create_spring`
- [ ] `remove_spring`
- [ ] `springs`
- [ ] `spring_state`
- [ ] `register_nodes`
- [ ] `create_parrot_nodes`
- [ ] `create_spike_recorder`
- [ ] `create_weight_recorder`
- [ ] `create_population`
- [ ] `create_connection`
- [ ] `create_relay`
- [ ] `set_relay_rate`
- [ ] `synapses`
- [ ] `synapse_count`
- [ ] `finalize_connections`
- [ ] `read_plastic_events`
- [ ] `sample_plastic_weight_deltas`
- [ ] `last_plastic_counts`
- [ ] `sample_plastic_weight_stats`
- [ ] `final_plastic_weights`
- [ ] `get`
- [ ] `get_population_cfg`
- [ ] `get_all_optimizable_params`
- [ ] `sample_all_actions`
- [ ] `apply_explored_parameters`
- [ ] `clamp_weight_sign` (module-level or re-exported)
- [ ] `plastic_delta_from_events` (shared, not reimplemented)

NEURON-only (declared on `NeuronRegistry`, not on `Adapter`):

- [ ] `__init__(*, seed=None)`
- [ ] `set_maxstep`
- [ ] `set_neuron_seed`
- [ ] `rng`
- [ ] `parallel_context`, `h`
- [ ] `register_cell_type`, `cell_type`
- [ ] `register_synapse_type`, `synapse_type`
- [ ] `register_recorder` (optional)

---

## 12. Environment and commands

WSL:

```bash
source ~/.virtualenvs/KinematicsCPG/bin/activate
cd /mnt/d/CPGvalidation
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1
python -c "import neuron; print(neuron.__version__)"
```

Compile the reference mechanisms (once, in a scratch copy of `memcpg/mod_files`
or a dedicated build directory; do not modify `memcpg/`):

```bash
cd /mnt/d/CPGvalidation
mkdir -p /tmp/nrn_build && cp -r memcpg/mod_files /tmp/nrn_build/
cd /tmp/nrn_build
nrnivmodl
```

Then load the compiled library in the NEURON session used by the smoke test.
Exact loading mechanism depends on NEURON's `nrnivmodl` output on WSL; document
it in the test fixture.

Run the focused tests:

```bash
pytest -q tests/test_neuron_api_surface.py tests/test_neuron_route.py \
         tests/test_neuron_groups.py tests/test_neuron_cell_types.py \
         tests/test_neuron_apply.py tests/test_neuron_stdp.py \
         tests/test_neuron_smoke.py
```

Run the full suite:

```bash
pytest -q
```

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| NEST `NodeCollection` semantics leak into the NEURON groups | The contract in spec §8 is the only allowed surface; no `nest`-specific methods in model code. |
| NEST registry and NEURON registry drift | API parity test; shared pure-Python site layer; one spec. |
| STDP recording is approximate | Use memcpg's `synweight` vector sampling (10 ms); test the delta formula, not the event count. |
| Sign conventions flip E/I | `weight_sign_mode="signed"` default; explicit test that a negative `static_inh` site maps to a positive `NetCon.weight[0]` and is classified as inhibitory by `PhysicalMap`. |
| Units copied from NEST | The spec §14.1 forbids conversion; model author sets native units; document the V-cell warning. |
| Pool exhaustion silently drops synapses | `allocate_synapse` raises; test. |
| Structural parameter set at runtime corrupts a cell | Path resolver rejects `nseg`/`insert`/topology; test. |
| `h` global state leaks between tests | `reset_kernel` fixture; one registry per process; test isolation. |
| MPI behavior is undefined | Decide scope; keep `stable_id`/group interfaces; document single-process if deferred. |
| NEST suite regresses from shared dataclass fields | Additive defaults; run the full NEST suite after Phase 0. |

---

## 14. Resolved decisions

The human partner answered these on 2026-10-23:

1. **Adapter shape:** one pure abstract `Adapter(ABC)` in `persona/adapter.py`;
   `NestAdapter(Adapter)` in `persona/nest_adapter.py`; `NeuronAdapter(Adapter)`
   in `persona/neuron_adapter.py`. `Registry`/`NeuronRegistry` are
   compatibility aliases. No `SiteBook` mixin. Do not rewrite NEST behavior.
2. **Cell scope:** all five memcpg types.
3. **STDP recording:** memcpg's `h.Vector().record(stdpmech._ref_synweight,
   10.0)` step-sampling; no `stdp.mod` change.
4. **MPI:** single-process v1, MPI-ready API.
5. **Parameter naming:** NEURON-native canonical; NEST names optional aliases.
6. **Shared site layer:** no `SiteBook`; `Adapter` declares the common methods
   abstract, and pure module-level helpers are shared by both concrete
   adapters. NEST behavior unchanged.
7. **Weight sign:** signed site values; NEURON factory converts to a positive
   conductance. Hard invariant: an inhibitory type stays inhibitory even if a
   positive magnitude is supplied; it is normalized to a negative site and can
   never write an excitatory sign.
8. **Kernel ownership:** `Adapter` declares the kernel methods; NEST delegates
   to `nest`, NEURON to `h`.

One technical requirement remains: exact-key-first resolution for
`dend[0].g_pas` paths (`fnmatch` bracket issue). Implement it in the shared
`_resolve_site_keys`/`Adapter` layer and keep the NEST suite green.

---

## 15. Hand-off checklist

- [ ] Spec approved; decisions in §20 resolved (2026-10-23).
- [ ] TODO phases assigned to owners.
- [ ] Environment verified (NEURON import, `nrnivmodl`, pytest).
- [ ] API parity test green.
- [ ] Single-route scan green.
- [ ] NEST regression green.
- [ ] A short README in `persona/neuron/` explaining how to register a cell
      type and a synapse type.
- [ ] A short migration note: "swap `Registry` for `NeuronRegistry`; the
      `Population`/`Connection` objects are unchanged; use NEURON-native
      parameter paths for non-NEST parameters."
