# NEURON registry adapter — interface specification

**Status:** approved 2026-10-23 (decisions in §20 resolved); implemented in
this branch. This document remains the contract; `persona/adapter.py`,
`persona/nest_adapter.py`, `persona/neuron_adapter.py` and `persona/neuron/`
are the implementation.

**Normative reference:** `persona/registry.py` (the NEST `Registry`).
**Reference model for parameter coverage:** `memcpg/` (all cell classes,
synapses, STDP, generators).

---

## 0. Reading guide

- **MUST / MUST NOT** are contract requirements. A conforming adapter is not
  complete if any MUST is unmet.
- **MAY** is an implementation choice.
- Sections marked *informative* describe the reference mapping and can be
  changed by the implementer as long as the contract holds.
- "Parity" means the same public name, signature, keys, semantics and
  duck-compatible return object. It does **not** mean the same numerical
  result as NEST.

---

## 1. Intent

The NEST `Registry` is the single creation, realization, recording and
plasticity route for PERSONA. It is what makes `Agent`, `Personalizer`,
`PhysicalMap`, freeze, springs and the noise channels work without knowing the
simulator.

The NEURON adapter MUST provide the same route at the same level of
integration. A PERSONA user MUST be able to swap

```python
from persona.registry import Registry
registry = Registry()
```

for

```python
from persona.neuron import NeuronRegistry
registry = NeuronRegistry()
```

and keep the rest of the PERSONA-side code unchanged: `Population`,
`Connection`, `PersonaParam`, `NoiseRecorder`, `Agent`, `Personalizer`,
`PhysicalMap`, `HomeostaticRegulator`, `StackGame`, `freeze`, `springs`.

The adapter MUST also be able to personalize every **runtime-settable numeric
parameter** of the `memcpg` reference, including parameters NEST does not have
and parameters whose names or units differ from NEST.

---

## 2. The non-negotiable invariant

> **Interface parity + single route.**

1. `persona/adapter.py` defines one abstract `Adapter` interface. `Registry`
   (NEST) and `NeuronRegistry` (NEURON) MUST implement every method. Names,
   parameter names, keyword names, defaults and semantics are the same.
2. No `Adapter` method may be omitted, renamed, or reduced to a stub. A method
   that is meaningless in NEURON (e.g. `finalize_connections`) MUST still exist
   and MUST be a correct no-op or cache refresh, not `pass` with missing state.
3. Return values MUST be duck-compatible with the NEST objects at every call
   site that exists today. The compatibility contract is in §8.
4. Additive NEURON-specific methods/fields are allowed only where NEURON has a
   capability NEST does not. They MUST NOT change the semantics of the NEST
   methods.
5. **Single route:** all NEURON network creation and connection MUST go through
   the registry. Raw `neuron.h` / `ParallelContext` calls are allowed only in
   the adapter package (`persona/neuron/`) and in registered cell/synapse
   factories. Model code, the CPG adapter and entry points MUST NOT call
   `h.NetCon`, `h.NetStim`, `h.Section`, `h.ExpSyn`, `h.Exp2Syn`,
   `pc.gid_connect`, `pc.cell` or `pc.set_gid2node` directly. This is the
   NEURON analogue of `tests/test_nest_route.py`.
6. `recorded = applied` MUST hold exactly: the site recorder stores the sample
   the network realized, once per realized step.
7. Failures MUST be surfaced in `last_apply_errors`; no silent fallback and no
   mean-of-draws substitution.
8. Canonical site keys MUST be identical:
   `pop_<population.name>_<param>` and
   `conn_<connection.key>_<param>`. Producer keys MUST be identical:
   `producer:stdp:<key>` and `producer:poisson:<group>`.

`Agent`, `Personalizer`, `PhysicalMap`, `HomeostaticRegulator`, `StackGame`,
`PersonaParam`, `Population`, `Connection`, `NoiseRecorder` and `SpringSpec`
MUST NOT need simulator-specific changes. If an implementation needs to change
them, the API is wrong.

---

## 3. Scope

### 3.1 In scope

- A `NeuronRegistry` class with the full NEST `Registry` public surface.
- NEURON cell/synapse type registration and the built-in types needed by
  `memcpg`: `interneuron`, `bursting_interneuron`, `motoneuron`, `muscle`,
  `afferent`; static excitatory/inhibitory synapses and the `STDP` point
  process.
- Per-cell, per-synapse and per-projection personalization, including
  mechanism/section attributes that NEST does not expose.
- Spike recording, weight recording, read-and-clear semantics.
- Relay/rate generators (`NetStim`) and readout relays (the NEURON counterpart
  of NEST parrots).
- The NEURON kernel lifecycle (`reset`, `dt`, `initialize`, `simulate`, `t`).
- Single-process execution first; the API must not preclude MPI.
- Conformance tests that enforce parity and the single route.

### 3.2 Non-goals

- Numerical equivalence with NEST, or bit-identical random connectivity.
- Porting `t2xcpg` or `memcpg` to the new registry. The other team writes the
  model on top of this API.
- Auto-translating NEST model names (`izhikevich`, `static_synapse`, ...) to
  NEURON mechanisms. Model names are resolved by the adapter's type registry.
- Changing `persona/registry.py`, `Agent`, `Strategy`, `Homeostasis`,
  `Personalizer`, `personalize.py`, `t2xcpg/`, or `memcpg/`.
- A CLI, a GUI, or HDF5 logging changes.
- Personalizing structural properties that require rebuilding a cell
  (`nseg`, channel insertion/removal, section topology, cell class, synapse
  class, `pc.gid` assignment). See §14.3.

---

## 4. Naming and placement

### 4.1 The `Adapter` abstract base class

`persona/adapter.py` defines a **pure abstract** common interface. There is no
`SiteBook` mixin, no second shared-state base class, and no multiple
inheritance trick. The concrete adapters own their implementations.

```text
persona/
    adapter.py               # class Adapter(ABC): all common methods abstract
    nest_adapter.py          # class NestAdapter(Adapter): NEST
    neuron_adapter.py        # class NeuronAdapter(Adapter): NEURON
    registry.py              # compatibility: Registry(NestAdapter)
    neuron/
        __init__.py          # exports NeuronAdapter/NeuronRegistry + protocols
        groups.py            # CellGroup, SynapseGroup, RateGroup, RelayGroup
        recorders.py         # SpikeRecorder, WeightRecorder
        kernel.py            # NEURON lifecycle facade (raw h allowed here)
        cells.py             # built-in cell types + path resolver
        synapses.py          # static_exc, static_inh, stdp_exc
```

```python
from persona.adapter import Adapter
from persona.nest_adapter import NestAdapter
from persona.neuron_adapter import NeuronAdapter

# Compatibility aliases; existing code keeps working unchanged.
from persona.registry import Registry              # Registry is a NestAdapter
from persona.neuron import NeuronRegistry          # alias of NeuronAdapter
```

`Adapter(ABC)` is the single source of truth for the interface. Every method of
the common surface (construction, kernel lifecycle, creation, realization,
recorders, site bookkeeping, freeze, springs, noise recorders, sampling,
plasticity, introspection) is decorated with `@abstractmethod`, including the
properties `time_ms` and `dt_ms` and the classmethod `reset_kernel`. It MUST
NOT define a partial concrete implementation of the registry state.

- `NestAdapter(Adapter)` is the complete NEST implementation, in its own file.
  The existing NEST logic and behavior are preserved; the class simply gains
  the interface and the kernel delegations.
- `NeuronAdapter(Adapter)` is the complete NEURON implementation, in its own
  file. Its helper modules live under `persona/neuron/`; they contain no
  registry abstraction and are not inherited by `NestAdapter`.
- `persona/registry.py` is a thin compatibility module:
  `class Registry(NestAdapter): pass` and re-exports of
  `clamp_weight_sign`/`plastic_delta_from_events`. This keeps every existing
  import working while the concrete NEST class is `NestAdapter`.
- `persona/neuron/__init__.py` re-exports `NeuronAdapter` as
  `NeuronRegistry` for compatibility. Both names refer to the same class.
- Methods with simulator-independent semantics (site bookkeeping, freeze,
  spring, `get_all_optimizable_params`, `sample_all_actions`) MUST behave
  identically in both implementations. The pure module-level helpers and the
  exact-key-first site resolver live in `persona/adapter.py` and are called by
  both concrete classes; this is not a state-bearing mixin.
- NEURON-only extension points (`register_cell_type`,
  `register_synapse_type`, `register_recorder`) are declared on
  `NeuronAdapter`, not on `Adapter`. `Adapter` covers the common surface that
  a simulator-agnostic CPG adapter or the PERSONA loop may call.

The parity test (§18.1) compares `Adapter`'s public surface to both concrete
classes. `NestAdapter` and `NeuronAdapter` MUST implement every abstract
method. A missing implementation is an import-time error from `ABCMeta`, not a
runtime surprise. `Registry` and `NeuronRegistry` MUST also satisfy
`issubclass(..., Adapter)` because they are aliases/subclasses.

**Hard inhibitory invariant.** An inhibitory projection MUST stay inhibitory
for the whole run. This holds whether the model expresses inhibition as a
negative signed weight or as an inhibitory synapse type (`static_inh`,
`stdp_inh`, or a registered type whose `kind == "inh"`). The adapter MUST:

- normalize an inhibitory `weight` site to a negative site value (a positive
  magnitude supplied with an inhibitory type is converted to the negative
  signed form before recording/learning);
- clamp the site bounds so samples can never become positive;
- write a positive hardware conductance (`NetCon.weight[0] = abs(value)`) only
  to the inhibitory point process;
- reject or clamp to zero any runtime weight whose sign would turn the
  projection excitatory.

The converse holds for excitatory types. `static_synapse` is sign-dispatched
before the point process is chosen (negative → inhibitory, positive →
excitatory, zero → `ValueError`). No code path may turn `static_inh` into an
`ExpSyn`/`static_exc` or let it write an excitatory sign.

`persona.registry.Registry` and `persona.nest_adapter` MUST remain importable
even when NEURON is not installed. The NEURON modules MUST be importable only
when NEURON is installed; they MUST NOT become a hard dependency of
`persona/__init__.py`.

---

## 5. Shared data model

The adapter MUST reuse the existing dataclasses unchanged in meaning:

```python
Population(
    model: str,                 # adapter-resolved cell type name
    n: int,
    params: Dict[str, PersonaParam],
    name: str,
    corr: np.ndarray | None = None,
    node_collection: Any = None,
)

Connection(
    source: str,
    target: str,
    model: str = "static_synapse",   # adapter-resolved synapse type name
    parameters: Dict[str, PersonaParam] = {},
    rule: str = "pairwise_bernoulli",
    p: float = 0.5,
    noise_recorder_id: str | None = None,
    extra_syn_spec: Dict[str, Any] = {},
    key: str | None = None,
    weight_recorder: Any = None,
)
```

### 5.1 Additive adapter fields (required for NEURON topology)

Two optional fields MUST be added. They default to empty and are ignored by the
NEST adapter; this is backward-compatible.

```python
@dataclass
class Population:
    ...
    neuron: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Connection:
    ...
    neuron: Dict[str, Any] = field(default_factory=dict)
```

`neuron` carries creation-only options that NEST cannot accept in
`nest.Connect`. It is **static metadata**, never sampled by PERSONA. Suggested
keys are in §10.2 and §17. Any key a model needs and NEST would reject belongs
here, not in `extra_syn_spec`.

- `Population.neuron` values MAY be scalars or per-cell sequences of length
  `cfg.n`. The cell factory receives the **whole `Population` config**, so it
  can read per-cell values (e.g. the `motodiams()` vector that seeds
  `motoneuron(diam)`).
- `Connection.neuron` values are per-projection. Per-synapse creation
  dispersion is expressed with `weight_cv`/`delay_cv`; per-synapse site values
  come from `Connection.parameters`, not from `neuron`.
- A value in `neuron` is creation-time only. To make it personalizable, the
  model MUST put it in `params`/`parameters` as a `PersonaParam` or expose it
  through a factory-level site (see §7.9).

`extra_syn_spec` remains the NEST-compatible syn-spec dictionary (e.g.
`{"delay": 3.0}` or `{"weight": 4.0}`). The NEURON adapter MUST read it where
it has an obvious meaning (weight seed, delay) and MUST NOT mutate it.

### 5.2 `model` is a PERSONA-level type name

`Population.model` and `Connection.model` are resolved by the adapter, not by
the simulator:

- NEST: `Registry` passes `Population.model` to `nest.Create` and
  `Connection.model` to `nest.Connect` (as today).
- NEURON: `NeuronRegistry` resolves the name through its registered cell /
  synapse type tables. The same string can therefore mean different mechanisms
  on the two adapters; the PERSONA-side `Population`/`Connection` object is
  identical.

Built-in NEURON names (normative for the reference model):

| `Population.model` | NEURON cell | memcpg reference |
|---|---|---|
| `interneuron` | `interneuron(delaytype=False)` | `memcpg/interneuron.py` |
| `bursting_interneuron` | `interneuron(False, bursting_mode=True)` | `memcpg/interneuron.py` |
| `motoneuron` | `motoneuron(diam)` | `memcpg/motoneuron.py` |
| `muscle` | `muscle()` | `memcpg/muscle.py` |
| `afferent` | `bioaffrat()` | `memcpg/bioaffrat.py` |
| `poisson_generator` | rate generator (`NetStim`) | `create_relay(rate_param=...)` |
| `izhikevich` | optional registered cell type | only if a NEURON Izhikevich mechanism is supplied |

`poisson_generator` is kept as the canonical PERSONA name for a rate generator
so `HomeostaticRegulator`/`PhysicalMap` continue to classify it as `poisson`.
The NEURON registry MUST map it to its `RateGroup`, not to a NEURON mechanism
literally named `poisson_generator`.

| `Connection.model` | NEURON realization | memcpg pool |
|---|---|---|
| `static_synapse` | sign-dispatched: `weight >= 0` → `static_exc`; `weight < 0` → `static_inh` | `synlistex` / `synlistinh` |
| `static_exc` | `ExpSyn` | `synlistex` |
| `static_inh` | `Exp2Syn` | `synlistinh` |
| `stdp_synapse` | alias for `stdp_exc` (STDP weights are positive) | `synlistexstdp` |
| `stdp_exc` | `ExpSyn` + `STDP` point process | `synlistexstdp` |
| `stdp_inh` | `Exp2Syn` + inhibitory STDP (only if supplied) | `synlistinhstdp` |

The aliases exist so a `Connection` built for NEST keeps working when swapped
to NEURON, provided the referenced cells/synapses exist. `static_synapse`
is a dispatcher, not a plain alias: NEST uses one model name for both
excitatory and inhibitory static synapses, selected by the weight sign. The
NEURON factory MUST inspect the signed seed and route to the correct pool. If
the sign is zero, raise `ValueError` (ambiguous). `stdp_synapse` is
excitatory by convention; an inhibitory STDP model must use `stdp_inh`.

---

## 6. Parameter addressing

A `PersonaParam` name is a **path** resolved by the cell or synapse type. This
is how the adapter covers parameters NEST does not have without adding a new
registry method for every mechanism.

### 6.1 Grammar

```text
param       := special | path
special     := "weight" | "delay" | "rate" | "interval"
path        := component ("." component)*
component   := section | "syn" | "nc" | "stdp" | "stim" | mechanism
section     := name | name "[" index "]" | name "(" x ")"
```

Examples:

| Path | Meaning | Scope |
|---|---|---|
| `soma.gnabar_fastchannels` | section biophysics / inserted mechanism | per cell |
| `soma(0.5).gnabar_fastchannels` | one segment of a section | per cell |
| `dend.g_pas` | all dend sections (broadcast) | per cell |
| `dend[0].g_pas` | one dend section | per cell |
| `axon.gnabar_hh` | axon channel | per cell |
| `syn.tau` | point-process attribute | per synapse |
| `syn.e` | reversal potential | per synapse |
| `nc.threshold` | `NetCon` attribute | per synapse |
| `stdp.tauhebb` | STDP mechanism attribute | per projection |
| `stim.interval` | `NetStim` attribute | per generator |
| `weight` | reserved: `NetCon.weight[0]` | per synapse |
| `delay` | reserved: `NetCon.delay` | per synapse |
| `rate` | reserved: converted to `stim.interval` | per generator |

### 6.2 Resolution rules

- The **cell type** resolves `Population.params` paths. The **synapse type**
  resolves `Connection.parameters` paths.
- A path that does not exist on a cell/synapse MUST raise
  `KeyError`/`AttributeError` at registration/creation time, not silently set
  a wrong attribute.
- A bare path is per-cell when used in `Population.params`, per-synapse when
  used in `Connection.parameters`.
- `dend.attr` broadcasts over all sections whose name matches `dend`; if none
  match, raise.
- `section(x).attr` sets one segment; `section.attr` sets every segment of that
  section (NEURON semantics). If a cell type declares a parameter as
  per-segment, the registry samples one value per segment? **No:** PERSONA has
  one scalar per site, so one value is applied to the declared scope. If a
  model needs independent per-segment values, it must declare separate sites
  (e.g. `dend[0].g_pas`, `dend[1].g_pas`). This is the same "one site = one
  scalar" rule as NEST.
- `Connection.parameters` MAY use `weight` and `delay` for the NEST-compatible
  static synapse path, and `syn.*`/`nc.*`/`stdp.*` for explicit control. The
  adapter MUST reject a path that resolves to a non-numeric or non-settable
  attribute.
- `extra_syn_spec` and `neuron` produce static values at creation and are not
  sites. To make a value personalizable, put it in `params`/`parameters` as a
  `PersonaParam` (or provide a type-level default site; see §7.9).

### 6.3 Canonical keys

The prefixing is unchanged:

```text
pop_<population.name>_<param>
conn_<connection.key or source_target>_<param>
```

Dots remain in the key (e.g.
`pop_mn_E_soma.gnabar_motoneuron`). `freeze`/`create_spring` MUST resolve an
**exact key first**, then fall back to `fnmatchcase`. This is required because
`fnmatchcase` treats `[0]` in `dend[0].g_pas` as a character class, not a
literal section index. The exact-first rule is a shared, behavior-preserving
fix (it only adds matches that were already spelled exactly) and MUST be
applied to the shared site resolver so NEST and NEURON behave identically. A
glob that still cannot match a bracketed key is a model-author error; the
error MUST name the pattern.

---

## 7. Public API

The class MUST expose the following. Signatures are normative: identical
parameter names, keyword-only markers and defaults to `persona.registry.Registry`.
Return annotations are informative; the return contracts are in §8.

### 7.1 Construction and kernel lifecycle (common)

`Adapter` declares the kernel methods so a simulator-agnostic CPG adapter can
call one interface. NEST delegates to `nest`; NEURON delegates to `h`.

```python
class Adapter(ABC):
    def __init__(self) -> None: ...

    @classmethod
    def reset_kernel(
        cls,
        *,
        resolution_ms: float | None = None,
        v_init: float | None = None,
        cvode: bool = False,
        maxstep_ms: float | None = None,
    ) -> None: ...

    def set_resolution(self, resolution_ms: float) -> None: ...
    def initialize(self, v_init: float | None = None) -> None: ...
    def simulate(self, dt_ms: float) -> float: ...
    @property
    def time_ms(self) -> float: ...
    @property
    def dt_ms(self) -> float: ...
    # NEURON only (declared on NeuronRegistry, not Adapter):
    # __init__(*, seed=None), set_maxstep, set_neuron_seed, rng,
    # parallel_context, h
```

- NEST `Registry` implements `reset_kernel` with `nest.ResetKernel()` (and
  optional `SetKernelStatus` for resolution/threads), `set_resolution` with
  `SetKernelStatus({"resolution": ms})`, `initialize` as a no-op, `simulate`
  with `nest.Simulate(dt_ms)`, `time_ms` with
  `GetKernelStatus("biological_time")`, and `dt_ms` with
  `GetKernelStatus("resolution")`. These are additive delegations; the
  existing NEST behavior is unchanged. Existing call sites may keep calling
  `nest.Simulate` directly.
- NEURON `reset_kernel` MUST reset the NEURON global state used by the
  adapter: time, `dt`, `cvode`, `pc`. It is kernel-only: it MUST be callable
  before a `NeuronRegistry` is constructed and MUST NOT reach into an
  instance. The supported run pattern is `NeuronRegistry.reset_kernel(...)`
  followed by a **new** `NeuronRegistry()`.
- An existing NEURON registry whose cells were destroyed by `reset_kernel` is
  stale; its methods MUST raise a clear `RuntimeError` instead of reading
  dangling handles. An optional `close()` method MAY mark an instance closed.
  Tests MUST create a new registry after every `reset_kernel`.
- `reset_kernel` MUST NOT clear the registered cell/synapse **type tables**
  (they are class-level or constructor-level), but it MUST not be expected to
  preserve instance-level `population_cfgs`/`population_nodes`; those belong to
  a discarded instance.
- `initialize` MUST call `h.finitialize(v_init)` (NEURON) or be a no-op (NEST)
  and leave the clock at its initialization value.
- `simulate(dt_ms)` MUST advance the network by exactly `dt_ms` and return the
  new `time_ms`. It is the counterpart of `nest.Simulate(dt_ms)`. The CPG
  adapter MUST call this instead of `h.fadvance`/`pc.psolve` on NEURON.
- `time_ms` MUST return `float(h.t)` (NEURON) or the NEST biological time.
  `_now_ms()` MUST use it.
- NEURON adds `__init__(*, seed=None)`, `set_maxstep`, `set_neuron_seed`,
  `rng`, `parallel_context` and `h`. `seed` seeds the adapter's
  connectivity/morphology RNG; `set_neuron_seed` seeds `h.Random`/`pc` for
  stochastic mechanisms. NEST does not need these.
- The NEURON `h`/`parallel_context` properties exist for registered
  cell/synapse factories. Model code MUST NOT use them; the source scan in
  §18 enforces this.

### 7.2 Noise recorders (identical)

```python
def register_noise_recorder(self, recorder: NoiseRecorder) -> None: ...
def get_noise_recorder(self, name: str) -> NoiseRecorder | None: ...
def site_recorder(self, full_key: str) -> NoiseRecorder: ...
def producer_recorder(
    self, name: str, scale: float | None = None) -> NoiseRecorder: ...
def producer_recorders(self) -> Dict[str, NoiseRecorder]: ...
```

Semantics are exactly `persona/registry.py`:

- site recorder name `site:<full_key>`;
- producer name prefixed with `producer:` if absent;
- `scale` fixed at registration; a later call without `scale` leaves it;
- `scale <= 0` or `None` means no influence.

No simulator involvement. The implementation MAY share this code with the NEST
registry (mixin/helper); the behavior MUST be identical.

### 7.3 Creation defaults, freeze, springs (identical)

```python
def recapture_defaults(self, keys=None) -> None: ...
def freeze(self, keys, attrs=("mean",)) -> tuple[str, ...]: ...
def unfreeze(self, keys, attrs=None) -> tuple[str, ...]: ...
def frozen_attrs(self, site_key: str) -> frozenset[str]: ...
def is_frozen(self, site_key: str, attr: str) -> bool: ...
def defaults(self, site_key: str) -> Dict[str, float]: ...

def create_spring(
    self, name: str, keys, *, power: float | None = None,
    gain: float = 1.0, max_power: float = 0.5) -> SpringSpec: ...
def remove_spring(self, name: str) -> None: ...
def springs(self) -> Dict[str, SpringSpec]: ...
def spring_state(
    self, means: Dict[str, float], sigmas: Dict[str, float],
    lr_t: float) -> Dict[str, Dict[str, float]]: ...
```

These MUST be copied from / shared with `Registry` without semantic change.
They are pure Python plus the site table. The NEURON adapter MUST record the
creation defaults at the same points as NEST: `create_population`,
`create_connection`, `create_relay`; `finalize_connections` recaptures them.
`_defaults`, `_frozen`, `_springs` MUST exist with the same shapes.

### 7.4 Nodes, populations, connections, relays (identical surface)

```python
def register_nodes(self, name: str, nodes: Any) -> Any: ...
def create_parrot_nodes(self, name: str, n: int) -> Any: ...
def create_spike_recorder(self, target: Any) -> Any: ...
def create_weight_recorder(self) -> Any: ...

def create_population(
    self, cfg: Population,
    explored_means: Dict[str, float] | None = None) -> Any: ...

def create_connection(
    self, cfg: Connection,
    explored_params: Dict[str, float] | None = None) -> str: ...

def create_relay(
    self, pg_key: str, in_key: str, n: int, *,
    rate: float = 0.0,
    rate_param: PersonaParam | None = None,
    pop_name: str | None = None,
    noise_name: str | None = None,
    noise_scale: float | None = None,
) -> tuple[Any, Any]: ...

def set_relay_rate(self, pg_key: str, rate: float) -> None: ...
```

Return contracts and NEURON-specific semantics are in §8 and §9.

### 7.5 Synapse views and plasticity (identical surface)

```python
def synapses(self, conn_key: str) -> Any: ...
def synapse_count(self, conn_key: str) -> int: ...
def finalize_connections(self) -> None: ...

def read_plastic_events(
    self, clear: bool = True) -> Dict[str, Dict[str, np.ndarray]]: ...
def sample_plastic_weight_deltas(
    self, t_before: float, t_now: float) -> Dict[str, float]: ...
def last_plastic_counts(self) -> Dict[str, int]: ...
def sample_plastic_weight_stats(self) -> Dict[str, Dict[str, float]]: ...
def final_plastic_weights(self) -> Dict[str, np.ndarray]: ...
```

### 7.6 Parameters, sampling, apply (identical surface)

```python
def get_all_optimizable_params(self) -> Dict[str, PersonaParam]: ...
def sample_all_actions(self) -> Dict[str, float]: ...
def apply_explored_parameters(self, sampled_values: Dict[str, float]) -> None: ...
```

`last_apply_errors: list[tuple[str, str]]` MUST exist and be cleared on every
`apply_explored_parameters` call.

### 7.7 Introspection (identical surface)

```python
def get(self, name: str) -> Any: ...
def get_population_cfg(self, name: str) -> Population: ...
```

Public attributes MUST exist with the same names:

```python
population_cfgs: Dict[str, Population]
connection_cfgs: Dict[str, Connection]
population_nodes: Dict[str, Any]
noise_recorders: Dict[str, NoiseRecorder]
last_apply_errors: list[tuple[str, str]]
```

### 7.8 Module-level helpers (identical)

`persona.registry` exports two module-level functions. They MUST also be
available from the NEURON module (same names, same behavior), or be imported
from the shared module:

```python
def clamp_weight_sign(param: PersonaParam) -> PersonaParam: ...
def plastic_delta_from_events(
    events: dict, last_w: dict, seed_weight: float,
    n_syn: int) -> tuple[float, dict]: ...
```

`plastic_delta_from_events` is simulator-independent and MUST be shared, not
reimplemented. `clamp_weight_sign` is reused for signed-weight types; the
weight-sign policy is in §14.2.

`persona/adapter.py` defines the pure module-level helpers
`clamp_weight_sign`, `plastic_delta_from_events`, `resolve_site_keys` (the
exact-key-first resolver) and `enforce_synaptic_sign` (the hard E/I identity
normalizer). Both concrete adapters call the same functions; there is no
`SiteBook` mixin and no shared state-bearing base class. `persona/registry.py`
MUST continue to export `clamp_weight_sign` and `plastic_delta_from_events`,
so existing imports keep working. This is a behavior-preserving move; the NEST
test suite MUST pass unchanged.

The abstract `Adapter` MUST NOT import `nest` or `neuron`. `NestAdapter`
imports `nest`; `NeuronAdapter` imports `neuron` directly (never via
`persona/__init__.py`).

### 7.9 Additive extension points (NEURON only)

These methods are NOT in the NEST `Registry`; they are required so the adapter
can cover parameters NEST lacks without changing the core API.

```python
def register_cell_type(self, name: str, factory: "CellFactory") -> None: ...
def register_synapse_type(self, name: str, factory: "SynapseFactory") -> None: ...
def register_recorder(self, name: str, recorder: Any) -> None: ...   # optional
def cell_type(self, name: str) -> "CellFactory": ...
def synapse_type(self, name: str) -> "SynapseFactory": ...
```

#### 7.9.1 `CellFactory` protocol

The cell factory is the only sanctioned place for raw NEURON section/mechanism
creation. It MUST NOT connect cells; connections go through the registry.

```python
@dataclass(frozen=True)
class SpikeSource:
    ref: Any          # e.g. cell.node[-1](0.5)._ref_v
    sec: Any          # the section that owns ref; required by h.NetCon
    threshold: float  # this cell's output threshold (-20 or +10 in memcpg)

class CellFactory(Protocol):
    name: str

    def create(
        self, registry: "NeuronRegistry", cfg: Population) -> "CellGroup": ...

    def param_paths(self) -> set[str]: ...
    def has_param(self, cell: Any, path: str) -> bool: ...
    def get_param(self, cell: Any, path: str) -> float: ...
    def set_param(self, cell: Any, path: str, value: float) -> None: ...

    def spike_source(self, cell: Any) -> SpikeSource: ...
    def synapse_pool(self, cell: Any, kind: str) -> list[Any]: ...
    def allocate_synapse(self, cell: Any, kind: str) -> Any: ...
    def stable_id(self, cell: Any) -> int: ...
```

- `create` receives the **whole `Population` config**, not just `n`. This is
  where `motoneuron(diam)` reads `cfg.neuron` (a scalar or a per-cell sequence
  produced by `motodiams`), where `muscle(n_synapses, nseg)` reads its options,
  where `afferent(neuron_type, n_synapses, n_axon_nodes)` reads its options,
  and where `interneuron(serotonin_mode, bursting_mode)` reads its flags. It
  returns a `CellGroup` whose `.cells` are opaque cell handles.
- The factory MUST use `registry.rng` for any morphology/geometry randomness,
  not Python's global `random`, so a run is reproducible from
  `NeuronRegistry(seed=...)`. See §16.3.
- `param_paths` is the declared personalizable set. `has_param` is used for
  validation. `get_param` is used by `sample_plastic_weight_stats` only if the
  synapse type delegates to a cell attribute; otherwise optional.
- `set_param` MUST be runtime-safe (no channel re-insertion, no `nseg`
  change, no topology change). If a path requires a rebuild, the factory MUST
  raise a clear error.
- `spike_source` returns a `SpikeSource` carrying both the voltage reference
  and the section, plus the cell's output threshold. A bare `_ref_v` is not
  enough for `h.NetCon`. `memcpg` uses `-20` for interneuron/motoneuron/
  afferent and `+10` for muscle; `Connection.neuron["threshold"]` overrides it.
- `synapse_pool` returns the pre-allocated point processes for a kind
  (`exc`, `inh`, `stdp_exc`, `stdp_inh`), matching memcpg's
  `synlistex`/`synlistinh`/`synlistexstdp`/`synlistinhstdp`.
- `allocate_synapse` returns the next free point process or raises when the
  pool is exhausted. The pool size is a build-time choice (`n_synapses` in
  memcpg).
- `stable_id` returns a process-stable integer used as `senders`/`targets` in
  recorder events.

#### 7.9.2 `SynapseFactory` protocol

```python
@dataclass(frozen=True)
class SynapseContext:
    registry: "NeuronRegistry"
    conn_key: str
    source: str
    target: str
    presyn_cell: Any
    postsyn_cell: Any
    presyn_source: SpikeSource
    postsyn_pool: Any              # an allocated point process from the cell
    weight: float                  # signed site value at creation (see §14.2)
    delay: float                   # ms
    params: Dict[str, float]       # all Connection.parameters values
    weight_recorder: Any | None
    threshold: float               # effective NetCon threshold

@dataclass
class SynapseHandle:
    netcon: Any                    # the functional NetCon
    synapse: Any                   # ExpSyn / Exp2Syn
    stdp: Any | None               # STDP mechanism, if plastic
    teaching_netcons: list[Any]    # pre/post teaching NetCons, if plastic
    presyn_id: int
    postsyn_id: int
    params: Dict[str, float]
    meta: Dict[str, Any]           # factory-private (dummy section, vectors, ...)

class SynapseFactory(Protocol):
    name: str
    kind: str                  # "exc" | "inh"
    plastic: bool
    weight_sign_mode: str      # "signed" (default) | "magnitude"

    def create(
        self, registry: "NeuronRegistry", ctx: SynapseContext) -> SynapseHandle: ...

    def param_paths(self) -> set[str]: ...
    def set_param(self, synapse: Any, path: str, value: float) -> None: ...
    def get_param(self, synapse: Any, path: str) -> float: ...
    def weight(self, synapse: Any) -> float: ...
    def set_weight(self, synapse: Any, value: float) -> None: ...
    def delay(self, synapse: Any) -> float: ...
    def set_delay(self, synapse: Any, value: float) -> None: ...
```

`create` is the only place a `NetCon`/point process is connected. It MUST NOT
add the handle to the projection table; it returns a `SynapseHandle`, and
`create_connection` stores it in the `SynapseGroup`. The context gives the
factory everything it needs: the connection key, both cells, the presynaptic
`SpikeSource`, the allocated postsynaptic point process, the effective
threshold, the creation weight/delay, and the weight recorder. This also fixes
the stable-ID problem: the factory does not need to look up cells; the context
carries `presyn_id`/`postsyn_id` via `stable_id`.

`SynapseGroup` stores `SynapseHandle` objects. `synapses()`, `synapse_count()`
and `read_plastic_events()` see them through the registry's projection table;
the factory does not need a back-reference.

Built-in factories MUST include:

- `static_exc` (`ExpSyn`, pool `synlistex`, signed weights);
- `static_inh` (`Exp2Syn`, pool `synlistinh`, signed weights);
- `static_synapse` (sign dispatcher: positive → `static_exc`, negative →
  `static_inh`, zero → `ValueError`); this is the NEST-compatible name;
- `stdp_exc` (`ExpSyn` + `STDP`, pool `synlistexstdp`, positive seed);
- `stdp_synapse` (alias for `stdp_exc`);
- `stdp_inh` only if an inhibitory STDP mechanism is supplied.

The default `weight_sign_mode` is `"signed"` for both static types. This is
required because `persona/homeostasis.py` classifies a connection from the sign
of the weight parameter; a positive-magnitude inhibitory weight would be seen
as excitatory. See §14.2. The dispatcher is what lets an unchanged NEST
`Connection(model="static_synapse", weight=<negative>)` work on NEURON.

#### 7.9.3 Type registration timing

A model MUST be able to register types before `create_population` /
`create_connection`:

```python
registry.register_cell_type("motoneuron", MotoneuronFactory())
registry.register_synapse_type("static_exc", StaticExcFactory())
```

The built-ins MUST be pre-registered at construction so the reference model
works without explicit registration.

---

## 8. Return-object contracts

The existing NEST call sites use only a subset of `NodeCollection` /
`SynapseCollection` / device methods. The NEURON objects MUST provide that
subset with the same names and semantics.

### 8.1 `CellGroup` (replaces `nest.NodeCollection`)

Used by `create_population`, `create_parrot_nodes`, `register_nodes`, `get`,
`create_spike_recorder`, `create_connection` (as endpoints).

```python
class CellGroup:
    name: str
    cells: list[Any]
    ids: list[int]
    cell_type: CellFactory | None

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int) -> Any: ...

    def set(self, values: Dict[str, Any]) -> None: ...
    def get(self, name: str) -> np.ndarray: ...
    def has_param(self, name: str) -> bool: ...
    def param_paths(self) -> set[str]: ...
    def spike_source(self, cell: Any | None = None) -> Any: ...
```

`set` semantics:

- A scalar value is broadcast to every cell.
- A sequence/array MUST have length `len(self)`; otherwise `ValueError`.
- Each value is passed to `cell_type.set_param(cell, path, value)`.
- The method MUST apply all updates or raise; partial application is allowed
  only if the caller catches and records the error (see §9.6). The registry
  uses a per-population try/except and records failures.
- `get` returns a `np.ndarray` of per-cell values in `cells` order.

### 8.2 `SynapseGroup` (replaces `nest.SynapseCollection`)

Used by `synapses`, `apply_explored_parameters`, `sample_plastic_weight_stats`,
`final_plastic_weights`.

```python
class SynapseGroup:
    key: str
    source: str
    target: str
    model: str
    handles: list["SynapseHandle"]
    weight_recorder: WeightRecorder | None

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator["SynapseHandle"]: ...
    def __getitem__(self, index: int) -> "SynapseHandle": ...

    def set(self, values: Dict[str, Any]) -> None: ...
    def get(self, name: str) -> np.ndarray: ...
    def param_paths(self) -> set[str]: ...
```

`set` semantics are the same as `CellGroup`: scalar broadcast, sequence length
must equal `len(self)`, resolved by the synapse type. `set({"weight": ...})`
and `set({"delay": ...})` are reserved and MUST work for every static
projection.

`weight` is a **signed site value** for static synapses, matching NEST:
positive = excitatory, negative = inhibitory. The factory converts it to the
hardware representation (`NetCon.weight[0]` is a non-negative conductance
multiplier for `ExpSyn`/`Exp2Syn`). `set({"weight": -4.0})` on a `static_inh`
projection sets `weight[0] = 4.0`; `set({"weight": +4.0})` on `static_exc` sets
`4.0`. `get("weight")` returns the **live hardware** values (`weight[0]`), not
the signed site values; `get("signed_weight")` reconstructs the signed site
values. `sample_plastic_weight_stats` and `final_plastic_weights` use the
hardware values, which are the STDP-owned weights.

### 8.3 `RateGroup` (replaces the `poisson_generator` NodeCollection)

```python
class RateGroup:
    name: str
    stims: list[Any]            # h.NetStim
    ids: list[int]

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int) -> Any: ...

    def set(self, values: Dict[str, Any]) -> None: ...
    def get(self, name: str) -> np.ndarray: ...
```

- `set({"rate": hz})` converts Hz to `NetStim.interval` (ms) per generator:
  `interval = 1000 / hz`; `hz <= 0` MUST disable the generator without
  dividing by zero.
- The reference recipe for a continuous rate generator is:
  `start = 0`, `number = 1_000_000_000`, `noise = 1.0` (Poisson-like),
  `interval = 1000 / rate`. `rate <= 0` sets `number = 0` and remembers the
  previous `number`; re-enabling restores it. `start`, `number`, `noise` are
  individually overridable through `neuron` / `stim.*` sites.
- A deterministic generator MAY use `noise = 0.0`; a finite train MAY use a
  finite `number`. The recipe is a creation default, not a hard-coded rule;
  `RateGroup` MUST keep `rate` and `interval` consistent after any `number` or
  `noise` change.
- `get("rate")` converts back. `get("interval")`, `get("number")` and
  `get("noise")` are also available.
- A scalar is broadcast; a sequence length must equal `len(self)`.
- `set({"interval": ms})` is allowed as a lower-level escape hatch; `rate` and
  `interval` MUST stay consistent.
- `create_relay(rate_param=...)` uses `RateGroup` as `pop.node_collection`, so
  `apply_explored_parameters` can call `.set({"rate": array})`.
- `RateGroup` MUST keep strong references to its `NetStim`s and any
  `h.Random` objects attached via `noiseFromRandom`.

### 8.4 `RelayGroup` (replaces `nest` parrots / relay parrots)

```python
class RelayGroup:
    name: str
    cells: list[Any]            # true relay cells or readout taps
    ids: list[int]
    kind: str                   # "forward" | "tap"
    can_source: bool
    can_target: bool

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, index: int) -> Any: ...

    def set(self, values: Dict[str, Any]) -> None: ...
    def get(self, name: str) -> np.ndarray: ...
    def spike_source(self, cell: Any | None = None) -> Any: ...
```

- It MUST be a valid `create_spike_recorder` target.
- A relay exposes `can_source: bool` and `can_target: bool`:
  - a **forwarding** relay has `can_source=True`, `can_target=False` (you do
    not connect into a `NetStim`);
  - a **tap** relay has `can_source=False`, `can_target=True`.
  `create_connection` MUST validate the endpoint capability and raise a clear
  `TypeError` when a relay is used on the wrong side.
- **Two relay kinds, two contracts:**
  - `create_relay` returns a **forwarding relay**: a spike arriving from the
    `RateGroup` MUST produce exactly one output spike that downstream
    `Connection`s see. The recommended implementation is the `NetStim` itself
    as the logical source (the `RelayGroup` wraps the `RateGroup`'s `NetStim`s
    and exposes them as the spike source), so no extra cell or delay is added.
    A true relay cell (`IntFire`/`Section` + threshold) is allowed only if the
    model explicitly needs one; document its added delay and threshold.
  - `create_parrot_nodes` returns a **readout tap**: it records the spikes of
    its inputs but is not required to forward them. This is the NEURON
    counterpart of the NEST muscle readout parrots in `t2xcpg/leg.py`, whose
    only use is `create_spike_recorder` + `get("events")`.
- The `RateGroup → RelayGroup` link in `create_relay` MUST be created by the
  registry (a one-to-one event path). It is not a user `Connection`.
- `create_parrot_nodes` MUST NOT be used as a functional source when the model
  needs downstream propagation; use `create_relay` or a direct `Connection`.
- Both kinds keep the `RelayGroup` name for API parity. The kind is visible as
  `relay.kind` (`"forward"` / `"tap"`).

### 8.5 `SpikeRecorder`

```python
class SpikeRecorder:
    def get(self, key: str) -> Any: ...
    def set(self, values: Dict[str, Any]) -> None: ...
    def __len__(self) -> int: ...          # n_events
```

- `get("events")` MUST return a dict with at least:
  `{"times": np.ndarray, "senders": np.ndarray}`.
  `senders` are `CellFactory.stable_id` values.
- `set({"n_events": 0})` MUST clear the accumulated events. A subsequent
  `get("events")` returns empty arrays.
- `get("n_events")` returns the number of recorded events.
- The recorder MUST keep strong references to its `NetCon`s and `h.Vector`s.
- Recording target: a `CellGroup`, `RelayGroup`, a single cell, or a sequence
  of cells.

### 8.6 `WeightRecorder`

```python
class WeightRecorder:
    def get(self, key: str) -> Any: ...
    def set(self, values: Dict[str, Any]) -> None: ...
    def __len__(self) -> int: ...
```

- `get("events")` MUST return a dict with at least:
  `{"times": np.ndarray, "weights": np.ndarray,
    "senders": np.ndarray, "targets": np.ndarray}`.
  `senders`/`targets` are stable cell IDs.
- `set({"n_events": 0})` clears the accumulated events.
- It is attached to a plastic projection via `Connection.weight_recorder`.
  The synapse factory receives it in `create` and MUST record every weight
  update event (see §11).
- `get("n_events")` returns the number of events since the last clear.

### 8.7 `create_connection` return value

`create_connection` MUST return a `str` connection key, exactly as NEST does.
The key is `cfg.key` when given, else `f"{cfg.source}_{cfg.target}"`.

### 8.8 `create_relay` return value

`create_relay` MUST return `(RateGroup, RelayGroup)`.

---

## 9. Method semantics

### 9.1 `create_population`

1. Resolve `cfg.model` through `cell_types`; unknown → `ValueError`.
2. Create `cfg.n` cells via the factory; assign stable IDs; wrap in a
   `CellGroup`.
3. Validate every path in `cfg.params` with `cell_type.has_param` on the first
   cell (and on every cell for heterogeneous types); unknown → `ValueError`.
4. Set `cfg.node_collection = group`; store `population_cfgs[cfg.name] = cfg`
   and `population_nodes[cfg.name] = group`.
5. For every `p_name, param_obj` in `cfg.params`:
   - key = `pop_<cfg.name>_<p_name>`;
   - `_record_defaults(key, param_obj)`;
   - `site_recorder(key).record_sample(_now_ms(), param_obj.mean, param_obj.sigma)`.
6. `sampled = cfg.sample_matrix(explored_means=explored_means)`; if non-empty,
   call `group.set(sampled)`. A failure is appended to `last_apply_errors` as
   `(f"pop_{cfg.name}", str(exc))`; it MUST NOT be swallowed.
7. Return the `CellGroup`.

The NEST implementation's vectorized-first/per-node fallback is a NEST detail.
NEURON always applies per cell; the observable contract is the same.

### 9.2 `create_connection`

1. Resolve `cfg.source` and `cfg.target` through `get`; unknown → `KeyError`.
2. Resolve `cfg.model` through `synapse_types`; unknown → `ValueError`. If the
   name is `static_synapse`, dispatch on the signed seed weight: positive →
   `static_exc`, negative → `static_inh`, zero → `ValueError`. The resolved
   concrete type is what the `SynapseGroup` stores; the original
   `Connection.model` string stays in `connection_cfgs` for introspection.
3. Realize connectivity according to §10. `cfg.p` is structural and MUST NOT
   appear in `get_all_optimizable_params`.
4. For each realized pre/post pair:
   - seed `weight` from `cfg.parameters["weight"].mean` if present, else from
     `cfg.extra_syn_spec.get("weight", 0.0)`;
   - seed `delay` from `cfg.extra_syn_spec.get("delay", 1.0)` or
     `cfg.parameters["delay"].mean` if present;
   - allocate the postsynaptic point process from
     `postsyn_cell_type.allocate_synapse(postsyn_cell, kind)`;
   - build a `SynapseContext` (§7.9.2) with `conn_key`, both cells, the
     presynaptic `SpikeSource`, the allocated point process, the signed seed
     weight, delay, all static `cfg.parameters` values, the weight recorder,
     and the effective threshold;
   - call `synapse_type.create(self, ctx)` and store the returned
     `SynapseHandle` in the `SynapseGroup`.
   - The effective threshold is
     `cfg.neuron.get("threshold", ctx.presyn_source.threshold)`. It is NOT a
     global `-20`: muscle cells use `+10`, and the STDP teaching NetCons use
     NEURON's default unless the factory explicitly overrides it.
5. Store `connection_cfgs[conn_key] = cfg`; store the `SynapseGroup` in the
   projection table.
6. Record creation defaults and site samples for every `cfg.parameters` entry,
   with the same key format as NEST.
7. If `cfg.weight_recorder is not None`, register the producer recorder
   `producer:stdp:<conn_key>` with the STDP weight-ceiling mean as its scale.
   The adapter MUST accept `Wmax`, `wmax`, and `stdp.wmax` (first present wins)
   so a NEST-style model and a NEURON-style model both get a scale. If none is
   present, `scale=None` (no influence), exactly as NEST does when `Wmax` is
   absent.
8. Apply the weight-sign policy (§14.2). The default is `"signed"`: a
   `static_inh` site has a negative `PersonaParam.mean` and the factory writes
   `abs(value)` to the inhibitory point process. This keeps
   `HomeostaticRegulator`/`PhysicalMap`'s sign-based classification correct
   without changing those modules.
9. Return `conn_key`.

`cfg.parameters["weight"]` is a site only for static synapses. For plastic
synapses the model MUST NOT put `weight` in `cfg.parameters`; the weight is
seeded via `extra_syn_spec` and owned by STDP, exactly as NEST.

### 9.3 `create_relay`

1. Create `n` `NetStim` generators with `rate` and a `RateGroup`, using the
   recipe in §8.3: `start=0`, `number=1_000_000_000`, `noise=1.0`,
   `interval=1000/rate`; `rate<=0` disables safely.
2. Create a **forwarding** `RelayGroup` of `n` relays (see §8.4). The
   recommended implementation wraps the `RateGroup`'s `NetStim`s as the
   logical spike source, so a downstream `Connection` sees the generated
   spikes without an extra cell or delay. A true relay cell is allowed only if
   the model needs one; it MUST be documented.
3. Register `pg_key` → `RateGroup`, `in_key` → `RelayGroup` in
   `population_nodes`. The registry MUST create the one-to-one
   `RateGroup → RelayGroup` event path; it is not a user `Connection`.
4. If `rate_param is not None`:
   - `name = pop_name or pg_key`;
   - build `Population(model="poisson_generator", n=n, params={"rate":
     rate_param}, name=name)`;
   - set `pop.node_collection = pg`;
   - store it in `population_cfgs`; record defaults and a site sample;
   - realize the seed dispersion per generator: `rate_param.sample_population(
     n, center=rate_param.mean)` → `pg.set({"rate": ...})`; failures go to
     `last_apply_errors`.
5. If `noise_name is not None`:
   - `rec_name = f"producer:poisson:{noise_name}"`;
   - `value = noise_scale if not None else max(0.0, rate)`;
   - `producer_recorder(f"poisson:{noise_name}", scale=value)`;
   - store `_relay_noise[pg_key] = rec_name`, `_relay_last[pg_key] = rate`.
6. Return `(pg, relay)`.

### 9.4 `set_relay_rate`

1. `r = float(max(0.0, rate))`.
2. `get(pg_key).set({"rate": r})`.
3. If `pg_key` has a producer recorder, record `r - previous` with
   `_now_ms()` and update `_relay_last`.
4. A relay's rate generator MUST accept a scalar. A `RateGroup` handles the
   per-generator conversion.

### 9.5 `create_parrot_nodes` and `register_nodes`

- `create_parrot_nodes(name, n)` creates a `RelayGroup` of `n` **readout taps**
  (`kind="tap"`, `can_source=False`, `can_target=True`), registers it, and
  returns it. It does not create `n` spiking cells unless the chosen tap
  implementation needs them.
- A tap can be the target of a `Connection` and a `create_spike_recorder`
  target; it cannot be a `Connection.source`.
- `create_relay` creates a **forwarding** relay (`kind="forward"`,
  `can_source=True`, `can_target=False`); it can be a `Connection.source` and
  a recorder target, but not a `Connection.target`.
- `register_nodes(name, nodes)` stores `nodes` in `population_nodes` and
  returns it. It accepts a `CellGroup`, `RelayGroup`, `RateGroup`, a raw cell,
  or a sequence.
- `create_connection` validates endpoint capability and raises `TypeError` on
  a wrong-side relay, instead of silently creating an inert projection.
- The NEST behavior of registering raw readout parrots is preserved for code
  that passes a NEST-like object; the NEURON implementation accepts its own
  group types.

### 9.6 `apply_explored_parameters`

The order and semantics MUST match `Registry.apply_explored_parameters`:

1. `t_now = _now_ms()`; `last_apply_errors = []`.
2. For every population in insertion order:
   - skip if `pop.node_collection is None`;
   - build `update_dict` for every sampled `pop_<name>_<p>` key:
     `param_obj.sample_population(pop.n, center=val)`;
   - if no updates, continue;
   - `pop.node_collection.set(update_dict)`; on exception append
     `(f"pop_{pop_name}", str(exc))` and continue;
   - for each applied key, `site_recorder(full_key).record_sample(t_now,
     float(val), param_obj.sigma)`.
3. For every connection in insertion order:
   - build updates for sampled `conn_<key>_<p>` keys;
   - if `cfg.weight_recorder is not None` (plastic): all sampled values are
     scalars; `self.synapses(conn_key).set(updates)`;
   - else (static):
     - `n_syn = self.synapse_count(conn_key)`;
     - if `n_syn <= 0`, append the same zero-synapse error as NEST and skip
       the parameter;
     - else `updates[p_name] = param_obj.sample_population(n_syn, center=val)`
       and `self.synapses(conn_key).set(updates)`;
   - on exception append `(f"conn_{conn_key}", str(exc))` and continue;
   - for every applied key, record the site sample at `t_now`.
4. A failure in one population/connection MUST NOT stop the others.
5. `recorded = applied` exactly. The recorder holds the realized `x̂`, not a
   fresh draw.

The NEURON `set` calls are per cell/synapse loops, but the observable behavior
is identical. `last_apply_errors` MUST be a list of `(key, message)` tuples.

### 9.7 `finalize_connections`

- Recompute and cache `_syn_counts[conn_key] = len(synapses(conn_key))`.
- Set `_plastic_n_syn[conn_key]` for plastic projections.
- `recapture_defaults()` for every registered site.
- There is no NEST connection-handle recycling problem, but the method MUST
  exist and MUST be safe to call more than once.

### 9.8 `synapses` and `synapse_count`

- `synapses(conn_key)` returns a stable `SynapseGroup`.
- If the group was not cached, build it from the projection table; do not
  return a fresh empty group.
- `synapse_count(conn_key)` returns the cached count if `finalize_connections`
  ran, else `len(synapses(conn_key))`.
- For a zero-synapse projection, return 0; do not raise.

### 9.9 `read_plastic_events` / `sample_plastic_weight_deltas`

Identical contract to NEST:

- Iterate every `connection_cfgs` entry with `weight_recorder is not None`.
- Read `events = recorder.get("events")`; convert to `np.ndarray` with
  float64 weights/times and int64 senders/targets.
- If `clear`, `recorder.set({"n_events": 0})`.
- Return `{conn_key: {"times", "weights", "senders", "targets"}}`.
- `sample_plastic_weight_deltas(t_before, t_now)`:
  - cache per `t_now` (same key, return a copy);
  - drain events;
  - per projection, seed weight = `cfg.extra_syn_spec.get("weight", 0.0)`;
  - `n_syn = _plastic_n_syn or max(1, len(events))`;
  - `delta = plastic_delta_from_events(...)` (shared helper);
  - record the producer delta `producer:stdp:<key>` at `t_now`;
  - store `_plastic_counts[key] = len(weights)`;
  - return the deltas.
- `last_plastic_counts` returns a copy.
- `sample_plastic_weight_stats` returns mean/min/max of live weights per
  plastic projection.
- `final_plastic_weights` returns arrays per plastic projection.

### 9.10 `_now_ms`

Private, but `Agent`/`Personalizer` indirectly rely on it through the
registry's site samples. It MUST return `float(h.t)` and MUST NOT raise when
NEURON has not been initialized (return 0.0 in that case).

---

## 10. Connectivity

### 10.1 `rule` values

The NEURON adapter MUST support the rule names used by the NEST model and the
memcpg reference:

| `Connection.rule` | Meaning | Parameters |
|---|---|---|
| `all_to_all` | every pre cell connects to every post cell | `p` ignored |
| `one_to_one` | pair pre[i] with post[i] | `p` ignored |
| `pairwise_bernoulli` | each pair connects with probability `p` | `p` |
| `fixed_indegree` | each post cell receives `indegree` synapses, drawn from the pre pool | `neuron["indegree"]`, `neuron["indegree_jitter"]` |
| `fixed_outdegree` | each pre cell sends `outdegree` synapses | `neuron["outdegree"]`, `neuron["outdegree_jitter"]` |

The NEST adapter currently uses only `pairwise_bernoulli`, `all_to_all` and
`one_to_one`; the extra rules are NEURON additions for memcpg. They do not
change existing NEST behavior.

### 10.2 Creation options in `Connection.neuron`

| Key | Type | Meaning |
|---|---|---|
| `kind` | `"exc" \| "inh" \| "stdp_exc" \| "stdp_inh"` | target synapse pool; defaults from `model` |
| `indegree` | int | fixed-indegree count |
| `indegree_jitter` | int | add `randint(0, jitter)` per post cell |
| `outdegree` | int | fixed-outdegree count |
| `outdegree_jitter` | int | add `randint(0, jitter)` per pre cell |
| `allow_autapses` | bool | default false |
| `allow_multapses` | bool | default true |
| `delay_cv` | float | per-synapse delay jitter, `gauss(delay, delay*delay_cv)`; default 0 |
| `weight_cv` | float | per-synapse seed jitter, `gauss(weight, weight*weight_cv)`; default 0 |
| `seed` | int | per-projection RNG seed; default registry seed |
| `threshold` | float | `NetCon.threshold`; default = the presynaptic cell's `SpikeSource.threshold` (memcpg: `-20` for neurons, `+10` for muscle), not a global constant |
| `synapse_pool` | str | explicit pool name, overrides `kind` |
| `weight_sign_mode` | `"signed" \| "magnitude"` | default `"signed"`; see §14.2 |
| `relay_kind` | `"forward" \| "tap"` | only for `create_relay`/`create_parrot_nodes`; see §8.4 |

`weight_cv`/`delay_cv` are creation-time dispersion. The default 0 reproduces
NEST's scalar seed. The PERSONA `sigma` still controls dispersion on every
`apply_explored_parameters` call; creation dispersion is separate and
optional.

The threshold rule is important for STDP: the functional NetCon uses the
source cell's threshold; the two teaching NetCons (`pre → STDP`,
`post → STDP`) use NEURON's default unless the synapse factory explicitly
sets them. `memcpg` never overrides the teaching NetCons, so the factory MUST
leave them at the default.

### 10.3 RNG and determinism

- The adapter MUST own one `np.random.Generator` (or `random.Random`) for
  connectivity, seeded by `NeuronRegistry(seed=...)`.
- `reset_kernel` MUST NOT reset the RNG unless explicitly requested; a run's
  connectivity seed is a construction input.
- The realized network is deterministic for a given adapter seed and creation
  order. It is NOT required to match NEST's realized network for the same
  numeric seed.
- `Connection.neuron["seed"]` overrides the registry seed for one projection.

### 10.4 Synapse pools

- Every postsynaptic cell type MUST pre-allocate pools matching memcpg:
  `synlistex`, `synlistinh`, `synlistexstdp`, `synlistinhstdp`.
- `create_connection` requests a point process from the pool via
  `cell_type.allocate_synapse(cell, kind)`.
- Exhaustion MUST raise a clear `RuntimeError` naming the cell, pool, and
  requested count. It MUST NOT silently drop synapses.
- `neuron["synapse_pool"]` can select a custom pool.

### 10.5 Delays

- `NetCon.delay` is in ms.
- A scalar `delay` is applied to every NetCon; `delay_cv` adds Gaussian jitter.
- A `PersonaParam("delay", ...)` in `Connection.parameters` is a site and is
  applied per NetCon by `SynapseGroup.set({"delay": ...})`.
- Delays MUST be > 0 and finite; the adapter MUST raise on invalid values.

### 10.6 Autapses and multapses

- Default `allow_autapses=False`, `allow_multapses=True`. This matches NEST's
  `pairwise_bernoulli` default and memcpg's behavior (a post cell can draw the
  same pre cell multiple times).
- For `one_to_one`, autapses are allowed only if `allow_autapses=True`.
- A multapse is represented by multiple `NetCon`s to the same or different
  point processes; `synapse_count` counts NetCons, not unique pairs.

---

## 11. Plasticity contract

### 11.1 STDP point process

The reference `memcpg/mod_files/stdp.mod` MUST be usable. It exposes:

- `RANGE`: `tauhebb`, `tauanti`, `hebbwt`, `antiwt`, `RLwindhebb`,
  `RLwindanti`, `useRLexp`, `RLlenhebb`, `RLlenanti`, `RLhebbwt`, `RLantiwt`,
  `wmax`, `softthresh`, `STDPon`, `RLon`, `verbose`, `tlastpre`, `tlastpost`,
  `tlasthebbelig`, `tlastantielig`, `interval`, `deltaw`, `newweight`, `skip`,
  `gv`;
- `POINTER synweight` pointing at the target `NetCon.weight[0]`;
- `NET_RECEIVE` updates `synweight` and clips to `[0, wmax]`.

The `stdp_exc` factory MUST wire the three connections memcpg uses:

1. `NetCon(pre_source, ExpSyn)` — the functional synapse, weight `w`;
2. `NetCon(pre_source, stdp_mech)` — pre spike, weight `+2`;
3. `NetCon(post_source, stdp_mech)` — post spike, weight `-2`;
4. `h.setpointer(netcon._ref_weight[0], "synweight", stdp_mech)`;
5. create the `STDP` mechanism on a dedicated dummy section.

The exact weights `+2/-2` are memcpg's; they are not PERSONA sites.

### 11.2 Personalizable STDP parameters

The factory MUST expose at least:

| Canonical site key | NEURON attribute | Notes |
|---|---|---|
| `tauhebb` | `stdp.tauhebb` | ms |
| `tauanti` | `stdp.tauanti` | ms |
| `hebbwt` | `stdp.hebbwt` | LTP step |
| `antiwt` | `stdp.antiwt` | LTD step |
| `wmax` | `stdp.wmax` | weight ceiling |
| `softthresh` | `stdp.softthresh` | 0/1 flag |
| `STDPon` | `stdp.STDPon` | 0/1 flag |
| `RLon` | `stdp.RLon` | 0/1 flag |
| `useRLexp` | `stdp.useRLexp` | 0/1 flag |
| `RLwindhebb` | `stdp.RLwindhebb` | ms |
| `RLwindanti` | `stdp.RLwindanti` | ms |
| `RLlenhebb` | `stdp.RLlenhebb` | ms |
| `RLlenanti` | `stdp.RLlenanti` | ms |
| `RLhebbwt` | `stdp.RLhebbwt` | gain |
| `RLantiwt` | `stdp.RLantiwt` | gain |

The canonical key is the **bare** name (`conn_<key>_wmax`), because the
`Connection.model` identifies the mechanism. `stdp.wmax` is accepted as an
alias and resolves to the same site; the registry MUST normalize aliases to
the bare canonical form in `get_all_optimizable_params` so there is one key per
parameter. A model MAY register a synapse factory that also maps NEST names
(`tau_plus`, `lambda`, `alpha`, `mu_plus`, `mu_minus`, `Wmax`) to these
attributes; the adapter does not require the translation.

The personalizable set is the table above. It does **not** include the
mechanism's state/diagnostic variables (`tlastpre`, `tlastpost`,
`tlasthebbelig`, `tlastantielig`, `interval`, `deltaw`, `newweight`, `skip`,
`gv`, `verbose`). They are readable for diagnostics but MUST NOT be exposed as
sites. `useRLexp` is a configuration flag and MAY be a site; flags are numeric
0/1 and the adapter MUST apply them as numbers, not booleans.

`Wmax`/`wmax`/`stdp.wmax` is also the producer-scale key (§9.2 step 7). The
canonical stored key is `wmax`; the producer scale uses the resolved value.

### 11.3 Weight events

The chosen implementation follows what `memcpg` already does, not a new
`stdp.mod` event stream. In `memcpg/utils_cpg.py` each STDP synapse records:

```python
weight_changes = h.Vector()
weight_changes.record(stdpmech._ref_synweight, 10.0)
leg.weight_changes_vectors.append((src_gid, post_gid, weight_changes))
```

The NEURON `WeightRecorder` MUST wrap the same mechanism:

- one `h.Vector` per plastic synapse, sampling `stdpmech._ref_synweight` at a
  fixed interval (default `10.0 ms`, matching memcpg; configurable through the
  factory);
- the recorder keeps `(pre_id, post_id)` and the previous sampled weight;
- on `get("events")` / `read_plastic_events`, it diffs each vector against the
  previous sample and synthesizes one event per changed synapse:
  `times` = the vector's sample times, `weights` = the new sampled weights,
  `senders` = `pre_id`, `targets` = `post_id`;
- `set({"n_events": 0})` truncates the vectors and resets the previous-weight
  baseline, so the next read starts a new batch;
- `sample_plastic_weight_deltas` then computes
  `Σ(w_event − last_w)/N_syn` exactly as the shared helper does.

This is **step-sample mode**, and it is the contract. No `stdp.mod` change is
required. The per-step `δ` is exact for the sampled weight trajectory; the
per-event semantics are approximated at the sampling interval. If a model
needs true per-event recording later, it MAY add an event recorder mechanism,
but that is an optional extension, not the default.

The sampling interval MUST be configurable (factory/`neuron` option) and
documented. `10.0 ms` reproduces memcpg. A registry `simulate(dt_ms)` shorter
than the interval may see zero new samples; `δ` is then 0 for that step, which
is the correct read-and-clear behavior. The conformance test in §18.6 checks
the event schema and the delta formula, not the exact event count.

### 11.4 Weight ownership

- For a plastic projection, PERSONA MUST NOT write `weight`. The weight is
  seeded once at creation and owned by STDP.
- `Connection.parameters` for a plastic projection MUST contain only the STDP
  hyper-parameters (e.g. `tauhebb`, `hebbwt`, `wmax`), not `weight`.
- `sample_plastic_weight_stats` and `final_plastic_weights` read the weights;
  they do not write them.
- `clamp_weight_sign` is not applied to a plastic weight.

### 11.5 Static weight sign policy

See §14.2.

---

## 12. Recorders

### 12.1 Spike recorder

- Attaches one `NetCon` per cell to a `h.Vector` (or a single vector with
  `NetCon.record`).
- `senders` are stable IDs.
- `times` are `h.t` values at spike time, in ms.
- Clearing sets `n_events = 0` and truncates the vectors.
- The recorder MUST be O(1) to read after clearing; do not rescan the whole
  history.
- A recorder attached to a `CellGroup` records all cells in the group. A
  recorder attached to a `RelayGroup` records the relay taps.
- A recorder attached to a single cell records that cell.
- Multiple recorders may attach to the same target.

### 12.2 Weight recorder

- Attached to a plastic projection's `SynapseGroup`.
- `get("events")` returns the four arrays.
- Clearing truncates.
- The recorder MUST be shared by all synapses of the projection; the
  `senders`/`targets` fields distinguish them.
- The recorder's `n_events` is the number of weight-update events, not the
  number of synapses.

### 12.3 Read-and-clear

The model code reads then clears (`get("events")`, then
`set({"n_events": 0})`) every step. The adapter MUST make clearing cheap and
must not retain references to cleared data. `create_spike_recorder` and
`create_weight_recorder` MUST keep strong references to their NetCons and
Vectors for the lifetime of the registry.

---

## 13. Kernel lifecycle

### 13.1 Reset

`NeuronRegistry.reset_kernel(...)` MUST:

1. Call `h.load_file("stdrun.hoc")` if not already loaded (or use the
   equivalent built-in setup).
2. Set `h.dt = resolution_ms`.
3. Set `h.cvode_active(1 if cvode else 0)`.
4. Set `pc.set_maxstep(maxstep_ms or 10.0)` (or an appropriate value).
5. Reset `h.t = 0` / call `h.finitialize(v_init)` as appropriate.
6. Not clear registered cell/synapse types (types are class-level or
   construction-level).
7. Not reach into an instance. Any existing `NeuronRegistry` is stale after
   the reset; its methods MUST raise `RuntimeError`. The caller constructs a
   new registry for the next run.

`reset_kernel` is a classmethod because it must be callable before a registry
exists. It MUST be safe to call repeatedly.

### 13.2 Initialize

`initialize(v_init=-65.0)` MUST call `h.finitialize(v_init)` and set the
registry's clock to `h.t`.

### 13.3 Simulate

`simulate(dt_ms)` MUST:

- advance by exactly `dt_ms` (within NEURON's integration accuracy);
- use `pc.psolve(h.t + dt_ms)` for MPI/parallel, or an equivalent
  `h.fadvance()` loop for single-process;
- update `_now_ms()` to the new `h.t`;
- NOT be called during network construction;
- return the new `time_ms`.

The CPG adapter MUST call `registry.simulate(dt_ms)`; it MUST NOT call
`h.fadvance`, `pc.psolve`, or `h.tstop` directly.

### 13.4 Resolution

- `dt_ms` is the NEURON integration step.
- PERSONA's `dt_ms` (the `Agent` step) and the NEURON `dt` are independent:
  the CPG adapter calls `simulate(dt_ms)` once per PERSONA step, and NEURON
  internally sub-steps at `h.dt`.
- The default `resolution_ms` SHOULD be 0.025 (memcpg's `h.dt`), not NEST's
  0.2. The model author chooses; the API does not hardcode a NEST value.
- Changing resolution after `initialize` is undefined; `set_resolution` MUST
  raise or require a re-initialize.

### 13.5 `tstop`

`simulate(dt_ms)` uses absolute `h.t + dt_ms`; the registry MUST NOT rely on a
global `h.tstop`. A model may still set `h.tstop` for its own loops, but the
registry's clock is authoritative.

---

## 14. Units, bounds, type identity

### 14.1 Units

- PERSONA values are in the **native NEURON units** of the target attribute:
  conductances in S/cm², potentials in mV, capacitance in µF/cm², resistance in
  Ω·cm, time in ms, rate in Hz, weight in the `NetCon` weight units (unitless
  conductance multiplier for `ExpSyn`/`Exp2Syn`).
- The registry MUST NOT convert units. The model author sets `mean`, `sigma`,
  `min`, `max` in native units.
- The NEST adapter's units are NEST current units; a numeric seed copied from
  NEST is not a conversion. The V-cell spec already warns about this
  (`2026-10-20-persona-variantB-cpg-vcells-design.md` §1).

### 14.2 Static weight sign

NEST uses signed weights: positive = excitatory, negative = inhibitory.
NEURON/memcpg uses positive magnitudes on an excitatory `ExpSyn` or an
inhibitory `Exp2Syn`; the synapse type carries E/I.

**The default MUST be `"signed"`**, not `"magnitude"`. Reason:
`persona/homeostasis.py` classifies a connection as `static_exc`/`static_inh`
from the sign of `parameters["weight"].mean` (or `extra_syn_spec["weight"]`
for STDP) and never reads `Connection.model`. If the site were a positive
magnitude on an inhibitory synapse, `PhysicalMap` would classify it as
excitatory and the H/game antagonism edges would be wrong. A `"magnitude"`
default would therefore require changing `homeostasis.py`, which is out of
scope.

The adapter MUST support both through `SynapseFactory.weight_sign_mode`:

- `"signed"` (default for `static_exc`/`static_inh`):
  - `clamp_weight_sign` is applied exactly as in NEST;
  - positive = excitatory, negative = inhibitory;
  - the `PersonaParam` bounds carry the sign identity (negative bounds for
    inhibitory sites);
  - the factory converts the signed site value to the hardware magnitude:
    `NetCon.weight[0] = abs(value)` for both `ExpSyn` and `Exp2Syn`;
  - `SynapseGroup.set({"weight": value})` validates that the sign agrees with
    the factory's `kind` and raises on a sign/type mismatch;
  - the site recorder records the signed value, so `recorded = applied` holds;
  - `get("weight")` returns the live positive `weight[0]`; `get("signed_weight")`
    reconstructs the signed value.
  - `Connection.model="static_synapse"` is a dispatcher on the signed seed:
    positive → `static_exc`, negative → `static_inh`, zero → `ValueError`.
    This is what keeps an unchanged NEST `Connection` working.
- `"magnitude"` (opt-in only):
  - the `PersonaParam` bounds SHOULD be `[0, max]`;
  - `clamp_weight_sign` is not applied;
  - the factory's `kind` decides E/I;
  - the model accepts that `PhysicalMap` will see all weights as positive
    unless it also changes the sign classifier. This mode is NOT recommended
    for models that use H or the conserved-stack game.

The mode MUST be visible to the model author and MUST be documented on the
built-in factories.

**Hard identity invariant (overrides the mode).** If a synapse type is
inhibitory (`kind == "inh"`, including `static_inh` and any registered
inhibitory type), the adapter MUST normalize the `weight` site to negative:
a positive mean/seed supplied with an inhibitory type is converted to
`-abs(value)`, `min` is preserved, `max` is clamped to `0.0`, and every runtime
set is rejected (or clamped to zero) if it would write a positive signed site.
The factory still writes `abs(value)` to the inhibitory `Exp2Syn` and never
swaps in an `ExpSyn`. Symmetrically, an excitatory type cannot write a
negative signed site. `static_synapse` resolves its sign before choosing the
concrete type. This invariant is independent of `weight_sign_mode`; the
`"magnitude"` mode is an opt-in convenience for models that do not use
`HomeostaticRegulator`/`PhysicalMap`, and it MUST NOT be used to flip an
inhibitory type to an excitatory one.

### 14.3 Runtime-safe vs structural parameters

Personalizable (runtime-safe) examples:

- section biophysics: `cm`, `Ra`;
- inserted mechanism parameters: `gnabar_*`, `gkbar_*`, `gl_*`, `el_*`,
  `g_pas`, `e_pas`, `gbar_nap`, `gmax_cac1`, ...;
- `NetCon`: `weight`, `delay`, `threshold`;
- `ExpSyn`: `tau`, `e`; `Exp2Syn`: `tau1`, `tau2`, `e`;
- `STDP`: all numeric `RANGE` values that are not pointers;
- `NetStim`: `start`, `interval`, `number`, `noise`;
- muscle/afferent-specific conductances and reversal potentials.

Not personalizable in v1 (structural):

- `nseg`;
- channel insertion/removal (`insert`, `uninsert`);
- section topology (`connect`, `disconnect`);
- cell class / synapse class;
- `pc.gid` assignment;
- morphology `L`, `diam` (settable before `finitialize`, but changing during a
  run invalidates geometry and point processes). A model MAY expose them as
  build-time parameters, but `apply_explored_parameters` MUST NOT change them
  mid-run.
- `POINTER`/`setpointer` targets, random generator state, and anything that
  would change the model type.

The `CellFactory`/`SynapseFactory` MUST raise a clear `ValueError` when asked to
set a structural path at runtime.

### 14.4 Bounds and type identity

- `PersonaParam.min/max` are the hard bounds; the registry clips samples to
  them exactly as NEST does.
- A population model and a connection model never change during a run.
- A static synapse stays on its E/I side: in `"signed"` mode via
  `clamp_weight_sign`; in `"magnitude"` mode via `min >= 0` and the fixed
  factory `kind`.
- An STDP hyper-parameter stays inside its biological range; the model author
  sets the bounds.

---

## 15. Error handling

### 15.1 `last_apply_errors`

- Type: `list[tuple[str, str]]`.
- Cleared at the start of every `apply_explored_parameters`.
- Appended by `create_population` (initial realization) and
  `apply_explored_parameters` (population and connection failures).
- A population failure key is `pop_<name>`; a connection failure key is
  `conn_<key>`.
- A zero-synapse static projection appends
  `(conn_<key>, "<param>: zero-synapse projection, nothing realized")`, the
  same shape as NEST.
- `Personalizer._log_new_apply_errors` must be able to print every entry
  without further parsing.

### 15.2 Exceptions

Raise (do not swallow) for:

- unknown cell/synapse type: `ValueError`;
- unknown population/connection name: `KeyError`;
- unknown parameter path at creation: `ValueError`;
- length mismatch in `CellGroup.set`/`SynapseGroup.set`: `ValueError`;
- pool exhaustion: `RuntimeError`;
- invalid delay/weight/rate: `ValueError`;
- `freeze` on an unknown site: `KeyError` (same as NEST);
- `freeze` with an unknown attribute: `ValueError` (same as NEST).

A single cell in a heterogeneous population failing to set a value is a
`last_apply_errors` entry, not a fatal exception, unless the path was invalid
at creation.

### 15.3 No silent fallbacks

The NEST code removed the mean-of-draws fallback and recorded failures. The
NEURON adapter MUST NOT reintroduce a fallback. In particular:

- a device that rejects an array is applied per device, not averaged;
- a missing mechanism is an error, not a skipped update;
- a weight outside bounds is clipped by the `PersonaParam` bounds, not
  silently coerced to a different type.

---

## 16. Parallelism and global state

### 16.1 Global `h`

NEURON's `h` is process-global. The registry is effectively a process-level
singleton:

- one `NeuronRegistry` per process in v1;
- `reset_kernel` clears the network state;
- two registries in one process are not supported (the second would see the
  first's `h` state). If needed, document and enforce a singleton guard.

### 16.2 MPI

v1 MAY be single-process. The API MUST NOT preclude MPI:

- `CellGroup`/`SynapseGroup`/`RateGroup`/`RelayGroup` MAY be local views;
- `ids`/`senders` MUST be globally stable across ranks;
- `get()`/`population_nodes` MUST be collective or replicated;
- `synapse_count` MUST be consistent across ranks (use an all-reduce or a
  replicated construction);
- `apply_explored_parameters` MUST use the same scalar sample on every rank
  (draw on rank 0 and broadcast, or use a deterministic per-site RNG);
- recorders MUST aggregate across ranks or return the local view with a
  documented rank tag.

If MPI is not in v1, the spec still requires `stable_id` and a single-process
implementation that can be extended without changing the API.

### 16.3 Determinism

- Connectivity and built-in cell morphology are deterministic for a given
  registry seed and creation order.
- `NeuronRegistry(seed=...)` seeds the adapter's RNG (`registry.rng`), which is
  used for connectivity and by built-in cell factories for morphology
  (`interneuron` dend count/diameters, `afferent` axon node count, etc.). It
  MUST NOT be the Python global `random` stream.
- NEURON's own RNG (`h.Random`, `NetStim.noiseFromRandom`, stochastic
  mechanisms) is separate. The adapter MUST expose
  `set_neuron_seed(seed: int) -> None` (or an equivalent documented method)
  that seeds `h.Random`/`pc` for models that use stochastic mechanisms. The
  `NeuronRegistry(seed=...)` constructor MAY call it as a convenience; if it
  does, document that one seed controls both streams.
- PERSONA's `PersonaParam` RNG streams are independent of the adapter.
- `reset_kernel` MUST NOT re-seed `PersonaParam`; it MAY reset the NEURON RNG
  if `set_neuron_seed` was called.
- `simulate` is deterministic for a given NEURON seed.

---

## 17. memcpg personalizable coverage

This section is the reference coverage check. The adapter MUST be able to
express every item below as a `PersonaParam` site. The model author decides
which are actually optimized; the API must not block them.

### 17.1 Cell types and attributes

#### `interneuron` (`memcpg/interneuron.py`)

| Path | Mechanism / section | Notes |
|---|---|---|
| `soma.gnabar_fastchannels` | `fastchannels` | Na conductance |
| `soma.gkbar_fastchannels` | `fastchannels` | K conductance |
| `soma.gl_fastchannels` | `fastchannels` | leak |
| `soma.el_fastchannels` | `fastchannels` | leak reversal |
| `soma.cm` | section | capacitance |
| `soma.Ra` | section | axial resistance |
| `soma.gbar_nap` | `NaP` (bursting only) | bursting current |
| `soma.gnabar_hh` | `hh` (axon in some variants) | if inserted |
| `soma.gkbar_hh` | `hh` | if inserted |
| `soma.gl_hh` | `hh` | if inserted |
| `soma.el_hh` | `hh` | if inserted |
| `dend.g_pas` | `pas` | all dend sections |
| `dend.e_pas` | `pas` | all dend sections |
| `dend.cm` / `dend.Ra` | section | optional |
| `axon.Ra` | section | optional |
| `axon.gnabar_hh` / `gkbar_hh` / `gl_hh` / `el_hh` | `hh` | if inserted |

`interneuron` with `serotonin_mode=True` inserts `fastchannels` in dend and
adds `slow_5HT`/`r5ht3a`; their numeric `RANGE` values are runtime-settable and
SHOULD be exposable with the same path grammar if the model needs them.

#### `motoneuron` (`memcpg/motoneuron.py`, `memcpg/axon.py`)

| Path | Mechanism / section | Notes |
|---|---|---|
| `soma.gnabar_motoneuron` | `motoneuron` | Na |
| `soma.gcaL_motoneuron` | `motoneuron` | L-type Ca |
| `soma.gl_motoneuron` | `motoneuron` | leak |
| `soma.gkrect_motoneuron` | `motoneuron` | rectifier K |
| `soma.gcak_motoneuron` | `motoneuron` | Ca-activated K |
| `soma.cm` / `soma.Ra` | section | capacitance / axial |
| `dend.g_pas` / `dend.e_pas` / `dend.Ra` / `dend.cm` | `pas` / section | |
| `node.gnabar_hh` / `gkbar_hh` / `gl_hh` / `el_hh` | `hh` | axon nodes |
| `MYSA.g_pas` / `MYSA.e_pas` | `pas` | axon paranodes |
| `FLUT.g_pas` / `FLUT.e_pas` | `pas` | axon |
| `STIN.g_pas` / `STIN.e_pas` | `pas` | axon internodes |
| `soma.ena` / `soma.ek` | section | reversal potentials if used |
| `diam` | build-time | diameter; NOT runtime-personalizable |

The `motoneuron` diameter (`diam`) selects conductance seeds in
`biophys` (`if self.diam > 50`). It is build-time structural; it can seed a
`PersonaParam` but cannot be changed mid-run.

#### `muscle` (`memcpg/muscle.py`)

| Path | Mechanism / section | Notes |
|---|---|---|
| `muscle_unit.cm` / `Ra` | section | |
| `muscle_unit.g_pas` / `e_pas` | `pas` | |
| `muscle_unit.gnabar_fastchannels` / `gkbar_fastchannels` | `fastchannels` | |
| `muscle_unit` `CaSP`/`fHill`/`xm` numeric RANGEs | mechanisms | if exposed |
| `soma.cm` / `Ra` | section | |
| `soma.gmax_cac1` | `cac1` | |
| `soma.gbar_na14a` | `na14a` | |
| `soma.gkbar_kir` | `kir` | |
| `soma.gnabar_fastchannels` / `gkbar_fastchannels` / `gl_fastchannels` / `el_fastchannels` | `fastchannels` | |
| `soma.gkmax_K_No` | `K_No` | |
| `soma.gcalbar_cal` | `cal` | |
| `soma.ena` / `soma.ek` | section | |
| `syn.tau` / `syn.e` | `ExpSyn` | excitatory pool |
| `syn.tau1` / `syn.tau2` / `syn.e` | `Exp2Syn` | inhibitory pool |

#### `afferent` (`memcpg/bioaffrat.py`, `memcpg/axon.py`)

| Path | Mechanism / section | Notes |
|---|---|---|
| `soma.gnabar_hh` / `gkbar_hh` / `gl_hh` / `el_hh` | `hh` | |
| `soma.cm` / `Ra` | section | |
| `node.gnabar_hh` / `gkbar_hh` / `gl_hh` / `el_hh` | `hh` | axon nodes |
| `MYSA.g_pas` / `MYSA.e_pas` | `pas` | |
| `FLUT.g_pas` / `FLUT.e_pas` | `pas` | |
| `STIN.g_pas` / `STIN.e_pas` | `pas` | |
| `syn.tau` / `syn.e` | `ExpSyn` | `synlistees`/`synlistex` |
| `syn.tau1` / `syn.tau2` / `syn.e` | `Exp2Syn` | `synlistinh` |

### 17.2 Synapses

| Path | Mechanism | Scope |
|---|---|---|
| `weight` | `NetCon.weight[0]` | per synapse |
| `delay` | `NetCon.delay` | per synapse |
| `nc.threshold` | `NetCon.threshold` | per synapse |
| `syn.tau` | `ExpSyn.tau` | per synapse |
| `syn.e` | `ExpSyn.e` / `Exp2Syn.e` | per synapse |
| `syn.tau1`, `syn.tau2` | `Exp2Syn` | per synapse |
| `stdp.*` | `STDP` mechanism | per projection |

### 17.3 Generators

| Path | Mechanism | Scope |
|---|---|---|
| `rate` | converted to `NetStim.interval` | per generator |
| `interval` | `NetStim.interval` | per generator |
| `start` | `NetStim.start` | per generator |
| `number` | `NetStim.number` | per generator |
| `noise` | `NetStim.noise` | per generator |

### 17.4 Players and keys

- A population player is a `Population` with `params`; site keys are
  `pop_<name>_<path>`.
- A synapse player is a `Connection` with `parameters`; site keys are
  `conn_<key>_<path>`.
- A rate generator registered by `create_relay(rate_param=...)` is a
  `Population(model="poisson_generator")`; its site is `pop_<name>_rate`.
- A readout relay (`create_parrot_nodes`) is a boundary node: it is in
  `population_nodes` but has no `Population` config, so `PhysicalMap` skips
  it. This matches NEST.
- STDP producers are `producer:stdp:<key>`; rate producers are
  `producer:poisson:<group>`. They are metric-only, not players.

---

## 18. Conformance tests

The other team MUST add these. They are the acceptance criteria for "same
level of integration".

### 18.1 API surface parity

- Import `Adapter`, `NestAdapter`, `NeuronAdapter`, `Registry` and
  `NeuronRegistry`.
- Assert `NestAdapter` and `NeuronAdapter` subclass `Adapter`; assert
  `Registry` and `NeuronRegistry` are compatible aliases/subclasses.
- For every public method/attribute of `Adapter`, assert both concrete classes
  implement it. Abstract methods are checked with
  `inspect.isabstract`/`__abstractmethods__`.
- Compare `inspect.signature` parameter names, kinds and defaults between
  `Adapter` and both concrete implementations. Return annotations may differ.
- `__init__` is the one allowed exception: `NeuronRegistry` MAY add
  keyword-only arguments (`seed`, `kernel`). The test MUST compare the shared
  parameters as a subset and require the extras to be keyword-only with
  defaults.
- Assert the public attributes in §7.7 exist on both concrete classes and have
  the right types.
- Assert `persona.adapter` does not import `nest` or `neuron`.
- This test can skip if NEST or NEURON is unavailable, but it MUST run in the
  WSL `KinematicsCPG` environment where both are installed.

### 18.2 Key parity

- Create one population and one connection with the same `Population`/
  `Connection` objects used by the NEST tests; assert the canonical keys are
  identical (`pop_*`, `conn_*`, `producer:*`, `site:*`).
- `freeze`/`create_spring` glob resolution works on dotted NEURON paths.

### 18.3 Creation and realization

- Create a small network with `interneuron` → `interneuron` using
  `static_exc`, `pairwise_bernoulli`, and a `PersonaParam` weight.
- `len(group) == n`, `synapse_count(key) > 0`.
- `apply_explored_parameters({...})` changes the live `NetCon.weight[0]`
  values and the site recorder matches the requested `x̂`.
- Repeat with `static_inh`, `fixed_indegree`, `delay`, and per-synapse
  `syn.tau`.

### 18.4 Freeze and spring

- The pure-Python freeze/spring tests from `tests/test_registry_freeze.py`
  and `tests/test_registry_spring.py` MUST pass against `NeuronRegistry` with
  a small NEURON network.
- `frozen_attrs`, `defaults`, `is_frozen`, `freeze`, `unfreeze`,
  `recapture_defaults` produce identical results to the NEST registry for the
  same site table.
- `Agent.step` with a frozen mean returns `delta_means[key] == 0.0`.

### 18.5 Recorders

- `create_spike_recorder(group)` records spikes; `get("events")` returns
  `times`/`senders`; `set({"n_events": 0})` clears.
- `create_weight_recorder()` attached to `stdp_exc` returns
  `times`/`weights`/`senders`/`targets`; clearing works; `sample_plastic_weight_deltas`
  matches `Σ Δw / N_syn` on a synthetic event batch.
- The recorder uses memcpg's `h.Vector().record(stdpmech._ref_synweight,
  interval_ms)` with `interval_ms=10.0` by default; no `stdp.mod` change.
- A recorder attached to a `RelayGroup` works.
- Read-and-clear is O(new events), not O(history).

### 18.6 Plasticity

- `stdp_exc` builds: functional `NetCon`, pre/post teaching `NetCon`s,
  `setpointer` to `weight[0]`.
- Changing `tauhebb`, `hebbwt`, `wmax` through `apply_explored_parameters`
  changes the live `STDP` mechanism.
- Weight updates stay within `[0, wmax]`.
- `sample_plastic_weight_stats` and `final_plastic_weights` match the live
  weights.
- Step-sample recording (memcpg's `synweight` vectors) yields the correct
  per-step `δ`; a step shorter than the sampling interval yields 0 with no
  stale read.

### 18.7 Single route

`tests/test_neuron_route.py`:

- Scan `persona/` (except `persona/neuron/` and `persona/neuron_adapter.py`),
  the NEURON CPG package, and the entry points.
- Forbid `import neuron`, `from neuron import`, `h.NetCon`, `h.NetStim`,
  `h.Section`, `h.ExpSyn`, `h.Exp2Syn`, `pc.gid_connect`, `pc.cell`,
  `pc.set_gid2node`.
- Assert the only allowed locations are `persona/neuron_adapter.py`,
  `persona/neuron/`, and registered factory modules.
- This is the NEURON analogue of `tests/test_nest_route.py`.

### 18.8 memcpg smoke

- Build a minimal network with `interneuron`, `motoneuron`, `muscle`,
  `afferent`; connect with `static_exc`/`static_inh`/`stdp_exc`; run
  `simulate(10)`; record spikes and weights.
- This proves the built-in types cover the reference mechanisms.
- It need not reproduce memcpg's full topology.

### 18.9 Agent integration

- `Agent(NeuronRegistry(), ...)` constructs.
- Run 5 steps with a dummy output/target; `means` move, `sigmas` stay in
  bounds, `F`/`Xbar` are finite.
- `Personalizer` can run a 2-step loop with a fake bench or a small NEURON
  network; it must not need simulator imports.

### 18.10 Regression

- The entire existing NEST test suite passes unchanged.
- The NEST implementation moves to `persona/nest_adapter.py`; `persona/
  registry.py` becomes a compatibility shim. The only semantic changes are:
  (a) `NestAdapter` implements `Adapter` (kernel delegations included);
  (b) the additive `neuron` fields in `Population`/`Connection`;
  (c) the pure helpers/exact-key-first resolver moving to
      `persona/adapter.py` while keeping the public exports;
  (d) exact-key-first resolution in `_resolve_site_keys` (required for
      `dend[0]` paths);
  (e) the hard inhibitory sign normalizer, which only activates for explicit
      inhibitory/excitatory model names or an explicit kind.
  Any other change is an API defect.
- `persona/adapter.py` MUST NOT import `nest` or `neuron`.
- `t2xcpg`, `memcpg`, `tinycpg` are not modified.

---

## 19. Suggested implementation order

This is for the other team; it is not part of the contract.

1. **Freeze the interface.** Add the `neuron` fields, the `CellGroup`/
   `SynapseGroup`/`RateGroup`/`RelayGroup`/recorder contracts, and the API
   parity test. No simulation yet.
2. **Kernel facade.** `reset_kernel`, `set_resolution`, `initialize`,
   `simulate`, `time_ms`.
3. **Groups and recorders.** `CellGroup`, `SpikeRecorder`, `RateGroup`,
   `RelayGroup`; test with a trivial `IntFire`/`Section`.
4. **Cell factories.** `interneuron`, `motoneuron`, `muscle`, `afferent`;
   attribute path resolver.
5. **Static synapses and connectivity.** `static_exc`, `static_inh`,
   `pairwise_bernoulli`, `all_to_all`, `one_to_one`, `fixed_indegree`.
6. **Apply and site bookkeeping.** Copy/share the NEST freeze/spring/site
   layer; run the parity tests.
7. **STDP and weight recorder.** `stdp_exc`, memcpg-style step-sampled
   `synweight` vectors, `sample_plastic_weight_deltas`.
8. **memcpg smoke and Agent integration.**
9. **MPI** (if in scope).
10. **Source-scan test and documentation.**

---

## 20. Decisions (resolved 2026-10-23)

The human partner answered the open questions. These are now contract, not
defaults.

| # | Decision | Answer |
|---|---|---|
| 1 | **Adapter shape** | One pure abstract `Adapter(ABC)` in `persona/adapter.py`; `NestAdapter(Adapter)` in `persona/nest_adapter.py`; `NeuronAdapter(Adapter)` in `persona/neuron_adapter.py`. `Registry` and `NeuronRegistry` remain compatibility aliases/subclasses. No `SiteBook` mixin. Do not rewrite NEST behavior; make the existing methods the NEST implementation. |
| 2 | **Built-in cell scope** | All five memcpg cell types: `interneuron`, `bursting_interneuron`, `motoneuron`, `muscle`, `afferent`. |
| 3 | **STDP recording** | Use what `memcpg` already does: `h.Vector().record(stdpmech._ref_synweight, 10.0)` step-sampling. No `stdp.mod` change. Event-level recording is an optional future extension only. |
| 4 | **MPI scope** | Single-process v1, API kept MPI-ready. |
| 5 | **Parameter naming** | NEURON-native names are canonical (`tauhebb`, `wmax`, `soma.*`, `syn.*`); NEST names are optional aliases. |
| 6 | **Shared site layer** | No `SiteBook`. `Adapter` declares the common methods as abstract; pure module-level helpers (`clamp_weight_sign`, `plastic_delta_from_events`, `resolve_site_keys`, `enforce_synaptic_sign`) are called by both concrete adapters. The NEST class keeps its behavior. |
| 7 | **Weight sign convention** | Signed site values (negative = inhibitory); the NEURON factory converts to a positive hardware conductance. `HomeostaticRegulator`/`PhysicalMap` stay unchanged. |
| 8 | **Kernel ownership** | `Adapter` declares `reset_kernel`/`set_resolution`/`initialize`/`simulate`/`time_ms`; NEST delegates to `nest`, NEURON to `h`. Model code calls the adapter, never `h.fadvance`. |

One technical requirement remains shared: exact-key-first resolution for
bracketed paths such as `dend[0].g_pas` (`fnmatch` treats `[0]` as a character
class). It MUST be implemented in the shared `_resolve_site_keys`/`Adapter`
layer; it is behavior-preserving for existing NEST keys.

---

## 21. References

- `persona/registry.py` — normative NEST API.
- `docs/superpowers/specs/2026-10-19-persona-single-sigma-persistent-strategy-design.md`
  §5.2, §5.3, §5.5 — realization, noise recorders, NEST integration.
- `docs/superpowers/specs/2026-10-22-persona-registry-usability-design.md`
  — freeze, cosine LR, springs.
- `docs/superpowers/specs/2026-10-21-persona-homeostasis-h-design.md`
  §3 — players, physical map, site keys.
- `docs/superpowers/specs/2026-10-20-persona-variantB-cpg-vcells-design.md`
  §1 — NEURON↔NEST units, V-cell reference.
- `memcpg/interneuron.py`, `memcpg/motoneuron.py`, `memcpg/muscle.py`,
  `memcpg/bioaffrat.py`, `memcpg/axon.py`, `memcpg/utils_cpg.py`,
  `memcpg/mod_files/stdp.mod` — reference cell/synapse parameters.
- `tests/test_nest_route.py` — single-route source scan precedent.
