# Design (revision 2): CPG circuit — scheme-authoritative V-cell relay

Date: 2026-10-20
Status: approved (chat, 2026-10-20)
Supersedes the CPG sections (§3, §5.3, §5.4 and the CPG part of §5.6) of
`2026-10-15-persona-variantB-cpg-mvp-design.md`. The PERSONA strategy sections
there remain as revised; where they conflict with
`2026-10-20-persona-recursive-fisher-inverse-design.md`, the latter wins.

## 1. Authority

The sources are ranked. When they disagree, the higher rank wins:

1. **The detailed scheme (2026-10-20, no Ib)** and the **previous scheme**
   (the same drawing plus Ib) are the *architecture* authority. The user
   confirmed the detailed scheme is newer than `cpg_2legs_fast`.
2. **`memcpg` (NEURON)** supplies populations and weights for the commissural
   V-cell relay. `cpg_2legs_fast` only abstracts that circuit as
   "V0v/V0d/V2a/In1 classes" and wires direct `RG↔RG` inhibition instead.
3. **`cpg_2legs_fast`** supplies values where the scheme is silent and the
   population exists there (half-centre, motor output, Ia reciprocal, CUT).
4. **Project additions** are listed explicitly in §4. They are not implied by
   the scheme and must not be inferred from it.

## 2. Circuit

### 2.1 Ipsilateral (per leg)

Half-centre, through the InE/InF relays:

| projection | weight | p | source |
|---|---:|---:|---|
| `RGE → InE` | +12.0 | 0.15 | `W_RG2INE` |
| `InE ⊣ RGF` | −8.0 | 0.15 | `W_INE2RGF` |
| `RGF → InF` | +18.0 | 0.30 | `W_RG2INF` |
| `InF ⊣ RGE` | −48.0 | 0.30 | `W_INF2RGE` |
| `RGE → RGE` | +4.0 | 0.12 | `W_RG_REC_E` |
| `RGF → RGF` | +5.5 | 0.12 | `W_RG_REC_F` |

Motor output: `RGE→M-E`, `RGF→M-F` at 2.75 (p 0.50); `M-E→mE`, `M-F→mF` at
1.0 (p 0.80). There is **no** `M-E↔M-F` projection (§3.2).

Ia (spindle):

| projection | weight | p | note |
|---|---:|---:|---|
| `Ia-E → M-E`, `Ia-F → M-F` | +12.0 | 0.40 | project addition (§3.3) |
| `Ia-E → RGE`, `Ia-F → RGF` | +12.0 | 0.40 | STDP, project addition (§3.3) |
| `Ia-E → Ia-IN-E`, `Ia-F → Ia-IN-F` | +6.0 | 0.40 | `W_IA_IN2INT` |
| `Ia-IN-E ⊣ M-F`, `Ia-IN-F ⊣ M-E` | −10.0 | 0.40 | `W_IA_INT2ANT` |
| `Ia-E → InE`, `Ia-F → InF` | +6.0 | 0.25 | `W_IA2IN` |

Renshaw (project addition; no `cpg_2legs_fast` counterpart):
`M→R` +3.0 (p 0.50), `R⊣M` −4.0 (p 0.50), `R⊣Ia-IN` −2.4 (p 0.40). The full
gain pinned the motor pools at the 0.2 ms step cap; 0.2× is the operating
point.

Ib (previous scheme; project values):

| projection | weight | p | note |
|---|---:|---:|---|
| `GTO-E → InIb-E`, `GTO-F → InIb-F` | +16.0 | 0.40 | force afferent |
| `InIb-E ⊣ M-E`, `InIb-F ⊣ M-F` | −18.0 | 0.40 | autogenic inhibition |
| `InIb-E → M-F`, `InIb-F → M-E` | +12.0 | 0.35 | antagonist MN |
| `InIb-E → RGF`, `InIb-F → RGE` | **+18.0** | 0.35 | phase switch (§3.1) |
| `Ia-E → InIb-E`, `Ia-F → InIb-F` | +8.0 | 0.30 | Ia/CPG convergence |

CUT: `CUT→RGE` +14.0 (p 0.35, STDP), `CUT→InE` +6.0 (p 0.30).

Commissural V-cells (NEURON `memcpg/leg.py`):

| projection | weight | p | delay | type |
|---|---:|---:|---:|---|
| `RGF → V2a` | +0.75 | 0.50 | 3 | exc |
| `RGF → V0d` | +0.75 | 0.50 | 3 | exc |
| `RGF → V3F` | +1.5 | 0.50 | 3 | exc |
| `V2a → V0v` | +1.2 | 0.50 | 3 | exc |
| `In1 ⊣ RGF` | −0.5 | 0.50 | 1 | inh |

V2a/V0v/V3F are regular-spiking (`a .02 b .2 c −65 d 8`); V0d/In1 are
fast-spiking (`a .1 b .2 c −65 d 2`). Each population has `n_int` neurons.

### 2.2 Commissural (replaces direct `RG↔RG`)

From `memcpg/utils_cpg.py::add_external_connections`, mirrored L↔R:

| projection | weight | p | delay | type |
|---|---:|---:|---:|---|
| `V0v → contralateral In1` | +1.3 | 0.40 | 3 | exc |
| `V0d ⊣ contralateral RGF` | −1.3 | 0.40 | 3 | inh |
| `V3F → contralateral RGF` | +0.5 | 0.40 | 3 | exc |

**Removed:** `RGF-L ⊣ RGF-R` (−20, p 0.22) and `RGE-L ⊣ RGE-R` (−8, p 0.10).
There is no direct `RG↔RG` commissural projection left. `V3F→RGF` is 0.0 in
NEURON's WALK mode and 0.5 in RUN; the scheme draws the projection, so 0.5 is
the seed and PERSONA may tune it to zero (walk-only configuration).

### 2.3 Drives

`BS_in_e→RGE` and `BS_in_f→RGF`: 4.0 × leading gain (1.5 on the leading leg L),
p 0.45. `BASE_in→RGE/RGF`: 1.0, p 0.10, delay 1.0. Reference rate 700 Hz
(`DEFAULT_REF_HZ`), chosen from the measured motor p95 (569 extensor / 342
flexor) rather than the old 100 Hz, which clipped 44 % of the drive.

## 3. Decisions (confirmed 2026-10-20)

### 3.1 Ib → RG is excitatory

The detailed scheme draws the Ib projection into the CPG as a "phase-switch
signal". §5.3 of the previous design resolves that as `contralateral-phase RG
exc`: rising extensor force must drive the flexor half-centre, not clamp it.
`W_IB2RG` is therefore **+18.0** (magnitude from the reference pair
`W_IN2RG`, sign resolved by the spec). Code and test are flipped together:
`tests/test_ib_halfcentre_sign.py::test_ib_excites_antagonist_rg` asserts
`> 0.0` and guards every `in_ib_* → antagonist RG` projection.

### 3.2 There is no `M-E ↔ M-F` reciprocal inhibition

The detailed scheme draws no projection between the two motor pools: M-E and
M-F connect only to their Renshaw cell, the Ia inhibitory cell and the muscle.
`W_MOTOR_RECIP = −22.0` belongs to the older `cpg_2legs_fast` architecture and
is **not** part of this circuit.

Measured consequence of adding it anyway: the motor pools go from 192/95 Hz to
2072/2079 Hz (L) and 2500/2500 Hz (R), i.e. saturation, not alternation. The
motor stage is bistable, so this is not a harmless missing wire.

### 3.3 Homonymous `Ia→M` and `Ia→RG` STDP are kept as project additions

The scheme draws only the reciprocal Ia pathway (`Ia-E ⊣ M-F`); it does not
draw the homonymous branches. Both branches nevertheless exist in the NEURON
model:

- `Ia_aff → mns` weight 1.55 (homonymous monosynaptic stretch reflex),
- `Ia_aff → RG` weight 0.3 with STDP.

Dropping them would remove the fastest plant feedback loop and the plastic half
of the Ia pathway. They stay as **explicit project additions**, documented
here so they are not mistaken for scheme wiring. Our seeds are project-tuned
and much larger than NEURON's (Ia→M 12.0 vs 1.55; Ia→RG 12.0 vs 0.3); they are
registered `PersonaParam`s and flagged for a future reference-alignment pass.

The `Ia→RG` projection remains always-on (STDP), not gated behind an option:
D12 makes it part of the learned Ia drive.

## 4. Deviations / project additions

| item | scheme | NEURON | code | status |
|---|---|---|---|---|
| V-cell relay | yes | yes | yes | implemented, `e7e9cf9` |
| direct `RG↔RG` commissural | no | no | no | removed, `e7e9cf9` |
| Ib arc | previous scheme | no | yes | project addition |
| Renshaw | yes | no | yes | project values, 0.2× gain |
| homonymous `Ia→M` | not drawn | yes (1.55) | yes (12.0) | project addition (§3.3) |
| `Ia→RG` STDP | not drawn | yes (0.3) | yes (12.0) | project addition (§3.3) |
| `M-E↔M-F` | no | yes (−22) | no | correctly absent (§3.2) |
| V3F cross | yes | WALK 0.0 / RUN 0.5 | 0.5 | flag for walk-only |
| BS drive | yes | regular desync spikes | Poisson relay re-commanded every 10 ms | open (T4) |

## 5. Tests

- `tests/test_v_cells.py` — populations, ipsilateral weights, commissural
  weights and signs, absence of direct `RG↔RG`, neuron types.
- `tests/test_ib_halfcentre_sign.py` — Ib→RG excitatory, no RG→Ib drive, the
  Ib reflex arc survives.
- `tests/test_cpg_topology.py` — half-centre via InE/InF, reference values,
  Renshaw gain, BS seed weights.
- `tests/test_cpg_build.py` — all 15 populations per leg.
- Full suite: **164 passed** (excluding the untracked
  `tests/test_bench_recursive_inverse.py`, which is concurrent work).

## 6. Measured behaviour at FINAL

`test_results/vcell_probe.py/.out`: 1 s simulated, 0.2 ms, BS 60 Hz, no
sensory input.

- V-cells are alive: V2a 37/24 Hz, V0v 13/11, V0d 90/54, V3F 64/42, In1 9/10.
- RG remains E/F co-active (L +0.58, R +0.77); L/R `RGF` correlates +0.685 at
  lag 0, i.e. the legs are in phase.
- The relay is roughly 10× weaker than the direct −20/−8 it replaced (the
  cascade multiplies 0.75 × 1.3, etc.).
- The V-cell swap is architecturally correct but does **not** fix C1. The
  co-activation is generated upstream in the half-centre / BS regime, not in
  the commissural path.

## 7. Open items

1. Walk-only configuration: set `V3F→RGF` to 0.0 (NEURON WALK) if the model is
   not meant to run.
2. Reference-alignment pass for the Ia project weights (12.0 → 1.55 / 0.3) once
   the motor stage is out of its bistable regime.
3. Decide whether the V-cell relay should be scaled up to have the functional
   strength of the old direct projection; PERSONA can tune the registered
   weights only if the tracking loss stops being flat.
4. C1 remains the half-centre / BS drive problem. The commissural architecture
   is no longer a suspect.
