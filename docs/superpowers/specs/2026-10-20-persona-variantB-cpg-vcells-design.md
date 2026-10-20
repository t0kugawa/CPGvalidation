# Design (revision 2): CPG V-cell commissural relay + scheme-confirmed Ia/Ib wiring

**Status:** accepted 2026-10-20.
**Supersedes:** §5.3 and the CPG value table in §5.4 of
`2026-10-15-persona-variantB-cpg-mvp-design.md`. Everything else in that
document (PERSONA, render, `σ_param`, optimised parameter set) still applies.

## 1. Authority order

The drawings are newer than both code references. When they disagree:

1. **Architecture** (which populations exist and which projections connect
   them) follows the *detailed scheme* (`Снимок экрана 2026-10-20 210356`),
   with the Ib overlay from the earlier scheme (`… 204446`).
2. **Values** follow `cpg_2legs_fast` where the scheme is silent, and the
   NEURON model (`memcpg/leg.py`, `memcpg/utils_cpg.py`) for populations and
   weights that the NEST reference only abstracts.
3. `cpg_2legs_fast`'s direct `RG↔RG` commissural projection is **not**
   architecture. It is the abstraction that the V-cell relay replaced
   (`cpg_2legs_fast.py:1479`: "abstracts V0v/V0d/V2a/In1 classes").

The port's synaptic weights are NEST current units; the NEURON model's are
conductances. The numbers below are copied from the reference as initial
values, exactly as the existing `RG→MN = 2.75` was; they are not a unit
conversion.

## 2. What changed in revision 2

- Added the commissural V-cell relay: **V2a, V0v, V0d, V3-F, In1**.
- Removed the direct commissural `RGF⊣RGF` (−20, p 0.22) and `RGE⊣RGE`
  (−8, p 0.10).
- **Ib→RG is excitatory (+18.0).** The scheme draws the InIb→RG projection
  with an excitatory arrow (the "CPG phase-switch signal"), and the spec text
  says "contralateral-phase RG exc". The previous −18.0 came from
  `W_IN2RG`, a `cpg_2legs_fast` constant that is defined but never connected;
  the sign was the error, the magnitude is kept.
- Confirmed **absent**: direct `M-E⊣M-F` / `M-F⊣M-E`.
- Confirmed **present**: homonymous `Ia→M` and `Ia→RG`.

## 3. Ipsilateral CPG (per leg)

### 3.1 Half-centre, self-recurrence, motor output

| Projection | Weight | p | Notes |
|---|---:|---:|---|
| `RGE→InE` | 12.0 | 0.15 | `W_RG2INE`, `P_RG_RECIP_E` |
| `InE⊣RGF` | −8.0 | 0.15 | `W_INE2RGF` |
| `RGF→InF` | 18.0 | 0.30 | `W_RG2INF` |
| `InF⊣RGE` | −48.0 | 0.30 | `W_INF2RGE` |
| `RGE→RGE` | 4.0 | 0.12 | `W_RG_REC_E` |
| `RGF→RGF` | 5.5 | 0.12 | `W_RG_REC_F` |
| `RGE→M-E` | 2.75 | 0.50 | `W_RG2MN` |
| `RGF→M-F` | 2.75 | 0.50 | `W_RG2MN` |
| `M-E→mE` | 1.0 | 0.80 | `W_M2MUS` |
| `M-F→mF` | 1.0 | 0.80 | `W_M2MUS` |

No direct `RGE⊣RGF` / `RGF⊣RGE`; no `InE/InF→MN`.

### 3.2 Ia (stretch) and Renshaw

| Projection | Weight | p | Notes |
|---|---:|---:|---|
| `Ia-E→M-E` | 12.0 | 0.40 | homonymous, **scheme-confirmed** |
| `Ia-F→M-F` | 12.0 | 0.40 | homonymous |
| `Ia-E→RGE` | 12.0 (STDP seed) | 0.40 | homonymous, **scheme-confirmed** |
| `Ia-F→RGF` | 12.0 (STDP seed) | 0.40 | homonymous |
| `Ia-E→Ia-IN-E` | 6.0 | 0.40 | `W_IA_IN2INT` |
| `Ia-F→Ia-IN-F` | 6.0 | 0.40 | |
| `Ia-IN-E⊣M-F` | −10.0 | 0.40 | `W_IA_INT2ANT` |
| `Ia-IN-F⊣M-E` | −10.0 | 0.40 | |
| `Ia-E→InE` | 6.0 | 0.25 | `W_IA2IN`, `P_IA2IN` |
| `Ia-F→InF` | 6.0 | 0.25 | |
| `M-E→R-E` | 3.0 | 0.50 | project value |
| `R-E⊣M-E` | −4.0 | 0.50 | project value |
| `R-E⊣Ia-IN-E` | −2.4 | 0.40 | project value |
| `M-F→R-F` | 3.0 | 0.50 | project value |
| `R-F⊣M-F` | −4.0 | 0.50 | project value |
| `R-F⊣Ia-IN-F` | −2.4 | 0.40 | project value |

`Ia→RG` is plastic (STDP); the weight is seeded, PERSONA optimises the STDP
hyper-parameters. `Ia→M` is static.

### 3.3 Ib (force), with the phase-switch sign corrected

| Projection | Weight | p | Notes |
|---|---:|---:|---|
| `GTO-E→InIb-E` | 16.0 | 0.40 | stretch/force afferent |
| `InIb-E⊣M-E` | −18.0 | 0.40 | autogenic inhibition |
| `InIb-E→M-F` | 12.0 | 0.35 | antagonist excitation |
| `InIb-E→RGF` | **+18.0** | 0.35 | **phase switch, excitatory** |
| `GTO-F→InIb-F` | 16.0 | 0.40 | |
| `InIb-F⊣M-F` | −18.0 | 0.40 | autogenic inhibition |
| `InIb-F→M-E` | 12.0 | 0.35 | antagonist excitation |
| `InIb-F→RGE` | **+18.0** | 0.35 | **phase switch, excitatory** |
| `Ia-E→InIb-E` | 8.0 | 0.30 | Ia/CPG convergence |
| `Ia-F→InIb-F` | 8.0 | 0.30 | |

### 3.4 CUT

| Projection | Weight | p | Notes |
|---|---:|---:|---|
| `CUT→RGE` | 14.0 (STDP seed) | 0.35 | `W_CUT2RGE_STATIC`, `P_CUT2RGE_STATIC` |
| `CUT→InE` | 6.0 | 0.30 | `W_CUT2INE` |

### 3.5 V-cell populations and ipsilateral inputs

Populations per leg (size `n_int`, 50 at FINAL):

- **V2a, V0v, V3-F** — excitatory (`a .02 b .2 c −65 d 8`).
- **V0d, In1** — inhibitory (`a .1 b .2 c −65 d 2`).

| Projection | Weight | Delay (ms) | p | Sign |
|---|---:|---:|---:|---|
| `RGF→V2a` | 0.75 | 3 | 0.50 | excitatory |
| `RGF→V0d` | 0.75 | 3 | 0.50 | excitatory |
| `RGF→V3-F` | 1.5 | 3 | 0.50 | excitatory |
| `V2a→V0v` | 1.2 | 3 | 0.50 | excitatory |
| `In1⊣RGF` | −0.5 | 1 | 0.50 | inhibitory |

`In1` has **no ipsilateral RG input**; it is driven only by the contralateral
`V0v` (see §4).

## 4. Commissural V-cell relay

Replaces the direct `RG↔RG` projections:

```
RGF_L → V2a_L → V0v_L → In1_R ⊣ RGF_R
RGF_L → V0d_L ⊣ RGF_R
RGF_L → V3-F_L → RGF_R
```

and the mirror image.

| Projection | Weight | Delay (ms) | p | Sign |
|---|---:|---:|---:|---|
| `V0v_L→In1_R` | 1.3 | 3 | 0.40 | excitatory |
| `V0v_R→In1_L` | 1.3 | 3 | 0.40 | excitatory |
| `V0d_L⊣RGF_R` | −1.3 | 3 | 0.40 | inhibitory |
| `V0d_R⊣RGF_L` | −1.3 | 3 | 0.40 | inhibitory |
| `V3-F_L→RGF_R` | +0.5 | 3 | 0.40 | excitatory |
| `V3-F_R→RGF_L` | +0.5 | 3 | 0.40 | excitatory |

**V3-F cross weight.** NEURON's WALK mode sets `v3f_cross_weight = 0.0` and
RUN sets `0.5` (`memcpg/bs_command.py`). The scheme draws the projection, so
the port seeds the enabled RUN value; a walk-only model may set it to zero.
PERSONA can tune it down (positive weights have `min = 0`).

**Density.** NEURON's `connectcells` gives each target ~50 synapses drawn
from the presynaptic pool; at `n_int = 50` that is ≈0.63 unique-pair
probability. The port uses the uniform `p = 0.50` for ipsilateral V-cell
projections and `p_comm = 0.40` for commissural ones. `p` is structural and is
never sampled by PERSONA.

## 5. Scheme confirmations (evidence)

Checked against the upscaled drawings, not from the old reference:

1. **`M-E↔M-F` is absent.** The detailed scheme has no line between the two
   motor-neuron diamonds. They connect only to their muscle, their Renshaw
   cell, and the Ia inhibitory cell. `W_MOTOR_RECIP = −22` from
   `cpg_2legs_fast` is old architecture and is **not** used. (A probe adding
   it saturated the motor pools, 192/95 Hz → 2072/2079 Hz, so it would also
   need the motor stage out of its bistable regime first.)
2. **Homonymous `Ia→M` and `Ia→RG` are present.** The yellow Ia-afferent line
   from the muscle spindle branches into three arrowheads: **M-E**, **RGE**,
   and **Ia-E**. The blue `Ia-E` cell projects only to the antagonist `M-F`.
   So the homonymous excitation is from the afferent, not from the inhibitory
   Ia interneuron. Both port projections are correct.
3. **`Ib→RG` is excitatory.** The InIb-E→RGF and InIb-F→RGE projections are
   drawn as dashed brown arrows with excitatory arrowheads, labelled
   "CPG phase-switch signal"; the same figure's text says "CPG phase-switch
   signal" and the 2026-10-15 spec says "contralateral-phase RG exc". The
   autogenic `InIb→homonymous MN` is the inhibitory one. Code and test now
   use `+18.0` (sign corrected; magnitude unchanged).

## 6. Implementation

- `t2xcpg/leg.py`: V-cell populations and ipsilateral projections;
  `W_IB2RG = +18.0`.
- `t2xcpg/cpg.py::_build_commissural_connections`: the V-cell relay.
- `tests/test_v_cells.py`: populations, ipsilateral weights, commissural
  weights/signs, absence of direct `RG↔RG`, neuron types.
- `tests/test_ib_halfcentre_sign.py`: asserts `InIb→antagonist RG` is
  excitatory and `InIb⊣homonymous MN` is inhibitory.

## 7. Validation status (2026-10-20)

Suite: **164 passed** (excluding the untracked
`tests/test_bench_recursive_inverse.py`, which imports a module not yet on the
path).

FINAL probe, 1 s simulated at 0.2 ms, BS 60 Hz, no sensory input
(`test_results/vcell_probe.out`):

- RG L 378/129 Hz (E/F corr +0.580), R 236/75 Hz (+0.772).
- Motor L 191/89 Hz (+0.829), R 144/53 Hz (+0.905).
- L/R `RGF` corr +0.685 at lag 0 → the legs are still in phase.
- V-cells fire: V2a 37/24, V0v 13/11, V0d 90/54, V3-F 64/42, In1 9/10 Hz.

The relay is alive but ~10× weaker than the removed direct `−20/−8`, and it
does not break E/F co-activation or create L/R alternation. The co-activation
is upstream (half-centre / BS drive), not commissural. If the relay is meant
to carry functional interlimb coupling, its weights need scaling after the
half-centre is out of the co-active regime.

## 8. Runtime resolution (2026-10-21)

The closed loop must run at **0.2 ms**, the `cpg_2legs_fast` default. The port
had hardcoded **1.0 ms** in `personalize.py`; at that step the izhikevich
network locks into a resolution-induced limit cycle and the drive readout
saturates. Same FINAL network, zero sensory input, seed 0:

| resolution (ms) | mE (Hz) | mF (Hz) | rgE (Hz) | rgF (Hz) |
|---:|---:|---:|---:|---:|
| 0.1 | 75 | 63 | 105 | 136 |
| 0.2 | 197 | 83 | 421 | 141 |
| 0.5 | 180 | 38 | 476 | 27 |
| **1.0** | **471** | **500** | **500** | **459** |
| 2.0 | 252 | 248 | 248 | 175 |

At 1.0 ms both motor pools sit at the step ceiling (~500 Hz = one spike every
two steps). Fixed sensory inputs barely move them (flexor: no sensory 500,
IaF 100 422, IaF 250 405 Hz; removing the homonymous `Ia_F→M_F` raised it to
476 — an attractor flip), so the **step, not the synaptic weights, sets the
rate**. That is why PERSONA could not move the flexor: the flexor drive was
pinned at 1.0 by numerics, not by the CPG.

`ref_hz=700` was calibrated at 0.2 ms (p95 569 extensor / 342 flexor); running
the loop at 1.0 ms made the calibration and the dynamics disagree.

The production path now uses `DEFAULT_RESOLUTION_MS = 0.2`, exposes
`--resolution-ms` in `run_validation.py` and `run_2min.py`, and records
`resolution_ms` in `run.h5`. `tests/test_run_resolution.py` guards the default
and checks that the flexor drive stays graded at 0.2 ms. The 33 s server log
(1.0 ms) is not a valid C1 measurement; rerun at 0.2 ms. The 0.2 ms NEST step
is 5× more work than 1.0 ms, so use `--realize-every-k` to bound wall time.

## 9. Open issues

- Half-centre co-activation (E/F corr +0.58…+0.91; reference alternates).
- V-cell relay gain: 0.75/1.2/1.3/1.5 vs the removed −20/−8. Scaling is a
  tuning decision; PERSONA can sample the weights, but the tracking loss is
  still flat while the plant input saturates.
- NEURON→NEST unit mapping for the V-cell weights is approximate (µS → current).
- The `M-E↔M-F` probe result (saturation) means any future re-introduction
  needs the motor stage re-based first.
