# CPG Drive Balance Implementation Plan (T1, T2, T11, T12)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the two tonic projections onto the rhythm generators back to a physiological operating point so the RG pools stop saturating at ≥100 Hz, and make the RG readout report its true magnitude instead of clipping at 1.0.

**Architecture:** Two static projections dominate the RG's input: `BS_in_{e,f} -> rg_e/rg_f` (built in `t2xcpg/cpg.py`) and `cut_in_{L,R} -> rg_e/rg_f` (built in `t2xcpg/leg.py`). Each one *alone* holds an RG pool at >100 Hz, so the reciprocal/commissural circuitry never gets to shape alternation and the muscle drive lands at a constant 1.0. Fix by lowering both initial weights into the measured working band, and by un-clipping the RG readout so the magnitude is visible.

**Tech Stack:** Python 3.12, NEST 3.10, pytest, MuJoCo 3.14 (for the A/B validation only).

**Spec:** `docs/superpowers/specs/2026-10-15-persona-variantB-cpg-mvp-design.md`

## Global Constraints

- Do not modify anything under `persona/`, `tinycpg/`, `memcpg/`, or legacy `persona.py`. This plan touches `t2xcpg/` and `tests/` only.
- Commit with explicit pathspecs only (`git commit -- <paths>`); never `git add -A`/`-u`.
- Git identity is passed per-command: `git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit ...`.
- WSL test command: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest -q"`.

## Measured Evidence (drives the chosen values)

Each projection measured alone against a bare RG-E pool of 100 Izhikevich neurons
(`test_results/bs_cut_sweep.py`); BS = 40 Hz Poisson, p 0.45; CUT = 90 Hz Poisson, p 0.35.

| initial weight | BS alone | CUT alone | BS+CUT |
|---|---|---|---|
| 27.0 (BS now, leading) | 243 Hz | — | — |
| 14.0 (CUT now) | 93 Hz | 146 Hz | 196 Hz |
| 3.5 | 19 Hz | 31 Hz | 46 Hz |
| **2.0 (chosen)** | **9.6 Hz** | **17 Hz** | **25 Hz** |
| 1.0 | 0.2 Hz | 6.1 Hz | 12 Hz |

Chosen: **`W_BS2RG = 2.0`** (keeps `leading_gain = 1.5` → leading leg 3.0, trailing 2.0) and **`W_CUT2RG = 2.0`**. Both sit far below the 100 Hz saturation band, their co-activation lands ≈25 Hz, and PERSONA is left to fine-tune from there (T11). The reference's own init constants (`W0_IN = 22.0`) are *not* a safe starting point — they reproduce the saturation.

## Review Focus

- **A silent threshold crossing is now possible.** Lowering the drives means the RG could sit at 0 Hz instead of saturating, and a permanently silent CPG looks like "working" to every existing test. Task 2's co-activation test must assert a *lower* bound, not just an upper one.
- **The leading/trailing asymmetry grows relatively.** `leading_gain = 1.5` on a base of 2.0 makes the leading leg 50 % stronger; on a base of 18 it was 50 % of a saturated signal. Task 1 pins both legs' rates.
- **`get_rg_rates` callers may assume [0, 1].** Un-clipping lets it exceed 1.0. Any consumer that treats it as an activation must be found. Task 4 greps for them.
- **Population-size sensitivity.** The mapping above was measured at n=100. At `--fast` sizes (n_rg=6) a single spike moves the readout by 1/6, so the band assertions must tolerate quantisation.
- **Q1 vs steady state.** In the current `--fast` run the first quarters show partial alternation and later quarters saturate, so a run-length assertion can pass or fail on duration alone. Every test fixes the number of steps explicitly.

---

### Task 1: Tune the brainstem→RG initial weight (T1)

**Files:**
- Modify: `t2xcpg/cpg.py` (the `_build_bs_connections` method)
- Test: `tests/test_cpg_drive_balance.py` (create)

**Interfaces:**
- Consumes: `t2xcpg.cpg.L2CPG(registry, bs, legs, dt_ms, leading_leg, leading_gain, base)`.
- Produces: `t2xcpg.cpg.W_BS2RG: float` — the per-leg base weight before `leading_gain`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cpg_drive_balance.py`:

```python
"""The tonic projections onto the rhythm generators must not saturate them.

Measured before the fix (100-neuron RG pool): the brainstem projection alone held
RG-E at 243 Hz and the cutaneous projection alone at 146 Hz, so `rg_e_L` read a
flat 1.0 for a whole trial and the muscle drive was pinned at 1.0.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from persona.parameter import reset_rng_stream
from persona.registry import Registry
from model.adapter import CPGInput
from t2xcpg.bs import BrainStem, BaseDrive
from t2xcpg.cpg import L2CPG
from t2xcpg.leg import Leg

DT_MS = 10.0
REF_HZ = 100.0
FAST = dict(n_rg=6, n_mn=5, n_int=5, n_aff=3, n_bs=5)


def _run(cut=0.0, ia=0.0, ib=0.0, bs_hz=40.0, steps=60, seed=0):
    """Build a two-leg CPG and return the per-step RG rate (units of ref_hz)."""
    import nest
    reset_rng_stream(seed)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=FAST["n_bs"], bs_e_hz=bs_hz,
                   bs_f_hz=bs_hz)
    base = BaseDrive(registry=reg, n_bs=FAST["n_bs"], base_hz=2.0)
    legs = {s: Leg(s, reg, n_rg=FAST["n_rg"], n_mn=FAST["n_mn"],
                   n_int=FAST["n_int"], n_aff=FAST["n_aff"], ref_hz=REF_HZ)
            for s in "LR"}
    cpg = L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=DT_MS, base=base)

    cpg_in = CPGInput()
    for side in "LR":
        setattr(cpg_in, f"cut_{side}", cut)
        for m in "EF":
            setattr(cpg_in, f"ia_rate_{m}_{side}", ia)
            setattr(cpg_in, f"ib_rate_{m}_{side}", ib)

    rates = []
    for _ in range(steps):
        cpg.update(cpg_in)
        out = cpg.step()
        rates.append(out.extra["rg"]["rg_e_L"])
    return np.asarray(rates, dtype=float)


def test_brainstem_alone_does_not_saturate_the_rhythm_generators(nest_kernel):
    rates = _run(cut=0.0, ia=0.0, ib=0.0, bs_hz=40.0)
    # ref_hz=100 -> 0.6 is 60 Hz. The RG must be driven, but not pinned.
    assert rates.mean() < 0.6, f"BS alone pins the RG at {rates.mean() * REF_HZ:.0f} Hz"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_cpg_drive_balance.py -q"`

Expected: FAIL — `BS alone pins the RG at ~200 Hz` (readout clips at 1.0 until Task 4 lands, so the assertion message reports ≥100 Hz).

- [ ] **Step 3: Add the constant and use it**

In `t2xcpg/cpg.py`, above `class L2CPG`, add:

```python
# Tonic brainstem -> rhythm generator weight. The reference (`cpg_2legs_fast`)
# relies on this projection being weak enough that the cutaneous co-activation
# gate decides when the RG fires. Measured: the previous value (18.0, x1.5 on the
# leading leg) held an isolated RG pool at 243 Hz, i.e. the gate was always open.
# 2.0 gives ~10 Hz on this pool and ~25 Hz when co-activated with CUT.
W_BS2RG = 2.0
```

Replace the two weight arguments in `_build_bs_connections`:

```python
            self._connect("BS_in_e", f"rg_e_{leg_key}", weight=W_BS2RG * gain, p=0.45)
            self._connect("BS_in_f", f"rg_f_{leg_key}", weight=W_BS2RG * gain, p=0.45)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_cpg_drive_balance.py -q"`

Expected: PASS. If the mean is still >0.6, lower `W_BS2RG` to 1.5 and re-run; do not go below 1.0 (the pool goes silent and Task 2's lower bound will fail).

- [ ] **Step 5: Commit**

```bash
git add t2xcpg/cpg.py tests/test_cpg_drive_balance.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "fix(cpg): stop the brainstem projection saturating the rhythm generators" -- t2xcpg/cpg.py tests/test_cpg_drive_balance.py
```

---

### Task 2: Tune the cutaneous→RG initial weight and pin the co-activation gate (T2, T11)

**Files:**
- Modify: `t2xcpg/leg.py` (the `_build_connections` method, the CUT block)
- Test: `tests/test_cpg_drive_balance.py` (append)

**Interfaces:**
- Consumes: `t2xcpg.cpg.W_BS2RG` from Task 1.
- Produces: `t2xcpg.leg.W_CUT2RG: float` — the cutaneous→RG base weight.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cpg_drive_balance.py`:

```python
def test_cut_alone_does_not_saturate_the_rhythm_generators(nest_kernel):
    rates = _run(cut=90.0, ia=0.0, ib=0.0, bs_hz=0.0)
    assert rates.mean() < 0.6, f"CUT alone pins the RG at {rates.mean() * REF_HZ:.0f} Hz"


def test_coactivation_drives_the_rhythm_generators_without_saturating(nest_kernel):
    """The reference's design: neither pathway alone decides, their sum does."""
    rates = _run(cut=90.0, ia=0.0, ib=0.0, bs_hz=40.0)
    assert 0.02 < rates.mean() < 0.6, (
        f"co-activation must drive the RG into a working band, got "
        f"{rates.mean() * REF_HZ:.0f} Hz")


def test_both_legs_land_in_the_same_band(nest_kernel):
    """`leading_gain` makes the leading leg stronger; it must not saturate it."""
    import nest
    from persona.parameter import reset_rng_stream
    from persona.registry import Registry
    from model.adapter import CPGInput
    from t2xcpg.bs import BrainStem, BaseDrive
    from t2xcpg.cpg import L2CPG
    from t2xcpg.leg import Leg

    reset_rng_stream(0)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=FAST["n_bs"], bs_e_hz=40.0, bs_f_hz=40.0)
    base = BaseDrive(registry=reg, n_bs=FAST["n_bs"], base_hz=2.0)
    legs = {s: Leg(s, reg, n_rg=FAST["n_rg"], n_mn=FAST["n_mn"],
                   n_int=FAST["n_int"], n_aff=FAST["n_aff"], ref_hz=REF_HZ)
            for s in "LR"}
    cpg = L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=DT_MS, base=base)
    cpg_in = CPGInput()
    for side in "LR":
        setattr(cpg_in, f"cut_{side}", 90.0)
        for m in "EF":
            setattr(cpg_in, f"ia_rate_{m}_{side}", 0.0)
            setattr(cpg_in, f"ib_rate_{m}_{side}", 0.0)
    seen = {"rg_e_L": [], "rg_e_R": []}
    for _ in range(60):
        cpg.update(cpg_in)
        out = cpg.step()
        for k in seen:
            seen[k].append(out.extra["rg"][k])
    for k, v in seen.items():
        assert 0.02 < np.mean(v) < 0.6, f"{k} mean {np.mean(v) * REF_HZ:.0f} Hz"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_cpg_drive_balance.py -q"`

Expected: `test_cut_alone_...` FAILS (CUT alone ≈146 Hz); `test_both_legs_...` FAILS (both ≈200 Hz). `test_coactivation_...` may already pass on the upper bound — confirm it fails on the upper bound *before* Task 2 changes the weight; if it passes, do not weaken it, just note that the co-activation guard is carried by the other two.

- [ ] **Step 3: Add the constant and use it**

In `t2xcpg/leg.py`, beside the existing `# W_M2MUS, P_M2MUS` comment (near line 276), add:

```python
# Cutaneous -> rhythm generator weight (`cpg_2legs_fast` W_CUT2RGE/P_CUT2RGE).
# The reference intends CUT alone to be sub-threshold and to gate the RG only in
# combination with the brainstem drive. Measured: the previous value (14.0) held
# an isolated RG pool at 146 Hz, and in the closed loop CUT sits near 90 Hz for
# the whole trial, so that gate was permanently open.
W_CUT2RG = 2.0
```

Replace the weight at the CUT→RG-E connection in `_build_connections`:

```python
        self._connect(f"cut_in_{leg}", f"rg_e_{leg}", weight=W_CUT2RG, p=0.35,
                      is_stdp=True, key=f"cut_rge_{leg}")   # W_CUT2RGE
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_cpg_drive_balance.py -q"`

Expected: 4 passed. If the co-activation rate falls below 0.02 (2 Hz), raise `W_CUT2RG` to 2.5 — the lower bound protects against a silent CPG.

- [ ] **Step 5: Commit**

```bash
git add t2xcpg/leg.py tests/test_cpg_drive_balance.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "fix(cpg): bring the cutaneous gate back to a working band" -- t2xcpg/leg.py tests/test_cpg_drive_balance.py
```

---

### Task 3: Confirm the muscle drive leaves 1.0 at full population sizes

**Files:**
- Test: `tests/test_cpg_drive_balance.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: nothing new.

This is the acceptance test for the whole plan. At `POPULATIONS_FINAL` the muscle
drive was exactly 1.0 for all 400 steps of a run and the last-10 loss was
identical to 15 significant digits across CPG variants. After Tasks 1–2 both must
change.

- [ ] **Step 1: Write the test**

```python
def test_muscle_drive_varies_at_full_population_size(nest_kernel):
    """Regression for the saturated-drive symptom (was exactly 1.0 for a run)."""
    from t2xcpg.cfg import POPULATIONS_FINAL

    sizes = POPULATIONS_FINAL
    import nest
    from persona.parameter import reset_rng_stream
    from persona.registry import Registry
    from model.adapter import CPGInput
    from t2xcpg.bs import BrainStem, BaseDrive
    from t2xcpg.cpg import L2CPG
    from t2xcpg.leg import Leg

    reset_rng_stream(0)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})
    reg = Registry()
    bs = BrainStem(registry=reg, n_bs=sizes["n_bs"], bs_e_hz=40.0, bs_f_hz=40.0)
    base = BaseDrive(registry=reg, n_bs=sizes["n_bs"], base_hz=2.0)
    legs = {s: Leg(s, reg, n_rg=sizes["n_rg"], n_mn=sizes["n_mn"],
                   n_int=sizes["n_int"], n_aff=sizes["n_aff"], ref_hz=REF_HZ)
            for s in "LR"}
    cpg = L2CPG(registry=reg, bs=bs, legs=legs, dt_ms=DT_MS, base=base)
    cpg_in = CPGInput()
    for side in "LR":
        setattr(cpg_in, f"cut_{side}", 90.0)
        for m in "EF":
            setattr(cpg_in, f"ia_rate_{m}_{side}", 40.0)
            setattr(cpg_in, f"ib_rate_{m}_{side}", 1.0)
    drives = []
    for _ in range(100):
        cpg.update(cpg_in)
        drives.append(cpg.step().mus_E_L)
    drives = np.asarray(drives, dtype=float)
    assert drives.std() > 0.0, "muscle drive is constant -> the plant cannot learn"
    assert drives.max() < 1.0 or drives.mean() < 0.95, (
        f"muscle drive still saturated: mean {drives.mean():.4f}")
```

- [ ] **Step 2: Run it**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_cpg_drive_balance.py -q"`

Expected: PASS. If it fails on `drives.std() > 0.0`, the RG is still saturating — return to Task 1 and lower `W_BS2RG` one step.

- [ ] **Step 3: Run the whole suite**

Run: the Global Constraints test command.

Expected: all pass. Report by name any failure you did not cause.

- [ ] **Step 4: Commit**

```bash
git add tests/test_cpg_drive_balance.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "test(cpg): pin the muscle drive leaving saturation at full sizes" -- tests/test_cpg_drive_balance.py
```

---

### Task 4: Let the RG readout report beyond 1.0 (T12)

**Files:**
- Modify: `t2xcpg/leg.py` (`_drive`, `get_rg_rates`)
- Test: `tests/test_cpg_drive_balance.py` (append)

**Interfaces:**
- Produces: `Leg._drive(spikes: int, dt_ms: float, n: int | None = None, clip: bool = True) -> float` — unchanged signature plus the keyword-only-ish `clip` parameter, default `True` so every existing caller keeps its behaviour.
- Produces: `Leg.get_rg_rates(dt_ms, current_time_ms=0.0) -> Tuple[float, float]` — now **unclipped**; 1.0 means 100 Hz and 2.27 means 227 Hz.

- [ ] **Step 1: Write the failing test**

```python
def test_drive_can_report_beyond_reference_rate(nest_kernel):
    leg = Leg("L", Registry(), n_rg=10, n_mn=10, n_int=6, n_aff=6, ref_hz=REF_HZ)
    # 30 spikes from a 10-neuron pool in 10 ms = 300 Hz, i.e. 3.0 x ref_hz
    assert np.isclose(leg._drive(30, 10.0, 10), 1.0)
    assert np.isclose(leg._drive(30, 10.0, 10, clip=False), 3.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `... -m pytest tests/test_cpg_drive_balance.py::test_drive_can_report_beyond_reference_rate -q`

Expected: FAIL — `TypeError: _drive() got an unexpected keyword argument 'clip'`.

- [ ] **Step 3: Implement**

In `t2xcpg/leg.py`, give `_drive` a `clip` parameter and apply it to the result;
leave `get_muscle_drives` untouched (it must stay in [0, 1] — it is a muscle
command). Change only the RG readout in `get_rg_rates` to pass `clip=False`, and
note in its docstring that values above 1.0 mean "above `ref_hz`".

- [ ] **Step 4: Relax the one existing assertion that depended on the ceiling**

The pre-flight scan already ran this search. The only consumers of `rg_*` outside
tests are `persona/personalization.py:361-364`, which append the value to the run
log — nothing downstream treats it as an activation. Inside tests, exactly one
assertion depended on the clip: `tests/test_review_fixes.py:241`

```python
assert 0.0 <= rg_e <= 1.0 and 0.0 <= rg_f <= 1.0
```

That test (`test_rg_recorders_are_read_and_cleared`) is about the recorders being
read and cleared, not about the scale, so relax it to the property that survives:

```python
assert rg_e >= 0.0 and rg_f >= 0.0
```

Do not delete the assertion — non-negativity is still the contract. The positive
proof that the ceiling is gone lives in this task's own unit case
(`_drive(30, 10.0, 10, clip=False) == 3.0`). Do not touch anything else in
`tests/test_review_fixes.py`; in particular its `18.0` reciprocal weight
assertion is unrelated to this plan.

- [ ] **Step 5: Run the suite and commit**

Run: the Global Constraints test command. Expected: all pass, including
`tests/test_review_fixes.py`.

```bash
git add t2xcpg/leg.py tests/test_cpg_drive_balance.py tests/test_review_fixes.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "feat(cpg): report the rhythm generator rate past the reference ceiling" -- t2xcpg/leg.py tests/test_cpg_drive_balance.py tests/test_review_fixes.py
```

- [ ] **Step 6: Re-run the A/B and record the numbers**

Run the fast A/B twice per Global Constraints, once with `--mode legacy`, saving
under `test_results/drv2_fast` and `test_results/drv2_final`, then:

```
/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python test_results/drive_granularity.py test_results/drv2_fast/run.h5 test_results/drv2_final/run.h5
/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python test_results/rhythm_check.py test_results/drv2_final/run.h5
```

Expected: `rg_e_L` spans many distinct values (not `{1.0}`) and the last-10 loss
differs from `0.8489750236571337`. Record both in the commit message of the final
task or in the plan's execution notes.
