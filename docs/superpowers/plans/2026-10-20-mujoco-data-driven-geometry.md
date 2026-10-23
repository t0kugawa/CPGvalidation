# Data-Driven MuJoCo Plant Geometry Implementation Plan (T8, T9)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive every trajectory-dependent MuJoCo parameter (joint ranges, muscle `lengthrange`) from the loaded dataset at build time, so no constant in `dual_rat_hindlimb.xml` can clip a different subject's trajectory, and so the SOL tendon stops being clipped for 45 % of the trial.

**Architecture:** Two new `Leg` methods compute what MuJoCo currently *author* by hand: `fit_joint_ranges` writes each hinge's `jnt_range` from the dataset's observed excursion, and `fit_muscle_lengthranges` writes each muscle actuator's `actuator_lengthrange` from the tendon excursion over those fitted ranges. `personalize.build_personalizer` calls both right after `set_limb_lengths`. The hardcoded `lengthrange` attributes are deleted from the XML, so the model cannot fall back to a stale window.

**Tech Stack:** Python 3.12, MuJoCo 3.14, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-persona-variantB-cpg-mvp-design.md`

## Global Constraints

- Do not modify anything under `persona/`, `tinycpg/`, `memcpg/`, or legacy `persona.py`. This plan touches `model/`, `personalize.py`, `dual_rat_hindlimb.xml`, and `tests/` only.
- Commit with explicit pathspecs only (`git commit -- <paths>`); never `git add -A`/`-u`.
- Git identity is passed per-command: `git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit ...`.
- WSL test command: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest -q"`.
- The dataset used by every test is the one `personalize.py` already loads (RatA x1p50 Lfast T1) via `loader.mocap.MocapLoader`.
- Segment naming is off-by-one relative to anatomy and must not be "fixed": the body named `shank_{p}_{s}` carries the *thigh* segment, `foot_{p}_{s}` the *shank*, `toe_{p}_{s}` the *foot*. `set_limb_lengths` already encodes this correctly.

## Root Cause (T9, measured)

`test_results/t8_tendon_probe.py` drives the model along the dataset trajectory and compares the tendon's real excursion with the actuator's window:

| rig | SOL excursion | authored window | steps outside the window | worst shortfall |
|---|---|---|---|---|
| XML defaults | 2.679 .. 15.714 mm | 2.700 .. 19.500 mm | 306 / 5995 (5 %) | 0.021 mm |
| after `set_limb_lengths` (ratios 0.822 / 0.892 / 0.934) | 2.009 .. 14.279 mm | 2.700 .. 19.500 mm | **2725 / 5995 (45 %)** | **0.691 mm** |

`test_results/mujoco_hooks_probe.py` shows where the authored numbers came from: deleting the attribute makes MuJoCo auto-compute `sol 2.67869 .. 20.2606 mm`, `ta 0.15714 .. 25.45536 mm` — the excursion over the **XML's hardcoded joint ranges** (`ankle -1.0 .. 2.6`) on the **unscaled** rig. So the authored `0.0027 0.0195` is that auto-range hand-rounded, with the SOL top additionally *lowered* (20.26 → 19.5). Three compounding causes:

1. The window was authored for the *joint-range* excursion, not the observed trajectory.
2. The SOL top was manually narrowed below even that.
3. `set_limb_lengths` rescales the bones from the subject (0.82–0.93×), which shifts the excursion to 2.014 .. 18.442 mm over the same joint ranges, and nothing recomputes the window.

**What the fault actually costs (corrected by measurement).** The first hypothesis — a clipped force — is *wrong*. `test_results/t9_gain_probe.py` evaluates MuJoCo's own `mju_muscleGain` / `mju_muscleBias` at every step of the trial: the passive bias is **0.0000 N for both windows**, so nothing is clamped. What `lengthrange` really does is set the abscissa that normalises the muscle's length. Over the trial the tendon occupies `l = -0.041 .. +0.689` of the authored window against `l = -0.000 .. +0.747` of a fitted one, and the resulting active gain is **0.904–0.990× (median 0.913)** the fitted value — a systematic, cycle-varying ~9 % under-drive of SOL.

So T9 is a real but *modest* numerical fault at this scale. It is worth fixing because of T8's requirement, not because of the 9 %: the window is a constant in the XML that cannot follow the subject's scale or joint ranges, so every new dataset re-introduces it. The TA window (0.5 .. 24.8) happens to be generous enough to stay outside this, which is exactly why a constant is the wrong design — the fault lands on whichever muscle the next subject shortens.

## Review Focus

- **Clamping width is a modelling choice with a real trade-off.** Fitting the window to the *observed* trajectory makes the force–length curve span exactly the operating range, but any closed-loop excursion past the data's range is then normalised outside `[0, 1]`. Measured, that is *forgiving*: the passive bias stays 0 N outside the window and only the active gain is extrapolated, so a too-narrow window degrades smoothly rather than clipping. That is why this plan fits over the **fitted joint ranges** (which guarantee every reachable pose is inside) instead of the tighter observed trajectory — the safer of two acceptable choices, not the only one.
- **The fitted joint ranges now bound the plant.** The ankle's authored `-1.0 .. 2.6` becomes ≈`-0.24 .. 2.04`, so the ankle is physically confined to roughly the trial's range. That is the point of "load any data", but it also removes overshoot headroom; `margin_frac` is the knob.
- **The toe joint cannot be data-derived.** `loader.mocap` emits `hip_*`, `knee_*`, `ankle_*` only — there is no toe/metatarsal angle, because the ML model has no toe DOF and the MH marker is the foot segment's endpoint. `toe_m_jnt_*` / `toe_ref_jnt_*` keep their XML ranges, and Task 1 must not silently pretend otherwise.
- **Muscle `force` is not derivable from this dataset.** The CSV has kinematics (and metadata), not forces, so `force`/`scale`/`timeconst` cannot be "calculated from the CSV". Task 3 keeps them in the XML as an explicit, commented physiological block keyed by muscle rather than leaving them unmarked.
- **Runtime writes work, but `qfrc_constraint` is a trap.** Verified in `test_results/plan_api_probe.py` and `test_results/jnt_range_probe2.py`: a joint pushed past a runtime-written `jnt_range` settles exactly on the new limit (`1.0 → -0.0500` for a `-0.05 .. 0.05` window), identical to recompiling the XML with that range; and writing `actuator_lengthrange` changes `actuator_force`. However `qfrc_constraint` reads **0.0000** on the first `mj_forward` after moving `qpos`, so a test that asserts on it instead of stepping will wrongly conclude the limit is ignored. This plan therefore asserts on `jnt_range` (config) and on `actuator_force` (observable), never on `qfrc_constraint`.
- **Grid resolution.** `fit_muscle_lengthranges` samples the joint space. Task 2 validates it against MuJoCo's own compile-time auto-computation, which is a genuine independent check of the sampler.

---

### Task 1: Fit the joint ranges from the dataset

**Files:**
- Modify: `model/leg.py` (add `fit_joint_ranges` after `set_limb_lengths`, ~line 361)
- Test: `tests/test_data_driven_geometry.py` (create)

**Interfaces:**
- Consumes: `angles: Dict[str, np.ndarray]` — the mapping returned by `MocapLoader.resample()`, with keys `hip_{s}`, `knee_{s}`, `ankle_{s}` (radians), `hip_x_{s}`, `hip_z_{s}`, `mh_z_{s}` (mm), `seg_thigh_{s}`, `seg_shank_{s}`, `seg_foot_{s}` (mm).
- Produces: `Leg.fit_joint_ranges(angles, margin_frac: float = 0.05) -> Dict[Tuple[str, str], Tuple[float, float]]` — sets `model.jnt_range` for `hip`/`knee`/`ankle` on both `m` and `ref` prefixes and both sides, widens the matching `position` servos' `actuator_ctrlrange` so they can reach the new limits, and returns `{(joint, side): (lo, hi)}`. `toe` joints are not in the dataset and are left untouched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data_driven_geometry.py`:

```python
"""Trajectory-dependent MuJoCo parameters must be computed from the dataset.

Before this, `dual_rat_hindlimb.xml` carried hand-authored `jnt_range` and muscle
`lengthrange` values tuned on one rig at one scale. `set_limb_lengths` then rescales
the bones to the subject, so the authored windows go stale: measured on RatA the
SOL tendon spent 2725 of 5995 steps below its window's floor (45 % of the trial).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

from loader.mocap import MocapLoader
from model.leg import Leg

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")
REF = os.path.join(HERE, "data/zenodo/DatasetMotionCaputure/RatA",
                   "RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv")
DT_MS = 10.0


@pytest.fixture(scope="module")
def angles():
    return MocapLoader(REF, dt_ms=DT_MS).resample()


@pytest.fixture()
def rig():
    return Leg(mujoco_cfg=XML, dt_ms=DT_MS)


def _scaled_rig(rig, angles):
    """The rig as `personalize.build_personalizer` leaves it."""
    for s in ("L", "R"):
        rig.set_limb_lengths(
            "m", s,
            float(np.mean(angles[f"seg_thigh_{s}"])) / 1000.0,
            float(np.mean(angles[f"seg_shank_{s}"])) / 1000.0,
            float(np.mean(angles[f"seg_foot_{s}"])) / 1000.0)
        rig.set_limb_lengths(
            "ref", s,
            float(np.mean(angles[f"seg_thigh_{s}"])) / 1000.0,
            float(np.mean(angles[f"seg_shank_{s}"])) / 1000.0,
            float(np.mean(angles[f"seg_foot_{s}"])) / 1000.0)
    return rig


def test_joint_ranges_cover_the_observed_trajectory(rig, angles):
    _scaled_rig(rig, angles)
    fitted = rig.fit_joint_ranges(angles, margin_frac=0.05)
    for joint in ("hip", "knee", "ankle"):
        for side in ("L", "R"):
            jid = {"hip": rig.j_hip_m, "knee": rig.j_knee_m,
                   "ankle": rig.j_ankle_m}[joint][side]
            lo, hi = rig.model.jnt_range[jid]
            obs = np.asarray(angles[f"{joint}_{side}"], dtype=float)
            assert lo <= obs.min() and obs.max() <= hi, (
                f"{joint}_{side} range {lo:.3f}..{hi:.3f} excludes observed "
                f"{obs.min():.3f}..{obs.max():.3f}")
            assert (lo, hi) == pytest.approx(fitted[(joint, side)])


def test_servo_ctrlranges_follow_the_fitted_joint_ranges(rig, angles):
    """A hardcoded `ctrlrange` on a position servo clips the command, so a trial
    with a wider excursion would be silently unfollowable."""
    import mujoco
    _scaled_rig(rig, angles)
    rig.fit_joint_ranges(angles, margin_frac=0.05)
    checked = 0
    for joint, jmap in (("hip", rig.j_hip_m), ("knee", rig.j_knee_m)):
        for side in ("L", "R"):
            for prefix in ("m", "ref"):
                name = f"{joint}_{prefix}_ctrl_{side}"
                aid = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if aid < 0:
                    continue
                lo, hi = rig.model.jnt_range[jmap[side]]
                ctrl = rig.model.actuator_ctrlrange[aid]
                assert ctrl[0] <= lo and hi <= ctrl[1], (
                    f"{name} ctrlrange {ctrl} cannot reach {lo:.3f}..{hi:.3f}")
                checked += 1
    assert checked >= 8, "expected the hip/knee servos of both limbs"


def test_joint_ranges_are_data_dependent(rig, angles):
    """A different trial must produce different limits (the 'load any data' ask)."""
    scaled = {"hip_L": np.asarray(angles["hip_L"]) * 0.0 - 0.9,
              "knee_L": np.asarray(angles["knee_L"]) * 0.0 + 0.1,
              "ankle_L": np.asarray(angles["ankle_L"]) * 0.0 + 0.4}
    # Only the L channels are supplied; the R channels fall back to their XML range.
    fitted = rig.fit_joint_ranges(
        {**{k: np.asarray(v, dtype=float) for k, v in angles.items()}, **scaled},
        margin_frac=0.0)
    lo, hi = fitted[("ankle", "L")]
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4), (
        "a constant trajectory must collapse the range to that constant")
    assert fitted[("ankle", "R")][0] < 0.0, "the untouched side keeps its range"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_data_driven_geometry.py -q"`

Expected: FAIL — `AttributeError: 'Leg' object has no attribute 'fit_joint_ranges'`.

- [ ] **Step 3: Implement `fit_joint_ranges`**

In `model/leg.py`, directly after `set_limb_lengths`, add:

```python
    #: Joints whose excursion the loader can supply. The ML mocap model has no toe
    #: DOF (the MH marker is the foot segment's endpoint), so `toe_*_jnt_*` keeps
    #: its XML range; see the module docstring in tests/test_data_driven_geometry.py.
    FIT_JOINTS = ("hip", "knee", "ankle")

    def fit_joint_ranges(self, angles, margin_frac: float = 0.05):
        """
        Sets each hinge's `jnt_range` from the dataset's observed excursion, so a
        different subject/trial can be loaded without editing the XML. The window
        is `[min - m, max + m]` with `m = margin_frac * (max - min)`.

        Returns `{(joint, side): (lo, hi)}` for the joints that were fitted.
        """
        jid_of = {"hip": self.j_hip_m, "knee": self.j_knee_m, "ankle": self.j_ankle_m}
        rid_of = {"hip": self.j_hip_ref, "knee": self.j_knee_ref,
                  "ankle": self.j_ankle_ref}
        fitted = {}
        for joint, per_side in jid_of.items():
            key = f"{joint}_{{}}"  # e.g. "ankle_{}"
            for side in SIDES:
                try:
                    obs = np.asarray(angles[key.format(side)], dtype=float)
                except KeyError:
                    continue
                if obs.size == 0:
                    continue
                lo, hi = float(obs.min()), float(obs.max())
                margin = margin_frac * (hi - lo)
                lo, hi = lo - margin, hi + margin
                fitted[(joint, side)] = (lo, hi)
                self.model.jnt_range[jid_of[joint][side]] = (lo, hi)
                self.model.jnt_range[rid_of[joint][side]] = (lo, hi)
                # Keep the position servos able to command the whole new range:
                # `actuator_ctrlrange` clips the setpoint before the servo ever
                # sees it, so a fitted range outside it would be unreachable.
                for prefix in ("m", "ref"):
                    name = f"{joint}_{prefix}_ctrl_{side}"
                    aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                            name)
                    if aid >= 0:
                        self.model.actuator_ctrlrange[aid] = (lo, hi)
        mujoco.mj_setConst(self.model, self.data)
        self.calibrate_spindles()
        return fitted
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_data_driven_geometry.py -q"`

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add model/leg.py tests/test_data_driven_geometry.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "feat(model): fit joint ranges from the dataset trajectory" -- model/leg.py tests/test_data_driven_geometry.py
```

---

### Task 2: Fit the muscle length ranges over the fitted joint ranges

**Files:**
- Modify: `model/leg.py` (add `fit_muscle_lengthranges` after `fit_joint_ranges`)
- Test: `tests/test_data_driven_geometry.py` (append)

**Interfaces:**
- Consumes: Task 1's `fit_joint_ranges`.
- Produces: `Leg.fit_muscle_lengthranges(samples: int = 41) -> Dict[str, Tuple[float, float]]` — sets `model.actuator_lengthrange` for `sol_m_{L,R}` and `ta_m_{L,R}` and returns `{"sol_L": (lo, hi), ...}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data_driven_geometry.py`:

```python
def _tendon_excursion(rig, side, samples=41):
    """Tendon length min/max over the model's CURRENT joint ranges."""
    knee = np.linspace(*rig.model.jnt_range[rig.j_knee_m[side]], samples)
    ank = np.linspace(*rig.model.jnt_range[rig.j_ankle_m[side]], samples)
    sol, ta = [], []
    kadr = int(rig.model.jnt_qposadr[rig.j_knee_m[side]])
    aadr = int(rig.model.jnt_qposadr[rig.j_ankle_m[side]])
    for k in knee:
        for a in ank:
            rig.data.qpos[kadr] = float(k)
            rig.data.qpos[aadr] = float(a)
            mujoco.mj_forward(rig.model, rig.data)
            sol.append(float(rig.data.ten_length[rig.ten_sol[side]]))
            ta.append(float(rig.data.ten_length[rig.ten_ta[side]]))
    return {"sol": (min(sol), max(sol)), "ta": (min(ta), max(ta))}


def test_fit_muscle_lengthranges_matches_mujocos_own_auto_range(rig, tmp_path):
    """Independent check of the sampler.

    With `lengthrange` omitted MuJoCo computes the tendon's excursion over the
    model's joint ranges itself (2.67869 .. 20.2606 mm for sol_m_L on the unscaled
    rig). The hand-computed sweep must agree, otherwise the sampler -- not the
    authoring -- is the problem. `abs=2e-4` is about one grid cell at 201 samples.
    """
    src = open(XML).read()
    for attr in ('lengthrange="0.0027 0.0195"', 'lengthrange="0.0005 0.0248"'):
        src = src.replace(attr, "")
    tmp = tmp_path / "_auto.xml"
    tmp.write_text(src)
    stock = mujoco.MjModel.from_xml_path(str(tmp))
    fitted = rig.fit_muscle_lengthranges(samples=201)
    for side in ("L", "R"):
        auto = stock.actuator_lengthrange[rig.act_sol_m[side]]
        got = fitted[f"sol_{side}"]
        assert got[0] == pytest.approx(float(auto[0]), abs=2e-4), (
            "sampler disagrees with MuJoCo's own auto-computed range")


def test_fit_muscle_lengthranges_contains_the_observed_excursion(rig, angles):
    _scaled_rig(rig, angles)
    rig.fit_joint_ranges(angles, margin_frac=0.05)
    fitted = rig.fit_muscle_lengthranges(samples=41)
    for side in ("L", "R"):
        obs = _trajectory_excursion(rig, angles, side)
        lo, hi = fitted[f"sol_{side}"]
        assert lo <= obs[0] and obs[1] <= hi, (
            f"sol_{side} window {lo * 1e3:.3f}..{hi * 1e3:.3f} mm excludes the "
            f"trajectory {obs[0] * 1e3:.3f}..{obs[1] * 1e3:.3f} mm")
        assert rig.model.actuator_lengthrange[rig.act_sol_m[side]] == pytest.approx(
            fitted[f"sol_{side}"])


def _trajectory_excursion(rig, angles, side):
    import mujoco
    knee, ank = [], []
    kadr = int(rig.model.jnt_qposadr[rig.j_knee_m[side]])
    aadr = int(rig.model.jnt_qposadr[rig.j_ankle_m[side]])
    for i in range(len(angles[f"knee_{side}"])):
        rig.data.qpos[kadr] = float(angles[f"knee_{side}"][i])
        rig.data.qpos[aadr] = float(angles[f"ankle_{side}"][i])
        mujoco.mj_forward(rig.model, rig.data)
        knee.append(float(rig.data.ten_length[rig.ten_sol[side]]))
    return min(knee), max(knee)


def test_the_window_actually_changes_the_muscle_force(rig):
    """A written `actuator_lengthrange` that MuJoCo ignores would pass every
    containment assertion above, so observe the force itself."""
    import mujoco
    aid = rig.act_sol_m["L"]
    aadr = int(rig.model.jnt_qposadr[rig.j_ankle_m["L"]])
    rig.data.qpos[aadr] = 1.2
    rig.data.ctrl[aid] = 1.0
    rig.fit_muscle_lengthranges(samples=41)
    mujoco.mj_forward(rig.model, rig.data)
    narrow = float(rig.data.actuator_force[aid])
    rig.model.actuator_lengthrange[aid] = (0.0, 0.005)  # 5 mm span, ankle at ~14 mm
    mujoco.mj_setConst(rig.model, rig.data)
    mujoco.mj_forward(rig.model, rig.data)
    wide = float(rig.data.actuator_force[aid])
    assert abs(wide - narrow) > 1e-4, "actuator_lengthrange had no effect on force"
```

Add `import mujoco` to the module header (after `import numpy as np`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_data_driven_geometry.py -q"`

Expected: FAIL — `AttributeError: 'Leg' object has no attribute 'fit_muscle_lengthranges'`.

- [ ] **Step 3: Implement `fit_muscle_lengthranges`**

In `model/leg.py`, after `fit_joint_ranges`:

```python
    #: (muscle, tendon attr, actuator attr) — the actuated muscle-tendon units.
    MUSCLES = (("sol", "ten_sol", "act_sol_m"), ("ta", "ten_ta", "act_ta_m"))

    def fit_muscle_lengthranges(self, samples: int = 41):
        """
        Sets each muscle actuator's `lengthrange` from the excursion of its tendon
        over the model's *current* joint ranges (which `fit_joint_ranges` derives
        from the dataset). Replaces the hand-authored XML window, which was
        computed for the unscaled rig and clipped the SOL tendon for 45 % of the
        trial once `set_limb_lengths` shortened the bones.

        The SOL/TA tendons span the knee and the ankle, so only those two hinges
        are swept; the hip rotates the whole sub-chain rigidly.

        Returns `{(muscle, side): (lo, hi)}`, i.e. `{"sol_L": (...), "ta_R": (...)}`.
        """
        fitted = {}
        for side in SIDES:
            knee = np.linspace(*self.model.jnt_range[self.j_knee_m[side]], samples)
            ank = np.linspace(*self.model.jnt_range[self.j_ankle_m[side]], samples)
            kadr = int(self.model.jnt_qposadr[self.j_knee_m[side]])
            aadr = int(self.model.jnt_qposadr[self.j_ankle_m[side]])
            keep = (float(self.data.qpos[kadr]), float(self.data.qpos[aadr]))
            lows = {m: np.inf for m, _, _ in self.MUSCLES}
            highs = {m: -np.inf for m, _, _ in self.MUSCLES}
            for k in knee:
                for a in ank:
                    self.data.qpos[kadr] = float(k)
                    self.data.qpos[aadr] = float(a)
                    mujoco.mj_fwdPosition(self.model, self.data)
                    for name, ten_attr, _ in self.MUSCLES:
                        ln = float(self.data.ten_length[getattr(self, ten_attr)[side]])
                        lows[name] = min(lows[name], ln)
                        highs[name] = max(highs[name], ln)
            self.data.qpos[kadr], self.data.qpos[aadr] = keep
            for name, _, act_attr in self.MUSCLES:
                lo, hi = lows[name], highs[name]
                self.model.actuator_lengthrange[
                    getattr(self, act_attr)[side]] = (lo, hi)
                fitted[f"{name}_{side}"] = (lo, hi)
        mujoco.mj_setConst(self.model, self.data)
        self.calibrate_spindles()
        return fitted
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_data_driven_geometry.py -q"`

Expected: 6 passed. If `test_fit_muscle_lengthranges_matches_mujocos_own_auto_range` misses by more than `abs=2e-4`, raise `samples` (41 → 201) before loosening the tolerance — the tolerance is a check on the sampler, not a fit parameter.

- [ ] **Step 5: Commit**

```bash
git add model/leg.py tests/test_data_driven_geometry.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "feat(model): fit muscle length ranges from the tendon excursion" -- model/leg.py tests/test_data_driven_geometry.py
```

---

### Task 3: Remove the authored windows and wire the fits into the build

**Files:**
- Modify: `dual_rat_hindlimb.xml` (delete the four authored `lengthrange` attributes; comment the muscle block)
- Modify: `personalize.py` (call the two fits after `set_limb_lengths`, ~line 105)
- Test: `tests/test_data_driven_geometry.py` (append)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: a `build_personalizer` rig whose joint ranges and muscle `lengthrange` are entirely dataset-derived.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_data_driven_geometry.py`:

```python
def test_built_rig_uses_dataset_derived_muscle_windows(angles):
    """`build_personalizer` must leave no XML-authored window in place.

    `build_personalizer` calls `nest.ResetKernel()` itself, so this test must not
    request the `nest_kernel` fixture. It returns `(Personalizer, MocapLoader)`.
    """
    import personalize

    personalizer, _loader = personalize.build_personalizer(
        reference_path=REF, max_steps=1, belt_mode="dataset")
    rig = personalizer.rig
    stock = mujoco.MjModel.from_xml_path(XML)
    for side in ("L", "R"):
        aid = rig.act_sol_m[side]
        assert not np.allclose(rig.model.actuator_lengthrange[aid],
                               stock.actuator_lengthrange[aid]), (
            "the authored XML window survived the build")
        obs = _trajectory_excursion(rig, angles, side)
        lo, hi = rig.model.actuator_lengthrange[aid]
        assert lo <= obs[0] and obs[1] <= hi
```

- [ ] **Step 2: Run it to verify it fails**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_data_driven_geometry.py -q"`

Expected: FAIL — the authored window matches (`the authored XML window survived the build`).

- [ ] **Step 3: Delete the authored attributes**

In `dual_rat_hindlimb.xml`, remove `lengthrange="0.0027 0.0195"` and `lengthrange="0.0005 0.0248"` from the four `<muscle>` actuators (lines ~200–225), and replace the surrounding comment with:

```xml
    <!-- Muscle-tendon units. `lengthrange` is deliberately NOT authored here:
         model.leg.Leg.fit_muscle_lengthranges computes it from the tendon
         excursion over the dataset-derived joint ranges at build time, so a new
         subject/trial needs no XML edit. `force`, `scale`, `timeconst` and the
         FLV curve breakpoints are physiological constants (the dataset carries
         kinematics only), so they are declared here explicitly. -->
```

- [ ] **Step 4: Call the fits from the build**

In `personalize.py`, after the `set_limb_lengths` loop (line ~105):

```python
    # Every remaining trajectory-dependent parameter comes from the dataset, so a
    # different subject/trial loads without editing the XML: the hinge limits from
    # the observed joint excursions, and the muscle length windows from the tendon
    # excursion those limits permit.
    rig.fit_joint_ranges(angles)
    rig.fit_muscle_lengthranges()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_data_driven_geometry.py -q"`

Expected: 7 passed.

- [ ] **Step 6: Run the whole suite and commit**

Run: the Global Constraints test command. Expected: all pass; report by name any failure you did not cause.

```bash
git add dual_rat_hindlimb.xml personalize.py tests/test_data_driven_geometry.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "feat(model): derive muscle length windows and joint limits from the dataset" -- dual_rat_hindlimb.xml personalize.py tests/test_data_driven_geometry.py
```

---

### Task 4: Re-run the validation and confirm the clipping is gone

**Files:**
- Modify: `test_results/t8_tendon_probe.py` (accept the new behaviour; it is a probe, not a test)

- [ ] **Step 1: Re-run the probe**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python test_results/t8_tendon_probe.py"`

Expected after the fix, on the `dataset` rows: `clipped 0/...` for both SOL and TA, and the authored window replaced by the fitted one. Before the fix the SOL row reads `clipped 2725/5995 (45 %)`.

- [ ] **Step 2: Re-run the fast A/B and record the loss**

Run `run_validation.py --fast` twice (`--mode legacy` for the A side) saving under `test_results/geo_fast`, then:

```
/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python test_results/compare_ab.py test_results/geo_fast
```

Record the last-10 loss pair. It will differ from `0.8489750236571337`; whether it improves is a T1/T2 (drive-balance plan) question, not this plan's — this plan only removes an unphysical force–length clipping. State that explicitly in the commit message rather than claiming an improvement.

- [ ] **Step 3: Record the result (no commit)**

`test_results/` is gitignored (`.gitignore:29`), so the probe stays a local
diagnostic — do not `git add` it. Record the before/after `clipped` counts and the
A/B loss pair in the plan's execution notes (or the next commit message), not in a
file under `test_results/`.
