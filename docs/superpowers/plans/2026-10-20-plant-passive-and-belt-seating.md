# Plant Passive Properties and Belt Seating Implementation Plan (T10, T13)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the simulated ankle a bio-plausible passive resistance, and make `--belt model` seat the treadmill from the *controlled* foot instead of the non-colliding target foot.

**Architecture:** Two independent, small corrections with no shared state. T10 is a pair of attributes on the ankle joints in `dual_rat_hindlimb.xml`. T13 generalises an existing FK sweep in `model/leg.py` from the target limb to either limb and points `personalize.build_personalizer`'s `--belt model` branch at the controlled one.

**Tech Stack:** Python 3.12, MuJoCo 3.14, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-persona-variantB-cpg-mvp-design.md`

## Global Constraints

- Do not modify anything under `persona/`, `tinycpg/`, `memcpg/`, or legacy `persona.py`.
- Commit with explicit pathspecs only (`git commit -- <paths>`); never `git add -A`/`-u`.
- Git identity is passed per-command: `git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit ...`.
- WSL test command: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest -q"`.
- `TREADMILL_PENETRATION_M = 0.0005` and `CONTACT_Z_EPS = 0.003` come from `model/model_cfg.py`; do not introduce new magic heights.

## Measured Baseline

- Ankles (`ankle_m_L` line 58, `ankle_m_R` line 95) currently carry `damping="0.001" frictionloss="0.0005"` and **zero** stiffness — no joint anywhere in the model has nonzero stiffness (`test_results/mujoco_hooks_probe.py`). `jnt_stiffness` is writable at runtime and survives `mj_setConst`.
- Peak active ankle torque measured in the trial: **SOL +0.0663 N·m**, **TA −0.0285 N·m**. Any passive term must stay well below that or it, not the CPG, will drive the joint.
- `--belt model` calls `rig.target_foot_lowest_percentile(...)`, which sweeps the **target** limb's joints (`j_*_ref`) and reads `ref_foot_geoms`. The target foot is a non-colliding kinematic mirror (`contype=0 conaffinity=0`), and over the cycle the controlled foot's lowest surface sits **+0.10 … +1.10 mm above** the target foot's, so the resulting belt leaves the controlled foot in permanent air — the same 0.1–1.1 mm offset already fixed for the default `--belt dataset` path in `d588cea`.

## Review Focus

- **Spring rest angle is `qpos0`, i.e. 0 rad.** MuJoCo's joint stiffness pulls q toward `qpos0`. The ankle's neutral is roughly mid-range plantarflexion (the trial's ankle averages ≈+0.9 rad), not 0, so this spring adds a constant dorsiflexion-resisting bias rather than a symmetric neutral zone. It *assists* SOL (the plantarflexor), which is directionally right but is a bias, not an unbiased stiffness. If the reviewer wants a symmetric neutral zone, that is `<joint springref="0.9">` or a tendon spring — a modelling decision, not a bug fix.
- **Stiffness value is asserted in a test.** `0.006 N·m/rad` is chosen so the passive torque at the ankle range extreme (≈1.7 rad) is ≈0.010 N·m, about 15 % of the measured peak active torque (0.0663 N·m). The test pins that ratio, so a later change to 0.06 (which would dominate the plant) fails loudly.
- **T13 changes only the `model` mode.** The default `dataset` mode already levels on the controlled foot via `_level_treadmill` (`d588cea`); this plan does not touch it.
- **`sweep_foot_lowest` is a pure refactor of existing behaviour.** `target_foot_lowest_percentile` must return bit-identical values after the extraction; Task 2 Step 3 verifies that on the real trajectory before any prefix is switched.

---

### Task 1: Add bio-plausible passive stiffness and damping to the ankles (T10)

**Files:**
- Modify: `dual_rat_hindlimb.xml` (ankle joints at lines 58 and 95)
- Test: `tests/test_plant_passive.py` (create)

**Interfaces:**
- Produces: `ankle_m_L` / `ankle_m_R` carry `stiffness` and a raised `damping`; nothing in Python changes.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plant_passive.py`:

```python
"""The ankle must have a bio-plausible passive resistance.

It had damping 0.001, frictionloss 0.0005 and no stiffness at all, so with no
muscle activation the joint was free: nothing but the (unclipped, and therefore
nearly constant) muscle drive could move or hold it. Measured peak active ankle
torque in the trial is 0.0663 N*m (SOL), so the passive term is pinned at a small
fraction of that rather than being free to take over the plant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import pytest

from model.leg import Leg

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")
DT_MS = 10.0

#: Measured in test_results/muscle_probe.py over the trial's trajectory.
PEAK_ACTIVE_TORQUE_NM = 0.0663
ANKLE_STIFFNESS_NM_PER_RAD = 0.006
ANKLE_DAMPING_NM_S_PER_RAD = 0.002


@pytest.fixture()
def rig():
    return Leg(mujoco_cfg=XML, dt_ms=DT_MS)


def _passive_torque(rig, side, q_rad):
    jid = rig.j_ankle_m[side]
    adr = int(rig.model.jnt_qposadr[jid])
    rig.data.qpos[:] = 0.0
    rig.data.qpos[adr] = float(q_rad)
    rig.data.qvel[:] = 0.0
    rig.data.ctrl[:] = 0.0
    mujoco.mj_forward(rig.model, rig.data)
    return float(rig.data.qfrc_passive[rig.model.jnt_dofadr[jid]])


def test_ankle_has_stiffness(rig):
    for side in ("L", "R"):
        assert rig.model.jnt_stiffness[rig.j_ankle_m[side]] == pytest.approx(
            ANKLE_STIFFNESS_NM_PER_RAD)


def test_ankle_damping_is_raised(rig):
    for side in ("L", "R"):
        dof = rig.model.jnt_dofadr[rig.j_ankle_m[side]]
        assert rig.model.dof_damping[dof] == pytest.approx(ANKLE_DAMPING_NM_S_PER_RAD)


def test_passive_torque_is_zero_at_the_spring_rest(rig):
    assert _passive_torque(rig, "L", 0.0) == pytest.approx(0.0, abs=1e-6)


def test_passive_torque_resists_dorsiflexion(rig):
    """Positive ankle angle = dorsiflexion; the spring must push back."""
    for q in (0.5, 1.0, 1.7):
        assert _passive_torque(rig, "L", q) < 0.0, f"no restoring torque at q={q}"


def test_passive_torque_stays_below_the_active_torque(rig):
    """The plant, not the spring, must decide the ankle."""
    for q in (-0.9, -0.5, 0.5, 1.0, 1.7, 2.5):
        mag = abs(_passive_torque(rig, "L", q))
        assert mag < 0.25 * PEAK_ACTIVE_TORQUE_NM, (
            f"passive torque {mag:.4f} N*m at q={q} is more than a quarter of the "
            f"measured peak active torque {PEAK_ACTIVE_TORQUE_NM} N*m")


def test_target_ankle_stays_kinematic(rig):
    """The target limb is a kinematic mirror; adding stiffness there would fight
    its servo and change the reference trajectory."""
    for side in ("L", "R"):
        assert rig.model.jnt_stiffness[rig.j_ankle_ref[side]] == pytest.approx(0.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_plant_passive.py -q"`

Expected: 4 failed (`test_ankle_has_stiffness`, `test_ankle_damping_is_raised`, `test_passive_torque_resists_dorsiflexion`, `test_passive_torque_stays_below_the_active_torque`). `test_passive_torque_is_zero_at_the_spring_rest` and `test_target_ankle_stays_kinematic` pass already.

- [ ] **Step 3: Add the attributes**

In `dual_rat_hindlimb.xml`, both ankle joints (lines 58 and 95) change from:

```xml
<joint name="ankle_m_L" type="hinge" axis="0 1 0" range="-1.0 2.6" damping="0.001" frictionloss="0.0005"/>
```

to:

```xml
<!-- Passive ankle resistance. The calf muscle-tendon unit is not the only thing
     resisting dorsiflexion: the joint capsule and the passive parallel
     elasticity do too, and with none of it the ankle was free-floating.
     stiffness 0.006 N*m/rad makes the passive torque at the range extreme
     (~1.7 rad) about 15 % of the trial's peak active ankle torque (0.0663 N*m),
     so it shapes the plant without governing it. Not applied to the target ankle,
     which is a servo-driven kinematic mirror. -->
<joint name="ankle_m_L" type="hinge" axis="0 1 0" range="-1.0 2.6" stiffness="0.006" damping="0.002" frictionloss="0.0005"/>
```

Do the same for `ankle_m_R` (line 95). Leave the knee/hip/toe joints and all `*_ref` joints unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_plant_passive.py -q"`

Expected: 6 passed. If `test_passive_torque_stays_below_the_active_torque` fails at `q=-0.9`, the frictionless asymmetries dominate — check that `frictionloss` did not change (0.0005) before adjusting `stiffness`.

- [ ] **Step 5: Run the whole suite and commit**

Run: the Global Constraints test command. Expected: all pass. The ankle's dynamics change, so any test that simulates an unactuated ankle may shift; report by name rather than loosening it.

```bash
git add dual_rat_hindlimb.xml tests/test_plant_passive.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "feat(model): give the ankle a bio-plausible passive resistance" -- dual_rat_hindlimb.xml tests/test_plant_passive.py
```

---

### Task 2: Seat the `--belt model` plane from the controlled foot (T13)

**Files:**
- Modify: `model/leg.py` (`target_foot_lowest_percentile`, ~line 305; add `sweep_foot_lowest` and `_foot_geoms`)
- Modify: `personalize.py` (the `belt_mode == "model"` branch, ~line 127)
- Test: `tests/test_belt_seating.py` (create)

**Interfaces:**
- Produces: `Leg.sweep_foot_lowest(hip_x, hip_z, hip, knee, ankle, prefix="ref") -> np.ndarray` — per-frame lowest surface height (m) of the `prefix` limb's foot over the trial trajectory, forward-kinematics only. `hip_x`/`hip_z`/`hip`/`knee`/`ankle` map side → per-frame array.
- Produces: `Leg._foot_geoms(prefix) -> dict` — `"m"` → `model_foot_geoms`, `"ref"` → `ref_foot_geoms`.
- Produces: `Leg.target_foot_lowest_percentile(..., prefix="ref") -> float` — unchanged defaults, now prefix-aware and delegating to `sweep_foot_lowest`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_belt_seating.py`:

```python
"""`--belt model` must seat the plane from the foot that can actually touch.

It seated it from the target foot, which is a non-colliding kinematic mirror and
whose lowest surface sits 0.1-1.1 mm below the controlled foot's over the cycle.
The controlled foot is the only one with collision geoms and the only one whose
CUT contact feeds the CPG, so the plane it produced left that foot in permanent
air: contact never happens and CUT stays at its 120 Hz ceiling for the run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import pytest

from loader.mocap import MocapLoader
from model.leg import Leg
from model.model_cfg import TREADMILL_PENETRATION_M

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML = os.path.join(HERE, "dual_rat_hindlimb.xml")
REF = os.path.join(HERE, "data/zenodo/DatasetMotionCaputure/RatA",
                   "RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv")
DT_MS = 10.0


@pytest.fixture(scope="module")
def angles():
    return MocapLoader(REF, dt_ms=DT_MS).resample()


@pytest.fixture()
def rig(angles):
    leg = Leg(mujoco_cfg=XML, dt_ms=DT_MS)
    for s in ("L", "R"):
        thigh = float(np.mean(angles[f"seg_thigh_{s}"])) / 1000.0
        shank = float(np.mean(angles[f"seg_shank_{s}"])) / 1000.0
        foot = float(np.mean(angles[f"seg_foot_{s}"])) / 1000.0
        leg.set_limb_lengths("m", s, thigh, shank, foot)
        leg.set_limb_lengths("ref", s, thigh, shank, foot)
    return leg


def _traj(angles):
    return dict(
        hip_x={s: (angles[f"hip_x_{s}"] - np.mean(angles[f"hip_x_{s}"])) / 1000.0
               for s in ("L", "R")},
        hip_z={s: angles[f"hip_z_{s}"] / 1000.0 for s in ("L", "R")},
        hip={s: np.asarray(angles[f"hip_{s}"], dtype=float) for s in ("L", "R")},
        knee={s: np.asarray(angles[f"knee_{s}"], dtype=float) for s in ("L", "R")},
        ankle={s: np.asarray(angles[f"ankle_{s}"], dtype=float) for s in ("L", "R")},
    )


def test_target_foot_percentile_is_unchanged_by_the_refactor(rig, angles):
    """The extraction must not move the existing number."""
    t = _traj(angles)
    before = rig.target_foot_lowest_percentile(
        t["hip_x"], t["hip_z"], t["hip"], t["knee"], t["ankle"], percentile=5.0)
    assert before == pytest.approx(np.percentile(
        rig.sweep_foot_lowest(t["hip_x"], t["hip_z"], t["hip"], t["knee"],
                              t["ankle"], prefix="ref"), 5.0))


def test_controlled_foot_sits_above_the_target_foot(rig, angles):
    """Documents why the target foot was the wrong reference."""
    t = _traj(angles)
    lo_m = float(rig.sweep_foot_lowest(t["hip_x"], t["hip_z"], t["hip"],
                                       t["knee"], t["ankle"], prefix="m").min())
    lo_ref = float(rig.sweep_foot_lowest(t["hip_x"], t["hip_z"], t["hip"],
                                         t["knee"], t["ankle"], prefix="ref").min())
    assert lo_m > lo_ref, "expected the controlled foot to ride above the target"


def test_target_seated_belt_leaves_the_controlled_foot_in_the_air(rig, angles):
    """The bug: the old placement produces no contact at all."""
    t = _traj(angles)
    old = rig.target_foot_lowest_percentile(
        t["hip_x"], t["hip_z"], t["hip"], t["knee"], t["ankle"], percentile=5.0)
    rig.set_treadmill_height(old + TREADMILL_PENETRATION_M)
    lo_m = float(rig.sweep_foot_lowest(t["hip_x"], t["hip_z"], t["hip"],
                                       t["knee"], t["ankle"], prefix="m").min())
    assert lo_m > rig.belt_z, "expected the controlled foot to stay above the belt"


def test_controlled_seated_belt_makes_contact(rig, angles):
    """The fix: the plane reaches the foot that has collision geoms."""
    t = _traj(angles)
    new = rig.target_foot_lowest_percentile(
        t["hip_x"], t["hip_z"], t["hip"], t["knee"], t["ankle"], percentile=5.0,
        prefix="m")
    rig.set_treadmill_height(new + TREADMILL_PENETRATION_M)
    lo_m = float(rig.sweep_foot_lowest(t["hip_x"], t["hip_z"], t["hip"],
                                       t["knee"], t["ankle"], prefix="m").min())
    assert lo_m <= rig.belt_z, (
        f"controlled foot bottoms out {1e3 * (lo_m - rig.belt_z):.3f} mm above the "
        f"belt; no contact is possible")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_belt_seating.py -q"`

Expected: FAIL — `AttributeError: 'Leg' object has no attribute 'sweep_foot_lowest'`.

- [ ] **Step 3: Extract the sweep and add the prefix**

In `model/leg.py`, add `_foot_geoms` and `sweep_foot_lowest`, and reduce
`target_foot_lowest_percentile` to a delegation. `sweep_foot_lowest` is the
existing loop body verbatim, with `self.j_hip_ref[s]`/`j_knee_ref`/`j_ankle_ref`
replaced by `getattr(self, f"j_hip_{prefix}")[s]` etc., and
`self.ref_foot_geoms[s]` by `self._foot_geoms(prefix)[s]`:

```python
    def _foot_geoms(self, prefix: str) -> dict:
        """Foot/toe geoms of the controlled (`"m"`) or target (`"ref"`) limb."""
        return self.model_foot_geoms if prefix == "m" else self.ref_foot_geoms

    def sweep_foot_lowest(self, hip_x, hip_z, hip, knee, ankle,
                          prefix: str = "ref") -> np.ndarray:
        """
        Per-frame lowest surface height (m) of the `prefix` limb's foot while the
        limb is swept through the trial's hip/joint trajectory (forward kinematics
        only, no dynamics and no contact). `hip_x`/`hip_z`/`hip`/`knee`/`ankle` map
        side -> per-frame array.
        """
        hip_j = getattr(self, f"j_hip_{prefix}")
        knee_j = getattr(self, f"j_knee_{prefix}")
        ankle_j = getattr(self, f"j_ankle_{prefix}")
        geoms = self._foot_geoms(prefix)
        lows = []
        for s in SIDES:
            hip_adr = self.model.jnt_qposadr[hip_j[s]]
            knee_adr = self.model.jnt_qposadr[knee_j[s]]
            ankle_adr = self.model.jnt_qposadr[ankle_j[s]]
            for i in range(len(np.asarray(ankle[s]))):
                self.data.qpos[hip_adr] = float(hip[s][i])
                self.data.qpos[knee_adr] = float(knee[s][i])
                self.data.qpos[ankle_adr] = float(ankle[s][i])
                self.set_target_hip_position(s, float(hip_x[s][i]), float(hip_z[s][i]))
                mujoco.mj_kinematics(self.model, self.data)
                lows.append(self._foot_lowest_z(geoms[s]))
        return np.asarray(lows, dtype=float)

    def target_foot_lowest_percentile(self, hip_x, hip_z, hip, knee, ankle,
                                      percentile: float = 5.0,
                                      prefix: str = "ref") -> float:
        """
        The given percentile of `sweep_foot_lowest(...)`: the model-derived
        alternative to placing the belt at the dataset's ground.

        Pass `prefix="m"` to seat the belt from the CONTROLLED foot. That foot is
        the only one with collision geoms and the only one whose CUT contact feeds
        the CPG, and it rides 0.1-1.1 mm above the target foot over the cycle, so
        the target foot is the wrong reference for a belt that must be touched.
        """
        lows = self.sweep_foot_lowest(hip_x, hip_z, hip, knee, ankle, prefix=prefix)
        if lows.size == 0:
            return float(self.model.geom_pos[self.treadmill_gid, 2])
        return float(np.percentile(lows, percentile))
```

- [ ] **Step 4: Run the tests to verify the refactor is clean**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_belt_seating.py -q"`

Expected: 3 passed, 1 failed — only `test_controlled_seated_belt_makes_contact` still fails, because `prefix="m"` is not yet passed by any caller. Confirm the other three (including the bit-identical refactor check) pass before touching `personalize.py`.

- [ ] **Step 5: Point the build at the controlled foot**

In `personalize.py`, in the `belt_mode == "model"` branch, add the prefix and correct the comment:

```python
    # The belt is one plane shared by both limbs. Two placements are supported:
    #   "dataset" — the dataset's ground (min foot-marker height over the trial);
    #   "model"   — the model-derived lowest surface of the CONTROLLED foot
    #               (5th percentile over the trajectory). The controlled foot is
    #               the one with collision geoms and the one whose CUT contact
    #               feeds the CPG; the target foot is a non-colliding mirror that
    #               sits below it, so levelling on the target leaves the controlled
    #               foot permanently in the air.
    if belt_mode == "model":
        belt_z = rig.target_foot_lowest_percentile(
            hip_x={s: hip_x[s] for s in ("L", "R")},
            hip_z={s: hip_z[s] for s in ("L", "R")},
            hip={s: angles[f"hip_{s}"] for s in ("L", "R")},
            knee={s: angles[f"knee_{s}"] for s in ("L", "R")},
            ankle={s: angles[f"ankle_{s}"] for s in ("L", "R")},
            percentile=5.0, prefix="m")
```

- [ ] **Step 6: Run the tests, then the whole suite, and commit**

Run: `wsl -e bash -lc "cd /mnt/d/CPGvalidation && MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_belt_seating.py -q"` → 4 passed.

Run: the Global Constraints test command. Expected: all pass; report by name any failure you did not cause.

```bash
git add model/leg.py personalize.py tests/test_belt_seating.py
git -c user.name=TOKUGAWA -c user.email=metamaskus@gmail.com commit -m "fix(model): seat the model-derived belt on the controlled foot" -- model/leg.py personalize.py tests/test_belt_seating.py
```

- [ ] **Step 7: Confirm in an actual run**

Run `run_validation.py --fast --belt model` and check the logged CUT rate is no longer pinned at its 120 Hz ceiling (which is the signature of permanent contact) or at 0 (the signature of permanent air):

```
/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python test_results/afferent_summary.py test_results/belt_model_fixed/run.h5
```

Record the CUT mean/min/max for both legs in the commit or the run notes. A mean near the 120 Hz ceiling means the foot is now in constant contact — better than air, but then the *plane height* needs `TREADMILL_PENETRATION_M` reduced, which is a separate decision to raise with the reviewer.
