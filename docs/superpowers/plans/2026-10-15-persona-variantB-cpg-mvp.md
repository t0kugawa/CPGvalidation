# PERSONA perturbation–advantage natural gradient + CPG bio-init + render fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PERSONA a true natural-gradient strategy whose impact metric is the covariance of each parameter's exploration perturbation with the reward advantage, start the CPG from correctly-named `cpg_2legs_fast` bio-plausible parameters (topology included), and make the render muscle geoms visible.

**Architecture:** `persona/strategy.py` estimates the perturbation covariance `F = Cov(Δθ) + λI` and the impact `G = Cov(Δθ, A)` from a rolling window of `(Δθ, A)`; the step is `Δμ = η·F−1·(G·1)` (natural-gradient ascent on the reward advantage = descent on the tracked error). `t2xcpg` is rewired to the `cpg_2legs_fast` topology and initialised from its constants. The XML muscle capsules are thinned and offset.

**Tech Stack:** Python 3.12 (WSL venv), NEST 3.10, MuJoCo 3.14, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-15-persona-variantB-cpg-mvp-design.md`

## Global Constraints

- Runtime: WSL venv `/home/t0kgw/.virtualenvs/KinematicsCPG`; repo `/mnt/d/CPGvalidation`.
- `OPENBLAS_NUM_THREADS=1` (and OMP/MKL/NUMEXPR = 1) must be set before numpy import (a 322×322 solve is 1305 ms multi-thread vs 1.7 ms single).
- Do NOT modify `tinycpg/`, `memcpg/`, or `persona.py`.
- Keep `TODO(advantage)` / `TODO(fisher)` comment hooks.
- Optimised set: every **static** connection weight, every **rate**, Izhikevich
  `a,b,c,d`, and the **STDP hyper-parameters** of `CUT→RGE` / `Ia→RG`. Exclude
  the STDP weight itself (owned by plasticity), connection `p`, and σ_param.
- Reward/output vector `y = [ankle_model_L, ankle_model_R]`.
- Ib pathway stays. Reference values from `cpg_2legs_fast` per spec §5.4; Ib/Renshaw keep project values.

## Decisions recorded 2026-10-18 (supersede earlier text in this plan)

- **A1 / D10** Marker→angle = point-based rigid-bone reconstruction (approach A):
  absolute segment orientations from each marker pair (`GT→KN`, `KN→EA`,
  `EA→MH`), **unwrapped** over time, no rest-pose zero alignment.
- **A2 / A5** `CUT→RGE` is **STDP**; the earlier "static" wording is superseded.
- **A3 / B2** Connection `p` is a structural constant, never registered or sampled.
- **A4** Add the `cpg_2legs_fast` base drive (`BASE_DRIVE_W/HZ/P`) wired through
  the PERSONA registry. **Rate value to confirm** — `cpg_2legs_fast` has
  `BASE_DRIVE_HZ = 2.0`, the request said 20 Hz.
- **A6** `w_scale = 1.0` everywhere (unscaled §5.4 reference values).
- **A7** `Ia→RG` = 12.0 / p 0.40 (reference `IA2RG_W`/`IA2RG_P`).
- **A8** Noise features are **scored** like ordinary parameters.
- **A9** Per-component, per-phase moving-average baselines (left stance vs left
  swing, independently of the right leg); output and target share one smoothing
  filter.
- **B4** PERSONA optimizes a plastic projection's STDP hyper-parameters only,
  never its weight.
- **B6** Acceptance key renamed `duty_factor_in_distribution`.
- **B7** `stride_length` removed, `TODO(stride_length)` left in `loader/stats.py`.
- **A10** `run_validation` reuses `Personalizer.run` through a step hook.
- **Bug** SynapseCollections are captured only once the network is complete
  (`Registry.finalize_connections`); NEST recycles ConnectionObject handles while
  `nest.Connect` is still being called.

## Decisions recorded 2026-10-18 (b): target-limb + learning-rule fixes

- **Target limb = kinematic mirror, no collision.** The target limb's foot/toe
  geoms carry the `<default contype=0 conaffinity=0>`, so they never contact; the
  target phase is a **geometric** classification (`foot_z_ref <= belt_z + CONTACT_Z_EPS`),
  not a contact force. `Leg._contact_force_ref` / `_contact_force_ref` removed.
  `contact_ref_*` is now a 0/1 geometric stance indicator (validation only).
  Confirmed `phase_ref`/`contact_ref` are validation-only: the loop passes the
  **model** `phase_L/R` into the advantage (`personalization.py`).
- **Belt = the dataset's ground, one plane for both limbs.** `belt_z = min over the
  run window of the foot-marker (MH) height` (dataset frame, /1000) + `TREADMILL_PENETRATION_M`.
  It is read from the dataset, not derived from the target limb's rig geometry (that
  reconstruction carries a ~2–3 mm fixed-length bias).
- **Output limb is built at the target's size (per side).** For each side, the
  loader's sagittal (x-z) mean segment lengths (thigh/shank/foot) are applied to both
  the controlled (`m`) and target (`ref`) limbs with `Leg.set_limb_lengths`, which
  rescales body offsets, geom pos/size and site pos while **preserving the rest
  directions**. L/R lengths are kept exactly as recorded (no symmetrisation).
- **Unwrap + physiological clamp.** `markers_to_angles` unwraps each segment
  direction over time (removes the 2π branch cuts) and clamps each joint to
  `JOINT_RANGES`; the rig's hip/knee/ankle ranges (both limbs, both sides) are
  widened to match, so every target value is reachable by the servo.
- **B3 fixed.** `Leg.update_plastic_weights` records the sampled STDP delta with
  `NoiseRecorder.record_direct`; previously it fed a *delta* into `observe_weight`,
  which differenced it a second time.
- **B5 fixed.** `PersonaParam.update_controlling_variance` updates `log(sigma_ctrl)`
  (`grad = ((a-mu)/sigma)^2 - 1`) with a per-step cap, removing the `1/sigma^3`
  blow-up (observed 1e-3 -> 0.65 in 200 steps).
- **B7 superseded.** `stride_length_L/R` is back, computed from the controlled
  foot's horizontal excursion over the detected stance phases (contact cycles),
  both compared against the Zenodo distributions. Foot x is logged (`foot_x_m_L/R`).
- **Pose stage capped.** `Personalizer.run` takes `pose_max_steps` so the loop
  always proceeds to tracking even when contact-driven CPG activity keeps the ankle
  error above `eps_pose` (`run_validation --pose-max-s`, default 1.0 s).
- **Wall-clock budget.** `Personalizer.run` takes `wall_clock_s`
  (`run_validation --budget-s`) so a run fits a server time limit.
- **MVP scope: RatA only.** `DEFAULT_REFERENCE` is a `RatA` trial; the per-side
  recorded lengths are used as-is.

## Decisions recorded 2026-10-18 (c): 2-minute simulation + evaluation outputs

- **"2 minutes" = simulated time.** `run_validation.py --duration-s` defaults to
  **120 s** and the target trajectory now **loops** (`MocapTargetDriver(loop=True)`,
  period = trial length), so a run longer than the 60 s reference trial keeps
  walking instead of freezing on the last frame. The wall-clock `--budget-s`
  remains available but is not the server contract.
- **Run outputs.** `run.h5` holds `log/` (every `RunLog` channel: both-limb CUT,
  Ia length/velocity/rate, Ib activation/force/rate, hip/knee/ankle model+target,
  muscle drives, phases, contacts, foot x/z, loss, advantage per leg),
  `stdp/` (per-step mean weight change and update count from the **weight
  recorders**, the sampled live weight distribution, and the final weight
  vectors) and `agent/` (means, σ_ctrl, σ_param, influence, advantage/error
  vectors, the final Fisher metric matrix and feature scales).
- **Figures.** tracking, model_vs_target, learning (loss + per-leg advantage),
  sensory (CUT/Ia/Ib + muscle drives), stdp, param_shift, fisher, influence,
  metric.
- **Weight recorders are drained every step.** `Registry.read_plastic_events`
  reads the new events and clears the recorder (`n_events = 0`), so a long run
  costs O(new events) per step instead of re-scanning (and growing) the whole
  history; the drained batch is shared by both limbs within a step.
- **Belt placement is selectable** (`--belt dataset|model`): `dataset` = the
  dataset's ground (min foot-marker height), `model` = the 5th percentile of the
  reconstructed target foot's lowest surface over the trajectory. Both are
  evaluated head-to-head.
- **`eps_pose`** is the per-step settling tolerance: the pose stage ends after
  `pose_hold_steps` consecutive steps with `max(|ankle_model − ankle_target|) <
  eps_pose` (or the `pose_max_steps` cap). It gates the *stage transition*, not
  the optimization.
- **`TODO(perf)`** marks the per-step exploration application
  (`Registry.apply_explored_parameters`), which dominates the step cost but is
  required for personalization.

## Review Focus

1. **A parameter that does not affect the error** — its perturbation does not covary with `A`: `influence()` ≈ 0 and its step ≈ 0 (must not drift to bounds). → Task 1/2.
2. **Short window / rank-deficient metric** — fewer samples than features: `F` and the step stay finite. → Task 1.
3. **Unit change** — rescaling a parameter's units leaves the physical step unchanged. → Task 1.
4. **Missing sensory input** (CUT=Ia=Ib=0) — CPG runs, no NaN, step finite. → Task 3/5.
5. **Muscle geoms over bone** — tendon capsules must not overlap bone geoms. → Task 4.

---

### Task 1: FisherMatrix — perturbation–advantage covariance (natural gradient)

**Files:**
- Modify: `persona/strategy.py`
- Test: `tests/test_variantb_metric.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `FisherMatrix.record_response(delta_theta: np.ndarray, advantage: np.ndarray)` — appends one `(Δθ, A)` sample to a rolling window.
  - `FisherMatrix.compute_metric() -> tuple[np.ndarray, np.ndarray]` returning `(F, scales)`, `F = Cov(Δθ) + λ_F I` shape `(n_θ, n_θ)`, symmetric, PD via ridge; `scales = sqrt(diag(F))` shape `(n_θ,)`.
  - `FisherMatrix.compute_gradient() -> np.ndarray` shape `(n_θ,)` = `Cov(Δθ, ΣA)`.
  - `FisherMatrix.influence() -> np.ndarray` shape `(n_θ,)`, normalised `|Cov(Δθ, A)|` in `[0,1]`.
  - `FisherMatrix.compute_natural_gradient_step(action_dict, mean_dict, sigma_ctrl_dict, advantage_vector) -> Dict[str, float]` (signature unchanged) using the recorded `(Δθ, A)` window; step `Δμ = η·F−1·g`.
- Existing `compute_correlation_matrix` becomes a thin wrapper returning `compute_metric()[0]` (kept so `validate`/`agent` callers still import).

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from persona.strategy import FisherMatrix

def _fm(n=5, window=200, lam=1e-6):
    names = [f"p{i}" for i in range(n)]
    return FisherMatrix(param_names=names, window_size=window, lambda_reg=lam)

def test_influence_ranks_high_impact_first():
    fm = _fm()
    rng = np.random.default_rng(0)
    for _ in range(50):
        dth = rng.standard_normal(5) * 0.1
        A = np.array([dth[0] * 10.0, dth[0] * 5.0])  # only p0 covaries with A
        fm.record_response(dth, A)
    infl = fm.influence()
    assert infl.shape == (5,)
    assert infl[0] > 0.9 and np.all(infl[1:] < 0.1)

def test_short_window_is_finite():
    fm = _fm(n=40, window=3)
    rng = np.random.default_rng(1)
    for _ in range(3):
        fm.record_response(rng.standard_normal(40), rng.standard_normal(2))
    F, s = fm.compute_metric()
    assert np.isfinite(F).all() and np.isfinite(s).all()
    d = fm.compute_natural_gradient_step(
        {f"p{i}": 0.1 for i in range(40)}, {f"p{i}": 0.0 for i in range(40)},
        {f"p{i}": 1.0 for i in range(40)}, np.array([1.0, 1.0]))
    assert all(np.isfinite(v) for v in d.values())

def test_step_is_unit_invariant():
    rng = np.random.default_rng(2)
    base = [rng.standard_normal(5) * 0.1 for _ in range(100)]
    def build(c):
        fm = _fm()
        for dth in base:
            A = np.array([dth[0] * 10.0, dth[1]])   # reward from the physical dth
            rec = dth.copy(); rec[0] *= c            # same data, p0 units rescaled
            fm.record_response(rec, A)
        return fm
    def step(fm):
        return fm.compute_natural_gradient_step(
            {f"p{i}": 1.0 for i in range(5)}, {f"p{i}": 0.0 for i in range(5)},
            {f"p{i}": 1.0 for i in range(5)}, np.array([1.0, 1.0]))
    d1 = step(build(1.0))["p0"]
    d2 = step(build(10.0))["p0"]
    # The physical step (Δμ·scale) is independent of the arbitrary unit scale.
    assert np.isclose(d2 * 10.0, d1, rtol=1e-6)
```

- [ ] **Step 2: Run to verify failure**

Run: `/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_variantb_metric.py -v`
Expected: FAIL (`record_response` / `compute_metric` / `influence` missing).

- [ ] **Step 3: Implement in `persona/strategy.py`**

- `__init__`: add `self.response_dtheta: List[np.ndarray] = []`, `self.response_da: List[np.ndarray] = []`.
- `record_response(delta_theta, advantage)`: append as float arrays (pad `Δθ` to `n_total`); trim to `window_size`.
- `_covariances()`: if `< 3` samples return `(np.eye(n_total), np.zeros((n_total, n_y)))`. Else `X = stack(dtheta)` (`W×n_total`), `Y = stack(A)` (`W×n_y`); centre columns → `Xc, Yc`; `F = (Xc.T @ Xc)/W + λ_F I`; `G = (Xc.T @ Yc)/W`. Cache both.
- `compute_metric()`: `(F, scales)` with `scales = sqrt(diag(F))` (per-parameter perturbation std).
- `compute_gradient()`: `g = G.sum(axis=1)` = `Cov(Δθ, ΣA)` (`n_total`).
- `influence()`: `row = ||G[i]||`; return `row / (row.max() + 1e-12)`.
- `compute_natural_gradient_step`: `g = compute_gradient()`; `nat = solve(F, g)` (fallback `pinv`); `deltas[name_i] = lr_mu * nat[i]` for `param_names`.
- `compute_correlation_matrix` → `return self.compute_metric()[0]`.

- [ ] **Step 4: Run to verify pass**

Run: `/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_variantb_metric.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/strategy.py tests/test_variantb_metric.py
git commit -m "feat(persona): perturbation-advantage covariance natural gradient + influence"
```

---

### Task 2: Agent wiring (record `(Δθ, A)`, log influence)

**Files:**
- Modify: `persona/agent.py`, `persona/parameter.py`
- Test: `tests/test_variantb_agent.py`

**Interfaces:**
- Consumes: `FisherMatrix.record_response`, `compute_natural_gradient_step`, `influence`.
- Produces:
  - `Agent.step(output, target, phase) -> dict` records the pair `(Δθ = action − mean, A = baseline(E) − E)` via `strategy.record_response`; return/history include `influence: np.ndarray` and `loss`.
  - `PersonaParam.log_sigma_param` property (`float`, `np.log(max(sigma_param,1e-8))`) and `set_log_sigma_param(value)` clipping `σ_param ∈ [1e-4, 2.0]` (kept; σ_param is fixed for the MVP).

- [ ] **Step 1: Write failing test** (synthetic plant: one parameter moves the output)

```python
import numpy as np, pytest
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.agent import Agent

@pytest.fixture
def make_agent(nest_kernel):
    def _make():
        reset_rng_stream(0)
        reg = Registry()
        reg.create_population(Population("izhikevich", 5,
                              {"a": PersonaParam(mean=1.0, sigma_ctrl=0.3, min=0.0, max=3.0)},
                              "pop"))
        return Agent(registry=reg, lr_mu=0.05, lr_sigma_ctrl=0.0, lr_sigma_param=0.0,
                     window_size=40), reg
    return _make

def test_loss_decreases_on_coupled_target(make_agent):
    agent, reg = make_agent()
    key = "pop_pop_a"
    for _ in range(120):
        agent.sample_actions()
        reg.apply_explored_parameters(agent.current_actions)
        a = agent.current_actions[key]
        out = np.array([abs(a - 2.0)] * 2)   # plant y = |a-2|; optimum at a=2.0
        agent.step(output=out, target=np.array([0.0, 0.0]),
                   phase=["stance", "swing"])
    assert abs(agent.params[key].mean - 2.0) < 0.5

def test_influence_logged(make_agent):
    agent, reg = make_agent(); agent.sample_actions()
    agent.step(np.zeros(2), np.ones(2), ["stance", "swing"])
    assert "influence" in agent.history and len(agent.history["influence"]) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `.../python -m pytest tests/test_variantb_agent.py -v`
Expected: FAIL (`influence` missing / loss does not decrease).

- [ ] **Step 3: Implement**

- `Agent.step`: after computing `advantage_vec`, build `dtheta` over `strategy.all_names` from `perturbations` and call `self.strategy.record_response(dtheta, advantage_vec)`; append `influence = self.strategy.influence()` to `history["influence"]`; return it. Remove the `set_plant_response` / `output_mean` machinery (the estimator consumes the advantage, not the output response).
- `PersonaParam`: keep `log_sigma_param` property + setter (clip `[1e-4, 2.0]`).
- Delete the `sign(adv)` parametric-variance hack; σ_param stays fixed (`TODO(sigma_param)`).

- [ ] **Step 4: Run to verify pass**

Run: `.../python -m pytest tests/test_variantb_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/agent.py persona/parameter.py tests/test_variantb_agent.py
git commit -m "feat(persona): agent records perturbation-advantage pairs, logs influence"
```

---

### Task 3: CPG topology, names, and initial values (cpg_2legs_fast)

**Files:**
- Modify: `t2xcpg/leg.py`, `t2xcpg/cpg.py`
- Test: `tests/test_cpg_topology.py`

**Interfaces:**
- Consumes: `Registry`, `Population`, `Connection`, `PersonaParam`.
- Produces: `Leg` wiring per spec §5.3; every connection `Connection.key` is the `cpg_2legs_fast` name; initial `weight`/`p` per spec §5.4.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from persona.registry import Registry
from t2xcpg.leg import Leg

def _leg():
    reg = Registry(); leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    return reg, leg

def test_halfcentre_is_via_in(nest_kernel):
    reg, leg = _leg()
    assert "rg_e_L_in_e_L" in reg.connection_cfgs
    assert "in_e_L_rg_f_L" in reg.connection_cfgs
    assert "rg_e_L_rg_f_L" not in reg.connection_cfgs      # no direct half-centre
    assert "in_e_L_m_e_L" not in reg.connection_cfgs       # InE does not drive MN

def test_rg_to_mn_direct(nest_kernel):
    reg, leg = _leg()
    assert "rg_e_L_m_e_L" in reg.connection_cfgs

def test_reference_values(nest_kernel):
    reg, leg = _leg()
    w = lambda k: reg.connection_cfgs[k].parameters["weight"].mean
    assert np.isclose(w("rg_e_L_in_e_L"), 12.0)
    assert np.isclose(w("in_e_L_rg_f_L"), -8.0)
    assert np.isclose(w("rg_f_L_in_f_L"), 18.0)
    assert np.isclose(w("in_f_L_rg_e_L"), -48.0)
```
(Skip `nest_kernel`-free runs; `test_*` take the `nest_kernel` fixture from `conftest.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `.../python -m pytest tests/test_cpg_topology.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Rewrite `Leg._build_populations` / `_build_connections` per spec §5.3–§5.4: remove direct `rg_e→rg_f`/`rg_f→rg_e` and `in_e→m_e`/`in_f→m_f`; add `rg_e→in_e` (W_RG2INE=12, p=P_RG_RECIP_E=0.15), `in_e→rg_f` (W_INE2RGF=−8), `rg_f→in_f` (W_RG2INF=18, p=P_RG_RECIP_F=0.30), `in_f→rg_e` (W_INF2RGE=−48), `rg_e→m_e`/`rg_f→m_f` (W_RG2MN=2.75), self-recurrence (W_RG_REC_E=4.0/F=5.5, P_RG_REC=0.12), Ia wires (IA2RG_W=12/IA2RG_P=0.4, W_IA_IN2INT=6, W_IA_INT2ANT=−10, W_IA2IN=6/P_IA2IN=0.25), CUT (`CUT→RGE` **STDP** seeded at W=14/P=0.35, `CUT→InE` W_CUT2INE=6/P=0.30), M→mus (W_M2MUS=1/P_M2MUS=0.8), BS `BS_REGULAR_HZ=60`, base drive 1.0/2.0/0.10. Keep Ib (project values) and Renshaw; rename the commissural/BS keys in `cpg.py` to reference names; izh params per spec (exc/inh/RG-F bursting). Set each `Connection.key` to the reference name.

- [ ] **Step 4: Run to verify pass**

Run: `.../python -m pytest tests/test_cpg_topology.py tests/test_cpg_build.py tests/test_stdp.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add t2xcpg/leg.py t2xcpg/cpg.py tests/test_cpg_topology.py
git commit -m "feat(t2xcpg): cpg_2legs_fast topology, reference names + initial values"
```

---

### Task 4: Render muscle geoms

**Files:**
- Modify: `model/render.py` (add `muscle_geoms_clear` check), `dual_rat_hindlimb.xml`
- Test: `tests/test_render_geoms.py`

**Interfaces:**
- Produces: thinner SOL/TA tendon `width` and offset origin/insert sites so the tendon path clears the bone capsule.

- [ ] **Step 1: Write failing test**

```python
import mujoco, numpy as np
def test_muscle_tendon_width_thin():
    m = mujoco.MjModel.from_xml_path("dual_rat_hindlimb.xml")
    w = float(m.tendon_width[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "sol_tendon_m_L")])
    assert w <= 0.0012
```

- [ ] **Step 2: Run to verify failure** — `.../python -m pytest tests/test_render_geoms.py -v` → FAIL.
- [ ] **Step 3: Implement** — set both `sol_tendon_*`/`ta_tendon_*` `width="0.0012"`, shrink the origin/insert site `size` to `0.0012`, and move the origin/insert sites outward (increase |y| by ~0.004) so the tendon clears the bone capsule.
- [ ] **Step 4: Run to verify pass** — PASS.
- [ ] **Step 5: Commit** — `git commit -m "fix(render): thinner, offset muscle tendons"`

---

### Task 5: Figures + fast validation run

**Files:**
- Modify: `persona/validation.py`, `run_validation.py`
- Test: `tests/test_validation.py` (extend)

**Interfaces:**
- Produces: `ValidationReport.plot_influence(agent)` (bar chart of `influence()`), `plot_metric(agent)` (metric heatmap with a caption noting the diagonal is impact), phase-shaded `plot_tracking`.

- [ ] **Step 1: Write failing test** — `plot_influence`/`plot_metric` produce files in `tmp_path`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the two plots; in `run_validation.run`, call them; shade stance/swing in `plot_tracking`.
- [ ] **Step 4: Run to verify pass**, then a fast end-to-end: `OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa python run_validation.py --fast --out /mnt/d/CPGvalidation/validation_mvp` → all outputs `ok`.
- [ ] **Step 5: Commit** — `git commit -m "feat(validation): influence + metric figures, phase-shaded tracking"`

---

## Self-Review

- Spec coverage: §5.1–5.2 → T1/T2; §5.3–5.4 → T3; §5.5–5.6 → T2 (σ_param) + T3 (set/names); §5.7 → T4; §8 → tests in each task. Review Focus items mapped: RF1→T1/T2, RF2→T1, RF3→T1, RF4→T3/T5, RF5→T4.
- Type consistency: `record_response(Δθ,Δy)`, `compute_metric()→(F,scales)`, `influence()→(n_θ,)`, `Plot.influence(agent)` consistent across tasks.
- Proportion: signatures + tests, bodies only where the algorithm is not determined.
- Open defaults: `λ_J=1e-3`, `λ_F=1e-3`; `σ_param∈[1e-4,2.0]`.