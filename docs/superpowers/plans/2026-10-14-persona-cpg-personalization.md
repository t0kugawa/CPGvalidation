# PERSONA–CPG–MuJoCo Personalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Personalize a biologically detailed NEST spiking CPG so a MuJoCo rat hindlimb tracks a mocap-driven target hindlimb, using the PERSONA natural-gradient (Fisher) scheme, and validate the result against rat locomotion statistics.

**Architecture:** `persona/` owns all optimizable parameters and the update rule (loss objective, correlation Fisher over parameters+noise, per-leg phase baselines, dual stddev). `t2xcpg/` builds the Diagram-2 CPG entirely through the persona `Registry`. `model/` runs a bilateral MuJoCo rig (controlled L/R + target L/R) and produces the CUT/Ia/Ib sensory vector. `loader/` turns Zenodo mocap + statistics into target trajectories and validation metrics. `persona/personalization.py` orchestrates the two-stage closed loop; `personalize.py` is the entry point; validation figures are produced at the end.

**Tech Stack:** Python 3.12 (WSL venv `/home/t0kgw/.virtualenvs/KinematicsCPG`), NEST 3.10, MuJoCo 3.14, numpy 2.5, scipy 1.18, pandas 3.0, h5py 3.16, matplotlib 3.11, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-persona-cpg-mujoco-design.md`

## Global Constraints

- Runtime is the WSL venv. All commands run from the repo root `/mnt/d/CPGvalidation` in WSL. From the Windows harness the wrapper is `wsl.exe -- /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python <args>` (never `bash -lc` with nested quotes). Plain form for a reader: `/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest <path>`.
- Do NOT modify `tinycpg/`, `memcpg/`, or the legacy `persona.py` script. They are references only.
- OOP, matching the existing class style. No comments except the `TODO(...)` hooks the spec requires at error/advantage sites.
- Objective = **loss** (minimize); `advantage = phase_baseline_loss − loss`; parameter step is natural-gradient **ascent** on advantage. Sign is provisional, marked `TODO(advantage)`.
- Vector error/advantage is **scalarized** for now via `mean(...)`, marked `TODO(advantage)`.
- `F` = empirical **correlation** matrix of standardized feature perturbations (params + noise), ridge `λI`; noise features contribute through `F⁻¹` only (no `score` term).
- Dual stddev: `sigma_ctrl` (exploration/exploitation) vs `sigma_param` (biological dispersion); both updated separately.
- No timer-driven afferents: every CUT/Ia/Ib rate comes from MuJoCo state.
- Ankle-only MVP: every non-ankle joint is servo-held to the target.
- `dt` consistent across NEST resolution, MuJoCo substeps, and mocap resample: `DT_MS = 10.0`.
- NEST resolution `1.0` ms; MuJoCo timestep `0.002` s (from the XML).

## Review Focus

The five inputs/conditions the spec implies but no task's tests fully exercise, most likely to bite a user; each gets a pinning test in the owning task:

1. **Rank-deficient / singular Fisher** (window < n_features): `F⁻¹` must stay finite (ridge), no `LinAlgError`, no NaN. → Task 3.
2. **Mirror/sign errors in the bilateral MuJoCo rig**: L and R limbs must be geometrically symmetric and every joint must stay within its declared range. → Task 8.
3. **Malformed mocap trial** (bad filename token, NaN or missing markers): loader must raise a clear error or skip, never silently substitute zeros. → Task 10.
4. **Empty NEST population / zero realized connections** (low `p`, small `n`): build must not crash and the `PersonaParam` must remain registered. → Task 6.
5. **Cold-start agent** (0–2 steps of history): identity Fisher, no division by zero, no NaN in `mu`/`sigma`. → Task 2.

---

### Task 1: Baseline commit + test harness

**Files:**
- Create: `pytest.ini`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_smoke.py`
- Commit: the entire current scaffold (see step 1)

**Interfaces:**
- Produces: a working `python -m pytest` from the repo root in WSL; `tests/conftest.py` inserts the repo root on `sys.path` and provides a `nest_kernel` fixture that calls `nest.ResetKernel()` + `nest.SetKernelStatus({"resolution": 1.0, "print_time": False})`.

- [ ] **Step 1: Commit the existing scaffold as the baseline**

```bash
git add -A
git commit -m "chore: commit PERSONA/CPG/MuJoCo scaffold baseline"
```
This includes the design doc, all `persona/`, `t2xcpg/`, `model/`, `loader/` modules and their `__init__.py`s, `personalize.py`, the XML, and the current `docs/`.

- [ ] **Step 2: Write the harness files**

`pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -q
```
`tests/conftest.py`: prepend repo root to `sys.path`; define fixture
`nest_kernel()` that returns the `nest` module after `ResetKernel()` and
`SetKernelStatus({"resolution": 1.0, "print_time": False})`.
`tests/test_smoke.py`:
```python
def test_paths():
    import persona, t2xcpg, model, loader  # noqa: F401
```

- [ ] **Step 3: Run the smoke test**

Run (WSL): `/home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest tests/test_smoke.py -v`
Expected: PASS. If `pytest` is missing, install it into the venv first (`.../python -m pip install pytest`) and note it.

- [ ] **Step 4: Commit**

```bash
git add pytest.ini tests
git commit -m "test: add pytest harness"
```

---

### Task 2: PERSONA loss objective + `TODO` hooks

**Files:**
- Modify: `persona/strategy.py`, `persona/agent.py`, `persona/parameter.py`
- Test: `tests/test_persona_objective.py`

**Interfaces:**
- Consumes: `PersonaParam.update_controlling_variance`, `PhaseBaselines`.
- Produces: `FisherMatrix.compute_advantage(loss_vector: np.ndarray, phase) -> np.ndarray` returns `baseline - loss`; `FisherMatrix.record_step(perturbation_dict, loss_vector, phase)`; `Agent.step(output, target, phase) -> dict` whose return and `Agent.history` include `loss` (a float) and `advantage_vector` (np.ndarray); `Agent.history["loss"]: list[float]`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from persona.strategy import PhaseBaselines, FisherMatrix

def test_advantage_is_baseline_minus_loss():
    ph = PhaseBaselines(alpha=0.0)  # baseline = last seen
    fm = FisherMatrix(param_names=["p"], window_size=10)
    fm.baselines = ph
    a1 = fm.compute_advantage(np.array([1.0]), "stance")   # baseline 0 - loss 1 = -1
    a2 = fm.compute_advantage(np.array([0.4]), "stance")   # baseline 1 - loss 0.4 = +0.6
    assert np.isclose(a1[0], -1.0) and np.isclose(a2[0], 0.6)

def test_agent_logs_loss(make_agent):
    agent, _ = make_agent()
    for _ in range(3):
        agent.sample_actions()
        out = agent.step(output=np.array([0.0, 0.0]), target=np.array([1.0, 1.0]), phase="stance")
    assert "loss" in out and len(agent.history["loss"]) == 3
    assert np.isfinite(agent.history["loss"]).all()
```

`make_agent` fixture (in the test file): builds a `Registry` with one izhikevich `Population` of n=5 with a single `PersonaParam` and returns `(Agent(registry=reg), reg)`.

- [ ] **Step 2: Run to verify failure**

Run: `.../python -m pytest tests/test_persona_objective.py -v`
Expected: FAIL (`compute_advantage` signature/objective mismatch, missing `loss`).

- [ ] **Step 3: Implement**

- `PhaseBaselines`: key baseline storage by an arbitrary phase **key** (string or per-component tuple), keep `get_baseline(phase_key, dim)` / `update(phase_key, loss_vector)`.
- `FisherMatrix.compute_advantage(self, loss_vector, phase)`: `base = self.baselines.get_baseline(phase, len(loss_vector))`; `adv = base - loss`; `self.baselines.update(phase, loss)`; return `adv`. Add `# TODO(advantage): component-wise vector advantage (see spec §5.1) — currently baseline−loss for the scalarized loss.` immediately above.
- `Agent.step`: compute `error_vec = |target-output|`, `loss = float(np.mean(error_vec))` (mark `# TODO(advantage)`), `advantage_vec = strategy.compute_advantage(error_vec_broadcast?...)`. For the MVP pass the **loss vector** `loss_vec = error_vec` into the per-phase baseline and use `adv = baseline(error_vec) - error_vec`; the scalar used in the step is `mean(adv)`. Append `loss` to `history["loss"]`. Keep the existing `update_mean` / `update_controlling_variance` / `update_parametric_variance` calls.
- `parameter.py`: no behavior change required; ensure `update_controlling_variance` clips and never NaNs for `advantage=0`.

- [ ] **Step 4: Run to verify pass + the cold-start pin**

Add and run `test_cold_start_no_nan` (Review Focus 5): with 1–2 steps of history, `Agent.step` yields finite `mu`, `sigma_ctrl`, `sigma_param`, and `FisherMatrix.compute_correlation_matrix()` is finite.
Run: `.../python -m pytest tests/test_persona_objective.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add persona/strategy.py persona/agent.py persona/parameter.py tests/test_persona_objective.py
git commit -m "feat(persona): loss objective + TODO(advantage) hooks"
```

---

### Task 3: Correlation Fisher + unit standardization + noise coupling

**Files:**
- Modify: `persona/strategy.py`
- Test: `tests/test_fisher.py`

**Interfaces:**
- Produces: `FisherMatrix.compute_correlation_matrix() -> np.ndarray` shape `(n_total, n_total)`, unit diagonal, symmetric, ridge-regularized; `FisherMatrix.feature_scales() -> np.ndarray` (per-feature std used for standardization, so the step can be mapped back to physical units); `FisherMatrix.compute_natural_gradient_step(action_dict, mean_dict, sigma_ctrl_dict, advantage_vector) -> Dict[str, float]` restricted to `param_names`, using `F` (correlation) and `score_i = (a_i-μ_i)/σ_ctrl,i²` with scalarized advantage.

- [ ] **Step 1: Write failing tests**

```python
def test_correlation_unit_diagonal_and_symmetric():
    fm = make_hist_fisher(n_feat=6, kind="mixed_units")
    F = fm.compute_correlation_matrix()
    assert F.shape == (fm.n_total, fm.n_total)
    assert np.allclose(np.diag(F), 1.0, atol=1e-6)
    assert np.allclose(F, F.T, atol=1e-9)
    assert np.linalg.eigvalsh(F).min() > 0  # ridge => PD even when rank-deficient

def test_mixed_units_unified():
    # features on wildly different scales => correlation identical up to permutation
    F1 = make_hist_fisher(n_feat=4, scales=[1e-3, 1.0, 1e3, 1e-1]).compute_correlation_matrix()
    F2 = make_hist_fisher(n_feat=4, scales=[1.0, 1.0, 1.0, 1.0]).compute_correlation_matrix()
    assert np.allclose(np.sort(F1.ravel()), np.sort(F2.ravel()), atol=1e-8)

def test_noise_features_enter_fisher_not_score():
    fm = make_hist_fisher(n_feat=3, noise_names=["noise_x"])
    deltas = fm.compute_natural_gradient_step(
        {"p0": 1.0, "p1": 1.0, "p2": 1.0},
        {"p0": 0.0, "p1": 0.0, "p2": 0.0},
        {"p0": 1.0, "p1": 1.0, "p2": 1.0},
        np.array([1.0]),
    )
    assert set(deltas) == {"p0", "p1", "p2"}  # never a noise key

def test_singular_window_no_crash():
    fm = make_hist_fisher(n_feat=40, window=5)  # Review Focus 1
    d = fm.compute_natural_gradient_step({f"p{i}": 0.1 for i in range(40)},
                                          {f"p{i}": 0.0 for i in range(40)},
                                          {f"p{i}": 1.0 for i in range(40)},
                                          np.array([1.0]))
    assert all(np.isfinite(v) for v in d.values())
```

`make_hist_fisher` helper: builds a `FisherMatrix` with `n_feat` params named `p0..`, optional `noise_names`, optional per-feature `scales`, and fills `record_step` history with random perturbations.

- [ ] **Step 2: Run to verify failure**

Run: `.../python -m pytest tests/test_fisher.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`compute_correlation_matrix`: `X = array(perturbation_history)`; column standardize `Xs = (X - mean)/max(std, eps)` (store `self._scales`); `C = Xs.T @ Xs / max(1, n-1)`; return `C + lambda_reg * I`. `feature_scales` returns `self._scales`. `compute_natural_gradient_step`: as today but `# TODO(fisher): recursive/Sherman–Morrison inverse or pruned subspace (see spec §5.1 well-posedness)`; solve `F x = grad` with `np.linalg.solve`, fall back to `pinv` on `LinAlgError`.

- [ ] **Step 4: Run to verify pass**

Run: `.../python -m pytest tests/test_fisher.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add persona/strategy.py tests/test_fisher.py
git commit -m "feat(persona): correlation Fisher + unit standardization + noise coupling"
```

---

### Task 4: Per-leg phase baselines

**Files:**
- Modify: `persona/strategy.py`, `persona/agent.py`
- Test: `tests/test_phase.py`

**Interfaces:**
- Consumes: `PhaseBaselines`.
- Produces: `Agent.step(output, target, phase)` accepts `phase` as either a `str` (broadcast) or a `Sequence[str]` of length `len(output)`; each component is baselined under its own key. `Agent.history["phase"]` stores the per-step phase value used.

- [ ] **Step 1: Write failing tests**

```python
def test_per_component_phase_independent():
    fm = FisherMatrix(param_names=["p"], window_size=10)
    fm.baselines = PhaseBaselines(alpha=0.0)
    fm.compute_advantage(np.array([2.0, 2.0]), ("stance", "swing"))
    a = fm.compute_advantage(np.array([1.0, 2.0]), ("stance", "swing"))
    assert np.isclose(a[0], 1.0) and np.isclose(a[1], 0.0)  # only stance moved

def test_agent_accepts_phase_vector(make_agent):
    agent, _ = make_agent()
    agent.sample_actions()
    agent.step(np.zeros(2), np.ones(2), phase=["stance", "swing"])
    assert agent.history["phase"][-1] == ("stance", "swing")
```

- [ ] **Step 2–4:** run (FAIL) → implement `PhaseBaselines` per-key dict storage and `Agent.step` phase normalization (`str` → tuple of same length; `list/tuple` kept) → run (PASS).
Run: `.../python -m pytest tests/test_phase.py -v`

- [ ] **Step 5: Commit**

```bash
git add persona/strategy.py persona/agent.py tests/test_phase.py
git commit -m "feat(persona): per-leg phase baselines"
```

---

### Task 5: Noise recorder captures STDP weight deltas

**Files:**
- Modify: `persona/noise_recorder.py`, `persona/agent.py`
- Test: `tests/test_noise.py`

**Interfaces:**
- Produces: `NoiseRecorder.observe_weight(producer_id, t, weight) -> NoiseSample` (delta vs previous); `NoiseRecorder.get_latest_step_deltas()` returns tags `noise_<producer>_weight` alongside the existing `noise_<producer>_rate`. `Agent.step` merges these into `perturbations` (already does).

- [ ] **Step 1: Write failing tests**

```python
def test_weight_delta_tagged():
    nr = NoiseRecorder("n")
    nr.observe_weight("cut_rge", 0.0, 10.0)
    nr.observe_weight("cut_rge", 1.0, 12.5)
    d = nr.get_latest_step_deltas()
    assert np.isclose(d["noise_cut_rge_weight"], 2.5)

def test_rate_and_weight_coexist():
    nr = NoiseRecorder("n")
    nr.observe("cut", 0.0, "rate", 5.0)
    nr.observe("cut", 1.0, "rate", 7.0)
    nr.observe_weight("cut_rge", 0.0, 1.0)
    nr.observe_weight("cut_rge", 1.0, 3.0)
    d = nr.get_latest_step_deltas()
    assert np.isclose(d["noise_cut_rate"], 2.0) and np.isclose(d["noise_cut_rge_weight"], 2.0)
```

- [ ] **Step 2–4:** run (FAIL) → implement `observe_weight` (reuse `_last_values` keyed by `(producer, "weight")`, tag `noise_<producer>_weight`) → run (PASS).
Run: `.../python -m pytest tests/test_noise.py -v`

- [ ] **Step 5: Commit**

```bash
git add persona/noise_recorder.py tests/test_noise.py
git commit -m "feat(persona): noise recorder captures STDP weight deltas"
```

---

### Task 6: CPG circuit completion (Diagram-2, registry-complete, no timers)

**Files:**
- Modify: `t2xcpg/leg.py`, `t2xcpg/utils.py`
- Test: `tests/test_cpg_build.py`

**Interfaces:**
- Consumes: `Registry`, `Population`, `Connection`, `PersonaParam`.
- Produces: `Leg` builds populations `rg_e,f`, `m_e,f`, `r_e,f` (Renshaw), `ia_e,f` (inhibitory interneurons), `ia_e_in,ia_f_in`, `ib_e_in,ib_f_in`, `cut_in`, reciprocal `in_e,f`, commissural `v2a,v0v,v0d,v3_f,in1`, `mus_e,f`; every `nest.Create` goes through `Registry` (or is registered via `registry.population_nodes`); `Leg.update_sensory_feedback(cut_hz, ia_e_hz, ia_f_hz, ib_e_hz, ib_f_hz, current_time_ms)` sets Poisson rates and records deltas. `L2CPG.update(CPGInput)` routes per-leg.

- [ ] **Step 1: Write failing tests**

```python
def test_populations_registered(nest_kernel):
    reg = Registry()
    leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    for name in ["rg_e_L", "rg_f_L", "m_e_L", "m_f_L", "r_e_L", "r_f_L",
                 "ia_e_L", "ia_f_L", "in_e_L", "in_f_L", "v2a_L", "v0v_L", "v0d_L", "v3_f_L", "in1_L"]:
        assert reg.get(name) is not None

def test_afferent_generators_present(nest_kernel):
    reg = Registry(); leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    for name in ["cut_in_L", "ia_e_in_L", "ia_f_in_L", "ib_e_in_L", "ib_f_in_L"]:
        assert name in reg.population_nodes

def test_optimizable_params_nonempty(nest_kernel):
    reg = Registry(); Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    params = reg.get_all_optimizable_params()
    assert len(params) > 20 and all(k.startswith(("pop_", "conn_")) for k in params)

def test_zero_realized_connections_no_crash(nest_kernel):
    # Review Focus 4: tiny probability must not crash and param stays registered
    reg = Registry(); Leg("L", reg, n_rg=4, n_mn=4, n_int=4, n_aff=4,
                          p_scale=1e-6)
    assert len(reg.get_all_optimizable_params()) > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.../python -m pytest tests/test_cpg_build.py -v`
Expected: FAIL (missing populations / params / `p_scale`).

- [ ] **Step 3: Implement**

Audit and complete `Leg._build_populations` / `_build_connections` per spec §5.2: keep the Renshaw loop (`m_e→r_e→m_e`, `m_f→r_f→m_f`, `r_e⊣ia_e`, `r_f⊣ia_f`), Ia separation (`ia_e_in→{m_e, ia_e, rg_e}`, `ia_f_in→{m_f, ia_f, rg_f}`), Ib autogenic inhibition (`ib_e_in⊣m_e`, `ib_f_in⊣m_f`), CUT (`cut_in→rg_e`, `cut_in→in_e`), commissural V-pathways, reciprocal InE/InF with asymmetric E/F gains seeded from `cpg_2legs_fast` constants. Add an optional `p_scale: float = 1.0` constructor arg multiplying every `p` (for the zero-connection test). Ensure sensory Poisson generators are registered.

- [ ] **Step 4: Run to verify pass**

Run: `.../python -m pytest tests/test_cpg_build.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add t2xcpg/leg.py t2xcpg/utils.py tests/test_cpg_build.py
git commit -m "feat(t2xcpg): registry-complete Diagram-2 circuit with separated Ia/Ib"
```

---

### Task 7: STDP plastic projections wired to weight recorders

**Files:**
- Modify: `t2xcpg/leg.py`, `persona/registry.py`, `persona/noise_recorder.py`
- Test: `tests/test_stdp.py`

**Interfaces:**
- Produces: `Leg.plastic_projections: list[str]` (keys `"cut_rge_L"`, `"ia_rg_L"`, …); each plastic `Connection` carries a NEST `weight_recorder` via `Connection.extra_syn_spec["weight_recorder"]`; `Registry.sample_plastic_weight_deltas(t_before, t_now) -> Dict[str, float]` returns mean per-synapse Δw per projection, which `L2CPG.step` feeds to the leg's `NoiseRecorder.observe_weight` via `Agent`.

- [ ] **Step 1: Write failing tests**

```python
def test_plastic_projections_exist(nest_kernel):
    reg = Registry(); leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    assert any("cut" in k and "rge" in k for k in leg.plastic_projections)
    for k in leg.plastic_projections:
        assert leg.registry.connection_cfgs[k].model.startswith("stdp")

def test_weight_delta_sampled(nest_kernel):
    reg = Registry(); leg = Leg("L", reg, n_rg=8, n_mn=8, n_int=6, n_aff=6)
    deltas = reg.sample_plastic_weight_deltas(0.0, 10.0)
    assert isinstance(deltas, dict) and all(np.isfinite(v) for v in deltas.values())
```

- [ ] **Step 2–4:** run (FAIL) → implement `create_stdp_synapse` wiring so `weight_recorder` is in `syn_spec`, keep `Connection.model` as the copied stdp model name, implement `Registry.sample_plastic_weight_deltas` reading recorder `events["weights"]` → run (PASS).
Run: `.../python -m pytest tests/test_stdp.py -v`

- [ ] **Step 5: Commit**

```bash
git add t2xcpg/leg.py persona/registry.py persona/noise_recorder.py tests/test_stdp.py
git commit -m "feat: STDP plastic projections reported as noise features"
```

---

### Task 8: Bilateral MuJoCo rig (4 limbs)

**Files:**
- Modify: `dual_rat_hindlimb.xml`
- Test: `tests/test_rig.py`

**Interfaces:**
- Produces: model with joints `hip_m_L/knee_m_L/ankle_m_L/toe_m_L`, `*_m_R`, `hip_ref_L/…`, `hip_ref_R/…`; actuators `sol_m_L/ta_m_L`, `sol_m_R/ta_m_R`, position servos for every non-ankle joint of both `_m_*` and `_ref_*`; plantar `touch` sensors on `_m_L` and `_m_R`; `framepos`/`framelinvel` on both `paw_contact_m_*`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np, mujoco
XML = "dual_rat_hindlimb.xml"

def _load():
    m = mujoco.MjModel.from_xml_path(XML); return m, mujoco.MjData(m)

def _jid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
def _aid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
def _sid(m, n): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)

def test_four_limb_joints_and_actuators():
    m, _ = _load()
    for side in "LR":
        for j in ["hip_m", "knee_m", "ankle_m", "hip_ref", "knee_ref", "ankle_ref"]:
            assert _jid(m, f"{j}_{side}") >= 0, f"{j}_{side}"
        for a in ["sol_m", "ta_m"]:
            assert _aid(m, f"{a}_{side}") >= 0, f"{a}_{side}"

def test_lr_symmetry():  # Review Focus 2
    m, d = _load()
    for j in ["hip_m", "knee_m", "ankle_m"]:
        loL, hiL = m.jnt_range[_jid(m, f"{j}_L")]; loR, hiR = m.jnt_range[_jid(m, f"{j}_R")]
        assert np.allclose([loL, hiL], [loR, hiR])

def test_touch_sensors_both_sides():
    m, _ = _load()
    for side in "LR":
        assert _sid(m, f"cut_heel_0_{side}") >= 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.../python -m pytest tests/test_rig.py -v`
Expected: FAIL (only `_m`/`_ref` exist).

- [ ] **Step 3: Implement**

Duplicate the existing `MODEL HINDLIMB` and `REFERENCE HINDLIMB` bodies, mirroring `y` (side offset) and renaming suffixes to `_L`/`_R`; duplicate tendons/actuators/sensors per side; give each `_m_*` a full set of position servos on hip/knee/toe (ankle stays muscle-driven) so only the ankle is free. Preserve per-side rest poses identical to the current ones.

- [ ] **Step 4: Run to verify pass**

Run: `.../python -m pytest tests/test_rig.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dual_rat_hindlimb.xml tests/test_rig.py
git commit -m "feat(model): bilateral 4-limb MuJoCo rig"
```

---

### Task 9: Bilateral adapter + ankle-only target servoing

**Files:**
- Modify: `model/leg.py`, `model/adapter.py`, `model/model.py`
- Test: `tests/test_adapter.py`

**Interfaces:**
- Produces: `CPGInput` gains per-side fields (already L/R; ensure all filled for both sides incl. `ia_len_*`, `ia_vel_*`, `ib_act_*`, `ib_force_*`, `phase_*`, `ankle_angle_m_*`, `ankle_angle_ref_*`). `CPGOutput` carries `mus_E_L,mus_F_L,mus_E_R,mus_F_R`. `Leg.step(cpg_output, target_ankle_L=None, target_ankle_R=None) -> CPGInput` drives `sol_m_*/ta_m_*` from the CPG, servos every non-ankle joint to the target trajectory, and reads both legs' afferents.

- [ ] **Step 1: Write failing tests**

```python
from model.adapter import CPGOutput
from model.leg import Leg

def test_both_legs_sensory_populated():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    out = CPGOutput(mus_E_L=0.5, mus_F_L=0.1, mus_E_R=0.1, mus_F_R=0.5)
    ci = leg.step(out, target_ankle_L=0.3, target_ankle_R=-0.1)
    for f in ["cut_L", "cut_R", "ia_rate_E_L", "ia_rate_E_R", "ib_force_E_L", "ib_force_E_R"]:
        assert getattr(ci, f) is not None

def test_non_ankle_joints_track_target():
    leg = Leg(mujoco_cfg="dual_rat_hindlimb.xml", dt_ms=10.0)
    out = CPGOutput()
    for _ in range(50):
        ci = leg.step(out, target_ankle_L=0.2, target_ankle_R=0.2)
    assert abs(ci.ankle_angle_ref_L - 0.2) < 0.05
```

- [ ] **Step 2–4:** run (FAIL) → implement bilateral ids (`*_L`,`*_R`), ankle-only servoing, per-side afferent extraction (no copying L into R) → run (PASS).
Run: `.../python -m pytest tests/test_adapter.py -v`

- [ ] **Step 5: Commit**

```bash
git add model/leg.py model/adapter.py model/model.py tests/test_adapter.py
git commit -m "feat(model): bilateral adapter, ankle-only target servoing"
```

---

### Task 10: Mocap loader (filename decode + marker→angle)

**Files:**
- Create: `loader/mocap.py`
- Modify: `loader/loader_cfg.py`
- Test: `tests/test_mocap.py`

**Interfaces:**
- Produces:
  - `decode_trial_name(name: str) -> TrialMeta` with fields `rat_id:str`, `speed_ratio:float` (`x1d50→1.5`), `fast_belt:str` (`L`/`R`), `condition:str` (`T`/`S`/`W`), `trial:int`.
  - `markers_to_angles(df) -> dict[str, np.ndarray]` returning per-side `hip_L,knee_L,ankle_L,hip_R,knee_R,ankle_R` in rad from marker triples (IC→GT, GT→KN, KN→EA, EA→MH), zero-aligned to a supplied rest pose.
  - `MocapLoader(path, rest_pose, dt_ms=10.0)` with `.resample()`, `.__len__`, `.ankle(side)`, `.__iter__`.
  - `treadmill_surfacevel(meta: TrialMeta, slow_mps=0.2) -> tuple[float, float]` (slow, fast).

- [ ] **Step 1: Write failing tests**

```python
import numpy as np, pandas as pd
from loader.mocap import decode_trial_name, markers_to_angles, treadmill_surfacevel

def test_decode():
    m = decode_trial_name("RatA_x1p75_Rfast_T2.csv")
    assert (m.rat_id, m.speed_ratio, m.fast_belt, m.condition, m.trial) == ("RatA", 1.75, "R", "T", 2)

def test_markers_to_angles_straight_leg():
    # synthetic: collinear markers => hip ~ knee ~ ankle ~ 0 offset
    d = {c: [0.0] for c in COLLINEAR_COLUMNS}
    a = markers_to_angles(pd.DataFrame(d), rest_pose=None)
    assert all(np.isfinite(v).all() for v in a.values())

def test_bad_filename_raises():  # Review Focus 3
    import pytest
    with pytest.raises(ValueError):
        decode_trial_name("nonsense.csv")

def test_nan_markers_raise():
    import pytest
    d = {c: [np.nan] for c in COLLINEAR_COLUMNS}
    with pytest.raises(ValueError):
        markers_to_angles(pd.DataFrame(d), rest_pose=None)

def test_treadmill_ratio():
    assert treadmill_surfacevel(decode_trial_name("RatA_x2p00_Lfast_S1.csv"))[1] == 0.4
```

- [ ] **Step 2–4:** run (FAIL) → implement pure-numpy angle math (no h5/osim dependency), validation that raises on unknown tokens / NaN markers → run (PASS).
Run: `.../python -m pytest tests/test_mocap.py -v`

- [ ] **Step 5: Commit**

```bash
git add loader/mocap.py loader/loader_cfg.py tests/test_mocap.py
git commit -m "feat(loader): mocap loader with filename decode + marker->angle"
```

---

### Task 11: Statistical loader + locomotion metrics

**Files:**
- Create: `loader/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces: `StatsLoader(path)` reading `StatisticalDataTable.csv`; `.distribution(field, rat_ids=None) -> np.ndarray`; `LocomotionMetrics.from_run(ankle_L, ankle_R, contact_L, contact_R, dt_ms) -> dict` computing `duty_factor_*, double_support, step_length_*, stride_length_*`; `in_distribution(value, dist, k=2.0) -> bool`.

- [ ] **Step 1: Write failing tests**

```python
from loader.stats import StatsLoader, LocomotionMetrics, in_distribution

def test_distribution_reads_all_rats():
    dl = StatsLoader("data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv")
    d = dl.distribution("LH_duty_factor")
    assert d.size >= 5 and 0.0 < d.mean() < 1.0

def test_metrics_from_synthetic_gait():
    import numpy as np
    n = 400; dt = 10.0
    t = np.arange(n) * dt
    contact = ((t // 500) % 2 == 0).astype(float)  # 50% duty
    m = LocomotionMetrics.from_run(np.zeros(n), np.zeros(n), contact, contact, dt)
    assert abs(m["duty_factor_L"] - 0.5) < 0.15

def test_in_distribution():
    import numpy as np
    assert in_distribution(0.6, np.array([0.55, 0.6, 0.65]))
```

- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS).
Run: `.../python -m pytest tests/test_stats.py -v`

- [ ] **Step 5: Commit**

```bash
git add loader/stats.py tests/test_stats.py
git commit -m "feat(loader): statistical loader + locomotion metrics"
```

---

### Task 12: Two-stage personalization orchestrator

**Files:**
- Create: `persona/personalization.py`
- Modify: `personalize.py` (make it call the orchestrator)
- Test: `tests/test_personalization.py`

**Interfaces:**
- Produces: `Personalizer(registry, cpg, rig, agent, target_driver, dt_ms, eps_pose=0.08, pose_hold_steps=20, loss_stop=0.05, loss_window=30, max_steps=1000)` with `.run() -> RunLog` where `RunLog` has arrays `time_ms, ankle_model_L, ankle_model_R, ankle_target_L, ankle_target_R, loss, advantage, phase_L, phase_R` and `stage: list[str]` (`"pose"`/`"track"`). Target driver is any object with `.target(t_ms) -> (ankle_L, ankle_R)` and, during Stage 1, a `.pose() -> (ankle_L, ankle_R)`.

- [ ] **Step 1: Write failing tests (synthetic target, real NEST/CPG, small net)**

```python
def test_stage1_pose_then_stage2_track():
    # small CPG + a synthetic target driver whose pose()=(0.2,0.2) and
    # target(t)=(0.2+0.05*sin, 0.2-0.05*sin)
    log = build_small_personalizer(max_steps=40).run()
    assert log.stage[0] == "pose"
    assert "track" in log.stage
    assert len(log.loss) == len(log.time_ms)
    assert np.isfinite(log.loss).all()

def test_early_stop_triggers():
    log = build_small_personalizer(loss_stop=1e9, max_steps=40).run()
    assert sum(1 for s in log.stage if s == "track") < 40
```

- [ ] **Step 2–4:** run (FAIL) → implement the loop: Stage 1 holds `target.pose()` until both ankle errors `< eps_pose` for `pose_hold_steps`, then Stage 2 iterates `target.target(t)`; each step calls `agent.sample_actions()`, `registry.apply_explored_parameters(...)`, `cpg.update(cpg_in)`, `cpg_out=cpg.step()`, `cpg_in=rig.step(cpg_out, *target)`, `agent.step([ankle_L,ankle_R], [tgt_L,tgt_R], phase=[phase_L,phase_R])`; early stop when windowed-mean loss `< loss_stop` → run (PASS).
Run: `.../python -m pytest tests/test_personalization.py -v`

- [ ] **Step 5: Commit**

```bash
git add persona/personalization.py personalize.py tests/test_personalization.py
git commit -m "feat(persona): two-stage personalization orchestrator"
```

---

### Task 13: Validation figures

**Files:**
- Create: `persona/validation.py`
- Test: `tests/test_validation.py`

**Interfaces:**
- Consumes: `RunLog`, `StatsLoader`, `LocomotionMetrics`.
- Produces: `ValidationReport` with `.plot_tracking(log)`, `.plot_inputs(log)`, `.plot_learning(log)`, `.plot_distribution(metrics, stats)`, `.save_all(path)`; `biological_acceptance(metrics, stats, angle_errors, phase_mask, margin=0.05, coverage=0.70) -> dict` returning `{"global_in_distribution": bool, "angle_coverage": float, "acceptable": bool}`.

- [ ] **Step 1: Write failing tests (Agg backend, synthetic RunLog)**

```python
import matplotlib; matplotlib.use("Agg")
def test_figures_written(tmp_path):
    log = synthetic_runlog(n=100)
    rep = ValidationReport(); rep.plot_tracking(log); rep.plot_learning(log)
    rep.save_all(tmp_path)
    assert (tmp_path / "tracking.png").exists() and (tmp_path / "learning.png").exists()

def test_angle_acceptance_rule():
    err = np.array([0.01]*70 + [0.3]*30)
    res = biological_acceptance({}, stats=None, angle_errors=err,
                                phase_mask=np.ones(100, bool), margin=0.05, coverage=0.70)
    assert res["angle_coverage"] == 0.70 and res["acceptable"] is True
```

- [ ] **Step 2–4:** run (FAIL) → implement → run (PASS).
Run: `.../python -m pytest tests/test_validation.py -v`

- [ ] **Step 5: Commit**

```bash
git add persona/validation.py tests/test_validation.py
git commit -m "feat(persona): validation figures + biological acceptance rule"
```

---

### Task 14: Wire the full experiment entry point

**Files:**
- Modify: `personalize.py`
- Test: `tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_personalization(reference_path, stats_path, output_dir, n_steps, device="cpu") -> dict` that builds the bilateral CPG + rig + Agent + `MocapTargetDriver` (RatA trial), runs `Personalizer`, writes `validation_results.png` + `run_log.npz`, and returns the acceptance dict.

- [ ] **Step 1: Write failing smoke test (short, small net, real NEST+MuJoCo)**

```python
def test_e2e_smoke(tmp_path):
    res = run_personalization(
        reference_path="data/zenodo/DatasetMotionCaputure/RatA/RatA_x1p50_Lfast/RatA_x1p50_Lfast_T1.csv",
        stats_path="data/zenodo/DatasetStatisticalAnalysis/StatisticalDataTable.csv",
        output_dir=str(tmp_path), n_steps=20, small=True)
    assert (tmp_path / "validation_results.png").exists()
    assert "acceptable" in res
```

- [ ] **Step 2–4:** run (FAIL) → implement wiring incl. `small=True` reducing populations/steps → run (PASS).
Run: `.../python -m pytest tests/test_e2e_smoke.py -v -s`

- [ ] **Step 5: Commit**

```bash
git add personalize.py tests/test_e2e_smoke.py
git commit -m "feat: end-to-end personalization entry point"
```

---

## Self-Review

**Spec coverage:**
- §5.1 objective/loss/TODO → T2; correlation Fisher/units/well-posedness → T3; per-leg phase → T4; noise STDP → T5,T7; dual stddev → already present + T2 assertions.
- §5.2 circuit, Ia/Ib separation, Renshaw, no timers, registry → T6; STDP plasticity → T7.
- §5.3 bilateral rig → T8; bilateral adapter + ankle-only servoing → T9.
- §5.4 mocap decode/angles/treadmill → T10; stats → T11.
- §5.5 two-stage + stop → T12; §5.6 figures + acceptance → T13; entry point → T14.
- §8 testing → the tests above; Review Focus items pinned: RF1→T3, RF2→T8, RF3→T10, RF4→T6, RF5→T2.

**Type consistency:** `phase` is `str | Sequence[str]` in T4 and consumed as `[phase_L, phase_R]` in T12; `RunLog` fields defined in T12 are the exact fields read in T13/T14; `Registry.sample_plastic_weight_deltas` (T7) is consumed by T12's loop via the leg's recorder; `markers_to_angles` returns the same six keys T12/T14 index.

**Proportion:** tasks specify signatures + test names/assertions, not full bodies; body detail appears only where the algorithm is not determined by the signature/test.

**Open defaults** (tunable, set in T12/T14): `eps_pose=0.08`, `pose_hold_steps=20`, `loss_stop=0.05`, angle margin `0.05` / coverage `0.70` (spec §5.6).