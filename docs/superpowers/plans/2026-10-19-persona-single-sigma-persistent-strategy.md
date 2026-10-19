# PERSONA single-σ + persisted two-type strategy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace PERSONA's dual σ with one recorded σ per site, make the recorded sample the applied sample, persist a β-merged metric+trace+credit strategy that updates every dt, route all NEST traffic through the Registry, and surface apply failures.

**Architecture:** `persona/parameter.py` keeps `(μ, σ)`; `persona/noise_recorder.py` becomes per-site/per-producer with read-and-clear freshness; `persona/strategy.py` keeps `F = βF+(1−β)ssᵀ`, `X̄ = βX̄+(1−β)a`, `G = βG+(1−β)X̄Aᵀ` (a in score units for μ rows), `Δμ = η(F+λI)⁻¹(G·1)`; `persona/registry.py` is the single NEST route and realizes every site per instance.

**Tech Stack:** Python 3.12 (WSL venv), NEST 3.10, MuJoCo 3.14, numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-10-19-persona-single-sigma-persistent-strategy-design.md`

## Global Constraints

- Runtime: WSL venv `/home/t0kgw/.virtualenvs/KinematicsCPG`; repo `/mnt/d/CPGvalidation`.
  All test commands run as
  `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python -m pytest ... -v"`.
- Do NOT modify `tinycpg/`, `memcpg/`, `persona.py`.
- Keep `TODO(advantage)`/`TODO(fisher)`; add `TODO(sigma-ewma)`; keep `TODO(perf)`.
- Output `y = [ankle_model_L, ankle_model_R]`; `dt = 10 ms`.
- β = 1 − dt/τ_mem, τ_mem default 2.0 s (β = 0.995); advantage baselines α = 0.90 unchanged.
- Trace μ rows are in score units (`Δθ/σ²`); σ trace row is `Δlogσ`; noise has no trace rows.
- σ update is variant A: `Δlogσ = η_σ·clip((z²−1)·mean(A), ±0.25)` with sample-time μ/σ.
- Channel layout everywhere: `[site μ-rows; site σ-rows; noise rows]`, sites in `param_names` order.

## Review Focus

1. Site scales span σ 0.005…12 — the step must not carry a per-site σ² gain; the tiniest-σ site stays finite.
2. A held sample under `realize_every_k > 1` — recorded = applied, no fresh draw sneaks in.
3. Poisson `rate` device populations (BS-E/F, BASE) — per-device realization, never the mean-of-draws fallback.
4. A producer with no events in a step — fresh `None`, never a stale value.
5. L/R key collisions (`cut`, `ia_*`, `ib_*`) — every recorder key unique.

---

### Task 1: PersonaParam — single σ, sample-time snapshot

**Files:**
- Modify: `persona/parameter.py`, `persona/population.py`
- Modify (minimal port): `persona/agent.py`
- Modify (ctor sweep): `t2xcpg/leg.py`, `t2xcpg/bs.py`
- Create: `tests/test_param_sigma.py`
- Modify: `tests/test_persona_objective.py`, `tests/test_phase.py`, `tests/test_review_fixes.py`

**Interfaces:**
- Produces:
  - `PersonaParam(mean, sigma=0.1, min=-1e6, max=1e6, sigma_min=1e-4, sigma_max=None, rng=...)`;
    `__post_init__` sets `sigma_max = max(2.0, sigma)` when `None`.
  - `sample_exploration() -> float` (x̂ = clip(μ+σε)); `sample_population(n, center=None)` (realization draws); `sample(n)` alias.
  - `sigma_step(sample, mu_at_sample, sigma_at_sample, advantage, lr_sigma=0.005) -> float` (pure; the clipped Δlogσ; carries `TODO(sigma-ewma)`).
  - `update_sigma(sample, mu_at_sample, sigma_at_sample, advantage, lr_sigma=0.005) -> float` (sets and returns σ = clip(σ_at_sample·exp(step), sigma_min, sigma_max)).
  - `update_mean(delta)` unchanged.
- Removed: `sigma_param`, `sigma_ctrl`, `sigma` property, `log_sigma_param`, `set_log_sigma_param`, `update_controlling_variance`, `update_parametric_variance`.

- [x] **Step 1: Write the failing tests — `tests/test_param_sigma.py`**

```python
import numpy as np
from persona.parameter import PersonaParam, reset_rng_stream


def test_sigma_drives_sample_and_realization():
    reset_rng_stream(0)
    p = PersonaParam(mean=0.0, sigma=0.5)
    x = p.sample_exploration()
    pop = p.sample_population(4000, center=x)
    assert abs(pop.mean() - x) < 0.05
    assert 0.4 < pop.std() < 0.6


def test_update_sigma_uses_sample_time_snapshot():
    p = PersonaParam(mean=9.9, sigma=1.0)          # current mean irrelevant
    assert np.isclose(p.sigma_step(sample=3.0, mu_at_sample=0.0,
                                   sigma_at_sample=1.0, advantage=1.0,
                                   lr_sigma=0.01), 0.08)   # 0.01*(9-1)
    new = p.update_sigma(sample=3.0, mu_at_sample=0.0, sigma_at_sample=1.0,
                         advantage=1.0, lr_sigma=0.01)
    assert np.isclose(new, np.exp(0.08)) and np.isclose(p.sigma, new)


def test_sigma_step_caps_and_bounds():
    p = PersonaParam(mean=0.0, sigma=1.0, sigma_max=2.0)
    assert np.isclose(p.sigma_step(sample=100.0, mu_at_sample=0.0,
                                   sigma_at_sample=1.0, advantage=1.0,
                                   lr_sigma=1.0), 0.25)
    assert np.isclose(p.sigma_step(sample=0.0, mu_at_sample=0.0,
                                   sigma_at_sample=1.0, advantage=1.0,
                                   lr_sigma=1.0), -0.25)
    p.sigma = 1.9
    assert p.update_sigma(sample=1e6, mu_at_sample=0.0, sigma_at_sample=1.9,
                          advantage=1.0, lr_sigma=1.0) == 2.0


def test_sigma_max_default_keeps_large_initial_sigma():
    p = PersonaParam(mean=-65.0, sigma=5.0)
    assert p.sigma == 5.0 and p.sigma_max == 5.0
    q = PersonaParam(mean=1.0, sigma=0.005)
    assert q.sigma_max == 2.0                                    # default floor


def test_bounds_still_clip():
    p = PersonaParam(mean=0.0, sigma=1.0, min=0.0, max=2.0)
    p.update_mean(+10.0)
    assert p.mean == 2.0
```

- [x] **Step 2: Run to verify failure**

Run the Global-Constraints pytest command on `tests/test_param_sigma.py -v`.
Expected: FAIL (`sigma_step` / `update_sigma` missing, `sigma_max` not accepted).

- [x] **Step 3: Implement `persona/parameter.py`**

Dataclass fields per the Interfaces block. `sample_population` draws
`rng.normal(center, max(1e-8, sigma), n)` and clips; `sample_exploration` draws
`rng.normal(mean, max(1e-6, sigma))` and clips. `sigma_step`:
`z2 = ((sample − mu_at_sample)/max(sigma_at_sample, 1e-12))²`;
`return float(np.clip(lr_sigma*(z2−1.0)*advantage, −0.25, 0.25))` with a
`TODO(sigma-ewma)` comment (variant B). `update_sigma`:
`new = clip(sigma_at_sample·exp(self.sigma_step(...)), self.sigma_min, self.sigma_max); self.sigma = new; return new`.

- [x] **Step 4: Sweep call sites**

- `persona/population.py`: `sigmas = np.array([self.params[k].sigma ...])`; docstring “the single σ governs dispersion”.
- `persona/agent.py` (minimal port, keeps the windowed strategy working):
  `sigma_ctrl_dict = {k: p.sigma ...}`; in the update loop capture
  `mu_before = param.mean` and `sig_before = param.sigma` **before** `update_mean`,
  then call
  `param.update_sigma(sample=action_val, mu_at_sample=mu_before, sigma_at_sample=sig_before, advantage=adv_scalar, lr_sigma=self.lr_sigma_ctrl)`.
- `t2xcpg/leg.py`: `_param(name, mean, sigma, min_val=-100.0, max_val=100.0)`;
  `_stdp_params` passes `sigma=max(abs(mean)*0.1, 1e-6)`;
  `_izhi` passes a 0.005 / b 0.04 / c 5.0 / d 2.0;
  static weights pass `sigma=abs(weight)*0.1`. Drop `sigma_param=`.
- `t2xcpg/bs.py`: the three `PersonaParam(...)` get `sigma=5.0` (drop both old fields).
- `tests/test_persona_objective.py`, `tests/test_phase.py`: construct with `sigma=...`;
  the two “finite and positive” assertions become `assert np.isfinite(p.sigma) and p.sigma > 0.0`.
- `tests/test_review_fixes.py` (B5 test): `PersonaParam(mean=0.0, sigma=1e-3)`;
  loop `x = p.sample_exploration(); p.update_sigma(sample=x, mu_at_sample=0.0,
  sigma_at_sample=p.sigma, advantage=0.5, lr_sigma=0.005)`; keep
  `assert 1e-4 <= p.sigma < 0.05` (no blow-up).

- [x] **Step 5: Run the suite**

Run: Global-Constraints pytest command on
`tests/test_param_sigma.py tests/test_persona_objective.py tests/test_phase.py tests/test_review_fixes.py -v`.
Expected: PASS. Then run the full suite; only failures that require later tasks
(noise/strategy/agent rewrites) are acceptable if they are *import-time* free —
prefer fixing ctor-level fallout now.

- [x] **Step 6: Commit**

```bash
git add persona/parameter.py persona/population.py persona/agent.py t2xcpg/leg.py t2xcpg/bs.py tests/test_param_sigma.py tests/test_persona_objective.py tests/test_phase.py tests/test_review_fixes.py
git commit -m "feat(persona): single sigma per site with sample-time snapshot update"
```

---

### Task 2: Noise recorders — per site, per producer; STDP delta Σ(Δw)/N; producer wiring

**Files:**
- Rewrite: `persona/noise_recorder.py`
- Modify: `persona/registry.py`, `persona/agent.py` (read path only)
- Modify: `t2xcpg/leg.py`, `t2xcpg/bs.py` (producer relays via a new `create_relay`)
- Create: `tests/test_noise_recorder.py`
- Modify: `tests/test_noise.py`, `tests/test_stdp.py`
- Test: `tests/test_registry_recorders.py` (new)

**Interfaces:**
- Produces (recorder):
  - `NoiseRecorder(name, beta=0.995)`; `record_sample(t, sample, sigma)`; `record_delta(t, delta)`;
    `latest_sample() -> tuple[float,float] | None`; `take_delta() -> float | None` (read-and-clear);
    `stats() -> {"mean","var","std","latest","t","n"}` (β-EWMA).
- Produces (registry):
  - `site_recorder(full_key) -> NoiseRecorder` (auto-created at `create_population`/`create_connection` for every
    `pop_{name}_{param}` / `conn_{key}_{param}`);
  - `producer_recorder(name) -> NoiseRecorder` (get-or-create; names `stdp:{conn_key}`, `poisson:{group}`);
  - `create_relay(pg_key, in_key, n, *, rate=0.0, rate_param=None, noise_name=None) -> (pg, parrot)` — replaces raw
    poisson+parrot creates; registers `population_nodes[pg_key]`/`[in_key]`; `rate_param` registers a `Population`
    with `params={"rate": rate_param}`;
  - `set_relay_rate(pg_key, rate)` — sets the rate, records the delta into `poisson:{noise_name}` when it exists;
  - `sample_plastic_weight_deltas(t_before, t_now)` — δ = Σ(Δw)/N_syn per projection (fresh implementation),
    recorded into `stdp:{key}` producer recorders; keep the per-step cache and `last_plastic_counts()`.
  - module-level `plastic_delta_from_events(events, last_w, seed_weight, n_syn) -> (delta, updated_last_w)` (pure, testable).

- [x] **Step 1: Write failing tests**

`tests/test_noise_recorder.py`:
```python
import numpy as np
from persona.noise_recorder import NoiseRecorder

def test_take_delta_is_fresh():
    nr = NoiseRecorder("producer:stdp:x")
    assert nr.take_delta() is None
    nr.record_delta(0.0, 0.5)
    assert nr.take_delta() == 0.5
    assert nr.take_delta() is None                    # cleared

def test_latest_sample_persists():
    nr = NoiseRecorder("site:pop_p_a")
    assert nr.latest_sample() is None
    nr.record_sample(3, 1.5, 0.2)
    assert nr.latest_sample() == (1.5, 0.2)
    assert nr.latest_sample() == (1.5, 0.2)           # never cleared

def test_stats_converge():
    nr = NoiseRecorder("producer:poisson:cut_L", beta=0.98)
    rng = np.random.default_rng(0)
    vals = rng.normal(2.0, 0.5, 4000)
    for i, v in enumerate(vals):
        nr.record_delta(float(i), float(v))
    s = nr.stats()
    assert abs(s["mean"] - 2.0) < 0.1 and 0.1 < s["std"] < 1.0
```

`tests/test_registry_recorders.py` (needs the `nest_kernel` fixture from `conftest.py`):
```python
import numpy as np
from persona.registry import Registry
from t2xcpg.leg import Leg

def test_site_recorder_per_site_unique_across_legs(nest_kernel):
    reg = Registry()
    Leg("L", reg, n_rg=3, n_mn=3, n_int=3, n_aff=3)
    Leg("R", reg, n_rg=3, n_mn=3, n_int=3, n_aff=3)
    params = reg.get_all_optimizable_params()
    for key in params:
        rec = reg.site_recorder(key)
        assert rec.name == f"site:{key}"
        assert rec not in [reg.site_recorder(k) for k in params if k != key]

def test_stdp_producer_delta_from_events():
    from persona.registry import plastic_delta_from_events
    events = {"senders": np.array([1, 1, 2]), "targets": np.array([5, 5, 6]),
              "weights": np.array([11.0, 12.0, 9.0])}
    last = {(1, 5): 10.5}                     # synapse (1,5) seen before; (2,6) first sight
    delta, last2 = plastic_delta_from_events(events, last, seed_weight=10.0, n_syn=4)
    # (11-10.5) + (12-11) + (9-10 seed)
    assert np.isclose(delta, (0.5 + 1.0 - 1.0)/4)
    assert last2[(1, 5)] == 12.0 and last2[(2, 6)] == 9.0
```

`tests/test_stdp.py` / `tests/test_noise.py`: replace `leg.noise_recorder` reads with
`reg.producer_recorder(f"stdp:{key}").take_delta()` and drop the old tag-style asserts.

- [x] **Step 2: Run to verify failure** (all three files).

- [x] **Step 3: Implement**

- `persona/noise_recorder.py`: state `_m`, `_q`, `_n`, `_last`, `_latest_sample`, `_pending`;
  `record_*` update the β-EWMA (`m ← βm+(1−β)v; q ← βq+(1−β)v²`; first value seeds m=q=v) and set `_pending`
  (`record_sample` sets `_latest_sample`); `stats()` returns `mean=m`, `var=max(q−m²,0)`, `std=sqrt(var)`, `latest`, `t`, `n`.
- `persona/registry.py`: call `site_recorder` creation in `create_population`/`create_connection`;
  `producer_recorder` in `create_connection` when `cfg.weight_recorder is not None`
  (`stdp:{conn_key}`) and in `create_relay(noise_name=...)`;
  implement `create_relay`/`set_relay_rate` (get-or-create recorder; first call records nothing);
  rewrite `sample_plastic_weight_deltas` with `_plastic_last_w` keyed `(key, sender, target)` and
  `_plastic_n_syn` filled by `finalize_connections` (`len(synapses(key))`, fallback `1`),
  recording each δ into `producer_recorder(f"stdp:{key}")`; extract `plastic_delta_from_events`.
- `t2xcpg/leg.py`: replace the raw `nest.Create`/`nest.Connect` of the five sensory pg/parrot pairs with
  `registry.create_relay(f"{grp}_pg_{leg}", f"{grp}_in_{leg}", n_aff, rate=..., noise_name=f"{grp}_{leg}")`;
  `update_sensory_feedback` calls `registry.set_relay_rate` (keep `leg.py`'s own diff out — `set_relay_rate` owns it);
  delete `self.noise_recorder` / `_connect(noise_recorder_id=...)` usage; `update_plastic_weights` returns deltas only.
- `t2xcpg/bs.py`: BS and BaseDrive raw creates → `create_relay("bs_pg_e","BS_in_e",n_bs,rate_param=self.p_rate_e,pop_name="BS_E")`,
  …_f, and `create_relay("base_pg","BASE_in",n_bs,rate_param=self.p_rate,pop_name="BASE_pg")`.
- `persona/agent.py` read path: instead of iterating `registry.noise_recorders` for tags, build noise values from
  production recorders: for each `name` in the strategy's noise names, `delta = rec.take_delta() or 0.0`
  (τ stays the old empirical path until Task 3), key them `f"noise_{name}"` for the old windowed strategy.

- [x] **Step 4: Run to verify pass** — the new tests + `tests/test_stdp.py tests/test_noise.py tests/test_cpg_build.py`.

- [x] **Step 5: Commit** — `git commit -m "feat(persona): per-site/per-producer noise recorders, STDP delta = sum(dw)/N, registry relays"`

---

### Task 3: Persisted two-type strategy + Agent rewrite

**Files:**
- Rewrite: `persona/strategy.py`
- Rewrite: `persona/agent.py`
- Modify: `persona/h5io.py`, `persona/validation.py` (single `sigmas` group)
- Create: `tests/test_strategy.py`, `tests/test_agent_strategy.py`
- Delete/supersede: `tests/test_variantb_metric.py`, `tests/test_variantb_agent.py`, `tests/test_fisher.py`

**Interfaces:**
- `FisherMatrix(param_names, noise_names=None, lr_mu=0.02, lr_sigma=0.005, lambda_reg=1e-3, beta=0.995, warmup=3, baseline_alpha=0.90)`
  - `ensure_names(param_names, noise_names=None) -> None` (zero-pads F/X̄/G, keeps order);
  - `compute_advantage(loss_vector, phase) -> np.ndarray` (unchanged semantics);
  - `step(sites: dict[str, tuple[float,float,float]], noise: dict[str, tuple[float,float]], advantage: np.ndarray) -> dict[str, float]`
    with `sites[key] = (Δθ, σ, Δlogσ)`, `noise[name] = (δ, τ)`;
  - `influence() -> np.ndarray` (normalised |G| row norms over the `2·n_sites` site rows);
  - attributes `F`, `Xbar`, `G`; `feature_scales() = sqrt(diag(F))`.
- `Agent(registry, strategy=None, lr_mu=0.02, lr_sigma=0.005, tau_mem_s=2.0, dt_ms=10.0, warmup=3, smooth_kernel_size=5)`;
  `refresh_params()`; `sample_actions(t_ms=None)`; `step(output, target, phase, t_ms=None)`;
  `history` groups `means`, `sigmas`, `influence`; snapshot storage `self._snap[key] = (mu_at_sample, sigma_at_sample, xhat)`.
- h5 `agent/` groups: `means`, `sigmas` (replaces `sigmas_ctrl`/`sigmas_param`); add final `trace_final`, `credit_final` in the same task.

- [x] **Step 1: Write failing tests — `tests/test_strategy.py`**

```python
import numpy as np
from persona.strategy import FisherMatrix


def _ref_run(steps, names, noise_names, beta, warmup, lam, lr_mu, lr_sig=0.0):
    n = 2*len(names) + len(noise_names)
    F = np.zeros((n, n)); Xb = np.zeros(n); G = np.zeros((n, 2))
    for (dth, sig, dlog, nd, A) in steps:
        nz = len(names)
        s = np.zeros(n); a = np.zeros(n)
        for i, (k, (d, sg, dl)) in enumerate(zip(names, zip(dth, sig, dlog))):
            s[i] = d/sg**2; s[nz+i] = (d/sg)**2 - 1
            a[i] = d/sg**2; a[nz+i] = dl
        for j, k in enumerate(noise_names):
            d, t = nd[j]
            s[2*nz+j] = d/t**2
        F = beta*F + (1-beta)*np.outer(s, s)
        Xb = beta*Xb + (1-beta)*a
        G = beta*G + (1-beta)*np.outer(Xb, A)
    return F, Xb, G


def test_recursions_match_reference():
    rng = np.random.default_rng(0)
    names = ["p0", "p1"]; noise = ["n0"]
    fm = FisherMatrix(param_names=names, noise_names=noise, beta=0.9,
                      warmup=0, lr_mu=0.1)
    site_list, noise_list, adv_list = [], [], []
    for t in range(50):
        d = rng.normal(0, 1, 2)*0.1; sg = rng.uniform(0.1, 1.0, 2)
        dl = rng.normal(0, 0.01, 2); nd = [(rng.normal(0, 0.1), 0.5)]
        A = rng.normal(0, 1, 2)
        fm.step({k: (float(d[i]), float(sg[i]), float(dl[i])) for i, k in enumerate(names)},
                {noise[0]: nd[0]}, A)
        site_list.append((d.tolist(), sg.tolist(), dl.tolist()))
        noise_list.append([nd[0]]); adv_list.append(A.tolist())
    F_ref, X_ref, G_ref = _ref_run(
        list(zip([s[0] for s in site_list], [s[1] for s in site_list],
                 [s[2] for s in site_list], noise_list, adv_list)),
        names, noise, 0.9, 0, 1e-3, 0.1)
    assert np.allclose(fm.F, F_ref, atol=1e-10)
    assert np.allclose(fm.Xbar, X_ref, atol=1e-10)
    assert np.allclose(fm.G, G_ref, atol=1e-10)


def test_delayed_credit_ranks_coupled_site():
    rng = np.random.default_rng(1)
    names = ["coupled", "inert"]
    fm = FisherMatrix(param_names=names, beta=0.95, warmup=0, lr_mu=0.1)
    k = 4                                    # plant: A_t <- sample_{t-k} of "coupled"
    hist = []
    for t in range(3000):
        d = np.array([rng.normal(0, 0.3), rng.normal(0, 0.3)])
        sg = np.array([0.3, 0.3])
        fm.step({"coupled": (d[0], sg[0], 0.0), "inert": (d[1], sg[1], 0.0)},
                {}, np.array([1.0, 1.0]) * hist[t-k][0] if t >= k else np.array([0.0, 0.0]))
        hist.append(d)
        steps = fm.step({"coupled": (d[0], sg[0], 0.0), "inert": (d[1], sg[1], 0.0)},
                        {}, np.array([1.0, 1.0]) * (hist[t-k][0] if t >= k else 0.0))
    deltas = fm.step({"coupled": (0.0, 0.3, 0.0), "inert": (0.0, 0.3, 0.0)}, {},
                     np.array([1.0, 1.0]))
    assert deltas["coupled"] > 0.05 and abs(deltas["inert"]) < 0.01


def test_no_sigma_square_gain_and_unit_invariance():
    rng = np.random.default_rng(2)
    def run(sa, sb):
        fm = FisherMatrix(param_names=["a", "b"], beta=0.9, warmup=0,
                          lambda_reg=1e-6)
        for t in range(400):
            d = np.array([rng.normal(0, sa), rng.normal(0, sb)])
            A = np.array([d[0]/sa, 0.0])          # equal physical coupling
            fm.step({"a": (d[0], sa, 0.0), "b": (d[1], sb, 0.0)},
                    {}, A)
        return fm.step({"a": (0.0, sa, 0.0), "b": (0.0, sb, 0.0)}, {},
                       np.array([1.0, 0.0]))
    d1 = run(0.01, 0.01); d2 = run(0.001, 0.01)
    assert abs(d2["a"]) > 0.5*abs(d1["a"])        # no 100x sigma^2 blow-up


def test_warmup_zero_and_noise_not_in_trace():
    fm = FisherMatrix(param_names=["p"], noise_names=["n"], warmup=3, beta=0.9)
    for t in range(3):
        out = fm.step({"p": (0.1, 0.1, 0.01)}, {"n": (0.5, 0.5)}, np.array([1.0, 1.0]))
        assert out["p"] == 0.0
    assert np.all(fm.Xbar[1] == 0.0)              # noise row never traced
    out = fm.step({"p": (0.1, 0.1, 0.01)}, {"n": (0.5, 0.5)}, np.array([1.0, 1.0]))
    assert abs(out["p"]) > 0.0
```
(Note for the implementer: the delayed-credit loop calls `step` once per t — the
duplicate in the sketch is a copy artifact; write a single call per step.)

`tests/test_agent_strategy.py`:
```python
import numpy as np
from persona.parameter import PersonaParam, reset_rng_stream
from persona.population import Population
from persona.registry import Registry
from persona.agent import Agent


def _agent(lr_mu=0.05):
    reset_rng_stream(0)
    reg = Registry()
    reg.create_population(Population("izhikevich", 5,
                          {"a": PersonaParam(mean=1.0, sigma=0.3, min=0.0, max=3.0)}, "pop"))
    return Agent(registry=reg, lr_mu=lr_mu, lr_sigma=0.0), reg


def test_loss_decreases_and_sigma_recorded(nest_kernel):
    agent, reg = _agent()
    key = "pop_pop_a"
    for t in range(120):
        agent.sample_actions(t_ms=t*10.0)
        actions = dict(agent.current_actions)
        reg.apply_explored_parameters(actions)
        rec = reg.site_recorder(key)
        assert rec.latest_sample() == (actions[key], agent.params[key].sigma)
        out = np.array([abs(actions[key] - 2.0)] * 2)
        agent.step(output=out, target=np.zeros(2), phase=["stance", "swing"],
                   t_ms=t*10.0)
    assert abs(agent.params[key].mean - 2.0) < 0.5
    assert len(agent.history["sigmas"][key]) == 120
```

- [x] **Step 2: Run to verify failure.**

- [x] **Step 3: Implement `persona/strategy.py`**

State: `param_names`, `noise_names`, `n_sites`, `n_noise`, `F`, `Xbar`, `G`,
`steps` counter, `warmup`, `beta`, `lambda_reg`, `lr_mu`, `lr_sigma`, baselines.
`step(sites, noise, advantage)`:
1. `ensure_names` with the given keys (noise too);
2. build `s`, `a` vectors (layout above; μ rows `Δθ/σ²`; σ rows `z²−1` / `Δlogσ`; noise `δ/τ²`, trace 0);
3. `F = βF+(1−β)s sᵀ`; `Xbar = βXbar+(1−β)a`; `G = βG+(1−β)·outer(Xbar, A)`;
4. `self.steps += 1`; if `self.steps <= self.warmup`: return `{k: 0.0}`;
5. `g = G.sum(axis=1)`, zero the noise rows; `dmu = lr_mu·solve(F+λI, g)` (pinv fallback);
   return `{name: float(dmu[i])}` for sites.

- [x] **Step 4: Implement `persona/agent.py`**

`beta = 1 − (dt_ms/1000)/tau_mem_s`. `sample_actions(t_ms)`: refresh params; for each site
`x̂ = p.sample_exploration()`; store `self._snap[key] = (p.mean, p.sigma, x̂)`;
record into `registry.site_recorder(key).record_sample(t, x̂, p.sigma)`; return the dict.
`step(...)`: smoothing/error/advantage as now; refresh params; site tuples from `_snap`
(Δθ = x̂−μ_at_sample, σ_at_sample); σ step per site `p.sigma_step(...)` with `adv = mean(A)`;
noise dict from `registry.producer_recorder(name)`: `(take_delta() or 0.0, max(stats()["std"],1e-6))`;
`deltas = strategy.step(sites, noise, A)`; apply `p.update_mean(deltas[key])` and
`p.update_sigma(sample=x̂, mu_at_sample=μ_at_sample, sigma_at_sample=σ_at_sample, advantage=adv, lr_sigma=self.lr_sigma)`;
history logs `means`, `sigmas`, `influence`; return `delta_means`, `dlog_sigma`.
Remove all legacy paths (`record_step`, `record_response`, `sigma_ctrl_dict`, `lr_sigma_ctrl`/`lr_sigma_param`).

- [x] **Step 5: Update `persona/h5io.py` + `persona/validation.py`**

Replace the `sigmas_ctrl`/`sigmas_param` groups with `sigmas`; store `agent/trace_final`,
`agent/credit_final` next to the final metric. `plot_param_shift` reads `sigmas`.

- [x] **Step 6: Run to verify pass**; update/delete the superseded test files listed above;
  run the full suite and fix fallout that belongs to this task.

- [x] **Step 7: Commit** — `git commit -m "feat(persona): persisted F+trace+credit strategy; agent update-every-dt with recorded samples"`

---

### Task 4: Registry — single NEST route, per-instance realization, surfaced failures

**Files:**
- Modify: `persona/registry.py`, `t2xcpg/leg.py`, `t2xcpg/bs.py`, `t2xcpg/utils.py`, `t2xcpg/cpg.py` (if it creates/connects)
- Create: `tests/test_nest_route.py`, `tests/test_apply_realization.py`

**Interfaces:**
- `register_nodes(name, nodes)`; `create_spike_recorder(target)`; `create_weight_recorder()`;
  `apply_explored_parameters(sampled_values)` → sets `self.last_apply_errors: list[tuple[str, str]]`
  (cleared per call); device-only populations (`poisson_generator`) realize per node in a loop
  (array first, loop fallback, error on failure); static synapses realize per synapse via
  `synapses(key).set({"weight": array})`; STDP connections (`weight_recorder is not None`) set scalars;
  `create_population`'s `set` uses the same array→loop→error ladder.
- `t2xcpg/utils.py` keeps only `nest.CopyModel`-based helpers (`copy_syn_model`, `create_stdp_synapse(registry, ...)`;
  `create_izhi_population`/`poisson_to_parrot`/`create_spike_recorder`/`create_weight_recorder` removed).

- [x] **Step 1: Write failing tests**

`tests/test_nest_route.py`:
```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_no_raw_nest_calls_outside_registry():
    bad = []
    for path in list((ROOT/"persona").glob("*.py")) + list((ROOT/"t2xcpg").glob("*.py")) \
            + [ROOT/"personalize.py", ROOT/"run_validation.py"]:
        if path.name == "registry.py" or path.name == "conftest.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"nest\.(Create|Connect)\(", line):
                bad.append(f"{path}:{i}: {line.strip()}")
    assert bad == [], "\n".join(bad)
```

`tests/test_apply_realization.py`: build a tiny Registry with one static Connection
(p=1.0) plus one Poisson Population; apply an action far from the population mean and assert
(a) per-synapse weights differ across synapses and have mean ≈ action; (b) each Poisson node's rate
is a draw around the action (not the mean of draws); (c) a monkeypatched failing `set` lands in
`registry.last_apply_errors` and the earlier entries stay.

- [x] **Step 2: Run to verify failure** (raw-call scan fails until utils/leg/bs are cleaned).

- [x] **Step 3: Implement** the registry/apply changes; rewrite `t2xcpg/leg.py` muscle/readout
  creation (`register_nodes` / `create_spike_recorder`), `t2xcpg/utils.py` cleanup,
  `t2xcpg/bs.py`/`t2xcpg/cpg.py` call sites.

- [x] **Step 4: Run to verify pass** + `tests/test_cpg_build.py tests/test_cpg_topology.py tests/test_stdp.py`.

- [ ] **Step 5: Commit** — `git commit -m "feat(persona): registry single NEST route, per-instance realization, apply errors surfaced"`

---

### Task 5: Loop, logging, CLI

**Files:**
- Modify: `persona/personalization.py`, `personalize.py`, `run_validation.py`
- Modify: `tests/test_personalization.py`

**Interfaces:**
- `Personalizer.run(..., realize_every_k: int = 1)`; every step calls
  `agent.step(..., t_ms=self.cpg.current_time_ms)`; sample+apply only when `step % realize_every_k == 0`;
  logs each new `registry.last_apply_errors` entry once.
- `personalize.py`: `Agent(..., lr_mu=0.03, lr_sigma=0.005, tau_mem_s=args.tau_mem_s, dt_ms=DT_MS)`.
- CLI: `--tau-mem-s` (default 2.0), `--realize-every-k` (default 1) on `personalize.py`/`run_validation.py`.

- [ ] **Step 1: Write failing tests** — `tests/test_personalization.py`: `realize_every_k=2` keeps the
  site recorder's sample stable across the odd step and the strategy step still runs (loss log length equals
  total steps); `t_ms` reaches the recorders.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(persona): realize-every-k loop gating, tau-mem CLI, apply-error logging"`

---

### Task 6: Integration smoke + full suite

**Files:**
- Create: `tests/test_persona_integration.py`
- Modify: any test still red from the API change

- [ ] **Step 1: Write the smoke test** — build the two-leg CPG tiny (n_rg=4, n_mn=4, n_int=4, n_aff=4) with
  `BrainStem` + `BaseDrive`; run 12 `agent.sample_actions/apply/step` iterations on synthetic outputs;
  assert: `strategy.F` finite and symmetric; `G` non-zero; every site/σ finite; σ within bounds;
  `registry.last_apply_errors == []`; every producer recorder's `stats()["n"] > 0`.
- [ ] **Step 2: Run the full suite** with the Global-Constraints command; fix remaining fallout.
- [ ] **Step 3: Run a fast end-to-end**:
  `wsl -e bash -lc "cd /mnt/d/CPGvalidation && OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa /home/t0kgw/.virtualenvs/KinematicsCPG/bin/python run_validation.py --fast --out /mnt/d/CPGvalidation/validation_mvp"`;
  verify `run.h5` carries `agent/means`, `agent/sigmas`, `agent/trace_final`, `agent/credit_final`
  and the plots are produced.
- [ ] **Step 4: Commit** — `git commit -m "test(persona): integration smoke; full-suite green after single-sigma rework"`

---

### Task 7: Fresh-eyes review

- [ ] **Step 1:** Dispatch `superpowers:requesting-code-review` on the branch diff
  (`git diff f611e78..HEAD` + the spec).
- [ ] **Step 2:** Apply fixes for every finding; re-run the full suite.
- [ ] **Step 3:** Commit — `git commit -m "fix(persona): review findings"`.

**Reviewer checklist (added during T3, from test calibration):**
- **Credit gain.** `G = βG + (1−β)X̄Aᵀ` has stationary mean `(1−β)·E[trace·A]`
  (pinned by `test_credit_gain_is_ewma_weighted`), while `F = E[s sᵀ]` is
  full-scale. So the effective step is `η·(1−β)βᵏ·Cov` at lag `k` — the trace
  weights a lag-`k` coupling by `(1−β)βᵏ`. Confirm on the T6 runs that the
  production `lr_mu` (0.03) still learns acceptably with `τ_mem = 2 s`
  (`(1−β) = 0.005`), or raise the effective-rate question with the user
  (do not silently rescale `g`).

## Self-Review

- Spec coverage: §5.1→T1; §5.2→T4; §5.3→T2; §5.4→T3; §5.5→T2+T4; §5.6→T3+T5; §8→spread; V22→T1;
  V23→T2; V24→T3; V25→T1+T3; V26→T2; V27→T4; V28→T5; V29→T3; V30→T3+T5.
- Type consistency: `site_recorder`/`producer_recorder` names, `(Δθ, σ, Δlogσ)` site tuples,
  `(δ, τ)` noise tuples, `latest_sample`/`take_delta` used identically in T2–T4.
- Review Focus lines each get a test: RF1→T3 (`test_no_sigma_square_gain_and_unit_invariance`, `test_sigma_max_default…`),
  RF2→T3+T5 (recorder equality, `realize_every_k` test), RF3→T4 (device loop), RF4→T2 (`test_take_delta_is_fresh`),
  RF5→T2 (`test_site_recorder_per_site_unique_across_legs`).
