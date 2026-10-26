# PERSONA strategy history, noise recorders, and trace sufficiency — design

**Status:** decisions approved 2026-10-26. This is an analysis + test +
optional-trace task. It does **not** change the strategy update, the noise
scoring, or any production default. The user may correct the recorders later.

**Spec index:** `docs/superpowers/specs/2026-10-26-persona-strategy-history-noise-design.md`

## 1. Intent

Two questions:

1. How does the current strategy history update work, and how are external and
   internal noise recorders written?
2. Do we need to save parameter traces, or are `F/Xbar/G` and the recursive
   Fisher merge enough?

The answer must be backed by tests with several noise-recorder
configurations, not only by reading the code. The approved conclusion is:

- parameter mean/sigma traces are **diagnostic**, not sufficient statistics;
- `F/Xbar/G` are the sufficient statistics for the strategy;
- exact replay additionally needs the sampling-time applied sample, sigma,
  advantage, and producer deltas;
- the current `merge` is a normalized ensemble average, not an exact
  continuation;
- exact strategy-state replay is available by saving a compact applied-sample
  trace or by checkpointing `F/Xbar/G`; full Agent resume additionally needs
  the phase/game/H/means/sigmas/RNG state.

## 2. Current implementation map

### 2.1 Per-step pipeline

`Agent.step` (`persona/agent.py:207-358`):

1. `current_step += 1`; read `lr_mu` (scheduled or fixed).
2. `refresh_params()` re-reads the registry sites and syncs every noise
   recorder's `beta` to the strategy's beta.
3. Smooth output and target with the same moving average; compute
   `error_vec = |target − output|` and `loss = mean(error_vec)`.
4. Normalize the phase key; compute the per-component advantage
   `A_i = baseline_i(phase_i) − loss_i`; the baseline is conditioned on the
   reference phase when supplied (`advantage_phase`).
5. Snapshot each site **before** sampling: `(mu_at, sigma_at)`; the sample
   `xhat` is the realized policy action. The strategy receives
   `sites[name] = (xhat − mu_at, sigma_at, dlogsigma_applied)`.
6. Read every `producer:` recorder with `take_delta()` (read-and-clear); pair
   it with the registration value `scale` as `(delta, R)`.
7. `strategy.step(...)` accumulates `F/Xbar/G`, refreshes the inverse, and
   returns the per-site `Δμ`.
8. Apply `Δμ` (plus spring), then the log-sigma update; append history.

### 2.2 Strategy recurrences

`FisherMatrix` (`persona/strategy.py:33-466`):

```
β      = 1 − dt/τ_mem
F      = β F      + (1−β) s sᵀ          # score Fisher
Xbar   = β Xbar   + (1−β) a              # applied-move trace
G      = β G      + (1−β) Xbar Aᵀ        # delayed credit (Xbar after update)
g      = G·1₂,  g[noise rows] = 0
Δμ     = lr_mu · P g,  P ≈ (F + λ I)⁻¹
```

Channel layout: `n = 2·|sites| + |producers|`; site μ rows, site σ rows,
producer rows. Producer rows have no trace rows (`a = 0`) and never receive
credit. `P` is refreshed exactly on a phase shift or every `refresh_every`
steps; between refreshes it is updated by Sherman–Morrison. `ensure_names`
grows `F/Xbar/G` when a site or producer appears, preserving the old block.

### 2.3 What is actually saved

- `Agent.history` (`persona/agent.py:57-65, 341-348`): `error_vector`, `loss`,
  `advantage_vector`, `phase`, `influence`, `means`, `sigmas`; optional
  `homeostasis`, `game`, `spring`, `lr_mu`.
- `RunLog` (`persona/personalization.py:72-216`): per-step angles, loss,
  advantage, phases, drives, afferents, contacts, foot positions, `stdp`
  trace. `Personalizer.run` fills it.
- `run.h5` (`persona/h5io.py:21-82`): `log/*`, `stdp/*`, `agent/means`,
  `agent/sigmas`, `agent/influence`, `agent/loss`, `agent/advantage_vector`,
  `agent/error_vector`, and the **final** `fisher`, `feature_scales`,
  `trace_final` (`Xbar`), `credit_final` (`G`).
- There are **no per-step snapshots** of `F/Xbar/G`, no applied `xhat`, no
  sampling-time `mu_at`/`sigma_at`, no producer deltas in `run.h5`.

## 3. External and internal noise recorders

### 3.1 Definitions

- **External noise** — the 10 Poisson sensory relays:
  `producer:poisson:cut_{L,R}`, `ia_e_{L,R}`, `ia_f_{L,R}`, `ib_e_{L,R}`,
  `ib_f_{L,R}`. They are created by `create_relay(..., noise_name=...,
  noise_scale=...)` (`persona/nest_adapter.py:408-473`) and written by
  `set_relay_rate` as the per-step **rate change** `r − prev`. The stochastic
  Poisson spikes themselves are not recorded; the recorder sees the commanded
  rate delta.
- **Internal noise** — the 6 STDP projections (`ia_rge_*`, `ia_rgf_*`,
  `cut_rge_*`). A recorder is created at connection registration with
  `scale = Wmax.mean` (`nest_adapter.py:399-405`). `sample_plastic_weight_deltas`
  drains the NEST weight recorder, computes the per-synapse mean weight
  increment (`plastic_delta_from_events`), caches it per `t_now`, and calls
  `record_delta` exactly once per projection (`nest_adapter.py:563-593`).
- **Not noise producers** — the BS/BASE rates (`pop_BS_E_rate`,
  `pop_BS_F_rate`, `pop_BASE_pg_rate`) and all site recorders. BS/BASE are
  optimizable populations with site recorders; site sigma is PERSONA
  exploration/dispersion noise, not a strategy producer.
- The CPG calls `sample_plastic_weight_deltas` from `L2CPG.readout` for L and
  R; the per-`t_now` cache prevents a double record.

### 3.2 Recorder object

`NoiseRecorder` (`persona/noise_recorder.py`):

- site recorder: `record_sample(t, sample, sigma)` stores the last applied
  `(sample, sigma)` and never clears it; `take_delta` is not used for sites;
- producer recorder: `record_delta(t, delta)` sets `_pending` and updates the
  β-EWMA statistics; `take_delta()` returns and clears `_pending`;
- `scale` is the producer's value at registration (`R`); `None` or `<= 0`
  means no influence;
- `stats()` returns the EWMA mean/variance, latest value, time, and count; it
  is diagnostic and never used for scoring;
- `Agent.refresh_params` forces every recorder's `beta` to the strategy beta.

The strategy scores a producer as `s = δ/R²` only when `R > 0`; otherwise the
channel is zero but still occupies an `F` row.

### 3.3 Known issues to test (not fix in this task)

1. `record_delta` **overwrites** `_pending`; two writes between `take_delta`
   calls lose the first delta. Current call paths write once per producer per
   step, but this is not enforced.
2. `scale` is fixed at registration; `Wmax` is itself an optimizable site and
   can move, so the score can use a stale reference.
3. `δ/R²` treats `R` as a noise scale. For STDP, `R = Wmax` (a weight ceiling);
   for Poisson, `R` is a rate in Hz. The dimensional meaning should be checked
   by the user; the task records it, it does not change it.
4. A site recorder stores the **policy action** (`sampled_values[full_key]`),
   not the realized per-neuron/per-synapse dispersion. "recorded = applied"
   holds at the action level, not the realized value level.
5. Zero-synapse projections still have sites and a `size=0` segment; their
   recorders exist but no delta is produced.
6. Poisson deltas are rate commands, not spike-count noise; the actual
   stochasticity is in NEST.

## 4. Trace sufficiency: analysis and probe

### 4.1 What "parameter trace" means

- **P-trace** — `agent.history["means"]` / `["sigmas"]`: post-step parameter
  means and sigmas. This is what `run.h5` currently saves.
- **S-trace** — `F/Xbar/G` plus `steps` and `noise_scales`: the strategy's
  sufficient statistics.
- **A-trace** — per-step applied sampling-time state:
  `(xhat, mu_at, sigma_at, dlogsigma, A, noise_delta, noise_scale)`. This is
  what exact replay needs.

### 4.2 Synthetic probe (already run)

One synthetic stream, 6 sites, 2 producers, 120 steps, β=0.95:

| rebuild source | F error | Xbar error | G error | `P g` error |
|---|---:|---:|---:|---:|
| exact A-trace | 0 | 0 | 0 | 0 |
| P-trace (post-step means/sigmas) | 0.993 | 0.992 | 0.999 | 2.20 |
| drop `Xbar` (instantaneous `a`) | 0 | 2.09 | 3.38 | 1607 |

The P-trace is not recoverable: the exploration draw is not in the mean path.
`Xbar` is not optional: `G` and the natural-gradient direction depend on it.

### 4.3 Merge

For one sequential stream, with segments `i` ending at time `e_i`:

```
F_T   = Σ_i β^(T−e_i) F_i           (exact, if each F_i started at zero)
Xbar_T= Σ_i β^(T−e_i) Xbar_i        (exact)
G_T   = Σ_i β^(T−e_i) G_i + cross terms between earlier Xbar and later A
```

The cross terms are the delayed-credit coupling. A 3-way split of the synthetic
stream gives 35% relative error in `G` when only per-segment `G_i` is merged;
`F` and `Xbar` recover to machine precision with decay-aware weights.

The current `FisherMatrix.merge` normalizes the weights, so it is a **weighted
ensemble average**, not the decay-aware sequential sum. That is a valid
definition for independent replicas, but it must not be described as exact
continuation. `step_batch` (average the sufficient statistics of several
observations, then one update) is a third definition; the existing test prints
a 49% difference between `merge` and `step_batch`. The design records this;
it does not choose a winner.

### 4.4 Approved parameter-trace update

The update is **additive and opt-in**:

1. Keep `agent/history["means"/"sigmas"]` exactly as they are (diagnostics).
2. Add `Agent(trace_strategy=False)`. When true, append one record per strategy
   step in **float64**:
   ```
   dtheta = xhat - mu_at, sigma_at, dlogsigma, advantage,
   noise_delta, noise_scale
   ```
   Site and producer order is `strategy.param_names` / `strategy.noise_names`.
   Storing `dtheta` directly (rather than `xhat` and `mu_at`) avoids losing
   precision to subtraction at replay time. A float64 trace is required for an
   exact rebuild; `--trace-float32` is an explicit approximate option with a
   documented tolerance (`1e-6` relative on `F/Xbar/G`).
3. Trace mode requires a **fixed site set**. If `refresh_params` discovers a
   new site after tracing has started, raise `RuntimeError` with the new key;
   do not silently grow the trace. Producers may still grow (a zero scale and a
   zero delta give a zero score); a producer growth event is recorded in the
   trace metadata. The current fast-CPG site set is fixed after construction,
   so this is not a limitation for the intended run.
4. Add `FisherMatrix.snapshot() -> dict` and
   `FisherMatrix.from_snapshot(data) -> FisherMatrix`. A snapshot is a
   **strategy checkpoint**, not a full Agent restart point. It is self-contained
   for the strategy:
   ```
   F, Xbar, G, steps, _steps_since_refresh, P,
   beta, lambda_reg, warmup, refresh_every, lr_mu, dt_s,
   param_names, noise_names, noise_scales
   ```
   Restoring it restores the exact strategy state, including the lazy inverse;
   it does not restore `PhaseBaselines`, `_prev_phase`, means/sigmas, game
   stacks, H state, spring state, or RNG streams. Full-Agent resume is a
   separate feature.
5. `--snapshot-strategy N` takes a snapshot every N strategy steps, buffers it
   in `agent.history["strategy_checkpoints"]`, and `save_run` writes
   `strategy_checkpoints/<step>` at the final save. It does **not** force a
   refresh. Default off; the size estimate is printed before the run (a
   full-size `F` is ~10 MB, so N should be ≥ 500 in production). Incremental
   crash-safe checkpointing is out of scope in v1.
6. Add `rebuild_strategy_from_trace(trace, ...) -> FisherMatrix` in
   `persona/strategy_trace.py`; it replays `_accumulate`/`_score_vectors` in
   the recorded site/noise order and must reproduce `F/Xbar/G` exactly with
   float64. The trace metadata stores `beta`, `lambda_reg`, `warmup`,
   `refresh_every`, and the names.
7. `run.h5` gains `agent/strategy_trace/*` only when `trace_strategy=True`;
   `run_validation.py`/`run_2min.py` pass the flags through. `personalize.py`
   keeps the trace in `agent.history`; when `--trace-strategy` is set and
   `--out` is given, it also writes `strategy_trace.npz` (HDF5 remains the
   validation path). `run_personalization` does not start writing `run.h5`.
8. CLI: `--trace-strategy`, `--trace-float32`, and `--snapshot-strategy N`
   live in `persona/cli.py`, `build_personalizer`, and the validation runner.

This is the approved "parameter traces update": the saved traces become
diagnostic plus an optional **applied-sample** trace that is actually
sufficient for replay. No default run changes size.

## 5. Noise-recorder configuration tests

Add `tests/test_noise_configurations.py` (simulator-free where possible, plus
one NEST case) and `bench_strategy_history.py`.

### 5.1 Unit matrix

| configuration | expectation |
|---|---|
| no producers | `strategy.noise_names == []`; `F` is `2n×2n`; Agent runs finite |
| all 16 producers | all channels present; each `noise_scales[name] > 0`; `F` is `(2n+16)²` |
| Poisson only | 10 channels; STDP absent; no `producer:stdp:*` |
| STDP only | 6 channels; Poisson absent |
| one relay at a time | only that channel has a nonzero score; others zero |
| missing scale | channel present, score zero, `stats()["n"] > 0` |
| zero/negative scale | no influence; no division by zero |
| stale delta | `record_delta` once, `take_delta` twice → second `None` |
| duplicate delta | two `record_delta` before one take → last wins (documented) |
| producer added mid-run | `ensure_names` grows `F/Xbar/G`; old block preserved |
| site added mid-run, trace on | `RuntimeError` naming the new site; no silent growth |
| site added mid-run, trace off | `ensure_names` grows `F/Xbar/G` as today |
| beta sync | `refresh_params` sets every recorder beta to the strategy beta |
| zero-synapse projection | recorder exists, no delta, no NaN |
| recorder memory | `stats()` is constant size (existing test) |
| site recorder | last `(sample, sigma)` preserved; never a producer |

### 5.2 Agent-level tests

- With no producers, `Agent.step` produces the same `F/Xbar/G` as a strategy
  constructed with no noise names.
- With all producers, the noise rows of `F` are nonzero and the noise rows of
  `G` are exactly zero (no credit).
- A producer delta is consumed exactly once: step 1 sees it, step 2 sees 0.
- A producer added after construction is scored from the step it appears.
- `trace_strategy=True` produces a trace from which `rebuild_strategy_from_trace`
  reproduces `F/Xbar/G` exactly over 50 synthetic steps.

### 5.3 A/B benchmark

`bench_strategy_history.py`:

```
python bench_strategy_history.py --quick
python bench_strategy_history.py --noise-config all|none|poisson|stdp|one-relay
python bench_strategy_history.py --fast-cpg --steps 400 --seeds 0 1
```

Synthetic mode writes `benchmarks/strategy_history/synthetic_*`. Fast-CPG mode
runs through `build_personalizer` and writes `benchmarks/strategy_history/cpg_*`.
Report per configuration: loss, noise-channel `F` norms, `G` noise rows (must
be zero), recorder counts, wall time, and whether the trace rebuild is exact.
The fast-CPG mode is the real check that external and internal recorders are
written on the production path.

The noise configuration is a **construction-time filter**, not a silencing
mask, so `F` really has 0/1/6/10/16 producer rows:

```
noise_config = all | none | poisson | stdp | one-relay:<name>
```

`Agent(noise_config=...)` and `build_personalizer(..., noise_config=...)`
default to `all` (exact current behavior). The filter is applied when the
strategy is constructed; recorders still exist in the registry for diagnostics,
but names outside the filter are not added to `strategy.noise_names` and are
not read. This is a test/benchmark knob, not a production default.

"Fast CPG" means `bench_homeostasis.FAST_SIZES` (6/5/5/3/5), `game=None`,
`homeostasis=None`, `eps_pose=0.10`, `pose_hold_steps=3`, `loss_stop=-1.0`,
400 steps. Two seeds per configuration; if the full matrix exceeds ~30 minutes,
drop to one seed and say so. Synthetic mode is the primary evidence; fast-CPG
mode is the required production-path confirmation.

## 6. Findings to record

The current findings are appended to `docs/backlog.md` under
"Strategy history / noise findings (2026-10-26)". The report repeats them with
the measured numbers and the real-CPG confirmation.

## 7. Non-goals

- No change to `F`, `Xbar`, `G`, the sigma rule, advantage, credit, the
  recursive inverse, or the noise scoring formula.
- No change to the default `merge` semantics; only documentation and tests.
- No fix for stale `scale`, overwritten deltas, or `δ/R²` semantics in this
  task; they are recorded for the user's later correction.
- No new dependency.

## 8. Risks and open questions

- Memory: `trace_strategy` is `O(T·(4S + 2 + 2N))`; for 554 sites and 12,000
  steps this is large. Default off; float64 by default for exact replay;
  `--trace-float32` is explicit and approximate; print an estimate; HDF5 only.
- Exact rebuild needs the same `beta`, `lambda`, `refresh_every`, `warmup`,
  and the same site/noise order; the trace stores them in metadata.
- The real-CPG trace test may reveal that `sample` and `mu_at` are not aligned
  with the realized action for populations; if so, report it, do not silently
  redefine "recorded = applied".
- `share_floor`/game and Evo are separate; this task does not touch them.

## 9. Implementation order

1. Record findings in the backlog (done).
2. `NoiseRecorder`/producer configuration tests and any small test-only
   helpers.
3. `Agent` no-producer/all-producer parity tests.
4. `FisherMatrix.snapshot/from_snapshot` + exact rebuild from A-trace.
5. `trace_strategy` history + HDF5 + CLI.
6. `bench_strategy_history.py` synthetic + fast-CPG A/B.
7. Report with measured numbers and recommendations.
