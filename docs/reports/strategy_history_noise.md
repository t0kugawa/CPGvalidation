# Strategy history, noise recorders, and trace sufficiency — report

**Date:** 2026-10-26
**Spec:** `docs/superpowers/specs/2026-10-26-persona-strategy-history-noise-design.md`
**Plan:** `docs/superpowers/plans/2026-10-26-persona-strategy-history-noise.md`

## 1. What the strategy stores

`Agent.step` (`persona/agent.py`) computes the sampling-time policy state and
the per-component advantage, then calls `FisherMatrix.step`
(`persona/strategy.py`). The persisted recurrences are

```
F    = β F    + (1−β) s sᵀ
Xbar = β Xbar + (1−β) a
G    = β G    + (1−β) Xbar Aᵀ
g    = G·1₂, noise rows zero
Δμ   = lr_mu · P g
```

with `β = 1 − dt/τ_mem`, `P ≈ (F + λI)⁻¹` refreshed exactly on a phase shift
or every `refresh_every` steps and updated by Sherman–Morrison between
refreshes. The layout is `[site μ; site σ; producer]`; producer rows have no
trace rows and never receive credit.

`run.h5` stores the per-step log, the STDP trace, the parameter
mean/sigma histories, the influence/loss/advantage/error histories, and the
**final** `fisher`, `trace_final` (`Xbar`) and `credit_final` (`G`). It did not
store per-step strategy snapshots or the applied sample.

## 2. Noise recorders

- **External** — 10 Poisson sensory relays: `producer:poisson:cut_{L,R}`,
  `ia_e_{L,R}`, `ia_f_{L,R}`, `ib_e_{L,R}`, `ib_f_{L,R}`. `create_relay`
  registers the producer with `scale` (nominal rate, or the explicit CUT
  value). `set_relay_rate` writes `record_delta(now, r − prev)`.
- **Internal** — 6 STDP projections (`ia_rge_*`, `ia_rgf_*`, `cut_rge_*`).
  Registration sets `scale = Wmax.mean`. `sample_plastic_weight_deltas` drains
  the NEST weight recorder, computes the per-synapse mean increment, caches per
  `t_now`, and records one delta per projection.
- **Not producers** — BS/BASE rates (`pop_BS_E_rate`, `pop_BS_F_rate`,
  `pop_BASE_pg_rate`) and every site recorder. Site sigma is PERSONA
  exploration/dispersion, not a strategy producer.
- `Agent` reads producers with `take_delta()` (read-and-clear) and scores
  `s = δ/R²` only when `R > 0`.

## 3. Is a parameter trace enough?

The synthetic probe (`bench_strategy_history.py`, 120 steps, 6 sites,
2 producers, β=0.95) compares four reconstructions:

| reconstruction source | F error | Xbar error | G error |
|---|---:|---:|---:|
| exact applied-sample trace (A-trace) | 0 | 0 | 0 |
| post-step parameter mean/sigma trace (P-trace) | ~1.00 | ~1.00 | ~1.00 |
| no trace (`Xbar := a` instantaneous) | — | ~4.24 | — |
| current weighted merge (3 segments) | 0.45 | 0.35 | 0.72 |

The A-trace is sufficient and exact in float64. The P-trace is not: the
exploration draw is absent, so `Δθ` cannot be recovered from the post-step
mean. Dropping `Xbar` destroys `G` and the natural-gradient direction.

The merge errors are not a bug in the arithmetic; they are the definition.
`merge` is a normalized weighted average of the segments' `F/Xbar/G` (an
ensemble average). `F` and `Xbar` can be combined exactly for a **sequential**
stream only with decay-aware weights (`β^(T−e_i)`), and `G` still needs the
cross terms between an earlier `Xbar` and a later `A`; per-segment `G` alone is
not sufficient. `step_batch` is a third definition (average the sufficient
statistics, then one update) and differs from `merge` by ~49% in the existing
test. The plan records this and does not change the default merge.

### Approved parameter-trace update

- `Agent.history["means"/"sigmas"]` stay diagnostic, unchanged.
- `Agent(trace_strategy=True)` records the applied sampling-time state in
  float64: `dtheta = xhat − mu_at`, `sigma_at`, `dlogsigma`, advantage,
  producer delta/scale. `--trace-float32` is explicit and approximate.
- Trace mode requires a fixed site set; a site added mid-run raises. Producers
  may grow; earlier rows are zero-padded.
- `FisherMatrix.snapshot()` / `from_snapshot()` store a self-contained strategy
  checkpoint (`F/Xbar/G/P`, refresh age, `β/λ/warmup`, names, scales). They are
  not a full Agent resume: phase/game/H/means/sigmas/RNG state live outside.
- `run.h5` writes `agent/strategy_trace/*` and
  `agent/strategy_checkpoints/*` only when enabled; `save_run`/`load_run` tolerate
  their absence. `personalize.py` writes `strategy_trace.npz` when tracing and
  an output directory are given.
- CLI: `--trace-strategy`, `--trace-float32`, `--snapshot-strategy N`,
  `--noise-config` in `persona/cli.py`, `run_validation.py` and `run_2min.py`.

## 4. Noise-recorder configuration results

Fast CPG, 400 steps, seed 0, `FAST_SIZES`, game/H off:

| config | n_noise | loss_eval | G-noise norm | trace error | wall |
|---|---:|---:|---:|---:|---:|
| all | 16 | 0.1632 | 0.0 | 0.0 | 171 s |
| none | 0 | 0.2732 | 0.0 | 0.0 | 207 s |
| poisson | 10 | 0.1859 | 0.0 | 0.0 | 202 s |
| stdp | 6 | 0.2604 | 0.0 | 0.0 | 196 s |
| one-relay:cut_L | 1 | 0.2730 | 0.0 | 0.0 | 225 s |

The trace rebuild is exact in every configuration. `G` noise rows are exactly
zero, as the design requires. The raw `noise_F_norm` is scale-dominated by the
site rows (at 100 steps `noise_F_rel ≈ 5e-20`), so it is reported as a
diagnostic, not a ranking. The loss differences are one-seed and are not a
performance claim. The 100-step `all` run also reproduced exact trace and zero
G-noise.

## 5. Open issues (not changed in this task)

1. Producer `scale` is fixed at registration; `Wmax` is itself an optimizable
   site and can move.
2. `record_delta` overwrites `_pending`; multiple writes between `take_delta`
   calls keep only the last. Current call paths write once per producer per
   step.
3. `δ/R²` treats `R` as a noise scale. For STDP `R = Wmax` (a weight ceiling);
   for Poisson `R` is a rate in Hz. The dimensional meaning needs review.
4. A site recorder stores the policy action, not the realized per-neuron or
   per-synapse dispersion.
5. Zero-synapse projections still have sites and recorders but produce no
   delta.

## 6. Recommendation

Keep `F/Xbar/G` as the strategy's sufficient statistics. Keep the parameter
mean/sigma histories as diagnostics. Use `trace_strategy=True` when exact
offline replay is needed; use snapshots for archival strategy checkpoints.
Do not describe `merge` as exact continuation; describe it as the ensemble
average it is. If exact sequential merge becomes a requirement, save the
applied-sample trace or add a cross-moment accumulator; re-deriving it from the
post-step parameter trace is not possible.
