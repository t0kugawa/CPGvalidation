# PERSONA multi-sample parallel design

**Date:** 2026-10-23
**Status:** approved for implementation
**Supersedes:** the "identical samples for every thread count" contract in
`persona/sampling.py` (commit `1e8d708`).

## 1. Intent

`--threads=N` must buy more PERSONA samples per wall-clock step, not merely
run the same sample faster. Each of the `N` threads owns a different parameter
sample and its own learning strategy. The strategies are merged at reference
phase boundaries, and the merged result updates one shared policy.

The SNN and MuJoCo plant are also per-thread: different parameter samples
produce different spike trains and different limb trajectories.

## 2. Hard constraint: one NEST kernel per process

NEST has a single global kernel and cannot host two independent networks in
two Python threads. We therefore build `N` **independent CPG replicas inside
one NEST kernel**:

- each replica has its own `Registry`, populations, connections, relays,
  recorders, STDP weights and MuJoCo rig;
- the kernel runs with `local_num_threads = N`;
- NEST distributes the replicas' nodes over its threads, so the SNN work is
  parallel even though a NEST thread is not pinned to one replica;
- one `nest.Simulate(dt)` advances all replicas, then each replica's recorders
  are drained.

`--threads=1` keeps the existing single-replica path and behaviour.

## 3. Replica construction

`build_personalizer(threads=N)`:

1. `configure_kernel(nest, resolution_ms, seed, threads=N)` exactly once.
2. `reset_rng_stream(seed)` and `np.random.seed(seed)` exactly once.
3. Load the mocap reference and compute targets, belt height, hip offsets and
   limb lengths once.
4. For `i in range(N)`, build `Registry(namespace=f"r{i}_")`, its brainstem,
   base drive, two legs, `L2CPG` and its own `MuJoCo_Leg` rig.
5. Build `MultiCPG(cpgs)` and one `ParallelPersonalizer`.

`NestAdapter` gains a `namespace` used only for `nest.CopyModel` names
(`stdp_<type>_<namespace><src>_<tgt>`). Registry keys remain unprefixed, so
every replica exposes the same PERSONA site names.

All replicas are reset to the same initial pose and receive the same target
trajectory. Their parameters differ only through their own exploratory draws.

## 4. Per-dt flow (N replicas)

For each control step `t`:

1. Synchronise the shared policy's `mean`/`sigma` into every replica's
   `PersonaParam` objects.
2. The shared policy draws `N` values per site in one vectorised call;
   replica `i` receives draw `i`. The samples differ across replicas by
   construction, and no per-site Python loop is paid per replica.
3. Apply `xhat_i` to replica `i`'s registry.
4. Update each replica's sensory relays, call `nest.Simulate(dt)` **once**,
   then drain each replica's `L2CPG` readouts (`L2CPG.readout()`).
5. Step the `N` MuJoCo rigs in a thread pool, one `CPGInput` per replica.
6. Each replica computes its own smoothed output/target, error vector, loss
   and strategy scores. The **phase baseline is shared**: advantages are
   `A_i = baseline(ref_phase) - loss_i`, and the shared baseline is updated
   once per dt from the mean loss.
7. Each replica's own `FisherMatrix` accumulates F/X̄/G from its own sample
   and advantage. Its Δμ is not applied at this point.
8. When the shared reference phase (`phase_ref_L/R`) changes stance↔swing,
   merge the per-replica strategies with advantage-dependent weights, apply
   one Δμ/Δlogσ to the shared policy, and copy the merged strategy back to
   every replica. The next phase starts from the merged state.

## 5. Per-replica strategy accumulation

For replica `i`, the existing score/trace definitions are unchanged:

```
s_i = [Δθ_i/σ² ; z_i²−1 ; δ_noise/R²]
a_i = [Δθ_i/σ² ; Δlogσ_i ; 0]
F_i = b F_i + (1-b) s_i s_iᵀ
X̄_i = b X̄_i + (1-b) a_i
G_i = b G_i + (1-b) X̄_i A_iᵀ
```

`FisherMatrix.step(..., accumulate_only=True)` performs this update, advances
the replica's step counter, and returns no Δμ. It does not touch the shared
policy. The main strategy is only the merge target; it is not updated every
dt.

## 6. Advantage-weighted merge

At a reference phase change, each replica has the mean advantage it
accumulated during the phase that is ending, `Ā_i = mean_phase(A_i)`.
Weights are a softmax over that advantage spread:

```
spread = max(std(Ā), 1e-6)
w_i    = softmax(Ā_i / spread)
```

so a better sample gets more weight, while equal samples get equal weight.
The merge is a weighted average of the reusable state:

```
F    = Σ_i w_i F_i
X̄    = Σ_i w_i X̄_i
G    = Σ_i w_i G_i
steps = max_i steps_i
```

The merged `P` is refreshed (or rebuilt lazily) from the merged `F`. The
merged Δμ is computed once from `G` with the weighted mean advantage
`Ā = Σ_i w_i A_i`; homeostasis/game/springs are applied exactly as in the
single-agent path. The per-site Δlogσ is the weighted mean of the replicas'
`Δlogσ_i`. The policy mean/sigma are updated once and synchronised to every
replica.

The merged `F/X̄/G/steps` are copied into every replica strategy, so the next
phase accumulates on top of the merged state.

## 7. Baseline

The `PhaseBaselines` object is shared by all replicas. Each dt uses the
reference phase to look up one baseline per limb; each replica's advantage is
that baseline minus its own loss. The baseline is updated once per dt from the
mean loss across replicas. Replica-local baselines are not used.

## 8. Merge vs combined update (must be tested)

The spec keeps both aggregation rules:

- **merge (production):** per-replica strategies accumulated every dt, merged
  at phase boundaries with advantage weights.
- **combined/batch (comparison):** one strategy updated per dt from the
  averaged sufficient statistics of all replicas.

`FisherMatrix.step_batch(observations)` implements the combined rule with
`S_outer = mean(s_i s_iᵀ)`, `ā = mean(a_i)`, `Ā = mean(A_i)`. Tests run both
rules on the same synthetic observation stream and report the resulting Δμ
and timing. The production default is merge.

## 9. Determinism and reproducibility

- Samples are **intentionally different** across replicas. The old
  "single-thread == parallel" equality contract is retired.
- For a fixed base seed and a fixed `N`, the policy's vectorised draws and
  therefore every replica's sample sequence are reproducible.
- A fixed `(seed, threads, NEST thread count)` produces reproducible NEST
  simulations. Different thread counts may use different NEST RNG streams;
  this is documented, not tested as equality.

## 10. Logging

`RunLog` records the **mean across replicas** of every numeric field (ankle
angles, drives, afferent rates, contacts, foot positions, loss, advantage).
Phase/stage strings and the step hook use replica 0. STDP deltas/counts are
averaged; live weight distributions and final weights come from replica 0
(the policy registry). The return dict carries `losses` and `advantages` per
replica for diagnostics.

## 11. Tests

Pure (no NEST/MuJoCo):

- `tests/test_replica_sampling.py`
  - replicas receive different samples.
  - a fixed `(seed, N)` is reproducible.
  - `N=1` matches the single-agent sampler.
  - vectorised replica sampling speed is measured and bounded.
- `tests/test_strategy_merge.py`
  - `accumulate_only` updates F/X̄/G without applying Δμ.
  - weighted merge equals the explicit weighted sums and refreshes `P`.
  - merge respects advantage weights (a dominant sample dominates).
  - `step_batch` with one observation equals `step`.
  - merge vs combined update both run and are reported on the same stream.
- `tests/test_threads_config.py`, `tests/test_cli_threads.py` (updated help).

NEST/MuJoCo integration (skipped without the simulators):

- `tests/test_threads_nest.py`
  - `local_num_threads == N`.
  - `N` replicas scale node/site counts.
  - one `Simulate(dt)` advances all replicas.
  - different samples produce different outputs.
  - a fixed `(seed, replica)` is reproducible.
  - per-sample throughput does not regress pathologically.
- `tests/test_mujoco_replicas.py`
  - two rigs with different CPG outputs produce different, uncoupled inputs.

## 12. Non-goals

- MPI or multi-process replicas.
- Per-NEST-thread pinning of a replica.
- Changing the single-replica (`--threads=1`) learning rule or logs.
- Parallelising the biological population sampling (`Population.sample_matrix`).
