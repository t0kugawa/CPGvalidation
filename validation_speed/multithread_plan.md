# True multithread / parallel PERSONA — implementation plan

## Why the current `--threads N` path is slow

Measured on the same 0.5 s, FINAL-size workload:

```text
threads=1  build 5.46 s   run 41.31 s / 50 steps   1.21 steps/s
threads=6  build 142.38 s (killed before the run)
```

`cProfile` of the 6-thread build/one step:

```text
build_parallel_personalizer   157.4 s
  finalize_connections        140.3 s   (1,374 nest.GetConnections, 138.7 s)
one apply_actions step         28.7 s   (6 x apply_explored_parameters; 26.9 s in nest.set)
```

The thread path is not parallel for the expensive work. NEST's Python API
(`GetConnections`, `set`, `Connect`) is called from the main thread and the GIL
serializes the per-replica Python loops. `local_num_threads` only helps
`nest.Simulate`; it does not help parameter application or connection
discovery. The `ParallelPersonalizer` thread pool parallelizes only the MuJoCo
rigs, which are a small fraction of the step.

## Step 1 — quick wins, no architecture change

1. **Cache connection handles.** `nest.Connect` returns a `SynapseCollection`;
   store it on the `Connection`/adapter and reuse it. Stop re-querying every
   projection with `nest.GetConnections` in `finalize_connections`. This is the
   140 s build cost.
2. **Split the flags.** Replace the single `--threads` with:
   - `--replicas N` — independent CPG/MuJoCo replicas;
   - `--nest-threads M` — NEST `local_num_threads` (default 1).
   The current code conflates them; `local_num_threads=6` adds contention
   without parallelizing the API calls.
3. **Reduce realization frequency.** `--realize-every-k 10` samples/realizes/
   records every 10th step while the strategy still updates every dt. This cuts
   the 28.7 s `apply_actions` cost roughly 10x.
4. **Apply only changed sites.** `apply_explored_parameters` currently re-sets
   every site on every replica every realized step. Skip frozen sites (already
   done in the agent) and batch the remaining `set` calls.

## Step 2 — process-per-replica (the real parallel path)

Threads cannot bypass the GIL/NEST API. Use processes:

- Spawn one worker process per replica (`multiprocessing` with `spawn`, never
  `fork`; NEST is one kernel per process).
- Each worker owns one registry/CPG/MuJoCo rig and one `FisherMatrix`
  (`F/X̄/G`) and runs its own sample/realize/step loop.
- The parent owns the shared policy (`PersonaParam.mean/sigma`) and the merge.
- Use `multiprocessing.Queue`/`shared_memory` for:
  - policy mean/sigma + sampled actions (parent → worker);
  - per-replica sufficient statistics `F`, `X̄`, `G`, `steps`, `noise_scales`
    and current sites/advantages (worker → parent) at reference-phase
    boundaries.
- The parent merges with `MultiAgent._merge` (already implemented) and
  broadcasts the merged policy. The game/phase logic stays in the parent.
- Pin one CPU per worker; keep BLAS single-threaded per worker.

## Step 3 — benchmark and verify

- Measure build, per-step, and total wall time for 1, 2, 4, 8 replicas.
- Report `replica-steps/s` as well as `steps/s`.
- Verify the same seed gives reproducible H5 output, the M-F/Renshaw freezes
  hold, the game diagnostics are present, and the merge weights/advantages are
  identical to the thread path for a fixed schedule.

## Practical now

Use the fast single-thread path with the selected game:

```powershell
python run_2min.py --out validation_2min_game --no-render --duration-s 120 --snapshot-s 60
```

`--game selected` and `--threads 1` are the defaults. Add
`--realize-every-k 10` if wall time matters more than per-step fidelity.
