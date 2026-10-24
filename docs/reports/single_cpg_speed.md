# Single-CPG / single-MuJoCo threading and speed plan

## Constraint

No multiple CPGs and no multiple MuJoCo models. Parallel **samples** would
require a separate network state per sample, so that path is removed. The only
threading that applies is NEST `local_num_threads` on the **single** CPG:

```text
one CPG (NEST)  ── local_num_threads = M ── neurons/synapses split across M threads
one MuJoCo      ── serial
one PERSONA sample per step ── applied to that one CPG
```

The sampled parameters are shared by all threads; each thread owns a subset of
the one network, not another CPG.

## Measured on one CPG (0.5 s, 50 steps, selected game)

| NEST threads | build | run | steps/s |
|---:|---:|---:|---:|
| **1** | **5.39 s** | **42.60 s** | **1.174** |
| 2 | 4.29 s | 65.29 s | 0.766 |
| 4 | 22.37 s | 93.06 s | 0.537 |
| 6 | 34.98 s | 95.60 s | 0.523 |

NEST threads are slower on this network: the Python/NEST API
(`apply_explored_parameters`, recorders, `GetConnections`) is serial, and the
per-thread work is too small. **Keep `--nest-threads 1`.**

## Real optimization: realize less often (speed only; currently rejected)

`--realize-every-k 10` samples/realizes/records every 10th control step while
the strategy still updates every dt:

```text
--realize-every-k 1 : 50 steps  42.60 s   1.174 steps/s   last10 loss 0.260
--realize-every-k 10: 50 steps  14.33 s   3.489 steps/s   last10 loss 0.200
```

**2026-10-24 update: k=10 is rejected for learning/validation.** In a matched
2 s run it raised last-10 loss from `0.276` (k=1) to `1.001` (k=10) and drove
the right ankle to the upper stop while the target was at the lower stop. The
120 s k=10 run also ends with both E/F saturated at 1.0. Use k=1 for learning;
keep k=10 only as a speed microbenchmark.

That is the practical speedup without extra CPG/MuJoCo resources, but not yet
acceptable as the learning path.

## Next serial optimizations, in order

1. Cache/batch connection handles: avoid re-querying every projection with
   `nest.GetConnections` in `finalize_connections`. This was the dominant cost
   in the replica experiment (140 s for 6 replicas); on one CPG the build is
   only ~5 s, so it is a smaller win.
2. Skip frozen sites in `apply_explored_parameters` when both mean and sigma
   are pinned (M-F Izhikevich sites).
3. Batch `SynapseCollection.set` calls where a projection shares one parameter
   array; avoid per-site `set` when nothing changed.
4. Keep `local_num_threads=1`; the one-CPG measurement above shows the
   overhead dominates.

## Recommended run

```powershell
python run_2min.py --out experiments/current/2min_120s --no-render --duration-s 120 --snapshot-s 60 --realize-every-k 1
```

Defaults: `--game selected`, `--nest-threads 1`, one CPG, one MuJoCo.
