# One-CPG threading sweep (2026-10-24)

Run outputs: `experiments/current/nest_threads/{t1,t2,t4,t6,realk10}`.

All runs: one CPG, one MuJoCo, 0.5 s (50 steps), selected game, seed 0,
`--snapshot-s 0`.

| NEST threads | build | run | steps/s |
|---:|---:|---:|---:|
| **1** | **5.39 s** | **42.60 s** | **1.174** |
| 2 | 4.29 s | 65.29 s | 0.766 |
| 4 | 22.37 s | 93.06 s | 0.537 |
| 6 | 34.98 s | 95.60 s | 0.523 |

Result: NEST threading the single CPG is slower. Keep `--nest-threads 1`.

## Serial realization optimization

| realize_every_k | run | steps/s | last10 loss |
|---:|---:|---:|---:|
| 1 | 42.60 s | 1.174 | 0.260 |
| 10 | **14.33 s** | **3.489** | **0.200** |

`--realize-every-k 10` is the practical speedup with one CPG/MuJoCo.
