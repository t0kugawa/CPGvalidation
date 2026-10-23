# Single-thread vs multithread short run (2026-10-24)

Same command, seed, duration, and checkpoint interval:

```text
run_2min.py --no-render --duration-s 0.5 --snapshot-s 20
```

| mode | steps | loss first10 | loss last10 | loss mean | M-F/Renshaw freezes | C1 drive L | C1 drive R |
|---|---:|---:|---:|---:|---|---:|---:|
| single (`--threads 1`) | 50 | 0.3031 | 0.2596 | 0.3085 | held | 2.75 → 3.47 | 2.75 → 5.49 |
| multi (`--threads 2`) | 50 | 0.3006 | 0.3224 | 0.3394 | held | 2.75 → 4.50 | 2.75 → 3.06 |

Frozen sites checked in both runs:

```text
pop_m_f_R_c = -65 -> -65
pop_m_f_L_c = -65 -> -65
conn_m_f_R_r_f_R_weight = 3 -> 3
conn_r_f_R_m_f_R_weight = -4 -> -4
```

Model-phase vs reference-phase agreement is low in both runs:

```text
single: L 0.620, R 0.120
multi:  L 0.640, R 0.120
```

That is the evidence for the reference-phase advantage fix: the model phase is not a
usable conditioning signal, especially on the right leg.

**Reading.** The single-thread run improves loss over this 0.5 s window; the
two-replica run is slightly worse in the last 10 steps. This is not a learning
verdict: 50 steps is too short, and `MultiAgent` applies the merged policy update
only at reference-phase boundaries, so its effective update cadence is lower. The
important checks pass in both modes: the H5/CSV/figures are complete, the frozen
M-F/Renshaw sites hold, and the C1 flexor drives no longer collapse in the first
50 steps.

Run the longer comparison with:

```powershell
python run_2min.py --out validation_2min_st --no-render --threads 1 --duration-s 120 --snapshot-s 60
python run_2min.py --out validation_2min_mt --no-render --threads 2 --duration-s 120 --snapshot-s 60
```
