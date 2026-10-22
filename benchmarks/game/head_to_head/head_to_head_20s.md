# Conserved-stack game, 20 s comparison

p90 is a diagnostic, not the objective. `viable` means no seed loses
more than 10% of `loss_eval` and mean p90 improves.

| arm | mean p90 | min p90 | mean loss | max loss | q max | viol | active | entropy | stack min | stack max | pot | viable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| prop_none | 12.5% | -0.8% | -16.5% | 1.8% | -29 | 0 | 7 | 0.147 | 1.0006e-12 | 0.69249 | 0 | True |
| prop_unc_05_a25 | 7.3% | 2.0% | -16.5% | 1.6% | -32 | 0 | 110 | 0.445 | 9.9755e-13 | 0.54954 | 0 | True |

## Raw runs

| seed | arm | steps | loss_eval | p90 | at_bound | q>1 | viol |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | off | 2000 | 0.66727 | 0.5024 | 25 | 57 | 0 |
| 0 | prop_none | 2000 | 0.67897 | 0.5063 | 9 | 21 | 0 |
| 0 | prop_unc_05_a25 | 2000 | 0.67766 | 0.4926 | 17 | 22 | 0 |
| 1 | off | 2000 | 1.0444 | 0.4514 | 29 | 51 | 0 |
| 1 | prop_none | 2000 | 0.68106 | 0.3347 | 9 | 22 | 0 |
| 1 | prop_unc_05_a25 | 2000 | 0.6838 | 0.394 | 13 | 19 | 0 |
