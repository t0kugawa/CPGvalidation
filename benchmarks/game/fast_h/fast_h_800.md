# Conserved-stack game, 20 s comparison

p90 is a diagnostic, not the objective. `viable` means no seed loses
more than 10% of `loss_eval` and mean p90 improves.

| arm | mean p90 | min p90 | mean loss | max loss | q max | viol | active | entropy | stack min | stack max | pot | viable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| a25 | 11.0% | 1.5% | -7.3% | 0.5% | -13 | 0 | 118 | 0.505 | 9.9727e-13 | 0.48263 | 0 | True |
| h_fisher | -0.2% | -0.5% | 0.4% | 1.5% | 1 | 0 | — | — | — | — | — | False |
| h_best | -0.1% | -0.2% | -12.3% | -1.0% | 5 | 0 | — | — | — | — | — | False |

## Raw runs

| seed | arm | steps | loss_eval | p90 | at_bound | q>1 | viol |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | off | 800 | 0.58202 | 0.6094 | 42 | 66 | 0 |
| 0 | a25 | 800 | 0.58509 | 0.6 | 18 | 53 | 0 |
| 0 | h_fisher | 800 | 0.57805 | 0.6126 | 42 | 67 | 0 |
| 0 | h_best | 800 | 0.57594 | 0.6106 | 64 | 66 | 0 |
| 1 | off | 800 | 0.72452 | 0.6667 | 61 | 83 | 0 |
| 1 | a25 | 800 | 0.61513 | 0.5308 | 24 | 50 | 0 |
| 1 | h_fisher | 800 | 0.73517 | 0.6653 | 40 | 71 | 0 |
| 1 | h_best | 800 | 0.55422 | 0.6667 | 88 | 88 | 0 |
