# Conserved-stack game, 20 s comparison

p90 is a diagnostic, not the objective. `viable` means no seed loses
more than 10% of `loss_eval` and mean p90 improves.

| arm | mean p90 | min p90 | mean loss | max loss | q max | viol | active | entropy | stack min | stack max | pot | viable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| a25 | 15.2% | 15.2% | -14.0% | -14.0% | -45 | 0 | 115 | 0.528 | 9.9846e-13 | 0.51113 | 0 | True |
| h_fisher | 0.9% | 0.9% | -0.6% | -0.6% | -3 | 0 | — | — | — | — | — | True |
| h_best | 6.0% | 6.0% | -1.5% | -1.5% | 2 | 0 | — | — | — | — | — | True |

## Raw runs

| seed | arm | steps | loss_eval | p90 | at_bound | q>1 | viol |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | off | 400 | 0.65642 | 0.786 | 122 | 96 | 0 |
| 0 | a25 | 400 | 0.56443 | 0.6667 | 34 | 51 | 0 |
| 0 | h_fisher | 400 | 0.65272 | 0.7785 | 118 | 93 | 0 |
| 0 | h_best | 400 | 0.64629 | 0.7386 | 118 | 98 | 0 |
