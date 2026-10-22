# Real 20 s comparison: H off vs H_opt vs centered+delay

Production `t2xcpg` path, `PROD_SIZES` (10/8/8/6/8), 2000 control steps,
last 1000 evaluated, seeds 0/1. `H_opt` is the current player-level H with
`fisher`, `lambda_H=1.0`. `centered+delay` is the centered site-zero-sum lift
with H disabled before step 500. All six arms ran concurrently.

| seed | variant | p90 | median | max | r>0.25 | q>1 | at bound | loss_eval | share RMS | site-sum max | phase agreement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | H off | .4990 | .0934 | .888 | 61 | 66 | 23 | 1.14347 | — | — | 1.000 |
| 0 | H_opt | .5416 | .0886 | .996 | 68 | 68 | 34 | 1.14987 | .4086 | 6.46e-01 | .796 |
| 0 | centered+delay | **.4774** | .0936 | .911 | 62 | 68 | **13** | 1.14355 | .4616 | 1.56e-17 | .875 |
| 1 | H off | .6461 | .1037 | 1.000 | 91 | 80 | 40 | 1.28073 | — | — | 1.000 |
| 1 | H_opt | **.6154** | .1081 | 1.000 | 89 | 78 | 39 | 1.25192 | .4806 | 3.16e-01 | .925 |
| 1 | centered+delay | .6585 | .1245 | 1.000 | 102 | 78 | **29** | 1.27556 | .5682 | 4.51e-17 | .977 |

## Paired result

- `H_opt`: seed 0 p90 **-8.5%** (worse), seed 1 **+4.8%** (better); mean
  **-1.9%**. Mean loss change -0.8%.
- `centered+delay`: seed 0 p90 **+4.3%** (better), seed 1 **-1.9%** (worse);
  mean **+1.2%**. Mean loss change -0.2%.
- Neither mode reaches the 20% primary criterion. `q_growth_max` is +2 for both.
- `centered+delay` is best on seed 0 and strongly reduces at-bound parameters
  (23→13 and 40→29); it also has better phase agreement than `H_opt`.
- `H_opt` has lower share RMS and better seed-1 p90, but worse seed-0 p90 and
  more at-bound parameters.
- `centered+delay` achieves exact site-sum zero (`~1e-17`); `H_opt` does not.

## Conclusion

Centered+delay is the better p90 candidate overall (+1.2% mean vs -1.9% for
H_opt) and the only one that improves seed 0, but it regresses seed 1 and is
far from 20%. H_opt remains the current opt-in implementation, not a production
default. No default or production semantics were changed.

The next discriminating experiment should be a direct trust-region/oracle test:
can p90 be reduced by 20% by constraining top-decile parameter movement at all,
without exceeding the loss budget? If yes, the core goal is feasible and the H
mechanism needs redesign; if no, the objective/metric conflict is fundamental.
