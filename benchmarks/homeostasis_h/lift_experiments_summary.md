# Homeostatic lift/timing experiments

Production settings: 2000 steps, last 1000 evaluated, `PROD_SIZES`
(10/8/8/6/8), `global_variance_mode="fisher"`, `lambda_H=1.0`, seed 0/1.
The current probe reproduced the acceptance result (seed 0 p90 .5416), so the
experimental regulator is comparable to the production path.

| variant | seed 0 p90 | seed 1 p90 | seed 0 loss | seed 1 loss | seed 0 bound | seed 1 bound | site sum zero |
|---|---:|---:|---:|---:|---:|---:|---|
| H off | .4990 | .6461 | 1.1435 | 1.2807 | 23 | 40 | n/a |
| current player lift | .5416 | .6154 | 1.1499 | 1.2519 | 34 | 39 | no |
| scaled site-zero-sum | .5539 | not run | 1.1689 | not run | 15 | not run | yes |
| centered site-zero-sum | .5394 | .5759 | 1.1255 | 1.5205 | 15 | 13 | yes |
| equal site-zero-sum | .5524 | not run | 1.1840 | not run | 20 | not run | yes |
| global-centered site lift | .5665 | not run | 1.1902 | not run | 27 | not run | yes |
| 500-step ramp | .5247 | .6145 | 1.1577 | 1.3394 | 31 | 40 | no |
| 500-step delay | .4901 | .6513 | 1.1434 | 1.2733 | 22 | 39 | no |
| high-pass `e_bar` | .5202 | .6415 | 1.2367 | 1.2773 | 34 | 40 | no |
| 2% bounds cap | .5040 | .6488 | 1.1484 | 1.3101 | 30 | 40 | no |
| population-only | .5554 | not run | 1.2472 | not run | 37 | not run | no |
| no-STDP | .5708 | not run | 1.2015 | not run | 40 | not run | no |
| centered + delay | .4774 | .6585 | 1.1435 | 1.2756 | 13 | 29 | yes |
| centered + ramp | .5562 | .6160 | 1.1584 | 1.2142 | 15 | 18 | yes |
| centered + high-pass | .5730 | not run | 1.3321 | not run | 53 | not run | yes |
| centered + bounds | .5723 | not run | 1.1757 | not run | 35 | not run | yes |

## Conclusion

No candidate reaches the 20% p90 reduction. `centered_site` is the best mean
candidate (~2.6% mean p90 improvement) and reduces at-bound count to 15/13, but
it regresses seed 0 and raises seed 1 loss. `centered_delay` is the best single
seed-0 result (.4774) but regresses seed 1 (.6585). Simple lift, timing,
high-pass, bounds, and type-filter changes do not compose into an acceptance
pass. The current player-level lift remains the production implementation; H
stays opt-in.

Untested directions that require a larger design change: role/type-aware target
shares, a full site-level H graph, or a different H objective that damps
parameter inflation rather than redistributing credit.

The real 20 s, two-seed comparison of H off, current H_opt, and centered+delay
is in `real_compare_20s.md`: centered+delay is the best p90 candidate
(+4.3% / -1.9%; mean +1.2%) and cuts at-bound parameters to 13/29, but still
does not reach 20%. H_opt is -8.5% / +4.8% (mean -1.9%).
