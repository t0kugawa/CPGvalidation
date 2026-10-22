# Oracle feasibility: direct parameter-space trust region

No H. After every mean update, clip each parameter to

```text
theta0 +/- R * (max - min)
```

Sigma and all other semantics are untouched. Production sizes
(10/8/8/6/8), 2000 control steps, last 1000 evaluated, seeds 0/1. Only 2.7-5.6%
of mean updates actually hit the radius, so this is a tail constraint, not a
global learning-rate cut.

Baseline H off: seed 0 p90 `.4990`, loss `1.1435`, q `66`, at-bound `23`;
seed 1 p90 `.6461`, loss `1.2807`, q `80`, at-bound `40`.

| R | seed | p90 | p90 improvement | loss | loss delta | q>1 | at bound |
|---:|---:|---:|---:|---:|---:|---:|---:|
| .40 | 0 | .3951 | 20.8% | 1.2910 | +12.9% | 60 | 38 |
| .40 | 1 | .3954 | 38.8% | 1.4269 | +11.4% | 51 | 56 |
| .35 | 0 | .3491 | 30.0% | 1.4453 | +26.4% | 56 | 34 |
| .35 | 1 | .3455 | 46.5% | 1.4732 | +15.0% | 56 | 45 |
| .30 | 0 | .3000 | 39.9% | 1.3820 | +20.9% | 62 | 32 |
| .30 | 1 | .2948 | 54.4% | 1.2648 | -1.2% | 56 | 13 |

## Reading

- **R=.40** reaches 29.8% mean p90 improvement but fails the mean loss budget
  (+12.2%).
- **R=.35** reaches 38.3% mean p90 improvement but loss is much worse
  (+20.7%).
- **R=.30** reaches 47.1% mean p90 improvement, mean loss +9.8% (inside the
  10% mean budget by 0.2 percentage points), and q growth is negative. But seed
  0 loss is +20.9%; the mean hides a strong seed asymmetry. Seed 0 at-bound
  count also rises 23 -> 32.
- All radii have zero `range_fallback` parameters. The trust region is inside
  the host bounds by construction.

## Conclusion

The p90 objective is **not impossible**: a direct constraint on parameter
displacement reaches it while keeping the mean loss near the budget. The failure
of current H is therefore a mechanism/objective problem, not evidence that the
metric cannot be controlled.

This oracle is deliberately blunt. It does not yet show that a causal,
online controller can match R=.30 at the same loss, or that the per-seed loss
regression can be removed. The next mechanism should act on parameter
displacement or a tail trust region (possibly per player), not only on credit
shares. A robust pass needs a seed-0 loss result below +10%, not just a mean
inside +10%.
