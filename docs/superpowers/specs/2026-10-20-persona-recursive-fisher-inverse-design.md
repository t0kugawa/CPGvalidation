# PERSONA recursive Fisher inverse — design

**Status:** implemented (2026-10-20). This is the normative design for the
performance follow-up. It does not change PERSONA's credit, noise, or sigma
semantics.

## 1. Context

`FisherMatrix.step` currently forms `M = F + lambda*I` and calls
`np.linalg.solve(M, g)` every control step. At `n = 458` (221 sites plus 16
producers), the solve is 6.76 ms/call and about 68% of the 10 ms online step.
`F` changes by one rank-1 term per step, so its inverse can be updated in
O(n^2) instead of refactorizing in O(n^3).

## 2. Goal

Maintain `P ~= (F + lambda*I)^-1` and compute `dmu = eta * P @ g` in O(n^2) per
step, refreshing `P` exactly with a Cholesky factorization at a bounded
interval. Preserve the existing `F`, `Xbar`, `G`, `g`, noise scoring, and sigma
updates.

Non-goals:

- low-rank/block approximation of `F` (measured unsafe for the mu step);
- changing `beta`, `lambda`, the credit gain, or the noise scales;
- phase-local `F`;
- changing the paper's algorithm or the public strategy semantics.

## 3. Recursion

Let `s_t` be the score vector and

```text
F_t = beta * F_{t-1} + (1 - beta) * s_t s_t^T
```

With `P_{t-1} = F_{t-1}^-1`, Sherman-Morrison gives

```text
u = P_{t-1} @ s_t
v = s_t^T @ u
P_t = (1/beta) * (P_{t-1} - (1-beta) * u u^T / (beta + (1-beta) * v))
```

Cost: two matrix-vector products and one symmetric outer product, O(n^2).
Then

```text
nat = P_t @ g_t
```

is another O(n^2) product; no factorization.

## 4. Ridge

Initialize `P_0 = (1/lambda) * I` (`F_0 = 0`). If no refresh occurs, the
recursion maintains exactly

```text
P_t = (F_t + lambda * beta^t * I)^-1
```

Proof by induction: if `P_{t-1} = (F_{t-1} + lambda*beta^(t-1)*I)^-1`, the
update above is the inverse of

```text
beta*(F_{t-1} + lambda*beta^(t-1)*I) + (1-beta)*s_t s_t^T
= F_t + lambda*beta^t*I.
```

The prior fades with the same `beta` as the data. Refresh at a phase shift or
every `K_max` steps, whichever comes first:

```text
P <- (F + lambda*I)^-1        via Cholesky
```

This restores the constant ridge and removes accumulated round-off. `K_max` is
chosen so the ridge retains at least 95% between refreshes:

```text
K_max = max(1, floor(log(0.95) / log(beta)))   for 0 < beta < 1
K_max = 10^9                                    for beta = 1 (no decay)
K_max = 1                                       for beta <= 0
```

| beta | K_max |
|---|---|
| 0.995 | 10 |
| 0.99 | 5 |
| 0.98 | 2 |
| 0.95 | 1 |
| 0.90 | 1 |

At a refresh step, refresh **after** updating `F` with the current score and
**before** computing `nat`. The step at the boundary then uses the exact
`(F_t + lambda*I)^-1`. Between refreshes the step uses
`(F_t + lambda*beta^k*I)^-1`; the difference is bounded by the ridge drift.

## 5. Trigger policy

- `FisherMatrix.step(..., *, phase_shift: bool = False)`.
- Refresh if `phase_shift` or `_steps_since_refresh >= K_max`.
- `Agent.step` detects a phase shift by comparing the normalized `phase_key`
  with the previous step's key. The first step never counts as a shift. A
  change in any component of a per-leg phase tuple counts as a shift.
- The phase shift is the semantic event; `K_max` is the safety cap for long
  phases, standing, missed phase detection, or a lowered `beta`.
- Chattering phase detectors: this design assumes a stable phase signal. A
  shift every step would refresh every step (O(n^3)). If that is possible,
  debounce upstream or add a minimum interval in a follow-up.

**Important:** the refresh restores the ridge, not `F`'s memory. `F` remains
the beta-EWMA with effective memory `1/(1-beta)`; a long phase still forgets
its first updates at that rate. `K_max` does not change that. If phase-local
`F` is desired, that is a separate design (reset `F` at phase boundaries, or
maintain per-phase metrics).

## 6. Numerical policy

- Require `lambda > 0`.
- Symmetrize `P` after each rank-1 update and refresh:
  `P = 0.5 * (P + P.T)`.
- Refresh with `scipy.linalg.cho_factor` / `cho_solve` on `F + lambda*I`; on
  `numpy.linalg.LinAlgError`, fall back to `np.linalg.pinv`.
- If `P @ g` is non-finite, refresh and retry; if still non-finite, fall back
  to `np.linalg.solve` / `pinv` on `F + lambda*I`.
- `ensure_names` rebuilds `P` after resizing `F` (new channels have zero `F`
  and the lambda ridge), and resets the refresh counter.
- Memory: `F` and `P` are each n x n; about 3.4 MB total at n = 458, growing
  O(n^2).

## 7. Interfaces

- `FisherMatrix.P: np.ndarray` (n x n).
- `FisherMatrix.refresh_every: int`.
- `FisherMatrix.step(sites, noise, advantage, *, phase_shift: bool = False)
  -> Dict[str, float]`.
- `FisherMatrix._rank1_update_inverse(s)` and `_refresh_inverse()`.
- `Agent._prev_phase`; `Agent.step` passes `phase_shift`.

## 8. Acceptance

- A1. At a refresh, `P == inv(F + lambda*I)` (atol 1e-9).
- A2. Between refreshes, `P == inv(F + lambda*beta^k*I)` (atol 1e-9).
- A3. `P @ g` matches `solve(F + lambda*beta^k*I, g)` (rtol 1e-8); the returned
  `dmu` matches the direct solve.
- A4. `P` is symmetric, finite, and positive definite over 200 random steps.
- A5. A phase shift triggers exactly one refresh; an unchanged phase does not;
  the first step does not.
- A6. `K_max` triggers at the expected steps; a phase shift before `K_max`
  refreshes earlier.
- A7. `ensure_names` rebuilds `P`.
- A8. Full suite green; fast e2e finite; recursive loss last-10 within 10% of
  the same-network constant-ridge A/B and `max|dmu|` no worse than that A/B.
  The historical `validation_mvp_b` (221 sites) differs from the current
  217-site network, so it is a reference, not an exact A/B.
- A9. Microbenchmark recorded at n = 458: old direct full step, recursive
  no-refresh, recursive `K_max=10`, and direct solve-only.
- A10. The benchmark harness writes `performance.csv`, `accuracy.csv`,
  `summary.csv` (and `.xlsx` when `openpyxl` is available), plus
  `performance.png` and `accuracy.png`.
- A11. A smoke test runs the harness at small n and asserts the files and
  columns exist, that the decaying-ridge error is < 1e-8, and that the
  constant-ridge error is zero at refresh steps and grows between them.

## 9. Results

Implemented on 2026-10-20 on `mujoco_sim`.

- **Full suite:** 159 tests pass.
- **Fast e2e:** `run_validation.py --fast` on the current network gives loss
  last-10 `0.9482`, `max|dmu|` `0.5151`, and no NaNs. The historical
  `validation_mvp_b` has 221 sites versus 217 now (a later network commit
  changed the site set), so its `0.8725`/`0.4383` is not an exact A/B. A
  current-network constant-ridge A/B (`refresh_every=1`, refreshed every
  step) gives loss `0.9291` and `max|dmu|` `0.5706`; the recursive run is
  +2.1% loss and -9.7% `max|dmu|` against it, inside the acceptance margin.
- **Microbenchmark, n = 458** (221 sites + 16 producers; median full-step
  timings on the WSL CPU): old direct `np.linalg.solve` every step
  `34.96 ms`; recursive no-refresh `24.46 ms`; recursive `K_max=10`
  `30.32 ms`; Cholesky every step `68.78 ms`; direct solve-only `10.05 ms`.
  The solve-only figure is not a full step and is shown only to match the
  original measurement. The fair full-step ordering is
  `no-refresh < K_max=10 < old direct`. Timings are BLAS/CPU-dependent.

## 10. Benchmark harness

The implementation plan adds a standalone, NEST-free harness
(`bench_recursive_inverse.py`) that measures performance and accuracy and
writes spreadsheet-ready tables and figures:

- **Performance:** median and spread of ms/step for the direct full step
  (`np.linalg.solve`), recursive inverse without refresh, and recursive
  inverse with `K_max=10`, over
  `n = [64, 128, 256, 458, 512, 1024]`.
- **Accuracy:** for multiple seeds, the relative error of `P @ g` against
  `solve(F + lambda*beta^k*I, g)` (should be ~machine precision) and against
  the constant-ridge `solve(F + lambda*I, g)` (the controlled ridge drift),
  with the cosine similarity of the resulting `dmu`.
- **Artifacts:** `performance.csv`, `accuracy.csv`, `summary.csv`,
  `performance.png`, `accuracy.png`, and `recursive_inverse_results.xlsx`
  when `openpyxl` is installed. The files open directly in Excel or any
  spreadsheet program.

## 11. Risks

- Ridge drift between refreshes (5% retention at `K_max=10`); measure on the
  fast e2e.
- Numerical drift in `P`; refresh, symmetrization, and the Cholesky fallback
  bound it.
- Phase chattering; assumed stable.
- O(n^2) memory and the O(n^2) `F` outer product remain. If `n` grows into the
  thousands, revisit the score-window/dual formulation or structured methods.
- Phase-local `F` is out of scope; the refresh does not restore forgotten
  phase data.
