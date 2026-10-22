# PERSONA conserved-stack game — design

**Status:** experiment, 2026-10-22. Opt-in only. It does not replace
`homeostasis=None`, H, or any production default.

## 1. Why this layer exists

PERSONA already has a working natural-gradient strategy. The Fisher field
controls itself, parameter couplings are real, and personalization works. What
is missing is a layer that **owns the allocation** of that control.

The H experiment tried to do this by redistributing credit shares. It failed
the production acceptance target, and the oracle probe showed why the direction
was wrong: a parameter-space trust region can force p90 down, but it is a clamp
in disguise. It only works because it forbids movement. It does not produce a
balance.

This design adds the missing layer as a **zero-sum game over a conserved
resource**:

- the resource is a player's share of the total advantage (`Σ s_i = 1`);
- Fisher is the environment that prices and routes transfers between players;
- players hold influence, not movement budgets;
- no setpoint, target band, trust radius, or per-parameter stability clamp is
  allowed.

The only hard constraints are **type/sign identity**: an inhibitory synapse
cannot become excitatory, an excitatory synapse cannot become inhibitory, a
Poisson population cannot become STDP, and a population model cannot change.
Those are properties of the player, not stability targets.

## 2. Players and physical properties

A player is the same strategy-bearing unit as in the H design
(`PlayerSpec`): a population or a connection. Its physical properties live in
two places:

- **parameters** — type, sign, model, and the hard bounds already attached to
  each `PersonaParam`;
- **strategy** — the local natural-gradient direction, the sampled sigma, and
  the player's current stack.

A player does not know the global advantage. It knows only its own influence
and its own local direction. Fisher aggregates local effects into cross-player
transfers.

## 3. Conserved quantity: stacks

Each player `i` holds a stack `s_i ≥ 0` with

```text
Σ_i s_i = 1.
```

The stack is the player's current share of the total advantage. It is a share,
not a tank: it can move between players, but it cannot vanish. There is no
"bet" and no voluntary wager. The environment reallocates on every action.

Initial state: uniform, `s_i = 1/n`.

## 4. One action

One action is one `FisherMatrix.step` (`dt_s = 10 ms` in production).

1. The strategy computes the natural-gradient proposal `Δ_i` for every site.
2. The game computes a per-player throttle `τ_i` from the current stacks and
   applies `Δ_i ← τ_i Δ_i`.
3. The game computes the player's first-order contribution
   `a_i = Σ_{p∈P_i} g_p Δ_p`.
4. The game transfers share between players (zero-sum).
5. The throttled `Δ_i` is returned to the Agent and applied to the parameters.

The transfer uses the first-order predicted advantage `g^T Δ`, which is what
Fisher can attribute locally. Realized task advantage can replace it later; the
interface stays the same.

## 5. Throttle modes

Both modes are **relative**, so they need no absolute reference. They are
neutral (all `τ_i = 1`) when the stacks are uniform.

| mode | formula | behaviour |
|---|---|---|
| `proportional` | `τ_i = s_i / max_j s_j` | the richest player moves most; a zero-stack player freezes |
| `inverse` | `τ_i = (1 − s_i) / max_j (1 − s_j)` | the poorest player moves most; a full-stack player freezes |

The throttle is an **entitlement**, not a fee. Unused entitlement is not
transferred, because it was never taken from anyone. This is the
"never-consumed" property of the stack.

`inverse` is expected to be self-correcting: a player that gains share slows
down, a player that loses share speeds up. `proportional` is expected to be
positive feedback. Both are tested; neither is declared correct in advance.

### 5.1 Physical base seat (added after the first comparison)

The first comparison showed that a purely contestable throttle has an absorbing
state: `proportional` drove 129–130 of 137 players to zero and froze them. That
is a shutdown, not a balance. The fix separates **selection** from **survival**.

Every player keeps a physical base entitlement, independent of its current
share. The throttle becomes a blend:

```text
τ_i = (1 − ε) · c_i^contest + ε · c_i^base,
```

where `c_i^contest` is the mode's contestable throttle from the table above
(`proportional` or `inverse`) and `c_i^base` is the base seat normalized so
`max_i c_i^base = 1`. `ε ∈ [0, 1]` is the liveness fraction: `ε = 0` reproduces
the original contestable throttle exactly; `ε = 1` removes the game and leaves
only the physical base.

Two base seats are tested:

| base seat | `c_i^base` | meaning |
|---|---|---|
| `uniform` | `1` | every player keeps the same minimal movement |
| `uncertainty` | `RMS(σ_p / σ_ref,p)` over the player's sites | an uncertain player keeps more base movement; a confident one keeps less |

The uncertainty seat uses the same reference sigma as the H regulator: the first
observed sigma per site. It is the player's own exploration state, not a
setpoint. A player that loses influence but is still uncertain keeps exploring;
a confident, unproductive player correctly goes quiet.

The base does not enter the conserved resource. The stack stays zero-sum and
can still concentrate; only the *right to move* is separated from the *claim on
advantage*. This is the "players are not parameters" correction: a player's
parameters exist and keep being tuned even when its influence share is zero.

## 6. Zero-sum transfer

Normalize the contributions by their absolute mass:

```text
â_i = a_i / (Σ_j |a_j| + ε)
```

The replicator transfer is

```text
Δs_i = η (â_i − s_i Σ_j â_j).
```

Because `Σ_i s_i = 1`, `Σ_i Δs_i = η (Σ_i â_i − Σ_i â_i) = 0`: the transfer is
exactly zero-sum before the simplex projection. A player that contributes above
its share gains; a player below its share loses. Negative contributions lose
share. `η` is `transfer_rate`.

After the transfer, project back onto the simplex:

```text
s ← max(s + Δs, share_floor);  s ← s / Σ s.
```

The projection can move at most `n · share_floor` of mass; diagnostics record
the pre-projection `transfer_sum` and the post-projection `share_sum`.

## 7. Two arms

### 7.1 No-consume arm (`movement_fee = 0`)

The stack moves only by the outcome transfer in §6. Movement itself is free.
This is the baseline for the game.

### 7.2 Common-pot arm (`movement_fee = κ > 0`)

Movement has a price, and the price is redistributed. The Fisher length of the
executed move is

```text
d_i = sqrt(Δ_i^T F_ii Δ_i),
```

where `F_ii` is the player's Fisher block. The fee is

```text
c_i = κ d_i / (Σ_j d_j + ε),   pot = Σ_i c_i ≤ κ.
```

Debit `s_i ← s_i − c_i`. Redistribute the pot by positive contribution
(`pot_mode = "contribution"`, weights `max(â_i, 0)`) or equally
(`pot_mode = "equal"`). The debit equals the credit by construction, so the pot
is zero-sum before the simplex projection.

The question the arm answers: does charging for movement suppress drift without
destroying the exploration that low-stack players need?

## 8. Registry sign clamp

The registry enforces type/sign identity when a connection is created. For a
static synapse with an optimizable `weight`:

- if `weight.mean > 0`, set `weight.min = max(weight.min, 0.0)`;
- if `weight.mean < 0`, set `weight.max = min(weight.max, 0.0)`;
- if `weight.mean == 0`, leave the bounds untouched.

STDP hyper-parameters are untouched; an STDP seed weight is not an optimizable
site. The clamp is a type/sign constraint, not a stability setpoint. It makes
the identity true even for callers that forget to set one-sided bounds.

## 9. Clock

v1 runs one action per strategy step. The H clock is unchanged.

After the experiments, decide whether the game needs a per-player asynchronous
clock (each player acts on its own timescale) or whether the H clock should be
changed instead. That decision is deliberately deferred and must be recorded in
this document before it is implemented.

## 10. Test plan

The first comparison ran four game arms plus the H-off baseline. The base-seat
comparison keeps the proportional ranking (the only arm with a real stability
effect), drops the pot (it cost loss and collapsed participation), and sweeps
the base seat:

| arm | throttle | base seat | ε |
|---|---|---|---|
| `off` | — | — | — |
| `prop_none` | proportional | `none` | 0 |
| `prop_uni_05` | proportional | `uniform` | 0.05 |
| `prop_uni_10` | proportional | `uniform` | 0.10 |
| `prop_uni_20` | proportional | `uniform` | 0.20 |
| `prop_unc_05` | proportional | `uncertainty` | 0.05 |
| `prop_unc_10` | proportional | `uncertainty` | 0.10 |
| `prop_unc_20` | proportional | `uncertainty` | 0.20 |

All arms: seeds 0 and 1, 2000 steps (20 s), the same production sizes and
1000-step evaluation window as the H benchmark.

Reported per arm: mean and minimum `r_p90` improvement, mean and maximum
`loss_eval` delta, `q_gt_1` growth, type/sign violations, stack entropy,
stack min/max, throttle min/max, pot mean, and `transfer_sum`.

`r_p90` is a **diagnostic**, not the objective. The decision rule is: an arm is
viable if no seed loses more than 10% of `loss_eval` and it lowers p90/at-bound
relative to the baseline. The p90 target from the H design (≥20%) is not an
acceptance gate here.

## 11. Non-goals

- No absolute per-parameter setpoints, target bands, trust regions, or stability
  clamps.
- No change to H defaults; H stays opt-in.
- No modification of `tinycpg/`, `memcpg/`, or root `persona.py`.
- No per-player asynchronous clock in v1.
- No change to `F`, `Xbar`, `G`, `g`, noise scoring, credit, sigma, or the
  recursive inverse.

## 12. Open questions

1. Does `inverse` beat `proportional`?
2. Does the common pot help, hurt, or do nothing?
3. Does the first-order `g^T Δ` transfer predict realized advantage well enough?
4. Is a per-player asynchronous clock needed?
5. Should the throttle apply to sigma as well as the mean coordinates?

## 13. Results (2026-10-22)

20 s, 2000 steps, seeds 0/1, 137 players, H-off baseline. Artifacts:
`benchmarks/game/real_compare_20s.{csv,json,md}`.

| arm | mean p90 | min p90 | mean loss | max loss | q max | viol | entropy | active players |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `prop_none` | 19.2% | -1.3% | +1.6% | +2.1% | -40 | 0 | 0.13 | 7–8 |
| `prop_pot` | 19.8% | -1.9% | +5.2% | +8.3% | -42 | 0 | 0.13 | 12–13 |
| `inv_none` | 6.9% | -0.1% | -1.5% | -0.2% | -1 | 0 | 0.73 | 122–126 |
| `inv_pot` | -1.6% | -2.5% | +0.2% | +1.6% | -2 | 0 | 0.66 | 47–85 |

`at_bound` (seed 0 / seed 1): H off 28 / 46; `prop_none` 16 / 14;
`prop_pot` 14 / 19; `inv_none` 28 / 46; `inv_pot` 35 / 44.

**What the numbers say.**

- `prop_none` nearly reaches the old 20% p90 target (+19.2% mean) with a
  +1.6% mean loss, a +2.1% worst-seed loss, 40 fewer `q > 1` parameters, and
  no violations. But the gain is not distributed: only 7–8 of 137 players keep
  a nonzero stack, and 129–130 freeze. The p90 improvement is bought by
  shutting down most of the network, which is close to the clamp this design
  was meant to avoid.
- `inv_none` is the healthy-participation arm: 122–126 of 137 players stay
  active, entropy is 0.73, and loss is slightly lower than H off. Its stability
  gain is small (+6.9% mean p90) and not robust across seeds (seed 0 +13.9%,
  seed 1 -0.1%).
- The common pot does not pay for itself. Under `inverse` it collapses
  participation (47–85 active) and makes p90 negative; under `proportional` it
  buys 0.6 points of p90 for 3.6 points of extra mean loss.
- The seed split matters. Most of the `proportional` p90 gain is seed 1
  (+39.7%); seed 0 is -1.3%. The `inverse` arm is the reverse: its gain is
  seed 0 (+13.9%), and seed 1 is flat.

**Interpretation.** The game mechanics are sound: zero-sum transfer, simplex
projection, registry sign identity, opt-in parity, and zero type/sign
violations all hold. The open problem is no longer the accounting. It is that
the p90 diagnostic rewards a collapsed stack distribution. A stable subset of
7–8 players can dominate the metric while the rest freeze. That is exactly the
kind of degenerate equilibrium this design was supposed to rule out.

**Reproducibility correction (2026-10-22).** The first two comparisons were not
reproducible. Two leaks were found:

- `build_personalizer` reset the NEST kernel but did not pin its RNG, so the
  same `(arm, seed)` gave different trajectories in different processes;
- arms shared a worker process, and an H-off arm run after a game arm produced
  a different baseline even with the kernel reset.

The fixes are now in place: `build_personalizer` sets
`rng_seed = 1 + seed` (NEST requires a positive seed), and `bench_game.py` runs
each arm in a fresh process. The numbers in this section predate those fixes
and are kept only as the qualitative first observation; the base-seat sweep in
§15 supersedes them. The H benchmark harness still shares workers and does not
pin the NEST seed, so its historical numbers carry the same caveat.

## 14. Decision after the first comparison

The first comparison does **not** justify a production default and does not
settle the mechanism. It rules out the common pot as implemented (too much loss
or collapse) and shows that the throttle direction is the dominant lever.

The collapse was judged a failure: turning off 129–130 of 137 players is not a
natural balance, even if it improves p90. The resolution is the physical base
seat in §5.1: selection stays contestable, survival becomes a physical right.
The base-seat sweep in §10 tests both seats and three liveness fractions.

§15 reports the base-seat sweep. `prop_unc_05` (uncertainty seat, ε = 0.05,
proportional, no pot) is the candidate for the next step; it is not a
production default. No second mechanism beyond the base seat is implemented.

## 15. Base-seat sweep (fresh-process, 2026-10-22)

20 s, 2000 steps, seeds 0/1, 137 players, one fresh process per arm, NEST
`rng_seed = 1 + seed`. The H-off baseline is reproducible: seed 0
`loss 0.66727 / p90 0.5024 / at_bound 25 / q 57`; seed 1
`loss 1.0444 / p90 0.4514 / at_bound 29 / q 51`. The seed-1 baseline was
re-run in isolation and reproduced to six digits.

| arm | mean p90 | min p90 | mean loss | max loss | q max | active | entropy | at_bound s0/s1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `prop_none` | +12.5% | -0.8% | -16.5% | +1.8% | -29 | 7 | 0.147 | 9 / 9 |
| `prop_uni_05` | +5.8% | -0.2% | -16.1% | +2.0% | -27 | 114.5 | 0.473 | 22 / 20 |
| `prop_uni_10` | -1.7% | -2.3% | -15.5% | +2.4% | -24 | 113.5 | 0.556 | 25 / 18 |
| `prop_uni_20` | -10.1% | -10.8% | -15.9% | +1.8% | -20 | 116 | 0.619 | 29 / 30 |
| `prop_unc_05` | +6.3% | +0.5% | -16.3% | +1.7% | -27 | 117.5 | 0.476 | 21 / 21 |
| `prop_unc_10` | -7.9% | -12.1% | -15.4% | +2.5% | -27 | 112.5 | 0.547 | 25 / 15 |
| `prop_unc_20` | -6.7% | -9.9% | -16.6% | +0.9% | -20 | 113 | 0.593 | 23 / 23 |

Zero type/sign violations in every arm.

**What the sweep says.**

- `prop_unc_05` is the only arm with a positive p90 improvement on **both**
  seeds (+6.3% mean, +0.5% worst seed). It keeps 117.5 of 137 players active,
  entropy is 0.476, `q > 1` falls from 57 to 22 (seed 0) and 51 to 24 (seed 1),
  and `at_bound` falls from 25 to 21 and 29 to 21. Its worst-seed loss cost is
  +1.7%.
- `prop_uni_05` is nearly the same result: +5.8% mean p90, -0.2% worst seed,
  114.5 active, at_bound 22/20. The uncertainty seat has a slightly better
  worst-seed p90 and more active players; the difference is small.
- `prop_none` still has the best mean p90 (+12.5%) and the best at_bound/q
  (9/9, q -29), but it collapses to 7 active players. It is the same
  shutdown-as-balance failure as before.
- ε = 0.10 and 0.20 are not viable. The base seat lets enough movement back in
  that p90 turns negative (-1.7% to -10.1%) even though participation rises.
  ε = 0.05 is the only tested fraction that keeps both stability and
  participation.
- The seed-1 H-off run is a genuinely worse attractor (`loss 1.0444`), and every
  game arm lands near `0.68`. The game prevents a high-loss regime; it does not
  merely shuffle p90. The mean loss delta is therefore dominated by that
  baseline, and the honest per-seed statement is: +1.7% on seed 0, -34% on
  seed 1.

**Interpretation.** The base seat solves the conceptual problem. Selection is
still contestable (the proportional ranking is unchanged), survival is a
physical right (115+ players keep moving), and every stability metric improves
on both seeds. The price is that the mean p90 gain is roughly half of the
collapsed arm's, because the game no longer buys p90 by switching players off.

**Candidate for the next step, not a production default:** uncertainty base
seat, `ε = 0.05`, proportional ranking, no pot. The uniform seat is the control.
The next open questions are the transfer rate, whether the throttle should also
apply to sigma, and the per-player clock.
