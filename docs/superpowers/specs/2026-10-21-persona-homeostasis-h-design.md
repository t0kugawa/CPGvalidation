# PERSONA homeostatic H — design

**Status:** implemented as an opt-in regulator (2026-10-22). The 20 s
production acceptance still fails; no mode is a production default. The
committed experimental best is `HomeostaticConfig.current_best()` —
`fisher` + centered site-zero-sum lift + H delayed to step 500.

The follow-up experimental direction is the conserved-stack game:
`docs/superpowers/specs/2026-10-22-persona-conserved-stack-game-design.md`.

This document defines a homeostatic regulator `H` for PERSONA's natural-gradient
strategy. It is derived from the same Fisher/SNN architecture, but it does not
use absolute per-parameter setpoints. Stability is relational: players compete
for a share of the advantage, and the equilibrium is a calibrated set of player
strategies and distributions, not one preferred parameter vector.

The design preserves the completed recursive Fisher inverse and the existing
`F`, `Xbar`, `G`, `g`, noise-scoring, credit, and sigma semantics.

## 1. Problem

The natural-gradient step

```text
dmu = eta * (F + lambda I)^-1 g
```

has no preferred operating point. When a direction is weakly identified, the
inverse can amplify it until parameters hit their biological bounds. Hard bounds
are currently the only brake. The 2-minute validation run showed the symptom:
large relative drift and saturation in weights, Izhikevich `c`/`d`, and STDP
`tau_plus`. That run is motivation only; this design does not depend on it as
evidence.

The missing ingredient is a regulator that asks a relational question:

> Given the architecture's players and their strategies, is the current
> distribution of advantage claims consistent with what the players can
> actually predict and deliver?

It must answer without saying "this parameter should stay near its default."

## 2. Design decisions

| Decision | Choice |
|---|---|
| Player scoring | Gaussian log score on each player's own standardized credit scalar |
| Constraint enforcement | Hybrid: hard type/sign identity, soft numerical balance |
| H in the update | Fisher metric for within-player geometry + player-level H projection |
| H structure | Player-block graph from the physical map |
| Global variance regulator | All four variants implemented and A/B tested: observed advantage variance, fixed architectural budget, Fisher trace, target band |
| Player prediction target | Its own credit scalar |
| H regulation target | Credit-share balance |
| H timescale | Slow EWMA, slower than task learning |
| Player Gaussian | Forecast mean + declared variance |
| Rollout | Opt-in for the first implementation; existing public behavior unchanged |

These choices are normative for the first implementation. The global-variance
mode is configurable and remains experimental until the A/B results are in.

## 3. Players and the physical map

### 3.1 Players

A player is one strategy-bearing unit, not one parameter:

- **Population player** — one `Population` configuration. Biological size `N_i`
  neurons, model/type, place in the system, parameter set (Izhikevich `a`, `b`,
  `c`, `d`, or Poisson `rate`).
- **Synapse player** — one `Connection` configuration. Biological size `N_i`
  connections, connection type (static excitatory, static inhibitory, or STDP),
  parameter set (`weight`, or `tau_plus`, `tau_minus`/`mu_minus`, `lambda`,
  `alpha`, `mu_plus`, `Wmax`), and the source/target populations it couples.

Let `P_i` be the set of **mean-coordinate parameter sites** of player `i`, and
`M_i = |P_i|`. `M_i` is the number of optimizable coordinates, not the
biological size. `N_i` (neurons or connections) is metadata and a strategy
bound; it is not a credit multiplier. A population has four Izhikevich
coordinates regardless of whether it has 15 or 20 neurons; an STDP player has
six hyper-parameters regardless of its synapse count. This keeps the game about
strategy quality rather than raw size.

A player's parameter set is fixed by its type. A parameter belongs to exactly
one player. Noise producers (`producer:stdp:*`, `producer:poisson:*`) are **not**
players: they remain metric-only Fisher rows, with zero trace and zero credit.

The physical map is built from the registry:

```text
population player:  pop_{population.name}_{param}
synapse player:     conn_{connection.key}_{param}
```

If a parameter name cannot be mapped (a later-created site), it becomes a
singleton `unknown` player with no edges and a warning. This keeps
`FisherMatrix.ensure_names` working without silently dropping sites.

### 3.2 Player type and size

Population type:

- `poisson_generator` → `poisson`;
- inhibitory names (`in_*`, `in1_*`, `r_*`, `ia_*`, `v0d_*`) → `inh`;
- all other neuron populations → `exc`.

Synapse type:

- `stdp` if the connection has a weight recorder or STDP hyper-parameters;
- otherwise `static_exc` or `static_inh` from the sign of the weight parameter
  (or `extra_syn_spec["weight"]`).

Synapse size `N_i` is `registry.synapse_count(key)` when available; otherwise
the expected count from the connection rule, source size, target size, and `p`.
`N_i` is stored in `PlayerSpec.size` and used for diagnostics. It is not used in
the credit aggregation or the H gain.

**Hard type/sign constraints** are the existing `PersonaParam.min/max` bounds:

- static weight stays on its initial side of zero;
- STDP hyper-parameters stay positive and inside their biological ranges;
- population parameters stay inside their model ranges;
- the population model and connection model never change, so a player cannot
  change type.

H never bypasses `PersonaParam.update_mean`, so the hard projection is the same
clip that already protects the host. The implementation adds tests that prove a
large H correction cannot flip a sign or change a type.

### 3.3 Physical graph

The graph has one node per player. Only `registry.population_cfgs` are player
nodes. A node in `registry.population_nodes` without a `Population`
configuration (a relay parrot, muscle readout, or other zero-parameter
environment node) is a **boundary node**, not a player; incidence edges to it are
skipped and logged. A synapse player with no player endpoint becomes a singleton
player, weakly connected by the connectivity regularizer below.

Edges:

1. **Incidence** — for each synapse player `s` from population `a` to population
   `b`, add `(a, s)` if `a` is a player and `(s, b)` if `b` is a player.
2. **Antagonism** — for each target population, incoming connections with
   opposite weight signs add edges between their source populations and between
   their synapse players. This makes E/I balance explicit rather than relying
   only on a shared target.
3. **Reciprocity** — if connections `a → b` and `b → a` both exist, add an edge
   between their synapse players.
4. **Connectivity regularizer** — add a weak complete-graph term so disconnected
   components cannot keep independent constant share errors.

The adjacency is **simple**: if incidence, antagonism, and reciprocity all
propose the same pair, `A_ij = 1` once; duplicate proposals do not accumulate
weight. This keeps `d_max`, `L_sym`, and H's gain independent of how many edge
types happen to coincide. Edge weights are fixed from the architecture (default
`1.0`). They do **not** depend on player sigma or advantage, so a player cannot
weaken H by inflating its variance. The graph operator is a degree-scaled
combinatorial Laplacian, not the
symmetric normalized Laplacian (the latter does not have zero row sums on an
irregular graph):

```text
L = D - A
L_sym = L / max(d_max, 1)
H = L_sym + eps_H (I - 11^T/n_p)
```

where `A` is the weighted adjacency, `D` the weighted degree matrix,
`d_max = max_i degree_i`, `n_p` the number of players, and `eps_H = 1e-3`.
`L` is symmetric positive semidefinite, `L 1 = 0`, and `L_sym` has eigenvalues
at most 2. Isolated nodes have a zero row in `L`; the regularizer is also
symmetric, PSD, and zero-row-sum. Thus `H 1 = 0` and `1^T H = 0` for every
graph, including degree-zero players (for example `BS_E`, `BS_F`, or `BASE_pg`
when their projections start at relay parrots). H is fixed for a given physical
map. An isolated player gets no local Laplacian correction but still receives
the weak global regularizer term.

## 4. The zero-sum game

### 4.1 Player credit

Let `g_mu` be the per-site mean-credit vector from the existing strategy,
`g_mu = G_mu · 1_2`. Raw `g_mu,p` scales roughly as `1/sigma_p^2`, so a raw mean
over sites would be dominated by the tightest coordinate (`a`, `sigma=0.005`,
would swamp `c`, `sigma=5`). The game therefore standardizes each site by a
fixed credit unit before averaging:

```text
g_ref,p = max(RMS of g_mu,p over the H burn-in, g_min)   fixed unit
C_{i,p} = 1/(M_i * g_ref,p)   if p in P_i,   else 0
M_i = |P_i|
c_i = (C g_mu)_i = (1/M_i) sum_{p in P_i} g_mu,p / g_ref,p.
```

`g_ref,p` is a unit of measurement, like `sigma_ref,p`; it is not a preferred
credit level and it does not appear as a target. It is fixed at the end of the H
burn-in. The mean is deliberate: `N_i` is a capacity bound, not a claim
multiplier, so a large pool does not win merely by having more neurons or
synapses. Each site contributes on its own observed credit scale, so a
population's tight `a` coordinate does not dominate its loose `c` coordinate.

The default lift is the adjoint of the same standardization:

```text
g_H,p = u_H,i * g_ref,p        for p in P_i,
```

so `C g_H = u_H` exactly: a player-level correction `u_H,i` changes its credit
`c_i` by `u_H,i`. Zero-sum is defined at the **player** level:
`sum_i u_H,i = 0` and `sum_i c'_i = sum_i c_i`. For this default (`lift_mode =
"player"`), the site-level sum `sum_p g_H,p` is not generally zero because
players have different `M_i` and `g_ref,p`; that is expected, because the
game's unit is player credit, not site credit.

The committed experimental best uses `lift_mode = "site_centered"`:

```text
w_p = g_ref,p - mean_{q in P_i} g_ref,q     for p in P_i
g_H,p = u_H,i * w_p
```

This makes each player's site correction sum to zero, so `sum_p g_H,p = 0`
exactly. It deliberately breaks the player-credit adjoint `C g_H = u_H`; it is
a within-player redistribution, not a player-credit transfer. It was selected
empirically because it lowered seed-0 p90 and bound pressure in the 20 s
comparison, and it remains experimental. The player-level `u_H` is still
zero-sum because `H 1 = 0`. The default remains `"player"`; the preset
`HomeostaticConfig.current_best()` selects `"site_centered"`.

The game is zero-sum in credit shares. `c_i` may be positive or negative; the
shares are normalized with a softmax, so they always sum to one.

### 4.2 Standardized credit and declared variance

`c_i` is already dimensionless (credit divided by its site unit). The game uses
a centered global credit scale and a fixed per-parameter sigma unit:

```text
S_c_raw = std_i(c_i)
S_c = EWMA over tau_S of max(S_c_raw, S_min),   S_c >= S_min
z_i = (c_i - mean_i c_i) / S_c_prev    centered standardized credit
sigma_ref,p = max(sigma_p(0), sigma_min)   fixed unit, not a target
v_i = (1/M_i) sum_{p in P_i} (sigma_p / sigma_ref,p)^2
v_i = clip(v_i, v_min, v_max)
```

`S_c` is the **centered** scale (standard deviation across players), and `z` is
explicitly centered, so `mean_i z_i = 0` at every step. Adding a common credit
level to every player leaves `z`, `pi`, `e`, and `u_H` unchanged; the game
regulates relative shares, not the common drive. `S_c_raw` is the unfloored
standard deviation used by the null-game gate; `S_c` is the floored/EWMA scale
used for normalization. `S_c` is a global unit of account, not a player; no
single player can set it. `S_c_prev` is the value from the previous step, so
standardizing the current credit uses no current-step information beyond `c`
itself. `sigma_ref,p` is only a unit of measurement, so the score is invariant
to a global rescaling of all sigmas. It is not a stability setpoint and does not
appear in the update as a preferred value.

Each player predicts its next standardized credit before observing it. At step
`t`, the score uses `mu_i(t-1)`, the EWMA of standardized credit through
`t-1`; the current `z_i(t)` is scored first and only then folded into the mean:

```text
ell_i(t) = -0.5 * ( log(2 pi v_i(t)) + (z_i(t) - mu_i(t-1))^2 / v_i(t) )
mu_i(t)  = (1 - alpha_m) mu_i(t-1) + alpha_m z_i(t)
```

`ell_i` is a proper log score for the standardized credit with declared
variance `v_i`. For a fixed mean, the score is maximized at
`v_i = (z_i - mu_i)^2`; increasing `v_i` above the realized squared error is
penalized by `log v_i`, while increasing it below that error improves
calibration. Inflating `sigma` therefore does not buy claim once the declared
variance is wider than the player's actual forecast error. This is the formal
version of "a player may inflate its own variance, but earns less until the
distribution actually fits." The mean forecast is causal; it uses only past
standardized credit. The first forecast is established by the H burn-in
protocol in section 4.6.

`v_max` is required. The default is `1e6`; it only prevents numerical overflow,
not a preferred variance.

### 4.3 Shares and share error

The target share is

```text
a_i = softmax(ell_i / T)
```

and the current share is

```text
pi_i = softmax(z_i / T),
```

where `T > 0` is the dimensionless global-variance temperature (section 4.4).
Both are distributions, so

```text
e_i = a_i - pi_i,        sum_i e_i = 0.
```

The error is smoothed on the homeostatic timescale:

```text
e_bar_i <- (1 - alpha_H) e_bar_i + alpha_H e_i,
alpha_H = 1 - exp(-dt / tau_H),
```

with `tau_H` default `5.0 s`, slower than `tau_mem = 2.0 s`.

### 4.4 Global variance regulator

The global variance is a unit of account, not a player. No single player can set
it. Four modes are implemented behind one config flag.

Let `A(t) = mean(advantage_vector(t))`. During the H burn-in (section 4.6),
collect `A` and `V_F = trace(F_mu_mu)/n_sites`, where `F_mu_mu` is the
mean-coordinate block. `V_A0` is the sample variance of `A` over burn-in;
`V_obs` is an active EWMA variance of `A` with time constant `tau_A = 2.0 s`,
initialized to `V_A0`. `V_F0` is the EWMA-stationary extrapolation of `V_F` at
the end of burn-in; `V_F_ewma` is an active EWMA initialized to `V_F0`. If
either reference is zero or non-finite, use `1.0`. Clamp `T` to `[0.1, 10.0]`
unless stated.

1. **`advantage`** — `T = clamp(V_obs / V_A0)`. The task's realized variability,
   relative to its own initial scale, is the unit of account.
2. **`budget`** — `T = 1`. The initial standardized variance unit is held fixed.
3. **`fisher`** — `T = clamp(V_F / V_F0)`. This measures how much score
   information the players currently carry, relative to its own initial scale.
4. **`band`** — `T = clamp(EWMA(V_obs / V_A0, tau_V), 0.5, 2.0)` with
   `tau_V = 5.0 s`. This is the slow target-band controller: the global scale
   may move, but only inside a biological band around the initial scale.

For `fisher`, `T` starts below 1 because `F` is still accumulating after the
strategy warm-up; `V_F0` is the stationary extrapolation, not the burn-in
endpoint.

The active `V_obs` uses the standard EWMA variance recursion. With
`beta_A = exp(-dt/tau_A)` and `A_t = A(t)`:

```text
delta = A_t - mean_A_prev
mean_A = mean_A_prev + (1 - beta_A) * delta
var_A = beta_A * (var_A_prev + (1 - beta_A) * delta^2)
V_obs = max(var_A, 0)
```

`V_F_ewma` uses the same `beta_A`:
`V_F_ewma <- beta_A V_F_ewma + (1 - beta_A) V_F`. The `band` mode uses
`R <- V_obs/V_A0`, `R_ewma <- beta_V R_ewma + (1 - beta_V) R` with
`beta_V = exp(-dt/tau_V)`, initialized to `1.0` at active start; `T` is the
clamped `R_ewma`. All four modes are tested on the same benchmark. The final
default is chosen from the comparison, not assumed now. The development default
is `advantage`.

### 4.5 H dynamics

The player-level homeostatic correction is

```text
u_H = +lambda_H * S_c * H * e_bar,        lambda_H default 1.0.

The development default was 0.1. The frozen synthetic fixture showed that
0.1 is below the closed-loop stability threshold (H on worsened the share RMS
by 13%); an independent direct implementation matched the regulator exactly.
The calibrated default is 1.0, which gives a 31% reduction on the frozen
fixture and margin above the 20% acceptance threshold. The fixture and
threshold were not changed.
```

Here `S_c` is the previous-step EWMA scale (`S_c_prev` in the step pseudocode),
so the current correction uses only information available before the step. The
sign follows from `e = a - pi`: if `e_i > 0`, player `i` is under its target
share, and increasing its credit increases `pi_i`, which reduces `e_i`. A graph
Laplacian has the same sign pattern as the error on an edge (`e = (delta,
-delta)` gives `H e = (2 w/d_max)(delta, -delta)`), so the positive feedback
`+lambda_H S_c H e_bar` moves credit toward the target. The opposite sign would
be anti-homeostatic. `S_c` is the centered credit standard deviation, so `u_H`
is in the same standardized-credit units as `c`.

Because H has zero row sums, `sum_i u_H,i = 0`: the correction is zero-sum at
the player level. It redistributes credit shares along the physical graph; it
never creates player credit in aggregate. H is gated off during the strategy
warm-up and the H burn-in, and it is also gated off when
`S_c_raw < eps_credit` (a null or uniform game cannot be used to manufacture
positive and negative claims from nothing). When gated, `u_H = 0`.

H does **not** rewrite the realized credit state `G`, `c`, `z`, or `pi`. It is a
one-step bias to the credit vector used by the natural gradient:

```text
c' = c + u_H          homeostatic credit target for this step
g_H,p = u_H,i * g_ref,p        for p in P_i
g'_mu = g_mu + g_H
dmu = eta P [g'_mu; g_sigma; 0]
```

The lift is the adjoint of the standardized aggregation, so `C g_H = u_H`: the
player-level credit target changes by exactly `u_H,i`. The next step's realized
`c` comes from `G` and the host response, not from writing `u_H` back into `G`.
The direct unit test therefore checks the projected share

```text
z'_i = (c_i + u_H,i - mean_i c_i) / S_c
pi'_i = softmax(z'_i / T)
```

and verifies that it is closer to `a` than `pi` for a small positive gain. The
closed-loop synthetic game checks the actual next-step share error after the
parameter update. This distinction matters: H regulates the credit allocated to
the step, while the realized advantage remains the environment's outcome.

Zero-sum is player-level: `sum_i u_H,i = 0` and `sum_i c'_i = sum_i c_i`. The
default site-level sum `sum_p g_H,p = sum_i u_H,i sum_{p in P_i} g_ref,p` is not
generally zero because players have different `M_i` and `g_ref,p`; that is
expected for the adjoint lift. With the experimental `lift_mode =
"site_centered"`, each player's site correction sums to zero, so the site-level
sum is zero by construction. Tests for one lift must not be applied to the
other. H acts on **mean credit only**. The sigma rows of `g`, the noise rows,
and the log-sigma update are untouched. This preserves the earlier decision
that sigma stays in the joint Fisher and is updated by its own rule.

The current-best preset also delays H: `h_delay_steps = 500` means no
correction is applied until the absolute strategy step reaches 500. The
regulator still updates `e_bar`, `S_c`, `mu`, and the global-variance state
during the delay; only `u_H` is zero. This is a timing gate, not a second
burn-in. The delay is a response to the path-dependent production failure: H
enabled early selected a worse, higher-inflation attractor.


### 4.6 Startup and H burn-in

H has three stages:

1. **Strategy warm-up** (`steps <= strategy.warmup`): H is inactive and its
   state is not updated. The strategy returns zero deltas as before.
2. **H burn-in** (`h_burnin_steps = 100` steps after warm-up, 1.0 s at
   `dt = 10 ms`): H still returns `u_H = 0`, but it collects the scales used by
   the first active correction. The burn-in uses batch estimates, not an EWMA
   initialized from the first sample:
   - For each burn-in step, store `g_mu(t)`, `A(t)`, and
     `V_F(t) = trace(F_mu_mu(t))/n_sites`.
   - At the end, set
     `g_ref,p = max(sqrt(mean_t g_mu,p(t)^2), g_min)`.
   - Recompute the burn-in standardized credit with the final `g_ref`:
     `c_i(t) = (1/M_i) sum_p g_mu,p(t)/g_ref,p`.
     Set `S_c0 = max(mean_t std_i(c_i(t)), S_min)` and
     `z_i0(t) = (c_i(t) - mean_i c_i(t))/S_c0`; set
     `mu_i0 = mean_t z_i0(t)`.
   - Compute `V_A0` as the sample variance of `A(t)` over burn-in, and
     `mean_A0` as its mean; if `V_A0` is zero or non-finite, use `1.0`.
     Compute `V_F0 = V_F(end) / max(1 - beta^h_burnin, 0.1)`, the
     EWMA-stationary extrapolation of the Fisher trace (F starts at zero and
     grows as `1 - beta^t`); if it is zero or non-finite, use `1.0`.
   - Initialize the active state from these batch values:
     `S_c = S_c0`, `mu = mu_i0`, `mean_A = mean_A0`, `var_A = V_A0`,
     `V_obs = V_A0`, `V_F_ewma = V_F0`. Set `e_bar = 0`.
3. **Active H**: compute `ell`, `a`, `pi`, `e`, update `e_bar`, apply `u_H`.
   `S_c`, `mu`, `V_obs`, and `V_F_ewma` continue their EWMAs initialized from
   the burn-in batch.

All EWMAs use `alpha_x = 1 - exp(-dt/tau_x)`. The burn-in batch avoids the
"EWMA initialized at the first sample is still far from stationary" bias.
`T = 1` during burn-in. The burn-in is deliberately short; it exists to set the
units and the first forecast, not to tune H. If
`h_burnin_steps = 0`, the first active step computes `g_ref`, `S_c`, and `mu`
from the current step, returns `u_H = 0`, and correction starts on the following
step. The burn-in is not counted as a strategy warm-up and does not change
`F`/`Xbar`/`G`.

## 5. Update math

### 5.1 Fisher metric (unchanged)

The existing strategy maintains

```text
F_t = beta F_{t-1} + (1 - beta) s_t s_t^T
P_t = (F_t + lambda I)^-1
```

with the recursive inverse and exact refresh policy already implemented. H does
not modify `F`, `P`, `Xbar`, `G`, or the noise rows.

### 5.2 Full step

For each active strategy step after warm-up and burn-in:

```text
g          = G · 1_2
g[noise]   = 0
c          = C g_mu
c_bar      = mean_i c_i
z          = (c - c_bar) / S_c_prev
S_c_raw    = std_i(c_i)
v_i        = clip(mean_p (sigma_p / sigma_ref,p)^2, v_min, v_max)
ell_i      = -0.5 (log(2 pi v_i) + (z_i - mu_prev,i)^2 / v_i)
update V_obs with A; update V_F_ewma with V_F; compute T
a          = softmax(ell / T)
pi         = softmax(z / T)
e          = a - pi
e_bar      = EWMA(e, tau_H)
u_H        = 0 if S_c_raw < eps_credit or steps < h_delay_steps
             else +lambda_H S_c_prev H e_bar
g_H,p      = u_H,i * g_ref,p              for p in P_i   # lift_mode=player
             or u_H,i * (g_ref,p - mean_q g_ref,q)       # site_centered
g'_mu      = g_mu + g_H
g'         = [g'_mu; g_sigma; 0_noise]
dmu        = eta P g'
mu         = EWMA(mu_prev, z, tau_m)      # after scoring, never before
S_c        = EWMA(S_c_prev, max(S_c_raw, S_min), tau_S)
```

During the H burn-in, H collects the batch scales in section 4.6 and returns
`u_H = 0`, so `g'_mu = g_mu` and `dmu` is the ordinary natural-gradient step.
During strategy warm-up, H is inactive and `dmu` is zero as before.

The hard projection is then the existing host update:

```text
theta <- Proj_S(theta + dmu),
```

where `Proj_S` is `PersonaParam.update_mean`'s clip to `[min, max]` and the
player's type is structurally fixed. If `lambda_H = 0`, the regulator
short-circuits to `g'_mu = g_mu` exactly and the step is the old step (no new RNG
calls).

### 5.3 Why this is a Fisher metric plus a projection

`F` remains the metric that decides how a player-level credit change is spread
across its parameters. H is a projection/diffusion in the player subspace: it
takes the share error and projects it onto the graph of covaried players. The
constrained form was chosen over adding a dense `C^T H C` term to `F` because it

- preserves the recursive inverse exactly;
- keeps the per-step cost at the existing `O(n^2)` natural-gradient product;
- makes the zero-sum property explicit at the player level
  (`sum_i u_H,i = 0` and `C g_H = u_H`);
- cannot be weakened by a player changing its sigma.

Adding a low-rank `C^T H C` term directly to F is a possible follow-up, but it
would require rebuilding the inverse every step or carrying a Woodbury
correction. That is not part of the first implementation.

### 5.4 Timescales

| Quantity | Default | Role |
|---|---|---|
| `dt` | 10 ms | strategy step |
| `beta` | 0.995 | Fisher/trace/credit memory (`tau_mem = 2 s`) |
| `tau_m` | 2.0 s | player credit forecast |
| `tau_S` | 2.0 s | global credit scale `S_c` |
| `tau_A` | 2.0 s | observed advantage variance |
| `tau_H` | 5.0 s | share-error EWMA; slower than task learning |
| `tau_V` | 5.0 s | band-mode global variance |
| `h_burnin_steps` | 100 | H batch-scale burn-in after strategy warm-up |
| `lambda_H` | 1.0 | H gain, dimensionless (calibrated; see §4.5) |
| `eps_H` | 1e-3 | graph connectivity regularizer |
| `eps_credit` | 1e-9 | raw-std null-game gate |

H is applied every step, but its error state moves on `tau_H`; the game cannot
chase step-level noise.

## 6. Integration

### 6.1 New module

Add `persona/homeostasis.py`:

- `PlayerSpec` — name, kind, parameter names, `M_i`, biological size `N_i`,
  type, source/target, covariance metadata.
- `PhysicalMap` — builds players and edges from `Registry`, exposes
  `player_of`, `player_variance(sigmas)`, and the degree-scaled combinatorial
  graph operator `H`. It also provides
  `PhysicalMap.from_specs(player_specs, edges)` for registry-free synthetic
  tests.
- `GlobalVarianceRegulator` — one small state machine for the four modes.
- `HomeostaticRegulator` — owns `g_ref`, `S_c`, `mu`, `e_bar`, `T`, H, config,
  and diagnostics. `adjust_credit(g_mu, sigmas, advantage_scalar, F, steps)`
  maps site credit to player credit and returns the corrected mean-credit
  vector. The shared core is
  `step_players(c, sigmas, advantage_scalar, F, steps) -> u_H`, which the
  synthetic benchmark can call directly. Both paths update the same state;
  `adjust_credit` additionally applies the `g_ref` lift. `g_ref` lives here,
  not in `PhysicalMap`, because it is a burn-in scale.

No changes to `persona/parameter.py` are required. The existing bounds are the
hard projector.

### 6.2 `FisherMatrix`

Add optional constructor arguments:

```text
homeostasis: HomeostaticRegulator | None = None
dt_s: float = 0.01
```

In `step`, after `g = G.sum(axis=1)` and the noise-row zeroing:

```python
if self.homeostasis is not None:
    g[:m] = self.homeostasis.adjust_credit(
        g[:m], sigmas, float(np.mean(A)), self.F, self.steps)
```

The sigma rows `g[m:2*m]` and noise rows are not changed. If no regulator is
configured, the code path is unchanged.

`ensure_names` must notify the regulator of new site names; unknown sites get
singleton players (section 3.1). The regulator rebuilds H lazily when the player
set changes. A site that first appears after burn-in gets
`g_ref,p = max(|g_mu,p|, g_min)` and `sigma_ref,p = max(sigma_p, sigma_min)` on
its first appearance; existing sites keep their burn-in units.

### 6.3 `Agent`

The public entry point is `Agent(homeostasis: HomeostaticConfig | None = None)`.
`None` or `HomeostaticConfig(enabled=False)` means no regulator, no physical-map
build, and no `history["homeostasis"]`; existing behavior is unchanged. If
`enabled=True`, `Agent` builds the map after `refresh_params()` and constructs
the `HomeostaticRegulator` with the config. It passes the regulator to the
default `FisherMatrix`; an externally supplied strategy may attach one through
`strategy.homeostasis = regulator`. `Agent.step` passes `dt_s` and logs
diagnostics under `history["homeostasis"]`.

`FisherMatrix(homeostasis: HomeostaticRegulator | None = None)` accepts a
prebuilt regulator for direct strategy tests. It does not read the config
itself. The benchmark enables each mode with a fresh
`HomeostaticConfig(enabled=True, global_variance_mode=mode)` and a fresh
regulator; baseline runs use `homeostasis=None`.

`Agent.refresh_params()` calls the regulator's `ensure_params` when new sites
appear. It does not change the existing `F`/`Xbar`/`G` behavior.

### 6.4 Diagnostics

At minimum, log per step:

```text
stage (warmup|burnin|active), mode, T, V_obs, V_A0, V_F, V_F0,
S_c, c, z, mu, v, ell, a, pi, e, e_bar, projected_pi,
u_H, lambda_H, tau_H, correction_norm, gate, skipped_endpoints
```

These go into `Agent.history["homeostasis"]`. HDF5 export is optional and must
be additive: if the writer does not know the group, the run still succeeds and
the strategy semantics are unchanged. The benchmark reads the history directly.

### 6.5 Compatibility

- `homeostasis=None` (the default) reproduces the current behavior exactly.
- `lambda_H = 0` short-circuits to the old credit exactly.
- `F`, `Xbar`, `G`, noise scoring, credit, sigma updates, and the recursive
  inverse are untouched.
- `tinycpg/`, `memcpg/`, and root `persona.py` are not modified.
- Existing tests must pass without changing their expected numeric outputs.

## 7. Configuration

Suggested defaults for the first implementation:

```text
enabled: false                     # opt-in for the first implementation
lambda_H: 1.0                      # calibrated on the frozen synthetic fixture
tau_H_s: 5.0
tau_m_s: 2.0
tau_S_s: 2.0
tau_A_s: 2.0
tau_V_s: 5.0
h_burnin_steps: 100
global_variance_mode: "advantage"  # advantage|budget|fisher|band
lift_mode: "player"                # player|site_centered
h_delay_steps: 0                   # absolute strategy steps before H acts
T_min: 0.1
T_max: 10.0
band_lo: 0.5
band_hi: 2.0
v_min: 1e-8
v_max: 1e6
S_min: 1e-12
g_min: 1e-12
eps_H: 1e-3
eps_credit: 1e-9
```

`enabled: false` is deliberate. The first implementation is opt-in so existing
Agent tests, validation runs, and numeric outputs are unchanged. The benchmark
and H acceptance runs enable it explicitly. Making H the production default is a
separate decision after the A/B comparison.

The committed experimental preset is:

```python
HomeostaticConfig.current_best()
# enabled=True, global_variance_mode="fisher",
# lift_mode="site_centered", h_delay_steps=500, lambda_H=1.0
```

It is a named opt-in preset, not the generic default and not a production
default. The generic defaults stay `advantage`/`player`/no delay so existing H
tests and the original parity contract remain explicit.

## 8. Tests

### 8.1 Unit tests — physical map

- Every optimizable parameter maps to exactly one player; noise producers are
  excluded.
- Population/synapse biological sizes `N_i` and parameter counts `M_i` match the
  registry; `M_i` is used for credit and `N_i` is metadata only.
- Population/synapse types match the registry.
- Incidence edges exist for every connection whose endpoint is a player;
  non-player relay/muscle endpoints are skipped and reported.
- A synapse player with no player endpoint becomes a singleton, not a crash.
- Antagonism edges exist for opposite-sign connections to the same target;
  reciprocal edges exist.
- Duplicate edge proposals form a simple graph (`A_ij = 1` once), so H does not
  change when an incidence and an antagonism edge hit the same pair.
- Unknown parameters become singleton players; `ensure_names` preserves the map.
- H is symmetric, PSD, has zero row sums, and is fixed for a given map.
- With isolated players (degree zero), `H 1 = 0` and `1^T H = 0`; the isolated
  row of `L = D - A` is zero and the `eps_H` regularizer is the only coupling.

### 8.2 Unit tests — score and shares

- `S_c` and `z_i` are finite; scaling all credit by a constant leaves `z`
  unchanged; adding a common credit level to every player also leaves `z`
  unchanged because `S_c` is centered.
- `g_ref` standardizes each site before aggregation, so a tight-sigma site does
  not dominate a loose-sigma site.
- `v_i` is dimensionless; scaling all sigmas and `sigma_ref` by a constant
  leaves `v_i` unchanged.
- The Gaussian log score is proper: for a fixed mean, the optimum over `v` is
  the true forecast variance; over-dispersion lowers the score; under-dispersion
  lowers it when the residual is large.
- At fixed `z` and `mu`, `ell` is maximized at `v = (z-mu)^2`; increasing `v`
  above the realized squared error lowers the score/share, while increasing it
  below that error improves calibration. The test covers over-dispersion, not
  all inflation.
- `a` and `pi` are distributions; `sum(e) = 0`; `T` clipping works.
- The forecast mean uses only past standardized credit: at step `t`, changing
  `z(t)` must not change `ell(t)` through `mu`; `mu` updates after scoring.
- `S_c` uses `S_c_prev` for the current `z` and updates after; `tau_S` controls
  the rate.
- The H burn-in runs `h_burnin_steps`, returns `u_H = 0`, sets `V_A0`/`V_F0`,
  and initializes `mu` without a first-step transient.
- All four global-variance modes produce finite `T`; band mode stays inside
  `[0.5, 2.0]`.

### 8.3 Unit tests — H

- Sign test: with `e = (delta, -delta)` and `H` from a two-player edge,
  `u_H = +lambda S_c H e` makes the projected share
  `softmax((c + u_H)/S_c/T)` closer to `a` than `softmax(c/S_c/T)`; the opposite
  sign increases the projected error.
- H is a one-step credit bias: it does not mutate `G`, `c`, `z`, or `pi`.
- Zero-sum is player-level: `sum_i u_H,i = 0`, `sum_i c'_i = sum_i c_i`, and
  `C g_H = u_H` for the default `"player"` lift. The site-level `sum_p g_H,p`
  need not be zero for that lift and is not used as a test.
- The experimental `"site_centered"` lift makes each player's site correction
  sum to zero (and therefore the whole site correction sums to zero), while
  single-site players receive no correction.
- `h_delay_steps` suppresses `u_H` and the returned correction before the
  absolute step threshold, while still updating the regulator state.
- `HomeostaticConfig.current_best()` is `fisher` + `site_centered` + delay 500,
  remains opt-in, and is not an acceptance pass.
- The null/uniform-game gate returns `u_H = 0` when `S_c_raw < eps_credit`.
- H diffusion reduces the closed-loop synthetic two-player share error.
- A disconnected graph is connected by the `eps_H` regularizer.
- `tau_H` EWMA converges at the expected rate; a larger `tau_H` is slower.
- `lambda_H = 0` produces exactly zero correction.

### 8.4 Unit tests — hard constraints

- A large positive H correction cannot make an inhibitory weight positive.
- A large negative H correction cannot make an excitatory weight negative.
- STDP hyper-parameters stay positive and inside bounds.
- Population parameters stay inside bounds.
- Connection/population types are unchanged by a step.

### 8.5 Integration tests — strategy

- `FisherMatrix` with `homeostasis=None` matches the current step exactly.
- `FisherMatrix` with `HomeostaticRegulator(lambda_H=0)` produces exactly the
  old deltas.
- With H enabled, deltas are finite; `P @ g` still matches the exact solve at
  refresh; the recursive inverse refresh policy is unchanged.
- Sigma rows and noise rows are not modified by H.
- Warm-up still returns zero deltas.
- H burn-in returns the ordinary natural-gradient deltas and no correction;
  active H starts only after burn-in.
- H never writes into `G`; the next step recomputes `c` from `G`.
- `ensure_names` with a new site does not crash and maps the new site.
- All four modes run; diagnostics have the expected keys.

### 8.6 Integration tests — Agent

- A fake registry builds the expected player map.
- Agent warm-up applies no mean/sigma updates.
- Agent history records homeostasis diagnostics when enabled.
- Hard bounds hold after an Agent step with an aggressive H gain.
- `Agent(homeostasis=None)` is the default and leaves the old history/behavior
  unchanged; existing Agent tests pass without edits.

### 8.7 Regression and performance

- Full existing suite: all 173 tests pass.
- Fast e2e harness still passes.
- H overhead in the strategy step is at most 20% at `n = 458`; the recursive
  inverse's stable-phase speedup over exact solve remains at least 1.0x.
- No new RNG calls when H is disabled.

## 9. A/B benchmark and acceptance

Add `bench_homeostasis.py` (or extend the existing harness) with three layers:

1. **Synthetic game** — canonical fixture frozen before H tuning: 12 players,
   one scalar credit each, a ring graph with two chords built by
   `PhysicalMap.from_specs`, and
   `c_{t+1} = c_t + 0.25 u_H,t + 0.005 sin(2 pi t / 200 + i) + 0.02 xi_t`,
   with `xi` from fixed seed `20261021`, for 1000 steps (10 s at `dt=10 ms`).
   Use `HomeostaticRegulator.step_players(c, sigmas, A=1.0, F=I, steps)` with
   `global_variance_mode="budget"` (`T=1`) so the synthetic game isolates the
   score/share/H math; the four global-variance modes are tested in the fast
   smoke and e2e layers. H off sets `u_H = 0`. The fixture and thresholds are
   committed before tuning; changing them after seeing results is not allowed.
   Measure the RMS closed-loop share error `||e||` over the last 500 steps, the
   projected share error, and a loss proxy. This isolates the math, runs in
   seconds, and is long enough to exercise `tau_H=5 s`.
2. **Fast smoke** — the existing 400-step/4 s fast path, one run per mode. This
   checks finiteness, diagnostics, and zero type/sign violations. It is **not**
   the 20% acceptance run; 4 s is less than one `tau_H`.
3. **H acceptance** — 2000 steps/20 s, two paired seeds (same reset RNG stream
   for baseline and each H mode), baseline H off plus the four H modes. This is
   the run that exercises the homeostatic timescale and the deferred production
   smoke. Evaluate on the last 1000 **control steps** (steps 1001–2000; not
   pose substeps), after the H transient.

   The H acceptance uses the same production `t2xcpg` leg model as the
   2-minute validation run and the existing control loop at `dt = 10 ms`,
   `lr_mu = 0.03`, `tau_mem_s = 2.0`, and `realize_every_k = 1`, with 2000
   `Agent.step` calls per run. Use two paired
   seeds, `0` and `1`: reset the PERSONA RNG stream and the environment seed
   for baseline and for each H mode. The 400-step fast-path sizes are smoke
   only and are not used for the 20% acceptance.

   Metrics:
   - `loss_eval`: mean loss over the eval window; `loss_full`: mean over the
     whole active window, reported for context. The 10% rule uses `loss_eval`.
   - `phase_agreement`: fraction of eval steps where the run's phase label equals
     the harness's reference phase label. If the harness has no independent
     reference, use the H-off baseline phase sequence and label the result
     "vs baseline."
   - Primary inflation metric:
     `r_p = max_{t in eval} |theta_p(t)-theta_p(0)| / (max_p-min_p)`, where
     `max_p`/`min_p` are the `PersonaParam` bounds. If a bound is infinite, use
     `denom_p = max(|theta_p(0)|, 1.0)` and mark the parameter as
     range-normalized-by-magnitude. Report median, p90, max, and count
     `r_p > 0.25`.
   - Secondary metric:
     `q_p = max_{t in eval} |theta_p(t)-theta_p(0)| / max(|theta_p(0)|, eps)`,
     reported as count `q_p > 1` (the earlier 71/269 observation). Final-value
     versions of both metrics are reported as context, not used for pass/fail.
   - Number of parameters at bounds, type/sign violations (must be zero),
     share-error trajectory, and step-time overhead.

   Paired comparison: report each seed and the mean. The p90 `r_p` improvement
   is accepted when the mean over seeds is at least 20% and no seed worsens by
   more than 10%. If seed variance is high, add a third paired seed before
   deciding.

Artifacts go to `benchmarks/homeostasis_h/` as CSV, XLSX, and PNG. They remain
untracked until the user asks to commit them. `bench_homeostasis.py` writes the
long-form share trajectory, the combined workbook, and the summary figures.
Because the acceptance runs are paired, each H mode is built with the same
`seed` as its H-off baseline; the summary records `q_growth_max` per seed (the
rule is "does not grow", not an average that can hide a regression). The
default production size preset is the one used by `run_validation.py`
(`n_rg=10, n_mn=8, n_int=8, n_aff=6, n_bs=8`); `--sizes final` selects the full
`n_*=100/100/50/100/100` preset for a longer run. The acceptance harness
chooses a passing mode from the comparison; it does not require all four modes
to clear the 20% bar, but it does require all four to be finite and the band
mode to remain in `[0.5, 2.0]`.

Acceptance for the first implementation:

- H off reproduces the baseline exactly.
- All hard type/sign checks pass; zero violations.
- Synthetic closed-loop RMS share error falls by at least 20% versus H off on
  the frozen fixture; 50% is the target.
- H acceptance mean p90 `r_p` falls by at least 20% versus H off without a
  `loss_eval` increase above 10%; the `q_p > 1` count does not grow.
- All four modes are finite; the band stays in its band.
- Existing tests and the recursive-inverse performance check pass.
- The final default mode and whether H becomes production default are chosen
  from the comparison and recorded in the design/ledger; no mode is declared
  "stable" a priori.

### Result (2026-10-21)

The first production acceptance **failed**. H off was deterministic (two
identical runs), all modes were finite, all had zero type/sign violations, and
band mode stayed in `[0.5, 2.0]`. But every mode increased mean p90 `r_p`
versus H off (seeds 0/1):

| mode | mean p90 change | seed 0 p90 | seed 1 p90 | mean loss change | worst q growth |
|---|---:|---:|---:|---:|---:|
| advantage | -2.92% | .5688 | .5934 | +10.79% | +4 |
| budget | -4.75% | .5771 | .6063 | +6.10% | +7 |
| fisher | -1.89% | .5416 | .6154 | -0.84% | +2 |
| band | -3.60% | .5693 | .6015 | +5.97% | +8 |

H off was .4990 / .6461. H modes also put more parameters at bounds on seed 0
(off 23; H 34-43). No mode passes; `fisher` is only the least harmful.

Follow-up checks isolated the failure to production size/dynamics, not pairing
or sign/type handling:
- Synthetic (12 players) and fast sizes (6/5/5/3/5) improve with H.
- A production-size window trace reproduced the failure: H off settles to p90
  about .485-.495 after step 800, while fisher stays at .522-.543; the H
  correction norm remains about .18-.32 and `e_bar` stays nonzero.
- Lowering `lambda_H` does not rescue it: fisher at 0.1/0.3/0.5/1.0 gives p90
  .559/.580/.579/.542 versus off .499, with 40/32/31/34 parameters at bounds
  versus 23. This is not a scalar over-gain.
- A switch test showed path dependence: fisher on for steps 0-1000 then off
  ends at p90 .548 and 34 at-bound parameters; off for 0-1000 then fisher on
  ends at p90 .500 and 26 at-bound parameters, essentially the H-off result.
  The early H transient steers the production system to a different,
  higher-inflation attractor. Turning H off later cannot recover the baseline,
  and applying H only late does not reproduce the damage. The root cause is the
  early-phase correction selecting a worse attractor, not an ongoing gain or a
  sign/type violation.

A follow-up probe tested a normalized site-level zero-sum lift,
`g_H,p = u_i g_ref,p / S_i`, which makes `sum(g_H)=0` exactly. It achieved the
site-sum invariant but worsened the production metrics (seed 0 p90 .571 vs
current .542 and off .499; seed 1 .667 vs current .615 and off .646). Loss,
median inflation, q-count, and seed-1 at-bound pressure also worsened, so
nonzero site sum alone is not the primary failure mechanism. That lift is
rejected and the player-level lift is retained. The probe is recorded in
`benchmarks/homeostasis_h/site_zero_sum_probe.json`.

A broader production screening (fisher, lambda 1.0, seeds 0/1, 2000 steps)
tested scaled, centered, equal, and global-centered site lifts; 500-step ramp
and delay; high-pass error; bounds cap; population-only and no-STDP filters;
and centered+delay/ramp/high-pass/bounds combinations. The current lift was
reproduced exactly. The best mean candidate was centered site zero-sum (.5394
seed 0, .5759 seed 1; ~2.6% p90 improvement, 15/13 at bounds), still far short
of 20%. Centered+delay was best on seed 0 (.4774) but regressed seed 1 (.6585).
No tested variant passes; the full table is in
`benchmarks/homeostasis_h/lift_experiments_summary.md`. Simple lift and timing
changes do not compose into an acceptance pass. Role-aware targets, a full
site-level H graph, or a different H objective remain untested design changes.

A direct real 20 s comparison of H off, current H_opt, and centered+delay
(seed 0/1, production sizes, concurrent) confirmed the pattern: H_opt p90
.5416/.6154 (mean -1.9% vs off), centered+delay .4774/.6585 (mean +1.2%).
Centered+delay cuts at-bound parameters to 13/29 and achieves exact site-sum
zero, while H_opt has lower share RMS. Neither reaches 20%; H_opt is -8.5% on
seed 0 and centered+delay is -1.9% on seed 1. H remains opt-in. Details:
`benchmarks/homeostasis_h/real_compare_20s.md`. The next test should be a
direct trust-region/oracle check of whether a 20% p90 reduction is feasible
under the loss budget before changing the core objective.

**Decision:** H remains opt-in; no mode is promoted to production default. The
code default mode stays `advantage` for explicitly opted-in experiments, and no
mode is declared stable. The artifacts are in `benchmarks/homeostasis_h/`.
A design change (for example, a decaying/high-pass correction or a target that
vanishes at the production optimum) and a fresh frozen acceptance run are
required before reconsidering promotion.

### 9.1 Current-best variant (2026-10-22)

`HomeostaticConfig.current_best()` is the committed experimental best:

```text
global_variance_mode = "fisher"
lift_mode            = "site_centered"
h_delay_steps        = 500
lambda_H             = 1.0
h_burnin_steps       = 100
```

The lift is `g_H,p = u_H,i (g_ref,p - mean_{q in P_i} g_ref,q)`. It is a
within-player redistribution: the site correction sums to zero, single-site
players receive no correction, and the player-level `u_H` remains zero-sum.
The delay suppresses correction before absolute step 500 while still updating
`e_bar`, `S_c`, `mu`, and the variance state. This is the variant measured in
the real 20 s comparison: seed 0 p90 `.4774` and seed 1 `.6585` versus H off
`.4990`/`.6461`; mean improvement `+1.2%`, at-bound `13/29` versus `23/40`.
It does not pass the 20% acceptance bar and `q_growth_max` is `+2`. It is a
preset for further experiments, not a stable mode.

### 9.2 Oracle feasibility test (next)

Before changing the core objective, test whether a 20% p90 reduction is
reachable at all with the loss budget. The oracle is a direct parameter-space
trust region around each parameter's initial mean:

```text
theta0,p = theta_p(0)
radius_p = R * (max_p - min_p)
theta_p(t) <- clip(theta_p(t), theta0,p - radius_p, theta0,p + radius_p)
```

Apply it after every mean update, leave sigma untouched, and run the same
2000-step production path, seeds 0/1, with no H. Test `R` values that bracket
the target (for example 0.40, 0.35, 0.30). Compare p90, `loss_eval`, at-bound
count, and `q_p > 1` against the H-off baseline.

- If some `R` reaches 20% p90 improvement with `loss_eval` within 10%, the
  objective is feasible and the H mechanism/objective needs redesign.
- If no `R` can do it, the p90 target and the loss budget are in tension; the
  core objective or the acceptance metric must be revisited.

The oracle is a feasibility probe, not a proposed production controller.

### 9.3 Oracle result (2026-10-22)

The trust-region probe completed at R = 0.40/0.35/0.30, seeds 0/1. Baseline
H off was `.4990`/`1.1435`/q66/bound23 (seed 0) and
`.6461`/`1.2807`/q80/bound40 (seed 1).

| R | p90 seed 0/1 | mean p90 improvement | loss seed 0/1 | mean loss delta | q seed 0/1 | bound seed 0/1 |
|---:|---|---:|---|---:|---|---|
| .40 | .3951/.3954 | 29.8% | 1.2910/1.4269 | +12.2% | 60/51 | 38/56 |
| .35 | .3491/.3455 | 38.3% | 1.4453/1.4732 | +20.7% | 56/56 | 34/45 |
| .30 | .3000/.2948 | 47.1% | 1.3820/1.2648 | +9.8% | 62/56 | 32/13 |

Only 2.7–5.6% of mean updates actually hit the radius. R=.30 is the only arm
inside the 10% mean loss budget, and it is inside by only 0.2 percentage points;
seed 0 loss is still +20.9%. All radii have zero range-fallback parameters.

The result changes the interpretation of the failed H run: the p90 target is
**controllable** by a direct parameter-space constraint, so the current H
mechanism/objective is the problem, not the feasibility of the metric. It does
not show that an online, causal controller can match R=.30 at acceptable
per-seed loss. The next mechanism should act on parameter displacement or a
tail trust region, possibly per player, rather than only redistributing credit
shares. A robust acceptance still needs seed-0 loss below +10%, not merely a
mean inside the budget.

Details: `benchmarks/homeostasis_h/oracle_trust_region_20s.md`.

## 10. Implementation order (TDD)

1. `PhysicalMap` tests, then implementation.
2. Score/share/unit-normalization tests, then `HomeostaticRegulator` math.
3. H graph/diffusion tests, then `u_H` and `g_H`.
4. `FisherMatrix` integration tests, then optional regulator path.
5. `Agent`/`Registry` wiring and diagnostics.
6. Four global-variance modes and their tests.
7. Synthetic and fast-smoke benchmarks; then the 20 s paired-seed acceptance.
8. Full suite, performance check, docs/ledger update.
9. Record the chosen default mode and the opt-in/default decision.

The design was approved on 2026-10-22 and implementation followed this order.

## 11. Risks and open questions

- **Type inference from names** is fragile. Mitigation: explicit prefix map,
  registry metadata where available, and a test that asserts every player type.
- **Temperature sensitivity**: the log score can be sharp. Mitigation: clip T,
  log T, and test all four modes.
- **H can fight learning** if `lambda_H` is too large. Mitigation: slow `tau_H`,
  small default gain, synthetic and e2e A/B.
- **Credit projection may not fully stop absolute inflation** if a player
  genuinely improves its forecast while growing. That is allowed by design; the
  hard bounds and the log-score variance penalty remain the backstop. If the
  benchmark shows residual saturation, the follow-up is a low-rank H term in F,
  not an absolute setpoint.
- **H is a one-step credit bias, not a rewrite of realized credit.** The direct
  sign test uses projected shares; the 20% acceptance (50% target) uses the
  closed-loop synthetic plant. A test that expects the same-step realized `c` to
  change is testing the wrong contract.
- **The default H zero-sum is player-level; the current-best lift is
  site-level.** The graph operator is a degree-scaled combinatorial Laplacian
  (`H 1 = 0`). The default lift is `g_H,p = u_H,i g_ref,p`, so `C g_H = u_H`.
  The experimental `"site_centered"` lift instead subtracts each player's mean
  `g_ref`, so it does not transfer player credit and its site correction sums
  to zero. Do not apply the non-zero site-sum assertion to the site-centered
  mode, or the adjoint assertion to it either.
- **Burn-in length** is fixed at 100 steps. If a mode is still transient at the
  active boundary, report it; do not silently extend the burn-in per mode.
- **Isolated players** are handled by zero Laplacian rows plus the `eps_H`
  regularizer; H still sums to zero on every row.
- **Non-player relay endpoints** are skipped, so some sensory synapse players
  may be less locally coupled than the full anatomical graph. The diagnostics
  report skipped endpoints; a relay-parent map is a possible follow-up.
- **Normalization units**: `sigma_ref`, `g_ref`, `S_c`, `V_A0`, and `V_F0` are
  units, not preferred values. The tests check scale invariance. If the user
  objects to using the initial sigma or burn-in scale as a unit, the
  alternative is an online robust scale; that would be a design change, not a
  tuning knob.
- **`a` is state-dependent**: it is the score-implied target, not a fixed
  setpoint. The direct sign test freezes `a`; the closed-loop test recomputes it.
  H is a slow, small-gain controller; no global stability theorem is claimed.
- **Disconnected graph components**: the `eps_H` regularizer prevents a player
  from escaping H by sitting in a separate component.
- **H is opt-in** for the first implementation. Making it the production default
  is a separate approval after the A/B.

## 12. Approval checklist

Approval confirmed on 2026-10-22:

- [x] the player definition and physical map in section 3, including `M_i` vs
      `N_i` and skipped non-player endpoints;
- [x] the standardized Gaussian log score, causal forecast mean, declared
      variance, shares, and the 100-step H burn-in in section 4;
- [x] H as a degree-scaled combinatorial player graph with zero rows for
      isolated players, a positive player-level zero-sum credit bias, not an
      absolute parameter prior, in section 5;
- [x] the four global-variance modes, all tested before a default is chosen;
- [x] hard type/sign preservation via existing bounds;
- [x] H opt-in for the first implementation;
- [x] the test plan and the exact 20 s paired-seed acceptance metrics in
      sections 8–9;
- [x] implementation after approval; the current-best preset is documented in
      section 9.1 and remains opt-in.
