# PERSONA — the mathematics, as implemented

- **Status:** reference for the working tree on 2026-10-31.
- **Scope:** every mathematical rule in `persona/`: the core strategy, the σ rule, the three optional allocation layers (H, the conserved-stack game, and Evo), and the validation metrics.
- **Rule of thumb:** the code is authoritative. This file explains what the code does, in the order it does it, and why.

---

## Contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [Objects and notation](#2-objects-and-notation)
3. [The control loop, step by step](#3-the-control-loop-step-by-step)
4. [The core strategy](#4-the-core-strategy)
5. [The σ rule](#5-the-σ-rule)
6. [Homeostatic regulator H](#6-homeostatic-regulator-h)
7. [The conserved-stack game](#7-the-conserved-stack-game)
8. [PERSONA-Evo](#8-persona-evo)
9. [Validation mathematics](#9-validation-mathematics)
10. [How the layers compose](#10-how-the-layers-compose)
11. [Current state and evidence](#11-current-state-and-evidence)
12. [A worked example](#12-a-worked-example)
13. [Glossary and code map](#13-glossary-and-code-map)
14. [What the framework deliberately does not do](#14-what-the-framework-deliberately-does-not-do)

---

## 1. The idea in one page

PERSONA personalizes a spiking central pattern generator (CPG) that drives a simulated rat hindlimb. It does not have a gradient of the plant. Instead it uses **online stochastic search with memory**:

1. **Wiggle every tunable parameter** a little.
2. **Watch the tracking error** on the next steps.
3. **Move the parameter mean** in directions that recently preceded a better-than-usual error.
4. **Remember the past with exponential forgetting**, so no window has to be stored and old evidence fades smoothly.
5. **Correct for parameter scale** with a Fisher metric, so a parameter with σ = 0.005 and one with σ = 5 get comparable steps.

That is the core strategy. Three optional layers sit on top and answer a different question: *who gets to move, and who earns the credit?*

- **H (homeostasis)** redistributes credit across a physical player graph, zero-sum.
- **The game** gives each player a conserved share of the advantage and lets the shares move by a replicator rule.
- **Evo** chooses the player partition itself, by running the game and pruning players that earn nothing.

A picture of one step:

```text
  μ, σ ──sample──▶ x̂ ──realize──▶ NEST CPG ──drive──▶ MuJoCo ──▶ ankle angles
   ▲                                                                  │
   │                                                                  ▼
   └──── Δμ = η P g ◀──── F, X̄, G ◀──── scores s, a ◀──── advantage A = b(phase) − E
```

The rest of this file unpacks each arrow.

---

## 2. Objects and notation

### 2.1 The four objects

| Object | What it is | Where it lives |
|---|---|---|
| **Site** | One optimizable scalar parameter: an Izhikevich `a,b,c,d`, a Poisson rate, a static weight, or an STDP hyper-parameter. State is `(μ_s, σ_s)`. | `PersonaParam` |
| **Player** | A strategy-bearing unit: one population or one connection. It owns one or more sites. | `PlayerSpec` |
| **Segment** | One registry object: a parameterized population or projection. Used by Evo. A player is a group of segments. | `Segment` |
| **Phase** | The gait phase of a limb, `"stance"` or `"swing"`. Used for the advantage baseline and for the validation gate. | `phase_L`, `phase_R` |

A **site** is the unit of optimization. A **player** is the unit of allocation (H and the game). A **segment** is the unit of partitioning (Evo). They are nested:

```text
site ⊂ player = group of segments
```

### 2.2 Symbols

| Symbol | Meaning |
|---|---|
| `t` | Control step, `dt = 10 ms` in production |
| `β` | Forgetting factor, `β = 1 − dt/τ_mem`, clipped to `[0, 0.999999]` |
| `τ_mem` | Memory time constant, default `2.0 s` → `β = 0.995` at `dt = 10 ms` |
| `μ_s, σ_s` | Mean and standard deviation of site `s` |
| `ε_s` | Standard normal draw, one per site per step, from the site's own RNG |
| `x̂_s` | Sampled/applied value: `clip(μ_s + σ_s ε_s)` |
| `Δθ_s` | Perturbation: `x̂_s − μ_s` |
| `z_s` | Standardized perturbation: `Δθ_s / σ_s` |
| `E_i` | Error of output component `i` (one per limb): `\|target_i − output_i\|` |
| `b_i(phase)` | Moving-average error baseline for component `i` in that phase |
| `A_i` | Advantage: `b_i(phase) − E_i` |
| `s` | Score vector (one entry per channel) |
| `a` | Trace vector (one entry per channel) |
| `F` | Forgetting Fisher matrix, `n × n` |
| `X̄` | Forgetting trace, length `n` |
| `G` | Forgetting credit matrix, `n × 2` |
| `g` | Credit vector: `G · 1₂`, with noise rows zeroed |
| `P` | Inverse Fisher, `(F + λI)⁻¹` (approximately, between refreshes) |
| `λ` | Ridge, default `1e-3` |
| `η_μ, η_σ` | Mean and σ learning rates. The CLI uses `0.03` and `0.005`; a directly constructed `Agent` defaults to `0.02` and `0.005`. |
| `n` | Number of channels: `2\|S\| + \|P\|` (μ, σ, noise) |
| `\|S\|` | Number of sites |
| `\|P\|` | Number of noise producers |

### 2.3 Channel layout

The strategy state is one flat vector/matrix with three blocks:

```text
index:   0 … |S|−1        |S| … 2|S|−1        2|S| … 2|S|+|P|−1
block:   μ channels       σ channels          noise channels
```

- **μ channels** carry the mean-coordinate score.
- **σ channels** carry the log-σ score.
- **Noise channels** carry external producer deltas (STDP, Poisson).

Noise channels have a Fisher row but **no trace row and no credit row**. They change the step length through `F`, never through `G`.

In this file `|P|` counts noise producers; the number of players is written `n_p`.

---

## 3. The control loop, step by step

### 3.1 The two stages

`Personalizer.run` has two stages:

1. **`pose`** — hold the target at its initial pose until both ankle errors stay below `eps_pose` for `pose_hold_steps` consecutive steps, or until `pose_max_steps` is reached.
2. **`track`** — follow the moving mocap target until the windowed-mean loss drops below `loss_stop`, or until `max_steps` / the wall-clock budget.

Every step in both stages is an ordinary PERSONA update. The stage only changes which target is fed to the rig.

### 3.2 One step in order

This is the exact order in `Personalizer.run` and `Agent.step`:

1. **Choose the target.**
   - In `pose`: the target at `t = 0`.
   - In `track`: the target at the current CPG time.

2. **Sample and apply, every `k` steps** (`realize_every_k`, default 1).
   - Snapshot `(μ_s, σ_s)` for every site.
   - Draw `x̂_s = clip(μ_s + σ_s ε_s)`.
   - Apply `x̂_s` to the host (populations, synapses, devices, STDP hyper-parameters).
   - The strategy still updates **every** step; when `k > 1`, it reuses the last applied sample.

3. **Advance the plant.**
   - `cpg.update(cpg_in)` then `cpg_out = cpg.step()`: one NEST step.
   - `cpg_in = rig.step(cpg_out, **targets)`: one MuJoCo step.
   - The rig returns the measured ankle angles, gait phase, contact, and sensory afferents.

4. **Filter the signals.**
   - Output: subtract `zero_shift`, then a moving average over the last `smooth_kernel_size = 5` steps.
   - Target: the same moving average.
   - Both use the same filter, so there is no differential lag.

5. **Compute the error and the loss.**
   $$
   E_i(t) = |\text{target}_i(t) - \text{output}_i(t)|, \qquad
   \text{loss}(t) = \frac{1}{2}\sum_{i=1}^{2} E_i(t).
   $$
   The loss is only used for logging and stopping; the strategy uses the vector `E`.

6. **Compute the advantage.**
   $$
   A_i(t) = b_i(\phi_i) - E_i(t),
   $$
   where `b_i(φ)` is the moving-average baseline for component `i` in phase `φ`. The baseline is updated **after** the advantage is computed:
   $$
   b_i(\phi_i) \leftarrow \alpha b_i(\phi_i) + (1-\alpha)E_i(t), \qquad \alpha = 0.90.
   $$
   The first observation has `b = 0`, so its advantage is `−E`.

   The phase used for the baseline is the **reference** phase (`phase_ref_*`) when it is available, not the model phase. The model phase is what the policy is changing, so conditioning the baseline on it would make the baseline endogenous. The model phase still drives logging and the Fisher refresh.

7. **Build the strategy channels.**
   - For each site: `sites[s] = (Δθ_s, σ_s, Δlogσ_s)`.
   - For each producer: `noise[p] = (δ_p, R_p)`.

8. **Run the strategy.** `FisherMatrix.step`:
   - grow the state if new sites/producers appeared;
   - build the score `s` and trace `a`;
   - update `F`, `X̄`, `G`;
   - update the inverse `P`;
   - compute `Δμ` (zero during warm-up).

9. **Apply the updates.**
   - Means: `μ_s ← clip(μ_s + Δμ_s + spring_force_s)`.
   - σ: `σ_s ← clip(σ_s exp(Δlogσ_s), σ_min, σ_max)`.
   - Both use the **sampling-time** snapshot, so neither depends on the other's change this step.

10. **Log.** Errors, loss, advantage, phase, influence, means, σ, and the diagnostics of H and the game.

### 3.3 "Recorded = applied"

A site recorder stores `(x̂_s, σ_s)` when the sample is applied. The strategy reads that record, not a fresh draw. This is what makes the score match the perturbation that actually reached the plant:

- the same `σ_s` is used for exploration, recording, and scoring;
- when `realize_every_k > 1`, the held sample is reused, and the strategy still sees exactly what is in force;
- a site's realization draws are `x_j = clip(x̂_s + σ_s ξ_j)`, so the marginal per-subject width is `√2 σ_s` (common sample plus the subject's own draw). `N` is a substrate, not a tuned quantity.

---

## 4. The core strategy

### 4.1 The policy and its sample

Each site is a Gaussian policy:

$$
x̂_s = \operatorname{clip}\!\left(\mu_s + \sigma_s \varepsilon_s\right),
\qquad \varepsilon_s \sim \mathcal{N}(0,1).
$$

The perturbation and its standardized form are

$$
\Delta\theta_s = x̂_s - \mu_s, \qquad z_s = \frac{\Delta\theta_s}{\sigma_s}.
$$

`z_s` is the natural unit: it says how many σ the sample moved.

### 4.2 Scores

For each site `s`:

$$
s^{\mu}_s = \frac{\Delta\theta_s}{\sigma_s^2}, \qquad
s^{\sigma}_s = z_s^2 - 1.
$$

These are the **scores** of the Gaussian policy with respect to `μ` and `log σ`:

$$
\nabla_\mu \log p(x;\mu,\sigma) = \frac{x-\mu}{\sigma^2}, \qquad
\nabla_{\log\sigma} \log p(x;\mu,\sigma) = \left(\frac{x-\mu}{\sigma}\right)^2 - 1.
$$

For each noise producer `p`:

$$
s^{N}_p = \frac{\delta_p}{R_p^2},
$$

where `R_p` is the producer's current value at registration (STDP: `Wmax`; Poisson: current rate; CUT: 10 Hz at rest). `R_p` is a fixed normalization, not an estimate. A missing or zero `R_p` means no influence.

### 4.3 The trace

The trace is what PERSONA actually did:

$$
a^{\mu}_s = \frac{\Delta\theta_s}{\sigma_s^2}, \qquad
a^{\sigma}_s = \Delta\log\sigma_s, \qquad
a^{N}_p = 0.
$$

The μ row is in **score units**, not raw perturbation units. This is deliberate. If the trace carried raw `Δθ`, every site's gain would acquire a spurious `σ_s²` factor; with the current site scales that is a spread of about `10⁶` (Izh `a` at 0.005 versus `Wmax` at 12). In score units the update is unit-invariant: rescaling `θ → cθ` and `σ → cσ` scales the step by `c`, not `c³`.

The σ row is the log-σ change that was **actually applied** this step, including the clip and the σ bounds. This keeps "traced = applied" exact.

### 4.4 The three memories

All three use the same forgetting factor `β`:

$$
F_t = \beta F_{t-1} + (1-\beta)\, s_t s_t^\top,
$$

$$
\bar X_t = \beta \bar X_{t-1} + (1-\beta)\, a_t,
$$

$$
G_t = \beta G_{t-1} + (1-\beta)\, \bar X_t A_t^\top.
$$

Read them as:

- `F` — the expected Fisher information: which directions the policy is actually exploring, and how strongly.
- `X̄` — a decaying trace of recent applied moves, per channel.
- `G` — delayed credit: the current trace paired with the current advantage.

The key design choice is that `G` pairs `X̄_t` with `A_t`, not `a_t` with `A_t`. A parameter change often affects the plant several steps later (sensory latency plus the parameter's own effect). Pairing the **trace** (what has recently been moving) with the current advantage credits a site even when the coupling lags. `X̄` is an exponential memory of the last few moves; `G` is the exponentially forgotten sum of those pairings.

`G` has two columns, one per limb. The credit vector is

$$
g = G \cdot \mathbf{1}_2,
$$

with the noise rows set to zero:

$$
g_{2|S|:} = 0.
$$

So a site's credit is the sum of its credit for the left and right legs.

### 4.5 The natural-gradient step

$$
\Delta\mu = \eta_\mu\, P\, g, \qquad P \approx (F + \lambda I)^{-1}.
$$

Why the inverse Fisher? Because `g` is a score-space gradient, and `F` is the metric of that space. Dividing by `F` turns a gradient into a natural gradient: a direction with a lot of exploration noise (large `F`) gets a smaller step, and a direction with little noise gets a larger step. `λ` is a ridge that keeps `P` finite when `F` is singular or nearly singular.

Only the **μ rows** of `P g` are applied. The σ rows of `g` and the noise rows are not applied as parameter moves:

- σ has its own rule (§5);
- noise is external and must not be "optimized";
- but both still shape the μ step through the off-diagonal blocks of `P`.

### 4.6 The recursive inverse

`P` is not recomputed from scratch every step. The code maintains it with the Sherman–Morrison identity.

Let `k` be the number of steps since the last exact refresh, and suppose

$$
P_{t-1} = \left(F_{t-1} + \lambda \beta^{k-1} I\right)^{-1}.
$$

Then

$$
F_t + \lambda \beta^k I
= \beta\left(F_{t-1} + \lambda \beta^{k-1} I\right) + (1-\beta)s_t s_t^\top
= \beta P_{t-1}^{-1} + (1-\beta)s_t s_t^\top.
$$

Apply Sherman–Morrison to `A = β P⁻¹`, `u = (1−β)s`, `v = s`:

$$
P_t = \frac{1}{\beta}\left[
P_{t-1} - \frac{(1-\beta)\,P_{t-1}s_t s_t^\top P_{t-1}}
{\beta + (1-\beta)\,s_t^\top P_{t-1}s_t}
\right].
$$

That is exactly `_rank1_update_inverse`:

```python
u = P @ s
v = s @ u
denom = beta + (1 - beta) * v
P = (P - ((1 - beta) / denom) * np.outer(u, u)) / beta
```

The ridge decays as `λβ^k` between refreshes. This is consistent with forgetting: old information fades, so the regularization of that old information fades with it.

**Refresh schedule.**

- Every `refresh_every` steps, and on a phase shift, `P` is recomputed exactly as `(F + λI)⁻¹` by Cholesky (with a pseudo-inverse fallback). This restores the constant ridge.
- The default `refresh_every = floor(log(0.95) / log(β))`, which is the number of steps for `β^k` to fall to 0.95. At `β = 0.995` that is 10 steps.
- If `β ≤ 0`, there is no finite recursive ridge; the code uses `refresh_every = 1` and solves directly every step.
- If `β ≥ 1`, `refresh_every` is effectively infinite.
- If the rank-1 update produces a non-positive or non-finite denominator, it falls back to an exact refresh.

At initialization, `P = (1/λ) I`.

### 4.7 Warm-up and freezing

- `warmup = 3` by default.
- `F`, `X̄`, and `G` accumulate from step 1.
- `Δμ` and `Δlogσ` are applied from step `warmup + 1 = 4`.
- During warm-up, the returned `Δμ` and `Δlogσ` are exactly zero. The trace still records the sampled perturbation `Δθ/σ²` in the μ row, which is what the strategy accumulates; the σ row records the applied `Δlogσ = 0`. So "traced = applied" still holds for the applied move.

A site can be **frozen** per attribute (`mean`, `sigma`, or both). A frozen mean applies no displacement this step; a frozen σ is not updated. Freezing is a host constraint, not part of the strategy math.

### 4.8 What the σ and noise channels do

- **σ channels** are in `F`, `X̄`, and `G`. Their `G` row gets credit from `Δlogσ`. They influence the μ step through the off-diagonal of `P`, but their own `P g` value is not applied; the σ rule is separate.
- **Noise channels** are in `F` only. They have `a = 0`, so they never enter `X̄` or `G`, and `g` is zero on those rows. They still change the step length through `F` and `P`: a noisy producer makes the metric larger in its direction, which shrinks steps that are correlated with it.

This is the formal meaning of "noise is in the metric, not in the credit."

### 4.9 Merge and batch (optional paths)

These exist for benchmarks and replica-style comparisons. Production uses `step`.

**`step_batch(observations)`** averages the sufficient statistics of several observations and performs one update:

$$
F_t = \beta F_{t-1} + (1-\beta)\frac{1}{T}\sum_{j=1}^{T} s_j s_j^\top,
$$

$$
\bar X_t = \beta \bar X_{t-1} + (1-\beta)\frac{1}{T}\sum_{j=1}^{T} a_j,
\qquad
G_t = \beta G_{t-1} + (1-\beta)\bar X_t \left(\frac{1}{T}\sum_{j=1}^{T} A_j\right)^\top.
$$

It then refreshes `P` exactly and computes `Δμ` from the last observation's site values. This path is `O(n³)` per call.

**`merge(others, weights)`** replaces `F`, `X̄`, and `G` by a weighted average of several strategies:

$$
F \leftarrow \sum_k w_k F^{(k)}, \qquad
\bar X \leftarrow \sum_k w_k \bar X^{(k)}, \qquad
G \leftarrow \sum_k w_k G^{(k)},
$$

with `steps = max_k steps_k`, then refreshes `P`. Weights are normalized to sum to 1. The strategies must have identical site and noise name lists.

### 4.10 Introspection

- `feature_scales() = sqrt(diag(F))` — the per-channel activity scale.
- `influence()` — the row norms of `G` over the `2|S|` site rows, normalized by the maximum. This is the per-channel impact on the tracked advantage.
- `snapshot()` / `from_snapshot()` — a self-contained strategy checkpoint (`F`, `X̄`, `G`, `P`, refresh age, hyper-parameters). It does **not** include the advantage baselines, the game, H, springs, or the RNG streams.

---

## 5. The σ rule

The σ update is a log-variance policy-gradient step:

$$
\Delta\log\sigma_s = \eta_\sigma \cdot \operatorname{clip}\!\left((z_s^2 - 1)\,\bar A,\; -0.25,\; +0.25\right),
$$

where

$$
\bar A = \operatorname{mean}(A)
$$

is the scalar advantage for the step. Then

$$
\sigma_s \leftarrow \operatorname{clip}\!\left(\sigma_s \exp(\Delta\log\sigma_s),\; \sigma_{\min},\; \sigma_{\max}\right).
$$

Properties:

- **Log scale.** `Δlogσ` is a relative change, so the same rule works for `σ = 0.005` and `σ = 5`.
- **Direction.** If `z² > 1`, the sample was farther than one σ; with a positive advantage, σ grows (more exploration was useful). If `z² < 1`, the sample was close to the mean; with a positive advantage, σ shrinks.
- **Clip.** `|Δlogσ| ≤ 0.25` per step, so σ can never jump by more than about 28%.
- **Bounds.** `σ_min = 1e-4` by default. `σ_max = max(2.0, σ_initial)` unless a site overrides it. A site never starts out of bounds, and σ can shrink but not grow beyond its starting scale unless explicitly allowed.
- **Sampling-time snapshot.** The update uses `μ` and `σ` from the moment the sample was drawn, not the values after the mean update. This makes the order of the mean and σ updates irrelevant.

The trace's σ row records the **applied** `Δlogσ` after the clip and the σ bounds, so the trace and the host stay in exact agreement.

---

## 6. Homeostatic regulator H

H is **opt-in** (`HomeostaticConfig(enabled=True)`). It is a relational regulator: it does not say "this parameter should stay near its default." It asks whether the current distribution of advantage claims is consistent with what the players can predict and deliver.

### 6.1 Players and the physical graph

A player is one population or one connection with at least one optimizable site. The graph has one node per player. Edges come from the registry:

1. **Incidence** — a synapse player is connected to the population players at its source and target.
2. **Antagonism** — for each target, incoming projections with opposite weight signs add edges between their source populations and between their synapse players.
3. **Reciprocity** — if `A → B` and `B → A` both exist, their synapse players are connected.

The adjacency is **simple**: a pair is either connected or not; duplicate proposals do not add weight. Edge weights are fixed at 1.0 and do not depend on σ or advantage, so a player cannot weaken H by inflating its variance.

The graph operator is a degree-scaled combinatorial Laplacian plus a weak global regularizer:

$$
L = D - A, \qquad
L_{\text{sym}} = \frac{L}{\max(d_{\max}, 1)},
$$

$$
H = L_{\text{sym}} + \varepsilon_H\left(I - \frac{\mathbf{1}\mathbf{1}^\top}{n_p}\right),
\qquad \varepsilon_H = 10^{-3}.
$$

Properties:

- `L` is symmetric positive semidefinite and `L 1 = 0`.
- `L_sym` has eigenvalues at most 2.
- The regularizer is symmetric, PSD, and zero-row-sum.
- Therefore `H 1 = 0` and `1ᵀ H = 0` for every graph, including isolated players.
- An isolated player gets no Laplacian correction but still receives the weak global term.

H is fixed for a given physical map.

### 6.2 Standardized credit

Raw site credit `g_p` scales roughly as `1/σ_p²`, so a raw mean over sites would be dominated by the tightest coordinate. H standardizes each site by a fixed credit unit before averaging:

$$
g_{\text{ref},p} = \max\left(\operatorname{RMS}_t(g_p(t)),\; g_{\min}\right),
$$

where the RMS is taken over the H burn-in. A site that first appears after the burn-in gets `g_ref,p = max(|g_p|, g_min)` from its first observation. Then

$$
c_i = \frac{1}{M_i}\sum_{p \in P_i} \frac{g_p}{g_{\text{ref},p}},
$$

where `M_i` is the number of sites in player `i`. `g_ref` is a unit of measurement, not a preferred credit level. `M_i` is the number of coordinates, not the biological size `N_i`; a large pool does not win merely by having more neurons or synapses.

The **declared variance** is the mean squared σ ratio:

$$
v_i = \frac{1}{M_i}\sum_{p \in P_i}\left(\frac{\sigma_p}{\sigma_{\text{ref},p}}\right)^2,
\qquad v_i \leftarrow \operatorname{clip}(v_i, v_{\min}, v_{\max}).
$$

`σ_ref,p` is the first observed σ at that site, a fixed unit of measurement. The score is therefore invariant to a global rescaling of all σ.

The **centered standardized credit** is

$$
z_i = \frac{c_i - \operatorname{mean}_j(c_j)}{S_{c,\text{prev}}},
$$

where `S_c` is an EWMA of the across-player standard deviation of `c`, floored at `S_min`:

$$
S_{c,\text{raw}} = \operatorname{std}_i(c_i), \qquad
S_c \leftarrow (1-\alpha_S)S_c + \alpha_S \max(S_{c,\text{raw}}, S_{\min}),
$$

with `α_S = 1 − exp(−dt/τ_S)` and `τ_S = 2.0 s` by default. Using `S_{c,prev}` means the current standardization uses no current-step information beyond `c` itself.

Because `z` is centered, `mean_i z_i = 0` at every step. Adding a common credit level to every player leaves `z`, `π`, `e`, and `u_H` unchanged: H regulates relative shares, not the common drive.

### 6.3 Forecast, score, and shares

Each player forecasts its next standardized credit with an EWMA:

$$
\ell_i = -\frac{1}{2}\left[\log(2\pi v_i) + \frac{(z_i - \mu_i)^2}{v_i}\right],
$$

$$
\mu_i \leftarrow (1-\alpha_m)\mu_i + \alpha_m z_i,
\qquad \alpha_m = 1 - e^{-dt/\tau_m}.
$$

`ℓ_i` is a proper Gaussian log score with declared variance `v_i`. For a fixed mean, it is maximized at `v_i = (z_i − μ_i)²`; declaring more variance than the realized error is penalized by `log v_i`. Inflating σ therefore does not buy claim once the declared variance is wider than the actual forecast error.

The **target share** and **current share** are softmaxes:

$$
a_i = \operatorname{softmax}_i(\ell_i / T), \qquad
\pi_i = \operatorname{softmax}_i(z_i / T),
$$

and the **share error** is

$$
e_i = a_i - \pi_i, \qquad \sum_i e_i = 0.
$$

It is smoothed on the homeostatic timescale:

$$
\bar e_i \leftarrow (1-\alpha_H)\bar e_i + \alpha_H e_i,
\qquad \alpha_H = 1 - e^{-dt/\tau_H},
$$

with `τ_H = 5.0 s` by default, slower than `τ_mem = 2.0 s`.

### 6.4 The global variance temperature `T`

`T` is a dimensionless unit of account, not a player. No single player can set it. Four modes are implemented:

| Mode | `T` |
|---|---|
| `advantage` | `clip(V_obs / V_A0, T_min, T_max)` |
| `budget` | `1` |
| `fisher` | `clip(V_F_ewma / V_F0, T_min, T_max)` |
| `band` | `clip(R_ewma, band_lo, band_hi)` |

Here:

$$
V_F = \frac{\operatorname{trace}(F_{\mu\mu})}{n_{\text{sites}}},
$$

`V_A0` is the burn-in sample variance of `A`, `V_F0` is the EWMA-stationary extrapolation of `V_F` at the end of burn-in, and

$$
\text{delta} = A - \mu_A, \qquad
\mu_A \leftarrow \mu_A + (1-\beta_A)\text{delta},
$$

$$
V_{\text{obs}} = \max\left(\beta_A\left(V_{\text{obs}} + (1-\beta_A)\text{delta}^2\right), 0\right),
$$

$$
V_{F,\text{ewma}} \leftarrow \beta_A V_{F,\text{ewma}} + (1-\beta_A)V_F,
$$

with `β_A = exp(−dt/τ_A)`, `τ_A = 2.0 s`. The `band` mode uses `R = V_obs / V_A0`, `R_ewma ← β_V R_ewma + (1−β_V)R`, `β_V = exp(−dt/τ_V)`, `τ_V = 5.0 s`.

Defaults: `T_min = 0.1`, `T_max = 10.0`, `band_lo = 0.5`, `band_hi = 2.0`.

### 6.5 The correction `u_H` and the lift

The player-level correction is

$$
u_H = \lambda_H\, S_{c,\text{prev}}\, H\, \bar e^{+},
\qquad \lambda_H = 1.0 \text{ by default},
$$

where `ē⁺` is the value after the update in §6.3. The code updates `ē` first and then applies `u_H`.

The sign follows from `e = a − π`: if `e_i > 0`, player `i` is below its target share, and increasing its credit increases `π_i`, which reduces `e_i`. A graph Laplacian has the same sign pattern as the error on an edge (`e = (δ, −δ)` gives `H e = (2w/d_max)(δ, −δ)`), so the positive feedback `+λ_H S_c H ē` moves credit toward the target.

Because `H 1 = 0`, `Σ_i u_{H,i} = 0`: the correction is zero-sum at the player level.

**Lifting to sites.** There are two modes:

- **`player`** (the adjoint lift):
  $$
  g_{H,p} = u_{H,i}\, g_{\text{ref},p}, \qquad p \in P_i.
  $$
  Then `C g_H = u_H` exactly: a player correction `u_H,i` changes its credit `c_i` by exactly `u_H,i`. The site-level sum is not generally zero, because players have different `M_i` and `g_ref,p`. That is expected: the game's unit is player credit.

- **`site_centered`** (the experimental current-best lift):
  $$
  w_p = g_{\text{ref},p} - \operatorname{mean}_{q \in P_i} g_{\text{ref},q},
  \qquad
  g_{H,p} = u_{H,i}\, w_p.
  $$
  Each player's site correction sums to zero, so `Σ_p g_{H,p} = 0` exactly. It deliberately breaks the adjoint `C g_H = u_H`; it is a within-player redistribution, not a player-credit transfer. Only players with at least two sites participate; a singleton player gets zero.

The corrected credit used by the natural gradient is

$$
g' = g + g_H.
$$

H does **not** rewrite `G`, `c`, `z`, or `π`. It is a one-step bias to the credit vector used by the step. The next step's realized credit comes from `G` and the host response.

### 6.6 Burn-in, gates, and delay

H has three stages:

1. **Strategy warm-up** (`steps ≤ strategy.warmup`): H is inactive and its state is not updated.
2. **H burn-in** (`h_burnin_steps = 100` after warm-up, 1.0 s at `dt = 10 ms`): H returns `u_H = 0` but collects the scales:
   - store `g(t)`, `A(t)`, and `V_F(t)` each step;
   - at the end, set `g_ref,p` to the RMS of `g_p`;
   - recompute `c_i(t)` with the final `g_ref`; set `S_c0 = max(mean_t std_i(c_i(t)), S_min)`;
   - set `z_i0(t) = (c_i(t) − mean_i c_i(t))/S_c0` and `μ_i0 = mean_t z_i0(t)`;
   - set `V_A0` to the sample variance of `A(t)`, `mean_A0` to its mean;
   - set `V_F0 = V_F(end) / max(1 − β^T, 0.1)`, the EWMA-stationary extrapolation;
   - initialize `S_c = S_c0`, `μ = μ_i0`, `mean_A = mean_A0`, `var_A = V_A0`, `V_obs = V_A0`, `V_F_ewma = V_F0`, `ē = 0`.
3. **Active H**: compute `ℓ`, `a`, `π`, `e`, update `ē`, apply `u_H`.

If `h_burnin_steps = 0`, the first active step computes `g_ref`, `S_c`, and `μ` from the current step, returns `u_H = 0`, and correction starts on the following step.

`u_H` is set to zero (gated) when any of these holds:

- `S_c_raw < eps_credit` (a null or uniform game cannot manufacture claims from nothing);
- `steps < h_delay_steps` (the current-best preset delays H to step 500);
- `λ_H = 0`;
- the number of players is zero.

The delay is a timing gate, not a second burn-in. The regulator still updates `ē`, `S_c`, `μ`, and the global-variance state during the delay; only `u_H` is zero.

### 6.7 The current-best preset

`HomeostaticConfig.current_best()` is:

```python
HomeostaticConfig(
    enabled=True,
    global_variance_mode="fisher",
    lift_mode="site_centered",
    h_delay_steps=500,
)
```

Other defaults: `λ_H = 1.0`, `τ_H = 5.0 s`, `τ_m = 2.0 s`, `τ_S = 2.0 s`, `τ_A = 2.0 s`, `τ_V = 5.0 s`, `h_burnin_steps = 100`, `T_min = 0.1`, `T_max = 10.0`, `band_lo = 0.5`, `band_hi = 2.0`, `v_min = 1e-8`, `v_max = 1e6`, `S_min = 1e-12`, `g_min = 1e-12`, `eps_H = 1e-3`, `eps_credit = 1e-9`.

This is an experimental preset, not a production default.

---

## 7. The conserved-stack game

The game is **opt-in** at the class level, but the CLI default is `--game selected`, which uses `GameConfig.current_best()`. It is a zero-sum game over a conserved resource: each player's **share of the total advantage**.

### 7.1 Stacks

Each player `i` holds a stack `s_i ≥ 0` with

$$
\sum_i s_i = 1.
$$

The stack is a share, not a tank: it can move between players, but it cannot vanish. Initial state is uniform, `s_i = 1/n`.

### 7.2 Throttle

The game computes a per-player entitlement `τ_i ∈ [0, 1]` and applies `Δ_i ← τ_i Δ_i`.

Two contestable modes (the table shows the raw ranking before the exponent and base seat):

| Mode | Formula | Behaviour |
|---|---|---|
| `proportional` | `τ_i = s_i / max_j s_j` | the richest player moves most; a zero-stack player's raw throttle is 0 |
| `inverse` | `τ_i = (1 − s_i) / max_j (1 − s_j)` | the poorest player moves most; a full-stack player's raw throttle is 0 |

In code, the raw quantity is divided by its maximum, clipped to `[0,1]`, raised to `throttle_exponent`, and then blended with a base seat:

$$
c_i^{\text{contest}} = \left(\operatorname{clip}\!\left(\frac{\text{raw}_i}{\max_j \text{raw}_j}, 0, 1\right)\right)^{\gamma},
$$

$$
\tau_i = \operatorname{clip}\!\left((1-\varepsilon)c_i^{\text{contest}} + \varepsilon c_i^{\text{base}}, 0, 1\right),
$$

where `γ = throttle_exponent` (default 1.0; current best 2.5) and `ε = liveness_fraction` (default 0.0; current best 0.05).

If the maximum is zero, the contestable part is uniform. The base seat is:

- `none` — no base;
- `uniform` — `c_i^base = 1` for every player;
- `uncertainty` — `c_i^base = RMS(σ_p / σ_ref,p)` over the player's sites, normalized so the maximum is 1. An uncertain player keeps more base movement; a confident one keeps less.

If there is no σ history, the uncertainty seat is uniform.

The throttle is an **entitlement**, not a fee. Unused entitlement is not transferred, because it was never taken from anyone.

### 7.3 Contribution

After throttling, the game computes each player's first-order contribution:

$$
a_i = \sum_{p \in P_i} g_p\, \Delta_p,
$$

where `Δ_p` is the throttled move. This is the local predicted advantage that Fisher can attribute.

### 7.4 Replicator transfer

Normalize the contributions by their absolute mass:

$$
\hat a_i = \frac{a_i}{\sum_j |a_j| + \varepsilon}.
$$

The transfer is

$$
\Delta s_i = \eta\,(\hat a_i - s_i \sum_j \hat a_j),
\qquad \eta = \text{transfer\_rate}.
$$

Because `Σ_i s_i = 1`,

$$
\sum_i \Delta s_i = \eta\left(\sum_i \hat a_i - \sum_i \hat a_i\right) = 0:
$$

the transfer is exactly zero-sum before projection. A player that contributes above its share gains; a player below its share loses. Negative contributions lose share.

Then project back onto the simplex:

$$
s_i \leftarrow \max(s_i + \Delta s_i,\; \text{share\_floor}),
\qquad
s \leftarrow \frac{s}{\sum_j s_j}.
$$

If the sum is non-finite or below `eps`, the game reverts to the previous stacks.

The transfer is active only when `steps > burnin_steps`, `n > 1`, and `transfer_rate > 0`.

### 7.5 The common-pot arm (optional fee)

When `movement_fee = κ > 0`, movement has a price. The Fisher length of the executed move is

$$
d_i = \sqrt{\Delta_i^\top F_{ii} \Delta_i},
$$

where `F_ii` is the player's diagonal Fisher block. The fee is

$$
c_i = \kappa \frac{d_i}{\sum_j d_j + \varepsilon},
\qquad
\text{pot} = \sum_i c_i \le \kappa.
$$

Debit `s_i ← s_i − c_i`. Redistribute the pot either by positive contribution (`pot_mode = "contribution"`, weights `max(â_i, 0)`, falling back to equal weights if those sum to zero) or equally (`pot_mode = "equal"`). The debit equals the credit by construction, so the pot is zero-sum before projection.

### 7.6 Diagnostics

The game records the stacks, throttle, contribution, `â`, transfer, fees, pot, transfer sum, share sum, entropy, normalized entropy (`entropy / log n`), HHI (`Σ s_i²`), and the min/max stack and throttle.

### 7.7 The current-best preset

`GameConfig.current_best()` is:

```python
GameConfig(
    enabled=True,
    throttle_mode="proportional",
    base_seat="uncertainty",
    liveness_fraction=0.05,
    throttle_exponent=2.5,
    transfer_rate=0.05,
    movement_fee=0.0,
    burnin_steps=100,
)
```

This is the `a25` arm from the tuning grid: uncertainty base seat, proportional ranking, liveness 0.05, curvature 2.5, transfer 0.05, no pot.

---

## 8. PERSONA-Evo

Evo chooses the **player partition** itself. It is opt-in (`--training evo`) and runs the game inside each generation.

### 8.1 The segment matrix

A **segment** is one registry object with at least one optimizable site: a population or a projection. The segment matrix has:

- `adjacency[i, j]` — the maximum edge weight over all edge types between segments `i` and `j`;
- `edge_types[(i, j)]` — the set of edge types that proposed the pair;
- `distance[i, j]` — the unweighted shortest-path distance on the binarized adjacency (`adjacency > 0`).

Edge types and weights:

| Edge type | Weight | Meaning |
|---|---:|---|
| `incidence` | 1.0 | projection touches a population at its source or target |
| `relay_drive` | 1.0 | projection is driven by a relay for a population |
| `antagonism` | 0.70 | opposite-sign incoming projections to the same target |
| `reciprocity` | 0.60 | `A → B` and `B → A` both exist |
| `shared_source` | 0.40 | two projections share a source |
| `shared_target` | 0.40 | two projections share a target |

When several types propose the same pair, the adjacency takes the **maximum** weight, not the sum. The distance ignores weights: it is the number of hops on the unweighted graph. Weights are used only to break distance ties when reassigning orphan segments.

The matrix is deliberately independent of the PERSONA update. The strategy remains site-level.

### 8.2 Partitions

A `PlayerPartition` is a tuple of groups of segment ids. It is normalized (each group sorted, groups ordered) and must satisfy:

- every segment id is in exactly one group;
- no group is empty;
- every segment in the matrix is covered.

A partition is turned into a `PhysicalMap` by giving each group a `PlayerSpec` whose `param_names` are the union of its segments' sites. That map has **no edges**; the game does not need the graph. (Only H uses the graph, and Evo and H are mutually exclusive.)

### 8.3 Initialization

**Territory growth** (default):

1. Find the connected components of the segment graph.
2. Require `n_players ≥ number of components`, with at least one seed per component.
3. Allocate seeds proportionally to component size, with at least one per component. The allocation uses floor plus largest remainder.
4. Choose the seeds uniformly at random within each component.
5. Assign every remaining segment to its nearest seed by segment-matrix distance. Ties are broken uniformly with the Evo RNG.

The result is deterministic from `(seed, P)`.

**Pure random** (ablation): assign each segment uniformly at random to a player, then fill empty players by moving a random segment from the largest group.

### 8.4 One generation

`run_generation` builds the normal personalizer with:

- `GameConfig.current_best()`, `share_floor = 0`;
- `homeostasis = None`;
- `player_map = partition`;
- `steps_per_generation` steps;
- `eps_pose = 0.10`, `pose_hold_steps = 3`, `loss_stop = −1` (never stop early);
- `loss_window = 20`, `ref_hz = 700`, `resolution_ms = 0.2`, the generation seed.

It runs with `pose_max_steps = 100` and `realize_every_k = 1`, then reads the game diagnostics:

- `fractions` — the final stacks;
- `fraction_mean` — the mean stacks over the last `eval_window` steps;
- `contribution` — the mean contribution over the window;
- `transfer` — the mean transfer over the window;
- `replicator_residual` — `max |transfer|` over the window;
- `movement` — the mean over the window of `max |transfer|` per step.

The generation metrics also include `loss_eval` (last `eval_window` mean), `loss_full`, `n_steps`, `active_players` (fractions above `1e-12`), normalized stack entropy, `n_players`, inflation metrics, type/sign violations, the phase gate, biological acceptance, and locomotion metrics.

### 8.5 Pruning and redistribution

After a generation:

1. A player is **dead** if its fraction is `≤ max(impact_eps, 1e-12)`.
2. If all players are dead, the partition collapses to one player containing every segment (`all_dead`).
3. If fewer than `min_players` survive, no pruning is done (`min_players`).
4. Otherwise, every orphan segment of a dead player is reassigned individually:
   - compute `(distance, weight)` to each surviving player with `player_link` (minimum segment distance, summed adjacency weight);
   - keep the finite distances; choose the minimum distance;
   - break ties by the larger summed weight, then uniformly with the Evo RNG;
   - if no surviving player is reachable, choose the largest incident weight, then the larger surviving fraction, then the lower index, and flag the assignment as a fallback.

The result is validated against the matrix.

### 8.6 Stop rules

For each generation:

- `no_dead = not any(dead)`;
- `stable_next = stable + 1` if `no_dead`, else 0;
- `partition_ok = no_dead and stable_next ≥ patience`;
- `nash_ok = no_dead and replicator_residual ≤ nash_tol`.

Then:

| Condition | Stop reason |
|---|---|
| no survivors | `all_dead` |
| survivors `< min_players` | `min_players` |
| `stop = "partition"` and `partition_ok` | `partition` |
| `stop = "nash"` and `nash_ok` | `nash` |
| `stop = "nash"`, `no_dead`, not `nash_ok` | `partition_stable_nash_failed` |
| `stop = "both"` and both | `both` |
| `stop = "both"`, `no_dead`, not `nash_ok` | `partition_stable_nash_failed` |
| the partition key was seen before and is not the last key | `cycle` |
| generation limit reached | `max_generations` |

The partition key is the tuple of groups. `fraction_stability` is `max |fractions − previous_fractions|` when the player counts match.

### 8.7 Partition metrics

Given a partition and a reference labelling of segments, the metrics are:

- **ARI** — adjusted Rand index, chance-corrected pair agreement.
- **NMI** — normalized mutual information, `MI / mean(H_true, H_pred)`.
- **Homogeneity** — `1 − H(true | pred) / H(true)`.
- **Completeness** — `1 − H(pred | true) / H(pred)`.
- **V-measure** — harmonic mean of homogeneity and completeness.
- **Purity** — `Σ_j max_i table[i,j] / n`.
- **Coverage** — fraction of segments assigned to some player.
- **Pairwise F1** — `2PR/(P+R)` with `P = ΣC / b_comb`, `R = ΣC / a_comb`, where `ΣC` is the number of true-pair co-clusters.

The reference hierarchy is evaluation-only and never trains Evo:

- **Level 1** — a population plus its efferent projections (about 37 units).
- **Level 2** — the functional modules of the CPG: global drive plus nine per-limb groups (about 19 units).

`global_drive_fate` classifies where the BS/BASE rate and projection segments ended up: `same_player`, `absorbed`, `rate_global_projections_limb`, `split_by_limb`, `missing`, or `other`.

### 8.8 The current state of Evo

The first empirical pass (2026-10-26, one seed, three generations, 400 steps, fast CPG) found:

- P=24 territory: 21 final players, ARI 0.154, NMI 0.564, purity 0.454, loss 0.148.
- P=24 random: 24 players, ARI 0.0008, NMI 0.412, purity 0.270, loss 0.176.
- The territory partition had the lowest loss of the compared arms (manual 0.263, reference Level-2 0.199, evo random 0.176).
- The global-drive hypothesis was **not** supported in that run: the classifier returned `other`.
- The phase gate failed for every arm. This is a CPG/phase issue, not an Evo result.

These are partition-level results, not a bio-plausibility claim. They are one seed and short runs.

---

## 9. Validation mathematics

Validation is separate from learning. It asks two questions:

1. **Phase gate** — did the run produce a stable, repeated gait pattern?
2. **Biological acceptance** — do the global locomotion parameters and ankle angles look like the rat data?

### 9.1 The phase gate

The gate is not a reference-tracking test. It asks whether the current run produced a stable repeated pattern.

**Stance onsets.** Take the rising edges of `phase == "stance"`, debounce with `min_cycle_s = 0.25 s`, and treat each onset as the start of a cycle.

**Period and variability.** For each side, take the last `min_cycles = 5` inter-onset intervals:

$$
\bar T = \operatorname{mean}(T_k), \qquad
CV = \frac{\operatorname{std}(T_k)}{\bar T}.
$$

Fail if `n_cycles < 5` or `CV > 0.20`.

**Phase from onsets.** For time `t` after the most recent onset `o`:

$$
\theta(t) = \frac{2\pi(t-o)}{\bar T} \bmod 2\pi.
$$

**Energy phase.** For a non-negative signal `u(t)` (E or F drive):

$$
\phi_u = \arg\left(\frac{\sum_t u(t)e^{i\theta(t)}}{\sum_t u(t)}\right).
$$

This is the circular mean phase of the signal's energy over the cycle. It is used instead of burst thresholding because the applied motor envelope is smooth and F may never cross a hard threshold.

**E/F anti-phase.** For each side,

$$
\Delta_{EF} = \operatorname{wrap}(\phi_E - \phi_F).
$$

Fail if `| |Δ_EF| − π | > 0.35π` or if the circular standard deviation of `Δ_EF` exceeds `0.35π`.

**Stance/E alignment.** The E energy is expected at the middle of stance, `π/2` after the onset:

$$
\Delta_{SE} = \operatorname{wrap}(\phi_E - \pi/2).
$$

Fail if `|Δ_SE| > 0.35π` or its circular standard deviation exceeds `0.35π`.

**L/R offset.** If `phase_clock_L/R` are logged, use them directly; otherwise use the onset phases. Compute the wrapped offset (the clock branch uses `clock_R − clock_L`; the onset fallback uses `θ_L − θ_R`)

$$
\Delta_{LR} = \operatorname{wrap}(\theta_L - \theta_R)
$$

over the measured cycles, then its circular mean and standard deviation. Fail if `|mean| < 0.02` (effectively zero) or `std > 0.10π`. Also fail if the 10th–90th percentile spread exceeds `0.20π`; this catches a slow monotone drift that a circular std can understate.

**Joint-angle periodicity.** For hip, knee, and ankle on both sides, compute the autocorrelation of the joint-angle trace at the measured period:

$$
r = \operatorname{corr}\big(x(t), x(t+\bar T)\big).
$$

Fail if the mean over the six traces is below `0.50`.

**Ankle stuck at a limit.** A run is stuck if the ankle is within `0.06 rad` of a joint limit, more than `0.15 rad` from its target, and moving slower than `0.01 rad/step`, for at least `0.5 s`.

**Foot inversion.** The foot's local `−z` axis should not point upward. Fail if `foot_sole_z > 0.05` for at least `0.2 s`.

**Render.** If a render path is supplied, it must exist and be non-empty.

### 9.2 Biological acceptance

The acceptance test uses both ankles and the Stage-2 (`track`) window only:

$$
\text{angle coverage} = \operatorname{mean}\left[\max(E_L, E_R) \le 0.05\right],
$$

and requires `angle coverage ≥ 0.70`. The duty factors must also lie inside the rat distribution from the statistics table. The result is `acceptable = global_ok and angle_coverage ≥ coverage`.

### 9.3 What the current numbers say

From the 2026-10-26 comparison and Evo reports:

- The phase gate fails for **every** arm (manual, no-game, diagonal, block, Evo). The failures are the known CPG issues: stance/E offset, L/R drift, and joint-angle cycle correlation.
- Biological acceptance is **false** for every arm; ankle angle coverage is low (about 0.06–0.25 depending on the arm).
- Duty factors and double support remain in distribution.
- Type/sign violations are zero in every arm.
- These are CPG/phase results, not algorithm results. Do not present them as evidence for or against a PERSONA variant.

---

## 10. How the layers compose

| Layer | What it changes | Default | Where |
|---|---|---|---|
| Core strategy | `μ`, `σ` of every site | Always on | `strategy.py`, `agent.py` |
| H | credit `g` before the natural gradient | Off | `homeostasis.py` |
| Game | throttles the `Δμ` proposal and moves stacks | On in the CLI (`--game selected`) | `game.py` |
| Evo | chooses the player partition | Off (`--training manual`) | `evo.py` |

Order inside one step:

```text
scores s, a
   ↓
F, X̄, G
   ↓
g = G·1₂
   ↓
H: g ← g + g_H          (if enabled)
   ↓
Δμ = η P g
   ↓
game: Δμ ← τ ⊙ Δμ, move stacks   (if enabled)
   ↓
apply μ and σ
```

Important consequences:

- H acts on **credit**, before the natural gradient.
- The game acts on the **move**, after the natural gradient.
- H and Evo are mutually exclusive. `Agent` raises if both `player_map` and `homeostasis.enabled` are set.
- The game and Evo are compatible: Evo runs the game inside each generation.
- The game does not need the physical graph; only H does. A partition-derived player map has no edges, which is fine for the game.

`persona/cli.py` defaults (the simulator-independent CLI):

```text
--steps 200
--tau-mem-s 2.0
--realize-every-k 1
--lr-mode constant
--lr-start 0.03
--game selected        → GameConfig.current_best()
--training manual      → H off, Evo off
--noise-config all
--seed 0
--nest-threads 1
```

`run.py` is the single entry point; presets (`full`, `half`, `fifth`, `fast`) select sizes and rendering, and `--set key=value` overrides any `t2xcpg`/`model` PARAMS value:

```text
--sim-time 120
--snapshot-s 900
--realize-every-k 1
--lr-mode constant
--lr-start 0.03
--game selected        → GameConfig.current_best()
--noise-config all
--seed 0
--nest-threads 1
```

`Agent` defaults: `lr_mu = 0.02` when constructed directly (the CLI passes `0.03`), `lr_sigma = 0.005`, `warmup = 3`, `smooth_kernel_size = 5`, `dt_ms = 10`.

---

## 11. Current state and evidence

As of 2026-10-31:

**Core strategy — production.** The persisted `F/X̄/G` strategy, the single-σ rule, per-site and per-producer recorders, the recursive Fisher inverse, the natural-gradient step, and the two-stage personalization loop are the production path.

**H — opt-in experiment, not a production default.** `HomeostaticConfig.current_best()` is the Fisher global-variance mode with the centered site-zero-sum lift and H delayed to step 500. The 20 s production acceptance still fails; no H mode is a production default.

**Game — opt-in class, selected in the CLI.** `GameConfig.current_best()` is the uncertainty base seat, proportional ranking, liveness 0.05, curvature 2.5, transfer 0.05, no pot. In the first comparison it reduced saturation on the CPG at a loss cost; on the 14-parameter toy it was weaker than the diagonal preconditioner and CMA-ES.

**Evo — opt-in experiment.** The first pass found the P=24 territory partition had the best CPG loss (0.148) among the compared arms, with ARI 0.154 and NMI 0.564 against the Level-2 reference. It did not recover the 19 modules exactly; the pattern looks like per-limb sensorimotor territories plus commissural bridges. One seed, three generations, 400 steps.

**Validation — red.** The phase gate fails for every arm; biological acceptance is false; duty factors are in distribution. These are CPG/phase issues and must not be reported as algorithm wins or losses.

**Known caveats:**

- The evidence is short-run and mostly one seed.
- The 141-segment / 273-site inventory reflects the current working tree; older reports use 137/269.
- The phase gate is deliberately not a reference-tracking test; reference matching remains a separate PERSONA validation task.
- `sole-up` is not measured because `RunLog` does not record `foot_sole_z_*`.
- The global-drive verdict is one run and may change with more generations or a stronger shared-source weight.

---

## 12. A worked example

This is a two-step toy to show the arithmetic, not a production setting. Use one site with two channels (μ and σ), `β = 0.5`, `λ = 0.01`, `η_μ = 0.1`, and a scalar advantage padded to `[A, 0]`. There is no noise, no H, and no game.

### Step 1

Sample `z = 1` and `Δlogσ = 0`. Then

$$
s_1 = \begin{bmatrix}1\\0\end{bmatrix}, \qquad
a_1 = \begin{bmatrix}1\\0\end{bmatrix}, \qquad
A_1 = \begin{bmatrix}1\\0\end{bmatrix}.
$$

Update:

$$
F_1 = 0.5\cdot 0 + 0.5\begin{bmatrix}1&0\\0&0\end{bmatrix}
= \begin{bmatrix}0.5&0\\0&0\end{bmatrix},
$$

$$
\bar X_1 = 0.5\begin{bmatrix}1\\0\end{bmatrix}
= \begin{bmatrix}0.5\\0\end{bmatrix},
$$

$$
G_1 = 0.5\cdot 0 + 0.5\begin{bmatrix}0.5\\0\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix}
= \begin{bmatrix}0.25&0\\0&0\end{bmatrix}.
$$

Credit:

$$
g_1 = G_1\mathbf{1}_2 = \begin{bmatrix}0.25\\0\end{bmatrix}.
$$

Inverse (with `β = 0.5`, `refresh_every = 1`, so this is a direct solve):

$$
P_1 = (F_1 + 0.01I)^{-1}
= \begin{bmatrix}1/0.51&0\\0&1/0.01\end{bmatrix}
\approx \begin{bmatrix}1.961&0\\0&100\end{bmatrix}.
$$

Step:

$$
\Delta\mu_1 = 0.1\,P_1 g_1
\approx 0.1\begin{bmatrix}0.490\\0\end{bmatrix}
= \begin{bmatrix}0.049\\0\end{bmatrix}.
$$

Only the μ row is applied; the σ row is not.

### Step 2

Sample `z = 0` (so `s^σ = −1`) and `Δlogσ = 0.2`. Then

$$
s_2 = \begin{bmatrix}0\\-1\end{bmatrix}, \qquad
a_2 = \begin{bmatrix}0\\0.2\end{bmatrix}, \qquad
A_2 = \begin{bmatrix}1\\0\end{bmatrix}.
$$

Update:

$$
F_2 = 0.5F_1 + 0.5\begin{bmatrix}0&0\\0&1\end{bmatrix}
= \begin{bmatrix}0.25&0\\0&0.5\end{bmatrix},
$$

$$
\bar X_2 = 0.5\bar X_1 + 0.5\begin{bmatrix}0\\0.2\end{bmatrix}
= \begin{bmatrix}0.25\\0.1\end{bmatrix},
$$

$$
G_2 = 0.5G_1 + 0.5\bar X_2\begin{bmatrix}1&0\end{bmatrix}
= \begin{bmatrix}0.25&0\\0.05&0\end{bmatrix}.
$$

Credit:

$$
g_2 = \begin{bmatrix}0.25\\0.05\end{bmatrix}.
$$

Inverse:

$$
P_2 = \begin{bmatrix}1/0.26&0\\0&1/0.51\end{bmatrix}
\approx \begin{bmatrix}3.846&0\\0&1.961\end{bmatrix}.
$$

Step:

$$
\Delta\mu_2 = 0.1\,P_2 g_2
\approx 0.1\begin{bmatrix}0.962\\0.098\end{bmatrix}
= \begin{bmatrix}0.096\\0.0098\end{bmatrix}.
$$

Again, only the μ row is applied; the σ row of `P g` is not.

### What the example shows

- `F` accumulates exploration in each direction. The σ channel has more Fisher after step 2, so `P` gives it a smaller step.
- `X̄` is a decaying memory, not a window.
- `G` carries credit forward; the μ channel keeps credit from step 1 even though step 2 sampled only the σ channel.
- The natural gradient divides by the metric: a well-explored direction moves less per unit of credit.
- In general, the σ row of `g` can affect the μ step through the off-diagonal blocks of `P`; it is never applied as a σ move.

---

## 13. Glossary and code map

### 13.1 Glossary

| Term | Meaning |
|---|---|
| **Advantage** | `baseline(phase) − error`; positive means better than the recent same-phase error |
| **Burn-in** | A startup period that sets units/scales but applies no correction (H) or only accumulates (strategy warm-up) |
| **Channel** | One row of `F/X̄/G`: a μ coordinate, a σ coordinate, or a noise producer |
| **Credit** | The scalar per player or per site that the natural gradient uses |
| **Fisher** | The expected score outer product; the metric of the policy space |
| **Forgetting factor** | `β`; the weight of the past in the EWMA recurrences |
| **Lift** | Mapping a player-level correction back to its sites |
| **Natural gradient** | `P g`, a gradient divided by the Fisher metric |
| **Noise producer** | An external signal (STDP, Poisson) that enters `F` but not `X̄` or `G` |
| **Player** | A population or connection; the unit of H and the game |
| **Segment** | A registry object; the unit of Evo partitioning |
| **Site** | One optimizable scalar parameter; the unit of the core strategy |
| **Stack** | A player's conserved share of the total advantage in the game |
| **Throttle** | A player's entitlement to move, in `[0, 1]` |

### 13.2 Equation → code

| Mathematics | Code |
|---|---|
| Sample, σ rule, bounds | `persona/parameter.py` (`PersonaParam`) |
| Exploration sampling | `persona/sampling.py` (`ExplorationSampler`) |
| Filters, loss, advantage, apply order | `persona/agent.py` (`Agent.step`) |
| Scores, `F/X̄/G`, natural gradient, inverse, merge | `persona/strategy.py` (`FisherMatrix`) |
| Advantage baselines | `persona/strategy.py` (`PhaseBaselines`) |
| Physical map, H, credit, lift, global variance | `persona/homeostasis.py` |
| Stacks, throttle, transfer, fee | `persona/game.py` |
| Segment matrix, edges, distance | `persona/segments.py` |
| Partitions | `persona/partition.py` |
| Territory/random init, pruning, stop rules | `persona/evo.py` |
| One generation, fractions, metrics | `persona/evo_runner.py` |
| Physiological reference and partition metrics | `persona/physo_reference.py` |
| Phase gate | `persona/phase_metrics.py` |
| Biological acceptance | `persona/validation.py` |
| Cosine LR | `persona/schedule.py` |
| Springs | `persona/spring.py`, `persona/nest_adapter.py` |
| Producer deltas, EWMA stats | `persona/noise_recorder.py` |
| Two-stage run loop, logs | `persona/personalization.py` |
| CLI defaults | `persona/cli.py`, `run.py` |

### 13.3 The remaining small pieces

**Realization.** Each subject draws

$$
x_j = \operatorname{clip}(x̂_s + \sigma_s \xi_j), \qquad \xi_j \sim \mathcal{N}(0,1).
$$

The marginal per-subject width is `√2 σ_s`. `N` is a substrate, not a tuned quantity.

**STDP producer delta.** For one drained weight-recorder batch:

$$
\Delta w = w_{\text{event}} - w_{\text{last}}, \qquad
\delta_{\text{key}} = \frac{\sum \Delta w}{N_{\text{syn}}}.
$$

The first sighting uses the seed weight. The batch is shared across both limbs within one step.

**Poisson producer delta.** `δ = rate_new − rate_old`, read-and-clear.

**Producer diagnostics.** A producer recorder keeps a β-EWMA mean and second moment:

$$
m \leftarrow \beta m + (1-\beta)\delta, \qquad
q \leftarrow \beta q + (1-\beta)\delta^2, \qquad
\operatorname{var} = \max(q - m^2, 0).
$$

These are diagnostics only; scoring uses `δ/R²`.

**Type/sign clamp.** For a static weight with `mean > 0`, `min ← max(min, 0)`; with `mean < 0`, `max ← min(max, 0)`; a zero-mean weight is ambiguous and left untouched. `enforce_synaptic_sign` normalizes an inhibitory type to a negative value and an excitatory type to a positive one.

**Springs.** An optional per-site force added to `Δμ`:

$$
\text{force}_p = \text{power}_p\,(\text{anchor}_p - \mu_p),
$$

with a fixed `power` or the dynamic form

$$
\text{power}_p = \operatorname{clip}\!\left(\text{gain}\cdot\eta_\mu\cdot
\frac{\sigma_{\text{ref},p}^2}{\max(\sigma_p^2, \sigma_{\min}^2)},\; 0,\; \text{max\_power}\right).
$$

The anchor is the site's default mean. Springs are not created by the production CPG path; they are an available hook.

**Cosine LR.** For a 1-based step `t`:

$$
\eta(t) =
\begin{cases}
\eta_{\text{start}}, & t \le w+1,\\
\eta_{\text{end}}, & t \ge T,\\
\eta_{\text{end}} + \frac{\eta_{\text{start}}-\eta_{\text{end}}}{2}
\left(1+\cos\left(\pi\frac{t-w-1}{T-w-1}\right)\right), & \text{otherwise},
\end{cases}
$$

where `w = warmup_steps` and `T = total_steps`. `constant` means no schedule.

**Snapshot.** `snapshot()` stores `F`, `X̄`, `G`, `P`, `_P_valid`, `steps`, `_steps_since_refresh`, and the hyper-parameters. It does not store baselines, game, H, springs, or RNG streams.

---

## 14. What the framework deliberately does not do

These are design choices, not omissions:

- **No per-phase strategy state.** Only the advantage baselines are per-phase; `F`, `X̄`, and `G` are shared across phases.
- **No windows.** The recurrences are infinite-horizon EWMAs. There is no rolling buffer to prune.
- **No setpoints in H.** `g_ref`, `σ_ref`, and `S_c` are units of measurement, not preferred values. H regulates relative shares, not absolute parameter values.
- **No trust region or target band in the game.** Beyond the ordinary per-parameter bounds and the simplex share floor, the hard constraints are type/sign identity (an inhibitory weight cannot become excitatory, a Poisson population cannot become STDP, a population model cannot change). The game has no setpoint, target band, trust radius, or per-parameter stability clamp.
- **No "bet" in the game.** The stack is a share, not a wager. The throttle is an entitlement, not a fee. Unused entitlement is not transferred.
- **No optimization of `N`.** Population size is the realization substrate.
- **No bio-plausibility claim from the algorithm comparison.** The phase gate and biological acceptance are separate CPG/phase gates.

---

*End of file.*
