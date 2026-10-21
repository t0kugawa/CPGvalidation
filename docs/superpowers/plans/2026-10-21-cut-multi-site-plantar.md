# CUT multi-site plantar plan (update)

**Status:** accepted for implementation 2026-10-21. Option A is in scope now; Option B follows the PERSONA update. T0 has been run.
**Supersedes:** the single-site CUT table in `docs/superpowers/specs/2026-10-20-persona-variantB-cpg-vcells-design.md` §3.4, and the stale `cut_heel_0_{side}` touch-sensor test in `docs/superpowers/plans/2026-10-14-persona-cpg-personalization.md` Task 8.
**Authority:** the `TODO(cut-multi-site)` in `model/cut.py` (user-specified) + the user's bio-CUT specification (2026-10-21) + the existing `rct_*` sites in `dual_rat_hindlimb.xml` + the 120 s run log.

---

## 1. What I found

### 1.1 Spec search

- `docs/superpowers/specs/`: **no multi-site CUT specification exists.** The current vcells spec §3.4 is single-site (`CUT→RGE 14.0`, `CUT→InE 6.0`); the deleted v2 circuit spec is also single-site.
- `docs/superpowers/plans/2026-10-14-persona-cpg-personalization.md` Task 8 test expects `cut_heel_0_{side}` **touch sensors**, but the XML has `rct_*` **sites** and `model/leg.py` sums contacts directly. That plan is stale.
- `model/cut.py` `TODO(cut-multi-site)` is the only multi-site design. It says: N Poisson generators across the sole (heel / metatarsal pads / toe), active-list gating by contact, per-site force/area → local Hz, per-site STDP to RG-E (or a shared projection gated by the active list).
- `dual_rat_hindlimb.xml` **already has the site layout**: 8 RCT sites per controlled foot — `rct_heel_0/1/2`, `rct_mid_0/1`, `rct_fore_0/1` on the foot body and `rct_toe_0` on the toe body. They are currently unused by the CUT. No new XML sites are needed.

So the plan updates the single-site CUT to the TODO design, reusing the existing RCT sites.

### 1.2 Run-log evidence (120 s, 0.2 ms, 12 000 steps, `ref_hz=700`, `realize_every_k=1`)

The h5 confirms `resolution_ms=0.2`, so this is not the old 1.0 ms numerical ceiling.

| channel | result |
|---|---|
| `mus_E_L` | clipped ≥0.999 for **98.0 %** of the run; first clip at **3.67 s** |
| `mus_F_L` | clipped **98.4 %**; first clip at **2.03 s** |
| `mus_E_R` | clipped **98.7 %**; first clip at **1.64 s** |
| `mus_F_R` | **never clipped** (no ≥0.2 s clipped interval); alive **112.4 s** of 120 s, mean 0.34 |
| `rg_e_L` / `rg_f_L` | means **0.999 / 0.992**; both >0.5 for **99 %** of the run; CI = min/max = **0.993** |
| `rg_e_R` / `rg_f_R` | means **0.989 / 0.234**; CI = **0.237** |
| `phase_L` / `phase_R` | stance **76.1 % / 77.9 %**; both stance **57.7 %**, both swing **3.7 %**; **2296 / 2424 transitions** in 120 s (~19/s) |
| `cut_L` / `cut_R` | mean **91.3 / 93.4 Hz**; **39 % / 42 %** at ≥119 Hz; **8 % / 6 %** at ≤1 Hz |
| ankle L / R | RMSE **1.067 / 1.240 rad**; model mean **1.99 / 2.08** vs target **1.13 / 1.01**; model >2.5 rad for **34 % / 38 %** |
| loss | 0.828 at 0–10 s → 1.054 at 110–120 s; **no downward trend** |

**Answer to “flexor R only alive for 10 seconds”:** the full log shows the opposite. Flexor R is the **only muscle alive for the whole run** (112.4 s, never clipped ≥0.2 s); the 10 s is the **figure window** (60–70 s), not the activity duration. The other three are the ones that die, after 1.6–3.7 s.

### 1.3 CPG bug table from this run

| id | observation | cause | in this plan? |
|---|---|---|---|
| B1 | left RG E/F co-contract (CI 0.993), right E dominates (E 0.99, F 0.23) | the **policy means** are inflated 4–20×: `BASE_in_rg_e_L` 1→15.4, `rg_e_L→rg_e_L` 4→20.6, `rg_f_L→rg_f_L` 5.5→30.5, `rg_f_L→m_f_L` 2.75→15.3, `rg_e_R→m_e_R` 2.75→15.6, while `rg_f_R→m_f_R` 2.75→0.92. The earlier open-loop seed probe alternated (RG L 378/129, R 236/75), so the **seed** half-centre is not latched; this run is latched by the policy drift. | no — persona (user) |
| B2 | phase chatters ~19/s; `both stance` 57.7 % | `phase = "stance" if total_contact_force > 0.005 else "swing"` on the raw, noisy contact force (`model/leg.py:466`) | **yes — T5** |
| B3 | CUT is a 0/120 contact switch (39–42 % at 120; mean 91–93) | one scalar `CUTAfferent` on total foot force; sigmoid saturates above ~0.4 N; no site/CoP code | **yes — T1–T4** |
| B4 | `cut_rge_L` depressed (~15–35), `cut_rge_R` potentiated (~112, near Wmax) | single-site CUT has one phase per leg, so L/R learn opposite signs; the STDP hyper-params are also policy-inflated (`alpha` 0.95→0.39/0.11, `tau_plus` 20→3.4/10.5) | **partly — T4** (per-site or graded aggregate); persona part is user |
| B5 | extensor Ib tonic 125–130 Hz; `InIb-E→RGF +18` is a tonic flexor drive | extensor saturation (B1) makes the excitatory Ib phase switch tonic | no — verify with T0; persona |
| B6 | flex R “only alive 10 s” | actually alive 112.4 s; the 10 s is the 60–70 s figure window; the other three clip after 1.6–3.7 s | no — persona |

**Conclusion:** the CPG **seed** is probably fine; this run’s latch is the persona drift (B1). The CPG work in this plan is B2 (phase), B3 (CUT), and the CUT part of B4; B1/B5/B6 stay with persona. T0 re-checks the seed before any half-centre weight is changed.

### 1.4 T0 result: PERSONA means are sufficient to cause the latch

`test_results/t0_persona_influence.py` (run 2026-10-21, FINAL sizes, 0.2 ms, 4 s, phasic CUT 120/0 alternating L/R, BS = BASE = 2 Hz, Ia = Ib = 0). The h5 policy means at 20/60/120 s were replayed as fixed values from t=0. `rg` rates are the normalized readout × 700 Hz.

| condition | E L | F L | E R | F R | CI L | CI R | both E+F L | both E+F R | L/R anti-phase | E transitions/s L/R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed | 310.9 | 5.9 | 303.9 | 5.6 | 0.02 | 0.02 | 0.02 | 0.02 | **0.92** | 4.3 / 2.0 |
| persona 20 s | 700.0 | 555.3 | 695.7 | 359.0 | 0.79 | 0.52 | **1.00** | **0.87** | **0.00** | 0 / 0 |
| persona 60 s | 637.8 | 700.0 | 578.2 | 164.3 | 0.91 | 0.28 | **1.00** | 0.42 | 0.03 | 0 / 6.0 |
| persona 120 s | 629.7 | 0.0 | 489.7 | 13.6 | 0.00 | 0.03 | 0.00 | 0.02 | 0.07 | 0 / 14.7 |

- **Seed:** L/R alternation survives (anti-phase 0.92), and there is no E/F co-contraction. Under this reduced sensory set the flexor is weak (BS 2 Hz is the only flexor drive; CUT drives the extensor), so this does **not** prove the seed flexor drive is adequate. It does show the seed is not the 120 s co-contraction.
- **Persona means:** L/R alternation collapses (anti-phase 0.00–0.07). At 20 s and 60 s both E and F are active on the left for 100 % of the sampled window (CI 0.79–0.91); the 120 s means hold both extensors on and silence F-L. The policy means are therefore **sufficient** to produce the latch; the closed-loop correlation is not merely a consequence of the latch.
- **Caveat:** the means co-evolve with the latch, so this replay is a causal intervention on fixed means, not a decomposition of the 120 s trajectory. The V-cell relay does not need a gain calibration to explain the latch; no half-centre seed weight is changed.

This is the "calculate how PERSONA influences the bug" result. The remaining CPG work is the CUT/phase fix, not a half-centre re-tune.

---

## 2. Design

### 2.0 Bio-CUT specification (user, 2026-10-21)

The target is not one sigmoid of total foot force. Every plantar site gets its own Poisson generator; the set of generators that are in contact is the **active buffer**. A generator fires from two local quantities:

1. **Contact area** — the site's effective plantar area in contact. MuJoCo contacts are point contacts, so v1 uses a fixed per-site tributary area `A_s` (a geometry prior, not a measured patch) and a binary contact gate. The area is not re-derived from total force; it is the site's share of the sole.
2. **Contact force** — the normal force assigned to that site. Local pressure is `p_s = F_s / A_s`; the site rate is a nonlinear function of pressure. The default is a saturating sigmoid, with a power-law/ Hill option for fitting the rat CUT receptor distribution:

```
lambda_s = lambda_max * rho_s * A_s * g(p_s)
g(p)     = 1 / (1 + exp(-beta * (p - p_theta)))     # default
active_s = (A_s > 0) and (F_s > F_active)
```

`rho_s` is the receptor density. v1 keeps it uniform (1.0) because we do not yet have a rat CUT density dataset; the field is exposed so a nonlinear heel/mid/fore/toe profile can be fitted later without changing the wiring. The aggregate is **not** a new receptor: it is the area-weighted active-site mean (below).

The CPG side has two stages:

- **A (now):** the model computes the 8 per-site rates and the active buffer; the CPG still has one Poisson generator per leg, driven by the aggregate. This removes the 0/120 square wave without changing the PERSONA dimension.
- **B (after the PERSONA update):** the CPG has one Poisson generator per site, each fed by its own rate and gated by the active buffer; each site has its own STDP projection to RG-E. This preserves the heel→toe spike sequence.

The user's T1 concern — "it won't work without environment feedback in the loop" — is satisfied by construction: T1 reads `data.contact` and `data.site_xpos` **after every `mj_step`** inside `Leg.step`. The site layout is fixed; the forces and the active set are live environment feedback.

### 2.1 Site layout (from the XML, not hardcoded positions)

Eight sites per controlled foot, read by name from the model:

| region | sites | body | current position (m, body frame) |
|---|---|---|---|
| heel | `rct_heel_0/1/2` | `foot_m_{side}` | `(-0.006, 0/±0.003, -0.0045)` |
| mid | `rct_mid_0/1` | `foot_m_{side}` | `(0.006, ±0.002, -0.0045)` |
| fore | `rct_fore_0/1` | `foot_m_{side}` | `(0.016, ±0.002, -0.0045)` |
| toe | `rct_toe_0` | `toe_m_{side}` | `(0.008, 0, -0.003)` |

The positions come from `data.site_xpos` after `mj_step`; the table above is only documentation. The per-site **area weight** `A_s` is a relative constant (equal 1.0 in v1). MuJoCo gives point contacts, so a measured area is not available; the TODO already accepts a fixed per-site weighting. `rho_s` (receptor density) is also 1.0 in v1 and is the hook for the nonlinear rat distribution; no fitted profile is claimed until data are supplied.

### 2.2 Contact → site assignment

Add `Leg._contact_forces_by_site(side)`:

1. Collect the 8 site IDs for the side once in `__init__`.
2. For each contact whose `geom1`/`geom2` body is in `foot_bodies[side]`, take the contact position `data.contact[i].pos` and the normal force `_contact_force_i(i)`.
3. Assign the contact to the nearest site by Euclidean distance in world coordinates (the sites are on the plantar surface).
4. Sum the normal force per site. The sum over sites equals the current `_contact_force(side)` (minus any contact that is closer to a site of the other foot; filter by side first).
5. Replace the scalar `CUTAfferent` per side with 8 `CUTAfferent` instances (one per site), each `get_cut_hz(local_force / A_s)`. The low-pass is per site, so a site that unloads decays independently.
6. Active list: `active = (A_s > 0) and (local_force > ACTIVE_FORCE_N)` (default 0.01 N). An inactive site’s output rate is **0**; the low-pass state is reset (or decayed) so a re-contact starts clean. The active buffer is the ordered tuple of site indices with `active=True`.

### 2.3 Aggregate CUT

The CPG still needs one scalar `cut_L/R` for the existing `CUT→InE` path and the noise producer. Define the **area×density-weighted mean of the active sites**:

```
w_s    = A_s * rho_s
cut_hz = sum(w_s * rates[s] for s active) / sum(w_s for s active)   if any active
       = 0.0                                                        otherwise
```

This stays on the current 0–120 Hz scale, weights a loaded heel more than an equally loaded toe, and reduces to the plain mean when all weights are equal. The per-site rates are the population code; the aggregate is only the back-compatible scalar. `max` and unweighted mean are not used because they either discard area or let a lightly loaded toe dominate.

### 2.4 CPG wiring — two options

**Option A (approved now): graded aggregate, no persona-dimension change.**
- Keep the single `cut_pg_{leg}` / `cut_in_{leg}` relay and the single `CUT→RGE` STDP (`cut_rge_{leg}`) and `CUT→InE`.
- Feed `cut_pg` the **aggregate** `cut_hz` from §2.3.
- The per-site rates, the active buffer, and the site weights are computed in the model; the CPG sees a graded, active-list-gated aggregate instead of a 0/120 switch.
- Persona impact: **none** (same producers, same STDP keys, same parameter count).
- Limitation: the CPG does not see the heel→toe **sequence** as separate spike trains; the per-site STDP cannot learn site-specific phases.

**Option B (next, after the PERSONA update): per-site Poisson generators, per-site STDP.**
- Replace the single relay with 8 per-site relays per leg, `n_aff_site = max(1, n_aff // 8)` = 12 at FINAL (so the total afferents per leg stay ~100 and the total CUT synapse count stays ~3500).
- Each site’s Poisson group is fed by its own rate and is **in the active buffer only while that site is in contact**; inactive sites get rate 0 and are not allowed to drive STDP.
- Each site → `rg_e` through its own STDP projection, key `cut_rge_{leg}_s{k}`; each site → `in_e` through a static projection (weight 6.0, p 0.30) so the total CUT→InE drive is preserved.
- Per-site rates: `cut_afferent[site].get_cut_hz(local_force / A_s)`, gated by `active`.
- Persona impact: **+16 STDP projections** (8 per leg) × 7 hyper-parameters = **+112 optimizable params**, and +16 STDP producer recorders. The registry/Fisher dimension grows; the user’s persona code must handle it. The Poisson groups can be created with `noise_name=None` so the CUT **noise** features stay `cut_L/R`; the STDP weight-recorder producers still grow.
- Limitation: this is the full phase-sequence design, but it requires the persona update first.

**Decision:** implement **A** now; implement **B** as the next step after the user’s persona update. A is not a substitute for B: it fixes the square wave and gives the model-side per-site population code, while B is what lets the CPG see the heel→toe sequence as separate spike trains.

### 2.5 Phase hysteresis (B2)

Replace the raw threshold at `model/leg.py:466` with a Schmitt trigger + debounce on the **total contact force** (or the aggregate CUT):

- `on` when `total_contact_force > 0.010 N`;
- `off` when `< 0.005 N`;
- require the new state to hold for `PHASE_DEBOUNCE_MS = 50` before switching.

The first 4 s probe used 20 ms and still chattered at 5.0/7.3 transitions per second. Replaying the recorded contact force through the detector at 50 ms gave 2.7/3.0 per second and left the stance fraction at 0.53/0.43, so 50 ms is the smallest measured setting that meets the 2–4/s acceptance. The target gait’s stance/swing is ~0.4–0.6 s, so a 50 ms debounce costs at most ~12 % of a phase; the persona baselines see a slightly delayed but clean phase. The phase is still a `str` in `CPGInput`.

### 2.6 Logging

- `CPGInput` gains `cut_sites_L: tuple[float, ...]` and `cut_sites_R` (8 values each). `cut_L/R` remains the aggregate for back-compat.
- `model/leg.py` fills them from the per-site rates; `model/adapter.py` declares them.
- The CSV/h5 logging lives in `persona/personalization.py` (`RunLog`), which is **user-owned**. This plan does **not** edit it. The model-side values are available for a standalone diagnostic; the user can add `cut_site_*` fields when ready.
- A `test_results` diagnostic plot: per-site rates (heel→toe), aggregate CUT, phase, RG E/F, ankle angle. Not part of the production figure set.

---

## 3. Tasks

| id | task | files | test |
|---|---|---|---|
| T0 | **Seed diagnostic (no persona):** drive the CPG at seed weights with a phasic CUT (120 Hz stance / 0 swing, alternating L/R) + BS 2 Hz at 0.2 ms; measure RG E/F and L/R phase. Confirms whether B1/B5 are persona-only and whether the V-cell relay needs gain calibration. | `test_results/t0_persona_influence.py` | **done 2026-10-21**; §1.4 |
| T1 | Site IDs + `_contact_forces_by_site` + per-site `CUTAfferent` array + active list | `model/leg.py`, `model/cut.py` | `tests/test_cut_sites.py` (synthetic contact → nearest site; force sums; inactive = 0) |
| T2 | Aggregate `cut_hz = area×density-weighted mean(active rates)`; keep `cut_L/R` scale | `model/leg.py` | aggregate bounds 0–120; no active → 0 |
| T3 | `CPGInput.cut_sites_L/R` + `cut_active_L/R` + `Leg.step` return | `model/adapter.py`, `model/leg.py` | adapter field test |
| T4a | **Option A:** feed the aggregate through the existing `cut_L/R` → `cut_pg`/`cut_in` path; **no `t2xcpg` change** | `model/leg.py`, `model/adapter.py` | CUT→RGE key unchanged; registry recorder test |
| T4b | **Option B (after persona):** 8 per-site Poisson relays, active buffer, `cut_rge_{leg}_s{k}` STDP, per-site CUT→InE | `t2xcpg/leg.py`, `t2xcpg/cpg.py` | per-site keys exist; active sites only; synapse count ~3500/leg; registry test |
| T5 | Phase Schmitt trigger + 20 ms debounce | `model/leg.py` | noisy contact sequence → transitions ≤ 1 per debounce |
| T6 | Per-site logging in `RunLog` (user-owned) | `persona/personalization.py` | **not in this plan**; user adds |
| T7 | **4 s only** diagnostic at 0.2 ms with A; check per-site sequence, aggregate, phase, RG | `run_validation.py --fast` | n/a |
| T8 | **120 s final validation after A/B and the persona fix; user runs it** | server | n/a |

---

## 4. Acceptance

- **No 0/120 square wave:** the aggregate CUT varies with local force/CoP within stance; the per-site rates show a heel→toe sequence (heel leads, toe lags).
- **Phase:** ~2–4 transitions/s per leg, not 19/s; `both stance` no longer 58 % from chatter.
- **CPG:** with the persona fix in place, RG E/F alternates (CI << 0.99) and the ankle RMSE falls; if the seed diagnostic already alternates, no half-centre seed change is made.
- **Tests:** the existing 174 tests pass; `tests/test_cut_sites.py` covers contact→site, active list, aggregate, phase debounce, and (Option B) per-site keys.
- **Cost:** Option B stays at ~100 CUT afferents and ~3500 CUT synapses per leg; the 4 s diagnostic measures wall time.

---

## 5. Decisions (resolved 2026-10-21)

1. **A then B.** Implement A now; B follows the PERSONA update. The user's bio-CUT description (per-site Poisson, active buffer, area × force, nonlinear receptor density) is the B target. A is the aggregate stepping stone.
2. **Site area / density:** equal `A_s = 1.0`, `rho_s = 1.0` in v1. `rho_s` is a configurable nonlinear hook, not a fitted rat distribution; no data are claimed. The site layout comes from the XML, not hardcoded indices.
3. **Aggregate:** area×density-weighted mean of the active sites (§2.3). This keeps the 0–120 Hz scale and uses contact area explicitly.
4. **Per-site afferent count (B):** `max(1, n_aff // 8)`, i.e. 12 per site at FINAL; total ~96 per leg, same cost as the current single relay's 100.
5. **Phase:** Schmitt trigger on 0.010 N, off 0.005 N, **50 ms** debounce. Approved concept (T5); 20 ms measured 5–7/s, 50 ms measured 2.7–3.0/s, so the tuning is data-driven. Cost if wrong: ~30 ms extra phase lag.
6. **Site names:** reuse the XML `rct_*` names as the ordered tuple in `CPGInput.cut_site_names`; no new aliases. The tuple order is the XML order, documented in the model.
7. **V-cell relay:** T0 shows the seed retains L/R alternation (anti-phase 0.92) and the persona means abolish it (0.00–0.07); no relay-gain calibration is in scope. The relay remains a separate CPG ticket if a future seed-only probe with physiological Ia/Ib shows it is too weak.
8. **T1 concern:** T1 is live environment feedback — it reads MuJoCo contacts after every `mj_step`. The only fixed quantities are the XML site geometry and the area/density priors.
9. **Run scope:** T7 is 4 s only. The 120 s FINAL run is the user's, after the PERSONA update; no `--fast` for the final run.

---

## 6. Out of scope

- Any `persona/` change (the user is editing the policy; B1/B5/B6 stay there).
- Half-centre seed weights, Ib gains, and muscle/tendon parameters — only revisit after T0 shows a seed-level CPG defect.
- Target-limb CUT (the target limb is a kinematic mirror; the XML has RCT sites only on the controlled feet).
