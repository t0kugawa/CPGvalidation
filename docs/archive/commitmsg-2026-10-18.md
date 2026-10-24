Docs: record the long-run readiness audit (2026-10-18 (d), V21)

Adds the 12000-step readiness findings to the plan and the Variant-B spec:
- the RGE/RGF spike recorders are read *and* cleared each step (they were
  accumulating every rhythm-generator spike for the whole run), and their
  normalised activity is now logged as rg_* and plotted;
- NoiseRecorder.samples is a bounded deque;
- measured memory: RSS 403 MB after build -> 448.8 MB at step 1550 (early slope
  41.5 kB/step including allocator warm-up, late slope 18.0 kB/step = the
  Agent.history), projecting ~640-760 MB at 12000 steps;
- measured step rate at --final sizes: 1550 steps in a 700 s budget = ~0.45 s/step
  (~90 min for 12000 steps);
- MuJoCo 3.14.0 enforces the joint limits (jnt_limited=True, soft), so the
  observed +2.947 rad ankle is a constraint violation, not a missing limit;
- in the 4 s dataset-belt run the cut_rge_L STDP weight grows 14.3 -> 27.9
  monotonically (Wmax=120) while ia_* stay flat at ~11.8, and the tracking-stage
  loss is flat at ~1.2: personalization moves the parameters but not the loss,
  which the saturated ankle (C1) dominates.
