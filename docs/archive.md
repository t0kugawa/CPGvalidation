# Archive map

`archive/` is local and ignored by Git. Old tracked files moved there remain in
Git history; untracked/ignored files are local only.

| Old path | New path | Notes |
|---|---|---|
| `validation_2min/` | `archive/validation_2min/` | stale attached 2-minute output |
| `validation_mvp/`, `validation_mvp_b/`, `validation_mvp_direct_current/`, `validation_mvp_recursive/`, `validation_mvp_recursive_repeat/` | `archive/validation_mvp*/` | historical MVP runs |
| `validation_speed/` | `archive/validation_speed/` | old thread profiles; report moved to `docs/reports/single_cpg_speed.md` |
| `validation_st_vs_mt/` | `archive/validation_st_vs_mt/` | removed replica comparison; report moved to `docs/reports/st_vs_mt.md` |
| `test_results/` | `archive/test_results/` | 836 MB of historical probes and runs; local only (2026-11-17: remaining subdirs merged in; only the Excel-locked `final_90s/run_log.csv` stayed behind) |
| `agent_sessions/` | `archive/agent_sessions/` | historical agent ledgers; local only |
| `rat_hindlimb_0.2/` | `archive/legacy_models/rat_hindlimb_0.2/` | unused OpenSim source model |
| root `*.png`, `run_log.*`, `MUJOCO_LOG.TXT` | `archive/root_artifacts/` | ignored generated artifacts (2026-11-17: `run_90s.log` added) |
| `.pytest_cache/`, `__pycache__/` | `archive/cache/` | local caches, delete freely |
| `.commitmsg` | `docs/archive/commitmsg-2026-10-18.md` | historical commit-message draft |
| `2026-10-23-persona-multi-sample-parallel-design.md` | `docs/superpowers/archive/` | rejected replica design; kept for history |

Current evidence lives in `experiments/current/`. Historical docs and specs may
still cite the old top-level paths; those references are snapshots, not live
layout.
