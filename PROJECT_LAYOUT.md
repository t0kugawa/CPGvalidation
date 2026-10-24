# Project layout

This is the working map for the repository. Paths that hold generated data are
ignored by Git unless explicitly noted.

## Production code

- `persona/` — PERSONA agent, strategy, registry, game, adapters, validation.
- `t2xcpg/` — production NEST CPG (brainstem, leg, Renshaw, M-F).
- `model/` — MuJoCo plant and model configuration.
- `loader/` — mocap, stats, and treadmill loaders.
- `dual_rat_hindlimb.xml` — the plant XML at the repository root (the default
  relative path used by the model and tests).
- `tests/` — pytest suite.

## Reference implementations (read-only)

- `tinycpg/` — NEST reference circuit. The paper and specs cite it; do not edit.
- `memcpg/` — NEURON reference circuit. The paper and specs cite it; do not edit.

These are deliberately kept at the root so the paths in `docs/main.tex` and the
design records remain valid.

## Entry points

- `run_2min.py` — primary 2-minute personalization run.
- `run_validation.py` — build/run validation with HDF5 provenance.
- `personalize.py` — library-level builder and runner.
- `bench_game.py` — conserved-stack game A/B harness (writes `benchmarks/game/`).
- `bench_homeostasis.py` — homeostasis benchmark (writes
  `benchmarks/homeostasis_h/`).
- `bench_recursive_inverse.py` — recursive-inverse benchmark (writes
  `benchmarks/recursive_inverse/`).
- `scripts/read_h5.py` — rebuild validation figures from a saved `run.h5`.

## Data, results, and history

- `data/` — input datasets (ignored; large).
- `benchmarks/` — tracked benchmark tables, JSON, and reports.
- `experiments/current/` — tracked current evidence:
  - `check_current_2s/` — 2 s game-off control.
  - `check_game_2s/` — 2 s selected-game run.
  - `nest_threads/` — one-CPG NEST-thread sweep and `realize_every_k` run.
- `archive/` — local, ignored archive of old runs and generated artifacts.
  Tracked files moved here remain in Git history; untracked/ignored files are
  local only. See `docs/archive.md` for the mapping.
- `docs/` — paper, figures, specs, plans, reports, and the backlog.
- `docs/superpowers/specs/`, `docs/superpowers/plans/` — design and
  implementation records.
- `docs/reports/` — operational summaries and measurements.
- `docs/superpowers/archive/` — superseded or rejected designs.

## Commands

```powershell
# Fast current run (3x faster than realize-every-k 1 in the short sweep)
python run_2min.py --out experiments/current/2min_120s --no-render `
  --duration-s 120 --snapshot-s 60 --realize-every-k 10

# Full-fidelity realization
python run_2min.py --out experiments/current/2min_120s_k1 --no-render `
  --duration-s 120 --snapshot-s 60 --realize-every-k 1

# Tests
python -m pytest -q

# Benchmark smoke
python bench_game.py --smoke

# Rebuild figures from a saved run
python scripts/read_h5.py --h5 experiments/current/2min_120s/run.h5 --out figures_out
```

Defaults: `--game selected`, one CPG, one MuJoCo, `--nest-threads 1`. NEST
threads greater than one measured slower on this network; see
`docs/reports/nest_threads.md`.

## Line endings

`.gitattributes` keeps text files in LF and marks binary artifacts. Use
`git restore .` if a Windows editor reintroduces CRLF-only diffs.
