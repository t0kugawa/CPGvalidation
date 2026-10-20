"""Performance and accuracy harness for the recursive Fisher inverse.

The harness is deliberately independent of NEST: it loads
``persona/strategy.py`` directly instead of importing the ``persona`` package
(whose ``__init__`` pulls in the NEST-backed registry). It writes
spreadsheet-ready CSV/XLSX tables and PNG figures.

Run the full suite of measurements with::

    python bench_recursive_inverse.py --out benchmarks/recursive_inverse

Use ``--quick`` for a fast smoke run.
"""
import argparse
import importlib.util
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

BETA = 0.995
LAMBDA = 1e-3
WARMUP = 3
# Observed validation path: 129 phase-key changes in 400 steps -> ~2.9 steps
# between refreshes. A 3-step interval is the conservative whole-number model.
CHATTER_EVERY = 3
MODES = ("direct", "recursive_no_refresh", "recursive_kmax",
         "recursive_chatter")


def _load_fisher_matrix():
    path = Path(__file__).resolve().parent / "persona" / "strategy.py"
    spec = importlib.util.spec_from_file_location("_persona_strategy_bench", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FisherMatrix


FisherMatrix = _load_fisher_matrix()


def _layout(n):
    """Split n channels into sites and noise producers for a synthetic run."""
    n = max(2, int(n))
    n_noise = min(16, max(1, n // 4))
    n_sites = max(1, (n - n_noise) // 2)
    return ([f"p{i}" for i in range(n_sites)],
            [f"n{i}" for i in range(n_noise)])


def _performance_rows(n_values, seeds, steps, refresh_every, repeats):
    rows = []
    for n in n_values:
        n = int(n)
        total = WARMUP + steps
        for seed in seeds:
            rng = np.random.default_rng(1000 * n + seed)
            scores = rng.normal(0.0, 0.1, size=(total, n))
            gradients = rng.normal(0.0, 1.0, size=(total, n))
            eye = np.eye(n)
            for mode in MODES:
                for repeat in range(max(1, repeats)):
                    F = np.zeros((n, n), dtype=np.float64)
                    P = np.eye(n, dtype=np.float64) / LAMBDA

                    def one_step(t, mode=mode):
                        nonlocal F, P
                        s = scores[t]
                        F = BETA * F + (1.0 - BETA) * np.outer(s, s)
                        if mode == "direct":
                            return np.linalg.solve(F + LAMBDA * eye, gradients[t])
                        refresh = (
                            (mode == "recursive_kmax"
                             and (t + 1) % refresh_every == 0)
                            or (mode == "recursive_chatter"
                                and ((t + 1) % CHATTER_EVERY == 0
                                     or (t + 1) % refresh_every == 0)))
                        if refresh:
                            cf = cho_factor(F + LAMBDA * eye, lower=True,
                                            check_finite=False)
                            P = cho_solve(cf, eye, check_finite=False)
                        else:
                            u = P @ s
                            v = float(s @ u)
                            denom = BETA + (1.0 - BETA) * v
                            if denom <= 0.0 or not np.isfinite(denom):
                                cf = cho_factor(F + LAMBDA * eye, lower=True,
                                                check_finite=False)
                                P = cho_solve(cf, eye, check_finite=False)
                            else:
                                P = (P - ((1.0 - BETA) / denom)
                                     * np.outer(u, u)) / BETA
                            P = 0.5 * (P + P.T)
                        return P @ gradients[t]

                    for t in range(WARMUP):
                        one_step(t)
                    t0 = time.perf_counter()
                    for t in range(WARMUP, total):
                        one_step(t)
                    elapsed_ms = (time.perf_counter() - t0) / steps * 1e3
                    rows.append({"n": n, "mode": mode, "seed": int(seed),
                                 "repeat": repeat,
                                 "step_ms": float(elapsed_ms)})
    return pd.DataFrame(rows)


def _accuracy_rows(n_values, seeds, steps, refresh_every):
    rows = []
    # Non-zero sum: g = G @ [1, 1] must not cancel when A = [1, -1].
    advantage = np.array([1.0, 0.25])
    for n in n_values:
        n = int(n)
        names, noise = _layout(n)
        for seed in seeds:
            rng = np.random.default_rng(9000 * n + seed)
            fm = FisherMatrix(param_names=names, noise_names=noise,
                              beta=BETA, lambda_reg=LAMBDA, warmup=0,
                              refresh_every=refresh_every, lr_mu=0.1)
            eye = np.eye(fm.n_total)
            for step in range(steps):
                sites = {
                    name: (float(rng.normal(0.0, 0.2)),
                           float(rng.uniform(0.2, 0.8)), 0.0)
                    for name in names
                }
                noise_values = {
                    name: (float(rng.normal(0.0, 0.1)), 1.0)
                    for name in noise
                }
                fm.step(sites, noise_values, advantage)

                g = fm.G.sum(axis=1)
                g[2 * fm.n_sites:] = 0.0
                k = int(fm._steps_since_refresh)
                ridge = LAMBDA if k == 0 else LAMBDA * BETA ** k
                expected_decay = np.linalg.solve(fm.F + ridge * eye, g)
                expected_const = np.linalg.solve(fm.F + LAMBDA * eye, g)
                got = fm.P @ g

                norm_decay = max(float(np.linalg.norm(expected_decay)), 1e-12)
                norm_const = max(float(np.linalg.norm(expected_const)), 1e-12)
                norm_got = float(np.linalg.norm(got))
                cosine = float(np.dot(got, expected_const)
                               / max(norm_got * norm_const, 1e-12))
                site_rows = 2 * fm.n_sites
                dmu_error = (np.linalg.norm(got[:site_rows]
                                            - expected_const[:site_rows])
                             / max(float(np.linalg.norm(
                                 expected_const[:site_rows])), 1e-12))
                rows.append({
                    "n": n,
                    "seed": int(seed),
                    "step": step + 1,
                    "k": k,
                    "rel_error_decaying": float(
                        np.linalg.norm(got - expected_decay) / norm_decay),
                    "rel_error_constant": float(
                        np.linalg.norm(got - expected_const) / norm_const),
                    "cosine_constant": cosine,
                    "dmu_rel_error_constant": float(dmu_error),
                })
    return pd.DataFrame(rows)


def _summarize(performance, accuracy):
    rows = []
    for keys, sub in performance.groupby(["n", "mode"], sort=True):
        group = f"n={keys[0]},mode={keys[1]}"
        vals = sub["step_ms"].to_numpy(dtype=float)
        rows.append(_summary_row("performance", group, "step_ms", vals))
    for keys, sub in accuracy.groupby(["n", "k"], sort=True):
        group = f"n={keys[0]},k={keys[1]}"
        for metric in ("rel_error_decaying", "rel_error_constant",
                       "cosine_constant", "dmu_rel_error_constant"):
            rows.append(_summary_row("accuracy", group, metric,
                                     sub[metric].to_numpy(dtype=float)))
    return pd.DataFrame(rows)


def _summary_row(table, group, metric, values):
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {
        "table": table,
        "group": group,
        "metric": metric,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(q25),
        "q75": float(q75),
    }


def _write_figures(performance, accuracy, out_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for mode in MODES:
        sub = (performance[performance["mode"] == mode]
               .groupby("n")["step_ms"].agg(["mean", "std"]).reset_index())
        ax.errorbar(sub["n"], sub["mean"],
                    yerr=sub["std"].fillna(0.0), marker="o",
                    capsize=3, label=mode)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("channels n")
    ax.set_ylabel("ms per step")
    ax.set_title("Recursive Fisher inverse: step cost")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "performance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    focus = min(accuracy["n"].unique(), key=lambda value: abs(value - 458))
    sub = (accuracy[accuracy["n"] == focus]
           .groupby("k")["rel_error_decaying"]
           .agg(["mean", "std"]).reset_index())
    axes[0].errorbar(sub["k"], np.maximum(sub["mean"], 1e-16),
                     yerr=sub["std"].fillna(0.0), marker="o", capsize=3,
                     label=f"n={focus}")
    axes[0].axhline(1e-12, color="black", linestyle="--", linewidth=1.0,
                    label="1e-12")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("steps since refresh")
    axes[0].set_ylabel("relative error vs direct solve")
    axes[0].set_title("Same-ridge recursion")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    for n in sorted(accuracy["n"].unique()):
        sub = (accuracy[accuracy["n"] == n]
               .groupby("k")["rel_error_constant"]
               .agg(["mean", "std"]).reset_index())
        mean = np.maximum(sub["mean"].to_numpy(dtype=float), 1e-16)
        std = sub["std"].fillna(0.0).to_numpy(dtype=float)
        axes[1].fill_between(sub["k"], np.maximum(mean - std, 1e-16),
                             mean + std, alpha=0.12)
        axes[1].plot(sub["k"], mean, marker="o", label=f"n={n}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("steps since refresh")
    axes[1].set_ylabel("relative error vs constant ridge")
    axes[1].set_title("Ridge drift between refreshes")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=150)
    plt.close(fig)


def _write_xlsx(out_dir, performance, accuracy, summary):
    try:
        import openpyxl  # noqa: F401
        engine = "openpyxl"
    except ImportError:
        try:
            import xlsxwriter  # noqa: F401
            engine = "xlsxwriter"
        except ImportError:
            return None
    path = out_dir / "recursive_inverse_results.xlsx"
    with pd.ExcelWriter(path, engine=engine) as writer:
        performance.to_excel(writer, sheet_name="performance", index=False)
        accuracy.to_excel(writer, sheet_name="accuracy", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
    return path


def _write_readme(out_dir):
    text = """# Recursive Fisher inverse benchmark

Generated by `bench_recursive_inverse.py`.

## Files

- `performance.csv` / `performance.png`: per-step cost for the direct
  `np.linalg.solve` update, the recursive inverse without refresh, the
  recursive inverse with a `K_max` Cholesky refresh, and the recursive
  inverse with a phase-chatter refresh every 3 steps (`recursive_chatter`).
  The chatter mode models the observed validation path (129 phase-key
  changes in 400 steps, an effective interval of about 2.9); `recursive_kmax`
  is the stable-phase upper bound. Times are milliseconds per step, measured
  after a short warm-up.
- `accuracy.csv` / `accuracy.png`: for each synthetic step, the relative
  error of `P @ g` against the direct solve with the same decaying ridge
  (`rel_error_decaying`) and against the constant-ridge solve
  (`rel_error_constant`). The latter is the controlled ridge drift; it is
  zero at a refresh and grows until the next refresh. `cosine_constant` and
  `dmu_rel_error_constant` measure the effect on the μ and σ blocks of the
  sites (the first `2*n_sites` channels); noise rows are excluded.
- `summary.csv`: mean, standard deviation, median, and quartiles of every
  metric, grouped by `(n, mode)` for performance and `(n, k)` for accuracy.
- `recursive_inverse_results.xlsx`: the same three tables in one workbook
  when `openpyxl` or `xlsxwriter` is installed.

## Reading the figures

`performance.png` is log-log. At large `n`, the direct solve grows fastest;
the stable-phase recursive modes should be lower. The chatter mode refreshes
every few steps and can approach the direct cost, so read it as the realistic
phase-churn case and `recursive_kmax` as the stable-phase upper bound.
`accuracy.png` has two panels: the left shows that the recursion matches the
direct solve with the same decaying ridge to near machine precision; the
right shows the ridge drift against the constant-ridge solve, with a shaded
standard-deviation band. The drift resets at each refresh.

## Run

```bash
python bench_recursive_inverse.py --out benchmarks/recursive_inverse
```

Use `--quick` for a small smoke run and `--no-figures` to skip plotting.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run_benchmark(out_dir, n_values=(64, 128, 256, 458, 512, 1024),
                  seeds=(0, 1, 2), steps=30, refresh_every=10, repeats=3,
                  make_figures=True, accuracy_steps=None,
                  accuracy_n_values=None):
    """Run the timing and accuracy measurements and write the artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_values = tuple(int(n) for n in n_values)
    seeds = tuple(int(s) for s in seeds)
    steps = max(1, int(steps))
    refresh_every = max(1, int(refresh_every))
    repeats = max(1, int(repeats))
    if accuracy_steps is None:
        accuracy_steps = max(40, 4 * steps)
    accuracy_steps = max(1, int(accuracy_steps))
    if accuracy_n_values is None:
        accuracy_n_values = n_values
    accuracy_n_values = tuple(int(n) for n in accuracy_n_values)

    performance = _performance_rows(n_values, seeds, steps,
                                    refresh_every, repeats)
    accuracy = _accuracy_rows(accuracy_n_values, seeds, accuracy_steps,
                              refresh_every)
    summary = _summarize(performance, accuracy)

    performance.to_csv(out_dir / "performance.csv", index=False)
    accuracy.to_csv(out_dir / "accuracy.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if make_figures:
        _write_figures(performance, accuracy, out_dir)
    _write_xlsx(out_dir, performance, accuracy, summary)
    _write_readme(out_dir)
    return {"performance": performance, "accuracy": accuracy,
            "summary": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="benchmarks/recursive_inverse",
                        help="output directory for CSVs, XLSX, and figures")
    parser.add_argument("--quick", action="store_true",
                        help="small smoke run")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip PNG generation")
    args = parser.parse_args(argv)

    if args.quick:
        result = run_benchmark(
            args.out, n_values=(64,), seeds=(0,), steps=8,
            refresh_every=10, repeats=1, make_figures=not args.no_figures,
            accuracy_steps=24, accuracy_n_values=(64,))
    else:
        result = run_benchmark(
            args.out, n_values=(64, 128, 256, 458, 512, 1024),
            seeds=(0, 1, 2), steps=30, refresh_every=10, repeats=3,
            make_figures=not args.no_figures, accuracy_steps=120,
            accuracy_n_values=(64, 128, 256, 458, 512))
    print(f"wrote {len(result['performance'])} performance rows, "
          f"{len(result['accuracy'])} accuracy rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
