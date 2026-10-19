"""HDF5 persistence for a personalisation/validation run.

One file holds everything a reader needs to reproduce the figures and evaluate
the run:
  * log/    — per-step arrays (angles, muscle drives, CUT/Ia/Ib afferents,
              phases, contacts, foot positions, loss and advantage),
  * stdp/   — the STDP weight-recorder trace (per-step mean delta and update
              count, sampled live weight distribution, final weight vectors),
  * agent/  — the parameter mean/spread histories, the per-step influence, the
              advantage/error vectors and the final Fisher metric matrix.
"""
import numpy as np
import h5py


def _channel_names(log):
    cls = type(log)
    return list(getattr(cls, "_NUMERIC", ())), list(getattr(cls, "_STR", ()))


def save_run(path, log, agent, meta=None):
    numeric, string = _channel_names(log)
    with h5py.File(path, "w") as f:
        for k, v in (meta or {}).items():
            f.attrs[k] = v

        g = f.create_group("log")
        for k in numeric:
            v = getattr(log, k, None)
            if v is not None and len(v):
                g.create_dataset(k, data=np.asarray(v, dtype=float))
        for k in string:
            v = getattr(log, k, None)
            if v is not None and len(v):
                g.create_dataset(k, data=np.asarray(v, dtype=h5py.string_dtype()))

        stdp = getattr(log, "stdp", None) or {}
        sg = f.create_group("stdp")
        for grp in ("delta", "n", "weight_mean", "weight_min", "weight_max",
                    "final_weights"):
            if grp not in stdp:
                continue
            gg = sg.create_group(grp)
            for k, v in stdp[grp].items():
                gg.create_dataset(k, data=np.asarray(v, dtype=float))
        for k in ("time_ms", "sample_time_ms"):
            if k in stdp:
                sg.create_dataset(k, data=np.asarray(stdp[k], dtype=float))

        hist = getattr(agent, "history", {}) or {}
        ag = f.create_group("agent")
        for grp in ("means", "sigmas"):
            gg = ag.create_group(grp)
            for k, v in hist.get(grp, {}).items():
                gg.create_dataset(k, data=np.asarray(v, dtype=float))
        for grp in ("influence",):
            v = hist.get(grp)
            if v is not None and len(v):
                ag.create_dataset(grp, data=np.asarray(v, dtype=float))
        for grp in ("loss", "advantage_vector", "error_vector"):
            v = hist.get(grp)
            if v is not None and len(v):
                ag.create_dataset(grp, data=np.asarray(v, dtype=float))
        strategy = getattr(agent, "strategy", None)
        if strategy is not None:
            ag.attrs["param_names"] = list(getattr(strategy, "param_names", []))
            ag.attrs["noise_names"] = list(getattr(strategy, "noise_names", []))
            ag.attrs["noise_scales"] = [
                float(getattr(strategy, "noise_scales", {}).get(n, 0.0))
                for n in getattr(strategy, "noise_names", [])]
            ag.attrs["all_names"] = list(getattr(strategy, "all_names", []))
            try:
                ag.create_dataset("fisher",
                                  data=np.asarray(strategy.F, dtype=float))
                ag.create_dataset("feature_scales",
                                  data=np.asarray(strategy.feature_scales(), dtype=float))
                ag.create_dataset("trace_final",
                                  data=np.asarray(strategy.Xbar, dtype=float))
                ag.create_dataset("credit_final",
                                  data=np.asarray(strategy.G, dtype=float))
            except Exception:
                pass


def load_run(path):
    """Return (data, meta); data['log'] holds arrays, data['agent'] the histories."""
    out = {"log": {}, "stdp": {}, "agent": {"means": {}, "sigmas": {}}}
    with h5py.File(path, "r") as f:
        meta = {k: f.attrs[k] for k in f.attrs}
        for top in ("log", "stdp"):
            if top not in f:
                continue
            for k, v in f[top].items():
                if isinstance(v, h5py.Group):
                    out[top].setdefault(k, {})
                    for kk, vv in v.items():
                        out[top][k][kk] = vv[()]
                    continue
                arr = v[()]
                if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
                    arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
                out[top][k] = arr
        if "agent" in f:
            for grp in ("means", "sigmas"):
                if grp in f["agent"]:
                    for k, v in f["agent"][grp].items():
                        out["agent"][grp][k] = v[()]
            for grp in ("influence", "loss", "advantage_vector", "error_vector",
                        "fisher", "feature_scales", "trace_final", "credit_final"):
                if grp in f["agent"]:
                    out["agent"][grp] = f["agent"][grp][()]
            out["agent"]["param_names"] = list(f["agent"].attrs.get("param_names", []))
            out["agent"]["noise_names"] = list(f["agent"].attrs.get("noise_names", []))
            out["agent"]["noise_scales"] = list(f["agent"].attrs.get("noise_scales", []))
            out["agent"]["all_names"] = list(f["agent"].attrs.get("all_names", []))
    return out, meta
