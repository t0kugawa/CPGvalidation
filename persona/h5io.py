"""HDF5 persistence for a personalisation/validation run.

One file holds everything a reader needs to reproduce the figures:
the per-step log arrays, the parameter mean/spread histories, the per-step
influence, and scalar metadata.
"""
import numpy as np
import h5py

_SCALAR_LOG = [
    "time_ms", "ankle_model_L", "ankle_model_R", "ankle_target_L",
    "ankle_target_R", "loss", "advantage", "mus_E_L", "mus_F_L",
    "mus_E_R", "mus_F_R", "cut_L", "ia_rate_E_L", "ib_rate_E_L",
]
_STR_LOG = ["phase_L", "phase_R", "stage"]


def save_run(path, log, agent, meta=None):
    with h5py.File(path, "w") as f:
        for k, v in (meta or {}).items():
            f.attrs[k] = v
        g = f.create_group("log")
        for k in _SCALAR_LOG:
            v = getattr(log, k, None)
            if v is not None and len(v):
                g.create_dataset(k, data=np.asarray(v, dtype=float))
        for k in _STR_LOG:
            v = getattr(log, k, None)
            if v is not None and len(v):
                g.create_dataset(k, data=np.asarray(v, dtype=h5py.string_dtype()))
        hist = getattr(agent, "history", {}) or {}
        ag = f.create_group("agent")
        for grp in ("means", "sigmas_ctrl"):
            gg = ag.create_group(grp)
            for k, v in hist.get(grp, {}).items():
                gg.create_dataset(k, data=np.asarray(v, dtype=float))
        infl = hist.get("influence")
        if infl is not None and len(infl):
            ag.create_dataset("influence", data=np.asarray(infl, dtype=float))
        ag.attrs["param_names"] = list(getattr(agent.strategy, "param_names", []))
        ag.attrs["all_names"] = list(getattr(agent.strategy, "all_names", []))


def load_run(path):
    """Return (data, meta); data['log'] holds arrays, data['agent'] the histories."""
    out = {"log": {}, "agent": {"means": {}, "sigmas_ctrl": {}}}
    with h5py.File(path, "r") as f:
        meta = {k: f.attrs[k] for k in f.attrs}
        if "log" in f:
            for k, v in f["log"].items():
                arr = v[()]
                if getattr(arr, "dtype", None) is not None and arr.dtype.kind in ("S", "O"):
                    arr = np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
                out["log"][k] = arr
        if "agent" in f:
            for grp in ("means", "sigmas_ctrl"):
                if grp in f["agent"]:
                    for k, v in f["agent"][grp].items():
                        out["agent"][grp][k] = v[()]
            if "influence" in f["agent"]:
                out["agent"]["influence"] = f["agent"]["influence"][()]
            out["agent"]["param_names"] = list(f["agent"].attrs.get("param_names", []))
            out["agent"]["all_names"] = list(f["agent"].attrs.get("all_names", []))
    return out, meta