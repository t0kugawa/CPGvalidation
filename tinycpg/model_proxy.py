import os
from model.adapter import CPGAdapter, CPGOutput, CPGInput
import importlib.util
import sys, builtins
import nest

from model.model_cfg import SIM_MS


class NESTAdapter(CPGAdapter):
    def __init__(self,
                 script_path: str,
                 sim_chunk_ms: float = 10.0,
                 threads: int = 10,
                 resolution_ms: float = 0.2,
                 debug: bool = False,
                 extra_argv: list[str] | None = None,
                 ):
        self.script_path = script_path
        self.sim_ms = SIM_MS
        self.sim_chunk_ms = sim_chunk_ms
        self.threads = threads
        self.resolution_ms = resolution_ms
        self.debug = debug
        self.extra_argv = extra_argv or []

        self.t_ms = 0.0
        self.step_idx = 0
        self.cut_frac_L = 0.0
        self.cut_frac_R = 0.0

        self.build_cpg()

        self.leg = builtins.CPG_LEG
        self.state = builtins.CPG_STATE
        self.logs = builtins.CPG_LOGS
        self.times = builtins.CPG_TIMES
        self.wstats = builtins.CPG_WSTATS
        self.cfg = builtins.CPG_CFG
        self.update_leg = builtins.CPG_UPDATE_LEG
        self.log_weights = builtins.CPG_LOG_WEIGHTS

        self.dt_couple = float(self.cfg["CHUNK_MS"])
        if abs(self.dt_couple - self.sim_chunk_ms) > 1e-6:
            print(f"[NESTAdapter] WARNING: requested chunk {self.sim_chunk_ms} ms "
                  f"was quantized to {self.dt_couple} ms (resolution={self.resolution_ms}).")

    def update(self, cpg_in: CPGInput):
        nest.SetStatus(self.leg["L"]["cut_pg"], {"rate": cpg_in.cut_L})
        nest.SetStatus(self.leg["R"]["cut_pg"], {"rate": cpg_in.cut_R})

        cut_on = float(self.cfg["CUT_RATE_ON_HZ"]) or 1.0
        self.cut_frac_L = min(1.0, max(0.0, float(cpg_in.cut_L) / cut_on))
        self.cut_frac_R = min(1.0, max(0.0, float(cpg_in.cut_R) / cut_on))

    def step(self) -> CPGOutput:
        if self.t_ms + self.dt_couple > self.sim_ms + 1e-6:
            raise RuntimeError(
                f"CPG simulation exceeded SIM_MS={self.sim_ms} ms "
                f"(t_ms={self.t_ms}, dt={self.dt_couple}). "
                f"Increase SIM_MS in model_cfg — BS spike_generators are pre-programmed "
                f"only up to SIM_MS and stop firing beyond it."
            )
        nest.Simulate(self.dt_couple)
        self.t_ms += self.dt_couple
        self.step_idx += 1

        do_rate = bool(self.step_idx % int(self.cfg["rate_every"]) == 0)
        self.update_leg("L", self.t_ms, self.dt_couple, self.cut_frac_L, do_rate)
        self.update_leg("R", self.t_ms, self.dt_couple, self.cut_frac_R, do_rate)
        self.log_weights(self.t_ms, self.step_idx)

        return CPGOutput(
            mus_E_L=self.logs["L"]["act_e"][-1],
            mus_F_L=self.logs["L"]["act_f"][-1],
            mus_E_R=self.logs["R"]["act_e"][-1],
            mus_F_R=self.logs["R"]["act_f"][-1],
            t_ms=self.t_ms,
        )

    def build_cpg(self):
        os.environ["CPG_DIAG_HOOK"] = "1"

        spec = importlib.util.spec_from_file_location("cpg_nest", self.script_path)
        cpg_nest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cpg_nest)

        argv = [
            "cpg_nest",
            "--sim-ms", str(self.sim_ms),
            "--simulate-chunk-ms", str(self.sim_chunk_ms),
            "--resolution-ms", str(self.resolution_ms),
            "--threads", str(self.threads),
        ]
        if self.debug:
            argv.append("--debug-small")
        argv.extend(self.extra_argv)

        old_argv = sys.argv
        sys.argv = argv
        try:
            cpg_nest.main()
        finally:
            sys.argv = old_argv

        if not getattr(builtins, "CPG_READY", False):
            raise RuntimeError(
                "CPG module did not set CPG_READY. "
                "Check that CPG_DIAG_HOOK=1 was set BEFORE main() and that "
                "the hook block exists in main()."
            )

    def _get_side_logs(self, side, default="N/A"):
        state_keys = ["force_e", "force_f", "act_e", "act_f"]
        side_data = {k: self.state[side][k] for k in state_keys}

        log_keys = ["ia_e", "ia_f", "mus_e", "mus_f", "rge", "rgf"]
        for k in log_keys:
            lst = self.logs[side].get(k, [])
            side_data[k] = lst[-1] if lst else default

        return side_data

    def get_logs(self):
        return {
            "t_ms": self.t_ms,
            "L": self._get_side_logs("L"),
            "R": self._get_side_logs("R"),
        }

    def get_stdp_summary(self) -> dict:
        out = {}
        for side in ("L", "R"):
            for key in builtins.CPG_PLASTIC_KEYS:
                means = self.wstats[side][key][0]
                stds = self.wstats[side][key][1]
                out[f"{side}_{key}_mean"] = float(means[-1]) if means else float("nan")
                out[f"{side}_{key}_std"] = float(stds[-1]) if stds else float("nan")
        return out


# Quick tests
if __name__ == "__main__":
    import os

    HERE = os.path.dirname(os.path.abspath(__file__))
    CPG_SCRIPT = os.path.join(HERE, "cpg_2legs_fast.py")

    import sys

    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    print(f"[smoke] using CPG script: {CPG_SCRIPT}")
    print(f"[smoke] script exists:   {os.path.isfile(CPG_SCRIPT)}")

    adapter = NESTAdapter(
        script_path=CPG_SCRIPT,
        sim_chunk_ms=10.0,
        threads=4,
        resolution_ms=0.2,
        debug=False,
    )

    dt = adapter.dt_couple
    print(f"[smoke] dt_couple = {dt} ms, SIM_MS = {adapter.sim_ms} ms")

    cpg_in = CPGInput(cut_L=50.0, cut_R=25.0)
    adapter.update(cpg_in)

    for i in range(3):
        out = adapter.step()
        print(
            f"[step {i + 1}] t={out.t_ms:.2f} ms | "
            f"mus_E_L={out.mus_E_L:.4f} mus_F_L={out.mus_F_L:.4f} | "
            f"mus_E_R={out.mus_E_R:.4f} mus_F_R={out.mus_F_R:.4f} | "
            f"cut_frac L/R = {adapter.cut_frac_L:.3f}/{adapter.cut_frac_R:.3f}"
        )

    print("\n[smoke] logs:")
    for side, data in adapter.get_logs().items():
        if side == "t_ms":
            print(f"  t_ms = {data}")
            continue
        pretty = ", ".join(f"{k}={v}" for k, v in data.items())
        print(f"  {side}: {pretty}")

    print("\n[smoke] OK")
