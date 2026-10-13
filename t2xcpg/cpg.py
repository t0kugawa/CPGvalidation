import nest
import numpy as np

from t2xcpg.bs import BrainStem
from t2xcpg.leg import Leg

class L2CPG:
    def __init__(self, bs: BrainStem, legs: dict[str, Leg], p_con: float = 0.5,
                 p_comm: float = 0.08, base_drive_p: float = 0.08, base_drive_w: float = 18.0,
                 w_comm_f_inh: float = -10.0, w_comm_e_inh: float = -8.0,
                 p_comm_f: float = 0.08, p_comm_e: float = 0.06,
                 enable_commissural: bool = True,
                 dt_ms: float = 10.0, window_ms: float = 1.0):

        self.dt_ms = dt_ms
        for leg_key, leg in legs.items():
            # bs_in_e -> rg_e (tonic extensor drive)
            nest.Connect(bs.bs_in_e, leg.rg_e,
                         conn_spec={"rule": "pairwise_bernoulli", "p": p_con},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})
            # bs_in_f -> rg_f (tonic flexor drive)
            nest.Connect(bs.bs_in_f, leg.rg_f,
                         conn_spec={"rule": "pairwise_bernoulli", "p": p_con},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})

            # base_in -> rg_e (background drive)
            nest.Connect(bs.base_in, leg.rg_e,
                         conn_spec={"rule": "pairwise_bernoulli", "p": base_drive_p},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})
            # base_in -> rg_f (background drive)
            nest.Connect(bs.base_in, leg.rg_f,
                         conn_spec={"rule": "pairwise_bernoulli", "p": base_drive_p},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})

        if enable_commissural and len(legs) == 2:
            LL = legs["L"]
            RR = legs["R"]

            # rg_f(L) -> rg_f(R)
            nest.Connect(LL.rg_f, RR.rg_f, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm_f},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_f_inh, "delay": DELAY_COMM_MS})
            # rg_f(R) -> rg_f(L)
            nest.Connect(RR.rg_f, LL.rg_f, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm_f},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_f_inh, "delay": DELAY_COMM_MS})

            # rg_e(L) -> rg_e(R)
            nest.Connect(LL.rg_e, RR.rg_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm_e},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_e_inh, "delay": DELAY_COMM_MS})
            # rg_e(R) -> rg_e(L)
            nest.Connect(RR.rg_e, LL.rg_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm_e},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_e_inh, "delay": DELAY_COMM_MS})

        self.legs = legs
        self.bs = bs

        self.multimeters = {}
        for leg_key, leg in self.legs.items():
            mm_e = nest.Create("multimeter", params={
                "record_from": ["V_m"],
                "interval": self.dt_ms,
                "record_to": "memory"
            })
            mm_f = nest.Create("multimeter", params={
                "record_from": ["V_m"],
                "interval": self.dt_ms,
                "record_to": "memory"
            })

            nest.Connect(mm_e, leg.m_e, conn_spec={"rule": "all_to_all"})
            nest.Connect(mm_f, leg.m_f, conn_spec={"rule": "all_to_all"})

            self.multimeters[leg_key] = {'e': mm_e, 'f': mm_f}

        self.current_time_ms = 0.0
        self.dt_ms = dt_ms
        self.dt_s = dt_ms / 1000.0
        self.window_ms = window_ms
        self.window_steps = int(window_ms / dt_ms)

    def step(self):
        nest.Simulate(self.dt_ms)
        self.current_time_ms += self.dt_ms

        results = {leg_key: {"F": 0.0, "E": 0.0} for leg_key in self.legs.keys()}

        for leg_key in self.legs.keys():
            mm_e = self.multimeters[leg_key]['e']
            mm_f = self.multimeters[leg_key]['f']

            events_e = mm_e.get("events")
            events_f = mm_f.get("events")

            v_m_e = events_e["V_m"]
            v_m_f = events_f["V_m"]

            if len(v_m_e) > 0:
                results[leg_key]['E'] = max(0.0, (np.mean(v_m_e) + 70.0) / 100.0) * 100.0
            if len(v_m_f) > 0:
                results[leg_key]['F'] = max(0.0, (np.mean(v_m_f) + 70.0) / 100.0) * 100.0

            mm_e.set({"n_events": 0})
            mm_f.set({"n_events": 0})

        return results
