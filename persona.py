# %% [markdown]
# # tinyCPG Personalization
# %% [markdown]
# ![fig1_schematic.png](fig1_schematic.png)
# %% [markdown]
# [Reference Model] Max-talanov/tinyCPG: the nest CPG model. (n.d.). Retrieved July 3, 2026, from https://github.com/max-talanov/tinyCPG
# 
# ~~[EMG Dataset] Kardassevitch, L., Burchielli, D., & Bonizzato, M. (2025, January 24). Motor Responses to Spinal Cord Stimulation at C4-C6 in Rats. Retrieved from https://osf.io/kpb7x~~
# 
# ~~[Synergy Dataset] S. Fujiki, R. Sakai, T. Funato, and D. Yanagihara, “Dataset of measured kinematics, analysed locomotor parameters, and kinematic synergies of rat hindlimb locomotion during split-belt treadmill walking,” Oct. 2025, doi: 10.5281/zenodo.17310509.~~
# 
# ~~[Locomotion Dataset] [1] S. Fujiki, S. Aoi, T. Funato, Y. Sato, K. Tsuchiya, and D. Yanagihara, “Adaptive hindlimb split-belt treadmill walking in rats by controlling basic muscle activation patterns via phase resetting,” Sci Rep, vol. 8, no. 1, p. 17341, Nov. 2018, doi: 10.1038/s41598-018-35714-8.~~
# 
# 
# %%
# %pip install
"numpy>=1.24"
"h5py>=3.8"
"matplotlib>=3.6"
"pandas>=3.0.3"
"nest-simulator>=3.10.0"
"scipy>=1.18.0"
# %% [markdown]
# ## Test Model (TinyTinyCPG, STDP)
# %%
from typing import Any
import nest
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

plt.close('all')
# %%
INIT_IZHI_PARAMS = {
    "a": 0.02,
    "b": 0.2,
    "c": -65.0,
    "d": 8.0,
    "V_th": 30.0,
    "V_min": -120.0,
}

DELAY_MS: float = 1.0
DELAY_COMM_MS: float = 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def poisson_to_parrot(poisson_n: int, parrot_n: int, hz: float) -> Any:
    assert poisson_n == parrot_n, f"one_to_one connection requires equal poisson_n (provided {poisson_n}) and parrot_n (provided {parrot_n})"

    poisson = nest.Create("poisson_generator", poisson_n)
    parrot = nest.Create("parrot_neuron", parrot_n)
    nest.Connect(poisson, parrot, conn_spec={"rule": "one_to_one"})
    nest.SetStatus(poisson, {"rate": hz})
    return poisson, parrot


def create_izhi_population(size: int, params=INIT_IZHI_PARAMS) -> Any:
    return nest.Create("izhikevich", size, params=params)


def create_spike_recorder(target):
    rec = nest.Create("spike_recorder", params={"record_to": "memory"})
    nest.Connect(target, rec)
    return rec
# %% [markdown]
# ### Leg
# %%
class Leg:  # L/R
    def __init__(self, n_rg_e: int = 10, n_rg_f: int = 10,
                 n_motor_e: int = 10, n_motor_f: int = 10,
                 n_mus_e: int = 10, n_mus_f: int = 10,
                 n_ia_e: int = 10, n_ia_f: int = 10,
                 n_ia_int: int = 5, n_in_e: int = 10,
                 n_in_f: int = 10, n_cut: int = 10,
                 p_rg2m: float = 0.5, p_rg2mus: float = 0.8,
                 p_rg_req: float = 0.12, p_ia2rg: float = 0.4,
                 p_rg_recip: float = 0.20, ia_base_hz: float = 10.0,
                 cut_rate_off_hz: float = 0.0, w_0in: float = 22.0,
                 w0_rm: float = 30.0, w_m2mus: float = 1.0,
                 w_rg: float = 8.0, w_rg_recip: float = -18.0,
                 ia2rg_w: float = 12.0):
        self.rg_e = create_izhi_population(n_rg_e)
        self.rg_f = create_izhi_population(n_rg_f)
        self.m_e = create_izhi_population(n_motor_e)
        self.m_f = create_izhi_population(n_motor_f)
        self.ia_int = create_izhi_population(n_ia_int)
        self.ia_e, self.in_e = poisson_to_parrot(n_ia_e, n_in_e, ia_base_hz)
        self.ia_f, self.in_f = poisson_to_parrot(n_ia_f, n_in_f, ia_base_hz)
        self.cut_pg, self.cut_in = poisson_to_parrot(n_cut, n_cut, cut_rate_off_hz)
        self.mus_e = nest.Create("parrot_neuron", n_mus_e)
        self.mus_f = nest.Create("parrot_neuron", n_mus_f)


        nest.Connect(self.cut_in, self.rg_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2m},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_0in, "delay": DELAY_MS})  # CUT -> RG_E

        nest.Connect(self.rg_f, self.m_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2m},
                     syn_spec={"synapse_model": "static_synapse", "weight": w0_rm,
                               "delay": DELAY_MS})  # RG_E -> MOTOR_E
        nest.Connect(self.rg_e, self.m_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2m},
                     syn_spec={"synapse_model": "static_synapse", "weight": w0_rm,
                               "delay": DELAY_MS})  # RG_F -> MOTOR_F

        nest.Connect(self.m_e, self.mus_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2mus},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_m2mus,
                               "delay": DELAY_MS})  # M_E -> EXTENSOR
        nest.Connect(self.m_f, self.mus_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2mus},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_m2mus,
                               "delay": DELAY_MS})  # M_F -> FLEXOR

        nest.Connect(self.rg_e, self.rg_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_req},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg,
                               "delay": DELAY_MS})  # RG_E -> RG_E (recursive)
        nest.Connect(self.rg_f, self.rg_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_req},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg,
                               "delay": DELAY_MS})  # RG_F -> RG_F (recursive)

        nest.Connect(self.rg_e, self.rg_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_recip},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg_recip,
                               "delay": DELAY_MS})  # RG_E -> RG_F
        nest.Connect(self.rg_f, self.rg_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_recip},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg_recip,
                               "delay": DELAY_MS})  # RG_F -> RG_E

        nest.Connect(self.ia_e, self.rg_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg},
                     syn_spec={"synapse_model": "static_synapse", "weight": ia2rg_w,
                               "delay": DELAY_MS})  # IA_E (spindle sensor extensor) -> RG_E
        nest.Connect(self.ia_f, self.rg_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg},
                     syn_spec={"synapse_model": "static_synapse", "weight": ia2rg_w,
                               "delay": DELAY_MS})  # IA_F (spindle sensor extensor) -> RG_F

        self.rec_muse = create_spike_recorder(self.mus_e)
        self.rec_musf = create_spike_recorder(self.mus_f)

    def mus_is_spiked(self, current_time):
        e_spiked = False
        if len(self.rec_muse.events['times']) > 0:
            last_spike = self.rec_muse.events['times'][-1]
            if abs(last_spike - current_time) < 1e-6:
                e_spiked = True

        f_spiked = False
        if len(self.rec_musf.events['times']) > 0:
            last_spike = self.rec_musf.events['times'][-1]
            if abs(last_spike - current_time) < 1e-6:
                f_spiked = True

        return e_spiked, f_spiked

    def get_spike_count(self):
        e_count = len(self.rec_muse.events['times'])
        f_count = len(self.rec_musf.events['times'])
        return e_count, f_count

# %% [markdown]
# ### Brain Stem
# %%
class BrainStem:
    def __init__(self, n_bs: int = 100, bs_rate_base_hz: float = 0.0,
                 bs_rate_amp_hz: float = 300.0, bs_noise_std_hz: float = 0.0,
                 bs_osc_hz: float = 1.0, base_drive_hz: float = 10.0,
                 bs_phase={"L": 0.0, "R": np.pi}):
        self.bs_pg_e, self.bs_in_e = poisson_to_parrot(n_bs, n_bs, bs_rate_base_hz)
        self.bs_pg_f, self.bs_in_f = poisson_to_parrot(n_bs, n_bs, bs_rate_base_hz)
        self.base_pg, self.base_in = poisson_to_parrot(n_bs, n_bs, base_drive_hz)

        self.bs_osc_hz = bs_osc_hz
        self.bs_rate_amp_hz = bs_rate_amp_hz
        self.bs_rate_base_hz = bs_rate_base_hz
        self.bs_noise_std_hz = bs_noise_std_hz
        self.bs_phase = bs_phase

    def bs_rates_counterphase(self, t_ms: float = 1, corr: float = 0.5, leg: str = "L") -> tuple[float, float]:
        t_s = t_ms / 1000.0
        s = np.sin(2.0 * np.pi * self.bs_osc_hz * t_s + self.bs_phase[leg])
        e = max(0.0, s)
        f = max(0.0, -s)

        common_noise = np.random.normal(0.0, self.bs_noise_std_hz * corr)
        indep_noise_e = np.random.normal(0.0, self.bs_noise_std_hz * (1 - corr))
        indep_noise_f = np.random.normal(0.0, self.bs_noise_std_hz * (1 - corr))

        noise_e = common_noise + indep_noise_e
        noise_f = common_noise + indep_noise_f

        r_e = self.bs_rate_base_hz + self.bs_rate_amp_hz * e + noise_e
        r_f = self.bs_rate_base_hz + self.bs_rate_amp_hz * f + noise_f

        r_e = clamp(r_e, self.bs_rate_base_hz,
                    self.bs_rate_base_hz + self.bs_rate_amp_hz + self.bs_noise_std_hz * 3)  # 3σ rule - 99.7% coverage
        r_f = clamp(r_f, self.bs_rate_base_hz,
                    self.bs_rate_base_hz + self.bs_rate_amp_hz + self.bs_noise_std_hz * 3)  # 3σ rule - 99.7% coverage
        return r_e, r_f
# %% [markdown]
# ## Main Model
# %% [markdown]
# ### L2CPG
# %%
class L2CPG:
    def __init__(self, bs: BrainStem, legs: dict[str, Leg], p_con: float = 0.5,
                 p_comm: float = 0.08, base_drive_p: float = 0.08, base_drive_w: float = 18.0,
                 w_comm_inh: float = -10.0, enable_commissural: bool = True,
                 dt_ms: float = 10.0, window_ms: float = 1.0):

        self.dt_ms = dt_ms
        for leg_key, leg in legs.items():
            nest.Connect(bs.bs_in_e, leg.rg_e,
                         conn_spec={"rule": "pairwise_bernoulli", "p": p_con},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})
            nest.Connect(bs.bs_in_f, leg.rg_f,
                         conn_spec={"rule": "pairwise_bernoulli", "p": p_con},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})

            nest.Connect(bs.base_in, leg.rg_e,
                         conn_spec={"rule": "pairwise_bernoulli", "p": base_drive_p},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})
            nest.Connect(bs.base_in, leg.rg_f,
                         conn_spec={"rule": "pairwise_bernoulli", "p": base_drive_p},
                         syn_spec={"synapse_model": "static_synapse", "weight": base_drive_w, "delay": DELAY_MS})

        if enable_commissural and len(legs) == 2:
            LL = legs["L"]
            RR = legs["R"]

            nest.Connect(LL.rg_e, RR.rg_f, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_inh, "delay": DELAY_COMM_MS})
            nest.Connect(RR.rg_e, LL.rg_f, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_inh, "delay": DELAY_COMM_MS})
            nest.Connect(LL.rg_f, RR.rg_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_inh, "delay": DELAY_COMM_MS})
            nest.Connect(RR.rg_f, LL.rg_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_comm},
                         syn_spec={"synapse_model": "static_synapse", "weight": w_comm_inh, "delay": DELAY_COMM_MS})

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

# %% [markdown]
# ## PERSONA (Parameter Exploration for Real-time Spiking Online Network Architecture)
# %%
import numpy as np


class PersonaModel:
    def __init__(self, params_default: np.ndarray,
                 sigma_default: float = 0.1,
                 sigma_min: float = 0.01,
                 sigma_max: float = 0.5,
                 lr_mu: float = 0.01,
                 lr_sigma: float = 0.005,
                 baseline_alpha: float = 0.9,
                 homeostasis: float = 0.01,
                 k: int = 20):
        self.n_params = len(params_default)
        self.bio_params = np.array(params_default, dtype=np.float64)
        self.mu = np.array(params_default, dtype=np.float64)
        self.sigma = np.ones(self.n_params, dtype=np.float64) * sigma_default

        self.W = np.ones(self.n_params) / self.n_params
        self.H = homeostasis

        self.baseline = 0.0
        self.baseline_alpha = baseline_alpha

        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.d_x_history = []
        self.d_r_history = []

        self.k = k
        self.history = {
            'mu': [self.mu.copy()],
            'sigma': [self.sigma.copy()],
            'baseline': [0.0],
            'rewards': []
        }

    def sample_action(self) -> np.ndarray:
        return np.random.normal(self.mu, self.sigma)

    def get_reward(self, output, target) -> float:
        error = np.mean(np.abs(np.asarray(target) - np.asarray(output)))
        return -float(np.clip(error, 0, 1e6))

    def get_influence(self):
        if len(self.d_x_history) < 10:
            return np.ones(self.n_params) / self.n_params

        X = np.array(self.d_x_history)
        Y = np.array(self.d_r_history)
        lambda_reg = 1e-4
        XTX = X.T @ X + lambda_reg * np.eye(self.n_params)
        XTY = X.T @ Y
        beta = np.linalg.solve(XTX, XTY)

        beta_abs = np.abs(beta)
        W = beta_abs / (np.sum(beta_abs) + 1e-8)

        return W

    def update_params(self, action, output, target):
        R = self.get_reward(output, target)
        A = R - self.baseline

        mu_old = self.mu.copy()
        R_old = self.history['rewards'][-1] if self.history['rewards'] else 0.0
        self.baseline = self.baseline_alpha * self.baseline + (1 - self.baseline_alpha) * R
        W = self.get_influence()

        bio_safe_factor = self.H * (self.mu - self.bio_params) / (self.sigma + 1e-8)
        pg_grad = (action - self.mu) / (self.sigma ** 2 + 1e-8) * A
        self.mu += W * (self.lr_mu * pg_grad - bio_safe_factor)

        sigma_grad = ((action - self.mu) ** 2 - self.sigma ** 2) / (self.sigma ** 3 + 1e-8) * A
        self.sigma += W * self.lr_sigma * sigma_grad
        self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        delta_mu = self.mu - mu_old
        delta_R = R - R_old
        self.d_x_history.append(delta_mu)
        self.d_r_history.append(delta_R)

        if len(self.d_x_history) > self.k:
            self.d_x_history = self.d_x_history[-self.k:]
            self.d_r_history = self.d_r_history[-self.k:]

        self.history['mu'].append(self.mu.copy())
        self.history['sigma'].append(self.sigma.copy())
        self.history['rewards'].append(R)
        self.history['baseline'].append(self.baseline)

        return self.mu.copy()

# %% [markdown]
# ## Environment (EMG)
# %%
DT_MS: float = 10.0
# %% [markdown]
# ### EMG Data Loader
# %%
# %pip install \
"h5py>=3.16.0" \
"seaborn"
# %%
import h5py
from pathlib import Path


class EMGLoader:
    def __init__(self, path: Path, field: str):
        self.file = h5py.File(path, "r")
        self.data = self.file[field]
        self.pointer = 0

    def step(self):
        if self.pointer >= len(self.data):
            self.reset()

        target = self.data[self.pointer]
        self.pointer += 1
        return target

    def reset(self):
        self.pointer = 0

    def close(self):
        self.file.close()
# %% [markdown]
# ### Main Environment
# %%
class Environment:
    def __init__(self, snn: L2CPG, persona: PersonaModel, loader: EMGLoader, n_steps: int = 50):
        self.snn = snn
        self.loader = loader
        self.persona = persona
        self.n_steps = n_steps
# %% [markdown]
# ## Test Model (TinyTinyCPG, STDP)
# %% [markdown]
# ## Experiment 1 (1 leg, Extensor, STDP, Persona)
# %% [markdown]
# Affected Parameters:
# * **Izhikevich neuronal params**
# * **Synaptic Probabilities**: p_rg2m, p_rg2mus, p_rg_recip, p_ia2rg
# * **Synaptic Weights**: w_0in, w_m2mus, w_rg_recip, ia2rg_w
# * **BrainStem (L2CPG)**: base_drive_p, base_drive_w, p_con
# %%
E_PATH = {"data/foot/4/bio_E_21cms_40Hz_i100_2pedal_no5ht_T.hdf5": [
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst7_Ton_6.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst8_Ton_6.fig"]}
F_PATH = {"data/foot/4/bio_F_21cms_40Hz_i100_2pedal_no5ht_T.hdf5": [
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst10_Ton_21.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst11_Ton_23.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst2_Ton_13.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst3_Ton_14.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst4_Ton_15.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst5_Ton_16.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst6_Ton_17.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst7_Ton_19.fig",
    "#4_112409_PRE_BIPEDAL_normal_21cms_burst9_Ton_20.fig"
]}

BRAINSTEM = BrainStem(base_drive_hz=40.0)
LEG_L = Leg()
LEG_R = Leg()

cpg = L2CPG(BRAINSTEM, legs={"L": LEG_L, "R": LEG_R})

PARAM_NAMES = [
    "p_rg2m", "p_rg2mus", "p_rg_recip", "p_ia2rg",
    "w_0in", "w_m2mus", "w_rg_recip", "ia2rg_w",
    "base_drive_p", "base_drive_w",
]

DEFAULT_PARAMS = np.array([0.5, 0.8, 0.20, 0.4, 22.0, 1.0, -18.0, 12.0, 0.08, 0.5])
PERSONA = PersonaModel(DEFAULT_PARAMS)

# %%
def build_network(params):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 1.0, "print_time": False})

    p = params
    p_rg2m, p_rg2mus, p_rg_recip, p_ia2rg = p[0], p[1], p[2], p[3]
    w_0in, w_m2mus, w_rg_recip, ia2rg_w = p[4], p[5], p[6], p[7]
    base_drive_p, base_drive_w = p[8], p[9]

    p_rg2m = np.clip(p_rg2m, 0.01, 0.99)
    p_rg2mus = np.clip(p_rg2mus, 0.01, 0.99)
    p_rg_recip = np.clip(p_rg_recip, 0.01, 0.99)
    p_ia2rg = np.clip(p_ia2rg, 0.01, 0.99)
    base_drive_p = np.clip(base_drive_p, 0.01, 0.99)
    base_drive_w = np.clip(base_drive_w, 0.1, 50.0)

    brainstem = BrainStem(base_drive_hz=40.0)
    leg_L = Leg(p_rg2m=p_rg2m, p_rg2mus=p_rg2mus, p_rg_recip=p_rg_recip,
                p_ia2rg=p_ia2rg, w_0in=abs(w_0in), w_m2mus=abs(w_m2mus),
                w_rg_recip=w_rg_recip, ia2rg_w=abs(ia2rg_w))
    leg_R = Leg(p_rg2m=p_rg2m, p_rg2mus=p_rg2mus, p_rg_recip=p_rg_recip,
                p_ia2rg=p_ia2rg, w_0in=abs(w_0in), w_m2mus=abs(w_m2mus),
                w_rg_recip=w_rg_recip, ia2rg_w=abs(ia2rg_w))

    cpg = L2CPG(brainstem, legs={"L": leg_L, "R": leg_R},
                base_drive_p=base_drive_p, base_drive_w=base_drive_w)
    return cpg


def run_episode(cpg, target_gen, n_steps=30):
    outputs, targets = [], []
    for _ in range(n_steps):
        result = cpg.step()
        out = result["L"]["E"]
        tgt = target_gen.step()
        outputs.append(out)
        targets.append(tgt)
    return np.array(outputs), np.array(targets)


def build_loaders(path_dict):
    loaders = []
    for path, fields in path_dict.items():
        for field in fields:
            loaders.append(EMGLoader(Path(path), field))
    return loaders


def run_experiment(n_episodes=1500, steps_per_ep=30,
                    persona_kwargs=None, seed=42,
                    path=None):
    np.random.seed(seed)
    if persona_kwargs is None:
        persona_kwargs = {}

    persona = PersonaModel(DEFAULT_PARAMS, **persona_kwargs)
    loaders = build_loaders(path)
    n_loaders = len(loaders)

    rewards, baselines = [], []
    mean_outputs, mean_targets = [], []
    all_mu_history = [DEFAULT_PARAMS.copy()]
    all_sigma_history = [np.ones(len(DEFAULT_PARAMS)) * persona_kwargs.get("sigma_default", 0.1)]

    print(f"Running {n_episodes} episodes x {steps_per_ep} steps "
          f"across {n_loaders} extensor bursts ...")
    try:
        for ep in range(n_episodes):
            if ep % 20 == 0:
                print(f"  Episode {ep}/{n_episodes}")

            target = loaders[ep % n_loaders]

            action = persona.sample_action()
            cpg = build_network(action)
            outputs, targets = run_episode(cpg, target, steps_per_ep)

            mean_out = float(np.mean(outputs))
            mean_tgt = float(np.mean(targets))

            persona.update_params(action, mean_out, mean_tgt)

            rewards.append(persona.history["rewards"][-1])
            baselines.append(persona.history["baseline"][-1])
            mean_outputs.append(mean_out)
            mean_targets.append(mean_tgt)
            all_mu_history.append(persona.mu.copy())
            all_sigma_history.append(persona.sigma.copy())
    finally:
        for loader in loaders:
            loader.close()

    return {
        "persona": persona,
        "rewards": np.array(rewards),
        "baselines": np.array(baselines),
        "mean_outputs": np.array(mean_outputs),
        "mean_targets": np.array(mean_targets),
        "mu_history": np.array(all_mu_history),
        "sigma_history": np.array(all_sigma_history),
        "n_episodes": n_episodes,
    }

# %% [markdown]
# ### Plotting
# %%
import matplotlib.gridspec as gridspec

COLORS = plt.cm.tab10(np.linspace(0, 0.9, 10))


def fig1_param_distributions(res):
    """Initial vs final parameter distributions."""
    mu_init = res["mu_history"][0]
    sig_init = res["sigma_history"][0]
    mu_final = res["mu_history"][-1]
    sig_final = res["sigma_history"][-1]

    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(2, n, figsize=(20, 5), sharey=False)
    fig.suptitle("Fig 1 - Parameter Distributions: Initial vs Final", fontsize=14, fontweight="bold")

    for i, name in enumerate(PARAM_NAMES):
        for row, (mu, sig, label, color) in enumerate([
            (mu_init[i], sig_init[i], "Initial", "#4C72B0"),
            (mu_final[i], sig_final[i], "Final", "#DD8452"),
        ]):
            ax = axes[row, i]
            vals = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
            pdf = np.exp(-0.5 * ((vals - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            ax.fill_between(vals, pdf, alpha=0.35, color=color)
            ax.plot(vals, pdf, color=color, lw=1.5)
            ax.axvline(mu, color=color, lw=1.2, ls="--")
            ax.set_title(name, fontsize=8)
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(label, fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    return fig


def fig2_baseline_rewards(res):
    """Reward, baseline, and output vs target over episodes."""
    ep = np.arange(res["n_episodes"])
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Fig 2 - Learning Dynamics Over Episodes", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(ep, res["rewards"], color="#4C72B0", lw=1.5, label="Reward R")
    ax.plot(ep, res["baselines"], color="#DD8452", lw=1.5, ls="--", label="Baseline b")
    ax.fill_between(ep, res["rewards"], res["baselines"],
                     where=res["rewards"] >= res["baselines"],
                     alpha=0.2, color="green", label="R > b (positive advantage)")
    ax.fill_between(ep, res["rewards"], res["baselines"],
                     where=res["rewards"] < res["baselines"],
                     alpha=0.2, color="red", label="R < b (negative advantage)")
    ax.set_ylabel("Reward / Baseline"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    window = 10
    rm = np.convolve(res["rewards"], np.ones(window) / window, mode="same")
    ax.plot(ep, res["rewards"], color="#9FC5E8", lw=0.8, alpha=0.6, label="Raw reward")
    ax.plot(ep, rm, color="#1A5276", lw=2, label=f"Rolling mean (w={window})")
    ax.set_ylabel("Reward"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(ep, res["mean_targets"], color="#2ECC71", lw=1.5, label="Target (mean)")
    ax.plot(ep, res["mean_outputs"], color="#E74C3C", lw=1.5, alpha=0.7, label="Output (mean)")
    ax.set_ylabel("Activity (norm.)"); ax.set_xlabel("Episode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig3_param_stabilization(res):
    """mu and sigma trajectories for each parameter."""
    mu_h = res["mu_history"]
    sig_h = res["sigma_history"]
    ep = np.arange(len(mu_h))
    n = len(PARAM_NAMES)

    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n))
    fig.suptitle("Fig 3 - Parameter Stabilization (mu and sigma over episodes)",
                 fontsize=14, fontweight="bold")

    for i, name in enumerate(PARAM_NAMES):
        c = COLORS[i]
        ax_mu = axes[i, 0]
        ax_mu.plot(ep, mu_h[:, i], color=c, lw=1.5)
        ax_mu.axhline(DEFAULT_PARAMS[i], color="gray", ls=":", lw=1, label="Default")
        ax_mu.set_ylabel(f"mu({name})", fontsize=8)
        ax_mu.grid(alpha=0.25)
        if i == 0: ax_mu.set_title("Mean (mu)", fontsize=10)

        ax_s = axes[i, 1]
        ax_s.plot(ep, sig_h[:, i], color=c, lw=1.5, ls="--")
        ax_s.set_ylabel(f"sigma({name})", fontsize=8)
        ax_s.grid(alpha=0.25)
        if i == 0: ax_s.set_title("Std-dev (sigma)", fontsize=10)

    axes[-1, 0].set_xlabel("Episode")
    axes[-1, 1].set_xlabel("Episode")
    fig.tight_layout()
    return fig


def fig4_accuracy_and_influence(res):
    persona = res["persona"]
    ep = np.arange(res["n_episodes"])

    error = np.abs(res["mean_outputs"] - res["mean_targets"])
    error_rm = np.convolve(error, np.ones(10) / 10, mode="same")
    W_final = persona.get_influence()
    delta_mu = np.abs(res["mu_history"][-1] - res["mu_history"][0])

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Fig 4 - Accuracy, Influence & Parameter Shift", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, error, color="#AAB7B8", lw=0.8, alpha=0.7, label="MAE per episode")
    ax1.plot(ep, error_rm, color="#1A5276", lw=2, label="Rolling mean (w=10)")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("|Output - Target|")
    ax1.set_title("Tracking Error"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(PARAM_NAMES, W_final, color=COLORS[:len(PARAM_NAMES)])
    ax2.set_xticklabels(PARAM_NAMES, rotation=40, ha="right", fontsize=8)
    ax2.set_ylabel("Influence weight W"); ax2.set_title("Final Influence Weights (PERSONA)")
    ax2.grid(alpha=0.3, axis="y")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep, np.cumsum(res["rewards"]), color="#2ECC71", lw=2)
    ax3.set_xlabel("Episode"); ax3.set_ylabel("Cumulative reward")
    ax3.set_title("Cumulative Reward"); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.barh(PARAM_NAMES, delta_mu, color=COLORS[:len(PARAM_NAMES)], alpha=0.8)
    ax4.set_xlabel("|delta mu| (final - initial)"); ax4.set_title("Total Parameter Shift")
    ax4.tick_params(axis="y", labelsize=8); ax4.grid(alpha=0.3, axis="x")

    return fig


def fig5_sigma_heatmap(res):
    """Heat-map of sigma evolution across all parameters."""
    sig_h = res["sigma_history"].T  # shape: (n_params, n_episodes+1)
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(sig_h, aspect="auto", cmap="viridis",
                    extent=[0, res["n_episodes"], -0.5, len(PARAM_NAMES) - 0.5])
    ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_yticklabels(PARAM_NAMES, fontsize=8)
    ax.set_xlabel("Episode"); ax.set_title("Fig 5 - sigma Heat-map: Exploration Dynamics", fontweight="bold")
    fig.colorbar(im, ax=ax, label="sigma (exploration width)")
    fig.tight_layout()
    return fig

# %% [markdown]
# ### Run the experiment and save figures
# %%
results = run_experiment(n_episodes=1000, steps_per_ep=1,
                          persona_kwargs={"sigma_default": 0.2,
                                          "lr_mu": 0.02,
                                          "lr_sigma": 0.02,
                                          "homeostasis": 0.01},
                          path=E_PATH)

figs = [
    ("fig1_param_distributions.png", fig1_param_distributions(results)),
    ("fig2_learning_dynamics.png", fig2_baseline_rewards(results)),
    ("fig3_stabilization.png", fig3_param_stabilization(results)),
    ("fig4_accuracy_influence.png", fig4_accuracy_and_influence(results)),
    ("fig5_sigma_heatmap.png", fig5_sigma_heatmap(results)),
]

import os
os.makedirs("/mnt/user-data/outputs", exist_ok=True)
for fname, fig in figs:
    out_path = f"/mnt/user-data/outputs/{fname}"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.show()

print("\n✓ All done. 5 figures written to /mnt/user-data/outputs/")
