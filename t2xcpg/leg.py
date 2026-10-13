import nest
from t2xcpg.utils import poisson_to_parrot, create_izhi_population, create_spike_recorder, create_stdp_synapse, \
    get_jitter_delay


class Leg:  # L/R
    def __init__(self, leg: str, n_rg_e: int = 10, n_rg_f: int = 10,
                 n_motor_e: int = 10, n_motor_f: int = 10,
                 n_mus_e: int = 10, n_mus_f: int = 10,
                 n_ia_in_e: int = 10, n_in_e: int = 20,
                 n_in_f: int = 10, n_cut: int = 10, n_ia_e: int = 10, n_ia_f: int = 10,
                 n_ia_int_e: int = 20, n_ia_int_f: int = 20,
                 # --- вероятности (приведены к основному файлу) ---
                 p_stdp: float = 0.5,          # P_IN_STDP
                 p_rg2m: float = 0.5,          # P_IN_STDP (в основном RG->M идёт с P_IN_STDP)
                 p_rg2mus: float = 0.8,        # P_M2MUS
                 p_rg_rec: float = 0.12,       # P_RG_REC
                 p_ia2rg: float = 0.4,         # IA2RG_P (статические Ia->int пути)
                 p_rg_recip_e: float = 0.15,   # P_RG_RECIP_E
                 p_rg_recip_f: float = 0.30,   # P_RG_RECIP_F
                 p_ia2in: float = 0.25,        # P_IA2IN
                 p_cut2ine: float = 0.30,      # P_CUT2INE
                 p_ia2rg_stdp: float = 0.4,    # P_IA2RG_STDP
                 ia_base_hz: float = 10.0,     # IA_BASE_HZ
                 cut_rate_off_hz: float = 0.0, # CUT_RATE_OFF_HZ
                 # --- веса (приведены к основному файлу) ---
                 w_0in: float = 22.0,          # W0_IN
                 w0_rm: float = 30.0,          # W0_RM
                 w_m2mus: float = 1.0,         # W_M2MUS
                 w_rg_rec_e: float = 4.0,      # W_RG_REC_E
                 w_rg_rec_f: float = 5.5,      # W_RG_REC_F
                 w_ia2rg: float = 12.0,        # IA2RG_W
                 w_rg2inf: float = 18.0,       # W_RG2INF
                 w_inf2rge: float = -48.0,     # W_INF2RGE (Zhang asym, было -28)
                 w_rg2ine: float = 12.0,       # W_RG2INE
                 w_ine2rgf: float = -8.0,      # W_INE2RGF (Zhang asym, было -28)
                 # --- доп. параметры Ia-путей (по образу основного файла) ---
                 w_ia2in: float = 6.0,         # W_IA2IN (было 10)
                 w_ia_in2int: float = 6.0,     # W_IA_IN2INT (было 10)
                 w_ia_int2ant: float = -10.0,  # W_IA_INT2ANT (было -20)
                 w_cut2ine: float = 6.0,       # W_CUT2INE (было 12)
                 w_init_ia: float = 5.0,       # W_INIT_IA (стартовый вес для STDP Ia->RG)
                 w_flex_aff2rgf: float = 6.0,  # W_FLEX_AFF2RGF (для swing-афферента)
                 stdp_ia_rg: bool = False):
        self.leg = leg

        self.rg_e = create_izhi_population(n_rg_e)
        self.rg_f = create_izhi_population(n_rg_f)
        self.in_e = create_izhi_population(n_in_e)
        self.in_f = create_izhi_population(n_in_f)
        self.m_e = create_izhi_population(n_motor_e)
        self.m_f = create_izhi_population(n_motor_f)
        self.ia_int_e = create_izhi_population(n_ia_int_e)
        self.ia_int_f = create_izhi_population(n_ia_int_f)   # FIX: было повторное создание ia_int_e
        self.ia_e, self.ia_in_e = poisson_to_parrot(n_ia_e, n_ia_in_e, ia_base_hz)  # TODO muscle input
        self.ia_f, self.ia_in_f = poisson_to_parrot(n_ia_f, n_ia_in_e, ia_base_hz)  # FIX: n_ia_in_e, было n_in_f
        self.cut_pg, self.cut_in = poisson_to_parrot(n_cut, n_cut, cut_rate_off_hz)
        self.mus_e = nest.Create("parrot_neuron", n_mus_e)
        self.mus_f = nest.Create("parrot_neuron", n_mus_f)

        # cut -> rg_e (STDP)
        nest.Connect(self.cut_in, self.rg_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_stdp},
                     syn_spec={"synapse_model": create_stdp_synapse("default", f"cut_rge_{self.leg}")})

        # rg_f -> m_f
        nest.Connect(self.rg_f, self.m_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2m},
                     syn_spec={"synapse_model": "static_synapse", "weight": w0_rm,
                               "delay": get_jitter_delay()})

        # rg_e -> m_e
        nest.Connect(self.rg_e, self.m_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2m},
                     syn_spec={"synapse_model": "static_synapse", "weight": w0_rm,
                               "delay": get_jitter_delay()})

        # m_e -> mus_e
        nest.Connect(self.m_e, self.mus_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2mus},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_m2mus,
                               "delay": get_jitter_delay()})

        # m_f -> mus_f
        nest.Connect(self.m_f, self.mus_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg2mus},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_m2mus,
                               "delay": get_jitter_delay()})

        # rg_e -> rg_e (recursive)
        nest.Connect(self.rg_e, self.rg_e,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_rec},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg_rec_e,
                               "delay": get_jitter_delay()})

        # rg_f -> rg_f (recursive)
        nest.Connect(self.rg_f, self.rg_f,
                     conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_rec},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg_rec_f,
                               "delay": get_jitter_delay()})

        # --- Ia-пути (по образу основного файла), пока закомментированы ---
        # # ia_in_e -> rg_e (STDP, MOD_IA_RG_STDP)
        # if stdp_ia_rg:
        #     nest.Connect(self.ia_in_e, self.rg_e,
        #                  conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg_stdp},
        #                  syn_spec={"synapse_model": create_stdp_synapse("ia", f"ia_rge_{self.leg}"),
        #                            "weight": w_init_ia, "delay": get_jitter_delay()})
        # # ia_in_f -> rg_f (STDP, MOD_IA_RG_STDP)
        # if stdp_ia_rg:
        #     nest.Connect(self.ia_in_f, self.rg_f,
        #                  conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg_stdp},
        #                  syn_spec={"synapse_model": create_stdp_synapse("ia", f"ia_rgf_{self.leg}"),
        #                            "weight": w_init_ia, "delay": get_jitter_delay()})
        #
        # # ia_in_e -> in_e (MOD_IA_LOOP)
        # nest.Connect(self.ia_in_e, self.in_e,
        #              conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2in},
        #              syn_spec={"synapse_model": "static_synapse", "weight": w_ia2in,
        #                        "delay": get_jitter_delay()})
        # # ia_in_f -> in_f (MOD_IA_LOOP)
        # nest.Connect(self.ia_in_f, self.in_f,
        #              conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2in},
        #              syn_spec={"synapse_model": "static_synapse", "weight": w_ia2in,
        #                        "delay": get_jitter_delay()})
        #
        # # ia_in_e -> ia_int_e (Ia extensor -> inhibitory interneuron)
        # nest.Connect(self.ia_in_e, self.ia_int_e,
        #              conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg},
        #              syn_spec={"synapse_model": "static_synapse", "weight": w_ia_in2int,
        #                        "delay": get_jitter_delay()})
        # # ia_int_e -> m_f (Ia extensor inhibits flexor motor pool)
        # nest.Connect(self.ia_int_e, self.m_f,
        #              conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg},
        #              syn_spec={"synapse_model": "static_synapse", "weight": w_ia_int2ant,
        #                        "delay": get_jitter_delay()})
        #
        # # ia_in_f -> ia_int_f (Ia flexor -> inhibitory interneuron)
        # nest.Connect(self.ia_in_f, self.ia_int_f,
        #              conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg},
        #              syn_spec={"synapse_model": "static_synapse", "weight": w_ia_in2int,
        #                        "delay": get_jitter_delay()})
        # # ia_int_f -> m_e (Ia flexor inhibits extensor motor pool)
        # nest.Connect(self.ia_int_f, self.m_e,
        #              conn_spec={"rule": "pairwise_bernoulli", "p": p_ia2rg},
        #              syn_spec={"synapse_model": "static_synapse", "weight": w_ia_int2ant,
        #                        "delay": get_jitter_delay()})

        # rg_f -> in_f (strong drive of the extensor-suppressing interneuron)
        nest.Connect(self.rg_f, self.in_f, conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_recip_f},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg2inf, "delay": get_jitter_delay()})

        # in_f -> rg_e (STRONG inhibition: F dominates and silences E)
        nest.Connect(self.in_f, self.rg_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_recip_f},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_inf2rge, "delay": get_jitter_delay()})

        # rg_e -> in_e (drives the flexor-suppressing interneuron)
        nest.Connect(self.rg_e, self.in_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_recip_e},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_rg2ine, "delay": get_jitter_delay()})

        # in_e -> rg_f (WEAK inhibition: preserves F's intrinsic rhythm)
        nest.Connect(self.in_e, self.rg_f, conn_spec={"rule": "pairwise_bernoulli", "p": p_rg_recip_e},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_ine2rgf, "delay": get_jitter_delay()})

        # cut -> in_e (MOD_CUT_REFLEX: cutaneous afferents reinforce stance-phase extensor reflex)
        nest.Connect(self.cut_in, self.in_e, conn_spec={"rule": "pairwise_bernoulli", "p": p_cut2ine},
                     syn_spec={"synapse_model": "static_synapse", "weight": w_cut2ine,
                               "delay": get_jitter_delay()})

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
