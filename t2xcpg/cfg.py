from t2xcpg.params import PARAMS

DT_MS: float = PARAMS["dt_ms"]

POPULATIONS_FINAL = {
    "n_rg": PARAMS["n_rg_final"],
    "n_mn": PARAMS["n_mn_final"],
    "n_int": PARAMS["n_int_final"],
    "n_aff": PARAMS["n_aff_final"],
    "n_bs": PARAMS["n_bs_final"],
}
